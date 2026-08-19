#!/usr/bin/env python3
"""
CrewAI VASP Quart asynchronous server
Features: task submission, execution history, detail view, live updates, parallel task queue management
Implemented on top of the CrewServer base class, with async operation support
"""

import os
import json
import uuid
import threading
import re
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from quart import Quart, render_template, request, jsonify, g, send_file, abort
import aiosqlite
import ctypes
from werkzeug.utils import secure_filename

# Directory of this module (quart_server/)
current_dir = Path(__file__).parent

# Import project modules
from ...listener.server_listener import CrewServer, ServerListener
from ...crew import VaspCrew
from ...tools.structure_application_boundary import StructureApplicationBoundary
from ...tools.structure_request_applicability import applicability_classifier_from_config
from ...tools.structure_request_coordinator import (
    InMemoryInvocationStore,
    StructureRequestCoordinator,
)
from ...tools.structure_request_parser import parser_from_config
from ...tools.structure_resolver import StructureResolver
from crewai import Task
from fastmcp.client import Client


def _structure_resolver_from_environment(output_directory: Path) -> StructureResolver:
    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MP_API_KEY is required for pure Materials Project structure requests"
        )
    return StructureResolver(api_key=api_key, output_dir=output_directory)


class TaskStatus(Enum):
    """Task status enum"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedTask:
    """A task waiting in the queue"""
    conversation_id: str
    task_description: str
    created_at: datetime
    status: TaskStatus = TaskStatus.QUEUED


class QuartCrewServer(CrewServer):
    """Quart-based asynchronous CrewServer implementation"""
    
    def __init__(self, crew_config: Dict[str, Any], title: str = "VASPilot Async Server", 
                 work_dir: str = ".", db_path: Optional[str] = None, 
                 allow_path: Optional[str] = None, max_concurrent_tasks: int = 3,
                 max_queue_size: int = 10,
                 structure_boundary: Optional[StructureApplicationBoundary] = None):
        super().__init__()
        self.title = title
        self.config = crew_config
        self.work_dir = os.path.abspath(work_dir)
        self.allow_path = allow_path
        
        # Concurrency control parameters
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_queue_size = max_queue_size

        # Task management
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: List[QueuedTask] = []
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._current_conversation_id: Optional[str] = None

        # Database path
        if db_path is None:
            db_path = os.path.join(work_dir, 'crew_tasks.db')
        self.db_path = os.path.abspath(db_path)

        # Create the Quart app
        template_folder = str(current_dir / "templates")
        self.app = Quart(__name__, template_folder=template_folder)
        self.app.secret_key = 'crew-ai-quart-server'

        # Upload directory
        self.upload_dir = os.path.join(self.work_dir, 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)

        self.generator = VaspCrew(self.config)
        self.structure_boundary = structure_boundary or self._create_structure_boundary()
        self.current_logger = ServerListener(self)
        # Mapping between concurrently running tasks: conversation_id <-> crew_fingerprint
        self._conversation_to_fingerprint: Dict[str, str] = {}
        self._fingerprint_to_conversation: Dict[str, str] = {}
        self._mapping_lock = threading.Lock()
        self._running_threads: Dict[str, threading.Thread] = {}
        self._crew_thread_ids: Dict[str, int] = {}

        # Logging infrastructure (initialized inside the event loop at startup)
        self._log_queue = None
        self._log_worker_task = None
        self._event_loop = None

        # Set up routes
        self._setup_routes()

    def _create_structure_boundary(self) -> StructureApplicationBoundary:
        classifier = applicability_classifier_from_config(self.config)
        parser = parser_from_config(self.config)
        store = InMemoryInvocationStore()

        coordinator = StructureRequestCoordinator(
            parser=parser,
            resolver_factory=_structure_resolver_from_environment,
            invocation_store=store,
        )
        return StructureApplicationBoundary(classifier, coordinator, store)

    async def _init_db(self):
        """Asynchronously initialize the database"""
        try:
            # Ensure the database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"📁 Created database directory: {db_dir}")

            print(f"🗄️ Initializing database: {self.db_path}")

            async with aiosqlite.connect(self.db_path) as conn:
                # Create the task_executions table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS task_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT UNIQUE NOT NULL,
                        task_description TEXT NOT NULL,
                        status TEXT DEFAULT 'queued',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        result TEXT,
                        error_message TEXT
                    )
                ''')
                
                # Create the activity_logs table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS activity_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        role_name TEXT,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (conversation_id) REFERENCES task_executions (conversation_id)
                    )
                ''')

                await conn.commit()

                # Verify the tables were created successfully
                async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
                    tables = [row[0] async for row in cursor]
                    expected_tables = ['task_executions', 'activity_logs']

                    for table in expected_tables:
                        if table in tables:
                            print(f"✅ Table '{table}' created successfully")
                        else:
                            raise Exception(f"Failed to create table '{table}'")

                print("🎉 Database initialization complete")

        except Exception as e:
            print(f"❌ Database initialization failed: {str(e)}")
            print(f"Database path: {self.db_path}")
            print(f"Working directory: {self.work_dir}")
            raise

    async def _get_db(self):
        """Get the database connection"""
        db = getattr(g, '_database', None)
        if db is None:
            try:
                db = g._database = await aiosqlite.connect(self.db_path)
                db.row_factory = aiosqlite.Row

                # Verify the table exists
                async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_executions'") as cursor:
                    result = await cursor.fetchone()
                    if not result:
                        # If the table doesn't exist, reinitialize the database
                        print("⚠️ Table not found, reinitializing database...")
                        await db.close()
                        await self._init_db()
                        db = g._database = await aiosqlite.connect(self.db_path)
                        db.row_factory = aiosqlite.Row

            except Exception as e:
                print(f"❌ Database connection failed: {str(e)}")
                raise
        return db

    async def _close_connection(self, exception):
        """Close the database connection"""
        db = getattr(g, '_database', None)
        if db is not None:
            await db.close()

    async def _get_recent_tasks(self, limit=10):
        """Get recent tasks"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM task_executions ORDER BY created_at DESC LIMIT ?',
                (limit,)
            ) as cursor:
                return await cursor.fetchall()

    async def _get_task_by_id(self, conversation_id):
        """Get a task by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM task_executions WHERE conversation_id = ?',
                (conversation_id,)
            ) as cursor:
                return await cursor.fetchone()

    async def _get_task_logs(self, conversation_id):
        """Get task logs"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                'SELECT * FROM activity_logs WHERE conversation_id = ? ORDER BY id ASC',
                (conversation_id,)
            ) as cursor:
                logs = await cursor.fetchall()

        # Format the logs
        formatted_logs = []
        for log in logs:
            type_names = {
                'system': 'System',
                'agent_input': 'Agent Input',
                'agent_output': 'Agent Output',
                'tool_input': 'Tool Input',
                'tool_output': 'Tool Output'
            }

            # Safely get the role_name field (for backward compatibility with old data)
            try:
                role_name = log['role_name'] if 'role_name' in log.keys() else None
            except (KeyError, TypeError):
                role_name = None
            
            formatted_logs.append({
                'type': log['type'],
                'type_name': type_names.get(log['type'], log['type']),
                'role_name': role_name,
                'content': log['content'],
                'timestamp': self._to_beijing_time_str(log['timestamp']),
                'preview': log['content'][:30] + '...' if len(log['content']) > 30 else log['content']
            })
        
        return formatted_logs

    async def _get_queue_status(self):
        """Get the queue status"""
        running_count = len(self.running_tasks)
        queued_count = len(self.task_queue)

        # Get the latest status from the database
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT COUNT(*) as count FROM task_executions WHERE status = 'running'") as cursor:
                db_running = await cursor.fetchone()
                db_running_count = db_running[0] if db_running else 0
            
            async with db.execute("SELECT COUNT(*) as count FROM task_executions WHERE status = 'queued'") as cursor:
                db_queued = await cursor.fetchone()
                db_queued_count = db_queued[0] if db_queued else 0
        
        return {
            'running_count': running_count,
            'queued_count': queued_count,
            'db_running_count': db_running_count,
            'db_queued_count': db_queued_count,
            'max_concurrent': self.max_concurrent_tasks,
            'max_queue_size': self.max_queue_size
        }

    def _to_beijing_time_str(self, value: Any) -> Optional[str]:
        """Convert an incoming UTC/local time string or datetime into a Beijing-time string.
        - Supports str: 'YYYY-MM-DD HH:MM:SS[.ffffff]' or ISO format;
        - Supports datetime: with or without tzinfo;
        Return format: 'YYYY-MM-DD HH:MM:SS'
        """
        if value is None:
            return None
        dt: Optional[datetime] = None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            s = value.strip()
            # Try common formats first
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except Exception:
                    dt = None
            if dt is None:
                # Fall back to ISO format
                try:
                    # Support a trailing Z
                    if s.endswith('Z'):
                        s = s[:-1]
                        dt = datetime.fromisoformat(s)
                    else:
                        dt = datetime.fromisoformat(s)
                except Exception:
                    return value
        else:
            return str(value)

        # Convert timezone-aware values to naive UTC
        if dt.tzinfo is not None and dt.utcoffset() is not None:
            dt_utc = dt - dt.utcoffset()
        else:
            # Assume the database's CURRENT_TIMESTAMP is UTC
            dt_utc = dt

        bj_dt = dt_utc + timedelta(hours=8)
        return bj_dt.strftime("%Y-%m-%d %H:%M:%S")

    def _format_task_row(self, row: aiosqlite.Row) -> Dict[str, Any]:
        """Convert a task record row into a dict with Beijing-time strings."""
        return {
            'conversation_id': row['conversation_id'],
            'task_description': row['task_description'],
            'status': row['status'],
            'created_at': self._to_beijing_time_str(row['created_at']),
            'started_at': self._to_beijing_time_str(row['started_at']) if 'started_at' in row.keys() else None,
            'completed_at': self._to_beijing_time_str(row['completed_at']) if 'completed_at' in row.keys() else None,
            'result': row['result'] if 'result' in row.keys() else None,
            'error_message': row['error_message'] if 'error_message' in row.keys() else None,
        }

    def _setup_routes(self):
        """Set up Quart routes"""

        @self.app.teardown_appcontext
        async def close_connection(exception):
            await self._close_connection(exception)

        # Return API errors as JSON to avoid HTML responses breaking frontend parsing
        @self.app.errorhandler(404)
        async def handle_404(error):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Not Found', 'path': request.path, 'status': 404}), 404
            return str(error), 404

        @self.app.errorhandler(405)
        async def handle_405(error):
            if request.path.startswith('/api/'):
                allowed = getattr(error, 'valid_methods', None)
                return jsonify({'error': 'Method Not Allowed', 'path': request.path, 'status': 405, 'allowed': allowed}), 405
            return str(error), 405

        @self.app.errorhandler(500)
        async def handle_500(error):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Internal Server Error', 'path': request.path, 'status': 500, 'message': str(error)}), 500
            return str(error), 500
        
        @self.app.route('/')
        async def index():
            """Home page"""
            recent_tasks_rows = await self._get_recent_tasks()
            recent_tasks = [self._format_task_row(row) for row in recent_tasks_rows]
            queue_status = await self._get_queue_status()
            return await render_template('base.html', 
                                       title=self.title,
                                       recent_tasks=recent_tasks,
                                       queue_status=queue_status)

        @self.app.route('/upload', methods=['POST'])
        async def upload_structure():
            """Upload a crystal structure file, returning the saved absolute path"""
            try:
                files = await request.files
                if 'file' not in files:
                    return jsonify({'error': 'File field not found'}), 400
                file = files['file']
                if not file or file.filename == '':
                    return jsonify({'error': 'No file uploaded'}), 400

                filename = secure_filename(file.filename)
                lower_name = filename.lower()
                if not (lower_name.endswith(('.vasp', '.cif', '.xyz')) or lower_name in ('poscar', 'contcar')):
                    return jsonify({'error': 'File type not supported. Only .vasp/.cif/.xyz or POSCAR/CONTCAR are allowed'}), 400

                unique_name = f"{uuid.uuid4().hex}_{filename}"
                save_path = os.path.join(self.upload_dir, unique_name)
                await file.save(save_path)
                abs_path = os.path.abspath(save_path)
                return jsonify({'success': True, 'path': abs_path, 'filename': filename})
            except Exception as e:
                return jsonify({'error': f'Upload failed: {str(e)}'}), 500

        @self.app.route('/submit', methods=['POST'])
        async def submit_task():
            """Submit a task"""
            try:
                data = await request.get_json()
                task_description = data.get('task_description', '').strip()

                if not task_description:
                    return jsonify({'error': 'Please enter a valid task description'}), 400

                # Check whether the queue is full
                current_queue_size = len(self.task_queue)
                current_running = len(self.running_tasks)

                if current_queue_size + current_running >= self.max_queue_size + self.max_concurrent_tasks:
                    return jsonify({'error': f'Queue is full. Currently running: {current_running}, queued: {current_queue_size}, max limit: {self.max_queue_size + self.max_concurrent_tasks}'}), 400

                # Create the task record
                conversation_id = str(uuid.uuid4())
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(
                        'INSERT INTO task_executions (conversation_id, task_description, status) VALUES (?, ?, ?)',
                        (conversation_id, task_description, TaskStatus.QUEUED.value)
                    )
                    await db.commit()

                # Add to the queue and try to process it
                queued_task = QueuedTask(
                    conversation_id=conversation_id,
                    task_description=task_description,
                    created_at=datetime.now()
                )
                self.task_queue.append(queued_task)

                # Process the queue asynchronously
                asyncio.create_task(self._process_queue())
                
                return jsonify({
                    'success': True,
                    'conversation_id': conversation_id,
                    'message': 'Task submitted successfully',
                    'queue_position': len(self.task_queue)
                })
                
            except Exception as e:
                return jsonify({'error': f'Server error: {str(e)}'}), 500

        @self.app.route('/task/<conversation_id>')
        async def task_detail(conversation_id):
            """Task detail page"""
            task = await self._get_task_by_id(conversation_id)
            if not task:
                return "Task not found", 404

            logs = await self._get_task_logs(conversation_id)
            recent_tasks_rows = await self._get_recent_tasks()
            recent_tasks = [self._format_task_row(row) for row in recent_tasks_rows]
            # Convert task detail timestamps to Beijing time
            task_dict = {
                'conversation_id': task['conversation_id'],
                'task_description': task['task_description'],
                'status': task['status'],
                'created_at': self._to_beijing_time_str(task['created_at']),
                'started_at': self._to_beijing_time_str(task['started_at']) if task['started_at'] else None,
                'completed_at': self._to_beijing_time_str(task['completed_at']) if task['completed_at'] else None,
                'result': task['result'],
                'error_message': task['error_message']
            }
            queue_status = await self._get_queue_status()
            
            return await render_template('task_detail.html',
                                       title=self.title,
                                       task=task_dict,
                                       logs=logs,
                                       recent_tasks=recent_tasks,
                                       queue_status=queue_status)

        @self.app.route('/api/task/<conversation_id>/status')
        async def get_task_status(conversation_id):
            """API to get task status"""
            task = await self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            
            return jsonify({
                'status': task['status'],
                'conversation_id': task['conversation_id'],
                'task_description': task['task_description']
            })

        @self.app.route('/api/task/<conversation_id>/logs')
        async def get_task_logs(conversation_id):
            """API to get task logs"""
            task = await self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404

            logs = await self._get_task_logs(conversation_id)

            # Convert the logs to dict format
            logs_data = []
            for log in logs:
                logs_data.append({
                    'type': log['type'],
                    'type_name': log['type_name'],
                    'role_name': log['role_name'],
                    'content': log['content'],
                    'timestamp': log['timestamp'],
                    'preview': log['preview']
                })
            
            return jsonify({
                'task': {
                    'status': task['status'],
                    'conversation_id': task['conversation_id'],
                    'task_description': task['task_description'],
                    'result': task['result'],
                    'error_message': task['error_message']
                },
                'logs': logs_data
            })

        @self.app.route('/api/tasks')
        async def get_tasks():
            """API to get the task list"""
            try:
                recent_tasks = await self._get_recent_tasks()
                tasks_data = []
                for task in recent_tasks:
                    tasks_data.append({
                        'conversation_id': task['conversation_id'],
                        'task_description': task['task_description'],
                        'status': task['status'],
                        'created_at': self._to_beijing_time_str(task['created_at']),
                        'started_at': self._to_beijing_time_str(task['started_at']) if task['started_at'] else None,
                        'completed_at': self._to_beijing_time_str(task['completed_at']) if task['completed_at'] else None
                    })
                return jsonify(tasks_data)
            except Exception as e:
                return jsonify({'error': f'Failed to get task list: {str(e)}'}), 500

        @self.app.route('/api/queue/status')
        async def get_queue_status_api():
            """API to get queue status"""
            try:
                status = await self._get_queue_status()
                # Add queue details
                queue_details = []
                for i, queued_task in enumerate(self.task_queue):
                    queue_details.append({
                        'conversation_id': queued_task.conversation_id,
                        'task_description': queued_task.task_description,
                        'position': i + 1,
                        'created_at': self._to_beijing_time_str(queued_task.created_at)
                    })
                
                status['queue_details'] = queue_details
                return jsonify(status)
            except Exception as e:
                return jsonify({'error': f'Failed to get queue status: {str(e)}'}), 500

        @self.app.route('/api/files/<conversation_id>/<path:filename>')
        async def serve_task_file(conversation_id, filename):
            """Serve file access for a specific task"""
            from urllib.parse import unquote

            try:
                # Decode the path segment by segment
                path_segments = filename.split('/')
                decoded_segments = [unquote(segment) for segment in path_segments]
                decoded_filename = '/'.join(decoded_segments)

                # Check for an absolute-path marker
                is_absolute_path = False
                if decoded_filename.startswith('__ABS__'):
                    decoded_filename = decoded_filename[7:]
                    is_absolute_path = True

                # Build the file path
                task_dir = os.path.join(self.work_dir, conversation_id)

                if is_absolute_path or (decoded_filename.startswith('/') and self.allow_path):
                    file_path = decoded_filename
                else:
                    file_path = os.path.join(task_dir, decoded_filename)

                # Security check
                file_path = os.path.abspath(file_path)
                task_dir = os.path.abspath(task_dir)

                if not is_absolute_path and not self.allow_path:
                    if not file_path.startswith(task_dir) and not file_path.startswith(self.work_dir):
                        abort(403, description="Access denied: file path not in allowed range")

                if not os.path.exists(file_path):
                    abort(404, description=f"File not found: {decoded_filename}")

                # Set the MIME type based on the file extension
                if decoded_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    mimetype = 'image/png' if decoded_filename.lower().endswith('.png') else 'image/jpeg'
                elif decoded_filename.lower().endswith(('.vasp', '.xyz', '.cif')):
                    mimetype = 'text/plain'
                else:
                    mimetype = 'application/octet-stream'
                
                return await send_file(file_path, mimetype=mimetype)
                
            except Exception as e:
                abort(500, description=f"File service error: {str(e)}")

        @self.app.route('/api/files/<conversation_id>/list')
        async def list_task_files(conversation_id):
            """List all files in the task directory"""
            from urllib.parse import quote
            
            try:
                task_dir = os.path.join(self.work_dir, conversation_id)
                if not os.path.exists(task_dir):
                    return jsonify({'files': []})
                
                files = []
                for root, dirs, filenames in os.walk(task_dir):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(file_path, task_dir)
                        file_size = os.path.getsize(file_path)
                        file_type = 'unknown'
                        
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            file_type = 'image'
                        elif filename.lower().endswith(('.vasp', '.xyz', '.cif')):
                            file_type = 'structure'
                        elif filename.lower().endswith(('.txt', '.log', '.out')):
                            file_type = 'text'
                        
                        # Encode the path segment by segment
                        path_segments = relative_path.split('/')
                        encoded_segments = [quote(segment, safe='') for segment in path_segments]
                        encoded_path = '/'.join(encoded_segments)
                        
                        files.append({
                            'filename': filename,
                            'path': relative_path,
                            'size': file_size,
                            'type': file_type,
                            'url': f'/api/files/{conversation_id}/{encoded_path}'
                        })
                
                return jsonify({'files': files})
                
            except Exception as e:
                return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

        @self.app.route('/api/task/<conversation_id>/stop', methods=['POST'])
        async def stop_task(conversation_id):
            """API to cancel a task"""
            try:
                # Capture the currently known fingerprint (if any)
                known_fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                # Check whether the task exists
                task = await self._get_task_by_id(conversation_id)
                if not task:
                    return jsonify({'error': 'Task not found', 'conversation_id': conversation_id, 'fingerprint': known_fingerprint}), 404

                # If the fingerprint isn't available yet and the task is running, wait briefly for the mapping to be established to reduce the race window
                if not known_fingerprint and task['status'] == 'running':
                    for _ in range(10):
                        await asyncio.sleep(0.05)
                        known_fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                        if known_fingerprint:
                            break

                # Check the task status
                if task['status'] not in ['running', 'queued']:
                    return jsonify({'error': f'Task status is {task["status"]}, cannot cancel', 'conversation_id': conversation_id, 'fingerprint': known_fingerprint}), 400

                success = False
                message = ""

                # If the task is still queued, remove it from the queue directly
                if task['status'] == 'queued':
                    self.task_queue = [t for t in self.task_queue if t.conversation_id != conversation_id]
                    success = True
                    message = f"Task removed from queue (conversation_id={conversation_id}, fingerprint={known_fingerprint})"
                    # Remove the mapping if it exists
                    self._unregister_mapping_by_conversation(conversation_id)

                # If the task is running, cancel the running task
                elif conversation_id in self.running_tasks:
                    running_task = self.running_tasks[conversation_id]
                    running_task.cancel()
                    try:
                        await running_task
                    except asyncio.CancelledError:
                        pass
                    self.running_tasks.pop(conversation_id, None)
                    success = True
                    message = f"Running task cancelled (conversation_id={conversation_id}, fingerprint={known_fingerprint})"

                    # Extract and cancel SLURM jobs
                    calc_ids = await self._extract_calc_ids_from_logs(conversation_id)
                    if calc_ids:
                        try:
                            cancel_results = await self._cancel_slurm_job(calc_ids)
                            # Prefer the already-captured fingerprint for logging
                            if known_fingerprint:
                                self.system_log(f"SLURM job cancellation result: {cancel_results}", known_fingerprint)
                            else:
                                # Without a mapping, log directly against the conversation ID
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                log_entry = f"[{timestamp}] SLURM job cancellation result: {cancel_results}"
                                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
                        except Exception as e:
                            if known_fingerprint:
                                self.system_log(f"Error while cancelling SLURM job: {str(e)}", known_fingerprint)
                            else:
                                timestamp = datetime.now().strftime("%H:%M:%S")
                                log_entry = f"[{timestamp}] Error while cancelling SLURM job: {str(e)}"
                                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

                    # Unregister the mapping after successful cancellation or removal from the queue
                    self._unregister_mapping_by_conversation(conversation_id)

                if success:
                    # Update the database status
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute(
                            'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                            ('cancelled', 'Task cancelled', conversation_id)
                        )
                        await db.commit()
                
                return jsonify({
                    'success': success,
                    'message': message,
                    'conversation_id': conversation_id,
                    'fingerprint': known_fingerprint
                })
                
            except Exception as e:
                error_msg = f"Failed to cancel task: {str(e)} (conversation_id={conversation_id}, fingerprint={known_fingerprint})"
                if known_fingerprint:
                    self.system_log(error_msg, known_fingerprint)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {error_msg}"
                    self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
                return jsonify({'error': error_msg, 'conversation_id': conversation_id, 'fingerprint': known_fingerprint}), 500

    async def _process_queue(self):
        """Process the task queue"""
        while self.task_queue and len(self.running_tasks) < self.max_concurrent_tasks:
            # Acquire the semaphore
            if self.task_semaphore.locked():
                break

            queued_task = self.task_queue.pop(0)

            # Create and launch the async task
            async_task = asyncio.create_task(
                self._execute_crew_task_async(queued_task.conversation_id, queued_task.task_description)
            )
            self.running_tasks[queued_task.conversation_id] = async_task

            # Don't wait for the task to finish, keep processing the queue
            asyncio.create_task(self._monitor_task(queued_task.conversation_id, async_task))

    async def _monitor_task(self, conversation_id: str, task: asyncio.Task):
        """Monitor task completion"""
        try:
            await task
        except asyncio.CancelledError:
            fingerprint = self._conversation_to_fingerprint.get(conversation_id)
            if fingerprint:
                self.system_log(f"Task cancelled", fingerprint)
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] Task {conversation_id} was cancelled"
                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
        except Exception as e:
            fingerprint = self._conversation_to_fingerprint.get(conversation_id)
            if fingerprint:
                self.system_log(f"Error during task execution: {str(e)}", fingerprint)
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] Error during execution of task {conversation_id}: {str(e)}"
                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
        finally:
            # Clean up the running task record
            if conversation_id in self.running_tasks:
                del self.running_tasks[conversation_id]
            # Unregister the fingerprint mapping
            self._unregister_mapping_by_conversation(conversation_id)

            # Continue processing the queue
            await self._process_queue()

    async def _extract_calc_ids_from_logs(self, conversation_id):
        """Extract calculation task IDs from task logs"""
        calc_ids = []
        try:
            logs = await self._get_task_logs(conversation_id)

            for log in logs:
                content = log['content']

                # Look for calculation_id in tool_output
                if log['type'] == 'tool_output':
                    try:
                        # Try to parse the content as JSON
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            tool_data = json.loads(json_match.group())
                            if isinstance(tool_data, dict):
                                # Look for the calculation_id field
                                if 'calculation_id' in tool_data:
                                    calc_ids.append(tool_data['calculation_id'])
                                # Also check for calculation_id in nested structures
                                elif isinstance(tool_data, dict):
                                    for key, value in tool_data.items():
                                        if isinstance(value, dict) and 'calculation_id' in value:
                                            calc_ids.append(value['calculation_id'])
                    except (json.JSONDecodeError, AttributeError):
                        # If JSON parsing fails, fall back to regex
                        calc_id_patterns = [
                            r'"calculation_id":\s*"([^"]+)"',
                            r"'calculation_id':\s*'([^']+)'",
                            r'calculation_id.*?([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
                        ]
                        for pattern in calc_id_patterns:
                            matches = re.findall(pattern, content, re.IGNORECASE)
                            calc_ids.extend(matches)

                # Look for UUID-formatted calculation IDs in other log types
                uuid_pattern = r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}'
                uuid_matches = re.findall(uuid_pattern, content, re.IGNORECASE)

                # Filter out the conversation ID itself, keeping only calculation IDs
                for match in uuid_matches:
                    if match != conversation_id and match not in calc_ids:
                        calc_ids.append(match)

            # Deduplicate and return
            return list(set(calc_ids))

        except Exception as e:
            self.system_log(f"Error while extracting calculation IDs: {str(e)}")
            return []

    def _run_crew_kickoff_thread(self, local_dir, task_description, result_container: Dict[str, Any], conversation_id: str) -> None:
        """Execute crew.kickoff in a dedicated thread and record the thread ID and result."""
        self._crew_thread_ids[conversation_id] = threading.get_ident()
        result_container['crew_constructed'] = False
        result_container['crew_kickoff_called'] = False
        try:
            boundary_result = self.structure_boundary.handle(
                source_text=task_description,
                output_directory=local_dir,
                invocation_key=conversation_id,
            )
            result_container['boundary_result'] = boundary_result
            if not boundary_result.should_run_crewai:
                result_container['result'] = boundary_result.rendered_response
                return

            self.system_log("Initializing crew...")
            crew = self.generator.crew(local_dir)
            result_container['crew_constructed'] = True
            # Register the mapping (register before logging so fingerprint is never None when unmapped)
            self._register_mapping(conversation_id, crew.fingerprint.uuid_str)
            self.system_log("Registered mapping", crew.fingerprint.uuid_str)
            self.system_log("Creating user task...", crew.fingerprint.uuid_str)

            # Create the task
            task = Task(
                description=task_description,
                # expected_output="A detailed report, including the execution process, calculation results, and the location of the drawn charts.",
                expected_output=(
                    "A concise, factual report addressing only the user's requested task. "
                    "Do not add calculations, plots, or additional work unless explicitly requested. "
                    "Only report agent and tool actions that actually occurred."),
                output_file=f'crew_output_{uuid.uuid4().hex[:8]}.md',
            )
            
            crew.tasks = [task]
            
            self.system_log("Starting task execution...", crew.fingerprint.uuid_str)
            result_container['crew_kickoff_called'] = True
            result_container['result'] = crew.kickoff()


            self.system_log("Task completed!", crew.fingerprint.uuid_str)
            self.agent_output("FinalResult", str(result_container['result']), crew.fingerprint.uuid_str)
        except BaseException as e:
            result_container['error'] = e
        finally:
            try:
                self._crew_thread_ids.pop(conversation_id, None)
            except Exception:
                pass

    def _inject_exception_into_thread(self, thread_id: int, exc_type=SystemExit) -> bool:
        """Asynchronously inject an exception into the target thread to try to force it to stop.
        Returns True if the injection succeeded, False if it failed or was rolled back.
        """
        try:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(exc_type))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
                return False
            return res == 1
        except Exception:
            return False

    async def _stop_and_join_crew_thread(self, conversation_id: str, timeout: float = 5.0) -> bool:
        """Try to forcibly stop and wait for the crew execution thread for the given session to exit."""
        thread = self._running_threads.get(conversation_id)
        thread_id = self._crew_thread_ids.get(conversation_id)
        stopped = False
        if thread_id is not None:
            stopped = self._inject_exception_into_thread(thread_id, SystemExit)
            print(f"injected, result={stopped}")
        print(f"result={stopped}")
        if thread is not None and thread.is_alive():
            loop = asyncio.get_event_loop()
            start = loop.time()
            while thread.is_alive() and (loop.time() - start) < timeout:
                await asyncio.sleep(0.05)
        self._running_threads.pop(conversation_id, None)
        self._crew_thread_ids.pop(conversation_id, None)
        return stopped

    async def _execute_crew_task_async(self, conversation_id, task_description):
        """Asynchronously execute a crew task"""
        async with self.task_semaphore:
            try:
                # Update the task status
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute(
                        'UPDATE task_executions SET status = ?, started_at = CURRENT_TIMESTAMP WHERE conversation_id = ?',
                        ('running', conversation_id)
                    )
                    await conn.commit()

                # System log (logged directly against the conversation ID, since the fingerprint doesn't exist yet)
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] conversation id:{conversation_id}"
                self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

                # Create the working directory
                local_dir = os.path.join(self.work_dir, conversation_id)
                os.makedirs(local_dir, exist_ok=True)
                old_cwd = os.getcwd()
                os.chdir(local_dir)

                try:
                    # Initialize and establish the mapping
                    # Run the synchronous kickoff in a thread, to allow forcibly stopping it later
                    result_container: Dict[str, Any] = {}
                    thread = threading.Thread(
                        target=self._run_crew_kickoff_thread,
                        args=(local_dir, task_description, result_container, conversation_id),
                        daemon=True,
                        name=f"crew-kickoff-{conversation_id[:8]}"
                    )
                    self._running_threads[conversation_id] = thread
                    thread.start()
                    # Asynchronously poll while waiting for the thread to finish
                    while thread.is_alive():
                        await asyncio.sleep(0.1)
                    # Once the thread finishes, retrieve the result or exception
                    if 'error' in result_container:
                        raise result_container['error']
                    result = result_container.get('result')

                    # Update the task status
                    async with aiosqlite.connect(self.db_path) as conn:
                        await conn.execute(
                            'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, result = ? WHERE conversation_id = ?',
                            ('completed', str(result), conversation_id)
                        )
                        await conn.commit()
                finally:
                    os.chdir(old_cwd)

            except asyncio.CancelledError:
                # Task was cancelled
                # Forcibly stop the background crew execution thread and wait for it to exit
                try:
                    await self._stop_and_join_crew_thread(conversation_id)
                except Exception as e:
                    print(f"Failed to stop crew thread: {e}")
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute(
                        'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                        ('cancelled', 'Task cancelled', conversation_id)
                    )
                    await conn.commit()
                raise
            except Exception as e:
                error_msg = f"Error occurred during execution: {str(e)}"

                # Log the error
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute(
                        'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                        ('failed', error_msg, conversation_id)
                    )
                    await conn.commit()

                # Log the error against the mapped fingerprint if available
                fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                if fingerprint:
                    self.system_log(error_msg, fingerprint)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] {error_msg}"
                    self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')
            finally:
                # Completion log
                fingerprint = self._conversation_to_fingerprint.get(conversation_id)
                if result_container.get('crew_constructed'):
                    try:
                        self.generator.stop()
                    except Exception as e:
                        print(f"Failed to stop mcp client: {e}")
                if fingerprint:
                    self.system_log("Mission ended", fingerprint)
                else:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_entry = f"[{timestamp}] Mission ended"
                    self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

    # CrewServer interface implementation (synchronous version)
    def system_log(self, message: str, crew_fingerprint: str = None):
        """Synchronous system log implementation (writes to the database internally, asynchronously)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Resolve the conversation ID via the fingerprint mapping
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'system', log_entry, role_name='system')

    def agent_input(self, agent_role: str, message: str, crew_fingerprint: str = None):
        """Synchronous agent-input implementation (writes to the database internally, asynchronously)"""
        log_content = f"[{agent_role}] {message}"
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'agent_input', log_content, role_name=agent_role)

    def agent_output(self, agent_role: str, message: str, crew_fingerprint: str = None):
        """Synchronous agent-output implementation (writes to the database internally, asynchronously)"""
        log_content = f"[{agent_role}] {message}"
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'agent_output', log_content, role_name=agent_role)

    def tool_input(self, tool_name: str, message: Any, crew_fingerprint: str = None):
        """Synchronous tool-input implementation (writes to the database internally, asynchronously)"""
        if isinstance(message, (dict, list)):
            log_content = json.dumps(message, ensure_ascii=False)
        else:
            try:
                parsed = json.loads(str(message))
                log_content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                log_content = json.dumps({"raw": str(message)}, ensure_ascii=False)
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'tool_input', log_content, role_name=tool_name)

    def tool_output(self, tool_name: str, message: Any, crew_fingerprint: str = None):
        """Synchronous tool-output implementation (writes to the database internally, asynchronously)"""
        if isinstance(message, (dict, list)):
            log_content = json.dumps(message, ensure_ascii=False)
        else:
            try:
                parsed = json.loads(str(message))
                log_content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                log_content = json.dumps({"raw": str(message)}, ensure_ascii=False)
        conversation_id = self._get_conversation_id_for_fingerprint(crew_fingerprint) if crew_fingerprint else None
        if conversation_id:
            self._schedule_log_to_db(conversation_id, 'tool_output', log_content, role_name=tool_name)

    def _schedule_log_to_db(self, conversation_id, log_type, content, role_name=None):
        """Enqueue onto the single in-event-loop log queue, ensuring write order matches call order."""
        self._enqueue_log_event(conversation_id, log_type, content, role_name)

    async def _log_worker(self):
        """Single-threaded async log-writing worker coroutine, writing strictly in queue order."""
        if self._log_queue is None:
            self._log_queue = asyncio.Queue()
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                while True:
                    item = await self._log_queue.get()
                    try:
                        conversation_id, log_type, content, role_name = item
                        await conn.execute(
                            'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                            (conversation_id, log_type, role_name, content)
                        )
                        await conn.commit()
                    finally:
                        self._log_queue.task_done()
        except asyncio.CancelledError:
            pass

    def _enqueue_log_event(self, conversation_id, log_type, content, role_name=None):
        """Deliver a log event to the main event loop's log queue.
        - Within the same event loop, use put_nowait directly to preserve call order;
        - When called from another thread, deliver via run_coroutine_threadsafe to preserve ordering as much as possible;
        - If the queue isn't initialized yet, fall back to a synchronous direct write (rarely happens before initialization).
        """
        if self._log_queue is not None and self._event_loop is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop is self._event_loop:
                    self._log_queue.put_nowait((conversation_id, log_type, content, role_name))
                else:
                    asyncio.run_coroutine_threadsafe(
                        self._log_queue.put((conversation_id, log_type, content, role_name)),
                        self._event_loop
                    )
            except RuntimeError:
                asyncio.run_coroutine_threadsafe(
                    self._log_queue.put((conversation_id, log_type, content, role_name)),
                    self._event_loop
                )
        else:
            # Fallback: write synchronously when the queue isn't ready yet, to avoid losing log entries
            import sqlite3
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                        (conversation_id, log_type, role_name, content)
                    )
                    conn.commit()
            except Exception as e:
                print(f"Direct log fallback failed: {e}")

    def _register_mapping(self, conversation_id: str, crew_fingerprint: str) -> None:
        """Register the conversation_id <-> crew_fingerprint mapping."""
        print(f"[Mapping] register {conversation_id} -> {crew_fingerprint}")
        with self._mapping_lock:
            self._conversation_to_fingerprint[conversation_id] = crew_fingerprint
            self._fingerprint_to_conversation[crew_fingerprint] = conversation_id
        print(f"[Mapping] size conv2fp={len(self._conversation_to_fingerprint)}, fp2conv={len(self._fingerprint_to_conversation)}")

    def _unregister_mapping_by_conversation(self, conversation_id: str) -> None:
        """Unregister the mapping by conversation_id."""
        with self._mapping_lock:
            print(f"[Mapping] unregister by conversation {conversation_id}")
            crew_fingerprint = self._conversation_to_fingerprint.pop(conversation_id, None)
            if crew_fingerprint:
                self._fingerprint_to_conversation.pop(crew_fingerprint, None)
        print(f"[Mapping] size conv2fp={len(self._conversation_to_fingerprint)}, fp2conv={len(self._fingerprint_to_conversation)}")

    def _get_conversation_id_for_fingerprint(self, crew_fingerprint: Optional[str]) -> Optional[str]:
        """Look up the conversation_id via crew_fingerprint."""
        if not crew_fingerprint:
            return None
        with self._mapping_lock:
            return self._fingerprint_to_conversation.get(crew_fingerprint)

    async def _log_to_db_async(self, conversation_id, log_type, content, role_name=None):
        """Asynchronously write a log entry to the database"""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                (conversation_id, log_type, role_name, content)
            )
            await conn.commit()

    async def _cancel_slurm_job(self, calc_ids: list[str]):
        """Asynchronously cancel SLURM jobs"""
        async with Client(self.config["mcp_server"]["url"]) as client:
            tool_result = await client.call_tool("cancel_slurm_job", {"calc_ids": calc_ids})
        if tool_result.data is None:
            return {"error": "No result from cancel_slurm_job"}
        else:
            return tool_result.data

    async def launch_async(self, host="127.0.0.1", port=5000, debug=False, **kwargs):
        """Asynchronously launch the Quart app"""
        print(f"🚀 Starting {self.title}...")
        print(f"💼 Working directory: {self.work_dir}")
        print(f"🗄️ Database: {self.db_path}")
        print(f"🌐 Server address: http://{host}:{port}")
        print(f"⚡ Max concurrent tasks: {self.max_concurrent_tasks}")
        print(f"📋 Max queue size: {self.max_queue_size}")
        print("=" * 50)
        print("✨ Quart Async Crew AI Server")
        print("📝 Parallel tasks, 📋 queue management, 🔍 live updates")
        print("=" * 50)

        # Initialize the database
        await self._init_db()

        # Initialize the log queue and worker coroutine (inside the main event loop)
        self._event_loop = asyncio.get_running_loop()
        self._log_queue = asyncio.Queue()
        self._log_worker_task = asyncio.create_task(self._log_worker())

        # Set the conversation context
        async def set_conversation_context(conversation_id):
            old_id = getattr(self, '_current_conversation_id', None)
            self._current_conversation_id = conversation_id
            return old_id

        # Wrap the task execution method to set the context
        original_execute = self._execute_crew_task_async
        async def execute_with_context(conversation_id, task_description):
            old_id = await set_conversation_context(conversation_id)
            try:
                await original_execute(conversation_id, task_description)
            finally:
                self._current_conversation_id = old_id

        self._execute_crew_task_async = execute_with_context

        try:
            await self.app.run_task(host=host, port=port, debug=debug, **kwargs)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

    def get_app(self):
        """Get the Quart app object"""
        return self.app

    # Synchronous launch method (for compatibility)
    def launch(self, host="127.0.0.1", port=5000, debug=False, **kwargs):
        """Launch the server (synchronous wrapper)"""
        asyncio.run(self.launch_async(host, port, debug, **kwargs))
