#!/usr/bin/env python3
"""
CrewAI VASP Flask server
Features: task submission, history, detail view, live updates
Implemented on top of the CrewServer base class, with templates kept separate
"""

import os
import json
import uuid
import sqlite3
import threading
import re
from datetime import datetime
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, g

import ctypes
from werkzeug.utils import secure_filename

current_dir = Path(__file__).parent  # flask_server/

from ...listener.server_listener import CrewServer, ServerListener
from ...crew import VaspCrew
from crewai import Task
from fastmcp.client import Client


class FlaskCrewServer(CrewServer):
    """CrewServer implementation based on Flask"""

    def __init__(self, crew_config: Dict[str, Any], title: str = "VASPilot Server",
                 work_dir: str = ".", db_path: Optional[str] = None, allow_path: Optional[str] = None):
        super().__init__()
        self.title = title
        self.config = crew_config
        self.work_dir = os.path.abspath(work_dir)
        self.running_tasks = {}
        self._current_conversation_id: Optional[str] = None
        self.allow_path = allow_path
        self._stop_flags = {}  # marks tasks that need to be stopped

        # Database path
        if db_path is None:
            db_path = os.path.join(work_dir, 'crew_tasks.db')
        self.db_path = os.path.abspath(db_path)

        # Create the Flask app
        template_folder = str(current_dir / "templates")
        self.app = Flask(__name__, template_folder=template_folder)
        self.app.secret_key = 'crew-ai-flask-server'

        # Upload directory
        self.upload_dir = os.path.join(self.work_dir, 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)

        self.generator = VaspCrew(self.config)

        self.current_logger = ServerListener(self, None)
        # Initialize the database
        self._init_db()

        # Set up routes
        self._setup_routes()

    def _raise_exception_in_thread(self, thread: threading.Thread, exception_type=SystemExit) -> bool:
        """Asynchronously inject an exception into the target thread.
        Returns whether the injection succeeded.
        """
        tid = getattr(thread, "ident", None)
        if not tid:
            return False
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exception_type))
        if res == 0:
            return False
        if res > 1:
            # Roll back and report failure
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
            return False
        return True

    def _init_db(self):
        """Initialize the database"""
        try:
            # Ensure the database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
                print(f"📁 Created database directory: {db_dir}")

            print(f"🗄️ Initializing database: {self.db_path}")

            with sqlite3.connect(self.db_path) as conn:
                # Create the task_executions table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS task_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT UNIQUE NOT NULL,
                        task_description TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        result TEXT,
                        error_message TEXT
                    )
                ''')

                # Create the activity_logs table
                conn.execute('''
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

                # Check whether the role_name column needs to be added (backward compatibility)
                cursor = conn.execute("PRAGMA table_info(activity_logs)")
                columns = [column[1] for column in cursor.fetchall()]
                if 'role_name' not in columns:
                    print("🔄 Adding role_name column to activity_logs table")
                    conn.execute('ALTER TABLE activity_logs ADD COLUMN role_name TEXT')

                conn.commit()

                # Verify the tables were created successfully
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
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

    def _get_db(self):
        """Get a database connection"""
        db = getattr(g, '_database', None)
        if db is None:
            try:
                db = g._database = sqlite3.connect(self.db_path)
                db.row_factory = sqlite3.Row

                # Verify the table exists
                cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_executions'")
                if not cursor.fetchone():
                    # If the table does not exist, reinitialize the database
                    print("⚠️ Table not found, reinitializing database...")
                    db.close()
                    self._init_db()
                    db = g._database = sqlite3.connect(self.db_path)
                    db.row_factory = sqlite3.Row

            except Exception as e:
                print(f"❌ Database connection failed: {str(e)}")
                raise
        return db

    def _close_connection(self, exception):
        """Close the database connection"""
        db = getattr(g, '_database', None)
        if db is not None:
            db.close()

    def _get_recent_tasks(self, limit=10):
        """Get the most recent tasks"""
        db = self._get_db()
        cursor = db.execute(
            'SELECT * FROM task_executions ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        return cursor.fetchall()

    def _get_task_by_id(self, conversation_id):
        """Get a task by ID"""
        db = self._get_db()
        cursor = db.execute(
            'SELECT * FROM task_executions WHERE conversation_id = ?',
            (conversation_id,)
        )
        return cursor.fetchone()

    def _get_task_logs(self, conversation_id):
        """Get task logs"""
        db = self._get_db()
        cursor = db.execute(
            'SELECT * FROM activity_logs WHERE conversation_id = ? ORDER BY timestamp',
            (conversation_id,)
        )
        logs = cursor.fetchall()

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

            # Safely access the role_name field (compatible with old data)
            try:
                role_name = log['role_name'] if 'role_name' in log.keys() else None
            except (KeyError, TypeError):
                role_name = None
            
            formatted_logs.append({
                'type': log['type'],
                'type_name': type_names.get(log['type'], log['type']),
                'role_name': role_name,
                'content': log['content'],
                'timestamp': log['timestamp'],
                'preview': log['content'][:30] + '...' if len(log['content']) > 30 else log['content']
            })
        
        return formatted_logs

    def _setup_routes(self):
        """Set up Flask routes"""

        @self.app.teardown_appcontext
        def close_connection(exception):
            self._close_connection(exception)

        @self.app.route('/')
        def index():
            """Home page"""
            recent_tasks = self._get_recent_tasks()
            return render_template('base.html',
                                 title=self.title,
                                 recent_tasks=recent_tasks)

        @self.app.route('/upload', methods=['POST'])
        def upload_structure():
            """Upload a crystal structure file, returning the saved absolute path"""
            try:
                if 'file' not in request.files:
                    return jsonify({'error': 'File field not found'}), 400
                file = request.files['file']
                if not file or file.filename == '':
                    return jsonify({'error': 'No file uploaded'}), 400

                filename = secure_filename(file.filename)
                lower_name = filename.lower()
                if not (lower_name.endswith(('.vasp', '.cif', '.xyz')) or lower_name in ('poscar', 'contcar')):
                    return jsonify({'error': 'File type not supported. Only .vasp/.cif/.xyz or POSCAR/CONTCAR are allowed'}), 400

                unique_name = f"{uuid.uuid4().hex}_{filename}"
                save_path = os.path.join(self.upload_dir, unique_name)
                file.save(save_path)
                abs_path = os.path.abspath(save_path)
                return jsonify({'success': True, 'path': abs_path, 'filename': filename})
            except Exception as e:
                return jsonify({'error': f'Upload failed: {str(e)}'}), 500

        @self.app.route('/submit', methods=['POST'])
        def submit_task():
            """Submit a task"""
            try:
                data = request.get_json()
                task_description = data.get('task_description', '').strip()

                if not task_description:
                    return jsonify({'error': 'Please enter a valid task description'}), 400

                # Check whether a task is already running
                db = self._get_db()
                cursor = db.execute("SELECT COUNT(*) as count FROM task_executions WHERE status = 'running'")
                running_count = cursor.fetchone()['count']

                if running_count > 0:
                    return jsonify({'error': 'There is a task running, please wait for it to complete before submitting a new task'}), 400

                # Create the task record
                conversation_id = str(uuid.uuid4())
                db.execute(
                    'INSERT INTO task_executions (conversation_id, task_description) VALUES (?, ?)',
                    (conversation_id, task_description)
                )
                db.commit()

                # Start the background task
                thread = threading.Thread(
                    target=self._execute_crew_task,
                    args=(conversation_id, task_description),
                    daemon=True
                )
                thread.start()
                self.running_tasks[conversation_id] = thread
                
                return jsonify({
                    'success': True,
                    'conversation_id': conversation_id,
                    'message': 'Task submitted, starting execution'
                })
                
            except Exception as e:
                return jsonify({'error': f'Server error: {str(e)}'}), 500

        @self.app.route('/task/<conversation_id>')
        def task_detail(conversation_id):
            """Task detail page"""
            task = self._get_task_by_id(conversation_id)
            if not task:
                return "Task not found", 404
            
            logs = self._get_task_logs(conversation_id)
            recent_tasks = self._get_recent_tasks()
            
            return render_template('task_detail.html',
                                 title=self.title,
                                 task=task,
                                 logs=logs,
                                 recent_tasks=recent_tasks)

        @self.app.route('/api/task/<conversation_id>/status')
        def get_task_status(conversation_id):
            """API to get task status"""
            task = self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404
            
            return jsonify({
                'status': task['status'],
                'conversation_id': task['conversation_id'],
                'task_description': task['task_description']
            })

        @self.app.route('/api/task/<conversation_id>/logs')
        def get_task_logs(conversation_id):
            """API to get task logs"""
            task = self._get_task_by_id(conversation_id)
            if not task:
                return jsonify({'error': 'Task not found'}), 404

            logs = self._get_task_logs(conversation_id)

            # Convert logs into dict format
            logs_data = []
            for log in logs:
                logs_data.append({
                    'type': log['type'],
                    'type_name': log['type_name'],
                    'role_name': log['role_name'],  # log is already a dict from formatted_logs, so this can be accessed directly
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
        def get_tasks():
            """API to get the task list"""
            try:
                recent_tasks = self._get_recent_tasks()
                tasks_data = []
                for task in recent_tasks:
                    tasks_data.append({
                        'conversation_id': task['conversation_id'],
                        'task_description': task['task_description'],
                        'status': task['status'],
                        'created_at': task['created_at'],
                        'started_at': task['started_at'],
                        'completed_at': task['completed_at']
                    })
                return jsonify(tasks_data)
            except Exception as e:
                return jsonify({'error': f'Failed to get task list: {str(e)}'}), 500

        @self.app.route('/api/files/<conversation_id>/<path:filename>')
        def serve_task_file(conversation_id, filename):
            """Serve file access for a specific task"""
            import os
            from flask import send_file, abort
            from urllib.parse import unquote

            try:

                # Decode the path segment by segment: split the path, decode each segment, then rejoin
                path_segments = filename.split('/')
                decoded_segments = [unquote(segment) for segment in path_segments]
                decoded_filename = '/'.join(decoded_segments)


                # Check whether the absolute-path marker is present
                is_absolute_path = False
                if decoded_filename.startswith('__ABS__'):
                    # Strip the marker to recover the absolute path
                    decoded_filename = decoded_filename[7:]  # strip '__ABS__'
                    is_absolute_path = True

                # Build the file path
                task_dir = os.path.join(self.work_dir, conversation_id)

                # If it's an absolute path, use it directly
                if is_absolute_path or (decoded_filename.startswith('/') and self.allow_path):
                    file_path = decoded_filename
                else:
                    file_path = os.path.join(task_dir, decoded_filename)

                # Safety check: ensure the file is within the task directory
                file_path = os.path.abspath(file_path)
                task_dir = os.path.abspath(task_dir)

                # Safety check: for absolute paths, allow access unless explicitly disallowed
                if not is_absolute_path and not self.allow_path:
                    if not file_path.startswith(task_dir) and not file_path.startswith(self.work_dir):
                        abort(403, description="Access denied: file path not in allowed range")
                elif is_absolute_path:
                    print(f"[DEBUG] Absolute path access allowed")

                # Check whether the file exists
                if not os.path.exists(file_path):
                    abort(404, description=f"File not found: {decoded_filename}")

                # Set the MIME type based on the file extension
                if decoded_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    mimetype = 'image/png' if decoded_filename.lower().endswith('.png') else 'image/jpeg'
                elif decoded_filename.lower().endswith(('.vasp', '.xyz', '.cif')):
                    mimetype = 'text/plain'
                else:
                    mimetype = 'application/octet-stream'
                
                
                return send_file(file_path, mimetype=mimetype)
                
            except Exception as e:
                abort(500, description=f"File service error: {str(e)}")

        @self.app.route('/api/files/<conversation_id>/list')
        def list_task_files(conversation_id):
            """List all files in the task directory"""
            import os
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
                        
                        # Encode the path segment by segment: split the path, encode each segment, then rejoin
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
        def stop_task(conversation_id):
            """API to stop a task"""
            try:
                # Check whether the task exists
                task = self._get_task_by_id(conversation_id)
                if not task:
                    return jsonify({'error': 'Task not found'}), 404

                # Check the task status
                if task['status'] not in ['running', 'pending']:
                    return jsonify({'error': f'Task status is {task["status"]}, cannot stop'}), 400

                self.system_log(f"Starting to stop task: {conversation_id}")

                # 1. Stop the crew process
                crew_stopped = self._stop_crew_process(conversation_id)

                # 2. Extract calculation task IDs from the logs
                calc_ids = self._extract_calc_ids_from_logs(conversation_id)
                self.system_log(f"Extracted calculation IDs: {calc_ids}")

                # 3. Cancel the related SLURM jobs
                cancel_results = {}
                if calc_ids:
                    try:
                        cancel_results = asyncio.run(self._cancel_slurm_job(calc_ids))
                        self.system_log(f"SLURM job cancellation result: {cancel_results}")
                    except Exception as e:
                        self.system_log(f"Error while cancelling SLURM job: {str(e)}")
                        cancel_results = {"error": str(e)}

                # 4. Update the database status
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                        ('cancelled', 'Task manually stopped by user', conversation_id)
                    )
                    conn.commit()

                # 5. Clean up the running task record
                if conversation_id in self.running_tasks:
                    del self.running_tasks[conversation_id]

                self.system_log(f"Task {conversation_id} has been stopped")

                return jsonify({
                    'success': True,
                    'message': 'Task has been stopped',
                    'details': {
                        'crew_stopped': crew_stopped,
                        'calc_ids_found': len(calc_ids),
                        'calc_ids': calc_ids,
                        'slurm_cancel_results': cancel_results
                    }
                })

            except Exception as e:
                error_msg = f'Error while stopping task: {str(e)}'
                self.system_log(error_msg)
                return jsonify({'error': error_msg}), 500

    def _extract_calc_ids_from_logs(self, conversation_id):
        """Extract calculation task IDs from the task logs"""
        calc_ids = []
        try:
            logs = self._get_task_logs(conversation_id)
            
            for log in logs:
                content = log['content']

                # Look for calculation_id in tool_output
                if log['type'] == 'tool_output':
                    try:
                        # Try to parse the JSON content
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
                        # If JSON parsing fails, fall back to regex matching
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

    def _stop_crew_process(self, conversation_id):
        """Stop the crew process"""
        try:
            if conversation_id in self.running_tasks:
                thread = self.running_tasks[conversation_id]
                if thread.is_alive():
                    self.system_log(f"Attempting to terminate task thread: {conversation_id}")
                    stopped = self._raise_exception_in_thread(thread, SystemExit)
                    if not stopped:
                        self.system_log(f"Unable to inject exception into task thread, marking task as stopped: {conversation_id}")
                        return False
                    # Wait for the thread to exit
                    thread.join(timeout=5)
                    if thread.is_alive():
                        self.system_log(f"Task thread did not exit within the timeout: {conversation_id}")
                        return False
                    self.system_log(f"Task {conversation_id} thread has terminated")
                    return True
                else:
                    self.system_log(f"Task {conversation_id} has already stopped")
                    return True
            else:
                self.system_log(f"No running task found for {conversation_id}")
                return False
        except Exception as e:
            self.system_log(f"Error while stopping crew process: {str(e)}")
            return False

    def _execute_crew_task(self, conversation_id, task_description):
        """Execute a crew task"""
        try:
            # Update the task status
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE task_executions SET status = ?, started_at = CURRENT_TIMESTAMP WHERE conversation_id = ?',
                    ('running', conversation_id)
                )
                conn.commit()

            # System log
            self.system_log(f"Conversation id: {conversation_id}")

            # Create the working directory
            local_dir = os.path.join(self.work_dir, conversation_id)
            os.makedirs(local_dir, exist_ok=True)
            os.chdir(local_dir)

            self.system_log("Initializing crew...")
            crew = self.generator.crew(local_dir)
            self.system_log("Setting up listener...")
            self.current_logger.crew_fingerprint = crew.fingerprint.uuid_str
            self.system_log("Creating user task...")

            # Create the task
            task = Task(
                description=task_description,
                expected_output="A detailed report including the task execution process, calculation results, and the locations of the generated plots.",
                output_file=f'crew_output_{uuid.uuid4().hex[:8]}.md',
            )

            crew.tasks = [task]

            self.system_log("Starting task execution...")
            # Execute the crew
            result = crew.kickoff()

            self.system_log("Task complete!")
            self.agent_output("FinalResult", str(result))

            # Update the task status
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, result = ? WHERE conversation_id = ?',
                    ('completed', str(result), conversation_id)
                )
                conn.commit()


        except Exception as e:
            error_msg = f"An error occurred during execution: {str(e)}"

            # Log the error
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    'UPDATE task_executions SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ? WHERE conversation_id = ?',
                    ('failed', error_msg, conversation_id)
                )
                conn.commit()

            self.system_log(error_msg)
        finally:
            # Clean up the running task record
            if conversation_id in self.running_tasks:
                del self.running_tasks[conversation_id]
            self.system_log("Task execution complete!")

    # CrewServer interface implementation
    def system_log(self, message: str, crew_fingerprint: str = None):
        """Implements the system log method"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Get the current conversation ID (the Flask version continues to use the original approach, since it handles a single task)
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'system', log_entry, role_name='system')

    def agent_input(self, agent_role: str, message: str, crew_fingerprint: str = None):
        """Implements the agent input method"""
        log_content = f"[{agent_role}] {message}"
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'agent_input', log_content, role_name=agent_role)

    def agent_output(self, agent_role: str, message: str, crew_fingerprint: str = None):
        """Implements the agent output method"""
        log_content = f"[{agent_role}] {message}"
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'agent_output', log_content, role_name=agent_role)

    def tool_input(self, tool_name: str, message: Any, crew_fingerprint: str = None):
        """Implements the tool input method"""
        if isinstance(message, (dict, list)):
            log_content = json.dumps(message, ensure_ascii=False)
        else:
            try:
                parsed = json.loads(str(message))
                log_content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                log_content = json.dumps({"raw": str(message)}, ensure_ascii=False)
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'tool_input', log_content, role_name=tool_name)

    def tool_output(self, tool_name: str, message: Any, crew_fingerprint: str = None):
        """Implements the tool output method"""
        if isinstance(message, (dict, list)):
            log_content = json.dumps(message, ensure_ascii=False)
        else:
            try:
                parsed = json.loads(str(message))
                log_content = json.dumps(parsed, ensure_ascii=False)
            except Exception:
                log_content = json.dumps({"raw": str(message)}, ensure_ascii=False)
        current_conversation_id = getattr(self, '_current_conversation_id', None)
        if current_conversation_id:
            self._log_to_db(current_conversation_id, 'tool_output', log_content, role_name=tool_name)

    def _log_to_db(self, conversation_id, log_type, content, role_name=None):
        """Write a log entry to the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO activity_logs (conversation_id, type, role_name, content) VALUES (?, ?, ?, ?)',
                (conversation_id, log_type, role_name, content)
            )
            conn.commit()

    async def _cancel_slurm_job(self, calc_ids: list[str]):
        async with Client(self.config["mcp_server"]["url"]) as client:
            # call tool
            tool_result = await client.call_tool("cancel_slurm_job", {"calc_ids": calc_ids})
        if tool_result.data is None:
            return {"error": "No result from check_calculation_status"}
        else:
            return tool_result.data

    def launch(self, host="127.0.0.1", port=5000, debug=False, **kwargs):
        """Launch the Flask application"""
        print(f"🚀 Starting {self.title}...")
        print(f"💼 Working directory: {self.work_dir}")
        print(f"🗄️ Database: {self.db_path}")
        print(f"🌐 Server address: http://{host}:{port}")
        print("=" * 50)
        print("✨ Flask Crew AI Server")
        print("📝 Submit tasks, 📋 view history, 🔍 live updates")
        print("=" * 50)

        # Set the conversation ID context while a task is executing
        def set_conversation_context(conversation_id):
            def wrapper(func):
                def inner(*args, **kwargs):
                    old_id = getattr(self, '_current_conversation_id', None)
                    self._current_conversation_id = conversation_id
                    try:
                        return func(*args, **kwargs)
                    finally:
                        self._current_conversation_id = old_id
                return inner
            return wrapper

        # Wrap the task execution method to set the context
        original_execute = self._execute_crew_task
        def execute_with_context(conversation_id, task_description):
            self._current_conversation_id = conversation_id
            try:
                original_execute(conversation_id, task_description)
            finally:
                self._current_conversation_id = None
        
        self._execute_crew_task = execute_with_context

        try:
            self.app.run(host=host, port=port, debug=debug, threaded=True, **kwargs)
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")

    def get_app(self):
        """Get the Flask application object"""
        return self.app
