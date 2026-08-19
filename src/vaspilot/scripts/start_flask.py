import argparse
from pathlib import Path
import os
import yaml

from ..server.flask_server.flask_server import FlaskCrewServer
def start_flask():
    """Main function - command-line entry point"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent        # project root directory

    parser = argparse.ArgumentParser(description="Start the CrewAI VASP Flask server")
    parser.add_argument("--config", default=f"{project_root}/configs/crew_config.yaml", help="Path to the config file")
    parser.add_argument("--host", default="0.0.0.0", help="Server address")
    parser.add_argument("--port", type=int, default=51293, help="Server port")
    parser.add_argument("--work-dir", default=os.getcwd(), help="Working directory")
    parser.add_argument("--allow-path", default=os.getcwd(), help="Directory allowed for access")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if not args.config:
        print(f"❌ Please set the config file path with --config")

    # Resolve paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = os.path.abspath(config_path)

    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Config file does not exist: {config_path}")
        return

    with open(config_path, "r", encoding='utf-8') as f:
        crew_config = yaml.load(f, Loader=yaml.FullLoader)

    # Create and launch the server
    server = FlaskCrewServer(
        crew_config=crew_config,
        title="VASPilot Web Server",
        work_dir=work_dir,
        db_path=f"{work_dir}/crew_tasks.db",
        allow_path=args.allow_path
    )
    
    server.launch(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    start_flask()
