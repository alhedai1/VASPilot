import argparse
from pathlib import Path
import os

from ..tools.mcp.mcp_server import main as mcp_main

def start_mcp():
    """Main function - command-line entry point"""
    parser = argparse.ArgumentParser(description="Start the VASP MCP server")
    parser.add_argument("--config", help="Path to the config file")
    parser.add_argument("--port", type=int, default=8933, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server address")
    parser.add_argument("--work-dir", default=f".", help="Working directory")
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

    # Check the config file (if needed)
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ Config file does not exist: {config_path}")
        return

    print(f"🚀 Starting VASP MCP server...")
    print(f"📁 Working directory: {work_dir}")

    # Start the MCP server
    mcp_main(config_path=config_path, port=args.port, host=args.host)


if __name__ == "__main__":
    start_mcp()
