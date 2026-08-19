import argparse
from pathlib import Path

import yaml

from vasp_crew.server.flask_server.flask_server import FlaskCrewServer
def main():
    """主函数 - 命令行启动入口"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent        # 项目根目录
    
    parser = argparse.ArgumentParser(description="启动CrewAI VASP Flask服务器")
    parser.add_argument("--config", default=f"{project_root}/configs/config.yaml", help="配置文件路径")
    parser.add_argument("--host", default="127.0.0.1", help="服务器地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--work-dir", default=f"{project_root}/crew_workspaces", help="工作目录")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    
    args = parser.parse_args()
    
    # 处理路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / args.config
    
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = current_dir / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = str(work_dir.absolute())
    
    # 加载配置
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, "r", encoding='utf-8') as f:
        crew_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建并启动服务器
    server = FlaskCrewServer(
        crew_config=crew_config,
        title="Flask Crew AI Server",
        work_dir="./work",
        db_path="./crew_tasks.db",
        allow_path=work_dir
    )
    
    server.launch(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
