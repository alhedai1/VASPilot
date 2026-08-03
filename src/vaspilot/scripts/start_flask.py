import argparse
from pathlib import Path
import os
import yaml

from ..server.flask_server.flask_server import FlaskCrewServer
def start_flask():
    """主函数 - 命令行启动入口"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent        # 项目根目录
    
    parser = argparse.ArgumentParser(description="启动CrewAI VASP Flask服务器")
    parser.add_argument("--config", default=f"{project_root}/configs/crew_config.yaml", help="配置文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=51293, help="服务器端口")
    parser.add_argument("--work-dir", default=os.getcwd(), help="工作目录")
    parser.add_argument("--allow-path", default=os.getcwd(), help="允许访问的目录")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    
    args = parser.parse_args()
    
    if not args.config:
        print(f"❌ 请用--config设置配置文件路径")

    # 处理路径
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = os.path.abspath(config_path)
    
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    
    # 加载配置
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, "r", encoding='utf-8') as f:
        crew_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 创建并启动服务器
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
