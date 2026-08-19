#!/bin/bash

# Flask Crew AI Server 快速启动脚本

echo "🚀 启动 Flask Crew AI Server (测试版)"
echo "========================================"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装或不在PATH中"
    exit 1
fi

# 进入正确目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "📂 当前目录: $SCRIPT_DIR"
echo "🔄 启动服务器..."
echo ""

# 启动Flask服务器
python3 run_flask_server.py "$@" 