#!/bin/bash
# 家具喷漆 Web 服务 — 一键启动脚本
# 用法: bash run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3, 请先安装 Python 3.8+"
    exit 1
fi

# 安装依赖 (首次运行)
if [ ! -f ".deps_installed" ]; then
    echo "=== 安装 Python 依赖 ==="
    pip install -r requirements.txt
    touch .deps_installed
    echo ""
fi

# 添加项目根目录到 PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "========================================="
echo "  雅格美天 喷漆机器人 — Web 服务"
echo "  地址: http://localhost:5000"
echo "  按 Ctrl+C 退出"
echo "========================================="
echo ""

# 启动服务
python3 webapp/app.py
