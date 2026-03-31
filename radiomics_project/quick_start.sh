#!/bin/bash
# 快速启动脚本

echo "========================================"
echo "影像组学项目 - 快速启动"
echo "========================================"
echo ""

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo ""
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "虚拟环境创建完成"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo ""
echo "安装依赖..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "环境设置完成！"
echo "========================================"
echo ""
echo "使用方法:"
echo "  1. 下载数据: python scripts/download_datasets.py --dataset sample"
echo "  2. 运行流程: python main.py --step 0"
echo "  3. 查看帮助: python main.py --help"
echo ""
echo "项目结构:"
echo "  - config/     配置文件"
echo "  - data/       数据目录"
echo "  - src/        源代码"
echo "  - results/    结果输出"
echo ""
