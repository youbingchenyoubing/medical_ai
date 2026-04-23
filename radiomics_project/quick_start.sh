#!/bin/bash

echo "========================================"
echo "影像组学项目 - 快速启动"
echo "========================================"
echo ""

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

if [ ! -d "venv" ]; then
    echo ""
    echo "创建虚拟环境..."
    python3 -m venv venv
    echo "虚拟环境创建完成"
fi

echo ""
echo "激活虚拟环境..."
source venv/bin/activate

echo ""
echo "Step 1/2: 安装基础依赖 (numpy等)..."
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.11.0 matplotlib>=3.4.0 \
    seaborn>=0.11.0 pyyaml>=6.0 tqdm>=4.62.0 joblib>=1.0.0 \
    SimpleITK>=2.3.0 scikit-learn>=1.3.0 \
    xgboost>=1.7.0 lightgbm>=3.3.0 catboost>=1.0.0 \
    shap>=0.40.0

echo ""
echo "Step 2/2: 安装 pyradiomics (需要 numpy 预装)..."
pip install --no-build-isolation pyradiomics>=3.0.0

echo ""
echo "Step 3/3: 安装深度学习框架..."
pip install torch>=2.0.0 monai>=1.3.0

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
