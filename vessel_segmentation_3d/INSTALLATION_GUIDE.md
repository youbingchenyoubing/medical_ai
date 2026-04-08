# 安装和运行指南

## 📦 安装步骤

### 方法1：使用conda（推荐）

```bash
# 1. 创建虚拟环境
conda create -n vessel_seg python=3.8 -y
conda activate vessel_seg

# 2. 安装PyTorch（根据你的CUDA版本选择）
# CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# CUDA 11.7
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# CPU版本
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 方法2：使用pip

```bash
# 1. 创建虚拟环境
python -m venv vessel_seg_env

# Windows激活
vessel_seg_env\Scripts\activate

# Linux/Mac激活
source vessel_seg_env/bin/activate

# 2. 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖
pip install -r requirements.txt
```

### 方法3：使用Docker

```bash
# 构建镜像
docker build -t vessel-seg:latest .

# 运行容器
docker run -it --rm --gpus all \
    -v /path/to/data:/data \
    -v /path/to/results:/results \
    vessel-seg:latest
```

## ✅ 验证安装

```bash
# 运行测试脚本
python test_modules.py
```

预期输出：
```
测试结果汇总
======================================================================
模块导入            : ✓ 通过
模型测试            : ✓ 通过
骨架化测试          : ✓ 通过
拓扑分析测试        : ✓ 通过
曲率计算测试        : ✓ 通过
扭率计算测试        : ✓ 通过
分支密度测试        : ✓ 通过
特征提取器测试      : ✓ 通过

总计: 8/8 测试通过
======================================================================

🎉 所有测试通过！系统运行正常。
```

## 🚀 快速开始

### 1. 运行示例

```bash
# 运行快速开始示例
python quick_start.py
```

这将演示所有核心功能。

### 2. 处理单个病例

```bash
# 使用命令行
python pipeline.py \
    --image data/case001.nii.gz \
    --case-id case001 \
    --output-dir results/ \
    --device cuda

# 或使用Python脚本
python -c "
from vessel_segmentation_3d import VesselSegmentationReconstructionPipeline

config = {
    'output_dir': 'results/',
    'spacing': [1.0, 1.0, 1.0]
}

pipeline = VesselSegmentationReconstructionPipeline(config)
features = pipeline.run_pipeline('data/case001.nii.gz', 'case001')

print('提取的特征:')
for key, value in features.items():
    print(f'{key}: {value}')
"
```

### 3. 批量处理

```python
# batch_process.py
from vessel_segmentation_3d import VesselSegmentationReconstructionPipeline
from pathlib import Path
import pandas as pd

# 配置
config = {
    'output_dir': 'results/',
    'segmentation': {'model_path': 'model.pth'}
}

# 初始化
pipeline = VesselSegmentationReconstructionPipeline(config)

# 病例列表
data_dir = Path('data/')
cases = list(data_dir.glob('*.nii.gz'))

# 批量处理
all_features = []

for case_path in cases:
    case_id = case_path.stem
    
    print(f"\n处理病例: {case_id}")
    
    try:
        features = pipeline.run_pipeline(str(case_path), case_id)
        all_features.append(features)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        continue

# 保存所有特征
df = pd.DataFrame(all_features)
df.to_csv('all_features.csv', index=False)

print(f"\n完成！共处理 {len(all_features)} 个病例")
```

## 📁 数据准备

### 数据格式要求

```
data/
├── case001.nii.gz          # NIfTI格式
├── case002.nii.gz
└── case003.nii.gz

或

data/
├── case001/
│   ├── 001.dcm            # DICOM系列
│   ├── 002.dcm
│   └── ...
├── case002/
│   └── ...
```

### 推荐参数

| 参数 | CT | MRI |
|------|----|----|
| 层厚 | ≤3mm | ≤3mm |
| 矩阵 | ≥512×512 | ≥256×256 |
| 对比剂 | 增强 | T1增强 |
| 序列 | 静脉期 | T1+C |

## ⚙️ 配置说明

### 基本配置

```python
config = {
    # 输出目录
    'output_dir': 'results/',
    
    # 体素间距 (mm)
    'spacing': [1.0, 1.0, 1.0],
    
    # 预处理参数
    'preprocessing': {
        'target_spacing': [1.0, 1.0, 1.0],  # 重采样目标间距
        'window_level': -600,               # CT窗位
        'window_width': 1500                # CT窗宽
    },
    
    # 分割参数
    'segmentation': {
        'model_path': 'model.pth',          # 模型路径
        'tumor_threshold': 0.5,             # 肿瘤分割阈值
        'vessel_threshold': 0.5             # 血管分割阈值
    }
}
```

### 高级配置

```python
config = {
    # ... 基本配置 ...
    
    # 骨架化参数
    'skeletonization': {
        'method': 'lee',                    # 骨架化方法
        'min_branch_length': 5              # 最小分支长度
    },
    
    # 形态量化参数
    'morphometry': {
        'smooth': 0.1,                      # 曲线平滑参数
        'radius_range': [1.0, 10.0]         # 半径范围 (mm)
    }
}
```

## 🔧 常见问题

### Q1: CUDA内存不足

```python
# 解决方案1: 减小batch size
config['segmentation']['batch_size'] = 1

# 解决方案2: 使用CPU
pipeline = VesselSegmentationReconstructionPipeline(config, device='cpu')

# 解决方案3: 使用混合精度
config['segmentation']['use_amp'] = True
```

### Q2: 找不到模型文件

```python
# 解决方案1: 指定正确的模型路径
config['segmentation']['model_path'] = '/path/to/model.pth'

# 解决方案2: 使用预训练模型（如果可用）
# 从项目Release页面下载

# 解决方案3: 训练自己的模型
# 参见训练脚本
```

### Q3: 处理速度慢

```python
# 解决方案1: 使用GPU
pipeline = VesselSegmentationReconstructionPipeline(config, device='cuda')

# 解决方案2: 调整预处理参数
config['preprocessing']['target_spacing'] = [2.0, 2.0, 2.0]  # 降低分辨率

# 解决方案3: 并行处理
from multiprocessing import Pool

def process_case(case_path):
    pipeline = VesselSegmentationReconstructionPipeline(config)
    return pipeline.run_pipeline(case_path, Path(case_path).stem)

with Pool(4) as p:  # 4个进程
    results = p.map(process_case, case_list)
```

### Q4: Windows路径问题

```python
# 使用原始字符串
image_path = r'd:\data\case001.nii.gz'

# 或使用Path
from pathlib import Path
image_path = Path('d:/data/case001.nii.gz')
```

## 📊 输出说明

### 文件结构

```
results/
├── segmentations/
│   ├── case001_tumor.nii.gz      # 肿瘤分割结果
│   └── case001_vessel.nii.gz     # 血管分割结果
├── skeletons/
│   └── case001_skeleton.nii.gz   # 血管骨架
├── features/
│   ├── case001_features.csv      # 单病例特征
│   └── all_features.csv          # 所有病例特征汇总
└── logs/
    └── case001.log               # 处理日志
```

### 特征CSV格式

```csv
case_id,curvature_mean,curvature_max,torsion_mean,branching_density,...
case001,0.1234,0.5678,0.0234,0.001234,...
case002,0.1456,0.6234,0.0345,0.001567,...
```

## 🎓 进阶使用

### 自定义模型

```python
from vessel_segmentation_3d.models.unet3d import UNet3D

# 创建自定义模型
model = UNet3D(
    in_channels=1,
    num_classes=3,
    base_channels=64,    # 调整通道数
    dropout=0.2          # 调整dropout
)

# 使用自定义模型
config['segmentation']['custom_model'] = model
```

### 特征选择

```python
from vessel_segmentation_3d.morphometry.feature_extractor import VesselMorphometryExtractor

# 选择特定特征
selected_features = [
    'curvature_mean',
    'torsion_mean',
    'branching_density',
    'vessel_density'
]

extractor = VesselMorphometryExtractor()
features = extractor.extract_all_features(...)

# 筛选特征
filtered_features = {k: v for k, v in features.items() 
                     if k in selected_features}
```

### 可视化

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 可视化骨架
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(skeleton_points[:, 0], 
           skeleton_points[:, 1], 
           skeleton_points[:, 2],
           c='blue', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Vessel Skeleton')
plt.savefig('skeleton_3d.png')
plt.show()
```

## 📞 获取帮助

- **文档**: 查看 `README.md` 和 `PROJECT_SUMMARY.md`
- **示例**: 运行 `quick_start.py`
- **问题**: 在GitHub Issues提交问题
- **邮件**: your.email@example.com

---

**祝您使用愉快！** 🎉
