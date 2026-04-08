# 血管自动化分割与三维重建系统

## 项目简介

本项目实现了完整的血管自动化分割与三维重建流程，包括：

- **肿瘤与血管协同分割**：基于3D U-Net/nnU-Net的深度学习分割
- **血管骨架化**：提取血管中轴线
- **拓扑结构分析**：识别分叉点、端点、分支
- **三维形态量化**：计算曲率、扭率、分支密度等特征
- **端到端流程**：从原始影像到定量特征的完整pipeline

## 项目结构

```
vessel_segmentation_3d/
├── __init__.py                   # 主模块初始化
├── pipeline.py                   # 主流程
├── models/                       # 模型定义
│   ├── unet3d.py                # 3D U-Net模型
│   └── nnunet.py                # nnU-Net模型
├── segmentation/                 # 分割模块
│   ├── tumor_vessel_seg.py      # 肿瘤血管分割
│   └── inference.py             # 推理脚本
├── skeletonization/              # 骨架化模块
│   ├── morphological.py         # 形态学骨架化
│   ├── distance_transform.py    # 距离变换方法
│   └── topology_analysis.py     # 拓扑分析
├── morphometry/                  # 形态量化模块
│   ├── curvature.py             # 曲率计算
│   ├── torsion.py               # 扭率计算
│   ├── branching.py             # 分支密度计算
│   └── feature_extractor.py     # 综合特征提取
├── preprocessing/                # 预处理模块
│   └── image_preprocessing.py   # 图像预处理
├── utils/                        # 工具函数
│   ├── visualization.py         # 可视化
│   └── io_utils.py              # 输入输出
├── README.md                     # 说明文档
└── requirements.txt              # 依赖包
```

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/vessel_segmentation_3d.git
cd vessel_segmentation_3d
```

### 2. 创建虚拟环境

```bash
conda create -n vessel_seg python=3.8
conda activate vessel_seg
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本使用

```python
from vessel_segmentation_3d import VesselSegmentationReconstructionPipeline

# 配置
config = {
    'output_dir': 'results/',
    'spacing': [1.0, 1.0, 1.0],
    'preprocessing': {
        'target_spacing': [1.0, 1.0, 1.0],
        'window_level': -600,
        'window_width': 1500
    },
    'segmentation': {
        'model_path': 'path/to/model.pth'
    }
}

# 初始化流程
pipeline = VesselSegmentationReconstructionPipeline(config)

# 运行
features = pipeline.run_pipeline(
    image_path='data/case001.nii.gz',
    case_id='case001'
)

# 查看特征
for key, value in features.items():
    print(f"{key}: {value}")
```

### 命令行使用

```bash
python pipeline.py \
    --image data/case001.nii.gz \
    --case-id case001 \
    --output-dir results/ \
    --device cuda
```

## 核心功能

### 1. 3D U-Net分割模型

```python
from vessel_segmentation_3d.models.unet3d import UNet3D

# 创建模型
model = UNet3D(
    in_channels=1,      # 单通道CT/MRI
    num_classes=3,      # 背景、肿瘤、血管
    base_channels=64
)

# 前向传播
import torch
x = torch.randn(2, 1, 64, 128, 128)
output = model(x)
print(output.shape)  # torch.Size([2, 3, 64, 128, 128])
```

### 8. 放射组学特征提取

```python
from vessel_segmentation_3d.morphometry.radiomics_extractor import RadiomicsFeatureExtractor

# 初始化提取器
extractor = RadiomicsFeatureExtractor()

# 提取特征
radiomics_features = extractor.extract_features_from_arrays(
    image_array,      # 图像数组
    mask_array,       # 分割掩码
    spacing=(1.0, 1.0, 1.0)  # 体素间距
)

print(f"提取放射组学特征数量: {len(radiomics_features)}")

# 保存特征
extractor.save_features(radiomics_features, 'radiomics_features.csv')
```

### 2. 血管骨架化

```python
from vessel_segmentation_3d.skeletonization.morphological import skeletonize_vessel_morphological

# 骨架化
skeleton_points, skeleton = skeletonize_vessel_morphological(vessel_mask)

print(f"骨架点数量: {len(skeleton_points)}")
```

### 3. 拓扑分析

```python
from vessel_segmentation_3d.skeletonization.topology_analysis import analyze_vessel_topology

# 分析拓扑
graph, branches, junctions, endpoints = analyze_vessel_topology(skeleton_points)

print(f"分支数量: {len(branches)}")
print(f"分叉点数量: {len(junctions)}")
```

### 4. 曲率计算

```python
from vessel_segmentation_3d.morphometry.curvature import calculate_curvature_3d

# 计算曲率
curvatures = calculate_curvature_3d(branch_points, smooth=0.1)

print(f"平均曲率: {np.mean(curvatures):.4f}")
print(f"最大曲率: {np.max(curvatures):.4f}")
```

### 5. 扭率计算

```python
from vessel_segmentation_3d.morphometry.torsion import calculate_torsion_3d

# 计算扭率
torsions = calculate_torsion_3d(branch_points, smooth=0.1)

print(f"平均扭率: {np.mean(torsions):.4f}")
print(f"扭率模式: {classify_torsion_pattern(torsions)}")
```

### 6. 分支密度计算

```python
from vessel_segmentation_3d.morphometry.branching import calculate_branching_density

# 计算分支密度
density, tumor_volume = calculate_branching_density(
    junctions=junctions,
    tumor_mask=tumor_mask,
    spacing=(1.0, 1.0, 1.0)
)

print(f"分支密度: {density:.6f} 分叉数/mm³")
```

### 7. 综合特征提取

```python
from vessel_segmentation_3d.morphometry.feature_extractor import VesselMorphometryExtractor

# 初始化提取器
extractor = VesselMorphometryExtractor(spacing=(1.0, 1.0, 1.0))

# 提取所有特征
features = extractor.extract_all_features(
    vessel_mask=vessel_mask,
    tumor_mask=tumor_mask,
    skeleton_points=skeleton_points,
    branches=branches,
    junctions=junctions,
    endpoints=endpoints
)

# 保存特征
extractor.save_features(features, 'features.csv', case_id='case001')
```

## 输出特征说明

### 形态学特征

| 特征名称 | 描述 | 单位 |
|---------|------|------|
| `curvature_mean` | 平均曲率 | 1/mm |
| `curvature_max` | 最大曲率 | 1/mm |
| `curvature_std` | 曲率标准差 | 1/mm |
| `torsion_mean` | 平均扭率 | 1/mm |
| `torsion_max` | 最大扭率 | 1/mm |
| `branching_density` | 分支密度 | 分叉数/mm³ |
| `vessel_density` | 血管密度 | 无量纲 |
| `radius_mean_mm` | 平均血管半径 | mm |
| `junction_count` | 分叉点数量 | 个 |
| `branch_count` | 分支数量 | 条 |

### 放射组学特征

PyRadiomics 提取的特征包括：

| 特征类别 | 描述 | 数量 |
|---------|------|------|
| 一阶统计特征 | 均值、方差、熵、偏度、峰度等 | ~15 |
| 形态学特征 | 体积、表面积、球形度、表面积/体积比等 | ~16 |
| 纹理特征 | GLCM（灰度共生矩阵） | ~24 |
| 纹理特征 | GLRLM（灰度游程长度矩阵） | ~16 |
| 纹理特征 | GLSZM（灰度大小区域矩阵） | ~16 |
| 纹理特征 | GLDM（灰度依赖矩阵） | ~14 |
| 纹理特征 | NGTDM（邻域灰度差分矩阵） | ~5 |
| 纹理特征 | 一阶直方图特征 | ~10 |

**特征命名格式**：
- 肿瘤区域：`tumor_{特征名}`
- 血管区域：`vessel_{特征名}`

**示例**：
- `tumor_firstorder_Mean` - 肿瘤区域的平均强度
- `vessel_shape_Sphericity` - 血管区域的球形度
- `tumor_glcm_Contrast` - 肿瘤区域的对比度纹理

### 临床意义

- **曲率**：反映血管弯曲程度，异常曲率可能提示血管病变
- **扭率**：反映血管扭曲程度，高扭率可能提示肿瘤侵袭
- **分支密度**：反映血管丰富程度，高密度提示血管生成活跃
- **血管密度**：反映肿瘤血供情况，与肿瘤恶性程度相关

## 数据要求

### 输入数据格式

- **格式**：NIfTI (.nii 或 .nii.gz) 或 DICOM系列
- **模态**：CT或MRI
- **推荐**：
  - CT：增强扫描，层厚≤3mm
  - MRI：T1增强序列

### 预处理

系统自动执行以下预处理：
1. 重采样到各向同性体素（默认1×1×1 mm³）
2. 窗宽窗位调整（CT默认肺窗）
3. 归一化到[0, 1]

## 模型训练

### 准备数据

```python
# 数据目录结构
data/
├── train/
│   ├── images/
│   │   ├── case001.nii.gz
│   │   └── ...
│   └── masks/
│       ├── case001_tumor.nii.gz
│       ├── case001_vessel.nii.gz
│       └── ...
└── val/
    ├── images/
    └── masks/
```

### 训练模型

```python
from vessel_segmentation_3d.segmentation.tumor_vessel_seg import CoSegmentationNet
import torch

# 创建模型
model = CoSegmentationNet(in_channels=1)

# 训练代码
# ... (详见训练脚本)
```

## 可视化

### 3D可视化

```python
from vessel_segmentation_3d.utils.visualization import visualize_3d

# 可视化血管骨架
visualize_3d(skeleton_points, junctions, endpoints)
```

### 特征可视化

```python
from vessel_segmentation_3d.utils.visualization import plot_curvature_torsion

# 绘制曲率和扭率
plot_curvature_torsion(curvatures, torsions)
```

## 性能优化

### GPU加速

```python
# 使用CUDA
pipeline = VesselSegmentationReconstructionPipeline(config, device='cuda')
```

### 批处理

```python
# 批量处理多个病例
case_list = ['case001', 'case002', 'case003']

for case_id in case_list:
    image_path = f'data/{case_id}.nii.gz'
    features = pipeline.run_pipeline(image_path, case_id)
```

## 常见问题

### Q1: 内存不足怎么办？

A: 减小batch size或使用滑动窗口推理：

```python
# 使用滑动窗口
from monai.inferers import sliding_window_inference

output = sliding_window_inference(
    inputs=image,
    roi_size=(64, 128, 128),
    sw_batch_size=4,
    predictor=model
)
```

### Q2: 如何调整分割阈值？

A: 在配置中设置：

```python
config = {
    'segmentation': {
        'tumor_threshold': 0.5,
        'vessel_threshold': 0.5
    }
}
```

### Q3: 如何处理不同分辨率的图像？

A: 系统会自动重采样，也可手动设置：

```python
config = {
    'preprocessing': {
        'target_spacing': [0.5, 0.5, 0.5]  # 更高分辨率
    }
}
```

## 引用

如果您使用本项目，请引用：

```bibtex
@software{vessel_segmentation_3d,
  title = {Vessel Segmentation and 3D Reconstruction System},
  author = {Medical Imaging AI Team},
  year = {2026},
  version = {1.0}
}
```

## 许可证

本项目采用 MIT 许可证。

## 联系方式

- 项目主页：https://github.com/yourusername/vessel_segmentation_3d
- 问题反馈：https://github.com/yourusername/vessel_segmentation_3d/issues
- 邮箱：your.email@example.com

## 致谢

感谢以下开源项目：
- [MONAI](https://monai.io/)
- [scikit-image](https://scikit-image.org/)
- [SimpleITK](https://simpleitk.org/)
- [NetworkX](https://networkx.org/)

---

**最后更新**: 2026-04-08
