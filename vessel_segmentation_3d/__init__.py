"""
血管自动化分割与三维重建完整实现
=====================================

项目结构：
- vessel_segmentation_3d/          # 主模块
  ├── models/                      # 模型定义
  │   ├── unet3d.py               # 3D U-Net模型
  │   └── nnunet.py               # nnU-Net模型
  ├── segmentation/                # 分割模块
  │   ├── tumor_vessel_seg.py     # 肿瘤血管分割
  │   └── inference.py            # 推理脚本
  ├── skeletonization/             # 骨架化模块
  │   ├── morphological.py        # 形态学骨架化
  │   ├── distance_transform.py   # 距离变换方法
  │   └── topology_analysis.py    # 拓扑分析
  ├── morphometry/                 # 形态量化模块
  │   ├── curvature.py            # 曲率计算
  │   ├── torsion.py              # 扭率计算
  │   ├── branching.py            # 分支密度计算
  │   └── feature_extractor.py    # 综合特征提取
  ├── preprocessing/               # 预处理模块
  │   └── image_preprocessing.py  # 图像预处理
  ├── utils/                       # 工具函数
  │   ├── visualization.py        # 可视化
  │   └── io_utils.py             # 输入输出
  └── pipeline.py                  # 主流程

作者：医学影像AI研究团队
日期：2026-04-08
版本：v1.0
"""

# 主模块初始化
__version__ = '1.0.0'
__author__ = 'Medical Imaging AI Team'

# 导入主要模块
from .models.unet3d import UNet3D, DoubleConv3D, Down3D, Up3D
from .models.nnunet import nnUNetSegmenter
from .segmentation.tumor_vessel_seg import CoSegmentationNet
from .skeletonization.morphological import skeletonize_vessel_morphological
# from .skeletonization.distance_transform import skeletonize_vessel_distance_transform
from .skeletonization.topology_analysis import analyze_vessel_topology
from .morphometry.curvature import calculate_curvature_3d, calculate_curvature_discrete
from .morphometry.torsion import calculate_torsion_3d, calculate_torsion_discrete
from .morphometry.branching import calculate_branching_density
from .morphometry.feature_extractor import VesselMorphometryExtractor
from .morphometry.radiomics_extractor import RadiomicsFeatureExtractor
from .pipeline import VesselSegmentationReconstructionPipeline

__all__ = [
    'UNet3D',
    'DoubleConv3D',
    'Down3D',
    'Up3D',
    'nnUNetSegmenter',
    'CoSegmentationNet',
    'skeletonize_vessel_morphological',
    'skeletonize_vessel_distance_transform',
    'analyze_vessel_topology',
    'calculate_curvature_3d',
    'calculate_curvature_discrete',
    'calculate_torsion_3d',
    'calculate_torsion_discrete',
    'calculate_branching_density',
    'VesselMorphometryExtractor',
    'RadiomicsFeatureExtractor',
    'VesselSegmentationReconstructionPipeline',
]
