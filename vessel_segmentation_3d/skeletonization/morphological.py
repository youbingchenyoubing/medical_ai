"""
血管骨架化模块 - 形态学方法
============================

实现基于形态学细化的血管骨架化算法。

方法：
1. 迭代细化算法
2. 保持拓扑结构
3. 提取中轴线

作者：医学影像AI研究团队
日期：2026-04-08
"""

import numpy as np
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def skeletonize_vessel_morphological(vessel_mask: np.ndarray,
                                     method: str = 'lee') -> Tuple[np.ndarray, np.ndarray]:
    """
    基于形态学细化的骨架化
    
    将三维血管结构简化为一维中轴线。
    
    算法步骤：
    1. 二值化处理
    2. 迭代形态学细化
    3. 保持连通性
    4. 提取骨架点
    
    参数:
        vessel_mask (np.ndarray): 血管二值mask (D, H, W)
        method (str): 细化方法 ('lee', 'zhang', 'guo')
    
    返回:
        skeleton_points (np.ndarray): 骨架点坐标 (N, 3)
        skeleton (np.ndarray): 骨架二值图像 (D, H, W)
    
    示例:
        >>> vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
        >>> vessel_mask[20:30, 20:30, 20:30] = 1  # 立方体血管
        >>> points, skeleton = skeletonize_vessel_morphological(vessel_mask)
        >>> print(f"骨架点数量: {len(points)}")
    """
    try:
        from skimage.morphology import skeletonize_3d
    except ImportError:
        logger.error("scikit-image未安装，请运行: pip install scikit-image")
        raise
    
    logger.info("开始形态学骨架化...")
    
    # 确保输入是二值图像
    vessel_binary = (vessel_mask > 0).astype(np.uint8)
    
    # 统计原始体素数量
    original_voxels = np.sum(vessel_binary)
    logger.info(f"原始血管体素数量: {original_voxels}")
    
    # 3D骨架化
    logger.info(f"使用 {method} 方法进行骨架化...")
    skeleton = skeletonize_3d(vessel_binary, method=method)
    
    # 提取骨架点坐标
    skeleton_points = np.argwhere(skeleton > 0)
    
    # 统计结果
    skeleton_voxels = len(skeleton_points)
    compression_ratio = original_voxels / skeleton_voxels if skeleton_voxels > 0 else 0
    
    logger.info(f"骨架点数量: {skeleton_voxels}")
    logger.info(f"压缩比: {compression_ratio:.2f}")
    
    return skeleton_points, skeleton


def skeletonize_with_pruning(vessel_mask: np.ndarray,
                             min_branch_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    带剪枝的骨架化
    
    去除短分支，保留主要血管结构。
    
    参数:
        vessel_mask (np.ndarray): 血管mask
        min_branch_length (int): 最小分支长度（体素数）
    
    返回:
        skeleton_points: 骨架点
        skeleton: 骨架图像
    """
    logger.info("执行带剪枝的骨架化...")
    
    # 基础骨架化
    skeleton_points, skeleton = skeletonize_vessel_morphological(vessel_mask)
    
    # 识别并去除短分支
    skeleton_pruned = prune_short_branches(skeleton, min_branch_length)
    
    # 提取剪枝后的骨架点
    skeleton_points_pruned = np.argwhere(skeleton_pruned > 0)
    
    logger.info(f"剪枝前骨架点: {len(skeleton_points)}")
    logger.info(f"剪枝后骨架点: {len(skeleton_points_pruned)}")
    
    return skeleton_points_pruned, skeleton_pruned


def prune_short_branches(skeleton: np.ndarray,
                         min_length: int) -> np.ndarray:
    """
    剪除短分支
    
    参数:
        skeleton: 骨架图像
        min_length: 最小分支长度
    
    返回:
        剪枝后的骨架
    """
    from scipy.ndimage import label
    
    # 标记连通区域
    labeled_array, num_features = label(skeleton)
    
    # 对每个连通区域进行处理
    skeleton_pruned = np.zeros_like(skeleton)
    
    for i in range(1, num_features + 1):
        component = (labeled_array == i)
        
        # 计算连通区域大小
        component_size = np.sum(component)
        
        # 保留大于最小长度的分支
        if component_size >= min_length:
            skeleton_pruned[component] = 1
    
    return skeleton_pruned


def calculate_skeleton_quality(skeleton: np.ndarray,
                               original_mask: np.ndarray) -> dict:
    """
    计算骨架质量指标
    
    参数:
        skeleton: 骨架图像
        original_mask: 原始血管mask
    
    返回:
        quality_metrics: 质量指标字典
    """
    from scipy.ndimage import distance_transform_edt
    
    # 骨架点数量
    skeleton_points = np.sum(skeleton)
    
    # 原始体素数量
    original_points = np.sum(original_mask)
    
    # 压缩比
    compression_ratio = original_points / skeleton_points if skeleton_points > 0 else 0
    
    # 计算骨架到原始血管的距离（应该很小）
    distance_map = distance_transform_edt(original_mask)
    skeleton_distances = distance_map[skeleton > 0]
    
    # 平均距离（应该接近0，表示骨架在血管中心）
    mean_distance = np.mean(skeleton_distances) if len(skeleton_distances) > 0 else 0
    max_distance = np.max(skeleton_distances) if len(skeleton_distances) > 0 else 0
    
    quality_metrics = {
        'skeleton_points': int(skeleton_points),
        'original_points': int(original_points),
        'compression_ratio': float(compression_ratio),
        'mean_distance_to_center': float(mean_distance),
        'max_distance_to_center': float(max_distance),
    }
    
    return quality_metrics


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试形态学骨架化")
    print("="*60)
    
    # 创建测试血管（圆柱体）
    vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    
    # 创建一个弯曲的血管
    for i in range(10, 40):
        y_center = 25 + int(5 * np.sin(i * 0.1))
        z_center = 25 + int(5 * np.cos(i * 0.1))
        
        # 创建圆形截面
        for y in range(y_center - 3, y_center + 4):
            for z in range(z_center - 3, z_center + 4):
                if (y - y_center)**2 + (z - z_center)**2 <= 9:
                    vessel_mask[i, y, z] = 1
    
    print(f"测试血管形状: {vessel_mask.shape}")
    print(f"血管体素数量: {np.sum(vessel_mask)}")
    
    # 执行骨架化
    skeleton_points, skeleton = skeletonize_vessel_morphological(vessel_mask)
    
    print(f"\n骨架点数量: {len(skeleton_points)}")
    
    # 计算质量指标
    quality = calculate_skeleton_quality(skeleton, vessel_mask)
    
    print("\n骨架质量指标:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
    
    print("\n✓ 测试完成!")
