"""
分支密度计算模块
==================

计算血管分支密度及相关特征。

分支密度定义：
- 单位体积内的分叉点数量
- 反映血管丰富程度
- 用于肿瘤血管生成评估

作者：医学影像AI研究团队
日期：2026-04-08
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy.ndimage import gaussian_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_branching_density(junctions: list,
                                tumor_mask: np.ndarray,
                                spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[float, float]:
    """
    计算分支密度
    
    分支密度 = 分叉点数量 / 肿瘤体积
    
    参数:
        junctions (list): 分叉点坐标列表 [(x,y,z), ...]
        tumor_mask (np.ndarray): 肿瘤二值mask (D, H, W)
        spacing (tuple): 体素间距 (mm)
    
    返回:
        density (float): 分支密度 (分叉数/mm³)
        tumor_volume (float): 肿瘤体积 (mm³)
    
    示例:
        >>> junctions = [[10, 10, 10], [20, 20, 20]]
        >>> tumor_mask = np.zeros((50, 50, 50))
        >>> tumor_mask[5:45, 5:45, 5:45] = 1
        >>> density, volume = calculate_branching_density(junctions, tumor_mask)
        >>> print(f"分支密度: {density:.4f} 分叉数/mm³")
    """
    # 计算体素体积
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³
    
    # 计算肿瘤体积
    tumor_voxels = np.sum(tumor_mask > 0)
    tumor_volume = tumor_voxels * voxel_volume
    
    # 分叉点数量
    num_junctions = len(junctions)
    
    # 分支密度
    if tumor_volume > 0:
        density = num_junctions / tumor_volume
    else:
        density = 0.0
    
    logger.info(f"分叉点数量: {num_junctions}")
    logger.info(f"肿瘤体积: {tumor_volume:.2f} mm³")
    logger.info(f"分支密度: {density:.6f} 分叉数/mm³")
    
    return density, tumor_volume


def calculate_local_branching_density(junctions: list,
                                      tumor_mask: np.ndarray,
                                      spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                      radius: float = 10.0) -> np.ndarray:
    """
    计算局部分支密度（空间变化）
    
    使用高斯核平滑分叉点分布，生成连续的密度图。
    
    参数:
        junctions (list): 分叉点坐标列表
        tumor_mask (np.ndarray): 肿瘤mask
        spacing (tuple): 体素间距 (mm)
        radius (float): 局部邻域半径 (mm)
    
    返回:
        density_map (np.ndarray): 局部分支密度图 (分叉数/mm³)
    
    示例:
        >>> junctions = [[10, 10, 10], [20, 20, 20]]
        >>> tumor_mask = np.zeros((50, 50, 50))
        >>> tumor_mask[5:45, 5:45, 5:45] = 1
        >>> density_map = calculate_local_branching_density(junctions, tumor_mask)
        >>> print(f"密度图形状: {density_map.shape}")
    """
    logger.info("计算局部分支密度...")
    
    # 创建分叉点分布图
    junction_map = np.zeros_like(tumor_mask, dtype=np.float32)
    
    for junction in junctions:
        # 转换为整数坐标
        z, y, x = [int(coord) for coord in junction]
        
        # 检查边界
        if (0 <= z < junction_map.shape[0] and
            0 <= y < junction_map.shape[1] and
            0 <= x < junction_map.shape[2]):
            junction_map[z, y, x] = 1.0
    
    # 高斯平滑（模拟局部邻域）
    # sigma = radius / spacing
    sigma = [radius / s for s in spacing]
    
    density_map = gaussian_filter(junction_map, sigma=sigma)
    
    # 归一化到体积
    # 计算球体体积
    sphere_volume = (4.0 / 3.0) * np.pi * radius**3
    
    # 归一化
    density_map = density_map / sphere_volume
    
    # 只在肿瘤区域内计算
    density_map = density_map * (tumor_mask > 0)
    
    logger.info(f"局部密度图范围: [{density_map.min():.6f}, {density_map.max():.6f}]")
    
    return density_map


def calculate_vessel_density(vessel_mask: np.ndarray,
                             tumor_mask: np.ndarray,
                             spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[float, float, float]:
    """
    计算血管密度
    
    血管密度 = 血管体积 / 肿瘤体积
    
    参数:
        vessel_mask (np.ndarray): 血管二值mask
        tumor_mask (np.ndarray): 肿瘤二值mask
        spacing (tuple): 体素间距 (mm)
    
    返回:
        vessel_density (float): 血管密度
        vessel_volume (float): 血管体积 (mm³)
        tumor_volume (float): 肿瘤体积 (mm³)
    
    示例:
        >>> vessel_mask = np.zeros((50, 50, 50))
        >>> vessel_mask[10:40, 10:40, 10:40] = 1
        >>> tumor_mask = np.zeros((50, 50, 50))
        >>> tumor_mask[5:45, 5:45, 5:45] = 1
        >>> density, v_vol, t_vol = calculate_vessel_density(vessel_mask, tumor_mask)
        >>> print(f"血管密度: {density:.4f}")
    """
    # 体素体积
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    # 血管体积
    vessel_voxels = np.sum(vessel_mask > 0)
    vessel_volume = vessel_voxels * voxel_volume
    
    # 肿瘤体积
    tumor_voxels = np.sum(tumor_mask > 0)
    tumor_volume = tumor_voxels * voxel_volume
    
    # 血管密度
    if tumor_volume > 0:
        vessel_density = vessel_volume / tumor_volume
    else:
        vessel_density = 0.0
    
    logger.info(f"血管体积: {vessel_volume:.2f} mm³")
    logger.info(f"肿瘤体积: {tumor_volume:.2f} mm³")
    logger.info(f"血管密度: {vessel_density:.4f}")
    
    return vessel_density, vessel_volume, tumor_volume


def calculate_branching_features(junctions: list,
                                  branches: list,
                                  tumor_mask: np.ndarray,
                                  vessel_mask: np.ndarray,
                                  spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                                  skeleton_points: Optional[np.ndarray] = None) -> Dict:
    """
    计算所有分支相关特征
    
    参数:
        junctions (list): 分叉点列表
        branches (list): 分支列表
        tumor_mask (np.ndarray): 肿瘤mask
        vessel_mask (np.ndarray): 血管mask
        spacing (tuple): 体素间距
    
    返回:
        features (dict): 分支特征字典
    """
    logger.info("计算分支特征...")
    
    features = {}
    
    # 1. 分支密度
    branching_density, tumor_volume = calculate_branching_density(
        junctions, tumor_mask, spacing
    )
    features['branching_density'] = float(branching_density)
    features['tumor_volume_mm3'] = float(tumor_volume)
    
    # 2. 血管密度
    vessel_density, vessel_volume, _ = calculate_vessel_density(
        vessel_mask, tumor_mask, spacing
    )
    features['vessel_density'] = float(vessel_density)
    features['vessel_volume_mm3'] = float(vessel_volume)
    
    # 3. 分叉点统计
    features['junction_count'] = len(junctions)
    
    # 4. 分支统计
    features['branch_count'] = len(branches)
    
    # 5. 平均分支长度
    if len(branches) > 0:
            branch_lengths = []
            for branch in branches:
                if len(branch) > 1:
                    # 计算分支长度
                    length = 0.0
                    # 检查分支是索引还是坐标
                    if isinstance(branch[0], (int, np.integer)) and skeleton_points is not None:
                        # 分支是索引，使用骨架点转换为坐标
                        for i in range(len(branch) - 1):
                            p1_idx = branch[i]
                            p2_idx = branch[i+1]
                            p1 = skeleton_points[p1_idx]
                            p2 = skeleton_points[p2_idx]
                            dist = np.sqrt(np.sum((p2 - p1)**2 * np.array(spacing)**2))
                            length += dist
                    elif isinstance(branch[0], (list, np.ndarray)):
                        # 分支已经是坐标
                        for i in range(len(branch) - 1):
                            p1 = np.array(branch[i])
                            p2 = np.array(branch[i+1])
                            dist = np.sqrt(np.sum((p2 - p1)**2 * np.array(spacing)**2))
                            length += dist
                    else:
                        for i in range(len(branch) - 1):
                            p1 = np.array(branch[i])
                            p2 = np.array(branch[i+1])
                            dist = np.sqrt(np.sum((p2 - p1)**2 * np.array(spacing)**2))
                            length += dist
                    branch_lengths.append(length)
        
        if len(branch_lengths) > 0:
            features['mean_branch_length_mm'] = float(np.mean(branch_lengths))
            features['std_branch_length_mm'] = float(np.std(branch_lengths))
            features['max_branch_length_mm'] = float(np.max(branch_lengths))
            features['total_branch_length_mm'] = float(np.sum(branch_lengths))
        else:
            features['mean_branch_length_mm'] = 0.0
            features['std_branch_length_mm'] = 0.0
            features['max_branch_length_mm'] = 0.0
            features['total_branch_length_mm'] = 0.0
    else:
        features['mean_branch_length_mm'] = 0.0
        features['std_branch_length_mm'] = 0.0
        features['max_branch_length_mm'] = 0.0
        features['total_branch_length_mm'] = 0.0
    
    # 6. 分叉点/分支比
    if len(branches) > 0:
        features['junction_to_branch_ratio'] = len(junctions) / len(branches)
    else:
        features['junction_to_branch_ratio'] = 0.0
    
    # 7. 血管复杂度指标
    if tumor_volume > 0:
        # 每mm³肿瘤的血管长度
        features['vessel_length_per_volume'] = (
            features['total_branch_length_mm'] / tumor_volume
        )
    else:
        features['vessel_length_per_volume'] = 0.0
    
    logger.info(f"计算完成，共 {len(features)} 个分支特征")
    
    return features


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试分支密度计算")
    print("="*60)
    
    # 创建测试数据
    tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    tumor_mask[10:40, 10:40, 10:40] = 1
    
    vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    vessel_mask[15:35, 15:35, 15:35] = 1
    
    junctions = [[20, 20, 20], [25, 25, 25], [30, 30, 30]]
    branches = [
        [[20, 20, 20], [21, 21, 21], [22, 22, 22]],
        [[25, 25, 25], [26, 26, 26], [27, 27, 27]],
    ]
    
    # 计算分支密度
    density, volume = calculate_branching_density(
        junctions, tumor_mask, spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"\n分支密度: {density:.6f} 分叉数/mm³")
    print(f"肿瘤体积: {volume:.2f} mm³")
    
    # 计算血管密度
    vessel_density, vessel_vol, tumor_vol = calculate_vessel_density(
        vessel_mask, tumor_mask, spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"\n血管密度: {vessel_density:.4f}")
    print(f"血管体积: {vessel_vol:.2f} mm³")
    
    # 计算所有分支特征
    features = calculate_branching_features(
        junctions, branches, tumor_mask, vessel_mask, spacing=(1.0, 1.0, 1.0)
    )
    
    print("\n所有分支特征:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\n✓ 测试完成!")
