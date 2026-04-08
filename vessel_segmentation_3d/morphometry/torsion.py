"""
扭率计算模块
============

实现血管扭率的计算方法。

扭率反映血管在三维空间中扭曲的程度。

作者：医学影像AI研究团队
日期：2026-04-08
"""

import numpy as np
from typing import Dict
from scipy.interpolate import splprep, splev
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_torsion_3d(points: np.ndarray,
                         smooth: float = 0.1,
                         k: int = 3) -> np.ndarray:
    """
    计算3D曲线的扭率（基于B样条拟合）
    
    数学公式：
        τ = (r' × r'') · r''' / |r' × r''|²
    
    其中：
        - r': 一阶导数
        - r'': 二阶导数
        - r''': 三阶导数
        - ×: 叉积
        - ·: 点积
    
    物理意义：
        - τ = 0: 平面曲线（不扭曲）
        - τ > 0: 右手螺旋
        - τ < 0: 左手螺旋
    
    参数:
        points (np.ndarray): 曲线点坐标 (N, 3)
        smooth (float): 平滑参数
        k (int): B样条阶数
    
    返回:
        torsions (np.ndarray): 每个点的扭率 (N,)
    
    示例:
        >>> # 创建螺旋线
        >>> t = np.linspace(0, 4*np.pi, 100)
        >>> points = np.column_stack([np.cos(t), np.sin(t), t])
        >>> torsions = calculate_torsion_3d(points)
        >>> print(f"平均扭率: {np.mean(torsions):.4f}")
    """
    n_points = len(points)
    
    if n_points < 5:
        logger.warning("点数太少，无法计算扭率（至少需要5个点）")
        return np.zeros(n_points)
    
    try:
        # B样条拟合
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], 
                         s=smooth, k=min(k, n_points - 1))
        
        u_new = np.linspace(0, 1, n_points)
        
        # 一阶、二阶、三阶导数
        d1 = np.array(splev(u_new, tck, der=1)).T  # r'
        d2 = np.array(splev(u_new, tck, der=2)).T  # r''
        d3 = np.array(splev(u_new, tck, der=3)).T  # r'''
        
        # 计算扭率
        # τ = (r' × r'') · r''' / |r' × r''|²
        cross_d1_d2 = np.cross(d1, d2)
        cross_norm_sq = np.sum(cross_d1_d2 ** 2, axis=1)
        
        torsions = np.sum(cross_d1_d2 * d3, axis=1) / (cross_norm_sq + 1e-10)
        
        return torsions
        
    except Exception as e:
        logger.error(f"扭率计算失败: {e}")
        return np.zeros(n_points)


def calculate_torsion_discrete(points: np.ndarray) -> np.ndarray:
    """
    离散扭率计算
    
    使用有限差分近似导数。
    
    参数:
        points (np.ndarray): 曲线点坐标 (N, 3)
    
    返回:
        torsions (np.ndarray): 扭率 (N,)
    """
    n = len(points)
    torsions = np.zeros(n)
    
    if n < 5:
        return torsions
    
    for i in range(2, n - 2):
        # 使用5点模板计算导数
        p0, p1, p2, p3, p4 = points[i-2:i+3]
        
        # 中心差分近似导数
        d1 = (-p4 + 8*p3 - 8*p1 + p0) / 12  # r'
        d2 = (-p4 + 16*p3 - 30*p2 + 16*p1 - p0) / 12  # r''
        d3 = (p4 - 2*p3 + p2)  # r''' (简化)
        
        # 计算扭率
        cross = np.cross(d1, d2)
        cross_norm_sq = np.sum(cross ** 2)
        
        if cross_norm_sq > 1e-10:
            torsions[i] = np.dot(cross, d3) / cross_norm_sq
    
    return torsions


def calculate_torsion_statistics(torsions: np.ndarray) -> Dict:
    """
    计算扭率统计特征
    
    参数:
        torsions (np.ndarray): 扭率数组
    
    返回:
        stats: 统计特征字典
    """
    # 计算绝对值统计
    abs_torsions = np.abs(torsions)
    
    stats = {
        'mean': float(np.mean(torsions)),
        'std': float(np.std(torsions)),
        'max': float(np.max(abs_torsions)),
        'min': float(np.min(torsions)),
        'median': float(np.median(torsions)),
        'abs_mean': float(np.mean(abs_torsions)),
        'abs_std': float(np.std(abs_torsions)),
        'positive_ratio': float(np.sum(torsions > 0) / len(torsions)),
        'negative_ratio': float(np.sum(torsions < 0) / len(torsions)),
    }
    
    return stats


def classify_torsion_pattern(torsions: np.ndarray,
                             threshold: float = 0.1) -> str:
    """
    分类扭率模式
    
    参数:
        torsions (np.ndarray): 扭率数组
        threshold (float): 分类阈值
    
    返回:
        pattern: 扭率模式 ('planar', 'right_handed', 'left_handed', 'mixed')
    """
    mean_torsion = np.mean(torsions)
    
    if np.abs(mean_torsion) < threshold:
        return 'planar'  # 平面曲线
    elif mean_torsion > threshold:
        return 'right_handed'  # 右手螺旋
    elif mean_torsion < -threshold:
        return 'left_handed'  # 左手螺旋
    else:
        return 'mixed'  # 混合模式


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试扭率计算")
    print("="*60)
    
    # 创建测试曲线1：螺旋线（右手螺旋）
    t = np.linspace(0, 4*np.pi, 100)
    radius = 5.0
    pitch = 2.0
    points_helix = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        pitch * t
    ])
    
    print("测试曲线1: 右手螺旋线")
    torsions_helix = calculate_torsion_3d(points_helix, smooth=0.0)
    
    print(f"  平均扭率: {np.mean(torsions_helix):.4f}")
    print(f"  扭率模式: {classify_torsion_pattern(torsions_helix)}")
    
    # 创建测试曲线2：平面曲线
    theta = np.linspace(0, np.pi, 100)
    points_planar = np.column_stack([
        10 * np.cos(theta),
        10 * np.sin(theta),
        np.zeros_like(theta)
    ])
    
    print("\n测试曲线2: 平面半圆")
    torsions_planar = calculate_torsion_3d(points_planar, smooth=0.0)
    
    print(f"  平均扭率: {np.mean(torsions_planar):.4f}")
    print(f"  扭率模式: {classify_torsion_pattern(torsions_planar)}")
    
    # 统计特征
    stats = calculate_torsion_statistics(torsions_helix)
    
    print("\n扭率统计特征（螺旋线）:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ 测试完成!")
