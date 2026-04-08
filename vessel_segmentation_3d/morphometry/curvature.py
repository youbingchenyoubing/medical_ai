"""
曲率计算模块
============

实现血管曲率的计算方法。

曲率反映血管弯曲的程度。

作者：医学影像AI研究团队
日期：2026-04-08
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import splprep, splev
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_curvature_3d(points: np.ndarray,
                           smooth: float = 0.1,
                           k: int = 3) -> np.ndarray:
    """
    计算3D曲线的曲率（基于B样条拟合）
    
    数学公式：
        κ = |r' × r''| / |r'|³
    
    其中：
        - r': 一阶导数（切向量）
        - r'': 二阶导数
        - ×: 叉积
    
    参数:
        points (np.ndarray): 曲线点坐标 (N, 3)
        smooth (float): 平滑参数（0表示插值，越大越平滑）
        k (int): B样条阶数（通常为3）
    
    返回:
        curvatures (np.ndarray): 每个点的曲率 (N,)
    
    示例:
        >>> points = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]])
        >>> curvatures = calculate_curvature_3d(points)
        >>> print(f"平均曲率: {np.mean(curvatures):.4f}")
    """
    n_points = len(points)
    
    if n_points < 4:
        logger.warning("点数太少，无法计算曲率")
        return np.zeros(n_points)
    
    try:
        # B样条拟合曲线
        # splprep返回参数tck和参数u
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], 
                         s=smooth, k=min(k, n_points - 1))
        
        # 在原始点位置计算导数
        u_new = np.linspace(0, 1, n_points)
        
        # 一阶导数 r'
        derivatives_1 = np.array(splev(u_new, tck, der=1)).T  # (N, 3)
        
        # 二阶导数 r''
        derivatives_2 = np.array(splev(u_new, tck, der=2)).T  # (N, 3)
        
        # 计算曲率
        # κ = |r' × r''| / |r'|³
        cross_product = np.cross(derivatives_1, derivatives_2)
        cross_norm = np.linalg.norm(cross_product, axis=1)
        
        d1_norm = np.linalg.norm(derivatives_1, axis=1)
        d1_norm_cubed = d1_norm ** 3
        
        # 避免除零
        curvatures = cross_norm / (d1_norm_cubed + 1e-10)
        
        return curvatures
        
    except Exception as e:
        logger.error(f"曲率计算失败: {e}")
        return np.zeros(n_points)


def calculate_curvature_discrete(points: np.ndarray) -> np.ndarray:
    """
    离散曲率计算（不需要曲线拟合）
    
    基于Menger曲率公式：
        κ = 4A / (|P1P2| × |P2P3| × |P3P1|)
    
    其中A是三角形P1P2P3的面积。
    
    优点：
        - 不需要曲线拟合
        - 计算简单快速
        - 对噪声敏感度较低
    
    参数:
        points (np.ndarray): 曲线点坐标 (N, 3)
    
    返回:
        curvatures (np.ndarray): 曲率 (N,)
    """
    n = len(points)
    curvatures = np.zeros(n)
    
    if n < 3:
        return curvatures
    
    for i in range(1, n - 1):
        # 三个连续点
        p1 = points[i - 1]
        p2 = points[i]
        p3 = points[i + 1]
        
        # 计算三角形边长
        a = np.linalg.norm(p2 - p1)  # |P1P2|
        b = np.linalg.norm(p3 - p2)  # |P2P3|
        c = np.linalg.norm(p3 - p1)  # |P3P1|
        
        # 海伦公式计算三角形面积
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)
        
        if area_squared > 0:
            area = np.sqrt(area_squared)
            
            # Menger曲率
            if a * b * c > 1e-10:
                curvatures[i] = 4 * area / (a * b * c)
    
    # 边界点处理
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    
    return curvatures


def calculate_curvature_oscillating_circle(points: np.ndarray,
                                           radius_range: Tuple[float, float] = (1.0, 10.0)) -> np.ndarray:
    """
    基于密切圆的曲率计算
    
    曲率 = 1 / 密切圆半径
    
    参数:
        points (np.ndarray): 曲线点坐标 (N, 3)
        radius_range (Tuple): 半径范围限制
    
    返回:
        curvatures (np.ndarray): 曲率 (N,)
    """
    n = len(points)
    curvatures = np.zeros(n)
    
    if n < 3:
        return curvatures
    
    for i in range(1, n - 1):
        # 三个点确定一个圆
        p1, p2, p3 = points[i-1], points[i], points[i+1]
        
        # 计算外接圆半径
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)
        
        if area_squared > 1e-10:
            area = np.sqrt(area_squared)
            radius = (a * b * c) / (4 * area)
            
            # 限制半径范围
            radius = np.clip(radius, radius_range[0], radius_range[1])
            
            curvatures[i] = 1.0 / radius
    
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    
    return curvatures


def calculate_curvature_statistics(curvatures: np.ndarray) -> Dict:
    """
    计算曲率统计特征
    
    参数:
        curvatures (np.ndarray): 曲率数组
    
    返回:
        stats: 统计特征字典
    """
    stats = {
        'mean': float(np.mean(curvatures)),
        'std': float(np.std(curvatures)),
        'max': float(np.max(curvatures)),
        'min': float(np.min(curvatures)),
        'median': float(np.median(curvatures)),
        'p90': float(np.percentile(curvatures, 90)),
        'p95': float(np.percentile(curvatures, 95)),
        'p99': float(np.percentile(curvatures, 99)),
    }
    
    return stats


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试曲率计算")
    print("="*60)
    
    # 创建测试曲线（半圆）
    theta = np.linspace(0, np.pi, 50)
    radius = 10.0
    points = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta)
    ])
    
    print(f"测试曲线: 半圆，半径 {radius}")
    print(f"理论曲率: {1/radius:.4f}")
    
    # 计算曲率
    curvatures_spline = calculate_curvature_3d(points, smooth=0.0)
    curvatures_discrete = calculate_curvature_discrete(points)
    
    print(f"\nB样条方法:")
    print(f"  平均曲率: {np.mean(curvatures_spline):.4f}")
    print(f"  标准差: {np.std(curvatures_spline):.4f}")
    
    print(f"\n离散方法:")
    print(f"  平均曲率: {np.mean(curvatures_discrete):.4f}")
    print(f"  标准差: {np.std(curvatures_discrete):.4f}")
    
    # 统计特征
    stats = calculate_curvature_statistics(curvatures_spline)
    
    print("\n曲率统计特征:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ 测试完成!")
