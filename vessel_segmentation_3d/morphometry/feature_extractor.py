"""
血管形态学特征综合提取器
========================

整合所有血管形态学特征的提取：
- 曲率特征
- 扭率特征
- 分支密度特征
- 血管密度特征
- 半径特征
- 拓扑特征

作者：医学影像AI研究团队
日期：2026-04-08
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VesselMorphometryExtractor:
    """
    血管形态学特征提取器
    
    从血管分割结果中提取全面的形态学特征。
    
    参数:
        spacing (tuple): 体素间距 (mm)
        smooth (float): 曲线平滑参数
        verbose (bool): 是否显示详细信息
    
    示例:
        >>> extractor = VesselMorphometryExtractor(spacing=(1.0, 1.0, 1.0))
        >>> features = extractor.extract_all_features(
        ...     vessel_mask=vessel_mask,
        ...     tumor_mask=tumor_mask,
        ...     skeleton_points=skeleton_points,
        ...     branches=branches,
        ...     junctions=junctions
        ... )
    """
    
    def __init__(self,
                 spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 smooth: float = 0.1,
                 verbose: bool = True):
        self.spacing = spacing
        self.smooth = smooth
        self.verbose = verbose
        
        logger.info("血管形态学特征提取器初始化完成")
    
    def extract_all_features(self,
                            vessel_mask: np.ndarray,
                            tumor_mask: np.ndarray,
                            skeleton_points: np.ndarray,
                            branches: List,
                            junctions: List,
                            endpoints: Optional[List] = None,
                            graph: Optional[object] = None) -> Dict:
        """
        提取所有血管形态学特征
        
        参数:
            vessel_mask (np.ndarray): 血管分割mask (D, H, W)
            tumor_mask (np.ndarray): 肿瘤分割mask (D, H, W)
            skeleton_points (np.ndarray): 骨架点坐标 (N, 3)
            branches (list): 分支列表
            junctions (list): 分叉点列表
            endpoints (list): 端点列表（可选）
            graph (object): 拓扑图（可选）
        
        返回:
            features (dict): 特征字典
        """
        logger.info("="*60)
        logger.info("开始提取血管形态学特征")
        logger.info("="*60)
        
        start_time = time.time()
        
        features = {}
        
        # 1. 曲率特征
        logger.info("\n[1/6] 提取曲率特征...")
        curvature_features = self._extract_curvature_features(branches, skeleton_points)
        features.update(curvature_features)
        
        # 2. 扭率特征
        logger.info("\n[2/6] 提取扭率特征...")
        torsion_features = self._extract_torsion_features(branches, skeleton_points)
        features.update(torsion_features)
        
        # 3. 分支密度特征
        logger.info("\n[3/6] 提取分支密度特征...")
        branching_features = self._extract_branching_features(
            junctions, branches, tumor_mask, vessel_mask
        )
        features.update(branching_features)
        
        # 4. 血管密度特征
        logger.info("\n[4/6] 提取血管密度特征...")
        density_features = self._extract_density_features(vessel_mask, tumor_mask)
        features.update(density_features)
        
        # 5. 半径特征
        logger.info("\n[5/6] 提取半径特征...")
        radius_features = self._extract_radius_features(vessel_mask, skeleton_points)
        features.update(radius_features)
        
        # 6. 拓扑特征
        logger.info("\n[6/6] 提取拓扑特征...")
        topology_features = self._extract_topology_features(
            branches, junctions, endpoints, skeleton_points
        )
        features.update(topology_features)
        
        # 添加元数据
        features['extraction_time'] = time.time() - start_time
        features['num_skeleton_points'] = len(skeleton_points)
        
        logger.info("\n" + "="*60)
        logger.info(f"特征提取完成，共 {len(features)} 个特征")
        logger.info(f"耗时: {features['extraction_time']:.2f} 秒")
        logger.info("="*60)
        
        return features
    
    def _extract_curvature_features(self, branches: List, skeleton_points: np.ndarray) -> Dict:
        """
        提取曲率特征
        
        从所有分支中计算曲率统计量。
        """
        from .curvature import calculate_curvature_3d, calculate_curvature_statistics
        
        all_curvatures = []
        
        for branch in branches:
            if len(branch) > 3:
                # 修复分支数据格式：分支是节点索引，需要转换为坐标
                branch_indices = branch
                if isinstance(branch_indices[0], (int, np.integer)):
                    # 分支是索引列表，转换为坐标
                    branch_points = skeleton_points[branch_indices]
                else:
                    # 分支已经是坐标
                    branch_points = np.array(branch)
                
                # 计算曲率
                curvatures = calculate_curvature_3d(
                    branch_points, smooth=self.smooth
                )
                
                all_curvatures.extend(curvatures)
        
        if len(all_curvatures) == 0:
            logger.warning("没有足够的点计算曲率")
            return {
                'curvature_mean': 0.0,
                'curvature_std': 0.0,
                'curvature_max': 0.0,
                'curvature_median': 0.0,
                'curvature_p90': 0.0,
                'curvature_p95': 0.0,
            }
        
        all_curvatures = np.array(all_curvatures)
        
        # 计算统计特征
        stats = calculate_curvature_statistics(all_curvatures)
        
        # 添加前缀
        features = {f'curvature_{k}': v for k, v in stats.items()}
        
        logger.info(f"  曲率统计: mean={stats['mean']:.4f}, max={stats['max']:.4f}")
        
        return features
    
    def _extract_torsion_features(self, branches: List, skeleton_points: np.ndarray) -> Dict:
        """
        提取扭率特征
        
        从所有分支中计算扭率统计量。
        """
        from .torsion import calculate_torsion_3d, calculate_torsion_statistics
        
        all_torsions = []
        
        for branch in branches:
            if len(branch) > 4:
                # 修复分支数据格式：分支是节点索引，需要转换为坐标
                branch_indices = branch
                if isinstance(branch_indices[0], (int, np.integer)):
                    # 分支是索引列表，转换为坐标
                    branch_points = skeleton_points[branch_indices]
                else:
                    # 分支已经是坐标
                    branch_points = np.array(branch)
                
                torsions = calculate_torsion_3d(
                    branch_points, smooth=self.smooth
                )
                
                all_torsions.extend(torsions)
        
        if len(all_torsions) == 0:
            logger.warning("没有足够的点计算扭率")
            return {
                'torsion_mean': 0.0,
                'torsion_std': 0.0,
                'torsion_abs_mean': 0.0,
                'torsion_max': 0.0,
                'torsion_positive_ratio': 0.0,
            }
        
        all_torsions = np.array(all_torsions)
        
        # 计算统计特征
        stats = calculate_torsion_statistics(all_torsions)
        
        features = {f'torsion_{k}': v for k, v in stats.items()}
        
        logger.info(f"  扭率统计: mean={stats['mean']:.4f}, abs_mean={stats['abs_mean']:.4f}")
        
        return features
    
    def _extract_branching_features(self,
                                    junctions: List,
                                    branches: List,
                                    tumor_mask: np.ndarray,
                                    vessel_mask: np.ndarray) -> Dict:
        """
        提取分支密度特征
        """
        from .branching import calculate_branching_features
        
        features = calculate_branching_features(
            junctions, branches, tumor_mask, vessel_mask, self.spacing, skeleton_points
        )
        
        logger.info(f"  分支密度: {features['branching_density']:.6f} 分叉数/mm³")
        
        return features
    
    def _extract_density_features(self,
                                  vessel_mask: np.ndarray,
                                  tumor_mask: np.ndarray) -> Dict:
        """
        提取血管密度特征
        """
        from .branching import calculate_vessel_density
        
        vessel_density, vessel_volume, tumor_volume = calculate_vessel_density(
            vessel_mask, tumor_mask, self.spacing
        )
        
        features = {
            'vessel_density': float(vessel_density),
            'vessel_volume_mm3': float(vessel_volume),
            'tumor_volume_mm3': float(tumor_volume),
        }
        
        logger.info(f"  血管密度: {vessel_density:.4f}")
        
        return features
    
    def _extract_radius_features(self,
                                 vessel_mask: np.ndarray,
                                 skeleton_points: np.ndarray) -> Dict:
        """
        提取血管半径特征
        
        使用距离变换计算骨架点处的血管半径。
        """
        from scipy.ndimage import distance_transform_edt
        
        # 距离变换
        distance_map = distance_transform_edt(vessel_mask)
        
        # 提取骨架点处的半径
        radii = []
        
        for point in skeleton_points:
            z, y, x = [int(coord) for coord in point]
            
            # 检查边界
            if (0 <= z < distance_map.shape[0] and
                0 <= y < distance_map.shape[1] and
                0 <= x < distance_map.shape[2]):
                radius = distance_map[z, y, x]
                # 转换为mm
                radius_mm = radius * np.mean(self.spacing)
                radii.append(radius_mm)
        
        if len(radii) == 0:
            logger.warning("没有有效的骨架点计算半径")
            return {
                'radius_mean_mm': 0.0,
                'radius_std_mm': 0.0,
                'radius_max_mm': 0.0,
                'radius_min_mm': 0.0,
                'radius_median_mm': 0.0,
            }
        
        radii = np.array(radii)
        
        features = {
            'radius_mean_mm': float(np.mean(radii)),
            'radius_std_mm': float(np.std(radii)),
            'radius_max_mm': float(np.max(radii)),
            'radius_min_mm': float(np.min(radii)),
            'radius_median_mm': float(np.median(radii)),
            'radius_p90_mm': float(np.percentile(radii, 90)),
            'radius_p95_mm': float(np.percentile(radii, 95)),
        }
        
        logger.info(f"  平均半径: {features['radius_mean_mm']:.4f} mm")
        
        return features
    
    def _extract_topology_features(self,
                                   branches: List,
                                   junctions: List,
                                   endpoints: List,
                                   skeleton_points: np.ndarray) -> Dict:
        """
        提取拓扑特征
        """
        features = {
            'junction_count': len(junctions),
            'branch_count': len(branches),
            'skeleton_point_count': len(skeleton_points),
        }
        
        # 分叉点/分支比
        if len(branches) > 0:
            features['junction_to_branch_ratio'] = len(junctions) / len(branches)
        else:
            features['junction_to_branch_ratio'] = 0.0
        
        # 端点统计
        if endpoints is not None:
            features['endpoint_count'] = len(endpoints)
        else:
            features['endpoint_count'] = 0
        
        logger.info(f"  分叉点数量: {features['junction_count']}")
        logger.info(f"  分支数量: {features['branch_count']}")
        
        return features
    
    def features_to_dataframe(self, features: Dict, case_id: str = 'case001') -> pd.DataFrame:
        """
        将特征字典转换为DataFrame
        
        参数:
            features (dict): 特征字典
            case_id (str): 病例ID
        
        返回:
            df (pd.DataFrame): 特征DataFrame
        """
        df_dict = {'case_id': case_id}
        df_dict.update(features)
        
        df = pd.DataFrame([df_dict])
        
        return df
    
    def save_features(self, 
                     features: Dict, 
                     output_path: str,
                     case_id: str = 'case001'):
        """
        保存特征到CSV文件
        
        参数:
            features (dict): 特征字典
            output_path (str): 输出文件路径
            case_id (str): 病例ID
        """
        df = self.features_to_dataframe(features, case_id)
        df.to_csv(output_path, index=False)
        
        logger.info(f"特征已保存到: {output_path}")


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试血管形态学特征提取器")
    print("="*60)
    
    # 创建测试数据
    vessel_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    vessel_mask[15:35, 15:35, 15:35] = 1
    
    tumor_mask = np.zeros((50, 50, 50), dtype=np.uint8)
    tumor_mask[10:40, 10:40, 10:40] = 1
    
    # 创建简单的骨架点
    skeleton_points = np.array([
        [20, 20, 20],
        [21, 21, 21],
        [22, 22, 22],
        [23, 23, 23],
        [24, 24, 24],
    ])
    
    branches = [
        [[20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24]]
    ]
    
    junctions = [[22, 22, 22]]
    endpoints = [[20, 20, 20], [24, 24, 24]]
    
    # 初始化提取器
    extractor = VesselMorphometryExtractor(
        spacing=(1.0, 1.0, 1.0),
        verbose=True
    )
    
    # 提取特征
    features = extractor.extract_all_features(
        vessel_mask=vessel_mask,
        tumor_mask=tumor_mask,
        skeleton_points=skeleton_points,
        branches=branches,
        junctions=junctions,
        endpoints=endpoints
    )
    
    # 打印特征
    print("\n" + "="*60)
    print("提取的特征:")
    print("="*60)
    
    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"  {key:30s}: {value:.6f}")
        else:
            print(f"  {key:30s}: {value}")
    
    # 保存特征
    extractor.save_features(features, 'test_features.csv', case_id='test001')
    
    print("\n✓ 测试完成!")
