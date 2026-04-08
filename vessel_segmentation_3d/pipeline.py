"""
血管分割与三维重建主流程
==========================

完整的端到端流程，整合所有模块：
1. 图像预处理
2. 肿瘤血管分割
3. 血管骨架化
4. 拓扑分析
5. 形态量化
6. 结果保存

作者：医学影像AI研究团队
日期：2026-04-08
版本：v1.0
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import SimpleITK as sitk
import torch

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入自定义模块
from models.unet3d import UNet3D
from segmentation.tumor_vessel_seg import CoSegmentationNet, TumorVesselSegmenter
from skeletonization.morphological import skeletonize_vessel_morphological
from skeletonization.topology_analysis import analyze_vessel_topology
from morphometry.feature_extractor import VesselMorphometryExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VesselSegmentationReconstructionPipeline:
    """
    血管分割与三维重建完整流程
    
    端到端处理医学影像数据，提取血管形态学特征。
    
    参数:
        config (dict): 配置字典
        device (str): 计算设备 ('cuda' 或 'cpu')
    
    示例:
        >>> config = {
        ...     'preprocessing': {'target_spacing': [1.0, 1.0, 1.0]},
        ...     'segmentation': {'model_path': 'model.pth'},
        ...     'output_dir': 'results/'
        ... }
        >>> pipeline = VesselSegmentationReconstructionPipeline(config)
        >>> features = pipeline.run_pipeline('image.nii.gz', 'case001')
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 初始化各模块
        self.segmenter = None
        self.feature_extractor = VesselMorphometryExtractor(
            spacing=tuple(config.get('spacing', [1.0, 1.0, 1.0]))
        )
        
        # 创建输出目录
        self.setup_directories()
        
        logger.info(f"初始化流程，设备: {self.device}")
    
    def setup_directories(self):
        """创建输出目录结构"""
        output_dir = Path(self.config.get('output_dir', 'results'))
        
        self.dirs = {
            'segmentations': output_dir / 'segmentations',
            'skeletons': output_dir / 'skeletons',
            'features': output_dir / 'features',
            'visualizations': output_dir / 'visualizations',
            'logs': output_dir / 'logs'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"输出目录: {output_dir}")
    
    def run_pipeline(self, image_path: str, case_id: str) -> Dict:
        """
        运行完整流程
        
        参数:
            image_path (str): 输入图像路径
            case_id (str): 病例ID
        
        返回:
            features (dict): 提取的特征
        """
        start_time = datetime.now()
        
        logger.info("="*70)
        logger.info(f"开始处理病例: {case_id}")
        logger.info("="*70)
        
        try:
            # 步骤1：加载和预处理
            image, original_image = self.step1_load_and_preprocess(image_path)
            
            # 步骤2：分割
            tumor_mask, vessel_mask = self.step2_segment(image, original_image)
            
            # 步骤3：骨架化
            skeleton_points, skeleton = self.step3_skeletonize(vessel_mask)
            
            # 步骤4：拓扑分析
            graph, branches, junctions, endpoints = self.step4_topology_analysis(skeleton_points)
            
            # 步骤5：形态量化
            features = self.step5_morphometry(
                vessel_mask, tumor_mask, preprocessed,
                skeleton_points, branches, junctions, endpoints, graph
            )
            
            # 步骤6：保存结果
            self.step6_save_results(
                case_id, original_image,
                tumor_mask, vessel_mask, skeleton,
                features
            )
            
            # 计算处理时间
            elapsed_time = (datetime.now() - start_time).total_seconds()
            features['processing_time_seconds'] = elapsed_time
            
            logger.info("="*70)
            logger.info(f"病例 {case_id} 处理完成")
            logger.info(f"总耗时: {elapsed_time:.2f} 秒")
            logger.info("="*70)
            
            return features
            
        except Exception as e:
            logger.error(f"处理病例 {case_id} 时出错: {str(e)}")
            raise
    
    def step1_load_and_preprocess(self, image_path: str) -> Tuple[np.ndarray, sitk.Image]:
        """
        步骤1：加载和预处理图像
        
        参数:
            image_path (str): 图像路径
        
        返回:
            preprocessed (np.ndarray): 预处理后的图像数组
            original (sitk.Image): 原始SimpleITK图像
        """
        logger.info("\n" + "="*60)
        logger.info("步骤1: 加载和预处理图像")
        logger.info("="*60)
        
        # 加载图像
        logger.info(f"加载图像: {image_path}")
        original = sitk.ReadImage(image_path)
        
        # 获取图像信息
        size = original.GetSize()
        spacing = original.GetSpacing()
        origin = original.GetOrigin()
        
        logger.info(f"图像大小: {size}")
        logger.info(f"体素间距: {spacing}")
        logger.info(f"图像原点: {origin}")
        
        # 转换为numpy数组
        image_array = sitk.GetArrayFromImage(original)
        logger.info(f"数组形状: {image_array.shape}")
        logger.info(f"数值范围: [{image_array.min():.2f}, {image_array.max():.2f}]")
        
        # 预处理
        preprocessed = self._preprocess_image(original)
        
        return preprocessed, original
    
    def _preprocess_image(self, image: sitk.Image) -> np.ndarray:
        """
        图像预处理
        
        包括：
        - 重采样
        - 窗宽窗位调整
        - 归一化
        """
        config = self.config.get('preprocessing', {})
        
        # 重采样
        target_spacing = config.get('target_spacing', [1.0, 1.0, 1.0])
        if target_spacing:
            logger.info(f"重采样到: {target_spacing}")
            
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            
            # 计算新大小
            new_size = [
                int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
                for i in range(3)
            ]
            
            # 重采样
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(target_spacing)
            resampler.SetSize(new_size)
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetDefaultPixelValue(-1000)
            
            image = resampler.Execute(image)
        
        # 窗宽窗位调整
        window_level = config.get('window_level', -600)
        window_width = config.get('window_width', 1500)
        
        logger.info(f"窗宽窗位: level={window_level}, width={window_width}")
        
        min_val = window_level - window_width / 2
        max_val = window_level + window_width / 2
        
        image = sitk.Clamp(image, lowerBound=min_val, upperBound=max_val)
        
        # 归一化
        image_array = sitk.GetArrayFromImage(image)
        image_array = (image_array - min_val) / (max_val - min_val)
        image_array = np.clip(image_array, 0, 1)
        
        logger.info(f"预处理后形状: {image_array.shape}")
        logger.info(f"预处理后范围: [{image_array.min():.4f}, {image_array.max():.4f}]")
        
        return image_array
    
    def step2_segment(self, 
                      image: np.ndarray,
                      original_image: sitk.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        步骤2：肿瘤和血管分割
        
        参数:
            image (np.ndarray): 预处理后的图像
            original_image (sitk.Image): 原始图像（用于获取spacing）
        
        返回:
            tumor_mask (np.ndarray): 肿瘤mask
            vessel_mask (np.ndarray): 血管mask
        """
        logger.info("\n" + "="*60)
        logger.info("步骤2: 肿瘤和血管分割")
        logger.info("="*60)
        
        # 初始化分割器
        if self.segmenter is None:
            model_path = self.config.get('segmentation', {}).get('model_path')
            self.segmenter = TumorVesselSegmenter(
                model_path=model_path,
                device=str(self.device)
            )
        
        # 执行分割 - 转换为 torch.Tensor
        import torch
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()  # (1, 1, D, H, W)
        tumor_mask, vessel_mask = self.segmenter.segment(image_tensor)
        
        # 统计
        tumor_voxels = np.sum(tumor_mask > 0)
        vessel_voxels = np.sum(vessel_mask > 0)
        
        logger.info(f"肿瘤体素: {tumor_voxels}")
        logger.info(f"血管体素: {vessel_voxels}")
        
        return tumor_mask, vessel_mask
    
    def step3_skeletonize(self, vessel_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        步骤3：血管骨架化
        
        参数:
            vessel_mask (np.ndarray): 血管mask
        
        返回:
            skeleton_points (np.ndarray): 骨架点坐标
            skeleton (np.ndarray): 骨架图像
        """
        logger.info("\n" + "="*60)
        logger.info("步骤3: 血管骨架化")
        logger.info("="*60)
        
        # 骨架化
        skeleton_points, skeleton = skeletonize_vessel_morphological(vessel_mask)
        
        logger.info(f"骨架点数量: {len(skeleton_points)}")
        
        return skeleton_points, skeleton
    
    def step4_topology_analysis(self, 
                                 skeleton_points: np.ndarray) -> Tuple:
        """
        步骤4：拓扑结构分析
        
        参数:
            skeleton_points (np.ndarray): 骨架点
        
        返回:
            graph: 拓扑图
            branches: 分支列表
            junctions: 分叉点列表
            endpoints: 端点列表
        """
        logger.info("\n" + "="*60)
        logger.info("步骤4: 拓扑结构分析")
        logger.info("="*60)
        
        # 分析拓扑
        graph, branches, junctions, endpoints = analyze_vessel_topology(skeleton_points)
        
        logger.info(f"分支数量: {len(branches)}")
        logger.info(f"分叉点数量: {len(junctions)}")
        logger.info(f"端点数量: {len(endpoints)}")
        
        return graph, branches, junctions, endpoints
    
    def step5_morphometry(self,
                         vessel_mask: np.ndarray,
                         tumor_mask: np.ndarray,
                         preprocessed: np.ndarray,
                         skeleton_points: np.ndarray,
                         branches: list,
                         junctions: list,
                         endpoints: list,
                         graph) -> Dict:
        """
        步骤5：形态学量化
        
        参数:
            vessel_mask: 血管mask
            tumor_mask: 肿瘤mask
            preprocessed: 预处理后的图像
            skeleton_points: 骨架点
            branches: 分支
            junctions: 分叉点
            endpoints: 端点
            graph: 拓扑图
        
        返回:
            features: 特征字典
        """
        logger.info("\n" + "="*60)
        logger.info("步骤5: 形态学量化")
        logger.info("="*60)
        
        # 提取传统形态学特征
        features = self.feature_extractor.extract_all_features(
            vessel_mask=vessel_mask,
            tumor_mask=tumor_mask,
            skeleton_points=skeleton_points,
            branches=branches,
            junctions=junctions,
            endpoints=endpoints,
            graph=graph
        )
        
        # 提取放射组学特征
        try:
            from morphometry.radiomics_extractor import RadiomicsFeatureExtractor
            
            logger.info("\n[7/7] 提取放射组学特征...")
            radiomics_extractor = RadiomicsFeatureExtractor()
            
            # 提取肿瘤区域的放射组学特征
            tumor_radiomics = radiomics_extractor.extract_features_from_arrays(
                preprocessed, tumor_mask, spacing=(1.0, 1.0, 1.0)
            )
            
            # 提取血管区域的放射组学特征
            vessel_radiomics = radiomics_extractor.extract_features_from_arrays(
                preprocessed, vessel_mask, spacing=(1.0, 1.0, 1.0)
            )
            
            # 为特征添加前缀以区分
            tumor_radiomics_features = {f'tumor_{k}': v for k, v in tumor_radiomics.items()}
            vessel_radiomics_features = {f'vessel_{k}': v for k, v in vessel_radiomics.items()}
            
            # 合并特征
            features.update(tumor_radiomics_features)
            features.update(vessel_radiomics_features)
            
            logger.info(f"成功提取 {len(tumor_radiomics) + len(vessel_radiomics)} 个放射组学特征")
            
        except ImportError as e:
            logger.warning(f"PyRadiomics 未安装，跳过放射组学特征提取: {e}")
        except Exception as e:
            logger.error(f"放射组学特征提取失败: {e}")
        
        logger.info(f"总提取特征数量: {len(features)}")
        
        return features
    
    def step6_save_results(self,
                          case_id: str,
                          original_image: sitk.Image,
                          tumor_mask: np.ndarray,
                          vessel_mask: np.ndarray,
                          skeleton: np.ndarray,
                          features: Dict):
        """
        步骤6：保存结果
        
        参数:
            case_id: 病例ID
            original_image: 原始图像
            tumor_mask: 肿瘤mask
            vessel_mask: 血管mask
            skeleton: 骨架
            features: 特征
        """
        logger.info("\n" + "="*60)
        logger.info("步骤6: 保存结果")
        logger.info("="*60)
        
        # 保存分割结果
        tumor_image = sitk.GetImageFromArray(tumor_mask.astype(np.uint8))
        tumor_image.CopyInformation(original_image)
        tumor_path = self.dirs['segmentations'] / f"{case_id}_tumor.nii.gz"
        sitk.WriteImage(tumor_image, str(tumor_path))
        logger.info(f"保存肿瘤分割: {tumor_path}")
        
        vessel_image = sitk.GetImageFromArray(vessel_mask.astype(np.uint8))
        vessel_image.CopyInformation(original_image)
        vessel_path = self.dirs['segmentations'] / f"{case_id}_vessel.nii.gz"
        sitk.WriteImage(vessel_image, str(vessel_path))
        logger.info(f"保存血管分割: {vessel_path}")
        
        # 保存骨架
        skeleton_image = sitk.GetImageFromArray(skeleton.astype(np.uint8))
        skeleton_image.CopyInformation(original_image)
        skeleton_path = self.dirs['skeletons'] / f"{case_id}_skeleton.nii.gz"
        sitk.WriteImage(skeleton_image, str(skeleton_path))
        logger.info(f"保存骨架: {skeleton_path}")
        
        # 保存特征
        import pandas as pd
        df = pd.DataFrame([features])
        df.insert(0, 'case_id', case_id)
        features_path = self.dirs['features'] / f"{case_id}_features.csv"
        df.to_csv(features_path, index=False)
        logger.info(f"保存特征: {features_path}")
        
        logger.info("结果保存完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='血管分割与三维重建流程')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--case-id', type=str, required=True, help='病例ID')
    parser.add_argument('--output-dir', type=str, default='results', help='输出目录')
    parser.add_argument('--model-path', type=str, default=None, help='模型权重路径')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'output_dir': args.output_dir,
        'spacing': [1.0, 1.0, 1.0],
        'preprocessing': {
            'target_spacing': [1.0, 1.0, 1.0],
            'window_level': -600,
            'window_width': 1500
        },
        'segmentation': {
            'model_path': args.model_path
        }
    }
    
    # 如果提供了配置文件，加载它
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config.update(yaml.safe_load(f))
    
    # 初始化流程
    pipeline = VesselSegmentationReconstructionPipeline(config, device=args.device)
    
    # 运行
    features = pipeline.run_pipeline(args.image, args.case_id)
    
    # 打印特征摘要
    print("\n" + "="*60)
    print("特征摘要")
    print("="*60)
    
    for key, value in sorted(features.items()):
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")
    
    print("\n✓ 处理完成!")


if __name__ == "__main__":
    main()
