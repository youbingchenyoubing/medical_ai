import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """影像组学特征提取器"""
    
    def __init__(self, config: dict):
        """
        初始化特征提取器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # 设置参数
        params = config['feature_extraction']
        self.extractor.settings['binWidth'] = params['bin_width']
        self.extractor.settings['resampledPixelSpacing'] = params['resampled_spacing']
        self.extractor.settings['interpolator'] = params['interpolator']
        self.extractor.settings['normalize'] = params['normalize']
        self.extractor.settings['force2D'] = params['force2D']
        
        logger.info("FeatureExtractor initialized")
        logger.info(f"Settings: {self.extractor.settings}")
    
    def extract_features_single_case(
        self, 
        image_path: str, 
        mask_path: str
    ) -> Optional[Dict]:
        """
        提取单个病例的特征
        
        Args:
            image_path: 图像路径
            mask_path: mask路径
            
        Returns:
            特征字典
        """
        try:
            # 读取图像和mask
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            
            # 提取特征
            features = self.extractor.execute(image, mask)
            
            # 过滤掉元数据，只保留特征值
            feature_dict = {}
            for key, value in features.items():
                if key.startswith('original_'):
                    feature_dict[key] = float(value) if not np.isnan(value) else 0.0
            
            logger.info(f"Extracted {len(feature_dict)} features from {image_path}")
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def extract_features_batch(
        self, 
        image_dir: str, 
        mask_dir: str, 
        output_csv: str
    ) -> pd.DataFrame:
        """
        批量提取特征
        
        Args:
            image_dir: 图像目录
            mask_dir: mask目录
            output_csv: 输出CSV文件路径
            
        Returns:
            特征DataFrame
        """
        logger.info(f"Batch extracting features from {image_dir}")
        
        all_features = []
        
        # 获取所有图像文件
        image_files = []
        for f in os.listdir(image_dir):
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                image_files.append(f)
        
        logger.info(f"Found {len(image_files)} images")
        
        # 提取特征
        for image_file in tqdm(image_files, desc="Extracting features"):
            case_id = image_file.replace('.nii.gz', '').replace('.nii', '')
            image_path = os.path.join(image_dir, image_file)
            
            # 查找对应的mask
            mask_file = f"{case_id}_mask.nii.gz"
            mask_path = os.path.join(mask_dir, mask_file)
            
            if not os.path.exists(mask_path):
                logger.warning(f"Mask not found for {case_id}")
                continue
            
            features = self.extract_features_single_case(image_path, mask_path)
            
            if features:
                features['case_id'] = case_id
                all_features.append(features)
        
        # 转换为DataFrame
        df = pd.DataFrame(all_features)
        
        # 保存
        ensure_dir(os.path.dirname(output_csv))
        df.to_csv(output_csv, index=False)
        
        logger.info(f"Features saved to {output_csv}")
        logger.info(f"Total cases: {len(df)}, Total features: {len(df.columns) - 1}")
        
        return df
    
    def extract_features_with_labels(
        self,
        image_dir: str,
        mask_dir: str,
        label_csv: str,
        output_csv: str
    ) -> pd.DataFrame:
        """
        提取特征并合并标签
        
        Args:
            image_dir: 图像目录
            mask_dir: mask目录
            label_csv: 标签CSV文件
            output_csv: 输出CSV文件路径
            
        Returns:
            特征DataFrame
        """
        # 提取特征
        features_df = self.extract_features_batch(image_dir, mask_dir, output_csv)
        
        # 加载标签
        labels_df = pd.read_csv(label_csv)
        
        # 合并
        merged_df = features_df.merge(labels_df, on='case_id', how='left')
        
        # 保存
        merged_df.to_csv(output_csv, index=False)
        
        logger.info(f"Merged features with labels. Total cases: {len(merged_df)}")
        
        return merged_df
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        # 返回所有启用的特征类别
        feature_names = []
        
        for feature_class in self.extractor.enabledFeatures.keys():
            if not self.extractor.enabledFeatures[feature_class]:
                # 如果为空，表示启用该类别的所有特征
                feature_names.append(f"{feature_class}_*")
            else:
                # 否则添加具体特征
                for feature in self.extractor.enabledFeatures[feature_class]:
                    feature_names.append(f"original_{feature_class}_{feature}")
        
        return feature_names
