"""
基于 PyRadiomics 的特征提取模块
==============================

使用 PyRadiomics 库从医学影像中提取放射组学特征。

PyRadiomics 是一个强大的开源库，可从医学影像中提取：
- 一阶统计特征（均值、方差、熵等）
- 形态学特征（体积、表面积、球形度等）
- 纹理特征（GLCM、GLRLM、GLSZM 等）

作者：医学影像AI研究团队
日期：2026-04-08
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Dict, Optional, Tuple
import radiomics
from radiomics import featureextractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RadiomicsFeatureExtractor:
    """
    基于 PyRadiomics 的特征提取器
    
    从医学影像和分割结果中提取放射组学特征。
    
    参数:
        params_path (str): PyRadiomics 参数文件路径
        features_to_extract (list): 要提取的特征类别
    """
    
    def __init__(self, 
                 params_path: Optional[str] = None,
                 features_to_extract: Optional[list] = None):
        """
        初始化放射组学特征提取器
        """
        # 设置 PyRadiomics 日志级别
        radiomics.setVerbosity(logging.ERROR)
        
        # 配置特征提取器
        if params_path and os.path.exists(params_path):
            self.extractor = featureextractor.RadiomicsFeatureExtractor(params_path)
        else:
            # 默认配置
            self.extractor = featureextractor.RadiomicsFeatureExtractor()
            
            # 启用所有特征类别
            if features_to_extract:
                for feature in features_to_extract:
                    self.extractor.enableFeatureClassByName(feature, True)
            else:
                # 启用所有特征
                self.extractor.enableAllFeatures()
        
        logger.info("PyRadiomics 特征提取器初始化完成")
    
    def extract_features(self, 
                        image: sitk.Image, 
                        mask: sitk.Image, 
                        label: int = 1) -> Dict:
        """
        从图像和掩码中提取特征
        
        参数:
            image (sitk.Image): 医学影像
            mask (sitk.Image): 分割掩码
            label (int): 掩码中的标签值
            
        返回:
            features (dict): 提取的特征字典
        """
        try:
            logger.info("开始提取放射组学特征...")
            
            # 提取特征
            features = self.extractor.execute(image, mask, label=label)
            
            # 过滤掉非特征信息
            feature_dict = {}
            for key, value in features.items():
                # 只保留特征值（排除元数据）
                if not key.startswith('diagnostics_'):
                    feature_dict[key] = value
            
            logger.info(f"成功提取 {len(feature_dict)} 个放射组学特征")
            return feature_dict
            
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            return {}
    
    def extract_features_from_arrays(self, 
                                   image_array: np.ndarray, 
                                   mask_array: np.ndarray, 
                                   spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict:
        """
        从 numpy 数组中提取特征
        
        参数:
            image_array (np.ndarray): 图像数组 (D, H, W)
            mask_array (np.ndarray): 掩码数组 (D, H, W)
            spacing (tuple): 体素间距
            
        返回:
            features (dict): 提取的特征字典
        """
        # 转换为 SimpleITK 图像
        image = sitk.GetImageFromArray(image_array)
        mask = sitk.GetImageFromArray(mask_array.astype(np.uint8))
        
        # 设置间距
        image.SetSpacing(spacing)
        mask.SetSpacing(spacing)
        
        # 提取特征
        return self.extract_features(image, mask)
    
    def extract_features_from_files(self, 
                                  image_path: str, 
                                  mask_path: str, 
                                  label: int = 1) -> Dict:
        """
        从文件中提取特征
        
        参数:
            image_path (str): 图像文件路径
            mask_path (str): 掩码文件路径
            label (int): 掩码中的标签值
            
        返回:
            features (dict): 提取的特征字典
        """
        # 读取图像
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        
        # 提取特征
        return self.extract_features(image, mask, label)
    
    def features_to_dataframe(self, features: Dict, case_id: str = 'case001') -> pd.DataFrame:
        """
        将特征字典转换为 DataFrame
        
        参数:
            features (dict): 特征字典
            case_id (str): 病例ID
            
        返回:
            df (pd.DataFrame): 特征 DataFrame
        """
        df_dict = {'case_id': case_id}
        df_dict.update(features)
        
        return pd.DataFrame([df_dict])
    
    def save_features(self, 
                     features: Dict, 
                     output_path: str, 
                     case_id: str = 'case001'):
        """
        保存特征到 CSV 文件
        
        参数:
            features (dict): 特征字典
            output_path (str): 输出文件路径
            case_id (str): 病例ID
        """
        df = self.features_to_dataframe(features, case_id)
        df.to_csv(output_path, index=False)
        
        logger.info(f"放射组学特征已保存到: {output_path}")


def get_default_params() -> dict:
    """
    获取默认的 PyRadiomics 参数配置
    
    返回:
        params (dict): 默认参数配置
    """
    return {
        'binWidth': 25,
        'resampledPixelSpacing': None,
        'interpolator': 'sitkBSpline',
        'normalize': True,
        'normalizeScale': 1000,
        'removeOutliers': 1.0,
        'label': 1,
        'additionalInfo': True
    }


def create_params_file(params: dict, output_path: str):
    """
    创建 PyRadiomics 参数文件
    
    参数:
        params (dict): 参数配置
        output_path (str): 输出文件路径
    """
    import yaml
    
    with open(output_path, 'w') as f:
        yaml.dump(params, f)
    
    logger.info(f"参数文件已创建: {output_path}")


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试放射组学特征提取")
    print("="*60)
    
    # 创建测试数据
    image_array = np.random.rand(32, 64, 64) * 1000
    mask_array = np.zeros((32, 64, 64), dtype=np.uint8)
    mask_array[10:22, 20:44, 20:44] = 1
    
    print(f"测试图像形状: {image_array.shape}")
    print(f"测试掩码形状: {mask_array.shape}")
    
    # 初始化提取器
    extractor = RadiomicsFeatureExtractor()
    
    # 提取特征
    features = extractor.extract_features_from_arrays(
        image_array, mask_array, spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"\n提取特征数量: {len(features)}")
    
    # 打印部分特征
    print("\n部分特征:")
    for i, (key, value) in enumerate(sorted(features.items())):
        if i >= 10:
            break
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # 保存特征
    extractor.save_features(features, 'test_radiomics_features.csv', case_id='test001')
    
    print("\n✓ 测试完成!")
