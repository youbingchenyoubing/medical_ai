import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """医学影像数据预处理器"""
    
    def __init__(self, config: dict):
        """
        初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.target_spacing = config['preprocessing']['target_spacing']
        self.window_level = config['preprocessing']['window_level']
        self.window_width = config['preprocessing']['window_width']
        self.normalize = config['preprocessing']['normalize']
        self.clip_values = config['preprocessing']['clip_values']
        
        logger.info("DataPreprocessor initialized")
    
    def preprocess_ct_image(self, image_path: str, output_path: Optional[str] = None) -> sitk.Image:
        """
        CT图像预处理
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            预处理后的图像
        """
        logger.info(f"Processing: {image_path}")
        
        # 读取DICOM系列或NIfTI文件
        if os.path.isdir(image_path):
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(image_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        else:
            image = sitk.ReadImage(image_path)
        
        # 重采样
        image = self._resample_image(image)
        
        # 窗宽窗位调整
        image = self._apply_windowing(image)
        
        # 裁剪值范围
        image = self._clip_image(image)
        
        # 归一化
        if self.normalize:
            image = self._normalize_image(image)
        
        # 保存
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            sitk.WriteImage(image, output_path)
            logger.info(f"Saved to: {output_path}")
        
        return image
    
    def _resample_image(self, image: sitk.Image) -> sitk.Image:
        """重采样图像"""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / self.target_spacing[i])))
            for i in range(3)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(self.target_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetDefaultPixelValue(-1000)
        
        return resampler.Execute(image)
    
    def _apply_windowing(self, image: sitk.Image) -> sitk.Image:
        """应用窗宽窗位"""
        min_val = self.window_level - self.window_width / 2
        max_val = self.window_level + self.window_width / 2
        
        return sitk.Clamp(image, lowerBound=min_val, upperBound=max_val)
    
    def _clip_image(self, image: sitk.Image) -> sitk.Image:
        """裁剪图像值范围"""
        return sitk.Clamp(
            image,
            lowerBound=self.clip_values[0],
            upperBound=self.clip_values[1]
        )
    
    def _normalize_image(self, image: sitk.Image) -> sitk.Image:
        """归一化图像"""
        image_array = sitk.GetArrayFromImage(image)
        
        min_val = image_array.min()
        max_val = image_array.max()
        
        normalized_array = (image_array - min_val) / (max_val - min_val)
        
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(image)
        
        return normalized_image
    
    def batch_preprocess(self, raw_dir: str, output_dir: str) -> None:
        """
        批量预处理
        
        Args:
            raw_dir: 原始数据目录
            output_dir: 输出目录
        """
        logger.info(f"Batch preprocessing from {raw_dir} to {output_dir}")
        
        ensure_dir(output_dir)
        
        # 获取所有病例
        cases = []
        for item in os.listdir(raw_dir):
            item_path = os.path.join(raw_dir, item)
            if os.path.isdir(item_path) or item.endswith('.nii') or item.endswith('.nii.gz'):
                cases.append(item)
        
        logger.info(f"Found {len(cases)} cases")
        
        # 处理每个病例
        for case in tqdm(cases, desc="Preprocessing"):
            input_path = os.path.join(raw_dir, case)
            
            # 确定输出文件名
            if os.path.isdir(input_path):
                output_filename = f"{case}.nii.gz"
            else:
                output_filename = case
            
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                self.preprocess_ct_image(input_path, output_path)
            except Exception as e:
                logger.error(f"Error processing {case}: {str(e)}")
                continue
        
        logger.info(f"Batch preprocessing completed. Processed {len(cases)} cases")
    
    def preprocess_mri_image(self, image_path: str, output_path: Optional[str] = None) -> sitk.Image:
        """
        MRI图像预处理
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            预处理后的图像
        """
        logger.info(f"Processing MRI: {image_path}")
        
        # 读取图像
        image = sitk.ReadImage(image_path)
        
        # 偏置场校正
        image = self._n4_bias_correction(image)
        
        # 重采样
        image = self._resample_image(image)
        
        # 归一化
        if self.normalize:
            image = self._normalize_image(image)
        
        # 保存
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            sitk.WriteImage(image, output_path)
            logger.info(f"Saved to: {output_path}")
        
        return image
    
    def _n4_bias_correction(self, image: sitk.Image) -> sitk.Image:
        """N4偏置场校正"""
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        return corrector.Execute(image)
