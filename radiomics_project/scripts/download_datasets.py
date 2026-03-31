#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公开数据集下载脚本

支持的数据集:
- LIDC-IDRI: 肺结节CT数据集
- NSCLC-Radiomics: 非小细胞肺癌影像组学数据集
- LiTS: 肝脏肿瘤分割数据集
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        初始化下载器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"DatasetDownloader initialized")
        print(f"Output directory: {output_dir}")
    
    def download_file(self, url: str, filename: str) -> str:
        """
        下载文件（带进度条）
        
        Args:
            url: 下载链接
            filename: 文件名
            
        Returns:
            文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            print(f"File already exists: {filepath}")
            return filepath
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"Downloaded: {filepath}")
        return filepath
    
    def download_lidc_idri(self):
        """
        下载LIDC-IDRI数据集
        
        数据集信息:
        - 病例数: 1,018例
        - 类型: 胸部CT
        - 标注: 肺结节
        - 大小: ~120GB
        """
        print("\n" + "="*70)
        print("Downloading LIDC-IDRI Dataset")
        print("="*70)
        
        print("\nDataset Information:")
        print("  - Name: LIDC-IDRI")
        print("  - Cases: 1,018")
        print("  - Type: Chest CT")
        print("  - Size: ~120GB")
        print("  - Annotations: Lung nodules")
        
        print("\nDownload Methods:")
        print("\n1. Official TCIA Website:")
        print("   https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI")
        
        print("\n2. Using TCIA Client (Recommended):")
        print("   pip install tcia-client")
        print("   ")
        print("   from tciaclient import TCIAClient")
        print("   tc = TCIAClient()")
        print("   tc.get_image(collection='LIDC-IDRI', downloadPath='data/raw/LIDC-IDRI')")
        
        print("\n3. Using NBIA Data Retriever:")
        print("   Download from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
        
        print("\n" + "="*70)
    
    def download_nsclc_radiomics(self):
        """
        下载NSCLC-Radiomics数据集
        
        数据集信息:
        - 病例数: 422例
        - 类型: NSCLC CT
        - 包含: 图像、分割、临床数据
        - 大小: ~30GB
        """
        print("\n" + "="*70)
        print("Downloading NSCLC-Radiomics Dataset")
        print("="*70)
        
        print("\nDataset Information:")
        print("  - Name: NSCLC-Radiomics")
        print("  - Cases: 422")
        print("  - Type: NSCLC CT")
        print("  - Size: ~30GB")
        print("  - Includes: Images, Segmentations, Clinical data")
        
        print("\nDownload Methods:")
        print("\n1. Official TCIA Website:")
        print("   https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics")
        
        print("\n2. Using TCIA Client:")
        print("   from tciaclient import TCIAClient")
        print("   tc = TCIAClient()")
        print("   tc.get_image(collection='NSCLC-Radiomics', downloadPath='data/raw/NSCLC-Radiomics')")
        
        print("\n" + "="*70)
    
    def download_lits(self):
        """
        下载LiTS数据集
        
        数据集信息:
        - 病例数: 201例 (131训练 + 70测试)
        - 类型: 腹部CT
        - 标注: 肝脏和肿瘤分割
        - 大小: ~20GB
        """
        print("\n" + "="*70)
        print("Downloading LiTS Dataset")
        print("="*70)
        
        print("\nDataset Information:")
        print("  - Name: LiTS (Liver Tumor Segmentation)")
        print("  - Cases: 201 (131 train + 70 test)")
        print("  - Type: Abdominal CT")
        print("  - Size: ~20GB")
        print("  - Annotations: Liver and tumor segmentation")
        
        print("\nDownload Methods:")
        print("\n1. Official Challenge Website:")
        print("   https://competitions.codalab.org/competitions/17094")
        print("   (Registration required)")
        
        print("\n2. Alternative Sources:")
        print("   - Kaggle: https://www.kaggle.com/datasets")
        print("   - Grand Challenge: https://grand-challenge.org/")
        
        print("\n" + "="*70)
    
    def download_sample_data(self):
        """
        下载示例数据（用于测试）
        """
        print("\n" + "="*70)
        print("Downloading Sample Data for Testing")
        print("="*70)
        
        # 创建示例数据目录
        sample_dir = os.path.join(self.output_dir, "sample_data")
        os.makedirs(sample_dir, exist_ok=True)
        
        print("\nCreating sample clinical data...")
        
        # 创建示例临床数据
        import pandas as pd
        import numpy as np
        
        n_samples = 10
        clinical_data = pd.DataFrame({
            'case_id': [f'case_{i:03d}' for i in range(n_samples)],
            'age': np.random.randint(40, 80, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'label': np.random.randint(0, 2, n_samples)
        })
        
        clinical_path = os.path.join(sample_dir, "clinical.csv")
        clinical_data.to_csv(clinical_path, index=False)
        
        print(f"Sample clinical data saved to: {clinical_path}")
        print("\nNote: This is sample data for testing the pipeline.")
        print("For real research, please download actual datasets.")
        
        print("\n" + "="*70)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Download Medical Imaging Datasets')
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['all', 'lidc', 'nsclc', 'lits', 'sample'],
                       help='Dataset to download')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(output_dir=args.output)
    
    if args.dataset == 'all':
        print("\nDownloading all datasets...")
        downloader.download_lidc_idri()
        downloader.download_nsclc_radiomics()
        downloader.download_lits()
    elif args.dataset == 'lidc':
        downloader.download_lidc_idri()
    elif args.dataset == 'nsclc':
        downloader.download_nsclc_radiomics()
    elif args.dataset == 'lits':
        downloader.download_lits()
    elif args.dataset == 'sample':
        downloader.download_sample_data()

if __name__ == "__main__":
    main()
