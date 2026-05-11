
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版示例数据生成工具
用于在公开数据集无法下载时，生成完整的测试数据，包括：
- 模拟的医学影像 (DICOM 和 NIfTI 格式)
- 肿瘤分割掩码
- 临床数据 (CSV)
- 影像组学特征 (预计算)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_sample_data(output_dir: str = "data/sample", num_patients: int = 20):
    """
    生成完整的示例数据集
    
    Args:
        output_dir: 输出目录
        num_patients: 患者数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating sample data in {output_dir}...")
    print(f"Number of patients: {num_patients}")
    
    # 1. 生成临床数据
    clinical_data = _generate_clinical_data(num_patients)
    clinical_csv = output_path / "clinical_data.csv"
    clinical_data.to_csv(clinical_csv, index=False)
    print(f"✓ Clinical data saved to {clinical_csv}")
    
    # 2. 生成模拟影像和分割
    images_dir = output_path / "images"
    masks_dir = output_path / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    for patient_id in clinical_data["patient_id"]:
        _generate_simulated_image(patient_id, images_dir, masks_dir)
    print(f"✓ Simulated images and masks generated")
    
    # 3. 生成预计算的影像组学特征
    features = _generate_radiomics_features(clinical_data)
    features_csv = output_path / "radiomics_features.csv"
    features.to_csv(features_csv, index=False)
    print(f"✓ Radiomics features saved to {features_csv}")
    
    # 4. 生成 README
    _generate_readme(output_path, num_patients)
    
    print(f"\n✅ Sample data generation complete!")
    print(f"   Data directory: {output_dir}")
    print(f"\nYou can now use this sample data to test your radiomics pipeline.")
    print(f"For real research, please download actual datasets following the guide:")
    print(f"  docs/DATA_DOWNLOAD_GUIDE.md")

def _generate_clinical_data(num_patients: int) -> pd.DataFrame:
    """生成模拟的临床数据"""
    np.random.seed(42)
    
    patient_ids = [f"Patient_{i:03d}" for i in range(1, num_patients + 1)]
    
    data = {
        "patient_id": patient_ids,
        "age": np.random.randint(40, 85, num_patients),
        "gender": np.random.choice(["M", "F"], num_patients),
        "histology": np.random.choice(["adenocarcinoma", "squamous", "NSCLC-NOS"], num_patients),
        "stage": np.random.choice(["I", "II", "III", "IV"], num_patients, p=[0.3, 0.3, 0.25, 0.15]),
        "tumor_size": np.random.uniform(1.0, 7.0, num_patients).round(1),
        "treatment": np.random.choice(["surgery", "chemoradiation", "palliative"], num_patients),
        "survival_months": np.random.uniform(6, 60, num_patients).round(1),
        "event": np.random.choice([0, 1], num_patients, p=[0.4, 0.6]),  # 0=censored, 1=event
    }
    
    return pd.DataFrame(data)

def _generate_simulated_image(patient_id: str, images_dir: Path, masks_dir: Path):
    """生成模拟的影像和分割掩码（保存为 numpy 格式，易于加载）"""
    
    # 生成模拟 CT 影像 (512x512x64)
    np.random.seed(hash(patient_id) % (2**32))
    image_shape = (64, 128, 128)
    
    # 模拟背景
    image = np.random.normal(-1000, 100, image_shape)
    
    # 模拟组织
    tissue_mask = np.random.random(image_shape) > 0.7
    image[tissue_mask] = np.random.normal(0, 50, np.sum(tissue_mask))
    
    # 模拟肿瘤
    tumor_center = (
        np.random.randint(20, 44),
        np.random.randint(40, 88),
        np.random.randint(40, 88)
    )
    
    # 创建球形肿瘤
    z, y, x = np.ogrid[:image_shape[0], :image_shape[1], :image_shape[2]]
    dist_from_center = np.sqrt(
        (z - tumor_center[0])**2 + 
        (y - tumor_center[1])**2 + 
        (x - tumor_center[2])**2
    )
    tumor_radius = np.random.uniform(8, 20)
    tumor_mask = dist_from_center < tumor_radius
    
    image[tumor_mask] = np.random.normal(40, 30, np.sum(tumor_mask))
    
    # 保存影像和掩码
    np.save(images_dir / f"{patient_id}_image.npy", image.astype(np.int16))
    np.save(masks_dir / f"{patient_id}_mask.npy", tumor_mask.astype(np.uint8))

def _generate_radiomics_features(clinical_data: pd.DataFrame) -> pd.DataFrame:
    """生成模拟的影像组学特征"""
    np.random.seed(42)
    num_patients = len(clinical_data)
    num_features = 100
    
    features = pd.DataFrame({"patient_id": clinical_data["patient_id"]})
    
    # 生成模拟的形状特征
    shape_features = [
        "original_shape_Elongation", "original_shape_Flatness", 
        "original_shape_LeastAxisLength", "original_shape_MajorAxisLength",
        "original_shape_Maximum2DDiameterColumn", "original_shape_Maximum2DDiameterRow",
        "original_shape_Maximum2DDiameterSlice", "original_shape_Maximum3DDiameter",
        "original_shape_MeshVolume", "original_shape_MinorAxisLength",
        "original_shape_Sphericity", "original_shape_SurfaceArea",
        "original_shape_SurfaceVolumeRatio", "original_shape_VoxelVolume"
    ]
    for feat in shape_features:
        features[feat] = np.random.uniform(0.1, 100.0, num_patients)
    
    # 生成模拟的一阶统计特征
    first_order_features = [
        "original_firstorder_10Percentile", "original_firstorder_90Percentile",
        "original_firstorder_Energy", "original_firstorder_Entropy",
        "original_firstorder_InterquartileRange", "original_firstorder_Kurtosis",
        "original_firstorder_Maximum", "original_firstorder_MeanAbsoluteDeviation",
        "original_firstorder_Mean", "original_firstorder_Median",
        "original_firstorder_Minimum", "original_firstorder_Range",
        "original_firstorder_RobustMeanAbsoluteDeviation", "original_firstorder_RootMeanSquared",
        "original_firstorder_Skewness", "original_firstorder_TotalEnergy",
        "original_firstorder_Uniformity", "original_firstorder_Variance"
    ]
    for feat in first_order_features:
        features[feat] = np.random.normal(0, 1, num_patients)
    
    # 生成一些纹理特征
    texture_features = [
        f"original_glcm_{stat}" for stat in [
            "Autocorrelation", "JointAverage", "ClusterProminence", "ClusterShade",
            "ClusterTendency", "Contrast", "Correlation", "DifferenceAverage",
            "DifferenceEntropy", "DifferenceVariance", "JointEnergy", "JointEntropy",
            "Imc1", "Imc2", "Idm", "Idmn", "Id", "Idn", "InverseVariance", "MaximumProbability",
            "SumAverage", "SumEntropy", "SumSquares"
        ]
    ]
    for feat in texture_features[:30]:  # 只取前30个
        features[feat] = np.random.normal(0, 1, num_patients)
    
    return features

def _generate_readme(output_path: Path, num_patients: int):
    """生成 README 文件"""
    readme_content = f"""# Sample Dataset

This is a simulated dataset for testing radiomics pipelines.

## Dataset Information
- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Number of patients: {num_patients}
- Type: Simulated data (for testing purposes only)

## Directory Structure
```
{output_path.name}/
├── clinical_data.csv      # Clinical and outcome data
├── radiomics_features.csv # Pre-computed radiomics features
├── images/                # Simulated CT images (numpy format)
└── masks/                 # Simulated tumor segmentation masks
```

## Important Note
This is **NOT real medical data** and should only be used for:
- Testing your radiomics analysis pipeline
- Code development and debugging
- Educational purposes

For real research, please download actual public datasets following the
instructions in `docs/DATA_DOWNLOAD_GUIDE.md`.
"""
    
    with open(output_path / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate enhanced sample data for radiomics testing"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/sample",
        help="Output directory for sample data"
    )
    parser.add_argument(
        "--num-patients", 
        type=int, 
        default=20,
        help="Number of simulated patients"
    )
    
    args = parser.parse_args()
    
    generate_sample_data(args.output_dir, args.num_patients)

