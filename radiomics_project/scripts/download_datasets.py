#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公开数据集下载脚本

支持的数据集:
肺部/胸部:
- LIDC-IDRI: 肺结节CT数据集 (1,018例)
- NSCLC-Radiomics: 非小细胞肺癌影像组学数据集 (422例)
- NSCLC-Radiogenomics: 非小细胞肺癌影像基因组学数据集 (211例)
- LUNA16: 肺结节分析挑战赛数据集 (888例)
- Lung-PET-CT-Dx: 肺癌PET-CT诊断数据集 (284例)

腹部/肝脏:
- LiTS: 肝脏肿瘤分割数据集 (201例)
- TCGA-LIHC: 肝细胞癌影像基因组学数据集 (186例)
- WAW-TACE: HCC TACE治疗MRI数据集 (117例)

头颈部:
- Head-Neck-PET-CT: 头颈部PET-CT数据集 (298例)
- OPC-Radiomics: 口咽癌影像组学数据集 (606例)
- HNSCC: 头颈鳞状细胞癌数据集 (364例)

乳腺:
- Breast-MRI-NACT-Pilot: 乳腺癌新辅助化疗MRI数据集 (64例)

综合:
- DeepLesion: 多器官多病灶CT数据集 (32,735病灶)
- sample: 随机生成示例数据 (10例)
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

DATASET_REGISTRY = {
    "lidc": {
        "name": "LIDC-IDRI",
        "category": "lung",
        "cases": 1018,
        "modality": "CT",
        "size": "~120GB",
        "description": "胸部CT肺结节数据集，4位医生标注+恶性度评分",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
    },
    "nsclc": {
        "name": "NSCLC-Radiomics",
        "category": "lung",
        "cases": 422,
        "modality": "CT",
        "size": "~30GB",
        "description": "非小细胞肺癌CT，含肿瘤分割+临床+生存数据",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics",
    },
    "nsclc_rgenomics": {
        "name": "NSCLC-Radiogenomics",
        "category": "lung",
        "cases": 211,
        "modality": "CT",
        "size": "~15GB",
        "description": "非小细胞肺癌CT，含肿瘤分割+基因表达+临床数据",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/NSCLC+Radiogenomics",
    },
    "luna16": {
        "name": "LUNA16",
        "category": "lung",
        "cases": 888,
        "modality": "CT",
        "size": "~50GB",
        "description": "肺结节分析挑战赛数据集，LIDC子集剔除<3mm结节",
        "url": "https://luna16.grand-challenge.org/",
    },
    "lung_pet_ct_dx": {
        "name": "Lung-PET-CT-Dx",
        "category": "lung",
        "cases": 284,
        "modality": "CT+PET",
        "size": "~25GB",
        "description": "肺癌PET-CT，含亚型诊断+生存数据",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/Lung+PET-CT+Dx",
    },
    "lits": {
        "name": "LiTS",
        "category": "liver",
        "cases": 201,
        "modality": "CT",
        "size": "~20GB",
        "description": "肝脏肿瘤分割数据集，131训练+70测试",
        "url": "https://competitions.codalab.org/competitions/17094",
    },
    "tcga_lihc": {
        "name": "TCGA-LIHC",
        "category": "liver",
        "cases": 186,
        "modality": "MRI/CT",
        "size": "~10GB",
        "description": "肝细胞癌，含基因+影像+临床数据",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/TCGA-LIHC",
    },
    "waw_tace": {
        "name": "WAW-TACE",
        "category": "liver",
        "cases": 117,
        "modality": "MRI",
        "size": "~8GB",
        "description": "HCC TACE治疗MRI数据集，含临床+疗效标注",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/WAW-TACE",
    },
    "head_neck_pet_ct": {
        "name": "Head-Neck-PET-CT",
        "category": "head_neck",
        "cases": 298,
        "modality": "CT+PET",
        "size": "~30GB",
        "description": "头颈部PET-CT，含肿瘤分割+临床数据",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT",
    },
    "opc_radiomics": {
        "name": "OPC-Radiomics",
        "category": "head_neck",
        "cases": 606,
        "modality": "CT",
        "size": "~20GB",
        "description": "口咽癌CT，含GTV分割+生存+HPV状态",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/OPC-Radiomics",
    },
    "hnscc": {
        "name": "HNSCC",
        "category": "head_neck",
        "cases": 364,
        "modality": "CT",
        "size": "~15GB",
        "description": "头颈鳞状细胞癌CT，含肿瘤分割+临床+生存数据",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/HNSCC",
    },
    "breast_mri_nact": {
        "name": "Breast-MRI-NACT-Pilot",
        "category": "breast",
        "cases": 64,
        "modality": "MRI",
        "size": "~5GB",
        "description": "乳腺癌新辅助化疗MRI，含病理缓解标注",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/Breast-MRI-NACT-Pilot",
    },
    "deeplesion": {
        "name": "DeepLesion",
        "category": "multi_organ",
        "cases": 10594,
        "modality": "CT",
        "size": "~80GB",
        "description": "多器官多病灶CT，32,735个标注病灶",
        "url": "https://wiki.cancerimagingarchive.net/display/Public/DeepLesion",
    },
}


class DatasetDownloader:
    """数据集下载器"""

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"DatasetDownloader initialized")
        print(f"Output directory: {output_dir}")

    def download_file(self, url: str, filename: str) -> str:
        filepath = os.path.join(self.output_dir, filename)

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

    def _print_dataset_info(self, key: str):
        info = DATASET_REGISTRY[key]
        print("\n" + "=" * 70)
        print(f"  {info['name']} Dataset")
        print("=" * 70)
        print(f"\n  Name:        {info['name']}")
        print(f"  Category:    {info['category']}")
        print(f"  Cases:       {info['cases']}")
        print(f"  Modality:    {info['modality']}")
        print(f"  Size:        {info['size']}")
        print(f"  Description: {info['description']}")

    def _print_tcia_download_guide(self, key: str, collection_name: str = None):
        info = DATASET_REGISTRY[key]
        col = collection_name or info['name']

        print("\n  Download Methods:")
        print(f"\n  1. Official TCIA Website:")
        print(f"     {info['url']}")
        print(f"\n  2. Using TCIA Client (Recommended):")
        print(f"     pip install tcia-client")
        print(f"     from tciaclient import TCIAClient")
        print(f"     tc = TCIAClient()")
        print(f"     tc.get_image(collection='{col}', downloadPath='data/raw/{info['name']}')")
        print(f"\n  3. Using NBIA Data Retriever:")
        print(f"     https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")

        print("\n" + "=" * 70)

    def download_lidc_idri(self):
        self._print_dataset_info("lidc")
        print("\n  Special Notes:")
        print("    - 4位放射科医生独立标注肺结节")
        print("    - 每个结节含恶性度评分(1-5)")
        print("    - 适合ICC一致性分析和结节分类研究")
        self._print_tcia_download_guide("lidc", "LIDC-IDRI")

    def download_nsclc_radiomics(self):
        self._print_dataset_info("nsclc")
        print("\n  Special Notes:")
        print("    - 含422例NSCLC患者CT图像")
        print("    - 含肿瘤分割掩码和临床生存数据")
        print("    - 适合影像组学特征提取和生存预测")
        self._print_tcia_download_guide("nsclc", "NSCLC-Radiomics")

    def download_nsclc_radiogenomics(self):
        self._print_dataset_info("nsclc_rgenomics")
        print("\n  Special Notes:")
        print("    - 含211例NSCLC患者CT图像")
        print("    - 含基因表达数据和临床信息")
        print("    - 适合影像基因组学研究和基因突变预测")
        print("\n  Download Methods:")
        print(f"\n  1. Official TCIA Website:")
        print(f"     {DATASET_REGISTRY['nsclc_rgenomics']['url']}")
        print(f"\n  2. Using TCIA Client:")
        print(f"     pip install tcia-client")
        print(f"     from tciaclient import TCIAClient")
        print(f"     tc = TCIAClient()")
        print(f"     tc.get_image(collection='NSCLC+Radiogenomics', downloadPath='data/raw/NSCLC-Radiogenomics')")
        print("\n" + "=" * 70)

    def download_luna16(self):
        self._print_dataset_info("luna16")
        print("\n  Special Notes:")
        print("    - LIDC-IDRI子集，剔除直径<3mm的结节")
        print("    - 含888例CT和1,186个标注结节")
        print("    - 适合肺结节检测和假阳性减少研究")
        print("\n  Download Methods:")
        print(f"\n  1. LUNA16 Challenge Website:")
        print(f"     {DATASET_REGISTRY['luna16']['url']}")
        print(f"\n  2. Alternative: 直接从TCIA下载LIDC-IDRI后使用LUNA16筛选列表")
        print("\n" + "=" * 70)

    def download_lung_pet_ct_dx(self):
        self._print_dataset_info("lung_pet_ct_dx")
        print("\n  Special Notes:")
        print("    - 含284例肺癌患者CT和PET图像")
        print("    - 含组织学亚型诊断和生存数据")
        print("    - 适合多模态影像组学研究")
        self._print_tcia_download_guide("lung_pet_ct_dx", "Lung-PET-CT-Dx")

    def download_lits(self):
        self._print_dataset_info("lits")
        print("\n  Special Notes:")
        print("    - 131训练+70测试，含肝脏和肿瘤分割标注")
        print("    - 适合肝脏肿瘤分割和影像组学特征提取")
        print("\n  Download Methods:")
        print(f"\n  1. Official Challenge Website:")
        print(f"     {DATASET_REGISTRY['lits']['url']}")
        print(f"     (Registration required)")
        print(f"\n  2. Alternative Sources:")
        print(f"     - Kaggle: https://www.kaggle.com/datasets")
        print(f"     - Grand Challenge: https://grand-challenge.org/")
        print("\n" + "=" * 70)

    def download_tcga_lihc(self):
        self._print_dataset_info("tcga_lihc")
        print("\n  Special Notes:")
        print("    - 含186例肝细胞癌患者影像")
        print("    - 含TCGA基因突变、表达和临床数据")
        print("    - 适合影像基因组学和HCC预后预测")
        print("\n  Download Methods:")
        print(f"\n  1. TCIA Website (Imaging):")
        print(f"     {DATASET_REGISTRY['tcga_lihc']['url']}")
        print(f"\n  2. GDC Data Portal (Genomics):")
        print(f"     https://portal.gdc.cancer.gov/projects/TCGA-LIHC")
        print(f"\n  3. Using TCIA Client:")
        print(f"     pip install tcia-client")
        print(f"     from tciaclient import TCIAClient")
        print(f"     tc = TCIAClient()")
        print(f"     tc.get_image(collection='TCGA-LIHC', downloadPath='data/raw/TCGA-LIHC')")
        print("\n" + "=" * 70)

    def download_waw_tace(self):
        self._print_dataset_info("waw_tace")
        print("\n  Special Notes:")
        print("    - 含117例HCC患者TACE治疗前后MRI")
        print("    - 含基线临床变量和疗效标注")
        print("    - 适合HCC治疗反应预测和Delta影像组学")
        print("\n  Download Methods:")
        print(f"\n  1. TCIA Website:")
        print(f"     {DATASET_REGISTRY['waw_tace']['url']}")
        print(f"\n  2. Using TCIA Client:")
        print(f"     pip install tcia-client")
        print(f"     from tciaclient import TCIAClient")
        print(f"     tc = TCIAClient()")
        print(f"     tc.get_image(collection='WAW-TACE', downloadPath='data/raw/WAW-TACE')")
        print("\n" + "=" * 70)

    def download_head_neck_pet_ct(self):
        self._print_dataset_info("head_neck_pet_ct")
        print("\n  Special Notes:")
        print("    - 含298例头颈部癌患者CT和PET图像")
        print("    - 含肿瘤分割和临床数据")
        print("    - 适合放疗疗效预测和多模态影像组学")
        self._print_tcia_download_guide("head_neck_pet_ct", "Head-Neck-PET-CT")

    def download_opc_radiomics(self):
        self._print_dataset_info("opc_radiomics")
        print("\n  Special Notes:")
        print("    - 含606例口咽癌患者CT")
        print("    - 含GTV分割、生存数据和HPV/p16状态")
        print("    - 适合口咽癌生存预测和影像组学")
        self._print_tcia_download_guide("opc_radiomics", "OPC-Radiomics")

    def download_hnscc(self):
        self._print_dataset_info("hnscc")
        print("\n  Special Notes:")
        print("    - 含364例头颈鳞状细胞癌患者CT")
        print("    - 含肿瘤分割、临床和生存数据")
        print("    - 适合头颈癌预后预测")
        self._print_tcia_download_guide("hnscc", "HNSCC")

    def download_breast_mri_nact(self):
        self._print_dataset_info("breast_mri_nact")
        print("\n  Special Notes:")
        print("    - 含64例乳腺癌患者新辅助化疗前后MRI")
        print("    - 含病理完全缓解(pCR)标注")
        print("    - 适合治疗反应预测和Delta影像组学")
        print("\n  Download Methods:")
        print(f"\n  1. TCIA Website:")
        print(f"     {DATASET_REGISTRY['breast_mri_nact']['url']}")
        print(f"\n  2. Using TCIA Client:")
        print(f"     pip install tcia-client")
        print(f"     from tciaclient import TCIAClient")
        print(f"     tc = TCIAClient()")
        print(f"     tc.get_image(collection='Breast-MRI-NACT-Pilot', downloadPath='data/raw/Breast-MRI-NACT')")
        print("\n" + "=" * 70)

    def download_deeplesion(self):
        self._print_dataset_info("deeplesion")
        print("\n  Special Notes:")
        print("    - 含10,594例CT中32,735个标注病灶")
        print("    - 覆盖多器官多病种")
        print("    - 适合大规模病灶检测和分类研究")
        print("\n  Download Methods:")
        print(f"\n  1. TCIA Website:")
        print(f"     {DATASET_REGISTRY['deeplesion']['url']}")
        print(f"\n  2. NIH DeepLesion Page:")
        print(f"     https://nihcc.app.box.com/v/DeepLesion")
        print("\n" + "=" * 70)

    def download_sample_data(self):
        print("\n" + "=" * 70)
        print("  Sample Data for Testing")
        print("=" * 70)

        sample_dir = os.path.join(self.output_dir, "sample_data")
        os.makedirs(sample_dir, exist_ok=True)

        print("\n  Creating sample clinical data...")

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

        print(f"  Sample clinical data saved to: {clinical_path}")
        print("\n  Note: This is sample data for testing the pipeline.")
        print("  For real research, please download actual datasets.")

        print("\n" + "=" * 70)

    def list_datasets(self):
        print("\n" + "=" * 70)
        print("  Available Public Datasets for Radiomics Research")
        print("=" * 70)

        categories = {
            "lung": "Lung / Chest",
            "liver": "Liver / Abdomen",
            "head_neck": "Head & Neck",
            "breast": "Breast",
            "multi_organ": "Multi-Organ",
        }

        for cat_key, cat_name in categories.items():
            print(f"\n  [{cat_name}]")
            for key, info in DATASET_REGISTRY.items():
                if info["category"] == cat_key:
                    print(f"    {key:<22s} | {info['name']:<25s} | {info['cases']:>5d} cases | {info['modality']:<8s} | {info['size']:<8s}")
                    print(f"    {'':22s}   {info['description']}")

        print("\n  [Testing]")
        print(f"    {'sample':<22s} | Sample Data                 |    10 cases | Random   | ~1KB")
        print(f"    {'':22s}   随机生成示例临床数据，用于测试流程")

        print("\n  Usage:")
        print("    python scripts/download_datasets.py --dataset <key>")
        print("    python scripts/download_datasets.py --dataset all")
        print("    python scripts/download_datasets.py --list")
        print("\n" + "=" * 70)


DATASET_DOWNLOADERS = {
    "lidc": "download_lidc_idri",
    "nsclc": "download_nsclc_radiomics",
    "nsclc_rgenomics": "download_nsclc_radiogenomics",
    "luna16": "download_luna16",
    "lung_pet_ct_dx": "download_lung_pet_ct_dx",
    "lits": "download_lits",
    "tcga_lihc": "download_tcga_lihc",
    "waw_tace": "download_waw_tace",
    "head_neck_pet_ct": "download_head_neck_pet_ct",
    "opc_radiomics": "download_opc_radiomics",
    "hnscc": "download_hnscc",
    "breast_mri_nact": "download_breast_mri_nact",
    "deeplesion": "download_deeplesion",
    "sample": "download_sample_data",
}


def main():
    all_choices = list(DATASET_DOWNLOADERS.keys()) + ["all", "list"]

    parser = argparse.ArgumentParser(
        description='Download Medical Imaging Datasets for Radiomics Research',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  Lung/Chest:     lidc, nsclc, nsclc_rgenomics, luna16, lung_pet_ct_dx
  Liver/Abdomen:  lits, tcga_lihc, waw_tace
  Head & Neck:    head_neck_pet_ct, opc_radiomics, hnscc
  Breast:         breast_mri_nact
  Multi-Organ:    deeplesion
  Testing:        sample
  Special:        all (download all), list (show dataset info)

Examples:
  python scripts/download_datasets.py --dataset lidc
  python scripts/download_datasets.py --dataset nsclc
  python scripts/download_datasets.py --dataset waw_tace
  python scripts/download_datasets.py --dataset all
  python scripts/download_datasets.py --list
        """
    )
    parser.add_argument('--dataset', type=str, default='list',
                        choices=all_choices,
                        help='Dataset to download (default: list)')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')

    args = parser.parse_args()

    downloader = DatasetDownloader(output_dir=args.output)

    if args.dataset == 'list':
        downloader.list_datasets()
    elif args.dataset == 'all':
        print("\nDownloading all datasets...")
        for key in DATASET_DOWNLOADERS:
            if key != "sample":
                method_name = DATASET_DOWNLOADERS[key]
                getattr(downloader, method_name)()
    else:
        method_name = DATASET_DOWNLOADERS[args.dataset]
        getattr(downloader, method_name)()


if __name__ == "__main__":
    main()
