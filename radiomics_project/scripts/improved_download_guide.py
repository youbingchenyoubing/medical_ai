#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""改进的公开数据集下载指南 - 提供准确链接和备选方案"""

import os
import sys
import argparse
from pathlib import Path

# 改进的数据集注册表，包含更准确的链接和备选方案
DATASET_REGISTRY = {
    "lidc": {
        "name": "LIDC-IDRI",
        "category": "lung",
        "cases": 1018,
        "modality": "CT",
        "size": "~120GB",
        "description": "胸部CT肺结节数据集，4位医生标注+恶性度评分",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/lidc-idri/",
            "https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI",
        ],
        "tcia_collection": "LIDC-IDRI",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "LIDC-XML-Annotations",
                "urls": ["https://www.cancerimagingarchive.net/collection/lidc-idri/"],
                "note": "肺结节标注XML文件，在Collection页面的Annotations下载",
            }
        ],
    },
    "nsclc": {
        "name": "NSCLC-Radiomics",
        "category": "lung",
        "cases": 422,
        "modality": "CT",
        "size": "~30GB",
        "description": "非小细胞肺癌CT，含肿瘤分割+临床+生存数据",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/",
            "https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics",
        ],
        "tcia_collection": "NSCLC-Radiomics",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "NSCLC-Radiomics-Annotations",
                "urls": ["https://www.cancerimagingarchive.net/collection/nsclc-radiomics/"],
                "note": "肿瘤分割轮廓和临床数据，在Collection页面下载",
            }
        ],
    },
    "nsclc_rgenomics": {
        "name": "NSCLC-Radiogenomics",
        "category": "lung",
        "cases": 211,
        "modality": "CT",
        "size": "~15GB",
        "description": "非小细胞肺癌CT，含肿瘤分割+基因表达+临床数据",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/",
            "https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiogenomics",
        ],
        "tcia_collection": "NSCLC-Radiogenomics",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "NSCLC-Radiogenomics-Clinical-Genomic",
                "urls": ["https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/"],
                "note": "临床和基因表达数据，在Collection页面下载",
            }
        ],
    },
    "luna16": {
        "name": "LUNA16",
        "category": "lung",
        "cases": 888,
        "modality": "CT",
        "size": "~50GB",
        "description": "肺结节分析挑战赛数据集，LIDC子集剔除<3mm结节",
        "urls": [
            "https://luna16.grand-challenge.org/",
            "https://zenodo.org/record/3723295",
        ],
        "tcia_collection": None,
        "requires_registration": True,
        "download_methods": [
            "Grand Challenge平台 (需要注册)",
            "Zenodo镜像站",
        ],
        "mirrors": [
            "https://zenodo.org/record/3723295",
        ],
        "annotations": [],
    },
    "lung_pet_ct_dx": {
        "name": "Lung-PET-CT-Dx",
        "category": "lung",
        "cases": 355,
        "modality": "CT+PET",
        "size": "~25GB",
        "description": "肺癌PET-CT，含亚型诊断+生存数据",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/",
            "https://wiki.cancerimagingarchive.net/display/Public/Lung-PET-CT-Dx",
        ],
        "tcia_collection": "Lung-PET-CT-Dx",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [],
    },
    "lits": {
        "name": "LiTS",
        "category": "liver",
        "cases": 201,
        "modality": "CT",
        "size": "~20GB",
        "description": "肝脏肿瘤分割数据集，131训练+70测试",
        "urls": [
            "https://competitions.codalab.org/competitions/17094",
            "https://zenodo.org/record/7019515",
        ],
        "tcia_collection": None,
        "requires_registration": True,
        "download_methods": [
            "Codalab竞赛平台 (需要注册)",
            "Zenodo镜像站",
        ],
        "mirrors": [
            "https://zenodo.org/record/7019515",
        ],
        "annotations": [],
    },
    "tcga_lihc": {
        "name": "TCGA-LIHC",
        "category": "liver",
        "cases": 186,
        "modality": "MRI/CT",
        "size": "~10GB",
        "description": "肝细胞癌，含基因+影像+临床数据",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/tcga-lihc/",
            "https://wiki.cancerimagingarchive.net/display/Public/TCGA-LIHC",
            "https://portal.gdc.cancer.gov/projects/TCGA-LIHC",
        ],
        "tcia_collection": "TCGA-LIHC",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "GDC Data Portal (基因数据)",
            "TCIA API",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "TCGA-LIHC-Clinical-Genomic",
                "urls": ["https://portal.gdc.cancer.gov/projects/TCGA-LIHC"],
                "note": "基因和临床数据需从GDC Portal下载",
            }
        ],
    },
    "waw_tace": {
        "name": "WAW-TACE",
        "category": "liver",
        "cases": 233,
        "modality": "CT",
        "size": "~8GB",
        "description": "HCC TACE治疗多期CT数据集，含临床+疗效标注",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/waw-tace/",
            "https://wiki.cancerimagingarchive.net/display/Public/WAW-TACE",
        ],
        "tcia_collection": "WAW-TACE",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [],
    },
    "head_neck_pet_ct": {
        "name": "Head-Neck-PET-CT",
        "category": "head_neck",
        "cases": 298,
        "modality": "CT+PET",
        "size": "~30GB",
        "description": "头颈部PET-CT，含肿瘤分割+临床数据",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/",
            "https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT",
        ],
        "tcia_collection": "Head-Neck-PET-CT",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "Head-Neck-PET-CT-Annotations",
                "urls": ["https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/"],
                "note": "肿瘤分割轮廓和临床数据，在Collection页面下载",
            }
        ],
    },
    "opc_radiomics": {
        "name": "OPC-Radiomics",
        "category": "head_neck",
        "cases": 606,
        "modality": "CT",
        "size": "~20GB",
        "description": "口咽癌CT，含GTV分割+生存+HPV状态",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/opc-radiomics/",
            "https://wiki.cancerimagingarchive.net/display/Public/OPC-Radiomics",
        ],
        "tcia_collection": "OPC-Radiomics",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "OPC-Radiomics-Clinical",
                "urls": ["https://www.cancerimagingarchive.net/collection/opc-radiomics/"],
                "note": "临床和生存数据，在Collection页面下载",
            }
        ],
    },
    "hnscc": {
        "name": "HNSCC",
        "category": "head_neck",
        "cases": 364,
        "modality": "CT",
        "size": "~15GB",
        "description": "头颈鳞状细胞癌CT，含肿瘤分割+临床+生存数据",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/hnscc/",
            "https://wiki.cancerimagingarchive.net/display/Public/HNSCC",
        ],
        "tcia_collection": "HNSCC",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "HNSCC-Clinical",
                "urls": ["https://www.cancerimagingarchive.net/collection/hnscc/"],
                "note": "临床和生存数据，在Collection页面下载",
            }
        ],
    },
    "breast_mri_nact": {
        "name": "Breast-MRI-NACT-Pilot",
        "category": "breast",
        "cases": 64,
        "modality": "MRI",
        "size": "~5GB",
        "description": "乳腺癌新辅助化疗MRI，含病理缓解标注",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/breast-mri-nact-pilot/",
            "https://wiki.cancerimagingarchive.net/display/Public/Breast-MRI-NACT-Pilot",
        ],
        "tcia_collection": "Breast-MRI-NACT-Pilot",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "TCIA API",
            "TCIA网站手动下载",
        ],
        "mirrors": [],
        "annotations": [],
    },
    "deeplesion": {
        "name": "DeepLesion",
        "category": "multi_organ",
        "cases": 10594,
        "modality": "CT",
        "size": "~80GB",
        "description": "多器官多病灶CT，32,735个标注病灶",
        "urls": [
            "https://www.cancerimagingarchive.net/collection/deeplesion/",
            "https://wiki.cancerimagingarchive.net/display/Public/DeepLesion",
            "https://nihcc.app.box.com/v/DeepLesion",
        ],
        "tcia_collection": "DeepLesion",
        "requires_registration": False,
        "download_methods": [
            "NBIA Data Retriever (推荐)",
            "NIH Box网盘 (标注文件)",
            "TCIA API",
        ],
        "mirrors": [],
        "annotations": [
            {
                "name": "DeepLesion-Annotations",
                "urls": ["https://nihcc.app.box.com/v/DeepLesion"],
                "note": "标注文件需从NIH Box下载",
            }
        ],
    },
}


def print_general_guide():
    """打印通用下载指南"""
    print("=" * 80)
    print("公开医学影像数据集 - 下载指南")
    print("=" * 80)
    
    print("\n📚 主要数据平台:")
    print("  1. The Cancer Imaging Archive (TCIA)")
    print("     - 网址: https://www.cancerimagingarchive.net/")
    print("     - 主要肿瘤影像数据集")
    
    print("\n🚀 推荐下载工具:")
    print("  1. NBIA Data Retriever (强烈推荐)")
    print("     - 下载地址: https://wiki.cancerimagingarchive.net/display/NBIA")
    print("     - 支持Windows/macOS/Linux")
    print("     - 支持断点续传，适合大文件下载")
    
    print("  2. TCIA Python Client")
    print("     - pip install tcia")
    print("     - 编程方式下载")
    
    print("  3. 手动下载")
    print("     - 在TCIA网站直接下载")
    
    print("\n🔧 网络问题解决方案:")
    print("  1. 检查网络连接")
    print("  2. 尝试使用代理或VPN")
    print("  3. 使用下载工具支持断点续传")
    print("  4. 查找镜像站(如Zenodo)")
    
    print("\n📋 数据集列表:")
    print("  使用 --dataset list 查看所有可用数据集")
    print("=" * 80)


def print_dataset_info(dataset_key):
    """打印单个数据集详细信息"""
    if dataset_key not in DATASET_REGISTRY:
        print(f"错误: 找不到数据集 '{dataset_key}'")
        return
    
    info = DATASET_REGISTRY[dataset_key]
    
    print("\n" + "=" * 80)
    print(f"📦 {info['name']}")
    print("=" * 80)
    
    print(f"\n📋 基本信息:")
    print(f"  分类: {info['category']}")
    print(f"  病例数: {info['cases']}")
    print(f"  模态: {info['modality']}")
    print(f"  大小: {info['size']}")
    print(f"  描述: {info['description']}")
    
    if info.get('requires_registration'):
        print(f"  ⚠️ 需要注册: 是")
    
    print(f"\n🔗 官方链接:")
    for i, url in enumerate(info['urls'], 1):
        print(f"  {i}. {url}")
    
    if info.get('mirrors'):
        print(f"\n🔄 镜像站:")
        for i, url in enumerate(info['mirrors'], 1):
            print(f"  {i}. {url}")
    
    print(f"\n📥 下载方式:")
    for i, method in enumerate(info['download_methods'], 1):
        print(f"  {i}. {method}")
    
    if info.get('tcia_collection'):
        print(f"\n🔌 TCIA Collection ID: {info['tcia_collection']}")
        print(f"   使用NBIA Data Retriever搜索此ID即可下载")
    
    if info.get('annotations'):
        print(f"\n📝 标注/临床数据:")
        for i, ann in enumerate(info['annotations'], 1):
            print(f"  {i}. {ann['name']}")
            print(f"     说明: {ann['note']}")
            for url in ann['urls']:
                print(f"     链接: {url}")
    
    print("\n" + "=" * 80)


def list_all_datasets():
    """列出所有数据集"""
    print("\n" + "=" * 80)
    print("📦 可用数据集列表")
    print("=" * 80)
    
    categories = {
        "lung": "🫁 肺部/胸部",
        "liver": "🫀 肝脏/腹部", 
        "head_neck": "👤 头颈部",
        "breast": "🩺 乳腺",
        "multi_organ": "🧬 多器官",
    }
    
    for cat_key, cat_name in categories.items():
        print(f"\n{cat_name}")
        print("-" * 80)
        
        for key, info in DATASET_REGISTRY.items():
            if info["category"] == cat_key:
                reg = "🔐" if info.get("requires_registration") else "📂"
                print(f"  {reg} {key:20s} - {info['name']:30s} | {info['cases']:5d}例 | {info['size']}")
                print(f"      {info['description']}")
    
    print("\n" + "=" * 80)
    print("💡 使用说明:")
    print("  --dataset [key]      查看指定数据集详情")
    print("  --dataset list       列出所有数据集")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="改进的公开医学影像数据集下载指南"
    )
    parser.add_argument('--dataset', type=str, default='list',
                        help="数据集名称或 'list' 查看所有")
    
    args = parser.parse_args()
    
    if args.dataset == 'list':
        list_all_datasets()
    else:
        print_dataset_info(args.dataset)


if __name__ == "__main__":
    main()

