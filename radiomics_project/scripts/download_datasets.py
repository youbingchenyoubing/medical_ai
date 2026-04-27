#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公开数据集下载脚本 - 支持自动下载和下载指引

支持的数据集:
肺部/胸部:
- LIDC-IDRI: 肺结节CT数据集 (1,018例)
- NSCLC-Radiomics: 非小细胞肺癌影像组学数据集 (422例)
- NSCLC-Radiogenomics: 非小细胞肺癌影像基因组学数据集 (211例)
- LUNA16: 肺结节分析挑战赛数据集 (888例, 需注册)
- Lung-PET-CT-Dx: 肺癌PET-CT诊断数据集 (355例)

腹部/肝脏:
- LiTS: 肝脏肿瘤分割数据集 (201例, 需注册)
- TCGA-LIHC: 肝细胞癌影像基因组学数据集 (186例)
- WAW-TACE: HCC TACE治疗多期CT数据集 (233例)

头颈部:
- Head-Neck-PET-CT: 头颈部PET-CT数据集 (298例)
- OPC-Radiomics: 口咽癌影像组学数据集 (606例)
- HNSCC: 头颈鳞状细胞癌数据集 (364例)

乳腺:
- Breast-MRI-NACT-Pilot: 乳腺癌新辅助化疗MRI数据集 (64例)

综合:
- DeepLesion: 多器官多病灶CT数据集 (32,735病灶)
- sample: 随机生成示例数据 (10例)

下载模式:
- 默认: 自动通过TCIA API下载影像数据
- --guide: 仅显示下载指引，不实际下载
- 需注册的数据集(LUNA16, LiTS)仅提供指引
"""

import os
import sys
import json
import time
import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

TCIA_API_BASE = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

DATASET_REGISTRY = {
    "lidc": {
        "name": "LIDC-IDRI",
        "category": "lung",
        "cases": 1018,
        "modality": "CT",
        "size": "~120GB",
        "description": "胸部CT肺结节数据集，4位医生标注+恶性度评分",
        "url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
        "tcia_collection": "LIDC-IDRI",
        "requires_registration": False,
        "annotations": [
            {
                "name": "LIDC-XML-Annotations",
                "url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
                "note": "肺结节标注XML文件，在Collection页面下载",
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
        "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/",
        "tcia_collection": "NSCLC-Radiomics",
        "requires_registration": False,
        "annotations": [
            {
                "name": "NSCLC-Radiomics-Annotations",
                "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/",
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
        "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/",
        "tcia_collection": "NSCLC-Radiogenomics",
        "requires_registration": False,
        "annotations": [
            {
                "name": "NSCLC-Radiogenomics-Clinical-Genomic",
                "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/",
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
        "url": "https://luna16.grand-challenge.org/",
        "tcia_collection": None,
        "requires_registration": True,
        "annotations": [],
    },
    "lung_pet_ct_dx": {
        "name": "Lung-PET-CT-Dx",
        "category": "lung",
        "cases": 355,
        "modality": "CT+PET",
        "size": "~25GB",
        "description": "肺癌PET-CT，含亚型诊断+生存数据",
        "url": "https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/",
        "tcia_collection": "Lung-PET-CT-Dx",
        "requires_registration": False,
        "annotations": [],
    },
    "lits": {
        "name": "LiTS",
        "category": "liver",
        "cases": 201,
        "modality": "CT",
        "size": "~20GB",
        "description": "肝脏肿瘤分割数据集，131训练+70测试",
        "url": "https://competitions.codalab.org/competitions/17094",
        "tcia_collection": None,
        "requires_registration": True,
        "annotations": [],
    },
    "tcga_lihc": {
        "name": "TCGA-LIHC",
        "category": "liver",
        "cases": 186,
        "modality": "MRI/CT",
        "size": "~10GB",
        "description": "肝细胞癌，含基因+影像+临床数据",
        "url": "https://www.cancerimagingarchive.net/collection/tcga-lihc/",
        "tcia_collection": "TCGA-LIHC",
        "requires_registration": False,
        "annotations": [
            {
                "name": "TCGA-LIHC-Clinical-Genomic",
                "url": "https://portal.gdc.cancer.gov/projects/TCGA-LIHC",
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
        "url": "https://www.cancerimagingarchive.net/collection/waw-tace/",
        "tcia_collection": "WAW-TACE",
        "requires_registration": False,
        "annotations": [],
    },
    "head_neck_pet_ct": {
        "name": "Head-Neck-PET-CT",
        "category": "head_neck",
        "cases": 298,
        "modality": "CT+PET",
        "size": "~30GB",
        "description": "头颈部PET-CT，含肿瘤分割+临床数据",
        "url": "https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/",
        "tcia_collection": "Head-Neck-PET-CT",
        "requires_registration": False,
        "annotations": [
            {
                "name": "Head-Neck-PET-CT-Annotations",
                "url": "https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/",
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
        "url": "https://www.cancerimagingarchive.net/collection/opc-radiomics/",
        "tcia_collection": "OPC-Radiomics",
        "requires_registration": False,
        "annotations": [
            {
                "name": "OPC-Radiomics-Clinical",
                "url": "https://www.cancerimagingarchive.net/collection/opc-radiomics/",
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
        "url": "https://www.cancerimagingarchive.net/collection/hnscc/",
        "tcia_collection": "HNSCC",
        "requires_registration": False,
        "annotations": [
            {
                "name": "HNSCC-Clinical",
                "url": "https://www.cancerimagingarchive.net/collection/hnscc/",
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
        "url": "https://www.cancerimagingarchive.net/collection/breast-mri-nact-pilot/",
        "tcia_collection": "Breast-MRI-NACT-Pilot",
        "requires_registration": False,
        "annotations": [],
    },
    "deeplesion": {
        "name": "DeepLesion",
        "category": "multi_organ",
        "cases": 10594,
        "modality": "CT",
        "size": "~80GB",
        "description": "多器官多病灶CT，32,735个标注病灶",
        "url": "https://www.cancerimagingarchive.net/collection/deeplesion/",
        "tcia_collection": "DeepLesion",
        "requires_registration": False,
        "annotations": [
            {
                "name": "DeepLesion-Annotations",
                "url": "https://nihcc.box.com/v/DeepLesion",
                "note": "标注文件需从NIH Box下载",
            }
        ],
    },
}


class TCIAClient:
    """TCIA REST API 客户端"""

    CONNECT_TIMEOUT = 15
    _connection_ok = None

    def __init__(self, api_base: str = TCIA_API_BASE, timeout: int = 120, retries: int = 3):
        self.api_base = api_base
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._test_connection()

    def _request_with_retry(self, url, params=None, stream=False, timeout=None):
        if timeout is None:
            timeout = self.timeout
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, stream=stream, timeout=timeout)
                r.raise_for_status()
                return r
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_error = e
                if attempt < self.retries:
                    wait = min(2 ** attempt, 30)
                    print(f"    [Retry {attempt}/{self.retries}] {type(e).__name__}, waiting {wait}s...")
                    time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                raise
        raise last_error

    def _test_connection(self):
        if TCIAClient._connection_ok is True:
            return True
        try:
            r = self.session.get(
                f"{self.api_base}/getCollectionValues",
                timeout=self.CONNECT_TIMEOUT,
            )
            r.raise_for_status()
            TCIAClient._connection_ok = True
            return True
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            print(f"\n  [WARNING] TCIA API connection failed: {e}")
            print(f"  The Cancer Imaging Archive may be unreachable from your network.")
            print(f"  Falling back to guide-only mode. Use --guide to skip this check.")
            TCIAClient._connection_ok = False
            return False

    def is_connected(self):
        return TCIAClient._connection_ok is True

    def get_series_list(self, collection: str) -> list:
        url = f"{self.api_base}/getSeries"
        params = {"Collection": collection}
        r = self._request_with_retry(url, params=params)
        return r.json()

    def get_patient_ids(self, collection: str) -> list:
        url = f"{self.api_base}/getPatient"
        params = {"Collection": collection}
        r = self._request_with_retry(url, params=params)
        return r.json()

    def download_series(self, series_uid: str, output_path: str) -> str:
        url = f"{self.api_base}/getImage"
        params = {"SeriesInstanceUID": series_uid}

        if os.path.exists(output_path):
            return output_path

        r = self._request_with_retry(url, params=params, stream=True)

        total_size = int(r.headers.get('content-length', 0))
        tmp_path = output_path + ".tmp"

        with open(tmp_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      desc=os.path.basename(output_path)[:40]) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        os.rename(tmp_path, output_path)
        return output_path


class DatasetDownloader:
    """数据集下载器"""

    def __init__(self, output_dir: str = "data/raw", guide_only: bool = False,
                 timeout: int = 120, retries: int = 3):
        self.output_dir = output_dir
        self.guide_only = guide_only
        self.timeout = timeout
        self.retries = retries
        os.makedirs(output_dir, exist_ok=True)

        print(f"DatasetDownloader initialized")
        print(f"Output directory: {output_dir}")
        if guide_only:
            print(f"Mode: guide-only (no actual download)")
        print(f"Timeout: {timeout}s | Retries: {retries}")

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
        if info.get("requires_registration"):
            print(f"\n  [!] This dataset requires registration to download.")

    def _print_download_guide(self, key: str):
        info = DATASET_REGISTRY[key]
        col = info.get("tcia_collection") or info["name"]

        print("\n  Download Methods:")
        print(f"\n  1. Official Website:")
        print(f"     {info['url']}")

        if info.get("tcia_collection"):
            print(f"\n  2. Using this script (auto download):")
            print(f"     python scripts/download_datasets.py --dataset {key}")
            print(f"\n  3. Using TCIA Client:")
            print(f"     pip install tcia-client")
            print(f"     from tciaclient import TCIAClient")
            print(f"     tc = TCIAClient()")
            print(f"     tc.get_image(collection='{col}', downloadPath='data/raw/{info['name']}')")
            print(f"\n  4. Using NBIA Data Retriever:")
            print(f"     https://www.cancerimagingarchive.net/access-data/")

        if info.get("annotations"):
            print(f"\n  Annotation / Clinical Data (separate download):")
            for ann in info["annotations"]:
                print(f"    - {ann['name']}")
                print(f"      {ann['url']}")
                print(f"      ({ann['note']})")

        print("\n" + "=" * 70)

    def _download_tcia_collection(self, key: str):
        info = DATASET_REGISTRY[key]
        collection = info["tcia_collection"]
        dataset_dir = os.path.join(self.output_dir, info["name"])
        zip_dir = os.path.join(dataset_dir, "zips")
        os.makedirs(zip_dir, exist_ok=True)

        print(f"\n  Connecting to TCIA API...")
        try:
            client = TCIAClient(timeout=self.timeout, retries=self.retries)
        except Exception as e:
            print(f"  [ERROR] Failed to connect to TCIA API: {e}")
            print(f"  Falling back to download guide...")
            self._print_download_guide(key)
            return

        if not client.is_connected():
            print(f"  Falling back to download guide...")
            self._print_download_guide(key)
            return

        print(f"  Querying series list for collection: {collection}...")
        try:
            series_list = client.get_series_list(collection)
        except Exception as e:
            print(f"  [ERROR] Failed to query series: {e}")
            print(f"  Falling back to download guide...")
            self._print_download_guide(key)
            return

        if not series_list:
            print(f"  [ERROR] No series found for collection: {collection}")
            self._print_download_guide(key)
            return

        total_series = len(series_list)
        print(f"  Found {total_series} series in collection '{collection}'")

        patient_ids = set()
        for s in series_list:
            patient_ids.add(s.get("PatientID", "unknown"))
        print(f"  Found {len(patient_ids)} patients")

        print(f"\n  Estimated total size: {info['size']}")
        print(f"  Download directory: {dataset_dir}")

        confirm = input(f"\n  Start downloading {total_series} series? [y/N]: ").strip().lower()
        if confirm != 'y':
            print(f"  Download cancelled.")
            self._print_download_guide(key)
            return

        progress_file = os.path.join(dataset_dir, "download_progress.json")
        completed = set()
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                completed = set(json.load(f).get("completed", []))
            print(f"  Resuming from previous download ({len(completed)} series already done)")

        failed = []
        for i, series in enumerate(series_list):
            uid = series.get("SeriesInstanceUID", "")
            patient_id = series.get("PatientID", "unknown")
            series_desc = series.get("SeriesDescription", "unknown")
            zip_filename = f"{patient_id}_{uid}.zip"
            zip_path = os.path.join(zip_dir, zip_filename)

            if uid in completed:
                continue

            print(f"\n  [{i + 1}/{total_series}] Patient: {patient_id} | Series: {series_desc[:40]}")

            try:
                client.download_series(uid, zip_path)
                completed.add(uid)
                with open(progress_file, 'w') as f:
                    json.dump({"completed": list(completed)}, f)
            except Exception as e:
                print(f"  [ERROR] Failed to download series {uid}: {e}")
                failed.append({"uid": uid, "patient": patient_id, "error": str(e)})
                if os.path.exists(zip_path + ".tmp"):
                    os.remove(zip_path + ".tmp")

        print(f"\n  Download Summary:")
        print(f"    Completed: {len(completed)}/{total_series}")
        print(f"    Failed:    {len(failed)}")

        if failed:
            failed_file = os.path.join(dataset_dir, "failed_downloads.json")
            with open(failed_file, 'w') as f:
                json.dump(failed, f, indent=2)
            print(f"    Failed list saved to: {failed_file}")

        if info.get("annotations"):
            print(f"\n  [!] Annotation/Clinical data must be downloaded separately:")
            for ann in info["annotations"]:
                print(f"    - {ann['name']}: {ann['url']}")

        print(f"\n  Files saved to: {dataset_dir}")
        print("=" * 70)

    def _download_dataset(self, key: str):
        info = DATASET_REGISTRY[key]

        if info.get("requires_registration"):
            print(f"\n  [!] This dataset requires registration and cannot be auto-downloaded.")
            self._print_download_guide(key)
            return

        if not info.get("tcia_collection"):
            print(f"\n  [!] This dataset is not available via TCIA API.")
            self._print_download_guide(key)
            return

        if self.guide_only:
            self._print_download_guide(key)
            return

        self._download_tcia_collection(key)

    def download_lidc_idri(self):
        self._print_dataset_info("lidc")
        print("\n  Special Notes:")
        print("    - 4位放射科医生独立标注肺结节")
        print("    - 每个结节含恶性度评分(1-5)")
        print("    - 适合ICC一致性分析和结节分类研究")
        self._download_dataset("lidc")

    def download_nsclc_radiomics(self):
        self._print_dataset_info("nsclc")
        print("\n  Special Notes:")
        print("    - 含422例NSCLC患者CT图像")
        print("    - 含肿瘤分割掩码和临床生存数据")
        print("    - 适合影像组学特征提取和生存预测")
        self._download_dataset("nsclc")

    def download_nsclc_radiogenomics(self):
        self._print_dataset_info("nsclc_rgenomics")
        print("\n  Special Notes:")
        print("    - 含211例NSCLC患者CT图像")
        print("    - 含基因表达数据和临床信息")
        print("    - 适合影像基因组学研究和基因突变预测")
        self._download_dataset("nsclc_rgenomics")

    def download_luna16(self):
        self._print_dataset_info("luna16")
        print("\n  Special Notes:")
        print("    - LIDC-IDRI子集，剔除直径<3mm的结节")
        print("    - 含888例CT和1,186个标注结节")
        print("    - 适合肺结节检测和假阳性减少研究")
        self._download_dataset("luna16")

    def download_lung_pet_ct_dx(self):
        self._print_dataset_info("lung_pet_ct_dx")
        print("\n  Special Notes:")
        print("    - 含355例肺癌患者CT和PET图像")
        print("    - 含组织学亚型诊断和生存数据")
        print("    - 适合多模态影像组学研究")
        self._download_dataset("lung_pet_ct_dx")

    def download_lits(self):
        self._print_dataset_info("lits")
        print("\n  Special Notes:")
        print("    - 131训练+70测试，含肝脏和肿瘤分割标注")
        print("    - 适合肝脏肿瘤分割和影像组学特征提取")
        self._download_dataset("lits")

    def download_tcga_lihc(self):
        self._print_dataset_info("tcga_lihc")
        print("\n  Special Notes:")
        print("    - 含186例肝细胞癌患者影像")
        print("    - 含TCGA基因突变、表达和临床数据")
        print("    - 适合影像基因组学和HCC预后预测")
        self._download_dataset("tcga_lihc")

    def download_waw_tace(self):
        self._print_dataset_info("waw_tace")
        print("\n  Special Notes:")
        print("    - 含233例HCC患者TACE治疗前后多期CT")
        print("    - 含基线临床变量和疗效标注")
        print("    - 适合HCC治疗反应预测和Delta影像组学")
        self._download_dataset("waw_tace")

    def download_head_neck_pet_ct(self):
        self._print_dataset_info("head_neck_pet_ct")
        print("\n  Special Notes:")
        print("    - 含298例头颈部癌患者CT和PET图像")
        print("    - 含肿瘤分割和临床数据")
        print("    - 适合放疗疗效预测和多模态影像组学")
        self._download_dataset("head_neck_pet_ct")

    def download_opc_radiomics(self):
        self._print_dataset_info("opc_radiomics")
        print("\n  Special Notes:")
        print("    - 含606例口咽癌患者CT")
        print("    - 含GTV分割、生存数据和HPV/p16状态")
        print("    - 适合口咽癌生存预测和影像组学")
        self._download_dataset("opc_radiomics")

    def download_hnscc(self):
        self._print_dataset_info("hnscc")
        print("\n  Special Notes:")
        print("    - 含364例头颈鳞状细胞癌患者CT")
        print("    - 含肿瘤分割、临床和生存数据")
        print("    - 适合头颈癌预后预测")
        self._download_dataset("hnscc")

    def download_breast_mri_nact(self):
        self._print_dataset_info("breast_mri_nact")
        print("\n  Special Notes:")
        print("    - 含64例乳腺癌患者新辅助化疗前后MRI")
        print("    - 含病理完全缓解(pCR)标注")
        print("    - 适合治疗反应预测和Delta影像组学")
        self._download_dataset("breast_mri_nact")

    def download_deeplesion(self):
        self._print_dataset_info("deeplesion")
        print("\n  Special Notes:")
        print("    - 含10,594例CT中32,735个标注病灶")
        print("    - 覆盖多器官多病种")
        print("    - 适合大规模病灶检测和分类研究")
        self._download_dataset("deeplesion")

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
                    auto = "Auto" if (info.get("tcia_collection") and not info.get("requires_registration")) else "Guide"
                    print(f"    {key:<22s} | {info['name']:<25s} | {info['cases']:>5d} cases | {info['modality']:<8s} | {info['size']:<8s} | {auto}")
                    print(f"    {'':22s}   {info['description']}")

        print("\n  [Testing]")
        print(f"    {'sample':<22s} | Sample Data                 |    10 cases | Random   | ~1KB     | Auto")
        print(f"    {'':22s}   随机生成示例临床数据，用于测试流程")

        print("\n  Download Modes:")
        print("    Auto  = Can be auto-downloaded via TCIA API (default)")
        print("    Guide = Registration required, only download guide provided")

        print("\n  Usage:")
        print("    python scripts/download_datasets.py --dataset <key>        # Auto download")
        print("    python scripts/download_datasets.py --dataset <key> --guide # Guide only")
        print("    python scripts/download_datasets.py --dataset all")
        print("    python scripts/download_datasets.py --dataset list")
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

Download modes:
  Default:  Auto-download via TCIA API (with confirmation prompt)
  --guide:  Only show download instructions, no actual download

Examples:
  python scripts/download_datasets.py --dataset nsclc           # Auto download
  python scripts/download_datasets.py --dataset nsclc --guide   # Guide only
  python scripts/download_datasets.py --dataset list            # List all datasets
  python scripts/download_datasets.py --dataset all             # Download all
        """
    )
    parser.add_argument('--dataset', type=str, default='list',
                        choices=all_choices,
                        help='Dataset to download (default: list)')
    parser.add_argument('--output', type=str, default='data/raw',
                        help='Output directory')
    parser.add_argument('--guide', action='store_true',
                        help='Only show download guide, do not actually download')
    parser.add_argument('--timeout', type=int, default=120,
                        help='API request timeout in seconds (default: 120)')
    parser.add_argument('--retries', type=int, default=3,
                        help='Number of retries for failed requests (default: 3)')

    args = parser.parse_args()

    downloader = DatasetDownloader(output_dir=args.output, guide_only=args.guide,
                                   timeout=args.timeout, retries=args.retries)

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
