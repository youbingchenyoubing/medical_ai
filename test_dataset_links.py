#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试公开数据集链接的可访问性"""

import requests
import time
from typing import Dict, List, Tuple

# TCIA API地址
TCIA_API_BASE = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

# 主要数据集链接
DATASET_LINKS = [
    # 肺部数据集
    {"name": "LIDC-IDRI", "url": "https://www.cancerimagingarchive.net/collection/lidc-idri/"},
    {"name": "NSCLC-Radiomics", "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiomics/"},
    {"name": "NSCLC-Radiogenomics", "url": "https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/"},
    {"name": "LUNG-PET-CT-Dx", "url": "https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/"},
    
    # 肝脏数据集
    {"name": "TCGA-LIHC", "url": "https://www.cancerimagingarchive.net/collection/tcga-lihc/"},
    {"name": "WAW-TACE", "url": "https://www.cancerimagingarchive.net/collection/waw-tace/"},
    
    # 头颈部数据集
    {"name": "Head-Neck-PET-CT", "url": "https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/"},
    {"name": "OPC-Radiomics", "url": "https://www.cancerimagingarchive.net/collection/opc-radiomics/"},
    {"name": "HNSCC", "url": "https://www.cancerimagingarchive.net/collection/hnscc/"},
    
    # 其他数据集
    {"name": "Breast-MRI-NACT", "url": "https://www.cancerimagingarchive.net/collection/breast-mri-nact-pilot/"},
    {"name": "DeepLesion", "url": "https://www.cancerimagingarchive.net/collection/deeplesion/"},
]

def test_tcia_api() -> Tuple[bool, str]:
    """测试TCIA API连接"""
    print("=" * 70)
    print("测试 TCIA API 连接")
    print("=" * 70)
    
    test_endpoints = [
        ("Get Collection Values", f"{TCIA_API_BASE}/getCollectionValues"),
    ]
    
    results = []
    for name, url in test_endpoints:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=15)
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                print(f"✓ {name}: 成功 ({elapsed:.1f}ms)")
                results.append(True)
            else:
                print(f"✗ {name}: HTTP {response.status_code}")
                results.append(False)
        except requests.exceptions.Timeout:
            print(f"✗ {name}: 连接超时")
            results.append(False)
        except requests.exceptions.ConnectionError as e:
            print(f"✗ {name}: 连接错误 - {e}")
            results.append(False)
        except Exception as e:
            print(f"✗ {name}: 其他错误 - {e}")
            results.append(False)
    
    return all(results), ""

def test_webpage_links() -> List[Dict]:
    """测试各数据集网页链接"""
    print("\n" + "=" * 70)
    print("测试数据集网页链接")
    print("=" * 70)
    
    results = []
    for dataset in DATASET_LINKS:
        try:
            start_time = time.time()
            # 使用head请求，避免下载整个页面
            response = requests.head(dataset["url"], timeout=10, allow_redirects=True)
            elapsed = (time.time() - start_time) * 1000
            
            success = response.status_code in [200, 301, 302, 304]
            result = {
                "name": dataset["name"],
                "url": dataset["url"],
                "success": success,
                "status_code": response.status_code,
                "elapsed_ms": elapsed,
            }
            results.append(result)
            
            status = "✓" if success else "✗"
            print(f"{status} {dataset['name']:25s} - {dataset['url']}")
            if success:
                print(f"    状态码: {response.status_code}, 耗时: {elapsed:.1f}ms")
            else:
                print(f"    状态码: {response.status_code}")
                
        except requests.exceptions.Timeout:
            results.append({
                "name": dataset["name"],
                "url": dataset["url"],
                "success": False,
                "error": "连接超时",
            })
            print(f"✗ {dataset['name']:25s} - 连接超时")
        except requests.exceptions.ConnectionError as e:
            results.append({
                "name": dataset["name"],
                "url": dataset["url"],
                "success": False,
                "error": f"连接错误: {e}",
            })
            print(f"✗ {dataset['name']:25s} - 连接错误: {e}")
        except Exception as e:
            results.append({
                "name": dataset["name"],
                "url": dataset["url"],
                "success": False,
                "error": f"其他错误: {e}",
            })
            print(f"✗ {dataset['name']:25s} - 其他错误: {e}")
    
    return results

def print_summary(api_ok: bool, link_results: List[Dict]):
    """打印总结"""
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    print(f"\nTCIA API: {'✓ 可访问' if api_ok else '✗ 不可访问'}")
    
    success_count = sum(1 for r in link_results if r.get("success"))
    total_count = len(link_results)
    print(f"数据集网页: {success_count}/{total_count} 可访问")
    
    print("\n详细状态:")
    for result in link_results:
        status = "✓ 可访问" if result.get("success") else "✗ 不可访问"
        print(f"  {result['name']:25s} - {status}")
    
    print("\n" + "=" * 70)

def main():
    api_ok, _ = test_tcia_api()
    link_results = test_webpage_links()
    print_summary(api_ok, link_results)

if __name__ == "__main__":
    main()

