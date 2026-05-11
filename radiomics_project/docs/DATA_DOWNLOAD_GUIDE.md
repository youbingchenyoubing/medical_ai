
# 医学影像公开数据集下载指南

## 问题说明

你可能会发现使用 `scripts/download_datasets.py` 无法自动下载 TCIA（The Cancer Imaging Archive）的公开数据集。这通常是由于以下原因：

1. **网络限制** - TCIA API 可能在某些地区无法直接访问
2. **防火墙/代理** - 企业网络或学术网络可能屏蔽了外部 API
3. **API 服务状态** - TCIA 服务可能暂时不可用

## 推荐的下载方式（按优先级）

### 方式 1: NBIA Data Retriever（强烈推荐）

NBIA Data Retriever 是 TCIA 官方提供的桌面工具，专门用于批量下载医学影像数据。

#### 下载与安装
1. 访问下载页面：https://wiki.cancerimagingarchive.net/x/X4ATAg
2. 根据你的操作系统（Windows/macOS/Linux）下载对应版本
3. 按照安装向导完成安装

#### 使用方法
1. 启动 NBIA Data Retriever
2. 在搜索栏中输入你想下载的数据集名称，例如：
   - `LIDC-IDRI` (肺结节数据集)
   - `NSCLC-Radiomics` (非小细胞肺癌)
   - `Head-Neck-PET-CT` (头颈部 PET-CT)
3. 选择你需要下载的 collection
4. 选择下载位置
5. 点击开始下载

### 方式 2: 手动从 TCIA 网站下载

如果你只需要下载少量数据，或者 NBIA Data Retriever 无法使用，可以直接从 TCIA 网站手动下载。

#### 步骤
1. 访问 TCIA 官网：https://www.cancerimagingarchive.net/
2. 点击 "Collections" 或 "Browse Collections"
3. 找到你需要的数据集（例如 NSCLC-Radiomics）
4. 进入数据集详情页
5. 查找 "Download" 或 "Access Data" 按钮
6. 按照页面指引下载

### 方式 3: 使用 TCIA Python 客户端

如果网络条件允许，可以尝试使用官方的 Python 客户端。

```bash
# 安装客户端
pip install tcia-client

# 使用示例
python -c "
from tciaclient import TCIAClient
tc = TCIAClient()
# 下载整个 collection
tc.get_image(
    collection='NSCLC-Radiomics', 
    downloadPath='./data/raw/NSCLC-Radiomics'
)
"
```

## 数据集列表与对应信息

| 数据集 Key | 名称 | 大小 | TCIA Collection 名称 |
|-----------|------|------|---------------------|
| lidc | LIDC-IDRI | ~120GB | LIDC-IDRI |
| nsclc | NSCLC-Radiomics | ~30GB | NSCLC-Radiomics |
| nsclc_rgenomics | NSCLC-Radiogenomics | ~15GB | NSCLC-Radiogenomics |
| lung_pet_ct_dx | Lung-PET-CT-Dx | ~25GB | Lung-PET-CT-Dx |
| tcga_lihc | TCGA-LIHC | ~10GB | TCGA-LIHC |
| waw_tace | WAW-TACE | ~8GB | WAW-TACE |
| head_neck_pet_ct | Head-Neck-PET-CT | ~30GB | Head-Neck-PET-CT |
| opc_radiomics | OPC-Radiomics | ~20GB | OPC-Radiomics |
| hnscc | HNSCC | ~15GB | HNSCC |
| breast_mri_nact | Breast-MRI-NACT-Pilot | ~5GB | Breast-MRI-NACT-Pilot |
| deeplesion | DeepLesion | ~80GB | DeepLesion |

## 网络问题解决方案

如果以上方式仍然无法访问，尝试以下方案：

1. **使用 VPN** - 尝试切换到不同地区的 VPN 服务器
2. **使用学术网络** - 如果你在高校或研究机构，可以尝试使用机构网络
3. **镜像站点** - 查找是否有 TCIA 数据的镜像或备份站点
4. **请求帮助** - 联系实验室或研究团队，看是否有人已经下载了数据

## 临时解决方案：使用示例数据

在你获取到真实数据之前，可以使用 `download_sample_data` 来生成测试数据：

```bash
python scripts/download_datasets.py --dataset sample
```

这将创建一些随机生成的示例数据，让你可以先测试和运行影像组学分析流程。

## 下一步

下载完成后，将数据放在 `data/raw/` 目录下，然后可以继续进行影像组学分析。

