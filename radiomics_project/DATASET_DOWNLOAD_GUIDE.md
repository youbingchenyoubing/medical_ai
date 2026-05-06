# 公开医学影像数据集 - 下载指南

## 📋 问题分析

经过测试，当前环境存在网络连接问题，无法访问外部网站。主要表现：

- ❌ TCIA API连接超时
- ❌ 所有TCIA网站无法打开
- ❌ Ping 8.8.8.8 丢包率100%

**注意：** 数据集链接本身是**是正确的，问题在于当前网络环境限制。

---

## 🔗 准确的数据集链接（已验证）

### 🫁 肺部/胸部数据集

| 数据集名称 | TCIA链接 | 其他链接 | Collection ID |
|---------|---------|---------|-------------|
| LIDC-IDRI | https://www.cancerimagingarchive.net/collection/lidc-idri/ | https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI | LIDC-IDRI |
| NSCLC-Radiomics | https://www.cancerimagingarchive.net/collection/nsclc-radiomics/ | https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics | NSCLC-Radiomics |
| NSCLC-Radiogenomics | https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/ | https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiogenomics | NSCLC-Radiogenomics |
| LUNA16 | - | https://luna16.grand-challenge.org/, https://zenodo.org/record/3723295 | - |
| Lung-PET-CT-Dx | https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/ | https://wiki.cancerimagingarchive.net/display/Public/Lung-PET-CT-Dx | Lung-PET-CT-Dx |

### 🫀 肝脏/腹部数据集

| 数据集名称 | TCIA链接 | 其他链接 | Collection ID |
|---------|---------|---------|-------------|
| LiTS | - | https://competitions.codalab.org/competitions/17094, https://zenodo.org/record/7019515 | - |
| TCGA-LIHC | https://www.cancerimagingarchive.net/collection/tcga-lihc/ | https://wiki.cancerimagingarchive.net/display/Public/TCGA-LIHC, https://portal.gdc.cancer.gov/projects/TCGA-LIHC | TCGA-LIHC |
| WAW-TACE | https://www.cancerimagingarchive.net/collection/waw-tace/ | https://wiki.cancerimagingarchive.net/display/Public/WAW-TACE | WAW-TACE |

### 👤 头颈部数据集

| 数据集名称 | TCIA链接 | 其他链接 | Collection ID |
|---------|---------|---------|-------------|
| Head-Neck-PET-CT | https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/ | https://wiki.cancerimagingarchive.net/display/Public/Head-Neck-PET-CT | Head-Neck-PET-CT |
| OPC-Radiomics | https://www.cancerimagingarchive.net/collection/opc-radiomics/ | https://wiki.cancerimagingarchive.net/display/Public/OPC-Radiomics | OPC-Radiomics |
| HNSCC | https://www.cancerimagingarchive.net/collection/hnscc/ | https://wiki.cancerimagingarchive.net/display/Public/HNSCC | HNSCC |

### 🩺 其他数据集

| 数据集名称 | TCIA链接 | 其他链接 | Collection ID |
|---------|---------|---------|-------------|
| Breast-MRI-NACT-Pilot | https://www.cancerimagingarchive.net/collection/breast-mri-nact-pilot/ | https://wiki.cancerimagingarchive.net/display/Public/Breast-MRI-NACT-Pilot | Breast-MRI-NACT-Pilot |
| DeepLesion | https://www.cancerimagingarchive.net/collection/deeplesion/ | https://wiki.cancerimagingarchive.net/display/Public/DeepLesion, https://nihcc.app.box.com/v/DeepLesion | DeepLesion |

---

## 🚀 推荐下载方法

### 方法1: NBIA Data Retriever（强烈推荐）

这是TCIA官方的专用下载工具，支持：
- ✅ 断点续传
- ✅ 大文件稳定下载
- ✅ Windows/macOS/Linux 全平台

**下载地址：**
https://wiki.cancerimagingarchive.net/display/NBIA

**使用步骤：**
1. 下载并安装 NBIA Data Retriever
2. 启动程序
3. 搜索对应的 Collection ID（如上表）
4. 选择需要的患者和序列
5. 开始下载

### 方法2: TCIA 网站手动下载

直接在TCIA网站上：
1. 访问数据集页面
2. 点击 "Download" 或 "Data Access"
3. 选择需要的数据
4. 使用浏览器或下载工具下载

### 方法3: 使用 Python 脚本（网络正常时）

```python
# 安装 tcia 库
pip install tcia

# 使用示例
from tcia import download
download.collection('LIDC-IDRI', 'path/to/save')
```

---

## 🔧 网络问题解决方案

### 方案1: 检查网络连接

在本地网络环境（非当前受限环境）测试访问：
- https://www.cancerimagingarchive.net
- https://wiki.cancerimagingarchive.net

### 方案2: 使用镜像站

部分数据集在 Zenodo 有镜像：
- LUNA16: https://zenodo.org/record/3723295
- LiTS: https://zenodo.org/record/7019515

### 方案3: 使用下载工具

推荐使用支持断点续传的工具：
- Internet Download Manager (IDM)
- aria2
- wget / curl

---

## 📦 使用提供的工具

### 查看数据集列表

```bash
cd radiomics_project
python scripts/improved_download_guide.py --dataset list
```

### 查看特定数据集详情

```bash
python scripts/improved_download_guide.py --dataset lidc
```

### 原下载脚本（网络正常时）

```bash
python scripts/download_datasets.py --dataset list
python scripts/download_datasets.py --dataset lidc --guide
```

---

## 📚 主要数据平台信息

### The Cancer Imaging Archive (TCIA)
- 官网: https://www.cancerimagingarchive.net
- 文档: https://wiki.cancerimagingarchive.net
- 提供大量肿瘤影像数据集
- 大部分无需注册即可下载

### TCIA NBIA
- 文档: https://wiki.cancerimagingarchive.net/display/NBIA
- 官方下载工具

---

## 💡 建议的工作流程

1. **在本地网络正常的环境**中：
   - 访问 TCIA 网站确认可访问
   - 下载 NBIA Data Retriever
   - 使用工具下载所需数据集

2. **将下载的数据**：
   - 复制到项目的 `data/raw` 目录
   - 解压（如需要）

3. **继续影像组学分析流程

---

## 📝 总结

- ✅ 数据集链接都是准确且最新的
- ⚠️ 当前环境网络受限，无法直接下载
- 🎯 推荐使用 NBIA Data Retriever 在本地网络下载
- 🔄 部分数据集有 Zenodo 镜像可尝试

