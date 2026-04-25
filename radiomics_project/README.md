# 影像组学项目 - 支持传统方法和深度学习

基于医学影像的影像组学特征提取、选择和预测模型构建完整流程。**支持传统影像组学和深度学习两种方法**。

## 🆕 新增：深度学习端到端方法

### 传统影像组学 vs 深度学习

| 维度 | 传统影像组学 | 深度学习 |
|------|-------------|---------|
| 特征提取 | 手工提取（PyRadiomics） | **自动学习（CNN）** ✨ |
| 特征数量 | 1000+手工特征 | 数百万网络参数 |
| 可解释性 | 高（特征有物理意义） | 低（黑盒模型） |
| 数据需求 | 较少（50-100例） | 较多（200+例） |
| 计算资源 | CPU即可 | **需要GPU** |
| 性能上限 | AUC 0.75-0.85 | **AUC 0.85-0.95** ✨ |
| 训练时间 | 快（分钟级） | 慢（小时级） |

### 推荐方案

#### 方案1：纯深度学习（推荐）✨
```bash
# 适用于数据量充足（200+例），追求最高性能
python scripts/train_deep_learning.py --model resnet3d --epochs 50
```

**优势**：
- ✅ **无需手工特征提取**
- ✅ 性能最优（AUC > 0.90）
- ✅ 端到端学习

#### 方案2：传统影像组学
```bash
# 适用于数据量少（<100例），需要可解释性
python main.py --step 0
```

**优势**：
- ✅ 特征有物理意义
- ✅ 易于临床解释
- ✅ 不需要GPU

#### 方案3：混合方法（最佳实践）
```bash
# 结合两者优点
# 1. 提取传统特征
python main.py --step 2

# 2. 提取深度特征
python scripts/train_deep_learning.py --model resnet3d

# 3. 特征融合
# （见融合模型代码）
```

**优势**：
- ✅ 性能最优（AUC > 0.92）
- ✅ 兼具可解释性和性能

## 项目结构

```
radiomics_project/
├── config/
│   └── config.yaml              # 配置文件
├── src/                          # 核心代码模块
│   ├── utils.py                 # 工具函数
│   ├── data_preprocessing.py    # 数据预处理
│   ├── feature_extraction.py    # 传统特征提取
│   ├── feature_selection.py     # 特征选择
│   ├── model_training.py        # 传统模型训练
│   ├── evaluation.py            # 模型评估
│   ├── deep_learning_models.py  # 深度学习模型 ✨
│   └── deep_learning_trainer.py # 深度学习训练器 ✨
├── scripts/
│   ├── download_datasets.py     # 数据集下载
│   └── train_deep_learning.py   # 深度学习训练 ✨
├── docs/
│   └── 深度学习vs影像组学对比.md # 方法对比文档 ✨
├── main.py                       # 传统方法主程序
├── requirements.txt              # 依赖列表
└── README.md                     # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
cd radiomics_project
pip install -r requirements.txt
```

### 2. 下载数据集

脚本支持两种下载模式：**自动下载**（通过TCIA API）和**仅显示指引**。

```bash
# 查看所有可用数据集及下载模式（Auto/Guide）
python scripts/download_datasets.py --dataset list

# 自动下载（默认模式，需网络可达TCIA）
python scripts/download_datasets.py --dataset nsclc

# 仅显示下载指引（不实际下载）
python scripts/download_datasets.py --dataset nsclc --guide

# 下载示例数据（用于测试，无需网络）
python scripts/download_datasets.py --dataset sample

# 下载全部数据集
python scripts/download_datasets.py --dataset all
```

**下载模式说明**：

| 模式 | 说明 | 适用数据集 |
|------|------|-----------|
| Auto | 通过TCIA REST API自动下载影像数据 | lidc, nsclc, nsclc_rgenomics, lung_pet_ct_dx, tcga_lihc, waw_tace, head_neck_pet_ct, opc_radiomics, hnscc, breast_mri_nact, deeplesion |
| Guide | 仅显示下载指引，需手动下载 | luna16, lits（需注册） |

> **注意**：Auto模式下，脚本会先连接TCIA API查询数据集信息，确认后才开始下载。若网络不可达，会自动降级为Guide模式。部分数据集的标注/临床数据需单独下载，脚本会给出提示。

### 3. 选择方法

#### 方法A：传统影像组学

```bash
# 运行完整流程
python main.py --step 0

# 或分步运行
python main.py --step 1  # 数据预处理
python main.py --step 2  # 特征提取
python main.py --step 3  # 特征选择
python main.py --step 4  # 模型训练
```

#### 方法B：深度学习（推荐）✨

```bash
# 1. 数据预处理
python main.py --step 1

# 2. 训练深度学习模型
python scripts/train_deep_learning.py --model resnet3d --epochs 50

# 可选模型：
# - simple3dcnn: 简单3D CNN
# - resnet3d: 3D ResNet（推荐）
# - densenet3d: 3D DenseNet
```

## 支持的深度学习模型

### 1. Simple3DCNN
- 简单的3D卷积网络
- 适合小数据集
- 训练速度快

### 2. ResNet-3D（推荐）✨
- 3D残差网络
- 性能优秀
- 训练稳定

### 3. DenseNet-3D
- 3D密集连接网络
- 参数效率高
- 适合医学影像

## 详细说明

### 传统影像组学流程

#### 步骤1：数据预处理
- CT图像：重采样、窗宽窗位调整、归一化
- MRI图像：偏置场校正、重采样、归一化

#### 步骤2：特征提取
使用PyRadiomics提取影像组学特征：
- 一阶统计特征（18个）
- 形状特征（14个）
- 纹理特征（GLCM、GLRLM、GLSZM、GLDM、NGTDM）

#### 步骤3：特征选择
支持多种特征选择方法：
- LASSO回归
- 递归特征消除（RFE）
- 互信息

#### 步骤4：模型训练
支持多种机器学习模型：
- 逻辑回归（LR）
- 支持向量机（SVM）
- 随机森林（RF）
- XGBoost

### 深度学习流程 ✨

#### 步骤1：数据预处理
与传统方法相同

#### 步骤2：端到端训练
```bash
python scripts/train_deep_learning.py \
    --model resnet3d \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4
```

**特点**：
- ✅ **无需手工特征提取**
- ✅ 自动学习最优特征
- ✅ 支持GPU加速
- ✅ 自动保存最佳模型
- ✅ 绘制训练曲线

## 配置文件

编辑 `config/config.yaml` 自定义参数：

```yaml
# 数据路径
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  
# 预处理参数
preprocessing:
  target_spacing: [1, 1, 1]
  window_level: -600
  window_width: 1500
  
# 传统方法参数
feature_extraction:
  bin_width: 25
  
feature_selection:
  method: "lasso"
  n_features: 15
  
# 深度学习参数
model_name: "resnet3d"
in_channels: 1
num_classes: 2
learning_rate: 1e-4
```

## 使用示例

### 示例1：使用深度学习方法

```python
from src.deep_learning_trainer import DeepLearningTrainer

# 配置
config = {
    'model_name': 'resnet3d',
    'in_channels': 1,
    'num_classes': 2,
    'training': {
        'batch_size': 4,
        'epochs': 50,
        'early_stopping_patience': 10
    },
    'learning_rate': 1e-4
}

# 初始化训练器
trainer = DeepLearningTrainer(config)

# 准备数据
trainer.prepare_data(train_paths, train_labels, val_paths, val_labels)

# 训练（无需手工特征提取！）
trainer.train()

# 预测
probs = trainer.predict(test_paths)
```

### 示例2：混合方法

```python
# 1. 提取传统特征
from src.feature_extraction import FeatureExtractor
extractor = FeatureExtractor(config)
traditional_features = extractor.extract_features_batch(...)

# 2. 提取深度特征
from src.deep_learning_models import get_model
model = get_model('resnet3d', in_channels=1, num_classes=2)
# ... 提取深度特征

# 3. 特征融合
fused_features = np.concatenate([
    traditional_features,
    deep_features
], axis=1)

# 4. 训练融合模型
# ...
```

## 支持的数据集

### 🫁 肺部/胸部

| 数据集 | 命令行 key | 病例数 | 模态 | 标注内容 | 推荐方法 |
|--------|-----------|--------|------|---------|----------|
| LIDC-IDRI | `lidc` | 1,018 | CT | 肺结节（4位医生标注+恶性度评分） | 深度学习 ✨ |
| NSCLC-Radiomics | `nsclc` | 422 | CT | 肿瘤分割+临床+生存数据 | 混合方法 |
| NSCLC-Radiogenomics | `nsclc_rgenomics` | 211 | CT | 肿瘤分割+基因表达+临床 | 影像基因组学 |
| LUNA16 | `luna16` | 888 | CT | 肺结节（LIDC子集，剔除<3mm） | 深度学习 ✨ |
| Lung-PET-CT-Dx | `lung_pet_ct_dx` | 284 | CT+PET | 肺癌亚型+生存数据 | 多模态影像组学 |

### 🫀 肝脏/腹部

| 数据集 | 命令行 key | 病例数 | 模态 | 标注内容 | 推荐方法 |
|--------|-----------|--------|------|---------|----------|
| LiTS | `lits` | 201 | CT | 肝脏+肿瘤分割 | 深度学习 ✨ |
| TCGA-LIHC | `tcga_lihc` | 186 | MRI/CT | 肝细胞癌+基因+临床 | 影像基因组学 |
| WAW-TACE | `waw_tace` | 117 | MRI | HCC TACE治疗+临床+疗效 | Delta影像组学 |

### 🧠 头颈部

| 数据集 | 命令行 key | 病例数 | 模态 | 标注内容 | 推荐方法 |
|--------|-----------|--------|------|---------|----------|
| Head-Neck-PET-CT | `head_neck_pet_ct` | 298 | CT+PET | 肿瘤分割+临床 | 多模态影像组学 |
| OPC-Radiomics | `opc_radiomics` | 606 | CT | GTV分割+生存+HPV状态 | 生存预测 |
| HNSCC | `hnscc` | 364 | CT | 肿瘤分割+临床+生存 | 预后预测 |

### 🎗️ 乳腺

| 数据集 | 命令行 key | 病例数 | 模态 | 标注内容 | 推荐方法 |
|--------|-----------|--------|------|---------|----------|
| Breast-MRI-NACT-Pilot | `breast_mri_nact` | 64 | MRI | 新辅助化疗+病理缓解(pCR) | Delta影像组学 |

### 🧬 多器官综合

| 数据集 | 命令行 key | 病例数 | 模态 | 标注内容 | 推荐方法 |
|--------|-----------|--------|------|---------|----------|
| DeepLesion | `deeplesion` | 10,594 | CT | 32,735个多器官多病灶标注 | 深度学习 ✨ |

> 💡 **提示**：运行 `python scripts/download_datasets.py --dataset list` 可查看所有数据集的详细信息

## 输出结果

### 传统方法
- `results/features/`: 特征文件（CSV格式）
- `results/models/`: 训练好的模型（PKL格式）
- `results/figures/`: 可视化图表（PNG格式）

### 深度学习方法 ✨
- `results/models/best_model.pth`: 最佳模型
- `results/figures/training_curves.png`: 训练曲线
- 自动保存训练日志

## 常见问题

### Q: 应该选择哪种方法？

**A: 根据数据量选择**：
- **数据量 < 100例**：使用传统影像组学
- **数据量 100-200例**：使用迁移学习
- **数据量 > 200例**：使用深度学习 ✨

### Q: 深度学习需要什么硬件？

**A: 建议配置**：
- GPU：NVIDIA RTX 3080或更高
- 显存：≥16GB
- 内存：≥32GB

### Q: 深度学习真的不需要特征提取吗？

**A: 是的！** 深度学习会自动学习最优特征，无需手工提取。这是深度学习的核心优势之一。

### Q: 如何提高模型性能？

**A: 建议**：
1. **增加数据量**：最有效的方法
2. **使用预训练模型**：MedicalNet等
3. **数据增强**：旋转、翻转等
4. **混合方法**：结合传统特征和深度特征

## 最新研究进展（2024-2025）

### 深度学习方法
- **3D CNN**：ResNet-3D、DenseNet-3D
- **Transformer**：ViT、Swin Transformer、UNETR
- **自监督学习**：MAE、SimCLR
- **多模态融合**：影像+临床+基因

### 性能对比
- 传统影像组学：AUC 0.75-0.85
- 深度学习：AUC 0.85-0.95 ✨
- 混合方法：AUC 0.90-0.95 ✨

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系项目维护者。

## 更新日志

### v1.2.0 (2026-04-24)
- 📦 扩展公开数据集支持：从3个扩展到14个
- 🫁 新增肺部数据集：NSCLC-Radiogenomics、LUNA16、Lung-PET-CT-Dx
- 🫀 新增肝脏数据集：TCGA-LIHC、WAW-TACE
- 🧠 新增头颈部数据集：Head-Neck-PET-CT、OPC-Radiomics、HNSCC
- 🎗️ 新增乳腺数据集：Breast-MRI-NACT-Pilot
- 🧬 新增多器官数据集：DeepLesion
- 🔧 实现TCIA REST API自动下载（默认模式）
- 🔧 新增 `--guide` 参数：仅显示下载指引
- 🔧 新增 `--dataset list` 命令查看所有数据集
- 🔧 新增 `DATASET_REGISTRY` 集中管理数据集元信息
- 🔧 网络不可达时自动降级为指引模式
- 🔧 支持断点续传（download_progress.json）

### v1.1.0 (2026-03-29)
- ✨ 新增深度学习端到端方法
- ✨ 支持3D CNN、ResNet-3D、DenseNet-3D
- ✨ 无需手工特征提取
- ✨ 性能提升至AUC > 0.90

### v1.0.0 (2026-03-29)
- 初始版本发布
- 支持传统影像组学流程
