# PyRadiomics 技术详解：从功能介绍到初步使用

> **PyRadiomics** — 医学影像放射组学特征提取的标准化开源平台
>
> 引用：van Griethuysen et al., *Computational Radiomics System to Decode the Radiographic Phenotype*, Cancer Research, 2017

---

## 一、什么是 PyRadiomics？

### 1.1 定义与背景

**PyRadiomics** 是一个用于从医学影像中提取**放射组学特征（Radiomics Features）**的开源 Python 库。它由哈佛大学 Dana-Farber 癌症研究所和 Brigham 妇女医院的研究团队开发，旨在为放射组学分析建立**参考标准**，并提供一个经过测试和维护的开源平台，以实现可重复的放射组学特征提取。

**核心目标**：
- 建立放射组学特征提取的**标准化参考**
- 提供**可重复、可验证**的特征提取流程
- 降低放射组学研究的**技术门槛**
- 扩大放射组学研究的**社区影响力**

### 1.2 放射组学是什么？

放射组学（Radiomics）是一种从医学影像（CT、MRI、PET 等）中提取大量定量特征的方法学。其核心思想是：**医学影像中蕴含着肉眼无法识别的深层信息**，通过高通量地提取定量特征，可以揭示疾病的影像学表型（Radiographic Phenotype），辅助临床决策。

```
放射组学工作流程：
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  影像获取  │ →  │  图像分割  │ →  │  特征提取  │ →  │  特征筛选  │ →  │  模型构建  │
│ CT/MRI/   │    │  ROI定义  │    │ PyRadiomics│   │  降维/选择  │    │  预测/分类  │
│ PET       │    │  肿瘤/器官 │    │  1000+特征 │    │  LASSO等   │    │  机器学习   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 1.3 PyRadiomics 的核心优势

| 优势 | 说明 |
|------|------|
| **标准化** | 遵循 IBSI（影像生物标志物标准化倡议）规范 |
| **全面性** | 支持 7 大类 100+ 个特征，2D 和 3D 均可 |
| **灵活性** | 支持原始图像和滤波后图像的特征提取 |
| **可重复性** | 提供参数文件机制，确保结果可复现 |
| **易用性** | 支持 Python API 和命令行两种使用方式 |
| **兼容性** | 支持 CT、MRI、PET 等多种影像模态 |

---

## 二、PyRadiomics 特征体系详解

PyRadiomics 共提供 **7 大类特征**，总计可提取 **100+ 个定量特征**。特征体系如下：

```
PyRadiomics 特征体系
├── 一阶统计特征 (First Order)          → 19 个
├── 形状特征 3D (Shape 3D)              → 16 个
├── 形状特征 2D (Shape 2D)              → 10 个
├── 灰度共生矩阵 (GLCM)                → 24 个
├── 灰度游程长度矩阵 (GLRLM)            → 16 个
├── 灰度大小区域矩阵 (GLSZM)            → 16 个
├── 邻域灰度差分矩阵 (NGTDM)            → 5 个
└── 灰度依赖矩阵 (GLDM)                → 14 个
```

### 2.1 一阶统计特征（First Order Statistics）— 19 个

一阶统计特征描述 ROI 内体素强度的**分布特征**，仅考虑单个体素的灰度值，不涉及体素间的空间关系。

| 特征名 | 英文名 | 公式/描述 | 临床意义 |
|--------|--------|----------|---------|
| 能量 | Energy | Σ(X(i)+c)² | 体素强度的大小量度 |
| 总能量 | Total Energy | V_voxel × Energy | 考虑体素体积的能量 |
| 熵 | Entropy | -Σp(i)log₂(p(i)) | 不确定性/随机性 |
| 最小值 | Minimum | min(X) | ROI内最低灰度 |
| 10th百分位 | 10Percentile | P10 | 低灰度分布 |
| 90th百分位 | 90Percentile | P90 | 高灰度分布 |
| 最大值 | Maximum | max(X) | ROI内最高灰度 |
| 均值 | Mean | ΣX(i)/Np | 平均灰度强度 |
| 中位数 | Median | P50 | 中位灰度 |
| 四分位距 | InterquartileRange | P75 - P25 | 灰度分布离散度 |
| 范围 | Range | max(X) - min(X) | 灰度值范围 |
| 平均绝对偏差 | MeanAbsoluteDeviation | Σ\|X(i)-X̄\|/Np | 灰度偏离均值的程度 |
| 鲁棒平均绝对偏差 | RobustMeanAbsoluteDeviation | 10-90百分位间的MAD | 抗异常值的离散度 |
| 均方根 | RootMeanSquared | √(Σ(X(i)+c)²/Np) | 强度幅度的另一量度 |
| 标准差 | StandardDeviation | √Variance | 灰度变异程度 |
| 偏度 | Skewness | μ₃/σ³ | 分布不对称性 |
| 峰度 | Kurtosis | μ₄/σ⁴ | 分布尖锐程度 |
| 方差 | Variance | Σ(X(i)-X̄)²/Np | 灰度分散程度 |
| 均匀度 | Uniformity | Σp(i)² | 灰度均匀性 |

**关键参数**：
- `binWidth`（默认25）：离散化灰度的直方图bin宽度，直接影响特征值
- `voxelArrayShift`（默认0）：灰度偏移量，防止负值影响 Energy/RMS 计算

### 2.2 形状特征 3D（Shape-based 3D）— 16 个

形状特征描述 ROI 的**三维几何形态**，与灰度值无关，仅从分割掩码中计算。使用 Marching Cubes 算法生成三角网格来近似 ROI 表面。

| 特征名 | 英文名 | 描述 | 临床意义 |
|--------|--------|------|---------|
| 网格体积 | MeshVolume | 三角网格计算的体积 | 肿瘤体积 |
| 体素体积 | VoxelVolume | 体素数×单个体素体积 | 近似体积 |
| 表面积 | SurfaceArea | 三角网格表面积 | 肿瘤表面积 |
| 表面积体积比 | SurfaceVolumeRatio | A/V | 形态紧凑度 |
| 球形度 | Sphericity | (36πV²)^(1/3)/A | 越接近1越像球体 |
| 紧凑度1 | Compactness1 | V/(√πA³) | 与球形度相关 |
| 紧凑度2 | Compactness2 | 36πV²/A³ | 球形度的立方 |
| 球形不对称度 | SphericalDisproportion | A/(4πR²) | 球形度的倒数 |
| 最大3D直径 | Maximum3DDiameter | 表面顶点间最大欧氏距离 | 肿瘤最大径 |
| 最大2D直径(层面) | Maximum2DDiameterSlice | 轴位面最大径 | 轴位最大径 |
| 最大2D直径(列) | Maximum2DDiameterColumn | 冠状面最大径 | 冠状最大径 |
| 最大2D直径(行) | Maximum2DDiameterRow | 矢状面最大径 | 矢状最大径 |
| 主轴长度 | MajorAxisLength | 4√λ_major | 最长主轴 |
| 副轴长度 | MinorAxisLength | 4√λ_minor | 次长主轴 |
| 最短轴长度 | LeastAxisLength | 4√λ_least | 最短主轴 |
| 伸长度 | Elongation | √(λ_minor/λ_major) | 形状细长程度 |
| 扁平度 | Flatness | √(λ_least/λ_major) | 形状扁平程度 |

### 2.3 灰度共生矩阵特征（GLCM）— 24 个

GLCM 描述图像中**相邻体素灰度值的空间组合关系**。矩阵 P(i,j|δ,θ) 表示在距离 δ、角度 θ 方向上，灰度 i 和 j 同时出现的次数。

```
GLCM 示例（2D, δ=1, θ=0°）：

原始图像:          对称化 GLCM:
1 2 5 2 3         i\j  1   2   3   4   5
3 2 1 3 1              1 [ 6   4   3   0   0 ]
1 3 5 5 2              2 [ 4   0   2   1   3 ]
1 1 2 1 2              3 [ 3   2   0   1   2 ]
4 2 2 3 5              4 [ 0   1   1   0   0 ]
3 3 5 3 2              5 [ 0   3   2   0   2 ]
```

| 特征名 | 描述 | 临床意义 |
|--------|------|---------|
| 自相关 | Autocorrelation | 纹理粗细程度 |
| 联合平均 | JointAverage | i分布的均值 |
| 聚类显著性 | ClusterProminence | GLCM不对称性 |
| 聚类阴影 | ClusterShade | GLCM偏斜度 |
| 聚类趋势 | ClusterTendency | 相似灰度聚集程度 |
| 对比度 | Contrast | 局部强度变化 |
| 相关性 | Correlation | 灰度线性依赖性 |
| 差异平均 | DifferenceAverage | 灰度差异的均值 |
| 差异熵 | DifferenceEntropy | 灰度差异的随机性 |
| 差异方差 | DifferenceVariance | 灰度差异的异质性 |
| 联合能量 | JointEnergy | 均匀模式频率 |
| 联合熵 | JointEntropy | 邻域灰度随机性 |
| 逆差矩 | IDM | 局部均匀性 |
| 最大相关系数 | MCC | 纹理复杂度 |
| 逆差矩归一化 | IDMN | 归一化局部均匀性 |
| 逆差异 | ID | 另一种局部均匀性量度 |
| 逆差异归一化 | IDN | 归一化逆差异 |
| 逆方差 | InverseVariance | 灰度差异的逆加权 |
| 最大概率 | MaximumProbability | 最主要灰度对频率 |
| 和平均 | SumAverage | 高低灰度对关系 |
| 和熵 | SumEntropy | 灰度和的随机性 |
| 平方和 | SumSquares | i分布的方差 |
| IMC1 | 信息相关性度量1 | 分布复杂度 |
| IMC2 | 信息相关性度量2 | 互信息量度 |

**关键参数**：
- `distances`（默认[1]）：计算共生的距离列表
- `symmetricalGLCM`（默认True）：是否对称化矩阵

### 2.4 灰度游程长度矩阵特征（GLRLM）— 16 个

GLRLM 量化**连续相同灰度值的游程长度**。P(i,j|θ) 表示沿角度 θ 方向，灰度 i 连续出现 j 次的游程数量。

| 特征名 | 描述 | 临床意义 |
|--------|------|---------|
| 短游程强调 | SRE | 短游程越多→细纹理 |
| 长游程强调 | LRE | 长游程越多→粗纹理 |
| 灰度非均匀性 | GLN | 灰度值相似性 |
| 灰度非均匀性归一化 | GLNN | 归一化灰度相似性 |
| 游程长度非均匀性 | RLN | 游程长度同质性 |
| 游程长度非均匀性归一化 | RLNN | 归一化游程同质性 |
| 游程百分比 | RP | 纹理粗糙度 |
| 灰度方差 | GLV | 游程灰度变异 |
| 游程方差 | RV | 游程长度变异 |
| 游程熵 | RE | 游程分布随机性 |
| 低灰度游程强调 | LGLRE | 低灰度游程集中度 |
| 高灰度游程强调 | HGLRE | 高灰度游程集中度 |
| 短游程低灰度强调 | SRLGLE | 短游程+低灰度联合分布 |
| 短游程高灰度强调 | SRHGLE | 短游程+高灰度联合分布 |
| 长游程低灰度强调 | LRLGLE | 长游程+低灰度联合分布 |
| 长游程高灰度强调 | LRHGLE | 长游程+高灰度联合分布 |

### 2.5 灰度大小区域矩阵特征（GLSZM）— 16 个

GLSZM 量化**相同灰度值的连通区域大小**。与 GLRLM 不同，GLSZM 是**旋转无关的**，只计算一个矩阵。

| 特征名 | 描述 | 临床意义 |
|--------|------|---------|
| 小区域强调 | SAE | 小区域越多→细纹理 |
| 大区域强调 | LAE | 大区域越多→粗纹理 |
| 灰度非均匀性 | GLN | 灰度值相似性 |
| 灰度非均匀性归一化 | GLNN | 归一化灰度相似性 |
| 大小区域非均匀性 | SZN | 区域体积同质性 |
| 大小区域非均匀性归一化 | SZNN | 归一化区域同质性 |
| 区域百分比 | ZP | 纹理粗糙度 |
| 灰度方差 | GLV | 区域灰度变异 |
| 区域方差 | ZV | 区域大小变异 |
| 区域熵 | ZE | 区域分布随机性 |
| 低灰度区域强调 | LGLZE | 低灰度区域集中度 |
| 高灰度区域强调 | HGLZE | 高灰度区域集中度 |
| 小区域低灰度强调 | SALGLE | 小区域+低灰度联合 |
| 小区域高灰度强调 | SAHGLE | 小区域+高灰度联合 |
| 大区域低灰度强调 | LALGLE | 大区域+低灰度联合 |
| 大区域高灰度强调 | LAHGLE | 大区域+高灰度联合 |

### 2.6 邻域灰度差分矩阵特征（NGTDM）— 5 个

NGTDM 量化**体素灰度值与其邻域平均灰度值的差异**。

| 特征名 | 描述 | 临床意义 |
|--------|------|---------|
| 粗糙度 | Coarseness | 空间变化率，越高越均匀 |
| 对比度 | Contrast | 空间强度变化程度 |
| 忙碌度 | Busyness | 像素间强度变化频率 |
| 复杂度 | Complexity | 图像非均匀性程度 |
| 强度 | Strength | 纹理基元的清晰程度 |

### 2.7 灰度依赖矩阵特征（GLDM）— 14 个

GLDM 量化**邻域内灰度值相近的体素数量**。若邻域体素 j 与中心体素 i 的灰度差 |i-j| ≤ α，则认为 j 依赖于 i。

| 特征名 | 描述 | 临床意义 |
|--------|------|---------|
| 小依赖强调 | SDE | 小依赖越多→异质性 |
| 大依赖强调 | LDE | 大依赖越多→均匀性 |
| 灰度非均匀性 | GLN | 灰度值相似性 |
| 依赖非均匀性 | DN | 依赖大小同质性 |
| 依赖非均匀性归一化 | DNN | 归一化依赖同质性 |
| 灰度方差 | GLV | 灰度变异 |
| 依赖方差 | DV | 依赖大小变异 |
| 依赖熵 | DE | 依赖分布随机性 |
| 低灰度强调 | LGLE | 低灰度集中度 |
| 高灰度强调 | HGLE | 高灰度集中度 |
| 小依赖低灰度强调 | SDLGLE | 小依赖+低灰度联合 |
| 小依赖高灰度强调 | SDHGLE | 小依赖+高灰度联合 |
| 大依赖低灰度强调 | LDLGLE | 大依赖+低灰度联合 |
| 大依赖高灰度强调 | LDHGLE | 大依赖+高灰度联合 |

---

## 三、图像滤波与特征增强

PyRadiomics 不仅可以在原始图像上提取特征，还支持在**滤波后的派生图像**上提取特征，从而捕获不同尺度和频率的纹理信息。

### 3.1 支持的图像类型

| 图像类型 | 参数 | 说明 |
|---------|------|------|
| **Original** | 无 | 原始图像 |
| **LoG** | sigma: [1.0, 2.0, ...] | Laplacian of Gaussian，多尺度边缘增强 |
| **Wavelet** | 无 | 小波变换，8个方向的高低频分解 |
| **Square** | 无 | 平方变换 |
| **SquareRoot** | 无 | 平方根变换 |
| **Logarithm** | 无 | 对数变换 |
| **Exponential** | 无 | 指数变换 |
| **Gradient** | 无 | 梯度变换 |

### 3.2 小波变换详解

小波变换是 PyRadiomics 中最常用的滤波方法，对原始图像进行 3D 小波分解，产生 **8 个子带**：

```
原始图像 → 小波变换 → LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH
                      (低频)                        (高频)
                      
L = Low-pass (低通滤波)
H = High-pass (高通滤波)
三个字母分别对应三个轴 (x, y, z) 的滤波方式

LLL: 三个轴都低通 → 近似图像（低频成分）
HHH: 三个轴都高通 → 细节图像（高频边缘）
```

**临床意义**：
- 低频子带（LLL）捕获大尺度结构信息
- 高频子带（HHH等）捕获微细纹理和边缘信息
- 每个子带都可以独立提取特征，极大扩展了特征空间

### 3.3 LoG 滤波详解

Laplacian of Gaussian (LoG) 是一种多尺度边缘检测滤波器：

```
LoG(x,y) = -1/(πσ⁴) × [1 - (x²+y²)/(2σ²)] × exp(-(x²+y²)/(2σ²))

σ 控制检测的尺度：
- 小 σ (如1.0) → 检测细小边缘
- 大 σ (如5.0) → 检测粗大结构
```

**典型配置**：使用 σ = [1.0, 2.0, 3.0, 4.0, 5.0] 提取多尺度特征。

---

## 四、安装与配置

### 4.1 安装

```bash
# 使用 pip 安装
pip install pyradiomics

# 使用 conda 安装
conda install -c conda-forge pyradiomics

# 从源码安装
git clone https://github.com/Radiomics/pyradiomics.git
cd pyradiomics
pip install .
```

### 4.2 依赖项

| 依赖 | 用途 |
|------|------|
| SimpleITK | 图像加载与预处理 |
| numpy | 特征计算 |
| PyWavelets | 小波滤波 |
| pykwalify | YAML参数文件校验 |
| six | Python 3 兼容性 |

### 4.3 支持的图像格式

PyRadiomics 支持所有 ITK 可读的图像格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| NRRD | .nrrd, .nhdr | 推荐格式 |
| NIfTI | .nii, .nii.gz | 神经影像常用 |
| MetaImage | .mha, .mhd | ITK原生格式 |
| DICOM | .dcm | 需要SimpleITK读取 |

---

## 五、初步使用

### 5.1 最简使用 — 默认参数

```python
from radiomics import featureextractor
import SimpleITK as sitk

# 初始化提取器（使用默认参数）
extractor = featureextractor.RadiomicsFeatureExtractor()

# 从文件提取特征
result = extractor.execute('image.nrrd', 'mask.nrrd')

# 打印结果
for key, value in result.items():
    print(f"{key}: {value}")
```

默认配置将提取：
- 原始图像上的所有特征类
- 不包含任何滤波图像
- binWidth = 25

### 5.2 自定义参数 — 字典方式

```python
from radiomics import featureextractor

# 定义参数
params = {
    'binWidth': 25,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',
    'normalize': True,
    'normalizeScale': 1000,
}

# 初始化提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# 自定义要提取的特征类
extractor.enableFeatureClassByName('firstorder')   # 启用一阶特征
extractor.enableFeatureClassByName('glcm')          # 启用GLCM
extractor.enableFeatureClassByName('shape')         # 启用形状特征

# 自定义要提取的具体特征
extractor.enableFeaturesByName(
    firstorder=['Mean', 'Entropy', 'Skewness', 'Kurtosis'],
    glcm=['Contrast', 'Correlation', 'Entropy']
)

# 提取特征
result = extractor.execute('image.nii.gz', 'mask.nii.gz')

for key, value in result.items():
    if not key.startswith('diagnostics_'):
        print(f"{key}: {value:.6f}")
```

### 5.3 自定义参数 — YAML 文件方式（推荐）

创建参数文件 `params.yaml`：

```yaml
# 图像类型设置
imageType:
  Original: {}                    # 原始图像
  LoG:                            # LoG 滤波
    sigma: [1.0, 2.0, 3.0, 4.0, 5.0]
  Wavelet: {}                     # 小波变换（8个子带）

# 特征类设置
featureClass:
  firstorder: []                  # 启用所有一阶特征
  shape: []                       # 启用所有形状特征
  glcm: ['Autocorrelation', 'Contrast', 'Correlation', 'Entropy',
         'JointEnergy', 'JointEntropy']
  glrlm: ['ShortRunEmphasis', 'LongRunEmphasis', 'GrayLevelNonUniformity']
  glszm: ['SmallAreaEmphasis', 'LargeAreaEmphasis']
  ngtdm: ['Coarseness', 'Contrast', 'Busyness']
  gldm: ['SmallDependenceEmphasis', 'LargeDependenceEmphasis']

# 全局设置
setting:
  binWidth: 25                    # 直方图bin宽度
  label: 1                        # 分割标签值
  resampledPixelSpacing: [1, 1, 1]  # 重采样体素间距
  interpolator: 'sitkBSpline'     # 插值方法
  normalize: true                 # 是否归一化
  normalizeScale: 1000            # 归一化尺度
  removeOutliers: 1.0             # 去除异常值（±3σ外的值裁剪到±1.0σ）

# 特征类特定设置
featureClassSetting:
  glcm:
    distances: [1]                # 共生距离
    symmetricalGLCM: true         # 对称化GLCM
```

使用参数文件：

```python
from radiomics import featureextractor

# 使用参数文件初始化
extractor = featureextractor.RadiomicsFeatureExtractor('params.yaml')

# 提取特征
result = extractor.execute('image.nii.gz', 'mask.nii.gz')
```

### 5.4 从 numpy 数组提取特征

```python
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

# 假设已有 numpy 数组
image_array = np.random.rand(64, 128, 128) * 1000  # 模拟CT图像
mask_array = np.zeros((64, 128, 128), dtype=np.uint8)
mask_array[20:44, 40:88, 40:88] = 1                 # 模拟肿瘤区域

# 转换为 SimpleITK 图像
image = sitk.GetImageFromArray(image_array)
mask = sitk.GetImageFromArray(mask_array)

# 设置体素间距（非常重要！）
spacing = (1.0, 1.0, 1.0)  # mm
image.SetSpacing(spacing)
mask.SetSpacing(spacing)

# 提取特征
extractor = featureextractor.RadiomicsFeatureExtractor()
result = extractor.execute(image, mask)

# 输出特征
features = {}
for key, value in result.items():
    if not key.startswith('diagnostics_'):
        features[key] = float(value)

print(f"提取特征数量: {len(features)}")
for key, value in sorted(features.items()):
    print(f"  {key}: {value:.6f}")
```

### 5.5 批量处理多个病例

```python
import pandas as pd
from radiomics import featureextractor
import SimpleITK as sitk
import os

# 初始化提取器
extractor = featureextractor.RadiomicsFeatureExtractor('params.yaml')

# 病例列表
cases = [
    {'id': 'case001', 'image': 'data/case001_image.nii.gz', 'mask': 'data/case001_mask.nii.gz'},
    {'id': 'case002', 'image': 'data/case002_image.nii.gz', 'mask': 'data/case002_mask.nii.gz'},
    {'id': 'case003', 'image': 'data/case003_image.nii.gz', 'mask': 'data/case003_mask.nii.gz'},
]

# 批量提取
all_features = []
for case in cases:
    print(f"处理 {case['id']}...")
    
    result = extractor.execute(case['image'], case['mask'])
    
    # 提取特征值
    feature_dict = {'case_id': case['id']}
    for key, value in result.items():
        if not key.startswith('diagnostics_'):
            feature_dict[key] = float(value)
    
    all_features.append(feature_dict)

# 转换为 DataFrame
df = pd.DataFrame(all_features)
df.to_csv('radiomics_features.csv', index=False)

print(f"\n完成！共提取 {len(df)} 个病例，{len(df.columns)-1} 个特征")
print(f"特征已保存到 radiomics_features.csv")
```

### 5.6 命令行使用

```bash
# 单个病例
pyradiomics image.nrrd mask.nrrd

# 使用参数文件
pyradiomics image.nrrd mask.nrrd --param params.yaml

# 输出到CSV
pyradiomics image.nrrd mask.nrrd -o results.csv -f csv

# 批量处理
pyradiomics batch_input.csv -o results.csv -f csv --param params.yaml

# 多进程加速
pyradiomics batch_input.csv -o results.csv -f csv --jobs 4

# 体素级特征提取（生成特征图）
pyradiomics image.nrrd mask.nrrd --mode voxel --out-dir feature_maps/
```

批量输入CSV格式：
```csv
Image,Mask
data/case001_image.nii.gz,data/case001_mask.nii.gz
data/case002_image.nii.gz,data/case002_mask.nii.gz
```

---

## 六、特征结果的解读

### 6.1 特征命名规则

PyRadiomics 的特征命名遵循统一格式：

```
<filter>_<featureClass>_<featureName>

示例：
original_firstorder_Mean          → 原始图像的一阶均值
original_glcm_Correlation         → 原始图像的GLCM相关性
log-sigma-1-mm_firstorder_Entropy → LoG(σ=1mm)图像的一阶熵
wavelet-LLH_glcm_Contrast         → 小波LLH子带的GLCM对比度
```

### 6.2 诊断信息

`execute()` 返回的结果中包含 `diagnostics_` 前缀的元数据：

```python
# 诊断信息类别
diagnostics_Versions_PyRadiomics          # PyRadiomics版本
diagnostics_Versions_Numpy                # NumPy版本
diagnostics_Configuration_Settings        # 配置参数
diagnostics_Image-original_Hash           # 图像哈希
diagnostics_Image-original_Spacing        # 图像间距
diagnostics_Image-original_Size           # 图像尺寸
diagnostics_Mask-original_Spacing         # 掩码间距
diagnostics_Mask-original_Size            # 掩码尺寸
diagnostics_Mask-original_Center          # 掩码中心
diagnostics_Mask-original_BoundingBox     # 边界框
diagnostics_Mask-original_VoxelNum        # ROI体素数
```

### 6.3 特征数量估算

| 配置 | 特征数量 |
|------|---------|
| 仅原始图像 + 所有特征类 | ~110 |
| 原始 + LoG(5个σ) + 所有特征类 | ~660 |
| 原始 + 小波(8子带) + 所有特征类 | ~990 |
| 原始 + LoG(5) + 小波(8) + 所有特征类 | ~1540 |

---

## 七、最佳实践与注意事项

### 7.1 图像预处理建议

```python
params = {
    # 重采样到统一间距（强烈推荐）
    'resampledPixelSpacing': [1, 1, 1],
    'interpolator': 'sitkBSpline',
    
    # 归一化（CT推荐）
    'normalize': True,
    'normalizeScale': 1000,
    
    # 去除异常值
    'removeOutliers': 3.0,
    
    # binWidth选择
    # CT: 25 (经验值)
    # MRI: 根据序列调整
    'binWidth': 25,
}
```

### 7.2 binWidth 的选择

`binWidth` 是影响特征值最敏感的参数之一：

| 影像类型 | 推荐 binWidth | 说明 |
|---------|--------------|------|
| CT | 25 | 灰度范围约 -1000~3000 HU |
| MRI T1 | 10-25 | 需根据序列调整 |
| MRI T2 | 10-25 | 需根据序列调整 |
| PET | 2.5-25 | SUV值范围较小 |

### 7.3 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 特征值为 NaN | ROI体素数太少 | 确保 ROI ≥ 64 个体素 |
| 特征不可复现 | 参数不一致 | 使用YAML参数文件 |
| 内存溢出 | 图像+滤波太大 | 减少滤波类型或裁剪ROI |
| 计算时间过长 | 特征+滤波太多 | 减少不必要的特征类 |
| 不同版本结果不同 | 算法更新 | 固定PyRadiomics版本 |

### 7.4 IBSI 合规性

PyRadiomics 尽量遵循 IBSI（影像生物标志物标准化倡议）的定义，但存在少数差异：

| 差异点 | PyRadiomics | IBSI |
|--------|------------|------|
| 峰度 | 未减3 | 超额峰度（减3） |
| 均匀度 | 在原始图像上计算 | 在重采样后计算 |
| Energy | 可选偏移量 c | 无偏移量 |

---

## 八、与血管分割项目的集成

在我们的 `vessel_segmentation_3d` 项目中，PyRadiomics 被用于提取肿瘤和血管区域的放射组学特征：

```python
from vessel_segmentation_3d.morphometry.radiomics_extractor import RadiomicsFeatureExtractor

# 初始化提取器
extractor = RadiomicsFeatureExtractor()

# 提取肿瘤区域特征
tumor_features = extractor.extract_features_from_arrays(
    image_array=ct_image,
    mask_array=tumor_mask,
    spacing=(1.0, 1.0, 1.0)
)

# 提取血管区域特征
vessel_features = extractor.extract_features_from_arrays(
    image_array=ct_image,
    mask_array=vessel_mask,
    spacing=(1.0, 1.0, 1.0)
)

# 合并特征（自动添加前缀）
all_features = {}
all_features.update({f'tumor_{k}': v for k, v in tumor_features.items()})
all_features.update({f'vessel_{k}': v for k, v in vessel_features.items()})
```

**特征融合策略**：
- 传统形态学特征（曲率、扭率、分支密度）→ 描述血管几何结构
- 放射组学特征（一阶、纹理、形状）→ 描述组织纹理和形态
- 两者互补，提供更全面的疾病表型描述

---

## 九、参考资源

| 资源 | 链接 |
|------|------|
| 官方文档 | https://pyradiomics.readthedocs.io/ |
| GitHub 仓库 | https://github.com/Radiomics/pyradiomics |
| 原始论文 | van Griethuysen et al., Cancer Research, 2017 |
| IBSI 标准 | https://theibsi.github.io/ |
| 3D Slicer 扩展 | https://github.com/Radiomics/SlicerRadiomics |
| 示例代码 | https://github.com/Radiomics/pyradiomics/tree/master/examples |

---

*本文档由医学影像AI研究团队整理，最后更新：2026-04-08*
