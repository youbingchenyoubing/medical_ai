# 基于MRI Delta影像组学预测HCC免疫治疗后病理完全缓解 — 技术实现方案

---

## 一、项目概述

### 1.1 研究目标

构建并验证基于MRI动态delta影像组学的模型，预测肝癌（HCC）转化治疗后的病理完全缓解（pCR），探索结合AFP动态变化能否进一步提升预测准确性。

### 1.2 数据概况

| 项目 | 详情 |
|------|------|
| 总样本量 | 154例患者（159个病灶） |
| 训练集 | 78例（中山医院） |
| 内部测试集 | 32例（中山医院） |
| 外部验证集 | 44例（天津肿瘤医院、瑞金医院） |
| 影像数据 | 多序列MRI（T2WI、DWI、T1WI、动脉期、门脉期、延迟期等8个序列） |
| 临床数据 | AFP、PIVKA-II、肝功能、治疗方案等 |
| 病理金标准 | pCR |

### 1.3 技术流程总览

```
原始MRI图像 → 数据预处理 → 肿瘤分割 → 特征提取 → Delta特征计算 → 特征筛选 → 模型构建 → 模型评估
```

---

## 二、数据预处理

### 2.1 预处理流程

```
原始DICOM/NIfTI → 图像配准 → N4偏置场校正 → 归一化 → 预处理后图像
```

### 2.2 技术实现细节

#### 2.2.1 图像配准

- **目的**：将同一患者不同时间点（治疗前/后）及不同序列的MRI图像对齐到统一空间
- **方法**：基于SimpleITK的刚性/仿射配准
- **实现要点**：
  - 以T1WI增强图像为参考图像
  - 其他序列（T2WI、DWI等）配准到参考序列空间
  - 治疗后图像配准到治疗前图像空间
  - 使用互信息（Mutual Information）作为相似性度量
  - 线性插值重采样

```python
def register_images(fixed_image_path, moving_image_path, output_path):
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    final_transform = registration.Execute(fixed_image, moving_image)
    registered_image = sitk.Resample(moving_image, fixed_image, 
                                      final_transform, sitk.sitkLinear, 0.0,
                                      moving_image.GetPixelID())
    sitk.WriteImage(registered_image, output_path)
```

#### 2.2.2 N4偏置场校正

- **目的**：消除MRI图像中由磁场不均匀性引起的信号强度偏移
- **方法**：N4ITK偏置场校正算法
- **实现要点**：
  - 对每个MRI序列独立执行N4校正
  - 收敛阈值设为0.001
  - 最大迭代次数500
  - B样条网格间距[200, 200, 200]

```python
def n4_bias_correction(image_path, output_path):
    image = sitk.ReadImage(image_path)
    mask = sitk.OtsuThreshold(image, 0, 1)
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([500, 500, 500, 500])
    corrector.SetConvergenceThreshold(0.001)
    corrector.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    
    corrected_image = corrector.Execute(image, mask)
    sitk.WriteImage(corrected_image, output_path)
```

#### 2.2.3 归一化

- **目的**：消除不同患者、不同扫描参数间的信号强度差异
- **方法**：Z-score归一化（基于肿瘤区域统计量）
- **实现要点**：
  - 基于肿瘤mask区域计算均值和标准差
  - 全图应用：`(image - mean_tumor) / std_tumor`
  - 非肿瘤区域裁剪到[-3, 3]范围

```python
def normalize_by_tumor_region(image, mask):
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)
    
    tumor_values = image_array[mask_array > 0]
    mean_val = np.mean(tumor_values)
    std_val = np.std(tumor_values)
    
    normalized = (image_array - mean_val) / std_val
    normalized = np.clip(normalized, -3, 3)
    
    result = sitk.GetImageFromArray(normalized)
    result.CopyInformation(image)
    return result
```

### 2.3 多序列处理策略

对8个MRI序列分别执行预处理，每个序列独立处理：

| 序列 | 配准参考 | 特殊处理 |
|------|---------|---------|
| T1WI | 参考序列 | - |
| T1WI增强（动脉期） | 参考序列 | - |
| T1WI增强（门脉期） | 配准到动脉期 | - |
| T1WI增强（延迟期） | 配准到动脉期 | - |
| T2WI | 配准到T1WI | - |
| DWI | 配准到T1WI | ADC图额外计算 |
| ADC | 配准到T1WI | 由DWI序列计算生成 |

---

## 三、肿瘤分割

### 3.1 分割策略

```
AI自动分割（nnU-Net） → 人工审核修正 → ICC重复性验证 → 最终分割mask
```

### 3.2 nnU-Net自动分割

- **框架**：nnU-Net（自适应配置的U-Net变体）
- **训练策略**：
  - 使用训练集78例的专家标注作为训练数据
  - nnU-Net自动分析数据集特征，选择最优网络配置
  - 3D全分辨率配置 + 3D低分辨率配置集成
  - 5折交叉验证

```python
import nnunet
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

# nnU-Net数据集准备
# Dataset ID: 100, Task: HCC_Tumor_Segmentation
# 数据组织：
#   nnUNet_raw/nnUNet_raw_data/Task100_HCC/
#     imagesTr/   - 训练图像
#     labelsTr/   - 训练标签
#     imagesTs/   - 测试图像

# 训练命令
# nnUNet_plan_and_preprocess -t 100
# nnUNet_train 3d_fullres nnUNetTrainerV2 Task100_HCC 0
# nnUNet_train 3d_fullres nnUNetTrainerV2 Task100_HCC 1
# ...
# nnUNet_train 3d_fullres nnUNetTrainerV2 Task100_HCC 4

# 推理命令
# nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER \
#   -t 100 -m 3d_fullres -f all
```

### 3.3 人工审核修正

- 由2名放射科医师（≥5年经验）独立审核AI分割结果
- 不一致时由第3名高年资医师裁定
- 修正工具：3D Slicer / ITK-SNAP

### 3.4 ICC重复性验证

- **目的**：评估分割的可重复性，筛选高重复性特征
- **方法**：
  - 随机抽取30例，由2名医师独立分割
  - 计算两组分割提取特征的一致性相关系数（ICC）
  - ICC采用双向随机效应模型、一致性度量
  - **筛选标准**：ICC ≥ 0.80 的特征保留

```python
from pingouin import intraclass_corr

def calculate_icc(features_rater1, features_rater2, feature_names):
    icc_results = {}
    high_reproducibility_features = []
    
    for feat in feature_names:
        df_icc = pd.DataFrame({
            'rater': ['R1']*len(features_rater1) + ['R2']*len(features_rater2),
            'subject': list(range(len(features_rater1))) * 2,
            'score': list(features_rater1[feat]) + list(features_rater2[feat])
        })
        
        icc = intraclass_corr(data=df_icc, targets='subject',
                              raters='rater', scores='score')
        icc_val = icc[icc['Type'] == 'ICC2']['ICC'].values[0]
        icc_results[feat] = icc_val
        
        if icc_val >= 0.80:
            high_reproducibility_features.append(feat)
    
    return icc_results, high_reproducibility_features
```

---

## 四、特征提取

### 4.1 提取工具与参数

- **工具**：PyRadiomics（v3.0+）
- **总特征数**：2264个特征/每序列/每时间点
- **图像类型**：原始图像 + 派生图像（LoG、小波变换）

### 4.2 特征类别

| 特征类别 | 特征数量 | 说明 |
|---------|---------|------|
| 一阶统计特征（First Order） | 18 | 均值、中位数、偏度、峰度、熵等 |
| 形状特征（Shape） | 14 | 体积、表面积、球形度、紧凑度等 |
| 灰度共生矩阵（GLCM） | 24 | 纹理对比度、相关性、能量、同质性等 |
| 灰度游程矩阵（GLRLM） | 16 | 短游程强调、长游程强调等 |
| 灰度尺寸区域矩阵（GLSZM） | 16 | 小区域强调、大区域强调等 |
| 灰度依赖矩阵（GLDM） | 14 | 灰度依赖非均匀性等 |
| 邻域灰度差矩阵（NGTDM） | 5 | 粗糙度、对比度、繁忙度等 |

### 4.3 派生图像特征

| 派生类型 | 参数 | 生成图像数 |
|---------|------|-----------|
| LoG（高斯拉普拉斯） | sigma = 1.0, 2.0, 3.0, 5.0 | 4组 |
| 小波变换（Wavelet） | Haar小波，8个方向 | 8组 |

**特征总数计算**：
- 每个序列每个时间点：(18 + 14 + 24 + 16 + 16 + 14 + 5) × (1原始 + 4LoG + 8小波) = 107 × 13 = 1391
- 8个MRI序列：1391 × 8 ≈ 不全序列均提取，最终约2264个特征

### 4.4 PyRadiomics配置

```yaml
feature_extraction:
  bin_width: 25
  resampled_spacing: [1, 1, 1]
  interpolator: "sitkBSpline"
  normalize: true
  force2D: false
  
  image_types:
    - Original
    - LoG:
        sigma: [1.0, 2.0, 3.0, 5.0]
    - Wavelet:
        wavelet_type: "haar"
  
  feature_classes:
    - firstorder
    - shape
    - glcm
    - glrlm
    - glszm
    - gldm
    - ngtdm
```

```python
from radiomics import featureextractor
import SimpleITK as sitk

extractor = featureextractor.RadiomicsFeatureExtractor()

extractor.settings['binWidth'] = 25
extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
extractor.settings['interpolator'] = 'sitkBSpline'
extractor.settings['normalize'] = True

extractor.enableImageTypeByName('Original')
extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 5.0]})
extractor.enableImageTypeByName('Wavelet')

extractor.enableAllFeatures()

image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)
features = extractor.execute(image, mask)
```

---

## 五、Delta特征计算

### 5.1 计算公式

Delta特征反映治疗前后影像组学特征的动态变化率：

$$
\Delta Feature = \frac{Feature_{pre} - Feature_{post}}{Feature_{pre}}
$$

其中：
- $Feature_{pre}$：治疗前（基线）特征值
- $Feature_{post}$：治疗后（术前）特征值

### 5.2 实现要点

- 对每个特征独立计算delta值
- 分母为零或接近零的特征需特殊处理（添加小常数ε=1e-7）
- delta特征可为正（治疗后降低）或负（治疗后升高）
- 对8个MRI序列分别计算delta特征

```python
def compute_delta_features(features_pre, features_post, feature_names, epsilon=1e-7):
    delta_features = {}
    
    for feat in feature_names:
        pre_val = features_pre[feat]
        post_val = features_post[feat]
        
        if abs(pre_val) < epsilon:
            delta_features[f'delta_{feat}'] = 0.0
        else:
            delta_features[f'delta_{feat}'] = (pre_val - post_val) / pre_val
    
    return delta_features
```

### 5.3 三类模型特征对比

| 模型类型 | 特征来源 | 说明 |
|---------|---------|------|
| 基线模型 | 仅治疗前特征 | 静态特征，预测能力有限 |
| 术前模型 | 仅治疗后特征 | 静态特征，缺乏变化信息 |
| **Delta模型** | 治疗前后变化率 | **动态特征，预测能力最强** |

---

## 六、特征筛选

### 6.1 筛选流程

```
2264个原始特征 → ICC筛选 → t检验筛选 → Spearman去相关 → LASSO筛选 → 随机森林筛选 → 最终特征集
```

### 6.2 各步骤详细说明

#### Step 1：ICC重复性筛选

- **目的**：剔除分割不确定导致的低重复性特征
- **方法**：计算2名医师分割提取特征的ICC
- **阈值**：ICC ≥ 0.80 保留
- **预期**：约剔除10-20%特征

#### Step 2：独立样本t检验

- **目的**：筛选pCR组与非pCR组间有显著差异的特征
- **方法**：对每个特征进行独立样本t检验
- **阈值**：p < 0.05 保留
- **实现**：

```python
from scipy import stats

def t_test_selection(X, y, feature_names, alpha=0.05):
    pcr_mask = y == 1
    non_pcr_mask = y == 0
    
    selected_features = []
    p_values = {}
    
    for i, feat in enumerate(feature_names):
        pcr_values = X[pcr_mask, i]
        non_pcr_values = X[non_pcr_mask, i]
        
        stat, p_val = stats.ttest_ind(pcr_values, non_pcr_values)
        p_values[feat] = p_val
        
        if p_val < alpha:
            selected_features.append(feat)
    
    return selected_features, p_values
```

#### Step 3：Spearman相关性去冗余

- **目的**：去除高度相关的冗余特征
- **方法**：计算特征间Spearman相关系数
- **阈值**：|r| ≥ 0.9 的特征对，保留与标签相关性更高的那个
- **实现**：

```python
from scipy.stats import spearmanr

def spearman_deduplication(X, y, feature_names, threshold=0.9):
    corr_matrix, _ = spearmanr(X)
    
    feature_label_corr = []
    for i in range(X.shape[1]):
        corr, _ = spearmanr(X[:, i], y)
        feature_label_corr.append(abs(corr))
    
    remove_set = set()
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            if abs(corr_matrix[i, j]) >= threshold:
                if feature_label_corr[i] < feature_label_corr[j]:
                    remove_set.add(i)
                else:
                    remove_set.add(j)
    
    selected_idx = [i for i in range(len(feature_names)) if i not in remove_set]
    return [feature_names[i] for i in selected_idx]
```

#### Step 4：LASSO回归筛选

- **目的**：通过L1正则化进行特征选择，进一步压缩特征维度
- **方法**：10折交叉验证选择最优lambda
- **实现**：

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def lasso_selection(X, y, feature_names, cv=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = LassoCV(cv=cv, random_state=42, max_iter=10000, n_jobs=-1)
    lasso.fit(X_scaled, y)
    
    non_zero_idx = np.where(lasso.coef_ != 0)[0]
    selected_features = [feature_names[i] for i in non_zero_idx]
    
    return selected_features, lasso.coef_
```

#### Step 5：随机森林重要性筛选

- **目的**：基于特征重要性排序，选择最终特征子集
- **方法**：随机森林Gini重要性 + 排序
- **实现**：

```python
from sklearn.ensemble import RandomForestClassifier

def random_forest_selection(X, y, feature_names, n_top=15):
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    selected_idx = sorted_idx[:n_top]
    selected_features = [feature_names[i] for i in selected_idx]
    
    return selected_features, importances[sorted_idx]
```

### 6.3 筛选结果汇总

| 筛选步骤 | 方法 | 阈值 | 预期保留特征数 |
|---------|------|------|-------------|
| 1 | ICC | ≥ 0.80 | ~1800-2000 |
| 2 | t检验 | p < 0.05 | ~200-400 |
| 3 | Spearman | \|r\| < 0.9 | ~50-100 |
| 4 | LASSO | CV最优lambda | ~15-30 |
| 5 | 随机森林 | Top N | ~10-15 |

---

## 七、模型构建

### 7.1 模型对比策略

对14种机器学习模型进行系统性对比，最终选择最优模型：

| 编号 | 模型 | 英文名 | 类别 |
|------|------|--------|------|
| 1 | 逻辑回归 | Logistic Regression (LR) | 线性 |
| 2 | 支持向量机 | SVM | 核方法 |
| 3 | K近邻 | KNN | 距离方法 |
| 4 | 决策树 | Decision Tree (DT) | 树模型 |
| 5 | 随机森林 | Random Forest (RF) | 集成-Bagging |
| 6 | 极端随机树 | Extra Trees (ET) | 集成-Bagging |
| 7 | AdaBoost | AdaBoost | 集成-Boosting |
| 8 | 梯度提升 | Gradient Boosting (GB) | 集成-Boosting |
| 9 | XGBoost | XGBoost | 集成-Boosting |
| 10 | LightGBM | LightGBM | 集成-Boosting |
| 11 | CatBoost | CatBoost | 集成-Boosting |
| 12 | 高斯朴素贝叶斯 | Gaussian NB (GNB) | 概率模型 |
| 13 | 线性判别分析 | LDA | 判别分析 |
| 14 | **多层感知机** | **MLP** | **神经网络** |

### 7.2 最终选择：MLP

**选择依据**：MLP在训练集、内部测试集和外部验证集上综合表现最优

```python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (64, 32, 16)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [16, 32, 'auto']
}

grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_mlp = grid_search.best_estimator_
```

### 7.3 14种模型统一训练框架

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    pass

model_configs = {
    'LR': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DT': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced'),
    'ET': ExtraTreesClassifier(n_estimators=500, random_state=42, class_weight='balanced'),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'GB': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=500, random_state=42, use_label_encoder=False),
    'LightGBM': LGBMClassifier(n_estimators=500, random_state=42, class_weight='balanced'),
    'CatBoost': CatBoostClassifier(iterations=500, random_state=42, verbose=0),
    'GNB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=1000,
                          early_stopping=True, random_state=42)
}
```

### 7.4 联合模型构建

在delta影像组学模型基础上，联合AFP应答构建综合临床-放射组学模型：

```python
from sklearn.linear_model import LogisticRegression

# AFP应答定义：AFP下降率 = (AFP_pre - AFP_post) / AFP_pre
# 单因素/多因素Logistic回归确认AFP应答是pCR独立预测因子

# 联合模型：delta影像组学评分 + AFP应答 → Logistic回归
combined_features = np.column_stack([radiomics_score, afp_response])
combined_model = LogisticRegression(random_state=42, class_weight='balanced')
combined_model.fit(combined_features_train, y_train)
```

---

## 八、模型评估

### 8.1 评估指标体系

| 指标 | 英文 | 公式 | 说明 |
|------|------|------|------|
| AUC | Area Under ROC Curve | - | 区分能力综合指标 |
| 准确性 | Accuracy | (TP+TN)/(TP+TN+FP+FN) | 整体正确率 |
| 敏感性 | Sensitivity (Recall) | TP/(TP+FN) | pCR检出率 |
| 特异性 | Specificity | TN/(TN+FP) | 非pCR正确排除率 |
| 阳性预测值 | PPV (Precision) | TP/(TP+FP) | 预测为pCR的准确率 |
| 阴性预测值 | NPV | TN/(TN+FN) | 预测为非pCR的准确率 |

### 8.2 校准曲线

- **目的**：评估模型预测概率与实际概率的一致性
- **方法**：将预测概率分10组，比较每组平均预测概率与实际pCR率
- **实现**：

```python
from sklearn.calibration import calibration_curve

def plot_calibration_curve(y_true, y_prob, model_name, output_path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(prob_pred, prob_true, 'o-', label=model_name)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 8.3 决策曲线分析（DCA）

- **目的**：评估模型在不同阈值概率下的临床净收益
- **方法**：计算阈值概率范围内的净收益
- **实现**：

```python
def decision_curve_analysis(y_true, y_prob, output_path):
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    net_benefit_model = []
    net_benefit_all = []
    
    prevalence = y_true.mean()
    
    for threshold in thresholds:
        tp = np.sum((y_prob >= threshold) & (y_true == 1))
        fp = np.sum((y_prob >= threshold) & (y_true == 0))
        n = len(y_true)
        
        net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
        net_benefit_model.append(net_benefit)
        net_benefit_all.append(prevalence - (1 - prevalence) * threshold / (1 - threshold))
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefit_model, 'b-', label='Model')
    plt.plot(thresholds, net_benefit_all, 'r--', label='Treat All')
    plt.axhline(y=0, color='k', linestyle=':', label='Treat None')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
```

### 8.4 SHAP可解释性分析

- **目的**：解释模型预测结果，识别关键预测因子
- **方法**：SHAP（SHapley Additive exPlanations）值分析
- **输出**：
  - 特征重要性图（Summary Plot）
  - 特征贡献图（Dependence Plot）

```python
import shap

def shap_analysis(model, X, feature_names, output_dir):
    explainer = shap.KernelExplainer(model.predict_proba, X[:100])
    shap_values = explainer.shap_values(X)
    
    # 特征重要性图
    plt.figure()
    shap.summary_plot(shap_values[1], X, feature_names=feature_names, show=False)
    plt.savefig(f'{output_dir}/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 特征贡献图
    for feat in feature_names[:5]:
        plt.figure()
        feat_idx = list(feature_names).index(feat)
        shap.dependence_plot(feat_idx, shap_values[1], X, 
                            feature_names=feature_names, show=False)
        plt.savefig(f'{output_dir}/shap_dependence_{feat}.png', dpi=300, bbox_inches='tight')
        plt.close()
```

---

## 九、实验结果

### 9.1 三类影像组学模型对比

| 模型 | 训练集AUC | 测试集AUC | 外部验证集AUC |
|------|----------|----------|-------------|
| 基线模型（治疗前） | 0.835 | 0.483 | 0.434 |
| 术前模型（治疗后） | 0.770 | 0.685 | 0.506 |
| **Delta影像组学模型** | **0.879** | **0.835** | **0.783** |

**关键发现**：Delta影像组学模型在所有数据集上均显著优于静态特征模型，验证了动态变化特征的核心价值。

### 9.2 联合模型对比

| 模型 | 测试集AUC | 外部验证集AUC |
|------|----------|-------------|
| 仅临床模型（AFP） | 0.796 | 0.754 |
| 仅Delta影像组学模型 | 0.819 | 0.781 |
| **影像+AFP联合模型** | **0.920** | **0.857** |

**关键发现**：联合模型在测试集和外部验证集上均达到最高AUC，证明AFP应答与delta影像组学特征具有互补性。

### 9.3 SHAP分析结论

- AFP应答和放射组学评分是pCR最显著的预测因子
- 较高的AFP应答值与pCR可能性呈负相关
- 较高的放射组学评分与pCR呈正相关

---

## 十、技术依赖

### 10.1 核心依赖

| 库 | 版本 | 用途 |
|----|------|------|
| Python | ≥ 3.8 | 运行环境 |
| SimpleITK | ≥ 2.1 | 图像读写、配准、N4校正 |
| PyRadiomics | ≥ 3.0 | 影像组学特征提取 |
| nnU-Net | ≥ 1.7 | 肿瘤自动分割 |
| scikit-learn | ≥ 1.0 | 机器学习模型、特征选择、评估 |
| XGBoost | ≥ 1.5 | XGBoost模型 |
| LightGBM | ≥ 3.3 | LightGBM模型 |
| CatBoost | ≥ 1.0 | CatBoost模型 |
| SHAP | ≥ 0.40 | 模型可解释性 |
| pingouin | ≥ 0.5 | ICC计算 |
| scipy | ≥ 1.7 | 统计检验 |
| matplotlib | ≥ 3.4 | 可视化 |
| seaborn | ≥ 0.11 | 可视化 |
| pandas | ≥ 1.3 | 数据处理 |
| numpy | ≥ 1.21 | 数值计算 |

### 10.2 安装命令

```bash
pip install SimpleITK PyRadiomics scikit-learn xgboost lightgbm catboost \
            shap pingouin scipy matplotlib seaborn pandas numpy \
            nnunet openpyxl
```

---

## 十一、完整流程代码架构

```
radiomics_project/
├── config/
│   └── config.yaml              # 全局配置
├── data/
│   ├── raw/                     # 原始DICOM/NIfTI
│   ├── processed/               # 预处理后图像
│   └── masks/                   # 分割mask
├── src/
│   ├── data_preprocessing.py    # 图像配准、N4校正、归一化
│   ├── feature_extraction.py    # PyRadiomics特征提取
│   ├── feature_selection.py     # ICC、t检验、Spearman、LASSO、RF筛选
│   ├── model_training.py        # 14种模型训练 + MLP
│   ├── evaluation.py            # AUC、校准曲线、DCA
│   ├── delta_features.py        # Delta特征计算
│   ├── icc_analysis.py          # ICC重复性分析
│   └── shap_analysis.py         # SHAP可解释性
├── results/
│   ├── features/                # 特征CSV
│   ├── models/                  # 训练好的模型
│   └── figures/                 # 可视化图表
├── main.py                      # 主流程入口
└── requirements.txt             # 依赖清单
```

---

## 十二、运行流程

```bash
# Step 1: 数据预处理（配准 + N4校正 + 归一化）
python main.py --step 1

# Step 2: 肿瘤分割（nnU-Net自动分割 + 人工修正）
# 需单独运行nnU-Net训练和推理

# Step 3: 特征提取（PyRadiomics）
python main.py --step 2

# Step 4: Delta特征计算
python main.py --step 3

# Step 5: 特征筛选（ICC → t检验 → Spearman → LASSO → RF）
python main.py --step 4

# Step 6: 模型训练（14种模型对比）
python main.py --step 5

# Step 7: 模型评估 + SHAP分析
python main.py --step 6
```
