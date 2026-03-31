import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, config: dict):
        """
        初始化特征选择器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.method = config['feature_selection']['method']
        self.n_features = config['feature_selection']['n_features']
        self.cv_folds = config['feature_selection']['cv_folds']
        self.random_state = config['feature_selection']['random_state']
        
        self.scaler = StandardScaler()
        self.selected_features = None
        self.selected_indices = None
        
        logger.info(f"FeatureSelector initialized with method: {self.method}")
    
    def fit_transform(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        拟合并转换数据
        
        Args:
            X: 特征矩阵
            y: 标签
            feature_names: 特征名称列表
            
        Returns:
            选择后的特征矩阵和特征名称
        """
        logger.info(f"Selecting features using {self.method}")
        
        if self.method == 'lasso':
            return self._select_lasso(X, y, feature_names)
        elif self.method == 'rfe':
            return self._select_rfe(X, y, feature_names)
        elif self.method == 'mutual_info':
            return self._select_mutual_info(X, y, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _select_lasso(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """LASSO特征选择"""
        logger.info("Performing LASSO feature selection")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # LASSO回归
        lasso = LassoCV(
            cv=self.cv_folds, 
            random_state=self.random_state, 
            max_iter=10000,
            n_jobs=-1
        )
        lasso.fit(X_scaled, y)
        
        logger.info(f"Optimal alpha: {lasso.alpha_:.6f}")
        
        # 获取非零系数的特征
        coef = lasso.coef_
        non_zero_idx = np.where(coef != 0)[0]
        
        # 如果特征太多，选择系数绝对值最大的n_features个
        if len(non_zero_idx) > self.n_features:
            top_idx = np.argsort(np.abs(coef))[::-1][:self.n_features]
            selected_idx = top_idx
        else:
            selected_idx = non_zero_idx
        
        self.selected_indices = selected_idx
        self.selected_features = [feature_names[i] for i in selected_idx]
        
        logger.info(f"Selected {len(selected_idx)} features")
        
        # 绘制系数图
        self._plot_coefficients(coef, feature_names, selected_idx)
        
        return X_scaled[:, selected_idx], self.selected_features
    
    def _select_rfe(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """RFE特征选择"""
        logger.info("Performing RFE feature selection")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # RFE
        estimator = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        rfe = RFE(
            estimator, 
            n_features_to_select=self.n_features, 
            step=1
        )
        rfe.fit(X_scaled, y)
        
        selected_idx = np.where(rfe.support_)[0]
        
        self.selected_indices = selected_idx
        self.selected_features = [feature_names[i] for i in selected_idx]
        
        logger.info(f"Selected {len(selected_idx)} features")
        
        return X_scaled[:, selected_idx], self.selected_features
    
    def _select_mutual_info(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """互信息特征选择"""
        logger.info("Performing mutual information feature selection")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 计算互信息
        mi_scores = mutual_info_classif(X_scaled, y, random_state=self.random_state)
        
        # 选择得分最高的n_features个特征
        selected_idx = np.argsort(mi_scores)[::-1][:self.n_features]
        
        self.selected_indices = selected_idx
        self.selected_features = [feature_names[i] for i in selected_idx]
        
        logger.info(f"Selected {len(selected_idx)} features")
        
        # 绘制互信息得分
        self._plot_mi_scores(mi_scores, feature_names, selected_idx)
        
        return X_scaled[:, selected_idx], self.selected_features
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据
        
        Args:
            X: 特征矩阵
            
        Returns:
            选择后的特征矩阵
        """
        if self.selected_indices is None:
            raise ValueError("FeatureSelector has not been fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_indices]
    
    def _plot_coefficients(
        self, 
        coef: np.ndarray, 
        feature_names: List[str], 
        selected_idx: np.ndarray
    ) -> None:
        """绘制特征系数图"""
        selected_coef = coef[selected_idx]
        selected_names = [feature_names[i] for i in selected_idx]
        
        # 排序
        sorted_idx = np.argsort(np.abs(selected_coef))[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), selected_coef[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [selected_names[i] for i in sorted_idx])
        plt.xlabel('Coefficient')
        plt.title('Selected Features (LASSO)')
        plt.tight_layout()
        
        output_dir = "results/figures"
        ensure_dir(output_dir)
        plt.savefig(f"{output_dir}/feature_coefficients.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature coefficients plot saved to {output_dir}/feature_coefficients.png")
    
    def _plot_mi_scores(
        self, 
        mi_scores: np.ndarray, 
        feature_names: List[str], 
        selected_idx: np.ndarray
    ) -> None:
        """绘制互信息得分图"""
        selected_scores = mi_scores[selected_idx]
        selected_names = [feature_names[i] for i in selected_idx]
        
        # 排序
        sorted_idx = np.argsort(selected_scores)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), selected_scores[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [selected_names[i] for i in sorted_idx])
        plt.xlabel('Mutual Information Score')
        plt.title('Selected Features (Mutual Information)')
        plt.tight_layout()
        
        output_dir = "results/figures"
        ensure_dir(output_dir)
        plt.savefig(f"{output_dir}/mi_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"MI scores plot saved to {output_dir}/mi_scores.png")
    
    def get_selected_features(self) -> List[str]:
        """
        获取选中的特征名称
        
        Returns:
            特征名称列表
        """
        if self.selected_features is None:
            raise ValueError("FeatureSelector has not been fitted yet")
        
        return self.selected_features
    
    def save_selected_features(self, output_path: str) -> None:
        """
        保存选中的特征
        
        Args:
            output_path: 输出文件路径
        """
        if self.selected_features is None:
            raise ValueError("FeatureSelector has not been fitted yet")
        
        ensure_dir(os.path.dirname(output_path))
        
        with open(output_path, 'w') as f:
            for feature in self.selected_features:
                f.write(f"{feature}\n")
        
        logger.info(f"Selected features saved to {output_path}")

import os
