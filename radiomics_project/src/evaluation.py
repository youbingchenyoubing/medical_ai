import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    calibration_curve
)
from typing import Dict, Any
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: dict):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.metrics = config['evaluation']['metrics']
        self.figure_format = config['output']['figure_format']
        self.figure_dpi = config['output']['figure_dpi']
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            model_name: 模型名称
            
        Returns:
            评估指标字典
        """
        logger.info(f"Evaluating {model_name}")
        
        metrics = {}
        
        # AUC
        if 'auc' in self.metrics:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        # Accuracy
        if 'accuracy' in self.metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Sensitivity & Specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        if 'sensitivity' in self.metrics:
            metrics['sensitivity'] = tp / (tp + fn)
        
        if 'specificity' in self.metrics:
            metrics['specificity'] = tn / (tn + fp)
        
        # 打印结果
        logger.info(f"\n{model_name} Performance:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.3f}")
        
        return metrics
    
    def plot_roc_curves(
        self, 
        results: Dict[str, Dict], 
        y_test: np.ndarray,
        output_path: str = "results/figures/roc_curves.png"
    ) -> None:
        """
        绘制ROC曲线
        
        Args:
            results: 结果字典
            y_test: 测试集标签
            output_path: 输出路径
        """
        logger.info("Plotting ROC curves")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_test_prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {output_path}")
    
    def plot_confusion_matrices(
        self, 
        results: Dict[str, Dict], 
        y_test: np.ndarray,
        output_path: str = "results/figures/confusion_matrices.png"
    ) -> None:
        """
        绘制混淆矩阵
        
        Args:
            results: 结果字典
            y_test: 测试集标签
            output_path: 输出路径
        """
        logger.info("Plotting confusion matrices")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(y_test, result['y_test_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name}', fontsize=12)
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
        
        plt.tight_layout()
        
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {output_path}")
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        output_path: str = "results/figures/calibration_curve.png"
    ) -> None:
        """
        绘制校准曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            model_name: 模型名称
            output_path: 输出路径
        """
        logger.info("Plotting calibration curve")
        
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curve', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curve saved to {output_path}")
    
    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list = None,
        output_path: str = "results/classification_report.txt"
    ) -> str:
        """
        生成分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            target_names: 类别名称
            output_path: 输出路径
            
        Returns:
            分类报告字符串
        """
        logger.info("Generating classification report")
        
        if target_names is None:
            target_names = ['Class 0', 'Class 1']
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Classification report saved to {output_path}")
        
        return report

import os
from sklearn.metrics import roc_auc_score, accuracy_score
