import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    calibration_curve, roc_auc_score, accuracy_score
)
from typing import Dict, Any, List, Optional, Tuple
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not installed, SHAP analysis will be skipped")


class ComprehensiveEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.eval_config = config['evaluation']
        self.metrics = self.eval_config['metrics']
        self.calibration_bins = self.eval_config['calibration_bins']
        self.dca_config = {
            'threshold_range': self.eval_config['dca_threshold_range'],
            'step': self.eval_config['dca_step']
        }
        self.shap_config = {
            'background_samples': self.eval_config['shap_background_samples'],
            'top_features': self.eval_config['shap_top_features']
        }
        self.figure_dpi = config['output']['figure_dpi']
        self.figure_format = config['output']['figure_format']
        logger.info("ComprehensiveEvaluator initialized")

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        logger.info(f"Evaluating {model_name}")

        metrics = {}

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        if 'auc' in self.metrics:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        if 'accuracy' in self.metrics:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
        if 'sensitivity' in self.metrics:
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        if 'specificity' in self.metrics:
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        if 'ppv' in self.metrics:
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        if 'npv' in self.metrics:
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

        logger.info(f"\n{model_name} Performance:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.3f}")

        return metrics

    def evaluate_all_models(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        y_external: Optional[np.ndarray] = None,
        output_dir: str = "results/figures"
    ) -> pd.DataFrame:
        logger.info("Evaluating all models")

        all_metrics = []
        for model_name, result in results.items():
            y_pred = result.get('y_test_pred')
            y_prob = result.get('y_test_prob')

            if y_pred is None or y_prob is None:
                continue

            metrics = self.evaluate_model(y_test, y_pred, y_prob, model_name)
            metrics['Model'] = model_name
            metrics['Dataset'] = 'Internal Test'

            if y_external is not None and 'y_ext_prob' in result:
                ext_pred = result.get('y_ext_pred')
                ext_prob = result.get('y_ext_prob')
                if ext_pred is not None and ext_prob is not None:
                    ext_metrics = self.evaluate_model(
                        y_external, ext_pred, ext_prob,
                        f"{model_name} (External)"
                    )
                    for k, v in ext_metrics.items():
                        metrics[f'external_{k}'] = v

            all_metrics.append(metrics)

        df_metrics = pd.DataFrame(all_metrics)
        output_path = os.path.join(output_dir, "model_comparison.csv")
        ensure_dir(output_dir)
        df_metrics.to_csv(output_path, index=False)
        logger.info(f"Model comparison saved to {output_path}")

        return df_metrics

    def plot_roc_curves(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        output_path: str = "results/figures/roc_curves.png",
        y_external: Optional[np.ndarray] = None,
        external_output_path: Optional[str] = None
    ) -> None:
        logger.info("Plotting ROC curves")

        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, result in results.items():
            y_prob = result.get('y_test_prob')
            if y_prob is None:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Internal Test Set', fontsize=14)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curves saved to {output_path}")

        if y_external is not None and external_output_path:
            fig, ax = plt.subplots(figsize=(10, 8))

            for model_name, result in results.items():
                y_prob = result.get('y_ext_prob')
                if y_prob is None:
                    continue

                fpr, tpr, _ = roc_curve(y_external, y_prob)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')

            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curves - External Validation Set', fontsize=14)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

            ensure_dir(os.path.dirname(external_output_path))
            plt.savefig(external_output_path, dpi=self.figure_dpi,
                        bbox_inches='tight')
            plt.close()
            logger.info(f"External ROC curves saved to {external_output_path}")

    def plot_confusion_matrices(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        output_dir: str = "results/figures"
    ) -> None:
        logger.info("Plotting confusion matrices")

        ensure_dir(output_dir)

        for model_name, result in results.items():
            y_pred = result.get('y_test_pred')
            if y_pred is None:
                continue

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['non-pCR', 'pCR'],
                yticklabels=['non-pCR', 'pCR']
            )
            ax.set_title(f'{model_name} - Confusion Matrix', fontsize=12)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"confusion_{model_name}.png"),
                dpi=self.figure_dpi, bbox_inches='tight'
            )
            plt.close()

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        output_path: str = "results/figures/calibration_curve.png"
    ) -> None:
        logger.info(f"Plotting calibration curve for {model_name}")

        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=self.calibration_bins
        )

        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(prob_pred, prob_true, 'o-', label=model_name, linewidth=2)
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curve', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Calibration curve saved to {output_path}")

    def plot_calibration_curves_comparison(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        output_path: str = "results/figures/calibration_comparison.png"
    ) -> None:
        logger.info("Plotting calibration curves comparison")

        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        for model_name, result in results.items():
            y_prob = result.get('y_test_prob')
            if y_prob is None:
                continue

            prob_true, prob_pred = calibration_curve(
                y_test, y_prob, n_bins=self.calibration_bins
            )
            plt.plot(prob_pred, prob_true, 'o-', label=model_name, linewidth=2)

        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title('Calibration Curves Comparison', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()

    def decision_curve_analysis(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        output_path: str = "results/figures/dca_curve.png"
    ) -> None:
        logger.info(f"Decision curve analysis for {model_name}")

        threshold_low, threshold_high = self.dca_config['threshold_range']
        step = self.dca_config['step']
        thresholds = np.arange(threshold_low, threshold_high, step)

        net_benefit_model = []
        net_benefit_all = []

        prevalence = y_true.mean()
        n = len(y_true)

        for threshold in thresholds:
            tp = np.sum((y_prob >= threshold) & (y_true == 1))
            fp = np.sum((y_prob >= threshold) & (y_true == 0))

            net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
            net_benefit_model.append(net_benefit)

            nb_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            net_benefit_all.append(nb_all)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, net_benefit_model, 'b-', linewidth=2,
                 label=model_name)
        plt.plot(thresholds, net_benefit_all, 'r--', linewidth=1.5,
                 label='Treat All')
        plt.axhline(y=0, color='k', linestyle=':', linewidth=1,
                    label='Treat None')
        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title('Decision Curve Analysis', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])

        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"DCA curve saved to {output_path}")

    def plot_dca_comparison(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        output_path: str = "results/figures/dca_comparison.png"
    ) -> None:
        logger.info("Plotting DCA comparison")

        threshold_low, threshold_high = self.dca_config['threshold_range']
        step = self.dca_config['step']
        thresholds = np.arange(threshold_low, threshold_high, step)

        plt.figure(figsize=(12, 8))

        prevalence = y_test.mean()
        n = len(y_test)

        net_benefit_all = []
        for threshold in thresholds:
            nb_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            net_benefit_all.append(nb_all)

        plt.plot(thresholds, net_benefit_all, 'r--', linewidth=1.5,
                 label='Treat All')
        plt.axhline(y=0, color='k', linestyle=':', linewidth=1,
                    label='Treat None')

        for model_name, result in results.items():
            y_prob = result.get('y_test_prob')
            if y_prob is None:
                continue

            net_benefit = []
            for threshold in thresholds:
                tp = np.sum((y_prob >= threshold) & (y_test == 1))
                fp = np.sum((y_prob >= threshold) & (y_test == 0))
                nb = tp / n - fp / n * (threshold / (1 - threshold))
                net_benefit.append(nb)

            plt.plot(thresholds, net_benefit, linewidth=2, label=model_name)

        plt.xlabel('Threshold Probability', fontsize=12)
        plt.ylabel('Net Benefit', fontsize=12)
        plt.title('Decision Curve Analysis - Model Comparison', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])

        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=self.figure_dpi, bbox_inches='tight')
        plt.close()

    def shap_analysis(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        output_dir: str = "results/figures/shap"
    ) -> None:
        if not HAS_SHAP:
            logger.warning("SHAP not installed, skipping SHAP analysis")
            return

        logger.info("Running SHAP analysis")

        ensure_dir(output_dir)

        n_background = min(self.shap_config['background_samples'], X.shape[0])
        background = X[:n_background]

        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]
        else:
            shap_values_positive = shap_values

        plt.figure()
        shap.summary_plot(
            shap_values_positive, X,
            feature_names=feature_names,
            show=False, max_display=self.shap_config['top_features']
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "shap_summary.png"),
            dpi=self.figure_dpi, bbox_inches='tight'
        )
        plt.close()
        logger.info("SHAP summary plot saved")

        plt.figure()
        shap.summary_plot(
            shap_values_positive, X,
            feature_names=feature_names,
            plot_type="bar",
            show=False, max_display=self.shap_config['top_features']
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "shap_importance.png"),
            dpi=self.figure_dpi, bbox_inches='tight'
        )
        plt.close()
        logger.info("SHAP importance plot saved")

        top_n = min(5, len(feature_names))
        mean_abs_shap = np.mean(np.abs(shap_values_positive), axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

        for idx in top_indices:
            feat_name = feature_names[idx]
            plt.figure()
            shap.dependence_plot(
                idx, shap_values_positive, X,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            safe_name = feat_name.replace('/', '_').replace('\\', '_')
            plt.savefig(
                os.path.join(output_dir, f"shap_dependence_{safe_name}.png"),
                dpi=self.figure_dpi, bbox_inches='tight'
            )
            plt.close()

        logger.info("SHAP dependence plots saved")

    def generate_full_report(
        self,
        results: Dict[str, Dict],
        y_test: np.ndarray,
        y_external: Optional[np.ndarray] = None,
        output_dir: str = "results/figures"
    ) -> pd.DataFrame:
        logger.info("Generating full evaluation report")

        ensure_dir(output_dir)

        df_metrics = self.evaluate_all_models(
            results, y_test, y_external, output_dir
        )

        self.plot_roc_curves(
            results, y_test,
            os.path.join(output_dir, "roc_curves_test.png"),
            y_external,
            os.path.join(output_dir, "roc_curves_external.png")
        )

        self.plot_confusion_matrices(results, y_test, output_dir)

        self.plot_calibration_curves_comparison(
            results, y_test,
            os.path.join(output_dir, "calibration_comparison.png")
        )

        self.plot_dca_comparison(
            results, y_test,
            os.path.join(output_dir, "dca_comparison.png")
        )

        for model_name, result in results.items():
            y_prob = result.get('y_test_prob')
            if y_prob is None:
                continue

            self.plot_calibration_curve(
                y_test, y_prob, model_name,
                os.path.join(output_dir, f"calibration_{model_name}.png")
            )

            self.decision_curve_analysis(
                y_test, y_prob, model_name,
                os.path.join(output_dir, f"dca_{model_name}.png")
            )

        logger.info("Full evaluation report generated")
        return df_metrics
