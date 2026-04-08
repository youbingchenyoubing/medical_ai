import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


class CascadeFeatureSelector:
    def __init__(self, config: dict):
        self.config = config
        self.fs_config = config['feature_selection']
        self.pipeline = self.fs_config['pipeline']
        self.ttest_alpha = self.fs_config['ttest_alpha']
        self.spearman_threshold = self.fs_config['spearman_threshold']
        self.lasso_cv_folds = self.fs_config['lasso_cv_folds']
        self.rf_n_top = self.fs_config['rf_n_top_features']
        self.random_state = self.fs_config['random_state']
        self.scaler = StandardScaler()
        self.selected_features = None
        self.selection_log = {}
        logger.info(
            f"CascadeFeatureSelector initialized: "
            f"pipeline={' -> '.join(self.pipeline)}"
        )

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        icc_results: Optional[Dict[str, float]] = None,
        output_dir: str = "results/figures"
    ) -> Tuple[np.ndarray, List[str]]:
        logger.info(f"Starting cascade feature selection: {X.shape[1]} features")

        current_X = X.copy()
        current_features = feature_names.copy()
        self.selection_log = {}

        for step in self.pipeline:
            logger.info(f"\n{'='*50}")
            logger.info(f"Step: {step} | Features: {len(current_features)}")
            logger.info(f"{'='*50}")

            if step == 'icc':
                if icc_results is not None:
                    current_X, current_features = self._select_by_icc(
                        current_X, y, current_features, icc_results
                    )
                else:
                    logger.warning("ICC results not provided, skipping ICC step")

            elif step == 'ttest':
                current_X, current_features = self._select_by_ttest(
                    current_X, y, current_features, output_dir
                )

            elif step == 'spearman':
                current_X, current_features = self._select_by_spearman(
                    current_X, y, current_features, output_dir
                )

            elif step == 'lasso':
                current_X, current_features = self._select_by_lasso(
                    current_X, y, current_features, output_dir
                )

            elif step == 'random_forest':
                current_X, current_features = self._select_by_random_forest(
                    current_X, y, current_features, output_dir
                )

            else:
                logger.warning(f"Unknown selection step: {step}")

            self.selection_log[step] = {
                'n_features': len(current_features),
                'features': current_features.copy()
            }

        self.selected_features = current_features
        logger.info(f"\nFinal selected features: {len(current_features)}")

        return current_X, current_features

    def _select_by_icc(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        icc_results: Dict[str, float]
    ) -> Tuple[np.ndarray, List[str]]:
        icc_threshold = self.config['icc_analysis']['icc_threshold']

        selected_idx = []
        for i, feat in enumerate(feature_names):
            if feat in icc_results and icc_results[feat] >= icc_threshold:
                selected_idx.append(i)

        if not selected_idx:
            logger.warning("No features passed ICC filter, keeping all")
            return X, feature_names

        logger.info(f"ICC: {len(feature_names)} -> {len(selected_idx)} features")
        return X[:, selected_idx], [feature_names[i] for i in selected_idx]

    def _select_by_ttest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        output_dir: str
    ) -> Tuple[np.ndarray, List[str]]:
        pcr_mask = y == 1
        non_pcr_mask = y == 0

        selected_idx = []
        p_values = []

        for i, feat in enumerate(feature_names):
            pcr_values = X[pcr_mask, i]
            non_pcr_values = X[non_pcr_mask, i]

            if np.std(pcr_values) == 0 and np.std(non_pcr_values) == 0:
                p_values.append(1.0)
                continue

            stat, p_val = stats.ttest_ind(pcr_values, non_pcr_values)
            p_values.append(p_val)

            if p_val < self.ttest_alpha:
                selected_idx.append(i)

        self._plot_ttest_results(
            p_values, feature_names, output_dir
        )

        if not selected_idx:
            logger.warning("No features passed t-test, keeping all")
            return X, feature_names

        logger.info(f"t-test: {len(feature_names)} -> {len(selected_idx)} features")
        return X[:, selected_idx], [feature_names[i] for i in selected_idx]

    def _select_by_spearman(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        output_dir: str
    ) -> Tuple[np.ndarray, List[str]]:
        if X.shape[1] < 2:
            return X, feature_names

        corr_matrix, _ = spearmanr(X)

        if corr_matrix.ndim == 0:
            return X, feature_names

        feature_label_corr = []
        for i in range(X.shape[1]):
            corr, _ = spearmanr(X[:, i], y)
            feature_label_corr.append(abs(corr))

        remove_set = set()
        for i in range(len(feature_names)):
            if i in remove_set:
                continue
            for j in range(i + 1, len(feature_names)):
                if j in remove_set:
                    continue
                if abs(corr_matrix[i, j]) >= self.spearman_threshold:
                    if feature_label_corr[i] < feature_label_corr[j]:
                        remove_set.add(i)
                    else:
                        remove_set.add(j)

        selected_idx = [i for i in range(len(feature_names))
                        if i not in remove_set]

        self._plot_spearman_heatmap(
            corr_matrix, feature_names, selected_idx, output_dir
        )

        logger.info(
            f"Spearman: {len(feature_names)} -> {len(selected_idx)} features "
            f"(removed {len(remove_set)} correlated pairs)"
        )
        return X[:, selected_idx], [feature_names[i] for i in selected_idx]

    def _select_by_lasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        output_dir: str
    ) -> Tuple[np.ndarray, List[str]]:
        X_scaled = self.scaler.fit_transform(X)

        lasso = LassoCV(
            cv=self.lasso_cv_folds,
            random_state=self.random_state,
            max_iter=10000,
            n_jobs=-1
        )
        lasso.fit(X_scaled, y)

        logger.info(f"LASSO optimal alpha: {lasso.alpha_:.6f}")

        coef = lasso.coef_
        non_zero_idx = np.where(coef != 0)[0]

        if len(non_zero_idx) == 0:
            logger.warning("LASSO selected 0 features, using top 5 by absolute coef")
            non_zero_idx = np.argsort(np.abs(coef))[::-1][:5]

        self._plot_lasso_coefficients(
            coef, feature_names, non_zero_idx, output_dir
        )

        logger.info(f"LASSO: {len(feature_names)} -> {len(non_zero_idx)} features")
        return X[:, non_zero_idx], [feature_names[i] for i in non_zero_idx]

    def _select_by_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        output_dir: str
    ) -> Tuple[np.ndarray, List[str]]:
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X, y)

        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        n_select = min(self.rf_n_top, len(feature_names))
        selected_idx = sorted_idx[:n_select]

        self._plot_rf_importances(
            importances, feature_names, selected_idx, output_dir
        )

        logger.info(
            f"Random Forest: {len(feature_names)} -> {len(selected_idx)} features"
        )
        return X[:, selected_idx], [feature_names[i] for i in selected_idx]

    def _plot_ttest_results(
        self,
        p_values: List[float],
        feature_names: List[str],
        output_dir: str
    ) -> None:
        ensure_dir(output_dir)

        sorted_pairs = sorted(zip(p_values, feature_names))
        sorted_p = [p for p, _ in sorted_pairs]
        sorted_names = [n for _, n in sorted_pairs]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2ecc71' if p < self.ttest_alpha else '#e74c3c'
                  for p in sorted_p]
        ax.bar(range(len(sorted_p)), [-np.log10(p) if p > 0 else 300
                                       for p in sorted_p], color=colors, alpha=0.7)
        ax.axhline(y=-np.log10(self.ttest_alpha), color='red', linestyle='--',
                   label=f'p = {self.ttest_alpha}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title('Independent Sample t-test Feature Selection')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(output_dir, "ttest_selection.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def _plot_spearman_heatmap(
        self,
        corr_matrix: np.ndarray,
        feature_names: List[str],
        selected_idx: List[int],
        output_dir: str
    ) -> None:
        ensure_dir(output_dir)

        if len(selected_idx) > 50:
            plot_idx = selected_idx[:50]
        else:
            plot_idx = selected_idx

        sub_corr = corr_matrix[np.ix_(plot_idx, plot_idx)]
        sub_names = [feature_names[i][:20] for i in plot_idx]

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            sub_corr, annot=False, cmap='RdBu_r', center=0,
            xticklabels=sub_names, yticklabels=sub_names,
            ax=ax, vmin=-1, vmax=1
        )
        ax.set_title('Spearman Correlation Heatmap (Selected Features)')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "spearman_heatmap.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def _plot_lasso_coefficients(
        self,
        coef: np.ndarray,
        feature_names: List[str],
        selected_idx: np.ndarray,
        output_dir: str
    ) -> None:
        ensure_dir(output_dir)

        selected_coef = coef[selected_idx]
        selected_names = [feature_names[i][:30] for i in selected_idx]

        sorted_idx = np.argsort(np.abs(selected_coef))[::-1]

        fig, ax = plt.subplots(figsize=(10, max(6, len(selected_idx) * 0.3)))
        ax.barh(
            range(len(sorted_idx)),
            selected_coef[sorted_idx],
            color=['#3498db' if c > 0 else '#e74c3c'
                   for c in selected_coef[sorted_idx]]
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([selected_names[i] for i in sorted_idx])
        ax.set_xlabel('Coefficient')
        ax.set_title('LASSO Selected Features')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "lasso_coefficients.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def _plot_rf_importances(
        self,
        importances: np.ndarray,
        feature_names: List[str],
        selected_idx: np.ndarray,
        output_dir: str
    ) -> None:
        ensure_dir(output_dir)

        selected_imp = importances[selected_idx]
        selected_names = [feature_names[i][:30] for i in selected_idx]

        sorted_idx = np.argsort(selected_imp)[::-1]

        fig, ax = plt.subplots(figsize=(10, max(6, len(selected_idx) * 0.3)))
        ax.barh(
            range(len(sorted_idx)),
            selected_imp[sorted_idx],
            color='#2ecc71', alpha=0.8
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([selected_names[i] for i in sorted_idx])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Random Forest Feature Importance')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "rf_importances.png"),
            dpi=300, bbox_inches='tight'
        )
        plt.close()

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features is None:
            raise ValueError("FeatureSelector has not been fitted yet")
        return X

    def get_selected_features(self) -> List[str]:
        if self.selected_features is None:
            raise ValueError("FeatureSelector has not been fitted yet")
        return self.selected_features

    def get_selection_log(self) -> Dict:
        return self.selection_log

    def save_selected_features(self, output_path: str) -> None:
        if self.selected_features is None:
            raise ValueError("FeatureSelector has not been fitted yet")

        ensure_dir(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            for feat in self.selected_features:
                f.write(f"{feat}\n")

        logger.info(f"Selected features saved to {output_path}")

    def save_selection_log(self, output_path: str) -> None:
        ensure_dir(os.path.dirname(output_path))

        log_df = pd.DataFrame([
            {
                'step': step,
                'n_features': info['n_features'],
                'features': '|'.join(info['features'][:20])
            }
            for step, info in self.selection_log.items()
        ])
        log_df.to_csv(output_path, index=False)
        logger.info(f"Selection log saved to {output_path}")
