import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


class ICCAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.icc_config = config['icc_analysis']
        self.icc_threshold = self.icc_config['icc_threshold']
        self.icc_type = self.icc_config['icc_type']
        self.n_samples = self.icc_config['n_samples']
        logger.info(
            f"ICCAnalyzer initialized (threshold={self.icc_threshold}, "
            f"type={self.icc_type})"
        )

    def calculate_icc_single_feature(
        self,
        scores_rater1: np.ndarray,
        scores_rater2: np.ndarray
    ) -> float:
        n = len(scores_rater1)
        k = 2

        all_scores = np.concatenate([scores_rater1, scores_rater2])
        all_raters = np.array([1] * n + [2] * n)
        all_subjects = np.array(list(range(n)) + list(range(n)))

        grand_mean = np.mean(all_scores)

        ss_between = 0
        for i in range(n):
            subject_mean = (scores_rater1[i] + scores_rater2[i]) / 2
            ss_between += (subject_mean - grand_mean) ** 2
        ss_between *= k

        ss_within = 0
        for i in range(n):
            ss_within += (scores_rater1[i] - (scores_rater1[i] + scores_rater2[i]) / 2) ** 2
            ss_within += (scores_rater2[i] - (scores_rater1[i] + scores_rater2[i]) / 2) ** 2

        ss_between_raters = 0
        mean_r1 = np.mean(scores_rater1)
        mean_r2 = np.mean(scores_rater2)
        ss_between_raters = n * (mean_r1 - grand_mean) ** 2 + n * (mean_r2 - grand_mean) ** 2

        ss_error = ss_within - ss_between_raters

        ms_between = ss_between / (n - 1) if n > 1 else 0
        ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 and k > 1 else 0

        if ms_error == 0:
            return 1.0

        icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)
        return max(0.0, icc)

    def calculate_icc_all_features(
        self,
        features_rater1: pd.DataFrame,
        features_rater2: pd.DataFrame,
        feature_names: List[str]
    ) -> Tuple[Dict[str, float], List[str]]:
        logger.info(
            f"Calculating ICC for {len(feature_names)} features "
            f"across {len(features_rater1)} samples"
        )

        icc_results = {}
        high_reproducibility_features = []

        for feat in feature_names:
            if feat not in features_rater1.columns or feat not in features_rater2.columns:
                logger.warning(f"Feature {feat} not found in both raters")
                continue

            scores_r1 = features_rater1[feat].values.astype(float)
            scores_r2 = features_rater2[feat].values.astype(float)

            mask = ~(np.isnan(scores_r1) | np.isnan(scores_r2))
            if mask.sum() < 3:
                logger.warning(f"Too few valid samples for {feat}: {mask.sum()}")
                continue

            icc_val = self.calculate_icc_single_feature(
                scores_r1[mask], scores_r2[mask]
            )
            icc_results[feat] = icc_val

            if icc_val >= self.icc_threshold:
                high_reproducibility_features.append(feat)

        logger.info(
            f"ICC results: {len(high_reproducibility_features)}/{len(feature_names)} "
            f"features passed threshold ({self.icc_threshold})"
        )

        return icc_results, high_reproducibility_features

    def filter_by_icc(
        self,
        features_df: pd.DataFrame,
        icc_results: Dict[str, float],
        feature_names: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        selected_features = [
            feat for feat in feature_names
            if feat in icc_results and icc_results[feat] >= self.icc_threshold
        ]

        logger.info(
            f"ICC filter: {len(feature_names)} -> {len(selected_features)} features"
        )

        keep_cols = [c for c in features_df.columns
                     if c not in feature_names or c in selected_features]
        filtered_df = features_df[keep_cols].copy()

        return filtered_df, selected_features

    def plot_icc_distribution(
        self,
        icc_results: Dict[str, float],
        output_path: str
    ) -> None:
        import matplotlib.pyplot as plt

        values = list(icc_results.values())
        feature_names = list(icc_results.keys())

        sorted_pairs = sorted(zip(values, feature_names), reverse=True)
        sorted_values = [p[0] for p in sorted_pairs]
        sorted_names = [p[1] for p in sorted_pairs]

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#2ecc71' if v >= self.icc_threshold else '#e74c3c'
                  for v in sorted_values]
        ax.bar(range(len(sorted_values)), sorted_values, color=colors, alpha=0.7)
        ax.axhline(y=self.icc_threshold, color='red', linestyle='--',
                   label=f'ICC threshold = {self.icc_threshold}')
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('ICC Value')
        ax.set_title('ICC Values for All Features')
        ax.legend()
        ax.grid(True, alpha=0.3)

        n_pass = sum(1 for v in sorted_values if v >= self.icc_threshold)
        n_fail = len(sorted_values) - n_pass
        ax.text(0.02, 0.98, f'Pass: {n_pass}\nFail: {n_fail}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ICC distribution plot saved to {output_path}")

    def save_icc_results(
        self,
        icc_results: Dict[str, float],
        output_path: str
    ) -> None:
        ensure_dir(os.path.dirname(output_path))
        df = pd.DataFrame([
            {'feature': k, 'icc': v, 'pass': v >= self.icc_threshold}
            for k, v in sorted(icc_results.items(), key=lambda x: x[1], reverse=True)
        ])
        df.to_csv(output_path, index=False)
        logger.info(f"ICC results saved to {output_path}")

    def load_rater_features_and_compute(
        self,
        rater1_csv: str,
        rater2_csv: str,
        output_dir: str
    ) -> Tuple[Dict[str, float], List[str]]:
        logger.info(f"Loading rater1 features from {rater1_csv}")
        df_r1 = pd.read_csv(rater1_csv)

        logger.info(f"Loading rater2 features from {rater2_csv}")
        df_r2 = pd.read_csv(rater2_csv)

        feature_cols = [
            c for c in df_r1.columns
            if c in df_r2.columns
            and c not in ['patient_id', 'timepoint']
            and df_r1[c].dtype in [np.float64, np.int64, float, int]
        ]

        icc_results, high_rep_features = self.calculate_icc_all_features(
            df_r1, df_r2, feature_cols
        )

        self.save_icc_results(
            icc_results,
            os.path.join(output_dir, "icc_results.csv")
        )
        self.plot_icc_distribution(
            icc_results,
            os.path.join(output_dir, "icc_distribution.png")
        )

        return icc_results, high_rep_features
