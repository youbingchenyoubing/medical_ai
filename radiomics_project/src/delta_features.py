import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)


class DeltaFeatureCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.delta_config = config['delta_features']
        self.epsilon = self.delta_config['epsilon']
        logger.info("DeltaFeatureCalculator initialized")

    def compute_delta(
        self,
        pre_value: float,
        post_value: float
    ) -> float:
        if abs(pre_value) < self.epsilon:
            return 0.0
        return (pre_value - post_value) / pre_value

    def compute_delta_features(
        self,
        features_pre: Dict[str, float],
        features_post: Dict[str, float],
        feature_names: List[str]
    ) -> Dict[str, float]:
        delta_features = {}

        for feat in feature_names:
            pre_val = features_pre.get(feat, 0.0)
            post_val = features_post.get(feat, 0.0)
            delta_features[f'delta_{feat}'] = self.compute_delta(pre_val, post_val)

        return delta_features

    def compute_delta_from_dataframes(
        self,
        df_pre: pd.DataFrame,
        df_post: pd.DataFrame,
        patient_col: str = 'patient_id'
    ) -> pd.DataFrame:
        logger.info("Computing delta features from pre/post DataFrames")

        feature_cols = [
            c for c in df_pre.columns
            if c not in [patient_col, 'timepoint']
            and c in df_post.columns
        ]

        logger.info(f"Computing delta for {len(feature_cols)} features")

        df_pre_indexed = df_pre.set_index(patient_col)
        df_post_indexed = df_post.set_index(patient_col)

        common_patients = df_pre_indexed.index.intersection(df_post_indexed.index)
        logger.info(f"Common patients: {len(common_patients)}")

        delta_records = []
        for patient_id in common_patients:
            pre_row = df_pre_indexed.loc[patient_id]
            post_row = df_post_indexed.loc[patient_id]

            if isinstance(pre_row, pd.DataFrame):
                pre_row = pre_row.iloc[0]
            if isinstance(post_row, pd.DataFrame):
                post_row = post_row.iloc[0]

            delta_record = {patient_col: patient_id}

            for feat in feature_cols:
                pre_val = float(pre_row[feat])
                post_val = float(post_row[feat])
                delta_record[f'delta_{feat}'] = self.compute_delta(pre_val, post_val)

            for feat in feature_cols:
                delta_record[f'baseline_{feat}'] = float(pre_row[feat])
                delta_record[f'preop_{feat}'] = float(post_row[feat])

            delta_records.append(delta_record)

        df_delta = pd.DataFrame(delta_records)
        logger.info(
            f"Delta features computed: {len(df_delta)} patients, "
            f"{len(df_delta.columns) - 1} columns"
        )
        return df_delta

    def build_three_model_datasets(
        self,
        df_pre: pd.DataFrame,
        df_post: pd.DataFrame,
        df_delta: pd.DataFrame,
        clinical_df: Optional[pd.DataFrame] = None,
        patient_col: str = 'patient_id'
    ) -> Dict[str, pd.DataFrame]:
        logger.info("Building three model datasets: baseline, preoperative, delta")

        feature_cols_pre = [
            c for c in df_pre.columns
            if c not in [patient_col, 'timepoint']
        ]
        feature_cols_post = [
            c for c in df_post.columns
            if c not in [patient_col, 'timepoint']
        ]

        df_baseline = df_pre[[patient_col] + feature_cols_pre].copy()
        df_baseline.columns = [patient_col] + [
            f'baseline_{c}' for c in feature_cols_pre
        ]

        df_preop = df_post[[patient_col] + feature_cols_post].copy()
        df_preop.columns = [patient_col] + [
            f'preop_{c}' for c in feature_cols_post
        ]

        datasets = {
            'baseline': df_baseline,
            'preoperative': df_preop,
            'delta': df_delta
        }

        if clinical_df is not None:
            for name, df in datasets.items():
                datasets[name] = df.merge(
                    clinical_df, on=patient_col, how='left'
                )

        for name, df in datasets.items():
            logger.info(f"  {name}: {len(df)} patients, {len(df.columns)} columns")

        return datasets

    def compute_afp_response(
        self,
        afp_pre: pd.Series,
        afp_post: pd.Series,
        epsilon: float = 1e-7
    ) -> pd.Series:
        afp_response = pd.Series(index=afp_pre.index, dtype=float)
        for idx in afp_pre.index:
            pre_val = afp_pre.loc[idx]
            post_val = afp_post.loc[idx]
            if abs(pre_val) < epsilon:
                afp_response.loc[idx] = 0.0
            else:
                afp_response.loc[idx] = (pre_val - post_val) / pre_val
        return afp_response

    def save_delta_features(
        self,
        df_delta: pd.DataFrame,
        output_path: str
    ) -> None:
        ensure_dir(os.path.dirname(output_path))
        df_delta.to_csv(output_path, index=False)
        logger.info(f"Delta features saved to {output_path}")

    def load_and_compute(
        self,
        pre_csv: str,
        post_csv: str,
        output_dir: str
    ) -> pd.DataFrame:
        logger.info(f"Loading pre features from {pre_csv}")
        df_pre = pd.read_csv(pre_csv)

        logger.info(f"Loading post features from {post_csv}")
        df_post = pd.read_csv(post_csv)

        df_delta = self.compute_delta_from_dataframes(df_pre, df_post)

        output_path = os.path.join(output_dir, "radiomics_features_delta.csv")
        self.save_delta_features(df_delta, output_path)

        return df_delta
