#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import MRIPreprocessor
from src.feature_extraction import PyRadiomicsExtractor
from src.delta_features import DeltaFeatureCalculator
from src.icc_analysis import ICCAnalyzer
from src.feature_selection import CascadeFeatureSelector
from src.model_training import MultiModelTrainer
from src.evaluation import ComprehensiveEvaluator
from src.utils import load_config, setup_logger, ensure_dir

import pandas as pd
import numpy as np

logger = setup_logger(__name__, log_file="logs/radiomics_pipeline.log")


class HCCpCRPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.setup_directories()

        logger.info("=" * 70)
        logger.info("HCC pCR Prediction Pipeline Initialized")
        logger.info(f"Project: {self.config['project']['name']}")
        logger.info("=" * 70)

    def setup_directories(self):
        dirs = [
            self.config['data']['raw_dir'],
            self.config['data']['processed_dir'],
            self.config['data']['mask_dir'],
            self.config['output']['save_dir'],
            "results/features",
            "results/models",
            "results/figures",
            "results/figures/shap",
            "logs"
        ]
        for dir_path in dirs:
            ensure_dir(dir_path)

    def run_step1_preprocessing(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 1: MRI Data Preprocessing")
        logger.info("=" * 70)

        preprocessor = MRIPreprocessor(self.config)

        raw_dir = self.config['data']['raw_dir']
        processed_dir = self.config['data']['processed_dir']

        pre_raw = os.path.join(raw_dir, "pre")
        post_raw = os.path.join(raw_dir, "post")
        pre_processed = os.path.join(processed_dir, "pre")
        post_processed = os.path.join(processed_dir, "post")

        if os.path.exists(pre_raw):
            preprocessor.batch_preprocess(pre_raw, pre_processed, timepoint="pre")
        else:
            logger.warning(f"Pre-treatment data not found: {pre_raw}")

        if os.path.exists(post_raw):
            preprocessor.batch_preprocess(post_raw, post_processed, timepoint="post")
        else:
            logger.warning(f"Post-treatment data not found: {post_raw}")

        logger.info("Step 1 completed")

    def run_step2_feature_extraction(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 2: PyRadiomics Feature Extraction")
        logger.info("=" * 70)

        extractor = PyRadiomicsExtractor(self.config)

        feature_counts = extractor.count_expected_features()
        logger.info(f"Expected features per sequence: {feature_counts['per_sequence']}")
        logger.info(f"Expected total features: {feature_counts['total']}")

        processed_dir = self.config['data']['processed_dir']
        output_dir = "results/features"

        results = extractor.extract_both_timepoints(processed_dir, output_dir)

        for tp, df in results.items():
            logger.info(f"  {tp}: {len(df)} patients, {len(df.columns) - 2} features")

        logger.info("Step 2 completed")
        return results

    def run_step3_delta_features(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 3: Delta Feature Computation")
        logger.info("=" * 70)

        calculator = DeltaFeatureCalculator(self.config)

        pre_csv = "results/features/radiomics_features_pre.csv"
        post_csv = "results/features/radiomics_features_post.csv"

        if not os.path.exists(pre_csv) or not os.path.exists(post_csv):
            logger.error("Pre/post feature files not found. Run step 2 first.")
            return None

        df_delta = calculator.load_and_compute(pre_csv, post_csv, "results/features")

        logger.info(f"Delta features: {len(df_delta)} patients, {len(df_delta.columns) - 1} columns")
        logger.info("Step 3 completed")
        return df_delta

    def run_step4_icc_analysis(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 4: ICC Reproducibility Analysis")
        logger.info("=" * 70)

        analyzer = ICCAnalyzer(self.config)

        rater1_csv = "results/features/radiomics_features_rater1.csv"
        rater2_csv = "results/features/radiomics_features_rater2.csv"

        if not os.path.exists(rater1_csv) or not os.path.exists(rater2_csv):
            logger.warning("Rater feature files not found. ICC analysis requires "
                           "two independent segmentations.")
            logger.info("Skipping ICC analysis (will be skipped in feature selection)")
            return None

        icc_results, high_rep_features = analyzer.load_rater_features_and_compute(
            rater1_csv, rater2_csv, "results/features"
        )

        logger.info(f"ICC: {len(high_rep_features)} features passed threshold")
        logger.info("Step 4 completed")
        return icc_results

    def run_step5_feature_selection(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 5: Cascade Feature Selection")
        logger.info("=" * 70)

        delta_csv = "results/features/radiomics_features_delta.csv"
        if not os.path.exists(delta_csv):
            logger.error("Delta features not found. Run step 3 first.")
            return None

        df = pd.read_csv(delta_csv)

        feature_cols = [c for c in df.columns
                        if c.startswith('delta_') and c != 'patient_id']

        clinical_file = self.config['data']['clinical_file']
        if os.path.exists(clinical_file):
            clinical_df = pd.read_csv(clinical_file)
            if 'patient_id' in clinical_df.columns:
                df = df.merge(clinical_df, on='patient_id', how='left')

        label_col = 'pCR'
        if label_col not in df.columns:
            logger.error(f"Label column '{label_col}' not found in data")
            return None

        X = df[feature_cols].values
        y = df[label_col].values

        icc_results = None
        icc_csv = "results/features/icc_results.csv"
        if os.path.exists(icc_csv):
            icc_df = pd.read_csv(icc_csv)
            icc_results = dict(zip(icc_df['feature'], icc_df['icc']))

        selector = CascadeFeatureSelector(self.config)
        X_selected, selected_names = selector.fit_transform(
            X, y, feature_cols,
            icc_results=icc_results,
            output_dir="results/figures"
        )

        selected_df = pd.DataFrame(X_selected, columns=selected_names)
        selected_df['patient_id'] = df['patient_id'].values
        selected_df[label_col] = y

        if 'afp_response' in df.columns:
            selected_df['afp_response'] = df['afp_response'].values

        selected_df.to_csv("results/features/selected_features.csv", index=False)
        selector.save_selected_features("results/features/selected_feature_names.txt")
        selector.save_selection_log("results/features/selection_log.csv")

        logger.info(f"Step 5 completed: {len(selected_names)} features selected")
        return selected_df

    def run_step6_model_training(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 6: Multi-Model Training (14 models)")
        logger.info("=" * 70)

        features_csv = "results/features/selected_features.csv"
        if not os.path.exists(features_csv):
            logger.error("Selected features not found. Run step 5 first.")
            return None, None

        df = pd.read_csv(features_csv)

        feature_cols = [c for c in df.columns
                        if c not in ['patient_id', 'pCR', 'afp_response',
                                     'site', 'label', 'response']]

        label_col = 'pCR'
        if label_col not in df.columns:
            logger.error(f"Label column '{label_col}' not found")
            return None, None

        external_csv = "results/features/selected_features_external.csv"
        external_df = None
        if os.path.exists(external_csv):
            external_df = pd.read_csv(external_csv)

        trainer = MultiModelTrainer(self.config)
        trainer.prepare_data(df, feature_cols, label_col=label_col,
                             external_df=external_df)
        models, results = trainer.train_all_models()

        best_model, best_name = trainer.get_best_model()
        trainer.save_model(best_model, f"best_model_{best_name}")
        trainer.save_scaler()

        summary = trainer.get_results_summary()
        summary.to_csv("results/models/model_comparison.csv", index=False)
        logger.info(f"\nModel comparison:\n{summary.to_string()}")

        logger.info("Step 6 completed")
        return models, results

    def run_step7_evaluation(self):
        logger.info("\n" + "=" * 70)
        logger.info("Step 7: Comprehensive Model Evaluation")
        logger.info("=" * 70)

        features_csv = "results/features/selected_features.csv"
        df = pd.read_csv(features_csv)

        feature_cols = [c for c in df.columns
                        if c not in ['patient_id', 'pCR', 'afp_response',
                                     'site', 'label', 'response']]
        label_col = 'pCR'

        trainer = MultiModelTrainer(self.config)
        trainer.prepare_data(df, feature_cols, label_col=label_col)

        models, results = trainer.train_all_models()

        evaluator = ComprehensiveEvaluator(self.config)

        y_external = None
        if hasattr(trainer, 'X_external'):
            y_external = trainer.y_external

        df_metrics = evaluator.generate_full_report(
            results, trainer.y_test, y_external,
            output_dir="results/figures"
        )

        best_model, best_name = trainer.get_best_model()

        evaluator.shap_analysis(
            best_model, trainer.X_test, feature_cols,
            output_dir="results/figures/shap"
        )

        if 'afp_response' in df.columns:
            logger.info("Building combined radiomics+AFP model")
            best_prob = results[best_name]['y_test_prob']
            radiomics_score_train = best_prob

            if hasattr(trainer, 'X_external'):
                ext_prob = results[best_name].get('y_ext_prob')
                if ext_prob is not None:
                    radiomics_score_ext = ext_prob

        logger.info("Step 7 completed")
        return df_metrics

    def run_all(self):
        logger.info("\n" + "=" * 70)
        logger.info("Running Complete Pipeline")
        logger.info("=" * 70)

        self.run_step1_preprocessing()
        self.run_step2_feature_extraction()
        self.run_step3_delta_features()
        self.run_step4_icc_analysis()
        self.run_step5_feature_selection()
        self.run_step6_model_training()
        self.run_step7_evaluation()

        logger.info("\n" + "=" * 70)
        logger.info("Pipeline Completed Successfully!")
        logger.info("=" * 70)

    def run_step(self, step: int):
        step_methods = {
            1: self.run_step1_preprocessing,
            2: self.run_step2_feature_extraction,
            3: self.run_step3_delta_features,
            4: self.run_step4_icc_analysis,
            5: self.run_step5_feature_selection,
            6: self.run_step6_model_training,
            7: self.run_step7_evaluation
        }

        if step not in step_methods:
            logger.error(f"Invalid step number: {step}")
            logger.info("Valid steps: 1-7")
            return

        step_methods[step]()


def main():
    parser = argparse.ArgumentParser(
        description='HCC pCR Prediction - MRI Delta Radiomics Pipeline'
    )
    parser.add_argument('--step', type=int, default=0,
                        help='Run specific step (1-7), 0 for all steps')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    pipeline = HCCpCRPipeline(config_path=args.config)

    if args.step == 0:
        pipeline.run_all()
    else:
        pipeline.run_step(args.step)


if __name__ == "__main__":
    main()
