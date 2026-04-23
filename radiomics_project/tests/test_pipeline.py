#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def _safe_import(module_path, attr_name):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name)
    except ImportError as e:
        return None


load_config = _safe_import("src.utils", "load_config")
ensure_dir = _safe_import("src.utils", "ensure_dir")
CascadeFeatureSelector = _safe_import("src.feature_selection", "CascadeFeatureSelector")
MultiModelTrainer = _safe_import("src.model_training", "MultiModelTrainer")


@unittest.skipIf(load_config is None, "src.utils 依赖缺失")
class TestUtils(unittest.TestCase):

    def test_load_config(self):
        config_path = project_root / "config" / "config.yaml"
        config = load_config(str(config_path))

        self.assertIsInstance(config, dict)
        self.assertIn('data', config)
        self.assertIn('preprocessing', config)

    def test_ensure_dir(self):
        test_dir = project_root / "test_dir"
        ensure_dir(str(test_dir))

        self.assertTrue(test_dir.exists())

        test_dir.rmdir()


@unittest.skipIf(CascadeFeatureSelector is None, "src.feature_selection 依赖缺失")
class TestCascadeFeatureSelector(unittest.TestCase):

    def setUp(self):
        self.config = {
            'feature_selection': {
                'pipeline': ['ttest', 'lasso'],
                'ttest_alpha': 0.05,
                'spearman_threshold': 0.9,
                'lasso_cv_folds': 3,
                'rf_n_top_features': 15,
                'random_state': 42
            }
        }

        np.random.seed(42)
        n_pos = 50
        n_neg = 50
        self.X = np.random.randn(n_pos + n_neg, 20)
        self.X[:n_pos, :3] += 1.5
        self.y = np.array([1] * n_pos + [0] * n_neg)
        self.feature_names = [f'feature_{i}' for i in range(20)]

    def test_cascade_selection(self):
        selector = CascadeFeatureSelector(self.config)
        X_selected, selected_names = selector.fit_transform(
            self.X, self.y, self.feature_names
        )

        self.assertLessEqual(X_selected.shape[1], 20)
        self.assertEqual(X_selected.shape[1], len(selected_names))
        self.assertIsInstance(selected_names, list)

    def test_selection_log(self):
        selector = CascadeFeatureSelector(self.config)
        selector.fit_transform(
            self.X, self.y, self.feature_names
        )

        log = selector.get_selection_log()
        self.assertIn('ttest', log)
        self.assertIn('lasso', log)


@unittest.skipIf(MultiModelTrainer is None, "src.model_training 依赖缺失")
class TestMultiModelTrainer(unittest.TestCase):

    def setUp(self):
        self.config = {
            'model': {
                'types': ['LR', 'SVM'],
                'cv_folds': 3,
                'random_state': 42
            },
            'training': {
                'test_size': 0.2,
                'val_size': 0.15
            }
        }

        np.random.seed(42)
        n_samples = 100
        n_features = 10

        self.df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.df['pCR'] = np.random.randint(0, 2, n_samples)
        self.feature_cols = [f'feature_{i}' for i in range(n_features)]

    def test_data_preparation(self):
        trainer = MultiModelTrainer(self.config)
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            self.df, self.feature_cols, label_col='pCR'
        )

        self.assertEqual(len(X_train) + len(X_test), 100)
        self.assertEqual(len(y_train) + len(y_test), 100)

    def test_model_training(self):
        trainer = MultiModelTrainer(self.config)
        trainer.prepare_data(
            self.df, self.feature_cols, label_col='pCR'
        )
        models, results = trainer.train_all_models()

        self.assertGreater(len(models), 0)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
