#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单元测试
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, ensure_dir
from src.feature_selection import FeatureSelector

class TestUtils(unittest.TestCase):
    """工具函数测试"""
    
    def test_load_config(self):
        """测试配置加载"""
        config_path = project_root / "config" / "config.yaml"
        config = load_config(str(config_path))
        
        self.assertIsInstance(config, dict)
        self.assertIn('data', config)
        self.assertIn('preprocessing', config)
    
    def test_ensure_dir(self):
        """测试目录创建"""
        test_dir = project_root / "test_dir"
        ensure_dir(str(test_dir))
        
        self.assertTrue(test_dir.exists())
        
        # 清理
        test_dir.rmdir()

class TestFeatureSelector(unittest.TestCase):
    """特征选择器测试"""
    
    def setUp(self):
        """设置测试数据"""
        self.config = {
            'feature_selection': {
                'method': 'lasso',
                'n_features': 5,
                'cv_folds': 3,
                'random_state': 42
            }
        }
        
        # 创建测试数据
        np.random.seed(42)
        self.X = np.random.randn(100, 20)
        self.y = np.random.randint(0, 2, 100)
        self.feature_names = [f'feature_{i}' for i in range(20)]
    
    def test_lasso_selection(self):
        """测试LASSO特征选择"""
        selector = FeatureSelector(self.config)
        X_selected, selected_names = selector.fit_transform(
            self.X, self.y, self.feature_names
        )
        
        self.assertEqual(X_selected.shape[1], 5)
        self.assertEqual(len(selected_names), 5)
        self.assertIsInstance(selected_names, list)

class TestModelTrainer(unittest.TestCase):
    """模型训练器测试"""
    
    def setUp(self):
        """设置测试数据"""
        self.config = {
            'model': {
                'types': ['lr'],
                'cv_folds': 3,
                'random_state': 42
            },
            'training': {
                'test_size': 0.2,
                'val_size': 0.2
            }
        }
        
        # 创建测试数据
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        self.df = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.df['label'] = np.random.randint(0, 2, n_samples)
        self.feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    def test_data_preparation(self):
        """测试数据准备"""
        from src.model_training import ModelTrainer
        
        trainer = ModelTrainer(self.config)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
            self.df, self.feature_cols, label_col='label'
        )
        
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 100)
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), 100)

if __name__ == '__main__':
    unittest.main()
