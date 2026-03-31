import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib
from typing import Dict, Tuple, Any
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: dict):
        """
        初始化模型训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_types = config['model']['types']
        self.cv_folds = config['model']['cv_folds']
        self.random_state = config['model']['random_state']
        self.test_size = config['training']['test_size']
        self.val_size = config['training']['val_size']
        
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        logger.info("ModelTrainer initialized")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: list, 
        label_col: str = 'label'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            df: 数据DataFrame
            feature_cols: 特征列名列表
            label_col: 标签列名
            
        Returns:
            训练集、验证集、测试集的特征和标签
        """
        logger.info("Preparing training data")
        
        X = df[feature_cols].values
        y = df[label_col].values
        
        # 分割数据
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size_adjusted, 
            random_state=self.random_state, 
            stratify=y_train_val
        )
        
        # 标准化
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_val = self.scaler.transform(X_val)
        self.X_test = self.scaler.transform(X_test)
        
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        logger.info(f"Training set: {self.X_train.shape[0]} samples")
        logger.info(f"Validation set: {self.X_val.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        logger.info(f"Class distribution (train): {np.bincount(self.y_train)}")
        logger.info(f"Class distribution (val): {np.bincount(self.y_val)}")
        logger.info(f"Class distribution (test): {np.bincount(self.y_test)}")
        
        return (self.X_train, self.X_val, self.X_test, 
                self.y_train, self.y_val, self.y_test)
    
    def train_models(self) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        """
        训练多个模型
        
        Returns:
            模型字典和结果字典
        """
        if self.X_train is None:
            raise ValueError("Data has not been prepared. Call prepare_data first.")
        
        logger.info("Training models")
        
        model_configs = self._get_model_configs()
        
        for name, model in model_configs.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {name} model...")
            logger.info(f"{'='*50}")
            
            # 交叉验证
            cv = StratifiedKFold(
                n_splits=self.cv_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
            cv_scores = cross_val_score(
                model, 
                self.X_train, 
                self.y_train, 
                cv=cv, 
                scoring='roc_auc',
                n_jobs=-1
            )
            logger.info(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # 训练
            model.fit(self.X_train, self.y_train)
            
            # 验证集评估
            y_val_pred = model.predict(self.X_val)
            y_val_prob = model.predict_proba(self.X_val)[:, 1]
            val_auc = roc_auc_score(self.y_val, y_val_prob)
            
            # 测试集评估
            y_test_pred = model.predict(self.X_test)
            y_test_prob = model.predict_proba(self.X_test)[:, 1]
            test_auc = roc_auc_score(self.y_test, y_test_prob)
            test_acc = accuracy_score(self.y_test, y_test_pred)
            
            # 混淆矩阵
            tn, fp, fn, tp = confusion_matrix(self.y_test, y_test_pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            
            logger.info(f"\nValidation AUC: {val_auc:.3f}")
            logger.info(f"Test AUC: {test_auc:.3f}")
            logger.info(f"Test Accuracy: {test_acc:.3f}")
            logger.info(f"Test Sensitivity: {sensitivity:.3f}")
            logger.info(f"Test Specificity: {specificity:.3f}")
            
            # 保存结果
            self.models[name] = model
            self.results[name] = {
                'cv_scores': cv_scores,
                'val_auc': val_auc,
                'test_auc': test_auc,
                'test_acc': test_acc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'y_test_pred': y_test_pred,
                'y_test_prob': y_test_prob
            }
        
        return self.models, self.results
    
    def _get_model_configs(self) -> Dict[str, Any]:
        """获取模型配置"""
        configs = {}
        
        if 'lr' in self.model_types:
            configs['LR'] = LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state, 
                class_weight='balanced'
            )
        
        if 'svm' in self.model_types:
            configs['SVM'] = SVC(
                kernel='rbf', 
                probability=True, 
                random_state=self.random_state, 
                class_weight='balanced'
            )
        
        if 'rf' in self.model_types:
            configs['RF'] = RandomForestClassifier(
                n_estimators=500, 
                max_depth=10, 
                random_state=self.random_state, 
                class_weight='balanced',
                n_jobs=-1
            )
        
        if 'xgboost' in self.model_types:
            configs['XGBoost'] = XGBClassifier(
                n_estimators=500, 
                max_depth=6, 
                learning_rate=0.01, 
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        return configs
    
    def get_best_model(self, metric: str = 'test_auc') -> Tuple[Any, str]:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            最佳模型和模型名称
        """
        if not self.results:
            raise ValueError("Models have not been trained yet")
        
        best_score = 0
        best_model_name = None
        
        for name, result in self.results.items():
            score = result[metric]
            if score > best_score:
                best_score = score
                best_model_name = name
        
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"{metric}: {best_score:.3f}")
        
        return self.models[best_model_name], best_model_name
    
    def save_model(self, model: Any, model_name: str, output_dir: str = "results/models") -> None:
        """
        保存模型
        
        Args:
            model: 模型对象
            model_name: 模型名称
            output_dir: 输出目录
        """
        ensure_dir(output_dir)
        
        model_path = f"{output_dir}/{model_name}.pkl"
        joblib.dump(model, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str) -> Any:
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            模型对象
        """
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        return model
