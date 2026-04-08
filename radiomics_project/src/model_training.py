import os
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import joblib
from typing import Dict, Tuple, Any, List, Optional
import logging

from .utils import ensure_dir, setup_logger

logger = setup_logger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, skipping XGBoost model")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed, skipping LightGBM model")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not installed, skipping CatBoost model")


class MultiModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.model_types = self.model_config['types']
        self.cv_folds = self.model_config['cv_folds']
        self.random_state = self.model_config['random_state']
        self.test_size = self.training_config['test_size']
        self.val_size = self.training_config['val_size']

        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model_name = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        logger.info(f"MultiModelTrainer initialized: {len(self.model_types)} models")

    def _get_model_configs(self) -> Dict[str, Any]:
        configs = {}

        if 'LR' in self.model_types:
            configs['LR'] = LogisticRegression(
                max_iter=1000, random_state=self.random_state,
                class_weight='balanced'
            )

        if 'SVM' in self.model_types:
            configs['SVM'] = SVC(
                kernel='rbf', probability=True,
                random_state=self.random_state, class_weight='balanced'
            )

        if 'KNN' in self.model_types:
            configs['KNN'] = KNeighborsClassifier(n_neighbors=5)

        if 'DT' in self.model_types:
            configs['DT'] = DecisionTreeClassifier(
                random_state=self.random_state, class_weight='balanced'
            )

        if 'RF' in self.model_types:
            configs['RF'] = RandomForestClassifier(
                n_estimators=500, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            )

        if 'ET' in self.model_types:
            configs['ET'] = ExtraTreesClassifier(
                n_estimators=500, random_state=self.random_state,
                class_weight='balanced', n_jobs=-1
            )

        if 'AdaBoost' in self.model_types:
            configs['AdaBoost'] = AdaBoostClassifier(
                n_estimators=100, random_state=self.random_state
            )

        if 'GB' in self.model_types:
            configs['GB'] = GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            )

        if 'XGBoost' in self.model_types and HAS_XGBOOST:
            configs['XGBoost'] = XGBClassifier(
                n_estimators=500, random_state=self.random_state,
                use_label_encoder=False, eval_metric='logloss'
            )

        if 'LightGBM' in self.model_types and HAS_LIGHTGBM:
            configs['LightGBM'] = LGBMClassifier(
                n_estimators=500, random_state=self.random_state,
                class_weight='balanced', verbose=-1
            )

        if 'CatBoost' in self.model_types and HAS_CATBOOST:
            configs['CatBoost'] = CatBoostClassifier(
                iterations=500, random_state=self.random_state, verbose=0
            )

        if 'GNB' in self.model_types:
            configs['GNB'] = GaussianNB()

        if 'LDA' in self.model_types:
            configs['LDA'] = LinearDiscriminantAnalysis()

        if 'MLP' in self.model_types:
            mlp_cfg = self.model_config.get('mlp', {})
            configs['MLP'] = MLPClassifier(
                hidden_layer_sizes=tuple(mlp_cfg.get('hidden_layer_sizes', [64, 32, 16])[0]),
                activation=mlp_cfg.get('activation', 'relu'),
                solver=mlp_cfg.get('solver', 'adam'),
                alpha=mlp_cfg.get('alpha', [0.001])[0],
                max_iter=mlp_cfg.get('max_iter', 1000),
                early_stopping=mlp_cfg.get('early_stopping', True),
                validation_fraction=mlp_cfg.get('validation_fraction', 0.1),
                n_iter_no_change=mlp_cfg.get('n_iter_no_change', 20),
                random_state=self.random_state
            )

        return configs

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = 'pCR',
        external_df: Optional[pd.DataFrame] = None
    ) -> Tuple:
        logger.info("Preparing training data")

        X = df[feature_cols].values
        y = df[label_col].values

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.X_train = self.scaler.fit_transform(X_train_val)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train_val
        self.y_test = y_test

        self.feature_cols = feature_cols

        logger.info(f"Training set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        logger.info(f"Class distribution (train): {np.bincount(self.y_train)}")
        logger.info(f"Class distribution (test): {np.bincount(self.y_test)}")

        if external_df is not None:
            X_ext = external_df[feature_cols].values
            y_ext = external_df[label_col].values
            self.X_external = self.scaler.transform(X_ext)
            self.y_external = y_ext
            logger.info(f"External validation set: {len(y_ext)} samples")

        return (self.X_train, self.X_test,
                self.y_train, self.y_test)

    def train_all_models(self) -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data first.")

        logger.info(f"Training {len(self._get_model_configs())} models")

        model_configs = self._get_model_configs()

        for name, model in model_configs.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {name}...")
            logger.info(f"{'='*50}")

            try:
                cv = StratifiedKFold(
                    n_splits=self.cv_folds,
                    shuffle=True,
                    random_state=self.random_state
                )
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring='roc_auc', n_jobs=-1
                )
                logger.info(f"CV AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

                model.fit(self.X_train, self.y_train)

                y_test_pred = model.predict(self.X_test)
                y_test_prob = model.predict_proba(self.X_test)[:, 1]
                test_auc = roc_auc_score(self.y_test, y_test_prob)
                test_acc = accuracy_score(self.y_test, y_test_pred)

                tn, fp, fn, tp = confusion_matrix(
                    self.y_test, y_test_pred
                ).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0

                result = {
                    'cv_scores': cv_scores,
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'test_auc': test_auc,
                    'test_acc': test_acc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'ppv': ppv,
                    'npv': npv,
                    'y_test_pred': y_test_pred,
                    'y_test_prob': y_test_prob
                }

                if hasattr(self, 'X_external'):
                    y_ext_pred = model.predict(self.X_external)
                    y_ext_prob = model.predict_proba(self.X_external)[:, 1]
                    ext_auc = roc_auc_score(self.y_external, y_ext_prob)
                    ext_acc = accuracy_score(self.y_external, y_ext_pred)

                    tn_e, fp_e, fn_e, tp_e = confusion_matrix(
                        self.y_external, y_ext_pred
                    ).ravel()
                    ext_sens = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0
                    ext_spec = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0

                    result.update({
                        'external_auc': ext_auc,
                        'external_acc': ext_acc,
                        'external_sensitivity': ext_sens,
                        'external_specificity': ext_spec,
                        'y_ext_pred': y_ext_pred,
                        'y_ext_prob': y_ext_prob
                    })

                self.models[name] = model
                self.results[name] = result

                logger.info(f"Test AUC: {test_auc:.3f}")
                logger.info(f"Test Accuracy: {test_acc:.3f}")
                logger.info(f"Sensitivity: {sensitivity:.3f}")
                logger.info(f"Specificity: {specificity:.3f}")

                if hasattr(self, 'X_external'):
                    logger.info(f"External AUC: {ext_auc:.3f}")

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue

        self._select_best_model()
        return self.models, self.results

    def _select_best_model(self, metric: str = 'test_auc') -> None:
        if not self.results:
            raise ValueError("No models trained yet")

        best_score = 0
        best_name = None

        for name, result in self.results.items():
            score = result.get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = name

        self.best_model_name = best_name
        logger.info(f"\nBest model: {best_name} ({metric}={best_score:.3f})")

    def get_best_model(self) -> Tuple[Any, str]:
        if self.best_model_name is None:
            self._select_best_model()
        return self.models[self.best_model_name], self.best_model_name

    def optimize_mlp(self) -> Any:
        logger.info("Optimizing MLP with GridSearchCV")

        mlp_cfg = self.model_config.get('mlp', {})

        param_grid = {
            'hidden_layer_sizes': [
                tuple(s) for s in mlp_cfg.get('hidden_layer_sizes', [[64, 32]])
            ],
            'alpha': mlp_cfg.get('alpha', [0.001]),
            'learning_rate_init': mlp_cfg.get('learning_rate_init', [0.001]),
            'batch_size': mlp_cfg.get('batch_size', [16, 32])
        }

        base_mlp = MLPClassifier(
            activation=mlp_cfg.get('activation', 'relu'),
            solver=mlp_cfg.get('solver', 'adam'),
            max_iter=mlp_cfg.get('max_iter', 1000),
            early_stopping=mlp_cfg.get('early_stopping', True),
            validation_fraction=mlp_cfg.get('validation_fraction', 0.1),
            n_iter_no_change=mlp_cfg.get('n_iter_no_change', 20),
            random_state=self.random_state
        )

        grid_search = GridSearchCV(
            base_mlp, param_grid,
            cv=self.cv_folds, scoring='roc_auc',
            n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        logger.info(f"Best MLP params: {grid_search.best_params_}")
        logger.info(f"Best MLP CV AUC: {grid_search.best_score_:.3f}")

        self.models['MLP_optimized'] = grid_search.best_estimator_

        y_test_prob = grid_search.best_estimator_.predict_proba(self.X_test)[:, 1]
        test_auc = roc_auc_score(self.y_test, y_test_prob)
        self.results['MLP_optimized'] = {
            'test_auc': test_auc,
            'y_test_prob': y_test_prob,
            'y_test_pred': grid_search.best_estimator_.predict(self.X_test),
            'best_params': grid_search.best_params_
        }

        return grid_search.best_estimator_

    def build_combined_model(
        self,
        radiomics_scores_train: np.ndarray,
        afp_response_train: np.ndarray,
        radiomics_scores_test: np.ndarray,
        afp_response_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Any:
        logger.info("Building combined clinical-radiomics model")

        combined_train = np.column_stack([radiomics_scores_train, afp_response_train])
        combined_test = np.column_stack([radiomics_scores_test, afp_response_test])

        combined_model = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000
        )
        combined_model.fit(combined_train, y_train)

        y_test_prob = combined_model.predict_proba(combined_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_prob)

        logger.info(f"Combined model test AUC: {test_auc:.3f}")

        self.models['Combined_Radiomics_AFP'] = combined_model
        self.results['Combined_Radiomics_AFP'] = {
            'test_auc': test_auc,
            'y_test_prob': y_test_prob,
            'y_test_pred': combined_model.predict(combined_test)
        }

        return combined_model

    def get_results_summary(self) -> pd.DataFrame:
        summary = []
        for name, result in self.results.items():
            row = {
                'Model': name,
                'CV_AUC_Mean': result.get('cv_auc_mean', np.nan),
                'CV_AUC_Std': result.get('cv_auc_std', np.nan),
                'Test_AUC': result.get('test_auc', np.nan),
                'Test_Acc': result.get('test_acc', np.nan),
                'Sensitivity': result.get('sensitivity', np.nan),
                'Specificity': result.get('specificity', np.nan),
                'PPV': result.get('ppv', np.nan),
                'NPV': result.get('npv', np.nan),
            }
            if 'external_auc' in result:
                row.update({
                    'External_AUC': result['external_auc'],
                    'External_Acc': result['external_acc'],
                    'External_Sensitivity': result['external_sensitivity'],
                    'External_Specificity': result['external_specificity'],
                })
            summary.append(row)

        df_summary = pd.DataFrame(summary)
        df_summary = df_summary.sort_values('Test_AUC', ascending=False)
        return df_summary

    def save_model(self, model: Any, model_name: str,
                   output_dir: str = "results/models") -> None:
        ensure_dir(output_dir)
        model_path = os.path.join(output_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    def save_scaler(self, output_dir: str = "results/models") -> None:
        ensure_dir(output_dir)
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

    def load_model(self, model_path: str) -> Any:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
