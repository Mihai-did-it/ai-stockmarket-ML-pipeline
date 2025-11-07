"""
Model Trainer
Trains and evaluates ML models with hyperparameter tuning and cross-validation.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available for hyperparameter tuning")

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models for stock prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model = None
        self.feature_importance = None
        self.best_params = None
        
        # Extract config
        self.algorithm = config.get('algorithm', 'lightgbm')
        self.random_seed = config.get('execution', {}).get('random_seed', 42)
        self.n_jobs = config.get('execution', {}).get('n_jobs', -1)
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info(f"Training {self.algorithm} model")
        logger.info(f"Train size: {X_train.shape}, Val size: {X_val.shape}")
        
        # Handle class imbalance if configured
        if self.config.get('handle_imbalance', {}).get('method') != 'none':
            X_train, y_train = self._handle_imbalance(X_train, y_train)
        
        # Hyperparameter tuning if enabled
        if self.config.get('hyperparameter_tuning', {}).get('enabled', False):
            self.best_params = self._tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            self.best_params = self._get_default_params()
        
        # Train model with best parameters
        self.model = self._train_model(X_train, y_train, X_val, y_val, self.best_params)
        
        # Calibrate if configured
        if self.config.get('calibration', {}).get('enabled', False):
            self.model = self._calibrate_model(self.model, X_val, y_val)
        
        # Calculate feature importance
        self.feature_importance = self._get_feature_importance(X_train.columns)
        
        logger.info("Training completed")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating model on test set")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'log_loss': log_loss(y_test, y_pred_proba),
        }
        
        # Multi-class ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, 
                                               multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            metrics['roc_auc'] = None
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=['SELL', 'HOLD', 'BUY'],
                                           output_dict=True)
        metrics['classification_report'] = class_report
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = conf_matrix
        
        # Per-class metrics
        for i, label in enumerate(['SELL', 'HOLD', 'BUY']):
            if label in class_report:
                metrics[f'{label.lower()}_precision'] = class_report[label]['precision']
                metrics[f'{label.lower()}_recall'] = class_report[label]['recall']
                metrics[f'{label.lower()}_f1'] = class_report[label]['f1-score']
        
        # Log metrics
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1 (weighted): {metrics['f1']:.4f}")
        logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "ROC AUC: N/A")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            predictions (class labels), probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def _handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance."""
        method = self.config.get('handle_imbalance', {}).get('method', 'none')
        
        logger.info(f"Handling class imbalance with method: {method}")
        logger.info(f"Original class distribution:\n{y.value_counts()}")
        
        if method == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=self.random_seed)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
        elif method == 'undersampling':
            # Random undersampling
            undersample = RandomUnderSampler(random_state=self.random_seed)
            X_resampled, y_resampled = undersample.fit_resample(X, y)
            
        elif method == 'class_weights':
            # Use class weights in model (no resampling needed)
            return X, y
        
        else:
            return X, y
        
        logger.info(f"Resampled class distribution:\n{pd.Series(y_resampled).value_counts()}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default hyperparameters for the algorithm."""
        
        if self.algorithm == 'lightgbm':
            return {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.0,
                'lambda_l2': 0.0,
                'verbosity': -1,
                'random_state': self.random_seed,
            }
        
        elif self.algorithm == 'xgboost':
            return {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': self.random_seed,
                'n_jobs': self.n_jobs,
            }
        
        elif self.algorithm == 'catboost':
            return {
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': self.random_seed,
                'verbose': False,
            }
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return self._get_default_params()
        
        logger.info("Starting hyperparameter tuning with Optuna")
        
        n_trials = self.config.get('hyperparameter_tuning', {}).get('n_trials', 100)
        timeout = self.config.get('hyperparameter_tuning', {}).get('timeout_hours', 2) * 3600
        
        def objective(trial):
            if self.algorithm == 'lightgbm':
                params = {
                    'objective': 'multiclass',
                    'num_class': 3,
                    'metric': 'multi_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                    'verbosity': -1,
                    'random_state': self.random_seed,
                }
                
                model = lgb.LGBMClassifier(**params, n_estimators=500)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50, verbose=False)])
                
                y_pred_proba = model.predict_proba(X_val)
                score = log_loss(y_val, y_pred_proba)
                
                return score
            
            # Add XGBoost and CatBoost tuning here similarly
            else:
                return 0.0
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        logger.info(f"Best trial: {study.best_trial.value:.4f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Merge with default params
        best_params = self._get_default_params()
        best_params.update(study.best_params)
        
        return best_params
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    params: Dict[str, Any]):
        """Train the model with given parameters."""
        
        if self.algorithm == 'lightgbm':
            model = lgb.LGBMClassifier(**params, n_estimators=1000)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)]
            )
            
        elif self.algorithm == 'xgboost':
            model = xgb.XGBClassifier(**params, n_estimators=1000)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=100
            )
            
        elif self.algorithm == 'catboost':
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=100
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return model
    
    def _calibrate_model(self, model, X_val: pd.DataFrame, y_val: pd.Series):
        """Calibrate model probabilities."""
        
        method = self.config.get('calibration', {}).get('method', 'isotonic')
        logger.info(f"Calibrating model with method: {method}")
        
        calibrated = CalibratedClassifierCV(
            model, method=method, cv='prefit'
        )
        calibrated.fit(X_val, y_val)
        
        return calibrated
    
    def _get_feature_importance(self, feature_names) -> pd.DataFrame:
        """Get feature importance."""
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'base_estimator'):
                # For calibrated classifier
                importances = self.model.base_estimator.feature_importances_
            else:
                logger.warning("Model does not have feature_importances_")
                return pd.DataFrame()
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            logger.info(f"Top 10 features:\n{importance_df.head(10)}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = path.with_suffix('.pkl')
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save feature importance
        if self.feature_importance is not None and not self.feature_importance.empty:
            importance_path = path.with_name(f"{path.stem}_feature_importance.csv")
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Save config
        config_path = path.with_name(f"{path.stem}_config.pkl")
        joblib.dump(self.config, config_path)
        logger.info(f"Config saved to {config_path}")
    
    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        
        path = Path(path)
        model_path = path.with_suffix('.pkl')
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load feature importance if exists
        importance_path = path.with_name(f"{path.stem}_feature_importance.csv")
        if importance_path.exists():
            self.feature_importance = pd.read_csv(importance_path)
            logger.info(f"Feature importance loaded from {importance_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]
    
    # Train model
    config = {
        'algorithm': 'lightgbm',
        'handle_imbalance': {'method': 'none'},
        'hyperparameter_tuning': {'enabled': False},
        'calibration': {'enabled': False},
        'execution': {'random_seed': 42, 'n_jobs': -1}
    }
    
    trainer = ModelTrainer(config)
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print(f"\nTest Metrics: {metrics}")
