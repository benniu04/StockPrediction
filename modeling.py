"""
Improved Modeling Framework
- Walk-forward validation to prevent overfitting
- Proper cross-validation
- Financial metrics for evaluation
- Hyperparameter tuning with validation
- Multi-stock training capability
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss
)
from xgboost import XGBClassifier
import lightgbm as lgb


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    
    # Trading-specific metrics
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    
    # Per-split metrics for stability check
    split_accuracies: List[float]
    split_sharpes: List[float]


class WalkForwardValidator:
    """
    Walk-forward validation for time series
    Prevents overfitting by simulating real trading conditions
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,  # ~1 year of trading days
        gap: int = 5  # gap between train and test to prevent leakage
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation
        
        Returns list of (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        min_train_size = 252 * 2  # At least 2 years of data
        
        if n_samples < min_train_size + self.test_size:
            raise ValueError("Not enough data for walk-forward validation")
        
        splits = []
        
        # Calculate step size for expanding window
        total_test_periods = self.n_splits * self.test_size
        available_samples = n_samples - min_train_size
        
        for i in range(self.n_splits):
            # Expanding window: train size grows with each split
            train_end = min_train_size + (i * (available_samples - total_test_periods) // self.n_splits)
            test_start = train_end + self.gap
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
                
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits


class ImprovedStockPredictor:
    """
    Improved stock prediction model with:
    - Robust cross-validation
    - Proper hyperparameter tuning
    - Financial metrics
    - Multi-model ensemble
    """
    
    def __init__(
        self,
        use_walk_forward: bool = True,
        n_cv_splits: int = 5,
        transaction_cost: float = 0.0005,
        random_state: int = 42
    ):
        self.use_walk_forward = use_walk_forward
        self.n_cv_splits = n_cv_splits
        self.transaction_cost = transaction_cost
        self.random_state = random_state
        
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.models = {}
        self.meta_model = None
        self.best_threshold = 0.5
        
    def _create_base_models(self) -> Dict[str, any]:
        """Create base models with regularization"""
        return {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,  # Limit depth to prevent overfitting
                min_samples_split=50,  # Require more samples per split
                min_samples_leaf=20,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgb': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,  # Lower learning rate
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                scale_pos_weight=1,
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'lgb': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                class_weight='balanced',
                random_state=self.random_state,
                verbosity=-1
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=self.random_state
            )
        }
    
    def _calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics"""
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'log_loss': log_loss(y_true, y_proba)
        }
        
        # Trading metrics
        if returns is not None:
            # Strategy returns: go long when predict up, flat when predict down
            position = y_pred.copy()
            strat_returns = position * returns
            
            # Account for transaction costs
            trades = np.abs(np.diff(position, prepend=0))
            strat_returns = strat_returns - trades * self.transaction_cost
            
            # Sharpe ratio (annualized)
            if len(strat_returns) > 0 and strat_returns.std() > 0:
                metrics['sharpe_ratio'] = (
                    np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252)
                )
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Maximum drawdown
            cumulative = (1 + strat_returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdown)
            
            # Win rate
            winning_trades = np.sum(strat_returns > 0)
            total_trades = np.sum(trades > 0)
            metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Profit factor
            gross_profit = np.sum(strat_returns[strat_returns > 0])
            gross_loss = np.abs(np.sum(strat_returns[strat_returns < 0]))
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Total return
            metrics['total_return'] = (cumulative[-1] - 1) if len(cumulative) > 0 else 0.0
        else:
            metrics.update({
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0
            })
        
        return metrics
    
    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> float:
        """
        Find optimal probability threshold based on Sharpe ratio
        If no returns provided, use F1 score
        """
        thresholds = np.arange(0.45, 0.80, 0.01)
        best_score = -np.inf
        best_thresh = 0.5
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if returns is not None:
                # Optimize for Sharpe ratio
                position = y_pred.copy()
                strat_returns = position * returns
                trades = np.abs(np.diff(position, prepend=0))
                strat_returns = strat_returns - trades * self.transaction_cost
                
                if strat_returns.std() > 0:
                    score = np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(252)
                else:
                    score = 0.0
            else:
                # Optimize for F1 score
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        return best_thresh
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'ImprovedStockPredictor':
        """
        Fit the model with proper cross-validation
        
        Args:
            X: Feature dataframe
            y: Target labels
            returns: Actual returns for trading metrics (optional)
            verbose: Print progress
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        returns_array = returns.values if returns is not None else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Setup cross-validation
        if self.use_walk_forward:
            cv_splitter = WalkForwardValidator(n_splits=self.n_cv_splits)
            splits = cv_splitter.split(X_scaled)
        else:
            cv_splitter = TimeSeriesSplit(n_splits=self.n_cv_splits)
            splits = list(cv_splitter.split(X_scaled))
        
        # Store out-of-fold predictions for meta-model
        oof_predictions = np.zeros((X_scaled.shape[0], 4))  # 4 base models
        
        # Train base models
        self.models = self._create_base_models()
        
        if verbose:
            print(f"\nTraining with {len(splits)} CV splits...")
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_array[train_idx], y_array[val_idx]
            
            if verbose:
                print(f"\nFold {fold_idx + 1}/{len(splits)}")
                print(f"  Train: {len(train_idx)} samples, Test: {len(val_idx)} samples")
            
            # Train each base model
            for model_idx, (name, model) in enumerate(self.models.items()):
                model.fit(X_train, y_train)
                
                # Get out-of-fold predictions for meta-model
                oof_predictions[val_idx, model_idx] = model.predict_proba(X_val)[:, 1]
                
                if verbose:
                    val_preds = (oof_predictions[val_idx, model_idx] > 0.5).astype(int)
                    acc = accuracy_score(y_val, val_preds)
                    print(f"  {name}: Validation Accuracy = {acc:.4f}")
        
        # Train meta-model on out-of-fold predictions
        # Use only samples that have predictions (from validation folds)
        valid_mask = oof_predictions.sum(axis=1) > 0
        
        self.meta_model = LogisticRegression(
            penalty='l2',
            C=0.1,  # Strong regularization
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        self.meta_model.fit(oof_predictions[valid_mask], y_array[valid_mask])
        
        # Optimize threshold on validation data
        meta_proba = self.meta_model.predict_proba(oof_predictions[valid_mask])[:, 1]
        val_returns = returns_array[valid_mask] if returns_array is not None else None
        self.best_threshold = self._optimize_threshold(
            y_array[valid_mask],
            meta_proba,
            val_returns
        )
        
        if verbose:
            print(f"\nOptimal threshold: {self.best_threshold:.3f}")
        
        # Retrain all models on full dataset
        for name, model in self.models.items():
            model.fit(X_scaled, y_array)
        
        # Retrain meta-model on full predictions
        full_base_preds = np.column_stack([
            model.predict_proba(X_scaled)[:, 1]
            for model in self.models.values()
        ])
        self.meta_model.fit(full_base_preds, y_array)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)
        
        # Get predictions from base models
        base_preds = np.column_stack([
            model.predict_proba(X_scaled)[:, 1]
            for model in self.models.values()
        ])
        
        # Meta-model prediction
        return self.meta_model.predict_proba(base_preds)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict class labels using optimal threshold"""
        if threshold is None:
            threshold = self.best_threshold
        
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        returns: Optional[pd.Series] = None
    ) -> ModelMetrics:
        """
        Comprehensive model evaluation with trading metrics
        """
        y_proba = self.predict_proba(X)
        y_pred = self.predict(X)
        
        y_array = y.values if isinstance(y, pd.Series) else y
        returns_array = returns.values if returns is not None else None
        
        metrics = self._calculate_trading_metrics(
            y_array, y_pred, y_proba, returns_array
        )
        
        # Calculate per-split metrics for stability
        if self.use_walk_forward:
            cv_splitter = WalkForwardValidator(n_splits=self.n_cv_splits)
        else:
            cv_splitter = TimeSeriesSplit(n_splits=self.n_cv_splits)
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)
        
        split_accs = []
        split_sharpes = []
        
        for train_idx, test_idx in cv_splitter.split(X_scaled):
            y_test = y_array[test_idx]
            X_test = X_scaled[test_idx]
            
            base_preds = np.column_stack([
                model.predict_proba(X_test)[:, 1]
                for model in self.models.values()
            ])
            test_proba = self.meta_model.predict_proba(base_preds)[:, 1]
            test_pred = (test_proba >= self.best_threshold).astype(int)
            
            split_accs.append(accuracy_score(y_test, test_pred))
            
            if returns_array is not None:
                test_returns = returns_array[test_idx]
                strat_rets = test_pred * test_returns
                if strat_rets.std() > 0:
                    sharpe = np.mean(strat_rets) / np.std(strat_rets) * np.sqrt(252)
                else:
                    sharpe = 0.0
                split_sharpes.append(sharpe)
        
        return ModelMetrics(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            roc_auc=metrics['roc_auc'],
            log_loss=metrics['log_loss'],
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_return=metrics['total_return'],
            split_accuracies=split_accs,
            split_sharpes=split_sharpes
        )


class MultiStockPredictor:
    """
    Train a single model across multiple stocks
    This improves generalization by learning universal patterns
    """
    
    def __init__(
        self,
        base_predictor: ImprovedStockPredictor,
        stock_symbols: List[str]
    ):
        self.base_predictor = base_predictor
        self.stock_symbols = stock_symbols
    
    def fit(
        self,
        stock_data: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series]],
        verbose: bool = True
    ):
        """
        Fit on multiple stocks
        
        Args:
            stock_data: Dict mapping symbol -> (X, y, returns) tuples
        """
        # Combine all stock data
        X_list = []
        y_list = []
        returns_list = []
        
        for symbol in self.stock_symbols:
            if symbol in stock_data:
                X, y, returns = stock_data[symbol]
                X_list.append(X)
                y_list.append(y)
                returns_list.append(returns)
        
        X_combined = pd.concat(X_list, axis=0)
        y_combined = pd.concat(y_list, axis=0)
        returns_combined = pd.concat(returns_list, axis=0)
        
        if verbose:
            print(f"\nTraining on {len(X_combined)} samples from {len(self.stock_symbols)} stocks")
        
        self.base_predictor.fit(X_combined, y_combined, returns_combined, verbose=verbose)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_predictor.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_predictor.predict(X)

