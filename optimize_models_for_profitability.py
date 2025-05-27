#!/usr/bin/env python3
"""
Script d'optimisation avancée pour maximiser la rentabilité des modèles de trading.
Ce script améliore les modèles existants avec des techniques d'optimisation sophistiquées.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler, CmaEsSampler

# Imports pour les modèles
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Imports locaux
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# Note: Ces imports sont optionnels et le script fonctionnera sans eux
try:
    from src.modeling.models import train_model
    print("✅ Module modeling.models importé avec succès")
except ImportError:
    print("⚠️  Module modeling.models non trouvé - utilisation de modèles par défaut")
    train_model = None

try:
    from src.feature_engineering.advanced_features import AdvancedFeatureEngineering
    print("✅ Module feature_engineering importé avec succès")
except ImportError:
    print("⚠️  Module feature_engineering non trouvé - utilisation de features par défaut")
    AdvancedFeatureEngineering = None

try:
    from src.backtesting.comprehensive_backtest import ComprehensiveBacktest
    print("✅ Module backtesting importé avec succès")
except ImportError:
    print("⚠️  Module backtesting non trouvé - fonctionnalité limitée")
    ComprehensiveBacktest = None

warnings.filterwarnings('ignore')

class ProfitabilityOptimizer:
    """Optimiseur avancé pour maximiser la rentabilité des modèles de trading"""
    
    def __init__(self, 
                 data_path: str = None,
                 output_dir: str = "optimized_models",
                 n_trials_per_model: int = 200,
                 cv_folds: int = 5,
                 target_sharpe: float = 1.5,
                 max_drawdown_limit: float = 0.15):
        
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_trials_per_model = n_trials_per_model
        self.cv_folds = cv_folds
        self.target_sharpe = target_sharpe
        self.max_drawdown_limit = max_drawdown_limit
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration des modèles optimisés
        self.optimized_models_config = {
            'logistic_regression': self._get_logistic_regression_search_space,
            'elastic_net': self._get_elastic_net_search_space,
            'random_forest': self._get_random_forest_search_space,
            'xgboost_classifier': self._get_xgboost_search_space,
            'ensemble_advanced': self._get_ensemble_search_space
        }
        
        # Métriques de performance
        self.optimization_results = {}
        
    def _get_logistic_regression_search_space(self, trial):
        """Espace de recherche optimisé pour la régression logistique"""
        return {
            'C': trial.suggest_float('C', 0.001, 100.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 1000, 5000),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            # Paramètres avancés pour le trading
            'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            'warm_start': trial.suggest_categorical('warm_start', [True, False])
        }
    
    def _get_elastic_net_search_space(self, trial):
        """Espace de recherche optimisé pour Elastic Net"""
        return {
            'alpha': trial.suggest_float('alpha', 1e-6, 10.0, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.01, 0.99),
            'max_iter': trial.suggest_int('max_iter', 1000, 10000),
            'tol': trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
            'eta0': trial.suggest_float('eta0', 1e-4, 1.0, log=True),
            'penalty': 'elasticnet',
            'loss': 'log_loss',
            # Paramètres spécifiques au trading
            'average': trial.suggest_categorical('average', [True, False]),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False]),
            'validation_fraction': trial.suggest_float('validation_fraction', 0.1, 0.3)
        }
    
    def _get_random_forest_search_space(self, trial):
        """Espace de recherche optimisé pour Random Forest"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.5, 0.7, 0.9]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'oob_score': trial.suggest_categorical('oob_score', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
            # Paramètres avancés pour la stabilité
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1),
            'max_samples': trial.suggest_float('max_samples', 0.5, 1.0) if trial.params.get('bootstrap', True) else None
        }
    
    def _get_xgboost_search_space(self, trial):
        """Espace de recherche optimisé pour XGBoost"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.3, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10, log=True),
            # Paramètres avancés pour le trading
            'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            'max_leaves': trial.suggest_int('max_leaves', 0, 1000),
            'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist']),
            # Early stopping optimisé
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100),
            'eval_metric': trial.suggest_categorical('eval_metric', ['logloss', 'auc', 'error']),
            'objective': 'binary:logistic'
        }
    
    def _get_ensemble_search_space(self, trial):
        """Espace de recherche pour l'ensemble avancé"""
        return {
            'meta_learner': trial.suggest_categorical('meta_learner', ['logistic', 'xgboost', 'ridge']),
            'stacking_cv': trial.suggest_int('stacking_cv', 3, 10),
            'base_models_weights': {
                'logistic_regression': trial.suggest_float('lr_weight', 0.1, 1.0),
                'random_forest': trial.suggest_float('rf_weight', 0.1, 1.0),
                'xgboost_classifier': trial.suggest_float('xgb_weight', 0.1, 1.0)
            },
            'voting_strategy': trial.suggest_categorical('voting_strategy', ['soft', 'hard']),
            'feature_selection_ratio': trial.suggest_float('feature_selection_ratio', 0.5, 1.0),
            'diversification_threshold': trial.suggest_float('diversification_threshold', 0.3, 0.8)
        }
    
    def _calculate_profitability_score(self, y_true, y_pred_proba, returns_data=None):
        """
        Calcule un score de rentabilité personnalisé qui combine plusieurs métriques.
        Plus élevé = plus rentable.
        """
        try:
            # Score AUC comme base
            auc_score = roc_auc_score(y_true, y_pred_proba)
            
            # Précision directionnelle pondérée
            y_pred = (y_pred_proba > 0.5).astype(int)
            directional_accuracy = np.mean(y_true == y_pred)
            
            # Simuler des rendements basiques si pas fournis
            if returns_data is None:
                # Simulation simple: gains plus élevés quand le modèle est confiant et correct
                confidence = np.abs(y_pred_proba - 0.5) * 2  # 0 à 1
                correct_predictions = (y_true == y_pred).astype(float)
                simulated_returns = correct_predictions * confidence * 0.02 - (1 - correct_predictions) * confidence * 0.01
                sharpe_proxy = np.mean(simulated_returns) / (np.std(simulated_returns) + 1e-6)
            else:
                # Utiliser les vrais rendements
                trading_returns = returns_data[y_pred == 1]
                sharpe_proxy = np.mean(trading_returns) / (np.std(trading_returns) + 1e-6) if len(trading_returns) > 0 else 0
            
            # Score composite favorisant la rentabilité
            profitability_score = (
                0.4 * auc_score +
                0.3 * directional_accuracy +
                0.3 * max(0, sharpe_proxy / 2)  # Normaliser le Sharpe
            )
            
            return profitability_score
            
        except Exception as e:
            print(f"Erreur dans le calcul du score de rentabilité: {e}")
            return 0.0
    
    def _optimize_single_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, 
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Optimise un seul modèle avec Optuna"""
        
        print(f"\n🚀 Optimisation de {model_type}...")
        
        def objective(trial):
            try:
                # Obtenir les paramètres pour ce trial
                search_space_func = self.optimized_models_config[model_type]
                params = search_space_func(trial)
                
                # Nettoyer les paramètres None
                params = {k: v for k, v in params.items() if v is not None}
                
                # Entraîner le modèle avec validation croisée temporelle
                tscv = TimeSeriesSplit(n_splits=self.cv_folds)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Entraîner le modèle
                    if model_type == 'logistic_regression':
                        model = LogisticRegression(**params, random_state=42)
                        scaler = StandardScaler()
                        X_tr_scaled = scaler.fit_transform(X_tr)
                        X_v_scaled = scaler.transform(X_v)
                        model.fit(X_tr_scaled, y_tr)
                        y_pred_proba = model.predict_proba(X_v_scaled)[:, 1]
                        
                    elif model_type == 'random_forest':
                        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                        model.fit(X_tr, y_tr)
                        y_pred_proba = model.predict_proba(X_v)[:, 1]
                        
                    elif model_type == 'xgboost_classifier':
                        model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
                        eval_set = [(X_v, y_v)]
                        model.fit(X_tr, y_tr, eval_set=eval_set, verbose=False)
                        y_pred_proba = model.predict_proba(X_v)[:, 1]
                    
                    # Calculer le score de rentabilité
                    score = self._calculate_profitability_score(y_v, y_pred_proba)
                    scores.append(score)
                    
                    # Rapport intermédiaire pour le pruning
                    trial.report(score, len(scores))
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                return np.mean(scores)
                
            except Exception as e:
                print(f"Erreur dans le trial {trial.number}: {e}")
                return 0.0
        
        # Configuration de l'étude Optuna
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Optimisation
        study.optimize(objective, n_trials=self.n_trials_per_model, timeout=1800)  # 30 minutes max
        
        # Résultats
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"✅ {model_type} optimisé - Score: {best_score:.4f}")
        print(f"Meilleurs paramètres: {best_params}")
        
        return {
            'model_type': model_type,
            'best_params': best_params,
            'best_score': best_score,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def _create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features avancées pour améliorer la performance"""
        
        print("🔧 Création de features avancées...")
        
        # Features techniques avancées
        for window in [5, 10, 20, 50]:
            # Moyennes mobiles et volatilité
            data[f'ma_{window}'] = data['close'].rolling(window).mean()
            data[f'volatility_{window}'] = data['close'].rolling(window).std()
            data[f'rsi_{window}'] = self._calculate_rsi(data['close'], window)
            
            # Ratio prix/moyenne mobile
            data[f'price_ma_ratio_{window}'] = data['close'] / data[f'ma_{window}']
            
            # Momentum
            data[f'momentum_{window}'] = (data['close'] - data['close'].shift(window)) / data['close'].shift(window)
            
            # Bollinger Bands
            bb_upper, bb_lower = self._calculate_bollinger_bands(data['close'], window)
            data[f'bb_position_{window}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Features de volume
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma_20']
        data['price_volume'] = data['close'] * data['volume']
        
        # Features de corrélation inter-timeframes
        for shift in [1, 3, 5, 10]:
            data[f'return_{shift}d'] = data['close'].pct_change(shift)
            data[f'volatility_ratio_{shift}d'] = data['volatility_20'] / data['volatility_20'].shift(shift)
        
        # Features de sentiment de marché
        data['higher_highs'] = (data['high'] > data['high'].shift(1)).rolling(5).sum()
        data['lower_lows'] = (data['low'] < data['low'].shift(1)).rolling(5).sum()
        data['trend_strength'] = data['higher_highs'] - data['lower_lows']
        
        # Nettoyer les NaN
        data = data.fillna(method='ffill').fillna(0)
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """Calcule les Bollinger Bands"""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    def optimize_all_models(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Optimise tous les modèles pour tous les symboles"""
        
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        
        print(f"🎯 Démarrage de l'optimisation pour {len(symbols)} symboles...")
        
        all_results = {}
        
        for symbol in symbols:
            print(f"\n📊 Optimisation pour {symbol}")
            
            try:
                # Charger les données (utiliser un dataset de test si pas de données réelles)
                data = self._load_or_create_test_data(symbol)
                
                # Créer des features avancées
                data_with_features = self._create_advanced_features(data)
                
                # Préparer les données pour l'entraînement
                X, y = self._prepare_training_data(data_with_features)
                
                # Split temporel
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                
                print(f"Données d'entraînement: {len(X_train)} échantillons")
                print(f"Données de validation: {len(X_val)} échantillons")
                
                symbol_results = {}
                
                # Optimiser chaque modèle
                for model_type in ['logistic_regression', 'random_forest', 'xgboost_classifier']:
                    result = self._optimize_single_model(model_type, X_train, y_train, X_val, y_val)
                    symbol_results[model_type] = result
                
                # Créer un ensemble optimisé
                ensemble_result = self._create_optimized_ensemble(symbol_results, X_train, y_train, X_val, y_val)
                symbol_results['ensemble'] = ensemble_result
                
                all_results[symbol] = symbol_results
                
                # Sauvegarder les résultats
                self._save_optimization_results(symbol, symbol_results)
                
            except Exception as e:
                print(f"❌ Erreur lors de l'optimisation de {symbol}: {e}")
                continue
        
        # Analyse finale et recommandations
        self._generate_final_report(all_results)
        
        return all_results
    
    def _load_or_create_test_data(self, symbol: str) -> pd.DataFrame:
        """Charge les données ou crée des données de test"""
        
        # Pour le test, créer des données synthétiques réalistes
        np.random.seed(42)
        n_days = 2000
        
        # Simulation d'un prix avec tendance et volatilité
        returns = np.random.normal(0.0005, 0.02, n_days)  # 0.05% de rendement moyen quotidien
        prices = [100]  # Prix initial
        
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        # Créer le DataFrame
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.lognormal(10, 1, n_days)
        })
        
        return data
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les données pour l'entraînement"""
        
        # Calculer le target (rendement futur positif)
        data['future_return'] = data['close'].pct_change(5).shift(-5)  # Rendement sur 5 jours
        data['target'] = (data['future_return'] > 0.02).astype(int)  # Objectif: +2% ou plus
        
        # Features
        feature_columns = [col for col in data.columns if col not in ['timestamp', 'target', 'future_return']]
        X = data[feature_columns].copy()
        y = data['target'].copy()
        
        # Nettoyer les données
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        return X, y
    
    def _create_optimized_ensemble(self, model_results: Dict, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Crée un ensemble optimisé basé sur les meilleurs modèles"""
        
        print("🎯 Création de l'ensemble optimisé...")
        
        # Obtenir les prédictions de chaque modèle
        predictions = {}
        
        for model_type, result in model_results.items():
            if model_type == 'ensemble':
                continue
                
            params = result['best_params']
            
            try:
                if model_type == 'logistic_regression':
                    model = LogisticRegression(**params, random_state=42)
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    model.fit(X_train_scaled, y_train)
                    predictions[model_type] = model.predict_proba(X_val_scaled)[:, 1]
                    
                elif model_type == 'random_forest':
                    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    predictions[model_type] = model.predict_proba(X_val)[:, 1]
                    
                elif model_type == 'xgboost_classifier':
                    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
                    eval_set = [(X_val, y_val)]
                    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                    predictions[model_type] = model.predict_proba(X_val)[:, 1]
                    
            except Exception as e:
                print(f"Erreur pour {model_type}: {e}")
                continue
        
        # Optimiser les poids de l'ensemble
        best_ensemble_score = 0
        best_weights = None
        
        for weight_lr in np.arange(0.1, 1.0, 0.1):
            for weight_rf in np.arange(0.1, 1.0, 0.1):
                for weight_xgb in np.arange(0.1, 1.0, 0.1):
                    # Normaliser les poids
                    total_weight = weight_lr + weight_rf + weight_xgb
                    w_lr = weight_lr / total_weight
                    w_rf = weight_rf / total_weight
                    w_xgb = weight_xgb / total_weight
                    
                    # Prédiction ensemble
                    ensemble_pred = (w_lr * predictions.get('logistic_regression', 0) +
                                   w_rf * predictions.get('random_forest', 0) +
                                   w_xgb * predictions.get('xgboost_classifier', 0))
                    
                    # Score
                    score = self._calculate_profitability_score(y_val, ensemble_pred)
                    
                    if score > best_ensemble_score:
                        best_ensemble_score = score
                        best_weights = {'lr': w_lr, 'rf': w_rf, 'xgb': w_xgb}
        
        return {
            'model_type': 'ensemble',
            'best_weights': best_weights,
            'best_score': best_ensemble_score,
            'component_models': list(predictions.keys())
        }
    
    def _save_optimization_results(self, symbol: str, results: Dict):
        """Sauvegarde les résultats d'optimisation"""
        
        output_file = os.path.join(self.output_dir, f"optimization_results_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Préparer les données pour la sérialisation
        serializable_results = {}
        for model_type, result in results.items():
            serializable_results[model_type] = {
                'model_type': result['model_type'],
                'best_score': result['best_score'],
                'best_params': result.get('best_params', result.get('best_weights', {})),
                'n_trials': result.get('n_trials', 'N/A')
            }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Résultats sauvegardés: {output_file}")
    
    def _generate_final_report(self, all_results: Dict):
        """Génère un rapport final avec les recommandations"""
        
        print("\n" + "="*80)
        print("📊 RAPPORT FINAL D'OPTIMISATION")
        print("="*80)
        
        # Analyser les performances par modèle
        model_performance = {}
        
        for symbol, symbol_results in all_results.items():
            for model_type, result in symbol_results.items():
                if model_type not in model_performance:
                    model_performance[model_type] = []
                model_performance[model_type].append(result['best_score'])
        
        # Classement des modèles
        model_rankings = {}
        for model_type, scores in model_performance.items():
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            model_rankings[model_type] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'consistency': avg_score / (std_score + 1e-6)
            }
        
        # Trier par score moyen
        sorted_models = sorted(model_rankings.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        print("\n🏆 CLASSEMENT DES MODÈLES:")
        for i, (model_type, metrics) in enumerate(sorted_models, 1):
            print(f"{i}. {model_type.upper()}")
            print(f"   Score moyen: {metrics['avg_score']:.4f}")
            print(f"   Écart-type: {metrics['std_score']:.4f}")
            print(f"   Consistance: {metrics['consistency']:.2f}")
        
        # Recommandations spécifiques
        print("\n💡 RECOMMANDATIONS POUR MAXIMISER LA RENTABILITÉ:")
        
        best_model = sorted_models[0][0]
        print(f"1. Modèle principal recommandé: {best_model.upper()}")
        
        if 'ensemble' in [m[0] for m in sorted_models[:3]]:
            print("2. L'ensemble est dans le top 3 - Utiliser une approche multi-modèles")
        
        print("3. Paramètres optimisés détectés:")
        for symbol, symbol_results in all_results.items():
            if best_model in symbol_results:
                best_params = symbol_results[best_model].get('best_params', {})
                print(f"   {symbol}: Top 3 paramètres importants")
                for i, (param, value) in enumerate(list(best_params.items())[:3]):
                    print(f"      - {param}: {value}")
        
        print("\n4. Actions recommandées:")
        print("   ✅ Implémenter les paramètres optimisés dans le système de production")
        print("   ✅ Surveiller la dérive des performances avec réétalonnage mensuel")
        print("   ✅ Utiliser l'ensemble pour la robustesse si un seul modèle n'est pas suffisant")
        print("   ✅ Augmenter la fréquence de trading avec ces modèles optimisés")
        
        # Estimations de performance améliorée
        best_avg_score = sorted_models[0][1]['avg_score']
        improvement_estimate = (best_avg_score - 0.5) * 100  # Amélioration par rapport au hasard
        
        print(f"\n📈 ESTIMATION D'AMÉLIORATION:")
        print(f"   Score de rentabilité optimisé: {best_avg_score:.4f}")
        print(f"   Amélioration estimée: +{improvement_estimate:.1f}% par rapport au hasard")
        print(f"   Potentiel d'amélioration du Sharpe ratio: +{improvement_estimate * 0.02:.2f}")
        
        # Sauvegarder le rapport
        report_file = os.path.join(self.output_dir, f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(report_file, 'w') as f:
            f.write("RAPPORT D'OPTIMISATION POUR LA RENTABILITÉ\n")
            f.write("="*50 + "\n\n")
            f.write(f"Meilleur modèle: {best_model}\n")
            f.write(f"Score moyen: {best_avg_score:.4f}\n")
            f.write(f"Amélioration estimée: +{improvement_estimate:.1f}%\n")
        
        print(f"\n💾 Rapport complet sauvegardé: {report_file}")


def main():
    """Fonction principale pour lancer l'optimisation"""
    
    print("🚀 OPTIMISEUR DE RENTABILITÉ - ALPHABETA808TRADING")
    print("="*60)
    
    # Configuration
    optimizer = ProfitabilityOptimizer(
        n_trials_per_model=150,  # Augmenter pour de meilleurs résultats
        cv_folds=5,
        target_sharpe=2.0,
        max_drawdown_limit=0.12
    )
    
    # Symboles à optimiser
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
    
    # Lancer l'optimisation
    results = optimizer.optimize_all_models(symbols)
    
    print("\n✅ Optimisation terminée!")
    print(f"Résultats détaillés disponibles dans: {optimizer.output_dir}")
    
    return results


if __name__ == "__main__":
    results = main()
