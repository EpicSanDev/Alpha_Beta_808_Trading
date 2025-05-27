#!/usr/bin/env python3
"""
Comprehensive Backtesting System for AlphaBeta808Trading
Système de backtests complet avec validation multi-modèles et métriques avancées
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
from pathlib import Path
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Suppression des warnings
warnings.filterwarnings("ignore")

# Ajout du répertoire src au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Imports des modules existants
from src.acquisition.connectors import load_binance_klines, generate_random_market_data
from src.feature_engineering.technical_features import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands,
    calculate_price_momentum, calculate_volume_features
)
from src.modeling.models import prepare_data_for_model, train_model, load_model_and_predict
from src.signal_generation.signal_generator import generate_signals_from_predictions, allocate_capital_simple
from src.execution.simulator import BacktestSimulator
from src.core.performance_analyzer import BacktestAnalyzer
from src.validation.walk_forward import WalkForwardValidator
from src.acquisition.preprocessing_utils import handle_missing_values_column
from src.acquisition.preprocessing import normalize_min_max

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveBacktester:
    """
    Système de backtests complet avec support multi-modèles, 
    validation temporelle et métriques financières avancées
    """
    
    def __init__(self, config_file: str = "config/backtest_config.json"):
        """
        Initialise le système de backtests
        
        Args:
            config_file: Fichier de configuration des paramètres de backtest
        """
        self.config = self._load_config(config_file)
        self.results_dir = self.config.get('results_dir', 'backtest_results')
        self.models_dir = self.config.get('models_dir', 'models_store')
        
        # Créer les répertoires nécessaires
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs("config", exist_ok=True)
        
        # Initialiser les composants
        self.analyzer = BacktestAnalyzer(self.results_dir)
        self.validator = WalkForwardValidator()
        
        # Métriques de trading financières
        self.trading_metrics = {}
        
        logger.info(f"ComprehensiveBacktester initialisé avec config: {config_file}")
    
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration depuis un fichier JSON"""
        
        default_config = {
            "data_settings": {
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                "intervals": ["1d", "4h"],
                "lookback_days": 1095,  # 3 ans
                "use_random_data": False
            },
            "model_settings": {
                "models_to_test": [
                    "logistic_regression",
                    "random_forest", 
                    "xgboost_classifier",
                    "elastic_net"
                ],
                "ensemble_mode": True,
                "cross_validation_folds": 5,
                "walk_forward_validation": True
            },
            "strategy_settings": {
                "initial_capital": 100000,
                "risk_per_trade": 0.02,
                "max_positions": 5,
                "stop_loss": 0.05,
                "take_profit": 0.10,
                "rebalance_frequency": "daily"
            },
            "backtest_settings": {
                "start_date": "2022-01-01",
                "end_date": "2024-12-31",
                "commission": 0.001,
                "slippage": 0.0005,
                "benchmark": "buy_and_hold"
            },
            "performance_settings": {
                "metrics": [
                    "total_return", "annualized_return", "volatility",
                    "sharpe_ratio", "information_ratio", "max_drawdown",
                    "calmar_ratio", "sortino_ratio", "win_rate"
                ],
                "risk_free_rate": 0.02
            },
            "results_dir": "backtest_results",
            "models_dir": "models_store"
        }
        
        try:
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                # Merger avec la config par défaut
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Fichier de config {config_file} non trouvé. Utilisation de la config par défaut.")
            # Créer le fichier de config par défaut
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def run_comprehensive_backtest(self, test_name: str = None) -> Dict[str, Any]:
        """
        Lance un backtest complet avec tous les modèles et configurations
        
        Args:
            test_name: Nom optionnel pour ce test
            
        Returns:
            Dict avec tous les résultats du backtest
        """
        test_name = test_name or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🚀 Démarrage du backtest complet: {test_name}")
        
        # 1. Acquisition et préparation des données
        logger.info("📊 Acquisition des données...")
        market_data = self._prepare_market_data()
        
        if market_data.empty:
            logger.error("❌ Aucune donnée de marché disponible")
            return {}
        
        # 2. Feature Engineering
        logger.info("🔧 Feature Engineering...")
        features_data = self._engineer_features(market_data)
        
        # 3. Tests multi-modèles
        logger.info("🤖 Tests multi-modèles...")
        model_results = self._run_multi_model_tests(features_data, test_name)
        
        # 4. Validation Walk-Forward
        if self.config['model_settings']['walk_forward_validation']:
            logger.info("📈 Validation Walk-Forward...")
            walk_forward_results = self._run_walk_forward_validation(features_data)
            model_results['walk_forward'] = walk_forward_results
        
        # 5. Tests d'ensemble
        if self.config['model_settings']['ensemble_mode']:
            logger.info("🎯 Tests d'ensemble...")
            ensemble_results = self._run_ensemble_tests(features_data, test_name)
            model_results['ensemble'] = ensemble_results
        
        # 6. Analyse comparative
        logger.info("📊 Analyse comparative...")
        comparative_analysis = self._run_comparative_analysis(model_results, features_data)
        
        # 7. Génération du rapport final
        logger.info("📝 Génération du rapport final...")
        final_report = self._generate_comprehensive_report(
            model_results, comparative_analysis, test_name
        )
        
        logger.info(f"✅ Backtest complet terminé: {test_name}")
        return final_report
    
    def _prepare_market_data(self) -> pd.DataFrame:
        """Prépare les données de marché selon la configuration"""
        
        symbols = self.config['data_settings']['symbols']
        intervals = self.config['data_settings']['intervals']
        lookback_days = self.config['data_settings']['lookback_days']
        use_random = self.config['data_settings']['use_random_data']
        
        all_data = []
        
        for symbol in symbols:
            logger.info(f"📈 Chargement des données pour {symbol}...")
            
            if use_random:
                # Utiliser des données aléatoires pour les tests
                symbol_data = generate_random_market_data(
                    num_rows=lookback_days, 
                    start_price=100.0, 
                    volatility=0.02, 
                    freq='D'
                )
                symbol_data['symbol'] = symbol
                symbol_data['interval'] = '1d'
                all_data.append(symbol_data)
            else:
                try:
                    # Charger les données réelles depuis Binance
                    from dotenv import load_dotenv
                    load_dotenv()
                    
                    api_key = os.getenv('BINANCE_API_KEY')
                    api_secret = os.getenv('BINANCE_API_SECRET')
                    
                    if not api_key or not api_secret:
                        logger.warning(f"Clés API manquantes pour {symbol}, utilisation de données aléatoires")
                        symbol_data = generate_random_market_data(
                            num_rows=lookback_days, 
                            start_price=100.0, 
                            volatility=0.02, 
                            freq='D'
                        )
                        symbol_data['symbol'] = symbol
                        symbol_data['interval'] = '1d'
                        all_data.append(symbol_data)
                        continue
                    
                    # Calculer la date de début
                    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
                    
                    # Charger les données pour tous les intervalles
                    symbol_data_dict = load_binance_klines(
                        api_key=api_key,
                        api_secret=api_secret,
                        symbol=symbol,
                        intervals=intervals,
                        start_date_str=start_date
                    )
                    
                    if symbol_data_dict:
                        for interval, data in symbol_data_dict.items():
                            if not data.empty:
                                all_data.append(data)
                    else:
                        logger.warning(f"Impossible de charger {symbol}, utilisation de données aléatoires")
                        symbol_data = generate_random_market_data(
                            num_rows=lookback_days, 
                            start_price=100.0, 
                            volatility=0.02, 
                            freq='D'
                        )
                        symbol_data['symbol'] = symbol
                        symbol_data['interval'] = '1d'
                        all_data.append(symbol_data)
                        
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de {symbol}: {e}")
                    # Fallback sur données aléatoires
                    symbol_data = generate_random_market_data(
                        num_rows=lookback_days, 
                        start_price=100.0, 
                        volatility=0.02, 
                        freq='D'
                    )
                    symbol_data['symbol'] = symbol
                    symbol_data['interval'] = '1d'
                    all_data.append(symbol_data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Données combinées: {len(combined_data)} lignes, {len(symbols)} symboles")
            return combined_data
        else:
            logger.error("❌ Aucune donnée chargée")
            return pd.DataFrame()
    
    def _engineer_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Applique le feature engineering sur les données de marché"""
        
        if market_data.empty:
            return market_data
        
        processed_data = []
        
        # Traiter chaque symbole/intervalle séparément
        for (symbol, interval), group in market_data.groupby(['symbol', 'interval']):
            logger.info(f"🔧 Feature engineering pour {symbol} ({interval})...")
            
            group = group.copy().sort_values('timestamp')
            
            # Gestion des valeurs manquantes
            if 'close' in group.columns:
                group = handle_missing_values_column(group, column='close', strategy='ffill')
                
                # Normalisation
                group = normalize_min_max(group, column='close')
                
                # Features techniques de base
                group = calculate_sma(group, column='close', windows=[10, 20, 50])
                group = calculate_ema(group, column='close', windows=[10, 20, 50])
                group = calculate_rsi(group, column='close', window=14)
                
                # Features avancées
                group = calculate_macd(group, column='close')
                group = calculate_bollinger_bands(group, column='close')
                group = calculate_price_momentum(group, column='close', windows=[5, 10, 20])
                
                # Features de volume si disponible
                if 'volume' in group.columns:
                    group = calculate_volume_features(
                        group, volume_col='volume', price_col='close', windows=[10, 20]
                    )
                
                # Supprimer les NaN
                group = group.dropna()
                
                if not group.empty:
                    processed_data.append(group)
        
        if processed_data:
            final_data = pd.concat(processed_data, ignore_index=True)
            logger.info(f"✅ Feature engineering terminé: {len(final_data)} lignes avec features")
            return final_data
        else:
            logger.error("❌ Aucune donnée après feature engineering")
            return pd.DataFrame()
    
    def _run_multi_model_tests(self, features_data: pd.DataFrame, test_name: str) -> Dict[str, Any]:
        """Lance les tests sur tous les modèles configurés"""
        
        models_to_test = self.config['model_settings']['models_to_test']
        results = {}
        
        # Préparer les données pour chaque symbole
        for symbol in features_data['symbol'].unique():
            symbol_data = features_data[features_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 100:  # Minimum de données requis
                logger.warning(f"Pas assez de données pour {symbol} ({len(symbol_data)} lignes)")
                continue
            
            logger.info(f"🤖 Tests multi-modèles pour {symbol}...")
            symbol_results = {}
            
            # Préparer X et y
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval']
            feature_cols = [col for col in symbol_data.columns if col not in exclude_cols]
            
            X, y = prepare_data_for_model(
                symbol_data,
                target_shift_days=1,
                feature_columns=feature_cols,
                price_change_threshold=0.01
            )
            
            if len(X) < 50:
                logger.warning(f"Pas assez de données préparées pour {symbol}")
                continue
            
            # Split temporel
            split_ratio = 0.8
            split_index = int(len(X) * split_ratio)
            
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Tester chaque modèle
            for model_type in models_to_test:
                logger.info(f"  📊 Test du modèle {model_type}...")
                
                try:
                    # Entraîner le modèle
                    model_path = f"{self.models_dir}/{model_type}_{symbol}_{test_name}.joblib"
                    
                    metrics = train_model(
                        X_train, y_train,
                        model_type=model_type,
                        model_path=model_path,
                        scale_features=True
                    )
                    
                    # Prédictions sur le test set
                    predictions = load_model_and_predict(
                        X_test, model_path=model_path, return_probabilities=True
                    )
                    
                    # Générer les signaux
                    signals = self._generate_trading_signals(predictions)
                    
                    # Simulation de trading
                    backtest_results = self._run_trading_simulation(
                        symbol_data.iloc[split_index:], signals, symbol, model_type
                    )
                    
                    symbol_results[model_type] = {
                        'training_metrics': metrics,
                        'predictions': predictions,
                        'signals': signals,
                        'backtest_results': backtest_results
                    }
                    
                    logger.info(f"  ✅ {model_type} terminé pour {symbol}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Erreur avec {model_type} pour {symbol}: {e}")
                    continue
            
            results[symbol] = symbol_results
        
        return results
    
    def _generate_trading_signals(self, predictions: np.ndarray) -> pd.Series:
        """Génère les signaux de trading à partir des prédictions"""
        
        # Gérer les valeurs NaN dans les prédictions
        if np.isnan(predictions).all():
            logger.warning("Toutes les prédictions sont NaN, génération de signaux HOLD par défaut")
            return pd.Series(['HOLD'] * len(predictions))
        
        # Filtrer les valeurs NaN pour le calcul des percentiles
        valid_predictions = predictions[~np.isnan(predictions)]
        if len(valid_predictions) == 0:
            logger.warning("Aucune prédiction valide, génération de signaux HOLD par défaut")
            return pd.Series(['HOLD'] * len(predictions))
        
        signals = pd.Series(['HOLD'] * len(predictions))
        
        # Seuils adaptatifs basés sur les percentiles des valeurs valides
        upper_threshold = np.percentile(valid_predictions, 75)  # Top 25% -> BUY
        lower_threshold = np.percentile(valid_predictions, 25)  # Bottom 25% -> SELL
        
        # Appliquer les seuils seulement aux prédictions valides
        valid_mask = ~np.isnan(predictions)
        signals[valid_mask & (predictions >= upper_threshold)] = 'BUY'
        signals[valid_mask & (predictions <= lower_threshold)] = 'SELL'
        
        return signals
    
    def _run_trading_simulation(self, market_data: pd.DataFrame, signals: pd.Series, 
                              symbol: str, model_type: str) -> Dict[str, Any]:
        """Lance une simulation de trading pour un symbole et modèle donnés"""
        
        try:
            # Préparer les données pour le simulateur
            market_data_sim = market_data.copy()
            market_data_sim = market_data_sim.sort_values('timestamp').set_index('timestamp')
            
            # Créer le DataFrame des signaux
            timestamps = market_data_sim.index
            signals_trimmed = signals[:len(timestamps)]
            
            # Remplacer les valeurs NaN dans les signaux par 'HOLD'
            signals_cleaned = []
            for sig in signals_trimmed:
                if pd.isna(sig) or sig == 'nan' or str(sig).lower() == 'nan':
                    signals_cleaned.append('HOLD')
                else:
                    signals_cleaned.append(sig)
            
            signals_df = pd.DataFrame({
                'signal': signals_cleaned,
                'position_to_allocate': [
                    self.config['strategy_settings']['initial_capital'] * 
                    self.config['strategy_settings']['risk_per_trade'] 
                    if sig in ['BUY', 'SELL'] else 0 
                    for sig in signals_cleaned
                ]
            }, index=timestamps)
            
            # Lancer la simulation
            simulator = BacktestSimulator(
                initial_capital=self.config['strategy_settings']['initial_capital'],
                market_data=market_data_sim
            )
            
            simulator.run_simulation(signals_df)
            
            # Récupérer les résultats
            portfolio_history = simulator.get_portfolio_history()
            trade_history = simulator.get_trades_history()
            
            # Calculer les métriques de performance
            performance_metrics = self._calculate_performance_metrics(
                portfolio_history, trade_history, market_data_sim
            )
            
            return {
                'portfolio_history': portfolio_history,
                'trade_history': trade_history,
                'performance_metrics': performance_metrics,
                'symbol': symbol,
                'model_type': model_type
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la simulation pour {symbol} - {model_type}: {e}")
            return {}
    
    def _calculate_performance_metrics(self, portfolio_history, 
                                     trade_history, 
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """Calcule des métriques de performance avancées"""
        
        # Convertir en DataFrame si nécessaire
        if isinstance(portfolio_history, list):
            if not portfolio_history:
                return {}
            portfolio_df = pd.DataFrame(portfolio_history)
        elif isinstance(portfolio_history, pd.DataFrame):
            if portfolio_history.empty:
                return {}
            portfolio_df = portfolio_history.copy()
        else:
            return {}
        
        # Vérifier que les colonnes nécessaires existent
        if 'portfolio_value' not in portfolio_df.columns:
            logger.warning("Colonne 'portfolio_value' manquante dans l'historique du portefeuille")
            return {}
        
        initial_capital = self.config['strategy_settings']['initial_capital']
        risk_free_rate = self.config['performance_settings']['risk_free_rate']
        
        # Métriques de base
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Rendements quotidiens
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_returns = portfolio_df['daily_return'].dropna()
        
        # Métriques de risque
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / (volatility) if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (daily_returns.mean() * 252 - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum Drawdown
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Calmar Ratio
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Information Ratio (vs benchmark)
        benchmark_return = self._calculate_benchmark_return(market_data)
        excess_returns = daily_returns.mean() * 252 - benchmark_return
        tracking_error = daily_returns.std() * np.sqrt(252)
        information_ratio = excess_returns / tracking_error if tracking_error > 0 else 0
        
        # Métriques de trading
        win_rate = 0
        total_trades = 0
        
        if trade_history is not None:
            # Convertir en DataFrame si nécessaire
            if isinstance(trade_history, list):
                if trade_history:
                    trades_df = pd.DataFrame(trade_history)
                else:
                    trades_df = pd.DataFrame()
            elif isinstance(trade_history, pd.DataFrame):
                trades_df = trade_history.copy()
            else:
                trades_df = pd.DataFrame()
            
            if not trades_df.empty and 'type' in trades_df.columns:
                total_trades = len(trades_df)
                buy_trades = trades_df[trades_df['type'] == 'BUY']
                sell_trades = trades_df[trades_df['type'] == 'SELL']
                
                # Win rate calculation based on actual trade outcomes
                if len(sell_trades) > 0 and len(buy_trades) > 0:
                    # Calculate actual win rate by comparing buy/sell pairs
                    winning_trades = 0
                    total_trade_pairs = 0
                    
                    # Simple approach: compare consecutive buy-sell pairs
                    for i in range(min(len(buy_trades), len(sell_trades))):
                        if 'price' in buy_trades.columns and 'price' in sell_trades.columns:
                            buy_price = buy_trades.iloc[i]['price']
                            sell_price = sell_trades.iloc[i]['price']
                            if sell_price > buy_price:
                                winning_trades += 1
                            total_trade_pairs += 1
                    
                    win_rate = winning_trades / total_trade_pairs if total_trade_pairs > 0 else 0
                else:
                    # Alternative: calculate based on daily returns
                    if len(daily_returns) > 0:
                        positive_returns = daily_returns[daily_returns > 0]
                        win_rate = len(positive_returns) / len(daily_returns)
                    else:
                        win_rate = 0
        
        return {
            'total_return': total_return,
            'annualized_return': daily_returns.mean() * 252,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_value': final_value
        }
    
    def _calculate_benchmark_return(self, market_data: pd.DataFrame) -> float:
        """Calcule le rendement du benchmark (Buy & Hold)"""
        
        if market_data.empty or 'close' not in market_data.columns:
            return 0.0
        
        initial_price = market_data['close'].iloc[0]
        final_price = market_data['close'].iloc[-1]
        
        return (final_price - initial_price) / initial_price
    
    def _run_walk_forward_validation(self, features_data: pd.DataFrame) -> Dict[str, Any]:
        """Lance la validation walk-forward"""
        
        logger.info("📈 Exécution de la validation walk-forward...")
        
        results = {}
        
        for symbol in features_data['symbol'].unique():
            symbol_data = features_data[features_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 200:  # Minimum pour walk-forward
                continue
            
            logger.info(f"  📊 Walk-forward pour {symbol}...")
            
            # Préparer les données
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval']
            feature_cols = [col for col in symbol_data.columns if col not in exclude_cols]
            
            X, y = prepare_data_for_model(
                symbol_data,
                target_shift_days=1,
                feature_columns=feature_cols,
                price_change_threshold=0.01
            )
            
            # Configuration walk-forward
            try:
                wf_results = self.validator.validate(
                    X, y,
                    model_type='logistic_regression',
                    train_window_size=100,
                    test_window_size=20,
                    step_size=10
                )
                
                results[symbol] = {
                    'mean_score': wf_results.mean_score,
                    'std_score': wf_results.std_score,
                    'scores': wf_results.scores,
                    'num_windows': len(wf_results.scores)
                }
                
                logger.info(f"  ✅ Walk-forward {symbol}: Score moyen = {wf_results.mean_score:.3f}")
                
            except Exception as e:
                logger.error(f"  ❌ Erreur walk-forward pour {symbol}: {e}")
                continue
        
        return results
    
    def _run_ensemble_tests(self, features_data: pd.DataFrame, test_name: str) -> Dict[str, Any]:
        """Lance les tests d'ensemble de modèles"""
        
        logger.info("🎯 Exécution des tests d'ensemble...")
        
        ensemble_results = {}
        models_to_ensemble = self.config['model_settings']['models_to_test']
        
        for symbol in features_data['symbol'].unique():
            symbol_data = features_data[features_data['symbol'] == symbol].copy()
            
            if len(symbol_data) < 100:
                continue
            
            logger.info(f"  🎯 Ensemble pour {symbol}...")
            
            # Préparer les données
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval']
            feature_cols = [col for col in symbol_data.columns if col not in exclude_cols]
            
            X, y = prepare_data_for_model(
                symbol_data,
                target_shift_days=1,
                feature_columns=feature_cols,
                price_change_threshold=0.01
            )
            
            # Split
            split_ratio = 0.8
            split_index = int(len(X) * split_ratio)
            
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
            
            # Entraîner plusieurs modèles
            model_predictions = {}
            
            for model_type in models_to_ensemble:
                try:
                    model_path = f"{self.models_dir}/ensemble_{model_type}_{symbol}_{test_name}.joblib"
                    
                    # Entraîner
                    train_model(
                        X_train, y_train,
                        model_type=model_type,
                        model_path=model_path,
                        scale_features=True
                    )
                    
                    # Prédire
                    predictions = load_model_and_predict(
                        X_test, model_path=model_path, return_probabilities=True
                    )
                    
                    model_predictions[model_type] = predictions
                    
                except Exception as e:
                    logger.error(f"    ❌ Erreur avec {model_type}: {e}")
                    continue
            
            if len(model_predictions) >= 2:
                # Créer l'ensemble (moyenne pondérée)
                weights = {model: 1.0 / len(model_predictions) for model in model_predictions}
                
                ensemble_pred = np.zeros(len(X_test))
                for model, pred in model_predictions.items():
                    ensemble_pred += weights[model] * pred
                
                # Tester l'ensemble
                ensemble_signals = self._generate_trading_signals(ensemble_pred)
                ensemble_backtest = self._run_trading_simulation(
                    symbol_data.iloc[split_index:], ensemble_signals, symbol, "ensemble"
                )
                
                ensemble_results[symbol] = {
                    'individual_predictions': model_predictions,
                    'ensemble_predictions': ensemble_pred,
                    'weights': weights,
                    'backtest_results': ensemble_backtest
                }
                
                logger.info(f"  ✅ Ensemble {symbol} créé avec {len(model_predictions)} modèles")
        
        return ensemble_results
    
    def _run_comparative_analysis(self, model_results: Dict, features_data: pd.DataFrame) -> Dict[str, Any]:
        """Lance l'analyse comparative entre modèles et stratégies"""
        
        logger.info("📊 Analyse comparative...")
        
        comparative_results = {
            'model_comparison': {},
            'strategy_comparison': {},
            'risk_analysis': {}
        }
        
        # Comparaison des modèles
        for symbol in model_results:
            if symbol in ['walk_forward', 'ensemble']:
                continue
                
            symbol_models = model_results[symbol]
            model_comparison = {}
            
            for model_type, results in symbol_models.items():
                if 'backtest_results' in results and results['backtest_results']:
                    perf_metrics = results['backtest_results'].get('performance_metrics', {})
                    model_comparison[model_type] = perf_metrics
            
            if model_comparison:
                # Trouver le meilleur modèle selon différents critères
                best_sharpe = max(model_comparison.items(), 
                                key=lambda x: x[1].get('sharpe_ratio', -999), default=(None, {}))[0]
                best_return = max(model_comparison.items(), 
                                key=lambda x: x[1].get('total_return', -999), default=(None, {}))[0]
                
                comparative_results['model_comparison'][symbol] = {
                    'all_models': model_comparison,
                    'best_sharpe': best_sharpe,
                    'best_return': best_return,
                    'model_count': len(model_comparison)
                }
        
        # Calculs de risque agrégés
        all_returns = []
        all_sharpe = []
        all_drawdowns = []
        
        for symbol_data in comparative_results['model_comparison'].values():
            for model_perf in symbol_data['all_models'].values():
                if model_perf:
                    all_returns.append(model_perf.get('total_return', 0))
                    all_sharpe.append(model_perf.get('sharpe_ratio', 0))
                    all_drawdowns.append(model_perf.get('max_drawdown', 0))
        
        if all_returns:
            comparative_results['risk_analysis'] = {
                'avg_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'avg_sharpe': np.mean(all_sharpe),
                'avg_drawdown': np.mean(all_drawdowns),
                'best_overall_return': max(all_returns),
                'worst_drawdown': min(all_drawdowns)
            }
        
        return comparative_results
    
    def _generate_comprehensive_report(self, model_results: Dict, 
                                     comparative_analysis: Dict, 
                                     test_name: str) -> Dict[str, Any]:
        """Génère un rapport complet des résultats"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rapport consolidé
        final_report = {
            'test_info': {
                'test_name': test_name,
                'timestamp': timestamp,
                'config': self.config
            },
            'data_summary': {
                'symbols_tested': list(model_results.keys()),
                'models_tested': self.config['model_settings']['models_to_test'],
                'total_tests': sum(len(v) for k, v in model_results.items() if k not in ['walk_forward', 'ensemble'])
            },
            'model_results': model_results,
            'comparative_analysis': comparative_analysis,
            'recommendations': self._generate_recommendations(comparative_analysis)
        }
        
        # Sauvegarder le rapport
        report_file = f"{self.results_dir}/comprehensive_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Générer un résumé lisible
        summary_file = f"{self.results_dir}/summary_report_{timestamp}.txt"
        self._generate_summary_report(final_report, summary_file)
        
        logger.info(f"📝 Rapport complet sauvegardé: {report_file}")
        logger.info(f"📄 Résumé sauvegardé: {summary_file}")
        
        return final_report
    
    def _generate_recommendations(self, comparative_analysis: Dict) -> Dict[str, str]:
        """Génère des recommandations basées sur l'analyse"""
        
        recommendations = {}
        
        # Analyse des modèles
        model_comp = comparative_analysis.get('model_comparison', {})
        if model_comp:
            # Compter les modèles gagnants
            sharpe_winners = {}
            return_winners = {}
            
            for symbol, data in model_comp.items():
                best_sharpe = data.get('best_sharpe')
                best_return = data.get('best_return')
                
                if best_sharpe:
                    sharpe_winners[best_sharpe] = sharpe_winners.get(best_sharpe, 0) + 1
                if best_return:
                    return_winners[best_return] = return_winners.get(best_return, 0) + 1
            
            if sharpe_winners:
                best_sharpe_model = max(sharpe_winners.items(), key=lambda x: x[1])[0]
                recommendations['best_sharpe_model'] = f"Modèle recommandé pour le Sharpe ratio: {best_sharpe_model}"
            
            if return_winners:
                best_return_model = max(return_winners.items(), key=lambda x: x[1])[0]
                recommendations['best_return_model'] = f"Modèle recommandé pour les rendements: {best_return_model}"
        
        # Analyse des risques
        risk_analysis = comparative_analysis.get('risk_analysis', {})
        if risk_analysis:
            avg_sharpe = risk_analysis.get('avg_sharpe', 0)
            avg_drawdown = risk_analysis.get('avg_drawdown', 0)
            
            if avg_sharpe > 1.0:
                recommendations['sharpe_assessment'] = "Excellents ratios de Sharpe observés - stratégie viable"
            elif avg_sharpe > 0.5:
                recommendations['sharpe_assessment'] = "Bons ratios de Sharpe - stratégie prometteuse"
            else:
                recommendations['sharpe_assessment'] = "Ratios de Sharpe faibles - révision stratégique recommandée"
            
            if abs(avg_drawdown) > 0.2:
                recommendations['risk_warning'] = "Drawdowns élevés détectés - gestion des risques à renforcer"
        
        return recommendations
    
    def _generate_summary_report(self, final_report: Dict, summary_file: str):
        """Génère un rapport de synthèse lisible"""
        
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RAPPORT DE BACKTEST COMPLET - ALPHABETA808TRADING\n")
            f.write("=" * 80 + "\n\n")
            
            # Informations du test
            test_info = final_report['test_info']
            f.write(f"Test: {test_info['test_name']}\n")
            f.write(f"Date: {test_info['timestamp']}\n")
            f.write(f"Configuration: {len(test_info['config'])} paramètres\n\n")
            
            # Résumé des données
            data_summary = final_report['data_summary']
            f.write("DONNÉES TESTÉES:\n")
            f.write(f"- Symboles: {', '.join(data_summary['symbols_tested'])}\n")
            f.write(f"- Modèles: {', '.join(data_summary['models_tested'])}\n")
            f.write(f"- Total tests: {data_summary['total_tests']}\n\n")
            
            # Analyse comparative
            comp_analysis = final_report['comparative_analysis']
            if 'risk_analysis' in comp_analysis:
                risk = comp_analysis['risk_analysis']
                f.write("ANALYSE DES RISQUES:\n")
                f.write(f"- Rendement moyen: {risk.get('avg_return', 0):.2%}\n")
                f.write(f"- Sharpe ratio moyen: {risk.get('avg_sharpe', 0):.3f}\n")
                f.write(f"- Drawdown moyen: {risk.get('avg_drawdown', 0):.2%}\n")
                f.write(f"- Meilleur rendement: {risk.get('best_overall_return', 0):.2%}\n\n")
            
            # Recommandations
            recommendations = final_report['recommendations']
            if recommendations:
                f.write("RECOMMANDATIONS:\n")
                for key, rec in recommendations.items():
                    f.write(f"- {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")


def main():
    """Fonction principale pour lancer un backtest complet"""
    
    # Créer et configurer le backtester
    backtester = ComprehensiveBacktester("config/backtest_config.json")
    
    # Lancer le backtest complet
    test_name = f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = backtester.run_comprehensive_backtest(test_name)
    
    if results:
        print("\n" + "=" * 80)
        print("🎉 BACKTEST COMPLET TERMINÉ AVEC SUCCÈS")
        print("=" * 80)
        
        # Afficher un résumé
        if 'comparative_analysis' in results and 'risk_analysis' in results['comparative_analysis']:
            risk = results['comparative_analysis']['risk_analysis']
            print(f"📊 Rendement moyen: {risk.get('avg_return', 0):.2%}")
            print(f"📈 Sharpe ratio moyen: {risk.get('avg_sharpe', 0):.3f}")
            print(f"📉 Drawdown moyen: {risk.get('avg_drawdown', 0):.2%}")
        
        print(f"📁 Résultats sauvegardés dans: {backtester.results_dir}")
        print("=" * 80)
    else:
        print("❌ Échec du backtest complet")


if __name__ == "__main__":
    main()
