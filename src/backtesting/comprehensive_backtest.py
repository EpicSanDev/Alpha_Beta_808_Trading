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
from typing import Dict, List, Tuple, Optional, Any, Union
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
# Ajout pour les features de futures
from src.feature_engineering.futures_features import add_all_futures_features
from src.modeling.models import prepare_data_for_model, train_model, load_model_and_predict
from src.signal_generation.signal_generator import generate_base_signals_from_predictions
from src.execution.simulator import BacktestSimulator
from src.core.performance_analyzer import BacktestAnalyzer
from src.validation.walk_forward import WalkForwardValidator
from src.acquisition.preprocessing_utils import handle_missing_values_column
from src.acquisition.preprocessing import normalize_min_max

# Import pour les métriques financières avancées
from src.backtesting.financial_metrics import calculate_portfolio_metrics

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
                "use_random_data": False,
                "use_futures_features": True # Nouveau paramètre pour contrôler l'utilisation des features futures
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
        Lance un backtest complet avec tous les modèles et configurations,
        potentiellement en deux passes (avec et sans features de futures).
        
        Args:
            test_name: Nom optionnel pour ce test
            
        Returns:
            Dict avec tous les résultats du backtest
        """
        base_test_name = test_name or f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"🚀 Démarrage du cycle de backtests complets: {base_test_name}")

        original_use_futures_features = self.config['data_settings'].get('use_futures_features', True)
        all_final_reports = {}

        # Passe 1: Baseline (sans features de futures)
        logger.info("--- PASS 1: BASELINE (SANS FEATURES FUTURES) ---")
        self.config['data_settings']['use_futures_features'] = False
        test_name_baseline = f"{base_test_name}_baseline"
        
        logger.info("📊 Acquisition des données (Baseline)...")
        market_data_baseline = self._prepare_market_data()
        
        if market_data_baseline.empty:
            logger.error("❌ Aucune donnée de marché disponible pour la baseline")
            # Continuer potentiellement avec la passe "avec features" si original_use_futures_features est True
        else:
            logger.info("🔧 Feature Engineering (Baseline)...")
            features_data_baseline = self._engineer_features(market_data_baseline) # _engineer_features respectera le flag de config
            
            logger.info("🤖 Tests multi-modèles (Baseline)...")
            model_results_baseline = self._run_multi_model_tests(features_data_baseline, test_name_baseline)
            
            if self.config['model_settings']['walk_forward_validation']:
                logger.info("📈 Validation Walk-Forward (Baseline)...")
                walk_forward_results_baseline = self._run_walk_forward_validation(features_data_baseline)
                model_results_baseline['walk_forward'] = walk_forward_results_baseline
            
            if self.config['model_settings']['ensemble_mode']:
                logger.info("🎯 Tests d'ensemble (Baseline)...")
                ensemble_results_baseline = self._run_ensemble_tests(features_data_baseline, test_name_baseline)
                model_results_baseline['ensemble'] = ensemble_results_baseline
            
            logger.info("📊 Analyse comparative (Baseline)...")
            comparative_analysis_baseline = self._run_comparative_analysis(model_results_baseline, features_data_baseline)
            
            logger.info("📝 Génération du rapport final (Baseline)...")
            final_report_baseline = self._generate_comprehensive_report(
                model_results_baseline, comparative_analysis_baseline, test_name_baseline
            )
            all_final_reports['baseline'] = final_report_baseline
            logger.info(f"✅ Backtest Baseline terminé: {test_name_baseline}")

        # Passe 2: Avec features de futures (si configuré ou différent de la baseline)
        if original_use_futures_features:
            logger.info("--- PASS 2: AVEC FEATURES FUTURES ---")
            self.config['data_settings']['use_futures_features'] = True # S'assurer que c'est True
            test_name_with_futures = f"{base_test_name}_with_futures"

            logger.info("📊 Acquisition des données (Avec Futures)...")
            market_data_with_futures = self._prepare_market_data() # Peut être les mêmes données brutes
            
            if market_data_with_futures.empty:
                logger.error("❌ Aucune donnée de marché disponible pour le test avec features futures")
            else:
                logger.info("🔧 Feature Engineering (Avec Futures)...")
                features_data_with_futures = self._engineer_features(market_data_with_futures)
                
                logger.info("🤖 Tests multi-modèles (Avec Futures)...")
                model_results_with_futures = self._run_multi_model_tests(features_data_with_futures, test_name_with_futures)
                
                if self.config['model_settings']['walk_forward_validation']:
                    logger.info("📈 Validation Walk-Forward (Avec Futures)...")
                    walk_forward_results_with_futures = self._run_walk_forward_validation(features_data_with_futures)
                    model_results_with_futures['walk_forward'] = walk_forward_results_with_futures
                
                if self.config['model_settings']['ensemble_mode']:
                    logger.info("🎯 Tests d'ensemble (Avec Futures)...")
                    ensemble_results_with_futures = self._run_ensemble_tests(features_data_with_futures, test_name_with_futures)
                    model_results_with_futures['ensemble'] = ensemble_results_with_futures
                
                logger.info("📊 Analyse comparative (Avec Futures)...")
                comparative_analysis_with_futures = self._run_comparative_analysis(model_results_with_futures, features_data_with_futures)
                
                logger.info("📝 Génération du rapport final (Avec Futures)...")
                final_report_with_futures = self._generate_comprehensive_report(
                    model_results_with_futures, comparative_analysis_with_futures, test_name_with_futures
                )
                all_final_reports['with_futures'] = final_report_with_futures
                logger.info(f"✅ Backtest Avec Features Futures terminé: {test_name_with_futures}")
        
        # Restaurer la configuration originale
        self.config['data_settings']['use_futures_features'] = original_use_futures_features
        
        if not all_final_reports:
            logger.error("❌ Aucun backtest n'a pu être exécuté.")
            return {}

        # Pour l'instant, retourner le rapport "avec futures" s'il existe, sinon la baseline, ou un rapport combiné plus tard.
        # L'objectif est d'avoir les deux jeux de résultats.
        # Une amélioration future serait de fusionner intelligemment ces rapports.
        logger.info(f"✅ Cycle de backtests complets terminé pour: {base_test_name}")
        return all_final_reports # Retourne un dict avec 'baseline' et/ou 'with_futures'
    
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

                # Ajout des caractéristiques de futures
                if self.config['data_settings'].get('use_futures_features', False): # Lire le paramètre de configuration
                    # S'assurer que les colonnes requises sont présentes.
                    # Les noms de colonnes par défaut dans add_all_futures_features correspondent
                    # à ceux attendus après le prétraitement.
                    required_futures_cols = ['open_interest', 'funding_rate', 'basis', 'volume', 'close']
                    if all(col in group.columns for col in required_futures_cols):
                        logger.info(f"📈 Ajout des caractéristiques de futures pour {symbol} ({interval})...")
                        group = add_all_futures_features(
                            group,
                            open_interest_col='open_interest', # Nom attendu après prétraitement
                            funding_rate_col='funding_rate',   # Nom attendu après prétraitement
                            basis_col='basis',                 # Nom attendu après prétraitement
                            volume_col='volume',               # Volume des klines, utilisé aussi pour OI/Volume
                            price_col='close'                  # Prix de clôture pour normalisations etc.
                            # Les fenêtres par défaut seront utilisées, peuvent être configurées si besoin
                        )
                    else:
                        missing_cols = [col for col in required_futures_cols if col not in group.columns]
                        logger.warning(f"⚠️ Colonnes manquantes pour les features de futures pour {symbol} ({interval}): {missing_cols}. Ces features ne seront pas ajoutées.")
                else:
                    logger.info(f"📉 Les caractéristiques de futures ne sont pas activées pour {symbol} ({interval}).")

                # Supprimer les NaN après toutes les features
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
                    symbol_results[model_type] = {
                        'training_metrics': {},
                        'predictions': np.array([]),
                        'signals': pd.Series([]),
                        'backtest_results': {}
                    }
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
        
        # Initialiser les signaux
        signals = ['HOLD'] * len(predictions)
        
        # Seuils adaptatifs basés sur les percentiles des valeurs valides
        upper_threshold = np.percentile(valid_predictions, 75)  # Top 25% -> BUY
        lower_threshold = np.percentile(valid_predictions, 25)  # Bottom 25% -> SELL
        
        # Appliquer les seuils en utilisant l'indexation directe pour éviter les problèmes d'index
        for i in range(len(predictions)):
            if not np.isnan(predictions[i]):
                if predictions[i] >= upper_threshold:
                    signals[i] = 'BUY'
                elif predictions[i] <= lower_threshold:
                    signals[i] = 'SELL'
        
        return pd.Series(signals)
    
    def _run_trading_simulation(self, market_data: pd.DataFrame, signals: pd.Series, 
                              symbol: str, model_type: str) -> Dict[str, Any]:
        """Lance une simulation de trading pour un symbole et modèle donnés"""
        
        try:
            # Préparer les données pour le simulateur
            market_data_sim = market_data.copy()
            market_data_sim = market_data_sim.sort_values('timestamp').set_index('timestamp')
            if not market_data_sim.index.is_unique:
                logger.warning(f"Index de market_data_sim non unique pour {symbol} - {model_type}. Doublons supprimés (en gardant le premier).")
                market_data_sim = market_data_sim[~market_data_sim.index.duplicated(keep='first')]
            
            # Créer le DataFrame des signaux avec alignement strict
            timestamps = market_data_sim.index
            min_length = min(len(timestamps), len(signals))
            
            # Tronquer les deux à la même longueur
            timestamps_aligned = timestamps[:min_length]
            signals_aligned = signals.iloc[:min_length] if hasattr(signals, 'iloc') else signals[:min_length]
            
            # Remplacer les valeurs NaN dans les signaux par 'HOLD'
            signals_cleaned = []
            for sig in signals_aligned:
                if pd.isna(sig) or sig == 'nan' or str(sig).lower() == 'nan':
                    signals_cleaned.append('HOLD')
                else:
                    signals_cleaned.append(str(sig))
            
            # S'assurer que market_data_sim est aussi aligné
            market_data_sim = market_data_sim.iloc[:min_length]
            
            signals_df = pd.DataFrame({
                'signal': signals_cleaned,
                'nominal_value_to_trade': [
                    self.config['strategy_settings']['initial_capital'] * 
                    self.config['strategy_settings']['risk_per_trade'] 
                    if sig in ['BUY', 'SELL'] else 0 
                    for sig in signals_cleaned
                ]
            }, index=timestamps_aligned)
            
            # Lancer la simulation
            simulator = BacktestSimulator(
                initial_capital=self.config['strategy_settings']['initial_capital'],
                market_data=market_data_sim,
                asset_symbol=symbol # Ajout du symbole de l'actif
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
    
    def _calculate_performance_metrics(self, portfolio_history: Union[List[Dict], pd.DataFrame],
                                     trade_history: Union[List[Dict], pd.DataFrame],
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcule des métriques de performance avancées en utilisant financial_metrics.calculate_portfolio_metrics.
        """
        
        if isinstance(portfolio_history, list):
            if not portfolio_history:
                logger.warning("Historique de portefeuille vide, impossible de calculer les métriques.")
                return {}
            portfolio_df = pd.DataFrame(portfolio_history)
        elif isinstance(portfolio_history, pd.DataFrame):
            if portfolio_history.empty:
                logger.warning("DataFrame de l'historique de portefeuille vide.")
                return {}
            portfolio_df = portfolio_history.copy()
        else:
            logger.error(f"Type d'historique de portefeuille non supporté: {type(portfolio_history)}")
            return {}

        if 'timestamp' in portfolio_df.columns: # S'assurer que le timestamp est l'index si ce n'est pas déjà le cas
            portfolio_df = portfolio_df.set_index('timestamp')


        trades_df = None
        if isinstance(trade_history, list):
            if trade_history:
                trades_df = pd.DataFrame(trade_history)
        elif isinstance(trade_history, pd.DataFrame):
            trades_df = trade_history.copy()
        
        # Préparer les données du benchmark si market_data est fourni
        benchmark_returns_df = None
        if market_data is not None and not market_data.empty and 'close' in market_data.columns:
            # S'assurer que market_data a un index de type datetime pour l'alignement
            if not isinstance(market_data.index, pd.DatetimeIndex) and 'timestamp' in market_data.columns:
                 market_data_bm = market_data.set_index('timestamp').copy()
            else:
                 market_data_bm = market_data.copy()

            # Créer un DataFrame pour le benchmark avec la colonne 'close'
            # Ceci est une simplification; idéalement, benchmark_data serait un DataFrame séparé
            # avec des rendements de benchmark. Ici, nous utilisons market_data comme proxy.
            benchmark_returns_df = pd.DataFrame({'close': market_data_bm['close']}, index=market_data_bm.index)


        initial_capital = self.config['strategy_settings'].get('initial_capital', 100000)
        
        # Appeler la fonction centralisée de calcul des métriques
        # Note: financial_metrics.calculate_portfolio_metrics s'attend à ce que
        # portfolio_history soit un DataFrame avec 'portfolio_value' et un index Datetime.
        # trades_df est optionnel. benchmark_data est aussi optionnel.
        
        metrics = calculate_portfolio_metrics(
            portfolio_history=portfolio_df, # Doit contenir 'portfolio_value' et être indexé par date/heure
            benchmark_data=benchmark_returns_df, # DataFrame avec 'close' et index Datetime
            trades_df=trades_df, # DataFrame des transactions
            initial_capital=initial_capital
        )
        
        # Ajouter des métriques qui pourraient ne pas être couvertes par calculate_portfolio_metrics
        # ou qui sont spécifiques à ce contexte de backtest.
        # Par exemple, si calculate_portfolio_metrics ne retourne pas 'total_trades' directement
        # (bien qu'il le fasse via _calculate_trading_metrics).
        if trades_df is not None and not trades_df.empty:
            metrics['total_trades'] = metrics.get('total_trades', len(trades_df)) # Assurer que total_trades est présent

        if not portfolio_df.empty and 'portfolio_value' in portfolio_df.columns:
             metrics['final_value'] = metrics.get('final_value', portfolio_df['portfolio_value'].iloc[-1])


        # S'assurer que les métriques clés sont présentes, même avec NaN si non calculables
        required_keys = self.config['performance_settings'].get('metrics', [])
        for key in required_keys:
            if key not in metrics:
                metrics[key] = np.nan
        
        return metrics

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
        
        # Générer un résumé lisible pour chaque rapport
        base_summary_filename = f"{self.results_dir}/summary_report_{timestamp}.txt" # Nom de base
        self._generate_summary_report(final_report, base_summary_filename) # final_report est ici all_final_reports
        
        logger.info(f"📝 Rapports JSON complets sauvegardés dans {self.results_dir} (préfixe: comprehensive_report_{timestamp})")
        # Le message pour les résumés est maintenant dans _generate_summary_report
        
        return final_report # final_report est ici all_final_reports
    
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
    
    def _generate_summary_report(self, all_reports: Dict[str, Dict], base_summary_filename: str):
        """Génère un rapport de synthèse lisible pour chaque passe de backtest."""
        
        for report_type, final_report_data in all_reports.items():
            if not final_report_data: # Skip si un rapport est vide (ex: une passe a échoué)
                logger.warning(f"Aucune donnée de rapport pour {report_type}, résumé non généré.")
                continue

            # Construire un nom de fichier unique pour chaque résumé
            # base_summary_filename pourrait être "summary_report_YYYYMMDD_HHMMSS.txt"
            # On veut "summary_report_YYYYMMDD_HHMMSS_baseline.txt" ou "_with_futures.txt"
            path_obj = Path(base_summary_filename)
            summary_file_path = path_obj.with_name(f"{path_obj.stem}_{report_type}{path_obj.suffix}")

            with open(summary_file_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"RAPPORT DE BACKTEST - {report_type.upper()} - ALPHABETA808TRADING\n")
                f.write("=" * 80 + "\n\n")
                
                # Informations du test
                test_info = final_report_data.get('test_info', {})
                f.write(f"Test: {test_info.get('test_name', 'N/A')}\n")
                f.write(f"Date: {test_info.get('timestamp', 'N/A')}\n")
                
                # Indiquer si les features futures ont été utilisées pour cette passe
                use_futures_features_for_pass = test_info.get('config', {}).get('data_settings', {}).get('use_futures_features', 'N/A')
                f.write(f"Utilisation des Features Futures: {use_futures_features_for_pass}\n")
                f.write(f"Configuration: {len(test_info.get('config', {}))} paramètres\n\n")
                
                # Résumé des données
                data_summary = final_report_data.get('data_summary', {})
                f.write("DONNÉES TESTÉES:\n")
                f.write(f"- Symboles: {', '.join(data_summary.get('symbols_tested', []))}\n")
                f.write(f"- Modèles: {', '.join(data_summary.get('models_tested', []))}\n")
                f.write(f"- Total tests: {data_summary.get('total_tests', 0)}\n\n")
                
                # Analyse comparative
                comp_analysis = final_report_data.get('comparative_analysis', {})
                if 'risk_analysis' in comp_analysis:
                    risk = comp_analysis['risk_analysis']
                    f.write("ANALYSE DES RISQUES (pour cette passe):\n")
                    f.write(f"- Rendement moyen: {risk.get('avg_return', 0):.2%}\n")
                    f.write(f"- Sharpe ratio moyen: {risk.get('avg_sharpe', 0):.3f}\n")
                    f.write(f"- Drawdown moyen: {risk.get('avg_drawdown', 0):.2%}\n")
                    f.write(f"- Meilleur rendement: {risk.get('best_overall_return', 0):.2%}\n\n")
                
                # Recommandations
                recommendations = final_report_data.get('recommendations', {})
                if recommendations:
                    f.write("RECOMMANDATIONS (basées sur cette passe):\n")
                    for key, rec in recommendations.items():
                        f.write(f"- {rec}\n")
                
                f.write("\n" + "=" * 80 + "\n")
            logger.info(f"📄 Résumé ({report_type}) sauvegardé: {summary_file_path}")


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
