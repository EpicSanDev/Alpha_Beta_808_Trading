#!/usr/bin/env python3
"""
AlphaBeta808 Continuous Trading Bot
Trading continu 24h/24 7j/7 avec gestion automatique du portfolio
"""

import os
import sys
import time
import asyncio
import logging
import signal
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import json

# Ajout du répertoire src au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Imports des modules existants
from src.acquisition.connectors import load_binance_klines
from src.feature_engineering.technical_features import (
    calculate_sma, calculate_ema, calculate_rsi,
    calculate_macd, calculate_bollinger_bands,
    calculate_price_momentum, calculate_volume_features
)
from src.modeling.models import prepare_data_for_model, load_model_and_predict
from src.signal_generation.signal_generator import generate_signals_from_predictions
from src.execution.real_time_trading import (
    BinanceRealTimeTrader, TradingStrategy, RiskManager,
    MarketData, TradingOrder, OrderSide, OrderType, OrderStatus
)
from src.risk_management.risk_controls import check_position_limit
from src.portfolio.multi_asset import MultiAssetPortfolioManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTrader:
    """
    Bot de trading continu 24h/24 7j/7
    Scanne les opportunités, génère des signaux et exécute les trades automatiquement
    """
    
    def __init__(self, config_file: str = "trader_config.json"):
        """
        Initialize the continuous trader
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.is_running = False
        self.trader = None
        self.strategy = None
        self.portfolio_manager = None
        self.risk_manager = None
        
        # État du trading
        self.last_model_update = None # Conserver pour la logique de _should_update_model
        self.trading_pairs = self.config.get('trading_pairs', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
        self.scan_interval = self.config.get('scan_interval', 60)  # seconds
        self.model_update_interval = self.config.get('model_update_interval', 3600)  # 1 heure
        
        # Historique de marché pour les features
        self.market_history: Dict[str, pd.DataFrame] = {} # Symbol -> DataFrame
        self.last_feature_update: Dict[str, datetime] = {}

        # Métriques de performance et statistiques
        self.stats = {
            'start_time': datetime.now(),
            'total_signals_generated': 0,
            'total_signals_processed': 0,
            'total_trades_executed': 0, # Compte les soumissions d'ordres
            'successful_trades': 0, # Compte les ordres FILLED
            'failed_trades': 0, # Compte les ordres REJECTED, CANCELED, EXPIRED ou échecs de soumission
            'errors': 0,
            'pnl_today': 0.0, # Profit and Loss journalier
            'last_health_check': None,
            'last_model_retrain_time': None,
            'current_exposure': 0.0,
            'open_positions': {} # symbol -> {quantity, entry_price_sum}
        }
        # Pour compatibilité avec le code existant qui pourrait utiliser ces attributs directement
        self.trades_today = 0
        self.profit_loss_today = 0.0
        self.start_time = self.stats['start_time']

        # Queue pour les signaux de trading
        self.signal_queue = asyncio.Queue()
        
        logger.info("ContinuousTrader initialisé")
    
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration depuis un fichier JSON"""
        default_config = {
            "initial_capital": 10000,
            "max_position_size": 0.1, # Pourcentage du capital total par position
            "max_daily_loss": 0.02, # Pourcentage du capital initial
            "max_total_exposure": 0.8, # Pourcentage du capital total investi
            "trading_pairs": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "SOLUSDT"],
            "scan_interval": 60, # secondes, pour _market_scanner
            "model_update_interval": 3600, # secondes, pour _model_updater
            "testnet": True, # true pour le mode testnet de Binance, false pour production

            "model": {
                "model_path": "models_store/logistic_regression_mvp.joblib",
            },

            "trading_thresholds": {
                "buy_threshold": 0.3,
                "sell_threshold": -0.3
            },

            "signal_filters": {
                "enabled": True,
                "volatility": { # Filtre basé sur la variation de prix sur 24h
                    "max_change_percent_24h": 10.0, # Si > 10% de variation, signal atténué
                    "signal_dampening_factor": 0.5
                },
                "spread": { # Filtre basé sur le spread bid-ask (nécessite données temps réel non kline)
                    "max_relative_spread": 0.001, # Si spread > 0.1%, signal atténué
                    "signal_dampening_factor": 0.7,
                    "enabled": False # Désactivé par défaut car nécessite appel API supplémentaire
                },
                "volume": { # Filtre basé sur le volume de trading sur 24h
                    "min_volume_24h_usdt": 1000000, # Volume en USDT pour comparaison entre paires
                    "signal_boost_factor": 1.1, # Si volume suffisant, signal renforcé
                    "use_kline_volume_proxy": True # Si true et volume 24h non dispo, utilise volume de la kline * prix comme proxy
                }
            },

            "risk_management": {
                "stop_loss_percent": 0.02,
                "take_profit_percent": 0.04,
                "max_orders_per_minute": 10
            },

            "features": {
                "sma_windows": [10, 20, 50],
                "ema_windows": [10, 20],
                "rsi_window": 14,
                "use_volume_features": True,
                "feature_lookback_periods": 200 # Nombre de klines à conserver pour calcul des features
            },
            "logging": {
                "level": "INFO",
                "file": "continuous_trader.log"
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Deep merge for nested dictionaries
                def deep_update(source, overrides):
                    for key, value in overrides.items():
                        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                            deep_update(source[key], value)
                        else:
                            source[key] = value
                    return source

                default_config = deep_update(default_config, loaded_config)
                logger.info(f"Configuration chargée depuis {config_file}")
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de {config_file}: {e}")
                logger.info("Utilisation de la configuration par défaut")
        else:
            # Créer le fichier de configuration par défaut
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Fichier de configuration par défaut créé: {config_file}")
        
        return default_config
    
    async def initialize(self):
        """Initialise tous les composants du trader"""
        try:
            load_dotenv()
            
            # Vérifier les clés API
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Clés API Binance manquantes dans les variables d'environnement")
            
            # Initialiser le risk manager
            self.risk_manager = RiskManager(
                max_position_size=self.config['max_position_size'],
                max_daily_loss=self.config['max_daily_loss'],
                max_total_exposure=self.config['max_total_exposure'],
                max_orders_per_minute=self.config['risk_management']['max_orders_per_minute']
            )
            
            # Initialiser le trader Binance
            self.trader = BinanceRealTimeTrader(
                api_key=api_key,
                api_secret=api_secret,
                testnet=self.config.get('testnet', True), # Utiliser la configuration
                risk_manager=self.risk_manager
            )
            
            # Initialiser le portfolio manager
            self.portfolio_manager = MultiAssetPortfolioManager(
                initial_capital=self.config['initial_capital']
            )
            
            # Initialiser la stratégie de trading
            self.strategy = TradingStrategy(self.trader)
            
            # Configurer les callbacks
            self.trader.set_callbacks(
                on_market_data=self._on_market_data,
                on_order_update=self._on_order_update,
                on_account_update=self._on_account_update
            )
            
            # Ajouter les paires de trading à la stratégie
            for symbol in self.trading_pairs:
                self.strategy.add_symbol(symbol)
            
            logger.info("Tous les composants ont été initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            raise
    
    async def start_trading(self):
        """Démarre le trading continu"""
        if self.is_running:
            logger.warning("Le trading est déjà en cours")
            return
        
        self.is_running = True
        logger.info("🚀 Démarrage du trading continu AlphaBeta808...")
        
        try:
            # Démarrer les tâches asynchrones
            tasks = [
                asyncio.create_task(self._market_scanner()),
                asyncio.create_task(self._signal_processor()),
                asyncio.create_task(self._model_updater()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._risk_monitor())
            ]
            
            # Démarrer le flux de données de marché
            self.strategy.start_trading(self.trading_pairs)
            
            # Attendre que toutes les tâches soient terminées
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Erreur dans le trading continu: {e}")
            await self.stop_trading()
    
    async def stop_trading(self):
        """Arrête le trading continu"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("🛑 Arrêt du trading continu...")
        
        try:
            # Arrêter la stratégie
            if self.strategy:
                self.strategy.stop_trading()
            
            # Annuler tous les ordres ouverts
            if self.trader:
                await self._cancel_all_open_orders()
            
            logger.info("Trading arrêté avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt: {e}")
    
    async def _market_scanner(self):
        """Scanne les marchés en continu pour détecter les opportunités"""
        logger.info("📡 Démarrage du scanner de marché")
        
        while self.is_running:
            try:
                for symbol in self.trading_pairs:
                    await self._analyze_symbol(symbol)
                
                # Attendre avant le prochain scan
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans le scanner de marché: {e}")
                await asyncio.sleep(30)  # Attendre un peu plus en cas d'erreur
    
    async def _analyze_symbol(self, symbol: str):
        """Analyse un symbole spécifique et génère un signal"""
        try:
            min_lookback_features = self.config['features'].get('feature_lookback_periods', 200)
            # Un seuil minimal absolu pour avoir assez de données pour les indicateurs (ex: RSI 14 + MACD ~26-30)
            # On prend le max des fenêtres de SMA/EMA/RSI etc. + une marge.
            # Pour RSI(14), EMA(20), SMA(50), il faut au moins 50 points.
            # min_data_points_for_features = max(
            #     max(self.config['features']['sma_windows'] if self.config['features']['sma_windows'] else [0]),
            #     max(self.config['features']['ema_windows'] if self.config['features']['ema_windows'] else [0]),
            #     self.config['features']['rsi_window']
            # ) + 20 # Marge pour stabilisation des indicateurs
            min_data_points_for_features = 70 # Valeur fixe pour simplifier, basée sur SMA50 + marge

            market_data_df = None
            use_history = False
            if symbol in self.market_history and len(self.market_history[symbol]) >= min_data_points_for_features:
                # Vérifier si la dernière donnée de l'historique est suffisamment récente
                # Cela dépend de la fréquence des klines reçues par _on_market_data
                # Si on scan toutes les minutes et qu'on reçoit des klines 1m, c'est bon.
                # Si on scan toutes les heures et qu'on reçoit des klines 1h, c'est bon.
                # Pour l'instant, on suppose que si l'historique est là, il est pertinent.
                market_data_df = self.market_history[symbol].copy()
                use_history = True
                logger.debug(f"Utilisation de l'historique de marché pour {symbol} ({len(market_data_df)} points)")
            
            if not use_history:
                logger.debug(f"Récupération de nouvelles données de marché pour {symbol} (historique insuffisant ou trop ancien)")
                # S'assurer de récupérer assez de données pour le calcul des features + la période de lookback du modèle si besoin.
                # min_lookback_features est pour le calcul des features.
                market_data_df = await self._get_recent_market_data(symbol, limit=min_lookback_features + 50) # +50 pour marge

            if market_data_df is None or len(market_data_df) < min_data_points_for_features:
                logger.warning(f"Données insuffisantes pour {symbol} après récupération/historique ({len(market_data_df) if market_data_df is not None else 0} points, besoin de {min_data_points_for_features}).")
                return
            
            # Calculer les features techniques
            required_cols = ['open', 'high', 'low', 'close', 'volume'] # Assurer que ces colonnes existent
            if not all(col in market_data_df.columns for col in required_cols):
                logger.warning(f"Colonnes OHLVC manquantes pour {symbol}. Tentative de récupération complète.")
                market_data_df = await self._get_recent_market_data(symbol, limit=min_lookback_features + 50)
                if market_data_df is None or not all(col in market_data_df.columns for col in required_cols):
                    logger.error(f"Échec final de récupération des colonnes OHLVC pour {symbol}.")
                    return

            market_data_with_features = self._calculate_features(market_data_df.copy()) # Toujours copier
            
            if market_data_with_features.empty or len(market_data_with_features) < 1: # S'assurer qu'il y a au moins une ligne après dropna
                logger.warning(f"Données vides après calcul des features pour {symbol}")
                return

            # Générer une prédiction avec le modèle
            signal_strength = await self._generate_signal(market_data_with_features, symbol)
            self.stats['total_signals_generated'] += 1
            
            if signal_strength is not None:
                current_price = market_data_with_features['close'].iloc[-1]
                last_row_dict = market_data_with_features.iloc[-1].to_dict()

                # Tenter de récupérer des données de ticker plus récentes pour les filtres (volume 24h, etc.)
                # Ceci est optionnel et ajoute un appel API.
                ticker_info = None
                try:
                    # Note: get_ticker_info n'est pas async dans BinanceRealTimeTrader,
                    # il faudrait l'adapter ou utiliser une version non-bloquante si disponible.
                    # Pour l'instant, on va supposer qu'on ne fait pas cet appel ici pour garder _analyze_symbol sync après les await initiaux.
                    # Les filtres devront se contenter des données des klines.
                    # Si BinanceRealTimeTrader.get_ticker_info() devient async:
                    # ticker_info = await self.trader.get_ticker_info(symbol)
                    pass
                except Exception as e_ticker:
                    logger.warning(f"Impossible de récupérer les infos du ticker pour {symbol} pour les filtres: {e_ticker}")

                market_snapshot_for_filter = last_row_dict.copy()
                if ticker_info:
                    # 'priceChangePercent' est utile pour le filtre de volatilité
                    market_snapshot_for_filter['price_change_percent_24h'] = float(ticker_info.get('priceChangePercent', 0.0))
                    # 'quoteVolume' est le volume en USDT sur 24h, utile pour le filtre de volume
                    market_snapshot_for_filter['volume_usdt_24h'] = float(ticker_info.get('quoteVolume', 0.0))
                    # 'askPrice', 'bidPrice' pour le filtre de spread
                    market_snapshot_for_filter['ask_price'] = float(ticker_info.get('askPrice', 0.0))
                    market_snapshot_for_filter['bid_price'] = float(ticker_info.get('bidPrice', 0.0))
                else:
                    # Si pas de ticker_info, les filtres devront s'adapter ou utiliser des proxies
                    # Le volume de la kline est dans last_row_dict['volume'] et last_row_dict['quote_asset_volume']
                    pass


                signal_payload = {
                    'symbol': symbol,
                    'signal': signal_strength,
                    'timestamp': datetime.now(),
                    'price': current_price,
                    'market_data_snapshot': market_snapshot_for_filter
                }
                await self.signal_queue.put(signal_payload)
                
                logger.debug(f"Signal généré pour {symbol}: {signal_strength:.3f} @ {current_price}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {symbol}: {e}")
            self.stats['errors'] += 1
    
    async def _get_recent_market_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Récupère les données de marché récentes pour un symbole"""
        try:
            # Utiliser l'API Binance pour récupérer les klines récentes
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            # Récupérer les 100 dernières bougies 1h
            market_data_dict = load_binance_klines(
                api_key=api_key,
                api_secret=api_secret,
                symbol=symbol,
                intervals=['1h'],
                start_date_str="7 days ago"
            )
            
            if '1h' in market_data_dict:
                return market_data_dict['1h']
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            return None
    
    def _calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les features techniques sur les données"""
        try:
            # Features de base
            data = calculate_sma(data, column='close', windows=self.config['features']['sma_windows'])
            data = calculate_ema(data, column='close', windows=self.config['features']['ema_windows'])
            data = calculate_rsi(data, column='close', window=self.config['features']['rsi_window'])
            
            # Features avancées
            data = calculate_macd(data, column='close')
            data = calculate_bollinger_bands(data, column='close')
            data = calculate_price_momentum(data, column='close', windows=[5, 10])
            
            # Features de volume si activées
            if self.config['features']['use_volume_features'] and 'volume' in data.columns:
                data = calculate_volume_features(data, volume_col='volume', price_col='close')
            
            # Nettoyer les NaN
            data = data.dropna()
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des features: {e}")
            return data
    
    async def _generate_signal(self, data: pd.DataFrame, symbol: str) -> Optional[float]:
        """Génère un signal de trading basé sur le modèle ML"""
        try:
            # Préparer les données pour le modèle
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval']
            feature_cols = [col for col in data.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                logger.warning(f"Aucune feature disponible pour {symbol}")
                return None
            
            # Utiliser seulement la dernière observation
            X = data[feature_cols].iloc[-1:].copy()
            
            # Charger le modèle et faire une prédiction
            model_path = self.config['model'].get('model_path', 'models_store/logistic_regression_mvp.joblib')
            if not os.path.exists(model_path):
                logger.warning(f"Modèle non trouvé: {model_path}")
                return None
            
            prediction = load_model_and_predict(X, model_path=model_path, return_probabilities=True)
            
            if prediction is not None and len(prediction) > 0:
                # Convertir la probabilité en signal (-1 à 1)
                prob = prediction[0]
                signal = (prob - 0.5) * 2  # Map [0,1] to [-1,1]
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du signal pour {symbol}: {e}")
            return None
    
    async def _signal_processor(self):
        """Traite les signaux de trading en continu"""
        logger.info("📊 Démarrage du processeur de signaux")
        
        while self.is_running:
            try:
                # Récupérer un signal de la queue (avec timeout)
                try:
                    signal_data = await asyncio.wait_for(self.signal_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Traiter le signal
                await self._process_trading_signal(signal_data)
                
            except Exception as e:
                logger.error(f"Erreur dans le processeur de signaux: {e}")
                await asyncio.sleep(1)
    
    async def _process_trading_signal(self, signal_data: Dict):
        """Traite un signal de trading individuel"""
        try:
            symbol = signal_data['symbol']
            original_signal = signal_data['signal']
            price = signal_data['price'] # Prix au moment de la génération du signal
            market_snapshot = signal_data.get('market_data_snapshot', {}) # Données de la dernière bougie

            self.stats['total_signals_processed'] += 1
            
            # Appliquer les filtres de signaux
            filtered_signal = original_signal
            if self.config['signal_filters']['enabled']:
                filtered_signal = self._apply_signal_filters(symbol, original_signal, price, market_snapshot)

            if filtered_signal != original_signal:
                logger.info(f"Signal pour {symbol} filtré de {original_signal:.3f} à {filtered_signal:.3f}")

            # Seuils de trading configurables
            buy_threshold = self.config['trading_thresholds']['buy_threshold']
            sell_threshold = self.config['trading_thresholds']['sell_threshold']
            
            # Vérifier les limites de risque
            if not self._check_risk_limits(symbol, filtered_signal): # Utiliser le signal filtré pour la décision de risque
                return
            
            # Déterminer l'action à prendre
            if filtered_signal > buy_threshold:
                await self._execute_buy_signal(symbol, filtered_signal, price)
            elif filtered_signal < sell_threshold:
                await self._execute_sell_signal(symbol, filtered_signal, price)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du signal: {e}")
            self.stats['errors'] += 1
    
    def _apply_signal_filters(self, symbol: str, signal: float, current_price: float, market_snapshot: Dict) -> float:
        """Applique des filtres au signal généré."""
        if not self.config['signal_filters']['enabled']:
            return signal

        cfg_filters = self.config['signal_filters']
        modified_signal = signal

        # 1. Filtre de Volatilité
        # Nécessite price_change_24h. Si non dispo dans market_snapshot, on ne peut pas l'appliquer.
        # 'market_snapshot' vient des klines, qui n'ont pas directement price_change_24h.
        # On pourrait essayer de le calculer à partir de l'historique, ou le récupérer via un appel ticker.
        # Pour l'instant, si 'price_change_24h' n'est pas dans market_snapshot, on saute ce filtre.
        # Alternative: utiliser une mesure de volatilité basée sur les klines (ex: ATR).
        # Simplification: si on a 'price_change_percent' dans market_snapshot (ex: si on l'ajoute depuis un ticker)
        price_change_24h_percent = market_snapshot.get('price_change_percent') # Supposons qu'on l'ait
        if price_change_24h_percent is not None:
            vol_cfg = cfg_filters.get('volatility', {})
            if abs(price_change_24h_percent) > vol_cfg.get('max_change_percent_24h', 10.0):
                modified_signal *= vol_cfg.get('signal_dampening_factor', 0.5)
                logger.debug(f"Filtre volatilité appliqué pour {symbol}. Signal: {modified_signal:.3f}")

        # 2. Filtre de Spread (Difficile sans données bid/ask en temps réel)
        # Les klines ne fournissent pas bid/ask. On pourrait le récupérer via self.trader.get_order_book_ticker(symbol)
        # Pour l'instant, on va sauter ce filtre ou le rendre optionnel.
        # spread_cfg = cfg_filters.get('spread', {})
        # try:
        #     ticker_data = await self.trader.get_ticker_data(symbol) # Nécessiterait que la méthode soit async
        #     if ticker_data and 'askPrice' in ticker_data and 'bidPrice' in ticker_data:
        #         ask_price = float(ticker_data['askPrice'])
        #         bid_price = float(ticker_data['bidPrice'])
        #         if current_price > 0: # current_price est le close de la kline, peut être différent
        #             spread = (ask_price - bid_price) / current_price
        #             if spread > spread_cfg.get('max_relative_spread', 0.001):
        #                 modified_signal *= spread_cfg.get('signal_dampening_factor', 0.7)
        #                 logger.debug(f"Filtre spread appliqué pour {symbol}. Signal: {modified_signal:.3f}")
        # except Exception as e:
        #     logger.warning(f"Impossible d'appliquer le filtre de spread pour {symbol}: {e}")
        pass # Sauter le filtre de spread pour l'instant car il nécessite un appel async dans une méthode sync

        # 3. Filtre de Volume
        # 'volume' dans market_snapshot est le volume de la kline, pas le volume 24h en USD.
        # On a besoin du volume 24h en USDT (ou équivalent).
        # On pourrait le récupérer via self.trader.get_ticker_data(symbol) -> quoteVolume
        # Ou si 'volume_usd_24h' est dans market_snapshot.
        volume_cfg = cfg_filters.get('volume', {})
        volume_24h_usd = market_snapshot.get('quote_asset_volume') # 'volume' est base asset, 'quote_asset_volume' est volume en quote (USDT)
                                                                # Ceci est le volume de la kline, pas 24h.
                                                                # Pour un vrai volume 24h, il faudrait un appel ticker.
        # Tentative d'utiliser le volume de la kline comme proxy si 'volume_24h_usd' n'est pas là.
        # C'est une approximation grossière.
        if volume_24h_usd is None and 'volume' in market_snapshot and current_price > 0:
             volume_24h_usd = market_snapshot['volume'] * current_price # Approximation du volume de la kline en USD

        if volume_24h_usd is not None:
            if volume_24h_usd < volume_cfg.get('min_volume_24h_usd', 1000000): # Si volume trop bas
                 # On pourrait réduire le signal ou l'ignorer. LTB le boostait.
                 # Ici, on va suivre la logique de LTB qui booste si volume élevé.
                 pass # Ne rien faire si volume bas, LTB boostait si volume HAUT.
            else: # Volume est HAUT (ou au moins pas bas)
                 modified_signal *= volume_cfg.get('signal_boost_factor', 1.1)
                 logger.debug(f"Filtre volume (boost) appliqué pour {symbol}. Volume: {volume_24h_usd:.2f}. Signal: {modified_signal:.3f}")
        
        return max(-1.0, min(1.0, modified_signal)) # Assurer que le signal reste dans [-1, 1]

    def _check_risk_limits(self, symbol: str, signal: float) -> bool:
        """Vérifie si le trade respecte les limites de risque"""
        try:
            # Vérifier les limites du risk manager
            current_exposure = self._calculate_current_exposure()
            max_exposure = self.config['initial_capital'] * self.config['max_total_exposure']
            
            if current_exposure >= max_exposure:
                logger.warning(f"Exposition maximale atteinte: {current_exposure:.2f}")
                return False
            
            # Vérifier les pertes journalières
            if self.stats['pnl_today'] <= -self.config['initial_capital'] * self.config['max_daily_loss']:
                logger.warning(f"Limite de perte journalière atteinte: {self.stats['pnl_today']:.2f}")
                return False
            
            # TODO: Vérifier d'autres limites spécifiques au symbole si nécessaire
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des risques: {e}")
            self.stats['errors'] += 1
            return False
    
    async def _execute_buy_signal(self, symbol: str, signal: float, price: float):
        """Exécute un signal d'achat"""
        try:
            # Calculer la taille de position
            position_value = self._calculate_position_size(symbol, signal)
            quantity = position_value / price
            
            # Arrondir la quantité selon les règles de Binance
            quantity = self._round_quantity(symbol, quantity)
            
            if quantity > 0:
                order = TradingOrder(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    client_order_id=f"buy_{symbol}_{int(time.time())}"
                )
                
                # Placer l'ordre
                success = await self.trader.place_order(order) # place_order devrait retourner l'ID de l'ordre ou une confirmation
                if success: # Supposons que success signifie que l'ordre a été accepté par l'API
                    logger.info(f"🟢 Ordre d'achat soumis: {symbol} - {quantity:.6f} @ market (ref price {price:.4f})")
                    # Les stats de trades exécutés/réussis seront mises à jour dans _on_order_update
                else:
                    self.stats['failed_trades'] += 1
                    logger.error(f"Échec de la soumission de l'ordre d'achat pour {symbol}")

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal d'achat: {e}")
            self.stats['errors'] += 1
            self.stats['failed_trades'] += 1
    
    async def _execute_sell_signal(self, symbol: str, signal: float, price: float):
        """Exécute un signal de vente"""
        try:
            # Vérifier la position actuelle
            current_balance = self.trader.account_balances.get(symbol.replace('USDT', ''), None)
            if not current_balance or current_balance.free <= 0:
                logger.debug(f"Pas de position à vendre pour {symbol}")
                return
            
            # Calculer la quantité à vendre
            sell_percentage = min(abs(signal), 1.0)  # Vendre selon la force du signal
            quantity = current_balance.free * sell_percentage
            quantity = self._round_quantity(symbol, quantity)
            
            if quantity > 0:
                order = TradingOrder(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    client_order_id=f"sell_{symbol}_{int(time.time())}"
                )
                
                # Placer l'ordre
                success = await self.trader.place_order(order)
                if success:
                    logger.info(f"🔴 Ordre de vente soumis: {symbol} - {quantity:.6f} @ market (ref price {price:.4f})")
                else:
                    self.stats['failed_trades'] += 1
                    logger.error(f"Échec de la soumission de l'ordre de vente pour {symbol}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal de vente: {e}")
            self.stats['errors'] += 1
            self.stats['failed_trades'] += 1
    
    def _calculate_position_size(self, symbol: str, signal: float) -> float:
        """Calcule la taille de position en fonction du signal"""
        # Taille de base proportionnelle au signal
        base_size = self.config['initial_capital'] * self.config['max_position_size']
        position_size = base_size * abs(signal)
        
        # Limiter la taille maximale
        max_position = self.config['initial_capital'] * self.config['max_position_size']
        return min(position_size, max_position)
    
    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Arrondit la quantité selon les règles de Binance"""
        # Règles simplifiées - dans un vrai système, utiliser les exchange info
        if 'USDT' in symbol:
            if 'BTC' in symbol:
                return round(quantity, 6)
            elif 'ETH' in symbol:
                return round(quantity, 5)
            else:
                return round(quantity, 4)
        return round(quantity, 8)
    
    def _calculate_current_exposure(self) -> float:
        """Calcule l'exposition actuelle du portfolio"""
        total_value = 0.0
        
        for asset, balance in self.trader.account_balances.items():
            if balance.total > 0 and asset != 'USDT':
                # Estimer la valeur en USDT (simplifié)
                # Dans un vrai système, utiliser les prix de marché actuels
                estimated_price = 100  # Placeholder
                total_value += balance.total * estimated_price
        
        return total_value
    
    async def _model_updater(self):
        """Met à jour le modèle ML périodiquement"""
        logger.info("🧠 Démarrage du gestionnaire de modèle")
        
        while self.is_running:
            try:
                # Attendre l'intervalle de mise à jour
                await asyncio.sleep(self.model_update_interval)
                
                # Vérifier si une mise à jour est nécessaire
                if self._should_update_model():
                    await self._retrain_model()
                
            except Exception as e:
                logger.error(f"Erreur dans le gestionnaire de modèle: {e}")
    
    def _should_update_model(self) -> bool:
        """Détermine si le modèle doit être mis à jour"""
        if self.last_model_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_model_update
        return time_since_update.total_seconds() >= self.model_update_interval
    
    async def _retrain_model(self):
        """Réentraîne le modèle avec les nouvelles données"""
        try:
            logger.info("🔄 Réentraînement du modèle en cours...")
            
            # Collecter des données récentes pour tous les symboles
            all_data = []
            for symbol in self.trading_pairs:
                data = await self._get_recent_market_data(symbol, limit=500)
                if data is not None:
                    data = self._calculate_features(data)
                    data['symbol'] = symbol
                    all_data.append(data)
            
            if not all_data:
                logger.warning("Pas de données pour réentraîner le modèle")
                return
            
            # Combiner toutes les données
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 100:
                logger.warning("Données insuffisantes pour le réentraînement")
                return
            
            # Préparer les données pour l'entraînement
            from src.modeling.models import train_model
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval']
            feature_cols = [col for col in combined_data.columns if col not in exclude_cols]
            
            X, y = prepare_data_for_model(
                combined_data,
                target_shift_days=1,
                feature_columns=feature_cols,
                price_change_threshold=0.01
            )
            
            # Réentraîner le modèle
            model_path = self.config['model'].get('model_path', 'models_store/logistic_regression_mvp.joblib')
            metrics = train_model(
                X, y,
                model_type='logistic_regression', # Pourrait être configurable aussi
                model_path=model_path,
                scale_features=True
            )
            
            self.last_model_update = datetime.now()
            self.stats['last_model_retrain_time'] = self.last_model_update
            logger.info(f"✅ Modèle réentraîné avec succès. Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
            
        except Exception as e:
            logger.error(f"Erreur lors du réentraînement du modèle: {e}")
            self.stats['errors'] += 1
    
    async def _performance_monitor(self):
        """Surveille les performances du trading"""
        logger.info("📈 Démarrage du moniteur de performance")
        
        while self.is_running:
            try:
                # Générer un rapport de performance toutes les heures
                await asyncio.sleep(3600)
                await self._generate_performance_report()
                
            except Exception as e:
                logger.error(f"Erreur dans le moniteur de performance: {e}")
    
    async def _generate_performance_report(self):
        """Génère un rapport de performance"""
        try:
            uptime = datetime.now() - self.start_time
            
            # Récupérer le résumé de trading
            trading_summary = self.trader.get_trading_summary()
            
            # Calculer les métriques
            total_orders = trading_summary['total_orders']
            fill_rate = trading_summary['fill_rate']
            
            report = f"""
📊 RAPPORT DE PERFORMANCE AlphaBeta808
═══════════════════════════════════════
⏱️  Temps d'activité: {uptime}
📉 Paires actives: {len(self.trading_pairs)}
Signals Générés: {self.stats['total_signals_generated']}
Signals Traités: {self.stats['total_signals_processed']}
Ordres Exécutés (API): {self.stats['total_trades_executed']} (ceci compte les soumissions, pas les fills)
Trades Réussis (Filled): {self.stats['successful_trades']}
Trades Échoués (API/Filled): {self.stats['failed_trades']}
💰 P&L Journalier (Estimé): ${self.stats['pnl_today']:.2f}
Exposition Actuelle (Estimée): ${self.stats['current_exposure']:.2f}
Erreurs: {self.stats['errors']}
Dernière MAJ Modèle: {self.stats['last_model_retrain_time']}
Dernier Health Check: {self.stats['last_health_check']}
═══════════════════════════════════════
            """
            
            logger.info(report)
            
            # Sauvegarder dans un fichier
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"reports/performance_{timestamp}.txt"
            os.makedirs("reports", exist_ok=True)
            
            with open(report_file, 'w') as f:
                f.write(report)
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            self.stats['errors'] += 1
    
    async def _risk_monitor(self):
        """Surveille les risques en continu"""
        logger.info("🛡️ Démarrage du moniteur de risques")
        
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Vérifier toutes les 5 minutes
                
                # Vérifier les limites de risque
                if not self._check_risk_limits("", 0):
                    logger.warning("⚠️ Limites de risque dépassées - Réduction des positions")
                    await self._reduce_positions()
                
            except Exception as e:
                logger.error(f"Erreur dans le moniteur de risques: {e}")
                self.stats['errors'] += 1
    # Correction de l'indentation: _reduce_positions est une méthode de la classe
    async def _reduce_positions(self):
        """Réduit les positions en cas de dépassement des limites de risque"""
        try:
            # Annuler tous les ordres en attente
            await self._cancel_all_open_orders()
            
            # Vendre partiellement les positions importantes
            # Utiliser self.stats['open_positions'] pour savoir ce qu'on détient réellement
            # ou self.trader.account_balances qui est mis à jour par le stream user data.
            logger.info("Tentative de réduction des positions en raison du risque.")
            for symbol_held, position_info in list(self.stats['open_positions'].items()): # list() pour copier si on modifie le dict
                if position_info['quantity'] > 0:
                    # Vendre un pourcentage de la position, ex: 50%
                    quantity_to_sell = position_info['quantity'] * 0.5
                    
                    # S'assurer que le symbole est tradable (ex: BTC si on a BTCUSDT)
                    # Le symbol_held est déjà le symbol de la paire (ex: BTCUSDT) si on stocke comme ça dans open_positions
                    # Sinon, il faut le reconstruire. Supposons que symbol_held est la paire.
                    
                    quantity_to_sell = self._round_quantity(symbol_held, quantity_to_sell)
                        
                    if quantity_to_sell > 0:
                        logger.info(f"Réduction de risque: Vente de {quantity_to_sell} de {symbol_held}")
                        order = TradingOrder(
                            symbol=symbol_held,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=quantity_to_sell,
                            client_order_id=f"risk_reduce_{symbol_held}_{int(time.time())}"
                        )
                        await self.trader.place_order(order)
                        # La mise à jour de self.stats['open_positions'] se fera via _on_order_update
                    else:
                        logger.info(f"Quantité calculée pour réduction de {symbol_held} est nulle ou négative.")
            # Alternative: utiliser self.trader.account_balances
            # for asset, balance in self.trader.account_balances.items():
            #     if balance.total > 0 and asset.upper() != 'USDT':
            #         symbol_pair = f"{asset.upper()}USDT"
            #         if symbol_pair in self.trading_pairs: # S'assurer que c'est une paire qu'on trade
            #             quantity_to_sell = balance.free * 0.5 # Vendre 50% du disponible
            #             quantity_to_sell = self._round_quantity(symbol_pair, quantity_to_sell)
            #             if quantity_to_sell > 0:
            #                 # ... placer l'ordre ...
            
        except Exception as e:
            logger.error(f"Erreur lors de la réduction des positions: {e}")
            self.stats['errors'] += 1
    
    async def _cancel_all_open_orders(self):
        """Annule tous les ordres ouverts"""
        try:
            for order in list(self.trader.open_orders.values()):
                await self.trader.cancel_order(order.order_id)
                logger.info(f"❌ Ordre annulé: {order.symbol} - {order.client_order_id}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation des ordres: {e}")
            self.stats['errors'] += 1
    
    def _update_market_history(self, symbol: str, kline_data: Dict):
        """
        Met à jour l'historique des données de marché pour un symbole à partir d'une kline.
        kline_data est un dictionnaire attendu de l'API stream de Binance pour les klines.
        Exemple: {'t': 1678886400000, 'o': '25000', 'h': '25100', 'l': '24900', 'c': '25050', 'v': '100', ...}
        """
        try:
            if symbol not in self.market_history:
                self.market_history[symbol] = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convertir les données de la kline au format attendu par le DataFrame
            # Le timestamp 't' est le début de la kline. 'T' est la fin. On utilise 't'.
            new_data_point = pd.DataFrame([{
                'timestamp': pd.to_datetime(kline_data['t'], unit='ms'),
                'open': float(kline_data['o']),
                'high': float(kline_data['h']),
                'low': float(kline_data['l']),
                'close': float(kline_data['c']),
                'volume': float(kline_data['v']),
                'quote_asset_volume': float(kline_data.get('q', 0)) # Volume de l'asset de cotation
                # Ajouter d'autres champs si nécessaire, ex: 'n' (nombre de trades)
            }])

            # Concaténer et gérer les doublons (basé sur le timestamp)
            # Si la kline n'est pas encore fermée, son timestamp de début peut se répéter.
            # On ne devrait ajouter que les klines fermées (kline_data['x'] == True)
            if kline_data.get('x', False): # 'x': Is this kline closed?
                # Supprimer l'ancienne entrée si elle existe pour ce timestamp pour éviter les doublons exacts
                self.market_history[symbol] = self.market_history[symbol][self.market_history[symbol]['timestamp'] != new_data_point['timestamp'].iloc[0]]
                self.market_history[symbol] = pd.concat([self.market_history[symbol], new_data_point], ignore_index=True)
                
                # Garder seulement les N dernières périodes
                max_periods = self.config['features'].get('feature_lookback_periods', 200) + 100 # Garder un peu plus pour calculs
                if len(self.market_history[symbol]) > max_periods:
                    self.market_history[symbol] = self.market_history[symbol].tail(max_periods).reset_index(drop=True)
                
                # logger.debug(f"Historique marché pour {symbol} mis à jour avec kline @ {new_data_point['timestamp'].iloc[0]}. Taille: {len(self.market_history[symbol])}")
            # else:
                # logger.debug(f"Kline non fermée pour {symbol} @ {pd.to_datetime(kline_data['t'], unit='ms')}, non ajoutée à l'historique principal.")

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'historique de marché pour {symbol}: {e}")
            self.stats['errors'] += 1

    async def _on_market_data(self, market_data: MarketData):
        """
        Callback pour les données de marché (flux de klines).
        market_data est un objet MarketData qui contient .data (le dictionnaire de la kline).
        """
        try:
            # market_data.data contient les informations de la kline du stream
            # ex: {'e': 'kline', 'E': 1678886460000, 's': 'BTCUSDT', 'k': {'t': 1678886400000, 'T': 1678886459999, 's': 'BTCUSDT', 'i': '1m', 'f': 100, 'L': 200, 'o': '0.0010', 'c': '0.0020', 'h': '0.0025', 'l': '0.0015', 'v': '1000', 'n': 100, 'x': False, 'q': '1.0000', 'V': '500', 'Q': '0.500', 'B': '123456'}}
            if market_data and market_data.data and 'k' in market_data.data:
                kline_details = market_data.data['k']
                symbol = market_data.data['s']
                
                # Mettre à jour l'historique de marché uniquement avec les klines fermées
                if kline_details.get('x', False): # 'x' est le booléen "is kline closed?"
                    self._update_market_history(symbol, kline_details)
                    # On pourrait déclencher une analyse ici si la kline est fermée,
                    # au lieu d'attendre le _market_scanner.
                    # Cela rendrait le bot plus réactif.
                    # await self._analyze_symbol(symbol) # Attention, _analyze_symbol est déjà appelé par _market_scanner
                                                       # Il faudrait une logique pour éviter les analyses concurrentes ou redondantes.
                                                       # Pour l'instant, on laisse _market_scanner faire son travail à intervalle régulier.
                # else:
                    # logger.debug(f"Kline non fermée reçue pour {symbol}: o={kline_details['o']}, c={kline_details['c']}")
                    # On pourrait stocker ces klines non fermées pour une vue "temps réel" si besoin.
            else:
                logger.warning(f"Données de marché reçues dans un format inattendu: {market_data.data if market_data else 'None'}")

        except Exception as e:
            logger.error(f"Erreur dans _on_market_data pour {market_data.symbol if market_data else 'N/A'}: {e}")
            self.stats['errors'] += 1
    
    async def _on_order_update(self, order: TradingOrder):
        """Callback pour les mises à jour d'ordres"""
        logger.info(f"Mise à jour d'ordre reçue: ID {order.order_id}, ClientID {order.client_order_id}, Symbole {order.symbol}, Statut {order.status.value}, Quantité Remplie {order.filled_quantity}, Prix Moyen {order.avg_fill_price}")
        self.stats['total_trades_executed'] +=1 # Compte chaque tentative de trade (soumission)

        if order.status == OrderStatus.FILLED:
            self.stats['successful_trades'] += 1
            logger.info(f"✅ Ordre exécuté et rempli: {order.symbol} {order.side.value} {order.filled_quantity:.6f} @ {order.avg_fill_price:.4f}")
            
            # Mettre à jour P&L et positions (simplifié)
            # Un vrai calcul de P&L nécessite de tracker le coût d'acquisition.
            # Pour l'instant, on va juste compter les trades.
            # On pourrait aussi mettre à jour self.stats['open_positions']
            if order.side == OrderSide.BUY:
                if order.symbol not in self.stats['open_positions']:
                    self.stats['open_positions'][order.symbol] = {'quantity': 0, 'entry_price_sum': 0}
                self.stats['open_positions'][order.symbol]['quantity'] += order.filled_quantity
                self.stats['open_positions'][order.symbol]['entry_price_sum'] += order.filled_quantity * order.avg_fill_price
            elif order.side == OrderSide.SELL:
                if order.symbol in self.stats['open_positions'] and self.stats['open_positions'][order.symbol]['quantity'] > 0:
                    # Calcul simple du P&L pour cette vente
                    avg_buy_price = self.stats['open_positions'][order.symbol]['entry_price_sum'] / self.stats['open_positions'][order.symbol]['quantity']
                    pnl_trade = (order.avg_fill_price - avg_buy_price) * order.filled_quantity
                    self.stats['pnl_today'] += pnl_trade
                    logger.info(f"Trade de vente pour {order.symbol} P&L: {pnl_trade:.2f}")
                    self.stats['open_positions'][order.symbol]['quantity'] -= order.filled_quantity
                    self.stats['open_positions'][order.symbol]['entry_price_sum'] -= order.filled_quantity * avg_buy_price # Ajuster la somme
                    if self.stats['open_positions'][order.symbol]['quantity'] <= 0.000001: # Tolérance pour float
                        del self.stats['open_positions'][order.symbol]
                else: # Vente à découvert ou vente sans position trackée
                    logger.warning(f"Vente de {order.symbol} sans position d'achat trackée.")


        elif order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.stats['failed_trades'] += 1
            logger.warning(f"⚠️ Ordre non abouti: {order.symbol} {order.side.value}, Statut: {order.status.value}, Raison: {order.reject_reason if order.reject_reason else 'N/A'}")
    
    async def _on_account_update(self, balances: Dict[str, any]):
        """Callback pour les mises à jour de compte"""
        # Mettre à jour l'exposition totale, P&L, etc.
        # logger.debug(f"Mise à jour du compte reçue: {balances}")
        # Cette fonction est appelée par BinanceRealTimeTrader avec les balances.
        # On peut l'utiliser pour mettre à jour self.stats['current_exposure']
        # et potentiellement un P&L plus précis si on a les prix actuels.
        current_exposure_val = 0
        if self.trader: # S'assurer que trader est initialisé
            for asset, balance_info in self.trader.account_balances.items(): # Utiliser les balances stockées dans trader
                if asset.upper() != 'USDT' and balance_info.total > 0:
                    symbol_pair = f"{asset.upper()}USDT"
                    # Essayer d'obtenir le prix actuel pour l'évaluation
                    # Pourrait utiliser self.market_history ou un appel ticker
                    current_price = None
                    if symbol_pair in self.market_history and not self.market_history[symbol_pair].empty:
                        current_price = self.market_history[symbol_pair]['close'].iloc[-1]
                    
                    if current_price:
                        current_exposure_val += balance_info.total * current_price
                    # else: logger.warning(f"Prix non trouvé pour {symbol_pair} pour calculer l'exposition")

        self.stats['current_exposure'] = current_exposure_val
        # logger.info(f"Exposition actuelle mise à jour: ${self.stats['current_exposure']:.2f}")
        pass

    def get_status(self) -> Dict:
        """Retourne le statut actuel du bot."""
        uptime_seconds = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        # Copier les stats pour éviter les modifications concurrentes pendant la lecture
        current_stats = self.stats.copy()
        current_stats['uptime_seconds'] = uptime_seconds
        current_stats['start_time'] = str(current_stats['start_time']) # Sérialisable
        if current_stats['last_health_check']:
             current_stats['last_health_check'] = str(current_stats['last_health_check'])
        if current_stats['last_model_retrain_time']:
             current_stats['last_model_retrain_time'] = str(current_stats['last_model_retrain_time'])


        trading_summary = {}
        if self.trader:
            try:
                trading_summary = self.trader.get_trading_summary()
            except Exception as e:
                logger.error(f"Erreur lors de la récupération du trading summary: {e}")
        
        subscribed_symbols = []
        if self.strategy and hasattr(self.strategy, 'trader') and self.strategy.trader:
            subscribed_symbols = list(self.strategy.trader.subscribed_symbols)


        return {
            'is_running': self.is_running,
            'config_name': self.config.get("name", "N/A"), # Si on ajoute un nom à la config
            'current_time': datetime.now().isoformat(),
            'stats': current_stats,
            'trading_summary': trading_summary,
            'subscribed_symbols': subscribed_symbols,
            'market_history_stats': {
                symbol: {
                    'count': len(df),
                    'last_update': df['timestamp'].iloc[-1].isoformat() if not df.empty else None
                } for symbol, df in self.market_history.items()
            },
            'signal_queue_size': self.signal_queue.qsize(),
            # Ne pas inclure self.config entier ici car il peut contenir des secrets si mal configuré.
            # Ou alors, filtrer les clés sensibles.
            # 'active_config_subset': {
            #     'trading_pairs': self.config.get('trading_pairs'),
            #     'testnet': self.config.get('testnet'),
            #     'model_path': self.config.get('model', {}).get('model_path')
            # }
        }

async def main():
    """Fonction principale"""
    # Configuration du signal handler pour arrêt propre
    trader = ContinuousTrader()
    
    def signal_handler(signum, frame):
        logger.info("Signal d'arrêt reçu...")
        asyncio.create_task(trader.stop_trading())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialiser et démarrer le trader
        await trader.initialize()
        await trader.start_trading()
        
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
    finally:
        await trader.stop_trading()

if __name__ == "__main__":
    print("🚀 AlphaBeta808 Continuous Trading Bot")
    print("═══════════════════════════════════════")
    print("Trading automatique 24h/24 7j/7")
    print("Ctrl+C pour arrêter")
    print("═══════════════════════════════════════")
    
    asyncio.run(main())
