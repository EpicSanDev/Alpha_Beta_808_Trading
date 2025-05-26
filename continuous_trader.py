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
        self.last_model_update = None
        self.trading_pairs = self.config.get('trading_pairs', ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
        self.scan_interval = self.config.get('scan_interval', 60)  # seconds
        self.model_update_interval = self.config.get('model_update_interval', 3600)  # 1 heure
        
        # Métriques de performance
        self.trades_today = 0
        self.profit_loss_today = 0.0
        self.start_time = datetime.now()
        
        # Queue pour les signaux de trading
        self.signal_queue = asyncio.Queue()
        
        logger.info("ContinuousTrader initialisé")
    
    def _load_config(self, config_file: str) -> Dict:
        """Charge la configuration depuis un fichier JSON"""
        default_config = {
            "initial_capital": 10000,
            "max_position_size": 0.1,
            "max_daily_loss": 0.02,
            "max_total_exposure": 0.8,
            "trading_pairs": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "SOLUSDT"],
            "scan_interval": 60,
            "model_update_interval": 3600,
            "risk_management": {
                "stop_loss_percent": 0.02,
                "take_profit_percent": 0.04,
                "max_orders_per_minute": 10
            },
            "features": {
                "sma_windows": [10, 20, 50],
                "ema_windows": [10, 20],
                "rsi_window": 14,
                "use_volume_features": True
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                default_config.update(loaded_config)
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
                testnet=False,  # Production mode
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
            # Récupérer les données de marché récentes
            market_data = await self._get_recent_market_data(symbol)
            
            if market_data is None or len(market_data) < 50:
                logger.warning(f"Données insuffisantes pour {symbol}")
                return
            
            # Calculer les features techniques
            market_data = self._calculate_features(market_data)
            
            # Générer une prédiction avec le modèle
            signal_strength = await self._generate_signal(market_data, symbol)
            
            if signal_strength is not None:
                # Ajouter le signal à la queue pour traitement
                await self.signal_queue.put({
                    'symbol': symbol,
                    'signal': signal_strength,
                    'timestamp': datetime.now(),
                    'price': market_data['close'].iloc[-1]
                })
                
                logger.debug(f"Signal généré pour {symbol}: {signal_strength:.3f}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de {symbol}: {e}")
    
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
                limit=limit
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
            model_path = 'models_store/logistic_regression_mvp.joblib'
            if not os.path.exists(model_path):
                logger.warning(f"Modèle non trouvé: {model_path}")
                return None
            
            prediction = load_model_and_predict(X, model_path=model_path, return_probabilities=True)
            
            if len(prediction) > 0:
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
            signal = signal_data['signal']
            price = signal_data['price']
            
            # Seuils de trading
            buy_threshold = 0.3
            sell_threshold = -0.3
            
            # Vérifier les limites de risque
            if not self._check_risk_limits(symbol, signal):
                return
            
            # Déterminer l'action à prendre
            if signal > buy_threshold:
                await self._execute_buy_signal(symbol, signal, price)
            elif signal < sell_threshold:
                await self._execute_sell_signal(symbol, signal, price)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du signal: {e}")
    
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
            if self.profit_loss_today <= -self.config['initial_capital'] * self.config['max_daily_loss']:
                logger.warning(f"Limite de perte journalière atteinte: {self.profit_loss_today:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des risques: {e}")
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
                success = await self.trader.place_order(order)
                if success:
                    logger.info(f"🟢 Ordre d'achat placé: {symbol} - {quantity:.6f} @ {price:.4f}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal d'achat: {e}")
    
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
                    logger.info(f"🔴 Ordre de vente placé: {symbol} - {quantity:.6f} @ {price:.4f}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal de vente: {e}")
    
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
            model_path = 'models_store/logistic_regression_mvp.joblib'
            metrics = train_model(
                X, y,
                model_type='logistic_regression',
                model_path=model_path,
                scale_features=True
            )
            
            self.last_model_update = datetime.now()
            logger.info(f"✅ Modèle réentraîné avec succès. Accuracy: {metrics.get('accuracy', 'N/A'):.3f}")
            
        except Exception as e:
            logger.error(f"Erreur lors du réentraînement du modèle: {e}")
    
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
🔄 Ordres totaux: {total_orders}
✅ Taux de remplissage: {fill_rate:.1%}
💰 P&L journalier: ${self.profit_loss_today:.2f}
📈 Paires actives: {len(self.trading_pairs)}
🎯 Signaux traités: {self.signal_queue.qsize()}
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
    
    async def _reduce_positions(self):
        """Réduit les positions en cas de dépassement des limites de risque"""
        try:
            # Annuler tous les ordres en attente
            await self._cancel_all_open_orders()
            
            # Vendre partiellement les positions importantes
            for asset, balance in self.trader.account_balances.items():
                if balance.total > 0 and asset != 'USDT':
                    symbol = f"{asset}USDT"
                    if symbol in self.trading_pairs:
                        # Vendre 50% de la position
                        quantity = balance.free * 0.5
                        quantity = self._round_quantity(symbol, quantity)
                        
                        if quantity > 0:
                            order = TradingOrder(
                                symbol=symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=quantity,
                                client_order_id=f"risk_sell_{symbol}_{int(time.time())}"
                            )
                            
                            await self.trader.place_order(order)
                            logger.info(f"🔻 Position réduite pour {symbol}: {quantity:.6f}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la réduction des positions: {e}")
    
    async def _cancel_all_open_orders(self):
        """Annule tous les ordres ouverts"""
        try:
            for order in list(self.trader.open_orders.values()):
                await self.trader.cancel_order(order.order_id)
                logger.info(f"❌ Ordre annulé: {order.symbol} - {order.client_order_id}")
        
        except Exception as e:
            logger.error(f"Erreur lors de l'annulation des ordres: {e}")
    
    async def _on_market_data(self, market_data: MarketData):
        """Callback pour les données de marché"""
        # Utiliser les données en temps réel pour ajuster les stratégies
        pass
    
    async def _on_order_update(self, order: TradingOrder):
        """Callback pour les mises à jour d'ordres"""
        if order.status == OrderStatus.FILLED:
            logger.info(f"✅ Ordre exécuté: {order.symbol} {order.side.value} {order.filled_quantity:.6f}")
            self.trades_today += 1
    
    async def _on_account_update(self, balances: Dict[str, any]):
        """Callback pour les mises à jour de compte"""
        # Mettre à jour les métriques de performance
        pass

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
