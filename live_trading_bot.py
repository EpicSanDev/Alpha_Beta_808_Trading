#!/usr/bin/env python3
"""
Bot de Trading AlphaBeta808 - Trading Live 24h/24 7j/7
Système de trading automatisé avec Binance
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ajouter le répertoire src au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Imports des modules du projet
from src.execution.real_time_trading import (
    BinanceRealTimeTrader, TradingStrategy, RiskManager,
    MarketData, TradingOrder, OrderType, OrderSide, OrderStatus
)
from src.feature_engineering.technical_features import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_price_momentum, calculate_volume_features
)
from src.modeling.models import load_model_and_predict
from src.signal_generation.signal_generator import generate_signals_from_predictions

class LiveTradingBot:
    """
    Bot de trading principal pour le trading 24h/24 7j/7
    """
    
    def __init__(self, config_path: str = "trader_config.json"):
        """
        Initialise le bot de trading
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialiser le logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Données de marché historiques pour les features
        self.market_history = {}  # Dict[str, pd.DataFrame]
        self.last_predictions = {}
        self.last_feature_update = {}
        
        # Statistiques
        self.stats = {
            'start_time': None,
            'total_signals': 0,
            'total_trades': 0,
            'successful_trades': 0,
            'errors': 0,
            'last_health_check': None
        }
        
        # Initialiser les composants
        self._initialize_components()
        
        # Gestionnaire de signaux pour arrêt propre
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis un fichier JSON"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Fichier de configuration {config_path} non trouvé, utilisation des valeurs par défaut")
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Configuration par défaut"""
        return {
            "trading": {
                "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
                "testnet": True,
                "max_position_size": 0.05,
                "signal_update_interval": 60,
                "feature_lookback_periods": 200
            },
            "risk_management": {
                "max_daily_loss": 0.02,
                "max_total_exposure": 0.8,
                "max_orders_per_minute": 5,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.10
            },
            "model": {
                "model_path": "models_store/logistic_regression_mvp.joblib",
                "retrain_interval_hours": 24,
                "min_confidence": 0.6
            },
            "logging": {
                "level": "INFO",
                "file": "logs/trading_bot.log",
                "max_size_mb": 50
            }
        }
    
    def _setup_logging(self):
        """Configure le système de logging"""
        log_level = getattr(logging, self.config["logging"]["level"].upper())
        log_file = self.config["logging"]["file"]
        
        # Créer le dossier de logs s'il n'existe pas
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _initialize_components(self):
        """Initialise tous les composants du bot"""
        load_dotenv()
        
        # Récupérer les clés API
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Clés API Binance non configurées. Vérifiez votre fichier .env")
        
        # Initialiser le gestionnaire de risques
        risk_config = self.config["risk_management"]
        self.risk_manager = RiskManager(
            max_position_size=risk_config["max_position_size"],
            max_daily_loss=risk_config["max_daily_loss"],
            max_total_exposure=risk_config["max_total_exposure"],
            max_orders_per_minute=risk_config["max_orders_per_minute"]
        )
        
        # Initialiser le trader Binance
        self.trader = BinanceRealTimeTrader(
            api_key=api_key,
            api_secret=api_secret,
            testnet=self.config["trading"]["testnet"],
            risk_manager=self.risk_manager
        )
        
        # Initialiser la stratégie de trading
        self.strategy = TradingStrategy(
            trader=self.trader,
            signal_generator=self._generate_trading_signal
        )
        
        self.logger.info("Composants du bot initialisés avec succès")
    
    def _generate_trading_signal(self, symbol: str, market_data: MarketData) -> float:
        """
        Génère un signal de trading basé sur les données de marché
        
        Args:
            symbol: Symbole de trading
            market_data: Données de marché en temps réel
            
        Returns:
            Signal de trading entre -1 et 1
        """
        try:
            # Mettre à jour l'historique des données
            self._update_market_history(symbol, market_data)
            
            # Vérifier si on a assez de données pour calculer les features
            if symbol not in self.market_history or len(self.market_history[symbol]) < 50:
                return 0.0  # Signal neutre si pas assez de données
            
            # Calculer les features techniques si nécessaire
            should_update = self._should_update_features(symbol)
            if should_update:
                self._calculate_technical_features(symbol)
                self.last_feature_update[symbol] = datetime.now()
            
            # Générer une prédiction avec le modèle
            signal = self._get_model_prediction(symbol)
            
            # Appliquer des filtres de signal
            signal = self._apply_signal_filters(symbol, signal, market_data)
            
            self.stats['total_signals'] += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du signal pour {symbol}: {e}")
            self.stats['errors'] += 1
            return 0.0
    
    def _update_market_history(self, symbol: str, market_data: MarketData):
        """Met à jour l'historique des données de marché"""
        if symbol not in self.market_history:
            self.market_history[symbol] = pd.DataFrame()
        
        # Ajouter la nouvelle donnée
        new_data = pd.DataFrame([{
            'timestamp': market_data.timestamp,
            'open': market_data.price,  # Approximation
            'high': market_data.price,  # Approximation
            'low': market_data.price,   # Approximation
            'close': market_data.price,
            'volume': market_data.volume_24h,
            'price_change': market_data.price_change_24h
        }])
        
        self.market_history[symbol] = pd.concat([self.market_history[symbol], new_data], ignore_index=True)
        
        # Garder seulement les N dernières périodes
        max_periods = self.config["trading"]["feature_lookback_periods"]
        if len(self.market_history[symbol]) > max_periods:
            self.market_history[symbol] = self.market_history[symbol].tail(max_periods).reset_index(drop=True)
    
    def _should_update_features(self, symbol: str) -> bool:
        """Détermine si les features doivent être recalculées"""
        if symbol not in self.last_feature_update:
            return True
        
        # Recalculer toutes les minutes
        time_since_update = datetime.now() - self.last_feature_update[symbol]
        return time_since_update.total_seconds() > self.config["trading"]["signal_update_interval"]
    
    def _calculate_technical_features(self, symbol: str):
        """Calcule les features techniques pour un symbole"""
        if symbol not in self.market_history or len(self.market_history[symbol]) < 20:
            return
        
        df = self.market_history[symbol].copy()
        
        # Features de base
        df = calculate_sma(df, column='close', windows=[10, 20]) # MODIFIÉ: sma_50 retiré
        df = calculate_ema(df, column='close', windows=[10, 20])
        df = calculate_rsi(df, column='close', window=14)
        
        # Features avancées
        df = calculate_macd(df, column='close')
        df = calculate_bollinger_bands(df, column='close')
        df = calculate_price_momentum(df, column='close', windows=[5, 10])

        # Features de volume si disponible (AJOUTÉ)
        if 'volume' in df.columns and 'close' in df.columns:
            df = calculate_volume_features(df, volume_col='volume', price_col='close', windows=[10, 20])
        
        # Nettoyer les NaN
        df.dropna(inplace=True)
        
        self.market_history[symbol] = df
    
    def _get_model_prediction(self, symbol: str) -> float:
        """Obtient une prédiction du modèle ML"""
        try:
            model_path = self.config["model"]["model_path"]
            
            if not os.path.exists(model_path):
                self.logger.warning(f"Modèle non trouvé: {model_path}")
                return 0.0
            
            df = self.market_history[symbol]
            if len(df) < 5:
                return 0.0
            
            # Préparer les features pour la prédiction
            exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'price_change']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            if not feature_cols:
                return 0.0
            
            # Utiliser les dernières données
            X = df[feature_cols].tail(1)
            
            if X.empty or X.isnull().any().any():
                return 0.0
            
            # Faire la prédiction
            prediction = load_model_and_predict(X, model_path=model_path, return_probabilities=True)
            
            if len(prediction) == 0:
                return 0.0
            
            # Convertir la probabilité en signal (-1 à 1)
            prob = prediction[0]
            signal = (prob - 0.5) * 2  # Normaliser de [0,1] à [-1,1]
            
            # Appliquer le seuil de confiance
            min_confidence = self.config["model"]["min_confidence"]
            if abs(signal) < min_confidence:
                signal = 0.0
            
            self.last_predictions[symbol] = {
                'probability': prob,
                'signal': signal,
                'timestamp': datetime.now()
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction pour {symbol}: {e}")
            return 0.0
    
    def _apply_signal_filters(self, symbol: str, signal: float, market_data: MarketData) -> float:
        """Applique des filtres au signal généré"""
        # Filtre de volatilité - réduire les signaux en cas de forte volatilité
        if abs(market_data.price_change_24h / market_data.price) > 0.1:  # >10% de variation
            signal *= 0.5
        
        # Filtre de spread - réduire les signaux si le spread est trop large
        spread = (market_data.ask_price - market_data.bid_price) / market_data.price
        if spread > 0.001:  # Spread > 0.1%
            signal *= 0.7
        
        # Filtre de volume - renforcer les signaux avec du volume
        # (Ici simplifié, dans un vrai cas on comparerait au volume moyen)
        if market_data.volume_24h > 1000000:  # Volume élevé
            signal *= 1.1
        
        # S'assurer que le signal reste dans [-1, 1]
        return max(-1.0, min(1.0, signal))
    
    async def _health_check(self):
        """Vérifie la santé du système de trading"""
        try:
            # Vérifier la connectivité
            account_info = self.trader.get_account_info()
            if not account_info:
                self.logger.error("Échec de la vérification de connectivité")
                return False
            
            # Vérifier l'état des ordres
            open_orders = self.trader.get_open_orders()
            self.logger.info(f"Ordres ouverts: {len(open_orders)}")
            
            # Vérifier les modèles
            model_path = self.config["model"]["model_path"]
            if not os.path.exists(model_path):
                self.logger.warning("Modèle de trading manquant")
            
            # Mettre à jour les statistiques
            self.stats['last_health_check'] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de santé: {e}")
            return False
    
    async def _periodic_tasks(self):
        """Tâches périodiques du bot"""
        while self.is_running:
            try:
                # Vérification de santé toutes les 5 minutes
                await self._health_check()
                
                # Reset des compteurs journaliers à minuit
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    self.risk_manager.reset_daily_counters()
                    self.logger.info("Compteurs journaliers remis à zéro")
                
                # Sauvegarde des statistiques
                self._save_stats()
                
                # Attendre 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Erreur dans les tâches périodiques: {e}")
                await asyncio.sleep(60)
    
    def _save_stats(self):
        """Sauvegarde les statistiques du bot"""
        try:
            stats_file = "logs/bot_stats.json"
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            
            current_stats = {
                **self.stats,
                'current_time': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0,
                'trading_summary': self.trader.get_trading_summary() if hasattr(self, 'trader') else {}
            }
            
            with open(stats_file, 'w') as f:
                json.dump(current_stats, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des stats: {e}")
    
    def _signal_handler(self, signum, frame):
        """Gestionnaire pour les signaux système"""
        self.logger.info(f"Signal {signum} reçu, arrêt du bot...")
        self.shutdown_event.set()
    
    async def start(self):
        """Démarre le bot de trading"""
        try:
            self.logger.info("Démarrage du bot de trading AlphaBeta808...")
            self.is_running = True
            self.stats['start_time'] = datetime.now()
            
            # Récupérer les informations du compte
            account_info = self.trader.get_account_info()
            if account_info:
                self.logger.info("Compte Binance connecté avec succès")
            else:
                raise Exception("Impossible de se connecter au compte Binance")
            
            # Démarrer la stratégie de trading
            symbols = self.config["trading"]["symbols"]
            self.strategy.start_trading(symbols)
            
            self.logger.info(f"Trading démarré pour les symboles: {symbols}")
            
            # Démarrer les tâches périodiques
            periodic_task = asyncio.create_task(self._periodic_tasks())
            
            # Boucle principale - attendre l'arrêt
            await self.shutdown_event.wait()
            
            # Arrêt propre
            self.logger.info("Arrêt du bot en cours...")
            self.is_running = False
            
            # Annuler les tâches
            periodic_task.cancel()
            
            # Arrêter la stratégie
            self.strategy.stop_trading()
            
            # Annuler tous les ordres ouverts
            open_orders = self.trader.get_open_orders()
            for order in open_orders:
                self.trader.cancel_order(order.symbol, order.order_id)
            
            # Sauvegarder les statistiques finales
            self._save_stats()
            
            self.logger.info("Bot de trading arrêté proprement")
            
        except Exception as e:
            self.logger.error(f"Erreur fatale dans le bot de trading: {e}")
            raise
    
    def get_status(self) -> Dict:
        """Retourne le statut actuel du bot"""
        return {
            'is_running': self.is_running,
            'uptime': (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0,
            'stats': self.stats,
            'trading_summary': self.trader.get_trading_summary() if hasattr(self, 'trader') else {},
            'subscribed_symbols': list(self.strategy.trader.subscribed_symbols) if hasattr(self, 'strategy') else [],
            'last_predictions': {k: v for k, v in self.last_predictions.items()},
            'config': self.config
        }


async def main():
    """Fonction principale"""
    try:
        # Créer et démarrer le bot
        bot = LiveTradingBot()
        await bot.start()
        
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Démarrer le bot
    asyncio.run(main())
