#!/usr/bin/env python3
"""
Live Trading Bot - AlphaBeta808
Bot de trading en temps réel avec intégration ML
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from typing import Dict, Optional

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from continuous_trader import ContinuousTrader

# Configuration du logging avec gestion des systèmes de fichiers en lecture seule
def setup_logging():
    handlers = [logging.StreamHandler()]
    
    # Essayer d'ajouter un FileHandler si possible
    try:
        os.makedirs('logs', exist_ok=True)
        handlers.append(logging.FileHandler('logs/trading_bot.log'))
    except (OSError, PermissionError) as e:
        print(f"Warning: Cannot create log file (read-only filesystem?): {e}")
        print("Logging to console only")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

setup_logging()
logger = logging.getLogger(__name__)

class LiveTradingBot:
    """
    Bot de trading en temps réel basé sur ContinuousTrader
    """
    
    def __init__(self, config_file: str = "trader_config.json"):
        """
        Initialise le bot de trading en temps réel
        
        Args:
            config_file: Chemin vers le fichier de configuration
        """
        self.config_file = config_file
        self.trader: Optional[ContinuousTrader] = None
        self.is_running = False
        
        logger.info(f"📈 Initialisation du Live Trading Bot avec config: {config_file}")
    
    async def start(self):
        """Démarre le bot de trading"""
        try:
            logger.info("🚀 Démarrage du Live Trading Bot...")
            
            # Créer et initialiser le trader
            self.trader = ContinuousTrader(self.config_file)
            await self.trader.initialize()
            
            # Configurer les signaux pour arrêt propre
            self._setup_signal_handlers()
            
            # Démarrer le trading
            self.is_running = True
            logger.info("✅ Bot de trading démarré avec succès!")
            
            await self.trader.start_trading()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"❌ Erreur fatale dans le bot: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Arrête le bot de trading"""
        if self.trader and self.is_running:
            logger.info("⏹️ Arrêt du bot de trading...")
            await self.trader.stop_trading()
            self.is_running = False
            logger.info("✅ Bot arrêté proprement")
    
    def _setup_signal_handlers(self):
        """Configure les gestionnaires de signaux pour arrêt propre"""
        def signal_handler(signum, frame):
            logger.info(f"📡 Signal {signum} reçu, arrêt en cours...")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_status(self) -> Dict:
        """Retourne le statut du bot"""
        if not self.trader:
            return {
                "status": "not_initialized",
                "is_running": False,
                "current_time": datetime.now().isoformat()
            }
        
        trader_status = self.trader.get_status()
        trader_status["bot_type"] = "live_trading_bot"
        trader_status["config_file"] = self.config_file
        
        return trader_status

async def main():
    """Fonction principale"""
    # Configuration par défaut
    config_file = "trader_config.json"
    
    # Permettre de spécifier un fichier de config via argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    # Créer et démarrer le bot
    bot = LiveTradingBot(config_file)
    await bot.start()

if __name__ == "__main__":
    print("🤖 AlphaBeta808 Live Trading Bot")
    print("=" * 50)
    print("Bot de trading automatique avec ML")
    print("Ctrl+C pour arrêter")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Arrêt du bot demandé")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
