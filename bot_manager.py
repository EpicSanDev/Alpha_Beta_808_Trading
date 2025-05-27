#!/usr/bin/env python3
"""
Gestionnaire du Bot de Trading AlphaBeta808
Script pour démarrer, arrêter et surveiller le bot de trading
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
# from datetime import datetime # F401: Inutilisé
from typing import Dict, Optional


class TradingBotManager:
    """
    Gestionnaire pour le bot de trading
    """

    def __init__(self):
        self.bot_process: Optional[subprocess.Popen] = None
        self.pid_file = "logs/trading_bot.pid"
        self.log_file = "logs/trading_bot.log"
        self.stats_file = "logs/bot_stats.json"

        # Créer le dossier de logs
        os.makedirs("logs", exist_ok=True)

    def start_bot(self, background: bool = False) -> bool:
        """
        Démarre le bot de trading

        Args:
            background: Démarrer en arrière-plan

        Returns:
            True si le démarrage a réussi
        """
        if self.is_running():
            print("Le bot de trading est déjà en cours d'exécution")
            return False

        print("Démarrage du bot de trading AlphaBeta808...")

        try:
            # Commande pour démarrer le bot
            cmd = [sys.executable, "live_trading_bot.py"]

            if background:
                # Démarrer en arrière-plan
                with open(self.log_file, 'a') as log:
                    self.bot_process = subprocess.Popen(
                        cmd,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True
                    )

                # Sauvegarder le PID
                with open(self.pid_file, 'w') as f:
                    f.write(str(self.bot_process.pid))

                print(f"Bot démarré en arrière-plan avec PID: "
                      f"{self.bot_process.pid}")
                print(f"Logs disponibles dans: {self.log_file}")

            else:
                # Démarrer en premier plan
                self.bot_process = subprocess.Popen(cmd)
                self.bot_process.wait()

            return True

        except Exception as e:
            print(f"Erreur lors du démarrage du bot: {e}")
            return False

    def stop_bot(self) -> bool:
        """
        Arrête le bot de trading

        Returns:
            True si l'arrêt a réussi
        """
        if not self.is_running():
            print("Le bot de trading n'est pas en cours d'exécution")
            return False

        print("Arrêt du bot de trading...")

        try:
            pid = self.get_bot_pid()
            if pid:
                # Envoyer signal SIGTERM pour arrêt propre
                os.kill(pid, signal.SIGTERM)

                # Attendre que le processus se termine
                timeout = 30
                for _ in range(timeout):
                    if not self.is_running():
                        break
                    time.sleep(1)

                # Si toujours en cours, forcer l'arrêt
                if self.is_running():
                    print("Arrêt forcé du bot...")
                    os.kill(pid, signal.SIGKILL)

                # Nettoyer le fichier PID
                if os.path.exists(self.pid_file):
                    os.remove(self.pid_file)

                print("Bot arrêté avec succès")
                return True

        except Exception as e:
            print(f"Erreur lors de l'arrêt du bot: {e}")
            return False

        return False

    def restart_bot(self) -> bool:
        """
        Redémarre le bot de trading

        Returns:
            True si le redémarrage a réussi
        """
        print("Redémarrage du bot de trading...")

        if self.is_running():
            if not self.stop_bot():
                return False

        time.sleep(2)  # Attendre un peu avant de redémarrer

        return self.start_bot(background=True)

    def is_running(self) -> bool:
        """
        Vérifie si le bot est en cours d'exécution

        Returns:
            True si le bot est en cours d'exécution
        """
        pid = self.get_bot_pid()
        if not pid:
            return False

        try:
            # Vérifier si le processus existe
            os.kill(pid, 0)
            return True
        except OSError:
            # Le processus n'existe plus
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            return False

    def get_bot_pid(self) -> Optional[int]:
        """
        Récupère le PID du bot

        Returns:
            PID du bot ou None
        """
        try:
            if os.path.exists(self.pid_file):
                with open(self.pid_file, 'r') as f:
                    return int(f.read().strip())
        except (ValueError, FileNotFoundError):
            pass

        return None

    def get_status(self) -> Dict:
        """
        Récupère le statut du bot

        Returns:
            Dictionnaire avec le statut
        """
        status = {
            'is_running': self.is_running(),
            'pid': self.get_bot_pid(),
            'start_time': None,
            'uptime': None,
            'stats': {}
        }

        # Charger les statistiques si disponibles
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    stats = json.load(f)
                    status['stats'] = stats

                    if 'start_time' in stats:
                        status['start_time'] = stats['start_time']

                    if 'uptime_seconds' in stats:
                        status['uptime'] = self._format_uptime(
                            stats['uptime_seconds']
                        )

        except Exception as e:
            print(f"Erreur lors du chargement des stats: {e}")

        return status

    def _format_uptime(self, seconds: float) -> str:
        """
        Formate la durée de fonctionnement

        Args:
            seconds: Durée en secondes

        Returns:
            Durée formatée
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def show_status(self):
        """Affiche le statut du bot"""
        status = self.get_status()

        print("="*60)
        print("STATUT DU BOT DE TRADING ALPHABETA808")
        print("="*60)

        if status['is_running']:
            print("État: ✅ EN COURS D'EXÉCUTION")  # F541 corrigé, E261 corrigé
            print(f"PID: {status['pid']}")

            if status['start_time']:
                print(f"Démarré: {status['start_time']}")

            if status['uptime']:
                print(f"Durée: {status['uptime']}")

        else:
            print("État: ❌ ARRÊTÉ")

        # Afficher les statistiques si disponibles
        if status['stats']:
            stats = status['stats']
            print("\n" + "-"*60)
            print("STATISTIQUES")
            print("-"*60)

            if 'total_signals' in stats:
                print(f"Signaux générés: {stats['total_signals']}")

            if 'total_trades' in stats:
                print(f"Trades exécutés: {stats['total_trades']}")

            if 'errors' in stats:
                print(f"Erreurs: {stats['errors']}")

            if 'trading_summary' in stats:
                summary = stats['trading_summary']
                if summary:
                    print(f"Ordres ouverts: {summary.get('open_orders', 0)}")
                    print(
                        f"Ordres complétés: "
                        f"{summary.get('filled_orders', 0)}"
                    )
                    subscribed_symbols = summary.get('subscribed_symbols', [])
                    print(f"Symboles surveillés: {subscribed_symbols}")

        print("="*60)

    def show_logs(self, lines: int = 50):
        """
        Affiche les dernières lignes du log

        Args:
            lines: Nombre de lignes à afficher
        """
        if not os.path.exists(self.log_file):
            print("Aucun fichier de log trouvé")
            return

        try:
            with open(self.log_file, 'r') as f:
                log_lines = f.readlines()

            # Afficher les dernières lignes
            recent_lines = log_lines[-lines:] \
                if len(log_lines) > lines else log_lines

            print(
                f"Dernières {len(recent_lines)} lignes du log:"
            )
            print("-" * 60)
            for line in recent_lines:
                print(line.rstrip())

        except Exception as e:
            print(f"Erreur lors de la lecture du log: {e}")

    def monitor(self, refresh_interval: int = 30):
        """
        Mode monitoring en temps réel

        Args:
            refresh_interval: Intervalle de rafraîchissement en secondes
        """
        print("Mode monitoring activé (Ctrl+C pour quitter)")

        try:
            while True:
                # Effacer l'écran
                os.system('clear' if os.name == 'posix' else 'cls')

                # Afficher le statut
                self.show_status()

                # E501: Ligne coupée
                print(f"\nMise à jour automatique toutes les "
                      f"{refresh_interval}s")
                print("Appuyez sur Ctrl+C pour quitter le monitoring")

                # Attendre avant la prochaine mise à jour
                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nMonitoring arrêté")


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Gestionnaire du Bot de Trading AlphaBeta808"
    )

    parser.add_argument(
        'action',
        choices=['start', 'stop', 'restart', 'status', 'logs', 'monitor'],
        help='Action à effectuer'
    )

    parser.add_argument(
        '--background', '-b',
        action='store_true',
        help='Démarrer en arrière-plan (pour start)'
    )

    parser.add_argument(
        '--lines', '-n',
        type=int,
        default=50,
        help='Nombre de lignes à afficher (pour logs)'
    )

    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=30,
        help='Intervalle de rafraîchissement en secondes (pour monitor)'
    )

    args = parser.parse_args()

    manager = TradingBotManager()

    if args.action == 'start':
        success = manager.start_bot(background=args.background)
        sys.exit(0 if success else 1)

    elif args.action == 'stop':
        success = manager.stop_bot()
        sys.exit(0 if success else 1)

    elif args.action == 'restart':
        success = manager.restart_bot()
        sys.exit(0 if success else 1)

    elif args.action == 'status':
        manager.show_status()

    elif args.action == 'logs':
        manager.show_logs(lines=args.lines)

    elif args.action == 'monitor':
        manager.monitor(refresh_interval=args.interval)


if __name__ == "__main__":
    main()
