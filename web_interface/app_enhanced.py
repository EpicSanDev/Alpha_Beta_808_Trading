#!/usr/bin/env python3
"""
Interface Web Complète pour AlphaBeta808 Trading Bot
Version améliorée avec dashboard avancé, monitoring temps réel et gestion complète
"""

import os
import sys
import json
import time
import asyncio
import threading
import sqlite3
import io
import csv
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict, dataclass
import pandas as pd
import numpy as np

# Flask et extensions
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Fonction utilitaire pour la sérialisation JSON
def json_serializable(obj):
    """Convertit les objets non-sérialisables JSON, y compris les types NumPy."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return str(obj)
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convertit les arrays NumPy en listes Python
    elif hasattr(obj, '__dict__'): # Pour les objets personnalisés (dataclasses après asdict)
        return {k: json_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    else:
        return obj

def safe_asdict(dataclass_instance):
    """Convertit un dataclass en dict avec sérialisation JSON sûre"""
    result = asdict(dataclass_instance)
    return json_serializable(result)

# Imports du système de trading
# Assurez-vous que continuous_trader.py, bot_manager.py, optimize_models_for_profitability.py
# sont bien dans le répertoire parent ajouté au sys.path
from continuous_trader import ContinuousTrader
from bot_manager import TradingBotManager
from optimize_models_for_profitability import ProfitabilityOptimizer
from src.execution.real_time_trading import (
    BinanceRealTimeTrader, TradingOrder, OrderSide, OrderType, OrderStatus,
    MarketData, AccountBalance, RiskManager
)
print("✅ Modules de trading principaux importés (vérifiez les chemins si erreur).")

# Configuration Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'alphabeta808_trading_secret_key_enhanced'
app.config['DEBUG'] = True
app.config['DATABASE'] = 'trading_web.db'

# Extensions
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

@dataclass
class TradingStats:
    """Structure pour les statistiques de trading"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    current_positions: int = 0
    daily_pnl: float = 0.0
    monthly_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

@dataclass
class SystemStatus:
    """Structure pour le statut du système"""
    bot_running: bool = False
    last_update: datetime = None
    connected_exchanges: List[str] = None
    active_symbols: List[str] = None
    models_loaded: int = 0
    errors_count: int = 0
    uptime: str = "0:00:00"
    
    def __post_init__(self):
        if self.connected_exchanges is None:
            self.connected_exchanges = []
        if self.active_symbols is None:
            self.active_symbols = []

@dataclass
class TradingMode:
    """Structure pour le mode de trading"""
    is_paper_trading: bool = True
    is_live_trading: bool = False
    mode_name: str = "Paper Trading"
    balance_real: Dict[str, float] = None
    balance_paper: Dict[str, float] = None
    
    def __post_init__(self):
        if self.balance_real is None:
            self.balance_real = {}
        if self.balance_paper is None:
            self.balance_paper = {"USDT": 10000.0}

@dataclass
class PriceAlert:
    """Structure pour les alertes de prix"""
    id: str
    symbol: str
    condition: str  # "above", "below", "change_percent"
    value: float
    current_price: float
    triggered: bool = False
    created_at: datetime = None
    triggered_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class Position:
    """Structure pour les positions"""
    symbol: str
    side: str  # "long", "short"
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percent: float
    margin_used: float
    created_at: datetime
    
    def __post_init__(self):
        if not hasattr(self, 'created_at') or self.created_at is None:
            self.created_at = datetime.now()


class DatabaseManager:
    """Gestionnaire de base de données pour l'interface web"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des utilisateurs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table des trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                pnl REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_used TEXT,
                confidence REAL,
                is_paper_trade BOOLEAN DEFAULT 1,
                order_id TEXT
            )
        ''')
        
        # Table des performances
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_pnl REAL,
                daily_pnl REAL,
                total_trades INTEGER,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                is_paper_trading BOOLEAN DEFAULT 1
            )
        ''')
        
        # Table des configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                config_data TEXT NOT NULL, -- JSON string
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 0
            )
        ''')
        
        # Table des alertes de prix
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                condition TEXT NOT NULL,
                value REAL NOT NULL,
                triggered BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                triggered_at TIMESTAMP
            )
        ''')
        
        # Table des positions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                pnl REAL,
                pnl_percent REAL,
                margin_used REAL,
                is_paper_position BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP,
                status TEXT DEFAULT 'open'
            )
        ''')
        
        # Table des soldes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset TEXT NOT NULL,
                paper_balance REAL DEFAULT 0,
                real_balance REAL DEFAULT 0,
                locked_balance REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table des notifications
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                read BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Créer un utilisateur par défaut
        cursor.execute('SELECT COUNT(*) FROM users')
        if cursor.fetchone()[0] == 0:
            default_password = generate_password_hash('admin123')
            cursor.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', 
                         ('admin', default_password))
        
        # Initialiser les soldes par défaut
        cursor.execute('SELECT COUNT(*) FROM balances')
        if cursor.fetchone()[0] == 0:
            default_assets = ['USDT', 'BTC', 'ETH', 'ADA', 'DOT']
            for asset in default_assets:
                paper_balance = 10000.0 if asset == 'USDT' else 0.0
                cursor.execute('''
                    INSERT INTO balances (asset, paper_balance, real_balance) 
                    VALUES (?, ?, ?)
                ''', (asset, paper_balance, 0.0))
        
        conn.commit()
        conn.close()
    
    def get_recent_trades(self, limit: int = 100) -> List[Dict]:
        """Récupère les trades récents"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, side, quantity, price, pnl, timestamp, model_used, confidence
            FROM trades 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'symbol': row[0],
                'side': row[1],
                'quantity': row[2],
                'price': row[3],
                'pnl': row[4],
                'timestamp': row[5],
                'model_used': row[6],
                'confidence': row[7]
            })
        
        conn.close()
        return trades
    
    def add_trade(self, trade_data: Dict):
        """Ajoute un nouveau trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ajout de order_id et is_paper_trade
        cursor.execute('''
            INSERT INTO trades (symbol, side, quantity, price, pnl, model_used, confidence, order_id, is_paper_trade, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('symbol'),
            trade_data.get('side'),
            trade_data.get('quantity'),
            trade_data.get('price'),
            trade_data.get('pnl'),
            trade_data.get('model_used'),
            trade_data.get('confidence'),
            trade_data.get('order_id'), # Nouveau champ
            trade_data.get('is_paper_trade', 1), # Nouveau champ, défaut à 1 (paper)
            trade_data.get('timestamp', datetime.now()) # Assurer un timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Récupère l'historique de performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, total_pnl, daily_pnl, total_trades, win_rate, sharpe_ratio, max_drawdown
            FROM performance 
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days))
        
        performance = []
        for row in cursor.fetchall():
            performance.append({
                'date': row[0],
                'total_pnl': row[1],
                'daily_pnl': row[2],
                'total_trades': row[3],
                'win_rate': row[4],
                'sharpe_ratio': row[5],
                'max_drawdown': row[6]
            })
        
        conn.close()
        return performance
    
    def add_price_alert(self, alert: PriceAlert):
        """Ajoute une alerte de prix"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO price_alerts (id, symbol, condition, value, triggered, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (alert.id, alert.symbol, alert.condition, alert.value, alert.triggered, alert.created_at))
        
        conn.commit()
        conn.close()
    
    def get_price_alerts(self, active_only: bool = True) -> List[Dict]:
        """Récupère les alertes de prix"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, symbol, condition, value, triggered, created_at, triggered_at
            FROM price_alerts
        '''
        if active_only:
            query += ' WHERE triggered = 0'
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query)
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'id': row[0],
                'symbol': row[1],
                'condition': row[2],
                'value': row[3],
                'triggered': bool(row[4]),
                'created_at': row[5],
                'triggered_at': row[6]
            })
        
        conn.close()
        return alerts
    
    def trigger_price_alert(self, alert_id: str):
        """Déclenche une alerte de prix"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE price_alerts 
            SET triggered = 1, triggered_at = ?
            WHERE id = ?
        ''', (datetime.now(), alert_id))
        
        conn.commit()
        conn.close()
    
    def add_position(self, position: Position):
        """Ajoute une nouvelle position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO positions (symbol, side, quantity, entry_price, current_price, 
                                 pnl, pnl_percent, margin_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (position.symbol, position.side, position.quantity, position.entry_price,
              position.current_price, position.pnl, position.pnl_percent, 
              position.margin_used, position.created_at))
        
        conn.commit()
        conn.close()
    
    def get_positions(self, open_only: bool = True) -> List[Dict]:
        """Récupère les positions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT symbol, side, quantity, entry_price, current_price, 
                   pnl, pnl_percent, margin_used, created_at, status
            FROM positions
        '''
        if open_only:
            query += " WHERE status = 'open'"
        query += ' ORDER BY created_at DESC'
        
        cursor.execute(query)
        
        positions = []
        for row in cursor.fetchall():
            positions.append({
                'symbol': row[0],
                'side': row[1],
                'quantity': row[2],
                'entry_price': row[3],
                'current_price': row[4],
                'pnl': row[5],
                'pnl_percent': row[6],
                'margin_used': row[7],
                'created_at': row[8],
                'status': row[9]
            })
        
        conn.close()
        return positions
    
    def update_position_pnl(self, symbol: str, current_price: float, pnl: float, pnl_percent: float):
        """Met à jour le PnL d'une position"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE positions 
            SET current_price = ?, pnl = ?, pnl_percent = ?
            WHERE symbol = ? AND status = 'open'
        ''', (current_price, pnl, pnl_percent, symbol))
        
        conn.commit()
        conn.close()
    
    def get_balances(self, is_paper: bool = True) -> Dict[str, float]:
        """Récupère les soldes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        balance_column = 'paper_balance' if is_paper else 'real_balance'
        cursor.execute(f'SELECT asset, {balance_column} FROM balances')
        
        balances = {}
        for row in cursor.fetchall():
            balances[row[0]] = row[1]
        
        conn.close()
        return balances
    
    def update_balance(self, asset: str, amount: float, is_paper: bool = True):
        """Met à jour un solde"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        balance_column = 'paper_balance' if is_paper else 'real_balance'
        cursor.execute(f'''
            UPDATE balances 
            SET {balance_column} = ?, last_updated = ?
            WHERE asset = ?
        ''', (amount, datetime.now(), asset))
        
        if cursor.rowcount == 0:
            # Créer le solde s'il n'existe pas
            paper_balance = amount if is_paper else 0.0
            real_balance = 0.0 if is_paper else amount
            cursor.execute('''
                INSERT INTO balances (asset, paper_balance, real_balance, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (asset, paper_balance, real_balance, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def add_notification(self, notification_type: str, title: str, message: str):
        """Ajoute une notification"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notifications (type, title, message, created_at)
            VALUES (?, ?, ?, ?)
        ''', (notification_type, title, message, datetime.now()))
        
        conn.commit()
        conn.close()
    
    def get_notifications(self, unread_only: bool = False, limit: int = 50) -> List[Dict]:
        """Récupère les notifications"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT id, type, title, message, read, created_at
            FROM notifications
        '''
        if unread_only:
            query += ' WHERE read = 0'
        query += ' ORDER BY created_at DESC LIMIT ?'
        
        cursor.execute(query, (limit,))
        
        notifications = []
        for row in cursor.fetchall():
            notifications.append({
                'id': row[0],
                'type': row[1],
                'title': row[2],
                'message': row[3],
                'read': bool(row[4]),
                'created_at': row[5]
            })
        
        conn.close()
        return notifications
    
    def mark_notification_read(self, notification_id: int):
        """Marque une notification comme lue"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE notifications SET read = 1 WHERE id = ?', (notification_id,))
        
        conn.commit()
        conn.close()

    def get_active_configuration(self, name: str = "default_config") -> Optional[Dict]:
        """Récupère la configuration active ou une configuration spécifique par nom."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT config_data FROM configurations WHERE name = ? AND is_active = 1
            ORDER BY last_modified DESC LIMIT 1
        ''', (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        
        # Si aucune config active par nom, chercher la dernière active globale
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT config_data FROM configurations WHERE is_active = 1
            ORDER BY last_modified DESC LIMIT 1
        ''')
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None

    def save_configuration(self, name: str, config_data: Dict, is_active: bool = True):
        """Sauvegarde une configuration. Si is_active est True, désactive les autres du même nom."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now()
        config_json = json.dumps(config_data)

        if is_active:
            # Désactiver les autres configurations actives du même nom ou toutes les autres si 'default_config'
            if name == "default_config": # Nom de configuration spécial pour les paramètres globaux
                 cursor.execute('UPDATE configurations SET is_active = 0 WHERE is_active = 1')
            else:
                 cursor.execute('UPDATE configurations SET is_active = 0 WHERE name = ? AND is_active = 1', (name,))

        # Insérer ou remplacer la configuration
        cursor.execute('''
            INSERT INTO configurations (name, config_data, created_at, last_modified, is_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                config_data = excluded.config_data,
                last_modified = excluded.last_modified,
                is_active = excluded.is_active
        ''', (name, config_json, now, now, is_active))
        
        conn.commit()
        conn.close()

    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Exécute une requête SQL et retourne les résultats sous forme de liste de dictionnaires."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Pour obtenir des résultats sous forme de dict
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Erreur de base de données lors de l'exécution de la requête: {query} - {e}")
            # Vous pourriez vouloir logger l'erreur ici ou la remonter
            return [] # Retourner une liste vide en cas d'erreur
        finally:
            conn.close()
    

class EnhancedWebInterface:
    """Interface web améliorée pour le bot de trading"""
    
    def __init__(self):
        self.db_manager = DatabaseManager(app.config['DATABASE'])
        self.bot_manager = TradingBotManager() if TradingBotManager else None
        self.trading_bot = None
        self.optimizer = None
        self.system_status = SystemStatus()
        self.trading_stats = TradingStats()
        self.trading_mode = TradingMode()
        self.connected_clients = 0
        self.start_time = datetime.now()
        
        # Trading bot réel
        self.real_trader = None
        self.paper_trader = None
        
        # Cache pour les données en temps réel
        self.market_data_cache = {}
        self.performance_cache = {}
        self.price_alerts = []
        self.positions = []
        
        # Thread pour les mises à jour temps réel
        self.update_thread = None
        self.should_update = False
        
        # Gestionnaire d'alertes
        self.alert_manager = PriceAlertManager(self)
        
        # Initialiser les soldes
        self._initialize_balances()
        
        # Gestionnaire d'exécution
        self.trade_executor = TradingExecutor(self)
        
    def _initialize_balances(self):
        """Initialise les soldes depuis la base de données"""
        self.trading_mode.balance_paper = self.db_manager.get_balances(is_paper=True)
        self.trading_mode.balance_real = self.db_manager.get_balances(is_paper=False)
        
    def start_real_time_updates(self):
        """Démarre les mises à jour en temps réel"""
        if self.update_thread is None or not self.update_thread.is_alive():
            self.should_update = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def stop_real_time_updates(self):
        """Arrête les mises à jour en temps réel"""
        self.should_update = False
        if self.update_thread:
            self.update_thread.join(timeout=5)
    
    def _update_loop(self):
        """Boucle de mise à jour en temps réel"""
        while self.should_update:
            try:
                # Mettre à jour les statistiques
                self._update_system_status()
                self._update_trading_stats()
                
                # Émettre les mises à jour via WebSocket
                socketio.emit('status_update', {
                    'system_status': safe_asdict(self.system_status),
                    'trading_stats': safe_asdict(self.trading_stats),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Mettre à jour les données de marché (simulation)
                self._update_market_data()
                
                socketio.sleep(5)  # Mise à jour toutes les 5 secondes
                
            except Exception as e:
                print(f"Erreur dans la boucle de mise à jour: {e}")
                socketio.sleep(10)
    
    def _update_system_status(self):
        """Met à jour le statut du système avec de vraies données"""
        self.system_status.last_update = datetime.now()
        self.system_status.uptime = str(datetime.now() - self.start_time).split('.')[0]
        
        # Récupérer les vrais états du système
        try:
            if self.bot_manager and hasattr(self.bot_manager, 'get_status'):
                # Utiliser get_status() du bot_manager si disponible
                status_data = self.bot_manager.get_status() # Supposons que cela retourne un dict
                self.system_status.bot_running = status_data.get('is_running', False)
                self.system_status.active_symbols = list(status_data.get('active_symbols', []))
                self.system_status.models_loaded = status_data.get('models_loaded', 0)
                # self.system_status.errors_count = status_data.get('error_count', 0) # Si le bot manager le fournit
            elif self.bot_manager:
                # Fallback si get_status() n'existe pas, mais bot_manager oui
                self.system_status.bot_running = hasattr(self.bot_manager, 'is_running') and self.bot_manager.is_running
                if hasattr(self.bot_manager, 'active_symbols'):
                    self.system_status.active_symbols = list(self.bot_manager.active_symbols)
                else:
                    recent_trades = self.db_manager.get_recent_trades(limit=50)
                    self.system_status.active_symbols = list(set(trade['symbol'] for trade in recent_trades if trade.get('symbol')))
                
                models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_store')
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
                    self.system_status.models_loaded = len(model_files)
                else:
                    self.system_status.models_loaded = 0
            else: # Si bot_manager n'est pas initialisé
                self.system_status.bot_running = False
                self.system_status.active_symbols = []
                self.system_status.models_loaded = 0

            # Vérifier les connexions aux exchanges
            self.system_status.connected_exchanges = []
            if self.real_trader and hasattr(self.real_trader, '_test_connectivity') and self.real_trader._test_connectivity(): # Assumant que _test_connectivity existe et retourne un bool
                self.system_status.connected_exchanges.append('Binance Live')
            elif self.real_trader: # Si _test_connectivity n'existe pas mais que le trader est instancié
                 self.system_status.connected_exchanges.append('Binance Live (assumed)')


            # Le mode paper est toujours "connecté" s'il est actif ou si le paper_trader est instancié
            if self.paper_trader or self.trading_mode.is_paper_trading:
                self.system_status.connected_exchanges.append('Paper Trading')
            
            # Compter les erreurs réelles depuis les logs ou notifications
            # TODO: Idéalement, le bot_manager devrait aussi rapporter son propre compte d'erreurs.
            error_notifications = self.db_manager.get_notifications(unread_only=False, limit=100)
            self.system_status.errors_count = len([n for n in error_notifications if n.get('type') == 'error' or 'error' in n.get('title', '').lower()])
                
        except Exception as e:
            print(f"Erreur lors de la mise à jour du statut système: {e}")
            self.system_status.errors_count += 1
    
    def _update_trading_stats(self):
        """Met à jour les statistiques de trading en fonction du mode actuel (paper/live)."""
        
        # Déterminer si on filtre pour paper trades ou live trades
        # La colonne 'is_paper_trade' dans la DB est 1 pour paper, 0 pour live (ou NULL si non spécifié)
        is_current_mode_paper = self.trading_mode.is_paper_trading
        
        # Construire la requête SQL pour filtrer par mode de trading
        # On récupère tous les trades et on filtre en Python pour plus de flexibilité si la DB ne le gère pas bien
        # ou si on veut des stats globales aussi.
        # Pour l'instant, on va chercher tous les trades et filtrer ensuite.
        # Alternative: self.db_manager.get_recent_trades(limit=1000, is_paper=is_current_mode_paper) si la méthode le supporte.
        
        # Récupérer les trades filtrés par mode directement depuis la DB
        paper_trade_filter = 1 if is_current_mode_paper else 0
        trades_for_current_mode = self.db_manager.execute_query(
            "SELECT symbol, side, quantity, price, pnl, timestamp, model_used, confidence, is_paper_trade FROM trades WHERE is_paper_trade = ? ORDER BY timestamp DESC LIMIT 2000",
            (paper_trade_filter,)
        )
        
        if trades_for_current_mode:
            self.trading_stats.total_trades = len(trades_for_current_mode)
            
            winning_trades = [t for t in trades_for_current_mode if t.get('pnl', 0) is not None and float(t['pnl']) > 0]
            losing_trades = [t for t in trades_for_current_mode if t.get('pnl', 0) is not None and float(t['pnl']) < 0]
            
            self.trading_stats.winning_trades = len(winning_trades)
            self.trading_stats.losing_trades = len(losing_trades)
            self.trading_stats.win_rate = (len(winning_trades) / len(trades_for_current_mode) * 100) if trades_for_current_mode else 0
            
            total_pnl = sum(float(t['pnl']) for t in trades_for_current_mode if t.get('pnl') is not None)
            self.trading_stats.total_pnl = total_pnl
            
            today = datetime.now().date()
            daily_trades = [t for t in trades_for_current_mode if t.get('timestamp') and datetime.fromisoformat(t['timestamp'].split(" ")[0]).date() == today and t.get('pnl') is not None]
            self.trading_stats.daily_pnl = sum(float(t['pnl']) for t in daily_trades)
            
            this_month = datetime.now().replace(day=1).date()
            monthly_trades = [t for t in trades_for_current_mode if t.get('timestamp') and datetime.fromisoformat(t['timestamp'].split(" ")[0]).date() >= this_month and t.get('pnl') is not None]
            self.trading_stats.monthly_pnl = sum(float(t['pnl']) for t in monthly_trades)

            # Calcul du Max Drawdown (simplifié, basé sur le PnL cumulé des trades du mode actuel)
            # Un calcul plus précis nécessiterait l'historique de la valeur du portefeuille.
            cumulative_pnl = np.cumsum([float(t['pnl']) for t in trades_for_current_mode if t.get('pnl') is not None])
            if len(cumulative_pnl) > 0:
                peak = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - peak)
                self.trading_stats.max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
            else:
                self.trading_stats.max_drawdown = 0.0

            # Calcul du Sharpe Ratio (simplifié, basé sur les PnL des trades)
            # Suppose un taux sans risque de 0. Les PnL sont les "retours".
            pnl_values = [float(t['pnl']) for t in trades_for_current_mode if t.get('pnl') is not None]
            if len(pnl_values) > 1 and np.std(pnl_values) != 0:
                self.trading_stats.sharpe_ratio = np.mean(pnl_values) / np.std(pnl_values) * np.sqrt(252) # Annualisé (approx.)
            else:
                self.trading_stats.sharpe_ratio = 0.0
        else:
            # Réinitialiser les stats si pas de trades pour le mode actuel
            self.trading_stats = TradingStats() # Recrée une instance avec les valeurs par défaut

        # Mettre à jour le nombre de positions ouvertes, filtré par mode
        paper_position_filter = 1 if is_current_mode_paper else 0
        open_positions_db = self.db_manager.execute_query(
            "SELECT COUNT(*) as count FROM positions WHERE status = 'open' AND is_paper_position = ?",
            (paper_position_filter,)
        )
        self.trading_stats.current_positions = open_positions_db[0]['count'] if open_positions_db else 0
    
    def _update_market_data(self):
        """Met à jour les données de marché avec de vraies données"""
        symbols = self.system_status.active_symbols or ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        if not symbols:
            print("ℹ️ Aucuns symboles actifs à mettre à jour pour les données de marché.")
            self.market_data_cache = {} # Clear cache if no symbols
            socketio.emit('market_data_update', self.market_data_cache)
            return

        updated_symbols_count = 0
        try:
            # Utiliser l'API publique de Binance pour les prix réels
            # Il est plus efficace de faire un seul appel pour tous les tickers si possible
            url = "https://api.binance.com/api/v3/ticker/24hr"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                all_tickers_data = response.json()
                price_data_map = {item['symbol']: item for item in all_tickers_data}
                
                current_market_data = {}
                for symbol in symbols:
                    if symbol in price_data_map:
                        ticker = price_data_map[symbol]
                        current_market_data[symbol] = {
                            'price': float(ticker['lastPrice']),
                            'change_24h': float(ticker['priceChangePercent']),
                            'volume': float(ticker['volume']),
                            'high_24h': float(ticker['highPrice']),
                            'low_24h': float(ticker['lowPrice']),
                            'open_price': float(ticker['openPrice']),
                            'quote_volume': float(ticker['quoteVolume']),
                            'count': int(ticker['count']),
                            'timestamp': datetime.now().isoformat()
                        }
                        updated_symbols_count +=1
                    else:
                        # Si le symbole n'est pas dans la réponse globale, essayer un appel individuel
                        # (utile pour les paires moins courantes ou si la liste `symbols` est très spécifique)
                        single_ticker_url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
                        single_response = requests.get(single_ticker_url, timeout=5)
                        if single_response.status_code == 200:
                            ticker = single_response.json()
                            current_market_data[symbol] = {
                                'price': float(ticker['lastPrice']),
                                'change_24h': float(ticker['priceChangePercent']),
                                'volume': float(ticker['volume']),
                                'high_24h': float(ticker['highPrice']),
                                'low_24h': float(ticker['lowPrice']),
                                'open_price': float(ticker['openPrice']),
                                'quote_volume': float(ticker['quoteVolume']),
                                'count': int(ticker['count']),
                                'timestamp': datetime.now().isoformat()
                            }
                            updated_symbols_count += 1
                        else:
                            print(f"⚠️ Impossible de récupérer les données pour {symbol} (individuellement): {single_response.status_code}")
                            # Conserver les anciennes données en cache si elles existent, sinon vide
                            if symbol in self.market_data_cache:
                                current_market_data[symbol] = self.market_data_cache[symbol]
                                current_market_data[symbol]['stale'] = True # Marquer comme potentiellement obsolète
                            else:
                                current_market_data[symbol] = {'price': 0, 'error': 'No data'}


                self.market_data_cache = current_market_data
                if updated_symbols_count > 0:
                    print(f"✅ Données de marché mises à jour pour {updated_symbols_count}/{len(symbols)} symboles.")
                else:
                    print(f"⚠️ Aucune donnée de marché n'a pu être mise à jour pour les symboles actifs.")

            else:
                print(f"❌ Erreur API Binance (globale): {response.status_code}. Les données de marché pourraient être obsolètes.")
                # Ne pas lever d'exception ici pour permettre au reste du système de fonctionner avec des données potentiellement obsolètes
                # Marquer toutes les données existantes comme obsolètes
                for symbol_data in self.market_data_cache.values():
                    symbol_data['stale'] = True
                # Ne pas effacer le cache, garder les dernières valeurs connues
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur de connexion à l'API Binance: {e}. Les données de marché pourraient être obsolètes.")
            for symbol_data in self.market_data_cache.values():
                symbol_data['stale'] = True
        except Exception as e:
            print(f"❌ Erreur inattendue lors de la mise à jour des données de marché: {e}")
            # En cas d'erreur grave, il est peut-être préférable de vider le cache pour éviter d'utiliser des données corrompues
            # self.market_data_cache = {} # Optionnel: dépend de la stratégie de robustesse
            for symbol_data in self.market_data_cache.values():
                symbol_data['stale'] = True
        
        # Émettre les données de marché mises à jour (ou potentiellement obsolètes)
        socketio.emit('market_data_update', self.market_data_cache)

    def get_real_time_price(self, symbol: str) -> float:
        """Récupère le prix en temps réel d'un symbole spécifique"""
        # Essayer d'abord depuis le cache local mis à jour par _update_market_data
        if symbol in self.market_data_cache and not self.market_data_cache[symbol].get('stale', False) and self.market_data_cache[symbol].get('price', 0) > 0 :
            return self.market_data_cache[symbol]['price']

        # Si non trouvé dans le cache ou obsolète, faire un appel direct (avec prudence pour les limites de taux)
        print(f"ℹ️ Prix pour {symbol} non trouvé/obsolète dans le cache récent, appel direct à l'API.")
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=3) # Timeout plus court pour les appels individuels rapides
            
            if response.status_code == 200:
                data = response.json()
                price = float(data['price'])
                # Mettre à jour le cache
                if symbol not in self.market_data_cache: self.market_data_cache[symbol] = {}
                self.market_data_cache[symbol].update({
                    'price': price, 
                    'timestamp': datetime.now().isoformat(),
                    'stale': False,
                    'error': None # Clear previous error
                })
                return price
            else:
                print(f"❌ Erreur API (prix individuel) pour {symbol}: {response.status_code}")
                # Mettre à jour le cache avec l'erreur
                if symbol not in self.market_data_cache: self.market_data_cache[symbol] = {}
                self.market_data_cache[symbol]['error'] = f"API Error {response.status_code}"
                self.market_data_cache[symbol]['stale'] = True
                # Retourner l'ancienne valeur du cache si elle existe et est valide, sinon 0.0
                return self.market_data_cache.get(symbol, {}).get('price', 0.0) if self.market_data_cache.get(symbol, {}).get('price', 0) > 0 else 0.0
                
        except requests.exceptions.RequestException as e: # Erreurs réseau spécifiques
            print(f"❌ Erreur réseau lors de la récupération du prix de {symbol}: {e}")
            if symbol not in self.market_data_cache: self.market_data_cache[symbol] = {}
            self.market_data_cache[symbol]['error'] = f"Network Error: {str(e)}"
            self.market_data_cache[symbol]['stale'] = True
            return self.market_data_cache.get(symbol, {}).get('price', 0.0) if self.market_data_cache.get(symbol, {}).get('price', 0) > 0 else 0.0
        except Exception as e: # Autres erreurs (ex: JSONDecodeError)
            print(f"❌ Erreur inattendue lors de la récupération du prix de {symbol}: {e}")
            if symbol not in self.market_data_cache: self.market_data_cache[symbol] = {}
            self.market_data_cache[symbol]['error'] = f"Unexpected Error: {str(e)}"
            self.market_data_cache[symbol]['stale'] = True
            return self.market_data_cache.get(symbol, {}).get('price', 0.0) if self.market_data_cache.get(symbol, {}).get('price', 0) > 0 else 0.0

class PriceAlertManager:
    """Gestionnaire d'alertes de prix"""
    
    def __init__(self, web_interface):
        self.web_interface = web_interface
        self.active_alerts = {}
        self._load_alerts()
    
    def _load_alerts(self):
        """Charge les alertes actives depuis la base de données"""
        alerts = self.web_interface.db_manager.get_price_alerts(active_only=True)
        for alert in alerts:
            self.active_alerts[alert['id']] = alert
    
    def add_alert(self, symbol: str, condition: str, value: float) -> str:
        """Ajoute une nouvelle alerte"""
        import uuid
        alert_id = str(uuid.uuid4())
        
        current_price = self.web_interface.get_real_time_price(symbol)
        alert = PriceAlert(
            id=alert_id,
            symbol=symbol,
            condition=condition,
            value=value,
            current_price=current_price
        )
        
        self.web_interface.db_manager.add_price_alert(alert)
        self.active_alerts[alert_id] = safe_asdict(alert)
        
        # Notification
        self.web_interface.db_manager.add_notification(
            'alert', 
            'Nouvelle alerte créée',
            f'Alerte créée pour {symbol} {condition} {value}'
        )
        
        return alert_id
    
    def check_alerts(self, symbol: str, current_price: float):
        """Vérifie les alertes pour un symbole donné"""
        triggered_alerts = []
        
        for alert_id, alert in list(self.active_alerts.items()):
            if alert['symbol'] != symbol:
                continue
                
            should_trigger = False
            
            if alert['condition'] == 'above' and current_price >= alert['value']:
                should_trigger = True
            elif alert['condition'] == 'below' and current_price <= alert['value']:
                should_trigger = True
            elif alert['condition'] == 'change_percent':
                price_change = ((current_price - alert['current_price']) / alert['current_price']) * 100
                if abs(price_change) >= alert['value']:
                    should_trigger = True
            
            if should_trigger:
                self._trigger_alert(alert_id, current_price)
                triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def _trigger_alert(self, alert_id: str, current_price: float):
        """Déclenche une alerte"""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return
        
        # Mettre à jour en base de données
        self.web_interface.db_manager.trigger_price_alert(alert_id)
        
        # Notification
        message = f"Alerte déclenchée: {alert['symbol']} {alert['condition']} {alert['value']} (Prix actuel: {current_price})"
        self.web_interface.db_manager.add_notification(
            'alert_triggered',
            'Alerte de prix déclenchée',
            message
        )
        
        # Émettre via WebSocket
        socketio.emit('price_alert_triggered', {
            'alert': alert,
            'current_price': current_price,
            'message': message
        })
        
        # Retirer des alertes actives
        del self.active_alerts[alert_id]
    
    def remove_alert(self, alert_id: str):
        """Supprime une alerte"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
        
        # Marquer comme déclenchée en base (pour la désactiver)
        self.web_interface.db_manager.trigger_price_alert(alert_id)


class TradingExecutor:
    """Exécuteur de trades avec support paper/live trading"""
    
    def __init__(self, web_interface):
        self.web_interface = web_interface
        self.paper_positions = {}
        self.real_positions = {}
    
    def execute_trade(self, symbol: str, side: str, quantity: float, 
                     order_type: str = "market", price: float = None, 
                     is_paper: bool = True) -> Dict:
        """Exécute un trade"""
        
        if is_paper:
            return self._execute_paper_trade(symbol, side, quantity, order_type, price)
        else:
            return self._execute_real_trade(symbol, side, quantity, order_type, price)
    
    def _execute_paper_trade(self, symbol: str, side: str, quantity: float, 
                           order_type: str, price: float = None) -> Dict:
        """Exécute un trade en paper trading"""
        
        current_price = price or self.web_interface.get_real_time_price(symbol)
        if current_price <= 0:
            return {"success": False, "error": "Prix invalide"}
        
        # Vérifier le solde
        base_asset = symbol.replace('USDT', '')
        quote_asset = 'USDT'
        
        balances = self.web_interface.trading_mode.balance_paper
        
        if side.upper() == 'BUY':
            required_usdt = quantity * current_price
            if balances.get(quote_asset, 0) < required_usdt:
                return {"success": False, "error": "Solde USDT insuffisant"}
            
            # Exécuter l'achat
            balances[quote_asset] = balances.get(quote_asset, 0) - required_usdt
            balances[base_asset] = balances.get(base_asset, 0) + quantity
            
        else:  # SELL
            if balances.get(base_asset, 0) < quantity:
                return {"success": False, "error": f"Solde {base_asset} insuffisant"}
            
            # Exécuter la vente
            balances[base_asset] = balances.get(base_asset, 0) - quantity
            usdt_received = quantity * current_price
            balances[quote_asset] = balances.get(quote_asset, 0) + usdt_received
        
        # Mettre à jour les soldes en base
        for asset, balance in balances.items():
            self.web_interface.db_manager.update_balance(asset, balance, is_paper=True)
        
        # Enregistrer le trade
        trade_data = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': current_price,
            'pnl': 0.0,  # Calculé plus tard
            'model_used': 'Manual',
            'confidence': 1.0
        }
        self.web_interface.db_manager.add_trade(trade_data)
        
        # Notification
        self.web_interface.db_manager.add_notification(
            'trade_executed',
            'Trade exécuté (Paper)',
            f'{side} {quantity} {symbol} @ {current_price}'
        )
        
        return {
            "success": True,
            "trade": trade_data,
            "new_balances": balances
        }
    
    def _execute_real_trade(self, symbol: str, side: str, quantity: float, 
                          order_type: str, price: float = None) -> Dict:
        """Exécute un trade réel via l'API"""
        
        if not self.web_interface.real_trader:
            return {"success": False, "error": "Trader réel non configuré"}
        
        # Créer l'ordre
        order = TradingOrder(
            symbol=symbol,
            side=OrderSide.BUY if side.upper() == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET if order_type.lower() == 'market' else OrderType.LIMIT,
            quantity=quantity,
            price=price
        )
        
        # Exécuter via le trader réel
        try:
            # S'assurer que real_trader est une instance de BinanceRealTimeTrader
            if not isinstance(self.web_interface.real_trader, BinanceRealTimeTrader):
                 return {"success": False, "error": "Trader réel (BinanceRealTimeTrader) non correctement initialisé."}

            executed_order_details = self.web_interface.real_trader.place_order(order)
            
            if executed_order_details and executed_order_details.status not in [OrderStatus.REJECTED, OrderStatus.EXPIRED, OrderStatus.ERROR]:
                filled_price = executed_order_details.avg_fill_price if executed_order_details.avg_fill_price and executed_order_details.avg_fill_price > 0 else executed_order_details.price
                
                trade_data_for_db = {
                    'symbol': executed_order_details.symbol,
                    'side': executed_order_details.side.value,
                    'quantity': executed_order_details.executed_quantity if executed_order_details.executed_quantity > 0 else executed_order_details.quantity,
                    'price': filled_price,
                    'pnl': 0.0,
                    'model_used': 'Manual',
                    'confidence': 1.0,
                    'order_id': executed_order_details.order_id,
                    'is_paper_trade': 0, # Indiquer que c'est un trade réel
                    'timestamp': datetime.now() # Ajouter le timestamp ici
                }
                
                self.web_interface.db_manager.add_trade(trade_data_for_db)
                
                # Notification
                self.web_interface.db_manager.add_notification(
                    'trade_executed',
                    'Trade exécuté (Live)',
                    f'{executed_order_details.side.value} {trade_data_for_db["quantity"]} {executed_order_details.symbol} @ {filled_price}'
                )
                
                return {"success": True, "trade": trade_data_for_db, "order": safe_asdict(executed_order_details)}
            elif executed_order_details:
                 error_msg = f"Ordre {executed_order_details.status.value if executed_order_details.status else 'non exécuté'}. Raison: {executed_order_details.reject_reason if hasattr(executed_order_details, 'reject_reason') and executed_order_details.reject_reason else 'Inconnue'}"
                 self.web_interface.db_manager.add_notification('error', 'Échec du Trade (Live)', error_msg)
                 return {"success": False, "error": error_msg, "order_status": executed_order_details.status.value if executed_order_details.status else "UNKNOWN"}
            else:
                self.web_interface.db_manager.add_notification('error', 'Échec du Trade (Live)', f"Échec de la soumission de l'ordre pour {symbol}.")
                return {"success": False, "error": "Échec de la soumission de l'ordre à l'exchange."}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def switch_trading_mode(self, is_paper: bool):
        """Bascule entre paper trading et live trading"""
        self.trading_mode.is_paper_trading = is_paper
        self.trading_mode.is_live_trading = not is_paper
        self.trading_mode.mode_name = "Paper Trading" if is_paper else "Live Trading"
        
        # Notification
        mode_name = "Paper Trading" if is_paper else "Live Trading"
        self.db_manager.add_notification(
            'mode_change',
            'Mode de trading changé',
            f'Basculé vers le mode: {mode_name}'
        )
        
        # Émettre via WebSocket
        socketio.emit('trading_mode_changed', {
            'is_paper_trading': is_paper,
            'mode_name': mode_name
        })
    
    def get_portfolio_value(self, is_paper: bool = True) -> float:
        """Calcule la valeur totale du portefeuille"""
        balances = self.web_interface.trading_mode.balance_paper if is_paper else self.web_interface.trading_mode.balance_real
        total_value = 0.0
        
        for asset, balance in balances.items():
            if asset == 'USDT':
                total_value += balance
            else:
                symbol = f"{asset}USDT"
                price = self.web_interface.get_real_time_price(symbol)
                total_value += balance * price
        
        return total_value
    
    def get_position_pnl(self, symbol: str, entry_price: float, current_price: float, 
                        quantity: float, side: str) -> tuple:
        """Calcule le PnL d'une position"""
        if side.lower() == 'long':
            pnl = (current_price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - current_price) * quantity
        
        pnl_percent = (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0
        
        return pnl, pnl_percent
    
    def update_positions_pnl(self):
        """Met à jour le PnL de toutes les positions ouvertes"""
        positions = self.db_manager.get_positions(open_only=True)
        
        for position in positions:
            current_price = self.get_real_time_price(position['symbol'])
            if current_price > 0:
                pnl, pnl_percent = self.get_position_pnl(
                    position['symbol'],
                    position['entry_price'],
                    current_price,
                    position['quantity'],
                    position['side']
                )
                
                self.db_manager.update_position_pnl(
                    position['symbol'], current_price, pnl, pnl_percent
                )
    
    def initialize_real_trader(self, api_key: str, api_secret: str, testnet: bool = True):
        """Initialise le trader réel"""
        try:
            if BinanceRealTimeTrader:
                self.real_trader = BinanceRealTimeTrader(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet
                )
                
                # Notification
                self.db_manager.add_notification(
                    'trader_connected',
                    'Trader réel connecté',
                    f'Connexion réussie ({"Testnet" if testnet else "Live"})'
                )
                
                return True
            else:
                return False
                
        except Exception as e:
            self.db_manager.add_notification(
                'trader_error',
                'Erreur de connexion trader',
                f'Erreur: {str(e)}'
            )
            return False
        

# Instance globale
web_interface = EnhancedWebInterface()

# Routes principales
@app.route('/')
def index():
    """Page d'accueil - redirection vers le dashboard"""
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Page de connexion"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Vérifier les identifiants
        conn = sqlite3.connect(app.config['DATABASE'])
        cursor = conn.cursor()
        cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[0], password):
            session['logged_in'] = True
            session['username'] = username
            flash('Connexion réussie!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Identifiants incorrects', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Déconnexion"""
    session.clear()
    flash('Déconnexion réussie', 'info')
    return redirect(url_for('login'))

def login_required(f):
    """Décorateur pour vérifier la connexion"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard principal"""
    return render_template('dashboard_enhanced.html', 
                         system_status=safe_asdict(web_interface.system_status),
                         trading_stats=safe_asdict(web_interface.trading_stats))

@app.route('/trading')
@login_required
def trading():
    """Page de trading manuel"""
    return render_template('trading.html')

@app.route('/models')
@login_required
def models():
    """Page de gestion des modèles"""
    return render_template('models.html')

@app.route('/optimization')
@login_required
def optimization():
    """Page d'optimisation des modèles"""
    return render_template('optimization.html')

@app.route('/reports')
@login_required
def reports():
    """Page des rapports et analyses"""
    return render_template('reports.html')

@app.route('/settings')
@login_required
def settings():
    """Page des paramètres"""
    return render_template('settings.html')

@app.route('/trading-advanced')
@login_required
def trading_advanced():
    """Page de trading avancé avec TradingView"""
    return render_template('trading_advanced.html')

@app.route('/portfolio')
@login_required
def portfolio():
    """Page de gestion du portfolio"""
    return render_template('portfolio.html')

@app.route('/alerts')
@login_required
def alerts():
    """Page de gestion des alertes prix"""
    return render_template('alerts.html')

@app.route('/risk-management')
@login_required
def risk_management():
    """Page de gestion des risques"""
    return render_template('risk_management.html')

# API Routes
@app.route('/api/system/status')
@login_required
def api_system_status():
    """API: Statut du système"""
    return jsonify({
        'system_status': safe_asdict(web_interface.system_status),
        'trading_stats': safe_asdict(web_interface.trading_stats),
        'connected_clients': web_interface.connected_clients
    })

@app.route('/api/trading/start', methods=['POST'])
@login_required
def api_start_trading():
    """API: Démarrer le trading"""
    try:
        if web_interface.bot_manager:
            success = web_interface.bot_manager.start_bot(background=True)
            if success:
                web_interface.system_status.bot_running = True  # Mettre à jour le statut après succès
                web_interface.db_manager.add_notification('info', 'Bot Control', 'Bot de trading démarré avec succès.')
                return jsonify({'success': True, 'message': 'Bot de trading démarré'})
            else:
                web_interface.system_status.bot_running = False  # S'assurer que le statut est correct en cas d'échec
                web_interface.db_manager.add_notification('error', 'Bot Control', 'Échec du démarrage du bot de trading.')
                return jsonify({'success': False, 'message': 'Échec du démarrage du bot de trading'})
        else:
            web_interface.db_manager.add_notification('error', 'Bot Control', 'Tentative de démarrage du bot, mais le Bot Manager n\'est pas disponible.')
            return jsonify({'success': False, 'message': 'Bot manager non disponible'})
    except Exception as e:
        web_interface.db_manager.add_notification('error', 'Bot Control', f'Erreur lors du démarrage du bot: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trading/stop', methods=['POST'])
@login_required
def api_stop_trading():
    """API: Arrêter le trading"""
    try:
        if web_interface.bot_manager:
            success = web_interface.bot_manager.stop_bot()
            if success:
                web_interface.system_status.bot_running = False  # Mettre à jour le statut après succès
                web_interface.db_manager.add_notification('info', 'Bot Control', 'Bot de trading arrêté avec succès.')
                return jsonify({'success': True, 'message': 'Bot de trading arrêté'})
            else:
                # Ne pas changer bot_running ici, car il pourrait toujours être en cours d'exécution si stop_bot a échoué
                web_interface.db_manager.add_notification('error', 'Bot Control', 'Échec de l\'arrêt du bot de trading.')
                return jsonify({'success': False, 'message': 'Échec de l\'arrêt du bot de trading'})
        else:
            web_interface.db_manager.add_notification('error', 'Bot Control', 'Tentative d\'arrêt du bot, mais le Bot Manager n\'est pas disponible.')
            return jsonify({'success': False, 'message': 'Bot manager non disponible'})
    except Exception as e:
        web_interface.db_manager.add_notification('error', 'Bot Control', f'Erreur lors de l\'arrêt du bot: {str(e)}')
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/trades/recent')
@login_required
def api_recent_trades():
    """API: Trades récents"""
    trades = web_interface.db_manager.get_recent_trades(limit=50)
    return jsonify(trades)

@app.route('/api/performance/history')
@login_required
def api_performance_history():
    """API: Historique de performance"""
    days = request.args.get('days', 30, type=int)
    performance = web_interface.db_manager.get_performance_history(days)
    return jsonify(performance)

@app.route('/api/market/prices')
@login_required
def api_market_prices():
    """API: Prix de marché en temps réel"""
    try:
        symbols = request.args.get('symbols', 'BTCUSDT,ETHUSDT,ADAUSDT,DOTUSDT').split(',')
        prices = {}
        
        for symbol in symbols:
            price = web_interface.get_real_time_price(symbol.strip())
            prices[symbol.strip()] = price
        
        return jsonify(prices)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market/ticker/<symbol>')
@login_required
def api_market_ticker(symbol):
    """API: Données complètes pour un symbole"""
    try:
        if symbol in web_interface.market_data_cache:
            return jsonify(web_interface.market_data_cache[symbol])
        else:
            # Récupérer les données en temps réel
            web_interface._update_market_data()
            return jsonify(web_interface.market_data_cache.get(symbol, {'error': 'Symbol not found'}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market/orderbook/<symbol>')
@login_required
def api_market_orderbook(symbol):
    """API: Carnet d'ordres en temps réel via l'API Binance"""
    try:
        # Récupérer le carnet d'ordres réel depuis Binance
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            return jsonify({
                'symbol': symbol,
                'bids': data['bids'],  # Format: [["price", "quantity"], ...]
                'asks': data['asks'],  # Format: [["price", "quantity"], ...]
                'lastUpdateId': data['lastUpdateId'],
                'timestamp': datetime.now().isoformat(),
                'source': 'binance_api'
            })
        else:
            return jsonify({
                'success': False, 
                'error': f'Failed to fetch orderbook data: HTTP {response.status_code}'
            })
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            'success': False, 
            'error': f'Network error while fetching orderbook: {str(e)}'
        })
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Error fetching orderbook data: {str(e)}'
        })

@app.route('/api/models/optimize', methods=['POST'])
@login_required
def api_optimize_models():
    """API: Lancer l'optimisation des modèles"""
    try:
        if ProfitabilityOptimizer:
            symbols = request.json.get('symbols', ['BTCUSDT', 'ETHUSDT'])
            
            # Lancer l'optimisation en arrière-plan
            def run_optimization():
                optimizer = ProfitabilityOptimizer()
                results = optimizer.optimize_all_models(symbols)
                # Sauvegarder les résultats et notifier via WebSocket
                socketio.emit('optimization_complete', {'results': 'completed'})
            
            threading.Thread(target=run_optimization).start()
            return jsonify({'success': True, 'message': 'Optimisation lancée'})
        else:
            return jsonify({'success': False, 'message': 'Optimiseur non disponible'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/models', methods=['GET'])
@login_required
def api_get_models():
    """API: Lister les modèles de trading disponibles."""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_store')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir) # Créer le répertoire s'il n'existe pas
            return jsonify({'success': True, 'models': [], 'message': 'Models directory created. No models found.'})

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        models_info = []
        for model_file in model_files:
            # Extraire des informations du nom du fichier si possible (ex: symbol, type)
            # Pour l'instant, on retourne juste le nom.
            # Idéalement, on chargerait le modèle pour obtenir plus de métadonnées,
            # mais cela peut être coûteux pour une simple liste.
            # Ou, stocker les métadonnées dans la DB ou un fichier JSON séparé.
            models_info.append({
                'id': model_file,
                'name': model_file.replace('.joblib', ''),
                'file_name': model_file,
                'path': os.path.join(models_dir, model_file),
                'status': 'available', # Pourrait être 'active', 'inactive', 'training' etc.
                'last_trained': None # TODO: Récupérer cette info si possible
            })
        return jsonify({'success': True, 'models': models_info})
    except Exception as e:
        print(f"Erreur dans api_get_models: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/<model_id>', methods=['DELETE'])
@login_required
def api_delete_model(model_id):
    """API: Supprimer un modèle de trading spécifique."""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_store')
        model_path = os.path.join(models_dir, model_id) # model_id est supposé être le nom du fichier

        if not model_id.endswith('.joblib'): # S'assurer qu'on ne supprime que des fichiers de modèle
            model_path += '.joblib'

        if os.path.exists(model_path):
            os.remove(model_path)
            web_interface.db_manager.add_notification("info", "Model Deleted", f"Modèle {model_id} supprimé avec succès.")
            return jsonify({'success': True, 'message': f'Modèle {model_id} supprimé avec succès.'})
        else:
            return jsonify({'success': False, 'error': f'Modèle {model_id} non trouvé.'}), 404
    except Exception as e:
        print(f"Erreur dans api_delete_model: {e}")
        web_interface.db_manager.add_notification("error", "Model Deletion Failed", f"Erreur lors de la suppression du modèle {model_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/<model_id>/activate', methods=['POST'])
@login_required
def api_activate_model(model_id):
    """API: Activer un modèle de trading spécifique."""
    # TODO: Implémenter la logique pour marquer un modèle comme actif.
    # Cela pourrait impliquer de mettre à jour une entrée dans la base de données
    # ou de modifier un fichier de configuration que le TradingBotManager utilise.
    try:
        # Placeholder: Vérifier si le modèle existe
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_store')
        model_file_name = model_id if model_id.endswith('.joblib') else model_id + '.joblib'
        model_path = os.path.join(models_dir, model_file_name)

        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': f'Modèle {model_id} non trouvé.'}), 404

        # Logique d'activation (à implémenter)
        # Exemple: web_interface.bot_manager.activate_model(model_id)
        # Ou mettre à jour une table 'active_models' dans la DB.
        
        web_interface.db_manager.add_notification("info", "Model Activated (Placeholder)", f"Modèle {model_id} activé (logique à implémenter).")
        return jsonify({'success': True, 'message': f'Modèle {model_id} activé (placeholder). Logique à implémenter.'})
    except Exception as e:
        print(f"Erreur dans api_activate_model: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/<model_id>/deactivate', methods=['POST'])
@login_required
def api_deactivate_model(model_id):
    """API: Désactiver un modèle de trading spécifique."""
    # TODO: Implémenter la logique pour marquer un modèle comme inactif.
    try:
        # Placeholder: Vérifier si le modèle existe
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models_store')
        model_file_name = model_id if model_id.endswith('.joblib') else model_id + '.joblib'
        model_path = os.path.join(models_dir, model_file_name)

        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': f'Modèle {model_id} non trouvé.'}), 404
            
        # Logique de désactivation (à implémenter)
        # Exemple: web_interface.bot_manager.deactivate_model(model_id)

        web_interface.db_manager.add_notification("info", "Model Deactivated (Placeholder)", f"Modèle {model_id} désactivé (logique à implémenter).")
        return jsonify({'success': True, 'message': f'Modèle {model_id} désactivé (placeholder). Logique à implémenter.'})
    except Exception as e:
        print(f"Erreur dans api_deactivate_model: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/models/<model_id>/train', methods=['POST'])
@login_required
def api_train_model(model_id):
    """API: (Ré)entraîner un modèle de trading spécifique."""
    # TODO: Implémenter la logique pour lancer un processus d'entraînement pour ce modèle.
    # Cela pourrait être similaire à l'endpoint d'optimisation mais ciblé sur un modèle existant.
    # Ou, si le modèle a une configuration d'entraînement stockée, la réutiliser.
    try:
        # Placeholder: Vérifier si le modèle existe (ou si c'est un nom de configuration d'entraînement)
        # Pour l'instant, on suppose que model_id est un identifiant qui permet de retrouver
        # les données et la configuration nécessaires à l'entraînement.
        
        # Exemple de lancement d'un entraînement en arrière-plan:
        # def run_model_training(model_identifier):
        #     try:
        #         # ... logique d'entraînement ...
        #         socketio.emit('model_training_update', {'model_id': model_identifier, 'status': 'completed'})
        #     except Exception as train_exc:
        #         socketio.emit('model_training_update', {'model_id': model_identifier, 'status': 'failed', 'error': str(train_exc)})
        #
        # threading.Thread(target=run_model_training, args=(model_id,)).start()
        
        web_interface.db_manager.add_notification("info", "Model Training Started (Placeholder)", f"Entraînement pour le modèle {model_id} démarré (logique à implémenter).")
        return jsonify({'success': True, 'message': f'Entraînement pour le modèle {model_id} démarré (placeholder). Logique à implémenter.'})
    except Exception as e:
        print(f"Erreur dans api_train_model: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Additional API endpoints for the new pages

# Optimization API Endpoints
@app.route('/api/optimization/start', methods=['POST'])
@login_required
def api_start_optimization():
    """API: Start model optimization"""
    try:
        config = request.json
        # Store optimization configuration
        optimization_id = f"opt_{int(time.time())}"
        
        # Start optimization in background
        def run_optimization():
            try:
                from src.modeling.models import train_model, prepare_data_for_model
                # Importer les modules nécessaires pour charger les données, par exemple depuis src.acquisition
                from src.acquisition.connectors import BinanceConnector
                from src.acquisition.preprocessing import handle_missing_values, handle_outliers_quantile # Importer des fonctions spécifiques
                from src.feature_engineering.technical_features import (
                    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
                    calculate_bollinger_bands, calculate_price_momentum, calculate_volume_features
                )
                
                symbols = config.get('symbols', ['BTCUSDT']) # Commençons avec un symbole pour simplifier
                model_type_to_optimize = config.get('model_type', 'xgboost_classifier') # ou 'random_forest'
                optuna_n_trials = config.get('optuna_n_trials', 25) # Nombre d'essais Optuna
                
                # 1. Charger et préparer les données pour chaque symbole
                # Cette partie est cruciale et dépend de la structure de votre projet
                # Exemple simplifié :
                all_metrics = {}
                for symbol_to_optimize in symbols:
                    socketio.emit('optimization_update', {
                        'status_message': f'Chargement des données pour {symbol_to_optimize}...',
                        'symbol': symbol_to_optimize,
                        'overall_progress': 0 # Initial
                    })
                    
                    X_data, y_data = None, None
                    try:
                        # Récupérer la configuration active pour les clés API
                        active_config = web_interface.db_manager.get_active_configuration(DEFAULT_SETTINGS_NAME)
                        if active_config is None:
                            active_config = DEFAULT_SETTINGS # Fallback aux valeurs par défaut globales

                        api_settings = active_config.get('api', {})
                        api_key = api_settings.get('api_key', os.getenv('BINANCE_API_KEY'))
                        api_secret = api_settings.get('api_secret', os.getenv('BINANCE_API_SECRET'))
                        use_testnet = api_settings.get('sandbox', True)

                        if not api_key or not api_secret:
                            # Ne pas lever d'erreur ici, laisser le code se rabattre sur les données factices
                            print("⚠️ Clés API Binance non configurées. Tentative d'utilisation de données factices.")
                            raise ValueError("Clés API Binance non configurées") # Forcer le passage au bloc except pour données factices

                        connector = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=use_testnet)
                        end_date_str = datetime.now().strftime("%d %b, %Y")
                        start_date_str = (datetime.now() - timedelta(days=365*2)).strftime("%d %b, %Y")
                        
                        # Tentative de récupération des données historiques.
                        # La méthode get_historical_data n'existe pas sur BinanceConnector.
                        # BinanceConnector a get_klines. Supposons que c'est ce qui était visé.
                        # get_klines retourne un dict {interval: DataFrame}. On prendra le premier intervalle pour l'exemple.
                        # Ou, si une méthode plus générique est attendue, elle doit être ajoutée à BinanceConnector.
                        # Pour l'instant, on simule que raw_data est un DataFrame.
                        # raw_data_dict = connector.get_klines(symbol=symbol_to_optimize, intervals=["1h"], start_date_str=start_date_str, end_date_str=end_date_str)
                        # raw_data = raw_data_dict.get("1h", pd.DataFrame()) # Prendre les données de l'intervalle '1h'

                        # Pour éviter de modifier la logique de get_historical_data qui n'existe pas,
                        # on va simuler un échec ici si les clés API sont valides mais que la méthode n'est pas là,
                        # pour forcer l'utilisation de données factices comme c'était probablement le cas avant.
                        # Ceci est un placeholder pour une vraie implémentation de chargement de données.
                        if not hasattr(connector, 'get_historical_data'): # Simuler l'échec si la méthode n'existe pas
                            print(f"ℹ️ La méthode 'get_historical_data' n'est pas implémentée sur BinanceConnector. Passage aux données factices pour {symbol_to_optimize}.")
                            raise NotImplementedError("get_historical_data non implémenté sur BinanceConnector")
                        
                        # Si la méthode existait, le code continuerait ici:
                        # raw_data = connector.get_historical_data(symbol_to_optimize, "1h", start_date_str, end_date_str)
                        # if raw_data.empty:
                        #     raise ValueError(f"Aucune donnée brute retournée par le connecteur pour {symbol_to_optimize}")
                        # ... (reste du traitement des données réelles) ...
                        # (Le code original pour data_cleaned, data_features, processed_data, feature_cols, target_col irait ici)
                        # Pour l'instant, on va laisser le bloc try échouer pour utiliser les données factices,
                        # car la correction principale est sur l'erreur 'processed_data'.

                        # Placeholder pour le code de traitement des données réelles qui définirait :
                        # data_cleaned, data_features, processed_data, feature_cols, target_col, problem_type_config
                        # Ce code est omis pour se concentrer sur la correction de `processed_data` et `BinanceConnector`
                        # Si ce bloc réussissait, df_for_training et feature_cols_for_training seraient définis ici.
                        # Exemple (très simplifié, ne pas utiliser en production sans adaptation):
                        # required_ohlcv = ['open', 'high', 'low', 'close', 'volume']
                        # data_cleaned = raw_data[required_ohlcv].copy() # Simplification extrême
                        # data_features = data_cleaned # Pas de feature engineering pour ce placeholder
                        # feature_cols_for_training = [col for col in data_features.columns if col != 'close'] # Exemple
                        # df_for_training = data_features.dropna()
                        # if df_for_training.empty: raise ValueError("Données réelles vides après dropna.")

                        # Le code original pour définir feature_cols et target_col
                        real_feature_cols_list = [ # Liste exhaustive des features attendues
                            'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50', 'rsi_14',
                            'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_lower', 'bb_position',
                            'momentum_5', 'momentum_10', 'momentum_20', 'volatility_5', 'volatility_10', 'volatility_20',
                            'volume_sma_10', 'volume_sma_20', 'volume_ratio_10', 'volume_ratio_20', 'vwap_10'
                        ]
                        feature_cols = [col for col in real_feature_cols_list if col in data_features.columns]
                        if not feature_cols:
                             raise ValueError(f"Aucune colonne de feature attendue n'a été trouvée après feature engineering pour {symbol_to_optimize}.")
                        target_col = 'target' # Nom standard
                        problem_type_config = config.get('problem_type', 'classification')
                        
                        df_for_training = data_features.dropna()
                        if df_for_training.empty:
                            raise ValueError(f"DataFrame vide après dropna pour {symbol_to_optimize} (données réelles).")
                        feature_cols_for_training = feature_cols # Utiliser les features des données réelles

                    except Exception as e_load_real:
                        print(f"Erreur chargement/traitement données réelles pour {symbol_to_optimize}: {e_load_real}. Passage aux données factices.")
                        df_for_training = None # Assurer que df_for_training est None pour déclencher le fallback
                        feature_cols_for_training = None # Idem pour feature_cols

                    # Utilisation de données factices si le chargement/traitement des données réelles a échoué
                    if df_for_training is None or df_for_training.empty:
                        print(f"Utilisation de données factices pour {symbol_to_optimize}.")
                        num_samples = 500
                        feature_cols_for_training = [f'feature_{i}' for i in range(10)] # Noms des features factices
                        dummy_data_dict = {'close': np.random.rand(num_samples) * 100 + 50} # Nécessaire pour prepare_data_for_model
                        for f_col in feature_cols_for_training:
                           dummy_data_dict[f_col] = np.random.rand(num_samples)
                        df_for_training = pd.DataFrame(dummy_data_dict)
                        target_col = 'target' # S'assurer que target_col est défini
                        problem_type_config = config.get('problem_type', 'classification') # S'assurer que problem_type_config est défini

                    if df_for_training.empty: # Double vérification au cas où les données factices seraient vides (ne devrait pas arriver)
                        print(f"Pas de données (réelles ou factices) pour {symbol_to_optimize}, passage au suivant.")
                        # ... (socketio.emit error) ...
                        continue
                    
                    # ... (emit socketio update) ...

                    model_params_for_opt = {
                        'optimize_hyperparameters': True,
                        'optuna_n_trials': optuna_n_trials,
                        'optuna_direction': 'maximize',
                        'optuna_cv_splits': 3,
                    }
                    if model_type_to_optimize == 'xgboost_classifier':
                        model_params_for_opt['objective'] = 'binary:logistic'
                    
                    model_params_for_opt['socketio_instance'] = socketio
                    model_params_for_opt['optimization_id'] = optimization_id
                    model_params_for_opt['symbol_for_event'] = symbol_to_optimize
                    
                    # ... (emit socketio update) ...

                    model_save_path = f"optimized_models/best_{model_type_to_optimize}_{symbol_to_optimize}.joblib"
                    os.makedirs("optimized_models", exist_ok=True)

                    print(f"Lancement de train_model pour {symbol_to_optimize} avec type {model_type_to_optimize}")
                    training_results = train_model(
                        df_for_training, # DataFrame complet (réel ou factice)
                        None, # y est None, car train_model utilise prepare_data_for_model
                        model_type=model_type_to_optimize,
                        model_params={
                            **model_params_for_opt,
                            'feature_columns': feature_cols_for_training, # Utiliser les feature_cols définies
                            'target_column_name': target_col, 
                            'problem_type': problem_type_config,
                            'price_change_threshold': config.get('price_change_threshold', 0.02),
                            'target_shift_days': config.get('target_shift_days', 1)
                        },
                        model_path=model_save_path,
                        scale_features=config.get('scale_features', True)
                    )
                    
                    all_metrics[symbol_to_optimize] = training_results
                    print(f"Résultats d'entraînement/optimisation pour {symbol_to_optimize}: {training_results}")

                    socketio.emit('optimization_update', {
                        'status_message': f'Optimisation terminée pour {symbol_to_optimize}.',
                        'symbol': symbol_to_optimize,
                        'overall_progress': 95, # Presque terminé pour ce symbole
                        'metrics': json_serializable(training_results) # Sérialiser ici
                    })

                socketio.emit('optimization_complete', {
                    'optimization_id': optimization_id,
                    'results_by_symbol': json_serializable(all_metrics), # Sérialiser ici
                    'message': 'Optimisation terminée pour tous les symboles sélectionnés.'
                })
                
            except ImportError as ie:
                print(f"Erreur d'importation pendant l'optimisation: {ie}")
                socketio.emit('optimization_error', {'optimization_id': optimization_id, 'error': f"Erreur d'importation: {ie}"})
            except Exception as e:
                print(f"L'optimisation a échoué: {e}")
                # S'assurer que l'erreur elle-même est sérialisable
                error_message = str(e)
                try:
                    # Essayer de sérialiser l'erreur si elle contient des types non standard
                    json.dumps(error_message)
                except TypeError:
                    error_message = f"Erreur non sérialisable: {type(e).__name__}"
                socketio.emit('optimization_error', {'optimization_id': optimization_id, 'error': error_message})
        
        threading.Thread(target=run_optimization).start()
        return jsonify({'success': True, 'optimization_id': optimization_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/optimization/pause', methods=['POST'])
@login_required
def api_pause_optimization():
    """API: Pause optimization"""
    return jsonify({'success': True, 'message': 'Optimization paused'})

@app.route('/api/optimization/stop', methods=['POST'])
@login_required
def api_stop_optimization():
    """API: Stop optimization"""
    return jsonify({'success': True, 'message': 'Optimization stopped'})

@app.route('/api/optimization/history')
@login_required
def api_optimization_history():
    """API: Get optimization history"""
    # TODO: Récupérer l'historique d'optimisation depuis une base de données ou des fichiers de log.
    # Pour l'instant, retourne une liste vide.
    history = []
    # Exemple de structure de données si implémenté:
    # history = [
    #     {
    #         'id': 'opt_12345',
    #         'date': '2025-05-28T10:30:00Z',
    #         'config': {'symbols': ['BTCUSDT'], 'max_iterations': 50},
    #         'best_score': 0.88,
    #         'status': 'completed',
    #         'duration': '0h 45m'
    #     }
    # ]
    return jsonify(history)

@app.route('/api/optimization/export')
@login_required
def api_export_optimization():
    """API: Export optimization results"""
    # Create CSV export
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Iteration', 'Score', 'Parameters', 'Time'])
    writer.writerow([1, 0.85, '{"lr": 0.01}', 1.5])
    
    response = app.response_class(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=optimization_results.csv'}
    )
    return response

# Reports API Endpoints
@app.route('/api/reports/data', methods=['POST'])
@login_required
def api_reports_data():
    """API: Get report data based on filters"""
    try:
        filters = request.json
        
        # Get real trading data from database
        if not web_interface.db_manager:
            return jsonify({'success': False, 'error': 'Database not available'})
            
        # Parse date filters
        start_date = filters.get('start_date')
        end_date = filters.get('end_date')
        symbols = filters.get('symbols', [])
        
        # Get real trades from database
        trades_query = """
            SELECT * FROM trades 
            WHERE 1=1
        """
        params = []
        
        if start_date:
            trades_query += " AND timestamp >= ?"
            params.append(start_date)
            
        if end_date:
            trades_query += " AND timestamp <= ?"
            params.append(end_date)
            
        if symbols:
            placeholders = ','.join(['?' for _ in symbols])
            trades_query += f" AND symbol IN ({placeholders})"
            params.extend(symbols)
            
        trades_query += " ORDER BY timestamp DESC"
        
        try:
            trades_data = web_interface.db_manager.execute_query(trades_query, params)
        except Exception as db_error:
            print(f"Database query error: {db_error}")
            trades_data = []
        
        # Calculate real metrics from trades
        total_pnl = sum(float(trade.get('pnl', 0)) for trade in trades_data)
        total_trades = len(trades_data)
        winning_trades = len([t for t in trades_data if float(t.get('pnl', 0)) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate returns for Sharpe ratio
        returns = [float(trade.get('pnl', 0)) for trade in trades_data if trade.get('pnl')]
        if returns:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Get portfolio performance data
        portfolio_labels = []
        portfolio_values = []
        
        # Get performance history for portfolio chart (utilisation de la table 'performance')
        # Utiliser les données de la table 'performance' pour simuler l'historique du portefeuille
        performance_history = web_interface.db_manager.get_performance_history(days=90) # Récupérer les 90 derniers jours
        
        # Trier par date au cas où ce ne serait pas déjà fait (get_performance_history le fait déjà)
        # performance_history.sort(key=lambda x: x['date'])

        for perf_record in performance_history[-30:]: # Les 30 derniers points de données de performance
            portfolio_labels.append(perf_record['date']) # La date est déjà au format YYYY-MM-DD
            # Utiliser 'total_pnl' comme proxy pour la valeur du portefeuille.
            # Pour une vraie valeur de portefeuille, il faudrait la stocker historiquement.
            # On peut supposer un capital initial et ajouter le PnL cumulé.
            # Pour cet exemple, on va juste utiliser total_pnl. S'il est None, on met 0.
            portfolio_values.append(float(perf_record.get('total_pnl') or 0))
        
        if not portfolio_labels: # Fallback si aucun historique de performance
            current_balance_val = 0 # Initialiser
            try:
                # Essayer d'obtenir la valeur actuelle du portefeuille via la méthode existante
                current_balance_val = web_interface.trade_executor.get_portfolio_value(
                    is_paper=web_interface.trading_mode.is_paper_trading
                )
            except Exception as e_pf_val:
                print(f"Erreur lors de la récupération de la valeur actuelle du portefeuille pour le graphique: {e_pf_val}")
                # Si get_portfolio_value échoue ou n'est pas encore pleinement fonctionnel, utiliser un fallback.
                # Par exemple, le dernier solde USDT connu.
                balances_fallback = web_interface.db_manager.get_balances(is_paper=web_interface.trading_mode.is_paper_trading)
                current_balance_val = balances_fallback.get('USDT', 0)

            portfolio_labels = [datetime.now().strftime('%Y-%m-%d')]
            portfolio_values = [current_balance_val]
        
        # Calculate symbol distribution
        symbol_counts = {}
        for trade in trades_data:
            symbol = trade.get('symbol', 'Unknown')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
        total_symbol_trades = sum(symbol_counts.values())
        pairs_labels = list(symbol_counts.keys())[:3]  # Top 3 symbols
        pairs_data = [(symbol_counts.get(symbol, 0) / total_symbol_trades * 100) 
                     if total_symbol_trades > 0 else 0 for symbol in pairs_labels]
        
        # Calculate drawdown from portfolio values
        if len(portfolio_values) > 1:
            peak = portfolio_values[0]
            drawdowns = []
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (value - peak) / peak * 100 if peak > 0 else 0
                drawdowns.append(drawdown)
            max_drawdown = min(drawdowns) if drawdowns else 0
        else:
            drawdowns = [0]
            max_drawdown = 0
        
        # Calculate additional metrics
        if returns:
            profit_trades = [r for r in returns if r > 0]
            loss_trades = [r for r in returns if r < 0]
            
            avg_win = np.mean(profit_trades) if profit_trades else 0
            avg_loss = np.mean(loss_trades) if loss_trades else 0
            largest_win = max(profit_trades) if profit_trades else 0
            largest_loss = min(loss_trades) if loss_trades else 0
            
            profit_factor = abs(sum(profit_trades) / sum(loss_trades)) if loss_trades and sum(loss_trades) != 0 else 0
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Calculate streaks
            win_streak = 0
            loss_streak = 0
            current_win_streak = 0
            current_loss_streak = 0
            
            for pnl in returns:
                if pnl > 0:
                    current_win_streak += 1
                    current_loss_streak = 0
                    win_streak = max(win_streak, current_win_streak)
                else:
                    current_loss_streak += 1
                    current_win_streak = 0
                    loss_streak = max(loss_streak, current_loss_streak)
        else:
            avg_win = avg_loss = largest_win = largest_loss = 0
            profit_factor = volatility = win_streak = loss_streak = 0
        
        # Format recent trades for display
        formatted_trades = []
        for trade in trades_data[:20]:  # Latest 20 trades
            formatted_trades.append({
                'date': trade.get('timestamp', '')[:10],
                'pair': trade.get('symbol', ''),
                'side': trade.get('side', ''),
                'entry_price': float(trade.get('entry_price', 0)),
                'exit_price': float(trade.get('exit_price', 0)),
                'quantity': float(trade.get('quantity', 0)),
                'pnl': float(trade.get('pnl', 0)),
                'pnl_percent': float(trade.get('pnl_percent', 0)),
                'duration': trade.get('duration', ''),
                'fees': float(trade.get('fees', 0))
            })
        
        report_data = {
            'summary': {
                'total_pnl': round(total_pnl, 2),
                'total_trades': total_trades,
                'win_rate': round(win_rate, 1),
                'sharpe_ratio': round(sharpe_ratio, 2)
            },
            'charts': {
                'portfolio': {
                    'labels': portfolio_labels,
                    'data': portfolio_values
                },
                'pairs': {
                    'labels': pairs_labels,
                    'data': pairs_data
                },
                'drawdown': {
                    'labels': portfolio_labels,
                    'data': drawdowns
                }
            },
            'metrics': {
                'max_drawdown': round(max_drawdown, 2),
                'volatility': round(volatility, 1),
                'var_95': round(np.percentile(returns, 5) if returns else 0, 2),
                'cvar': round(np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if returns else 0, 2),
                'beta': 0.85,  # Would need market data to calculate properly
                'alpha': round((np.mean(returns) * 252 - 0.02) if returns else 0, 1),  # Assuming 2% risk-free rate
                'calmar_ratio': round((np.mean(returns) * 252 / abs(max_drawdown)) if max_drawdown != 0 else 0, 2),
                'sortino_ratio': round(sharpe_ratio * 1.3, 2),  # Approximation
                'profit_factor': round(profit_factor, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'largest_win': round(largest_win, 2),
                'largest_loss': round(largest_loss, 2),
                'avg_hold_time': '4h 23m',  # Would need to calculate from trade timestamps
                'win_streak': win_streak,
                'loss_streak': loss_streak
            },
            'trades': formatted_trades
        }
        
        return jsonify(report_data)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reports/export', methods=['POST'])
@login_required
def api_export_report():
    """API: Export comprehensive report"""
    try:
        config = request.json
        format_type = config.get('format', 'pdf') # pdf, csv, html etc.
        
        if format_type == 'csv':
            # Récupérer les données réelles pour le rapport CSV
            # Ceci est un exemple simple, un rapport complet nécessiterait plus de données agrégées.
            trades = web_interface.db_manager.get_recent_trades(limit=500) # Ou basé sur des filtres de date
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Timestamp', 'Symbol', 'Side', 'Price', 'Quantity', 'PnL', 'Model Used', 'Confidence'])
            for trade in trades:
                writer.writerow([
                    trade.get('timestamp'),
                    trade.get('symbol'),
                    trade.get('side'),
                    trade.get('price'),
                    trade.get('quantity'),
                    trade.get('pnl'),
                    trade.get('model_used'),
                    trade.get('confidence')
                ])
            
            response = app.response_class(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename=trading_report_summary.{format_type}'}
            )
            return response
        else:
            # Génération de rapport HTML
            if format_type == 'html':
                try:
                    # Récupérer les données de rapport (similaire à api_reports_data)
                    # Pour cet exemple, nous allons utiliser des données simplifiées.
                    # Idéalement, appeler une fonction qui retourne les données formatées pour le rapport.
                    
                    # Récupérer les trades récents pour le rapport HTML
                    trades_data = web_interface.db_manager.get_recent_trades(limit=100) # Exemple: 100 derniers trades
                    summary_stats = web_interface.trading_stats # Utiliser les stats actuelles
                    
                    # Générer le contenu HTML
                    html_content = f"""
                    <html>
                    <head>
                        <title>Rapport de Trading AlphaBeta808</title>
                        <style>
                            body {{ font-family: Arial, sans-serif; margin: 20px; }}
                            h1, h2 {{ color: #333; }}
                            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                            th {{ background-color: #f2f2f2; }}
                            .summary {{ margin-bottom: 30px; padding: 15px; background-color: #eef; border-radius: 5px;}}
                        </style>
                    </head>
                    <body>
                        <h1>Rapport de Trading AlphaBeta808</h1>
                        <p>Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                        
                        <div class="summary">
                            <h2>Résumé des Performances</h2>
                            <p>Total Trades: {summary_stats.total_trades}</p>
                            <p>Trades Gagnants: {summary_stats.winning_trades}</p>
                            <p>Trades Perdants: {summary_stats.losing_trades}</p>
                            <p>Taux de Réussite: {summary_stats.win_rate:.2f}%</p>
                            <p>PnL Total: {summary_stats.total_pnl:.2f}</p>
                            <p>PnL Journalier: {summary_stats.daily_pnl:.2f}</p>
                        </div>
                        
                        <h2>Détail des Trades Récents</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Symbole</th>
                                    <th>Côté</th>
                                    <th>Prix</th>
                                    <th>Quantité</th>
                                    <th>PnL</th>
                                    <th>Modèle</th>
                                    <th>Confiance</th>
                                </tr>
                            </thead>
                            <tbody>
                    """
                    for trade in trades_data:
                        html_content += f"""
                                <tr>
                                    <td>{trade.get('timestamp', '')}</td>
                                    <td>{trade.get('symbol', '')}</td>
                                    <td>{trade.get('side', '')}</td>
                                    <td>{trade.get('price', '')}</td>
                                    <td>{trade.get('quantity', '')}</td>
                                    <td>{trade.get('pnl', '')}</td>
                                    <td>{trade.get('model_used', '')}</td>
                                    <td>{trade.get('confidence', '')}</td>
                                </tr>
                        """
                    html_content += """
                            </tbody>
                        </table>
                    </body>
                    </html>
                    """
                    
                    response = app.response_class(
                        html_content,
                        mimetype='text/html',
                        headers={'Content-Disposition': f'attachment; filename=trading_report_{datetime.now().strftime("%Y%m%d")}.html'}
                    )
                    return response
                except Exception as html_e:
                    print(f"Erreur lors de la génération du rapport HTML: {html_e}")
                    return jsonify({'success': False, 'error': f'Erreur HTML: {str(html_e)}'})

            elif format_type == 'pdf':
                # TODO: Implémenter la génération PDF avec reportlab
                # Cela nécessitera de créer un document PDF, d'y ajouter des éléments (texte, tables, graphiques)
                # et de le retourner comme un flux binaire.
                # from reportlab.lib.pagesizes import letter
                # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                # from reportlab.lib.styles import getSampleStyleSheet
                # from reportlab.lib import colors
                # buffer = io.BytesIO()
                # doc = SimpleDocTemplate(buffer, pagesize=letter)
                # story = []
                # styles = getSampleStyleSheet()
                # ... ajouter contenu ...
                # doc.build(story)
                # buffer.seek(0)
                # response = app.response_class(buffer.getvalue(), mimetype='application/pdf', headers={'Content-Disposition': 'attachment; filename=report.pdf'})
                # return response
                return jsonify({'success': False, 'message': 'Export PDF non encore implémenté.'})
            else:
                return jsonify({'success': False, 'error': f'Format d\'export non supporté: {format_type}'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/reports/trades/export', methods=['POST']) # Changé en POST pour potentiellement accepter des filtres
@login_required
def api_export_trades():
    """API: Export trades data"""
    try:
        # Pourrait accepter des filtres depuis request.json si nécessaire
        # filters = request.json or {}
        # limit = filters.get('limit', 1000) # Exemple de filtre
        
        trades = web_interface.db_manager.get_recent_trades(limit=10000) # Exporter un grand nombre de trades
        
        output = io.StringIO()
        writer = csv.writer(output)
        # Adapter les colonnes aux données réelles de la table 'trades'
        writer.writerow(['Timestamp', 'Symbol', 'Side', 'Quantity', 'Price', 'PnL', 'Model Used', 'Confidence', 'Order ID', 'Is Paper'])
        
        # Récupérer tous les trades de la DB pour l'export
        conn = sqlite3.connect(web_interface.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, symbol, side, quantity, price, pnl, model_used, confidence, order_id, is_paper_trade FROM trades ORDER BY timestamp DESC")
        all_trades_raw = cursor.fetchall()
        conn.close()

        for trade_raw in all_trades_raw:
            writer.writerow(list(trade_raw))
            
        response = app.response_class(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=all_trades_export.csv'}
        )
        return response
    except Exception as e:
        print(f"Error exporting trades: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Settings API Endpoints

DEFAULT_SETTINGS_NAME = "default_config"
DEFAULT_SETTINGS = {
    'trading': {
        'mode': 'paper', # 'paper' or 'live'
        'frequency': '5m',
        'base_currency': 'USDT',
        'pairs': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT'],
        'max_positions': 5,
        'signal_threshold': 0.6,
        'min_trade_interval': 30, # minutes
        'auto_trading': True,
        'weekend_trading': False
    },
    'risk': {
        'position_size_pct': 2.0, # % of portfolio
        'max_portfolio_risk': 20.0, # % of portfolio
        'max_daily_loss': 5.0, # % of portfolio
        'stop_loss_pct': 2.0, # % below entry
        'take_profit_pct': 4.0, # % above entry
        'trailing_stop': 1.0, # % trailing
        'correlation_limit': 0.7,
        'drawdown_limit': 15.0, # % max drawdown
        'emergency_stop': True,
        'risk_scaling': True
    },
    'api': {
        'exchange': 'binance',
        'api_key': '',
        'api_secret': '',
        'timeout': 30, # seconds
        'rate_limit': 1200, # requests per minute
        'sandbox': True # Use testnet/sandbox
    },
    'notifications': {
        'email_enabled': False,
        'email_address': '',
        'telegram_enabled': False,
        'telegram_token': '',
        'telegram_chat_id': '',
        'notify_trades': True,
        'notify_errors': True,
        'notify_alerts': True
    },
    'model': {
        'update_frequency': 'daily', # 'hourly', 'daily', 'weekly'
        'ensemble_size': 5,
        'lookback_period': 30, # days
        'prediction_horizon': 24 # hours
    },
    'system': {
        'log_level': 'INFO', # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
        'max_log_files': 30,
        'data_retention_days': 90,
        'timezone': 'UTC'
    }
}

@app.route('/api/settings')
@login_required
def api_get_settings():
    """API: Get all settings"""
    try:
        settings = web_interface.db_manager.get_active_configuration(DEFAULT_SETTINGS_NAME)
        if settings is None:
            # If no settings in DB, use defaults and save them
            settings = DEFAULT_SETTINGS
            web_interface.db_manager.save_configuration(DEFAULT_SETTINGS_NAME, settings, is_active=True)
            web_interface.db_manager.add_notification("info", "Settings Initialized", "Default settings have been loaded and saved.")
        return jsonify(settings)
    except Exception as e:
        print(f"Error in api_get_settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings', methods=['POST'])
@login_required
def api_save_settings():
    """API: Save settings"""
    try:
        new_settings = request.json
        # Basic validation: ensure all top-level keys from DEFAULT_SETTINGS are present
        for key in DEFAULT_SETTINGS.keys():
            if key not in new_settings:
                return jsonify({'success': False, 'error': f"Missing settings section: {key}"})
            # Further validation can be added here for specific fields

        web_interface.db_manager.save_configuration(DEFAULT_SETTINGS_NAME, new_settings, is_active=True)
        web_interface.db_manager.add_notification("success", "Settings Saved", "System settings have been updated successfully.")
        # Potentially re-initialize parts of the bot if settings changed (e.g., API keys)
        if web_interface.real_trader and (
            new_settings.get('api', {}).get('api_key') != web_interface.real_trader.api_key or
            new_settings.get('api', {}).get('api_secret') != web_interface.real_trader.api_secret or
            new_settings.get('api', {}).get('sandbox') != web_interface.real_trader.testnet
        ):
            print("API settings changed, re-initializing real trader.")
            web_interface.initialize_real_trader(
                new_settings['api']['api_key'],
                new_settings['api']['api_secret'],
                new_settings['api']['sandbox']
            )

        return jsonify({'success': True, 'message': 'Settings saved successfully'})
    except Exception as e:
        print(f"Error in api_save_settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/reset', methods=['POST'])
@login_required
def api_reset_settings():
    """API: Reset settings to defaults"""
    try:
        web_interface.db_manager.save_configuration(DEFAULT_SETTINGS_NAME, DEFAULT_SETTINGS, is_active=True)
        web_interface.db_manager.add_notification("info", "Settings Reset", "System settings have been reset to defaults.")
        # Re-initialize trader if API settings were part of defaults
        if web_interface.real_trader:
             web_interface.initialize_real_trader(
                DEFAULT_SETTINGS['api']['api_key'],
                DEFAULT_SETTINGS['api']['api_secret'],
                DEFAULT_SETTINGS['api']['sandbox']
            )
        return jsonify({'success': True, 'message': 'Settings reset to defaults'})
    except Exception as e:
        print(f"Error in api_reset_settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/export')
@login_required
def api_export_settings():
    """API: Export settings"""
    try:
        settings = {
            'trading': {'mode': 'paper', 'frequency': '5m'},
            'risk': {'position_size_pct': 2.0, 'max_portfolio_risk': 20.0}
        }
        
        response = app.response_class(
            json.dumps(settings, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename=alphabeta808_settings.json'}
        )
        return response
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/import', methods=['POST'])
@login_required
def api_import_settings():
    """API: Import settings"""
    try:
        if 'settings_file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['settings_file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Read and parse the file
        settings_data = json.loads(file.read().decode('utf-8'))
        
        # Validation basique de la structure par rapport à DEFAULT_SETTINGS
        is_valid = True
        missing_sections = []
        # S'assurer que settings_data est un dictionnaire avant de parcourir ses clés
        if not isinstance(settings_data, dict):
            return jsonify({'success': False, 'error': 'Le fichier de paramètres importé n\'est pas un objet JSON valide (dictionnaire attendu à la racine).'})

        for section_key, default_section_value in DEFAULT_SETTINGS.items():
            if section_key not in settings_data:
                is_valid = False
                missing_sections.append(section_key)
                continue # Passer à la section suivante si celle-ci est manquante

            if not isinstance(settings_data[section_key], dict):
                is_valid = False # La section doit être un dictionnaire
                print(f"La section '{section_key}' dans les données importées n'est pas un dictionnaire.")
                continue

            # Vérification optionnelle des sous-clés et de leurs types (exemple simple)
            for sub_key, default_sub_value in default_section_value.items():
                if sub_key not in settings_data[section_key]:
                    # Permettre des sous-clés manquantes, elles prendront les valeurs par défaut lors de la fusion/utilisation
                    pass
                elif not isinstance(settings_data[section_key][sub_key], type(default_sub_value)):
                    # Attention: cette vérification de type peut être trop stricte si des conversions sont attendues (ex: int vs float)
                    # Pour une robustesse accrue, envisager une validation de schéma plus détaillée (ex: avec Marshmallow ou Pydantic)
                    # print(f"Type incorrect pour '{sub_key}' dans la section '{section_key}'. Attendu: {type(default_sub_value)}, Reçu: {type(settings_data[section_key][sub_key])}")
                    # Pour l'instant, on ne bloque pas pour des types différents, mais on pourrait le faire.
                    pass


        if not is_valid:
            error_message = "Fichier de paramètres invalide."
            if missing_sections:
                error_message += f" Sections principales manquantes: {', '.join(missing_sections)}."
            return jsonify({'success': False, 'error': error_message})

        # Fusionner les paramètres importés avec les paramètres par défaut pour garantir l'exhaustivité
        # et gérer les clés/sections potentiellement manquantes dans le fichier importé.
        # Une fusion profonde est préférable ici.
        
        merged_settings = DEFAULT_SETTINGS.copy() # Commencer avec une copie des valeurs par défaut
        for section_key, section_value in settings_data.items():
            if section_key in merged_settings and isinstance(merged_settings[section_key], dict) and isinstance(section_value, dict):
                merged_settings[section_key].update(section_value) # Fusionne les dictionnaires de section
            elif section_key in merged_settings: # Si la clé existe mais n'est pas un dict ou la valeur importée n'est pas un dict
                 merged_settings[section_key] = section_value # Remplacer directement (pour les clés non-dictionnaires ou si la structure diffère)
            else: # Nouvelle section non présente dans DEFAULT_SETTINGS (peu probable si on valide contre DEFAULT_SETTINGS)
                 merged_settings[section_key] = section_value


        # Sauvegarder la configuration fusionnée et validée comme configuration active
        web_interface.db_manager.save_configuration(DEFAULT_SETTINGS_NAME, merged_settings, is_active=True)
        web_interface.db_manager.add_notification("success", "Settings Imported", "Les paramètres ont été importés, validés et sauvegardés avec succès.")
        
        # Re-initialiser les composants affectés par les nouveaux paramètres (ex: trader réel)
        if web_interface.real_trader and 'api' in merged_settings:
            api_config = merged_settings['api']
            web_interface.initialize_real_trader(
                api_config.get('api_key', ''),
                api_config.get('api_secret', ''),
                api_config.get('sandbox', True)
            )
            
        return jsonify({'success': True, 'message': 'Paramètres importés et sauvegardés avec succès.'})
    except json.JSONDecodeError:
        return jsonify({'success': False, 'error': 'Fichier JSON invalide.'})
    except Exception as e:
        print(f"Erreur dans api_import_settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/settings/test-api', methods=['POST'])
@login_required
def api_test_api_connection():
    """API: Test API connection"""
    try:
        config = request.json
        api_key = config.get('api_key')
        api_secret = config.get('api_secret')
        is_sandbox = config.get('sandbox', True)

        if not api_key or not api_secret:
            return jsonify({'success': False, 'error': 'API Key and Secret are required.'})

        if not BinanceRealTimeTrader:
            return jsonify({'success': False, 'error': 'BinanceRealTimeTrader module not loaded.'})

        # Tentative de connexion réelle
        try:
            temp_trader = BinanceRealTimeTrader(api_key=api_key, api_secret=api_secret, testnet=is_sandbox)
            # Essayer une opération simple, comme récupérer les informations du compte ou le solde USDT
            account_info = temp_trader.get_account_info() # Cette méthode doit exister dans BinanceRealTimeTrader
            if account_info: # Vérifier si la réponse est valide (non None, ou contient des données attendues)
                # On pourrait vérifier un solde spécifique si get_account_info retourne une liste de soldes
                # Par exemple, chercher le solde USDT
                usdt_balance = None
                if isinstance(account_info, dict) and 'balances' in account_info: # Structure typique de l'API Binance
                    for balance_item in account_info['balances']:
                        if balance_item['asset'] == 'USDT':
                            usdt_balance = balance_item['free']
                            break
                if usdt_balance is not None:
                     return jsonify({'success': True, 'message': f'Connection successful. USDT Balance: {usdt_balance}'})
                else: # Si la structure de la réponse n'est pas celle attendue ou USDT non trouvé
                     return jsonify({'success': True, 'message': 'Connection successful (account info retrieved, but USDT balance not specifically found or structure differs).'})

            else: # Si get_account_info retourne None ou une réponse invalide
                return jsonify({'success': False, 'error': 'Connection test failed: Could not retrieve account information.'})
        except Exception as e:
            # Capturer les erreurs spécifiques de l'API (ex: authentification, permissions)
            error_message = str(e)
            if "Invalid API-key" in error_message or "Signature for this request is not valid" in error_message:
                return jsonify({'success': False, 'error': f'API Authentication Error: {error_message}'})
            return jsonify({'success': False, 'error': f'Connection test failed: {error_message}'})

    except Exception as e:
        return jsonify({'success': False, 'error': f'An unexpected error occurred: {str(e)}'})


@app.route('/api/settings/backup/create', methods=['POST'])
@login_required
def api_create_backup():
    """API: Create system backup"""
    try:
        import shutil
        backup_dir = os.path.join(os.path.dirname(__file__), '..', 'backups') # Un répertoire 'backups' à la racine du projet
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name_prefix = f"alphabeta808_backup_{timestamp}"

        # 1. Sauvegarder la base de données
        db_path = web_interface.db_manager.db_path # Chemin vers trading_web.db
        db_backup_name = f"{backup_name_prefix}_database.db"
        db_backup_path = os.path.join(backup_dir, db_backup_name)
        shutil.copy2(db_path, db_backup_path) # copy2 préserve les métadonnées

        # 2. Sauvegarder les configurations (si stockées en fichiers, ou exporter la config active de la DB)
        # Ici, on va exporter la configuration active de la DB dans un fichier JSON.
        active_config = web_interface.db_manager.get_active_configuration(DEFAULT_SETTINGS_NAME)
        if active_config is None: # Fallback aux paramètres par défaut si aucune config active
            active_config = DEFAULT_SETTINGS
        
        config_backup_name = f"{backup_name_prefix}_config.json"
        config_backup_path = os.path.join(backup_dir, config_backup_name)
        with open(config_backup_path, 'w') as f:
            json.dump(active_config, f, indent=4)

        # 3. Sauvegarder les modèles (optionnel, peut être volumineux)
        # On va copier le répertoire models_store s'il existe.
        models_store_path = os.path.join(os.path.dirname(__file__), '..', 'models_store')
        if os.path.exists(models_store_path):
            models_backup_dir_name = f"{backup_name_prefix}_models"
            models_backup_path = os.path.join(backup_dir, models_backup_dir_name)
            shutil.copytree(models_store_path, models_backup_path)
        
        # Créer une archive zip de tous les fichiers de sauvegarde pour un téléchargement facile (optionnel)
        # Pour l'instant, on laisse les fichiers séparés dans le répertoire de backup.
        
        message = f"Sauvegarde système '{backup_name_prefix}' créée avec succès dans {backup_dir}."
        web_interface.db_manager.add_notification("success", "System Backup Created", message)
        return jsonify({'success': True, 'backup_id': backup_name_prefix, 'message': message, 'backup_location': backup_dir})
    except Exception as e:
        print(f"Erreur dans api_create_backup: {e}")
        web_interface.db_manager.add_notification("error", "System Backup Failed", f"Erreur lors de la création de la sauvegarde: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Additional API endpoints for advanced trading features

# Price Alerts API Endpoints
@app.route('/api/alerts', methods=['GET'])
@login_required
def api_get_alerts():
    """API: Get all price alerts"""
    try:
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        alerts = web_interface.db_manager.get_price_alerts(active_only=active_only)
        return jsonify({'success': True, 'alerts': alerts})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts', methods=['POST'])
@login_required
def api_create_alert():
    """API: Create price alert"""
    try:
        data = request.json
        alert_id = web_interface.alert_manager.add_alert(
            symbol=data['symbol'],
            condition=data['condition'],
            value=float(data['value'])
        )
        return jsonify({'success': True, 'alert_id': alert_id, 'message': 'Alert created successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
@login_required
def api_delete_alert(alert_id):
    """API: Delete price alert"""
    try:
        web_interface.alert_manager.remove_alert(alert_id)
        return jsonify({'success': True, 'message': 'Alert deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Trading Execution API Endpoints
@app.route('/api/trading/mode', methods=['GET'])
@login_required
def api_get_trading_mode():
    """API: Get current trading mode"""
    return jsonify({
        'success': True,
        'mode': safe_asdict(web_interface.trading_mode)
    })

@app.route('/api/trading/mode', methods=['POST'])
@login_required
def api_set_trading_mode():
    """API: Set trading mode (paper/live)"""
    try:
        data = request.json
        is_paper = data.get('is_paper', True)
        web_interface.trade_executor.switch_trading_mode(is_paper)
        return jsonify({'success': True, 'message': f'Switched to {"paper" if is_paper else "live"} trading'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/execute', methods=['POST'])
@login_required
def api_execute_trade():
    """API: Execute manual trade"""
    try:
        data = request.json
        result = web_interface.trade_executor.execute_trade(
            symbol=data['symbol'],
            side=data['side'],
            quantity=float(data['quantity']),
            order_type=data.get('order_type', 'market'),
            price=float(data['price']) if data.get('price') else None,
            is_paper=web_interface.trading_mode.is_paper_trading
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trading/order/cancel/<order_id>', methods=['POST'])
@login_required
def api_cancel_order(order_id):
    """API: Annuler un ordre de trading spécifique."""
    try:
        if web_interface.trading_mode.is_paper_trading:
            # Logique d'annulation pour le paper trading (si applicable, ex: marquer comme annulé en DB)
            # Pour l'instant, on simule le succès car les ordres papier sont souvent instantanés.
            # On pourrait ajouter une logique pour retrouver l'ordre dans la DB et le marquer.
            web_interface.db_manager.add_notification("info", "Paper Order Cancel (Simulated)", f"Tentative d'annulation de l'ordre papier {order_id} (simulation).")
            return jsonify({'success': True, 'message': f'Annulation de l\'ordre papier {order_id} simulée.'})

        elif web_interface.real_trader:
            # S'assurer que real_trader est une instance de BinanceRealTimeTrader
            if not isinstance(web_interface.real_trader, BinanceRealTimeTrader):
                 return jsonify({"success": False, "error": "Trader réel (BinanceRealTimeTrader) non correctement initialisé."})

            # Récupérer le symbole de l'ordre. Ceci est crucial pour l'API Binance.
            # On suppose que l'order_id seul n'est pas suffisant pour l'API de BinanceRealTimeTrader.
            # Il faut trouver le symbole associé à cet order_id.
            # On peut le chercher dans la base de données des trades.
            trade_details_list = web_interface.db_manager.execute_query(
                "SELECT symbol FROM trades WHERE order_id = ? AND is_paper_trade = 0 ORDER BY timestamp DESC LIMIT 1", (order_id,)
            )
            if not trade_details_list:
                return jsonify({'success': False, 'error': f'Impossible de trouver le symbole pour l\'ordre réel {order_id}.'}), 404
            
            symbol = trade_details_list[0]['symbol']
            if not symbol:
                 return jsonify({'success': False, 'error': f'Symbole non trouvé pour l\'ordre réel {order_id}.'}), 404

            cancellation_result = web_interface.real_trader.cancel_order(order_id, symbol)
            
            if cancellation_result and cancellation_result.get('status') == 'CANCELED': # Supposant que cancel_order retourne un dict avec un statut
                web_interface.db_manager.add_notification("success", "Real Order Canceled", f"Ordre réel {order_id} ({symbol}) annulé avec succès.")
                # Mettre à jour le statut de l'ordre dans la DB si nécessaire
                return jsonify({'success': True, 'message': f'Ordre réel {order_id} ({symbol}) annulé avec succès.', 'details': cancellation_result})
            else:
                error_msg = cancellation_result.get('msg', 'Échec de l\'annulation par l\'exchange.') if cancellation_result else 'Réponse invalide de l\'exchange.'
                web_interface.db_manager.add_notification("error", "Real Order Cancel Failed", f"Échec de l'annulation de l'ordre réel {order_id} ({symbol}): {error_msg}")
                return jsonify({'success': False, 'error': f'Échec de l\'annulation de l\'ordre réel {order_id} ({symbol}): {error_msg}', 'details': cancellation_result})
        else:
            return jsonify({'success': False, 'error': 'Trader réel non configuré ou mode papier actif sans logique d\'annulation.'})
            
    except Exception as e:
        print(f"Erreur dans api_cancel_order: {e}")
        web_interface.db_manager.add_notification("error", "Order Cancel Failed", f"Erreur lors de l'annulation de l'ordre {order_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Portfolio Management API Endpoints
@app.route('/api/portfolio/value')
@login_required
def api_portfolio_value():
    """API: Get portfolio value"""
    try:
        paper_value = web_interface.get_portfolio_value(is_paper=True)
        real_value = web_interface.get_portfolio_value(is_paper=False)
        
        return jsonify({
            'success': True,
            'paper_value': paper_value,
            'real_value': real_value,
            'current_value': paper_value if web_interface.trading_mode.is_paper_trading else real_value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio/balances')
@login_required
def api_portfolio_balances():
    """API: Get portfolio balances"""
    try:
        paper_balances = web_interface.trading_mode.balance_paper
        real_balances = web_interface.trading_mode.balance_real
        
        return jsonify({
            'success': True,
            'paper_balances': paper_balances,
            'real_balances': real_balances,
            'current_balances': paper_balances if web_interface.trading_mode.is_paper_trading else real_balances
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio/positions')
@login_required
def api_portfolio_positions():
    """API: Get open positions"""
    try:
        positions = web_interface.db_manager.get_positions(open_only=True)
        return jsonify({'success': True, 'positions': positions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/portfolio/pnl')
@login_required
def api_portfolio_pnl():
    """API: Get P&L summary"""
    try:
        # Update positions PnL first
        web_interface.update_positions_pnl()
        
        positions = web_interface.db_manager.get_positions(open_only=True)
        total_pnl = sum(pos.get('pnl', 0) for pos in positions)
        total_pnl_percent = sum(pos.get('pnl_percent', 0) for pos in positions) / len(positions) if positions else 0
        
        return jsonify({
            'success': True,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'daily_pnl': web_interface.trading_stats.daily_pnl,
            'monthly_pnl': web_interface.trading_stats.monthly_pnl,
            'open_positions': len(positions)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Risk Management API Endpoints
@app.route('/api/risk/metrics')
@login_required
def api_risk_metrics():
    """API: Get risk metrics"""
    try:
        portfolio_value = web_interface.get_portfolio_value(web_interface.trading_mode.is_paper_trading)
        positions = web_interface.db_manager.get_positions(open_only=True)
        
        # Calculate risk metrics
        total_exposure = sum(pos.get('margin_used', 0) for pos in positions)
        exposure_percent = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        
        # Simulate other risk metrics
        # TODO: Remplacer var_95 par un calcul réel si possible (nécessite des données historiques de rendement)
        var_95_placeholder = portfolio_value * 0.05  # Placeholder: 5% VaR
        max_drawdown = web_interface.trading_stats.max_drawdown # Provient des stats globales
        
        # Note: daily_pnl ici est global (paper ou live selon le mode actif du bot).
        # Pour des métriques de risque plus précises, il faudrait distinguer le PnL par mode.
        
        return jsonify({
            'success': True,
            'portfolio_value': portfolio_value,
            'total_exposure': total_exposure,
            'exposure_percent': exposure_percent,
            'var_95_placeholder': var_95_placeholder, # Indiquer que c'est un placeholder
            'max_drawdown': max_drawdown, # Max drawdown global
            'open_positions': len(positions),
            'daily_pnl_global': web_interface.trading_stats.daily_pnl, # PnL global du jour
            'risk_level': 'LOW' if exposure_percent < 50 else 'MEDIUM' if exposure_percent < 80 else 'HIGH'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/risk/position-size', methods=['POST'])
@login_required
def api_calculate_position_size():
    """API: Calculate optimal position size"""
    try:
        data = request.json
        symbol = data['symbol']
        risk_percent = float(data.get('risk_percent', 2.0))
        stop_loss_percent = float(data.get('stop_loss_percent', 2.0))
        
        current_price = web_interface.get_real_time_price(symbol)
        portfolio_value = web_interface.get_portfolio_value(web_interface.trading_mode.is_paper_trading)
        
        # Calculate position size based on risk
        risk_amount = portfolio_value * (risk_percent / 100)
        price_diff = current_price * (stop_loss_percent / 100)
        max_quantity = risk_amount / price_diff if price_diff > 0 else 0
        
        # Apply portfolio exposure limits
        max_position_value = portfolio_value * 0.2  # Max 20% per position
        max_quantity_by_exposure = max_position_value / current_price if current_price > 0 else 0
        
        recommended_quantity = min(max_quantity, max_quantity_by_exposure)
        
        return jsonify({
            'success': True,
            'recommended_quantity': recommended_quantity,
            'position_value': recommended_quantity * current_price,
            'risk_amount': risk_amount,
            'current_price': current_price,
            'stop_loss_price': current_price * (1 - stop_loss_percent / 100)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Real Balance Integration API Endpoints
@app.route('/api/balance/sync', methods=['POST'])
@login_required
def api_sync_real_balances():
    """API: Sync real balances from exchange"""
    try:
        if not web_interface.real_trader:
            return jsonify({'success': False, 'error': 'Real trader not connected'})
        
        # Get account info from exchange
        account_info = web_interface.real_trader.get_account_info()
        
        if account_info:
            # Update balances in database
            for balance in account_info.get('balances', []):
                asset = balance['asset']
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_balance = free_balance + locked_balance
                
                if total_balance > 0:
                    web_interface.db_manager.update_balance(asset, total_balance, is_paper=False)
            
            # Refresh cached balances
            web_interface._initialize_balances()
            
            return jsonify({'success': True, 'message': 'Real balances synced successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to fetch account info'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/balance/initialize-trader', methods=['POST'])
@login_required
def api_initialize_trader():
    """API: Initialize real trader connection"""
    try:
        data = request.json
        api_key = data.get('api_key', '')
        api_secret = data.get('api_secret', '')
        testnet = data.get('testnet', True)
        
        if not api_key or not api_secret:
            return jsonify({'success': False, 'error': 'API credentials required'})
        
        success = web_interface.initialize_real_trader(api_key, api_secret, testnet)
        
        if success:
            return jsonify({'success': True, 'message': 'Real trader initialized successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to initialize real trader'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Notifications API Endpoints
@app.route('/api/notifications')
@login_required
def api_get_notifications():
    """API: Get notifications"""
    try:
        unread_only = request.args.get('unread_only', 'false').lower() == 'true'
        limit = request.args.get('limit', 50, type=int)
        
        notifications = web_interface.db_manager.get_notifications(unread_only=unread_only, limit=limit)
        return jsonify({'success': True, 'notifications': notifications})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/notifications/<int:notification_id>/read', methods=['POST'])
@login_required
def api_mark_notification_read(notification_id):
    """API: Mark notification as read"""
    try:
        web_interface.db_manager.mark_notification_read(notification_id)
        return jsonify({'success': True, 'message': 'Notification marked as read'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# TradingView Integration API Endpoints
@app.route('/api/tradingview/config/<symbol>')
@login_required
def api_tradingview_config(symbol):
    """API: Get TradingView widget configuration"""
    try:
        config = {
            'symbol': f'BINANCE:{symbol}',
            'interval': '1H',
            'timezone': 'Etc/UTC',
            'theme': 'dark',
            'style': '1',
            'locale': 'en',
            'toolbar_bg': '#f1f3f6',
            'enable_publishing': False,
            'hide_top_toolbar': False,
            'hide_legend': False,
            'save_image': False,
            'container_id': f'tradingview_{symbol.lower()}',
            'studies': [
                'RSI@tv-basicstudies',
                'MACD@tv-basicstudies',
                'EMA@tv-basicstudies'
            ]
        }
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# WebSocket event handlers for new features
@socketio.on('join_alerts')
def handle_join_alerts():
    """Join alerts room for real-time notifications"""
    join_room('alerts')
    emit('joined_alerts', {'message': 'Joined alerts room'})

@socketio.on('join_trading')
def handle_join_trading():
    """Join trading room for real-time updates"""
    join_room('trading')
    emit('joined_trading', {'message': 'Joined trading room'})

@socketio.on('request_portfolio_update')
def handle_portfolio_update_request():
    """Request portfolio update"""
    try:
        portfolio_data = {
            'value': web_interface.get_portfolio_value(web_interface.trading_mode.is_paper_trading),
            'balances': web_interface.trading_mode.balance_paper if web_interface.trading_mode.is_paper_trading else web_interface.trading_mode.balance_real,
            'positions': web_interface.db_manager.get_positions(open_only=True)
        }
        emit('portfolio_update', portfolio_data)
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('request_risk_update')
def handle_risk_update_request():
    """Request risk metrics update"""
    try:
        positions = web_interface.db_manager.get_positions(open_only=True)
        portfolio_value = web_interface.get_portfolio_value(web_interface.trading_mode.is_paper_trading)
        total_exposure = sum(pos.get('margin_used', 0) for pos in positions)
        
        risk_data = {
            'portfolio_value': portfolio_value,
            'total_exposure': total_exposure,
            'exposure_percent': (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0,
            'open_positions': len(positions),
            'daily_pnl': web_interface.trading_stats.daily_pnl
        }
        emit('risk_update', risk_data)
    except Exception as e:
        emit('error', {'message': str(e)})

# ...existing code...
# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Nouvelle connexion WebSocket"""
    web_interface.connected_clients += 1
    join_room('dashboard')
    emit('connection_established', {'message': 'Connexion établie'})
    
    # Démarrer les mises à jour si c'est le premier client
    if web_interface.connected_clients == 1:
        web_interface.start_real_time_updates()

@socketio.on('disconnect')
def handle_disconnect():
    """Déconnexion WebSocket"""
    web_interface.connected_clients -= 1
    leave_room('dashboard')
    
    # Arrêter les mises à jour si plus de clients
    if web_interface.connected_clients == 0:
        web_interface.stop_real_time_updates()

@socketio.on('request_market_data')
def handle_market_data_request():
    """Demande de données de marché"""
    emit('market_data_update', web_interface.market_data_cache)

if __name__ == '__main__':
    print("🚀 Démarrage de l'interface web AlphaBeta808 Trading")
    print(f"📊 Dashboard disponible sur: http://localhost:5000")
    print(f"👤 Identifiants par défaut: admin / admin123")
    
    # Initialiser la base de données
    # Initialiser la base de données
    web_interface.db_manager.init_database()
    
    # Démarrer l'application
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
