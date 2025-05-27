#!/usr/bin/env python3
"""
Real-Time Trading Module with Binance Integration
Implements live trading capabilities with advanced order management
"""

import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import hmac
import hashlib
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from queue import Queue, Empty
import warnings

# Suppression des warnings
warnings.filterwarnings("ignore")

class OrderType(Enum):
    """Types d'ordres supportés"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    OCO = "OCO"  # One-Cancels-Other

class OrderSide(Enum):
    """Côtés d'ordre"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Statuts d'ordre"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class TradingOrder:
    """Représente un ordre de trading"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Canceled
    client_order_id: Optional[str] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: Optional[datetime] = None
    update_time: Optional[datetime] = None

@dataclass
class MarketData:
    """Données de marché en temps réel"""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    volume_24h: float
    price_change_24h: float
    timestamp: datetime

@dataclass
class AccountBalance:
    """Solde d'un actif"""
    asset: str
    free: float
    locked: float
    total: float

class RiskManager:
    """Gestionnaire de risques pour le trading en temps réel"""
    
    def __init__(self,
                 max_position_size: float = 0.1,  # 10% du capital max par position
                 max_daily_loss: float = 0.02,    # 2% de perte max par jour
                 max_total_exposure: float = 0.8,  # 80% d'exposition max
                 max_orders_per_minute: int = 10):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_total_exposure = max_total_exposure
        self.max_orders_per_minute = max_orders_per_minute
        
        self.daily_pnl = 0.0
        self.total_exposure = 0.0
        self.orders_this_minute = []
        
        self.logger = logging.getLogger(__name__)
    
    def check_order_validity(self, 
                           order: TradingOrder,
                           account_balance: Dict[str, AccountBalance],
                           portfolio_value: float) -> Tuple[bool, str]:
        """
        Vérifie si un ordre est valide selon les règles de risque
        
        Args:
            order: Ordre à vérifier
            account_balance: Soldes du compte
            portfolio_value: Valeur totale du portefeuille
            
        Returns:
            Tuple (is_valid, reason)
        """
        # Vérifier la taille de position
        order_value = order.quantity * (order.price or 0)
        position_size_pct = order_value / portfolio_value if portfolio_value > 0 else 0
        
        if position_size_pct > self.max_position_size:
            return False, f"Position trop large: {position_size_pct:.2%} > {self.max_position_size:.2%}"
        
        # Vérifier la perte journalière
        if self.daily_pnl < -self.max_daily_loss * portfolio_value:
            return False, f"Limite de perte journalière atteinte: {self.daily_pnl:.2f}"
        
        # Vérifier l'exposition totale
        if self.total_exposure + position_size_pct > self.max_total_exposure:
            return False, f"Exposition totale trop élevée: {self.total_exposure + position_size_pct:.2%}"
        
        # Vérifier le nombre d'ordres par minute
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)
        self.orders_this_minute = [t for t in self.orders_this_minute if t > minute_ago]
        
        if len(self.orders_this_minute) >= self.max_orders_per_minute:
            return False, f"Trop d'ordres cette minute: {len(self.orders_this_minute)}"
        
        # Vérifier le solde disponible
        base_asset, quote_asset = order.symbol[:-4], order.symbol[-4:]  # Simplification pour USDT
        
        if order.side == OrderSide.BUY:
            required_balance = order_value
            available_balance = account_balance.get(quote_asset, AccountBalance(quote_asset, 0, 0, 0)).free
            
            if required_balance > available_balance:
                return False, f"Solde insuffisant: {required_balance:.2f} > {available_balance:.2f} {quote_asset}"
        
        else:  # SELL
            required_quantity = order.quantity
            available_quantity = account_balance.get(base_asset, AccountBalance(base_asset, 0, 0, 0)).free
            
            if required_quantity > available_quantity:
                return False, f"Quantité insuffisante: {required_quantity:.6f} > {available_quantity:.6f} {base_asset}"
        
        return True, "OK"
    
    def update_daily_pnl(self, pnl_change: float):
        """Met à jour le P&L journalier"""
        self.daily_pnl += pnl_change
    
    def reset_daily_counters(self):
        """Remet à zéro les compteurs journaliers"""
        self.daily_pnl = 0.0
        self.orders_this_minute = []


class BinanceRealTimeTrader:
    """
    Trader en temps réel connecté à l'API Binance
    """
    
    def __init__(self,
                 api_key: str,
                 api_secret: str,
                 testnet: bool = True,
                 risk_manager: Optional[RiskManager] = None):
        """
        Initialise le trader Binance
        
        Args:
            api_key: Clé API Binance
            api_secret: Secret API Binance
            testnet: Utiliser le testnet (recommandé pour les tests)
            risk_manager: Gestionnaire de risques
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # URLs API
        if testnet:
            self.base_url = "https://testnet.binance.vision/"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/"
            self.ws_url = "wss://stream.binance.com:9443/ws"
        
        # Gestionnaire de risques
        self.risk_manager = risk_manager or RiskManager()
        
        # État du trader
        self.is_running = False
        self.subscribed_symbols = set()
        self.market_data = {}  # Dict[str, MarketData]
        self.account_balances = {}  # Dict[str, AccountBalance]
        self.open_orders = {}  # Dict[str, TradingOrder]
        self.order_history = []
        
        # WebSocket et threads
        self.websocket = None
        self.ws_thread = None
        self.data_queue = Queue()
        
        # Callbacks
        self.on_market_data_callback: Optional[Callable[[MarketData], None]] = None
        self.on_order_update_callback: Optional[Callable[[TradingOrder], None]] = None
        self.on_account_update_callback: Optional[Callable[[Dict[str, AccountBalance]], None]] = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Vérifier la connexion
        self._test_connectivity()
    
    def _generate_signature(self, query_string: str) -> str:
        """Génère la signature HMAC pour l'authentification"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, 
                     method: str, 
                     endpoint: str, 
                     params: Dict = None,
                     signed: bool = False) -> requests.Response:
        """
        Fait une requête à l'API Binance
        
        Args:
            method: Méthode HTTP (GET, POST, DELETE)
            endpoint: Endpoint de l'API
            params: Paramètres de la requête
            signed: Si la requête doit être signée
            
        Returns:
            Réponse de la requête
        """
        if params is None:
            params = {}
        
        headers = {
            'X-MBX-APIKEY': self.api_key
        }
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        url = f"{self.base_url}{endpoint}"
        
        if method == 'GET':
            response = requests.get(url, headers=headers, params=params)
        elif method == 'POST':
            response = requests.post(url, headers=headers, params=params)
        elif method == 'DELETE':
            response = requests.delete(url, headers=headers, params=params)
        else:
            raise ValueError(f"Méthode HTTP non supportée: {method}")
        
        return response
    
    def _test_connectivity(self):
        """Test la connectivité avec l'API Binance"""
        try:
            response = self._make_request('GET', 'api/v3/ping')
            if response.status_code == 200:
                self.logger.info("Connexion à l'API Binance établie")
            else:
                self.logger.error(f"Erreur de connexion: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Erreur de connectivité: {e}")
    
    def get_account_info(self) -> Dict:
        """Récupère les informations du compte"""
        try:
            response = self._make_request('GET', 'api/v3/account', signed=True)
            if response.status_code == 200:
                account_data = response.json()
                
                # Mettre à jour les soldes
                self.account_balances = {}
                for balance in account_data['balances']:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    
                    if total > 0:  # Ne garder que les actifs avec un solde
                        self.account_balances[asset] = AccountBalance(
                            asset=asset,
                            free=free,
                            locked=locked,
                            total=total
                        )
                
                return account_data
            else:
                self.logger.error(f"Erreur lors de la récupération du compte: {response.status_code}")
                return {}
        except Exception as e:
            self.logger.error(f"Erreur get_account_info: {e}")
            return {}
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Récupère les informations d'un symbole"""
        try:
            response = self._make_request('GET', 'api/v3/exchangeInfo')
            if response.status_code == 200:
                exchange_info = response.json()
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        return symbol_info
            return {}
        except Exception as e:
            self.logger.error(f"Erreur get_symbol_info: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> float:
        """Récupère le prix actuel d'un symbole"""
        try:
            response = self._make_request('GET', 'api/v3/ticker/price', {'symbol': symbol})
            if response.status_code == 200:
                price_data = response.json()
                return float(price_data['price'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Erreur get_current_price: {e}")
            return 0.0
    
    def place_order(self, order: TradingOrder) -> Optional[TradingOrder]:
        """
        Place un ordre sur Binance
        
        Args:
            order: Ordre à placer
            
        Returns:
            Ordre mis à jour avec les informations de Binance
        """
        # Vérifier les règles de risque
        portfolio_value = sum(balance.total * self.get_current_price(f"{balance.asset}USDT") 
                            for balance in self.account_balances.values()
                            if balance.asset != 'USDT')
        portfolio_value += self.account_balances.get('USDT', AccountBalance('USDT', 0, 0, 0)).total
        
        is_valid, reason = self.risk_manager.check_order_validity(
            order, self.account_balances, portfolio_value)
        
        if not is_valid:
            self.logger.warning(f"Ordre rejeté par le risk manager: {reason}")
            order.status = OrderStatus.REJECTED
            return order
        
        # Préparer les paramètres de l'ordre
        params = {
            'symbol': order.symbol,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': f"{order.quantity:.6f}"
        }

        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT, OrderType.OCO]:
            params['timeInForce'] = order.time_in_force
        
        if order.price is not None:
            params['price'] = f"{order.price:.8f}"
        
        if order.stop_price is not None:
            params['stopPrice'] = f"{order.stop_price:.8f}"
        
        if order.client_order_id:
            params['newClientOrderId'] = order.client_order_id
        
        try:
            response = self._make_request('POST', 'api/v3/order', params, signed=True)
            
            if response.status_code == 200:
                order_response = response.json()
                
                # Mettre à jour l'ordre avec la réponse
                order.order_id = order_response['orderId']
                order.status = OrderStatus(order_response['status'])
                order.filled_quantity = float(order_response.get('executedQty', 0))
                order.timestamp = datetime.fromtimestamp(order_response['transactTime'] / 1000)
                
                # Ajouter aux ordres ouverts
                self.open_orders[order.order_id] = order
                
                # Mettre à jour le compteur d'ordres
                self.risk_manager.orders_this_minute.append(datetime.now())
                
                self.logger.info(f"Ordre placé: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
                
                return order
            
            else:
                error_data = response.json()
                self.logger.error(f"Erreur lors du placement d'ordre: {error_data}")
                order.status = OrderStatus.REJECTED
                return order
                
        except Exception as e:
            self.logger.error(f"Erreur place_order: {e}")
            order.status = OrderStatus.REJECTED
            return order
    
    async def place_order_async(self, order: TradingOrder) -> bool:
        """Place un ordre de manière asynchrone"""
        try:
            # Valider l'ordre avec le risk manager
            portfolio_value = sum(balance.total * self.get_current_price(f"{balance.asset}USDT")
                                for balance in self.account_balances.values()
                                if balance.asset != 'USDT')
            portfolio_value += self.account_balances.get('USDT', AccountBalance('USDT', 0, 0, 0)).total

            is_valid, reason = self.risk_manager.check_order_validity(
                order, self.account_balances, portfolio_value)
            
            if not is_valid:
                self.logger.warning(f"Ordre rejeté par le risk manager: {reason} ({order.symbol})")
                order.status = OrderStatus.REJECTED # Assigner un statut en cas de rejet
                return False
            
            # Préparer les paramètres pour l'API Binance
            params = {
                'symbol': order.symbol,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'newClientOrderId': order.client_order_id
            }
            
            if order.order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
                params['price'] = order.price
                params['timeInForce'] = order.time_in_force
            
            if order.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
                params['stopPrice'] = order.stop_price
            
            # Placer l'ordre via l'API
            if self.testnet:
                # Simuler l'ordre en mode testnet
                order.order_id = f"test_{int(time.time())}"
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                order.avg_fill_price = order.price or 0
                self.logger.info(f"Ordre simulé (testnet): {order.symbol} {order.side.value}")
            else:
                # Mode production
                response = self.client.create_order(**params)
                order.order_id = str(response['orderId'])
                order.status = OrderStatus(response['status'])
                
                self.logger.info(f"Ordre placé: {order.symbol} {order.side.value} - ID: {order.order_id}")
            
            # Ajouter à la liste des ordres
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                self.open_orders[order.order_id] = order
            else:
                self.order_history.append(order)
            
            # Callback
            if self.on_order_update_callback:
                await self.on_order_update_callback(order)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Annule un ordre"""
        try:
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            response = self._make_request('DELETE', 'api/v3/order', params, signed=True)
            
            if response.status_code == 200:
                # Mettre à jour le statut de l'ordre
                if order_id in self.open_orders:
                    self.open_orders[order_id].status = OrderStatus.CANCELED
                    del self.open_orders[order_id]
                
                self.logger.info(f"Ordre annulé: {order_id}")
                return True
            else:
                self.logger.error(f"Erreur lors de l'annulation: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Erreur cancel_order: {e}")
            return False
    
    async def cancel_order_async(self, order_id: str) -> bool:
        """Annule un ordre de manière asynchrone"""
        try:
            if order_id not in self.open_orders:
                self.logger.warning(f"Ordre {order_id} non trouvé")
                return False
            
            order = self.open_orders[order_id]
            
            if not self.testnet:
                # Annuler via l'API Binance
                response = self.client.cancel_order(
                    symbol=order.symbol,
                    orderId=order_id
                )
                self.logger.info(f"Ordre annulé: {order_id}")
            else:
                self.logger.info(f"Ordre annulé (testnet): {order_id}")
            
            # Mettre à jour le statut
            order.status = OrderStatus.CANCELED
            
            # Déplacer vers l'historique
            del self.open_orders[order_id]
            self.order_history.append(order)
            
            # Callback
            if self.on_order_update_callback:
                await self.on_order_update_callback(order)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'annulation de l'ordre {order_id}: {e}")
            return False

    def get_open_orders(self, symbol: str = None) -> List[TradingOrder]:
        """Récupère les ordres ouverts"""
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            response = self._make_request('GET', 'api/v3/openOrders', params, signed=True)
            
            if response.status_code == 200:
                orders_data = response.json()
                orders = []
                
                for order_data in orders_data:
                    order = TradingOrder(
                        symbol=order_data['symbol'],
                        side=OrderSide(order_data['side']),
                        order_type=OrderType(order_data['type']),
                        quantity=float(order_data['origQty']),
                        price=float(order_data['price']) if order_data['price'] != '0.00000000' else None,
                        stop_price=float(order_data['stopPrice']) if order_data['stopPrice'] != '0.00000000' else None,
                        time_in_force=order_data['timeInForce'],
                        client_order_id=order_data['clientOrderId'],
                        order_id=order_data['orderId'],
                        status=OrderStatus(order_data['status']),
                        filled_quantity=float(order_data['executedQty']),
                        timestamp=datetime.fromtimestamp(order_data['time'] / 1000),
                        update_time=datetime.fromtimestamp(order_data['updateTime'] / 1000)
                    )
                    orders.append(order)
                
                # Mettre à jour les ordres ouverts
                self.open_orders = {order.order_id: order for order in orders}
                
                return orders
            else:
                self.logger.error(f"Erreur lors de la récupération des ordres: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Erreur get_open_orders: {e}")
            return []
    
    async def _handle_websocket_message(self, message: str):
        """Traite les messages WebSocket"""
        try:
            data = json.loads(message)
            
            # Pour le format stream, les données sont dans data['data']
            if 'stream' in data and 'data' in data:
                stream_data = data['data']
            else:
                stream_data = data
            
            # Ticker de prix
            if 'c' in stream_data and 'b' in stream_data and 'a' in stream_data:
                market_data = MarketData(
                    symbol=stream_data['s'],
                    price=float(stream_data['c']),
                    bid_price=float(stream_data['b']),
                    ask_price=float(stream_data['a']),
                    bid_quantity=float(stream_data['B']),
                    ask_quantity=float(stream_data['A']),
                    volume_24h=float(stream_data['v']),
                    price_change_24h=float(stream_data['p']),
                    timestamp=datetime.now()
                )
                
                self.market_data[stream_data['s']] = market_data
                
                if self.on_market_data_callback:
                    self.on_market_data_callback(market_data)
            
            # Mise à jour d'ordre (user data stream)
            elif 'e' in stream_data and stream_data['e'] == 'executionReport':
                if stream_data['i'] in self.open_orders:
                    order = self.open_orders[stream_data['i']]
                    order.status = OrderStatus(stream_data['X'])
                    order.filled_quantity = float(stream_data['z'])
                    order.avg_fill_price = float(stream_data['Z']) / float(stream_data['z']) if float(stream_data['z']) > 0 else 0
                    order.update_time = datetime.fromtimestamp(stream_data['T'] / 1000)
                    
                    if self.on_order_update_callback:
                        self.on_order_update_callback(order)
                    
                    # Si l'ordre est complètement exécuté, le retirer des ordres ouverts
                    if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
                        del self.open_orders[stream_data['i']]
                        self.order_history.append(order)
            
            # Mise à jour du compte
            elif 'e' in stream_data and stream_data['e'] == 'outboundAccountPosition':
                for balance_data in stream_data['B']:
                    asset = balance_data['a']
                    free = float(balance_data['f'])
                    locked = float(balance_data['l'])
                    total = free + locked
                    
                    self.account_balances[asset] = AccountBalance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=total
                    )
                
                if self.on_account_update_callback:
                    self.on_account_update_callback(self.account_balances)
                    
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du message WebSocket: {e}")
    
    async def _websocket_listener(self):
        """Écoute les messages WebSocket"""
        try:
            # Construire l'URL du stream
            streams = []
            for symbol in self.subscribed_symbols:
                streams.append(f"{symbol.lower()}@ticker")
            
            if streams:
                # Corriger l'URL WebSocket pour Binance
                if self.testnet:
                    # Pour testnet, utiliser l'endpoint testnet correct
                    if len(streams) == 1:
                        stream_url = f"wss://stream.testnet.binance.vision/ws/{streams[0]}"
                    else:
                        # Pour multiple streams sur testnet
                        stream_url = f"wss://stream.testnet.binance.vision/stream?streams={'/'.join(streams)}"
                else:
                    # Pour production, utiliser le format standard
                    if len(streams) == 1:
                        stream_url = f"wss://stream.binance.com:9443/ws/{streams[0]}"
                    else:
                        stream_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
                
                self.logger.info(f"Tentative de connexion WebSocket: {stream_url}")
                
                async with websockets.connect(stream_url) as websocket:
                    self.websocket = websocket
                    self.logger.info(f"WebSocket connecté: {len(streams)} streams")
                    
                    async for message in websocket:
                        await self._handle_websocket_message(message)
                        
        except Exception as e:
            self.logger.error(f"Erreur WebSocket: {e}")
    
    def subscribe_to_market_data(self, symbols: List[str]):
        """S'abonne aux données de marché en temps réel"""
        self.subscribed_symbols.update(symbols)
        self.logger.info(f"Abonnement aux symboles: {symbols}")
    
    def start_market_data_stream(self):
        """Démarre le flux de données de marché"""
        if self.subscribed_symbols and not self.is_running:
            self.is_running = True
            
            # Démarrer le thread WebSocket
            def run_websocket():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._websocket_listener())
            
            self.ws_thread = threading.Thread(target=run_websocket, daemon=True)
            self.ws_thread.start()
            
            self.logger.info("Flux de données de marché démarré")
    
    def stop_market_data_stream(self):
        """Arrête le flux de données de marché"""
        self.is_running = False
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        self.logger.info("Flux de données de marché arrêté")
    
    def set_callbacks(self,
                     on_market_data: Callable[[MarketData], None] = None,
                     on_order_update: Callable[[TradingOrder], None] = None,
                     on_account_update: Callable[[Dict[str, AccountBalance]], None] = None):
        """Configure les callbacks pour les événements"""
        self.on_market_data_callback = on_market_data
        self.on_order_update_callback = on_order_update
        self.on_account_update_callback = on_account_update
    
    def get_trading_summary(self) -> Dict:
        """Génère un résumé des activités de trading"""
        total_orders = len(self.order_history) + len(self.open_orders)
        filled_orders = len([o for o in self.order_history if o.status == OrderStatus.FILLED])
        
        return {
            'total_orders': total_orders,
            'open_orders': len(self.open_orders),
            'filled_orders': filled_orders,
            'canceled_orders': len([o for o in self.order_history if o.status == OrderStatus.CANCELED]),
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'subscribed_symbols': list(self.subscribed_symbols),
            'account_assets': list(self.account_balances.keys()),
            'is_streaming': self.is_running
        }


class TradingStrategy:
    """
    Stratégie de trading intégrée avec le trader en temps réel
    """
    
    def __init__(self, 
                 trader: BinanceRealTimeTrader,
                 signal_generator: Callable[[str, MarketData], float] = None):
        self.trader = trader
        self.signal_generator = signal_generator
        self.position_sizes = {}  # Dict[str, float]
        self.last_signals = {}    # Dict[str, float]
        
        # Configurer les callbacks
        self.trader.set_callbacks(
            on_market_data=self._on_market_data,
            on_order_update=self._on_order_update,
            on_account_update=self._on_account_update
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _on_market_data(self, market_data: MarketData):
        """Callback pour les nouvelles données de marché"""
        symbol = market_data.symbol
        
        if self.signal_generator:
            # Générer un signal
            signal = self.signal_generator(symbol, market_data)
            self.last_signals[symbol] = signal
            
            # Analyser si un trade est nécessaire
            self._analyze_trading_opportunity(symbol, market_data, signal)
    
    def _on_order_update(self, order: TradingOrder):
        """Callback pour les mises à jour d'ordres"""
        self.logger.info(f"Ordre mis à jour: {order.symbol} {order.status.value}")
        
        # Mettre à jour les tailles de position
        if order.status == OrderStatus.FILLED:
            if order.symbol not in self.position_sizes:
                self.position_sizes[order.symbol] = 0.0
            
            if order.side == OrderSide.BUY:
                self.position_sizes[order.symbol] += order.filled_quantity
            else:
                self.position_sizes[order.symbol] -= order.filled_quantity
    
    def _on_account_update(self, balances: Dict[str, AccountBalance]):
        """Callback pour les mises à jour de compte"""
        self.logger.debug(f"Compte mis à jour: {len(balances)} actifs")
    
    def _analyze_trading_opportunity(self, 
                                   symbol: str, 
                                   market_data: MarketData, 
                                   signal: float):
        """
        Analyse une opportunité de trading basée sur le signal
        
        Args:
            symbol: Symbole de l'actif
            market_data: Données de marché
            signal: Signal de trading (-1 à 1)
        """
        # Seuils de signal pour déclencher des trades
        buy_threshold = 0.6
        sell_threshold = -0.6
        
        current_position = self.position_sizes.get(symbol, 0.0)
        
        # Calculer la taille de position cible basée sur le signal
        max_position_value = 1000  # $1000 max par position
        target_position_value = signal * max_position_value
        target_quantity = target_position_value / market_data.price
        
        quantity_diff = target_position_value - current_position
        
        # Ignorer les petites différences
        if abs(quantity_diff) < 0.001:
            return
        
        # Créer l'ordre si nécessaire
        if quantity_diff > 0 and signal > buy_threshold:
            # Signal d'achat
            order = TradingOrder(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=abs(quantity_diff),
                client_order_id=f"buy_{symbol}_{int(time.time())}"
            )
            
            self.trader.place_order(order)
            
        elif quantity_diff < 0 and signal < sell_threshold:
            # Signal de vente
            order = TradingOrder(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=abs(quantity_diff),
                client_order_id=f"sell_{symbol}_{int(time.time())}"
            )
            
            self.trader.place_order(order)
    
    def start_trading(self, symbols: List[str]):
        """Démarre la stratégie de trading"""
        self.trader.subscribe_to_market_data(symbols)
        self.trader.start_market_data_stream()
        self.logger.info(f"Stratégie de trading démarrée pour {symbols}")
    
    def stop_trading(self):
        """Arrête la stratégie de trading"""
        self.trader.stop_market_data_stream()
        self.logger.info("Stratégie de trading arrêtée")

    def add_symbol(self, symbol: str):
        """
        Add a symbol to the trading strategy
        
        Args:
            symbol: Trading symbol to add
        """
        if symbol not in self.position_sizes:
            self.position_sizes[symbol] = 0.0
        if symbol not in self.last_signals:
            self.last_signals[symbol] = 0.0
        
        self.logger.info(f"Symbol {symbol} added to trading strategy")

    def remove_symbol(self, symbol: str):
        """
        Remove a symbol from the trading strategy
        
        Args:
            symbol: Trading symbol to remove
        """
        if symbol in self.position_sizes:
            del self.position_sizes[symbol]
        if symbol in self.last_signals:
            del self.last_signals[symbol]
        
        self.logger.info(f"Symbol {symbol} removed from trading strategy")

    def get_position_size(self, symbol: str) -> float:
        """
        Get current position size for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position size
        """
        return self.position_sizes.get(symbol, 0.0)

    def get_last_signal(self, symbol: str) -> float:
        """
        Get last signal for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Last signal value
        """
        return self.last_signals.get(symbol, 0.0)
