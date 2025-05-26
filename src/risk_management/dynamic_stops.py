#!/usr/bin/env python3
"""
Dynamic Stop-Loss and Take-Profit Module
Implements advanced risk management with adaptive stop-loss mechanisms
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from enum import Enum
from dataclasses import dataclass

class StopLossType(Enum):
    """Types de stop-loss disponibles"""
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    TIME_BASED = "time_based"
    SUPPORT_RESISTANCE = "support_resistance"
    DYNAMIC_PERCENTILE = "dynamic_percentile"

@dataclass
class StopLossOrder:
    """Représente un ordre de stop-loss"""
    symbol: str
    position_type: str  # 'long' ou 'short'
    entry_price: float
    current_price: float
    stop_loss_price: float
    take_profit_price: Optional[float]
    stop_loss_type: StopLossType
    entry_timestamp: datetime
    last_update: datetime
    quantity: float
    max_loss_pct: float
    trailing_distance: Optional[float] = None
    highest_price: Optional[float] = None  # Pour trailing stop sur long
    lowest_price: Optional[float] = None   # Pour trailing stop sur short
    volatility_factor: float = 1.0
    is_active: bool = True

class DynamicStopLossManager:
    """
    Gestionnaire de stop-loss dynamiques pour optimiser la gestion des risques
    """
    
    def __init__(self,
                 default_stop_loss_pct: float = 0.02,  # 2% par défaut
                 default_take_profit_pct: float = 0.04,  # 4% par défaut
                 atr_multiplier: float = 2.0,
                 volatility_lookback: int = 20,
                 trailing_activation_pct: float = 0.01,  # 1% de profit avant activation
                 max_holding_days: int = 30,
                 risk_per_trade: float = 0.01):  # 1% du capital par trade
        """
        Initialise le gestionnaire de stop-loss dynamiques
        
        Args:
            default_stop_loss_pct: Pourcentage de stop-loss par défaut
            default_take_profit_pct: Pourcentage de take-profit par défaut
            atr_multiplier: Multiplicateur pour les stop-loss basés sur l'ATR
            volatility_lookback: Période de lookback pour le calcul de volatilité
            trailing_activation_pct: Profit minimum avant activation du trailing stop
            max_holding_days: Durée maximale de détention d'une position
            risk_per_trade: Risque maximum par trade (% du capital)
        """
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.atr_multiplier = atr_multiplier
        self.volatility_lookback = volatility_lookback
        self.trailing_activation_pct = trailing_activation_pct
        self.max_holding_days = max_holding_days
        self.risk_per_trade = risk_per_trade
        
        # Tracking des ordres actifs
        self.active_stops = {}  # Dict[str, StopLossOrder]
        self.executed_stops = []
        self.performance_metrics = {
            'total_stops_triggered': 0,
            'profitable_stops': 0,
            'total_pnl': 0.0,
            'avg_holding_time': 0.0
        }
        
        # Configuration du logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def calculate_atr(self, 
                     price_data: pd.DataFrame, 
                     period: int = 14,
                     high_col: str = 'high',
                     low_col: str = 'low',
                     close_col: str = 'close') -> pd.Series:
        """
        Calcule l'Average True Range (ATR)
        
        Args:
            price_data: DataFrame avec les données OHLC
            period: Période pour le calcul de l'ATR
            high_col: Nom de la colonne high
            low_col: Nom de la colonne low
            close_col: Nom de la colonne close
            
        Returns:
            Série avec les valeurs ATR
        """
        high = price_data[high_col]
        low = price_data[low_col]
        close = price_data[close_col]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR comme moyenne mobile exponentielle du True Range
        atr = true_range.ewm(span=period).mean()
        
        return atr
    
    def calculate_volatility(self, 
                           returns: pd.Series, 
                           period: int = None) -> float:
        """
        Calcule la volatilité des rendements
        
        Args:
            returns: Série des rendements
            period: Période de calcul (utilise self.volatility_lookback par défaut)
            
        Returns:
            Volatilité annualisée
        """
        if period is None:
            period = self.volatility_lookback
            
        recent_returns = returns.tail(period)
        volatility = recent_returns.std() * np.sqrt(252)  # Annualisée
        
        return volatility if not np.isnan(volatility) else 0.02  # Défaut à 2%
    
    def calculate_support_resistance(self, 
                                   price_data: pd.DataFrame,
                                   lookback_periods: int = 50,
                                   min_touches: int = 2) -> Tuple[float, float]:
        """
        Calcule les niveaux de support et résistance
        
        Args:
            price_data: DataFrame avec les données de prix
            lookback_periods: Nombre de périodes à analyser
            min_touches: Nombre minimum de touches pour valider un niveau
            
        Returns:
            Tuple (support_level, resistance_level)
        """
        recent_data = price_data.tail(lookback_periods)
        
        # Méthode simplifiée: utiliser les percentiles
        support_level = recent_data['low'].quantile(0.1)
        resistance_level = recent_data['high'].quantile(0.9)
        
        return support_level, resistance_level
    
    def create_stop_loss_order(self,
                             symbol: str,
                             position_type: str,
                             entry_price: float,
                             quantity: float,
                             price_data: pd.DataFrame,
                             stop_loss_type: StopLossType = StopLossType.VOLATILITY_BASED,
                             custom_stop_pct: Optional[float] = None,
                             custom_tp_pct: Optional[float] = None) -> StopLossOrder:
        """
        Crée un ordre de stop-loss adaptatif
        
        Args:
            symbol: Symbole de l'actif
            position_type: Type de position ('long' ou 'short')
            entry_price: Prix d'entrée
            quantity: Quantité
            price_data: Données historiques de prix
            stop_loss_type: Type de stop-loss à utiliser
            custom_stop_pct: Pourcentage custom de stop-loss
            custom_tp_pct: Pourcentage custom de take-profit
            
        Returns:
            Ordre de stop-loss créé
        """
        current_price = price_data['close'].iloc[-1]
        
        # Calculer le stop-loss selon le type
        if stop_loss_type == StopLossType.FIXED_PERCENTAGE:
            stop_pct = custom_stop_pct or self.default_stop_loss_pct
            if position_type == 'long':
                stop_loss_price = entry_price * (1 - stop_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_pct)
                
        elif stop_loss_type == StopLossType.ATR_BASED:
            atr = self.calculate_atr(price_data).iloc[-1]
            if position_type == 'long':
                stop_loss_price = entry_price - (atr * self.atr_multiplier)
            else:
                stop_loss_price = entry_price + (atr * self.atr_multiplier)
                
        elif stop_loss_type == StopLossType.VOLATILITY_BASED:
            returns = price_data['close'].pct_change().dropna()
            volatility = self.calculate_volatility(returns)
            # Adapter le stop-loss à la volatilité (plus volatile = stop plus large)
            vol_adjusted_pct = self.default_stop_loss_pct * (1 + volatility)
            if position_type == 'long':
                stop_loss_price = entry_price * (1 - vol_adjusted_pct)
            else:
                stop_loss_price = entry_price * (1 + vol_adjusted_pct)
                
        elif stop_loss_type == StopLossType.SUPPORT_RESISTANCE:
            support, resistance = self.calculate_support_resistance(price_data)
            if position_type == 'long':
                stop_loss_price = max(support * 0.99, entry_price * (1 - self.default_stop_loss_pct))
            else:
                stop_loss_price = min(resistance * 1.01, entry_price * (1 + self.default_stop_loss_pct))
                
        elif stop_loss_type == StopLossType.DYNAMIC_PERCENTILE:
            returns = price_data['close'].pct_change().tail(self.volatility_lookback).dropna()
            if position_type == 'long':
                percentile_loss = returns.quantile(0.05)  # 5ème percentile des pertes
                stop_loss_price = entry_price * (1 + percentile_loss)
            else:
                percentile_loss = returns.quantile(0.95)  # 95ème percentile des gains
                stop_loss_price = entry_price * (1 + percentile_loss)
        
        else:  # TRAILING ou défaut
            stop_pct = custom_stop_pct or self.default_stop_loss_pct
            if position_type == 'long':
                stop_loss_price = entry_price * (1 - stop_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_pct)
        
        # Calculer le take-profit
        tp_pct = custom_tp_pct or self.default_take_profit_pct
        if position_type == 'long':
            take_profit_price = entry_price * (1 + tp_pct)
        else:
            take_profit_price = entry_price * (1 - tp_pct)
        
        # Calculer le pourcentage de perte maximum
        if position_type == 'long':
            max_loss_pct = (entry_price - stop_loss_price) / entry_price
        else:
            max_loss_pct = (stop_loss_price - entry_price) / entry_price
        
        # Créer l'ordre
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        stop_order = StopLossOrder(
            symbol=symbol,
            position_type=position_type,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_type=stop_loss_type,
            entry_timestamp=datetime.now(),
            last_update=datetime.now(),
            quantity=quantity,
            max_loss_pct=max_loss_pct,
            highest_price=current_price if position_type == 'long' else None,
            lowest_price=current_price if position_type == 'short' else None
        )
        
        # Pour les trailing stops, calculer la distance
        if stop_loss_type == StopLossType.TRAILING:
            if position_type == 'long':
                stop_order.trailing_distance = current_price - stop_loss_price
            else:
                stop_order.trailing_distance = stop_loss_price - current_price
        
        self.active_stops[order_id] = stop_order
        
        self.logger.info(f"Stop-loss créé pour {symbol}: {stop_loss_type.value}")
        self.logger.info(f"  Entry: {entry_price:.4f}, Stop: {stop_loss_price:.4f}, TP: {take_profit_price:.4f}")
        
        return stop_order
    
    def update_trailing_stop(self, order_id: str, current_price: float) -> bool:
        """
        Met à jour un trailing stop
        
        Args:
            order_id: ID de l'ordre
            current_price: Prix actuel
            
        Returns:
            True si le stop a été modifié
        """
        if order_id not in self.active_stops:
            return False
            
        order = self.active_stops[order_id]
        
        if order.stop_loss_type != StopLossType.TRAILING:
            return False
        
        updated = False
        
        if order.position_type == 'long':
            # Pour une position longue, suivre le prix vers le haut
            if current_price > order.highest_price:
                order.highest_price = current_price
                new_stop = current_price - order.trailing_distance
                
                # Ne jamais baisser le stop
                if new_stop > order.stop_loss_price:
                    order.stop_loss_price = new_stop
                    updated = True
                    
        else:  # short position
            # Pour une position courte, suivre le prix vers le bas
            if current_price < order.lowest_price:
                order.lowest_price = current_price
                new_stop = current_price + order.trailing_distance
                
                # Ne jamais augmenter le stop
                if new_stop < order.stop_loss_price:
                    order.stop_loss_price = new_stop
                    updated = True
        
        if updated:
            order.last_update = datetime.now()
            order.current_price = current_price
            self.logger.debug(f"Trailing stop mis à jour pour {order.symbol}: {order.stop_loss_price:.4f}")
        
        return updated
    
    def check_stop_triggers(self, 
                          price_data: Dict[str, float], 
                          timestamp: datetime = None) -> List[Dict]:
        """
        Vérifie si des stops doivent être déclenchés
        
        Args:
            price_data: Dict avec les prix actuels {symbol: price}
            timestamp: Timestamp actuel
            
        Returns:
            Liste des ordres déclenchés
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        triggered_orders = []
        orders_to_remove = []
        
        for order_id, order in self.active_stops.items():
            if not order.is_active:
                continue
                
            symbol = order.symbol
            if symbol not in price_data:
                continue
                
            current_price = price_data[symbol]
            order.current_price = current_price
            
            # Mettre à jour les trailing stops
            if order.stop_loss_type == StopLossType.TRAILING:
                self.update_trailing_stop(order_id, current_price)
            
            # Vérifier les conditions de déclenchement
            stop_triggered = False
            tp_triggered = False
            time_stop_triggered = False
            
            # Stop-loss
            if order.position_type == 'long' and current_price <= order.stop_loss_price:
                stop_triggered = True
            elif order.position_type == 'short' and current_price >= order.stop_loss_price:
                stop_triggered = True
            
            # Take-profit
            if order.take_profit_price:
                if order.position_type == 'long' and current_price >= order.take_profit_price:
                    tp_triggered = True
                elif order.position_type == 'short' and current_price <= order.take_profit_price:
                    tp_triggered = True
            
            # Stop temporel
            holding_time = (timestamp - order.entry_timestamp).days
            if holding_time >= self.max_holding_days:
                time_stop_triggered = True
            
            # Traiter les déclenchements
            if stop_triggered or tp_triggered or time_stop_triggered:
                exit_reason = 'stop_loss' if stop_triggered else ('take_profit' if tp_triggered else 'time_stop')
                
                # Calculer le P&L
                if order.position_type == 'long':
                    pnl_pct = (current_price - order.entry_price) / order.entry_price
                else:
                    pnl_pct = (order.entry_price - current_price) / order.entry_price
                
                pnl_absolute = pnl_pct * order.quantity * order.entry_price
                
                # Créer l'ordre de sortie
                exit_order = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'action': 'sell' if order.position_type == 'long' else 'buy',
                    'quantity': order.quantity,
                    'price': current_price,
                    'exit_reason': exit_reason,
                    'entry_price': order.entry_price,
                    'pnl_pct': pnl_pct,
                    'pnl_absolute': pnl_absolute,
                    'holding_time_days': holding_time,
                    'timestamp': timestamp
                }
                
                triggered_orders.append(exit_order)
                orders_to_remove.append(order_id)
                
                # Mettre à jour les métriques
                self.performance_metrics['total_stops_triggered'] += 1
                if pnl_absolute > 0:
                    self.performance_metrics['profitable_stops'] += 1
                self.performance_metrics['total_pnl'] += pnl_absolute
                
                # Archiver l'ordre
                order.is_active = False
                self.executed_stops.append({
                    'order': order,
                    'exit_details': exit_order
                })
                
                self.logger.info(f"Stop déclenché pour {symbol}: {exit_reason}")
                self.logger.info(f"  P&L: {pnl_pct*100:.2f}% ({pnl_absolute:.2f})")
        
        # Supprimer les ordres déclenchés
        for order_id in orders_to_remove:
            del self.active_stops[order_id]
        
        return triggered_orders
    
    def adjust_stops_for_volatility(self, 
                                  price_data: pd.DataFrame, 
                                  volatility_threshold: float = 0.3) -> int:
        """
        Ajuste les stops en fonction de la volatilité du marché
        
        Args:
            price_data: Données de prix récentes
            volatility_threshold: Seuil de volatilité pour déclencher l'ajustement
            
        Returns:
            Nombre d'ordres ajustés
        """
        returns = price_data['close'].pct_change().dropna()
        current_volatility = self.calculate_volatility(returns)
        
        adjusted_count = 0
        
        if current_volatility > volatility_threshold:
            vol_multiplier = min(2.0, current_volatility / volatility_threshold)
            
            for order_id, order in self.active_stops.items():
                if not order.is_active:
                    continue
                
                # Élargir les stops en période de haute volatilité
                if order.position_type == 'long':
                    new_stop = order.entry_price * (1 - order.max_loss_pct * vol_multiplier)
                    if new_stop < order.stop_loss_price:  # Ne jamais resserrer le stop
                        order.stop_loss_price = new_stop
                        adjusted_count += 1
                else:
                    new_stop = order.entry_price * (1 + order.max_loss_pct * vol_multiplier)
                    if new_stop > order.stop_loss_price:  # Ne jamais resserrer le stop
                        order.stop_loss_price = new_stop
                        adjusted_count += 1
                
                order.volatility_factor = vol_multiplier
                order.last_update = datetime.now()
        
        if adjusted_count > 0:
            self.logger.info(f"Ajustement de {adjusted_count} stops pour volatilité élevée ({current_volatility:.2%})")
        
        return adjusted_count
    
    def get_portfolio_risk_metrics(self) -> Dict:
        """
        Calcule les métriques de risque du portefeuille
        
        Returns:
            Dict avec les métriques de risque
        """
        active_orders = [order for order in self.active_stops.values() if order.is_active]
        
        if not active_orders:
            return {
                'total_positions': 0,
                'total_risk_exposure': 0.0,
                'avg_max_loss_pct': 0.0,
                'largest_position_risk': 0.0
            }
        
        total_positions = len(active_orders)
        total_risk_exposure = sum(order.max_loss_pct * order.quantity * order.entry_price 
                                for order in active_orders)
        avg_max_loss_pct = np.mean([order.max_loss_pct for order in active_orders])
        largest_position_risk = max(order.max_loss_pct * order.quantity * order.entry_price 
                                  for order in active_orders)
        
        return {
            'total_positions': total_positions,
            'total_risk_exposure': total_risk_exposure,
            'avg_max_loss_pct': avg_max_loss_pct,
            'largest_position_risk': largest_position_risk,
            'active_symbols': [order.symbol for order in active_orders]
        }
    
    def get_performance_summary(self) -> Dict:
        """
        Génère un résumé de performance des stops
        
        Returns:
            Dict avec les statistiques de performance
        """
        if self.performance_metrics['total_stops_triggered'] == 0:
            return {
                'total_stops_triggered': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0
            }
        
        win_rate = self.performance_metrics['profitable_stops'] / self.performance_metrics['total_stops_triggered']
        avg_pnl = self.performance_metrics['total_pnl'] / self.performance_metrics['total_stops_triggered']
        
        # Calcul des métriques détaillées des ordres exécutés
        executed_details = []
        for executed in self.executed_stops:
            executed_details.append({
                'pnl_pct': executed['exit_details']['pnl_pct'],
                'holding_time': executed['exit_details']['holding_time_days'],
                'exit_reason': executed['exit_details']['exit_reason']
            })
        
        avg_holding_time = np.mean([detail['holding_time'] for detail in executed_details]) if executed_details else 0
        
        return {
            'total_stops_triggered': self.performance_metrics['total_stops_triggered'],
            'profitable_stops': self.performance_metrics['profitable_stops'],
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': self.performance_metrics['total_pnl'],
            'avg_holding_time_days': avg_holding_time,
            'executed_details': executed_details
        }

    def set_atr_stop_loss(self, symbol: str, atr_value: float, entry_price: float, 
                         position_type: str, atr_multiplier: float = None):
        """
        Set ATR-based stop-loss for a symbol
        
        Args:
            symbol: Trading symbol
            atr_value: Current ATR value
            entry_price: Entry price of position
            position_type: 'long' or 'short'
            atr_multiplier: ATR multiplier (uses default if None)
        """
        if atr_multiplier is None:
            atr_multiplier = self.atr_multiplier
            
        # Create price data with dummy values for ATR calculation
        price_data = pd.DataFrame({
            'high': [entry_price * 1.01],
            'low': [entry_price * 0.99],
            'close': [entry_price]
        })
        
        # Create stop-loss order
        stop_order = self.create_stop_loss_order(
            symbol=symbol,
            position_type=position_type,
            entry_price=entry_price,
            quantity=1.0,  # Default quantity
            price_data=price_data,
            stop_loss_type=StopLossType.ATR_BASED
        )
        
        self.logger.info(f"ATR stop-loss set for {symbol}: {stop_order.stop_loss_price:.4f}")

    def set_trailing_stop_loss(self, symbol: str, current_price: float, 
                              position_type: str, trail_amount: float):
        """
        Set trailing stop-loss for a symbol
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            position_type: 'long' or 'short'
            trail_amount: Trailing distance
        """
        # Create price data
        price_data = pd.DataFrame({
            'high': [current_price * 1.01],
            'low': [current_price * 0.99],
            'close': [current_price]
        })
        
        # Create trailing stop order
        stop_order = self.create_stop_loss_order(
            symbol=symbol,
            position_type=position_type,
            entry_price=current_price,
            quantity=1.0,
            price_data=price_data,
            stop_loss_type=StopLossType.TRAILING
        )
        
        # Set trailing distance
        stop_order.trailing_distance = trail_amount
        
        self.logger.info(f"Trailing stop-loss set for {symbol}: trail={trail_amount:.4f}")

    def set_volatility_stop_loss(self, symbol: str, price_series: pd.Series, 
                                entry_price: float, position_type: str, 
                                volatility_multiplier: float = 1.5):
        """
        Set volatility-based stop-loss for a symbol
        
        Args:
            symbol: Trading symbol
            price_series: Historical price series
            entry_price: Entry price of position
            position_type: 'long' or 'short'
            volatility_multiplier: Volatility multiplier
        """
        # Create price data
        price_data = pd.DataFrame({
            'high': price_series * 1.01,
            'low': price_series * 0.99,
            'close': price_series
        })
        
        # Create volatility-based stop order
        stop_order = self.create_stop_loss_order(
            symbol=symbol,
            position_type=position_type,
            entry_price=entry_price,
            quantity=1.0,
            price_data=price_data,
            stop_loss_type=StopLossType.VOLATILITY_BASED
        )
        
        self.logger.info(f"Volatility stop-loss set for {symbol}: {stop_order.stop_loss_price:.4f}")

    def check_exit_conditions(self, symbol: str, current_price: float, 
                             position_type: str) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for a position
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            position_type: 'long' or 'short'
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        # Check if symbol has active stops
        active_order = None
        for order in self.active_stops.values():
            if order.symbol == symbol and order.is_active:
                active_order = order
                break
        
        if not active_order:
            return False, "No active stops"
        
        # Check stop-loss conditions
        if position_type == 'long' and current_price <= active_order.stop_loss_price:
            return True, "Stop-loss triggered"
        elif position_type == 'short' and current_price >= active_order.stop_loss_price:
            return True, "Stop-loss triggered"
            
        # Check take-profit conditions
        if active_order.take_profit_price:
            if position_type == 'long' and current_price >= active_order.take_profit_price:
                return True, "Take-profit triggered"
            elif position_type == 'short' and current_price <= active_order.take_profit_price:
                return True, "Take-profit triggered"
        
        return False, "No exit conditions met"

    def update_trailing_stop(self, symbol: str, current_price: float, position_type: str):
        """
        Update trailing stop for a symbol (wrapper for existing method)
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            position_type: 'long' or 'short'
        """
        # Find the order ID for this symbol
        order_id = None
        for oid, order in self.active_stops.items():
            if order.symbol == symbol and order.is_active:
                order_id = oid
                break
        
        if order_id:
            # Call the original update_trailing_stop method with order_id
            for oid, order in self.active_stops.items():
                if oid == order_id and order.symbol == symbol and order.is_active:
                    # Update trailing stop for this order
                    if hasattr(order, 'trailing_distance'):
                        if position_type == 'long':
                            new_stop = current_price - order.trailing_distance
                            if new_stop > order.stop_loss_price:
                                order.stop_loss_price = new_stop
                                return True
                        else:  # short
                            new_stop = current_price + order.trailing_distance
                            if new_stop < order.stop_loss_price:
                                order.stop_loss_price = new_stop
                                return True
                    break
        
        return False

    def get_stop_levels(self, symbol: str) -> Dict:
        """
        Get current stop levels for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with stop levels
        """
        for order in self.active_stops.values():
            if order.symbol == symbol and order.is_active:
                return {
                    'stop_loss': order.stop_loss_price,
                    'take_profit': order.take_profit_price,
                    'entry_price': order.entry_price,
                    'stop_type': order.stop_loss_type.value
                }
        
        return {}

    def calculate_portfolio_risk_metrics(self, positions: Dict) -> Dict:
        """
        Calculate portfolio risk metrics (wrapper for existing method)
        
        Args:
            positions: Dict of positions {symbol: {'size': float, 'entry_price': float}}
            
        Returns:
            Dict with risk metrics
        """
        return self.get_portfolio_risk_metrics()
