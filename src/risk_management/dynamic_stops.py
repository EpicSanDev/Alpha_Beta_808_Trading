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
    quantity: float # Nombre de contrats/unités
    max_loss_pct: float # Pourcentage de perte sur la valeur notionnelle au prix d'entrée
    contract_multiplier: float = 1.0 # Multiplicateur de contrat pour les futures
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
        
        self.active_stops: Dict[str, StopLossOrder] = {}
        self.executed_stops: List[Dict] = []
        self.performance_metrics: Dict[str, Union[int, float]] = {
            'total_stops_triggered': 0,
            'profitable_stops': 0,
            'total_pnl': 0.0,
            'avg_holding_time': 0.0  # Sera calculé plus tard
        }
        
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
        """
        high = price_data[high_col]
        low = price_data[low_col]
        close = price_data[close_col]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean() # adjust=False pour correspondre à la plupart des plateformes
        return atr
    
    def calculate_volatility(self, 
                           returns: pd.Series, 
                           period: Optional[int] = None) -> float:
        """
        Calcule la volatilité des rendements (écart-type annualisé).
        """
        if period is None:
            period = self.volatility_lookback
        if len(returns) < period:
            # Pas assez de données pour une volatilité significative
            return self.default_stop_loss_pct # Retourne un % de stop par défaut comme proxy
            
        recent_returns = returns.tail(period)
        volatility = recent_returns.std() * np.sqrt(252)  # Annualisée
        
        return volatility if pd.notna(volatility) else self.default_stop_loss_pct

    def calculate_support_resistance(self, 
                                   price_data: pd.DataFrame,
                                   lookback_periods: int = 50,
                                   min_touches: int = 2) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcule les niveaux de support et résistance (méthode simplifiée par percentiles).
        """
        if len(price_data) < lookback_periods:
            return None, None # Pas assez de données
        recent_data = price_data.tail(lookback_periods)
        support_level = recent_data['low'].quantile(0.1)
        resistance_level = recent_data['high'].quantile(0.9)
        return support_level, resistance_level
    
    def create_stop_loss_order(self,
                             symbol: str,
                             position_type: str,
                             entry_price: float,
                             quantity: float,
                             price_data: pd.DataFrame, # Doit avoir 'close', et 'high'/'low' pour ATR/SupportResistance
                             contract_multiplier: float = 1.0,
                             stop_loss_type: StopLossType = StopLossType.VOLATILITY_BASED,
                             custom_stop_pct: Optional[float] = None,
                             custom_tp_pct: Optional[float] = None) -> Optional[StopLossOrder]:
        """
        Crée un ordre de stop-loss adaptatif.
        """
        if price_data.empty or 'close' not in price_data.columns or price_data['close'].isna().all():
            self.logger.warning(f"Price data for {symbol} is empty or lacks 'close' data. Cannot create stop order.")
            return None
        current_price = price_data['close'].iloc[-1]
        
        stop_loss_price = 0.0
        
        if stop_loss_type == StopLossType.FIXED_PERCENTAGE:
            stop_pct = custom_stop_pct if custom_stop_pct is not None else self.default_stop_loss_pct
            if position_type == 'long':
                stop_loss_price = entry_price * (1 - stop_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_pct)
                
        elif stop_loss_type == StopLossType.ATR_BASED:
            if not all(col in price_data.columns for col in ['high', 'low', 'close']):
                self.logger.warning(f"ATR stop requires 'high', 'low', 'close' in price_data for {symbol}.")
                return None
            atr_series = self.calculate_atr(price_data)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                 self.logger.warning(f"ATR calculation failed or resulted in NaN for {symbol}.")
                 return None
            atr = atr_series.iloc[-1]
            if position_type == 'long':
                stop_loss_price = entry_price - (atr * self.atr_multiplier)
            else:
                stop_loss_price = entry_price + (atr * self.atr_multiplier)
                
        elif stop_loss_type == StopLossType.VOLATILITY_BASED:
            returns = price_data['close'].pct_change().dropna()
            if len(returns) < self.volatility_lookback:
                self.logger.warning(f"Not enough return data for volatility stop for {symbol}. Using fixed.")
                stop_pct = custom_stop_pct if custom_stop_pct is not None else self.default_stop_loss_pct
                if position_type == 'long': stop_loss_price = entry_price * (1 - stop_pct)
                else: stop_loss_price = entry_price * (1 + stop_pct)
            else:
                volatility = self.calculate_volatility(returns)
                vol_adjusted_pct = self.default_stop_loss_pct * (1 + volatility * 2) # Multiplier l'impact de la vol
                if position_type == 'long':
                    stop_loss_price = entry_price * (1 - vol_adjusted_pct)
                else:
                    stop_loss_price = entry_price * (1 + vol_adjusted_pct)
                    
        elif stop_loss_type == StopLossType.SUPPORT_RESISTANCE:
            if not all(col in price_data.columns for col in ['high', 'low']):
                self.logger.warning(f"Support/Resistance stop requires 'high', 'low' in price_data for {symbol}.")
                return None
            support, resistance = self.calculate_support_resistance(price_data)
            if support is None or resistance is None:
                 self.logger.warning(f"Could not calculate S/R levels for {symbol}.")
                 return None
            if position_type == 'long':
                stop_loss_price = max(support * 0.99, entry_price * (1 - self.default_stop_loss_pct * 1.5)) # Wider fixed backup
            else:
                stop_loss_price = min(resistance * 1.01, entry_price * (1 + self.default_stop_loss_pct * 1.5))
                
        elif stop_loss_type == StopLossType.DYNAMIC_PERCENTILE:
            returns = price_data['close'].pct_change().dropna()
            if len(returns) < self.volatility_lookback:
                self.logger.warning(f"Not enough return data for dynamic percentile stop for {symbol}. Using fixed.")
                stop_pct = custom_stop_pct if custom_stop_pct is not None else self.default_stop_loss_pct
                if position_type == 'long': stop_loss_price = entry_price * (1 - stop_pct)
                else: stop_loss_price = entry_price * (1 + stop_pct)
            else:
                relevant_returns = returns.tail(self.volatility_lookback)
                if position_type == 'long':
                    percentile_loss = relevant_returns.quantile(0.05) 
                    stop_loss_price = entry_price * (1 + percentile_loss) 
                else:
                    percentile_gain_as_loss = relevant_returns.quantile(0.95) 
                    stop_loss_price = entry_price * (1 + percentile_gain_as_loss)
        
        else:  # TRAILING or default to FIXED_PERCENTAGE if type is unrecognized
            stop_pct = custom_stop_pct if custom_stop_pct is not None else self.default_stop_loss_pct
            if position_type == 'long':
                stop_loss_price = entry_price * (1 - stop_pct)
            else:
                stop_loss_price = entry_price * (1 + stop_pct)
        
        tp_pct = custom_tp_pct if custom_tp_pct is not None else self.default_take_profit_pct
        take_profit_price = None
        if tp_pct > 0: # Allow disabling TP with 0 or None
            if position_type == 'long':
                take_profit_price = entry_price * (1 + tp_pct)
            else:
                take_profit_price = entry_price * (1 - tp_pct)
        
        max_loss_val = 0.0
        if position_type == 'long':
            max_loss_val = (entry_price - stop_loss_price) / entry_price if entry_price != 0 else 0
        else:
            max_loss_val = (stop_loss_price - entry_price) / entry_price if entry_price != 0 else 0
        
        order_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
        
        stop_order = StopLossOrder(
            symbol=symbol,
            position_type=position_type,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_type=stop_loss_type,
            entry_timestamp=datetime.now(), # Consider passing timestamp if backtesting
            last_update=datetime.now(),
            quantity=quantity,
            max_loss_pct=max_loss_val,
            contract_multiplier=contract_multiplier,
            highest_price=entry_price if position_type == 'long' else current_price, # Initialize for trailing
            lowest_price=entry_price if position_type == 'short' else current_price  # Initialize for trailing
        )
        
        if stop_loss_type == StopLossType.TRAILING:
            # Initial trailing distance is based on the fixed stop percentage or custom_stop_pct
            initial_stop_pct_for_trail = custom_stop_pct if custom_stop_pct is not None else self.default_stop_loss_pct
            if position_type == 'long':
                # Stop is below entry, distance is positive
                stop_order.trailing_distance = entry_price * initial_stop_pct_for_trail
                stop_order.stop_loss_price = entry_price - stop_order.trailing_distance
                stop_order.highest_price = entry_price # Start tracking from entry
            else: # short
                # Stop is above entry, distance is positive
                stop_order.trailing_distance = entry_price * initial_stop_pct_for_trail
                stop_order.stop_loss_price = entry_price + stop_order.trailing_distance
                stop_order.lowest_price = entry_price # Start tracking from entry
        
        self.active_stops[order_id] = stop_order
        self.logger.info(f"Stop-loss created for {symbol} ({order_id}): Type={stop_loss_type.value}, Entry={entry_price:.4f}, SL={stop_order.stop_loss_price:.4f}, TP={stop_order.take_profit_price if stop_order.take_profit_price else 'N/A'}")
        return stop_order
    
    def update_trailing_stop(self, order_id: str, current_price: float) -> bool:
        """
        Met à jour un trailing stop pour un ordre spécifique.
        """
        if order_id not in self.active_stops:
            self.logger.warning(f"Order ID {order_id} not found for trailing stop update.")
            return False
            
        order = self.active_stops[order_id]
        
        if order.stop_loss_type != StopLossType.TRAILING or not order.is_active:
            return False # Not a trailing stop or not active
        
        if order.trailing_distance is None or order.trailing_distance <= 0:
            self.logger.warning(f"Trailing distance not set or invalid for order {order_id}.")
            return False

        updated = False
        new_stop_price = order.stop_loss_price

        if order.position_type == 'long':
            if current_price > order.highest_price: # type: ignore
                order.highest_price = current_price
                potential_new_stop = current_price - order.trailing_distance
                if potential_new_stop > order.stop_loss_price:
                    new_stop_price = potential_new_stop
                    updated = True
        else:  # short position
            if current_price < order.lowest_price: # type: ignore
                order.lowest_price = current_price
                potential_new_stop = current_price + order.trailing_distance
                if potential_new_stop < order.stop_loss_price:
                    new_stop_price = potential_new_stop
                    updated = True
        
        if updated:
            order.stop_loss_price = new_stop_price
            order.last_update = datetime.now() # Consider passing timestamp
            order.current_price = current_price # Update current price in order state
            self.logger.info(f"Trailing stop for {order.symbol} ({order_id}) updated to: {order.stop_loss_price:.4f}")
        
        return updated
    
    def check_stop_triggers(self, 
                          price_updates: Dict[str, float], # Dict {symbol: current_price}
                          timestamp: Optional[datetime] = None) -> List[Dict]:
        """
        Vérifie si des stops doivent être déclenchés.
        price_updates: Dictionnaire des prix actuels pour les symboles concernés.
        """
        ts = timestamp if timestamp is not None else datetime.now()
            
        triggered_orders_details = []
        orders_to_deactivate = [] # Store order_id to deactivate
        
        for order_id, order in list(self.active_stops.items()): # Iterate on a copy if modifying dict
            if not order.is_active or order.symbol not in price_updates:
                continue
                
            current_price = price_updates[order.symbol]
            order.current_price = current_price # Update order's view of current price
            
            if order.stop_loss_type == StopLossType.TRAILING:
                self.update_trailing_stop(order_id, current_price) # Update SL based on new current_price
            
            stop_triggered = False
            tp_triggered = False
            time_stop_triggered = False
            exit_price_for_calc = current_price # Default exit price
            
            if order.position_type == 'long':
                if current_price <= order.stop_loss_price:
                    stop_triggered = True
                    exit_price_for_calc = order.stop_loss_price # Assume execution at stop price
                elif order.take_profit_price is not None and current_price >= order.take_profit_price:
                    tp_triggered = True
                    exit_price_for_calc = order.take_profit_price # Assume execution at TP price
            else: # short
                if current_price >= order.stop_loss_price:
                    stop_triggered = True
                    exit_price_for_calc = order.stop_loss_price
                elif order.take_profit_price is not None and current_price <= order.take_profit_price:
                    tp_triggered = True
                    exit_price_for_calc = order.take_profit_price
            
            holding_duration = (ts - order.entry_timestamp)
            if holding_duration.days >= self.max_holding_days:
                time_stop_triggered = True
                # For time stop, exit_price_for_calc remains current_price
            
            if stop_triggered or tp_triggered or time_stop_triggered:
                exit_reason = "stop_loss" if stop_triggered else ("take_profit" if tp_triggered else "time_stop")
                
                # P&L Calculation using exit_price_for_calc
                if order.position_type == 'long':
                    pnl_per_contract = (exit_price_for_calc - order.entry_price)
                else: # short
                    pnl_per_contract = (order.entry_price - exit_price_for_calc)
                
                pnl_absolute = pnl_per_contract * order.quantity * order.contract_multiplier
                pnl_pct_on_notional_entry = (pnl_absolute / (order.entry_price * order.quantity * order.contract_multiplier)) if (order.entry_price * order.quantity * order.contract_multiplier) != 0 else 0

                exit_details = {
                    'order_id': order_id, 'symbol': order.symbol,
                    'action': 'sell' if order.position_type == 'long' else 'buy',
                    'quantity': order.quantity, 'exit_price': exit_price_for_calc,
                    'exit_reason': exit_reason, 'entry_price': order.entry_price,
                    'pnl_pct': pnl_pct_on_notional_entry, 'pnl_absolute': pnl_absolute,
                    'holding_time_seconds': holding_duration.total_seconds(), 'timestamp': ts,
                    'contract_multiplier': order.contract_multiplier
                }
                
                triggered_orders_details.append(exit_details)
                orders_to_deactivate.append(order_id)
                
                self.performance_metrics['total_stops_triggered'] += 1
                if pnl_absolute > 0:
                    self.performance_metrics['profitable_stops'] += 1
                self.performance_metrics['total_pnl'] += pnl_absolute
                
                self.logger.info(f"Triggered: {order.symbol} ({order_id}) Reason: {exit_reason}, P&L: {pnl_absolute:.2f} ({pnl_pct_on_notional_entry*100:.2f}%)")
        
        for order_id_to_remove in orders_to_deactivate:
            if order_id_to_remove in self.active_stops:
                executed_order_copy = self.active_stops[order_id_to_remove] # Get a copy before deactivating/deleting
                # Find corresponding exit_details for this order_id_to_remove
                final_exit_details = next((details for details in triggered_orders_details if details['order_id'] == order_id_to_remove), None)

                self.executed_stops.append({
                    'order_snapshot': executed_order_copy, # Store the state of StopLossOrder object
                    'exit_details': final_exit_details if final_exit_details else {}
                })
                del self.active_stops[order_id_to_remove] # Or mark as inactive: self.active_stops[order_id].is_active = False
                                                       # Deleting is cleaner if not needed for other lookups.
        
        return triggered_orders_details
    
    def adjust_stops_for_volatility(self, 
                                  symbol_price_data: Dict[str, pd.DataFrame], 
                                  volatility_threshold_pct: float = 0.30) -> int: # Expects dict of DataFrames
        """
        Ajuste les stops pour les symboles donnés si leur volatilité dépasse un seuil.
        symbol_price_data: Dictionnaire {symbol: pd.DataFrame_with_OHLC}
        """
        adjusted_count = 0
        for symbol, price_data_df in symbol_price_data.items():
            if price_data_df.empty or 'close' not in price_data_df.columns:
                continue

            returns = price_data_df['close'].pct_change().dropna()
            if len(returns) < self.volatility_lookback:
                continue # Pas assez de données

            current_volatility = self.calculate_volatility(returns) # Annualized std dev
            
            # Convert threshold to match annualized std dev if it's daily, or ensure comparison is apples-to-apples
            # Assuming volatility_threshold_pct is an annualized target.
            if current_volatility > volatility_threshold_pct:
                # Volatility is higher than threshold, consider widening stops
                # Factor by how much current vol exceeds threshold, capped.
                vol_expansion_factor = min(1.5, 1 + (current_volatility - volatility_threshold_pct) / volatility_threshold_pct if volatility_threshold_pct > 0 else 1)

                for order_id, order in self.active_stops.items():
                    if order.symbol == symbol and order.is_active:
                        original_stop_loss_price = order.stop_loss_price
                        
                        # Recalculate stop based on expanded loss percentage
                        # max_loss_pct is the original acceptable loss from entry
                        expanded_loss_pct = order.max_loss_pct * vol_expansion_factor
                        
                        if order.position_type == 'long':
                            new_stop_price = order.entry_price * (1 - expanded_loss_pct)
                            # Only widen, never tighten due to this adjustment
                            if new_stop_price < order.stop_loss_price:
                                order.stop_loss_price = new_stop_price
                        else: # short
                            new_stop_price = order.entry_price * (1 + expanded_loss_pct)
                            # Only widen
                            if new_stop_price > order.stop_loss_price:
                                order.stop_loss_price = new_stop_price
                        
                        if order.stop_loss_price != original_stop_loss_price:
                            adjusted_count += 1
                            order.volatility_factor = vol_expansion_factor # Store the factor used
                            order.last_update = datetime.now() # Consider passing timestamp
                            self.logger.info(f"Stop for {symbol} ({order_id}) adjusted for high volatility ({current_volatility:.2%}) to {order.stop_loss_price:.4f}. Factor: {vol_expansion_factor:.2f}")
        
        if adjusted_count > 0:
            self.logger.info(f"Total {adjusted_count} stops adjusted for volatility.")
        return adjusted_count
    
    def get_portfolio_risk_metrics(self) -> Dict:
        """
        Calcule les métriques de risque du portefeuille basé sur les stops actifs.
        """
        active_orders_list = [order for order in self.active_stops.values() if order.is_active]
        
        if not active_orders_list:
            return {
                'total_positions_with_stops': 0, 'total_potential_risk_value': 0.0,
                'avg_max_loss_pct_on_notional': 0.0, 'largest_single_position_risk_value': 0.0,
                'active_stop_symbols': []
            }
        
        total_positions = len(active_orders_list)
        total_potential_risk_value = 0.0
        sum_max_loss_pct = 0.0
        largest_single_risk = 0.0
        
        for order in active_orders_list:
            # Potential loss if stop is hit, based on notional value at entry
            notional_at_entry = order.entry_price * order.quantity * order.contract_multiplier
            position_risk_value = order.max_loss_pct * notional_at_entry
            total_potential_risk_value += position_risk_value
            sum_max_loss_pct += order.max_loss_pct
            if position_risk_value > largest_single_risk:
                largest_single_risk = position_risk_value
        
        avg_max_loss_pct = sum_max_loss_pct / total_positions if total_positions > 0 else 0.0
        
        return {
            'total_positions_with_stops': total_positions,
            'total_potential_risk_value': total_potential_risk_value,
            'avg_max_loss_pct_on_notional': avg_max_loss_pct,
            'largest_single_position_risk_value': largest_single_risk,
            'active_stop_symbols': list(set(order.symbol for order in active_orders_list))
        }
    
    def get_performance_summary(self) -> Dict:
        """
        Génère un résumé de performance des stops exécutés.
        """
        num_executed = len(self.executed_stops)
        if num_executed == 0:
            return {
                'total_executed_stops': 0, 'win_rate': 0.0,
                'avg_pnl_per_stop': 0.0, 'total_pnl_from_stops': 0.0,
                'avg_holding_time_seconds': 0.0
            }
        
        profitable_stops_count = sum(1 for exec_stop in self.executed_stops if exec_stop['exit_details'].get('pnl_absolute', 0) > 0)
        total_pnl = sum(exec_stop['exit_details'].get('pnl_absolute', 0) for exec_stop in self.executed_stops)
        total_holding_time_seconds = sum(exec_stop['exit_details'].get('holding_time_seconds', 0) for exec_stop in self.executed_stops)
        
        win_rate = profitable_stops_count / num_executed if num_executed > 0 else 0.0
        avg_pnl = total_pnl / num_executed if num_executed > 0 else 0.0
        avg_holding_time = total_holding_time_seconds / num_executed if num_executed > 0 else 0.0
        
        # Update internal metrics (optional, could be redundant if performance_metrics is primary)
        self.performance_metrics['total_stops_triggered'] = num_executed # Assuming this tracks executed stops
        self.performance_metrics['profitable_stops'] = profitable_stops_count
        self.performance_metrics['total_pnl'] = total_pnl
        self.performance_metrics['avg_holding_time'] = avg_holding_time


        return {
            'total_executed_stops': num_executed,
            'profitable_executed_stops': profitable_stops_count,
            'win_rate': win_rate,
            'avg_pnl_per_stop': avg_pnl,
            'total_pnl_from_stops': total_pnl,
            'avg_holding_time_seconds': avg_holding_time,
            # 'executed_stop_details': self.executed_stops # Could be very verbose
        }

    # Les méthodes set_atr_stop_loss, set_trailing_stop_loss, set_volatility_stop_loss
    # sont des helpers pour créer des types spécifiques de StopLossOrder.
    # Elles devraient correctement passer le contract_multiplier à create_stop_loss_order.

    def set_atr_stop_loss(self, symbol: str, entry_price: float, quantity: float,
                         position_type: str, price_data_for_atr: pd.DataFrame,
                         contract_multiplier: float = 1.0, atr_multiplier_override: Optional[float] = None):
        atr_mult = atr_multiplier_override if atr_multiplier_override is not None else self.atr_multiplier
        # Ensure price_data_for_atr is passed to create_stop_loss_order
        # The original calculate_atr is inside create_stop_loss_order if type is ATR_BASED
        # This helper might be redundant if create_stop_loss_order is used directly with ATR_BASED type.
        # For it to be useful, it should perhaps pre-calculate ATR and pass it as custom_stop_pct or similar.
        # However, create_stop_loss_order already handles ATR calculation.
        
        # Let's make this helper simpler: it just calls create_stop_loss_order with ATR type.
        return self.create_stop_loss_order(
            symbol=symbol, position_type=position_type, entry_price=entry_price, quantity=quantity,
            price_data=price_data_for_atr, contract_multiplier=contract_multiplier,
            stop_loss_type=StopLossType.ATR_BASED
            # atr_multiplier is used internally by create_stop_loss_order
        )

    def set_trailing_stop_loss(self, symbol: str, entry_price: float, quantity: float,
                              position_type: str, price_data_for_init: pd.DataFrame, # Used for current_price
                              contract_multiplier: float = 1.0,
                              trailing_stop_pct: Optional[float] = None): # Percentage of entry_price for initial trail
        # trailing_stop_pct will be used as custom_stop_pct to set initial trailing distance
        initial_trail_pct = trailing_stop_pct if trailing_stop_pct is not None else self.default_stop_loss_pct
        
        return self.create_stop_loss_order(
            symbol=symbol, position_type=position_type, entry_price=entry_price, quantity=quantity,
            price_data=price_data_for_init, contract_multiplier=contract_multiplier,
            stop_loss_type=StopLossType.TRAILING,
            custom_stop_pct=initial_trail_pct # This sets the initial distance for trailing
        )

    def set_volatility_stop_loss(self, symbol: str, entry_price: float, quantity: float,
                                position_type: str, price_data_for_vol: pd.DataFrame,
                                contract_multiplier: float = 1.0):
        return self.create_stop_loss_order(
            symbol=symbol, position_type=position_type, entry_price=entry_price, quantity=quantity,
            price_data=price_data_for_vol, contract_multiplier=contract_multiplier,
            stop_loss_type=StopLossType.VOLATILITY_BASED
        )

    def check_exit_conditions(self, symbol: str, current_price: float, 
                             position_type: str) -> Tuple[bool, str, Optional[str]]: # Added order_id
        """
        Vérifie si les conditions de sortie sont remplies pour un symbole donné.
        Retourne (should_exit, exit_reason, order_id_if_found).
        """
        active_order_found: Optional[StopLossOrder] = None
        found_order_id: Optional[str] = None

        for order_id_iter, order_iter in self.active_stops.items():
            if order_iter.symbol == symbol and order_iter.is_active and order_iter.position_type == position_type:
                active_order_found = order_iter
                found_order_id = order_id_iter
                break # Assume one active stop per symbol/type for this check
        
        if not active_order_found or not found_order_id:
            return False, "No active stop found for symbol/type", None
        
        # Update current price for the order (important for trailing stop logic if called externally)
        active_order_found.current_price = current_price
        if active_order_found.stop_loss_type == StopLossType.TRAILING:
             self.update_trailing_stop(found_order_id, current_price)


        if position_type == 'long':
            if current_price <= active_order_found.stop_loss_price:
                return True, "Stop-loss triggered", found_order_id
            if active_order_found.take_profit_price is not None and current_price >= active_order_found.take_profit_price:
                return True, "Take-profit triggered", found_order_id
        else: # short
            if current_price >= active_order_found.stop_loss_price:
                return True, "Stop-loss triggered", found_order_id
            if active_order_found.take_profit_price is not None and current_price <= active_order_found.take_profit_price:
                return True, "Take-profit triggered", found_order_id
        
        # Check time-based stop (optional, might be better handled in check_stop_triggers)
        # holding_time = (datetime.now() - active_order_found.entry_timestamp).days
        # if holding_time >= self.max_holding_days:
        #    return True, "Time-based stop triggered", found_order_id

        return False, "No exit conditions met", found_order_id

    def get_stop_levels(self, symbol: str, position_type: Optional[str] = None) -> Optional[Dict]:
        """
        Récupère les niveaux de stop pour un symbole et optionnellement un type de position.
        """
        for order in self.active_stops.values():
            if order.symbol == symbol and order.is_active:
                if position_type is None or order.position_type == position_type:
                    return {
                        'order_id': next((oid for oid, o in self.active_stops.items() if o == order), None), # Find order_id
                        'stop_loss_price': order.stop_loss_price,
                        'take_profit_price': order.take_profit_price,
                        'entry_price': order.entry_price,
                        'stop_loss_type': order.stop_loss_type.value,
                        'contract_multiplier': order.contract_multiplier,
                        'quantity': order.quantity
                    }
        return None

    # La méthode calculate_portfolio_risk_metrics est renommée en get_portfolio_risk_metrics
    # et est déjà définie plus haut. Si une version prenant `positions` est nécessaire,
    # elle devrait être implémentée différemment pour ne pas juste appeler celle qui utilise self.active_stops.
    # Pour l'instant, on considère que get_portfolio_risk_metrics est la méthode principale.
