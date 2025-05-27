import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# Nouveaux imports
from ..risk_management.risk_controls import (
    check_single_trade_notional_vs_capital, # Renommé et potentiellement logique modifiée
    check_total_leverage_limit,
    check_max_notional_exposure_per_asset,
    check_max_total_notional_exposure, # Nouvelle fonction
    check_max_leverage_per_trade,      # Nouvelle fonction
    check_margin_usage_limit           # Nouvelle fonction
)

class BacktestSimulator:
    """
    Simulateur pour le backtesting de stratégies de trading, adapté pour les futures.

    Ce simulateur prend des signaux (ACHETER, VENDRE, CONSERVER) et des tailles de position,
    et simule leur exécution sur des données de marché historiques. Il calcule les
    profits/pertes, gère l'effet de levier, la marge, et suit l'évolution du capital.
    """

    def __init__(self,
                 initial_capital: float,
                 market_data: pd.DataFrame, 
                 asset_symbol: str = "DEFAULT_ASSET", 
                 default_leverage: float = 1.0,
                 maintenance_margin_pct: float = 0.05, 
                 commission_pct_per_trade: float = 0.0005,
                 liquidation_slippage_pct: float = 0.005, 
                 funding_fee_check_interval_hours: Optional[int] = 8, 
                 risk_manager_config: Optional[Dict[str, Any]] = None
                 ):
        """
        Initialise le BacktestSimulator.
        """
        if not isinstance(market_data.index, pd.DatetimeIndex):
            raise ValueError("L'index de market_data doit être de type DatetimeIndex (timestamp).")
        if not all(col in market_data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("market_data doit contenir les colonnes 'open', 'high', 'low', 'close'.")

        self.initial_capital: float = initial_capital
        self.market_data: pd.DataFrame = market_data.sort_index() 
        self.asset_symbol: str = asset_symbol
        print(f"DEBUG SIMULATOR: Initialized with asset_symbol = {self.asset_symbol}")

        self.default_leverage: float = default_leverage
        self.maintenance_margin_pct: float = maintenance_margin_pct
        self.commission_pct_per_trade: float = commission_pct_per_trade
        self.liquidation_slippage_pct: float = liquidation_slippage_pct
        self.funding_fee_check_interval: Optional[pd.Timedelta] = pd.Timedelta(hours=funding_fee_check_interval_hours) if funding_fee_check_interval_hours is not None else None
        self.risk_manager_config: Dict[str, Any] = risk_manager_config if risk_manager_config is not None else {}

        self.available_funds: float = initial_capital
        self.used_margin: float = 0.0 
        self.portfolio_value: float = initial_capital 
        self.last_price_seen: Dict[str, float] = {} 

        self.open_positions: Dict[str, Dict[str, Any]] = {} 

        self.trades_history: List[Dict[str, Any]] = []
        
        initial_timestamp = pd.Timestamp('now', tz='UTC') 
        initial_market_price = 0.0
        if not self.market_data.empty:
            initial_timestamp = self.market_data.index.min() - pd.Timedelta(days=1)
            if 'open' in self.market_data.columns:
                 initial_market_price = self.market_data['open'].iloc[0]
        self.last_price_seen[self.asset_symbol] = initial_market_price

        self.portfolio_history: List[Dict[str, Any]] = [{
            'timestamp': initial_timestamp,
            'available_funds': self.available_funds,
            'used_margin': self.used_margin,
            'unrealized_pnl': 0.0,
            'portfolio_value': self.portfolio_value,
            'num_open_positions': 0,
            'total_notional_value': 0.0,
            f'market_price_{self.asset_symbol}': initial_market_price
        }]
        self.liquidation_events: List[Dict[str, Any]] = []
        self.funding_fee_history: List[Dict[str, Any]] = []
        self.last_funding_fee_check_ts: Optional[pd.Timestamp] = None

    def _calculate_unrealized_pnl(self, current_prices: Dict[str, float], target_asset_symbol: Optional[str] = None) -> float:
        """
        Calcule le P&L non réalisé.
        Si target_asset_symbol est fourni, calcule pour cette position uniquement.
        Sinon, calcule le P&L total de toutes les positions ouvertes.
        """
        total_pnl = 0.0
        positions_to_calculate = {}

        if target_asset_symbol:
            if target_asset_symbol in self.open_positions:
                positions_to_calculate = {target_asset_symbol: self.open_positions[target_asset_symbol]}
        else:
            positions_to_calculate = self.open_positions

        for asset_symbol, pos_details in positions_to_calculate.items():
            current_price = current_prices.get(asset_symbol)
            if current_price is None:
                current_price = self.last_price_seen.get(asset_symbol)
                if current_price is None:
                    print(f"Warning: Prix actuel manquant pour {asset_symbol} lors du calcul du P&L.")
                    continue

            price_diff = current_price - pos_details['entry_price']
            if pos_details['type'] == 'SHORT': # quantity is negative for short
                price_diff = pos_details['entry_price'] - current_price
            
            contract_multiplier = pos_details.get('contract_multiplier', 1.0)
            pnl_for_position = price_diff * pos_details['quantity'] * contract_multiplier
            total_pnl += pnl_for_position
        return total_pnl

    def _update_portfolio_value(self, timestamp: pd.Timestamp, current_prices: Dict[str, float]):
        """Met à jour la valeur du portefeuille et l'historique."""
        unrealized_pnl = self._calculate_unrealized_pnl(current_prices)
        self.portfolio_value = self.available_funds + self.used_margin + unrealized_pnl
        
        for asset_sym, price in current_prices.items():
            self.last_price_seen[asset_sym] = price

        total_notional_value = 0.0
        for asset_symbol_pos, pos_details in self.open_positions.items():
            price_for_notional = current_prices.get(asset_symbol_pos, self.last_price_seen.get(asset_symbol_pos))
            if price_for_notional is not None:
                contract_multiplier = pos_details.get('contract_multiplier', 1.0)
                total_notional_value += abs(pos_details['quantity']) * price_for_notional * contract_multiplier
            else:
                print(f"Warning: Prix manquant pour {asset_symbol_pos} pour calculer la valeur notionnelle.")

        history_entry = {
            'timestamp': timestamp,
            'available_funds': self.available_funds,
            'used_margin': self.used_margin,
            'unrealized_pnl': unrealized_pnl,
            'portfolio_value': self.portfolio_value,
            'num_open_positions': len(self.open_positions),
            'total_notional_value': total_notional_value,
        }
        main_asset_price = current_prices.get(self.asset_symbol, self.last_price_seen.get(self.asset_symbol, 0))
        history_entry[f'market_price_{self.asset_symbol}'] = main_asset_price
        
        self.portfolio_history.append(history_entry)

    def _liquidate_position(self, timestamp: pd.Timestamp, asset_symbol: str, liquidation_price: float, reason: str):
        """Liquide une position spécifique."""
        if asset_symbol not in self.open_positions:
            return

        pos_details = self.open_positions[asset_symbol]
        contract_multiplier = pos_details.get('contract_multiplier', 1.0)
        
        actual_liquidation_price = liquidation_price
        if pos_details['type'] == 'LONG':
            actual_liquidation_price *= (1 - self.liquidation_slippage_pct)
        else: # SHORT
            actual_liquidation_price *= (1 + self.liquidation_slippage_pct)

        print(f"LIQUIDATION: {asset_symbol} à {timestamp}. Raison: {reason}. Prix demandé: {liquidation_price:.2f}, Prix réel après slippage: {actual_liquidation_price:.2f}")

        contracts_to_close = pos_details['quantity'] # This is signed
        notional_value_closed = abs(contracts_to_close) * actual_liquidation_price * contract_multiplier
        commission = notional_value_closed * self.commission_pct_per_trade

        price_diff = actual_liquidation_price - pos_details['entry_price']
        realized_pnl = price_diff * contracts_to_close * contract_multiplier # contracts_to_close is signed
        
        margin_returned_from_position = pos_details['initial_margin'] 

        self.available_funds += margin_returned_from_position + realized_pnl - commission 
        self.used_margin -= pos_details['initial_margin']
        
        self.trades_history.append({
            'timestamp': timestamp, 'symbol': asset_symbol, 'type': f'LIQUIDATION_{pos_details["type"]}', 
            'price': actual_liquidation_price, 'quantity': contracts_to_close, # Signed quantity
            'notional_value': notional_value_closed, 'realized_pnl': realized_pnl,
            'commission': commission, 'margin_freed': margin_returned_from_position, 
            'leverage': pos_details['leverage'], 'contract_multiplier': contract_multiplier
        })
        
        pnl_before_slippage_calc = self._calculate_unrealized_pnl({asset_symbol: liquidation_price}, asset_symbol)

        liquidation_details = {
            'position_equity_before_liq': pos_details['initial_margin'] + pnl_before_slippage_calc,
            'liquidation_price_no_slippage': liquidation_price,
            'actual_liquidation_price': actual_liquidation_price,
            'slippage_amount': abs(liquidation_price - actual_liquidation_price) * abs(contracts_to_close) * contract_multiplier,
            'realized_pnl_after_slippage_commission': realized_pnl - commission,
            **pos_details 
        }
        self.liquidation_events.append({
            'timestamp': timestamp, 'symbol': asset_symbol, 'reason': reason,
            'details': liquidation_details
        })
        del self.open_positions[asset_symbol]

    def _check_and_handle_margin_calls(self, timestamp: pd.Timestamp, current_prices: Dict[str, float]):
        """Vérifie et gère les appels de marge pour les positions ouvertes."""
        for asset_symbol_iter in list(self.open_positions.keys()):
            pos_details = self.open_positions.get(asset_symbol_iter)
            if not pos_details:
                continue

            current_price_asset = current_prices.get(asset_symbol_iter, self.last_price_seen.get(asset_symbol_iter))
            if current_price_asset is None:
                print(f"Warning: Prix manquant pour {asset_symbol_iter} lors de la vérification de marge.")
                continue

            pnl_for_position = self._calculate_unrealized_pnl({asset_symbol_iter: current_price_asset}, asset_symbol_iter)
            position_equity = pos_details['initial_margin'] + pnl_for_position
            
            contract_multiplier = pos_details.get('contract_multiplier', 1.0)
            current_notional_value = abs(pos_details['quantity']) * current_price_asset * contract_multiplier
            maintenance_margin_required_for_pos = current_notional_value * self.maintenance_margin_pct

            if position_equity < maintenance_margin_required_for_pos:
                print(f"ALERT: Appel de marge pour {asset_symbol_iter} à {timestamp}! Équité: {position_equity:.2f}, Maintien requis: {maintenance_margin_required_for_pos:.2f}. Liquidation en cours...")
                self._liquidate_position(timestamp, asset_symbol_iter, current_price_asset, "Appel de marge non couvert (maintenance)")

    def _apply_funding_fees(self, timestamp: pd.Timestamp, current_prices_all_assets: Dict[str, float], funding_rates: Dict[str, float]):
        """Applique les frais de financement aux positions ouvertes."""
        for asset_symbol, pos_details in self.open_positions.items():
            funding_rate = funding_rates.get(asset_symbol)
            current_price = current_prices_all_assets.get(asset_symbol, self.last_price_seen.get(asset_symbol))

            if funding_rate is None or pd.isna(funding_rate) or funding_rate == 0 or current_price is None:
                continue
            
            contract_multiplier = pos_details.get('contract_multiplier', 1.0)
            position_notional_value_signed = pos_details['quantity'] * current_price * contract_multiplier
            funding_payment = -position_notional_value_signed * funding_rate
            
            self.available_funds += funding_payment
            
            self.funding_fee_history.append({
                'timestamp': timestamp, 'symbol': asset_symbol, 'funding_rate': funding_rate,
                'notional_value': abs(position_notional_value_signed), 
                'funding_fee': funding_payment, 
                'position_type': pos_details['type']
            })

    def execute_trade(self, timestamp: pd.Timestamp, asset_symbol_trade: str, signal: str, nominal_value_to_trade: float, current_price: float, leverage: Optional[float] = None, contract_multiplier: float = 1.0) -> None:
        print(f"DEBUG SIMULATOR: execute_trade called with asset_symbol_trade = {asset_symbol_trade}, signal = {signal}")
        trade_leverage = leverage if leverage is not None else self.default_leverage

        if trade_leverage <= 0:
            print(f"Warning: Levier non positif ({trade_leverage}) à {timestamp} pour {asset_symbol_trade}. Trade ignoré.")
            return

        if signal not in ['BUY', 'SELL', 'HOLD', 'CLOSE_LONG', 'CLOSE_SHORT']:
            print(f"Warning: Signal '{signal}' non reconnu à {timestamp} pour {asset_symbol_trade}. Trade ignoré.")
            return

        if signal == 'BUY' or signal == 'SELL':
            # 1. Contrôle: Exposition notionnelle du trade vs capital total
            max_notional_trade_pct_capital = self.risk_manager_config.get('max_notional_trade_pct_capital') # Nouveau nom de config possible
            if max_notional_trade_pct_capital is None: # Fallback sur l'ancien nom pour compatibilité
                max_notional_trade_pct_capital = self.risk_manager_config.get('max_pos_pct_capital')

            if max_notional_trade_pct_capital is not None:
                if not check_single_trade_notional_vs_capital(
                    trade_notional_value=nominal_value_to_trade, # nominal_value_to_trade est déjà la valeur notionnelle
                    total_capital=self.portfolio_value,
                    max_notional_as_pct_of_capital=max_notional_trade_pct_capital
                ):
                    print(f"RiskWarning: Trade {signal} {asset_symbol_trade} à {timestamp} (notionnel: {nominal_value_to_trade:.2f}) dépasse la limite notionnelle par trade ({max_notional_trade_pct_capital*100}% du capital {self.portfolio_value:.2f}). Trade ignoré.")
                    return

            # 2. Contrôle: Exposition notionnelle maximale par actif
            max_notional_per_asset_key = f'max_notional_per_asset_{asset_symbol_trade}' # Plus spécifique
            max_notional_per_asset_val = self.risk_manager_config.get(max_notional_per_asset_key)
            if max_notional_per_asset_val is None: # Fallback sur l'ancien nom générique
                 max_notional_per_asset_val = self.risk_manager_config.get(f'max_notional_{asset_symbol_trade}')

            if max_notional_per_asset_val is not None:
                # Exposition actuelle sur l'actif + trade proposé
                current_notional_on_asset = 0
                if asset_symbol_trade in self.open_positions:
                    pos = self.open_positions[asset_symbol_trade]
                    price = self.last_price_seen.get(asset_symbol_trade, pos['entry_price'])
                    cm = pos.get('contract_multiplier', 1.0)
                    current_notional_on_asset = abs(pos['quantity']) * price * cm
                
                projected_notional_on_asset = current_notional_on_asset + nominal_value_to_trade # Supposant que nominal_value_to_trade est pour une NOUVELLE position
                
                is_allowed, reason = check_max_notional_exposure_per_asset(projected_notional_on_asset, max_notional_per_asset_val)
                if not is_allowed:
                    print(f"RiskWarning: Trade {signal} {asset_symbol_trade} à {timestamp}. {reason} Trade ignoré.")
                    return

            # Calcul de l'exposition notionnelle totale actuelle et projetée
            current_total_notional = sum(
                abs(pos['quantity']) * self.last_price_seen.get(sym, pos['entry_price']) * pos.get('contract_multiplier', 1.0)
                for sym, pos in self.open_positions.items()
            )
            projected_total_notional_after_trade = current_total_notional + nominal_value_to_trade

            # 3. Contrôle: Levier maximum par trade
            max_leverage_trade_config = self.risk_manager_config.get('max_leverage_per_trade')
            if max_leverage_trade_config is not None:
                is_allowed, reason = check_max_leverage_per_trade(trade_leverage, max_leverage_trade_config)
                if not is_allowed:
                    print(f"RiskWarning: Trade {signal} {asset_symbol_trade} à {timestamp}. {reason} Trade ignoré.")
                    return

            # 4. Contrôle: Levier total du portefeuille
            max_portfolio_leverage_config = self.risk_manager_config.get('max_portfolio_leverage')
            if max_portfolio_leverage_config is not None:
                is_allowed, reason = check_total_leverage_limit(self.portfolio_value, projected_total_notional_after_trade, max_portfolio_leverage_config)
                if not is_allowed:
                     print(f"RiskWarning: Trade {signal} {asset_symbol_trade} à {timestamp} dépasserait le levier max du portefeuille. {reason} Trade ignoré.")
                     return
                elif self.portfolio_value <= 0 and projected_total_notional_after_trade > 0 :
                     print(f"RiskWarning: Tentative d'ouvrir une position {signal} {asset_symbol_trade} à {timestamp} avec une équité de portefeuille nulle ou négative. Trade ignoré.")
                     return
            
            # 5. Contrôle: Exposition notionnelle totale maximale du portefeuille
            max_total_notional_config = self.risk_manager_config.get('max_total_notional_exposure')
            if max_total_notional_config is not None:
                is_allowed, reason = check_max_total_notional_exposure(projected_total_notional_after_trade, max_total_notional_config)
                if not is_allowed:
                    print(f"RiskWarning: Trade {signal} {asset_symbol_trade} à {timestamp}. {reason} Trade ignoré.")
                    return

            # 6. Contrôle: Utilisation de la marge
            required_margin_for_trade = nominal_value_to_trade / trade_leverage
            projected_used_margin = self.used_margin + required_margin_for_trade
            max_margin_usage_pct_config = self.risk_manager_config.get('max_margin_usage_pct_limit')
            if max_margin_usage_pct_config is not None:
                is_allowed, reason = check_margin_usage_limit(projected_used_margin, self.portfolio_value, max_margin_usage_pct_config)
                if not is_allowed:
                    print(f"RiskWarning: Trade {signal} {asset_symbol_trade} à {timestamp}. {reason} Trade ignoré.")
                    return

        if signal == 'BUY' or signal == 'SELL':
            if nominal_value_to_trade <= 0:
                print(f"Warning: Valeur notionnelle non positive ({nominal_value_to_trade}) pour {signal} à {timestamp} pour {asset_symbol_trade}. Trade ignoré.")
                return
            if asset_symbol_trade in self.open_positions:
                print(f"Warning: Position déjà ouverte pour {asset_symbol_trade} lors d'une tentative d'{signal} à {timestamp}. Trade ignoré.")
                return

            contracts_to_trade_abs = nominal_value_to_trade / (current_price * contract_multiplier)
            required_margin = nominal_value_to_trade / trade_leverage
            commission = nominal_value_to_trade * self.commission_pct_per_trade

            if required_margin + commission > self.available_funds:
                print(f"Warning: Fonds disponibles ({self.available_funds:.2f}) insuffisants pour marge ({required_margin:.2f}) et commission ({commission:.2f}) pour {signal} {asset_symbol_trade} à {timestamp}. Trade ignoré.")
                return

            self.available_funds -= (required_margin + commission)
            self.used_margin += required_margin
            
            position_type = 'LONG' if signal == 'BUY' else 'SHORT'
            quantity_signed = contracts_to_trade_abs if position_type == 'LONG' else -contracts_to_trade_abs
            
            self.open_positions[asset_symbol_trade] = {
                'type': position_type, 'quantity': quantity_signed, 'entry_price': current_price,
                'leverage': trade_leverage, 'initial_margin': required_margin,
                'notional_value_at_entry': nominal_value_to_trade, 'timestamp': timestamp,
                'contract_multiplier': contract_multiplier
            }
            self.trades_history.append({
                'timestamp': timestamp, 'symbol': asset_symbol_trade, 'type': signal,
                'price': current_price, 'quantity': quantity_signed, 
                'notional_value': nominal_value_to_trade, 'leverage': trade_leverage,
                'margin_used': required_margin, 'commission': commission, 'realized_pnl': 0,
                'contract_multiplier': contract_multiplier
            })

        elif signal == 'CLOSE_LONG' or signal == 'CLOSE_SHORT':
            if asset_symbol_trade not in self.open_positions:
                print(f"Warning: Aucune position {asset_symbol_trade} ouverte à fermer avec {signal} à {timestamp}. Trade ignoré.")
                return

            current_pos = self.open_positions[asset_symbol_trade]
            pos_contract_multiplier = current_pos.get('contract_multiplier', 1.0)

            if (signal == 'CLOSE_LONG' and current_pos['type'] != 'LONG') or \
               (signal == 'CLOSE_SHORT' and current_pos['type'] != 'SHORT'):
                print(f"Warning: Incompatibilité de signal de fermeture pour {asset_symbol_trade}. Signal: {signal}, Type: {current_pos['type']} à {timestamp}. Trade ignoré.")
                return
            
            contracts_to_close_abs = abs(current_pos['quantity']) 
            if nominal_value_to_trade > 0: 
                 if current_price * pos_contract_multiplier > 1e-9: 
                    contracts_to_close_abs = min(nominal_value_to_trade / (current_price * pos_contract_multiplier), abs(current_pos['quantity']))
                 else: 
                    print(f"Warning: Prix ou multiplicateur de contrat nul pour {asset_symbol_trade} à {timestamp} lors de la fermeture partielle. Fermeture totale forcée.")
            
            if nominal_value_to_trade < 0:
                 print(f"Warning: Valeur notionnelle invalide ({nominal_value_to_trade}) pour {signal} {asset_symbol_trade} à {timestamp}. Trade ignoré.")
                 return
            
            if contracts_to_close_abs <= 1e-9: 
                print(f"Warning: Tentative de fermer une quantité négligeable de contrats ({contracts_to_close_abs:.8f}) pour {asset_symbol_trade} à {timestamp}. Trade ignoré.")
                return
            
            signed_contracts_to_close = contracts_to_close_abs if current_pos['type'] == 'LONG' else -contracts_to_close_abs
            if abs(signed_contracts_to_close) > abs(current_pos['quantity']): 
                signed_contracts_to_close = current_pos['quantity']


            notional_value_closed = abs(signed_contracts_to_close) * current_price * pos_contract_multiplier
            commission = notional_value_closed * self.commission_pct_per_trade
            
            price_diff = current_price - current_pos['entry_price']
            realized_pnl = price_diff * signed_contracts_to_close * pos_contract_multiplier

            fraction_closed = abs(signed_contracts_to_close / current_pos['quantity']) if abs(current_pos['quantity']) > 1e-9 else 1.0
            margin_to_return = current_pos['initial_margin'] * fraction_closed
            
            self.available_funds += margin_to_return + realized_pnl - commission
            self.used_margin -= margin_to_return
            
            self.trades_history.append({
                'timestamp': timestamp, 'symbol': asset_symbol_trade, 'type': signal,
                'price': current_price, 'quantity': signed_contracts_to_close, 
                'notional_value': notional_value_closed, 'realized_pnl': realized_pnl,
                'commission': commission, 'margin_freed': margin_to_return, 
                'leverage': current_pos['leverage'], 'contract_multiplier': pos_contract_multiplier
            })

            remaining_quantity = current_pos['quantity'] - signed_contracts_to_close
            if abs(remaining_quantity) <= 1e-9:
                del self.open_positions[asset_symbol_trade]
            else:
                current_pos['quantity'] = remaining_quantity
                current_pos['initial_margin'] -= margin_to_return 
                current_pos_notional_at_entry_old = current_pos['notional_value_at_entry']
                current_pos['notional_value_at_entry'] = current_pos_notional_at_entry_old * (abs(remaining_quantity) / abs(current_pos['quantity'] + signed_contracts_to_close)) if abs(current_pos['quantity'] + signed_contracts_to_close) > 1e-9 else 0


    def run_simulation(self, signals_df: pd.DataFrame, leverage_series: Optional[pd.Series] = None, contract_multipliers: Optional[Dict[str, float]] = None) -> None:
        if not isinstance(signals_df.index, pd.DatetimeIndex):
            raise ValueError("L'index de signals_df doit être de type DatetimeIndex (timestamp).")
        if not all(col in signals_df.columns for col in ['signal', 'nominal_value_to_trade']):
            raise ValueError("signals_df doit contenir les colonnes 'signal' et 'nominal_value_to_trade'.")

        asset_cm = 1.0
        if contract_multipliers and self.asset_symbol in contract_multipliers:
            asset_cm = contract_multipliers[self.asset_symbol]
        
        if 'funding_rate' not in self.market_data.columns:
            self.market_data['funding_rate'] = np.nan 

        merged_data = pd.merge_asof(signals_df.sort_index(),
                                    self.market_data[['open', 'funding_rate']].sort_index(), 
                                    left_index=True, right_index=True,
                                    direction='forward', tolerance=pd.Timedelta(days=1))
        merged_data['open'] = merged_data['open'].ffill().bfill()
        merged_data['funding_rate'] = merged_data['funding_rate'].ffill() 

        if leverage_series is not None:
            if not isinstance(leverage_series.index, pd.DatetimeIndex):
                 raise ValueError("L'index de leverage_series doit être de type DatetimeIndex (timestamp).")
            leverage_series_renamed = leverage_series.rename('custom_leverage').to_frame()
            merged_data = pd.merge_asof(merged_data, leverage_series_renamed,
                                        left_index=True, right_index=True,
                                        direction='backward', tolerance=pd.Timedelta(days=1))
            merged_data['custom_leverage'] = merged_data['custom_leverage'].ffill()
        else:
            merged_data['custom_leverage'] = np.nan

        if not self.market_data.empty:
            initial_market_price = self.market_data['open'].iloc[0] if not self.market_data.empty else 0
            if not self.portfolio_history or self.portfolio_history[0]['timestamp'] > (self.market_data.index.min() - pd.Timedelta(microseconds=1)):
                self.portfolio_history = [{
                    'timestamp': self.market_data.index.min() - pd.Timedelta(microseconds=1),
                    'available_funds': self.initial_capital, 'used_margin': 0.0, 'unrealized_pnl': 0.0,
                    'portfolio_value': self.initial_capital, 'num_open_positions': 0,
                    'total_notional_value': 0.0, f'market_price_{self.asset_symbol}': initial_market_price 
                }]
            self.last_price_seen[self.asset_symbol] = initial_market_price
        
        all_relevant_timestamps = pd.Index(self.market_data.index.tolist() + signals_df.index.tolist()).unique().sort_values()
        if not self.market_data.empty:
            # Fix boolean indexing by using proper filtering
            min_time = self.market_data.index.min()
            max_time = self.market_data.index.max()
            mask = (all_relevant_timestamps >= min_time) & (all_relevant_timestamps <= max_time)
            all_relevant_timestamps = all_relevant_timestamps[mask]
        
        last_known_market_price_main_asset = self.market_data['open'].iloc[0] if not self.market_data.empty else 0.0

        for current_ts in all_relevant_timestamps:
            current_market_price_main = self.market_data.loc[current_ts, 'open'] if current_ts in self.market_data.index else last_known_market_price_main_asset
            last_known_market_price_main_asset = current_market_price_main
            
            current_market_prices_dict = {self.asset_symbol: current_market_price_main}
            
            funding_rate_main = self.market_data.get('funding_rate', pd.Series(dtype=float)).get(current_ts, np.nan)
            if pd.isna(funding_rate_main) and current_ts in merged_data.index:
                 row_data = merged_data.loc[current_ts]
                 funding_rate_main = row_data['funding_rate'] if isinstance(row_data, pd.Series) else row_data['funding_rate'].iloc[0] if not isinstance(row_data['funding_rate'], (float, np.floating)) else row_data['funding_rate']

            funding_rates_dict = {self.asset_symbol: funding_rate_main}

            self._check_and_handle_margin_calls(current_ts, current_market_prices_dict)

            if current_ts in merged_data.index:
                signal_rows_at_ts = merged_data.loc[[current_ts]] if isinstance(merged_data.loc[[current_ts]], pd.DataFrame) else pd.DataFrame([merged_data.loc[current_ts]])
                for _, signal_row in signal_rows_at_ts.iterrows():
                    signal_asset_symbol_exec = self.asset_symbol 
                    
                    cm_exec = asset_cm 
                    if contract_multipliers and signal_asset_symbol_exec in contract_multipliers:
                        cm_exec = contract_multipliers[signal_asset_symbol_exec]

                    signal = signal_row['signal']
                    nominal_value = signal_row['nominal_value_to_trade']
                    
                    execution_price = signal_row['open'] 
                    if pd.isna(execution_price):
                        execution_price = current_market_prices_dict.get(signal_asset_symbol_exec, last_known_market_price_main_asset)
                        print(f"Warning: Prix d'exécution non disponible pour signal {signal_asset_symbol_exec} à {current_ts}. Utilisation du prix de marché: {execution_price}.")
                    
                    custom_leverage_for_trade = signal_row.get('custom_leverage', np.nan)
                    if pd.isna(custom_leverage_for_trade): custom_leverage_for_trade = None
                    
                    if signal != 'HOLD':
                        self.execute_trade(current_ts, signal_asset_symbol_exec, signal, nominal_value, execution_price, leverage=custom_leverage_for_trade, contract_multiplier=cm_exec)
            
            if self.funding_fee_check_interval:
                if self.last_funding_fee_check_ts is None or (current_ts - self.last_funding_fee_check_ts) >= self.funding_fee_check_interval:
                    if len(self.open_positions) > 0:
                         self._apply_funding_fees(current_ts, current_market_prices_dict, funding_rates_dict)
                    self.last_funding_fee_check_ts = current_ts
            elif any(pd.notna(rate) for rate in funding_rates_dict.values()) and len(self.open_positions) > 0 :
                 self._apply_funding_fees(current_ts, current_market_prices_dict, funding_rates_dict)

            self._update_portfolio_value(current_ts, current_market_prices_dict)
            
        if not self.market_data.empty and (not self.portfolio_history or self.portfolio_history[-1]['timestamp'] < self.market_data.index.max()):
             final_price_main = self.market_data['close'].iloc[-1]
             final_prices_dict = {self.asset_symbol: final_price_main}
             self._update_portfolio_value(self.market_data.index.max(), final_prices_dict)

    def get_portfolio_history(self) -> pd.DataFrame:
        if not self.portfolio_history:
            cols = ['timestamp', 'available_funds', 'used_margin', 'unrealized_pnl', 
                    'portfolio_value', 'num_open_positions', 'total_notional_value']
            if hasattr(self, 'asset_symbol') and self.asset_symbol: 
                 cols.append(f'market_price_{self.asset_symbol}')
            else: 
                 cols.append('market_price_DEFAULT_ASSET') 
            return pd.DataFrame(columns=cols)
            
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='last')] 
        return df

    def get_trades_history(self) -> pd.DataFrame:
        if not self.trades_history:
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'type', 'price', 'quantity', 
                'notional_value', 'leverage', 'margin_used', 'commission',
                'realized_pnl', 'margin_freed', 'contract_multiplier'
            ])
        df = pd.DataFrame(self.trades_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'symbol' in df.columns:
             df = df.set_index(['timestamp', 'symbol']).sort_index()
        else:
            df = df.set_index('timestamp').sort_index()
        return df

    def get_liquidation_events(self) -> pd.DataFrame:
        if not self.liquidation_events:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'reason', 'details'])
        df = pd.DataFrame(self.liquidation_events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'symbol' in df.columns:
            df = df.set_index(['timestamp', 'symbol']).sort_index()
        else:
            df = df.set_index('timestamp').sort_index()
        return df

    def get_funding_fee_history(self) -> pd.DataFrame:
        if not self.funding_fee_history:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'funding_rate', 'notional_value', 'funding_fee', 'position_type'])
        df = pd.DataFrame(self.funding_fee_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'symbol' in df.columns:
            df = df.set_index(['timestamp', 'symbol']).sort_index()
        else:
            df = df.set_index('timestamp').sort_index()
        return df


if __name__ == '__main__':
    ASSET_ID = "BTCUSDT_FUTURE"
    CONTRACT_MULT = 0.001 

    data = {
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 10:00:00', '2023-01-01 11:00:00', 
                                      '2023-01-02 09:00:00', '2023-01-03 09:00:00', '2023-01-04 09:00:00', 
                                      '2023-01-05 09:00:00', '2023-01-06 09:00:00']),
        'open': [100.0, 101.0, 100.5, 102.0, 101.0, 98.0, 95.0, 97.0], 
        'high': [103.0, 101.5, 101.0, 104.0, 102.5, 99.0, 98.0, 98.5],
        'low': [99.0, 100.0, 100.0, 101.0, 100.0, 94.0, 93.0, 96.0],
        'close': [101.0, 100.5, 102.0, 101.5, 98.0, 95.0, 97.0, 96.5],
        'funding_rate': [0.0001, np.nan, -0.0002, np.nan, 0.0001, np.nan, -0.0001, np.nan] 
    }
    market_df = pd.DataFrame(data).set_index('timestamp')

    signals_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-03 09:00:00', '2023-01-05 09:00:00']),
        'signal': ['BUY', 'CLOSE_LONG', 'SELL'], 
        'nominal_value_to_trade': [10000.0, 0, 5000.0] 
    }
    signals_df = pd.DataFrame(signals_data).set_index('timestamp')
    
    leverage_s = pd.Series([10.0, 20.0], index=pd.to_datetime(['2023-01-01 09:00:00', '2023-01-04 09:00:00']))
    leverage_s.name = 'leverage'

    print("Market Data:")
    print(market_df)
    print("\nSignals Data:")
    print(signals_df)
    print("\nLeverage Series:")
    print(leverage_s)

    risk_conf = {
        'max_pos_pct_capital': 0.5, 
        'max_portfolio_leverage': 20.0,
        f'max_notional_{ASSET_ID}': 50000.0 
    }

    simulator = BacktestSimulator(initial_capital=1000.0, 
                                  market_data=market_df,
                                  asset_symbol=ASSET_ID,
                                  default_leverage=5.0,
                                  maintenance_margin_pct=0.02, 
                                  commission_pct_per_trade=0.0005,
                                  liquidation_slippage_pct=0.005,
                                  funding_fee_check_interval_hours=8,
                                  risk_manager_config=risk_conf
                                  ) 
    
    sim_contract_multipliers = {ASSET_ID: CONTRACT_MULT}
    simulator.run_simulation(signals_df, leverage_series=leverage_s, contract_multipliers=sim_contract_multipliers)

    print("\nPortfolio History:")
    portfolio_hist_df = simulator.get_portfolio_history()
    print(portfolio_hist_df)

    print("\nTrades History:")
    trades_hist_df = simulator.get_trades_history()
    print(trades_hist_df)

    print("\nLiquidation Events:")
    liquidation_df = simulator.get_liquidation_events()
    print(liquidation_df)
    
    print("\nFunding Fee History:")
    funding_df = simulator.get_funding_fee_history()
    print(funding_df)

    print("\n--- Test Marge Insuffisante & Liquidation ---")
    market_liq_data = {
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 10:00:00', '2023-01-01 11:00:00', '2023-01-01 12:00:00']),
        'open': [100.0, 90.0, 80.0, 70.0], 
        'high': [100.0, 90.0, 80.0, 70.0],
        'low': [100.0, 90.0, 80.0, 70.0],
        'close': [100.0, 90.0, 80.0, 70.0],
        'funding_rate': [0,0,0,0]
    }
    market_liq_df = pd.DataFrame(market_liq_data).set_index('timestamp')
    signals_liq_data = {
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00']),
        'signal': ['BUY'],
        'nominal_value_to_trade': [1000.0] 
    }
    signals_liq_df = pd.DataFrame(signals_liq_data).set_index('timestamp')
        
    simulator_liq = BacktestSimulator(initial_capital=100.0, 
                                      market_data=market_liq_df, 
                                      asset_symbol=ASSET_ID, 
                                      default_leverage=10.0, 
                                      maintenance_margin_pct=0.05, 
                                      commission_pct_per_trade=0.0 
                                     )
    simulator_liq.run_simulation(signals_liq_df, contract_multipliers={ASSET_ID: 1.0}) 
    
    print("\nPortfolio History (Liquidation Test):")
    print(simulator_liq.get_portfolio_history())
    print("\nTrades History (Liquidation Test):")
    print(simulator_liq.get_trades_history()) 
    print("\nLiquidation Events (Liquidation Test):")
    print(simulator_liq.get_liquidation_events())

    print("\n--- Fin des tests ---")