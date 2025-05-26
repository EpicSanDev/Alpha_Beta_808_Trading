import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

class BacktestSimulator:
    """
    Simulateur simple pour le backtesting de stratégies de trading.

    Ce simulateur prend des signaux (ACHETER, VENDRE, CONSERVER) et des tailles de position,
    et simule leur exécution sur des données de marché historiques. Il calcule les
    profits/pertes et suit l'évolution du capital.
    """

    def __init__(self, initial_capital: float, market_data: pd.DataFrame):
        """
        Initialise le BacktestSimulator.

        Args:
            initial_capital (float): Le capital de départ pour la simulation.
            market_data (pd.DataFrame): Un DataFrame Pandas contenant les données de marché
                                        historiques. Doit contenir les colonnes 'timestamp',
                                        'open', 'high', 'low', 'close'.
                                        'timestamp' doit être l'index du DataFrame.
        """
        if not isinstance(market_data.index, pd.DatetimeIndex):
            raise ValueError("L'index de market_data doit être de type DatetimeIndex (timestamp).")
        if not all(col in market_data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("market_data doit contenir les colonnes 'open', 'high', 'low', 'close'.")

        self.initial_capital: float = initial_capital
        self.market_data: pd.DataFrame = market_data.sort_index() # Assurer que les données sont triées par date

        self.cash: float = initial_capital
        self.current_shares: float = 0.0
        self.portfolio_value: float = initial_capital
        self.last_price_seen: float = 0.0 # Utilisé pour évaluer le portefeuille si aucune transaction n'est faite

        self.trades_history: List[Dict[str, Any]] = []
        self.portfolio_history: List[Dict[str, Any]] = [{
            'timestamp': self.market_data.index.min() - pd.Timedelta(days=1) if not self.market_data.empty else pd.Timestamp('now'), # Avant le début
            'cash': self.cash,
            'current_shares': self.current_shares,
            'portfolio_value': self.portfolio_value,
            'market_price': 0 # Pas de prix de marché avant le début
        }]

    def _update_portfolio_value(self, timestamp: pd.Timestamp, current_price: float):
        """Met à jour la valeur du portefeuille."""
        self.portfolio_value = self.cash + (self.current_shares * current_price)
        self.last_price_seen = current_price
        self.portfolio_history.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'current_shares': self.current_shares,
            'portfolio_value': self.portfolio_value,
            'market_price': current_price
        })

    def execute_trade(self, timestamp: pd.Timestamp, signal: str, position_size: float, current_price: float) -> None:
        """
        Exécute un trade basé sur le signal et la taille de position.

        Args:
            timestamp (pd.Timestamp): Le moment du trade.
            signal (str): Le signal de trading ('BUY', 'SELL', 'HOLD').
            position_size (float): La quantité de capital à allouer (pour 'BUY'/'SELL').
                                   Pour 'SELL', cela représente la fraction des actions détenues à vendre,
                                   ou le nombre d'actions à shorter si la gestion du short est implémentée.
                                   Pour le MVP, nous vendons une fraction des actions détenues.
            current_price (float): Le prix d'exécution supposé.

        Returns:
            None
        """
        if signal not in ['BUY', 'SELL', 'HOLD']:
            print(f"Warning: Signal '{signal}' non reconnu à {timestamp}. Trade ignoré.")
            self._update_portfolio_value(timestamp, current_price) # Mettre à jour la valeur du portefeuille même si pas de trade
            return

        trade_executed = False
        if signal == 'BUY':
            if self.cash <= 0:
                print(f"Warning: Cash insuffisant pour acheter à {timestamp}. Trade ignoré.")
            elif position_size <= 0:
                 print(f"Warning: Taille de position non positive pour BUY à {timestamp}. Trade ignoré.")
            else:
                shares_to_buy = position_size / current_price
                cost = shares_to_buy * current_price
                if cost > self.cash: # Ajuster si la position_size est trop grande
                    shares_to_buy = self.cash / current_price
                    cost = self.cash # Utiliser tout le cash restant
                    if shares_to_buy == 0: # Si le cash est trop faible pour acheter même une fraction
                        print(f"Warning: Cash ({self.cash}) trop faible pour acheter des actions à {current_price} à {timestamp}. Trade ignoré.")
                        self._update_portfolio_value(timestamp, current_price)
                        return

                self.cash -= cost
                self.current_shares += shares_to_buy
                self.trades_history.append({
                    'timestamp': timestamp,
                    'type': 'BUY',
                    'price': current_price,
                    'quantity': shares_to_buy,
                    'cost': cost
                })
                trade_executed = True
                # print(f"BUY: {shares_to_buy:.4f} shares at {current_price:.2f} on {timestamp}. Cash: {self.cash:.2f}, Shares: {self.current_shares:.4f}")

        elif signal == 'SELL':
            if self.current_shares <= 0:
                print(f"Warning: Aucune action à vendre à {timestamp}. Trade ignoré.")
            elif position_size <= 0:
                 print(f"Warning: Taille de position non positive pour SELL à {timestamp}. Trade ignoré.")
            else:
                # Si position_size <= 1, on l'interprète comme une fraction des actions détenues
                # Si position_size > 1, on l'interprète comme un nombre d'actions à vendre
                if position_size <= 1:
                    shares_to_sell = self.current_shares * position_size
                else:
                    shares_to_sell = min(position_size, self.current_shares)
                
                shares_to_sell = min(shares_to_sell, self.current_shares)
                
                if shares_to_sell == 0:
                    print(f"Warning: Tentative de vendre 0 actions à {timestamp}. Trade ignoré.")
                    self._update_portfolio_value(timestamp, current_price)
                    return

                proceeds = shares_to_sell * current_price
                self.cash += proceeds
                self.current_shares -= shares_to_sell
                self.trades_history.append({
                    'timestamp': timestamp,
                    'type': 'SELL',
                    'price': current_price,
                    'quantity': shares_to_sell,
                    'proceeds': proceeds
                })
                trade_executed = True
                # print(f"SELL: {shares_to_sell:.4f} shares at {current_price:.2f} on {timestamp}. Cash: {self.cash:.2f}, Shares: {self.current_shares:.4f}")

        self._update_portfolio_value(timestamp, current_price)

    def run_simulation(self, signals_df: pd.DataFrame) -> None:
        """
        Exécute la simulation de backtesting sur une série de signaux.

        Args:
            signals_df (pd.DataFrame): DataFrame contenant les signaux de trading.
                                       Doit avoir les colonnes 'timestamp', 'signal',
                                       et 'position_to_allocate'.
                                       'timestamp' doit être l'index du DataFrame.
        Returns:
            None
        """
        if not isinstance(signals_df.index, pd.DatetimeIndex):
            raise ValueError("L'index de signals_df doit être de type DatetimeIndex (timestamp).")
        if not all(col in signals_df.columns for col in ['signal', 'position_to_allocate']):
            raise ValueError("signals_df doit contenir les colonnes 'signal' et 'position_to_allocate'.")

        # Fusionner les signaux avec les données de marché pour obtenir les prix d'exécution
        # S'assurer que les deux DataFrames ont un index DatetimeIndex nommé 'timestamp' ou sont alignés
        merged_data = pd.merge_asof(signals_df.sort_index(),
                                    self.market_data[['open']], # Utiliser 'open' comme prix d'exécution
                                    left_index=True,
                                    right_index=True,
                                    direction='forward', # Prendre le prochain 'open' disponible si le timestamp exact du signal n'est pas dans market_data
                                    tolerance=pd.Timedelta(days=1)) # Tolérance pour la fusion

        # Remplir les prix manquants (si un signal est en dehors des heures de marché ou un jour férié)
        # avec le dernier prix 'open' connu (ffill) puis le premier (bfill)
        merged_data['open'] = merged_data['open'].ffill().bfill()


        # S'assurer que le premier enregistrement de portfolio_history correspond au premier signal
        # ou à la première donnée de marché si aucun signal n'est donné.
        first_event_timestamp = signals_df.index.min() if not signals_df.empty else self.market_data.index.min()
        if not self.market_data.empty and first_event_timestamp >= self.market_data.index.min():
            initial_market_price = self.market_data.loc[self.market_data.index.min(), 'open']
            self.portfolio_history = [{
                'timestamp': self.market_data.index.min() - pd.Timedelta(microseconds=1), # Juste avant le premier tick
                'cash': self.initial_capital,
                'current_shares': 0.0,
                'portfolio_value': self.initial_capital,
                'market_price': initial_market_price # ou 0 si on préfère
            }]
            self.last_price_seen = initial_market_price


        # Itérer sur les timestamps uniques des données de marché pour mettre à jour la valeur du portefeuille quotidiennement
        # même les jours sans trade.
        all_relevant_timestamps = sorted(list(set(self.market_data.index.tolist() + signals_df.index.tolist())))
        all_relevant_timestamps = [ts for ts in all_relevant_timestamps if ts >= self.market_data.index.min() and ts <= self.market_data.index.max()]


        last_known_price = self.market_data['open'].iloc[0] if not self.market_data.empty else 0

        for current_ts in all_relevant_timestamps:
            current_market_price = self.market_data.loc[current_ts, 'open'] if current_ts in self.market_data.index else last_known_price
            last_known_price = current_market_price # Mettre à jour le dernier prix connu

            if current_ts in merged_data.index:
                signal_row = merged_data.loc[current_ts]
                # Si plusieurs signaux au même timestamp, prendre le premier (ou gérer autrement)
                if isinstance(signal_row, pd.DataFrame):
                    signal_row = signal_row.iloc[0]

                signal = signal_row['signal']
                position_to_allocate = signal_row['position_to_allocate'] # Pour BUY, c'est le montant en capital. Pour SELL, le nombre d'actions.
                execution_price = signal_row['open'] # Prix d'exécution du jour du signal

                if pd.isna(execution_price):
                    # Essayer de trouver un prix valide si celui du jour est NaN (ex: jour férié pour le signal)
                    # On pourrait prendre le 'close' précédent ou le 'open' suivant.
                    # Pour l'instant, on logue un warning et on utilise le dernier prix connu.
                    print(f"Warning: Prix d'exécution non disponible pour le signal à {current_ts}. Utilisation du dernier prix connu: {last_known_price}.")
                    execution_price = last_known_price # Fallback

                if signal != 'HOLD':
                    self.execute_trade(current_ts, signal, position_to_allocate, execution_price)
                else:
                    # Même pour HOLD, mettre à jour la valeur du portefeuille avec le prix actuel du marché
                    self._update_portfolio_value(current_ts, current_market_price)
            else:
                # Pas de signal à ce timestamp, mais il est dans market_data. Mettre à jour la valeur du portefeuille.
                 self._update_portfolio_value(current_ts, current_market_price)

        # Assurer que le dernier état du portefeuille est enregistré
        if not self.market_data.empty and (not self.portfolio_history or self.portfolio_history[-1]['timestamp'] < self.market_data.index.max()):
             final_price = self.market_data['close'].iloc[-1] # Utiliser le dernier 'close' pour la valorisation finale
             self._update_portfolio_value(self.market_data.index.max(), final_price)


    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Retourne l'historique de la valeur du portefeuille.

        Returns:
            pd.DataFrame: Un DataFrame avec l'historique du portefeuille
                          (timestamp, cash, current_shares, portfolio_value, market_price).
        """
        if not self.portfolio_history:
            return pd.DataFrame(columns=['timestamp', 'cash', 'current_shares', 'portfolio_value', 'market_price'])
        df = pd.DataFrame(self.portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        # Supprimer les doublons de timestamp en gardant la dernière entrée, ce qui est important
        # si _update_portfolio_value est appelé plusieurs fois pour le même timestamp (ex: trade + update général)
        df = df[~df.index.duplicated(keep='last')]
        return df

    def get_trades_history(self) -> pd.DataFrame:
        """
        Retourne l'historique des trades exécutés.

        Returns:
            pd.DataFrame: Un DataFrame avec l'historique des trades
                          (timestamp, type, price, quantity, cost/proceeds).
        """
        if not self.trades_history:
            return pd.DataFrame(columns=['timestamp', 'type', 'price', 'quantity', 'cost', 'proceeds'])
        df = pd.DataFrame(self.trades_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df

if __name__ == '__main__':
    # Exemple d'utilisation (pourrait être dans des tests unitaires)
    # Créer des données de marché factices
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-02 09:00:00', '2023-01-03 09:00:00',
                                      '2023-01-04 09:00:00', '2023-01-05 09:00:00', '2023-01-06 09:00:00']),
        'open': [100.0, 102.0, 101.0, 103.0, 105.0, 104.0],
        'high': [103.0, 104.0, 102.5, 105.0, 106.0, 104.5],
        'low': [99.0, 101.0, 100.0, 102.0, 103.5, 103.0],
        'close': [102.0, 101.5, 102.0, 104.5, 104.0, 103.5]
    }
    market_df = pd.DataFrame(data).set_index('timestamp')

    # Créer des signaux factices
    signals_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-03 11:00:00', '2023-01-05 09:30:00']),
        'signal': ['BUY', 'SELL', 'BUY'],
        # Pour BUY: montant de capital à allouer. Pour SELL: nombre d'actions à vendre.
        'position_to_allocate': [5000.0, 20.0, 3000.0] # Acheter pour 5000, Vendre 20 actions, Acheter pour 3000
    }
    signals_df = pd.DataFrame(signals_data).set_index('timestamp')

    print("Market Data:")
    print(market_df)
    print("\nSignals Data:")
    print(signals_df)

    simulator = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator.run_simulation(signals_df)

    print("\nPortfolio History:")
    portfolio_hist_df = simulator.get_portfolio_history()
    print(portfolio_hist_df)

    print("\nTrades History:")
    trades_hist_df = simulator.get_trades_history()
    print(trades_hist_df)

    # Test avec des signaux vides
    print("\n--- Test avec signaux vides ---")
    empty_signals_df = pd.DataFrame(columns=['signal', 'position_to_allocate']).set_index(pd.to_datetime([]))
    empty_signals_df.index.name = 'timestamp'

    simulator_empty = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_empty.run_simulation(empty_signals_df)
    print("\nPortfolio History (empty signals):")
    print(simulator_empty.get_portfolio_history())
    print("\nTrades History (empty signals):")
    print(simulator_empty.get_trades_history())


    # Test avec données de marché vides
    print("\n--- Test avec données de marché vides ---")
    empty_market_df = pd.DataFrame(columns=['open', 'high', 'low', 'close']).set_index(pd.to_datetime([]))
    empty_market_df.index.name = 'timestamp'
    try:
        simulator_empty_market = BacktestSimulator(initial_capital=10000.0, market_data=empty_market_df.copy()) # Copie pour éviter modif
        # run_simulation va probablement échouer ou ne rien faire d'utile, mais le constructeur doit gérer
        simulator_empty_market.run_simulation(signals_df) # Devrait loguer des erreurs ou ne rien faire
        print("\nPortfolio History (empty market):")
        print(simulator_empty_market.get_portfolio_history())
        print("\nTrades History (empty market):")
        print(simulator_empty_market.get_trades_history())
    except Exception as e:
        print(f"Erreur attendue avec données de marché vides: {e}")

    # Test avec un signal BUY mais pas assez de cash
    print("\n--- Test BUY avec cash insuffisant ---")
    signals_low_cash_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00']),
        'signal': ['BUY'],
        'position_to_allocate': [15000.0] # Plus que le capital initial
    }
    signals_low_cash_df = pd.DataFrame(signals_low_cash_data).set_index('timestamp')
    simulator_low_cash = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_low_cash.run_simulation(signals_low_cash_df)
    print("\nPortfolio History (low cash BUY):")
    print(simulator_low_cash.get_portfolio_history())
    print("\nTrades History (low cash BUY):") # Devrait montrer un achat partiel ou aucun achat
    print(simulator_low_cash.get_trades_history())


    # Test avec un signal SELL mais pas d'actions
    print("\n--- Test SELL sans actions ---")
    signals_no_shares_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00']),
        'signal': ['SELL'],
        'position_to_allocate': [10.0]
    }
    signals_no_shares_df = pd.DataFrame(signals_no_shares_data).set_index('timestamp')
    simulator_no_shares = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_no_shares.run_simulation(signals_no_shares_df)
    print("\nPortfolio History (no shares SELL):")
    print(simulator_no_shares.get_portfolio_history())
    print("\nTrades History (no shares SELL):") # Ne devrait montrer aucun trade
    print(simulator_no_shares.get_trades_history())

    # Test avec signal HOLD
    print("\n--- Test HOLD ---")
    signals_hold_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 10:00:00']),
        'signal': ['BUY', 'HOLD'],
        'position_to_allocate': [5000.0, 0.0]
    }
    signals_hold_df = pd.DataFrame(signals_hold_data).set_index('timestamp')
    simulator_hold = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_hold.run_simulation(signals_hold_df)
    print("\nPortfolio History (HOLD):")
    print(simulator_hold.get_portfolio_history())
    print("\nTrades History (HOLD):")
    print(simulator_hold.get_trades_history())

    # Test avec des timestamps de signaux qui ne sont pas dans market_data
    print("\n--- Test avec timestamps de signaux désalignés ---")
    signals_offset_ts_data = {
        'timestamp': pd.to_datetime(['2023-01-01 15:00:00', '2023-01-03 08:00:00']), # En dehors des heures de marché simulées
        'signal': ['BUY', 'SELL'],
        'position_to_allocate': [5000.0, 10.0]
    }
    signals_offset_ts_df = pd.DataFrame(signals_offset_ts_data).set_index('timestamp')
    simulator_offset_ts = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_offset_ts.run_simulation(signals_offset_ts_df)
    print("\nPortfolio History (offset timestamps):")
    print(simulator_offset_ts.get_portfolio_history()) # Devrait utiliser le 'open' suivant
    print("\nTrades History (offset timestamps):")
    print(simulator_offset_ts.get_trades_history())

    # Test avec un signal BUY où position_to_allocate est trop petit pour acheter une action
    print("\n--- Test BUY avec position_to_allocate trop petit ---")
    market_high_price_data = {
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00']),
        'open': [10000.0], 'high': [10000.0], 'low': [10000.0], 'close': [10000.0]
    }
    market_high_price_df = pd.DataFrame(market_high_price_data).set_index('timestamp')
    signals_tiny_alloc_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00']),
        'signal': ['BUY'],
        'position_to_allocate': [10.0] # Capital de 10, prix de 10000
    }
    signals_tiny_alloc_df = pd.DataFrame(signals_tiny_alloc_data).set_index('timestamp')
    simulator_tiny_alloc = BacktestSimulator(initial_capital=100.0, market_data=market_high_price_df)
    simulator_tiny_alloc.run_simulation(signals_tiny_alloc_df)
    print("\nPortfolio History (tiny allocation BUY):")
    print(simulator_tiny_alloc.get_portfolio_history())
    print("\nTrades History (tiny allocation BUY):") # Devrait montrer aucun trade
    print(simulator_tiny_alloc.get_trades_history())

    # Test avec un signal SELL où position_to_allocate est 0
    print("\n--- Test SELL avec position_to_allocate = 0 ---")
    signals_sell_zero_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 10:00:00']),
        'signal': ['BUY', 'SELL'],
        'position_to_allocate': [5000.0, 0.0] # Acheter, puis essayer de vendre 0 actions
    }
    signals_sell_zero_df = pd.DataFrame(signals_sell_zero_data).set_index('timestamp')
    simulator_sell_zero = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_sell_zero.run_simulation(signals_sell_zero_df)
    print("\nPortfolio History (SELL zero):")
    print(simulator_sell_zero.get_portfolio_history())
    print("\nTrades History (SELL zero):") # Le SELL ne devrait pas apparaître
    print(simulator_sell_zero.get_trades_history())

    # Test avec un signal BUY où position_to_allocate est 0
    print("\n--- Test BUY avec position_to_allocate = 0 ---")
    signals_buy_zero_data = {
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00']),
        'signal': ['BUY'],
        'position_to_allocate': [0.0] # Essayer d'acheter pour 0
    }
    signals_buy_zero_df = pd.DataFrame(signals_buy_zero_data).set_index('timestamp')
    simulator_buy_zero = BacktestSimulator(initial_capital=10000.0, market_data=market_df)
    simulator_buy_zero.run_simulation(signals_buy_zero_df)
    print("\nPortfolio History (BUY zero):")
    print(simulator_buy_zero.get_portfolio_history())
    print("\nTrades History (BUY zero):") # Le BUY ne devrait pas apparaître
    print(simulator_buy_zero.get_trades_history())

    print("\n--- Fin des tests ---")