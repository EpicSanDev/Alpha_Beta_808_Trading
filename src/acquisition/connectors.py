import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import numpy as np

# Import optionnel de binance
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    Client = None
    BINANCE_AVAILABLE = False

class BinanceConnector:
    """
    Connecteur pour interagir avec l'API Binance.
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        Initialise le connecteur Binance.

        Args:
            api_key (str): Clé API Binance.
            api_secret (str): Clé secrète API Binance.
            testnet (bool): True pour utiliser l'environnement de test Binance, False sinon.
        """
        if not BINANCE_AVAILABLE:
            raise ImportError("La librairie python-binance n'est pas installée. Veuillez l'installer avec 'pip install python-binance'.")
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

    def get_klines(self, symbol: str, intervals: List[str], start_date_str: str, end_date_str: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Charge les données de klines historiques depuis Binance pour plusieurs intervalles.

        Args:
            symbol (str): La paire de trading (ex: 'BTCUSDT').
            intervals (List[str]): Liste des intervalles de klines (ex: ['1d', '4h', '30m']).
            start_date_str (str): Date de début pour l'historique (format 'YYYY-MM-DD' ou 'X days/months/years ago').
            end_date_str (Optional[str]): Date de fin pour l'historique (format 'YYYY-MM-DD').
                                          Si None, utilise la date actuelle.

        Returns:
            Dict[str, pd.DataFrame]: Un dictionnaire où les clés sont les intervalles et
                                     les valeurs sont les DataFrames correspondants.
        """
        return load_binance_klines(
            api_key=self.api_key,
            api_secret=self.api_secret,
            symbol=symbol,
            intervals=intervals,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            testnet=self.testnet
        )

    def get_balance(self, specific_assets: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Récupère les soldes des actifs spécifiés depuis Binance.

        Args:
            specific_assets (Optional[List[str]]): Une liste d'actifs à récupérer (ex: ['USDC', 'USDT']).
                                                 Si None, tente de récupérer les stablecoins communs.

        Returns:
            Dict[str, float]: Un dictionnaire avec les actifs comme clés et leurs soldes libres comme valeurs.
        """
        return get_binance_balance(
            api_key=self.api_key,
            api_secret=self.api_secret,
            specific_assets=specific_assets,
            testnet=self.testnet
        )

def load_csv_data(
    file_path: str,
    columns: Optional[List[str]] = None,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        file_path (str): Le chemin vers le fichier CSV.
        columns (Optional[List[str]]): Liste des colonnes à charger.
                                         Si None, toutes les colonnes sont chargées.
                                         Par défaut ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        timestamp_col (str): Le nom de la colonne contenant les timestamps.
                             Cette colonne sera convertie en datetime.

    Returns:
        pd.DataFrame: Un DataFrame Pandas avec les données chargées.

    Raises:
        FileNotFoundError: Si le fichier spécifié n'est pas trouvé.
        ValueError: Si la colonne timestamp n'est pas trouvée ou ne peut pas être convertie.
    """
    if columns is None:
        columns_to_use = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    else:
        columns_to_use = columns

    try:
        data = pd.read_csv(file_path, usecols=lambda c: c in columns_to_use if columns_to_use else True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Le fichier {file_path} n'a pas été trouvé.")
    except ValueError as e:
        # Pandas peut lever un ValueError si usecols ne correspond à rien
        raise ValueError(f"Erreur lors de la lecture des colonnes du CSV : {e}")


    if timestamp_col not in data.columns:
        raise ValueError(f"La colonne timestamp '{timestamp_col}' n'est pas présente dans le fichier CSV.")

    try:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    except Exception as e:
        raise ValueError(f"Impossible de convertir la colonne '{timestamp_col}' en datetime: {e}")

    # S'assurer que les colonnes numériques attendues sont bien numériques
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce') # 'coerce' met NaN pour les erreurs

    return data

def generate_random_market_data(
    num_rows: int = 1000,
    start_price: float = 100.0,
    volatility: float = 0.01,
    start_date: str = "2023-01-01",
    freq: str = "T"  # 'T' pour minute, 'H' pour heure, 'D' pour jour
) -> pd.DataFrame:
    """
    Génère des données de marché OHLCV aléatoires mais réalistes.

    Args:
        num_rows (int): Nombre de lignes de données à générer.
        start_price (float): Prix de départ pour la simulation.
        volatility (float): Volatilité utilisée pour simuler les variations de prix.
        start_date (str): Date de début pour les timestamps.
        freq (str): Fréquence des barres de données (ex: 'T' pour minute, 'H', 'D').

    Returns:
        pd.DataFrame: DataFrame avec les colonnes 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
    """
    import numpy as np

    dates = pd.to_datetime(start_date) + pd.to_timedelta(np.arange(num_rows), unit=freq)
    prices = np.zeros(num_rows)
    prices[0] = start_price

    # Simuler les prix de clôture avec un mouvement brownien géométrique simple
    for i in range(1, num_rows):
        prices[i] = prices[i-1] * np.exp(np.random.normal(0, volatility))

    # Générer open, high, low à partir de close
    open_prices = np.roll(prices, 1) # open est le close précédent
    open_prices[0] = start_price # Ajuster le premier open

    high_prices = np.maximum(open_prices, prices) + np.abs(np.random.normal(0, volatility/2, num_rows)) * prices
    low_prices = np.minimum(open_prices, prices) - np.abs(np.random.normal(0, volatility/2, num_rows)) * prices

    for i in range(num_rows):
        current_open = open_prices[i]
        current_close = prices[i]
        min_oc = min(current_open, current_close)
        max_oc = max(current_open, current_close)
        if low_prices[i] > min_oc:
            low_prices[i] = min_oc - np.abs(np.random.normal(0, volatility/10)) * prices[i]
        if high_prices[i] < max_oc:
            high_prices[i] = max_oc + np.abs(np.random.normal(0, volatility/10)) * prices[i]
        if low_prices[i] > high_prices[i]:
            low_prices[i], high_prices[i] = high_prices[i], low_prices[i]

    volume = np.random.randint(100, 5000, num_rows)

    data = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': prices,
        'volume': volume
    })
    return data

def load_binance_klines(
    api_key: str,
    api_secret: str,
    symbol: str,
    intervals: List[str],
    start_date_str: str,
    end_date_str: Optional[str] = None,
    testnet: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Charge les données de klines historiques depuis Binance pour plusieurs intervalles.

    Args:
        api_key (str): Votre clé API Binance.
        api_secret (str): Votre clé secrète API Binance.
        symbol (str): La paire de trading (ex: 'BTCUSDC').
        intervals (List[str]): Liste des intervalles de klines (ex: ['1d', '4h', '30m']).
                                 Format Binance: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.
        start_date_str (str): Date de début pour l'historique (format 'YYYY-MM-DD' ou 'X days/months/years ago').
        end_date_str (Optional[str]): Date de fin pour l'historique (format 'YYYY-MM-DD'). 
                                      Si None, utilise la date actuelle.
        testnet (bool): Utiliser le testnet Binance (défaut: False).

    Returns:
        Dict[str, pd.DataFrame]: Un dictionnaire où les clés sont les intervalles et
                                 les valeurs sont les DataFrames correspondants avec les colonnes
                                 ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
                                 Retourne un dictionnaire vide si une erreur survient.
    """
    client = Client(api_key, api_secret, testnet=testnet)
    all_klines_data = {}

    # Convertir start_date_str en un format utilisable par Binance
    # La librairie python-binance gère bien "X days/months/years ago"
    
    # Si end_date_str n'est pas fourni, utiliser la date actuelle
    if end_date_str is None:
        end_date_dt = datetime.utcnow()
    else:
        try:
            end_date_dt = datetime.strptime(end_date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Format de end_date_str invalide: {end_date_str}. Utilisation de la date actuelle.")
            end_date_dt = datetime.utcnow()

    print(f"Chargement des données pour {symbol} depuis {start_date_str} jusqu'à {end_date_dt.strftime('%Y-%m-%d')}")

    for interval_str in intervals:
        print(f"  Chargement de l'intervalle: {interval_str}...")
        try:
            # La fonction get_historical_klines prend start_str et end_str
            # Convertir end_date_dt en string pour l'API
            end_str_api = end_date_dt.strftime("%d %b, %Y %H:%M:%S")

            klines = client.get_historical_klines(symbol, interval_str, start_date_str, end_str=end_str_api)
            
            if not klines:
                print(f"    Aucune donnée retournée pour l'intervalle {interval_str}.")
                continue

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Sélectionner et convertir les colonnes nécessaires
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            
            df['symbol'] = symbol # Ajout de la colonne symbol
            df['interval'] = interval_str # Ajout de la colonne interval pour référence future si besoin

            all_klines_data[interval_str] = df
            print(f"    Données pour l'intervalle {interval_str} chargées: {len(df)} lignes.")

        except Exception as e:
            print(f"Erreur lors du chargement des données pour l'intervalle {interval_str} de {symbol}: {e}")
            # Continuer avec les autres intervalles même si un échoue
            continue
            
    return all_klines_data

def get_binance_balance(
    api_key: str,
    api_secret: str,
    specific_assets: Optional[List[str]] = None,
    testnet: bool = False
) -> Dict[str, float]:
    """
    Récupère les soldes des actifs spécifiés depuis Binance.

    Args:
        api_key (str): Votre clé API Binance.
        api_secret (str): Votre clé secrète API Binance.
        specific_assets (Optional[List[str]]): Une liste d'actifs à récupérer (ex: ['USDC', 'USDT']).
                                             Si None, tente de récupérer les stablecoins communs.
        testnet (bool): Utiliser le testnet Binance (défaut: False).

    Returns:
        Dict[str, float]: Un dictionnaire avec les actifs comme clés et leurs soldes libres comme valeurs.
                          Retourne un dictionnaire vide si une erreur survient ou si BINANCE_AVAILABLE est False.
    """
    if not BINANCE_AVAILABLE:
        print("La librairie Binance n'est pas installée. Impossible de récupérer les soldes.")
        return {}

    client = Client(api_key, api_secret, testnet=testnet)
    balances = {}
    
    try:
        account_info = client.get_account()
        
        if 'balances' not in account_info:
            print("Impossible de trouver les informations de solde dans la réponse de l'API.")
            return {}

        assets_to_check = specific_assets
        if assets_to_check is None:
            # Liste par défaut de stablecoins si specific_assets n'est pas fourni
            assets_to_check = ['USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX'] # Ajoutez d'autres stablecoins si besoin

        for asset_balance in account_info['balances']:
            asset_code = asset_balance['asset']
            if asset_code in assets_to_check:
                free_balance = float(asset_balance['free'])
                if free_balance > 0: # On ne stocke que les soldes non nuls
                    balances[asset_code] = free_balance
        
        if not balances:
            print(f"Aucun solde trouvé pour les actifs spécifiés: {assets_to_check}")

    except Exception as e:
        print(f"Erreur lors de la récupération des soldes Binance: {e}")
        return {}
        
    return balances

if __name__ == '__main__':
    # Exemple d'utilisation pour load_csv_data (décommenter et créer sample_data.csv pour tester)
    # try:
    #     sample_df = pd.DataFrame({
    #         'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00']),
    #         'open': [100, 101, 100.5], 'high': [102, 103, 102.5], 'low': [99, 100, 99.5],
    #         'close': [101, 102, 101.5], 'volume': [1000, 1200, 1100], 'other_col': [1,2,3]
    #     })
    #     sample_df.to_csv('sample_data.csv', index=False)
    #     loaded_data_default = load_csv_data('sample_data.csv')
    #     print("\\nDonnées CSV chargées (défaut):\n", loaded_data_default.head())
    # except Exception as e:
    #     print(f"Erreur test load_csv_data: {e}")

    # Exemple d'utilisation pour generate_random_market_data
    # try:
    #     random_data = generate_random_market_data(num_rows=5, freq='H')
    #     print("\\nDonnées aléatoires générées:\n", random_data.head())
    # except Exception as e:
    #     print(f"Erreur test generate_random_market_data: {e}")

    # Exemple d'utilisation pour load_binance_klines
    # ATTENTION: Nécessite des clés API valides dans les variables d'environnement BINANCE_API_KEY et BINANCE_API_SECRET
    # ou de les passer directement. Pour un test public, on peut utiliser une instance de Client sans clés
    # mais get_historical_klines pourrait ne pas fonctionner ou être limité.
    # Pour un test réel, assurez-vous que vos clés sont configurées.
    print("\\n--- Test de load_binance_klines ---")
    # Note: Pour exécuter ce test, vous devez avoir vos clés API Binance configurées
    # comme variables d'environnement BINANCE_API_KEY et BINANCE_API_SECRET
    # ou les fournir directement.
    import os
    from dotenv import load_dotenv
    load_dotenv() # Charge les variables depuis .env s'il existe

    api_key_test = os.getenv('BINANCE_API_KEY_TEST') # Utilisez des clés de test si possible
    api_secret_test = os.getenv('BINANCE_API_SECRET_TEST')

    if api_key_test and api_secret_test:
        try:
            symbol_to_test = 'BTCUSDT' # Binance utilise USDT, pas USDC pour les paires majeures souvent
            # Vérifiez la disponibilité de BTCUSDC ou adaptez à BTCUSDC si c'est ce que vous avez.
            # Pour l'exemple, je vais utiliser BTCUSDT qui est plus commun.
            # Si vous voulez spécifiquement BTCUSDC, assurez-vous que la paire existe et a de l'historique.
            
            intervals_to_test = ['1d', '4h'] # Test avec moins d'intervalles pour la rapidité
            # start_date_test = "30 days ago" # Test avec un historique plus court
            three_years_ago = (datetime.utcnow() - timedelta(days=3*365)).strftime("%Y-%m-%d")
            
            print(f"Test avec le symbole {symbol_to_test}, intervalles {intervals_to_test}, début: {three_years_ago}")

            # binance_data = load_binance_klines(
            #     api_key=api_key_test,
            #     api_secret=api_secret_test,
            #     symbol=symbol_to_test,
            #     intervals=intervals_to_test,
            #     start_date_str=three_years_ago # Test avec 3 ans d'historique
            # )

            # if binance_data:
            #     for interval, df_interval in binance_data.items():
            #         print(f"\\nDonnées pour l'intervalle {interval} (premières 3 lignes):")
            #         print(df_interval.head(3))
            #         print(f"Données pour l'intervalle {interval} (dernières 3 lignes):")
            #         print(df_interval.tail(3))
            #         print(f"Shape: {df_interval.shape}")
            #         print(df_interval.dtypes)
            # else:
            #     print("Aucune donnée Binance n'a été chargée.")
            pass # Commenté pour éviter l'exécution automatique sans clés valides

        except Exception as e:
            print(f"Une erreur est survenue lors du test de load_binance_klines: {e}")
    else:
        print("Clés API Binance de test (BINANCE_API_KEY_TEST, BINANCE_API_SECRET_TEST) non trouvées dans les variables d'environnement.")
        print("Le test de load_binance_klines ne sera pas exécuté.")
    pass
