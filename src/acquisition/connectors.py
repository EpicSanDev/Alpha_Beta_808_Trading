import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone
import numpy as np
import requests # Ajout pour BitgetConnector
import time # Ajout pour la gestion des limites de taux Bitget
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from ..core.database import SessionLocal, OpenInterest, FundingRate, MarkPrice # Ajout pour le stockage DB

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

class BitgetConnector:
    """
    Connecteur pour interagir avec l'API Bitget pour la récupération des données de marché.
    Documentation API Klines (Futures): Non fournie directement, mais basée sur l'exemple utilisateur:
    GET /api/mix/v1/market/candles
    Params: symbol, granularity, startTime, endTime, limit
    Réponse: [[timestamp, open, high, low, close, base_volume, quote_volume], ...]
    """
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, passphrase: Optional[str] = None):
        """
        Initialise le connecteur Bitget.

        Args:
            api_key (Optional[str]): Clé API Bitget. Non utilisé pour les klines publiques.
            api_secret (Optional[str]): Clé secrète API Bitget. Non utilisé pour les klines publiques.
            passphrase (Optional[str]): Passphrase Bitget. Non utilisé pour les klines publiques.
        """
        self.base_url = "https://api.bitget.com"
        self.api_key = api_key # Conservé pour une éventuelle utilisation future (endpoints privés)
        self.api_secret = api_secret # Conservé pour une éventuelle utilisation future
        self.passphrase = passphrase # Conservé pour une éventuelle utilisation future
        # Aucune initialisation de client spécifique nécessaire pour les appels HTTP simples avec `requests`.

    def get_klines(self, symbol: str, intervals: List[str], start_date_str: str, end_date_str: Optional[str] = None, limit_per_request: int = 100) -> Dict[str, pd.DataFrame]:
        """
        Charge les données de klines (OHLCV) historiques depuis l'API Bitget pour les contrats à terme.

        Gère la pagination pour récupérer les données sur la période demandée, en respectant
        les limites de l'API Bitget (ex: 1000 bougies par requête, contrainte de 30 jours pour '1m').

        Args:
            symbol (str): Le symbole du contrat à terme (ex: 'BTCUSDT_UMCBL').
                          Doit être en majuscules comme requis par l'API Bitget.
            intervals (List[str]): Liste des granularités/intervalles de klines (ex: ['1m', '5m', '1H', '1D']).
                                   Ces valeurs doivent correspondre à celles acceptées par l'API Bitget.
            start_date_str (str): Date de début pour l'historique.
                                  Peut être un format 'YYYY-MM-DD' (sera converti en UTC)
                                  ou un timestamp en millisecondes (chaîne de chiffres).
            end_date_str (Optional[str]): Date de fin pour l'historique.
                                          Mêmes formats que start_date_str.
                                          Si None, la date et l'heure actuelles (UTC) sont utilisées.
            limit_per_request (int): Nombre de bougies à demander par requête à l'API.
                                     Le défaut est 100, maximum 1000 selon la documentation Bitget.

        Returns:
            Dict[str, pd.DataFrame]: Un dictionnaire où chaque clé est une chaîne d'intervalle (ex: '1H')
                                     et chaque valeur est un DataFrame Pandas contenant les données OHLCV.
                                     Les colonnes du DataFrame sont:
                                     ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'interval'].
                                     'timestamp' est en UTC.
                                     Retourne un dictionnaire vide si une erreur majeure survient
                                     ou si aucune donnée n'est récupérée.
        
        Raises:
            Pas d'exceptions directes, les erreurs sont logguées et un dictionnaire vide est retourné.
        """
        all_klines_data = {}
        # Endpoint pour les klines de contrats à terme (mix product)
        api_endpoint = f"{self.base_url}/api/mix/v1/market/candles"

        try:
            # Convertir start_date_str en timestamp millisecondes UTC
            if start_date_str.isdigit():
                start_ts_ms = int(start_date_str)
            else:
                # Assumer que la date est en UTC si pas de fuseau horaire spécifié
                start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                start_ts_ms = int(start_dt.timestamp() * 1000)

            # Convertir end_date_str en timestamp millisecondes UTC
            if end_date_str:
                if end_date_str.isdigit():
                    end_ts_ms = int(end_date_str)
                else:
                    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    end_ts_ms = int(end_dt.timestamp() * 1000)
            else:
                # Si pas de date de fin, utiliser maintenant en UTC
                end_ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        except ValueError as e:
            print(f"Erreur de format de date pour Bitget: {e}. Utilisez 'YYYY-MM-DD' ou un timestamp en ms.")
            return {}

        for granularity in intervals:
            print(f"  Chargement des données Bitget pour {symbol}, intervalle: {granularity}...")
            klines_for_interval_list = [] # Liste pour accumuler les données de toutes les pages
            
            # Initialiser current_request_start_ts pour la première requête de cet intervalle
            current_request_start_ts = start_ts_ms

            # Contrainte spécifique de Bitget: "Could only get data within 30 days for '1m' data"
            # Cela signifie que la différence entre endTime et startTime dans une requête pour '1m'
            # ne doit pas excéder 30 jours. Notre boucle de pagination doit en tenir compte.
            # La pagination se fait en ajustant le 'startTime' de la requête suivante.
            # 'endTime' dans la requête peut rester le 'end_ts_ms' global.

            # Si la période totale demandée pour '1m' est > 30 jours, on doit la segmenter.
            # Cependant, l'API elle-même limite la plage par requête.
            # La pagination se fera en demandant des lots jusqu'à ce que end_ts_ms soit atteint.

            loop_safety_counter = 0 # Pour éviter les boucles infinies en cas de logique de pagination erronée
            MAX_LOOPS = 2000 # Environ 2000 * 1000 bougies = 2M bougies, devrait être suffisant

            while current_request_start_ts < end_ts_ms and loop_safety_counter < MAX_LOOPS:
                loop_safety_counter += 1
                
                # Déterminer le endTime pour cette requête spécifique pour respecter la contrainte de 30 jours pour '1m'
                current_request_end_ts = end_ts_ms
                if granularity == '1m':
                    thirty_days_ms = 30 * 24 * 60 * 60 * 1000
                    if current_request_start_ts + thirty_days_ms < end_ts_ms:
                        current_request_end_ts = current_request_start_ts + thirty_days_ms -1 # -1ms pour être inclusif

                params = {
                    'symbol': symbol.upper(), # L'API requiert des majuscules
                    'granularity': granularity,
                    'startTime': str(current_request_start_ts),
                    'endTime': str(current_request_end_ts),
                    'limit': str(min(limit_per_request, 1000)) # Max 1000 pour Bitget, défaut 100
                }
                
                try:
                    # print(f"    DEBUG Bitget Request: {api_endpoint} PARAMS: {params}") # Pour le débogage
                    response = requests.get(api_endpoint, params=params, timeout=15) # Timeout augmenté à 15s
                    response.raise_for_status()  # Lève une HTTPError pour les codes d'erreur HTTP
                    
                    raw_data_page = response.json() # Bitget retourne une liste de listes

                    if not raw_data_page or not isinstance(raw_data_page, list) or not all(isinstance(item, list) for item in raw_data_page):
                        # print(f"    Aucune donnée valide reçue de Bitget ou format inattendu pour {symbol} ({granularity}) avec startTime {current_request_start_ts}. Réponse: {raw_data_page}")
                        break # Sortir de la boucle de pagination pour cet intervalle

                    klines_for_interval_list.extend(raw_data_page)
                    
                    # Condition d'arrêt de la pagination:
                    # 1. Si moins de 'limit' bougies sont retournées, on a atteint la fin des données disponibles pour cette période.
                    # 2. Si le timestamp de la dernière bougie retournée est >= endTime de la requête.
                    if len(raw_data_page) < int(params['limit']):
                        break
                    
                    last_kline_ts_page = int(raw_data_page[-1][0])
                    if last_kline_ts_page >= current_request_end_ts : # On a atteint la fin de la fenêtre de cette requête
                        break

                    # Mettre à jour current_request_start_ts pour la prochaine requête de pagination.
                    # Le nouveau startTime sera le timestamp de la dernière bougie reçue + 1ms pour éviter les doublons.
                    current_request_start_ts = last_kline_ts_page + 1

                    # Respecter la limite de taux de l'API Bitget (ex: 20 requêtes/seconde)
                    time.sleep(0.05) # Pause de 50ms entre les requêtes

                except requests.exceptions.HTTPError as http_err:
                    print(f"    Erreur HTTP lors de l'appel à Bitget API pour {symbol} ({granularity}): {http_err}. Réponse: {response.text}")
                    break
                except requests.exceptions.RequestException as req_err:
                    print(f"    Erreur réseau lors de l'appel à Bitget API pour {symbol} ({granularity}): {req_err}")
                    break
                except ValueError as json_err: # Erreur de parsing JSON
                    print(f"    Erreur de parsing JSON de la réponse Bitget pour {symbol} ({granularity}): {json_err}. Réponse: {response.text if 'response' in locals() else 'N/A'}")
                    break
                except Exception as e: # Autres erreurs inattendues
                    print(f"    Erreur inattendue lors de la récupération des klines Bitget pour {symbol} ({granularity}): {e}")
                    break
            
            if loop_safety_counter >= MAX_LOOPS:
                print(f"    Avertissement: Limite de boucle de pagination atteinte pour {symbol} ({granularity}).")

            if klines_for_interval_list:
                # Convertir la liste de listes en DataFrame
                df = pd.DataFrame(klines_for_interval_list, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'base_volume', # Nommé 'volume' dans notre standard
                    'quote_volume'   # Ignoré pour l'instant pour correspondre au format BinanceConnector
                ])
                
                # Sélectionner et renommer les colonnes pour standardisation
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'base_volume']]
                df.rename(columns={'base_volume': 'volume'}, inplace=True)
                
                # Convertir les types de données
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), unit='ms', utc=True)
                
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # errors='coerce' mettra NaN si la conversion échoue
                
                # Supprimer les lignes où la conversion numérique a échoué pour les colonnes clés
                df.dropna(subset=numeric_cols, inplace=True)
                if df.empty:
                    print(f"    Toutes les lignes pour {symbol} ({granularity}) ont été supprimées après la conversion numérique et la suppression des NaN.")
                    continue


                df['symbol'] = symbol.upper()
                df['interval'] = granularity
                
                # Trier par timestamp et supprimer les doublons potentiels (si la pagination en a introduit)
                df.sort_values(by='timestamp', inplace=True)
                df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True) # Garder la première occurrence
                
                df.reset_index(drop=True, inplace=True)

                all_klines_data[granularity] = df
                print(f"    Données Bitget pour {symbol} ({granularity}) chargées et traitées: {len(df)} lignes.")
            else:
                print(f"    Aucune donnée Bitget n'a pu être chargée pour {symbol} ({granularity}).")

        if not all_klines_data:
            print(f"Aucune donnée n'a pu être récupérée pour {symbol} avec les intervalles {intervals} depuis Bitget.")
        return all_klines_data

    def _make_request(self, endpoint_path: str, params: Optional[Dict] = None, error_context: str = "") -> Optional[Dict]:
        """
        Méthode utilitaire pour effectuer des requêtes GET à l'API Bitget et gérer les erreurs communes.
        """
        full_url = f"{self.base_url}{endpoint_path}"
        retries = 3
        for attempt in range(retries):
            try:
                # print(f"DEBUG Bitget Request: {full_url} PARAMS: {params}") # Pour le débogage
                response = requests.get(full_url, params=params, timeout=15)
                response.raise_for_status()
                
                if not response.content:
                    print(f"Réponse vide de Bitget pour {error_context} à {full_url} avec params {params}.")
                    return None

                data = response.json()
                if data.get("code") != "00000":
                    print(f"Erreur API Bitget pour {error_context} ({data.get('code')}): {data.get('msg')}. URL: {full_url}, Params: {params}")
                    return None
                return data.get("data")
            
            except requests.exceptions.HTTPError as http_err:
                print(f"Erreur HTTP lors de l'appel à Bitget API pour {error_context}: {http_err}. URL: {full_url}, Params: {params}, Réponse: {response.text if 'response' in locals() else 'N/A'}")
                if attempt == retries - 1: return None
                time.sleep(2 ** attempt) 
            except requests.exceptions.RequestException as req_err:
                print(f"Erreur réseau lors de l'appel à Bitget API pour {error_context}: {req_err}. URL: {full_url}, Params: {params}")
                if attempt == retries - 1: return None
                time.sleep(2 ** attempt)
            except ValueError as json_err: 
                print(f"Erreur de parsing JSON de la réponse Bitget pour {error_context}: {json_err}. URL: {full_url}, Params: {params}, Réponse: {response.text if 'response' in locals() else 'N/A'}")
                return None 
            except Exception as e:
                print(f"Erreur inattendue lors de la requête Bitget pour {error_context}: {e}. URL: {full_url}, Params: {params}")
                if attempt == retries - 1: return None
                time.sleep(2 ** attempt)
        return None

    def get_open_interest(self, symbol: str, db_session: Optional[Session] = None) -> Optional[pd.DataFrame]:
        """
        Récupère les données d'Open Interest pour un symbole de contrat à terme et les stocke en BDD.

        Args:
            symbol (str): Le symbole du contrat à terme (ex: 'BTCUSDT_UMCBL').
            db_session (Optional[Session]): Session SQLAlchemy pour l'interaction avec la BDD.

        Returns:
            Optional[pd.DataFrame]: Un DataFrame avec les colonnes ['timestamp', 'symbol', 'open_interest']
                                     ou None si une erreur survient.
                                     'timestamp' est en UTC.
        """
        endpoint_path = "/api/mix/v1/market/open-interest"
        params = {'symbol': symbol.upper()}
        error_context = f"Open Interest pour {symbol}"
        
        data = self._make_request(endpoint_path, params, error_context)
        
        if data and isinstance(data, dict):
            try:
                df = pd.DataFrame([data])
                df.rename(columns={'amount': 'open_interest'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), unit='ms', utc=True)
                df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
                # L'API renvoie le symbole, donc on l'utilise directement.
                # df['symbol'] = symbol.upper() # Assurer la cohérence si l'API ne le renvoyait pas
                df['symbol'] = df['symbol'].astype(str)


                df = df[['timestamp', 'symbol', 'open_interest']]
                
                df.dropna(inplace=True)
                if df.empty:
                    print(f"Aucune donnée d'open interest valide après traitement pour {symbol}.")
                    return None

                if db_session:
                    for _, row in df.iterrows():
                        oi_entry = OpenInterest(
                            symbol=row['symbol'],
                            timestamp=row['timestamp'].to_pydatetime(), # Convertir Timestamp Pandas en datetime Python
                            value=row['open_interest']
                        )
                        try:
                            db_session.add(oi_entry)
                            db_session.commit()
                        except IntegrityError:
                            db_session.rollback()
                            # print(f"Doublon détecté pour OpenInterest: {row['symbol']} @ {row['timestamp']}. Non inséré.")
                        except Exception as e_db:
                            db_session.rollback()
                            print(f"Erreur DB lors de l'insertion de OpenInterest pour {row['symbol']}: {e_db}")
                return df
            except Exception as e:
                print(f"Erreur lors de la transformation ou stockage des données d'open interest pour {symbol}: {e}. Données brutes: {data}")
                return None
        else:
            print(f"Aucune donnée ou format de données incorrect reçu pour l'open interest de {symbol}.")
            return None

    def get_current_funding_rate(self, symbol: str, db_session: Optional[Session] = None) -> Optional[pd.DataFrame]:
        """
        Récupère le taux de financement actuel pour un symbole de contrat à terme.

        Args:
            symbol (str): Le symbole du contrat à terme (ex: 'BTCUSDT_UMCBL').

        Returns:
            Optional[pd.DataFrame]: Un DataFrame avec les colonnes ['timestamp', 'symbol', 'funding_rate']
                                     ou None si une erreur survient.
                                     'timestamp' est l'heure actuelle de la requête en UTC.
        """
        endpoint_path = "/api/mix/v1/market/current-fundRate"
        params = {'symbol': symbol.upper()}
        error_context = f"Taux de financement actuel pour {symbol}"
        
        data = self._make_request(endpoint_path, params, error_context)
        
        if data and isinstance(data, dict):
            try:
                df = pd.DataFrame([data])
                df.rename(columns={'fundingRate': 'funding_rate'}, inplace=True)
                # Le timestamp du funding rate actuel est celui de la requête, car l'API ne le fournit pas.
                # Cependant, pour l'historique, l'API fournit 'settleTime'.
                # Pour la cohérence du stockage, il est préférable d'utiliser le 'settleTime' si disponible
                # ou un timestamp de la donnée elle-même. Ici, c'est le "current", donc now() est ok.
                current_api_timestamp = pd.to_datetime(data.get('timestamp'), unit='ms', utc=True) if data.get('timestamp') else pd.Timestamp.now(tz='UTC').round('ms')
                df['timestamp'] = current_api_timestamp
                df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
                df['symbol'] = df['symbol'].astype(str)

                df = df[['timestamp', 'symbol', 'funding_rate']]
                
                df.dropna(inplace=True)
                if df.empty:
                    print(f"Aucune donnée de taux de financement actuel valide après traitement pour {symbol}.")
                    return None

                if db_session:
                    for _, row in df.iterrows():
                        fr_entry = FundingRate(
                            symbol=row['symbol'],
                            timestamp=row['timestamp'].to_pydatetime(),
                            value=row['funding_rate']
                        )
                        try:
                            db_session.add(fr_entry)
                            db_session.commit()
                        except IntegrityError:
                            db_session.rollback()
                            # print(f"Doublon détecté pour FundingRate (current): {row['symbol']} @ {row['timestamp']}. Non inséré.")
                        except Exception as e_db:
                            db_session.rollback()
                            print(f"Erreur DB lors de l'insertion de FundingRate (current) pour {row['symbol']}: {e_db}")
                return df
            except Exception as e:
                print(f"Erreur lors de la transformation ou stockage des données de taux de financement actuel pour {symbol}: {e}. Données brutes: {data}")
                return None
        else:
            print(f"Aucune donnée ou format de données incorrect reçu pour le taux de financement actuel de {symbol}.")
            return None

    def get_historical_funding_rates(self, symbol: str, page_size: int = 100, page_no: int = 1, db_session: Optional[Session] = None) -> Optional[pd.DataFrame]:
        """
        Récupère l'historique des taux de financement pour un symbole de contrat à terme.

        Args:
            symbol (str): Le symbole du contrat à terme (ex: 'BTCUSDT_UMCBL').
            page_size (int): Nombre d'enregistrements par page.
            page_no (int): Numéro de la page à récupérer.

        Returns:
            Optional[pd.DataFrame]: Un DataFrame avec les colonnes ['timestamp', 'symbol', 'funding_rate']
                                     ou None si une erreur survient.
                                     'timestamp' (settleTime) est en UTC.
        """
        endpoint_path = "/api/mix/v1/market/history-fundRate"
        params = {
            'symbol': symbol.upper(),
            'pageSize': str(page_size),
            'pageNo': str(page_no)
        }
        error_context = f"Historique des taux de financement pour {symbol} (page {page_no})"
        
        data_list = self._make_request(endpoint_path, params, error_context)
        
        if data_list and isinstance(data_list, list):
            if not data_list:
                # print(f"Aucune donnée d'historique de taux de financement retournée par l'API pour {symbol}, page {page_no}.")
                return pd.DataFrame(columns=['timestamp', 'symbol', 'funding_rate']) # Retourner un DF vide est ok

            try:
                df = pd.DataFrame(data_list)
                df.rename(columns={'fundingRate': 'funding_rate', 'settleTime': 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), unit='ms', utc=True)
                df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
                df['symbol'] = symbol.upper() # L'API ne renvoie pas le symbole dans la liste, on le réassigne

                df = df[['timestamp', 'symbol', 'funding_rate']]
                
                df.dropna(inplace=True)
                if df.empty:
                     # print(f"Aucune donnée d'historique de FR valide après traitement pour {symbol} page {page_no}.")
                     return df # Retourner DF vide

                if db_session:
                    for _, row in df.iterrows():
                        fr_entry = FundingRate(
                            symbol=row['symbol'],
                            timestamp=row['timestamp'].to_pydatetime(),
                            value=row['funding_rate']
                        )
                        try:
                            db_session.add(fr_entry)
                            db_session.commit()
                        except IntegrityError:
                            db_session.rollback()
                            # print(f"Doublon détecté pour FundingRate (hist): {row['symbol']} @ {row['timestamp']}. Non inséré.")
                        except Exception as e_db:
                            db_session.rollback()
                            print(f"Erreur DB lors de l'insertion de FundingRate (hist) pour {row['symbol']}: {e_db}")
                return df
            except Exception as e:
                print(f"Erreur lors de la transformation ou stockage des données d'historique de taux de financement pour {symbol}: {e}. Données brutes: {data_list}")
                return None
        else:
            if data_list is None: # Erreur déjà logguée par _make_request
                pass
            elif not isinstance(data_list, list):
                 print(f"Format de données incorrect reçu pour l'historique des taux de financement de {symbol} (attendu: list, reçu: {type(data_list)}).")
            return None

    def get_mark_price(self, symbol: str, db_session: Optional[Session] = None) -> Optional[pd.DataFrame]:
        """
        Récupère le Mark Price actuel pour un symbole de contrat à terme.

        Args:
            symbol (str): Le symbole du contrat à terme (ex: 'BTCUSDT_UMCBL').

        Returns:
            Optional[pd.DataFrame]: Un DataFrame avec les colonnes ['timestamp', 'symbol', 'mark_price']
                                     ou None si une erreur survient.
                                     'timestamp' est en UTC.
        """
        endpoint_path = "/api/mix/v1/market/mark-price"
        params = {'symbol': symbol.upper()}
        error_context = f"Mark Price pour {symbol}"
        
        data = self._make_request(endpoint_path, params, error_context)
        
        if data and isinstance(data, dict):
            try:
                df = pd.DataFrame([data])
                df.rename(columns={'markPrice': 'mark_price'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), unit='ms', utc=True)
                df['mark_price'] = pd.to_numeric(df['mark_price'], errors='coerce')
                df['symbol'] = df['symbol'].astype(str)

                df = df[['timestamp', 'symbol', 'mark_price']]
                
                df.dropna(inplace=True)
                if df.empty:
                    print(f"Aucune donnée de mark price valide après traitement pour {symbol}.")
                    return None

                if db_session:
                    for _, row in df.iterrows():
                        mp_entry = MarkPrice(
                            symbol=row['symbol'],
                            timestamp=row['timestamp'].to_pydatetime(),
                            value=row['mark_price']
                        )
                        try:
                            db_session.add(mp_entry)
                            db_session.commit()
                        except IntegrityError:
                            db_session.rollback()
                            # print(f"Doublon détecté pour MarkPrice: {row['symbol']} @ {row['timestamp']}. Non inséré.")
                        except Exception as e_db:
                            db_session.rollback()
                            print(f"Erreur DB lors de l'insertion de MarkPrice pour {row['symbol']}: {e_db}")
                return df
            except Exception as e:
                print(f"Erreur lors de la transformation ou stockage des données de mark price pour {symbol}: {e}. Données brutes: {data}")
                return None
        else:
            print(f"Aucune donnée ou format de données incorrect reçu pour le mark price de {symbol}.")
            return None

    def fetch_and_store_all_metrics(self, symbol: str, fetch_historical_funding: bool = True, funding_pages_to_fetch: int = 5):
        """
        Récupère toutes les nouvelles métriques (Open Interest, Funding Rate, Mark Price)
        pour un symbole donné et les stocke dans la base de données.

        Args:
            symbol (str): Le symbole du contrat à terme (ex: 'BTCUSDT_UMCBL').
            fetch_historical_funding (bool): Si True, tente de récupérer l'historique des funding rates.
            funding_pages_to_fetch (int): Nombre de pages à récupérer pour l'historique des funding rates.
        """
        db = SessionLocal()
        try:
            print(f"Début de la récupération et stockage des métriques pour {symbol}...")

            # Open Interest
            print(f"  Récupération de l'Open Interest pour {symbol}...")
            oi_df = self.get_open_interest(symbol=symbol, db_session=db)
            if oi_df is not None and not oi_df.empty:
                print(f"    Open Interest pour {symbol} récupéré et traité pour stockage: {len(oi_df)} ligne(s).")
            elif oi_df is not None and oi_df.empty:
                 print(f"    Aucune donnée d'Open Interest retournée pour {symbol} après traitement.")
            else:
                print(f"    Échec de la récupération de l'Open Interest pour {symbol}.")

            # Mark Price (souvent utile d'avoir le plus récent)
            print(f"  Récupération du Mark Price pour {symbol}...")
            mp_df = self.get_mark_price(symbol=symbol, db_session=db)
            if mp_df is not None and not mp_df.empty:
                print(f"    Mark Price pour {symbol} récupéré et traité pour stockage: {len(mp_df)} ligne(s).")
            elif mp_df is not None and mp_df.empty:
                print(f"    Aucune donnée de Mark Price retournée pour {symbol} après traitement.")
            else:
                print(f"    Échec de la récupération du Mark Price pour {symbol}.")

            # Current Funding Rate
            # print(f"  Récupération du Taux de Financement Actuel pour {symbol}...")
            # cfr_df = self.get_current_funding_rate(symbol=symbol, db_session=db) # Déjà stocké si pertinent
            # if cfr_df is not None and not cfr_df.empty:
            #     print(f"    Taux de Financement Actuel pour {symbol} récupéré et traité pour stockage: {len(cfr_df)} ligne(s).")
            # else:
            #     print(f"    Échec de la récupération du Taux de Financement Actuel pour {symbol}.")

            # Historical Funding Rates
            if fetch_historical_funding:
                print(f"  Récupération de l'historique des Taux de Financement pour {symbol} (jusqu'à {funding_pages_to_fetch} pages)...")
                all_historical_fr_dfs = []
                for page_num in range(1, funding_pages_to_fetch + 1):
                    print(f"    Page {page_num}/{funding_pages_to_fetch}...")
                    hfr_df_page = self.get_historical_funding_rates(symbol=symbol, page_no=page_num, db_session=db)
                    if hfr_df_page is not None and not hfr_df_page.empty:
                        all_historical_fr_dfs.append(hfr_df_page)
                        print(f"      Page {page_num} récupérée et traitée pour stockage: {len(hfr_df_page)} ligne(s).")
                        if len(hfr_df_page) < 100: # Moins que la page size par défaut, probablement la fin
                            print(f"      Moins de 100 résultats sur la page {page_num}, arrêt de la pagination pour les funding rates.")
                            break
                    elif hfr_df_page is not None and hfr_df_page.empty:
                        print(f"      Aucune donnée sur la page {page_num} pour l'historique des funding rates. Arrêt.")
                        break
                    else:
                        print(f"      Échec de la récupération de la page {page_num} pour l'historique des funding rates. Arrêt.")
                        break # Arrêter si une page échoue
                
                if all_historical_fr_dfs:
                    combined_hfr_df = pd.concat(all_historical_fr_dfs).drop_duplicates(subset=['timestamp', 'symbol']).sort_values(by='timestamp')
                    print(f"    Total de {len(combined_hfr_df)} enregistrements d'historique de Taux de Financement uniques récupérés et traités pour {symbol}.")
                else:
                    print(f"    Aucun enregistrement d'historique de Taux de Financement n'a été récupéré pour {symbol}.")
            
            print(f"Fin de la récupération et stockage des métriques pour {symbol}.")

        except Exception as e:
            print(f"Erreur majeure dans fetch_and_store_all_metrics pour {symbol}: {e}")
        finally:
            db.close()

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
