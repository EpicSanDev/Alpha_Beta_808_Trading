import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any, Optional, Type
from sqlalchemy.orm import Session
from ..core.database import SessionLocal, OpenInterest, FundingRate, MarkPrice, Base as DBBaseModel  # Renommé Base en DBBaseModel pour éviter conflit
from datetime import datetime

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto',
    fill_value: Optional[Union[float, str, int]] = None,
    numeric_strategy: str = 'mean', # 'mean', 'median', 'ffill', 'bfill', 'drop_row', 'drop_col'
    category_strategy: str = 'mode', # 'mode', 'ffill', 'bfill', 'drop_row', 'drop_col', 'constant'
    specific_column_strategies: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes dans un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        strategy (str): Stratégie globale. Si 'auto', tente de déduire par type de colonne.
                        Si 'specific', utilise specific_column_strategies.
                        Si 'fill', utilise fill_value pour toutes les colonnes.
        fill_value (Optional[Union[float, str, int]]): Valeur à utiliser si strategy='fill'.
        numeric_strategy (str): Stratégie pour les colonnes numériques si strategy='auto'.
        category_strategy (str): Stratégie pour les colonnes catégorielles/objets si strategy='auto'.
        specific_column_strategies (Optional[Dict[str, Dict[str, Any]]]):
            Dictionnaire pour spécifier des stratégies par colonne.
            Ex: {'col_name': {'strategy': 'median'}, 'col2_name': {'strategy': 'constant', 'value': 'Missing'}}

    Returns:
        pd.DataFrame: DataFrame avec les valeurs manquantes traitées.
    """
    df_processed = df.copy()

    if strategy == 'fill' and fill_value is not None:
        df_processed = df_processed.fillna(fill_value)
        return df_processed
    elif strategy == 'fill' and fill_value is None:
        raise ValueError("fill_value ne peut pas être None si strategy='fill'")

    columns_to_process = df_processed.columns

    if specific_column_strategies:
        for col, col_strat_info in specific_column_strategies.items():
            if col not in df_processed.columns:
                print(f"Avertissement: La colonne '{col}' spécifiée dans specific_column_strategies n'existe pas.")
                continue

            strat = col_strat_info.get('strategy')
            val = col_strat_info.get('value') # Pour 'constant'

            if strat == 'mean':
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                else:
                    print(f"Avertissement: La stratégie 'mean' ne peut pas être appliquée à la colonne non numérique '{col}'.")
            elif strat == 'median':
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                else:
                    print(f"Avertissement: La stratégie 'median' ne peut pas être appliquée à la colonne non numérique '{col}'.")
            elif strat == 'mode':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else np.nan)
            elif strat == 'ffill':
                df_processed[col] = df_processed[col].fillna(method='ffill')
            elif strat == 'bfill':
                df_processed[col] = df_processed[col].fillna(method='bfill')
            elif strat == 'drop_row':
                df_processed = df_processed.dropna(subset=[col])
            elif strat == 'drop_col':
                if col in df_processed.columns: # Vérifier si elle n'a pas déjà été supprimée
                    df_processed = df_processed.drop(columns=[col])
            elif strat == 'constant':
                if val is not None:
                    df_processed[col] = df_processed[col].fillna(val)
                else:
                    raise ValueError(f"La stratégie 'constant' pour la colonne '{col}' nécessite une 'value'.")
            else:
                print(f"Avertissement: Stratégie '{strat}' non reconnue pour la colonne '{col}'.")
        # Mettre à jour columns_to_process au cas où des colonnes ont été supprimées
        columns_to_process = [c for c in df_processed.columns if c not in specific_column_strategies or specific_column_strategies[c].get('strategy') not in ['drop_col', 'drop_row']]


    if strategy == 'auto':
        for col in columns_to_process:
            if col not in df_processed.columns: # Peut avoir été supprimé par une stratégie spécifique
                continue
            if df_processed[col].isnull().any(): # Traiter seulement si des NaN existent
                if pd.api.types.is_numeric_dtype(df_processed[col]):
                    strat_to_use = numeric_strategy
                    if strat_to_use == 'mean':
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    elif strat_to_use == 'median':
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    elif strat_to_use == 'ffill':
                        df_processed[col] = df_processed[col].fillna(method='ffill')
                    elif strat_to_use == 'bfill':
                        df_processed[col] = df_processed[col].fillna(method='bfill')
                    elif strat_to_use == 'drop_row':
                        df_processed = df_processed.dropna(subset=[col])
                    elif strat_to_use == 'drop_col':
                         if col in df_processed.columns:
                            df_processed = df_processed.drop(columns=[col])
                    elif strat_to_use == 'constant' and fill_value is not None: # Utilise fill_value global si numeric_strategy est 'constant'
                        df_processed[col] = df_processed[col].fillna(fill_value)
                    elif strat_to_use == 'constant' and fill_value is None:
                        print(f"Avertissement: numeric_strategy='constant' sans fill_value global pour la colonne '{col}'. Remplissage par 0 par défaut.")
                        df_processed[col] = df_processed[col].fillna(0)


                elif pd.api.types.is_object_dtype(df_processed[col]) or pd.api.types.is_categorical_dtype(df_processed[col]):
                    strat_to_use = category_strategy
                    if strat_to_use == 'mode':
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mode().iloc[0] if not df_processed[col].mode().empty else "Unknown")
                    elif strat_to_use == 'ffill':
                        df_processed[col] = df_processed[col].fillna(method='ffill')
                    elif strat_to_use == 'bfill':
                        df_processed[col] = df_processed[col].fillna(method='bfill')
                    elif strat_to_use == 'drop_row':
                        df_processed = df_processed.dropna(subset=[col])
                    elif strat_to_use == 'drop_col':
                        if col in df_processed.columns:
                            df_processed = df_processed.drop(columns=[col])
                    elif strat_to_use == 'constant' and fill_value is not None: # Utilise fill_value global
                        df_processed[col] = df_processed[col].fillna(fill_value)
                    elif strat_to_use == 'constant' and fill_value is None:
                        print(f"Avertissement: category_strategy='constant' sans fill_value global pour la colonne '{col}'. Remplissage par 'Unknown' par défaut.")
                        df_processed[col] = df_processed[col].fillna("Unknown")
    return df_processed


def handle_outliers_quantile(
    df: pd.DataFrame,
    columns: List[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    clip: bool = True
) -> pd.DataFrame:
    """
    Gère les valeurs aberrantes en utilisant la méthode des quantiles.
    Les valeurs en dehors de [lower_quantile, upper_quantile] sont soit clippées (par défaut)
    soit remplacées par NaN.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        columns (List[str]): Liste des colonnes numériques sur lesquelles appliquer le traitement.
        lower_quantile (float): Quantile inférieur.
        upper_quantile (float): Quantile supérieur.
        clip (bool): Si True, clippe les valeurs aux limites des quantiles.
                     Si False, remplace les valeurs aberrantes par NaN.

    Returns:
        pd.DataFrame: DataFrame avec les valeurs aberrantes traitées.
    """
    df_processed = df.copy()
    for col in columns:
        if col not in df_processed.columns or not pd.api.types.is_numeric_dtype(df_processed[col]):
            print(f"Avertissement: La colonne '{col}' n'est pas numérique ou n'existe pas. Elle sera ignorée pour la gestion des outliers.")
            continue

        lower_bound = df_processed[col].quantile(lower_quantile)
        upper_bound = df_processed[col].quantile(upper_quantile)

        if clip:
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        else:
            df_processed[col] = np.where(
                (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound),
                np.nan,
                df_processed[col]
            )
    return df_processed


def normalize_min_max(df: pd.DataFrame, columns: Union[List[str], str] = None, column: str = None) -> pd.DataFrame:
    """
    Applique la normalisation Min-Max (mise à l'échelle entre 0 et 1)
    aux colonnes spécifiées.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        columns (Union[List[str], str], optional): Liste des colonnes numériques à normaliser ou une seule colonne.
        column (str, optional): Colonne unique à normaliser (pour compatibilité avec les versions précédentes).

    Returns:
        pd.DataFrame: DataFrame avec les colonnes normalisées.
    """
    df_processed = df.copy()
    
    # Déterminer quelles colonnes normaliser
    cols_to_normalize = []
    if column is not None:
        cols_to_normalize = [column]
    elif columns is not None:
        if isinstance(columns, str):
            cols_to_normalize = [columns]
        else:
            cols_to_normalize = columns
    else:
        # Si aucune colonne n'est spécifiée, normaliser toutes les colonnes numériques
        cols_to_normalize = [col for col in df_processed.columns if pd.api.types.is_numeric_dtype(df_processed[col])]
    
    for col in cols_to_normalize:
        if col not in df_processed.columns or not pd.api.types.is_numeric_dtype(df_processed[col]):
            print(f"Avertissement: La colonne '{col}' n'est pas numérique ou n'existe pas. Elle sera ignorée pour la normalisation.")
            continue
        min_val = df_processed[col].min()
        max_val = df_processed[col].max()
        if max_val == min_val: # Éviter la division par zéro si toutes les valeurs sont identiques
            df_processed[col] = 0.0 if max_val == 0 else 0.5 # ou une autre valeur constante appropriée
            print(f"Avertissement: La colonne '{col}' a des valeurs constantes. Normalisée à 0.0 ou 0.5.")
        else:
            df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
    return df_processed


def standardize_z_score(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Applique la standardisation Z-score (moyenne 0, écart-type 1)
    aux colonnes spécifiées.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        columns (List[str]): Liste des colonnes numériques à standardiser.

    Returns:
        pd.DataFrame: DataFrame avec les colonnes standardisées.
    """
    df_processed = df.copy()
    for col in columns:
        if col not in df_processed.columns or not pd.api.types.is_numeric_dtype(df_processed[col]):
            print(f"Avertissement: La colonne '{col}' n'est pas numérique ou n'existe pas. Elle sera ignorée pour la standardisation.")
            continue
        mean_val = df_processed[col].mean()
        std_val = df_processed[col].std()
        if std_val == 0: # Éviter la division par zéro
            df_processed[col] = 0.0 # Toutes les valeurs sont identiques, donc z-score est 0
            print(f"Avertissement: La colonne '{col}' a un écart-type de 0. Standardisée à 0.0.")
        else:
            df_processed[col] = (df_processed[col] - mean_val) / std_val
    return df_processed


def load_data_from_db(
    db_session: Session,
    model: Type[DBBaseModel],
    symbol: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Charge les données depuis la base de données pour un modèle, un symbole et une période donnés.
    """
    query = db_session.query(model).filter(model.symbol == symbol)
    if start_date:
        query = query.filter(model.timestamp >= start_date)
    if end_date:
        query = query.filter(model.timestamp <= end_date)
    
    df = pd.read_sql(query.statement, db_session.bind)
    if df.empty:
        # print(f"Aucune donnée trouvée en BDD pour {model.__tablename__}, symbole {symbol} entre {start_date} et {end_date}")
        # Retourner un DataFrame vide avec les colonnes attendues pour éviter les erreurs en aval
        if model == OpenInterest:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'value']).rename(columns={'value': 'open_interest'})
        elif model == FundingRate:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'value']).rename(columns={'value': 'funding_rate'})
        elif model == MarkPrice:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'value']).rename(columns={'value': 'mark_price'})
        else:
            return pd.DataFrame()

    # Renommer la colonne 'value' générique en nom de métrique spécifique
    if 'value' in df.columns:
        if model == OpenInterest:
            df.rename(columns={'value': 'open_interest'}, inplace=True)
        elif model == FundingRate:
            df.rename(columns={'value': 'funding_rate'}, inplace=True)
        elif model == MarkPrice:
            df.rename(columns={'value': 'mark_price'}, inplace=True)
            
    if 'timestamp' in df.columns:
         df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
         df = df.set_index('timestamp').sort_index()
    
    return df


def resample_and_align_data(
    df_klines: pd.DataFrame,
    df_metric: pd.DataFrame,
    metric_name: str,
    resample_rule: Optional[str] = None, # Ex: '1H', '4H', '1D'. Si None, utilise l'index des klines.
    interpolation_method: str = 'ffill' # 'ffill', 'bfill', 'linear'
) -> pd.DataFrame:
    """
    Rééchantillonne df_metric à la fréquence de df_klines (ou à resample_rule) et l'aligne.
    Gère les valeurs manquantes introduites par le rééchantillonnage via interpolation.
    """
    if df_metric.empty or metric_name not in df_metric.columns:
        # print(f"DataFrame de métrique vide ou colonne {metric_name} manquante. Retourne un DataFrame klines inchangé.")
        # Ajoute une colonne de NaNs pour la métrique si elle n'existe pas pour assurer la cohérence
        if metric_name not in df_klines.columns:
            df_klines_aligned = df_klines.copy()
            df_klines_aligned[metric_name] = np.nan
            return df_klines_aligned
        return df_klines

    if df_klines.empty:
        # print("DataFrame klines vide. Impossible d'aligner.")
        return pd.DataFrame() # ou df_metric si on veut retourner quelque chose

    # S'assurer que les deux DataFrames ont un index DateTime
    if not isinstance(df_klines.index, pd.DatetimeIndex):
        raise ValueError("df_klines doit avoir un DatetimeIndex.")
    if not isinstance(df_metric.index, pd.DatetimeIndex):
        # Si df_metric n'a pas d'index datetime (ex: chargé sans set_index), on le fait
        if 'timestamp' in df_metric.columns:
            df_metric['timestamp'] = pd.to_datetime(df_metric['timestamp'], utc=True)
            df_metric = df_metric.set_index('timestamp')
        else:
            raise ValueError("df_metric doit avoir un DatetimeIndex ou une colonne 'timestamp'.")

    # 1. Rééchantillonner la métrique si une règle est fournie
    df_metric_resampled = df_metric
    if resample_rule:
        # Pour le rééchantillonnage, on veut souvent la dernière valeur (pour OI, Mark Price) ou la moyenne/somme.
        # Pour funding rate, c'est une valeur ponctuelle, donc 'last' ou 'asfreq' puis ffill est bon.
        # Utilisons 'asfreq' pour prendre la valeur à la fin de la période, puis ffill.
        df_metric_resampled = df_metric[[metric_name]].asfreq(resample_rule, method=None) # Pas de méthode ici, on gère après

    # 2. Joindre avec l'index des klines pour aligner
    # On utilise un left join pour garder tous les timestamps des klines
    df_aligned = df_klines.copy()
    
    # Si la métrique n'est pas déjà dans les klines (cas typique)
    # On réindexe la métrique resamplée sur l'index des klines, puis on interpole
    if metric_name not in df_aligned.columns:
        # S'assurer que la colonne existe dans df_metric_resampled avant de la réindexer
        if metric_name in df_metric_resampled.columns:
            metric_reindexed = df_metric_resampled[[metric_name]].reindex(df_aligned.index)
        else: # Si la colonne n'existe pas (ex: df_metric était vide), créer une colonne de NaN
            metric_reindexed = pd.Series(np.nan, index=df_aligned.index, name=metric_name)
    else: # Si la métrique est déjà là (moins probable ici, mais pour être robuste)
        metric_reindexed = df_metric_resampled[[metric_name]].reindex(df_aligned.index)


    # 3. Gérer les NaN après réindexation/jointure avec la méthode spécifiée
    if interpolation_method == 'ffill':
        metric_interpolated = metric_reindexed[metric_name].ffill()
    elif interpolation_method == 'bfill':
        metric_interpolated = metric_reindexed[metric_name].bfill()
    elif interpolation_method == 'linear':
        metric_interpolated = metric_reindexed[metric_name].interpolate(method='linear')
    else:
        metric_interpolated = metric_reindexed[metric_name] # Pas d'interpolation

    df_aligned[metric_name] = metric_interpolated
    
    # Optionnel: bfill initial pour les NaN au début si ffill a été utilisé
    if interpolation_method == 'ffill' and df_aligned[metric_name].isnull().any():
        df_aligned[metric_name] = df_aligned[metric_name].bfill()

    return df_aligned


def calculate_basis(df_merged: pd.DataFrame, future_price_col: str = 'mark_price', spot_price_col: str = 'close') -> pd.DataFrame:
    """
    Calcule le Basis (différence entre prix future et prix spot).
    Assure que les colonnes nécessaires existent et sont numériques.
    """
    df_with_basis = df_merged.copy()
    
    if future_price_col not in df_with_basis.columns:
        print(f"Avertissement: Colonne du prix future '{future_price_col}' non trouvée. Impossible de calculer le basis.")
        df_with_basis['basis'] = np.nan
        return df_with_basis
        
    if spot_price_col not in df_with_basis.columns:
        print(f"Avertissement: Colonne du prix spot '{spot_price_col}' non trouvée. Impossible de calculer le basis.")
        df_with_basis['basis'] = np.nan
        return df_with_basis

    # S'assurer que les colonnes sont numériques
    df_with_basis[future_price_col] = pd.to_numeric(df_with_basis[future_price_col], errors='coerce')
    df_with_basis[spot_price_col] = pd.to_numeric(df_with_basis[spot_price_col], errors='coerce')
    
    # Calculer le basis
    df_with_basis['basis'] = df_with_basis[future_price_col] - df_with_basis[spot_price_col]
    
    # Gérer les NaN qui pourraient résulter de la soustraction si l'une des colonnes avait des NaN
    # La gestion des NaN pour 'basis' peut être faite en aval si nécessaire, ou ici.
    # Par exemple, un ffill simple si on s'attend à ce que le basis soit relativement stable.
    # df_with_basis['basis'] = df_with_basis['basis'].ffill()
    
    return df_with_basis


def preprocess_new_metrics_for_symbol(
    df_klines_orig: pd.DataFrame, # DataFrame des klines pour UN symbole
    symbol: str,
    db_session: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    kline_interval_for_resampling: Optional[str] = None, # Ex: '1H', '4H'. Si None, utilise la fréquence des klines.
    include_basis: bool = True,
    spot_price_col_for_basis: str = 'close' # Colonne des klines à utiliser comme prix spot
) -> pd.DataFrame:
    """
    Charge, prétraite (rééchantillonne, aligne, gère NaN) les nouvelles métriques
    (Open Interest, Funding Rate, Mark Price) pour un symbole donné et les fusionne avec les klines.
    Calcule également le basis si demandé.

    Args:
        df_klines_orig (pd.DataFrame): DataFrame des klines (doit avoir un DatetimeIndex).
        symbol (str): Symbole du contrat.
        db_session (Session): Session SQLAlchemy.
        start_date (Optional[datetime]): Date de début pour charger les métriques.
        end_date (Optional[datetime]): Date de fin pour charger les métriques.
        kline_interval_for_resampling (Optional[str]): Règle de rééchantillonnage (ex: '1H').
                                                       Si None, aligne sur l'index des klines tel quel.
        include_basis (bool): Si True, calcule et ajoute le basis.
        spot_price_col_for_basis (str): Nom de la colonne dans df_klines à utiliser comme prix spot.

    Returns:
        pd.DataFrame: DataFrame des klines fusionné avec les nouvelles métriques prétraitées.
    """
    if df_klines_orig.empty:
        print(f"DataFrame klines vide pour {symbol}. Retourne un DataFrame vide.")
        return pd.DataFrame()
    
    if not isinstance(df_klines_orig.index, pd.DatetimeIndex):
        # Essayer de définir l'index si une colonne 'timestamp' existe
        if 'timestamp' in df_klines_orig.columns:
            df_klines_temp = df_klines_orig.copy()
            df_klines_temp['timestamp'] = pd.to_datetime(df_klines_temp['timestamp'], utc=True)
            df_klines_temp = df_klines_temp.set_index('timestamp').sort_index()
            df_merged = df_klines_temp
        else:
            raise ValueError("df_klines_orig doit avoir un DatetimeIndex ou une colonne 'timestamp'.")
    else:
        df_merged = df_klines_orig.copy()

    # Déterminer la règle de rééchantillonnage si non fournie
    # Cela suppose que les klines sont à fréquence régulière.
    resample_rule_to_use = kline_interval_for_resampling
    if not resample_rule_to_use and len(df_merged.index) > 1:
         # Tenter d'inférer la fréquence si possible, sinon laisser None (alignement direct)
        inferred_freq = pd.infer_freq(df_merged.index)
        if inferred_freq:
            resample_rule_to_use = inferred_freq
        # else: print(f"Impossible d'inférer la fréquence pour {symbol}, alignement direct sera utilisé.")


    # 1. Open Interest
    # print(f"  Chargement Open Interest pour {symbol}...")
    df_oi = load_data_from_db(db_session, OpenInterest, symbol, start_date, end_date)
    if not df_oi.empty:
        df_merged = resample_and_align_data(df_merged, df_oi, 'open_interest', resample_rule=resample_rule_to_use, interpolation_method='ffill')
    elif 'open_interest' not in df_merged.columns: # Assurer que la colonne existe même si vide
        df_merged['open_interest'] = np.nan


    # 2. Funding Rate
    # print(f"  Chargement Funding Rates pour {symbol}...")
    df_fr = load_data_from_db(db_session, FundingRate, symbol, start_date, end_date)
    if not df_fr.empty:
        # Les funding rates sont typiquement constants entre les périodes de funding. 'ffill' est approprié.
        df_merged = resample_and_align_data(df_merged, df_fr, 'funding_rate', resample_rule=resample_rule_to_use, interpolation_method='ffill')
    elif 'funding_rate' not in df_merged.columns:
        df_merged['funding_rate'] = np.nan

    # 3. Mark Price
    # print(f"  Chargement Mark Prices pour {symbol}...")
    df_mp = load_data_from_db(db_session, MarkPrice, symbol, start_date, end_date)
    if not df_mp.empty:
        df_merged = resample_and_align_data(df_merged, df_mp, 'mark_price', resample_rule=resample_rule_to_use, interpolation_method='ffill')
    elif 'mark_price' not in df_merged.columns:
        df_merged['mark_price'] = np.nan

    # 4. Calcul du Basis (si demandé et si mark_price est disponible)
    if include_basis:
        if 'mark_price' in df_merged.columns and not df_merged['mark_price'].isnull().all():
            # print(f"  Calcul du Basis pour {symbol}...")
            df_merged = calculate_basis(df_merged, future_price_col='mark_price', spot_price_col=spot_price_col_for_basis)
        else:
            # print(f"  Mark price non disponible ou entièrement NaN pour {symbol}, impossible de calculer le basis.")
            if 'basis' not in df_merged.columns: df_merged['basis'] = np.nan
    elif 'basis' not in df_merged.columns: # Assurer que la colonne existe même si non calculée
         df_merged['basis'] = np.nan


    # Gestion finale des NaN pour les nouvelles colonnes si elles existent encore après ffill/bfill initial
    new_metric_cols = ['open_interest', 'funding_rate', 'mark_price', 'basis']
    for col in new_metric_cols:
        if col in df_merged.columns and df_merged[col].isnull().any():
            # Une stratégie simple : ffill puis bfill pour combler tous les trous restants
            df_merged[col] = df_merged[col].ffill().bfill()
            # Si encore des NaN (ex: df entièrement vide au début), on peut remplir avec 0 ou laisser NaN
            # df_merged[col] = df_merged[col].fillna(0) # Optionnel

    return df_merged


if __name__ == '__main__':
    # --- Configuration pour les tests ---
    TEST_SYMBOL = "BTCUSDT_UMCBL" # Assurez-vous que des données existent pour ce symbole en BDD
    
    # Créer une session de base de données pour les tests
    # Assurez-vous que src/core/database.py peut être exécuté pour créer trading_web.db
    # et que src/acquisition/data_collector.py a été exécuté pour peupler des données.
    try:
        from ..core.database import create_db_and_tables
        create_db_and_tables() # S'assurer que les tables existent
        print("Tables de la base de données vérifiées/créées.")
    except ImportError:
        print("Impossible d'importer create_db_and_tables. Assurez-vous que le PYTHONPATH est correct.")
        print("Ou exécutez ce script depuis le répertoire racine du projet avec python -m src.acquisition.preprocessing")
    except Exception as e_db_init:
        print(f"Erreur lors de l'initialisation de la BDD pour les tests: {e_db_init}")


    db_session_test = SessionLocal()

    # 0. Créer des données de klines d'exemple (simulées)
    print(f"\n--- Préparation des données de klines pour {TEST_SYMBOL} ---")
    
    # Générer des klines horaires pour les 10 derniers jours
    end_dt_test = datetime.utcnow().replace(tzinfo=None) # Naive datetime for pandas date_range
    start_dt_test = end_dt_test - pd.Timedelta(days=10)
    
    # Créer un index de klines horaires
    kline_index = pd.date_range(start=start_dt_test, end=end_dt_test, freq='1H', tz='UTC')
    if kline_index.empty and start_dt_test < end_dt_test : # Si date_range est vide mais devrait pas l'être
        kline_index = pd.DatetimeIndex([start_dt_test + pd.Timedelta(hours=i) for i in range(int((end_dt_test-start_dt_test).total_seconds() / 3600) +1)], tz='UTC')


    sample_klines_df = pd.DataFrame(index=kline_index)
    sample_klines_df['symbol'] = TEST_SYMBOL
    sample_klines_df['open'] = np.random.uniform(29000, 31000, size=len(sample_klines_df))
    sample_klines_df['high'] = sample_klines_df['open'] + np.random.uniform(0, 500, size=len(sample_klines_df))
    sample_klines_df['low'] = sample_klines_df['open'] - np.random.uniform(0, 500, size=len(sample_klines_df))
    sample_klines_df['close'] = (sample_klines_df['high'] + sample_klines_df['low']) / 2
    sample_klines_df['volume'] = np.random.uniform(100, 10000, size=len(sample_klines_df))
    
    print(f"Klines d'exemple générées pour {TEST_SYMBOL} ({len(sample_klines_df)} lignes):")
    print(sample_klines_df.head(2))
    print(sample_klines_df.tail(2))


    # --- Test de preprocess_new_metrics_for_symbol ---
    print(f"\n--- Test de preprocess_new_metrics_for_symbol pour {TEST_SYMBOL} ---")
    # Utiliser les dates des klines pour filtrer les métriques de la BDD
    db_start_date = sample_klines_df.index.min().to_pydatetime()
    db_end_date = sample_klines_df.index.max().to_pydatetime()

    df_processed_metrics = preprocess_new_metrics_for_symbol(
        df_klines_orig=sample_klines_df,
        symbol=TEST_SYMBOL,
        db_session=db_session_test,
        start_date=db_start_date,
        end_date=db_end_date,
        kline_interval_for_resampling='1H', # Correspond à la fréquence de nos klines d'exemple
        include_basis=True,
        spot_price_col_for_basis='close'
    )

    if not df_processed_metrics.empty:
        print(f"\nDataFrame après preprocess_new_metrics_for_symbol pour {TEST_SYMBOL}:")
        print(df_processed_metrics.head())
        print(df_processed_metrics.tail())
        print(f"Shape: {df_processed_metrics.shape}")
        print("Colonnes:", df_processed_metrics.columns.tolist())
        print("\nInfos sur les NaN par colonne:")
        print(df_processed_metrics[['open_interest', 'funding_rate', 'mark_price', 'basis']].isnull().sum())
        
        # Vérifier si les nouvelles colonnes ont été ajoutées
        expected_new_cols = ['open_interest', 'funding_rate', 'mark_price', 'basis']
        for col in expected_new_cols:
            if col not in df_processed_metrics.columns:
                print(f"ERREUR: La colonne attendue '{col}' est manquante dans le résultat.")
            elif df_processed_metrics[col].isnull().all():
                 print(f"AVERTISSEMENT: La colonne '{col}' est entièrement NaN. Vérifiez les données en BDD ou la logique de chargement/alignement.")

    else:
        print(f"preprocess_new_metrics_for_symbol a retourné un DataFrame vide pour {TEST_SYMBOL}.")

    # Fermer la session de test
    db_session_test.close()

    # --- Tests des fonctions unitaires (déjà présents, mais peuvent être étendus) ---
    print("\n--- Tests unitaires existants (handle_missing_values, etc.) ---")
    # Créer des données d'exemple pour les tests unitaires
    data_dict_unit = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00', '2023-01-01 00:03:00', '2023-01-01 00:04:00']),
        'price': [100, 101, np.nan, 103, 200],
        'volume': [1000, np.nan, 1200, 1150, 10000],
        'category': ['A', 'B', 'A', np.nan, 'C'],
        'constant_val': [5, 5, 5, 5, 5]
    }
    sample_df_unit = pd.DataFrame(data_dict_unit)
    print("\nDataFrame Original pour tests unitaires:")
    print(sample_df_unit)
    
    # ... (le reste des tests unitaires existants peuvent suivre ici) ...
    # (Les copier-coller ici pour la complétude si nécessaire, ou les laisser tels quels s'ils sont déjà robustes)
    # Exemple:
    df_na_handled_auto_unit = handle_missing_values(sample_df_unit.copy(), strategy='auto', numeric_strategy='median', category_strategy='mode')
    print("\nAprès gestion auto (num=median, cat=mode) - Test Unitaire:")
    print(df_na_handled_auto_unit)