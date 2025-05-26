import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any, Optional

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


if __name__ == '__main__':
    # Créer des données d'exemple
    data_dict = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00', '2023-01-01 00:03:00', '2023-01-01 00:04:00']),
        'price': [100, 101, np.nan, 103, 200], # Contient NaN et outlier
        'volume': [1000, np.nan, 1200, 1150, 10000], # Contient NaN et outlier
        'category': ['A', 'B', 'A', np.nan, 'C'],
        'constant_val': [5, 5, 5, 5, 5]
    }
    sample_df = pd.DataFrame(data_dict)
    print("DataFrame Original:")
    print(sample_df)
    print(sample_df.dtypes)

    # 1. Gérer les valeurs manquantes
    print("\n--- Gestion des Valeurs Manquantes ---")
    # Stratégie auto
    df_na_handled_auto = handle_missing_values(sample_df.copy(), strategy='auto', numeric_strategy='median', category_strategy='mode')
    print("\nAprès gestion auto (num=median, cat=mode):")
    print(df_na_handled_auto)

    # Stratégie de remplissage spécifique
    df_na_filled = handle_missing_values(sample_df.copy(), strategy='fill', fill_value=0)
    print("\nAprès remplissage avec 0:")
    print(df_na_filled)

    # Stratégies spécifiques par colonne
    specific_strats = {
        'price': {'strategy': 'mean'},
        'volume': {'strategy': 'constant', 'value': 100},
        'category': {'strategy': 'ffill'}
    }
    df_na_specific = handle_missing_values(sample_df.copy(), specific_column_strategies=specific_strats)
    print("\nAprès stratégies spécifiques par colonne:")
    print(df_na_specific)


    # 2. Gérer les valeurs aberrantes (sur un df sans NaN pour la démo des outliers)
    # Utilisons df_na_handled_auto qui a déjà traité les NaN
    # ou df_na_specific
    df_for_outliers = df_na_specific.copy() # ou df_na_handled_auto.copy()
    # S'assurer que les colonnes pour outliers sont numériques
    df_for_outliers['price'] = pd.to_numeric(df_for_outliers['price'], errors='coerce')
    df_for_outliers['volume'] = pd.to_numeric(df_for_outliers['volume'], errors='coerce')


    print("\n--- Gestion des Valeurs Aberrantes (sur df_na_specific) ---")
    df_outliers_handled = handle_outliers_quantile(df_for_outliers.copy(), columns=['price', 'volume'], lower_quantile=0.05, upper_quantile=0.95, clip=True)
    print("\nAprès gestion des outliers (clipping):")
    print(df_outliers_handled)

    df_outliers_nan = handle_outliers_quantile(df_for_outliers.copy(), columns=['price', 'volume'], lower_quantile=0.05, upper_quantile=0.95, clip=False)
    print("\nAprès gestion des outliers (remplacement par NaN):")
    print(df_outliers_nan) # Peut réintroduire des NaN

    # 3. Normalisation Min-Max (sur un df sans NaN et sans outliers extrêmes pour la démo)
    # Utilisons df_outliers_handled
    print("\n--- Normalisation Min-Max (sur df_outliers_handled) ---")
    df_normalized = normalize_min_max(df_outliers_handled.copy(), columns=['price', 'volume'])
    print("\nAprès normalisation Min-Max:")
    print(df_normalized[['price', 'volume']])

    # Test normalisation avec colonne constante
    df_norm_const = normalize_min_max(df_outliers_handled.copy(), columns=['constant_val'])
    print("\nAprès normalisation Min-Max (colonne constante):")
    print(df_norm_const[['constant_val']])


    # 4. Standardisation Z-score (sur un df sans NaN et sans outliers extrêmes pour la démo)
    # Utilisons df_outliers_handled
    print("\n--- Standardisation Z-score (sur df_outliers_handled) ---")
    df_standardized = standardize_z_score(df_outliers_handled.copy(), columns=['price', 'volume'])
    print("\nAprès standardisation Z-score:")
    print(df_standardized[['price', 'volume']])

    # Test standardisation avec colonne constante
    df_std_const = standardize_z_score(df_outliers_handled.copy(), columns=['constant_val'])
    print("\nAprès standardisation Z-score (colonne constante):")
    print(df_std_const[['constant_val']])

    # Test avec une colonne qui n'existe pas ou non numérique
    print("\n--- Tests avec colonnes invalides ---")
    df_test_invalid = sample_df.copy()
    handle_outliers_quantile(df_test_invalid, columns=['non_existent_col', 'category'])
    normalize_min_max(df_test_invalid, columns=['non_existent_col', 'category'])
    standardize_z_score(df_test_invalid, columns=['non_existent_col', 'category'])

    print("\n--- Test handle_missing_values avec drop_row/drop_col ---")
    df_missing_drop = sample_df.copy()
    df_dropped_rows_price = handle_missing_values(df_missing_drop.copy(), specific_column_strategies={'price': {'strategy': 'drop_row'}})
    print("\nAprès drop_row pour NaN dans 'price':")
    print(df_dropped_rows_price)

    df_dropped_col_category = handle_missing_values(df_missing_drop.copy(), specific_column_strategies={'category': {'strategy': 'drop_col'}})
    print("\nAprès drop_col pour 'category' (si elle a des NaN, sinon elle reste):")
    print(df_dropped_col_category) # Note: 'category' sera supprimée si elle a des NaN, sinon elle reste.

    # Pour forcer la suppression de la colonne 'category' via auto strategy
    df_auto_drop_col = sample_df.copy()
    # Mettre un NaN dans category si ce n'est pas déjà le cas pour le test
    # df_auto_drop_col.loc[0, 'category'] = np.nan
    df_auto_dropped = handle_missing_values(df_auto_drop_col, strategy='auto', category_strategy='drop_col')
    print("\nAprès gestion auto avec category_strategy='drop_col':")
    print(df_auto_dropped)