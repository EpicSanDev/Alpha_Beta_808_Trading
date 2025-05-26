import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any, Optional

def handle_missing_values_column(
    df: pd.DataFrame,
    column: str,
    strategy: str = 'ffill'
) -> pd.DataFrame:
    """
    Wrapper simplifié pour handle_missing_values qui se concentre sur une seule colonne.

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        column (str): Nom de la colonne à traiter.
        strategy (str): Stratégie à appliquer ('ffill', 'bfill', 'mean', 'median', 'mode').

    Returns:
        pd.DataFrame: DataFrame avec les valeurs manquantes traitées.
    """
    if column not in df.columns:
        raise ValueError(f"Colonne '{column}' non trouvée dans le DataFrame.")
    
    df_copy = df.copy()
    
    if strategy == 'ffill':
        df_copy[column] = df_copy[column].ffill()
    elif strategy == 'bfill':
        df_copy[column] = df_copy[column].bfill()
    elif strategy == 'mean':
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        else:
            raise ValueError(f"Stratégie 'mean' ne peut pas être appliquée à la colonne non numérique '{column}'.")
    elif strategy == 'median':
        if pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        else:
            raise ValueError(f"Stratégie 'median' ne peut pas être appliquée à la colonne non numérique '{column}'.")
    elif strategy == 'mode':
        df_copy[column] = df_copy[column].fillna(df_copy[column].mode().iloc[0] if not df_copy[column].mode().empty else np.nan)
    else:
        raise ValueError(f"Stratégie '{strategy}' non reconnue. Utilisez 'ffill', 'bfill', 'mean', 'median' ou 'mode'.")
    
    return df_copy
