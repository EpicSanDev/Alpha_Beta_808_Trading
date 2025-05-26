import pandas as pd
from typing import List

def calculate_sma(df: pd.DataFrame, column: str = 'close', windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Calcule les moyennes mobiles simples (SMA) pour une colonne donnée et une liste de fenêtres.

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant les données de prix.
                           Doit contenir la colonne spécifiée par le paramètre 'column'.
        column (str): Nom de la colonne sur laquelle calculer les SMA (par défaut 'close').
        windows (List[int]): Liste des fenêtres (en jours) pour calculer les SMA (par défaut [10, 20, 50]).

    Returns:
        pd.DataFrame: DataFrame enrichi avec les nouvelles colonnes SMA (ex: 'sma_10', 'sma_20').
    """
    df_copy = df.copy()
    for window in windows:
        df_copy[f'sma_{window}'] = df_copy[column].rolling(window=window).mean()
    return df_copy

def calculate_ema(df: pd.DataFrame, column: str = 'close', windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Calcule les moyennes mobiles exponentielles (EMA) pour une colonne donnée et une liste de fenêtres.

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant les données de prix.
                           Doit contenir la colonne spécifiée par le paramètre 'column'.
        column (str): Nom de la colonne sur laquelle calculer les EMA (par défaut 'close').
        windows (List[int]): Liste des fenêtres (en jours) pour calculer les EMA (par défaut [10, 20, 50]).

    Returns:
        pd.DataFrame: DataFrame enrichi avec les nouvelles colonnes EMA (ex: 'ema_10', 'ema_20').
    """
    df_copy = df.copy()
    for window in windows:
        df_copy[f'ema_{window}'] = df_copy[column].ewm(span=window, adjust=False).mean()
    return df_copy

def calculate_rsi(df: pd.DataFrame, column: str = 'close', window: int = 14) -> pd.DataFrame:
    """
    Calcule l'Indice de Force Relative (RSI) pour une colonne donnée et une fenêtre spécifiée.

    Args:
        df (pd.DataFrame): DataFrame d'entrée contenant les données de prix.
                           Doit contenir la colonne spécifiée par le paramètre 'column'.
        column (str): Nom de la colonne sur laquelle calculer le RSI (par défaut 'close').
        window (int): Fenêtre (en jours) pour calculer le RSI (par défaut 14).

    Returns:
        pd.DataFrame: DataFrame enrichi avec la nouvelle colonne RSI (ex: 'rsi_14').
    """
    df_copy = df.copy()
    delta = df_copy[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df_copy[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    return df_copy

# MACD - Implémentation optionnelle si le temps le permet
def calculate_macd(df: pd.DataFrame, column: str = 'close', short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> pd.DataFrame:
    """
    Calcule la Moving Average Convergence Divergence (MACD).

    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        column (str): Colonne sur laquelle calculer le MACD (par défaut 'close').
        short_window (int): Fenêtre courte pour l'EMA (par défaut 12).
        long_window (int): Fenêtre longue pour l'EMA (par défaut 26).
        signal_window (int): Fenêtre pour la ligne de signal EMA (par défaut 9).

    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes 'macd', 'macd_signal', 'macd_hist'.
    """
    df_copy = df.copy()
    ema_short = df_copy[column].ewm(span=short_window, adjust=False).mean()
    ema_long = df_copy[column].ewm(span=long_window, adjust=False).mean()
    df_copy['macd'] = ema_short - ema_long
    df_copy['macd_signal'] = df_copy['macd'].ewm(span=signal_window, adjust=False).mean()
    df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
    return df_copy

def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'close', window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calcule les Bandes de Bollinger.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        column (str): Colonne sur laquelle calculer les bandes (par défaut 'close').
        window (int): Fenêtre pour la moyenne mobile (par défaut 20).
        num_std (float): Nombre d'écarts-types pour les bandes (par défaut 2.0).
    
    Returns:
        pd.DataFrame: DataFrame enrichi avec 'bb_upper', 'bb_lower', 'bb_position'.
    """
    df_copy = df.copy()
    sma = df_copy[column].rolling(window=window).mean()
    std = df_copy[column].rolling(window=window).std()
    
    df_copy['bb_upper'] = sma + (num_std * std)
    df_copy['bb_lower'] = sma - (num_std * std)
    df_copy['bb_position'] = (df_copy[column] - df_copy['bb_lower']) / (df_copy['bb_upper'] - df_copy['bb_lower'])
    
    return df_copy

def calculate_price_momentum(df: pd.DataFrame, column: str = 'close', windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Calcule les indicateurs de momentum des prix.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        column (str): Colonne sur laquelle calculer le momentum.
        windows (List[int]): Fenêtres pour calculer le momentum.
    
    Returns:
        pd.DataFrame: DataFrame enrichi avec les colonnes de momentum.
    """
    df_copy = df.copy()
    
    for window in windows:
        # Changement de prix relatif sur N jours
        df_copy[f'momentum_{window}'] = (df_copy[column] / df_copy[column].shift(window) - 1) * 100
        
        # Volatilité sur N jours
        df_copy[f'volatility_{window}'] = df_copy[column].pct_change().rolling(window=window).std() * 100
    
    return df_copy

def calculate_volume_features(df: pd.DataFrame, volume_col: str = 'volume', price_col: str = 'close', windows: List[int] = [10, 20]) -> pd.DataFrame:
    """
    Calcule les features basées sur le volume.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée.
        volume_col (str): Colonne du volume.
        price_col (str): Colonne du prix.
        windows (List[int]): Fenêtres pour les moyennes mobiles.
    
    Returns:
        pd.DataFrame: DataFrame enrichi avec les features de volume.
    """
    df_copy = df.copy()
    
    for window in windows:
        # Volume moyen
        df_copy[f'volume_sma_{window}'] = df_copy[volume_col].rolling(window=window).mean()
        # Ratio volume actuel / volume moyen
        df_copy[f'volume_ratio_{window}'] = df_copy[volume_col] / df_copy[f'volume_sma_{window}']
    
    # Volume pondéré par le prix (approximation du VWAP)
    df_copy['vwap_10'] = (df_copy[price_col] * df_copy[volume_col]).rolling(10).sum() / df_copy[volume_col].rolling(10).sum()
    
    return df_copy