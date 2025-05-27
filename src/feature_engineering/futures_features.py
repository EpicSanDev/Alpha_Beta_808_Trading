import pandas as pd
from typing import List

def calculate_open_interest_features(df: pd.DataFrame, open_interest_col: str = 'open_interest', volume_col: str = 'volume', windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Calcule les caractéristiques basées sur l'Open Interest.

    Args:
        df (pd.DataFrame): DataFrame d'entrée. Doit contenir les colonnes spécifiées
                           par open_interest_col et volume_col.
        open_interest_col (str): Nom de la colonne de l'Open Interest (par défaut 'open_interest').
        volume_col (str): Nom de la colonne du volume (par défaut 'volume').
        windows (List[int]): Liste des fenêtres pour les variations et moyennes mobiles.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les nouvelles caractéristiques d'Open Interest.
    """
    df_copy = df.copy()

    # Open Interest brut (est déjà présent, mais on s'assure qu'il est là)
    if open_interest_col not in df_copy.columns:
        raise ValueError(f"La colonne {open_interest_col} est manquante dans le DataFrame.")

    # Variation de l'Open Interest (absolue et en pourcentage)
    for window in windows:
        df_copy[f'oi_change_{window}'] = df_copy[open_interest_col].diff(window)
        df_copy[f'oi_pct_change_{window}'] = df_copy[open_interest_col].pct_change(periods=window) * 100

    # Moyennes mobiles de l'Open Interest
    for window in windows:
        df_copy[f'oi_sma_{window}'] = df_copy[open_interest_col].rolling(window=window).mean()
        df_copy[f'oi_ema_{window}'] = df_copy[open_interest_col].ewm(span=window, adjust=False).mean()

    # Ratio Open Interest / Volume
    if volume_col in df_copy.columns:
        # Remplacer les volumes nuls ou NaN par une petite valeur pour éviter la division par zéro
        # ou s'assurer que le volume est positif.
        # Une meilleure approche pourrait être de laisser NaN si le volume est 0 ou NaN.
        volume_series = df_copy[volume_col].replace(0, float('nan'))
        df_copy['oi_volume_ratio'] = df_copy[open_interest_col] / volume_series
        # Gérer les infinis potentiels si open_interest est non nul et volume_series était 0 (maintenant NaN)
        # df_copy['oi_volume_ratio'].replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    else:
        print(f"Avertissement : La colonne de volume '{volume_col}' est manquante. Le ratio OI/Volume ne sera pas calculé.")

    return df_copy

def calculate_funding_rate_features(df: pd.DataFrame, funding_rate_col: str = 'funding_rate', windows: List[int] = [8, 24, 72]) -> pd.DataFrame:
    """
    Calcule les caractéristiques basées sur le Funding Rate.
    Les fenêtres sont typiquement en nombre de périodes de funding (ex: toutes les 8 heures).

    Args:
        df (pd.DataFrame): DataFrame d'entrée. Doit contenir la colonne funding_rate_col.
        funding_rate_col (str): Nom de la colonne du Funding Rate (par défaut 'funding_rate').
        windows (List[int]): Liste des fenêtres pour les moyennes mobiles et autres indicateurs.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les nouvelles caractéristiques de Funding Rate.
    """
    df_copy = df.copy()

    if funding_rate_col not in df_copy.columns:
        raise ValueError(f"La colonne {funding_rate_col} est manquante dans le DataFrame.")

    # Funding Rate brut (déjà présent)

    # Moyennes mobiles du Funding Rate
    for window in windows:
        df_copy[f'fr_sma_{window}'] = df_copy[funding_rate_col].rolling(window=window).mean()
        df_copy[f'fr_ema_{window}'] = df_copy[funding_rate_col].ewm(span=window, adjust=False).mean()

    # Momentum du Funding Rate
    for window in windows:
        df_copy[f'fr_momentum_{window}'] = df_copy[funding_rate_col].diff(window)

    # Indicateur de sentiment simple basé sur le signe et l'amplitude (exemple)
    # Pourrait être affiné, par exemple, en regardant les changements de signe consécutifs
    df_copy['fr_sentiment_simple'] = df_copy[funding_rate_col].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    return df_copy

def calculate_basis_features(df: pd.DataFrame, basis_col: str = 'basis', price_col: str = 'close', windows: List[int] = [10, 20, 50]) -> pd.DataFrame:
    """
    Calcule les caractéristiques basées sur le Basis (Mark Price - Spot Price).

    Args:
        df (pd.DataFrame): DataFrame d'entrée. Doit contenir basis_col et price_col.
        basis_col (str): Nom de la colonne du Basis (par défaut 'basis').
        price_col (str): Nom de la colonne du prix de référence (par défaut 'close').
        windows (List[int]): Liste des fenêtres pour les moyennes mobiles et autres indicateurs.

    Returns:
        pd.DataFrame: DataFrame enrichi avec les nouvelles caractéristiques de Basis.
    """
    df_copy = df.copy()

    if basis_col not in df_copy.columns:
        raise ValueError(f"La colonne {basis_col} est manquante dans le DataFrame.")
    if price_col not in df_copy.columns:
        # Nécessaire pour certaines features comme le basis normalisé
        print(f"Avertissement : La colonne de prix '{price_col}' est manquante. Certaines features du basis pourraient ne pas être calculées.")


    # Basis brut (déjà présent)

    # Moyennes mobiles du Basis
    for window in windows:
        df_copy[f'basis_sma_{window}'] = df_copy[basis_col].rolling(window=window).mean()
        df_copy[f'basis_ema_{window}'] = df_copy[basis_col].ewm(span=window, adjust=False).mean()

    # Volatilité du Basis
    for window in windows:
        df_copy[f'basis_volatility_{window}'] = df_copy[basis_col].rolling(window=window).std()

    # Basis normalisé (en pourcentage du prix)
    if price_col in df_copy.columns:
        # S'assurer que le prix n'est pas nul pour éviter la division par zéro
        price_series = df_copy[price_col].replace(0, float('nan'))
        df_copy['basis_normalized'] = (df_copy[basis_col] / price_series) * 100
        # df_copy['basis_normalized'].replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Convergence/Divergence du Basis (exemple simple: diff de moyennes mobiles du basis)
    if len(windows) >= 2:
        short_window = windows[0]
        long_window = windows[-1] # Prend la plus longue fenêtre pour la divergence
        if f'basis_sma_{short_window}' in df_copy.columns and f'basis_sma_{long_window}' in df_copy.columns:
            df_copy[f'basis_convergence_sma_{short_window}_{long_window}'] = df_copy[f'basis_sma_{short_window}'] - df_copy[f'basis_sma_{long_window}']

    return df_copy

# TODO: Ajouter des fonctions pour les caractéristiques de Volume spécifiques aux Futures (si nécessaire)
# TODO: Ajouter des fonctions pour les caractéristiques combinées

def add_all_futures_features(df: pd.DataFrame,
                             open_interest_col: str = 'open_interest',
                             funding_rate_col: str = 'funding_rate',
                             basis_col: str = 'basis',
                             volume_col: str = 'volume', # Volume des klines, peut être le volume des futures
                             price_col: str = 'close',
                             oi_windows: List[int] = [5, 10, 20],
                             fr_windows: List[int] = [8, 24, 72], # Périodes de funding
                             basis_windows: List[int] = [10, 20, 50]
                            ) -> pd.DataFrame:
    """
    Ajoute toutes les caractéristiques de futures au DataFrame.
    C'est une fonction d'orchestration.

    Args:
        df (pd.DataFrame): DataFrame d'entrée avec les données prétraitées
                           (OHLCV, Open Interest, Funding Rate, Basis).
        open_interest_col (str): Colonne Open Interest.
        funding_rate_col (str): Colonne Funding Rate.
        basis_col (str): Colonne Basis.
        volume_col (str): Colonne Volume.
        price_col (str): Colonne Prix (pour normalisation, etc.).
        oi_windows (List[int]): Fenêtres pour les features d'Open Interest.
        fr_windows (List[int]): Fenêtres pour les features de Funding Rate.
        basis_windows (List[int]): Fenêtres pour les features de Basis.

    Returns:
        pd.DataFrame: DataFrame enrichi avec toutes les caractéristiques de futures.
    """
    df_featured = df.copy()

    if open_interest_col in df_featured.columns and volume_col in df_featured.columns:
        df_featured = calculate_open_interest_features(df_featured,
                                                       open_interest_col=open_interest_col,
                                                       volume_col=volume_col,
                                                       windows=oi_windows)
    elif open_interest_col in df_featured.columns:
        # Calculer sans le ratio OI/Volume si le volume manque
        df_featured = calculate_open_interest_features(df_featured,
                                                       open_interest_col=open_interest_col,
                                                       volume_col="dummy_volume_col_non_existent", # pour éviter erreur
                                                       windows=oi_windows)
        print(f"Avertissement: Colonne '{volume_col}' manquante, certaines features d'OI ne seront pas calculées.")


    if funding_rate_col in df_featured.columns:
        df_featured = calculate_funding_rate_features(df_featured,
                                                      funding_rate_col=funding_rate_col,
                                                      windows=fr_windows)
    else:
        print(f"Avertissement: Colonne '{funding_rate_col}' manquante, les features de FR ne seront pas calculées.")

    if basis_col in df_featured.columns:
        df_featured = calculate_basis_features(df_featured,
                                               basis_col=basis_col,
                                               price_col=price_col,
                                               windows=basis_windows)
    else:
        print(f"Avertissement: Colonne '{basis_col}' manquante, les features de Basis ne seront pas calculées.")

    # Ici, on pourrait ajouter des appels à d'autres fonctions pour des features de volume spécifiques aux futures
    # ou des features combinées.

    return df_featured