# src/signal_generation/signal_generator.py
import pandas as pd
import numpy as np
from typing import List, Union

def generate_signals_from_predictions(
    predictions: Union[pd.Series, np.ndarray, List[float]],
    threshold: float = 0.5,
    prediction_type: str = 'probability' # 'probability' or 'class'
) -> pd.Series:
    """
    Génère des signaux de trading (BUY, SELL, HOLD) à partir des prédictions d'un modèle.

    Args:
        predictions (Union[pd.Series, np.ndarray, List[float]]): 
            Une série, un array NumPy ou une liste de prédictions.
            Si prediction_type est 'probability', ce sont des probabilités (ex: de hausse).
            Si prediction_type est 'class', ce sont des classes (ex: 0 pour SELL, 1 pour BUY).
        threshold (float, optional): 
            Le seuil pour convertir les probabilités en signaux. 
            Ignoré si prediction_type est 'class'. Par défaut à 0.5.
        prediction_type (str, optional):
            Le type de prédictions fournies. Peut être 'probability' ou 'class'.
            Par défaut à 'probability'.

    Returns:
        pd.Series: Une série de signaux de trading ('BUY', 'SELL', 'HOLD').
    """
    if not isinstance(predictions, pd.Series):
        predictions_series = pd.Series(predictions)
    else:
        predictions_series = predictions

    signals = pd.Series(['HOLD'] * len(predictions_series), index=predictions_series.index)

    if prediction_type == 'probability':
        signals[predictions_series > threshold] = 'BUY'
        signals[predictions_series < (1 - threshold)] = 'SELL' # Assuming symmetrical threshold for SELL
    elif prediction_type == 'class':
        # Assuming 1 means BUY and 0 means SELL. This might need adjustment
        # based on the specific model's target definition in models.py.
        signals[predictions_series == 1] = 'BUY'
        signals[predictions_series == 0] = 'SELL'
    else:
        raise ValueError("prediction_type must be 'probability' or 'class'")

    return signals

def allocate_capital_simple(
    signals: Union[pd.Series, List[str]],
    total_capital: float,
    fraction_per_trade: float = 0.1,
    risk_per_trade: float = None
) -> pd.Series:
    """
    Alloue une fraction fixe du capital total pour chaque signal de trading.

    Args:
        signals (Union[pd.Series, List[str]]): 
            Une série ou une liste de signaux de trading ('BUY', 'SELL', 'HOLD').
        total_capital (float): Le capital total disponible pour le trading.
        fraction_per_trade (float, optional): 
            La fraction du capital total à allouer par trade. 
            Par défaut à 0.1 (10%).
        risk_per_trade (float, optional):
            Alternative à fraction_per_trade. La proportion du capital total à risquer par trade.
            Si spécifié, remplace fraction_per_trade.

    Returns:
        pd.Series: Une série de tailles de position (montant de capital alloué).
                   Pour 'HOLD', la taille est 0.
                   Pour 'SELL', la valeur indique la fraction des positions à vendre (1.0 = 100%).
    """
    if not isinstance(signals, pd.Series):
        signals_series = pd.Series(signals)
    else:
        signals_series = signals
    
    # Utiliser risk_per_trade s'il est fourni
    actual_fraction = risk_per_trade if risk_per_trade is not None else fraction_per_trade

    position_sizes = pd.Series([0.0] * len(signals_series), index=signals_series.index)
    
    allocation_amount = total_capital * actual_fraction
    
    position_sizes[signals_series == 'BUY'] = allocation_amount
    # Pour SELL, on utilise 1.0 pour indiquer la vente complète des positions détenues
    # Le simulateur interprétera cela comme le nombre de shares à vendre
    position_sizes[signals_series == 'SELL'] = 1.0  # Fraction ou nombre d'actions à vendre
    
    return position_sizes

if __name__ == '__main__':
    # Exemple d'utilisation pour generate_signals_from_predictions (probabilités)
    sample_probabilities = pd.Series([0.1, 0.4, 0.6, 0.9, 0.5])
    print("Prédictions (probabilités):")
    print(sample_probabilities)
    signals_prob = generate_signals_from_predictions(sample_probabilities, threshold=0.55)
    print("\nSignaux générés (à partir de probabilités, seuil 0.55):")
    print(signals_prob)

    # Exemple d'utilisation pour generate_signals_from_predictions (classes)
    sample_classes = pd.Series([0, 1, 0, 1, 1]) # 0 pour VENDRE, 1 pour ACHETER
    print("\nPrédictions (classes):")
    print(sample_classes)
    signals_class = generate_signals_from_predictions(sample_classes, prediction_type='class')
    print("\nSignaux générés (à partir de classes):")
    print(signals_class)

    # Exemple d'utilisation pour allocate_capital_simple
    current_capital = 100000  # 100,000 EUR
    trade_fraction = 0.05     # 5% du capital par trade
    
    print(f"\nCapital total: {current_capital}, Fraction par trade: {trade_fraction}")
    
    allocations_prob = allocate_capital_simple(signals_prob, current_capital, trade_fraction)
    print("\nAllocations de capital (basées sur signaux prob):")
    print(allocations_prob)

    allocations_class = allocate_capital_simple(signals_class, current_capital, trade_fraction)
    print("\nAllocations de capital (basées sur signaux class):")
    print(allocations_class)

    # Cas où il n'y a que des HOLD
    hold_signals = pd.Series(['HOLD', 'HOLD', 'HOLD'])
    print("\nSignaux (uniquement HOLD):")
    print(hold_signals)
    allocations_hold = allocate_capital_simple(hold_signals, current_capital, trade_fraction)
    print("\nAllocations de capital (signaux HOLD):")
    print(allocations_hold)