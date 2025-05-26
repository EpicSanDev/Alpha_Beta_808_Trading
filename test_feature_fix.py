#!/usr/bin/env python3
"""
Script de test pour vérifier que le correctif des features fonctionne
"""

import sys
import os
sys.path.append('/Users/bastienjavaux/Desktop/AlphaBeta808Trading/src')

try:
    # Test d'import des modules nécessaires
    print("=== Test d'import des modules ===")
    from src.feature_engineering.technical_features import calculate_sma, calculate_ema, calculate_rsi
    print("✓ Modules de feature engineering importés")
    
    # Test avec des données simulées
    print("\n=== Test de calcul des features ===")
    import pandas as pd
    import numpy as np
    
    # Créer des données de test plus longues pour éviter les NaN
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    data = {
        'timestamp': dates,
        'open': np.random.rand(50) * 10 + 100,
        'high': np.random.rand(50) * 10 + 105,
        'low': np.random.rand(50) * 10 + 95,
        'close': np.random.rand(50) * 10 + 100,
        'volume': np.random.rand(50) * 1000 + 500
    }
    test_data = pd.DataFrame(data)
    print(f"Données créées: {len(test_data)} lignes")
    
    # Test des fonctions individuelles
    print("\n=== Test des fonctions de features ===")
    result = calculate_sma(test_data, windows=[10])
    print(f"SMA calculé: {'sma_10' in result.columns}")
    
    result = calculate_ema(result, windows=[10])
    print(f"EMA calculé: {'ema_10' in result.columns}")
    
    result = calculate_rsi(result, window=14)
    print(f"RSI calculé: {'rsi_14' in result.columns}")
    
    # Ajouter other_feature
    if 'sma_10' in result.columns and 'ema_10' in result.columns and 'rsi_14' in result.columns:
        result['other_feature'] = (
            (result['close'] - result['sma_10']) / result['sma_10'] * 100 +
            (result['close'] - result['ema_10']) / result['ema_10'] * 100 +
            (result['rsi_14'] - 50) / 50
        ) / 3
        print(f"Other feature créé: {'other_feature' in result.columns}")
    
    # Nettoyer les NaN
    result = result.dropna()
    print(f"Données après nettoyage: {len(result)} lignes")
    
    # Vérifier les features attendues
    expected_features = ['sma_10', 'ema_10', 'rsi_14', 'other_feature']
    missing = [f for f in expected_features if f not in result.columns]
    
    print(f"\n=== Résultat final ===")
    print(f"Features attendues: {expected_features}")
    print(f"Features présentes: {[f for f in expected_features if f in result.columns]}")
    print(f"Features manquantes: {missing}")
    
    if not missing:
        print("✓ SUCCÈS: Toutes les features attendues sont présentes!")
        print("\nExemple de données:")
        print(result[expected_features].tail())
    else:
        print("✗ ÉCHEC: Des features sont manquantes")
        
except Exception as e:
    print(f"Erreur: {e}")
    import traceback
    traceback.print_exc()
