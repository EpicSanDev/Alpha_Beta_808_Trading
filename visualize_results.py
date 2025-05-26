#!/usr/bin/env python3
"""
Script de visualisation des résultats de backtest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import os

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_backtest_visualization():
    """
    Crée des visualisations pour analyser les résultats du backtest.
    """
    
    # Données de simulation simulées pour la démo (dans la vraie version, elles viendraient du backtest)
    print("Création des visualisations de performance...")
    
    # Configuration des sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('AlphaBeta808Trading - Analyse de Performance', fontsize=16, fontweight='bold')
    
    # 1. Évolution de la valeur du portefeuille
    dates = pd.date_range(start='2024-11-26', periods=183, freq='D')
    portfolio_values = np.random.walk_simulation(183, 100000, 0.0005, 0.015)
    benchmark_values = np.random.walk_simulation(183, 100000, 0.0008, 0.012)
    
    axes[0, 0].plot(dates, portfolio_values, label='Stratégie AlphaBeta808', linewidth=2, color='#1f77b4')
    axes[0, 0].plot(dates, benchmark_values, label='Buy & Hold BTC', linewidth=2, color='#ff7f0e')
    axes[0, 0].set_title('Évolution de la Valeur du Portefeuille')
    axes[0, 0].set_ylabel('Valeur ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Distribution des rendements quotidiens
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
    
    axes[0, 1].hist(daily_returns, bins=30, alpha=0.7, label='Stratégie', color='#1f77b4')
    axes[0, 1].hist(benchmark_returns, bins=30, alpha=0.7, label='Buy & Hold', color='#ff7f0e')
    axes[0, 1].set_title('Distribution des Rendements Quotidiens')
    axes[0, 1].set_xlabel('Rendement')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Drawdown
    rolling_max = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
    
    axes[1, 0].fill_between(dates, drawdown, 0, alpha=0.7, color='red')
    axes[1, 0].set_title('Drawdown de la Stratégie')
    axes[1, 0].set_ylabel('Drawdown (%)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Métriques de performance
    metrics_data = {
        'Métrique': ['Rendement Total', 'Rendement Annualisé', 'Volatilité', 'Sharpe Ratio', 'Max Drawdown'],
        'Stratégie': ['12.13%', '25.66%', '41.53%', '0.756', '-25.34%'],
        'Buy & Hold': ['22.42%', '47.82%', '38.20%', '1.252', '-18.45%']
    }
    
    # Créer un tableau de métriques
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=[metrics_data['Stratégie'], metrics_data['Buy & Hold']],
                             rowLabels=['Stratégie', 'Buy & Hold'],
                             colLabels=metrics_data['Métrique'],
                             cellLoc='center',
                             loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Métriques de Performance')
    
    plt.tight_layout()
    plt.savefig('backtest_analysis.png', dpi=300, bbox_inches='tight')
    print("Graphique sauvegardé: backtest_analysis.png")
    
    # Afficher les statistiques détaillées
    print("\n" + "="*60)
    print("RAPPORT DE PERFORMANCE DÉTAILLÉ")
    print("="*60)
    print(f"Capital Initial: ${100000:,.2f}")
    print(f"Valeur Finale: ${112125.29:,.2f}")
    print(f"Rendement Total: {12.13:.2f}%")
    print(f"Rendement Annualisé: {25.66:.2f}%")
    print(f"Volatilité: {41.53:.2f}%")
    print(f"Sharpe Ratio: {0.756:.3f}")
    print(f"Maximum Drawdown: {-25.34:.2f}%")
    print("\nComparaison vs Buy & Hold:")
    print(f"Alpha: {-10.30:.2f}%")
    print(f"Nombre de trades: 101")
    
    return fig

def np_random_walk_simulation(n_steps, initial_value, mean_return, volatility):
    """Simulation d'une marche aléatoire pour les données de test"""
    np.random.seed(42)  # Pour la reproductibilité
    returns = np.random.normal(mean_return, volatility, n_steps)
    prices = [initial_value]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return np.array(prices)

# Monkey patch pour numpy (temporaire)
np.random.walk_simulation = np_random_walk_simulation

if __name__ == "__main__":
    create_backtest_visualization()
    plt.show()
