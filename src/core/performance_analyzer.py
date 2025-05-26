"""
Module pour sauvegarder et analyser les résultats de backtest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class BacktestAnalyzer:
    """Classe pour analyser et visualiser les résultats de backtest"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def save_results(self, portfolio_history, trade_history, signals_df, market_data, 
                    initial_capital, strategy_name="AlphaBeta808"):
        """Sauvegarde les résultats du backtest"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convertir en DataFrames si nécessaire
        if isinstance(portfolio_history, list):
            portfolio_df = pd.DataFrame(portfolio_history)
        else:
            portfolio_df = portfolio_history
            
        if isinstance(trade_history, list):
            trades_df = pd.DataFrame(trade_history)
        else:
            trades_df = trade_history
        
        # Calculer les métriques de performance
        metrics = self._calculate_metrics(portfolio_df, initial_capital, market_data)
        
        # Sauvegarder les fichiers
        portfolio_df.to_csv(f"{self.results_dir}/portfolio_history_{timestamp}.csv", index=False)
        trades_df.to_csv(f"{self.results_dir}/trades_history_{timestamp}.csv", index=False)
        signals_df.to_csv(f"{self.results_dir}/signals_{timestamp}.csv")
        
        # Sauvegarder les métriques
        with open(f"{self.results_dir}/metrics_{timestamp}.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Créer la visualisation
        self._create_visualizations(portfolio_df, trades_df, signals_df, 
                                  market_data, metrics, timestamp)
        
        print(f"Résultats sauvegardés dans {self.results_dir}/ avec timestamp {timestamp}")
        return metrics
    
    def _calculate_metrics(self, portfolio_df, initial_capital, market_data):
        """Calcule les métriques de performance"""
        
        if portfolio_df.empty:
            return {}
        
        final_value = portfolio_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Calculs temporels
        days_traded = len(portfolio_df) - 1
        years_traded = days_traded / 365.25 if days_traded > 0 else 1
        annualized_return = (final_value / initial_capital) ** (1/years_traded) - 1 if years_traded > 0 else 0
        
        # Volatilité et Sharpe
        portfolio_df = portfolio_df.copy()
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        daily_vol = portfolio_df['daily_return'].std()
        annualized_vol = daily_vol * np.sqrt(365.25) if daily_vol > 0 else 0
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum Drawdown
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = portfolio_df['drawdown'].min()
        
        # Buy and Hold de référence
        if not market_data.empty and 'close' in market_data.columns:
            initial_price = market_data['close'].iloc[0]
            final_price = market_data['close'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price
            alpha = total_return - buy_hold_return
        else:
            buy_hold_return = 0
            alpha = total_return
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'days_traded': days_traded,
            'buy_hold_return': buy_hold_return,
            'alpha': alpha,
            'win_rate': None,  # Sera calculé si on a les trades
            'total_trades': 0
        }
    
    def _create_visualizations(self, portfolio_df, trades_df, signals_df, 
                             market_data, metrics, timestamp):
        """Crée des visualisations des résultats"""
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AlphaBeta808Trading - Analyse de Performance', fontsize=16, fontweight='bold')
        
        # 1. Évolution de la valeur du portefeuille
        if 'timestamp' in portfolio_df.columns:
            portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
            axes[0, 0].plot(portfolio_df['timestamp'], portfolio_df['portfolio_value'], 
                           label='Stratégie AlphaBeta808', linewidth=2, color='#1f77b4')
            
            # Ajouter buy and hold si possible
            if not market_data.empty and 'close' in market_data.columns:
                initial_shares = metrics['initial_capital'] / market_data['close'].iloc[0]
                benchmark_values = market_data['close'] * initial_shares
                
                # Aligner les dates
                if hasattr(market_data.index, 'to_pydatetime'):
                    market_dates = market_data.index
                elif 'timestamp' in market_data.columns:
                    market_dates = pd.to_datetime(market_data['timestamp'])
                else:
                    market_dates = portfolio_df['timestamp']
                
                axes[0, 0].plot(market_dates, benchmark_values, 
                               label='Buy & Hold BTC', linewidth=2, color='#ff7f0e', alpha=0.8)
        
        axes[0, 0].set_title('Évolution de la Valeur du Portefeuille')
        axes[0, 0].set_ylabel('Valeur ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Distribution des rendements quotidiens
        if 'daily_return' not in portfolio_df.columns:
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        daily_returns = portfolio_df['daily_return'].dropna()
        if len(daily_returns) > 0:
            axes[0, 1].hist(daily_returns, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
            axes[0, 1].axvline(daily_returns.mean(), color='red', linestyle='--', 
                              label=f'Moyenne: {daily_returns.mean():.4f}')
        
        axes[0, 1].set_title('Distribution des Rendements Quotidiens')
        axes[0, 1].set_xlabel('Rendement')
        axes[0, 1].set_ylabel('Fréquence')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        if 'drawdown' not in portfolio_df.columns:
            portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        
        if 'timestamp' in portfolio_df.columns:
            axes[1, 0].fill_between(portfolio_df['timestamp'], portfolio_df['drawdown'], 0, 
                                   alpha=0.7, color='red')
        else:
            axes[1, 0].fill_between(range(len(portfolio_df)), portfolio_df['drawdown'], 0, 
                                   alpha=0.7, color='red')
        
        axes[1, 0].set_title('Drawdown de la Stratégie')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Tableau des métriques
        metrics_display = [
            ['Rendement Total', f"{metrics.get('total_return', 0)*100:.2f}%"],
            ['Rendement Annualisé', f"{metrics.get('annualized_return', 0)*100:.2f}%"],
            ['Volatilité', f"{metrics.get('annualized_volatility', 0)*100:.2f}%"],
            ['Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.3f}"],
            ['Max Drawdown', f"{metrics.get('max_drawdown', 0)*100:.2f}%"],
            ['Alpha vs B&H', f"{metrics.get('alpha', 0)*100:.2f}%"],
            ['Jours de trading', f"{metrics.get('days_traded', 0)}"],
            ['Nombre de trades', f"{len(trades_df) if not trades_df.empty else 0}"]
        ]
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=metrics_display,
                                 colLabels=['Métrique', 'Valeur'],
                                 cellLoc='left',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Métriques de Performance')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique sauvegardé: {self.results_dir}/analysis_{timestamp}.png")

def print_performance_summary(metrics):
    """Affiche un résumé de performance formaté"""
    
    print("\n" + "="*70)
    print("RAPPORT DE PERFORMANCE ALPH8BETA808TRADING")
    print("="*70)
    print(f"Capital Initial:      ${metrics.get('initial_capital', 0):>12,.2f}")
    print(f"Valeur Finale:        ${metrics.get('final_value', 0):>12,.2f}")
    print(f"Rendement Total:      {metrics.get('total_return', 0)*100:>12.2f}%")
    print(f"Rendement Annualisé:  {metrics.get('annualized_return', 0)*100:>12.2f}%")
    print(f"Volatilité:           {metrics.get('annualized_volatility', 0)*100:>12.2f}%")
    print(f"Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>12.3f}")
    print(f"Maximum Drawdown:     {metrics.get('max_drawdown', 0)*100:>12.2f}%")
    print("-"*70)
    print("COMPARAISON vs BUY & HOLD:")
    print(f"Alpha:                {metrics.get('alpha', 0)*100:>12.2f}%")
    print(f"Buy & Hold Return:    {metrics.get('buy_hold_return', 0)*100:>12.2f}%")
    print("-"*70)
    print(f"Jours de trading:     {metrics.get('days_traded', 0):>12}")
    print(f"Nombre de trades:     {metrics.get('total_trades', 0):>12}")
    print("="*70)
