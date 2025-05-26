#!/usr/bin/env python3
"""
Advanced Financial Metrics Module
Métriques financières avancées pour l'évaluation des stratégies de trading
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


class FinancialMetricsCalculator:
    """
    Calculateur de métriques financières avancées pour le trading algorithmique
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialise le calculateur de métriques
        
        Args:
            risk_free_rate: Taux sans risque annualisé (défaut: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, returns: pd.Series, 
                            benchmark_returns: Optional[pd.Series] = None,
                            prices: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calcule toutes les métriques financières disponibles
        
        Args:
            returns: Série des rendements quotidiens
            benchmark_returns: Rendements du benchmark (optionnel)
            prices: Série des prix (optionnel, pour certaines métriques)
            
        Returns:
            Dictionnaire contenant toutes les métriques
        """
        metrics = {}
        
        if returns.empty:
            return metrics
        
        # Métriques de rendement
        metrics.update(self._calculate_return_metrics(returns))
        
        # Métriques de risque
        metrics.update(self._calculate_risk_metrics(returns, prices))
        
        # Métriques ajustées au risque
        metrics.update(self._calculate_risk_adjusted_metrics(returns))
        
        # Métriques de drawdown
        if prices is not None:
            metrics.update(self._calculate_drawdown_metrics(prices))
        
        # Métriques de benchmark (si disponible)
        if benchmark_returns is not None:
            metrics.update(self._calculate_benchmark_metrics(returns, benchmark_returns))
        
        # Métriques de distribution
        metrics.update(self._calculate_distribution_metrics(returns))
        
        # Métriques de trading spécifiques
        metrics.update(self._calculate_trading_metrics(returns))
        
        return metrics
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de rendement"""
        
        metrics = {}
        
        # Rendements de base
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns.mean()) ** 252 - 1
        metrics['geometric_mean_return'] = stats.gmean(1 + returns) - 1 if (returns > -1).all() else np.nan
        
        # Rendements cumulés
        cumulative_returns = (1 + returns).cumprod()
        metrics['final_cumulative_return'] = cumulative_returns.iloc[-1] - 1
        
        # Statistiques de base
        metrics['mean_daily_return'] = returns.mean()
        metrics['median_daily_return'] = returns.median()
        
        return metrics
    
    def _calculate_risk_metrics(self, returns: pd.Series, 
                              prices: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calcule les métriques de risque"""
        
        metrics = {}
        
        # Volatilité
        metrics['daily_volatility'] = returns.std()
        metrics['annualized_volatility'] = returns.std() * np.sqrt(252)
        
        # Volatilité à la baisse (downside volatility)
        downside_returns = returns[returns < 0]
        metrics['downside_volatility'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Value at Risk (VaR)
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR/Expected Shortfall)
        var_95 = metrics['var_95']
        var_99 = metrics['var_99']
        metrics['cvar_95'] = returns[returns <= var_95].mean() if any(returns <= var_95) else 0
        metrics['cvar_99'] = returns[returns <= var_99].mean() if any(returns <= var_99) else 0
        
        # Semi-déviation
        mean_return = returns.mean()
        negative_deviations = returns[returns < mean_return] - mean_return
        metrics['semi_deviation'] = np.sqrt((negative_deviations ** 2).mean()) if len(negative_deviations) > 0 else 0
        
        return metrics
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques ajustées au risque"""
        
        metrics = {}
        
        # Sharpe Ratio
        excess_returns = returns.mean() - self.risk_free_rate / 252
        volatility = returns.std()
        metrics['sharpe_ratio'] = (excess_returns * 252) / (volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < self.risk_free_rate / 252]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        metrics['sortino_ratio'] = (excess_returns * 252) / (downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0
        
        # Calmar Ratio (nécessite les prix pour le drawdown)
        if 'max_drawdown' in metrics:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Modified Sharpe (avec VaR)
        var_95 = np.percentile(returns, 5)
        metrics['modified_sharpe'] = excess_returns * 252 / abs(var_95 * np.sqrt(252)) if var_95 != 0 else 0
        
        return metrics
    
    def _calculate_drawdown_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de drawdown"""
        
        metrics = {}
        
        # Calculer les drawdowns
        cumulative_max = prices.expanding().max()
        drawdowns = (prices - cumulative_max) / cumulative_max
        
        # Maximum Drawdown
        metrics['max_drawdown'] = drawdowns.min()
        
        # Drawdown moyen
        metrics['average_drawdown'] = drawdowns[drawdowns < 0].mean() if any(drawdowns < 0) else 0
        
        # Durée du maximum drawdown
        max_dd_end = drawdowns.idxmin()
        max_dd_start = prices[:max_dd_end].idxmax()
        metrics['max_drawdown_duration'] = (max_dd_end - max_dd_start).days if max_dd_end != max_dd_start else 0
        
        # Recovery factor
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        metrics['recovery_factor'] = total_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Calmar Ratio (maintenant qu'on a le max drawdown)
        annualized_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1
        metrics['calmar_ratio'] = annualized_return / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Ulcer Index
        squared_drawdowns = drawdowns ** 2
        metrics['ulcer_index'] = np.sqrt(squared_drawdowns.mean())
        
        return metrics
    
    def _calculate_benchmark_metrics(self, returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques relatives au benchmark"""
        
        metrics = {}
        
        # Alpha et Beta
        if len(returns) == len(benchmark_returns):
            # Régression linéaire pour Beta
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            metrics['beta'] = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Alpha (Jensen's Alpha)
            portfolio_return = returns.mean() * 252
            benchmark_return = benchmark_returns.mean() * 252
            metrics['alpha'] = portfolio_return - (self.risk_free_rate + metrics['beta'] * (benchmark_return - self.risk_free_rate))
            
            # Information Ratio
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            metrics['information_ratio'] = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            # Tracking Error
            metrics['tracking_error'] = tracking_error
            
            # Up/Down Capture Ratios
            up_market = benchmark_returns > 0
            down_market = benchmark_returns < 0
            
            if up_market.any():
                up_capture = (returns[up_market].mean() / benchmark_returns[up_market].mean()) if benchmark_returns[up_market].mean() != 0 else 0
                metrics['up_capture_ratio'] = up_capture
            
            if down_market.any():
                down_capture = (returns[down_market].mean() / benchmark_returns[down_market].mean()) if benchmark_returns[down_market].mean() != 0 else 0
                metrics['down_capture_ratio'] = down_capture
        
        return metrics
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de distribution des rendements"""
        
        metrics = {}
        
        # Moments statistiques
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        metrics['excess_kurtosis'] = metrics['kurtosis'] - 3
        
        # Tests de normalité
        _, p_value_shapiro = stats.shapiro(returns[:5000] if len(returns) > 5000 else returns)  # Limite pour Shapiro-Wilk
        metrics['shapiro_p_value'] = p_value_shapiro
        metrics['is_normal_distribution'] = p_value_shapiro > 0.05
        
        # Jarque-Bera test
        jb_stat, jb_p_value = stats.jarque_bera(returns)
        metrics['jarque_bera_p_value'] = jb_p_value
        
        # Tail ratio
        percentile_95 = np.percentile(returns, 95)
        percentile_5 = np.percentile(returns, 5)
        metrics['tail_ratio'] = abs(percentile_95) / abs(percentile_5) if percentile_5 != 0 else 0
        
        return metrics
    
    def _calculate_trading_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calcule les métriques spécifiques au trading"""
        
        metrics = {}
        
        # Win Rate (approximation basée sur les rendements positifs)
        positive_returns = returns > 0
        metrics['win_rate'] = positive_returns.mean()
        
        # Average Win/Loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        metrics['average_win'] = wins.mean() if len(wins) > 0 else 0
        metrics['average_loss'] = losses.mean() if len(losses) > 0 else 0
        
        # Profit Factor
        total_wins = wins.sum() if len(wins) > 0 else 0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        # Expectancy
        win_rate = metrics['win_rate']
        avg_win = metrics['average_win']
        avg_loss = abs(metrics['average_loss'])
        metrics['expectancy'] = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Consecutive wins/losses
        returns_binary = (returns > 0).astype(int)
        runs = []
        current_run = 1
        
        for i in range(1, len(returns_binary)):
            if returns_binary[i] == returns_binary[i-1]:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
        runs.append(current_run)
        
        metrics['max_consecutive_wins'] = max([run for i, run in enumerate(runs) if returns_binary[i] == 1], default=0)
        metrics['max_consecutive_losses'] = max([run for i, run in enumerate(runs) if returns_binary[i] == 0], default=0)
        
        return metrics
    
    def calculate_information_coefficient(self, predictions: np.ndarray, 
                                        actual_returns: np.ndarray) -> float:
        """
        Calcule l'Information Coefficient (corrélation de rang)
        
        Args:
            predictions: Prédictions du modèle
            actual_returns: Rendements réels
            
        Returns:
            Information Coefficient
        """
        if len(predictions) != len(actual_returns):
            return 0.0
        
        try:
            ic, _ = stats.spearmanr(predictions, actual_returns)
            return ic if not np.isnan(ic) else 0.0
        except:
            return 0.0
    
    def calculate_hit_rate(self, predictions: np.ndarray, 
                          actual_returns: np.ndarray,
                          threshold: float = 0.0) -> float:
        """
        Calcule le taux de réussite directionnel
        
        Args:
            predictions: Prédictions du modèle
            actual_returns: Rendements réels
            threshold: Seuil pour considérer une prédiction correcte
            
        Returns:
            Hit rate (taux de réussite)
        """
        if len(predictions) != len(actual_returns):
            return 0.0
        
        pred_direction = predictions > threshold
        actual_direction = actual_returns > threshold
        
        return (pred_direction == actual_direction).mean()
    
    def calculate_maximum_adverse_excursion(self, prices: pd.Series, 
                                          entry_points: List[int],
                                          exit_points: List[int],
                                          position_type: str = 'long') -> Dict[str, float]:
        """
        Calcule la Maximum Adverse Excursion (MAE)
        
        Args:
            prices: Série des prix
            entry_points: Indices des points d'entrée
            exit_points: Indices des points de sortie
            position_type: 'long' ou 'short'
            
        Returns:
            Métriques MAE
        """
        if len(entry_points) != len(exit_points):
            return {}
        
        mae_values = []
        mfe_values = []  # Maximum Favorable Excursion
        
        for entry, exit in zip(entry_points, exit_points):
            if entry >= len(prices) or exit >= len(prices) or entry >= exit:
                continue
            
            entry_price = prices.iloc[entry]
            exit_price = prices.iloc[exit]
            trade_prices = prices.iloc[entry:exit+1]
            
            if position_type == 'long':
                # Pour une position longue
                mae = (trade_prices.min() - entry_price) / entry_price  # Pire excursion (négative)
                mfe = (trade_prices.max() - entry_price) / entry_price  # Meilleure excursion (positive)
            else:
                # Pour une position courte
                mae = (entry_price - trade_prices.max()) / entry_price  # Pire excursion (négative)
                mfe = (entry_price - trade_prices.min()) / entry_price  # Meilleure excursion (positive)
            
            mae_values.append(mae)
            mfe_values.append(mfe)
        
        return {
            'average_mae': np.mean(mae_values) if mae_values else 0,
            'max_mae': min(mae_values) if mae_values else 0,  # Plus négatif
            'average_mfe': np.mean(mfe_values) if mfe_values else 0,
            'max_mfe': max(mfe_values) if mfe_values else 0,
            'mae_to_profit_ratio': abs(np.mean(mae_values)) / np.mean(mfe_values) if mfe_values and np.mean(mfe_values) != 0 else 0
        }
    
    def generate_performance_report(self, metrics: Dict[str, float]) -> str:
        """
        Génère un rapport de performance lisible
        
        Args:
            metrics: Dictionnaire des métriques calculées
            
        Returns:
            Rapport formaté en string
        """
        report = []
        report.append("=" * 80)
        report.append("RAPPORT DE PERFORMANCE DÉTAILLÉ")
        report.append("=" * 80)
        
        # Métriques de rendement
        if any(key in metrics for key in ['total_return', 'annualized_return']):
            report.append("\n📈 MÉTRIQUES DE RENDEMENT:")
            report.append("-" * 40)
            if 'total_return' in metrics:
                report.append(f"Rendement Total:        {metrics['total_return']:>12.2%}")
            if 'annualized_return' in metrics:
                report.append(f"Rendement Annualisé:    {metrics['annualized_return']:>12.2%}")
            if 'geometric_mean_return' in metrics:
                report.append(f"Moyenne Géométrique:    {metrics['geometric_mean_return']:>12.2%}")
        
        # Métriques de risque
        if any(key in metrics for key in ['annualized_volatility', 'max_drawdown']):
            report.append("\n📉 MÉTRIQUES DE RISQUE:")
            report.append("-" * 40)
            if 'annualized_volatility' in metrics:
                report.append(f"Volatilité Annuelle:    {metrics['annualized_volatility']:>12.2%}")
            if 'max_drawdown' in metrics:
                report.append(f"Drawdown Maximum:       {metrics['max_drawdown']:>12.2%}")
            if 'var_95' in metrics:
                report.append(f"VaR 95%:               {metrics['var_95']:>12.2%}")
        
        # Métriques ajustées au risque
        if any(key in metrics for key in ['sharpe_ratio', 'sortino_ratio']):
            report.append("\n⚖️ MÉTRIQUES AJUSTÉES AU RISQUE:")
            report.append("-" * 40)
            if 'sharpe_ratio' in metrics:
                report.append(f"Ratio de Sharpe:        {metrics['sharpe_ratio']:>12.3f}")
            if 'sortino_ratio' in metrics:
                report.append(f"Ratio de Sortino:       {metrics['sortino_ratio']:>12.3f}")
            if 'calmar_ratio' in metrics:
                report.append(f"Ratio de Calmar:        {metrics['calmar_ratio']:>12.3f}")
        
        # Métriques de benchmark
        if any(key in metrics for key in ['alpha', 'beta', 'information_ratio']):
            report.append("\n📊 MÉTRIQUES VS BENCHMARK:")
            report.append("-" * 40)
            if 'alpha' in metrics:
                report.append(f"Alpha:                  {metrics['alpha']:>12.2%}")
            if 'beta' in metrics:
                report.append(f"Beta:                   {metrics['beta']:>12.3f}")
            if 'information_ratio' in metrics:
                report.append(f"Information Ratio:      {metrics['information_ratio']:>12.3f}")
        
        # Métriques de trading
        if any(key in metrics for key in ['win_rate', 'profit_factor']):
            report.append("\n🎯 MÉTRIQUES DE TRADING:")
            report.append("-" * 40)
            if 'win_rate' in metrics:
                report.append(f"Taux de Réussite:       {metrics['win_rate']:>12.2%}")
            if 'profit_factor' in metrics:
                report.append(f"Facteur de Profit:      {metrics['profit_factor']:>12.3f}")
            if 'expectancy' in metrics:
                report.append(f"Espérance:              {metrics['expectancy']:>12.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def calculate_portfolio_metrics(portfolio_history: Union[List[Dict], pd.DataFrame],
                              benchmark_data: Optional[pd.DataFrame] = None,
                              initial_capital: float = 100000) -> Dict[str, float]:
    """
    Fonction utilitaire pour calculer les métriques d'un portefeuille
    
    Args:
        portfolio_history: Historique du portefeuille
        benchmark_data: Données du benchmark (optionnel)
        initial_capital: Capital initial
        
    Returns:
        Dictionnaire des métriques calculées
    """
    if isinstance(portfolio_history, list):
        portfolio_df = pd.DataFrame(portfolio_history)
    else:
        portfolio_df = portfolio_history
    
    if portfolio_df.empty or 'portfolio_value' not in portfolio_df.columns:
        return {}
    
    # Calculer les rendements
    portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change().dropna()
    returns = portfolio_df['returns'].dropna()
    
    # Calculer les prix normalisés
    prices = portfolio_df['portfolio_value'] / initial_capital
    
    # Initialiser le calculateur
    calculator = FinancialMetricsCalculator()
    
    # Calculer les métriques de benchmark si disponible
    benchmark_returns = None
    if benchmark_data is not None and 'close' in benchmark_data.columns:
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        # Aligner les séries
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns.iloc[:min_len]
        benchmark_returns = benchmark_returns.iloc[:min_len]
    
    # Calculer toutes les métriques
    metrics = calculator.calculate_all_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        prices=prices
    )
    
    return metrics
