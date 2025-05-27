# -*- coding: utf-8 -*-
"""
Module for calculating financial and performance metrics specific to futures trading strategies.
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# TODO: Implement futures-specific metrics as discussed in Phase 4.

def calculate_profit_per_contract(trades_df: pd.DataFrame, contract_size: float = 1.0) -> float:
    """
    Calculates the average profit (or loss) per contract traded.

    Args:
        trades_df (pd.DataFrame): DataFrame of trades. Expected columns:
                                  'pnl' (profit and loss of the trade)
                                  'quantity' (number of contracts traded, absolute value).
        contract_size (float): The size of one contract (e.g., 0.01 for BTC if quantity is in BTC).
                               This is used to normalize the quantity if 'quantity' represents
                               the underlying amount rather than number of contracts.
                               If 'quantity' is already in number of contracts, set to 1.0.

    Returns:
        float: Average profit per contract. Returns np.nan if no trades or total quantity is zero.
    """
    if trades_df.empty or 'pnl' not in trades_df.columns or 'quantity' not in trades_df.columns:
        return np.nan

    total_pnl = trades_df['pnl'].sum()
    total_contracts_traded = trades_df['quantity'].abs().sum() / contract_size

    if total_contracts_traded == 0:
        return np.nan

    return total_pnl / total_contracts_traded

def calculate_returns_skewness(returns: pd.Series) -> float:
    """
    Calculates the skewness of the strategy returns.

    Skewness measures the asymmetry of the probability distribution of returns.
    - Positive skew: frequent small losses and a few extreme gains.
    - Negative skew: frequent small gains and a few extreme losses.

    Args:
        returns (pd.Series): Series of strategy returns (e.g., daily, hourly).

    Returns:
        float: Skewness of returns. Returns np.nan if returns series is too short or has no variance.
    """
    if returns.empty or len(returns) < 3: # Skewness is typically undefined for less than 3 data points
        return np.nan
    if returns.std() == 0: # Avoid division by zero if all returns are the same
        return 0.0 # Or np.nan, depending on desired behavior for zero variance
    return skew(returns.dropna())

def calculate_returns_kurtosis(returns: pd.Series) -> float:
    """
    Calculates the kurtosis (excess kurtosis) of the strategy returns.

    Kurtosis measures the "tailedness" of the probability distribution of returns.
    Excess kurtosis is kurtosis - 3.
    - Positive excess kurtosis (leptokurtic): "fatter tails" and a sharper peak than a normal distribution,
      indicating higher probability of extreme outcomes.
    - Negative excess kurtosis (platykurtic): "thinner tails" and a flatter peak.

    Args:
        returns (pd.Series): Series of strategy returns.

    Returns:
        float: Excess kurtosis of returns. Returns np.nan if returns series is too short or has no variance.
    """
    if returns.empty or len(returns) < 4: # Kurtosis is typically undefined for less than 4 data points
        return np.nan
    if returns.std() == 0: # Avoid division by zero
        return 0.0 # Or np.nan
    return kurtosis(returns.dropna(), fisher=True) # fisher=True gives excess kurtosis

# Placeholder for funding cost impact - will require more detailed data
def calculate_funding_cost_impact(trades_df: pd.DataFrame, total_pnl: float) -> float:
    """
    Calculates the impact of funding costs as a percentage of total P&L.
    This is a placeholder and requires 'funding_fees' column in trades_df or aggregated funding data.

    Args:
        trades_df (pd.DataFrame): DataFrame of trades, ideally with a 'funding_fees' column.
        total_pnl (float): Total profit and loss of the strategy.

    Returns:
        float: Funding costs as a percentage of total P&L. np.nan if data is insufficient.
    """
    if 'funding_fees' not in trades_df.columns or total_pnl == 0:
        return np.nan
    
    total_funding_fees = trades_df['funding_fees'].sum()
    return (total_funding_fees / total_pnl) * 100 if total_pnl != 0 else np.nan

# Placeholder for Sortino Ratio adjusted for futures - might need specific target return for futures
def calculate_sortino_ratio_futures(returns: pd.Series,
                                    risk_free_rate: float = 0.0,
                                    target_return: float = 0.0,
                                    period_annualization_factor: int = 252) -> float:
    """
    Calculates the Sortino Ratio, potentially adjusted for futures context.
    The Sortino ratio measures the risk-adjusted return of an investment asset,
    but only considers downside volatility.

    Args:
        returns (pd.Series): Series of strategy returns.
        risk_free_rate (float): Annual risk-free rate.
        target_return (float): Target return or minimum acceptable return (MAR), per period.
                               For futures, this might be set differently than for spot.
        period_annualization_factor (int): Factor to annualize returns and downside deviation
                                           (e.g., 252 for daily, 12 for monthly).

    Returns:
        float: Annualized Sortino Ratio. np.nan if calculation is not possible.
    """
    if returns.empty or len(returns) < 2:
        return np.nan

    # Calculate downside returns
    downside_returns = returns[returns < target_return]

    if downside_returns.empty:
        # No returns below target, conventionally Sortino can be infinite or very high.
        # Or, if average return is also <= target_return, it could be 0 or negative.
        # For simplicity, if no downside, and mean return > target, treat as very good (e.g. large positive).
        # If mean return <= target, treat as 0 or negative.
        mean_return = returns.mean()
        if mean_return > target_return:
            return np.inf # Or a large number if np.inf is problematic for reporting
        else:
            # If mean return is not above target and no downside, it implies all returns are at target.
            # Or if mean return is below target but no single return is, this case is tricky.
            # Let's return 0 if mean_return <= target_return and no downside_returns.
            return 0.0


    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean((downside_returns - target_return)**2))

    if downside_deviation == 0:
        # If downside deviation is zero, but there were returns below target (should not happen if target_return is MAR)
        # Or if all returns are exactly target_return.
        # If mean_return > target_return, this is a very good scenario.
        mean_return = returns.mean()
        if mean_return > target_return:
            return np.inf
        else:
            return 0.0 # No excess return and no downside risk relative to target.

    # Calculate average period return
    average_return = returns.mean()

    # Calculate Sortino Ratio for the period
    sortino_ratio_period = (average_return - risk_free_rate / period_annualization_factor) / downside_deviation
    
    # Annualize Sortino Ratio
    # Note: Annualizing Sortino is debated. Common practice is to multiply by sqrt(annualization_factor)
    # if returns and downside deviation are for the period.
    # However, if average_return and risk_free_rate are already period-based,
    # and downside_deviation is also period-based, the ratio itself is period-based.
    # To annualize the numerator (excess return) and denominator (downside deviation) separately:
    # Annualized Excess Return = (average_return - risk_free_rate / period_annualization_factor) * period_annualization_factor
    # Annualized Downside Deviation = downside_deviation * np.sqrt(period_annualization_factor)
    # Annualized Sortino = Annualized Excess Return / Annualized Downside Deviation
    
    annualized_excess_return = (average_return * period_annualization_factor) - risk_free_rate
    annualized_downside_deviation = downside_deviation * np.sqrt(period_annualization_factor)

    if annualized_downside_deviation == 0:
        return np.inf if annualized_excess_return > 0 else 0.0

    return annualized_excess_return / annualized_downside_deviation

# Potential other metrics to consider:
# - Leverage Impact on Returns
# - Leverage Impact on Drawdown
# - Liquidation Event Frequency (requires simulation capabilities beyond simple returns)

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("Testing futures_metrics.py...")

    # Test calculate_profit_per_contract
    trades_data = {
        'pnl': [100, -50, 200, -20],
        'quantity': [10, 5, 20, 2] # Number of contracts
    }
    trades_df_test = pd.DataFrame(trades_data)
    profit_per_contract = calculate_profit_per_contract(trades_df_test)
    print(f"Profit per Contract: {profit_per_contract}") # Expected: (100-50+200-20) / (10+5+20+2) = 230 / 37 = 6.216...

    trades_data_underlying = {
        'pnl': [100, -50, 200, -20],
        'quantity': [0.1, 0.05, 0.2, 0.02] # Quantity in underlying (e.g. BTC)
    }
    trades_df_test_underlying = pd.DataFrame(trades_data_underlying)
    profit_per_contract_normalized = calculate_profit_per_contract(trades_df_test_underlying, contract_size=0.01)
    # Expected: 230 / ( (0.1+0.05+0.2+0.02)/0.01 ) = 230 / (0.37/0.01) = 230 / 37 = 6.216...
    print(f"Profit per Contract (normalized): {profit_per_contract_normalized}")


    # Test calculate_returns_skewness and calculate_returns_kurtosis
    np.random.seed(42)
    # Normal distribution: skewness ~0, excess kurtosis ~0
    returns_normal = pd.Series(np.random.normal(0, 1, 1000))
    # Skewed distribution
    returns_skewed = pd.Series(np.random.beta(2, 8, 1000) * 10 - 4) # Positively skewed example
    # Leptokurtic (fat tails)
    returns_fat_tails = pd.Series(np.random.standard_t(df=5, size=1000))


    skew_normal = calculate_returns_skewness(returns_normal)
    kurt_normal = calculate_returns_kurtosis(returns_normal)
    print(f"Normal Returns: Skewness={skew_normal:.4f}, Excess Kurtosis={kurt_normal:.4f}")

    skew_skewed = calculate_returns_skewness(returns_skewed)
    kurt_skewed = calculate_returns_kurtosis(returns_skewed) # Kurtosis will also be affected
    print(f"Skewed Returns: Skewness={skew_skewed:.4f}, Excess Kurtosis={kurt_skewed:.4f}")

    skew_fat_tails = calculate_returns_skewness(returns_fat_tails)
    kurt_fat_tails = calculate_returns_kurtosis(returns_fat_tails)
    print(f"Fat-tailed Returns: Skewness={skew_fat_tails:.4f}, Excess Kurtosis={kurt_fat_tails:.4f}")

    # Test Sortino Ratio
    returns_strategy = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.015, 0.005, -0.005])
    sortino = calculate_sortino_ratio_futures(returns_strategy, risk_free_rate=0.01, target_return=0.0, period_annualization_factor=252)
    print(f"Sortino Ratio (Futures): {sortino:.4f}")

    returns_all_positive = pd.Series([0.01, 0.02, 0.005, 0.03])
    sortino_all_pos = calculate_sortino_ratio_futures(returns_all_positive, target_return=0.0)
    print(f"Sortino Ratio (all positive returns, target=0): {sortino_all_pos:.4f}")
    
    returns_all_at_target = pd.Series([0.00, 0.00, 0.00])
    sortino_all_target = calculate_sortino_ratio_futures(returns_all_at_target, target_return=0.0)
    print(f"Sortino Ratio (all returns at target=0): {sortino_all_target:.4f}")

    returns_below_target_no_variance = pd.Series([-0.01, -0.01, -0.01])
    sortino_neg_novar = calculate_sortino_ratio_futures(returns_below_target_no_variance, target_return=0.0)
    print(f"Sortino Ratio (all returns -0.01, target=0): {sortino_neg_novar:.4f}") # Expected negative or 0

    # Test with empty or short series
    print(f"Skewness (empty): {calculate_returns_skewness(pd.Series([], dtype=float))}")
    print(f"Kurtosis (short): {calculate_returns_kurtosis(pd.Series([0.1, 0.2], dtype=float))}")
    print(f"Sortino (short): {calculate_sortino_ratio_futures(pd.Series([0.1], dtype=float))}")
    print(f"Profit per contract (empty df): {calculate_profit_per_contract(pd.DataFrame())}")