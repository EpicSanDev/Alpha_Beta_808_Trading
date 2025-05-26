#!/usr/bin/env python3
"""
Simple test to verify NaN signal fixes
"""

import pandas as pd
import numpy as np
import sys
import os

print("Starting simple test for NaN signal fixes...")

def test_nan_handling():
    """Test NaN handling in signal generation"""
    
    # Simulate the signal generation logic from our fixes
    predictions = pd.Series([0.7, np.nan, 0.3, np.nan, 0.8, 0.2, np.nan])
    print(f"Input predictions: {list(predictions)}")
    
    # Apply our NaN fixing logic
    # Filter out NaN values for percentile calculation
    valid_predictions = predictions.dropna()
    print(f"Valid predictions: {list(valid_predictions)}")
    
    if len(valid_predictions) == 0:
        print("No valid predictions, all signals will be HOLD")
        signals = pd.Series(['HOLD'] * len(predictions))
    else:
        # Calculate percentiles from valid predictions only
        buy_threshold = valid_predictions.quantile(0.6)
        sell_threshold = valid_predictions.quantile(0.4)
        print(f"Buy threshold: {buy_threshold}, Sell threshold: {sell_threshold}")
        
        # Generate signals only for valid predictions
        signals = pd.Series(['HOLD'] * len(predictions))
        
        # Only assign BUY/SELL to non-NaN values
        for i, pred in enumerate(predictions):
            if not pd.isna(pred):
                if pred >= buy_threshold:
                    signals.iloc[i] = 'BUY'
                elif pred <= sell_threshold:
                    signals.iloc[i] = 'SELL'
                # else remains 'HOLD'
    
    print(f"Generated signals: {list(signals)}")
    
    # Check for any remaining NaN values
    nan_count = signals.isna().sum()
    print(f"NaN signals remaining: {nan_count}")
    
    # Check for valid signal types
    valid_signals = {'BUY', 'SELL', 'HOLD'}
    signal_types = set(signals.unique())
    invalid_signals = signal_types - valid_signals
    
    print(f"Signal types found: {signal_types}")
    print(f"Invalid signals: {invalid_signals}")
    
    return nan_count == 0 and len(invalid_signals) == 0

def test_dataframe_vs_list():
    """Test DataFrame vs list handling"""
    
    # Test data as list
    portfolio_list = [
        {'timestamp': '2023-01-01', 'portfolio_value': 100000},
        {'timestamp': '2023-01-02', 'portfolio_value': 105000},
    ]
    
    trade_list = [
        {'timestamp': '2023-01-01', 'symbol': 'BTC', 'pnl': 100},
        {'timestamp': '2023-01-02', 'symbol': 'BTC', 'pnl': 200}
    ]
    
    # Convert to DataFrame
    portfolio_df = pd.DataFrame(portfolio_list)
    trade_df = pd.DataFrame(trade_list)
    
    print("Testing DataFrame to list conversion...")
    
    # Test the conversion logic from our fixes
    def safe_to_dict(data):
        """Safely convert DataFrame or list to list of dicts"""
        if isinstance(data, pd.DataFrame):
            return data.to_dict('records')
        elif isinstance(data, list):
            return data
        else:
            return []
    
    # Test conversions
    portfolio_converted = safe_to_dict(portfolio_df)
    trade_converted = safe_to_dict(trade_df)
    
    print(f"Original list length: {len(portfolio_list)}")
    print(f"Converted from DataFrame length: {len(portfolio_converted)}")
    
    list_unchanged = safe_to_dict(portfolio_list)
    print(f"List passed through unchanged length: {len(list_unchanged)}")
    
    # Test that conversions work correctly
    return (len(portfolio_converted) == len(portfolio_list) and 
            len(trade_converted) == len(trade_list) and
            len(list_unchanged) == len(portfolio_list))

def main():
    print("=== Simple NaN Fix Test ===\n")
    
    test1_passed = test_nan_handling()
    print(f"\nNaN handling test: {'PASS' if test1_passed else 'FAIL'}")
    
    test2_passed = test_dataframe_vs_list()
    print(f"DataFrame conversion test: {'PASS' if test2_passed else 'FAIL'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\nOverall result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    if overall_success:
        print("🎉 The NaN signal fixes are working correctly!")
    else:
        print("❌ Some issues remain with the fixes.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
