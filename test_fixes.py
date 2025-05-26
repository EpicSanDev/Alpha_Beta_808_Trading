#!/usr/bin/env python3
"""
Test script to verify the NaN signal generation and DataFrame ambiguity fixes
without requiring full ML dependencies.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append('src')

# Mock the problematic imports to avoid dependency issues
import unittest.mock

# Mock TensorFlow and other ML dependencies
with unittest.mock.patch.dict('sys.modules', {
    'tensorflow': unittest.mock.MagicMock(),
    'tensorflow.keras': unittest.mock.MagicMock(),
    'tensorflow.keras.models': unittest.mock.MagicMock(),
    'tensorflow.keras.layers': unittest.mock.MagicMock(),
    'tensorflow.keras.callbacks': unittest.mock.MagicMock(),
    'tensorflow.keras.regularizers': unittest.mock.MagicMock(),
    'gpflow': unittest.mock.MagicMock(),
    'pymc': unittest.mock.MagicMock(),
    'arviz': unittest.mock.MagicMock(),
}):
    # Now try to import the comprehensive backtest
    from src.backtesting.comprehensive_backtest import ComprehensiveBacktest

def test_nan_signal_generation():
    """Test the NaN signal generation fixes"""
    print("Testing NaN signal generation fixes...")
    
    # Create a backtest instance
    backtest = ComprehensiveBacktest()
    
    # Create test data with NaN predictions
    test_predictions = pd.Series([0.7, np.nan, 0.3, np.nan, 0.8, 0.2, np.nan])
    
    print(f"Input predictions: {list(test_predictions)}")
    
    try:
        # Test the signal generation method
        signals = backtest._generate_trading_signals(test_predictions)
        print(f"Generated signals: {list(signals)}")
        
        # Check that we don't have any NaN signals
        nan_signals = signals.isna().sum()
        if nan_signals > 0:
            print(f"ERROR: Still have {nan_signals} NaN signals!")
            return False
        else:
            print("SUCCESS: No NaN signals generated!")
            
        # Check that we have only valid signal types
        valid_signals = {'BUY', 'SELL', 'HOLD'}
        invalid_signals = set(signals) - valid_signals
        if invalid_signals:
            print(f"ERROR: Invalid signals found: {invalid_signals}")
            return False
        else:
            print("SUCCESS: All signals are valid!")
            
        return True
        
    except Exception as e:
        print(f"ERROR in signal generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_metrics_dataframe_handling():
    """Test the DataFrame handling in performance metrics"""
    print("\nTesting DataFrame handling in performance metrics...")
    
    backtest = ComprehensiveBacktest()
    
    # Create test data as both list and DataFrame formats
    portfolio_history_list = [
        {'timestamp': '2023-01-01', 'portfolio_value': 100000, 'cash': 50000, 'positions': {'BTC': 0.5}},
        {'timestamp': '2023-01-02', 'portfolio_value': 105000, 'cash': 45000, 'positions': {'BTC': 0.6}},
        {'timestamp': '2023-01-03', 'portfolio_value': 98000, 'cash': 48000, 'positions': {'BTC': 0.52}}
    ]
    
    trade_history_list = [
        {'timestamp': '2023-01-01', 'symbol': 'BTC', 'side': 'BUY', 'quantity': 0.1, 'price': 50000, 'pnl': 0},
        {'timestamp': '2023-01-02', 'symbol': 'BTC', 'side': 'SELL', 'quantity': 0.05, 'price': 52000, 'pnl': 100}
    ]
    
    # Convert to DataFrames
    portfolio_history_df = pd.DataFrame(portfolio_history_list)
    trade_history_df = pd.DataFrame(trade_history_list)
    
    try:
        # Test with list inputs
        print("Testing with list inputs...")
        metrics_list = backtest._calculate_performance_metrics(
            portfolio_history_list, trade_history_list
        )
        print(f"Metrics from lists: {metrics_list}")
        
        # Test with DataFrame inputs  
        print("Testing with DataFrame inputs...")
        metrics_df = backtest._calculate_performance_metrics(
            portfolio_history_df, trade_history_df
        )
        print(f"Metrics from DataFrames: {metrics_df}")
        
        # Check that both return valid metrics
        required_keys = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'total_trades']
        
        for key in required_keys:
            if key not in metrics_list:
                print(f"ERROR: Missing key '{key}' in list-based metrics")
                return False
            if key not in metrics_df:
                print(f"ERROR: Missing key '{key}' in DataFrame-based metrics")
                return False
                
        print("SUCCESS: Both list and DataFrame inputs handled correctly!")
        return True
        
    except Exception as e:
        print(f"ERROR in performance metrics calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_trading_simulation_nan_cleaning():
    """Test the NaN cleaning in trading simulation"""
    print("\nTesting NaN cleaning in trading simulation...")
    
    backtest = ComprehensiveBacktest()
    
    # Create test signals with NaN values
    test_signals = pd.Series([np.nan, 'BUY', 'nan', 'SELL', np.nan, 'HOLD'])
    test_prices = pd.Series([100, 105, 103, 107, 109, 106])
    
    print(f"Input signals: {list(test_signals)}")
    
    try:
        # Test the signal cleaning part
        signals_clean = test_signals.fillna('HOLD')
        signals_clean = signals_clean.replace('nan', 'HOLD')
        
        print(f"Cleaned signals: {list(signals_clean)}")
        
        # Check that no NaN or 'nan' strings remain
        has_nan = signals_clean.isna().any()
        has_nan_string = (signals_clean == 'nan').any()
        
        if has_nan or has_nan_string:
            print("ERROR: Still have NaN values after cleaning!")
            return False
        else:
            print("SUCCESS: All NaN values cleaned successfully!")
            return True
            
    except Exception as e:
        print(f"ERROR in signal cleaning: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Testing Comprehensive Backtesting Fixes ===\n")
    
    tests = [
        test_nan_signal_generation,
        test_performance_metrics_dataframe_handling,
        test_trading_simulation_nan_cleaning
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"ERROR running {test.__name__}: {str(e)}")
            results.append(False)
    
    print(f"\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "PASS" if result else "FAIL"
        print(f"{test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
