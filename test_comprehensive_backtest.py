#!/usr/bin/env python3
"""
Test script to verify comprehensive backtesting system fixes.
This focuses on testing the NaN signal generation and DataFrame ambiguity fixes.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def test_comprehensive_backtesting():
    """Test the comprehensive backtesting system with realistic data"""
    print("Testing Comprehensive Backtesting System")
    print("=" * 50)
    
    try:
        from backtesting.comprehensive_backtest import ComprehensiveBacktester
        from data_processing.feature_engineering import FeatureEngineer
        from modeling.models import ModelManager
        
        # Create test configuration
        test_config = {
            "data": {
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "timeframe": "1d"
            },
            "features": {
                "technical_indicators": ["sma", "ema", "rsi", "macd", "bb"],
                "window_sizes": [5, 10, 20],
                "include_volume": True,
                "price_features": ["close", "high", "low", "open"]
            },
            "modeling": {
                "models": ["logistic_regression", "random_forest", "xgboost"],
                "target_type": "classification",
                "train_test_split": 0.8,
                "cross_validation": {
                    "enabled": True,
                    "folds": 3
                }
            },
            "backtesting": {
                "initial_capital": 100000,
                "position_size": 0.2,
                "transaction_cost": 0.001,
                "slippage": 0.0005,
                "signal_thresholds": {
                    "buy": 0.6,
                    "sell": 0.4
                }
            }
        }
        
        print("✓ Configuration loaded successfully")
        
        # Initialize components
        feature_engineer = FeatureEngineer(test_config['features'])
        model_manager = ModelManager(test_config['modeling'])
        backtester = ComprehensiveBacktester(test_config)
        
        print("✓ Components initialized successfully")
        
        # Create synthetic test data with some NaN values to test handling
        date_range = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n_days = len(date_range)
        
        # Generate realistic price data
        np.random.seed(42)
        initial_price = 150.0
        price_changes = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
        prices = [initial_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        synthetic_data = pd.DataFrame({
            'date': date_range,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n_days)
        })
        
        # Introduce some NaN values to test handling
        nan_indices = np.random.choice(n_days, size=10, replace=False)
        synthetic_data.loc[nan_indices, 'close'] = np.nan
        
        print(f"✓ Created synthetic data with {len(synthetic_data)} days")
        print(f"✓ Introduced {len(nan_indices)} NaN values for testing")
        
        # Test feature engineering
        print("\nTesting Feature Engineering...")
        features = feature_engineer.engineer_features(synthetic_data)
        print(f"✓ Generated {len(features.columns)} features")
        print(f"✓ Features shape: {features.shape}")
        
        # Check for NaN handling in features
        nan_count = features.isnull().sum().sum()
        print(f"✓ NaN values in features: {nan_count}")
        
        # Test model training and prediction
        print("\nTesting Model Training and Prediction...")
        
        # Prepare target variable
        features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
        features = features.dropna()
        
        if len(features) < 50:
            print("⚠ Warning: Not enough data for meaningful testing")
            return False
            
        X = features.drop(['target'], axis=1)
        y = features['target']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        # Train models
        trained_models = model_manager.train_models(X_train, y_train)
        print(f"✓ Trained {len(trained_models)} models")
        
        # Test predictions with potential NaN handling
        print("\nTesting Prediction and Signal Generation...")
        
        predictions = {}
        for model_name, model in trained_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_test)[:, 1]
                else:
                    pred = model.predict(X_test)
                
                # Introduce some NaN predictions to test handling
                nan_pred_indices = np.random.choice(len(pred), size=3, replace=False)
                pred[nan_pred_indices] = np.nan
                
                predictions[model_name] = pred
                print(f"✓ {model_name}: Generated {len(pred)} predictions with {np.isnan(pred).sum()} NaN values")
                
            except Exception as e:
                print(f"✗ Error with {model_name}: {str(e)}")
        
        # Test comprehensive backtesting
        print("\nTesting Comprehensive Backtesting...")
        
        try:
            # Create test data structure for backtesting
            backtest_data = {
                'data': X_test,
                'predictions': predictions,
                'prices': features['close'][split_idx:].values,
                'dates': features.index[split_idx:]
            }
            
            # Test signal generation specifically
            print("Testing signal generation with NaN handling...")
            test_predictions = np.array([0.7, 0.3, np.nan, 0.8, 0.2, np.nan, 0.6])
            signals = backtester._generate_trading_signals(test_predictions)
            
            print(f"✓ Input predictions: {test_predictions}")
            print(f"✓ Generated signals: {signals}")
            
            # Verify no NaN signals were generated
            unique_signals = set(signals)
            expected_signals = {'BUY', 'SELL', 'HOLD'}
            
            if unique_signals.issubset(expected_signals):
                print("✓ All signals are valid (no NaN signals)")
            else:
                print(f"✗ Invalid signals found: {unique_signals - expected_signals}")
                return False
            
            # Test full backtesting run
            print("\nRunning full backtesting simulation...")
            
            results = backtester.run_backtest(backtest_data)
            
            print("✓ Backtesting completed successfully")
            print("\nResults Summary:")
            print(f"  - Total Return: {results.get('total_return', 'N/A'):.2%}")
            print(f"  - Sharpe Ratio: {results.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  - Max Drawdown: {results.get('max_drawdown', 'N/A'):.2%}")
            print(f"  - Win Rate: {results.get('win_rate', 'N/A'):.2%}")
            print(f"  - Total Trades: {results.get('total_trades', 'N/A')}")
            
            # Check if performance metrics are meaningful (not all zeros)
            key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
            non_zero_metrics = sum(1 for metric in key_metrics if results.get(metric, 0) != 0)
            
            if non_zero_metrics > 0:
                print("✓ Performance metrics show meaningful values")
            else:
                print("⚠ Warning: All performance metrics are zero - may indicate issues")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during backtesting: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Comprehensive Backtesting System Test")
    print("=" * 50)
    
    success = test_comprehensive_backtesting()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED - Fixes are working correctly!")
    else:
        print("✗ TESTS FAILED - Issues remain")
    
    return success

if __name__ == "__main__":
    main()
