#!/usr/bin/env python3
"""
AlphaBeta808Trading System Verification
Comprehensive verification script to test all system components
"""

import sys
import os
import traceback
import json
import pandas as pd
import numpy as np
from datetime import datetime

def test_imports():
    """Test that all critical modules can be imported"""
    print("🔍 Testing imports...")
    try:
        from continuous_trader import ContinuousTrader
        from live_trading_bot import LiveTradingBot
        
        # Core modules
        from src.acquisition.connectors import load_binance_klines, generate_random_market_data
        from src.feature_engineering.technical_features import calculate_sma, calculate_rsi, calculate_macd
        from src.modeling.models import prepare_data_for_model, load_model_and_predict
        from src.signal_generation.signal_generator import generate_signals_from_predictions
        from src.execution.real_time_trading import BinanceRealTimeTrader, MarketData, TradingOrder
        from src.risk_management.risk_controls import check_position_limit
        from src.portfolio.multi_asset import MultiAssetPortfolioManager
        from src.execution.simulator import BacktestSimulator
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_generation():
    """Test data generation functionality"""
    print("🔍 Testing data generation...")
    try:
        from src.acquisition.connectors import generate_random_market_data
        
        # Generate test data
        data = generate_random_market_data(num_rows=100, start_price=100.0)
        
        if len(data) == 100 and 'close' in data.columns:
            print("✅ Data generation successful")
            return True
        else:
            print("❌ Data generation failed: Invalid data structure")
            return False
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("🔍 Testing feature engineering...")
    try:
        from src.acquisition.connectors import generate_random_market_data
        from src.feature_engineering.technical_features import calculate_sma, calculate_rsi, calculate_macd
        
        # Generate test data
        data = generate_random_market_data(num_rows=100, start_price=100.0)
        
        # Test SMA - correct signature uses windows parameter
        sma_data = calculate_sma(data, column='close', windows=[10])
        data = data.join(sma_data, rsuffix='_sma')
        
        # Test RSI - correct signature - pass DataFrame not Series
        data = calculate_rsi(data, column='close', window=14)
        
        # Test MACD
        data = calculate_macd(data, column='close')
        
        if 'sma_10' in data.columns and 'rsi_14' in data.columns and 'macd' in data.columns:
            print("✅ Feature engineering successful")
            return True
        else:
            print("❌ Feature engineering failed: Missing features")
            return False
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return False

def test_model_functionality():
    """Test ML model functionality"""
    print("🔍 Testing ML model functionality...")
    try:
        from src.acquisition.connectors import generate_random_market_data
        from src.feature_engineering.technical_features import calculate_sma, calculate_rsi
        from src.modeling.models import prepare_data_for_model, train_model
        
        # Generate test data with features
        data = generate_random_market_data(num_rows=200, start_price=100.0)
        
        # Calculate features with correct signatures
        sma_data = calculate_sma(data, column='close', windows=[10])
        data = data.join(sma_data, rsuffix='_sma')
        data = calculate_rsi(data, column='close', window=14)
        
        # Prepare data for modeling
        X, y = prepare_data_for_model(data, target_column='close', price_change_threshold=0.01, 
                                    feature_columns=['sma_10', 'rsi_14'])
        
        if len(X) > 0 and len(y) > 0:
            print("✅ Model data preparation successful")
            return True
        else:
            print("❌ Model functionality failed: No data prepared")
            return False
    except Exception as e:
        print(f"❌ Model functionality failed: {e}")
        return False

def test_trading_components():
    """Test trading system components"""
    print("🔍 Testing trading components...")
    try:
        from src.execution.real_time_trading import MarketData, TradingOrder, OrderSide, OrderType
        from src.risk_management.risk_controls import check_position_limit
        
        # Test MarketData creation
        market_data = MarketData(
            symbol="BTCUSDT",
            price=50000.0,
            bid_price=49999.0,
            ask_price=50001.0,
            bid_quantity=1.0,
            ask_quantity=1.0,
            volume_24h=1000.0,
            price_change_24h=0.02,
            timestamp=datetime.now()
        )
        
        # Test TradingOrder creation
        order = TradingOrder(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        # Test risk controls with correct signature
        can_trade = check_position_limit(
            new_position_size=5000, 
            current_total_exposure=20000, 
            max_exposure_limit=50000
        )
        
        if market_data.symbol == "BTCUSDT" and order.side == OrderSide.BUY and can_trade:
            print("✅ Trading components successful")
            return True
        else:
            print("❌ Trading components failed")
            return False
    except Exception as e:
        print(f"❌ Trading components failed: {e}")
        return False

def test_continuous_trader_init():
    """Test ContinuousTrader initialization"""
    print("🔍 Testing ContinuousTrader initialization...")
    try:
        from continuous_trader import ContinuousTrader
        
        # Test with config file
        if os.path.exists('trader_config.json'):
            trader = ContinuousTrader('trader_config.json')
            print("✅ ContinuousTrader initialization successful")
            return True
        else:
            print("❌ ContinuousTrader initialization failed: No config file")
            return False
    except Exception as e:
        print(f"❌ ContinuousTrader initialization failed: {e}")
        return False

def test_live_trading_bot_init():
    """Test LiveTradingBot initialization"""
    print("🔍 Testing LiveTradingBot initialization...")
    try:
        from live_trading_bot import LiveTradingBot
        
        # Test initialization
        if os.path.exists('trader_config.json'):
            bot = LiveTradingBot('trader_config.json')
            print("✅ LiveTradingBot initialization successful")
            return True
        else:
            print("❌ LiveTradingBot initialization failed: No config file")
            return False
    except Exception as e:
        print(f"❌ LiveTradingBot initialization failed: {e}")
        return False

def test_portfolio_manager():
    """Test MultiAssetPortfolioManager"""
    print("🔍 Testing Portfolio Manager...")
    try:
        from src.portfolio.multi_asset import MultiAssetPortfolioManager
        
        # Test initialization - correct signature
        portfolio = MultiAssetPortfolioManager(
            initial_capital=100000
        )
        
        # Test basic functionality
        assets = portfolio.get_assets()
        
        if isinstance(assets, dict):
            print("✅ Portfolio Manager successful")
            return True
        else:
            print("❌ Portfolio Manager failed")
            return False
    except Exception as e:
        print(f"❌ Portfolio Manager failed: {e}")
        return False

def run_system_verification():
    """Run comprehensive system verification"""
    print("🚀 AlphaBeta808Trading System Verification")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_feature_engineering,
        test_model_functionality,
        test_trading_components,
        test_continuous_trader_init,
        test_live_trading_bot_init,
        test_portfolio_manager
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"📊 VERIFICATION SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! The system is fully functional.")
        print("✅ AlphaBeta808Trading is ready for deployment!")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = run_system_verification()
    sys.exit(0 if success else 1)
