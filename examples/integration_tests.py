"""
Integration Tests for Advanced Features

This script tests the integration of all four advanced features to ensure
they work correctly with the existing AlphaBeta808Trading system.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

class TestAdvancedFeaturesIntegration(unittest.TestCase):
    """Test suite for advanced features integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and common objects"""
        # Generate sample market data
        np.random.seed(42)  # For reproducible tests
        
        cls.sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
            'open': np.random.uniform(45, 55, 1000),
            'high': np.random.uniform(50, 60, 1000),
            'low': np.random.uniform(40, 50, 1000),
            'close': np.random.uniform(45, 55, 1000),
            'volume': np.random.uniform(1000, 10000, 1000),
        })
        
        # Calculate returns
        cls.sample_data['returns'] = cls.sample_data['close'].pct_change()
        cls.sample_data = cls.sample_data.dropna()
        
        print(f"Test setup complete. Generated {len(cls.sample_data)} data points.")
    
    def test_walk_forward_validation_import(self):
        """Test that Walk-Forward Validation modules can be imported"""
        try:
            from validation.walk_forward import WalkForwardValidator, AdaptiveModelManager
            self.assertTrue(True, "Walk-Forward Validation modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import Walk-Forward Validation modules: {e}")
    
    def test_dynamic_stops_import(self):
        """Test that Dynamic Stop-Loss modules can be imported"""
        try:
            from risk_management.dynamic_stops import DynamicStopLossManager
            self.assertTrue(True, "Dynamic Stop-Loss modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import Dynamic Stop-Loss modules: {e}")
    
    def test_multi_asset_portfolio_import(self):
        """Test that Multi-Asset Portfolio modules can be imported"""
        try:
            from portfolio.multi_asset import MultiAssetPortfolioManager
            self.assertTrue(True, "Multi-Asset Portfolio modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import Multi-Asset Portfolio modules: {e}")
    
    def test_real_time_trading_import(self):
        """Test that Real-Time Trading modules can be imported"""
        try:
            from execution.real_time_trading import BinanceRealTimeTrader, TradingStrategy
            self.assertTrue(True, "Real-Time Trading modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import Real-Time Trading modules: {e}")
    
    def test_walk_forward_validation_basic_functionality(self):
        """Test basic functionality of Walk-Forward Validation"""
        try:
            from validation.walk_forward import WalkForwardValidator
            
            validator = WalkForwardValidator(
                training_window_days=90,  # 3 months equivalent  
                validation_window_days=30,  # 1 month equivalent
                retrain_frequency_days=30
            )
            
            # Test initialization
            self.assertIsNotNone(validator)
            self.assertEqual(validator.training_window_days, 90)
            self.assertEqual(validator.validation_window_days, 30)
            
            print("✓ Walk-Forward Validation basic functionality test passed")
            
        except Exception as e:
            self.fail(f"Walk-Forward Validation basic functionality test failed: {e}")
    
    def test_dynamic_stops_basic_functionality(self):
        """Test basic functionality of Dynamic Stop-Loss"""
        try:
            from risk_management.dynamic_stops import DynamicStopLossManager
            
            stop_manager = DynamicStopLossManager()
            
            # Test ATR stop-loss setting
            stop_manager.set_atr_stop_loss(
                symbol='BTCUSDT',
                atr_value=100.0,
                entry_price=50000.0,
                position_type='long'
            )
            
            # Test stop level retrieval
            stops = stop_manager.get_stop_levels('BTCUSDT')
            self.assertIsInstance(stops, dict)
            
            print("✓ Dynamic Stop-Loss basic functionality test passed")
            
        except Exception as e:
            self.fail(f"Dynamic Stop-Loss basic functionality test failed: {e}")
    
    def test_multi_asset_portfolio_basic_functionality(self):
        """Test basic functionality of Multi-Asset Portfolio Management"""
        try:
            from portfolio.multi_asset import MultiAssetPortfolioManager
            
            portfolio_manager = MultiAssetPortfolioManager(
                initial_capital=100000.0,
                rebalancing_frequency_days=30  # monthly equivalent
            )
            
            # Test asset addition
            portfolio_manager.add_asset(
                symbol='BTCUSDT',
                price_data=self.sample_data['close'],
                returns_data=self.sample_data['returns']
            )
            
            # Test asset retrieval
            assets = portfolio_manager.get_assets()
            self.assertIn('BTCUSDT', assets)
            
            print("✓ Multi-Asset Portfolio basic functionality test passed")
            
        except Exception as e:
            self.fail(f"Multi-Asset Portfolio basic functionality test failed: {e}")
    
    def test_real_time_trading_basic_functionality(self):
        """Test basic functionality of Real-Time Trading"""
        try:
            from execution.real_time_trading import BinanceRealTimeTrader, TradingStrategy
            
            # Test trader initialization (demo mode)
            # SECURITY: Use environment variables for demo credentials
            import os
            demo_api_key = os.getenv('DEMO_API_KEY', 'demo_key_placeholder')
            demo_api_secret = os.getenv('DEMO_API_SECRET', 'demo_secret_placeholder')
            
            trader = BinanceRealTimeTrader(
                api_key=demo_api_key,
                api_secret=demo_api_secret,
                testnet=True
            )
            
            # Test strategy initialization
            strategy = TradingStrategy(trader)
            strategy.add_symbol('BTCUSDT')
            
            self.assertIsNotNone(trader)
            self.assertIsNotNone(strategy)
            
            print("✓ Real-Time Trading basic functionality test passed")
            
        except Exception as e:
            self.fail(f"Real-Time Trading basic functionality test failed: {e}")
    
    def test_integration_workflow(self):
        """Test that all features can work together in an integrated workflow"""
        try:
            from validation.walk_forward import WalkForwardValidator
            from risk_management.dynamic_stops import DynamicStopLossManager
            from portfolio.multi_asset import MultiAssetPortfolioManager
            from execution.real_time_trading import BinanceRealTimeTrader
            
            # Initialize all components
            validator = WalkForwardValidator()
            stop_manager = DynamicStopLossManager()
            portfolio_manager = MultiAssetPortfolioManager(initial_capital=100000.0)
            trader = BinanceRealTimeTrader("demo", "demo", testnet=True)
            
            # Test data flow between components
            symbols = ['BTCUSDT', 'ETHUSDT']
            
            for symbol in symbols:
                # Add to portfolio
                portfolio_manager.add_asset(
                    symbol=symbol,
                    price_data=self.sample_data['close'],
                    returns_data=self.sample_data['returns']
                )
                
                # Set stop-loss
                stop_manager.set_atr_stop_loss(
                    symbol=symbol,
                    atr_value=50.0,
                    entry_price=self.sample_data['close'].iloc[-1],
                    position_type='long'
                )
            
            # Test portfolio optimization
            weights = portfolio_manager.optimize_portfolio(method='equal_weight')
            self.assertIsInstance(weights, dict)
            self.assertEqual(len(weights), len(symbols))
            
            # Test risk calculations
            portfolio_risk = portfolio_manager.calculate_risk_metrics()
            self.assertIsInstance(portfolio_risk, dict)
            
            print("✓ Integration workflow test passed")
            
        except Exception as e:
            self.fail(f"Integration workflow test failed: {e}")
    
    def test_configuration_compatibility(self):
        """Test that configuration examples work with the system"""
        try:
            # Import configuration
            sys.path.append(os.path.dirname(__file__))
            from configuration_examples import (
                WALK_FORWARD_CONFIG,
                DYNAMIC_STOPS_CONFIG,
                PORTFOLIO_CONFIG,
                REAL_TIME_TRADING_CONFIG,
                validate_configuration
            )
            
            # Test configuration structure
            self.assertIsInstance(WALK_FORWARD_CONFIG, dict)
            self.assertIsInstance(DYNAMIC_STOPS_CONFIG, dict)
            self.assertIsInstance(PORTFOLIO_CONFIG, dict)
            self.assertIsInstance(REAL_TIME_TRADING_CONFIG, dict)
            
            # Test configuration keys
            self.assertIn('train_window_months', WALK_FORWARD_CONFIG)
            self.assertIn('atr_multiplier', DYNAMIC_STOPS_CONFIG)
            self.assertIn('initial_capital', PORTFOLIO_CONFIG)
            self.assertIn('signal_threshold', REAL_TIME_TRADING_CONFIG)
            
            print("✓ Configuration compatibility test passed")
            
        except Exception as e:
            self.fail(f"Configuration compatibility test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling in advanced features"""
        try:
            from validation.walk_forward import WalkForwardValidator
            from risk_management.dynamic_stops import DynamicStopLossManager
            from portfolio.multi_asset import MultiAssetPortfolioManager
            
            # Test invalid parameters handling
            validator = WalkForwardValidator()
            stop_manager = DynamicStopLossManager()
            portfolio_manager = MultiAssetPortfolioManager(initial_capital=100000.0)
            
            # These should not crash the system
            try:
                # Test with invalid symbol
                stop_manager.get_stop_levels('INVALID_SYMBOL')
            except:
                pass  # Expected to handle gracefully
            
            try:
                # Test with empty data - should handle gracefully
                portfolio_manager.add_asset(symbol='TEST')
            except Exception:
                pass  # Expected to handle gracefully
            
            print("✓ Error handling test passed")
            
        except Exception as e:
            self.fail(f"Error handling test failed: {e}")
    
    def test_performance_metrics(self):
        """Test that performance metrics can be calculated"""
        try:
            from portfolio.multi_asset import MultiAssetPortfolioManager
            
            portfolio_manager = MultiAssetPortfolioManager(initial_capital=100000.0)
            
            # Add sample asset
            portfolio_manager.add_asset(
                symbol='BTCUSDT',
                price_data=self.sample_data['close'],
                returns_data=self.sample_data['returns']
            )
            
            # Calculate performance metrics
            performance = portfolio_manager.calculate_portfolio_performance(
                start_date=self.sample_data['timestamp'].iloc[0],
                end_date=self.sample_data['timestamp'].iloc[-1]
            )
            
            # Check that key metrics are present
            expected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
            for metric in expected_metrics:
                self.assertIn(metric, performance)
            
            print("✓ Performance metrics test passed")
            
        except Exception as e:
            self.fail(f"Performance metrics test failed: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test integration with existing system components"""
    
    def test_existing_modules_compatibility(self):
        """Test that advanced features work with existing modules"""
        try:
            # Import existing modules
            from acquisition.connectors import generate_random_market_data
            from feature_engineering.technical_features import calculate_sma, calculate_rsi
            from signal_generation.signal_generator import generate_base_signals_from_predictions
            
            # Import advanced features
            from portfolio.multi_asset import MultiAssetPortfolioManager
            from risk_management.dynamic_stops import DynamicStopLossManager
            
            # Generate test data using existing connector
            test_data = generate_random_market_data(num_rows=100)
            
            # Add technical features
            test_data = calculate_sma(test_data, column='close', windows=[20])
            test_data = calculate_rsi(test_data, column='close')
            
            # Use advanced features with existing data
            portfolio_manager = MultiAssetPortfolioManager(initial_capital=10000.0)
            portfolio_manager.add_asset(
                symbol='TEST',
                price_data=test_data['close'],
                returns_data=test_data['close'].pct_change().dropna()
            )
            
            print("✓ Existing modules compatibility test passed")
            
        except Exception as e:
            self.fail(f"Existing modules compatibility test failed: {e}")

def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS FOR ADVANCED FEATURES")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [TestAdvancedFeaturesIntegration, TestSystemIntegration]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("The advanced features are properly integrated and functional.")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        for test, error in result.failures + result.errors:
            print(f"\nFailed: {test}")
            print(f"Error: {error}")
    
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
