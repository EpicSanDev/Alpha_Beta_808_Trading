#!/usr/bin/env python3
"""
Quick System Check - Verify module imports and basic functionality
This script tests the core functionality without external dependencies.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic module imports"""
    print("🔧 Testing basic module imports...")
    
    try:
        # Test core modules that should always work
        import acquisition.connectors as connectors
        print("✅ acquisition.connectors imported successfully")
        
        import feature_engineering.technical_features as tech_features
        print("✅ feature_engineering.technical_features imported successfully")
        
        import signal_generation.signal_generator as signal_gen
        print("✅ signal_generation.signal_generator imported successfully")
        
        import execution.simulator as simulator
        print("✅ execution.simulator imported successfully")
        
        import core.performance_analyzer as performance
        print("✅ core.performance_analyzer imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_advanced_imports():
    """Test advanced module imports"""
    print("\n🚀 Testing advanced module imports...")
    
    try:
        import validation.walk_forward as walk_forward
        print("✅ validation.walk_forward imported successfully")
        
        import risk_management.dynamic_stops as dynamic_stops
        print("✅ risk_management.dynamic_stops imported successfully")
        
        import portfolio.multi_asset as multi_asset
        print("✅ portfolio.multi_asset imported successfully")
        
        import execution.real_time_trading as real_time
        print("✅ execution.real_time_trading imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Advanced import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    print("\n⚡ Testing basic functionality...")
    
    try:
        # Test data generation
        import acquisition.connectors as connectors
        data = connectors.generate_random_market_data(num_rows=100)
        print(f"✅ Generated sample data with {len(data)} rows")
        
        # Test technical features
        import feature_engineering.technical_features as tech_features
        data_with_features = tech_features.calculate_sma(data, column='close', windows=[10, 20])
        print("✅ Technical features calculation working")
        
        # Test signal generation
        import signal_generation.signal_generator as signal_gen
        # Create dummy predictions for testing
        import numpy as np
        predictions = np.random.choice([0, 1], size=len(data_with_features))
        signals = signal_gen.generate_signals_from_predictions(
            predictions, 
            threshold=0.5,
            prediction_type='class'
        )
        print("✅ Signal generation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_walk_forward_module():
    """Test walk-forward validation module specifically"""
    print("\n📊 Testing Walk-Forward Validation...")
    
    try:
        from validation.walk_forward import WalkForwardValidator, WalkForwardResults
        
        validator = WalkForwardValidator(
            training_window_days=50,
            validation_window_days=10,
            retrain_frequency_days=10
        )
        print("✅ WalkForwardValidator created successfully")
        print("✅ WalkForwardResults class available")
        
        return True
        
    except Exception as e:
        print(f"❌ Walk-Forward test error: {e}")
        return False

def main():
    """Main test function"""
    print("🎯 AlphaBeta808Trading System Check")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # Run tests
    if test_basic_imports():
        success_count += 1
    
    if test_advanced_imports():
        success_count += 1
        
    if test_basic_functionality():
        success_count += 1
        
    if test_walk_forward_module():
        success_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📈 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All systems operational!")
        return True
    else:
        print("⚠️  Some issues detected. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
