#!/usr/bin/env python3
"""
Test script to verify ensemble functionality in continuous_trader.py
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_helper_methods():
    """Test the helper methods for model type detection and parameter generation"""
    from continuous_trader import ContinuousTrader
    
    # Create a dummy instance to test the methods
    class TestTrader(ContinuousTrader):
        def __init__(self):
            # Skip the full initialization, just need the methods
            pass
    
    trader = TestTrader()
    
    # Test model type detection
    test_cases = [
        ("models_store/logistic_regression_mvp.joblib", "logistic_regression"),
        ("models_store/test_elasticnet_model.joblib", "elastic_net"),
        ("models_store/test_rf_opt_model.joblib", "random_forest"),
        ("models_store/test_xgb_opt_model.joblib", "xgboost_classifier"),
        ("models_store/test_quantile_model.joblib", "quantile_regression"),
        ("models_store/unknown_model.joblib", "logistic_regression")  # fallback
    ]
    
    print("Testing _get_model_type_from_path:")
    for path, expected in test_cases:
        result = trader._get_model_type_from_path(path)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"  {path} -> {result} (expected: {expected}) {status}")
    
    print("\nTesting _get_model_params_for_type:")
    model_types = ['logistic_regression', 'elastic_net', 'random_forest', 'xgboost_classifier', 'quantile_regression']
    
    for model_type in model_types:
        params = trader._get_model_params_for_type(model_type)
        has_params = len(params) > 0
        status = "✅ PASS" if has_params else "❌ FAIL"
        print(f"  {model_type}: {len(params)} parameters {status}")
        if has_params:
            print(f"    Sample params: {list(params.keys())[:3]}")

def test_model_availability():
    """Test that all expected models exist in the models_store directory"""
    models_dir = Path("models_store")
    
    expected_models = [
        "logistic_regression_mvp.joblib",
        "test_elasticnet_model.joblib", 
        "test_rf_opt_model.joblib",
        "test_xgb_opt_model.joblib",
        "test_quantile_model.joblib"
    ]
    
    print("\nTesting model availability:")
    for model_file in expected_models:
        model_path = models_dir / model_file
        exists = model_path.exists()
        status = "✅ AVAILABLE" if exists else "❌ MISSING"
        print(f"  {model_file}: {status}")

def test_config_structure():
    """Test that the trader configuration has the correct ensemble structure"""
    import json
    
    print("\nTesting configuration structure:")
    
    try:
        with open("trader_config.json", "r") as f:
            config = json.load(f)
        
        # Check ensemble configuration
        model_config = config.get("model", {})
        ensemble_enabled = model_config.get("ensemble_enabled", False)
        models = model_config.get("models", [])
        fallback_model = model_config.get("fallback_model", "")
        
        print(f"  Ensemble enabled: {ensemble_enabled} {'✅' if ensemble_enabled else '❌'}")
        print(f"  Number of models configured: {len(models)} {'✅' if len(models) > 1 else '❌'}")
        print(f"  Fallback model configured: {'✅' if fallback_model else '❌'}")
        
        # Check individual model configurations
        total_weight = 0
        for i, model in enumerate(models):
            name = model.get("name", "")
            path = model.get("path", "")
            weight = model.get("weight", 0)
            enabled = model.get("enabled", False)
            total_weight += weight if enabled else 0
            
            print(f"  Model {i+1} ({name}): weight={weight}, enabled={enabled}, path={'✅' if path else '❌'}")
        
        weight_status = "✅" if 0.95 <= total_weight <= 1.05 else "❌"
        print(f"  Total enabled weights: {total_weight:.2f} {weight_status}")
        
    except Exception as e:
        print(f"  ❌ Error reading config: {e}")

if __name__ == "__main__":
    print("🧪 Testing AlphaBeta808 Ensemble Functionality")
    print("=" * 50)
    
    test_helper_methods()
    test_model_availability()
    test_config_structure()
    
    print("\n" + "=" * 50)
    print("✅ Ensemble testing completed!")
    print("\nNext steps:")
    print("1. Run the continuous trader to test live ensemble functionality")
    print("2. Monitor logs for ensemble prediction behavior")
    print("3. Verify that signals use weighted averages from multiple models")
