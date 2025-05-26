"""
Configuration Example for Advanced Features

This file demonstrates how to configure the advanced features of the
AlphaBeta808Trading system for different use cases.
"""

# =============================================================================
# 1. WALK-FORWARD VALIDATION CONFIGURATION
# =============================================================================

WALK_FORWARD_CONFIG = {
    # Training window configuration
    'train_window_months': 12,           # 12 months of training data
    'test_window_months': 1,             # 1 month of testing data
    'retrain_frequency_days': 30,        # Retrain every 30 days
    
    # Performance thresholds
    'performance_threshold': 0.05,       # 5% minimum performance threshold
    'drift_threshold': 0.1,              # 10% threshold for concept drift
    
    # Model configuration
    'models_to_validate': [
        'RandomForest',
        'GradientBoosting', 
        'LinearRegression',
        'SVM'
    ],
    
    # Cross-validation settings
    'cv_folds': 5,
    'scoring_metric': 'neg_mean_squared_error',
    
    # Adaptive model management
    'model_decay_factor': 0.95,          # Weight decay for older performance
    'min_performance_history': 10,       # Minimum performance records required
    'performance_window': 30,            # Days to consider for performance calculation
}

# =============================================================================
# 2. DYNAMIC STOP-LOSS CONFIGURATION
# =============================================================================

DYNAMIC_STOPS_CONFIG = {
    # ATR-based stop-loss
    'atr_period': 14,                    # ATR calculation period
    'atr_multiplier': 2.0,               # ATR multiplier for stop distance
    
    # Volatility-based stop-loss
    'volatility_period': 20,             # Volatility calculation period
    'volatility_multiplier': 1.5,        # Volatility multiplier
    
    # Trailing stop-loss
    'trailing_percentage': 0.05,         # 5% trailing stop
    'min_trail_amount': 0.02,            # Minimum 2% trail amount
    
    # Support/Resistance stops
    'support_resistance_periods': [20, 50], # Periods for S/R calculation
    'sr_buffer': 0.01,                   # 1% buffer around S/R levels
    
    # Time-based exits
    'max_hold_period_hours': 168,        # Maximum 7 days (168 hours)
    'intraday_exit_time': '16:00',       # Exit time for intraday positions
    
    # Risk management
    'max_loss_per_trade': 0.02,          # Maximum 2% loss per trade
    'portfolio_heat': 0.06,              # Maximum 6% portfolio at risk
    
    # Update frequencies
    'stop_update_frequency_minutes': 5,   # Update stops every 5 minutes
    'trailing_update_threshold': 0.005,   # Update trailing stop if price moves 0.5%
}

# =============================================================================
# 3. MULTI-ASSET PORTFOLIO CONFIGURATION
# =============================================================================

PORTFOLIO_CONFIG = {
    # Portfolio initialization
    'initial_capital': 100000.0,         # Starting capital
    'base_currency': 'USDT',             # Base currency for calculations
    
    # Asset universe
    'asset_universe': [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
        'BNBUSDT', 'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'ATOMUSDT'
    ],
    
    # Portfolio optimization
    'optimization_methods': {
        'mean_variance': {
            'risk_aversion': 1.0,
            'expected_return_method': 'historical',
            'covariance_method': 'sample'
        },
        'risk_parity': {
            'risk_budget_method': 'equal',
            'max_iterations': 1000,
            'tolerance': 1e-6
        },
        'black_litterman': {
            'risk_aversion': 3.0,
            'tau': 0.025,
            'confidence_level': 0.95
        }
    },
    
    # Rebalancing configuration
    'rebalance_frequency': 'monthly',     # 'daily', 'weekly', 'monthly', 'quarterly'
    'rebalance_threshold': 0.05,          # Rebalance if allocation drifts > 5%
    'volatility_rebalance_trigger': 0.3,  # Rebalance if volatility increases > 30%
    
    # Risk constraints
    'max_asset_weight': 0.3,              # Maximum 30% allocation per asset
    'min_asset_weight': 0.05,             # Minimum 5% allocation per asset
    'max_sector_concentration': 0.4,      # Maximum 40% in any sector
    
    # Performance calculation
    'benchmark': 'equal_weight',          # Benchmark for performance comparison
    'return_frequency': 'daily',          # Return calculation frequency
    'risk_free_rate': 0.02,               # 2% annual risk-free rate
    
    # Transaction costs
    'transaction_cost': 0.001,            # 0.1% transaction cost
    'slippage_model': 'linear',           # 'linear', 'square_root', 'fixed'
    'slippage_parameter': 0.0005,         # Slippage parameter
}

# =============================================================================
# 4. REAL-TIME TRADING CONFIGURATION
# =============================================================================

REAL_TIME_TRADING_CONFIG = {
    # API Configuration
    'exchange': 'binance',
    'testnet': True,                      # Use testnet for testing
    'api_timeout': 30,                    # API timeout in seconds
    'max_retries': 3,                     # Maximum API retry attempts
    
    # WebSocket Configuration
    'websocket_streams': [
        'ticker',                         # 24hr ticker statistics
        'trade',                          # Trade stream
        'kline_1m',                       # 1-minute kline data
        'depth',                          # Order book depth
    ],
    'reconnect_attempts': 5,              # WebSocket reconnection attempts
    'ping_interval': 30,                  # WebSocket ping interval
    
    # Trading Configuration
    'symbols_to_trade': [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT'
    ],
    'order_types': ['MARKET', 'LIMIT'],   # Supported order types
    'default_order_type': 'LIMIT',        # Default order type
    
    # Position Management
    'max_positions': 10,                  # Maximum concurrent positions
    'max_position_size_usd': 5000,        # Maximum position size in USD
    'min_position_size_usd': 100,         # Minimum position size in USD
    
    # Risk Management
    'max_daily_loss': 0.02,               # Maximum 2% daily loss
    'max_total_exposure': 0.8,            # Maximum 80% capital exposure
    'max_orders_per_minute': 10,          # Rate limiting
    'position_size_method': 'fixed_usd',  # 'fixed_usd', 'percentage', 'kelly'
    
    # Signal Processing
    'signal_threshold': 0.6,              # Minimum signal strength to trade
    'signal_timeout_minutes': 30,         # Signal validity timeout
    'conflicting_signal_resolution': 'latest', # How to handle conflicting signals
    
    # Execution Settings
    'execution_algorithm': 'twap',        # 'market', 'twap', 'vwap', 'iceberg'
    'slice_size_percentage': 0.1,         # For TWAP/VWAP: slice as % of total order
    'execution_time_limit_minutes': 60,   # Maximum execution time
    
    # Monitoring and Alerts
    'performance_monitoring': True,
    'alert_channels': ['email', 'telegram'], # Alert notification channels
    'alert_thresholds': {
        'drawdown': 0.05,                 # Alert if drawdown > 5%
        'api_errors': 5,                  # Alert if > 5 API errors per hour
        'position_pnl': -0.03,            # Alert if position loss > 3%
    },
    
    # Data Storage
    'store_market_data': True,
    'store_execution_data': True,
    'data_retention_days': 90,            # Keep data for 90 days
}

# =============================================================================
# 5. INTEGRATED SYSTEM CONFIGURATION
# =============================================================================

INTEGRATED_CONFIG = {
    # System coordination
    'master_frequency': '1min',           # Master system update frequency
    'component_sync': True,               # Synchronize all components
    
    # Feature interaction settings
    'walk_forward_retrain_triggers': [
        'performance_degradation',
        'concept_drift_detected',
        'scheduled_retrain'
    ],
    
    'portfolio_rebalance_triggers': [
        'allocation_drift',
        'volatility_spike',
        'new_model_deployment'
    ],
    
    'stop_loss_override_conditions': [
        'portfolio_heat_exceeded',
        'market_stress_detected',
        'liquidity_crisis'
    ],
    
    # Cross-component communication
    'message_queue': 'redis',             # Inter-component messaging
    'state_persistence': 'postgresql',    # System state storage
    'cache_backend': 'redis',             # Caching system
    
    # Monitoring and logging
    'log_level': 'INFO',
    'enable_metrics': True,
    'metrics_backend': 'prometheus',
    'dashboard_enabled': True,
    
    # Failover and recovery
    'enable_circuit_breakers': True,
    'auto_recovery': True,
    'backup_frequency': 'hourly',
    'health_check_interval': 60,         # Health check every 60 seconds
}

# =============================================================================
# 6. ENVIRONMENT-SPECIFIC CONFIGURATIONS
# =============================================================================

# Development Environment
DEV_CONFIG = {
    'use_simulated_data': True,
    'enable_debug_logging': True,
    'testnet_only': True,
    'reduced_capital': 10000.0,
    'fast_execution': True,
}

# Staging Environment
STAGING_CONFIG = {
    'use_simulated_data': False,
    'testnet_only': True,
    'full_capital': True,
    'performance_monitoring': True,
    'alert_testing': True,
}

# Production Environment
PROD_CONFIG = {
    'use_simulated_data': False,
    'testnet_only': False,
    'full_capital': True,
    'high_availability': True,
    'enhanced_monitoring': True,
    'regulatory_compliance': True,
}

# =============================================================================
# 7. EXAMPLE USAGE CONFIGURATIONS
# =============================================================================

# Conservative Trading Configuration
CONSERVATIVE_CONFIG = {
    'walk_forward': {
        **WALK_FORWARD_CONFIG,
        'performance_threshold': 0.08,    # Higher performance threshold
        'retrain_frequency_days': 14,     # More frequent retraining
    },
    'stops': {
        **DYNAMIC_STOPS_CONFIG,
        'atr_multiplier': 1.5,            # Tighter stops
        'max_loss_per_trade': 0.015,      # Lower loss limit
    },
    'portfolio': {
        **PORTFOLIO_CONFIG,
        'max_asset_weight': 0.2,          # More diversification
        'rebalance_threshold': 0.03,      # More frequent rebalancing
    },
    'trading': {
        **REAL_TIME_TRADING_CONFIG,
        'signal_threshold': 0.8,          # Higher signal threshold
        'max_daily_loss': 0.015,          # Lower daily loss limit
    }
}

# Aggressive Trading Configuration
AGGRESSIVE_CONFIG = {
    'walk_forward': {
        **WALK_FORWARD_CONFIG,
        'performance_threshold': 0.03,    # Lower performance threshold
        'retrain_frequency_days': 60,     # Less frequent retraining
    },
    'stops': {
        **DYNAMIC_STOPS_CONFIG,
        'atr_multiplier': 3.0,            # Wider stops
        'max_loss_per_trade': 0.03,       # Higher loss limit
    },
    'portfolio': {
        **PORTFOLIO_CONFIG,
        'max_asset_weight': 0.4,          # More concentration
        'rebalance_threshold': 0.08,      # Less frequent rebalancing
    },
    'trading': {
        **REAL_TIME_TRADING_CONFIG,
        'signal_threshold': 0.4,          # Lower signal threshold
        'max_daily_loss': 0.03,           # Higher daily loss limit
    }
}

# =============================================================================
# CONFIGURATION VALIDATION FUNCTIONS
# =============================================================================

def validate_configuration(config_dict):
    """
    Validate configuration parameters
    
    Args:
        config_dict: Configuration dictionary to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Add validation logic here
    required_keys = ['walk_forward', 'stops', 'portfolio', 'trading']
    
    for key in required_keys:
        if key not in config_dict:
            raise ValueError(f"Missing required configuration section: {key}")
    
    return True

def get_config_for_environment(environment='development'):
    """
    Get configuration for specific environment
    
    Args:
        environment: 'development', 'staging', or 'production'
        
    Returns:
        dict: Environment-specific configuration
    """
    base_config = {
        'walk_forward': WALK_FORWARD_CONFIG,
        'stops': DYNAMIC_STOPS_CONFIG,
        'portfolio': PORTFOLIO_CONFIG,
        'trading': REAL_TIME_TRADING_CONFIG,
        'integrated': INTEGRATED_CONFIG,
    }
    
    if environment == 'development':
        base_config.update(DEV_CONFIG)
    elif environment == 'staging':
        base_config.update(STAGING_CONFIG)
    elif environment == 'production':
        base_config.update(PROD_CONFIG)
    
    return base_config

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Get conservative configuration for production
    conservative_prod_config = {
        **CONSERVATIVE_CONFIG,
        **PROD_CONFIG
    }
    
    print("Conservative Production Configuration:")
    print(f"Signal Threshold: {conservative_prod_config['trading']['signal_threshold']}")
    print(f"Max Daily Loss: {conservative_prod_config['trading']['max_daily_loss']}")
    print(f"Max Asset Weight: {conservative_prod_config['portfolio']['max_asset_weight']}")
    
    # Validate configuration
    try:
        validate_configuration(conservative_prod_config)
        print("Configuration validation: PASSED")
    except ValueError as e:
        print(f"Configuration validation: FAILED - {e}")
