# Advanced Features Examples

This directory contains examples and demonstrations of the four advanced features implemented in the AlphaBeta808Trading system:

1. **Walk-Forward Validation** with rolling window model retraining
2. **Dynamic Stop-Loss Mechanisms** with multiple stop-loss strategies  
3. **Multi-Asset Portfolio Management** with optimization algorithms
4. **Real-Time Trading** with Binance integration

## 📁 Files Overview

### 🚀 `advanced_features_demo.py`
**Main demonstration script** that showcases all four advanced features.

**Features demonstrated:**
- Walk-Forward Validation with concept drift detection
- Dynamic Stop-Loss with ATR, volatility, and trailing stops
- Multi-Asset Portfolio optimization and rebalancing
- Real-Time Trading setup and strategy implementation
- Integrated workflow showing all features working together

**Usage:**
```bash
cd /Users/bastienjavaux/Desktop/AlphaBeta808Trading/examples
python advanced_features_demo.py
```

### ⚙️ `configuration_examples.py`
**Comprehensive configuration examples** for all advanced features.

**Includes:**
- Individual feature configurations
- Environment-specific settings (dev/staging/prod)
- Conservative vs Aggressive trading profiles
- Configuration validation functions

**Key configurations:**
- Walk-Forward: Training windows, retraining frequency, drift thresholds
- Dynamic Stops: ATR multipliers, trailing percentages, risk limits
- Portfolio: Asset weights, rebalancing triggers, optimization methods
- Real-Time: Signal thresholds, position limits, risk management

### 🧪 `integration_tests.py`
**Integration test suite** to verify all features work correctly.

**Tests include:**
- Module import verification
- Basic functionality testing
- Integration workflow testing
- Error handling validation
- Performance metrics calculation
- Compatibility with existing system

**Usage:**
```bash
cd /Users/bastienjavaux/Desktop/AlphaBeta808Trading/examples
python integration_tests.py
```

## 🚀 Quick Start Guide

### 1. Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install -r ../requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root with your API credentials (optional for demos):
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

### 3. Run the Demo

Execute the main demonstration:
```bash
python advanced_features_demo.py
```

This will run through all four advanced features and show how they integrate together.

### 4. Run Tests

Verify everything is working:
```bash
python integration_tests.py
```

## 📋 Feature Details

### 1. Walk-Forward Validation (`/src/validation/walk_forward.py`)

**Purpose:** Implement rolling window model validation with concept drift detection and adaptive model management.

**Key Components:**
- `WalkForwardValidator`: Main validation engine
- `AdaptiveModelManager`: Dynamic model selection and performance tracking

**Configuration Example:**
```python
from configuration_examples import WALK_FORWARD_CONFIG
validator = WalkForwardValidator(
    train_window_months=WALK_FORWARD_CONFIG['train_window_months'],
    test_window_months=WALK_FORWARD_CONFIG['test_window_months'],
    retrain_frequency_days=WALK_FORWARD_CONFIG['retrain_frequency_days']
)
```

**Key Features:**
- Rolling window training and testing
- Concept drift detection using statistical tests
- Automatic model retraining triggers
- Performance-based model selection
- Real-time model adaptation

### 2. Dynamic Stop-Loss (`/src/risk_management/dynamic_stops.py`)

**Purpose:** Advanced stop-loss mechanisms with multiple strategies and dynamic adjustments.

**Key Components:**
- `DynamicStopLossManager`: Main stop-loss management system

**Stop-Loss Types:**
- ATR-based stops
- Volatility-adjusted stops  
- Trailing stops
- Support/Resistance levels
- Time-based exits
- Fixed percentage stops
- Portfolio heat stops

**Configuration Example:**
```python
from configuration_examples import DYNAMIC_STOPS_CONFIG
stop_manager = DynamicStopLossManager()

stop_manager.set_atr_stop_loss(
    symbol='BTCUSDT',
    atr_value=atr_value,
    entry_price=entry_price,
    position_type='long',
    atr_multiplier=DYNAMIC_STOPS_CONFIG['atr_multiplier']
)
```

### 3. Multi-Asset Portfolio Management (`/src/portfolio/multi_asset.py`)

**Purpose:** Portfolio optimization, rebalancing, and risk management across multiple assets.

**Key Components:**
- `MultiAssetPortfolioManager`: Main portfolio management system

**Optimization Methods:**
- Mean-Variance Optimization
- Risk Parity
- Black-Litterman
- Equal Weight
- Minimum Variance

**Configuration Example:**
```python
from configuration_examples import PORTFOLIO_CONFIG
portfolio_manager = MultiAssetPortfolioManager(
    initial_capital=PORTFOLIO_CONFIG['initial_capital'],
    rebalance_frequency=PORTFOLIO_CONFIG['rebalance_frequency']
)

# Optimize portfolio
weights = portfolio_manager.optimize_portfolio(
    method='mean_variance',
    target_return=0.15
)
```

### 4. Real-Time Trading (`/src/execution/real_time_trading.py`)

**Purpose:** Live trading integration with Binance, WebSocket data streams, and automated execution.

**Key Components:**
- `BinanceRealTimeTrader`: Main trading interface
- `TradingStrategy`: Strategy execution framework
- `RiskManager`: Real-time risk controls
- `MarketDataStream`: WebSocket data handling

**Configuration Example:**
```python
from configuration_examples import REAL_TIME_TRADING_CONFIG

trader = BinanceRealTimeTrader(
    api_key=api_key,
    api_secret=api_secret,
    testnet=REAL_TIME_TRADING_CONFIG['testnet']
)

strategy = TradingStrategy(trader)
strategy.add_symbol('BTCUSDT')
```

## 🔧 Configuration Profiles

### Conservative Trading
- Higher signal thresholds (0.8)
- Tighter stop-losses (1.5x ATR)
- Lower position limits (2% max loss)
- More frequent rebalancing

### Aggressive Trading  
- Lower signal thresholds (0.4)
- Wider stop-losses (3.0x ATR)
- Higher position limits (3% max loss)
- Less frequent rebalancing

### Example Usage:
```python
from configuration_examples import CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG

# Use conservative settings
config = CONSERVATIVE_CONFIG
```

## 📊 Integration Example

Here's how all four features work together:

```python
# 1. Initialize all components
validator = WalkForwardValidator()
stop_manager = DynamicStopLossManager() 
portfolio_manager = MultiAssetPortfolioManager(initial_capital=100000)
trader = BinanceRealTimeTrader(api_key, api_secret, testnet=True)

# 2. Add assets to portfolio
for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
    portfolio_manager.add_asset(symbol, price_data, returns_data)

# 3. Optimize portfolio allocation
weights = portfolio_manager.optimize_portfolio(method='mean_variance')

# 4. For each asset, generate signals and manage risk
for symbol in symbols:
    # Model validation
    model_performance = validator.get_current_model_performance()
    
    # Generate trading signal
    signal = generate_trading_signal(symbol)
    
    # Risk management
    if signal_strength > threshold and model_performance > min_performance:
        # Set stop-loss
        stop_manager.set_dynamic_stops(symbol, current_price, signal_direction)
        
        # Execute trade
        trader.place_order(symbol, signal, weights[symbol])
```

## 🔍 Monitoring and Debugging

### Logging
All modules include comprehensive logging. Set log level in configuration:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Performance Monitoring
Track system performance:
```python
# Portfolio performance
performance = portfolio_manager.calculate_portfolio_performance()

# Risk metrics  
risk_metrics = portfolio_manager.calculate_risk_metrics()

# Stop-loss effectiveness
stop_metrics = stop_manager.calculate_portfolio_risk_metrics()
```

### Health Checks
Verify system status:
```python
# Run integration tests
python integration_tests.py

# Check API connectivity (for real-time trading)
trader.test_connectivity()

# Validate configurations
from configuration_examples import validate_configuration
validate_configuration(your_config)
```

## 🚨 Important Notes

### 1. **API Keys**
- Keep API keys secure and never commit them to version control
- Use testnet for development and testing
- Set proper permissions on production API keys

### 2. **Risk Management**
- Always test strategies in simulation before live trading
- Set appropriate position limits and stop-losses
- Monitor portfolio heat and risk metrics

### 3. **Performance**
- The system is designed for high-frequency operation
- Consider computational resources for complex optimizations
- Use appropriate rebalancing frequencies

### 4. **Data Quality**
- Ensure clean, validated market data
- Handle missing data and outliers appropriately
- Implement data quality checks

## 📞 Support

For questions or issues with the advanced features:

1. Run the integration tests to verify functionality
2. Check the configuration examples for proper setup
3. Review the demo script for usage patterns
4. Examine log files for error details

## 🎯 Next Steps

After familiarizing yourself with these examples:

1. **Customize configurations** for your specific trading strategy
2. **Backtest your strategy** using the Walk-Forward Validation
3. **Test in simulation** before live trading
4. **Monitor performance** and adjust parameters as needed
5. **Scale up gradually** from small positions to full capital

The AlphaBeta808Trading system with these advanced features provides a robust foundation for sophisticated algorithmic trading strategies. Use these examples as a starting point and adapt them to your specific requirements.
