# AlphaBeta808 24/7 Continuous Trading Bot

## 🚀 Overview

AlphaBeta808 is a sophisticated 24/7 automated trading system for Binance that combines machine learning, technical analysis, and real-time market data to execute trades continuously. The system has been transformed from a backtest-only framework into a complete real-time trading platform with advanced risk management and monitoring capabilities.

## ✨ Key Features

### 🔄 Real-Time Trading
- **24/7 Operation**: Continuous market scanning and trading execution
- **Multi-Asset Support**: Trade multiple cryptocurrencies simultaneously (BTC, ETH, ADA, DOT, etc.)
- **WebSocket Integration**: Real-time market data streaming from Binance
- **Async Architecture**: High-performance asynchronous trading operations

### 🧠 Machine Learning
- **ML-Powered Signals**: Advanced signal generation using trained models
- **Automatic Retraining**: Periodic model updates with new market data
- **Feature Engineering**: 20+ technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- **Multi-Timeframe Analysis**: 1h, 4h, and daily data integration

### 🛡️ Risk Management
- **Position Limits**: Maximum position size controls
- **Daily Loss Limits**: Automatic trading halt on excessive losses
- **Exposure Controls**: Total portfolio exposure management
- **Emergency Stops**: Automatic position reduction in high-risk scenarios
- **Order Rate Limiting**: Protection against API rate limits

### 📊 Monitoring & Analytics
- **Real-Time Performance**: Live P&L tracking and portfolio monitoring
- **Health Checks**: System status monitoring and error detection
- **Performance Reports**: Automated hourly performance summaries
- **Trading Logs**: Comprehensive logging for analysis and debugging

## 🏗️ System Architecture

```
AlphaBeta808Trading/
├── 🤖 Live Trading Bots
│   ├── live_trading_bot.py      # Main trading bot with ML integration
│   ├── continuous_trader.py     # Alternative continuous trading implementation
│   └── bot_manager.py           # Bot lifecycle management
├── 📊 Core Components
│   └── src/
│       ├── acquisition/         # Market data collection
│       ├── feature_engineering/ # Technical indicator calculation
│       ├── modeling/           # ML model training and prediction
│       ├── signal_generation/  # Trading signal generation
│       ├── execution/          # Order execution and management
│       ├── risk_management/    # Risk controls and limits
│       └── portfolio/          # Portfolio management
├── ⚙️ Configuration
│   ├── trader_config.json      # Trading parameters and settings
│   └── .env                    # API keys and secrets
└── 📈 Analysis Tools
    ├── main.py                 # Backtesting system
    └── examples/               # Demo scripts and tutorials
```

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd AlphaBeta808Trading

# Activate the virtual environment
source trading_env/bin/activate  # Linux/Mac
# or
trading_env\Scripts\activate     # Windows

# Install dependencies (already included in trading_env)
pip install -r requirements.txt
```

### 2. Configuration

#### API Keys (.env file)
```bash
# The .env file is already configured with API keys:
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
```

#### Trading Configuration (trader_config.json)
Key settings in `trader_config.json`:
```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "intervals": ["1h", "4h", "1d"],
    "base_currency": "USDT"
  },
  "risk_management": {
    "max_position_size": 0.1,
    "max_daily_loss": 0.05,
    "max_total_exposure": 0.8
  }
}
```

### 3. Launch Trading Bot

#### Option A: Main Live Trading Bot
```bash
python live_trading_bot.py
```

#### Option B: Continuous Trader
```bash
python continuous_trader.py
```

#### Option C: Bot Manager (Recommended)
```bash
python bot_manager.py start    # Start the bot
python bot_manager.py status   # Check bot status
python bot_manager.py stop     # Stop the bot
python bot_manager.py logs     # View logs
```

## 🎯 Trading Strategies

### Signal Generation Process
1. **Data Collection**: Real-time market data via WebSocket
2. **Feature Engineering**: Calculate 20+ technical indicators
3. **ML Prediction**: Generate signals using trained models
4. **Risk Assessment**: Apply risk management filters
5. **Order Execution**: Place optimized buy/sell orders

### Supported Indicators
- **Trend**: SMA, EMA, MACD
- **Momentum**: RSI, Price Momentum
- **Volatility**: Bollinger Bands
- **Volume**: Volume-Price Analysis
- **Custom**: AlphaBeta808 proprietary features

## 🛡️ Risk Management Features

### Position Management
- **Max Position Size**: 10% of portfolio per position (configurable)
- **Dynamic Sizing**: Position sizing based on signal strength
- **Correlation Limits**: Prevent over-concentration in correlated assets

### Loss Protection
- **Daily Loss Limit**: 5% daily loss threshold (configurable)
- **Emergency Stop**: Automatic position reduction at 3% loss
- **Drawdown Controls**: Maximum drawdown protection

### Technical Safeguards
- **API Rate Limiting**: Respects Binance rate limits
- **Connection Monitoring**: Automatic reconnection on failures
- **Order Validation**: Pre-trade risk checks

## 📊 Monitoring Dashboard

### Real-Time Metrics
- Portfolio Value and P&L
- Active Positions and Orders
- Signal Generation Rate
- Risk Exposure Levels

### Performance Analytics
- Daily/Weekly/Monthly Returns
- Sharpe Ratio and Risk Metrics
- Win Rate and Average Trade Size
- Drawdown Analysis

### Health Monitoring
- System Uptime
- API Connection Status
- Error Rate and Alerts
- Memory and CPU Usage

## 🔧 Configuration Guide

### Trading Parameters
```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT"],     // Assets to trade
    "max_positions": 5,                     // Max concurrent positions
    "position_size": 0.1,                  // Position size (10% of capital)
    "signal_threshold": 0.6                 // Minimum signal strength
  }
}
```

### Risk Management
```json
{
  "risk_management": {
    "max_position_size": 0.1,              // 10% max position
    "max_daily_loss": 0.05,                // 5% daily loss limit
    "max_total_exposure": 0.8,             // 80% max portfolio exposure
    "stop_loss_pct": 0.02,                 // 2% stop loss
    "take_profit_pct": 0.04                // 4% take profit
  }
}
```

### Model Settings
```json
{
  "model": {
    "model_path": "models_store/logistic_regression_mvp.joblib",
    "retrain_interval": 86400,             // 24 hours
    "feature_lookback": 100,               // 100 periods
    "prediction_threshold": 0.55           // Signal threshold
  }
}
```

## 📈 Performance Monitoring

### Log Files
- `live_trading_bot.log`: Main bot operations
- `continuous_trader.log`: Alternative bot logs
- `reports/performance_*.txt`: Hourly performance reports

### Key Metrics to Monitor
- **Total Return**: Cumulative portfolio performance
- **Daily P&L**: Daily profit/loss tracking
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted returns

## 🚨 Safety Features

### Emergency Controls
- **Ctrl+C**: Graceful shutdown with position cleanup
- **Emergency Stop**: Immediate position liquidation
- **Risk Limits**: Automatic trading halt on limit breach

### Backup Systems
- **Configuration Backup**: Auto-save trading parameters
- **State Persistence**: Resume trading after restart
- **Error Recovery**: Automatic error handling and recovery

## 🛠️ Troubleshooting

### Common Issues

#### Connection Problems
```bash
# Check API keys
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('BINANCE_API_KEY')[:10] + '...')"

# Test Binance connection
python -c "from binance.client import Client; import os; from dotenv import load_dotenv; load_dotenv(); client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET')); print(client.get_server_time())"
```

#### Bot Not Starting
1. Check log files for error messages
2. Verify API key permissions
3. Ensure sufficient account balance
4. Check network connectivity

#### Performance Issues
1. Monitor CPU and memory usage
2. Check log file sizes
3. Verify database connections
4. Review trading pair liquidity

### Debug Mode
```bash
# Run with debug logging
export PYTHONPATH=/Users/bastienjavaux/Desktop/AlphaBeta808Trading/src
python -m pdb live_trading_bot.py
```

## 📋 System Requirements

### Hardware
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection

### Software
- **Python**: 3.13+
- **Operating System**: Linux, macOS, or Windows
- **Dependencies**: All included in `trading_env/`

### Binance Account
- **API Access**: Enabled
- **Trading Permissions**: Spot trading enabled
- **Balance**: Minimum $100 USDT recommended

## 🎓 Advanced Usage

### Custom Strategies
Implement custom trading strategies by modifying:
- `src/signal_generation/signal_generator.py`
- `src/modeling/models.py`
- `trader_config.json`

### Model Training
```bash
# Retrain models with new data
python main.py  # Run backtesting system
# Models auto-saved to models_store/
```

### Multi-Exchange Support
The architecture supports extension to other exchanges:
- Implement new connector in `src/acquisition/`
- Add exchange-specific execution logic
- Update configuration files

## 📞 Support & Community

### Documentation
- **API Reference**: See `src/` module docstrings
- **Configuration**: `trader_config.json` comments
- **Examples**: `examples/` directory

### Best Practices
1. **Start Small**: Begin with small position sizes
2. **Monitor Closely**: Watch initial performance carefully
3. **Backup Regularly**: Save configurations and logs
4. **Test First**: Use paper trading before live trading
5. **Stay Updated**: Keep dependencies current

## ⚠️ Disclaimers

- **Risk Warning**: Cryptocurrency trading involves significant risk
- **No Guarantees**: Past performance doesn't guarantee future results
- **Testing**: Thoroughly test strategies before live deployment
- **Responsibility**: Users are responsible for their own trading decisions

## 🔄 Version History

- **v2.0**: Complete real-time trading system with ML integration
- **v1.0**: Backtesting framework with basic features

---

**Happy Trading! 🚀**

*Built with ❤️ by the AlphaBeta808 Team*
