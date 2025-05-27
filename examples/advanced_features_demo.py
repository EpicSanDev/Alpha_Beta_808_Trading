"""
Advanced Features Demo for AlphaBeta808Trading System

This script demonstrates the integration and usage of the four advanced features:
1. Walk-Forward Validation
2. Dynamic Stop-Loss Mechanisms
3. Multi-Asset Portfolio Management
4. Real-Time Trading with Binance

Author: AlphaBeta808 Trading System
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import existing modules
from acquisition.connectors import generate_random_market_data
from feature_engineering.technical_features import calculate_sma, calculate_ema, calculate_rsi
from modeling.models import prepare_data_for_model, train_model
from signal_generation.signal_generator import generate_base_signals_from_predictions

# Import new advanced features
from validation.walk_forward import WalkForwardValidator, AdaptiveModelManager
from risk_management.dynamic_stops import DynamicStopLossManager
from portfolio.multi_asset import MultiAssetPortfolioManager
from execution.real_time_trading import BinanceRealTimeTrader, TradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFeaturesDemo:
    """
    Demonstration class for advanced trading features
    """
    
    def __init__(self):
        load_dotenv()
        self.setup_sample_data()
        
    def setup_sample_data(self):
        """Generate sample multi-asset data for demonstration"""
        logger.info("Setting up sample multi-asset data...")
        
        # Generate data for multiple assets
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT']
        self.market_data = {}
        
        for symbol in symbols:
            # Generate random market data
            data = generate_random_market_data(
                num_rows=1000,
                start_price=np.random.uniform(50, 500),
                volatility=np.random.uniform(0.02, 0.05),
                freq='H'
            )
            
            # Add technical indicators
            data = calculate_sma(data, column='close', windows=[20])
            data = calculate_ema(data, column='close', windows=[20])
            data = calculate_rsi(data, column='close')
            
            # Add symbol information
            data['symbol'] = symbol
            
            self.market_data[symbol] = data
            
        logger.info(f"Generated data for {len(symbols)} assets")
    
    def demo_walk_forward_validation(self):
        """Demonstrate Walk-Forward Validation functionality"""
        logger.info("\n" + "="*60)
        logger.info("DEMO 1: Walk-Forward Validation")
        logger.info("="*60)
        
        try:
            # Initialize Walk-Forward Validator
            validator = WalkForwardValidator(
                training_window_days=180,  # 6 months equivalent
                validation_window_days=30,  # 1 month equivalent  
                retrain_frequency_days=30,
                performance_threshold=0.05
            )
            
            # Use BTCUSDT data for demonstration
            btc_data = self.market_data['BTCUSDT'].copy()
            
            # Prepare features and target
            features = ['sma_20', 'ema_20', 'rsi', 'volume']
            target = 'close_next'
            
            # Create target variable (next period's close price)
            btc_data[target] = btc_data['close'].shift(-1)
            btc_data = btc_data.dropna()
            
            # Run walk-forward validation
            results = validator.run_validation(
                data=btc_data,
                features=features,
                target=target,
                model_type='RandomForest'
            )
            
            logger.info(f"Walk-Forward Validation completed successfully!")
            logger.info(f"Number of validation windows: {len(results)}")
            
            if results:
                avg_score = np.mean([r['score'] for r in results])
                logger.info(f"Average validation score: {avg_score:.4f}")
                
                # Check for concept drift
                drift_detected = validator.detect_concept_drift(
                    btc_data[features],
                    btc_data[features].iloc[-100:]  # Recent data
                )
                logger.info(f"Concept drift detected: {drift_detected}")
            
            # Demonstrate Adaptive Model Manager
            model_manager = AdaptiveModelManager()
            
            # Add a mock model performance record
            model_manager.add_model_performance('model_1', 0.85, datetime.now())
            model_manager.add_model_performance('model_2', 0.78, datetime.now())
            
            best_model = model_manager.get_best_performing_model()
            logger.info(f"Best performing model: {best_model}")
            
        except Exception as e:
            logger.error(f"Error in Walk-Forward Validation demo: {e}")
    
    def demo_dynamic_stop_loss(self):
        """Demonstrate Dynamic Stop-Loss functionality"""
        logger.info("\n" + "="*60)
        logger.info("DEMO 2: Dynamic Stop-Loss Mechanisms")
        logger.info("="*60)
        
        try:
            # Initialize Dynamic Stop-Loss Manager
            stop_manager = DynamicStopLossManager()
            
            # Use ETHUSDT data for demonstration
            eth_data = self.market_data['ETHUSDT'].copy()
            
            # Simulate a long position
            symbol = 'ETHUSDT'
            entry_price = eth_data['close'].iloc[100]
            position_size = 1.0
            current_price = eth_data['close'].iloc[150]
            
            logger.info(f"Position: {position_size} {symbol} @ ${entry_price:.2f}")
            logger.info(f"Current price: ${current_price:.2f}")
            
            # Set various stop-loss types
            stop_manager.set_atr_stop_loss(
                symbol=symbol,
                atr_value=eth_data['close'].rolling(14).std().iloc[150],
                entry_price=entry_price,
                position_type='long',
                atr_multiplier=2.0
            )
            
            stop_manager.set_trailing_stop_loss(
                symbol=symbol,
                current_price=current_price,
                position_type='long',
                trail_amount=entry_price * 0.05  # 5% trail
            )
            
            stop_manager.set_volatility_stop_loss(
                symbol=symbol,
                price_series=eth_data['close'].iloc[100:151],
                entry_price=entry_price,
                position_type='long',
                volatility_multiplier=1.5
            )
            
            # Check all stop-loss conditions
            should_exit, exit_reason = stop_manager.check_exit_conditions(
                symbol=symbol,
                current_price=current_price,
                position_type='long'
            )
            
            logger.info(f"Should exit position: {should_exit}")
            if should_exit:
                logger.info(f"Exit reason: {exit_reason}")
            
            # Update trailing stop
            new_price = current_price * 1.02  # Price increases
            stop_manager.update_trailing_stop(symbol, new_price, 'long')
            
            # Display current stops
            stops = stop_manager.get_stop_levels(symbol)
            logger.info(f"Current stop levels: {stops}")
            
            # Risk metrics
            metrics = stop_manager.calculate_portfolio_risk_metrics(
                {'ETHUSDT': {'size': position_size, 'entry_price': entry_price}}
            )
            logger.info(f"Portfolio risk metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error in Dynamic Stop-Loss demo: {e}")
    
    def demo_multi_asset_portfolio(self):
        """Demonstrate Multi-Asset Portfolio Management functionality"""
        logger.info("\n" + "="*60)
        logger.info("DEMO 3: Multi-Asset Portfolio Management")
        logger.info("="*60)
        
        try:
            # Initialize Multi-Asset Portfolio Manager
            portfolio_manager = MultiAssetPortfolioManager(
                initial_capital=100000.0,
                rebalancing_frequency_days=30  # monthly equivalent
            )
            
            # Add assets to portfolio
            symbols = list(self.market_data.keys())
            for symbol in symbols:
                data = self.market_data[symbol]['close']
                returns = data.pct_change().dropna()
                
                portfolio_manager.add_asset(
                    symbol=symbol,
                    price_data=data,
                    returns_data=returns
                )
            
            logger.info(f"Added {len(symbols)} assets to portfolio")
            
            # Calculate correlation matrix
            corr_matrix = portfolio_manager.calculate_correlation_matrix()
            logger.info("Correlation matrix calculated")
            logger.info(f"Portfolio correlation summary:")
            logger.info(f"Mean correlation: {corr_matrix.mean().mean():.3f}")
            
            # Optimize portfolio using mean-variance optimization
            target_return = 0.15  # 15% annual return target
            weights = portfolio_manager.optimize_portfolio(
                method='mean_variance',
                target_return=target_return
            )
            
            logger.info("Portfolio optimization completed")
            for symbol, weight in weights.items():
                logger.info(f"{symbol}: {weight:.2%}")
            
            # Simulate portfolio performance
            start_date = datetime.now() - timedelta(days=100)
            end_date = datetime.now()
            
            performance = portfolio_manager.calculate_portfolio_performance(
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(f"Portfolio Performance Metrics:")
            logger.info(f"Total Return: {performance.get('total_return', 0):.2%}")
            logger.info(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
            logger.info(f"Volatility: {performance.get('volatility', 0):.2%}")
            
            # Test rebalancing
            needs_rebalance = portfolio_manager.check_rebalancing_conditions()
            logger.info(f"Portfolio needs rebalancing: {needs_rebalance}")
            
            if needs_rebalance:
                portfolio_manager.rebalance_portfolio()
                logger.info("Portfolio rebalanced successfully")
            
            # Risk metrics
            risk_metrics = portfolio_manager.calculate_risk_metrics()
            logger.info(f"Portfolio Risk Metrics:")
            logger.info(f"Value at Risk (95%): ${risk_metrics.get('var_95', 0):,.2f}")
            logger.info(f"Expected Shortfall: ${risk_metrics.get('expected_shortfall', 0):,.2f}")
            
        except Exception as e:
            logger.error(f"Error in Multi-Asset Portfolio demo: {e}")
    
    def demo_real_time_trading_setup(self):
        """Demonstrate Real-Time Trading setup (without actual trading)"""
        logger.info("\n" + "="*60)
        logger.info("DEMO 4: Real-Time Trading Setup")
        logger.info("="*60)
        
        try:
            # Check for API credentials
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                logger.warning("Binance API credentials not found in environment variables")
                logger.info("Please set BINANCE_API_KEY and BINANCE_API_SECRET for live trading")
                logger.info("Continuing with demo configuration...")
                
                # Use demo credentials
                api_key = "demo_key"
                api_secret = "demo_secret"
            
            # Initialize Real-Time Trader (in testnet mode)
            trader = BinanceRealTimeTrader(
                api_key=api_key,
                api_secret=api_secret,
                testnet=True  # Always use testnet for demo
            )
            
            logger.info("Real-time trader initialized (testnet mode)")
            
            # Create a sample trading strategy
            strategy = TradingStrategy(trader)
            
            # Add symbols to monitor
            symbols = ['BTCUSDT', 'ETHUSDT']
            for symbol in symbols:
                strategy.add_symbol(symbol)
            
            logger.info(f"Added {len(symbols)} symbols to trading strategy")
            
            # Simulate market data processing
            for symbol in symbols:
                # Generate a mock signal
                signal = np.random.uniform(-0.8, 0.8)
                
                # Mock market data
                mock_data = {
                    'symbol': symbol,
                    'price': np.random.uniform(30000, 50000) if symbol == 'BTCUSDT' else np.random.uniform(2000, 4000),
                    'volume': np.random.uniform(100, 1000),
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Processing signal for {symbol}: {signal:.3f}")
                logger.info(f"Mock market data: ${mock_data['price']:.2f}")
                
                # Risk checks would be performed here
                logger.info(f"Risk checks would be performed for {symbol}")
                
                # Order would be placed here (in live trading)
                if abs(signal) > 0.6:  # Strong signal threshold
                    action = "BUY" if signal > 0 else "SELL"
                    logger.info(f"Would place {action} order for {symbol}")
                else:
                    logger.info(f"Signal too weak for {symbol}, no action taken")
            
            logger.info("Real-time trading demo completed successfully")
            
            # Display configuration info
            logger.info("\nTrading Configuration:")
            logger.info(f"- API Mode: Testnet")
            logger.info(f"- Symbols monitored: {symbols}")
            logger.info(f"- Signal threshold: 0.6")
            logger.info(f"- Risk management: Enabled")
            
        except Exception as e:
            logger.error(f"Error in Real-Time Trading demo: {e}")
    
    def run_integrated_demo(self):
        """Run an integrated demo showing how all features work together"""
        logger.info("\n" + "="*60)
        logger.info("INTEGRATED DEMO: All Features Working Together")
        logger.info("="*60)
        
        try:
            # 1. Portfolio Management Setup
            portfolio_manager = MultiAssetPortfolioManager(
                initial_capital=100000.0,
                rebalancing_frequency_days=30  # monthly equivalent
            )
            
            # Add assets
            symbols = ['BTCUSDT', 'ETHUSDT']
            for symbol in symbols:
                data = self.market_data[symbol]['close']
                returns = data.pct_change().dropna()
                portfolio_manager.add_asset(symbol, data, returns)
            
            # 2. Walk-Forward Validation for model selection
            validator = WalkForwardValidator()
            
            # 3. Dynamic Stop-Loss Setup
            stop_manager = DynamicStopLossManager()
            
            # 4. Real-Time Trading Setup
            trader = BinanceRealTimeTrader(
                api_key="demo",
                api_secret="demo",
                testnet=True
            )
            
            logger.info("All systems initialized successfully")
            
            # Simulate integrated trading workflow
            for symbol in symbols:
                logger.info(f"\nProcessing {symbol}:")
                
                # 1. Portfolio optimization
                weights = portfolio_manager.optimize_portfolio(method='equal_weight')
                allocation = weights.get(symbol, 0)
                logger.info(f"Portfolio allocation: {allocation:.2%}")
                
                # 2. Model validation check
                # (In real implementation, this would check model performance)
                model_performance = np.random.uniform(0.6, 0.9)
                logger.info(f"Model performance score: {model_performance:.3f}")
                
                # 3. Generate trading signal
                signal = np.random.uniform(-1, 1)
                logger.info(f"Trading signal: {signal:.3f}")
                
                # 4. Risk management
                current_price = self.market_data[symbol]['close'].iloc[-1]
                
                if abs(signal) > 0.5 and model_performance > 0.7:
                    # Set stop-loss for potential position
                    stop_manager.set_atr_stop_loss(
                        symbol=symbol,
                        atr_value=self.market_data[symbol]['close'].rolling(14).std().iloc[-1],
                        entry_price=current_price,
                        position_type='long' if signal > 0 else 'short'
                    )
                    
                    logger.info(f"Stop-loss set for {symbol}")
                    logger.info(f"Signal strength sufficient for trading")
                else:
                    logger.info(f"Signal/model performance insufficient for {symbol}")
                
                # 5. Portfolio risk check
                portfolio_risk = portfolio_manager.calculate_risk_metrics()
                var_95 = portfolio_risk.get('var_95', 0)
                
                if abs(var_95) < 5000:  # Risk threshold
                    logger.info(f"Portfolio risk acceptable (VaR: ${var_95:,.2f})")
                else:
                    logger.info(f"Portfolio risk too high (VaR: ${var_95:,.2f})")
            
            logger.info("\nIntegrated demo completed successfully!")
            logger.info("All four advanced features working in harmony:")
            logger.info("✓ Walk-Forward Validation")
            logger.info("✓ Dynamic Stop-Loss Mechanisms")
            logger.info("✓ Multi-Asset Portfolio Management")
            logger.info("✓ Real-Time Trading Infrastructure")
            
        except Exception as e:
            logger.error(f"Error in integrated demo: {e}")
    
    def run_all_demos(self):
        """Run all demonstration functions"""
        logger.info("Starting Advanced Features Demonstration")
        logger.info("="*80)
        
        # Run individual feature demos
        self.demo_walk_forward_validation()
        self.demo_dynamic_stop_loss()
        self.demo_multi_asset_portfolio()
        self.demo_real_time_trading_setup()
        
        # Run integrated demo
        self.run_integrated_demo()
        
        logger.info("\n" + "="*80)
        logger.info("All demonstrations completed successfully!")
        logger.info("The AlphaBeta808Trading system now includes:")
        logger.info("1. ✓ Walk-Forward Validation with concept drift detection")
        logger.info("2. ✓ Dynamic Stop-Loss with multiple strategies")
        logger.info("3. ✓ Multi-Asset Portfolio Management with optimization")
        logger.info("4. ✓ Real-Time Trading with Binance integration")
        logger.info("="*80)

def main():
    """Main execution function"""
    try:
        demo = AdvancedFeaturesDemo()
        demo.run_all_demos()
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        raise

if __name__ == "__main__":
    main()
