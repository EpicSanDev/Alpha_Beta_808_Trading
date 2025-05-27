#!/usr/bin/env python3
"""
Test the actual simulator's run_simulation method to verify the boolean indexing fix.
This creates a minimal simulation test without the complex dependencies.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_simulator_boolean_indexing():
    """Test the simulator's run_simulation method with the boolean indexing fix."""
    print("Testing Simulator Boolean Indexing Fix")
    print("=" * 60)
    
    try:
        # Import the simulator with proper path and class name
        from src.execution.simulator import BacktestSimulator
        
        # Create minimal test data
        start_time = datetime(2024, 1, 1, 9, 0)
        
        # Market data (ETHUSDT format to match the original error)
        market_times = [start_time + timedelta(minutes=i) for i in range(100)]
        market_data = pd.DataFrame({
            'open': 2000 + np.random.randn(100).cumsum(),
            'high': 2010 + np.random.randn(100).cumsum(), 
            'low': 1990 + np.random.randn(100).cumsum(),
            'close': 2000 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=pd.DatetimeIndex(market_times))
        
        # Trading signals with required columns
        signal_times = [start_time + timedelta(minutes=i*3) for i in range(35)]  # Every 3 minutes
        signals_df = pd.DataFrame({
            'signal': np.random.choice([1, -1, 0], 35),
            'nominal_value_to_trade': np.random.uniform(100, 1000, 35)  # Required column
        }, index=pd.DatetimeIndex(signal_times))
        
        print(f"Market data: {len(market_data)} rows from {market_data.index.min()} to {market_data.index.max()}")
        print(f"Signals data: {len(signals_df)} rows from {signals_df.index.min()} to {signals_df.index.max()}")
        
        # Create simulator instance
        simulator = BacktestSimulator(
            initial_capital=10000.0,
            market_data=market_data,
            asset_symbol="ETHUSDT",
            commission_pct_per_trade=0.001,
            default_leverage=1.0
        )
        
        print("\nRunning simulation with boolean indexing fix...")
        
        # This should not raise the "Unalignable boolean Series provided as indexer" error
        simulator.run_simulation(signals_df)
        
        print(f"✓ Simulation completed successfully!")
        
        # Get results using the simulator's methods
        portfolio_history = simulator.get_portfolio_history()
        trades_history = simulator.get_trades_history()
        
        print(f"  - Portfolio history recorded: {len(portfolio_history)} entries")
        print(f"  - Trades executed: {len(trades_history)} trades")
        if len(portfolio_history) > 0:
            final_value = portfolio_history.iloc[-1]['capital'] if 'capital' in portfolio_history.columns else "N/A"
            print(f"  - Final capital: {final_value}")
        
        # Test with edge case: signals outside market data range
        print("\nTesting edge case: signals outside market data range...")
        
        edge_case_signals = pd.DataFrame({
            'signal': [1, -1, 0],
            'nominal_value_to_trade': [500, 750, 300]  # Required column
        }, index=pd.DatetimeIndex([
            start_time - timedelta(minutes=30),  # Before market data
            start_time + timedelta(minutes=50),   # Within market data
            start_time + timedelta(minutes=200)   # After market data
        ]))
        
        # Create a new simulator for the edge case
        edge_simulator = BacktestSimulator(
            initial_capital=10000.0,
            market_data=market_data,
            asset_symbol="ETHUSDT",
            commission_pct_per_trade=0.001,
            default_leverage=1.0
        )
        
        edge_simulator.run_simulation(edge_case_signals)
        edge_portfolio = edge_simulator.get_portfolio_history()
        print(f"✓ Edge case handled successfully!")
        print(f"  - Portfolio history: {len(edge_portfolio)} entries")
        
        print("\n" + "=" * 60)
        print("✓ SIMULATOR BOOLEAN INDEXING FIX TEST PASSED!")
        print("The simulator successfully handles the boolean indexing scenario")
        print("that was causing 'Unalignable boolean Series provided as indexer' errors.")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Simulator could not be imported. This may be due to missing dependencies.")
        return False
        
    except Exception as e:
        print(f"\n✗ Simulator test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simulator_boolean_indexing()
    sys.exit(0 if success else 1)
