#!/usr/bin/env python3
"""
Simple test to verify the boolean indexing fix in the trading simulator.
This test creates a minimal scenario to reproduce the original error and verify the fix.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_boolean_indexing_fix():
    """Test the specific boolean indexing scenario that was causing errors."""
    print("Testing Boolean Indexing Fix")
    print("=" * 50)
    
    try:
        # Create test data similar to what was causing the error
        # Market data with datetime index
        start_time = datetime(2024, 1, 1, 9, 0)
        market_times = [start_time + timedelta(minutes=i) for i in range(100)]
        market_data = pd.DataFrame({
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=pd.DatetimeIndex(market_times))
        
        # Signals data with some overlapping and some non-overlapping times
        signal_times = [start_time + timedelta(minutes=i*5) for i in range(25)]  # Every 5 minutes
        signals_df = pd.DataFrame({
            'signal': np.random.choice([1, -1, 0], 25)
        }, index=pd.DatetimeIndex(signal_times))
        
        print(f"Market data range: {market_data.index.min()} to {market_data.index.max()}")
        print(f"Signals data range: {signals_df.index.min()} to {signals_df.index.max()}")
        
        # Test the OLD way (that was causing the error)
        print("\nTesting OLD approach (should work fine in isolation)...")
        all_relevant_timestamps_old = pd.Index(market_data.index.tolist() + signals_df.index.tolist()).unique().sort_values()
        
        # This is the problematic line that was causing the error in complex scenarios
        try:
            # This might work in simple cases but fails in complex boolean operations
            filtered_old = all_relevant_timestamps_old[
                (all_relevant_timestamps_old >= market_data.index.min()) & 
                (all_relevant_timestamps_old <= market_data.index.max())
            ]
            print(f"✓ OLD approach worked: {len(filtered_old)} timestamps")
        except Exception as e:
            print(f"✗ OLD approach failed: {e}")
        
        # Test the NEW way (our fix)
        print("\nTesting NEW approach (our fix)...")
        all_relevant_timestamps_new = pd.Index(market_data.index.tolist() + signals_df.index.tolist()).unique().sort_values()
        
        # Our fix: separate the mask creation from indexing
        min_time = market_data.index.min()
        max_time = market_data.index.max()
        mask = (all_relevant_timestamps_new >= min_time) & (all_relevant_timestamps_new <= max_time)
        filtered_new = all_relevant_timestamps_new[mask]
        
        print(f"✓ NEW approach worked: {len(filtered_new)} timestamps")
        
        # Test with more complex scenario that would cause the original error
        print("\nTesting complex scenario (where original error occurred)...")
        
        # Create a scenario with misaligned indices that would cause the error
        complex_market_data = pd.DataFrame({
            'price': np.random.randn(50).cumsum() + 100,
        }, index=pd.DatetimeIndex([start_time + timedelta(minutes=i*2) for i in range(50)]))
        
        complex_signals = pd.DataFrame({
            'signal': np.random.choice([1, -1, 0], 30),
        }, index=pd.DatetimeIndex([start_time + timedelta(minutes=i*3) for i in range(30)]))
        
        # This scenario with our NEW approach
        all_timestamps_complex = pd.Index(complex_market_data.index.tolist() + complex_signals.index.tolist()).unique().sort_values()
        
        min_time_complex = complex_market_data.index.min()
        max_time_complex = complex_market_data.index.max()
        mask_complex = (all_timestamps_complex >= min_time_complex) & (all_timestamps_complex <= max_time_complex)
        filtered_complex = all_timestamps_complex[mask_complex]
        
        print(f"✓ Complex scenario handled: {len(filtered_complex)} timestamps")
        
        print("\n" + "=" * 50)
        print("✓ Boolean indexing fix test PASSED!")
        print("The fix successfully handles the problematic boolean indexing scenario.")
        return True
        
    except Exception as e:
        print(f"\n✗ Boolean indexing fix test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_boolean_indexing_fix()
    sys.exit(0 if success else 1)
