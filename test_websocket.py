#!/usr/bin/env python3
"""
Test WebSocket connection for Binance trading bot
"""

import asyncio
import sys
import os
import signal
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')
from execution.real_time_trading import BinanceRealTimeTrader

# Load environment variables
load_dotenv()

# Global flag to handle interruption
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\nTest interrupted by user")

# Set up signal handler
signal.signal(signal.SIGINT, signal_handler)

async def test_websocket():
    """Test WebSocket connection"""
    global interrupted
    print("Starting WebSocket connection test...")
    
    try:
        trader = BinanceRealTimeTrader(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=True
        )
        
        print("Trader created successfully")
        
        # Subscribe to market data
        symbols = ['BTCUSDT', 'ETHUSDT']
        trader.subscribe_to_market_data(symbols)
        print(f"Subscribed to: {symbols}")
        
        # Start market data stream
        trader.start_market_data_stream()
        print("Market data stream started")
        
        # Wait to see if WebSocket connects and receives data
        for i in range(20):
            if interrupted:
                break
            await asyncio.sleep(1)
            if trader.market_data:
                print(f"✓ Received market data for {len(trader.market_data)} symbols:")
                for symbol, data in trader.market_data.items():
                    print(f"  {symbol}: ${data.price:.2f}")
                break
            if i % 5 == 0:
                print(f"  Waiting for data... ({i+1}/20)")
        
        if not trader.market_data and not interrupted:
            print("❌ No market data received")
        elif trader.market_data:
            print("✓ WebSocket connection successful!")
        
        print("Test completed")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        print("\nTest interrupted")
