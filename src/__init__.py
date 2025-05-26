# src/__init__.py
"""
AlphaBeta808Trading - Advanced Algorithmic Trading System

This package provides a comprehensive framework for algorithmic trading
with machine learning, featuring advanced risk management, portfolio 
optimization, and real-time execution capabilities.
"""

__version__ = "1.0.0"
__author__ = "AlphaBeta808"

# Import main modules for easier access
try:
    from .acquisition import connectors, preprocessing
    from .feature_engineering import technical_features
    from .modeling import models
    from .signal_generation import signal_generator
    from .execution import simulator, real_time_trading
    from .risk_management import dynamic_stops, risk_controls
    from .portfolio import multi_asset
    from .validation import walk_forward
    from .core import performance_analyzer
except ImportError as e:
    # Allow imports to fail gracefully for development
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}", ImportWarning)
