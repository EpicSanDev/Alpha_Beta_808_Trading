"""
TensorFlow compatibility layer for production deployment.
Handles graceful fallback when TensorFlow is not available.
"""

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
    from tensorflow.keras.layers import Reshape, Add, Activation, BatchNormalization
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # Create mock objects for TensorFlow components
    TENSORFLOW_AVAILABLE = False
    
    class MockTensorFlow:
        """Mock TensorFlow for when it's not available"""
        def __getattr__(self, name):
            return MockTensorFlow()
        
        def __call__(self, *args, **kwargs):
            return MockTensorFlow()
    
    class MockKeras:
        """Mock Keras for when TensorFlow is not available"""
        models = MockTensorFlow()
        layers = MockTensorFlow()
        regularizers = MockTensorFlow()
        callbacks = MockTensorFlow()
    
    tf = MockTensorFlow()
    Sequential = MockTensorFlow
    LSTM = MockTensorFlow
    Bidirectional = MockTensorFlow
    Dense = MockTensorFlow
    Dropout = MockTensorFlow
    Conv1D = MockTensorFlow
    MaxPooling1D = MockTensorFlow
    GlobalAveragePooling1D = MockTensorFlow
    Reshape = MockTensorFlow
    Add = MockTensorFlow
    Activation = MockTensorFlow
    BatchNormalization = MockTensorFlow
    regularizers = MockKeras()
    EarlyStopping = MockTensorFlow

def require_tensorflow(func):
    """Decorator to check if TensorFlow is available before running a function"""
    def wrapper(*args, **kwargs):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for this functionality but is not installed. "
                "Install TensorFlow with: pip install tensorflow>=2.13.0"
            )
        return func(*args, **kwargs)
    return wrapper

def is_tensorflow_available():
    """Check if TensorFlow is available"""
    return TENSORFLOW_AVAILABLE
