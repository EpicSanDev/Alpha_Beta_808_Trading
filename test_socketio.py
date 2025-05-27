#!/usr/bin/env python3
"""
Test script to verify flask_socketio installation
"""

def test_imports():
    """Test all critical imports"""
    try:
        print("Testing flask_socketio import...")
        import flask_socketio
        print(f"✓ flask_socketio version: {flask_socketio.__version__}")
        
        print("Testing SocketIO class import...")
        from flask_socketio import SocketIO, emit, join_room, leave_room
        print("✓ SocketIO class imported successfully")
        
        print("Testing eventlet import...")
        import eventlet
        print(f"✓ eventlet version: {eventlet.__version__}")
        
        print("Testing Flask import...")
        from flask import Flask
        print("✓ Flask imported successfully")
        
        print("Creating test Flask app with SocketIO...")
        app = Flask(__name__)
        socketio = SocketIO(app)
        print("✓ Flask app with SocketIO created successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ General error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Flask-SocketIO Installation Test")
    print("=" * 50)
    
    if test_imports():
        print("\n✅ ALL TESTS PASSED - flask_socketio is working correctly!")
        exit(0)
    else:
        print("\n❌ TESTS FAILED - flask_socketio installation issues detected")
        exit(1)
