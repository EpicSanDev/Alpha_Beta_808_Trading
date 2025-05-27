#!/usr/bin/env python3
"""
Railway startup script for AlphaBeta808 Trading Bot
Handles missing dependencies gracefully and provides fallback options
"""

import os
import sys
import subprocess
import importlib
from datetime import datetime

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install critical dependencies"""
    critical_packages = [
        'flask',
        'flask-socketio',
        'flask-cors',
        'eventlet',
        'python-dotenv',
        'requests'
    ]
    
    missing_packages = []
    
    for package in critical_packages:
        try:
            # Try to import the package (handle package name differences)
            if package == 'flask-socketio':
                importlib.import_module('flask_socketio')
            elif package == 'flask-cors':
                importlib.import_module('flask_cors')
            elif package == 'python-dotenv':
                importlib.import_module('dotenv')
            else:
                importlib.import_module(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    # Install missing packages
    for package in missing_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")
    
    return len(missing_packages) == 0

def start_application():
    """Start the Flask application"""
    try:
        # Set environment variables
        os.environ['FLASK_ENV'] = 'production'
        os.environ['PYTHONPATH'] = '/app/src:/app'
        
        # Import and run the main application
        sys.path.insert(0, '/app')
        sys.path.insert(0, '/app/web_interface')
        
        from web_interface.app_enhanced import app, socketio
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get('PORT', 5000))
        
        print(f"Starting AlphaBeta808 Trading Bot on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Trying to start with minimal Flask app...")
        start_minimal_app()
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

def start_minimal_app():
    """Start a minimal Flask app if main app fails"""
    try:
        from flask import Flask, jsonify
        
        app = Flask(__name__)
        
        @app.route('/')
        def home():
            return jsonify({
                'status': 'running',
                'message': 'AlphaBeta808 Trading Bot - Minimal Mode',
                'timestamp': str(datetime.now())
            })
        
        @app.route('/health')
        def health():
            return jsonify({'status': 'healthy'})
        
        port = int(os.environ.get('PORT', 5000))
        print(f"Starting minimal app on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except Exception as e:
        print(f"Minimal app failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    print("AlphaBeta808 Trading Bot - Railway Startup")
    print("=" * 50)
    
    # Check dependencies first
    if check_and_install_dependencies():
        print("All dependencies available, starting main application...")
        start_application()
    else:
        print("Some dependencies missing, starting with available components...")
        start_application()
