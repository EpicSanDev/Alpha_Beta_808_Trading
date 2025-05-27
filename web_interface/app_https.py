#!/usr/bin/env python3
"""
HTTPS-enabled Web Interface for AlphaBeta808 Trading Bot
Production version with SSL/TLS support
"""

import os
import ssl
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main Flask app
from web_interface.app_enhanced import create_app, socketio

def main():
    """Main function to run the HTTPS-enabled web interface"""
    
    # Create Flask app
    app = create_app()
    
    # SSL configuration
    ssl_dir = Path(__file__).parent.parent / 'ssl'
    cert_file = ssl_dir / 'server.crt'
    key_file = ssl_dir / 'server.key'
    
    # Check if SSL certificates exist
    if not cert_file.exists() or not key_file.exists():
        print("❌ SSL certificates not found!")
        print(f"📁 Expected files:")
        print(f"   - {cert_file}")
        print(f"   - {key_file}")
        print("")
        print("🔧 Generate certificates by running:")
        print("   ./ssl/generate_ssl.sh")
        print("")
        print("🌐 Starting HTTP server instead on port 5000...")
        
        # Start HTTP server as fallback
        socketio.run(
            app,
            host='0.0.0.0',
            port=5000,
            debug=False,
            allow_unsafe_werkzeug=True
        )
        return
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
    context.load_cert_chain(str(cert_file), str(key_file))
    
    print("🔐 Starting HTTPS server with SSL/TLS encryption...")
    print("🌐 Access the interface at: https://localhost:5443")
    print("⚠️  You may see a security warning for self-signed certificates")
    print("")
    
    # Start HTTPS server
    socketio.run(
        app,
        host='0.0.0.0',
        port=5443,
        debug=False,
        ssl_context=context,
        allow_unsafe_werkzeug=True
    )

if __name__ == '__main__':
    main()
