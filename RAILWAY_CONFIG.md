# AlphaBeta808 Trading Bot - Railway Configuration

# Railway will automatically detect the Dockerfile.railway and use it for building
# This project uses Dockerfile.railway for optimized Railway deployment

# Health check endpoint: /health
# Start command: python web_interface/app_enhanced.py

# Environment variables to set in Railway dashboard:
# - SECRET_KEY (required)
# - WEB_ADMIN_USER (required) 
# - WEB_ADMIN_PASSWORD (required)
# - BINANCE_API_KEY (required)
# - BINANCE_API_SECRET (required)
# - WEBHOOK_SECRET (required)
# - FLASK_ENV=production
# - LOG_LEVEL=INFO
