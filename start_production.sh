#!/bin/bash
# AlphaBeta808 Trading Bot - Production Startup Script
# Complete production environment initialization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_DIR="/Users/bastienjavaux/Desktop/AlphaBeta808Trading"
VENV_NAME="trading_env"

echo -e "${BLUE}🚀 AlphaBeta808 Trading Bot - Production Startup${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "system_verification.py" ]; then
    print_error "Please run this script from the AlphaBeta808Trading directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    print_error "Python 3.11+ required. Current version: $python_version"
    exit 1
fi
print_status "Python version check passed ($python_version)"

# Activate virtual environment
if [ ! -d "$VENV_NAME" ]; then
    print_warning "Virtual environment not found. Creating..."
    python3 -m venv $VENV_NAME
fi

source $VENV_NAME/bin/activate
print_status "Virtual environment activated"

# Install/update dependencies
print_status "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Check environment file
if [ ! -f ".env" ]; then
    if [ -f ".env.production" ]; then
        print_warning "No .env file found. Using .env.production template..."
        cp .env.production .env
        print_warning "Please edit .env file with your actual API keys!"
        echo ""
        echo "Required changes:"
        echo "  - BINANCE_API_KEY=your_actual_api_key"
        echo "  - BINANCE_API_SECRET=your_actual_api_secret"
        echo "  - WEB_ADMIN_PASSWORD=your_secure_password"
        echo ""
        echo "Press Enter to continue after editing .env file..."
        read -r
    else
        print_error ".env file not found and no template available"
        exit 1
    fi
fi
print_status "Environment configuration loaded"

# Generate SSL certificates if needed
if [ ! -f "ssl/server.crt" ] || [ ! -f "ssl/server.key" ]; then
    print_warning "SSL certificates not found. Generating self-signed certificates..."
    mkdir -p ssl
    cd ssl
    
    # Generate certificates
    openssl genrsa -out server.key 2048 2>/dev/null
    openssl req -new -key server.key -out server.csr -subj "/C=US/ST=State/L=City/O=AlphaBeta808/OU=Trading/CN=localhost" 2>/dev/null
    openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt 2>/dev/null
    
    chmod 600 server.key
    chmod 644 server.crt
    cd ..
    
    print_status "SSL certificates generated"
else
    print_status "SSL certificates found"
fi

# Create necessary directories
mkdir -p logs
mkdir -p backups
mkdir -p models_store
print_status "Directory structure verified"

# Run system verification
echo ""
echo -e "${BLUE}🔍 Running system verification...${NC}"
python system_verification.py

if [ $? -eq 0 ]; then
    print_status "System verification passed"
else
    print_error "System verification failed"
    exit 1
fi

# Setup monitoring
if [ -f "monitoring_config.json" ]; then
    print_status "Monitoring configuration found"
fi

# Setup log rotation (macOS specific)
if command -v newsyslog >/dev/null 2>&1; then
    print_status "Log rotation available (newsyslog)"
fi

echo ""
echo -e "${GREEN}🎉 Production environment setup complete!${NC}"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo "1. Review your .env file configuration"
echo "2. Test with paper trading first"
echo "3. Monitor logs for any issues"
echo ""
echo "🚀 Startup options:"
echo "   Standard HTTP:  python web_interface/app_enhanced.py"
echo "   Secure HTTPS:   python web_interface/app_https.py"
echo "   Trading Bot:    python live_trading_bot.py"
echo "   Monitoring:     python src/monitoring/production_monitor.py"
echo ""
echo "🌐 Web Interface URLs:"
echo "   HTTP:  http://localhost:5000"
echo "   HTTPS: https://localhost:5443"
echo ""
echo "🔐 Default credentials: admin / admin123 (change in .env)"
echo ""
echo -e "${YELLOW}⚠️  IMPORTANT: Start with paper trading and verify everything works!${NC}"
