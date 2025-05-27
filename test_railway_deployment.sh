#!/bin/bash

# Railway Deployment Test Script
# Tests the Railway configuration before actual deployment

set -e

echo "🚂 Railway Deployment Configuration Test"
echo "========================================="

PROJECT_DIR="/Users/bastienjavaux/Desktop/AlphaBeta808Trading"
cd "$PROJECT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test 1: Check required files exist
print_status "Checking required Railway files..."

required_files=(
    "Dockerfile.railway"
    "Procfile"
    ".railwayignore"
    "railway_startup.py"
    "requirements-railway.txt"
    ".env.railway"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_success "✓ $file exists"
    else
        print_error "✗ $file missing"
        exit 1
    fi
done

# Test 2: Validate Dockerfile syntax
print_status "Validating Dockerfile.railway syntax..."
if docker run --rm -i hadolint/hadolint < Dockerfile.railway > /dev/null 2>&1; then
    print_success "✓ Dockerfile.railway syntax is valid"
else
    print_warning "! Dockerfile.railway has warnings (continuing anyway)"
fi

# Test 3: Check Python dependencies
print_status "Testing Python dependencies..."
if python3 -c "import flask, flask_socketio, flask_cors; print('Core dependencies available')" 2>/dev/null; then
    print_success "✓ Core Python dependencies available"
else
    print_warning "! Some Python dependencies missing locally"
fi

# Test 4: Test startup script
print_status "Testing railway_startup.py..."
if python3 -c "
import sys
sys.path.insert(0, '.')
from railway_startup import check_and_install_dependencies
print('Startup script syntax OK')
" 2>/dev/null; then
    print_success "✓ railway_startup.py syntax is valid"
else
    print_error "✗ railway_startup.py has syntax errors"
    exit 1
fi

# Test 5: Check environment variables template
print_status "Checking environment variables..."
if [[ -f ".env.railway" ]]; then
    required_vars=("FLASK_ENV" "SECRET_KEY" "DATABASE_URL")
    for var in "${required_vars[@]}"; do
        if grep -q "^$var=" .env.railway; then
            print_success "✓ $var defined in .env.railway"
        else
            print_warning "! $var not found in .env.railway"
        fi
    done
fi

# Test 6: Build Docker image locally (optional)
print_status "Testing Docker build (this may take a few minutes)..."
if command -v docker &> /dev/null; then
    echo "Building Railway Docker image..."
    if docker build -f Dockerfile.railway -t alphabeta808-railway-test . > build.log 2>&1; then
        print_success "✓ Docker image builds successfully"
        
        # Test container startup
        print_status "Testing container startup..."
        export PORT=5000
        if timeout 30s docker run --rm -p 5000:5000 -e PORT=5000 alphabeta808-railway-test > container.log 2>&1 &
        then
            sleep 5
            if curl -s http://localhost:5000/health > /dev/null 2>&1; then
                print_success "✓ Container starts and responds to health check"
            else
                print_warning "! Container started but health check failed"
            fi
            docker stop $(docker ps -q --filter ancestor=alphabeta808-railway-test) 2>/dev/null || true
        else
            print_warning "! Container startup test failed or timed out"
        fi
    else
        print_error "✗ Docker build failed. Check build.log for details"
        print_error "Build log:"
        tail -20 build.log
        exit 1
    fi
else
    print_warning "! Docker not available, skipping build test"
fi

# Test 7: Check Railway CLI (if available)
print_status "Checking Railway CLI..."
if command -v railway &> /dev/null; then
    print_success "✓ Railway CLI is available"
    
    # Check if logged in
    if railway whoami > /dev/null 2>&1; then
        print_success "✓ Logged into Railway"
    else
        print_warning "! Not logged into Railway (run: railway login)"
    fi
else
    print_warning "! Railway CLI not installed"
    echo "  Install with: npm install -g @railway/cli"
fi

# Test 8: Validate critical app files
print_status "Checking application files..."
critical_files=(
    "web_interface/app_enhanced.py"
    "requirements.txt"
    "README.md"
)

for file in "${critical_files[@]}"; do
    if [[ -f "$file" ]]; then
        print_success "✓ $file exists"
    else
        print_error "✗ Critical file $file missing"
        exit 1
    fi
done

# Summary and next steps
echo ""
echo "🎉 Railway Configuration Test Complete!"
echo "======================================"

print_success "Configuration looks good for Railway deployment!"
echo ""
echo "Next steps:"
echo "1. Make sure you have Railway CLI installed: npm install -g @railway/cli"
echo "2. Login to Railway: railway login"
echo "3. Create a new project: railway new"
echo "4. Deploy: railway up"
echo ""
echo "Environment variables to set in Railway dashboard:"
echo "- SECRET_KEY (generate a secure random key)"
echo "- BINANCE_API_KEY (your Binance API key)"
echo "- BINANCE_SECRET_KEY (your Binance secret)"
echo "- FLASK_ENV=production"
echo ""
echo "For detailed deployment instructions, see RAILWAY_DEPLOYMENT.md"

# Cleanup
rm -f build.log container.log 2>/dev/null || true
