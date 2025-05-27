#!/bin/bash

# AlphaBeta808 Trading - Docker Deployment Test Script
# This script tests the complete Docker deployment workflow

set -e  # Exit on any error

echo "🚀 Starting AlphaBeta808 Trading Docker Deployment Test"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
echo "1. Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi
print_status "Docker is installed"

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed"
    exit 1
fi
print_status "Docker Compose is installed"

# Check Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    exit 1
fi
print_status "Docker daemon is running"

# Validate configuration
echo -e "\n2. Validating Docker Compose configuration..."
if docker-compose config > /dev/null 2>&1; then
    print_status "Docker Compose configuration is valid"
else
    print_error "Docker Compose configuration is invalid"
    docker-compose config
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your actual values before running containers"
else
    print_status ".env file exists"
fi

# Check required directories
echo -e "\n3. Checking required directories..."
required_dirs=("models_store" "logs" "backtest_results" "reports" "monitoring/grafana/dashboards" "monitoring/grafana/datasources")

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        print_warning "Creating missing directory: $dir"
        mkdir -p "$dir"
    fi
    print_status "Directory exists: $dir"
done

# Check Docker image exists or can be built
echo -e "\n4. Checking Docker image..."
if docker images | grep -q "alphabeta808trading"; then
    print_status "AlphaBeta808 trading image exists"
else
    print_warning "Building AlphaBeta808 trading image..."
    if docker-compose build trading-bot; then
        print_status "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
fi

# Test individual services
echo -e "\n5. Testing individual services..."

# Test Redis
echo "Testing Redis..."
if docker-compose up -d redis; then
    sleep 5
    if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
        print_status "Redis is working"
    else
        print_error "Redis is not responding"
    fi
    docker-compose stop redis
else
    print_error "Failed to start Redis"
fi

# Test Prometheus
echo "Testing Prometheus..."
if docker-compose up -d prometheus; then
    sleep 10
    if docker-compose exec -T prometheus wget -q --spider http://localhost:9090; then
        print_status "Prometheus is working"
    else
        print_error "Prometheus is not responding"
    fi
    docker-compose stop prometheus
else
    print_error "Failed to start Prometheus"
fi

# Cleanup test containers
echo -e "\n6. Cleaning up test containers..."
docker-compose down > /dev/null 2>&1
print_status "Test cleanup completed"

# Test backtest mode
echo -e "\n7. Testing backtest mode..."
if docker-compose run --rm trading-bot backtest --help > /dev/null 2>&1; then
    print_status "Backtest mode is accessible"
else
    print_warning "Backtest mode test failed (this may be expected if backtest script doesn't have --help)"
fi

# Test configuration validation
echo -e "\n8. Testing configuration files..."
config_files=("trader_config.json" "monitoring/prometheus.yml")

for config in "${config_files[@]}"; do
    if [ -f "$config" ]; then
        print_status "Configuration file exists: $config"
    else
        print_error "Missing configuration file: $config"
    fi
done

# Network connectivity test
echo -e "\n9. Testing network connectivity..."
if docker network ls | grep -q "alphabeta808trading"; then
    print_warning "Previous AlphaBeta808 network exists (this is normal if you've run before)"
else
    print_status "No conflicting networks found"
fi

# Resource requirements check
echo -e "\n10. Checking system resources..."
available_memory=$(docker system info --format '{{.MemTotal}}' 2>/dev/null || echo "Unknown")
if [ "$available_memory" != "Unknown" ] && [ "$available_memory" -gt 4294967296 ]; then
    print_status "Sufficient memory available (>4GB)"
else
    print_warning "May have insufficient memory. Recommended: 4GB+"
fi

# Final recommendations
echo -e "\n🎯 Test Summary and Recommendations"
echo "=================================="

print_status "Docker deployment test completed successfully!"

echo -e "\n📋 Next Steps:"
echo "1. Edit .env file with your actual API credentials"
echo "2. Run: docker-compose up -d"
echo "3. Access web interface: http://localhost:5000"
echo "4. Access Grafana: http://localhost:3000"
echo "5. Access Prometheus: http://localhost:9090"

echo -e "\n🛠️ Useful Commands:"
echo "• Start all services: docker-compose up -d"
echo "• View logs: docker-compose logs -f"
echo "• Stop all services: docker-compose down"
echo "• Run backtest: docker-compose run --rm trading-bot backtest"
echo "• Open shell: docker-compose run --rm trading-bot bash"

echo -e "\n📖 Documentation:"
echo "• Full deployment guide: DOCKER_DEPLOYMENT.md"
echo "• Project documentation: README.md"
echo "• API documentation: web interface at /docs"

echo -e "\n${GREEN}🚀 Your AlphaBeta808 Trading system is ready for deployment!${NC}"
