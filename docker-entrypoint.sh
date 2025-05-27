#!/bin/bash

# AlphaBeta808 Trading Bot - Docker Entrypoint Script
# This script handles the startup logic for different container modes

set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to wait for dependencies
wait_for_dependencies() {
    # Wait for any external dependencies if needed
    log_info "Checking dependencies..."
    
    # Example: wait for database or redis if configured
    # if [ "$DATABASE_URL" ]; then
    #     log_info "Waiting for database..."
    # fi
    
    log_success "Dependencies check complete"
}

# Function to initialize the application
initialize_app() {
    log_info "Initializing AlphaBeta808 Trading Bot..."
    
    # Create necessary directories if they don't exist
    for dir in logs models_store backtest_results reports results optimized_models; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir" && log_info "Created directory: $dir"
        fi
    done
    
    # Set proper permissions only if we can write to the directories
    for dir in logs models_store backtest_results reports results optimized_models; do
        if [ -w "$dir" ]; then
            chmod 755 "$dir" 2>/dev/null && log_info "Set permissions for: $dir" || log_warning "Cannot change permissions for $dir (read-only mount?)"
        else
            log_info "Directory $dir is read-only, skipping permission change"
        fi
    done
    
    # Check if configuration exists
    if [ ! -f "trader_config.json" ]; then
        log_warning "trader_config.json not found, using default configuration"
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_warning ".env file not found, using environment variables"
    fi
    
    log_success "Application initialized"
}

# Function to run system verification
run_system_check() {
    log_info "Running system verification..."
    
    if [ -f "system_verification.py" ]; then
        python system_verification.py || {
            log_error "System verification failed"
            exit 1
        }
        log_success "System verification passed"
    else
        log_warning "system_verification.py not found, skipping verification"
    fi
}

# Main startup logic
main() {
    log_info "Starting AlphaBeta808 Trading Bot Docker Container"
    log_info "Command: $1"
    
    # Initialize the application
    initialize_app
    
    # Wait for dependencies
    wait_for_dependencies
    
    # Run system check for production modes
    if [ "$1" = "web" ] || [ "$1" = "bot" ] || [ "$1" = "production" ]; then
        run_system_check
    fi
    
    # Execute the appropriate command based on the argument
    case "$1" in
        "web")
            log_info "Starting web interface..."
            if python -c "import flask_socketio" 2>/dev/null; then
                cd web_interface
                exec python app_enhanced.py
            else
                log_error "Flask dependencies not installed. Installing now..."
                pip install flask flask-socketio flask-cors werkzeug eventlet
                cd web_interface
                exec python app_enhanced.py
            fi
            ;;
        "bot")
            log_info "Starting trading bot..."
            exec python live_trading_bot.py
            ;;
        "production")
            log_info "Starting production mode (bot + web)..."
            # Start both bot and web interface
            python continuous_trader.py &
            if python -c "import flask_socketio" 2>/dev/null; then
                cd web_interface
                exec python app_enhanced.py
            else
                log_warning "Web interface dependencies missing, running bot only..."
                wait
            fi
            ;;
        "backtest")
            log_info "Running backtest..."
            exec python main.py
            ;;
        "comprehensive-backtest")
            log_info "Running comprehensive backtest..."
            exec python test_comprehensive_backtest.py
            ;;
        "optimize")
            log_info "Running model optimization..."
            exec python optimize_models_for_profitability.py
            ;;
        "test")
            log_info "Running tests..."
            exec python -m pytest tests/ -v
            ;;
        "bash")
            log_info "Starting interactive bash session..."
            exec /bin/bash
            ;;
        "help")
            echo "AlphaBeta808 Trading Bot - Available commands:"
            echo "  web                   - Start web interface only"
            echo "  bot                   - Start trading bot only"
            echo "  production            - Start both bot and web interface"
            echo "  backtest              - Run simple backtest (main.py)"
            echo "  comprehensive-backtest - Run comprehensive backtest"
            echo "  optimize              - Run model optimization"
            echo "  test                  - Run test suite"
            echo "  bash                  - Interactive bash session"
            echo "  help                  - Show this help"
            ;;
        *)
            if [ -n "$1" ]; then
                log_info "Executing custom command: $@"
                exec "$@"
            else
                log_info "No command specified, running default backtest..."
                exec python main.py
            fi
            ;;
    esac
}

# Handle signals for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
    log_success "Cleanup complete"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Run main function with all arguments
main "$@"
