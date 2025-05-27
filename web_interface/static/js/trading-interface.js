// Enhanced Trading Interface JavaScript Utilities
// Common functions and WebSocket handling for the trading interface

class TradingInterface {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.isConnected = false;
        
        // Trading state
        this.currentMode = 'paper';
        this.currentSymbol = 'BTCUSDT';
        this.currentPrice = 0;
        
        // Initialize on page load
        this.init();
    }
    
    init() {
        this.initializeWebSocket();
        this.setupEventListeners();
        this.loadInitialData();
    }
    
    // WebSocket Management
    initializeWebSocket() {
        try {
            this.socket = io();
            
            this.socket.on('connect', () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.showConnectionStatus('connected');
            });
            
            this.socket.on('disconnect', () => {
                console.log('WebSocket disconnected');
                this.isConnected = false;
                this.showConnectionStatus('disconnected');
                this.attemptReconnect();
            });
            
            // Market data events
            this.socket.on('price_update', (data) => {
                this.handlePriceUpdate(data);
            });
            
            // Trading events
            this.socket.on('order_update', (data) => {
                this.handleOrderUpdate(data);
            });
            
            // Portfolio events
            this.socket.on('portfolio_update', (data) => {
                this.handlePortfolioUpdate(data);
            });
            
            // Alert events
            this.socket.on('alert_triggered', (data) => {
                this.handleAlertTriggered(data);
            });
            
            // Risk events
            this.socket.on('risk_alert', (data) => {
                this.handleRiskAlert(data);
            });
            
        } catch (error) {
            console.error('WebSocket initialization error:', error);
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, this.reconnectDelay * this.reconnectAttempts);
        } else {
            console.log('Max reconnection attempts reached');
            this.showConnectionStatus('failed');
        }
    }
    
    // Event Handlers
    handlePriceUpdate(data) {
        const { symbol, price, change24h } = data;
        
        // Update price displays
        const priceElements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
        priceElements.forEach(element => {
            if (element.classList.contains('price-ticker')) {
                element.textContent = this.formatPrice(price);
                element.classList.remove('price-positive', 'price-negative');
                element.classList.add(change24h >= 0 ? 'price-positive' : 'price-negative');
            }
        });
        
        // Update current price for calculations
        if (symbol === this.currentSymbol) {
            this.currentPrice = price;
            this.updateCalculations();
        }
        
        // Animate price change
        this.animatePriceChange(symbol, change24h >= 0);
    }
    
    handleOrderUpdate(data) {
        const { order, status } = data;
        
        // Show notification
        this.showToast(`Order ${status}: ${order.symbol} ${order.side}`, 
                      status === 'filled' ? 'success' : 'info');
        
        // Update UI elements
        this.refreshOrders();
        this.refreshPortfolio();
    }
    
    handlePortfolioUpdate(data) {
        // Update portfolio displays
        this.updatePortfolioDisplay(data);
    }
    
    handleAlertTriggered(data) {
        const { alert } = data;
        
        // Show prominent notification
        this.showToast(`Alert Triggered: ${alert.symbol} ${alert.condition} ${this.formatPrice(alert.target_price)}`, 
                      'warning', 5000);
        
        // Play sound if enabled
        this.playNotificationSound();
        
        // Update alerts display
        this.refreshAlerts();
    }
    
    handleRiskAlert(data) {
        const { type, message, severity } = data;
        
        this.showToast(`Risk Alert: ${message}`, 
                      severity === 'high' ? 'error' : 'warning', 10000);
        
        // Update risk metrics
        this.refreshRiskMetrics();
    }
    
    // UI Update Methods
    updatePortfolioDisplay(data) {
        // Update total value
        const totalValueElement = document.getElementById('portfolioValue');
        if (totalValueElement && data.total_value !== undefined) {
            totalValueElement.textContent = this.formatCurrency(data.total_value);
            this.animateValueChange(totalValueElement);
        }
        
        // Update cash balance
        const cashElement = document.getElementById('portfolioCash');
        if (cashElement && data.cash !== undefined) {
            cashElement.textContent = this.formatCurrency(data.cash);
        }
        
        // Update positions count
        const positionsElement = document.getElementById('activePositionsCount');
        if (positionsElement && data.positions_count !== undefined) {
            positionsElement.textContent = data.positions_count;
        }
    }
    
    animatePriceChange(symbol, isPositive) {
        const priceElements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
        priceElements.forEach(element => {
            element.style.transition = 'background-color 0.3s ease';
            element.style.backgroundColor = isPositive ? 
                'rgba(5, 150, 105, 0.2)' : 'rgba(220, 38, 38, 0.2)';
            
            setTimeout(() => {
                element.style.backgroundColor = '';
            }, 300);
        });
    }
    
    animateValueChange(element) {
        element.classList.add('slide-in');
        setTimeout(() => {
            element.classList.remove('slide-in');
        }, 300);
    }
    
    // Trading Functions
    async executeTrade(orderData) {
        try {
            this.showLoadingState(true);
            
            const response = await this.apiCall('/api/trading/execute', 'POST', orderData);
            
            if (response.success) {
                this.showToast(`Order placed successfully: ${orderData.symbol} ${orderData.side}`, 'success');
                this.resetOrderForm();
                this.refreshOrders();
            } else {
                this.showToast(`Order failed: ${response.error}`, 'error');
            }
        } catch (error) {
            console.error('Trade execution error:', error);
            this.showToast('Trade execution failed', 'error');
        } finally {
            this.showLoadingState(false);
        }
    }
    
    async switchTradingMode(mode) {
        try {
            const response = await this.apiCall('/api/trading/mode', 'POST', { mode });
            
            if (response.success) {
                this.currentMode = mode;
                this.updateTradingModeUI(mode);
                this.showToast(`Switched to ${mode} trading`, 'success');
            }
        } catch (error) {
            console.error('Mode switch error:', error);
            this.showToast('Failed to switch mode', 'error');
        }
    }
    
    updateTradingModeUI(mode) {
        // Update mode display
        const modeElements = document.querySelectorAll('.trading-mode-display');
        modeElements.forEach(element => {
            element.textContent = mode === 'live' ? 'Live Trading' : 'Paper Trading';
            element.className = `badge ${mode === 'live' ? 'bg-danger' : 'bg-secondary'}`;
        });
        
        // Update toggle buttons
        document.querySelectorAll('.trading-mode-toggle button').forEach(btn => {
            btn.classList.remove('active');
        });
        
        const activeButton = document.querySelector(`[data-mode="${mode}"]`);
        if (activeButton) {
            activeButton.classList.add('active');
        }
    }
    
    // Position Size Calculator
    calculatePositionSize(entryPrice, stopLoss, riskPercent, capital) {
        if (!entryPrice || !stopLoss || !riskPercent || !capital) {
            return null;
        }
        
        const riskAmount = capital * (riskPercent / 100);
        const stopLossDistance = Math.abs(entryPrice - stopLoss);
        const positionSize = riskAmount / stopLossDistance;
        const positionValue = positionSize * entryPrice;
        
        return {
            size: positionSize,
            value: positionValue,
            risk: riskAmount,
            stopLossDistance: stopLossDistance
        };
    }
    
    updateCalculations() {
        const entryPrice = parseFloat(document.getElementById('entryPrice')?.value || 0);
        const stopLoss = parseFloat(document.getElementById('stopLoss')?.value || 0);
        const riskPercent = parseFloat(document.getElementById('riskPercent')?.value || 2);
        const capital = parseFloat(document.getElementById('availableCapital')?.value || 0);
        
        const calculation = this.calculatePositionSize(entryPrice, stopLoss, riskPercent, capital);
        
        if (calculation) {
            this.updateCalculationDisplay(calculation);
        }
    }
    
    updateCalculationDisplay(calc) {
        const elements = {
            positionSize: document.getElementById('calculatedPositionSize'),
            positionValue: document.getElementById('calculatedPositionValue'),
            riskAmount: document.getElementById('calculatedRiskAmount')
        };
        
        if (elements.positionSize) {
            elements.positionSize.textContent = calc.size.toFixed(8);
        }
        if (elements.positionValue) {
            elements.positionValue.textContent = this.formatCurrency(calc.value);
        }
        if (elements.riskAmount) {
            elements.riskAmount.textContent = this.formatCurrency(calc.risk);
        }
    }
    
    // Utility Functions
    async apiCall(endpoint, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(endpoint, options);
        return await response.json();
    }
    
    formatPrice(price) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 8
        }).format(price);
    }
    
    formatCurrency(amount) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(amount);
    }
    
    formatPercentage(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 1,
            maximumFractionDigits: 2
        }).format(value / 100);
    }
    
    showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type} slide-in`;
        toast.innerHTML = `
            <div class="toast-content">
                <span>${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        // Create toast container if it doesn't exist
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        
        container.appendChild(toast);
        
        // Auto remove after duration
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, duration);
    }
    
    showConnectionStatus(status) {
        const statusIndicator = document.getElementById('connectionStatus');
        if (statusIndicator) {
            statusIndicator.className = `connection-status ${status}`;
            statusIndicator.textContent = status === 'connected' ? 'Connected' : 
                                        status === 'disconnected' ? 'Reconnecting...' : 'Connection Failed';
        }
    }
    
    showLoadingState(isLoading) {
        const loadingElements = document.querySelectorAll('.loading-state');
        loadingElements.forEach(element => {
            if (isLoading) {
                element.classList.add('loading-pulse');
                element.disabled = true;
            } else {
                element.classList.remove('loading-pulse');
                element.disabled = false;
            }
        });
    }
    
    playNotificationSound() {
        // Simple notification sound
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    }
    
    // Data Refresh Methods
    async refreshOrders() {
        // Implementation depends on your specific needs
        // This would typically reload the orders table/list
    }
    
    async refreshPortfolio() {
        // Refresh portfolio data
    }
    
    async refreshAlerts() {
        // Refresh alerts display
    }
    
    async refreshRiskMetrics() {
        // Refresh risk metrics display
    }
    
    resetOrderForm() {
        const forms = document.querySelectorAll('.order-form form');
        forms.forEach(form => form.reset());
    }
    
    setupEventListeners() {
        // Symbol change listeners
        document.addEventListener('change', (e) => {
            if (e.target.matches('.symbol-selector')) {
                this.currentSymbol = e.target.value;
                this.loadSymbolData(this.currentSymbol);
            }
        });
        
        // Calculation input listeners
        document.addEventListener('input', (e) => {
            if (e.target.matches('.calculation-input')) {
                this.updateCalculations();
            }
        });
        
        // Trading mode toggle listeners
        document.addEventListener('click', (e) => {
            if (e.target.matches('.trading-mode-btn')) {
                const mode = e.target.dataset.mode;
                this.switchTradingMode(mode);
            }
        });
    }
    
    async loadInitialData() {
        try {
            // Load trading mode
            const modeResponse = await this.apiCall('/api/trading/mode');
            if (modeResponse.success) {
                this.currentMode = modeResponse.data.mode;
                this.updateTradingModeUI(this.currentMode);
            }
            
            // Load initial portfolio data
            this.refreshPortfolio();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }
    
    async loadSymbolData(symbol) {
        try {
            const response = await this.apiCall(`/api/market/price/${symbol}`);
            if (response.success) {
                this.currentPrice = response.data.price;
                this.updatePriceDisplays(symbol, response.data);
            }
        } catch (error) {
            console.error('Failed to load symbol data:', error);
        }
    }
    
    updatePriceDisplays(symbol, data) {
        const priceElements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
        priceElements.forEach(element => {
            if (element.classList.contains('current-price')) {
                element.textContent = this.formatPrice(data.price);
            }
        });
    }
}

// Initialize the trading interface when the page loads
let tradingInterface;

document.addEventListener('DOMContentLoaded', function() {
    tradingInterface = new TradingInterface();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TradingInterface;
}
