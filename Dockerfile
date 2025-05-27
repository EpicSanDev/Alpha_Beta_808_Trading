# AlphaBeta808 Trading Bot - Production Dockerfile
FROM python:3.11-slim

LABEL maintainer="AlphaBeta808Trading"
LABEL description="Automated cryptocurrency trading bot with ML integration"
LABEL version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    FLASK_ENV=production \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    cmake \
    make \
    build-essential \
    libpq-dev \
    curl \
    wget \
    ca-certificates \
    pkg-config \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading -d /app -s /bin/bash trading

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with timeout and retry settings
# Split installation into batches to avoid network timeouts
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core scientific computing packages first
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    "pandas>=1.5.0" \
    "numpy>=1.22.4,<1.27.0" \
    "scipy>=1.7.0"

# Install ML and visualization packages
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    "scikit-learn>=1.0.0" \
    "joblib>=1.1.0" \
    "matplotlib>=3.5.0" \
    "seaborn>=0.11.0" \
    "xgboost>=1.6.0"

# Install optimization and statistical packages
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    "cvxpy>=1.1.0" \
    "optuna>=2.10.0" \
    "pymc>=4.0.0"

# Install trading and data packages
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    "ta>=0.10.0" \
    "yfinance>=0.2.0" \
    "python-binance>=1.0.0"

# Install web and visualization packages
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    "plotly>=5.0.0" \
    "dash>=2.10.0"

# Install development and utility packages
RUN pip install --no-cache-dir --timeout=300 --retries=3 \
    "pytest>=7.0.0" \
    "jupyter>=1.0.0" \
    "python-dotenv>=0.20.0" \
    "flake8>=6.0.0"

# Copy application code
COPY --chown=trading:trading . .

# Create necessary directories with proper permissions
RUN mkdir -p logs models_store backtest_results reports results optimized_models && \
    chown -R trading:trading /app

# Switch to non-root user
USER trading

# Health check for the web interface
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/api/system/status || exit 1

# Expose ports
EXPOSE 5000

# Add startup script
COPY --chown=trading:trading docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Default command
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["web"]
