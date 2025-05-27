# AlphaBeta808 Trading Bot - Dockerfile
FROM python:3.11-slim

LABEL maintainer="AlphaBeta808Trading"
LABEL description="Automated cryptocurrency trading bot with ML integration"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY web_interface/requirements.txt ./web_interface/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r web_interface/requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs models_store backtest_results reports

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading trading
RUN chown -R trading:trading /app
USER trading

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:5000/api/status')" || exit 1

# Expose ports
EXPOSE 5000

# Default command - can be overridden in deployment
CMD ["python", "web_interface/app_enhanced.py"]
