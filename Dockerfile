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
    libpq-dev \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r trading && useradd -r -g trading -d /app -s /bin/bash trading

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

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
