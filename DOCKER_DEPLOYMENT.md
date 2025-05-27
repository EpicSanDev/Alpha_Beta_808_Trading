# AlphaBeta808 Trading - Docker Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AlphaBeta808 Trading system using Docker and Docker Compose. The containerized deployment includes the trading bot, web interface, monitoring stack (Prometheus + Grafana), and Redis for caching.

## 🚀 Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- Valid Binance API credentials

### 1. Environment Setup

```bash
# Clone and enter the project directory
cd /path/to/AlphaBeta808Trading

# Copy environment template
cp .env.example .env

# Edit environment variables (REQUIRED!)
nano .env
```

### 2. Configure Environment Variables

Edit `.env` file with your actual values:

```bash
# REQUIRED: Binance API credentials
BINANCE_API_KEY=your_actual_api_key
BINANCE_API_SECRET=your_actual_api_secret

# REQUIRED: Web interface security
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=YourStrongPassword123!
SECRET_KEY=$(openssl rand -base64 32)
WEBHOOK_SECRET=$(openssl rand -hex 32)

# Optional: Monitoring credentials
GRAFANA_USER=admin
GRAFANA_PASSWORD=SecureGrafanaPassword123!
```

### 3. Deploy the Stack

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

## 📋 Service Architecture

| Service | Port | Description | Health Check |
|---------|------|-------------|--------------|
| **trading-bot** | - | Core trading engine | Internal API |
| **web-interface** | 5000 | Web dashboard & API | HTTP endpoint |
| **prometheus** | 9090 | Metrics collection | Web UI |
| **grafana** | 3000 | Monitoring dashboards | Web UI |
| **redis** | 6379 | Caching & session storage | Redis ping |

## 🔧 Execution Modes

The trading bot supports multiple execution modes via the entrypoint script:

### Live Trading Mode (Default)
```bash
docker-compose up -d trading-bot
```

### Backtest Mode
```bash
# Run historical backtest
docker-compose run --rm trading-bot backtest

# With custom config
docker-compose run --rm trading-bot backtest --config /app/config/backtest_config.json
```

### Model Training Mode
```bash
# Train new models
docker-compose run --rm trading-bot train

# Train specific model
docker-compose run --rm trading-bot train --model ensemble
```

### Web Interface Only
```bash
# Start just the web interface
docker-compose up -d web-interface redis
```

### Custom Commands
```bash
# Run any Python script
docker-compose run --rm trading-bot python your_script.py

# Interactive shell
docker-compose run --rm trading-bot bash
```

## 📊 Accessing Services

### Web Interface
- **URL**: http://localhost:5000
- **Login**: admin / YourStrongPassword123!
- **Features**: Trading dashboard, performance metrics, configuration

### Monitoring Stack
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
  - Login: admin / SecureGrafanaPassword123!
  - Pre-configured dashboards for trading metrics

### API Endpoints
```bash
# System status
curl http://localhost:5000/api/system/status

# Trading metrics
curl http://localhost:5000/api/metrics

# Bot status
curl http://localhost:5000/api/bot/status
```

## 🗂️ Data Persistence

The following directories are mounted for data persistence:

```
./trader_config.json → /app/trader_config.json (read-only)
./models_store → /app/models_store (trading models)
./logs → /app/logs (application logs)
./backtest_results → /app/backtest_results (backtest outputs)
./reports → /app/reports (trading reports)
```

### Backup Important Data
```bash
# Create backup
tar -czf trading_backup_$(date +%Y%m%d).tar.gz \
  models_store logs backtest_results reports trader_config.json

# Restore from backup
tar -xzf trading_backup_YYYYMMDD.tar.gz
```

## 🔍 Monitoring & Logging

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f trading-bot
docker-compose logs -f web-interface

# Last 100 lines
docker-compose logs --tail=100 trading-bot
```

### Resource Monitoring
```bash
# Container resource usage
docker stats

# Service health status
docker-compose ps
```

### Built-in Health Checks
All services include health checks:
- **Trading Bot**: Internal API status
- **Web Interface**: HTTP endpoint response
- **Redis**: Connection test
- **Prometheus/Grafana**: Web UI availability

## ⚡ Performance Optimization

### Resource Limits
Add to docker-compose.yml for production:

```yaml
services:
  trading-bot:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Scaling
```bash
# Scale web interface (if needed)
docker-compose up -d --scale web-interface=2

# Use nginx for load balancing
docker-compose -f docker-compose.yml -f docker-compose.nginx.yml up -d
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs for errors
docker-compose logs trading-bot

# Verify environment variables
docker-compose config

# Rebuild image
docker-compose build --no-cache trading-bot
```

#### 2. API Connection Issues
```bash
# Test Binance API connectivity
docker-compose run --rm trading-bot python -c "
from src.binance_client import BinanceClient
client = BinanceClient()
print(client.test_connectivity())
"
```

#### 3. Permission Issues
```bash
# Fix volume permissions
sudo chown -R 1000:1000 models_store logs backtest_results reports
sudo chmod -R 755 models_store logs backtest_results reports
```

#### 4. Memory Issues
```bash
# Check memory usage
docker stats --no-stream

# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 6GB+
```

### Debug Mode
```bash
# Run in debug mode
docker-compose run --rm -e LOG_LEVEL=DEBUG trading-bot

# Interactive debugging
docker-compose run --rm trading-bot bash
```

## 🔒 Security Considerations

### Environment Variables
- Never commit `.env` files to version control
- Use strong passwords for all admin interfaces
- Rotate API keys regularly
- Use testnet for development

### Network Security
```bash
# Restrict external access (production)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### SSL/TLS (Production)
Add nginx proxy with SSL certificates:
```bash
# Generate SSL certificates
./scripts/generate_ssl.sh

# Deploy with SSL
docker-compose -f docker-compose.yml -f docker-compose.ssl.yml up -d
```

## 🚀 Production Deployment

### 1. System Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+ 
- **Storage**: 50GB+ SSD
- **Network**: Stable internet connection

### 2. Production Configuration
```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Enable log rotation
./scripts/setup_log_rotation.sh

# Setup automated backups
./scripts/setup_backups.sh
```

### 3. Monitoring Setup
```bash
# Setup alerting
./scripts/setup_alerts.sh

# Configure external monitoring
./scripts/setup_external_monitoring.sh
```

## 📈 Maintenance

### Regular Maintenance Tasks
```bash
# Update to latest version
git pull
docker-compose build --no-cache
docker-compose up -d

# Clean up old images
docker image prune -a

# Database maintenance (if using PostgreSQL)
docker-compose exec db pg_dump trading_db > backup.sql
```

### Model Updates
```bash
# Retrain models
docker-compose run --rm trading-bot train

# Deploy new models
docker-compose restart trading-bot
```

## 📞 Support

### Getting Help
1. Check logs: `docker-compose logs -f`
2. Verify configuration: `docker-compose config`
3. Test connectivity: Run debug commands above
4. Review documentation in `/DOCS` directory

### Useful Commands Reference
```bash
# Complete restart
docker-compose down && docker-compose up -d

# Rebuild specific service
docker-compose build --no-cache web-interface

# Run one-off commands
docker-compose run --rm trading-bot python script.py

# Export/import configuration
docker-compose run --rm trading-bot python -m src.config_manager export
```

---

## 📋 Deployment Checklist

- [ ] Docker and Docker Compose installed
- [ ] Environment variables configured in `.env`
- [ ] Binance API credentials validated
- [ ] Strong passwords set for admin interfaces
- [ ] Data directories have proper permissions
- [ ] Firewall configured for required ports
- [ ] SSL certificates configured (production)
- [ ] Monitoring dashboards accessible
- [ ] Backup strategy implemented
- [ ] Alert notifications configured

**🎉 Your AlphaBeta808 Trading system is now ready for deployment!**
