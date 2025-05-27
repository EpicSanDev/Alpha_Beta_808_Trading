# 🚀 AlphaBeta808 Trading Bot - Railway Deployment Guide

## 📋 Prerequisites

1. **Railway Account**: Create account at [railway.app](https://railway.app)
2. **GitHub Repository**: Push your code to GitHub
3. **API Credentials**: Have your Binance API credentials ready

## 🔧 Deployment Steps

### Step 1: Create Railway Project

1. Login to Railway
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your AlphaBeta808Trading repository
4. Railway will detect the configuration automatically

### Step 2: Configure Environment Variables

In Railway dashboard, add these environment variables:

#### Required Variables
```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=your-secret-key-here-generate-with-openssl-rand-base64-32

# Authentication (CHANGE THESE!)
WEB_ADMIN_USER=admin
WEB_ADMIN_PASSWORD=change-this-secure-password

# Trading API
BINANCE_API_KEY=your-binance-api-key
BINANCE_API_SECRET=your-binance-api-secret

# Security
WEBHOOK_SECRET=your-webhook-secret-here
```

#### Optional Variables
```bash
# Notifications
EMAIL_ENABLED=false
TELEGRAM_ENABLED=false

# Database
DATABASE_URL=sqlite:///trading_web.db

# Monitoring
LOG_LEVEL=INFO
HEALTH_CHECK_INTERVAL=60
```

### Step 3: Generate Secure Keys

Generate secure keys before deployment:

```bash
# Secret key for Flask sessions
openssl rand -base64 32

# Webhook secret
openssl rand -base64 32
```

### Step 4: Deploy

1. Railway will automatically build using `Dockerfile.railway`
2. The web interface will be available at your Railway-generated URL
3. Health check endpoint: `https://your-app.railway.app/health`

## 🔍 Post-Deployment Verification

### 1. Check Application Status
```bash
curl https://your-app.railway.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-05-27T...",
  "version": "1.0.0",
  "checks": {
    "database": {"status": "ok"},
    "bot_manager": {"status": "ok"},
    "disk_space": {"status": "ok"},
    "memory_usage": {"status": "ok"}
  }
}
```

### 2. Access Web Interface
1. Go to your Railway app URL
2. Login with your admin credentials
3. Verify dashboard loads correctly

### 3. Test API Connection
1. Go to Settings page
2. Add your Binance API credentials
3. Click "Test Connection"

## 📊 Monitoring

### Railway Built-in Monitoring
- **Logs**: View in Railway dashboard
- **Metrics**: CPU, Memory, Network usage
- **Health Checks**: Automatic monitoring of `/health` endpoint

### Application Monitoring
- **Dashboard**: Real-time trading metrics
- **Alerts**: Configure email/Telegram notifications
- **Performance**: Monitor trade execution and profitability

## 🛠️ Configuration Management

### Database Persistence
- SQLite database is persistent on Railway
- Backups can be configured through Railway CLI

### Environment Updates
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link project
railway login
railway link

# Update environment variables
railway variables set BINANCE_API_KEY=new-key
```

## 🔒 Security Best Practices

### 1. Secure Your Credentials
- Use Railway's environment variables for sensitive data
- Never commit secrets to GitHub
- Rotate API keys regularly

### 2. Access Control
- Change default admin password immediately
- Use strong passwords (recommended: 16+ characters)
- Enable 2FA on your Binance account

### 3. API Security
- Use testnet/sandbox initially
- Set appropriate API permissions
- Monitor API usage regularly

## 🚨 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check build logs in Railway dashboard
# Common causes:
# - Missing environment variables
# - Docker build timeout
# - Package installation errors
```

#### Application Not Starting
```bash
# Check application logs
# Common causes:
# - Missing SECRET_KEY
# - Database connection issues
# - Port binding problems
```

#### Health Check Failures
```bash
# Test health endpoint locally
curl https://your-app.railway.app/health

# Check for:
# - Application startup errors
# - Database connectivity
# - Memory/disk issues
```

### Support Commands
```bash
# View logs
railway logs

# Check service status
railway status

# Restart service
railway redeploy
```

## 📈 Scaling and Optimization

### Performance Tuning
- Monitor memory usage (Railway provides metrics)
- Optimize model loading for faster startup
- Configure appropriate health check timeouts

### Cost Optimization
- Use Railway's free tier for testing
- Monitor resource usage to optimize costs
- Scale down during maintenance periods

## 🔄 Updates and Maintenance

### Updating the Application
1. Push changes to GitHub
2. Railway automatically redeploys
3. Monitor deployment logs for issues

### Backup Strategy
```bash
# Database backup (using Railway CLI)
railway run python scripts/backup_database.py

# Configuration backup
railway variables > environment_backup.txt
```

## 📞 Support

### Documentation
- [Railway Documentation](https://docs.railway.app/)
- [AlphaBeta808 Documentation](./DOCS/)

### Community
- [Railway Discord](https://discord.gg/railway)
- [GitHub Issues](https://github.com/yourusername/AlphaBeta808Trading/issues)

---

**🎯 Quick Start Summary:**
1. Fork repository to GitHub
2. Create Railway project from GitHub
3. Set environment variables (see Step 2)
4. Deploy and verify health endpoint
5. Access web interface and configure trading

**⚠️ Important:** Always test with small amounts and sandbox mode first!
