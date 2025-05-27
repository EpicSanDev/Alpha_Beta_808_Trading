# 🚂 Railway Deployment - AlphaBeta808 Trading Bot

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/yourusername/AlphaBeta808Trading)

## 🚀 One-Click Deploy to Railway

This AlphaBeta808 Trading Bot is optimized for deployment on Railway platform with automatic configuration.

### ⚡ Quick Deploy

1. Click the "Deploy on Railway" button above
2. Connect your GitHub account
3. Set your environment variables (see below)
4. Deploy automatically!

## 🔧 Environment Variables Required

Set these in your Railway dashboard before deployment:

### 🔑 Authentication & Security
```bash
SECRET_KEY=your-secret-key-32-chars          # Generate with: openssl rand -base64 32
WEB_ADMIN_USER=admin                         # Change this!
WEB_ADMIN_PASSWORD=secure-password-here      # Change this!
WEBHOOK_SECRET=webhook-secret-32-chars       # Generate with: openssl rand -base64 32
```

### 📈 Trading Configuration
```bash
BINANCE_API_KEY=your-binance-api-key
BINANCE_API_SECRET=your-binance-api-secret
```

### 🛠️ Optional Configuration
```bash
FLASK_ENV=production
LOG_LEVEL=INFO
EMAIL_ENABLED=false
TELEGRAM_ENABLED=false
```

## 📊 Features on Railway

- ✅ **Automatic Health Checks**: Railway monitors `/health` endpoint
- ✅ **Zero-Downtime Deployments**: Seamless updates
- ✅ **Built-in Monitoring**: CPU, memory, and network metrics
- ✅ **Persistent Storage**: SQLite database persistence
- ✅ **Environment Management**: Secure variable storage
- ✅ **Custom Domains**: Connect your own domain
- ✅ **SSL/TLS**: Automatic HTTPS certificates

## 🔍 After Deployment

### 1. Verify Health Status
Your app will be available at: `https://your-app-name.railway.app`

Check health: `https://your-app-name.railway.app/health`

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-05-27T...",
  "version": "1.0.0"
}
```

### 2. Access Dashboard
1. Go to your Railway app URL
2. Login with your configured credentials
3. Configure your trading parameters

### 3. Start Trading
1. Navigate to Settings → API Configuration
2. Test your Binance API connection
3. Enable trading mode (start with paper trading!)

## 🛠️ Configuration Files

Railway deployment uses these optimized files:

- **`railway.toml`**: Railway platform configuration
- **`Dockerfile.railway`**: Optimized Docker build for Railway
- **`Procfile`**: Service startup configuration
- **`.env.railway`**: Environment variables template

## 📈 Monitoring

### Railway Dashboard
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, network usage
- **Deployments**: Deployment history and rollbacks

### Trading Dashboard
- **Performance**: Real-time P&L and statistics
- **Positions**: Active trades and portfolio
- **Signals**: ML model predictions and signals

## 🔒 Security Best Practices

1. **Change Default Credentials**: Update admin username/password
2. **Use Testnet First**: Test with Binance sandbox before live trading
3. **Monitor API Usage**: Keep track of API calls and limits
4. **Regular Backups**: Export configurations regularly
5. **Key Rotation**: Rotate API keys periodically

## 🚨 Troubleshooting

### Common Issues

#### Deployment Fails
- Check build logs in Railway dashboard
- Verify all required environment variables are set
- Ensure your GitHub repository is public or properly connected

#### Health Check Fails
- Verify `/health` endpoint is accessible
- Check application logs for startup errors
- Ensure PORT environment variable is set correctly

#### Can't Access Dashboard
- Verify your app URL is correct
- Check admin credentials in environment variables
- Look for authentication errors in logs

### Support Commands

Using Railway CLI:
```bash
# Install CLI
npm install -g @railway/cli

# Login and link
railway login
railway link

# View logs
railway logs

# Check status
railway status

# Update environment
railway variables set SECRET_KEY=new-secret
```

## 🔄 Updates

Railway automatically redeploys when you push to your connected GitHub repository.

### Manual Redeploy
1. Go to Railway dashboard
2. Click "Deploy" → "Redeploy"
3. Monitor deployment logs

### Environment Updates
Update variables in Railway dashboard → Variables section

## 💰 Railway Pricing

- **Hobby Plan**: $5/month for personal projects
- **Pro Plan**: $20/month for production use
- **Usage-based**: Pay for what you use (CPU, memory, network)

See [Railway Pricing](https://railway.app/pricing) for current rates.

## 📞 Support

### Documentation
- [Railway Documentation](https://docs.railway.app/)
- [AlphaBeta808 Deployment Guide](./RAILWAY_DEPLOYMENT.md)

### Community
- [Railway Discord](https://discord.gg/railway)
- [Railway GitHub](https://github.com/railwayapp/railway)

---

## 🎯 Quick Start Checklist

- [ ] Click "Deploy on Railway" button
- [ ] Connect GitHub repository
- [ ] Set `SECRET_KEY` environment variable
- [ ] Set `WEB_ADMIN_USER` and `WEB_ADMIN_PASSWORD`
- [ ] Set `BINANCE_API_KEY` and `BINANCE_API_SECRET`
- [ ] Deploy and wait for health check to pass
- [ ] Access dashboard and configure trading
- [ ] Start with paper trading mode
- [ ] Monitor performance and logs

**⚠️ Important**: Always test with small amounts first!

---

**🚀 Ready to deploy?** [Click here to deploy on Railway](https://railway.app/new/template?template=https://github.com/yourusername/AlphaBeta808Trading)
