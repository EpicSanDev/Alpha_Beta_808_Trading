"""
AlphaBeta808 Trading Bot - Production Monitoring and Alerting System
This module provides comprehensive monitoring, alerting, and health checks for production deployment.
"""

import logging
import time
import threading
import smtplib
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import psutil
import os
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    details: Dict[str, Any] = None

class ProductionMonitor:
    """
    Comprehensive production monitoring system for the trading bot.
    Monitors system health, trading performance, and sends alerts.
    """
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config = self._load_config(config_path)
        self.alerts: List[Alert] = []
        self.is_running = False
        self.monitor_thread = None
        
        # Performance tracking
        self.performance_metrics = {
            "trades_executed": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "total_pnl": 0.0,
            "uptime_start": datetime.now(),
            "last_trade_time": None,
            "model_predictions": 0,
            "api_calls": 0,
            "errors": 0
        }
        
        # System health thresholds
        self.thresholds = {
            "cpu_usage": 80.0,  # %
            "memory_usage": 85.0,  # %
            "disk_usage": 90.0,  # %
            "api_response_time": 5.0,  # seconds
            "max_consecutive_errors": 5,
            "min_free_disk_gb": 1.0,
            "max_trade_loss": -1000.0  # USD
        }
        
        logger.info("Production monitor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration from file."""
        default_config = {
            "check_interval": 60,  # seconds
            "email_notifications": True,
            "slack_notifications": False,
            "telegram_notifications": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email_recipients": [],
            "health_check_endpoints": [
                "http://localhost:5000/api/system/status",
                "http://localhost:5000/api/bot/status"
            ]
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return default_config
    
    def start_monitoring(self):
        """Start the monitoring system in a separate thread."""
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Production monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._check_system_health()
                self._check_application_health()
                self._check_trading_performance()
                self._process_alerts()
                
                time.sleep(self.config.get("check_interval", 60))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._create_alert(
                    AlertLevel.ERROR,
                    f"Monitoring system error: {e}",
                    "monitor"
                )
                time.sleep(30)  # Shorter sleep on error
    
    def _check_system_health(self):
        """Check system resource usage and health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.thresholds["cpu_usage"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"High CPU usage: {cpu_percent:.1f}%",
                    "system",
                    {"cpu_usage": cpu_percent}
                )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            if memory_percent > self.thresholds["memory_usage"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"High memory usage: {memory_percent:.1f}%",
                    "system",
                    {"memory_usage": memory_percent, "available_gb": memory.available / (1024**3)}
                )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            free_gb = disk.free / (1024**3)
            
            if disk_percent > self.thresholds["disk_usage"]:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"High disk usage: {disk_percent:.1f}%",
                    "system",
                    {"disk_usage": disk_percent, "free_gb": free_gb}
                )
            elif free_gb < self.thresholds["min_free_disk_gb"]:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"Low disk space: {free_gb:.2f} GB remaining",
                    "system",
                    {"free_gb": free_gb}
                )
            
            # Network connectivity
            try:
                response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
                if response.status_code != 200:
                    self._create_alert(
                        AlertLevel.ERROR,
                        "Binance API connectivity issue",
                        "network"
                    )
            except Exception as e:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"Network connectivity error: {e}",
                    "network"
                )
            
            logger.debug(f"System health check: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%, Disk={disk_percent:.1f}%")
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
    
    def _check_application_health(self):
        """Check application endpoints and service health."""
        for endpoint in self.config.get("health_check_endpoints", []):
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=10)
                response_time = time.time() - start_time
                
                if response.status_code != 200:
                    self._create_alert(
                        AlertLevel.ERROR,
                        f"Health check failed for {endpoint}: HTTP {response.status_code}",
                        "application"
                    )
                elif response_time > self.thresholds["api_response_time"]:
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Slow response from {endpoint}: {response_time:.2f}s",
                        "application",
                        {"response_time": response_time}
                    )
                
                logger.debug(f"Health check {endpoint}: {response.status_code} in {response_time:.2f}s")
                
            except requests.exceptions.RequestException as e:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"Health check failed for {endpoint}: {e}",
                    "application"
                )
            except Exception as e:
                logger.error(f"Error in health check for {endpoint}: {e}")
    
    def _check_trading_performance(self):
        """Monitor trading performance and metrics."""
        try:
            # Check for recent trading activity
            if self.performance_metrics["last_trade_time"]:
                time_since_last_trade = datetime.now() - self.performance_metrics["last_trade_time"]
                if time_since_last_trade > timedelta(hours=2):  # No trades in 2 hours
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"No trading activity for {time_since_last_trade}",
                        "trading"
                    )
            
            # Check success rate
            total_trades = self.performance_metrics["trades_executed"]
            if total_trades > 10:  # Only check if we have enough data
                success_rate = self.performance_metrics["successful_trades"] / total_trades
                if success_rate < 0.3:  # Less than 30% success rate
                    self._create_alert(
                        AlertLevel.WARNING,
                        f"Low trading success rate: {success_rate:.1%}",
                        "trading",
                        {"success_rate": success_rate, "total_trades": total_trades}
                    )
            
            # Check for significant losses
            if self.performance_metrics["total_pnl"] < self.thresholds["max_trade_loss"]:
                self._create_alert(
                    AlertLevel.CRITICAL,
                    f"Significant trading losses: ${self.performance_metrics['total_pnl']:.2f}",
                    "trading",
                    {"total_pnl": self.performance_metrics["total_pnl"]}
                )
            
            # Check error rate
            if self.performance_metrics["errors"] > self.thresholds["max_consecutive_errors"]:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"High error rate: {self.performance_metrics['errors']} recent errors",
                    "trading"
                )
            
            logger.debug(f"Trading performance: {total_trades} trades, PnL: ${self.performance_metrics['total_pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"Error checking trading performance: {e}")
    
    def _create_alert(self, level: AlertLevel, message: str, component: str, details: Dict[str, Any] = None):
        """Create and queue an alert."""
        alert = Alert(
            level=level,
            message=message,
            timestamp=datetime.now(),
            component=component,
            details=details or {}
        )
        
        self.alerts.append(alert)
        logger.log(
            logging.CRITICAL if level == AlertLevel.CRITICAL else
            logging.ERROR if level == AlertLevel.ERROR else
            logging.WARNING if level == AlertLevel.WARNING else
            logging.INFO,
            f"[{component.upper()}] {message}"
        )
    
    def _process_alerts(self):
        """Process queued alerts and send notifications."""
        if not self.alerts:
            return
        
        # Group alerts by level for batching
        alerts_by_level = {}
        for alert in self.alerts:
            if alert.level not in alerts_by_level:
                alerts_by_level[alert.level] = []
            alerts_by_level[alert.level].append(alert)
        
        # Send notifications for critical and error alerts immediately
        critical_and_errors = []
        if AlertLevel.CRITICAL in alerts_by_level:
            critical_and_errors.extend(alerts_by_level[AlertLevel.CRITICAL])
        if AlertLevel.ERROR in alerts_by_level:
            critical_and_errors.extend(alerts_by_level[AlertLevel.ERROR])
        
        if critical_and_errors:
            self._send_notifications(critical_and_errors, urgent=True)
        
        # Send daily summary for warnings and info
        warnings_and_info = []
        if AlertLevel.WARNING in alerts_by_level:
            warnings_and_info.extend(alerts_by_level[AlertLevel.WARNING])
        if AlertLevel.INFO in alerts_by_level:
            warnings_and_info.extend(alerts_by_level[AlertLevel.INFO])
        
        if warnings_and_info:
            self._send_notifications(warnings_and_info, urgent=False)
        
        # Clear processed alerts
        self.alerts.clear()
    
    def _send_notifications(self, alerts: List[Alert], urgent: bool = False):
        """Send notifications for alerts."""
        try:
            if self.config.get("email_notifications", True):
                self._send_email_notification(alerts, urgent)
            
            if self.config.get("slack_notifications", False):
                self._send_slack_notification(alerts, urgent)
                
            if self.config.get("telegram_notifications", False):
                self._send_telegram_notification(alerts, urgent)
                
        except Exception as e:
            logger.error(f"Error sending notifications: {e}")
    
    def _send_email_notification(self, alerts: List[Alert], urgent: bool):
        """Send email notifications."""
        try:
            recipients = self.config.get("email_recipients", [])
            if not recipients:
                return
            
            smtp_server = self.config.get("smtp_server")
            smtp_port = self.config.get("smtp_port", 587)
            email_user = os.getenv("EMAIL_USER")
            email_password = os.getenv("EMAIL_PASSWORD")
            
            if not all([smtp_server, email_user, email_password]):
                logger.warning("Email configuration incomplete, skipping email notification")
                return
            
            subject = f"{'🚨 URGENT' if urgent else '📊'} AlphaBeta808 Trading Bot Alert"
            
            # Create HTML email content
            html_content = self._create_email_html(alerts, urgent)
            
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = email_user
            msg['To'] = ", ".join(recipients)
            
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(email_user, email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _create_email_html(self, alerts: List[Alert], urgent: bool) -> str:
        """Create HTML content for email notifications."""
        
        # Color coding by alert level
        level_colors = {
            AlertLevel.CRITICAL: "#dc3545",
            AlertLevel.ERROR: "#fd7e14", 
            AlertLevel.WARNING: "#ffc107",
            AlertLevel.INFO: "#17a2b8"
        }
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: {'#dc3545' if urgent else '#007bff'}; color: white; padding: 20px; border-radius: 5px; }}
                .alert {{ margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 5px solid; }}
                .metrics {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{'🚨 URGENT ALERT' if urgent else '📊 System Report'}</h2>
                <p>AlphaBeta808 Trading Bot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <h3>System Metrics</h3>
                <ul>
                    <li>Uptime: {datetime.now() - self.performance_metrics['uptime_start']}</li>
                    <li>Total Trades: {self.performance_metrics['trades_executed']}</li>
                    <li>Success Rate: {(self.performance_metrics['successful_trades'] / max(1, self.performance_metrics['trades_executed'])):.1%}</li>
                    <li>Total P&L: ${self.performance_metrics['total_pnl']:.2f}</li>
                    <li>Recent Errors: {self.performance_metrics['errors']}</li>
                </ul>
            </div>
        """
        
        for alert in alerts:
            color = level_colors.get(alert.level, "#6c757d")
            html += f"""
            <div class="alert" style="border-left-color: {color}; background-color: {color}15;">
                <strong>{alert.level.value.upper()}</strong> - {alert.component.upper()}<br>
                <strong>{alert.message}</strong><br>
                <small>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
                {f'<br><small>Details: {alert.details}</small>' if alert.details else ''}
            </div>
            """
        
        html += """
            <div class="footer">
                <p>This is an automated message from AlphaBeta808 Trading Bot monitoring system.</p>
                <p>For support, please check the application logs or contact the system administrator.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _send_slack_notification(self, alerts: List[Alert], urgent: bool):
        """Send Slack notifications."""
        # Implement Slack webhook notification
        # This would require SLACK_WEBHOOK_URL environment variable
        pass
    
    def _send_telegram_notification(self, alerts: List[Alert], urgent: bool):
        """Send Telegram notifications."""
        # Implement Telegram bot notification
        # This would require TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
        pass
    
    def update_performance_metric(self, metric: str, value: Any):
        """Update a performance metric."""
        if metric in self.performance_metrics:
            self.performance_metrics[metric] = value
            logger.debug(f"Updated metric {metric}: {value}")
    
    def increment_metric(self, metric: str, increment: int = 1):
        """Increment a numeric metric."""
        if metric in self.performance_metrics:
            self.performance_metrics[metric] += increment
            logger.debug(f"Incremented metric {metric} by {increment}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        uptime = datetime.now() - self.performance_metrics["uptime_start"]
        
        return {
            "system_status": "healthy",
            "uptime": str(uptime),
            "uptime_seconds": uptime.total_seconds(),
            "performance_metrics": self.performance_metrics.copy(),
            "recent_alerts": len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=1)]),
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            },
            "thresholds": self.thresholds
        }

# Global monitor instance
production_monitor = ProductionMonitor()

def start_production_monitoring():
    """Start the production monitoring system."""
    production_monitor.start_monitoring()

def stop_production_monitoring():
    """Stop the production monitoring system."""
    production_monitor.stop_monitoring()

if __name__ == "__main__":
    # Example usage
    start_production_monitoring()
    
    try:
        # Keep the monitor running
        while True:
            time.sleep(60)
            print(f"Monitor status: {production_monitor.get_status_report()}")
    except KeyboardInterrupt:
        print("Stopping monitor...")
        stop_production_monitoring()
