#!/usr/bin/env python3
"""
Production Health Check and Monitoring Service
Comprehensive monitoring for AlphaBeta808 Trading Bot
"""

import os
import sys
import time
import json
import psutil
import requests
import smtplib
import logging
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
from typing import Dict, List, Optional
import sqlite3

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class ProductionHealthMonitor:
    def __init__(self):
        self.setup_logging()
        self.load_config()
        self.alert_cooldown = {}  # Prevent spam alerts
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'health_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('HealthMonitor')
        
    def load_config(self):
        """Load configuration from environment and config files"""
        # Load from environment variables
        self.config = {
            'email_enabled': os.getenv('EMAIL_ENABLED', 'false').lower() == 'true',
            'email_host': os.getenv('EMAIL_HOST', 'smtp.gmail.com'),
            'email_port': int(os.getenv('EMAIL_PORT', 587)),
            'email_username': os.getenv('EMAIL_USERNAME', ''),
            'email_password': os.getenv('EMAIL_PASSWORD', ''),
            'email_recipients': os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(','),
            'check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', 60)),
            'web_interface_port': int(os.getenv('WEB_PORT', 5000)),
            'max_cpu_usage': float(os.getenv('MAX_CPU_USAGE', 80.0)),
            'max_memory_usage': float(os.getenv('MAX_MEMORY_USAGE', 80.0)),
            'max_disk_usage': float(os.getenv('MAX_DISK_USAGE', 90.0)),
        }
        
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Process count
            process_count = len(psutil.pids())
            
            return {
                'status': 'healthy',
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'process_count': process_count,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def check_web_interface(self) -> Dict:
        """Check if web interface is responsive"""
        try:
            # Try HTTP first
            url = f"http://localhost:{self.config['web_interface_port']}/health"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return {
                    'status': 'healthy',
                    'response_time': response.elapsed.total_seconds(),
                    'protocol': 'http'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'status_code': response.status_code,
                    'protocol': 'http'
                }
                
        except requests.exceptions.ConnectionError:
            # Try HTTPS if HTTP fails
            try:
                url = "https://localhost:5443/health"
                # For localhost, we can disable SSL verification with a warning
                # In production, you should use proper SSL certificates
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(url, timeout=10, verify=False)
                
                if response.status_code == 200:
                    return {
                        'status': 'healthy',
                        'response_time': response.elapsed.total_seconds(),
                        'protocol': 'https'
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'status_code': response.status_code,
                        'protocol': 'https'
                    }
            except Exception as e:
                return {
                    'status': 'error',
                    'error': f"Web interface not responding: {str(e)}"
                }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_trading_bot(self) -> Dict:
        """Check if trading bot is running and healthy"""
        try:
            # Check PID file
            pid_file = Path("logs/trading_bot.pid")
            if pid_file.exists():
                pid = int(pid_file.read_text().strip())
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    return {
                        'status': 'running',
                        'pid': pid,
                        'cpu_percent': process.cpu_percent(),
                        'memory_percent': process.memory_percent(),
                        'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
                    }
                else:
                    return {'status': 'stopped', 'error': 'PID not found in process list'}
            else:
                return {'status': 'stopped', 'error': 'PID file not found'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_database(self) -> Dict:
        """Check database connectivity and integrity"""
        try:
            db_path = "trading_web.db"
            if not os.path.exists(db_path):
                return {'status': 'missing', 'error': 'Database file not found'}
            
            # Test connection
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            
            # Check file size
            file_size = os.path.getsize(db_path)
            
            return {
                'status': 'healthy',
                'table_count': table_count,
                'file_size_mb': round(file_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def check_log_files(self) -> Dict:
        """Check log file status and recent activity"""
        try:
            log_file = Path("logs/trading_bot.log")
            if not log_file.exists():
                return {'status': 'missing', 'error': 'Log file not found'}
            
            # Check file size and modification time
            stat = log_file.stat()
            file_size = stat.st_size
            last_modified = datetime.fromtimestamp(stat.st_mtime)
            
            # Check if logs are recent (within last hour)
            time_since_update = datetime.now() - last_modified
            is_recent = time_since_update < timedelta(hours=1)
            
            return {
                'status': 'healthy' if is_recent else 'stale',
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'last_modified': last_modified.isoformat(),
                'minutes_since_update': int(time_since_update.total_seconds() / 60)
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def send_alert(self, subject: str, message: str, alert_type: str = 'warning'):
        """Send email alert if configured"""
        if not self.config['email_enabled']:
            return
            
        # Check cooldown to prevent spam
        cooldown_key = f"{alert_type}_{subject}"
        now = time.time()
        if cooldown_key in self.alert_cooldown:
            if now - self.alert_cooldown[cooldown_key] < 300:  # 5-minute cooldown
                return
        
        self.alert_cooldown[cooldown_key] = now
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['email_username']
            msg['To'] = ', '.join(self.config['email_recipients'])
            msg['Subject'] = f"[AlphaBeta808] {subject}"
            
            body = f"""
AlphaBeta808 Trading Bot Alert

Type: {alert_type.upper()}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

{message}

This is an automated alert from your trading bot monitoring system.
"""
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['email_host'], self.config['email_port'])
            server.starttls()
            server.login(self.config['email_username'], self.config['email_password'])
            text = msg.as_string()
            server.sendmail(self.config['email_username'], self.config['email_recipients'], text)
            server.quit()
            
            self.logger.info(f"Alert sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def run_health_check(self) -> Dict:
        """Run comprehensive health check"""
        self.logger.info("Running health check...")
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # System resources
        system_check = self.check_system_resources()
        health_status['checks']['system'] = system_check
        
        # Check for resource alerts
        if system_check.get('cpu_percent', 0) > self.config['max_cpu_usage']:
            self.send_alert(
                "High CPU Usage",
                f"CPU usage is {system_check['cpu_percent']:.1f}% (threshold: {self.config['max_cpu_usage']}%)",
                "warning"
            )
            health_status['overall_status'] = 'warning'
        
        if system_check.get('memory_percent', 0) > self.config['max_memory_usage']:
            self.send_alert(
                "High Memory Usage",
                f"Memory usage is {system_check['memory_percent']:.1f}% (threshold: {self.config['max_memory_usage']}%)",
                "warning"
            )
            health_status['overall_status'] = 'warning'
        
        # Web interface
        web_check = self.check_web_interface()
        health_status['checks']['web_interface'] = web_check
        
        if web_check['status'] != 'healthy':
            self.send_alert(
                "Web Interface Down",
                f"Web interface is not responding: {web_check.get('error', 'Unknown error')}",
                "critical"
            )
            health_status['overall_status'] = 'critical'
        
        # Trading bot
        bot_check = self.check_trading_bot()
        health_status['checks']['trading_bot'] = bot_check
        
        if bot_check['status'] != 'running':
            self.send_alert(
                "Trading Bot Stopped",
                f"Trading bot is not running: {bot_check.get('error', 'Unknown error')}",
                "critical"
            )
            health_status['overall_status'] = 'critical'
        
        # Database
        db_check = self.check_database()
        health_status['checks']['database'] = db_check
        
        if db_check['status'] != 'healthy':
            self.send_alert(
                "Database Issue",
                f"Database problem detected: {db_check.get('error', 'Unknown error')}",
                "warning"
            )
            if health_status['overall_status'] == 'healthy':
                health_status['overall_status'] = 'warning'
        
        # Log files
        log_check = self.check_log_files()
        health_status['checks']['logs'] = log_check
        
        return health_status
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop"""
        self.logger.info("Starting production health monitoring...")
        self.logger.info(f"Check interval: {self.config['check_interval']} seconds")
        
        while True:
            try:
                health_status = self.run_health_check()
                
                # Save status to file
                status_file = Path("logs/health_status.json")
                with open(status_file, 'w') as f:
                    json.dump(health_status, f, indent=2)
                
                # Log summary
                overall = health_status['overall_status']
                self.logger.info(f"Health check complete - Status: {overall}")
                
                if overall != 'healthy':
                    self.logger.warning(f"System status: {overall}")
                
                time.sleep(self.config['check_interval'])
                
            except KeyboardInterrupt:
                self.logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait before retrying

def main():
    """Main function"""
    monitor = ProductionHealthMonitor()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        # Run single health check
        health_status = monitor.run_health_check()
        print(json.dumps(health_status, indent=2))
    else:
        # Run continuous monitoring
        monitor.run_monitoring_loop()

if __name__ == '__main__':
    main()
