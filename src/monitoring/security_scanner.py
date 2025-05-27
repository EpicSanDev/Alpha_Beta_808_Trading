#!/usr/bin/env python3
"""
AlphaBeta808 Trading Bot - Production Security Scanner
Automated security assessment and vulnerability detection for production deployment
"""

import os
import sys
import json
import logging
import subprocess
import re
import hashlib
import socket
import ssl
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sqlite3

class SecurityScanner:
    def __init__(self, config_path: str = "security_config.json"):
        """Initialize security scanner"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.vulnerabilities = []
        self.security_score = 100.0
        
        # Security patterns
        self.sensitive_patterns = [
            r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?[\w\-@!#$%^&*()+={}[\]|\\:";\'<>?,./]+["\']?',
            r'(?i)(api[_\-]?key|apikey)\s*[=:]\s*["\']?[\w\-]+["\']?',
            r'(?i)(secret[_\-]?key|secretkey)\s*[=:]\s*["\']?[\w\-]+["\']?',
            r'(?i)(token)\s*[=:]\s*["\']?[\w\-\._]+["\']?',
            r'(?i)(access[_\-]?key|accesskey)\s*[=:]\s*["\']?[\w\-]+["\']?'
        ]
        
        self.insecure_patterns = [
            r'(?i)ssl[_\-]?verify\s*[=:]\s*false',
            r'(?i)verify\s*[=:]\s*false',
            r'(?i)debug\s*[=:]\s*true',
            r'(?i)test[_\-]?mode\s*[=:]\s*true'
        ]

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load security scanner configuration"""
        default_config = {
            "scan_types": {
                "file_permissions": True,
                "sensitive_data": True,
                "ssl_certificates": True,
                "network_security": True,
                "dependency_scan": True,
                "code_analysis": True,
                "database_security": True
            },
            "excluded_paths": [
                ".git/",
                "__pycache__/",
                ".venv/",
                "venv/",
                "node_modules/",
                ".pytest_cache/"
            ],
            "severity_weights": {
                "critical": 30,
                "high": 20,
                "medium": 10,
                "low": 5,
                "info": 1
            },
            "ssl_check": {
                "hosts": ["localhost:5443", "localhost:443"],
                "min_tls_version": "TLSv1.2"
            },
            "file_scan": {
                "extensions": [".py", ".js", ".json", ".yaml", ".yml", ".env", ".conf", ".cfg"],
                "max_file_size": 10485760  # 10MB
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load security config: {e}")
        
        return default_config

    def setup_logging(self):
        """Setup logging for security scanner"""
        self.logger = logging.getLogger('SecurityScanner')
        self.logger.setLevel(logging.INFO)
        
        # Console handler (always available)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # Try to add file handler if possible
        try:
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler('logs/security_scan.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Cannot create log file (read-only filesystem?): {e}")
            self.logger.info("Logging to console only")

    def add_vulnerability(self, severity: str, category: str, description: str, 
                         location: str = "", recommendation: str = "", cve: str = None):
        """Add a vulnerability to the scan results"""
        vulnerability = {
            'severity': severity,
            'category': category,
            'description': description,
            'location': location,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat(),
            'cve': cve
        }
        
        self.vulnerabilities.append(vulnerability)
        
        # Reduce security score based on severity
        weight = self.config['severity_weights'].get(severity, 5)
        self.security_score = max(0, self.security_score - weight)
        
        self.logger.warning(f"VULNERABILITY [{severity.upper()}]: {description} at {location}")

    def check_file_permissions(self) -> List[Dict[str, Any]]:
        """Check for insecure file permissions"""
        self.logger.info("Checking file permissions...")
        issues = []
        
        try:
            sensitive_files = [
                '.env', '.env.production', '.env.local',
                'config/trading_config.json',
                'ssl/server.key',
                'logs/',
                'backups/'
            ]
            
            for file_path in sensitive_files:
                if os.path.exists(file_path):
                    stat_info = os.stat(file_path)
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    # Check for world-readable permissions
                    if int(mode[2]) > 0:
                        self.add_vulnerability(
                            'high',
                            'file_permissions',
                            f"Sensitive file '{file_path}' is world-readable",
                            file_path,
                            f"Run: chmod 600 {file_path}"
                        )
                    
                    # Check for world-writable permissions
                    if int(mode[2]) > 1:
                        self.add_vulnerability(
                            'critical',
                            'file_permissions',
                            f"Sensitive file '{file_path}' is world-writable",
                            file_path,
                            f"Run: chmod 600 {file_path}"
                        )
                    
                    # Check for group permissions on sensitive files
                    if file_path.endswith('.key') and int(mode[1]) > 0:
                        self.add_vulnerability(
                            'medium',
                            'file_permissions',
                            f"Private key '{file_path}' has group permissions",
                            file_path,
                            f"Run: chmod 600 {file_path}"
                        )
        
        except Exception as e:
            self.logger.error(f"File permissions check failed: {e}")
        
        return issues

    def scan_for_sensitive_data(self) -> List[Dict[str, Any]]:
        """Scan for hardcoded sensitive data"""
        self.logger.info("Scanning for sensitive data...")
        issues = []
        
        try:
            excluded_paths = self.config['excluded_paths']
            file_extensions = self.config['file_scan']['extensions']
            max_file_size = self.config['file_scan']['max_file_size']
            
            for root, dirs, files in os.walk('.'):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(excl in os.path.join(root, d) for excl in excluded_paths)]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip files that don't match our criteria
                    if not any(file.endswith(ext) for ext in file_extensions):
                        continue
                    
                    try:
                        if os.path.getsize(file_path) > max_file_size:
                            continue
                        
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Check for sensitive patterns
                        for pattern in self.sensitive_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_number = content[:match.start()].count('\n') + 1
                                
                                self.add_vulnerability(
                                    'high',
                                    'sensitive_data',
                                    f"Potential hardcoded credential found",
                                    f"{file_path}:line {line_number}",
                                    "Move sensitive data to environment variables"
                                )
                        
                        # Check for insecure configurations
                        for pattern in self.insecure_patterns:
                            matches = re.finditer(pattern, content, re.IGNORECASE)
                            for match in matches:
                                line_number = content[:match.start()].count('\n') + 1
                                
                                self.add_vulnerability(
                                    'medium',
                                    'insecure_config',
                                    f"Insecure configuration detected: {match.group()}",
                                    f"{file_path}:line {line_number}",
                                    "Review and secure configuration"
                                )
                    
                    except Exception as e:
                        self.logger.debug(f"Could not scan file {file_path}: {e}")
        
        except Exception as e:
            self.logger.error(f"Sensitive data scan failed: {e}")
        
        return issues

    def check_ssl_certificates(self) -> List[Dict[str, Any]]:
        """Check SSL/TLS certificate security"""
        self.logger.info("Checking SSL certificates...")
        issues = []
        
        try:
            # Check local certificate files
            cert_files = ['ssl/server.crt', 'ssl/server.key']
            for cert_file in cert_files:
                if os.path.exists(cert_file):
                    # Check certificate expiration
                    if cert_file.endswith('.crt'):
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['openssl', 'x509', '-in', cert_file, '-text', '-noout'],
                                capture_output=True, text=True, timeout=10
                            )
                            
                            if result.returncode == 0:
                                # Extract expiration date
                                for line in result.stdout.split('\n'):
                                    if 'Not After' in line:
                                        # Parse expiration date
                                        # This is a simplified check
                                        if 'Dec 2024' in line:  # Self-signed certs expire quickly
                                            self.add_vulnerability(
                                                'medium',
                                                'ssl_certificate',
                                                "SSL certificate expires soon",
                                                cert_file,
                                                "Replace with production certificate"
                                            )
                        except Exception as e:
                            self.logger.debug(f"Could not check certificate {cert_file}: {e}")
                    
                    # Check key file permissions
                    if cert_file.endswith('.key'):
                        stat_info = os.stat(cert_file)
                        mode = oct(stat_info.st_mode)[-3:]
                        if mode != '600':
                            self.add_vulnerability(
                                'high',
                                'ssl_certificate',
                                f"SSL private key has insecure permissions: {mode}",
                                cert_file,
                                f"Run: chmod 600 {cert_file}"
                            )
            
            # Check SSL/TLS configuration for running services
            hosts = self.config['ssl_check']['hosts']
            for host in hosts:
                try:
                    hostname, port = host.split(':')
                    port = int(port)
                    
                    context = ssl.create_default_context()
                    with socket.create_connection((hostname, port), timeout=5) as sock:
                        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                            cert = ssock.getpeercert()
                            cipher = ssock.cipher()
                            
                            # Check TLS version
                            if cipher and cipher[1] < 'TLSv1.2':
                                self.add_vulnerability(
                                    'high',
                                    'ssl_configuration',
                                    f"Weak TLS version: {cipher[1]}",
                                    host,
                                    "Upgrade to TLS 1.2 or higher"
                                )
                            
                            # Check cipher strength
                            if cipher and cipher[2] < 128:
                                self.add_vulnerability(
                                    'medium',
                                    'ssl_configuration',
                                    f"Weak cipher key length: {cipher[2]} bits",
                                    host,
                                    "Use stronger ciphers (≥128 bits)"
                                )
                
                except Exception as e:
                    self.logger.debug(f"Could not check SSL for {host}: {e}")
        
        except Exception as e:
            self.logger.error(f"SSL certificate check failed: {e}")
        
        return issues

    def check_network_security(self) -> List[Dict[str, Any]]:
        """Check network security configuration"""
        self.logger.info("Checking network security...")
        issues = []
        
        try:
            # Check for open ports
            common_ports = [22, 80, 443, 3306, 5432, 6379, 27017]
            
            for port in common_ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex(('localhost', port))
                    sock.close()
                    
                    if result == 0:
                        if port in [3306, 5432, 6379, 27017]:  # Database ports
                            self.add_vulnerability(
                                'high',
                                'network_security',
                                f"Database port {port} is open to localhost",
                                f"localhost:{port}",
                                "Ensure database is properly secured"
                            )
                        elif port == 22:  # SSH
                            self.add_vulnerability(
                                'medium',
                                'network_security',
                                f"SSH port {port} is open",
                                f"localhost:{port}",
                                "Ensure SSH is properly configured"
                            )
                
                except Exception as e:
                    self.logger.debug(f"Could not check port {port}: {e}")
            
            # Check for default credentials in config
            config_files = ['.env', '.env.production', 'config/trading_config.json']
            for config_file in config_files:
                if os.path.exists(config_file):
                    try:
                        with open(config_file, 'r') as f:
                            content = f.read()
                        
                        # Check for default passwords
                        default_patterns = [
                            r'(?i)password.*=.*admin',
                            r'(?i)password.*=.*123',
                            r'(?i)password.*=.*password',
                            r'(?i)secret.*=.*secret'
                        ]
                        
                        for pattern in default_patterns:
                            if re.search(pattern, content):
                                self.add_vulnerability(
                                    'critical',
                                    'authentication',
                                    f"Default/weak credentials detected",
                                    config_file,
                                    "Change default passwords immediately"
                                )
                    
                    except Exception as e:
                        self.logger.debug(f"Could not check config {config_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"Network security check failed: {e}")
        
        return issues

    def check_dependency_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for known vulnerabilities in dependencies"""
        self.logger.info("Checking dependency vulnerabilities...")
        issues = []
        
        try:
            # Check Python packages
            if os.path.exists('requirements.txt'):
                try:
                    # Simple package version check (would need CVE database for full check)
                    with open('requirements.txt', 'r') as f:
                        packages = f.read()
                    
                    # Check for packages with known vulnerabilities (simplified)
                    vulnerable_packages = {
                        'requests<2.20.0': 'CVE-2018-18074',
                        'flask<1.0': 'Multiple CVEs',
                        'numpy<1.16.0': 'CVE-2019-6446'
                    }
                    
                    for vuln_pkg, cve in vulnerable_packages.items():
                        pkg_name = vuln_pkg.split('<')[0]
                        if pkg_name in packages:
                            self.add_vulnerability(
                                'high',
                                'dependency_vulnerability',
                                f"Potentially vulnerable package: {pkg_name}",
                                'requirements.txt',
                                f"Update {pkg_name} to latest version",
                                cve
                            )
                
                except Exception as e:
                    self.logger.debug(f"Could not check requirements.txt: {e}")
        
        except Exception as e:
            self.logger.error(f"Dependency vulnerability check failed: {e}")
        
        return issues

    def check_database_security(self) -> List[Dict[str, Any]]:
        """Check database security configuration"""
        self.logger.info("Checking database security...")
        issues = []
        
        try:
            db_files = ['trading_web.db', 'trading_web_production.db']
            
            for db_file in db_files:
                if os.path.exists(db_file):
                    # Check file permissions
                    stat_info = os.stat(db_file)
                    mode = oct(stat_info.st_mode)[-3:]
                    
                    if int(mode[2]) > 0:
                        self.add_vulnerability(
                            'high',
                            'database_security',
                            f"Database file '{db_file}' is world-readable",
                            db_file,
                            f"Run: chmod 600 {db_file}"
                        )
                    
                    # Check database content security
                    try:
                        conn = sqlite3.connect(db_file)
                        cursor = conn.cursor()
                        
                        # Check for tables with sensitive data
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [row[0] for row in cursor.fetchall()]
                        
                        if 'users' in tables:
                            # Check if passwords are hashed
                            cursor.execute("SELECT COUNT(*) FROM users WHERE password NOT LIKE '$%'")
                            unhashed_count = cursor.fetchone()[0]
                            
                            if unhashed_count > 0:
                                self.add_vulnerability(
                                    'critical',
                                    'database_security',
                                    f"Found {unhashed_count} users with unhashed passwords",
                                    db_file,
                                    "Hash all passwords using bcrypt or similar"
                                )
                        
                        conn.close()
                    
                    except Exception as e:
                        self.logger.debug(f"Could not check database content: {e}")
        
        except Exception as e:
            self.logger.error(f"Database security check failed: {e}")
        
        return issues

    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        self.logger.info("Starting comprehensive security scan...")
        
        scan_start = datetime.now()
        scan_types = self.config['scan_types']
        
        # Reset vulnerabilities and score
        self.vulnerabilities = []
        self.security_score = 100.0
        
        # Run individual scans
        if scan_types.get('file_permissions', True):
            self.check_file_permissions()
        
        if scan_types.get('sensitive_data', True):
            self.scan_for_sensitive_data()
        
        if scan_types.get('ssl_certificates', True):
            self.check_ssl_certificates()
        
        if scan_types.get('network_security', True):
            self.check_network_security()
        
        if scan_types.get('dependency_scan', True):
            self.check_dependency_vulnerabilities()
        
        if scan_types.get('database_security', True):
            self.check_database_security()
        
        scan_end = datetime.now()
        scan_duration = (scan_end - scan_start).total_seconds()
        
        # Categorize vulnerabilities by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        for vuln in self.vulnerabilities:
            severity_counts[vuln['severity']] += 1
        
        # Generate security grade
        if self.security_score >= 90:
            grade = 'A'
        elif self.security_score >= 80:
            grade = 'B'
        elif self.security_score >= 70:
            grade = 'C'
        elif self.security_score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        results = {
            'scan_timestamp': scan_start.isoformat(),
            'scan_duration_seconds': round(scan_duration, 2),
            'security_score': round(self.security_score, 1),
            'security_grade': grade,
            'total_vulnerabilities': len(self.vulnerabilities),
            'severity_breakdown': severity_counts,
            'vulnerabilities': self.vulnerabilities,
            'scan_summary': {
                'critical_issues': severity_counts['critical'],
                'high_issues': severity_counts['high'],
                'needs_immediate_attention': severity_counts['critical'] + severity_counts['high'] > 0,
                'production_ready': severity_counts['critical'] == 0 and severity_counts['high'] <= 2
            }
        }
        
        # Save scan results
        results_file = f"security_scan_{scan_start.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('reports', exist_ok=True)
        
        with open(f"reports/{results_file}", 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Security scan completed. Score: {self.security_score}/100 (Grade: {grade})")
        self.logger.info(f"Found {len(self.vulnerabilities)} vulnerabilities: "
                        f"{severity_counts['critical']} critical, "
                        f"{severity_counts['high']} high, "
                        f"{severity_counts['medium']} medium, "
                        f"{severity_counts['low']} low")
        
        return results

    def generate_security_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable security report"""
        report = f"""
# AlphaBeta808 Trading Bot - Security Assessment Report

**Generated:** {results['scan_timestamp']}
**Scan Duration:** {results['scan_duration_seconds']} seconds
**Security Score:** {results['security_score']}/100 (Grade: {results['security_grade']})

## Executive Summary

Total vulnerabilities found: **{results['total_vulnerabilities']}**
- Critical: {results['severity_breakdown']['critical']}
- High: {results['severity_breakdown']['high']}  
- Medium: {results['severity_breakdown']['medium']}
- Low: {results['severity_breakdown']['low']}

Production readiness: **{'✅ Ready' if results['scan_summary']['production_ready'] else '❌ Not Ready'}**

## Critical Issues (Immediate Action Required)
"""
        
        critical_vulns = [v for v in results['vulnerabilities'] if v['severity'] == 'critical']
        if critical_vulns:
            for vuln in critical_vulns:
                report += f"""
### {vuln['description']}
- **Location:** {vuln['location']}
- **Category:** {vuln['category']}
- **Recommendation:** {vuln['recommendation']}
"""
        else:
            report += "\nNo critical issues found. ✅\n"
        
        report += f"""
## High Priority Issues
"""
        
        high_vulns = [v for v in results['vulnerabilities'] if v['severity'] == 'high']
        if high_vulns:
            for vuln in high_vulns:
                report += f"""
### {vuln['description']}
- **Location:** {vuln['location']}
- **Category:** {vuln['category']}
- **Recommendation:** {vuln['recommendation']}
"""
        else:
            report += "\nNo high priority issues found. ✅\n"
        
        report += f"""
## Recommendations

1. **Immediate Actions:**
   - Fix all critical vulnerabilities before production deployment
   - Review and secure file permissions
   - Change any default credentials

2. **Security Best Practices:**
   - Enable encryption for sensitive data
   - Implement proper access controls
   - Regular security updates
   - Monitor for suspicious activity

3. **Ongoing Security:**
   - Schedule regular security scans
   - Keep dependencies updated
   - Monitor security advisories
   - Implement backup and disaster recovery

## Next Steps

{'🚨 **CRITICAL:** Do not deploy to production until critical issues are resolved!' if results['severity_breakdown']['critical'] > 0 else ''}
{'⚠️ **WARNING:** Address high priority issues before production deployment.' if results['severity_breakdown']['high'] > 0 else ''}
{'✅ **GOOD:** No critical security issues found. Review medium/low issues for best practices.' if results['severity_breakdown']['critical'] == 0 and results['severity_breakdown']['high'] == 0 else ''}
"""
        
        return report

def main():
    """Main entry point for security scanner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaBeta808 Security Scanner')
    parser.add_argument('--config', default='security_config.json',
                       help='Configuration file path')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick scan (skip time-consuming checks)')
    
    args = parser.parse_args()
    
    scanner = SecurityScanner(args.config)
    
    if args.quick:
        # Quick scan - disable some time-consuming checks
        scanner.config['scan_types']['dependency_scan'] = False
        scanner.config['scan_types']['code_analysis'] = False
    
    results = scanner.run_comprehensive_scan()
    
    print(f"\n🔒 Security Scan Results")
    print(f"Score: {results['security_score']}/100 (Grade: {results['security_grade']})")
    print(f"Vulnerabilities: {results['total_vulnerabilities']} total")
    print(f"  Critical: {results['severity_breakdown']['critical']}")
    print(f"  High: {results['severity_breakdown']['high']}")
    print(f"  Medium: {results['severity_breakdown']['medium']}")
    print(f"  Low: {results['severity_breakdown']['low']}")
    
    if args.report:
        report = scanner.generate_security_report(results)
        report_file = f"reports/security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nDetailed report saved to: {report_file}")
    
    # Exit with error code if critical issues found
    if results['severity_breakdown']['critical'] > 0:
        print("\n🚨 CRITICAL SECURITY ISSUES FOUND - DO NOT DEPLOY TO PRODUCTION!")
        sys.exit(1)
    elif results['severity_breakdown']['high'] > 0:
        print("\n⚠️ High priority security issues found - review before production deployment")
        sys.exit(2)
    else:
        print("\n✅ No critical security issues found")
        sys.exit(0)

if __name__ == "__main__":
    main()
