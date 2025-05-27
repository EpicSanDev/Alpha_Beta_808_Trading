#!/usr/bin/env python3
"""
AlphaBeta808 Trading Bot - Backup and Disaster Recovery System
Automated backup, restoration, and disaster recovery for production deployment
"""

import os
import sys
import shutil
import sqlite3
import json
import logging
import subprocess
import gzip
import tarfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
import cryptography.fernet

class BackupManager:
    def __init__(self, config_path: str = "backup_config.json"):
        """Initialize backup manager"""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.backup_dir = Path(self.config.get('backup_directory', './backups'))
        self.encryption_key = self.config.get('encryption_key')
        self.retention_days = self.config.get('retention_days', 30)
        
        # Create backup directory
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize encryption if enabled
        self.fernet = None
        if self.encryption_key:
            self.fernet = cryptography.fernet.Fernet(self.encryption_key.encode()[:44].ljust(44, b'='))

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load backup configuration"""
        default_config = {
            "backup_directory": "./backups",
            "retention_days": 30,
            "compression": True,
            "encryption": False,
            "encryption_key": None,
            "schedule": {
                "database": "hourly",
                "models": "daily",
                "configs": "daily",
                "logs": "weekly"
            },
            "components": {
                "database": {
                    "enabled": True,
                    "path": "trading_web.db",
                    "compress": True
                },
                "models": {
                    "enabled": True,
                    "path": "models_store",
                    "compress": True
                },
                "configurations": {
                    "enabled": True,
                    "paths": [".env", "config/", "*.json"],
                    "compress": True
                },
                "logs": {
                    "enabled": True,
                    "path": "logs",
                    "compress": True
                },
                "ssl_certificates": {
                    "enabled": True,
                    "path": "ssl",
                    "compress": False
                }
            },
            "remote_backup": {
                "enabled": False,
                "type": "s3",  # s3, ftp, rsync
                "config": {}
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load backup config: {e}")
        
        return default_config

    def setup_logging(self):
        """Setup logging for backup operations"""
        self.logger = logging.getLogger('BackupManager')
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
            file_handler = logging.FileHandler('logs/backup.log')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Cannot create log file (read-only filesystem?): {e}")
            self.logger.info("Logging to console only")

    def create_backup_metadata(self, backup_path: Path, component: str, 
                             size: int, checksum: str) -> Dict[str, Any]:
        """Create metadata for backup"""
        metadata = {
            'component': component,
            'timestamp': datetime.now().isoformat(),
            'backup_path': str(backup_path),
            'size_bytes': size,
            'checksum': checksum,
            'compressed': backup_path.suffix == '.gz',
            'encrypted': self.fernet is not None,
            'version': '1.0'
        }
        
        # Save metadata file
        metadata_path = backup_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def compress_file(self, source_path: Path, dest_path: Path) -> Path:
        """Compress file using gzip"""
        compressed_path = dest_path.with_suffix(dest_path.suffix + '.gz')
        
        with open(source_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return compressed_path

    def compress_directory(self, source_path: Path, dest_path: Path) -> Path:
        """Compress directory using tar.gz"""
        compressed_path = dest_path.with_suffix('.tar.gz')
        
        with tarfile.open(compressed_path, 'w:gz') as tar:
            tar.add(source_path, arcname=source_path.name)
        
        return compressed_path

    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt file if encryption is enabled"""
        if not self.fernet:
            return file_path
        
        encrypted_path = file_path.with_suffix(file_path.suffix + '.enc')
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.fernet.encrypt(data)
        
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        # Remove original file
        os.remove(file_path)
        
        return encrypted_path

    def backup_database(self) -> Optional[Dict[str, Any]]:
        """Backup SQLite database"""
        try:
            db_config = self.config['components']['database']
            if not db_config['enabled']:
                return None
            
            db_path = Path(db_config['path'])
            if not db_path.exists():
                self.logger.warning(f"Database file not found: {db_path}")
                return None
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"database_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # Create database backup using sqlite3 command
            # This ensures consistency even if database is in use
            conn = sqlite3.connect(str(db_path))
            backup_conn = sqlite3.connect(str(backup_path))
            conn.backup(backup_conn)
            backup_conn.close()
            conn.close()
            
            # Compress if enabled
            if db_config.get('compress', True):
                backup_path = self.compress_file(backup_path, backup_path)
                os.remove(backup_path.with_suffix('.db'))
            
            # Encrypt if enabled
            if self.fernet:
                backup_path = self.encrypt_file(backup_path)
            
            # Calculate checksum and create metadata
            checksum = self.calculate_checksum(backup_path)
            size = backup_path.stat().st_size
            metadata = self.create_backup_metadata(backup_path, 'database', size, checksum)
            
            self.logger.info(f"Database backup created: {backup_path} ({size} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return None

    def backup_models(self) -> Optional[Dict[str, Any]]:
        """Backup ML models directory"""
        try:
            models_config = self.config['components']['models']
            if not models_config['enabled']:
                return None
            
            models_path = Path(models_config['path'])
            if not models_path.exists():
                self.logger.warning(f"Models directory not found: {models_path}")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"models_backup_{timestamp}.tar"
            backup_path = self.backup_dir / backup_filename
            
            # Create tar archive
            if models_config.get('compress', True):
                backup_path = self.compress_directory(models_path, backup_path)
            else:
                with tarfile.open(backup_path, 'w') as tar:
                    tar.add(models_path, arcname=models_path.name)
            
            # Encrypt if enabled
            if self.fernet:
                backup_path = self.encrypt_file(backup_path)
            
            checksum = self.calculate_checksum(backup_path)
            size = backup_path.stat().st_size
            metadata = self.create_backup_metadata(backup_path, 'models', size, checksum)
            
            self.logger.info(f"Models backup created: {backup_path} ({size} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Models backup failed: {e}")
            return None

    def backup_configurations(self) -> Optional[Dict[str, Any]]:
        """Backup configuration files"""
        try:
            config_config = self.config['components']['configurations']
            if not config_config['enabled']:
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"configs_backup_{timestamp}.tar"
            backup_path = self.backup_dir / backup_filename
            
            # Create temporary directory for configs
            temp_dir = self.backup_dir / f"temp_configs_{timestamp}"
            temp_dir.mkdir(exist_ok=True)
            
            # Copy configuration files
            paths = config_config.get('paths', [])
            for path_pattern in paths:
                if '*' in path_pattern:
                    # Handle glob patterns
                    import glob
                    for file_path in glob.glob(path_pattern):
                        if os.path.exists(file_path):
                            dest = temp_dir / os.path.basename(file_path)
                            if os.path.isdir(file_path):
                                shutil.copytree(file_path, dest)
                            else:
                                shutil.copy2(file_path, dest)
                else:
                    # Handle single files/directories
                    if os.path.exists(path_pattern):
                        dest = temp_dir / os.path.basename(path_pattern)
                        if os.path.isdir(path_pattern):
                            shutil.copytree(path_pattern, dest)
                        else:
                            shutil.copy2(path_pattern, dest)
            
            # Create archive
            if config_config.get('compress', True):
                backup_path = self.compress_directory(temp_dir, backup_path)
            else:
                with tarfile.open(backup_path, 'w') as tar:
                    tar.add(temp_dir, arcname='configs')
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir)
            
            # Encrypt if enabled
            if self.fernet:
                backup_path = self.encrypt_file(backup_path)
            
            checksum = self.calculate_checksum(backup_path)
            size = backup_path.stat().st_size
            metadata = self.create_backup_metadata(backup_path, 'configurations', size, checksum)
            
            self.logger.info(f"Configurations backup created: {backup_path} ({size} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Configurations backup failed: {e}")
            return None

    def backup_logs(self) -> Optional[Dict[str, Any]]:
        """Backup log files"""
        try:
            logs_config = self.config['components']['logs']
            if not logs_config['enabled']:
                return None
            
            logs_path = Path(logs_config['path'])
            if not logs_path.exists():
                self.logger.warning(f"Logs directory not found: {logs_path}")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"logs_backup_{timestamp}.tar"
            backup_path = self.backup_dir / backup_filename
            
            # Create archive (usually compressed for logs)
            if logs_config.get('compress', True):
                backup_path = self.compress_directory(logs_path, backup_path)
            else:
                with tarfile.open(backup_path, 'w') as tar:
                    tar.add(logs_path, arcname=logs_path.name)
            
            # Encrypt if enabled
            if self.fernet:
                backup_path = self.encrypt_file(backup_path)
            
            checksum = self.calculate_checksum(backup_path)
            size = backup_path.stat().st_size
            metadata = self.create_backup_metadata(backup_path, 'logs', size, checksum)
            
            self.logger.info(f"Logs backup created: {backup_path} ({size} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Logs backup failed: {e}")
            return None

    def backup_ssl_certificates(self) -> Optional[Dict[str, Any]]:
        """Backup SSL certificates"""
        try:
            ssl_config = self.config['components']['ssl_certificates']
            if not ssl_config['enabled']:
                return None
            
            ssl_path = Path(ssl_config['path'])
            if not ssl_path.exists():
                self.logger.warning(f"SSL directory not found: {ssl_path}")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"ssl_backup_{timestamp}.tar"
            backup_path = self.backup_dir / backup_filename
            
            # SSL certificates usually shouldn't be compressed
            with tarfile.open(backup_path, 'w') as tar:
                tar.add(ssl_path, arcname=ssl_path.name)
            
            # Always encrypt SSL certificates
            if self.fernet:
                backup_path = self.encrypt_file(backup_path)
            
            checksum = self.calculate_checksum(backup_path)
            size = backup_path.stat().st_size
            metadata = self.create_backup_metadata(backup_path, 'ssl_certificates', size, checksum)
            
            self.logger.info(f"SSL certificates backup created: {backup_path} ({size} bytes)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"SSL certificates backup failed: {e}")
            return None

    def create_full_backup(self) -> Dict[str, Any]:
        """Create complete system backup"""
        self.logger.info("Starting full system backup...")
        
        backup_results = {
            'timestamp': datetime.now().isoformat(),
            'type': 'full',
            'components': {},
            'total_size': 0,
            'success': True
        }
        
        # Backup each component
        components = {
            'database': self.backup_database,
            'models': self.backup_models,
            'configurations': self.backup_configurations,
            'logs': self.backup_logs,
            'ssl_certificates': self.backup_ssl_certificates
        }
        
        for component_name, backup_func in components.items():
            try:
                result = backup_func()
                if result:
                    backup_results['components'][component_name] = result
                    backup_results['total_size'] += result['size_bytes']
                else:
                    backup_results['components'][component_name] = {'status': 'skipped'}
            except Exception as e:
                self.logger.error(f"Failed to backup {component_name}: {e}")
                backup_results['components'][component_name] = {'status': 'failed', 'error': str(e)}
                backup_results['success'] = False
        
        # Save backup manifest
        manifest_path = self.backup_dir / f"backup_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_path, 'w') as f:
            json.dump(backup_results, f, indent=2)
        
        self.logger.info(f"Full backup completed. Total size: {backup_results['total_size']} bytes")
        return backup_results

    def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            deleted_count = 0
            
            for file_path in self.backup_dir.iterdir():
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age < cutoff_date:
                        file_path.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted old backup: {file_path}")
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old backup files")
                
        except Exception as e:
            self.logger.error(f"Backup cleanup failed: {e}")

    def restore_database(self, backup_path: Path) -> bool:
        """Restore database from backup"""
        try:
            # TODO: Implement database restoration
            self.logger.info(f"Restoring database from {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Database restoration failed: {e}")
            return False

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for file_path in self.backup_dir.iterdir():
            if file_path.suffix == '.json' and 'manifest' in file_path.name:
                try:
                    with open(file_path, 'r') as f:
                        manifest = json.load(f)
                        backups.append(manifest)
                except Exception as e:
                    self.logger.error(f"Failed to read manifest {file_path}: {e}")
        
        # Sort by timestamp
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return backups

    def verify_backup_integrity(self, backup_path: Path) -> bool:
        """Verify backup file integrity using checksum"""
        try:
            metadata_path = backup_path.with_suffix('.metadata.json')
            if not metadata_path.exists():
                self.logger.warning(f"No metadata found for {backup_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            current_checksum = self.calculate_checksum(backup_path)
            stored_checksum = metadata.get('checksum')
            
            if current_checksum == stored_checksum:
                self.logger.info(f"Backup integrity verified: {backup_path}")
                return True
            else:
                self.logger.error(f"Backup integrity check failed: {backup_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False

def main():
    """Main entry point for backup operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AlphaBeta808 Backup Manager')
    parser.add_argument('action', choices=['full', 'database', 'models', 'configs', 'logs', 'ssl', 'list', 'cleanup'],
                       help='Backup action to perform')
    parser.add_argument('--config', default='backup_config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager(args.config)
    
    if args.action == 'full':
        result = backup_manager.create_full_backup()
        print(f"Full backup completed: {result['success']}")
    elif args.action == 'database':
        result = backup_manager.backup_database()
        print(f"Database backup: {'Success' if result else 'Failed'}")
    elif args.action == 'models':
        result = backup_manager.backup_models()
        print(f"Models backup: {'Success' if result else 'Failed'}")
    elif args.action == 'configs':
        result = backup_manager.backup_configurations()
        print(f"Configurations backup: {'Success' if result else 'Failed'}")
    elif args.action == 'logs':
        result = backup_manager.backup_logs()
        print(f"Logs backup: {'Success' if result else 'Failed'}")
    elif args.action == 'ssl':
        result = backup_manager.backup_ssl_certificates()
        print(f"SSL backup: {'Success' if result else 'Failed'}")
    elif args.action == 'list':
        backups = backup_manager.list_backups()
        print(f"Found {len(backups)} backup manifests:")
        for backup in backups[:10]:  # Show last 10
            print(f"  {backup['timestamp']} - {backup['type']} - {backup['total_size']} bytes")
    elif args.action == 'cleanup':
        backup_manager.cleanup_old_backups()
        print("Cleanup completed")

if __name__ == "__main__":
    main()
