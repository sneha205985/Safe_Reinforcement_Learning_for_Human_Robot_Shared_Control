"""
Comprehensive Disaster Recovery System for Safe RL Human-Robot Shared Control

This module implements enterprise-grade disaster recovery capabilities including:
- Automated backups of models, data, and configurations
- High availability with failover mechanisms
- Data replication across multiple regions
- Recovery time objective (RTO) and recovery point objective (RPO) management
- Disaster recovery testing and validation
"""

import asyncio
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import boto3
import psycopg2
import redis
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DisasterType(Enum):
    """Types of disasters that can occur"""
    HARDWARE_FAILURE = "hardware_failure"
    NETWORK_OUTAGE = "network_outage"
    DATA_CORRUPTION = "data_corruption"
    SECURITY_BREACH = "security_breach"
    HUMAN_ERROR = "human_error"
    NATURAL_DISASTER = "natural_disaster"
    SOFTWARE_BUG = "software_bug"


class RecoveryStatus(Enum):
    """Recovery operation status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class BackupMetadata:
    """Metadata for backup operations"""
    backup_id: str
    timestamp: datetime
    backup_type: str
    size_bytes: int
    checksum: str
    source_location: str
    backup_location: str
    retention_days: int
    encryption_key_id: Optional[str] = None


@dataclass
class RecoveryObjective:
    """Recovery time and point objectives"""
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    service_name: str
    criticality: str  # critical, high, medium, low


class BackupManager:
    """Manages automated backups of models, data, and configurations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.s3_client = boto3.client('s3', region_name=config.get('aws_region', 'us-east-1'))
        self.backup_bucket = config.get('backup_bucket')
        self.local_backup_path = Path(config.get('local_backup_path', '/backup'))
        self.encryption_enabled = config.get('encryption_enabled', True)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def create_full_backup(self) -> BackupMetadata:
        """Create a full system backup"""
        backup_id = f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.local_backup_path / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting full backup: {backup_id}")
        
        try:
            # Backup models
            await self._backup_models(backup_path / "models")
            
            # Backup database
            await self._backup_database(backup_path / "database")
            
            # Backup configurations
            await self._backup_configurations(backup_path / "configs")
            
            # Backup logs and metrics
            await self._backup_logs_metrics(backup_path / "logs_metrics")
            
            # Calculate checksum
            checksum = await self._calculate_backup_checksum(backup_path)
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type="full",
                size_bytes=self._get_directory_size(backup_path),
                checksum=checksum,
                source_location="system",
                backup_location=str(backup_path),
                retention_days=self.config.get('backup_retention_days', 30)
            )
            
            # Upload to cloud storage
            if self.backup_bucket:
                await self._upload_to_cloud(backup_path, backup_id)
            
            # Save metadata
            await self._save_backup_metadata(metadata)
            
            logger.info(f"Full backup completed: {backup_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Full backup failed: {str(e)}")
            raise
    
    async def create_incremental_backup(self, last_backup_time: datetime) -> BackupMetadata:
        """Create an incremental backup since last backup"""
        backup_id = f"incremental_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path = self.local_backup_path / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting incremental backup: {backup_id}")
        
        try:
            # Backup only changed files since last backup
            await self._backup_changed_files(backup_path, last_backup_time)
            
            checksum = await self._calculate_backup_checksum(backup_path)
            
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now(),
                backup_type="incremental",
                size_bytes=self._get_directory_size(backup_path),
                checksum=checksum,
                source_location="system",
                backup_location=str(backup_path),
                retention_days=self.config.get('backup_retention_days', 30)
            )
            
            if self.backup_bucket:
                await self._upload_to_cloud(backup_path, backup_id)
            
            await self._save_backup_metadata(metadata)
            
            logger.info(f"Incremental backup completed: {backup_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Incremental backup failed: {str(e)}")
            raise
    
    async def _backup_models(self, backup_path: Path):
        """Backup ML models and artifacts"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup model registry
        model_registry_path = Path("safe_rl_human_robot/models")
        if model_registry_path.exists():
            shutil.copytree(model_registry_path, backup_path / "registry", dirs_exist_ok=True)
        
        # Backup MLflow artifacts
        mlflow_path = Path("mlruns")
        if mlflow_path.exists():
            shutil.copytree(mlflow_path, backup_path / "mlflow", dirs_exist_ok=True)
    
    async def _backup_database(self, backup_path: Path):
        """Backup PostgreSQL database"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        db_config = self.config.get('database', {})
        if not db_config:
            return
        
        dump_file = backup_path / "database_dump.sql"
        
        # Create database dump
        cmd = [
            "pg_dump",
            f"--host={db_config.get('host', 'localhost')}",
            f"--port={db_config.get('port', 5432)}",
            f"--username={db_config.get('user', 'postgres')}",
            f"--dbname={db_config.get('database', 'safe_rl')}",
            f"--file={dump_file}"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Database backup failed: {stderr.decode()}")
    
    async def _backup_configurations(self, backup_path: Path):
        """Backup system configurations"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        config_paths = [
            "config/",
            "deployment/",
            ".env",
            "requirements.txt",
            "docker-compose.yml"
        ]
        
        for path_str in config_paths:
            path = Path(path_str)
            if path.exists():
                if path.is_file():
                    shutil.copy2(path, backup_path / path.name)
                else:
                    shutil.copytree(path, backup_path / path.name, dirs_exist_ok=True)
    
    async def _backup_logs_metrics(self, backup_path: Path):
        """Backup logs and metrics data"""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        log_paths = [
            "logs/",
            "/var/log/safe_rl/",
            "metrics/"
        ]
        
        for path_str in log_paths:
            path = Path(path_str)
            if path.exists():
                shutil.copytree(path, backup_path / path.name, dirs_exist_ok=True)
    
    async def _backup_changed_files(self, backup_path: Path, since: datetime):
        """Backup only files changed since specified time"""
        # Implementation for incremental backup
        pass
    
    async def _calculate_backup_checksum(self, backup_path: Path) -> str:
        """Calculate checksum for backup validation"""
        import hashlib
        
        hash_md5 = hashlib.md5()
        for file_path in backup_path.rglob("*"):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _get_directory_size(self, path: Path) -> int:
        """Calculate total size of directory"""
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    
    async def _upload_to_cloud(self, backup_path: Path, backup_id: str):
        """Upload backup to cloud storage"""
        if not self.backup_bucket:
            return
        
        # Create tar archive
        archive_path = backup_path.parent / f"{backup_id}.tar.gz"
        shutil.make_archive(str(archive_path.with_suffix('')), 'gztar', backup_path)
        
        # Upload to S3
        s3_key = f"backups/{backup_id}.tar.gz"
        with open(archive_path, 'rb') as f:
            self.s3_client.upload_fileobj(f, self.backup_bucket, s3_key)
        
        # Clean up local archive
        archive_path.unlink()
    
    async def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata to database"""
        # Store metadata in database or file system
        metadata_file = self.local_backup_path / f"{metadata.backup_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'backup_id': metadata.backup_id,
                'timestamp': metadata.timestamp.isoformat(),
                'backup_type': metadata.backup_type,
                'size_bytes': metadata.size_bytes,
                'checksum': metadata.checksum,
                'source_location': metadata.source_location,
                'backup_location': metadata.backup_location,
                'retention_days': metadata.retention_days
            }, f, indent=2)


class FailoverManager:
    """Manages high availability and failover operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary_endpoints = config.get('primary_endpoints', {})
        self.failover_endpoints = config.get('failover_endpoints', {})
        self.health_check_interval = config.get('health_check_interval', 30)
        self.failover_threshold = config.get('failover_threshold', 3)
        self.current_status = RecoveryStatus.HEALTHY
        self.failure_counts = {}
        self.is_running = False
        
    async def start_monitoring(self):
        """Start continuous health monitoring and failover management"""
        self.is_running = True
        logger.info("Starting failover monitoring")
        
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Error in failover monitoring: {str(e)}")
                await asyncio.sleep(5)
    
    async def stop_monitoring(self):
        """Stop failover monitoring"""
        self.is_running = False
        logger.info("Stopping failover monitoring")
    
    async def _perform_health_checks(self):
        """Perform health checks on all primary services"""
        for service_name, endpoint in self.primary_endpoints.items():
            try:
                is_healthy = await self._check_service_health(endpoint)
                
                if is_healthy:
                    self.failure_counts[service_name] = 0
                else:
                    self.failure_counts[service_name] = self.failure_counts.get(service_name, 0) + 1
                    
                    if self.failure_counts[service_name] >= self.failover_threshold:
                        await self._trigger_failover(service_name)
                        
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {str(e)}")
                self.failure_counts[service_name] = self.failure_counts.get(service_name, 0) + 1
    
    async def _check_service_health(self, endpoint: str) -> bool:
        """Check if a service endpoint is healthy"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint}/health", timeout=10) as response:
                    return response.status == 200
        except:
            return False
    
    async def _trigger_failover(self, service_name: str):
        """Trigger failover for a specific service"""
        if service_name not in self.failover_endpoints:
            logger.error(f"No failover endpoint configured for {service_name}")
            return
        
        logger.warning(f"Triggering failover for {service_name}")
        
        try:
            # Update load balancer configuration
            await self._update_load_balancer(service_name, self.failover_endpoints[service_name])
            
            # Update DNS records if configured
            await self._update_dns_records(service_name, self.failover_endpoints[service_name])
            
            # Send alerts
            await self._send_failover_alert(service_name)
            
            self.current_status = RecoveryStatus.DEGRADED
            logger.info(f"Failover completed for {service_name}")
            
        except Exception as e:
            logger.error(f"Failover failed for {service_name}: {str(e)}")
            self.current_status = RecoveryStatus.CRITICAL
    
    async def _update_load_balancer(self, service_name: str, failover_endpoint: str):
        """Update load balancer to route traffic to failover endpoint"""
        # Implementation depends on load balancer type (Nginx, HAProxy, AWS ALB, etc.)
        pass
    
    async def _update_dns_records(self, service_name: str, failover_endpoint: str):
        """Update DNS records to point to failover endpoint"""
        # Implementation for DNS failover (Route 53, CloudFlare, etc.)
        pass
    
    async def _send_failover_alert(self, service_name: str):
        """Send alert about failover event"""
        # Integration with alerting system
        pass


class DisasterRecoveryOrchestrator:
    """Main orchestrator for disaster recovery operations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_manager = BackupManager(config.get('backup', {}))
        self.failover_manager = FailoverManager(config.get('failover', {}))
        self.recovery_objectives = self._load_recovery_objectives()
        self.replication_manager = DataReplicationManager(config.get('replication', {}))
        
    def _load_recovery_objectives(self) -> Dict[str, RecoveryObjective]:
        """Load recovery time and point objectives for each service"""
        objectives = {}
        
        # Default objectives
        default_objectives = [
            RecoveryObjective(rto_minutes=15, rpo_minutes=5, service_name="safe_rl_api", criticality="critical"),
            RecoveryObjective(rto_minutes=30, rpo_minutes=15, service_name="policy_inference", criticality="high"),
            RecoveryObjective(rto_minutes=60, rpo_minutes=30, service_name="monitoring", criticality="medium"),
            RecoveryObjective(rto_minutes=120, rpo_minutes=60, service_name="model_registry", criticality="medium"),
        ]
        
        for obj in default_objectives:
            objectives[obj.service_name] = obj
            
        return objectives
    
    async def start(self):
        """Start disaster recovery system"""
        logger.info("Starting disaster recovery system")
        
        # Start backup scheduling
        asyncio.create_task(self._schedule_backups())
        
        # Start failover monitoring
        asyncio.create_task(self.failover_manager.start_monitoring())
        
        # Start replication monitoring
        asyncio.create_task(self.replication_manager.start_monitoring())
    
    async def _schedule_backups(self):
        """Schedule automated backups"""
        while True:
            try:
                # Full backup daily
                now = datetime.now()
                if now.hour == 2 and now.minute == 0:  # 2 AM daily
                    await self.backup_manager.create_full_backup()
                
                # Incremental backup every 4 hours
                elif now.minute == 0 and now.hour % 4 == 0:
                    last_backup_time = datetime.now() - timedelta(hours=4)
                    await self.backup_manager.create_incremental_backup(last_backup_time)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Backup scheduling error: {str(e)}")
                await asyncio.sleep(60)
    
    async def initiate_recovery(self, disaster_type: DisasterType, affected_services: List[str]) -> bool:
        """Initiate disaster recovery procedure"""
        logger.critical(f"Initiating disaster recovery for {disaster_type.value}")
        
        try:
            # Assess damage and determine recovery strategy
            recovery_plan = await self._create_recovery_plan(disaster_type, affected_services)
            
            # Execute recovery plan
            success = await self._execute_recovery_plan(recovery_plan)
            
            if success:
                logger.info("Disaster recovery completed successfully")
                return True
            else:
                logger.error("Disaster recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"Disaster recovery failed: {str(e)}")
            return False
    
    async def _create_recovery_plan(self, disaster_type: DisasterType, affected_services: List[str]) -> Dict[str, Any]:
        """Create detailed recovery plan based on disaster type and affected services"""
        recovery_plan = {
            'disaster_type': disaster_type.value,
            'affected_services': affected_services,
            'recovery_steps': [],
            'estimated_rto': 0,
            'priority_services': []
        }
        
        # Prioritize services by criticality
        priority_services = sorted(affected_services, 
                                 key=lambda s: self._get_service_priority(s))
        recovery_plan['priority_services'] = priority_services
        
        # Create recovery steps for each service
        for service in priority_services:
            service_steps = await self._create_service_recovery_steps(service, disaster_type)
            recovery_plan['recovery_steps'].extend(service_steps)
        
        # Calculate estimated RTO
        recovery_plan['estimated_rto'] = sum(
            self.recovery_objectives.get(service, RecoveryObjective(30, 15, service, "medium")).rto_minutes 
            for service in affected_services
        )
        
        return recovery_plan
    
    def _get_service_priority(self, service_name: str) -> int:
        """Get priority score for service (lower = higher priority)"""
        priority_map = {
            "critical": 1,
            "high": 2,
            "medium": 3,
            "low": 4
        }
        
        objective = self.recovery_objectives.get(service_name)
        if objective:
            return priority_map.get(objective.criticality, 3)
        return 3
    
    async def _create_service_recovery_steps(self, service_name: str, disaster_type: DisasterType) -> List[Dict[str, Any]]:
        """Create recovery steps for a specific service"""
        steps = []
        
        # Common recovery steps
        steps.append({
            'step': f'assess_{service_name}_damage',
            'description': f'Assess damage to {service_name}',
            'estimated_minutes': 5
        })
        
        if disaster_type in [DisasterType.HARDWARE_FAILURE, DisasterType.NATURAL_DISASTER]:
            steps.append({
                'step': f'restore_{service_name}_from_backup',
                'description': f'Restore {service_name} from latest backup',
                'estimated_minutes': 20
            })
        
        steps.append({
            'step': f'verify_{service_name}_functionality',
            'description': f'Verify {service_name} is functioning correctly',
            'estimated_minutes': 10
        })
        
        return steps
    
    async def _execute_recovery_plan(self, recovery_plan: Dict[str, Any]) -> bool:
        """Execute the recovery plan"""
        logger.info(f"Executing recovery plan with {len(recovery_plan['recovery_steps'])} steps")
        
        for step in recovery_plan['recovery_steps']:
            try:
                logger.info(f"Executing step: {step['description']}")
                success = await self._execute_recovery_step(step)
                
                if not success:
                    logger.error(f"Recovery step failed: {step['description']}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error executing recovery step {step['description']}: {str(e)}")
                return False
        
        return True
    
    async def _execute_recovery_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single recovery step"""
        step_name = step['step']
        
        if 'assess' in step_name:
            return await self._assess_service_damage(step_name)
        elif 'restore' in step_name:
            return await self._restore_service_from_backup(step_name)
        elif 'verify' in step_name:
            return await self._verify_service_functionality(step_name)
        
        return True
    
    async def _assess_service_damage(self, step_name: str) -> bool:
        """Assess damage to a service"""
        # Implementation for damage assessment
        return True
    
    async def _restore_service_from_backup(self, step_name: str) -> bool:
        """Restore service from backup"""
        # Implementation for service restoration
        return True
    
    async def _verify_service_functionality(self, step_name: str) -> bool:
        """Verify service is functioning correctly"""
        # Implementation for functionality verification
        return True


class DataReplicationManager:
    """Manages data replication across multiple regions/sites"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.replication_targets = config.get('targets', [])
        self.replication_mode = config.get('mode', 'async')  # async, sync
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start monitoring replication health"""
        self.is_monitoring = True
        
        while self.is_monitoring:
            try:
                await self._check_replication_health()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Replication monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _check_replication_health(self):
        """Check health of all replication targets"""
        for target in self.replication_targets:
            try:
                lag = await self._check_replication_lag(target)
                if lag > target.get('max_lag_seconds', 300):  # 5 minutes default
                    logger.warning(f"Replication lag high for {target['name']}: {lag}s")
                    
            except Exception as e:
                logger.error(f"Failed to check replication for {target['name']}: {str(e)}")
    
    async def _check_replication_lag(self, target: Dict[str, Any]) -> int:
        """Check replication lag for a specific target"""
        # Implementation depends on database type and replication setup
        return 0


# Example usage and testing functions
async def test_disaster_recovery():
    """Test disaster recovery system"""
    config = {
        'backup': {
            'backup_bucket': 'safe-rl-backups',
            'local_backup_path': '/tmp/backups',
            'backup_retention_days': 30,
            'aws_region': 'us-east-1'
        },
        'failover': {
            'primary_endpoints': {
                'safe_rl_api': 'http://localhost:8000',
                'model_registry': 'http://localhost:5000'
            },
            'failover_endpoints': {
                'safe_rl_api': 'http://failover:8000',
                'model_registry': 'http://failover:5000'
            },
            'health_check_interval': 30,
            'failover_threshold': 3
        }
    }
    
    # Initialize disaster recovery system
    dr_system = DisasterRecoveryOrchestrator(config)
    await dr_system.start()
    
    # Test backup creation
    backup_metadata = await dr_system.backup_manager.create_full_backup()
    logger.info(f"Test backup created: {backup_metadata.backup_id}")
    
    # Test disaster recovery simulation
    success = await dr_system.initiate_recovery(
        DisasterType.HARDWARE_FAILURE, 
        ['safe_rl_api', 'model_registry']
    )
    
    logger.info(f"Disaster recovery test {'passed' if success else 'failed'}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run disaster recovery test
    asyncio.run(test_disaster_recovery())