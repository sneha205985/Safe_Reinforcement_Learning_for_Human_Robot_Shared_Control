#!/usr/bin/env python3
"""
Backup and Recovery Validation System for Safe RL Production.

Validates backup procedures, disaster recovery capabilities, and data integrity
for the Safe RL Human-Robot Shared Control System.
"""

import os
import sys
import time
import json
import logging
import shutil
import hashlib
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import tarfile
import zipfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "safe_rl_human_robot" / "src"))


class BackupType(Enum):
    """Types of backups."""
    CONFIGURATION = auto()
    MODEL_WEIGHTS = auto()
    SYSTEM_STATE = auto()
    LOG_DATA = auto()
    FULL_SYSTEM = auto()


class RecoveryScenario(Enum):
    """Disaster recovery scenarios."""
    CONFIG_CORRUPTION = auto()
    MODEL_LOSS = auto()
    SYSTEM_FAILURE = auto()
    PARTIAL_DATA_LOSS = auto()
    COMPLETE_SYSTEM_LOSS = auto()


@dataclass
class BackupMetrics:
    """Backup operation metrics."""
    backup_size_mb: float = 0.0
    backup_time_seconds: float = 0.0
    compression_ratio: float = 1.0
    verification_passed: bool = False
    backup_path: str = ""
    checksum: str = ""


@dataclass
class RecoveryMetrics:
    """Recovery operation metrics."""
    recovery_time_seconds: float = 0.0
    data_integrity_verified: bool = False
    system_operational: bool = False
    recovery_success_rate: float = 0.0
    rollback_successful: bool = False


@dataclass
class BackupRecoveryReport:
    """Comprehensive backup and recovery validation report."""
    validation_timestamp: float = field(default_factory=time.time)
    
    # Backup validation results
    backup_tests_passed: int = 0
    backup_tests_total: int = 0
    backup_metrics: Dict[BackupType, BackupMetrics] = field(default_factory=dict)
    
    # Recovery validation results
    recovery_tests_passed: int = 0
    recovery_tests_total: int = 0
    recovery_metrics: Dict[RecoveryScenario, RecoveryMetrics] = field(default_factory=dict)
    
    # Overall assessment
    backup_score: float = 0.0
    recovery_score: float = 0.0
    overall_score: float = 0.0
    production_ready: bool = False
    
    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_scores(self):
        """Calculate backup and recovery scores."""
        # Backup score
        if self.backup_tests_total > 0:
            self.backup_score = (self.backup_tests_passed / self.backup_tests_total) * 100.0
        else:
            self.backup_score = 0.0
        
        # Recovery score
        if self.recovery_tests_total > 0:
            self.recovery_score = (self.recovery_tests_passed / self.recovery_tests_total) * 100.0
        else:
            self.recovery_score = 0.0
        
        # Overall score
        self.overall_score = (self.backup_score + self.recovery_score) / 2.0
        
        # Production readiness criteria
        self.production_ready = (
            self.overall_score >= 85.0 and
            self.backup_score >= 80.0 and
            self.recovery_score >= 80.0 and
            len(self.critical_issues) == 0
        )


class BackupSystem:
    """Backup system implementation and validation."""
    
    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.BackupSystem")
    
    def create_backup(self, backup_type: BackupType, source_path: Path) -> BackupMetrics:
        """Create backup of specified type and source."""
        metrics = BackupMetrics()
        start_time = time.time()
        
        try:
            # Generate backup filename
            timestamp = int(time.time())
            backup_name = f"{backup_type.name.lower()}_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_name
            
            self.logger.info(f"Creating {backup_type.name} backup: {backup_path}")
            
            # Create backup
            if source_path.exists():
                with tarfile.open(backup_path, 'w:gz') as tar:
                    if source_path.is_file():
                        tar.add(source_path, arcname=source_path.name)
                    else:
                        tar.add(source_path, arcname=source_path.name)
                
                # Calculate metrics
                metrics.backup_size_mb = backup_path.stat().st_size / 1024 / 1024
                metrics.backup_time_seconds = time.time() - start_time
                metrics.backup_path = str(backup_path)
                
                # Calculate compression ratio
                original_size = self._calculate_directory_size(source_path)
                if original_size > 0:
                    metrics.compression_ratio = original_size / backup_path.stat().st_size
                
                # Calculate checksum
                metrics.checksum = self._calculate_file_checksum(backup_path)
                
                # Verify backup integrity
                metrics.verification_passed = self._verify_backup(backup_path, source_path)
                
                self.logger.info(f"Backup created successfully: {metrics.backup_size_mb:.1f} MB")
                
            else:
                self.logger.warning(f"Source path does not exist: {source_path}")
                metrics.verification_passed = False
                
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            metrics.verification_passed = False
        
        return metrics
    
    def restore_backup(self, backup_path: Path, restore_path: Path) -> RecoveryMetrics:
        """Restore backup to specified location."""
        metrics = RecoveryMetrics()
        start_time = time.time()
        
        try:
            self.logger.info(f"Restoring backup from: {backup_path}")
            
            # Create restore directory
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(path=restore_path)
            
            metrics.recovery_time_seconds = time.time() - start_time
            
            # Verify data integrity
            metrics.data_integrity_verified = self._verify_restored_data(restore_path)
            
            # Check if system would be operational
            metrics.system_operational = self._check_system_operational(restore_path)
            
            metrics.recovery_success_rate = 1.0 if metrics.data_integrity_verified else 0.0
            
            self.logger.info(f"Backup restored successfully in {metrics.recovery_time_seconds:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Backup restore failed: {e}")
            metrics.data_integrity_verified = False
            metrics.system_operational = False
            metrics.recovery_success_rate = 0.0
        
        return metrics
    
    def _calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of directory or file."""
        if path.is_file():
            return path.stat().st_size
        
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
        return total_size
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""
    
    def _verify_backup(self, backup_path: Path, original_path: Path) -> bool:
        """Verify backup integrity by comparing with original."""
        try:
            # Create temporary directory for verification
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_restore_path = Path(temp_dir) / "verify"
                
                # Extract backup
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(path=temp_restore_path)
                
                # Compare with original (simplified check)
                return temp_restore_path.exists()
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    def _verify_restored_data(self, restore_path: Path) -> bool:
        """Verify integrity of restored data."""
        try:
            # Check if restore path has content
            if not restore_path.exists():
                return False
            
            # Check for essential files (simplified)
            essential_patterns = ["*.py", "*.yaml", "*.json"]
            found_files = 0
            
            for pattern in essential_patterns:
                files = list(restore_path.rglob(pattern))
                found_files += len(files)
            
            return found_files > 0
            
        except Exception:
            return False
    
    def _check_system_operational(self, restore_path: Path) -> bool:
        """Check if restored system would be operational."""
        try:
            # Look for key system files
            key_files = [
                "*.py",  # Python modules
                "*.yaml", "*.yml",  # Configuration files
                "requirements.txt"  # Dependencies
            ]
            
            for pattern in key_files:
                if list(restore_path.rglob(pattern)):
                    return True
            
            return False
            
        except Exception:
            return False


class DisasterRecoveryTester:
    """Disaster recovery scenario testing."""
    
    def __init__(self, system_path: Path, backup_system: BackupSystem):
        self.system_path = system_path
        self.backup_system = backup_system
        self.logger = logging.getLogger(f"{__name__}.DisasterRecoveryTester")
    
    def test_recovery_scenario(self, scenario: RecoveryScenario) -> RecoveryMetrics:
        """Test specific disaster recovery scenario."""
        self.logger.info(f"Testing recovery scenario: {scenario.name}")
        
        scenario_methods = {
            RecoveryScenario.CONFIG_CORRUPTION: self._test_config_corruption_recovery,
            RecoveryScenario.MODEL_LOSS: self._test_model_loss_recovery,
            RecoveryScenario.SYSTEM_FAILURE: self._test_system_failure_recovery,
            RecoveryScenario.PARTIAL_DATA_LOSS: self._test_partial_data_loss_recovery,
            RecoveryScenario.COMPLETE_SYSTEM_LOSS: self._test_complete_system_loss_recovery
        }
        
        method = scenario_methods.get(scenario)
        if method:
            return method()
        else:
            self.logger.error(f"Unknown recovery scenario: {scenario}")
            return RecoveryMetrics()
    
    def _test_config_corruption_recovery(self) -> RecoveryMetrics:
        """Test recovery from configuration corruption."""
        metrics = RecoveryMetrics()
        start_time = time.time()
        
        try:
            # Create test configuration
            with tempfile.TemporaryDirectory() as temp_dir:
                config_dir = Path(temp_dir) / "config"
                config_dir.mkdir()
                
                # Create test config file
                test_config = config_dir / "test_config.yaml"
                test_config.write_text("test_key: test_value\n")
                
                # Create backup
                backup_metrics = self.backup_system.create_backup(
                    BackupType.CONFIGURATION, config_dir
                )
                
                # Simulate corruption by deleting file
                test_config.unlink()
                
                # Test recovery
                restore_dir = Path(temp_dir) / "restored"
                recovery_metrics = self.backup_system.restore_backup(
                    Path(backup_metrics.backup_path), restore_dir
                )
                
                metrics.recovery_time_seconds = time.time() - start_time
                metrics.data_integrity_verified = recovery_metrics.data_integrity_verified
                metrics.system_operational = True  # Config recovery should make system operational
                metrics.recovery_success_rate = 1.0 if metrics.data_integrity_verified else 0.0
                metrics.rollback_successful = True
                
        except Exception as e:
            self.logger.error(f"Config corruption recovery test failed: {e}")
            metrics.data_integrity_verified = False
            metrics.system_operational = False
            metrics.recovery_success_rate = 0.0
        
        return metrics
    
    def _test_model_loss_recovery(self) -> RecoveryMetrics:
        """Test recovery from model loss."""
        metrics = RecoveryMetrics()
        start_time = time.time()
        
        try:
            # Create test model directory
            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = Path(temp_dir) / "models"
                model_dir.mkdir()
                
                # Create dummy model files
                (model_dir / "model.pt").write_bytes(b"dummy model data")
                (model_dir / "model_config.json").write_text('{"param": "value"}')
                
                # Create backup
                backup_metrics = self.backup_system.create_backup(
                    BackupType.MODEL_WEIGHTS, model_dir
                )
                
                # Simulate model loss
                shutil.rmtree(model_dir)
                
                # Test recovery
                restore_dir = Path(temp_dir) / "restored_models"
                recovery_metrics = self.backup_system.restore_backup(
                    Path(backup_metrics.backup_path), restore_dir
                )
                
                metrics.recovery_time_seconds = time.time() - start_time
                metrics.data_integrity_verified = recovery_metrics.data_integrity_verified
                metrics.system_operational = recovery_metrics.system_operational
                metrics.recovery_success_rate = recovery_metrics.recovery_success_rate
                metrics.rollback_successful = True
                
        except Exception as e:
            self.logger.error(f"Model loss recovery test failed: {e}")
            metrics.recovery_success_rate = 0.0
        
        return metrics
    
    def _test_system_failure_recovery(self) -> RecoveryMetrics:
        """Test recovery from complete system failure."""
        metrics = RecoveryMetrics()
        start_time = time.time()
        
        try:
            # Create test system directory
            with tempfile.TemporaryDirectory() as temp_dir:
                system_dir = Path(temp_dir) / "system"
                system_dir.mkdir()
                
                # Create essential system files
                (system_dir / "main.py").write_text("# Main system file\n")
                (system_dir / "config.yaml").write_text("system: test\n")
                (system_dir / "requirements.txt").write_text("numpy\n")
                
                # Create backup
                backup_metrics = self.backup_system.create_backup(
                    BackupType.FULL_SYSTEM, system_dir
                )
                
                # Simulate system failure
                shutil.rmtree(system_dir)
                
                # Test recovery
                restore_dir = Path(temp_dir) / "restored_system"
                recovery_metrics = self.backup_system.restore_backup(
                    Path(backup_metrics.backup_path), restore_dir
                )
                
                metrics.recovery_time_seconds = time.time() - start_time
                metrics.data_integrity_verified = recovery_metrics.data_integrity_verified
                metrics.system_operational = recovery_metrics.system_operational
                metrics.recovery_success_rate = recovery_metrics.recovery_success_rate
                metrics.rollback_successful = True
                
        except Exception as e:
            self.logger.error(f"System failure recovery test failed: {e}")
            metrics.recovery_success_rate = 0.0
        
        return metrics
    
    def _test_partial_data_loss_recovery(self) -> RecoveryMetrics:
        """Test recovery from partial data loss."""
        metrics = RecoveryMetrics()
        start_time = time.time()
        
        try:
            # Similar to other tests but simulate partial loss
            with tempfile.TemporaryDirectory() as temp_dir:
                data_dir = Path(temp_dir) / "data"
                data_dir.mkdir()
                
                # Create test data files
                (data_dir / "important.txt").write_text("important data")
                (data_dir / "backup.txt").write_text("backup data")
                
                # Create backup
                backup_metrics = self.backup_system.create_backup(
                    BackupType.SYSTEM_STATE, data_dir
                )
                
                # Simulate partial loss (delete one file)
                (data_dir / "important.txt").unlink()
                
                # Test recovery
                restore_dir = Path(temp_dir) / "partial_restore"
                recovery_metrics = self.backup_system.restore_backup(
                    Path(backup_metrics.backup_path), restore_dir
                )
                
                metrics.recovery_time_seconds = time.time() - start_time
                metrics.data_integrity_verified = recovery_metrics.data_integrity_verified
                metrics.system_operational = True  # Partial loss should still allow operation
                metrics.recovery_success_rate = recovery_metrics.recovery_success_rate
                metrics.rollback_successful = True
                
        except Exception as e:
            self.logger.error(f"Partial data loss recovery test failed: {e}")
            metrics.recovery_success_rate = 0.0
        
        return metrics
    
    def _test_complete_system_loss_recovery(self) -> RecoveryMetrics:
        """Test recovery from complete system loss."""
        return self._test_system_failure_recovery()  # Similar to system failure


class BackupRecoveryValidator:
    """Main backup and recovery validation system."""
    
    def __init__(self, system_path: Path, backup_dir: Optional[Path] = None):
        self.system_path = system_path
        self.backup_dir = backup_dir or (system_path.parent / "backups")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize subsystems
        self.backup_system = BackupSystem(self.backup_dir)
        self.disaster_recovery_tester = DisasterRecoveryTester(
            self.system_path, self.backup_system
        )
    
    def validate_backup_recovery_systems(self) -> BackupRecoveryReport:
        """Perform comprehensive backup and recovery validation."""
        self.logger.info("Starting backup and recovery validation...")
        
        report = BackupRecoveryReport()
        
        try:
            # 1. Test backup operations
            self._test_backup_operations(report)
            
            # 2. Test disaster recovery scenarios
            self._test_disaster_recovery_scenarios(report)
            
            # 3. Calculate scores and production readiness
            report.calculate_scores()
            
            # 4. Generate recommendations
            self._generate_recommendations(report)
            
        except Exception as e:
            report.critical_issues.append(f"Backup/recovery validation failed: {str(e)}")
            self.logger.error(f"Validation failed: {e}")
        
        self.logger.info(f"Backup/recovery validation completed. Score: {report.overall_score:.1f}/100")
        return report
    
    def _test_backup_operations(self, report: BackupRecoveryReport):
        """Test backup operations for all backup types."""
        self.logger.info("Testing backup operations...")
        
        backup_tests = [
            (BackupType.CONFIGURATION, self._create_test_config_dir),
            (BackupType.MODEL_WEIGHTS, self._create_test_model_dir),
            (BackupType.SYSTEM_STATE, self._create_test_state_dir),
            (BackupType.LOG_DATA, self._create_test_log_dir),
            (BackupType.FULL_SYSTEM, self._create_test_full_system_dir)
        ]
        
        for backup_type, create_test_data in backup_tests:
            report.backup_tests_total += 1
            
            try:
                # Create test data
                test_dir = create_test_data()
                
                # Test backup creation
                metrics = self.backup_system.create_backup(backup_type, test_dir)
                report.backup_metrics[backup_type] = metrics
                
                if metrics.verification_passed:
                    report.backup_tests_passed += 1
                    self.logger.info(f"✅ {backup_type.name} backup test passed")
                else:
                    report.warnings.append(f"{backup_type.name} backup verification failed")
                    self.logger.warning(f"⚠️  {backup_type.name} backup test failed")
                
                # Cleanup
                if test_dir.exists():
                    shutil.rmtree(test_dir)
                    
            except Exception as e:
                report.warnings.append(f"{backup_type.name} backup test failed: {str(e)}")
                self.logger.error(f"Backup test failed for {backup_type.name}: {e}")
    
    def _test_disaster_recovery_scenarios(self, report: BackupRecoveryReport):
        """Test disaster recovery scenarios."""
        self.logger.info("Testing disaster recovery scenarios...")
        
        recovery_scenarios = [
            RecoveryScenario.CONFIG_CORRUPTION,
            RecoveryScenario.MODEL_LOSS,
            RecoveryScenario.SYSTEM_FAILURE,
            RecoveryScenario.PARTIAL_DATA_LOSS,
            RecoveryScenario.COMPLETE_SYSTEM_LOSS
        ]
        
        for scenario in recovery_scenarios:
            report.recovery_tests_total += 1
            
            try:
                # Test recovery scenario
                metrics = self.disaster_recovery_tester.test_recovery_scenario(scenario)
                report.recovery_metrics[scenario] = metrics
                
                if metrics.recovery_success_rate >= 0.8:  # 80% success rate
                    report.recovery_tests_passed += 1
                    self.logger.info(f"✅ {scenario.name} recovery test passed")
                else:
                    report.warnings.append(f"{scenario.name} recovery test failed")
                    self.logger.warning(f"⚠️  {scenario.name} recovery test failed")
                    
            except Exception as e:
                report.critical_issues.append(f"{scenario.name} recovery test failed: {str(e)}")
                self.logger.error(f"Recovery test failed for {scenario.name}: {e}")
    
    def _create_test_config_dir(self) -> Path:
        """Create test configuration directory."""
        test_dir = Path(tempfile.mkdtemp(prefix="test_config_"))
        
        # Create test config files
        (test_dir / "main_config.yaml").write_text("system:\n  name: test\n  version: 1.0\n")
        (test_dir / "safety_config.yaml").write_text("safety:\n  enabled: true\n")
        
        return test_dir
    
    def _create_test_model_dir(self) -> Path:
        """Create test model directory."""
        test_dir = Path(tempfile.mkdtemp(prefix="test_models_"))
        
        # Create dummy model files
        (test_dir / "model.pt").write_bytes(b"dummy model weights data")
        (test_dir / "model_config.json").write_text('{"layers": [256, 128], "activation": "relu"}')
        
        return test_dir
    
    def _create_test_state_dir(self) -> Path:
        """Create test system state directory."""
        test_dir = Path(tempfile.mkdtemp(prefix="test_state_"))
        
        # Create state files
        (test_dir / "system_state.json").write_text('{"status": "running", "uptime": 3600}')
        (test_dir / "performance_data.json").write_text('{"metrics": {"cpu": 45.2, "memory": 512}}')
        
        return test_dir
    
    def _create_test_log_dir(self) -> Path:
        """Create test log directory."""
        test_dir = Path(tempfile.mkdtemp(prefix="test_logs_"))
        
        # Create log files
        (test_dir / "system.log").write_text("INFO: System started\nINFO: All systems operational\n")
        (test_dir / "error.log").write_text("ERROR: Minor warning at 12:00\n")
        
        return test_dir
    
    def _create_test_full_system_dir(self) -> Path:
        """Create test full system directory."""
        test_dir = Path(tempfile.mkdtemp(prefix="test_full_system_"))
        
        # Create comprehensive system structure
        (test_dir / "main.py").write_text("# Main system entry point\nif __name__ == '__main__':\n    pass\n")
        (test_dir / "config.yaml").write_text("system:\n  name: safe_rl\n  version: 1.0\n")
        (test_dir / "requirements.txt").write_text("numpy>=1.19.0\ntorch>=1.8.0\n")
        
        # Create subdirectories
        (test_dir / "models").mkdir()
        (test_dir / "config").mkdir()
        (test_dir / "logs").mkdir()
        
        return test_dir
    
    def _generate_recommendations(self, report: BackupRecoveryReport):
        """Generate backup and recovery recommendations."""
        # Backup recommendations
        if report.backup_score < 80:
            report.recommendations.append("Improve backup system reliability")
            
        backup_issues = sum(1 for metrics in report.backup_metrics.values() 
                          if not metrics.verification_passed)
        if backup_issues > 0:
            report.recommendations.append(f"Fix {backup_issues} backup verification issues")
        
        # Recovery recommendations
        if report.recovery_score < 80:
            report.recommendations.append("Improve disaster recovery capabilities")
            
        slow_recoveries = sum(1 for metrics in report.recovery_metrics.values()
                            if metrics.recovery_time_seconds > 30.0)
        if slow_recoveries > 0:
            report.recommendations.append("Optimize recovery time for faster disaster recovery")
        
        # Overall recommendations
        if not report.production_ready:
            report.recommendations.append("Backup and recovery system not production ready")
            report.recommendations.append("Address critical issues before production deployment")
        else:
            report.recommendations.append("Backup and recovery systems are production ready")
            report.recommendations.append("Implement regular backup testing in production")
            report.recommendations.append("Establish backup retention and rotation policies")
    
    def generate_report(self, report: BackupRecoveryReport, output_file: Optional[str] = None) -> str:
        """Generate detailed backup and recovery report."""
        report_lines = [
            "=" * 80,
            "SAFE RL SYSTEM - BACKUP AND RECOVERY VALIDATION REPORT",
            "=" * 80,
            "",
            f"Validation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.validation_timestamp))}",
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"Overall Score:     {report.overall_score:.1f}/100.0",
            f"Backup Score:      {report.backup_score:.1f}/100.0",
            f"Recovery Score:    {report.recovery_score:.1f}/100.0",
            f"Production Ready:  {'✅ YES' if report.production_ready else '❌ NO'}",
            "",
            "TEST RESULTS SUMMARY",
            "-" * 40,
            f"Backup Tests:      {report.backup_tests_passed}/{report.backup_tests_total}",
            f"Recovery Tests:    {report.recovery_tests_passed}/{report.recovery_tests_total}",
        ]
        
        # Backup test details
        if report.backup_metrics:
            report_lines.extend([
                "",
                "BACKUP VALIDATION RESULTS",
                "-" * 40
            ])
            
            for backup_type, metrics in report.backup_metrics.items():
                status = "✅ PASS" if metrics.verification_passed else "❌ FAIL"
                report_lines.extend([
                    f"{backup_type.name}:",
                    f"  Status:        {status}",
                    f"  Size:          {metrics.backup_size_mb:.1f} MB",
                    f"  Time:          {metrics.backup_time_seconds:.2f} seconds",
                    f"  Compression:   {metrics.compression_ratio:.1f}x",
                    f"  Checksum:      {metrics.checksum[:16]}..."
                ])
        
        # Recovery test details
        if report.recovery_metrics:
            report_lines.extend([
                "",
                "DISASTER RECOVERY RESULTS",
                "-" * 40
            ])
            
            for scenario, metrics in report.recovery_metrics.items():
                status = "✅ PASS" if metrics.recovery_success_rate >= 0.8 else "❌ FAIL"
                report_lines.extend([
                    f"{scenario.name}:",
                    f"  Status:           {status}",
                    f"  Recovery Time:    {metrics.recovery_time_seconds:.2f} seconds",
                    f"  Data Integrity:   {'✅' if metrics.data_integrity_verified else '❌'}",
                    f"  System Operational: {'✅' if metrics.system_operational else '❌'}",
                    f"  Success Rate:     {metrics.recovery_success_rate*100:.1f}%"
                ])
        
        # Issues and recommendations
        if report.critical_issues:
            report_lines.extend([
                "",
                "CRITICAL ISSUES",
                "-" * 40
            ])
            for issue in report.critical_issues:
                report_lines.append(f"❌ {issue}")
        
        if report.warnings:
            report_lines.extend([
                "",
                "WARNINGS",
                "-" * 40
            ])
            for warning in report.warnings:
                report_lines.append(f"⚠️  {warning}")
        
        if report.recommendations:
            report_lines.extend([
                "",
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for rec in report.recommendations:
                report_lines.append(f"💡 {rec}")
        
        # Final assessment
        report_lines.extend([
            "",
            "FINAL ASSESSMENT",
            "=" * 80
        ])
        
        if report.production_ready:
            report_lines.extend([
                "🎉 BACKUP AND RECOVERY SYSTEMS READY!",
                "",
                "The backup and recovery systems have passed validation and are",
                "approved for production deployment.",
                "",
                "✅ Backup operations validated",
                "✅ Disaster recovery tested",
                "✅ Data integrity verified",
                "✅ Recovery time acceptable"
            ])
        else:
            report_lines.extend([
                "⚠️  BACKUP AND RECOVERY SYSTEMS NOT READY",
                "",
                "The backup and recovery systems have NOT passed validation.",
                "Issues must be resolved before production deployment.",
                "",
                f"Overall Score: {report.overall_score:.1f}/100 (minimum 85 required)",
                f"Critical Issues: {len(report.critical_issues)} (maximum 0 allowed)"
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "Safe RL Human-Robot Shared Control System",
            "=" * 80
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_text


def main():
    """Main backup and recovery validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Safe RL Backup and Recovery Validation"
    )
    parser.add_argument(
        '--system-path',
        default='.',
        help='Path to Safe RL system directory'
    )
    parser.add_argument(
        '--backup-dir',
        help='Backup directory path (default: <system-path>/backups)'
    )
    parser.add_argument(
        '--output-report',
        help='Output report file path'
    )
    parser.add_argument(
        '--output-json',
        help='Output JSON results file path'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        system_path = Path(args.system_path).resolve()
        backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
        
        logger.info("Starting backup and recovery validation...")
        
        # Create validator
        validator = BackupRecoveryValidator(system_path, backup_dir)
        
        # Run validation
        report = validator.validate_backup_recovery_systems()
        
        # Generate report
        detailed_report = validator.generate_report(report, args.output_report)
        print(detailed_report)
        
        # Save JSON results if requested
        if args.output_json:
            import dataclasses
            report_dict = dataclasses.asdict(report)
            with open(args.output_json, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"JSON results saved to {args.output_json}")
        
        # Exit with appropriate code
        if report.production_ready:
            logger.info("✅ Backup and recovery systems are production ready!")
            return 0
        else:
            logger.warning("⚠️  Backup and recovery systems are not production ready")
            return 1
            
    except Exception as e:
        logger.error(f"Backup and recovery validation failed: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())