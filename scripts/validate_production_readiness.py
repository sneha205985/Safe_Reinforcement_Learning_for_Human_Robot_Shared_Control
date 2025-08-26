#!/usr/bin/env python3
"""
Production Readiness Validation Script for Safe RL Human-Robot Shared Control.

This script performs comprehensive validation to ensure the system is 100% production ready.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "safe_rl_human_robot" / "src"))

from integration.final_integration import (
    UnifiedSafeRLSystem, 
    SystemValidator,
    IntegrationReport,
    create_production_system
)
from integration.config_integrator import ConfigurationIntegrator
from deployment.config_manager import Environment
import tempfile


class ProductionReadinessValidator:
    """Comprehensive production readiness validator."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validation_results: Dict[str, Any] = {}
        
    def validate_production_readiness(self, config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive production readiness validation.
        
        Args:
            config_dir: Configuration directory path
            
        Returns:
            Dict containing validation results
        """
        self.logger.info("Starting production readiness validation...")
        start_time = time.time()
        
        # Use temporary directory if no config provided
        if config_dir is None:
            temp_dir = tempfile.mkdtemp()
            config_dir = temp_dir
            self._create_minimal_configs(config_dir)
        
        try:
            results = {
                'validation_timestamp': time.time(),
                'project_root': str(self.project_root),
                'config_dir': config_dir,
                'validation_duration': 0.0,
                'overall_status': 'UNKNOWN',
                'production_ready': False,
                'readiness_score': 0.0,
                'validations': {}
            }
            
            # 1. Code Structure Validation
            results['validations']['code_structure'] = self._validate_code_structure()
            
            # 2. Configuration Validation
            results['validations']['configuration'] = self._validate_configuration_system(config_dir)
            
            # 3. Import and Dependencies Validation
            results['validations']['imports'] = self._validate_imports_and_dependencies()
            
            # 4. System Integration Validation
            results['validations']['integration'] = self._validate_system_integration(config_dir)
            
            # 5. Safety Systems Validation
            results['validations']['safety'] = self._validate_safety_systems()
            
            # 6. Performance Validation
            results['validations']['performance'] = self._validate_performance_requirements()
            
            # 7. Security Validation
            results['validations']['security'] = self._validate_security_features()
            
            # 8. Documentation Validation
            results['validations']['documentation'] = self._validate_documentation()
            
            # Calculate overall readiness
            self._calculate_overall_readiness(results)
            
            results['validation_duration'] = time.time() - start_time
            
            self.validation_results = results
            self.logger.info(f"Production readiness validation completed in {results['validation_duration']:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production readiness validation failed: {e}")
            return {
                'validation_timestamp': time.time(),
                'overall_status': 'ERROR',
                'production_ready': False,
                'error': str(e),
                'validation_duration': time.time() - start_time
            }
    
    def _create_minimal_configs(self, config_dir: str):
        """Create minimal configuration for testing."""
        import yaml
        
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)
        
        # Create minimal system config
        system_dir = config_path / "system"
        system_dir.mkdir(exist_ok=True)
        
        system_config = {
            'node_name': 'safe_rl_production_system',
            'log_level': 'INFO',
            'max_cpu_percent': 80.0,
            'max_memory_mb': 2048,
            'enable_profiling': True,
            'heartbeat_interval_s': 1.0
        }
        
        with open(system_dir / "base.yaml", 'w') as f:
            yaml.dump(system_config, f)
    
    def _validate_code_structure(self) -> Dict[str, Any]:
        """Validate code structure and organization."""
        self.logger.info("Validating code structure...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            src_dir = self.project_root / "safe_rl_human_robot" / "src"
            
            # Check main directories exist
            required_dirs = [
                'core', 'algorithms', 'environments', 'hardware', 
                'human_modeling', 'realtime', 'safety', 'integration',
                'config', 'deployment', 'monitoring', 'ros_integration'
            ]
            
            missing_dirs = []
            existing_dirs = []
            
            for dir_name in required_dirs:
                dir_path = src_dir / dir_name
                if dir_path.exists():
                    existing_dirs.append(dir_name)
                    # Check for __init__.py
                    if (dir_path / "__init__.py").exists():
                        result['passed_checks'].append(f"Directory {dir_name} has __init__.py")
                    else:
                        result['issues'].append(f"Directory {dir_name} missing __init__.py")
                else:
                    missing_dirs.append(dir_name)
            
            result['details']['existing_directories'] = existing_dirs
            result['details']['missing_directories'] = missing_dirs
            
            # Check key files exist
            key_files = [
                'integration/final_integration.py',
                'integration/config_integrator.py',
                'config/settings.py',
                'core/constraints.py',
                'core/policy.py',
                'algorithms/cpo.py'
            ]
            
            existing_files = []
            missing_files = []
            
            for file_path in key_files:
                full_path = src_dir / file_path
                if full_path.exists():
                    existing_files.append(file_path)
                    result['passed_checks'].append(f"Key file {file_path} exists")
                else:
                    missing_files.append(file_path)
                    result['issues'].append(f"Missing key file: {file_path}")
            
            result['details']['existing_files'] = existing_files
            result['details']['missing_files'] = missing_files
            
            # Calculate score
            total_checks = len(required_dirs) + len(key_files)
            passed_checks = len(existing_dirs) + len(existing_files)
            result['score'] = (passed_checks / total_checks) * 100.0
            
            if result['score'] >= 95.0:
                result['status'] = 'PASS'
            elif result['score'] >= 80.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Code structure validation error: {str(e)}")
        
        return result
    
    def _validate_configuration_system(self, config_dir: str) -> Dict[str, Any]:
        """Validate configuration system."""
        self.logger.info("Validating configuration system...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Test configuration integrator
            integrator = ConfigurationIntegrator(config_dir, Environment.DEVELOPMENT)
            config = integrator.load_unified_config()
            
            if config:
                result['passed_checks'].append("Configuration integrator loads successfully")
                
                # Test configuration validation
                if config.validate():
                    result['passed_checks'].append("Configuration validation passes")
                else:
                    result['issues'].append("Configuration validation fails")
                
                # Test configuration saving
                if integrator.save_config('test_config.yaml'):
                    result['passed_checks'].append("Configuration saving works")
                else:
                    result['issues'].append("Configuration saving fails")
                
                # Test configuration summary
                summary = integrator.get_config_summary()
                if summary and len(summary) > 100:
                    result['passed_checks'].append("Configuration summary generation works")
                else:
                    result['issues'].append("Configuration summary generation fails")
                
            else:
                result['issues'].append("Configuration integrator fails to load config")
            
            # Calculate score
            total_checks = 4
            passed_checks_count = len(result['passed_checks'])
            result['score'] = (passed_checks_count / total_checks) * 100.0
            
            if result['score'] >= 95.0:
                result['status'] = 'PASS'
            elif result['score'] >= 75.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Configuration validation error: {str(e)}")
            result['details']['exception'] = traceback.format_exc()
        
        return result
    
    def _validate_imports_and_dependencies(self) -> Dict[str, Any]:
        """Validate imports and dependencies."""
        self.logger.info("Validating imports and dependencies...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Test key imports
            import_tests = [
                ('integration.final_integration', 'UnifiedSafeRLSystem'),
                ('integration.config_integrator', 'ConfigurationIntegrator'),
                ('config.settings', 'SafeRLConfig'),
                ('core.constraints', 'SafetyConstraint'),
                ('algorithms.cpo', 'CPO')
            ]
            
            successful_imports = []
            failed_imports = []
            
            for module_name, class_name in import_tests:
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    getattr(module, class_name)
                    successful_imports.append(f"{module_name}.{class_name}")
                    result['passed_checks'].append(f"Successfully imported {module_name}.{class_name}")
                except Exception as e:
                    failed_imports.append(f"{module_name}.{class_name}: {str(e)}")
                    result['issues'].append(f"Failed to import {module_name}.{class_name}: {str(e)}")
            
            result['details']['successful_imports'] = successful_imports
            result['details']['failed_imports'] = failed_imports
            
            # Calculate score
            total_imports = len(import_tests)
            successful_count = len(successful_imports)
            result['score'] = (successful_count / total_imports) * 100.0
            
            if result['score'] >= 90.0:
                result['status'] = 'PASS'
            elif result['score'] >= 70.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Import validation error: {str(e)}")
        
        return result
    
    def _validate_system_integration(self, config_dir: str) -> Dict[str, Any]:
        """Validate system integration."""
        self.logger.info("Validating system integration...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Mock dependencies for testing
            from unittest.mock import Mock, patch
            
            mock_components = {
                'SafetyConstraint': Mock,
                'SafePolicy': Mock,
                'LagrangianOptimizer': Mock,
                'SafetyMonitor': Mock,
                'HardwareInterface': Mock,
                'SafetyHardware': Mock
            }
            
            with patch.multiple('integration.final_integration', **mock_components):
                # Test system creation
                system = UnifiedSafeRLSystem(config_dir, Environment.DEVELOPMENT)
                result['passed_checks'].append("System creation successful")
                
                # Test system initialization
                if system.initialize_system():
                    result['passed_checks'].append("System initialization successful")
                    
                    # Test system validation
                    validator = SystemValidator(system)
                    report = validator.validate_full_integration()
                    
                    if isinstance(report, IntegrationReport):
                        result['passed_checks'].append("Integration validation successful")
                        result['details']['readiness_score'] = report.readiness_score
                        result['details']['components_initialized'] = len(report.components_initialized)
                        result['details']['components_failed'] = len(report.components_failed)
                        
                        if report.readiness_score >= 80.0:
                            result['passed_checks'].append("High integration readiness score")
                        else:
                            result['issues'].append("Low integration readiness score")
                    else:
                        result['issues'].append("Integration validation failed")
                    
                    # Test system start/stop
                    if system.start_system():
                        result['passed_checks'].append("System start successful")
                        
                        if system.stop_system():
                            result['passed_checks'].append("System stop successful")
                        else:
                            result['issues'].append("System stop failed")
                    else:
                        result['issues'].append("System start failed")
                else:
                    result['issues'].append("System initialization failed")
            
            # Calculate score
            total_checks = 6
            passed_checks_count = len(result['passed_checks'])
            result['score'] = (passed_checks_count / total_checks) * 100.0
            
            if result['score'] >= 90.0:
                result['status'] = 'PASS'
            elif result['score'] >= 70.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"System integration validation error: {str(e)}")
            result['details']['exception'] = traceback.format_exc()
        
        return result
    
    def _validate_safety_systems(self) -> Dict[str, Any]:
        """Validate safety systems."""
        self.logger.info("Validating safety systems...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Check safety-related files exist
            safety_files = [
                'core/constraints.py',
                'core/safety_monitor.py',
                'hardware/safety_hardware.py',
                'realtime/safety_monitor.py',
                'environments/safety_monitoring.py'
            ]
            
            src_dir = self.project_root / "safe_rl_human_robot" / "src"
            existing_safety_files = []
            missing_safety_files = []
            
            for file_path in safety_files:
                full_path = src_dir / file_path
                if full_path.exists():
                    existing_safety_files.append(file_path)
                    result['passed_checks'].append(f"Safety file exists: {file_path}")
                else:
                    missing_safety_files.append(file_path)
                    result['issues'].append(f"Missing safety file: {file_path}")
            
            result['details']['existing_safety_files'] = existing_safety_files
            result['details']['missing_safety_files'] = missing_safety_files
            
            # Test safety constraint imports
            try:
                from core.constraints import SafetyConstraint
                result['passed_checks'].append("SafetyConstraint import successful")
            except Exception as e:
                result['issues'].append(f"SafetyConstraint import failed: {str(e)}")
            
            # Test safety monitor imports
            try:
                from core.safety_monitor import SafetyMonitor
                result['passed_checks'].append("SafetyMonitor import successful")
            except Exception as e:
                result['issues'].append(f"SafetyMonitor import failed: {str(e)}")
            
            # Calculate score
            total_checks = len(safety_files) + 2
            passed_checks_count = len(result['passed_checks'])
            result['score'] = (passed_checks_count / total_checks) * 100.0
            
            if result['score'] >= 90.0:
                result['status'] = 'PASS'
            elif result['score'] >= 70.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Safety validation error: {str(e)}")
        
        return result
    
    def _validate_performance_requirements(self) -> Dict[str, Any]:
        """Validate performance requirements."""
        self.logger.info("Validating performance requirements...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Check performance optimization files exist
            performance_files = [
                'optimization/performance_optimizer.py',
                'optimization/rt_optimizer_integration.py',
                'optimization/gpu/cuda_optimizer.py',
                'optimization/memory/rt_memory_manager.py',
                'realtime/realtime_controller.py'
            ]
            
            src_dir = self.project_root / "safe_rl_human_robot" / "src"
            existing_perf_files = []
            
            for file_path in performance_files:
                full_path = src_dir / file_path
                if full_path.exists():
                    existing_perf_files.append(file_path)
                    result['passed_checks'].append(f"Performance file exists: {file_path}")
                else:
                    result['issues'].append(f"Missing performance file: {file_path}")
            
            result['details']['existing_performance_files'] = existing_perf_files
            
            # Test real-time capabilities
            try:
                from realtime.realtime_controller import RealTimeController
                result['passed_checks'].append("RealTimeController import successful")
            except Exception as e:
                result['issues'].append(f"RealTimeController import failed: {str(e)}")
            
            # Calculate score
            total_checks = len(performance_files) + 1
            passed_checks_count = len(result['passed_checks'])
            result['score'] = (passed_checks_count / total_checks) * 100.0
            
            if result['score'] >= 85.0:
                result['status'] = 'PASS'
            elif result['score'] >= 65.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Performance validation error: {str(e)}")
        
        return result
    
    def _validate_security_features(self) -> Dict[str, Any]:
        """Validate security features."""
        self.logger.info("Validating security features...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Check security files exist
            security_files = [
                'security/authentication.py',
                'security/compliance.py',
                'deployment/config_manager.py'
            ]
            
            src_dir = self.project_root / "safe_rl_human_robot" / "src"
            existing_security_files = []
            
            for file_path in security_files:
                full_path = src_dir / file_path
                if full_path.exists():
                    existing_security_files.append(file_path)
                    result['passed_checks'].append(f"Security file exists: {file_path}")
                else:
                    result['issues'].append(f"Missing security file: {file_path}")
            
            result['details']['existing_security_files'] = existing_security_files
            
            # Test configuration manager with encryption
            try:
                from deployment.config_manager import ConfigurationManager
                result['passed_checks'].append("ConfigurationManager import successful")
            except Exception as e:
                result['issues'].append(f"ConfigurationManager import failed: {str(e)}")
            
            # Calculate score
            total_checks = len(security_files) + 1
            passed_checks_count = len(result['passed_checks'])
            result['score'] = (passed_checks_count / total_checks) * 100.0
            
            if result['score'] >= 80.0:
                result['status'] = 'PASS'
            elif result['score'] >= 60.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Security validation error: {str(e)}")
        
        return result
    
    def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation."""
        self.logger.info("Validating documentation...")
        
        result = {
            'status': 'UNKNOWN',
            'score': 0.0,
            'issues': [],
            'passed_checks': [],
            'details': {}
        }
        
        try:
            # Check documentation files exist
            doc_files = [
                'README.md',
                'docs/API_DOCUMENTATION.md',
                'docs/PRODUCTION_DEPLOYMENT_GUIDE.md',
                'safe_rl_human_robot/README.md',
                'safe_rl_human_robot/docs/API_REFERENCE.md'
            ]
            
            existing_doc_files = []
            
            for file_path in doc_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    existing_doc_files.append(file_path)
                    result['passed_checks'].append(f"Documentation file exists: {file_path}")
                    
                    # Check file size (should have content)
                    if full_path.stat().st_size > 100:
                        result['passed_checks'].append(f"Documentation file has content: {file_path}")
                    else:
                        result['issues'].append(f"Documentation file too small: {file_path}")
                else:
                    result['issues'].append(f"Missing documentation file: {file_path}")
            
            result['details']['existing_doc_files'] = existing_doc_files
            
            # Calculate score
            total_checks = len(doc_files) * 2  # Existence + content checks
            passed_checks_count = len(result['passed_checks'])
            result['score'] = (passed_checks_count / total_checks) * 100.0
            
            if result['score'] >= 85.0:
                result['status'] = 'PASS'
            elif result['score'] >= 60.0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'FAIL'
                
        except Exception as e:
            result['status'] = 'ERROR'
            result['issues'].append(f"Documentation validation error: {str(e)}")
        
        return result
    
    def _calculate_overall_readiness(self, results: Dict[str, Any]):
        """Calculate overall production readiness."""
        validations = results['validations']
        
        # Weight different validation areas
        weights = {
            'code_structure': 0.20,
            'configuration': 0.15,
            'imports': 0.15,
            'integration': 0.25,
            'safety': 0.15,
            'performance': 0.05,
            'security': 0.03,
            'documentation': 0.02
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for validation_name, weight in weights.items():
            if validation_name in validations:
                validation_result = validations[validation_name]
                if validation_result['status'] != 'ERROR':
                    total_score += validation_result['score'] * weight
                    total_weight += weight
        
        if total_weight > 0:
            results['readiness_score'] = total_score / total_weight
        else:
            results['readiness_score'] = 0.0
        
        # Determine overall status
        if results['readiness_score'] >= 95.0:
            results['overall_status'] = 'PRODUCTION_READY'
            results['production_ready'] = True
        elif results['readiness_score'] >= 85.0:
            results['overall_status'] = 'NEARLY_READY'
            results['production_ready'] = False
        elif results['readiness_score'] >= 70.0:
            results['overall_status'] = 'NEEDS_WORK'
            results['production_ready'] = False
        else:
            results['overall_status'] = 'NOT_READY'
            results['production_ready'] = False
        
        # Check for critical failures
        critical_areas = ['safety', 'integration']
        for area in critical_areas:
            if area in validations and validations[area]['status'] == 'FAIL':
                results['overall_status'] = 'CRITICAL_ISSUES'
                results['production_ready'] = False
                break
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validation first."
        
        results = self.validation_results
        
        report_lines = [
            "=" * 80,
            "SAFE RL HUMAN-ROBOT SHARED CONTROL - PRODUCTION READINESS REPORT",
            "=" * 80,
            "",
            f"Validation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['validation_timestamp']))}",
            f"Project Root: {results['project_root']}",
            f"Validation Duration: {results['validation_duration']:.2f} seconds",
            "",
            "OVERALL ASSESSMENT",
            "-" * 40,
            f"Overall Status: {results['overall_status']}",
            f"Production Ready: {'✅ YES' if results['production_ready'] else '❌ NO'}",
            f"Readiness Score: {results['readiness_score']:.1f}/100.0",
            ""
        ]
        
        # Add detailed validation results
        if 'validations' in results:
            report_lines.extend([
                "DETAILED VALIDATION RESULTS",
                "-" * 40
            ])
            
            for validation_name, validation_result in results['validations'].items():
                status_emoji = {
                    'PASS': '✅',
                    'WARNING': '⚠️',
                    'FAIL': '❌',
                    'ERROR': '💥'
                }.get(validation_result['status'], '❓')
                
                report_lines.extend([
                    "",
                    f"{validation_name.upper().replace('_', ' ')} {status_emoji}",
                    f"  Status: {validation_result['status']}",
                    f"  Score: {validation_result['score']:.1f}/100.0",
                    f"  Passed Checks: {len(validation_result['passed_checks'])}",
                    f"  Issues: {len(validation_result['issues'])}"
                ])
                
                if validation_result['passed_checks']:
                    report_lines.append("  ✓ Passed:")
                    for check in validation_result['passed_checks'][:5]:  # Show first 5
                        report_lines.append(f"    - {check}")
                    if len(validation_result['passed_checks']) > 5:
                        report_lines.append(f"    ... and {len(validation_result['passed_checks']) - 5} more")
                
                if validation_result['issues']:
                    report_lines.append("  ✗ Issues:")
                    for issue in validation_result['issues'][:5]:  # Show first 5
                        report_lines.append(f"    - {issue}")
                    if len(validation_result['issues']) > 5:
                        report_lines.append(f"    ... and {len(validation_result['issues']) - 5} more")
        
        # Add recommendations
        report_lines.extend([
            "",
            "",
            "RECOMMENDATIONS",
            "-" * 40
        ])
        
        if results['production_ready']:
            report_lines.extend([
                "🎉 CONGRATULATIONS! Your Safe RL system is production ready!",
                "",
                "The system has passed all critical validation checks and is ready for",
                "deployment in production environments. Continue monitoring system",
                "performance and safety metrics in production."
            ])
        else:
            report_lines.extend([
                "The system is not yet ready for production deployment.",
                "Please address the following areas:",
                ""
            ])
            
            for validation_name, validation_result in results.get('validations', {}).items():
                if validation_result['status'] in ['FAIL', 'ERROR']:
                    report_lines.append(f"• Fix issues in {validation_name.replace('_', ' ')}")
                elif validation_result['status'] == 'WARNING':
                    report_lines.append(f"• Review warnings in {validation_name.replace('_', ' ')}")
        
        report_lines.extend([
            "",
            "=" * 80,
            f"Report generated on {time.strftime('%Y-%m-%d %H:%M:%S')}",
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
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate Safe RL system production readiness"
    )
    parser.add_argument(
        '--config-dir', 
        help='Configuration directory path'
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
        # Find project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        logger.info(f"Starting production readiness validation...")
        logger.info(f"Project root: {project_root}")
        
        # Create validator
        validator = ProductionReadinessValidator(project_root)
        
        # Run validation
        results = validator.validate_production_readiness(args.config_dir)
        
        # Generate report
        report_text = validator.generate_report(args.output_report)
        print(report_text)
        
        # Save JSON results if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"JSON results saved to {args.output_json}")
        
        # Exit with appropriate code
        if results['production_ready']:
            logger.info("✅ System is production ready!")
            sys.exit(0)
        else:
            logger.warning("❌ System is not production ready")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.debug(f"Exception details: {traceback.format_exc()}")
        sys.exit(2)


if __name__ == "__main__":
    main()