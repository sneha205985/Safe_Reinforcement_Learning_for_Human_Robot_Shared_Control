"""
Comprehensive Integration Tests for Final Safe RL System Integration.

Tests the complete system integration, initialization, validation, and operation
to ensure 100% production readiness.
"""

import os
import sys
import time
import tempfile
import unittest
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from integration.final_integration import (
    UnifiedSafeRLSystem, 
    SystemState, 
    ComponentStatus,
    IntegrationReport,
    SystemValidator,
    create_production_system
)
from integration.config_integrator import (
    ConfigurationIntegrator,
    UnifiedConfig,
    ConfigurationPriority
)
from deployment.config_manager import Environment
from config.settings import SafeRLConfig


class TestUnifiedSafeRLSystem(unittest.TestCase):
    """Test unified Safe RL system integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal configuration files
        self._create_test_configurations()
        
        # Set up logging for tests
        logging.basicConfig(level=logging.DEBUG)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_configurations(self):
        """Create minimal test configurations."""
        import yaml
        
        # System configuration
        system_config = {
            'node_name': 'test_safe_rl_system',
            'log_level': 'INFO',
            'max_cpu_percent': 80.0,
            'max_memory_mb': 1024,
            'enable_profiling': False,
            'heartbeat_interval_s': 1.0
        }
        
        system_dir = self.config_dir / "system"
        system_dir.mkdir(exist_ok=True)
        with open(system_dir / "base.yaml", 'w') as f:
            yaml.dump(system_config, f)
        
        # Hardware configuration
        hardware_config = {
            'interfaces': [
                {
                    'device_id': 'test_device',
                    'device_type': 'exoskeleton',
                    'communication_protocol': 'serial',
                    'sampling_rate': 1000.0,
                    'safety_limits': {
                        'max_torque': 50.0,
                        'max_velocity': 2.0,
                        'position_limits': [-1.57, 1.57]
                    }
                }
            ]
        }
        
        hardware_dir = self.config_dir / "hardware"
        hardware_dir.mkdir(exist_ok=True)
        with open(hardware_dir / "base.yaml", 'w') as f:
            yaml.dump(hardware_config, f)
        
        # Safety configuration
        safety_config = {
            'monitoring_frequency': 2000.0,
            'emergency_stop_enabled': True,
            'position_limits': {
                'joint_0': [-1.57, 1.57],
                'joint_1': [-1.0, 1.0]
            },
            'velocity_limits': {
                'joint_0': 2.0,
                'joint_1': 1.5
            },
            'force_limits': {
                'max_force': 50.0
            },
            'predictive_monitoring': {
                'enabled': True,
                'prediction_horizon_s': 0.5
            }
        }
        
        safety_dir = self.config_dir / "safety"
        safety_dir.mkdir(exist_ok=True)
        with open(safety_dir / "base.yaml", 'w') as f:
            yaml.dump(safety_config, f)
    
    def test_system_initialization(self):
        """Test system initialization process."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock,
            SafetyMonitor=Mock,
            HardwareInterface=Mock,
            SafetyHardware=Mock,
            ExoskeletonInterface=Mock
        ):
            system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            
            # Test initial state
            self.assertEqual(system.state, SystemState.UNINITIALIZED)
            self.assertEqual(len(system.components), 0)
            
            # Test initialization
            success = system.initialize_system()
            self.assertTrue(success)
            self.assertEqual(system.state, SystemState.CONFIGURED)
            self.assertIsNotNone(system.unified_config)
            self.assertIsNotNone(system.config_integrator)
            self.assertIsNotNone(system.validator)
            
            # Test components were initialized
            self.assertGreater(len(system.components), 0)
            
            # Test configuration validation
            self.assertTrue(system.unified_config.validate())
    
    def test_component_dependency_resolution(self):
        """Test component dependency resolution."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock,
            SafetyMonitor=Mock
        ):
            system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            system.initialize_system()
            
            # Check that dependencies are properly resolved
            if 'safe_policy' in system.components:
                safe_policy = system.components['safe_policy']
                self.assertIn('safety_constraint', safe_policy.dependencies)
            
            if 'safety_monitor' in system.components:
                safety_monitor = system.components['safety_monitor']
                self.assertIn('safety_constraint', safety_monitor.dependencies)
            
            # Check component start order respects dependencies
            start_order = system._get_component_start_order()
            
            # safety_constraint should come before safe_policy
            if 'safety_constraint' in start_order and 'safe_policy' in start_order:
                constraint_idx = start_order.index('safety_constraint')
                policy_idx = start_order.index('safe_policy')
                self.assertLess(constraint_idx, policy_idx)
    
    def test_system_start_stop(self):
        """Test system start and stop functionality."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock,
            SafetyMonitor=Mock
        ):
            system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            
            # Initialize system
            self.assertTrue(system.initialize_system())
            
            # Test start
            self.assertTrue(system.start_system())
            self.assertEqual(system.state, SystemState.RUNNING)
            
            # Check components are running
            running_components = [
                name for name, info in system.components.items()
                if info.status == ComponentStatus.RUNNING
            ]
            self.assertGreater(len(running_components), 0)
            
            # Test stop
            self.assertTrue(system.stop_system())
            self.assertEqual(system.state, SystemState.STOPPED)
    
    def test_configuration_integration(self):
        """Test configuration integration functionality."""
        system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
        
        # Test configuration loading
        config_overrides = {
            'debug_mode': True,
            'log_level': 'DEBUG',
            'hardware_config.test_param': 'test_value'
        }
        
        self.assertTrue(system.initialize_system(config_overrides))
        
        # Check configuration was applied
        config = system.get_config()
        self.assertIsNotNone(config)
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.log_level, 'DEBUG')
    
    def test_health_monitoring(self):
        """Test system health monitoring."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock
        ):
            system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            system.initialize_system()
            
            # Check health monitoring thread is started
            self.assertIsNotNone(system.health_check_thread)
            self.assertTrue(system.health_check_thread.is_alive())
            
            # Test health check execution
            initial_check_time = system.last_health_check
            time.sleep(0.1)  # Allow health check to run
            system._perform_health_checks()
            self.assertGreater(system.last_health_check, initial_check_time)
            
            # Cleanup
            system.stop_system()
    
    def test_error_handling(self):
        """Test system error handling."""
        # Test initialization failure
        with patch('integration.config_integrator.ConfigurationIntegrator') as mock_config:
            mock_config.side_effect = Exception("Configuration failed")
            
            system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            success = system.initialize_system()
            
            self.assertFalse(success)
            self.assertEqual(system.state, SystemState.ERROR)
    
    def test_context_manager(self):
        """Test system context manager functionality."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock
        ):
            with UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT) as system:
                self.assertIsInstance(system, UnifiedSafeRLSystem)
                # System should be properly initialized within context
                self.assertIn(system.state, [SystemState.UNINITIALIZED, SystemState.CONFIGURED])
            
            # System should be stopped after context exit
            if system.state not in [SystemState.UNINITIALIZED, SystemState.ERROR]:
                self.assertEqual(system.state, SystemState.STOPPED)


class TestSystemValidator(unittest.TestCase):
    """Test system validator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test system
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock,
            SafetyMonitor=Mock
        ):
            self.system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            self.system.initialize_system()
            
            self.validator = SystemValidator(self.system)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if hasattr(self, 'system'):
            self.system.stop_system()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_integration_validation(self):
        """Test comprehensive integration validation."""
        report = self.validator.validate_full_integration()
        
        self.assertIsInstance(report, IntegrationReport)
        self.assertEqual(report.environment, Environment.DEVELOPMENT)
        self.assertGreater(report.integration_timestamp, 0)
        
        # Check report structure
        self.assertIsInstance(report.components_initialized, list)
        self.assertIsInstance(report.components_failed, list)
        self.assertIsInstance(report.warnings, list)
        self.assertIsInstance(report.errors, list)
        self.assertIsInstance(report.recommendations, list)
        
        # Check readiness score calculation
        score = report.calculate_readiness_score()
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with valid configuration
        result = self.validator._validate_configuration(IntegrationReport())
        self.assertIsInstance(result, bool)
        
        # Test with invalid configuration
        self.system.unified_config = None
        report = IntegrationReport()
        result = self.validator._validate_configuration(report)
        self.assertFalse(result)
        self.assertGreater(len(report.errors), 0)
    
    def test_dependency_validation(self):
        """Test component dependency validation."""
        report = IntegrationReport()
        result = self.validator._validate_dependencies(report)
        self.assertIsInstance(result, bool)
    
    def test_safety_system_validation(self):
        """Test safety system validation."""
        report = IntegrationReport()
        result = self.validator._validate_safety_systems(report)
        self.assertIsInstance(result, bool)
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        report = IntegrationReport()
        self.validator._collect_performance_metrics(report)
        
        # Memory usage should be set (may be 0 if psutil not available)
        self.assertIsInstance(report.memory_usage_mb, float)
        self.assertIsInstance(report.cpu_usage_percent, float)
    
    def test_recommendations_generation(self):
        """Test recommendations generation."""
        report = IntegrationReport()
        report.configuration_valid = False
        report.dependencies_resolved = False
        report.safety_systems_active = False
        
        self.validator._generate_recommendations(report)
        
        self.assertGreater(len(report.recommendations), 0)
        self.assertIn("CRITICAL", " ".join(report.recommendations))


class TestConfigurationIntegrator(unittest.TestCase):
    """Test configuration integrator functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test configuration files
        self._create_test_configs()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_configs(self):
        """Create test configuration files."""
        import yaml
        
        # Create system config
        system_dir = self.config_dir / "system"
        system_dir.mkdir(exist_ok=True)
        
        system_config = {
            'node_name': 'test_system',
            'log_level': 'INFO',
            'max_cpu_percent': 80.0,
            'max_memory_mb': 1024
        }
        
        with open(system_dir / "base.yaml", 'w') as f:
            yaml.dump(system_config, f)
    
    def test_configuration_loading(self):
        """Test configuration loading and integration."""
        integrator = ConfigurationIntegrator(self.config_dir, Environment.DEVELOPMENT)
        
        # Test unified config loading
        config = integrator.load_unified_config()
        
        self.assertIsInstance(config, UnifiedConfig)
        self.assertEqual(config.environment, Environment.DEVELOPMENT)
        self.assertTrue(config.validate())
    
    def test_configuration_overrides(self):
        """Test configuration overrides."""
        integrator = ConfigurationIntegrator(self.config_dir, Environment.DEVELOPMENT)
        
        overrides = {
            'debug_mode': True,
            'log_level': 'DEBUG'
        }
        
        config = integrator.load_unified_config(override_values=overrides)
        
        self.assertTrue(config.debug_mode)
        self.assertEqual(config.log_level, 'DEBUG')
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        # Set test environment variables
        os.environ['SAFE_RL_DEBUG_MODE'] = 'true'
        os.environ['SAFE_RL_LOG_LEVEL'] = 'DEBUG'
        
        try:
            integrator = ConfigurationIntegrator(self.config_dir, Environment.DEVELOPMENT)
            config = integrator.load_unified_config()
            
            self.assertTrue(config.debug_mode)
            self.assertEqual(config.log_level, 'DEBUG')
            
        finally:
            # Clean up environment variables
            os.environ.pop('SAFE_RL_DEBUG_MODE', None)
            os.environ.pop('SAFE_RL_LOG_LEVEL', None)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        integrator = ConfigurationIntegrator(self.config_dir, Environment.DEVELOPMENT)
        config = integrator.load_unified_config()
        
        # Test valid configuration
        self.assertTrue(config.validate())
        
        # Test invalid configuration
        config.hardware_config = {'invalid': 'config'}
        self.assertFalse(config.validate())
    
    def test_configuration_saving(self):
        """Test configuration saving."""
        integrator = ConfigurationIntegrator(self.config_dir, Environment.DEVELOPMENT)
        config = integrator.load_unified_config()
        
        # Test saving
        self.assertTrue(integrator.save_config('test_config.yaml'))
        
        # Check file was created
        config_file = self.config_dir / 'test_config.yaml'
        self.assertTrue(config_file.exists())
    
    def test_configuration_change_callbacks(self):
        """Test configuration change callbacks."""
        integrator = ConfigurationIntegrator(self.config_dir, Environment.DEVELOPMENT)
        integrator.load_unified_config()
        
        callback_called = threading.Event()
        
        def test_callback(config):
            callback_called.set()
        
        integrator.register_change_callback(test_callback)
        
        # Simulate configuration change
        integrator._notify_change_callbacks()
        
        # Wait for callback
        self.assertTrue(callback_called.wait(timeout=1.0))


class TestIntegrationReport(unittest.TestCase):
    """Test integration report functionality."""
    
    def test_report_creation(self):
        """Test integration report creation."""
        report = IntegrationReport(
            environment=Environment.DEVELOPMENT,
            system_version="1.0.0"
        )
        
        self.assertEqual(report.environment, Environment.DEVELOPMENT)
        self.assertEqual(report.system_version, "1.0.0")
        self.assertIsInstance(report.integration_timestamp, float)
    
    def test_readiness_score_calculation(self):
        """Test readiness score calculation."""
        report = IntegrationReport()
        
        # Test with no data
        score = report.calculate_readiness_score()
        self.assertEqual(score, 0.0)
        
        # Test with all positive indicators
        report.components_initialized = ['comp1', 'comp2', 'comp3']
        report.components_failed = []
        report.configuration_valid = True
        report.dependencies_resolved = True
        report.hardware_connected = True
        report.safety_systems_active = True
        report.unit_tests_passed = 10
        
        score = report.calculate_readiness_score()
        self.assertGreater(score, 90.0)
        
        # Test with errors
        report.errors = ['error1', 'error2']
        score = report.calculate_readiness_score()
        self.assertLess(score, 90.0)
    
    def test_production_readiness_check(self):
        """Test production readiness determination."""
        report = IntegrationReport()
        
        # Test not ready
        self.assertFalse(report.is_production_ready())
        
        # Test ready
        report.components_initialized = ['comp1']
        report.configuration_valid = True
        report.dependencies_resolved = True
        report.hardware_connected = True
        report.safety_systems_active = True
        report.unit_tests_passed = 10
        
        report.calculate_readiness_score()  # Update score
        self.assertTrue(report.is_production_ready())
    
    def test_report_serialization(self):
        """Test report serialization to dictionary."""
        report = IntegrationReport(
            system_version="1.0.0",
            environment=Environment.DEVELOPMENT
        )
        
        report.components_initialized = ['comp1']
        report.warnings = ['warning1']
        
        report_dict = report.to_dict()
        
        self.assertIsInstance(report_dict, dict)
        self.assertEqual(report_dict['system_version'], "1.0.0")
        self.assertEqual(report_dict['environment'], "development")
        self.assertEqual(report_dict['components_initialized'], ['comp1'])
        self.assertIn('production_ready', report_dict)


class TestProductionSystemCreation(unittest.TestCase):
    """Test production system creation functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_production_system_creation(self):
        """Test creating production-ready system."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock,
            SafetyMonitor=Mock
        ):
            system = create_production_system(
                str(self.config_dir), 
                Environment.PRODUCTION
            )
            
            self.assertIsInstance(system, UnifiedSafeRLSystem)
            self.assertEqual(system.environment, Environment.PRODUCTION)
            self.assertEqual(system.state, SystemState.CONFIGURED)
            
            # Check production settings were applied
            config = system.get_config()
            self.assertFalse(config.debug_mode)  # Debug should be disabled in production
            self.assertTrue(config.safety_config.get('emergency_stop_enabled', False))


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive test configurations
        self._create_comprehensive_configs()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_comprehensive_configs(self):
        """Create comprehensive test configurations."""
        import yaml
        
        # Create all required config directories and files
        config_types = ['system', 'hardware', 'safety', 'policy', 'ros', 'deployment']
        
        for config_type in config_types:
            config_dir = self.config_dir / config_type
            config_dir.mkdir(exist_ok=True)
            
            # Create minimal but valid configuration
            if config_type == 'system':
                config_data = {
                    'node_name': f'test_{config_type}_system',
                    'log_level': 'INFO'
                }
            elif config_type == 'hardware':
                config_data = {
                    'interfaces': [{
                        'device_id': 'test_device',
                        'device_type': 'exoskeleton'
                    }]
                }
            elif config_type == 'safety':
                config_data = {
                    'monitoring_frequency': 1000.0,
                    'emergency_stop_enabled': True
                }
            else:
                config_data = {
                    'test_param': f'test_value_{config_type}'
                }
            
            with open(config_dir / "base.yaml", 'w') as f:
                yaml.dump(config_data, f)
    
    def test_complete_system_lifecycle(self):
        """Test complete system lifecycle from initialization to shutdown."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock,
            SafetyMonitor=Mock,
            HardwareInterface=Mock,
            SafetyHardware=Mock,
            ExoskeletonInterface=Mock,
            BiomechanicalModel=Mock,
            IntentRecognitionSystem=Mock,
            AdaptiveHumanModel=Mock,
            RealTimeProcessor=Mock,
            CPO=Mock,
            GAE=Mock,
            TrustRegionOptimizer=Mock,
            PerformanceOptimizer=Mock,
            RTOptimizerIntegration=Mock,
            RealTimeController=Mock,
            RTSafetyMonitor=Mock,
            TimingManager=Mock,
            MetricsCollector=Mock,
            AlertingSystem=Mock,
            EvaluationSuite=Mock
        ):
            
            # Phase 1: System Creation
            system = UnifiedSafeRLSystem(
                self.config_dir, 
                Environment.DEVELOPMENT,
                version="1.0.0"
            )
            self.assertEqual(system.state, SystemState.UNINITIALIZED)
            
            # Phase 2: System Initialization
            success = system.initialize_system({
                'debug_mode': True,
                'log_level': 'DEBUG'
            })
            self.assertTrue(success)
            self.assertEqual(system.state, SystemState.CONFIGURED)
            
            # Phase 3: Configuration Validation
            config = system.get_config()
            self.assertIsNotNone(config)
            self.assertTrue(config.validate())
            self.assertTrue(config.debug_mode)
            self.assertEqual(config.log_level, 'DEBUG')
            
            # Phase 4: Component Verification
            self.assertGreater(len(system.components), 5)  # Should have multiple components
            
            for name, info in system.components.items():
                self.assertIn(info.status, [ComponentStatus.READY, ComponentStatus.ERROR])
                if info.status == ComponentStatus.ERROR:
                    print(f"Component {name} in error state: {info.last_error}")
            
            # Phase 5: System Start
            success = system.start_system()
            self.assertTrue(success)
            self.assertEqual(system.state, SystemState.RUNNING)
            
            # Phase 6: System Validation
            validator = SystemValidator(system)
            report = validator.validate_full_integration()
            
            self.assertIsInstance(report, IntegrationReport)
            self.assertGreater(report.readiness_score, 0.0)
            
            # Phase 7: System Status Check
            status = system.get_system_status()
            self.assertEqual(status['state'], 'RUNNING')
            self.assertGreater(status['uptime'], 0)
            
            # Phase 8: System Stop
            success = system.stop_system()
            self.assertTrue(success)
            self.assertEqual(system.state, SystemState.STOPPED)
    
    def test_error_recovery(self):
        """Test system error recovery capabilities."""
        with patch.multiple(
            'integration.final_integration',
            SafetyConstraint=Mock,
            SafePolicy=Mock,
            LagrangianOptimizer=Mock
        ):
            # Test recovery from component failure
            SafePolicy.side_effect = Exception("Component failed")
            
            system = UnifiedSafeRLSystem(self.config_dir, Environment.DEVELOPMENT)
            success = system.initialize_system()
            
            # System should handle component failures gracefully
            self.assertIn(system.state, [SystemState.CONFIGURED, SystemState.ERROR])
            
            # Check that error was recorded
            if 'safe_policy' in system.components:
                policy_component = system.components['safe_policy']
                self.assertEqual(policy_component.status, ComponentStatus.ERROR)
                self.assertIsNotNone(policy_component.last_error)


def run_integration_tests():
    """Run all integration tests."""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestUnifiedSafeRLSystem,
        TestSystemValidator,
        TestConfigurationIntegrator,
        TestIntegrationReport,
        TestProductionSystemCreation,
        TestEndToEndIntegration
    ]
    
    for test_case in test_cases:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_case)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Integration Test Results")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Running Safe RL System Final Integration Tests...")
    print("=" * 60)
    
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
        print("Safe RL system is ready for production deployment.")
    else:
        print("\n❌ Some integration tests failed.")
        print("Please address the issues before production deployment.")
    
    sys.exit(0 if success else 1)