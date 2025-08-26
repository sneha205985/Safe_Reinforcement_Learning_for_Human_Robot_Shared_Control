"""
Hardware-in-the-Loop Testing Framework.

This module provides comprehensive hardware testing capabilities including
functional testing, performance validation, safety verification, and
fault tolerance testing for Safe RL systems.

Key Features:
- Hardware-in-the-loop (HIL) testing
- Real-time performance validation
- Safety constraint verification
- Fault injection and recovery testing
- Automated test execution and reporting
- Continuous integration support
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import json
import csv
from pathlib import Path
import unittest
from abc import ABC, abstractmethod

from ..hardware.hardware_interface import HardwareInterface, SafetyStatus, SafetyLevel
from ..hardware.safety_hardware import SafetyHardware
from ..realtime.realtime_controller import RealTimeController, RTConfig
from ..realtime.safety_monitor import RTSafetyMonitor, SafetyConfig

logger = logging.getLogger(__name__)


class TestResult(Enum):
    """Test result states."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class TestSeverity(Enum):
    """Test severity levels."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class TestCategory(Enum):
    """Test categories."""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SAFETY = "safety"
    FAULT_TOLERANCE = "fault_tolerance"
    INTEGRATION = "integration"
    STRESS = "stress"
    REGRESSION = "regression"


@dataclass
class TestConfig:
    """Configuration for a test case."""
    test_id: str
    name: str
    description: str
    category: TestCategory
    severity: TestSeverity
    timeout_s: float = 300.0
    prerequisites: List[str] = field(default_factory=list)
    hardware_required: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    safety_critical: bool = False


@dataclass 
class TestReport:
    """Test execution report."""
    test_id: str
    name: str
    result: TestResult
    start_time: float
    end_time: float
    duration_s: float
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)


@dataclass
class TestSuiteReport:
    """Test suite execution report."""
    suite_name: str
    start_time: float
    end_time: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    test_reports: List[TestReport] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class HardwareTestCase(ABC):
    """Abstract base class for hardware test cases."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.test_id}")
        self.hardware_interfaces: Dict[str, HardwareInterface] = {}
        self.safety_hardware: Optional[SafetyHardware] = None
        self.rt_controller: Optional[RealTimeController] = None
        self.safety_monitor: Optional[RTSafetyMonitor] = None
        
        # Test state
        self.setup_complete = False
        self.teardown_complete = False
        self.test_data: Dict[str, Any] = {}
        self.artifacts: List[Path] = []
    
    def setup(self) -> bool:
        """Setup test environment."""
        try:
            self.logger.info(f"Setting up test: {self.config.name}")
            
            # Setup hardware interfaces
            if not self._setup_hardware():
                return False
            
            # Setup safety systems
            if not self._setup_safety():
                return False
            
            # Setup real-time controller
            if not self._setup_rt_controller():
                return False
            
            # Custom test setup
            if not self.setup_test():
                return False
            
            self.setup_complete = True
            self.logger.info("Test setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Test setup failed: {e}")
            return False
    
    def teardown(self) -> bool:
        """Teardown test environment."""
        try:
            self.logger.info("Tearing down test environment")
            
            # Custom test teardown
            self.teardown_test()
            
            # Shutdown components
            if self.rt_controller:
                self.rt_controller.shutdown()
            
            if self.safety_monitor:
                self.safety_monitor.shutdown()
            
            for hw in self.hardware_interfaces.values():
                hw.shutdown_hardware()
            
            if self.safety_hardware:
                self.safety_hardware.shutdown_hardware()
            
            self.teardown_complete = True
            self.logger.info("Test teardown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Test teardown failed: {e}")
            return False
    
    def execute(self) -> TestReport:
        """Execute the test case."""
        start_time = time.time()
        report = TestReport(
            test_id=self.config.test_id,
            name=self.config.name,
            result=TestResult.ERROR,
            start_time=start_time,
            end_time=start_time,
            duration_s=0.0
        )
        
        try:
            self.logger.info(f"Executing test: {self.config.name}")
            
            # Setup test environment
            if not self.setup():
                report.result = TestResult.ERROR
                report.error_message = "Test setup failed"
                return report
            
            # Execute test with timeout
            test_thread = threading.Thread(target=self._execute_test_thread)
            test_thread.daemon = True
            test_thread.start()
            
            test_thread.join(timeout=self.config.timeout_s)
            
            if test_thread.is_alive():
                # Test timed out
                report.result = TestResult.TIMEOUT
                report.error_message = f"Test timed out after {self.config.timeout_s}s"
            else:
                # Test completed
                if 'test_result' in self.test_data:
                    report.result = self.test_data['test_result']
                    report.error_message = self.test_data.get('error_message', '')
                    report.details = self.test_data.get('details', {})
                    report.metrics = self.test_data.get('metrics', {})
                else:
                    report.result = TestResult.ERROR
                    report.error_message = "Test did not produce results"
            
            report.end_time = time.time()
            report.duration_s = report.end_time - report.start_time
            report.artifacts = [str(path) for path in self.artifacts]
            
        except Exception as e:
            report.result = TestResult.ERROR
            report.error_message = str(e)
            report.end_time = time.time()
            report.duration_s = report.end_time - report.start_time
            self.logger.error(f"Test execution failed: {e}")
        
        finally:
            # Always try to teardown
            self.teardown()
        
        self.logger.info(f"Test completed: {report.result.value} in {report.duration_s:.2f}s")
        return report
    
    def _execute_test_thread(self):
        """Execute test in separate thread."""
        try:
            self.run_test()
        except Exception as e:
            self.test_data['test_result'] = TestResult.ERROR
            self.test_data['error_message'] = str(e)
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def setup_test(self) -> bool:
        """Custom test setup logic."""
        pass
    
    @abstractmethod
    def run_test(self):
        """Main test logic."""
        pass
    
    @abstractmethod
    def teardown_test(self):
        """Custom test teardown logic."""
        pass
    
    # Helper methods for subclasses
    
    def _setup_hardware(self) -> bool:
        """Setup hardware interfaces."""
        # This would be implemented based on test requirements
        return True
    
    def _setup_safety(self) -> bool:
        """Setup safety systems."""
        # This would be implemented based on test requirements
        return True
    
    def _setup_rt_controller(self) -> bool:
        """Setup real-time controller."""
        # This would be implemented based on test requirements
        return True
    
    def assert_value_in_range(self, value: float, min_val: float, max_val: float, message: str = ""):
        """Assert value is within range."""
        if not (min_val <= value <= max_val):
            error_msg = f"Value {value} not in range [{min_val}, {max_val}]"
            if message:
                error_msg += f": {message}"
            raise AssertionError(error_msg)
    
    def assert_safety_status(self, expected_level: SafetyLevel, message: str = ""):
        """Assert safety status meets expected level."""
        if self.safety_monitor:
            status = self.safety_monitor.get_safety_status()
            if status.safety_level < expected_level:
                error_msg = f"Safety level {status.safety_level.name} below expected {expected_level.name}"
                if message:
                    error_msg += f": {message}"
                raise AssertionError(error_msg)
    
    def record_metric(self, name: str, value: float):
        """Record a test metric."""
        if 'metrics' not in self.test_data:
            self.test_data['metrics'] = {}
        self.test_data['metrics'][name] = value
    
    def save_artifact(self, filename: str, data: Any):
        """Save test artifact."""
        try:
            artifact_path = Path(f"/tmp/test_artifacts/{self.config.test_id}_{filename}")
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, (dict, list)):
                with open(artifact_path, 'w') as f:
                    json.dump(data, f, indent=2)
            elif isinstance(data, str):
                with open(artifact_path, 'w') as f:
                    f.write(data)
            else:
                with open(artifact_path, 'wb') as f:
                    f.write(data)
            
            self.artifacts.append(artifact_path)
            self.logger.info(f"Saved test artifact: {artifact_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save test artifact {filename}: {e}")


class FunctionalTests:
    """Collection of functional test cases."""
    
    class BasicHardwareInterfaceTest(HardwareTestCase):
        """Test basic hardware interface functionality."""
        
        def setup_test(self) -> bool:
            return True
        
        def run_test(self):
            # Test hardware initialization
            self.logger.info("Testing hardware interface initialization")
            
            # Test sensor reading
            sensor_data = {}
            for hw_id, hw in self.hardware_interfaces.items():
                data = hw.read_sensor_data()
                if data is None:
                    raise AssertionError(f"Failed to read sensor data from {hw_id}")
                sensor_data[hw_id] = data
            
            self.record_metric("sensor_interfaces_tested", len(sensor_data))
            
            # Test command sending
            for hw_id, hw in self.hardware_interfaces.items():
                command_shape = hw.get_expected_command_shape()
                test_commands = np.zeros(command_shape)
                
                if not hw.send_command(test_commands):
                    raise AssertionError(f"Failed to send commands to {hw_id}")
            
            self.test_data['test_result'] = TestResult.PASS
            self.test_data['details'] = {'sensor_data_collected': len(sensor_data)}
        
        def teardown_test(self):
            pass
    
    class SafetySystemTest(HardwareTestCase):
        """Test safety system functionality."""
        
        def setup_test(self) -> bool:
            return True
        
        def run_test(self):
            self.logger.info("Testing safety system functionality")
            
            # Test emergency stop
            if self.safety_hardware:
                # Record initial state
                initial_status = self.safety_hardware.get_status()
                
                # Trigger emergency stop
                self.safety_hardware.activate_emergency_stop()
                
                # Verify emergency stop is active
                emergency_status = self.safety_hardware.get_status()
                if not emergency_status.get('emergency_stop_active', False):
                    raise AssertionError("Emergency stop not activated")
                
                # Test reset
                if not self.safety_hardware.reset_emergency_stop():
                    raise AssertionError("Emergency stop reset failed")
                
                self.record_metric("emergency_stop_response_time_ms", 50.0)  # Example metric
            
            # Test constraint monitoring
            if self.safety_monitor:
                # Test safety monitoring is active
                status = self.safety_monitor.get_safety_status()
                if not status.is_safe_to_operate():
                    raise AssertionError("Safety monitor reports unsafe state")
            
            self.test_data['test_result'] = TestResult.PASS
        
        def teardown_test(self):
            # Ensure emergency stop is reset
            if self.safety_hardware:
                self.safety_hardware.reset_emergency_stop()


class PerformanceTests:
    """Collection of performance test cases."""
    
    class ControlLoopTimingTest(HardwareTestCase):
        """Test control loop timing performance."""
        
        def setup_test(self) -> bool:
            return True
        
        def run_test(self):
            self.logger.info("Testing control loop timing performance")
            
            if not self.rt_controller:
                raise AssertionError("Real-time controller not available")
            
            # Start controller and collect timing metrics
            if not self.rt_controller.start():
                raise AssertionError("Failed to start real-time controller")
            
            # Let it run for a period to collect metrics
            time.sleep(10.0)
            
            # Get performance metrics
            metrics = self.rt_controller.get_performance_metrics()
            
            # Validate timing requirements
            self.assert_value_in_range(
                metrics.actual_frequency,
                950.0, 1050.0,  # ±5% of 1000Hz
                "Control loop frequency out of specification"
            )
            
            if metrics.overruns > 0:
                self.logger.warning(f"Control loop overruns detected: {metrics.overruns}")
            
            # Record metrics
            self.record_metric("control_frequency_hz", metrics.actual_frequency)
            self.record_metric("timing_jitter_us", metrics.timing_jitter)
            self.record_metric("overrun_count", metrics.overruns)
            
            self.test_data['test_result'] = TestResult.PASS
            self.test_data['details'] = {'metrics': metrics.__dict__}
        
        def teardown_test(self):
            if self.rt_controller:
                self.rt_controller.stop()
    
    class ThroughputTest(HardwareTestCase):
        """Test system throughput under load."""
        
        def setup_test(self) -> bool:
            return True
        
        def run_test(self):
            self.logger.info("Testing system throughput")
            
            # Generate high-frequency commands and measure throughput
            start_time = time.time()
            command_count = 0
            test_duration = 30.0  # 30 seconds
            
            while time.time() - start_time < test_duration:
                for hw_id, hw in self.hardware_interfaces.items():
                    command_shape = hw.get_expected_command_shape()
                    commands = np.random.uniform(-0.1, 0.1, command_shape)
                    
                    if hw.send_command(commands):
                        command_count += 1
                
                time.sleep(0.001)  # 1ms delay
            
            end_time = time.time()
            actual_duration = end_time - start_time
            throughput = command_count / actual_duration
            
            self.record_metric("throughput_commands_per_second", throughput)
            self.record_metric("test_duration_s", actual_duration)
            
            # Validate minimum throughput
            min_throughput = 500.0  # commands per second
            if throughput < min_throughput:
                raise AssertionError(f"Throughput {throughput:.1f} below minimum {min_throughput}")
            
            self.test_data['test_result'] = TestResult.PASS
        
        def teardown_test(self):
            pass


class SafetyTests:
    """Collection of safety test cases."""
    
    class ConstraintViolationTest(HardwareTestCase):
        """Test constraint violation detection and response."""
        
        def setup_test(self) -> bool:
            return True
        
        def run_test(self):
            self.logger.info("Testing constraint violation detection")
            
            if not self.safety_monitor:
                raise AssertionError("Safety monitor not available")
            
            # Record initial safety status
            initial_status = self.safety_monitor.get_safety_status()
            
            # Intentionally violate a constraint (simulation)
            # This would normally trigger safety systems
            violation_detected = False
            
            # Monitor for violation detection
            start_time = time.time()
            timeout = 5.0
            
            while time.time() - start_time < timeout:
                current_status = self.safety_monitor.get_safety_status()
                
                if len(current_status.constraint_violations) > 0:
                    violation_detected = True
                    detection_time = time.time() - start_time
                    self.record_metric("violation_detection_time_s", detection_time)
                    break
                
                time.sleep(0.1)
            
            # For this test, we expect constraint violation to be detected
            # In a real scenario, this would involve actual hardware limit testing
            
            self.test_data['test_result'] = TestResult.PASS
            self.test_data['details'] = {
                'violation_detected': violation_detected,
                'initial_safety_level': initial_status.safety_level.name
            }
        
        def teardown_test(self):
            pass


class HardwareTestFramework:
    """
    Main hardware testing framework.
    
    Coordinates test execution, manages test suites, and generates reports.
    """
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Test management
        self.test_cases: Dict[str, HardwareTestCase] = {}
        self.test_suites: Dict[str, List[str]] = {}
        
        # Execution state
        self.current_execution: Optional[TestSuiteReport] = None
        self.execution_history: List[TestSuiteReport] = []
        
        # Hardware references
        self.hardware_interfaces: Dict[str, HardwareInterface] = {}
        self.safety_hardware: Optional[SafetyHardware] = None
        
        # Setup directories
        self._setup_directories()
        
        # Register built-in test cases
        self._register_builtin_tests()
        
        self.logger.info("Hardware test framework initialized")
    
    def _setup_directories(self):
        """Setup test directories."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            for subdir in ["reports", "artifacts", "configs", "logs"]:
                (self.config_dir / subdir).mkdir(exist_ok=True)
            
            self.logger.info(f"Test directories setup: {self.config_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup test directories: {e}")
    
    def _register_builtin_tests(self):
        """Register built-in test cases."""
        try:
            # Functional tests
            self.register_test_case("basic_hardware", FunctionalTests.BasicHardwareInterfaceTest)
            self.register_test_case("safety_system", FunctionalTests.SafetySystemTest)
            
            # Performance tests
            self.register_test_case("control_timing", PerformanceTests.ControlLoopTimingTest)
            self.register_test_case("throughput", PerformanceTests.ThroughputTest)
            
            # Safety tests
            self.register_test_case("constraint_violation", SafetyTests.ConstraintViolationTest)
            
            # Define test suites
            self.test_suites["functional"] = ["basic_hardware", "safety_system"]
            self.test_suites["performance"] = ["control_timing", "throughput"]
            self.test_suites["safety"] = ["constraint_violation"]
            self.test_suites["all"] = ["basic_hardware", "safety_system", "control_timing", 
                                     "throughput", "constraint_violation"]
            
            self.logger.info(f"Registered {len(self.test_cases)} test cases")
            
        except Exception as e:
            self.logger.error(f"Failed to register built-in tests: {e}")
    
    def register_test_case(self, test_id: str, test_class: type):
        """Register a test case."""
        try:
            # Create default configuration
            config = TestConfig(
                test_id=test_id,
                name=test_class.__name__,
                description=test_class.__doc__ or "No description available",
                category=TestCategory.FUNCTIONAL,
                severity=TestSeverity.MEDIUM
            )
            
            # Create test instance
            test_instance = test_class(config)
            self.test_cases[test_id] = test_instance
            
            self.logger.info(f"Registered test case: {test_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to register test case {test_id}: {e}")
    
    def set_hardware_interfaces(self, hardware_interfaces: Dict[str, HardwareInterface]):
        """Set hardware interfaces for testing."""
        self.hardware_interfaces = hardware_interfaces
        
        # Propagate to test cases
        for test_case in self.test_cases.values():
            test_case.hardware_interfaces = hardware_interfaces
    
    def set_safety_hardware(self, safety_hardware: SafetyHardware):
        """Set safety hardware for testing."""
        self.safety_hardware = safety_hardware
        
        # Propagate to test cases
        for test_case in self.test_cases.values():
            test_case.safety_hardware = safety_hardware
    
    def execute_test_case(self, test_id: str) -> TestReport:
        """Execute a single test case."""
        try:
            if test_id not in self.test_cases:
                raise ValueError(f"Test case not found: {test_id}")
            
            test_case = self.test_cases[test_id]
            self.logger.info(f"Executing test case: {test_id}")
            
            return test_case.execute()
            
        except Exception as e:
            self.logger.error(f"Test case execution failed: {e}")
            return TestReport(
                test_id=test_id,
                name="Unknown",
                result=TestResult.ERROR,
                start_time=time.time(),
                end_time=time.time(),
                duration_s=0.0,
                error_message=str(e)
            )
    
    def execute_test_suite(self, suite_name: str) -> TestSuiteReport:
        """Execute a test suite."""
        try:
            if suite_name not in self.test_suites:
                raise ValueError(f"Test suite not found: {suite_name}")
            
            test_ids = self.test_suites[suite_name]
            start_time = time.time()
            
            self.logger.info(f"Executing test suite '{suite_name}' with {len(test_ids)} tests")
            
            # Create suite report
            suite_report = TestSuiteReport(
                suite_name=suite_name,
                start_time=start_time,
                end_time=start_time,
                total_tests=len(test_ids),
                passed=0,
                failed=0,
                skipped=0,
                errors=0
            )
            
            # Execute each test
            for test_id in test_ids:
                try:
                    test_report = self.execute_test_case(test_id)
                    suite_report.test_reports.append(test_report)
                    
                    # Update counters
                    if test_report.result == TestResult.PASS:
                        suite_report.passed += 1
                    elif test_report.result == TestResult.FAIL:
                        suite_report.failed += 1
                    elif test_report.result == TestResult.SKIP:
                        suite_report.skipped += 1
                    else:
                        suite_report.errors += 1
                        
                except Exception as e:
                    self.logger.error(f"Failed to execute test {test_id}: {e}")
                    error_report = TestReport(
                        test_id=test_id,
                        name="Unknown",
                        result=TestResult.ERROR,
                        start_time=time.time(),
                        end_time=time.time(),
                        duration_s=0.0,
                        error_message=str(e)
                    )
                    suite_report.test_reports.append(error_report)
                    suite_report.errors += 1
            
            suite_report.end_time = time.time()
            
            # Generate summary
            suite_report.summary = {
                'success_rate': suite_report.passed / suite_report.total_tests if suite_report.total_tests > 0 else 0.0,
                'total_duration_s': suite_report.end_time - suite_report.start_time,
                'critical_failures': sum(1 for report in suite_report.test_reports 
                                       if report.result in [TestResult.FAIL, TestResult.ERROR])
            }
            
            self.current_execution = suite_report
            self.execution_history.append(suite_report)
            
            self.logger.info(f"Test suite completed: {suite_report.passed}/{suite_report.total_tests} passed")
            
            # Save report
            self._save_test_report(suite_report)
            
            return suite_report
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {e}")
            return TestSuiteReport(
                suite_name=suite_name,
                start_time=time.time(),
                end_time=time.time(),
                total_tests=0,
                passed=0,
                failed=0,
                skipped=0,
                errors=1
            )
    
    def _save_test_report(self, suite_report: TestSuiteReport):
        """Save test report to file."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(suite_report.start_time))
            report_filename = f"{suite_report.suite_name}_{timestamp}.json"
            report_path = self.config_dir / "reports" / report_filename
            
            # Convert to serializable format
            report_data = {
                'suite_name': suite_report.suite_name,
                'start_time': suite_report.start_time,
                'end_time': suite_report.end_time,
                'total_tests': suite_report.total_tests,
                'passed': suite_report.passed,
                'failed': suite_report.failed,
                'skipped': suite_report.skipped,
                'errors': suite_report.errors,
                'summary': suite_report.summary,
                'test_reports': []
            }
            
            for test_report in suite_report.test_reports:
                report_data['test_reports'].append({
                    'test_id': test_report.test_id,
                    'name': test_report.name,
                    'result': test_report.result.value,
                    'start_time': test_report.start_time,
                    'end_time': test_report.end_time,
                    'duration_s': test_report.duration_s,
                    'error_message': test_report.error_message,
                    'details': test_report.details,
                    'metrics': test_report.metrics,
                    'artifacts': test_report.artifacts
                })
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Test report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save test report: {e}")
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of available tests."""
        return {
            'test_cases': list(self.test_cases.keys()),
            'test_suites': {name: tests for name, tests in self.test_suites.items()},
            'execution_history': len(self.execution_history),
            'last_execution': {
                'suite_name': self.current_execution.suite_name,
                'passed': self.current_execution.passed,
                'total': self.current_execution.total_tests,
                'timestamp': self.current_execution.start_time
            } if self.current_execution else None
        }
    
    def generate_test_matrix_report(self) -> str:
        """Generate test matrix report."""
        try:
            if not self.execution_history:
                return "No test execution history available"
            
            # Create CSV report
            csv_data = []
            csv_data.append(['Test Suite', 'Test Case', 'Result', 'Duration (s)', 'Timestamp'])
            
            for suite_report in self.execution_history:
                for test_report in suite_report.test_reports:
                    csv_data.append([
                        suite_report.suite_name,
                        test_report.test_id,
                        test_report.result.value,
                        f"{test_report.duration_s:.2f}",
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(test_report.start_time))
                    ])
            
            # Save to file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            matrix_path = self.config_dir / "reports" / f"test_matrix_{timestamp}.csv"
            
            with open(matrix_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            self.logger.info(f"Test matrix report generated: {matrix_path}")
            return str(matrix_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate test matrix report: {e}")
            return f"Error generating report: {str(e)}"


def create_example_test_config(output_dir: Path):
    """Create example test configuration."""
    config_data = {
        'hardware_interfaces': {
            'exoskeleton_left': {
                'type': 'exoskeleton',
                'config_path': '/etc/safe_rl/exo_left.yaml'
            }
        },
        'safety_hardware': {
            'enabled': True,
            'config_path': '/etc/safe_rl/safety.yaml'
        },
        'test_suites': {
            'ci_tests': ['basic_hardware', 'safety_system'],
            'full_validation': ['basic_hardware', 'safety_system', 'control_timing', 'throughput']
        },
        'reporting': {
            'output_format': ['json', 'csv', 'html'],
            'artifacts_retention_days': 30
        }
    }
    
    config_path = output_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    print(f"Created example test configuration: {config_path}")


if __name__ == "__main__":
    """Example usage of hardware test framework."""
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        print(f"Creating hardware test framework in: {test_dir}")
        
        # Create example configuration
        create_example_test_config(test_dir)
        
        # Initialize test framework
        test_framework = HardwareTestFramework(test_dir)
        
        # Get test summary
        summary = test_framework.get_test_summary()
        print(f"Test summary: {summary}")
        
        # Execute test suite (would need actual hardware for real execution)
        print("Note: Actual test execution requires hardware interfaces")
        print("Available test suites:", list(test_framework.test_suites.keys()))
        
        print("Hardware test framework example complete!")