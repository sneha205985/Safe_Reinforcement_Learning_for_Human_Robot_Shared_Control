"""
Abstract hardware interface for Safe RL robotic systems.

This module defines the base hardware interface that all robot-specific
implementations must follow for production deployment.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import numpy as np
import time
import threading
import logging
from pathlib import Path
import json
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyLevel(IntEnum):
    """Safety level enumeration for hardware status."""
    CRITICAL_FAILURE = 0    # Immediate emergency stop required
    MAJOR_WARNING = 1       # Significant safety concern, reduce capabilities
    MINOR_WARNING = 2       # Minor issue, log and monitor
    NORMAL = 3             # Normal operation
    OPTIMAL = 4            # All systems optimal


class HardwareState(Enum):
    """Hardware system states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    CALIBRATING = "calibrating"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    SHUTDOWN = "shutdown"


@dataclass
class SafetyStatus:
    """Safety status information from hardware systems."""
    safety_level: SafetyLevel = SafetyLevel.NORMAL
    emergency_stop_active: bool = False
    constraint_violations: List[str] = field(default_factory=list)
    hardware_faults: List[str] = field(default_factory=list)
    last_update_time: float = field(default_factory=time.time)
    diagnostic_data: Dict[str, Any] = field(default_factory=dict)
    
    def is_safe_to_operate(self) -> bool:
        """Check if it's safe to continue operation."""
        return (self.safety_level >= SafetyLevel.MINOR_WARNING and 
                not self.emergency_stop_active and
                len(self.hardware_faults) == 0)
    
    def get_safety_score(self) -> float:
        """Get normalized safety score (0-1)."""
        base_score = float(self.safety_level) / 4.0
        
        # Penalties for violations and faults
        violation_penalty = len(self.constraint_violations) * 0.1
        fault_penalty = len(self.hardware_faults) * 0.2
        
        # Emergency stop override
        if self.emergency_stop_active:
            return 0.0
            
        return max(0.0, base_score - violation_penalty - fault_penalty)


@dataclass
class HardwareConfig:
    """Configuration for hardware interface."""
    device_id: str
    device_type: str
    communication_protocol: str = "serial"
    communication_params: Dict[str, Any] = field(default_factory=dict)
    sampling_rate: float = 1000.0  # Hz
    control_rate: float = 1000.0   # Hz
    safety_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 10
    retry_attempts: int = 3
    enable_watchdog: bool = True
    watchdog_timeout_ms: int = 100
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'HardwareConfig':
        """Load configuration from file."""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                data = json.load(f)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**data)
    
    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_path)
        
        data = {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'communication_protocol': self.communication_protocol,
            'communication_params': self.communication_params,
            'sampling_rate': self.sampling_rate,
            'control_rate': self.control_rate,
            'safety_limits': self.safety_limits,
            'calibration_params': self.calibration_params,
            'timeout_ms': self.timeout_ms,
            'retry_attempts': self.retry_attempts,
            'enable_watchdog': self.enable_watchdog,
            'watchdog_timeout_ms': self.watchdog_timeout_ms
        }
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2)
            elif config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")


class HardwareInterface(ABC):
    """
    Abstract base class for all hardware interfaces in the Safe RL system.
    
    This class defines the standard interface that all robot-specific
    hardware implementations must follow for production deployment.
    """
    
    def __init__(self, config: HardwareConfig):
        """
        Initialize hardware interface.
        
        Args:
            config: Hardware configuration parameters
        """
        self.config = config
        self.state = HardwareState.UNINITIALIZED
        self.safety_status = SafetyStatus()
        self.last_command_time = 0.0
        self.last_sensor_update = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Monitoring and diagnostics
        self._diagnostic_thread = None
        self._watchdog_thread = None
        self._performance_metrics = {
            'command_latency': [],
            'sensor_latency': [],
            'control_frequency': [],
            'dropped_commands': 0,
            'communication_errors': 0
        }
        
        # Safety monitoring
        self._safety_violations = []
        self._hardware_faults = []
        
        logger.info(f"Hardware interface initialized for {config.device_type} ({config.device_id})")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def initialize(self) -> bool:
        """
        Initialize hardware system and start monitoring threads.
        
        Returns:
            True if initialization successful, False otherwise
        """
        with self._lock:
            if self.state != HardwareState.UNINITIALIZED:
                logger.warning(f"Hardware already initialized (state: {self.state})")
                return self.state == HardwareState.READY
            
            self.state = HardwareState.INITIALIZING
            logger.info(f"Initializing hardware interface for {self.config.device_id}")
            
            try:
                # Initialize hardware-specific components
                if not self.initialize_hardware():
                    self.state = HardwareState.ERROR
                    logger.error("Hardware initialization failed")
                    return False
                
                # Start monitoring threads
                self._start_monitoring_threads()
                
                # Perform calibration if required
                if self.config.calibration_params:
                    self.state = HardwareState.CALIBRATING
                    if not self.calibrate():
                        self.state = HardwareState.ERROR
                        logger.error("Hardware calibration failed")
                        return False
                
                self.state = HardwareState.READY
                logger.info("Hardware interface ready")
                return True
                
            except Exception as e:
                self.state = HardwareState.ERROR
                logger.error(f"Hardware initialization error: {str(e)}")
                return False
    
    def shutdown(self):
        """Safely shutdown hardware interface."""
        with self._lock:
            if self.state == HardwareState.SHUTDOWN:
                return
            
            logger.info("Shutting down hardware interface")
            self.state = HardwareState.SHUTDOWN
            self._shutdown_event.set()
            
            # Stop monitoring threads
            if self._diagnostic_thread and self._diagnostic_thread.is_alive():
                self._diagnostic_thread.join(timeout=2.0)
            
            if self._watchdog_thread and self._watchdog_thread.is_alive():
                self._watchdog_thread.join(timeout=2.0)
            
            # Hardware-specific shutdown
            try:
                self.shutdown_hardware()
            except Exception as e:
                logger.error(f"Hardware shutdown error: {str(e)}")
            
            logger.info("Hardware interface shutdown complete")
    
    def send_command(self, commands: np.ndarray, timeout_ms: Optional[int] = None) -> bool:
        """
        Send commands to hardware with safety checking.
        
        Args:
            commands: Control commands to send
            timeout_ms: Timeout in milliseconds (default: use config timeout)
            
        Returns:
            True if commands sent successfully, False otherwise
        """
        if timeout_ms is None:
            timeout_ms = self.config.timeout_ms
        
        with self._lock:
            # Check system state
            if self.state not in [HardwareState.READY, HardwareState.RUNNING]:
                logger.warning(f"Cannot send commands in state {self.state}")
                return False
            
            # Safety check
            if not self.safety_status.is_safe_to_operate():
                logger.warning("Unsafe to operate - blocking commands")
                self._performance_metrics['dropped_commands'] += 1
                return False
            
            # Validate commands
            if not self._validate_commands(commands):
                logger.warning("Command validation failed")
                self._performance_metrics['dropped_commands'] += 1
                return False
            
            # Record timing
            start_time = time.perf_counter()
            
            try:
                # Send commands to hardware
                success = self.send_commands(commands)
                
                if success:
                    self.last_command_time = time.time()
                    self.state = HardwareState.RUNNING
                    
                    # Record performance metrics
                    latency = (time.perf_counter() - start_time) * 1000  # ms
                    self._performance_metrics['command_latency'].append(latency)
                    
                    # Keep only recent metrics (sliding window)
                    if len(self._performance_metrics['command_latency']) > 1000:
                        self._performance_metrics['command_latency'] = \
                            self._performance_metrics['command_latency'][-1000:]
                
                return success
                
            except Exception as e:
                logger.error(f"Command sending error: {str(e)}")
                self._performance_metrics['communication_errors'] += 1
                return False
    
    def read_sensors(self, timeout_ms: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Read sensor data from hardware.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Dictionary of sensor data, None if read failed
        """
        if timeout_ms is None:
            timeout_ms = self.config.timeout_ms
        
        with self._lock:
            if self.state not in [HardwareState.READY, HardwareState.RUNNING]:
                return None
            
            start_time = time.perf_counter()
            
            try:
                sensor_data = self.read_sensor_data()
                
                if sensor_data is not None:
                    self.last_sensor_update = time.time()
                    
                    # Record performance metrics
                    latency = (time.perf_counter() - start_time) * 1000  # ms
                    self._performance_metrics['sensor_latency'].append(latency)
                    
                    # Keep sliding window
                    if len(self._performance_metrics['sensor_latency']) > 1000:
                        self._performance_metrics['sensor_latency'] = \
                            self._performance_metrics['sensor_latency'][-1000:]
                    
                    # Update safety status based on sensor data
                    self._update_safety_status(sensor_data)
                
                return sensor_data
                
            except Exception as e:
                logger.error(f"Sensor reading error: {str(e)}")
                self._performance_metrics['communication_errors'] += 1
                return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get hardware performance metrics."""
        with self._lock:
            metrics = {}
            
            if self._performance_metrics['command_latency']:
                cmd_latencies = np.array(self._performance_metrics['command_latency'])
                metrics['command_latency'] = {
                    'mean_ms': float(np.mean(cmd_latencies)),
                    'max_ms': float(np.max(cmd_latencies)),
                    'std_ms': float(np.std(cmd_latencies)),
                    'p95_ms': float(np.percentile(cmd_latencies, 95))
                }
            
            if self._performance_metrics['sensor_latency']:
                sensor_latencies = np.array(self._performance_metrics['sensor_latency'])
                metrics['sensor_latency'] = {
                    'mean_ms': float(np.mean(sensor_latencies)),
                    'max_ms': float(np.max(sensor_latencies)),
                    'std_ms': float(np.std(sensor_latencies)),
                    'p95_ms': float(np.percentile(sensor_latencies, 95))
                }
            
            metrics['dropped_commands'] = self._performance_metrics['dropped_commands']
            metrics['communication_errors'] = self._performance_metrics['communication_errors']
            metrics['uptime_seconds'] = time.time() - getattr(self, '_start_time', time.time())
            
            return metrics
    
    def emergency_stop(self) -> None:
        """Immediately stop all hardware motion."""
        with self._lock:
            logger.critical("EMERGENCY STOP ACTIVATED")
            self.state = HardwareState.EMERGENCY_STOP
            self.safety_status.emergency_stop_active = True
            self.safety_status.safety_level = SafetyLevel.CRITICAL_FAILURE
            
            try:
                self.execute_emergency_stop()
            except Exception as e:
                logger.critical(f"Emergency stop execution failed: {str(e)}")
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop if conditions are safe."""
        with self._lock:
            if not self.safety_status.emergency_stop_active:
                return True
            
            logger.info("Attempting to reset emergency stop")
            
            # Check if it's safe to reset
            if not self._check_safe_to_reset():
                logger.warning("Not safe to reset emergency stop")
                return False
            
            try:
                if self.reset_hardware_emergency_stop():
                    self.safety_status.emergency_stop_active = False
                    self.safety_status.safety_level = SafetyLevel.MINOR_WARNING
                    self.state = HardwareState.READY
                    logger.info("Emergency stop reset successfully")
                    return True
                else:
                    logger.error("Failed to reset hardware emergency stop")
                    return False
            except Exception as e:
                logger.error(f"Error resetting emergency stop: {str(e)}")
                return False
    
    def _start_monitoring_threads(self):
        """Start background monitoring threads."""
        self._start_time = time.time()
        
        # Start diagnostic monitoring thread
        self._diagnostic_thread = threading.Thread(
            target=self._diagnostic_monitor,
            name=f"diagnostic_{self.config.device_id}",
            daemon=True
        )
        self._diagnostic_thread.start()
        
        # Start watchdog thread if enabled
        if self.config.enable_watchdog:
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_monitor,
                name=f"watchdog_{self.config.device_id}",
                daemon=True
            )
            self._watchdog_thread.start()
    
    def _diagnostic_monitor(self):
        """Background thread for system diagnostics."""
        logger.info("Diagnostic monitoring started")
        
        while not self._shutdown_event.is_set():
            try:
                # Update diagnostic data
                with self._lock:
                    self.safety_status.diagnostic_data = self._collect_diagnostic_data()
                    self.safety_status.last_update_time = time.time()
                
                # Sleep for diagnostic interval
                self._shutdown_event.wait(0.1)  # 10Hz diagnostic rate
                
            except Exception as e:
                logger.error(f"Diagnostic monitoring error: {str(e)}")
                self._shutdown_event.wait(1.0)
    
    def _watchdog_monitor(self):
        """Background watchdog for detecting communication failures."""
        logger.info("Watchdog monitoring started")
        watchdog_timeout = self.config.watchdog_timeout_ms / 1000.0
        
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Check for communication timeouts
                if (self.state == HardwareState.RUNNING and 
                    current_time - self.last_command_time > watchdog_timeout):
                    logger.warning("Watchdog timeout - no recent commands")
                    with self._lock:
                        self.safety_status.safety_level = SafetyLevel.MAJOR_WARNING
                        self._hardware_faults.append("watchdog_timeout")
                
                if (self.state == HardwareState.RUNNING and 
                    current_time - self.last_sensor_update > watchdog_timeout):
                    logger.warning("Sensor timeout - no recent data")
                    with self._lock:
                        self.safety_status.safety_level = SafetyLevel.MAJOR_WARNING
                        self._hardware_faults.append("sensor_timeout")
                
                self._shutdown_event.wait(watchdog_timeout / 2)
                
            except Exception as e:
                logger.error(f"Watchdog monitoring error: {str(e)}")
                self._shutdown_event.wait(1.0)
    
    def _validate_commands(self, commands: np.ndarray) -> bool:
        """Validate commands against safety limits."""
        try:
            # Check command dimensions
            if commands.shape != self.get_expected_command_shape():
                logger.warning(f"Invalid command shape: {commands.shape}")
                return False
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(commands)):
                logger.warning("Commands contain NaN or infinite values")
                return False
            
            # Check safety limits
            for i, (cmd_name, (min_val, max_val)) in enumerate(self.config.safety_limits.items()):
                if i < len(commands):
                    if not (min_val <= commands[i] <= max_val):
                        logger.warning(f"Command {cmd_name} out of safety limits: {commands[i]} not in [{min_val}, {max_val}]")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Command validation error: {str(e)}")
            return False
    
    def _update_safety_status(self, sensor_data: Dict[str, np.ndarray]):
        """Update safety status based on sensor data."""
        try:
            # Check for constraint violations
            violations = self.check_constraint_violations(sensor_data)
            
            # Update safety status
            if violations:
                self.safety_status.constraint_violations = violations
                if len(violations) > 2:
                    self.safety_status.safety_level = SafetyLevel.MAJOR_WARNING
                else:
                    self.safety_status.safety_level = SafetyLevel.MINOR_WARNING
            else:
                self.safety_status.constraint_violations = []
                if not self._hardware_faults:
                    self.safety_status.safety_level = SafetyLevel.OPTIMAL
            
            # Update hardware faults list
            self.safety_status.hardware_faults = list(self._hardware_faults)
            
        except Exception as e:
            logger.error(f"Safety status update error: {str(e)}")
    
    def _collect_diagnostic_data(self) -> Dict[str, Any]:
        """Collect diagnostic data from hardware."""
        try:
            diagnostics = {
                'uptime_seconds': time.time() - self._start_time,
                'state': self.state.value,
                'last_command_age_ms': (time.time() - self.last_command_time) * 1000,
                'last_sensor_age_ms': (time.time() - self.last_sensor_update) * 1000,
                'performance_metrics': self.get_performance_metrics()
            }
            
            # Add hardware-specific diagnostics
            hw_diagnostics = self.get_hardware_diagnostics()
            diagnostics.update(hw_diagnostics)
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Diagnostic data collection error: {str(e)}")
            return {}
    
    def _check_safe_to_reset(self) -> bool:
        """Check if conditions are safe to reset from emergency stop."""
        # Override in subclasses for hardware-specific safety checks
        return True
    
    # Abstract methods to be implemented by subclasses
    
    @abstractmethod
    def initialize_hardware(self) -> bool:
        """Initialize hardware-specific components."""
        pass
    
    @abstractmethod
    def shutdown_hardware(self) -> None:
        """Shutdown hardware-specific components."""
        pass
    
    @abstractmethod
    def send_commands(self, commands: np.ndarray) -> bool:
        """Send commands to hardware."""
        pass
    
    @abstractmethod
    def read_sensor_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Read sensor data from hardware."""
        pass
    
    @abstractmethod
    def execute_emergency_stop(self) -> None:
        """Execute emergency stop procedure."""
        pass
    
    @abstractmethod
    def reset_hardware_emergency_stop(self) -> bool:
        """Reset hardware from emergency stop state."""
        pass
    
    @abstractmethod
    def get_expected_command_shape(self) -> Tuple[int, ...]:
        """Get expected shape of command array."""
        pass
    
    @abstractmethod
    def check_constraint_violations(self, sensor_data: Dict[str, np.ndarray]) -> List[str]:
        """Check for constraint violations in sensor data."""
        pass
    
    @abstractmethod
    def get_hardware_diagnostics(self) -> Dict[str, Any]:
        """Get hardware-specific diagnostic information."""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Perform hardware calibration."""
        pass