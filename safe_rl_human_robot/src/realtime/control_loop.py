"""
Real-time Control Loop Implementation.

This module provides deterministic control loop execution with precise timing,
performance monitoring, and safety integration for robotics applications.

Key Features:
- Deterministic timing with <1ms latency
- Lock-free data structures for high performance
- Integrated safety monitoring
- Phase-aligned multi-loop coordination
- Performance metrics and diagnostics
- Hardware synchronization
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from collections import deque
import queue
import ctypes

from .timing_manager import TimingManager
from ..hardware.hardware_interface import HardwareInterface
from ..hardware.safety_hardware import SafetyHardware

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """Control loop states."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()
    SHUTDOWN = auto()


class ControlMode(Enum):
    """Control loop modes."""
    POSITION_CONTROL = auto()
    VELOCITY_CONTROL = auto()
    TORQUE_CONTROL = auto()
    IMPEDANCE_CONTROL = auto()
    HYBRID_CONTROL = auto()


@dataclass
class LoopConfig:
    """Configuration for control loop."""
    
    # Basic configuration
    loop_id: str
    frequency: float = 1000.0  # Hz
    control_mode: ControlMode = ControlMode.TORQUE_CONTROL
    
    # Timing configuration
    phase_offset_deg: float = 0.0  # Phase offset for multi-loop coordination
    timing_tolerance_us: float = 100.0  # Timing tolerance
    
    # Hardware interfaces
    hardware_interfaces: List[HardwareInterface] = field(default_factory=list)
    safety_hardware: Optional[SafetyHardware] = None
    
    # Control parameters
    control_gains: Dict[str, float] = field(default_factory=dict)
    safety_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Performance monitoring
    enable_diagnostics: bool = True
    performance_window_size: int = 1000
    
    # Advanced features
    enable_feedforward: bool = True
    enable_disturbance_observer: bool = True
    adaptive_control: bool = False
    
    # Real-time configuration
    rt_priority: int = 75
    cpu_affinity: Optional[List[int]] = None


@dataclass
class LoopMetrics:
    """Control loop performance metrics."""
    actual_frequency: float = 0.0
    cycle_time_mean_us: float = 0.0
    cycle_time_std_us: float = 0.0
    cycle_time_max_us: float = 0.0
    overrun_count: int = 0
    missed_deadlines: int = 0
    control_latency_us: float = 0.0
    sensor_latency_us: float = 0.0
    safety_violations: int = 0
    timestamp: float = field(default_factory=time.time)


class LockFreeRingBuffer:
    """Lock-free ring buffer for high-performance data exchange."""
    
    def __init__(self, size: int, element_size: int):
        self.size = size
        self.element_size = element_size
        self.buffer_size = size * element_size
        
        # Create shared memory buffer
        self.buffer = (ctypes.c_uint8 * self.buffer_size)()
        
        # Atomic counters for lock-free operation
        self.write_index = ctypes.c_size_t(0)
        self.read_index = ctypes.c_size_t(0)
        
    def write(self, data: bytes) -> bool:
        """Write data to ring buffer."""
        if len(data) != self.element_size:
            return False
        
        current_write = self.write_index.value
        current_read = self.read_index.value
        
        # Check if buffer is full
        next_write = (current_write + 1) % self.size
        if next_write == current_read:
            return False  # Buffer full
        
        # Write data
        offset = current_write * self.element_size
        ctypes.memmove(
            ctypes.addressof(self.buffer) + offset,
            data,
            self.element_size
        )
        
        # Update write index atomically
        self.write_index.value = next_write
        return True
    
    def read(self) -> Optional[bytes]:
        """Read data from ring buffer."""
        current_read = self.read_index.value
        current_write = self.write_index.value
        
        # Check if buffer is empty
        if current_read == current_write:
            return None
        
        # Read data
        offset = current_read * self.element_size
        data = bytes(self.buffer[offset:offset + self.element_size])
        
        # Update read index atomically
        self.read_index.value = (current_read + 1) % self.size
        return data


class ControlLoop:
    """
    Real-time control loop with deterministic timing and safety integration.
    
    Provides high-frequency, low-latency control execution with integrated
    safety monitoring and performance diagnostics.
    """
    
    def __init__(self, config: LoopConfig, timing_manager: TimingManager):
        self.config = config
        self.timing_manager = timing_manager
        self.logger = logging.getLogger(f"ControlLoop_{config.loop_id}")
        
        # Control loop state
        self.state = LoopState.UNINITIALIZED
        self.control_thread: Optional[threading.Thread] = None
        
        # Timing
        self.period_ns = int(1e9 / config.frequency)
        self.phase_offset_ns = int(config.phase_offset_deg * self.period_ns / 360.0)
        self.next_execution_ns = 0
        self.last_execution_ns = 0
        
        # Performance monitoring
        self.metrics = LoopMetrics()
        self.cycle_times = deque(maxlen=config.performance_window_size)
        self.overrun_count = 0
        self.missed_deadline_count = 0
        
        # Control state
        self.sensor_data: Dict[str, np.ndarray] = {}
        self.control_output: Optional[np.ndarray] = None
        self.reference_input: Optional[np.ndarray] = None
        self.error_state: Optional[np.ndarray] = None
        
        # Controllers
        self.position_controller = PositionController(config)
        self.velocity_controller = VelocityController(config)
        self.torque_controller = TorqueController(config)
        self.impedance_controller = ImpedanceController(config)
        
        # Safety monitoring
        self.safety_violations: List[str] = []
        self.emergency_stop_triggered = False
        
        # Lock-free communication
        if config.hardware_interfaces:
            sensor_size = sum(hw.get_expected_command_shape()[0] for hw in config.hardware_interfaces) * 8  # 8 bytes per float64
            self.sensor_buffer = LockFreeRingBuffer(100, sensor_size)
            self.command_buffer = LockFreeRingBuffer(100, sensor_size)
        else:
            self.sensor_buffer = None
            self.command_buffer = None
        
        # Control flags
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        
        # Callbacks
        self.pre_control_callbacks: List[Callable] = []
        self.post_control_callbacks: List[Callable] = []
        
        self.logger.info(f"Control loop initialized: {config.loop_id} @ {config.frequency}Hz")
    
    def initialize(self) -> bool:
        """Initialize control loop."""
        try:
            self.state = LoopState.INITIALIZING
            self.logger.info(f"Initializing control loop: {self.config.loop_id}")
            
            # Initialize controllers
            if not self._initialize_controllers():
                return False
            
            # Register with timing manager
            self.timing_manager.register_callback(
                f"control_loop_{self.config.loop_id}",
                self._timing_callback
            )
            
            # Validate hardware interfaces
            for hw in self.config.hardware_interfaces:
                if not hw.get_safety_status().is_safe_to_operate():
                    self.logger.error(f"Hardware not safe to operate: {hw.config.device_id}")
                    return False
            
            self.state = LoopState.READY
            self.logger.info(f"Control loop ready: {self.config.loop_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Control loop initialization failed: {e}")
            self.state = LoopState.ERROR
            return False
    
    def start(self) -> bool:
        """Start control loop execution."""
        try:
            if self.state != LoopState.READY:
                self.logger.error(f"Cannot start loop in state: {self.state}")
                return False
            
            self.logger.info(f"Starting control loop: {self.config.loop_id}")
            
            # Reset performance metrics
            self._reset_metrics()
            
            # Calculate first execution time
            current_time_ns = self.timing_manager.get_time_ns()
            self.next_execution_ns = current_time_ns + self.phase_offset_ns
            
            # Start control thread
            self.stop_flag.clear()
            self.pause_flag.clear()
            
            self.control_thread = threading.Thread(
                target=self._control_thread_main,
                name=f"ControlLoop_{self.config.loop_id}",
                daemon=True
            )
            
            # Set real-time priority
            self._configure_thread_priority()
            
            self.control_thread.start()
            
            self.state = LoopState.RUNNING
            self.logger.info(f"Control loop started: {self.config.loop_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Control loop start failed: {e}")
            self.state = LoopState.ERROR
            return False
    
    def stop(self) -> bool:
        """Stop control loop execution."""
        try:
            self.logger.info(f"Stopping control loop: {self.config.loop_id}")
            
            self.stop_flag.set()
            
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=2.0)
                if self.control_thread.is_alive():
                    self.logger.warning(f"Control loop thread did not stop gracefully: {self.config.loop_id}")
            
            # Send zero commands to hardware
            self._send_zero_commands()
            
            self.state = LoopState.READY
            self.logger.info(f"Control loop stopped: {self.config.loop_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Control loop stop failed: {e}")
            return False
    
    def pause(self):
        """Pause control loop execution."""
        self.pause_flag.set()
        self.state = LoopState.PAUSED
        self.logger.info(f"Control loop paused: {self.config.loop_id}")
    
    def resume(self):
        """Resume control loop execution."""
        self.pause_flag.clear()
        self.state = LoopState.RUNNING
        self.logger.info(f"Control loop resumed: {self.config.loop_id}")
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.emergency_stop_triggered = True
        self.state = LoopState.EMERGENCY_STOP
        self.stop_flag.set()
        self.logger.critical(f"Emergency stop triggered: {self.config.loop_id}")
        
        # Send emergency commands to hardware
        for hw in self.config.hardware_interfaces:
            hw.emergency_stop()
    
    def set_reference(self, reference: np.ndarray):
        """Set reference input for control loop."""
        self.reference_input = reference.copy()
    
    def get_state(self) -> LoopState:
        """Get current control loop state."""
        return self.state
    
    def get_actual_frequency(self) -> float:
        """Get actual execution frequency."""
        return self.metrics.actual_frequency
    
    def get_overrun_count(self) -> int:
        """Get number of timing overruns."""
        return self.overrun_count
    
    def get_average_cycle_time(self) -> float:
        """Get average cycle time in microseconds."""
        return self.metrics.cycle_time_mean_us
    
    def get_metrics(self) -> LoopMetrics:
        """Get performance metrics."""
        return self.metrics
    
    def execute_cycle(self) -> bool:
        """Execute single control cycle (for external execution)."""
        try:
            if self.state != LoopState.RUNNING:
                return False
            
            cycle_start_ns = self.timing_manager.get_time_ns()
            
            # Check for emergency stop
            if self.emergency_stop_triggered:
                return False
            
            # Execute control cycle
            success = self._execute_control_cycle(cycle_start_ns)
            
            # Update performance metrics
            cycle_end_ns = self.timing_manager.get_time_ns()
            cycle_time_us = (cycle_end_ns - cycle_start_ns) / 1000.0
            self._update_performance_metrics(cycle_time_us)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Control cycle execution failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown control loop."""
        try:
            self.logger.info(f"Shutting down control loop: {self.config.loop_id}")
            
            self.stop()
            
            # Unregister from timing manager
            self.timing_manager.unregister_callback(f"control_loop_{self.config.loop_id}")
            
            self.state = LoopState.SHUTDOWN
            self.logger.info(f"Control loop shutdown complete: {self.config.loop_id}")
            
        except Exception as e:
            self.logger.error(f"Control loop shutdown failed: {e}")
    
    def add_pre_control_callback(self, callback: Callable):
        """Add pre-control callback."""
        self.pre_control_callbacks.append(callback)
    
    def add_post_control_callback(self, callback: Callable):
        """Add post-control callback."""
        self.post_control_callbacks.append(callback)
    
    # Private implementation methods
    
    def _control_thread_main(self):
        """Main control thread loop."""
        try:
            self.logger.info(f"Control thread started: {self.config.loop_id}")
            
            while not self.stop_flag.is_set() and not self.emergency_stop_triggered:
                # Wait for next execution time
                current_time_ns = self.timing_manager.get_time_ns()
                
                if current_time_ns >= self.next_execution_ns:
                    cycle_start_ns = current_time_ns
                    
                    # Check for overrun
                    if current_time_ns > self.next_execution_ns + self.config.timing_tolerance_us * 1000:
                        self.overrun_count += 1
                        self.logger.warning(f"Control loop overrun: {self.config.loop_id}")
                    
                    # Execute control cycle
                    if not self.pause_flag.is_set():
                        self._execute_control_cycle(cycle_start_ns)
                    
                    # Update timing
                    self.last_execution_ns = cycle_start_ns
                    self.next_execution_ns += self.period_ns
                    
                    # Update performance metrics
                    cycle_end_ns = self.timing_manager.get_time_ns()
                    cycle_time_us = (cycle_end_ns - cycle_start_ns) / 1000.0
                    self._update_performance_metrics(cycle_time_us)
                
                else:
                    # Sleep until next execution time
                    self.timing_manager.sleep_until_ns(self.next_execution_ns)
                    
        except Exception as e:
            self.logger.error(f"Control thread failed: {e}")
            self.state = LoopState.ERROR
        finally:
            self.logger.info(f"Control thread terminated: {self.config.loop_id}")
    
    def _execute_control_cycle(self, timestamp_ns: int) -> bool:
        """Execute single control cycle."""
        try:
            # Execute pre-control callbacks
            for callback in self.pre_control_callbacks:
                callback(timestamp_ns)
            
            # Read sensor data
            sensor_start_ns = self.timing_manager.get_time_ns()
            if not self._read_sensor_data():
                return False
            sensor_end_ns = self.timing_manager.get_time_ns()
            
            # Check safety constraints
            if not self._check_safety_constraints():
                self.safety_violations.append("constraint_violation")
                return False
            
            # Compute control output
            control_start_ns = self.timing_manager.get_time_ns()
            if not self._compute_control_output():
                return False
            control_end_ns = self.timing_manager.get_time_ns()
            
            # Send control commands
            if not self._send_control_commands():
                return False
            
            # Update latency metrics
            self.metrics.sensor_latency_us = (sensor_end_ns - sensor_start_ns) / 1000.0
            self.metrics.control_latency_us = (control_end_ns - control_start_ns) / 1000.0
            
            # Execute post-control callbacks
            for callback in self.post_control_callbacks:
                callback(timestamp_ns)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Control cycle execution error: {e}")
            return False
    
    def _read_sensor_data(self) -> bool:
        """Read sensor data from hardware interfaces."""
        try:
            self.sensor_data.clear()
            
            for hw in self.config.hardware_interfaces:
                hw_sensor_data = hw.read_sensor_data()
                if hw_sensor_data is None:
                    self.logger.error(f"Failed to read sensor data from {hw.config.device_id}")
                    return False
                
                # Merge sensor data
                for key, value in hw_sensor_data.items():
                    self.sensor_data[f"{hw.config.device_id}_{key}"] = value
            
            return True
            
        except Exception as e:
            self.logger.error(f"Sensor data reading error: {e}")
            return False
    
    def _check_safety_constraints(self) -> bool:
        """Check safety constraints."""
        try:
            # Check hardware safety status
            for hw in self.config.hardware_interfaces:
                safety_status = hw.get_safety_status()
                if not safety_status.is_safe_to_operate():
                    self.logger.warning(f"Hardware safety violation: {hw.config.device_id}")
                    return False
            
            # Check configured safety limits
            for limit_name, (min_val, max_val) in self.config.safety_limits.items():
                if limit_name in self.sensor_data:
                    value = self.sensor_data[limit_name]
                    if np.any(value < min_val) or np.any(value > max_val):
                        self.logger.warning(f"Safety limit violated: {limit_name}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety constraint checking error: {e}")
            return False
    
    def _compute_control_output(self) -> bool:
        """Compute control output based on current mode."""
        try:
            if self.reference_input is None:
                self.control_output = np.zeros(self._get_output_size())
                return True
            
            # Select controller based on mode
            if self.config.control_mode == ControlMode.POSITION_CONTROL:
                self.control_output = self.position_controller.compute(
                    self.sensor_data, self.reference_input, self.error_state
                )
            elif self.config.control_mode == ControlMode.VELOCITY_CONTROL:
                self.control_output = self.velocity_controller.compute(
                    self.sensor_data, self.reference_input, self.error_state
                )
            elif self.config.control_mode == ControlMode.TORQUE_CONTROL:
                self.control_output = self.torque_controller.compute(
                    self.sensor_data, self.reference_input, self.error_state
                )
            elif self.config.control_mode == ControlMode.IMPEDANCE_CONTROL:
                self.control_output = self.impedance_controller.compute(
                    self.sensor_data, self.reference_input, self.error_state
                )
            else:
                self.control_output = np.zeros(self._get_output_size())
            
            # Update error state for next iteration
            if 'joint_positions' in self.sensor_data and len(self.reference_input) == len(self.sensor_data['joint_positions']):
                self.error_state = self.reference_input - self.sensor_data['joint_positions']
            
            return True
            
        except Exception as e:
            self.logger.error(f"Control computation error: {e}")
            return False
    
    def _send_control_commands(self) -> bool:
        """Send control commands to hardware."""
        try:
            if self.control_output is None:
                return False
            
            # Send commands to each hardware interface
            output_offset = 0
            for hw in self.config.hardware_interfaces:
                hw_output_size = hw.get_expected_command_shape()[0]
                hw_commands = self.control_output[output_offset:output_offset + hw_output_size]
                
                if not hw.send_command(hw_commands):
                    self.logger.error(f"Failed to send commands to {hw.config.device_id}")
                    return False
                
                output_offset += hw_output_size
            
            return True
            
        except Exception as e:
            self.logger.error(f"Control command sending error: {e}")
            return False
    
    def _send_zero_commands(self):
        """Send zero commands to all hardware."""
        try:
            for hw in self.config.hardware_interfaces:
                zero_commands = np.zeros(hw.get_expected_command_shape()[0])
                hw.send_command(zero_commands)
                
        except Exception as e:
            self.logger.error(f"Zero command sending error: {e}")
    
    def _get_output_size(self) -> int:
        """Get total output size for all hardware interfaces."""
        return sum(hw.get_expected_command_shape()[0] for hw in self.config.hardware_interfaces)
    
    def _initialize_controllers(self) -> bool:
        """Initialize all controllers."""
        try:
            self.position_controller.initialize()
            self.velocity_controller.initialize()
            self.torque_controller.initialize()
            self.impedance_controller.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Controller initialization error: {e}")
            return False
    
    def _configure_thread_priority(self):
        """Configure real-time thread priority."""
        try:
            if hasattr(self.control_thread, 'native_id') and self.control_thread.native_id:
                import os
                param = os.sched_param(self.config.rt_priority)
                os.sched_setscheduler(self.control_thread.native_id, os.SCHED_FIFO, param)
        except Exception as e:
            self.logger.warning(f"Failed to set thread priority: {e}")
    
    def _timing_callback(self, timestamp_ns: int):
        """Timing manager callback."""
        # This callback is used for synchronized execution
        pass
    
    def _reset_metrics(self):
        """Reset performance metrics."""
        self.metrics = LoopMetrics()
        self.cycle_times.clear()
        self.overrun_count = 0
        self.missed_deadline_count = 0
    
    def _update_performance_metrics(self, cycle_time_us: float):
        """Update performance metrics."""
        try:
            self.cycle_times.append(cycle_time_us)
            
            if len(self.cycle_times) >= 10:
                cycle_times_array = np.array(list(self.cycle_times))
                
                self.metrics.cycle_time_mean_us = float(np.mean(cycle_times_array))
                self.metrics.cycle_time_std_us = float(np.std(cycle_times_array))
                self.metrics.cycle_time_max_us = float(np.max(cycle_times_array))
                
                # Calculate frequency from recent cycles
                if len(self.cycle_times) > 1:
                    period_us = np.mean(np.diff(list(self.cycle_times)[-100:]))  # Last 100 cycles
                    if period_us > 0:
                        self.metrics.actual_frequency = 1e6 / period_us
                
                self.metrics.overrun_count = self.overrun_count
                self.metrics.missed_deadlines = self.missed_deadline_count
                self.metrics.safety_violations = len(self.safety_violations)
                self.metrics.timestamp = time.time()
                
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")


# Controller implementations

class BaseController:
    """Base class for control algorithms."""
    
    def __init__(self, config: LoopConfig):
        self.config = config
        self.gains = config.control_gains
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize controller."""
        self.initialized = True
        return True
    
    def compute(self, sensor_data: Dict[str, np.ndarray], 
                reference: np.ndarray, error: Optional[np.ndarray]) -> np.ndarray:
        """Compute control output."""
        raise NotImplementedError


class PositionController(BaseController):
    """PID position controller."""
    
    def __init__(self, config: LoopConfig):
        super().__init__(config)
        self.integral_error = None
        self.last_error = None
        self.dt = 1.0 / config.frequency
    
    def compute(self, sensor_data: Dict[str, np.ndarray], 
                reference: np.ndarray, error: Optional[np.ndarray]) -> np.ndarray:
        """Compute PID position control."""
        if 'joint_positions' not in sensor_data:
            return np.zeros_like(reference)
        
        current_position = sensor_data['joint_positions']
        position_error = reference - current_position
        
        # Initialize integral and derivative terms
        if self.integral_error is None:
            self.integral_error = np.zeros_like(position_error)
            self.last_error = position_error
        
        # PID computation
        kp = self.gains.get('kp', 1.0)
        ki = self.gains.get('ki', 0.0)
        kd = self.gains.get('kd', 0.0)
        
        # Proportional term
        proportional = kp * position_error
        
        # Integral term with windup protection
        self.integral_error += position_error * self.dt
        max_integral = self.gains.get('max_integral', 10.0)
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        integral = ki * self.integral_error
        
        # Derivative term
        derivative_error = (position_error - self.last_error) / self.dt
        derivative = kd * derivative_error
        
        self.last_error = position_error
        
        return proportional + integral + derivative


class VelocityController(BaseController):
    """PI velocity controller."""
    
    def __init__(self, config: LoopConfig):
        super().__init__(config)
        self.integral_error = None
        self.dt = 1.0 / config.frequency
    
    def compute(self, sensor_data: Dict[str, np.ndarray], 
                reference: np.ndarray, error: Optional[np.ndarray]) -> np.ndarray:
        """Compute PI velocity control."""
        if 'joint_velocities' not in sensor_data:
            return np.zeros_like(reference)
        
        current_velocity = sensor_data['joint_velocities']
        velocity_error = reference - current_velocity
        
        if self.integral_error is None:
            self.integral_error = np.zeros_like(velocity_error)
        
        # PI computation
        kp = self.gains.get('kp_vel', 1.0)
        ki = self.gains.get('ki_vel', 0.1)
        
        proportional = kp * velocity_error
        
        self.integral_error += velocity_error * self.dt
        max_integral = self.gains.get('max_integral_vel', 5.0)
        self.integral_error = np.clip(self.integral_error, -max_integral, max_integral)
        integral = ki * self.integral_error
        
        return proportional + integral


class TorqueController(BaseController):
    """Direct torque controller."""
    
    def compute(self, sensor_data: Dict[str, np.ndarray], 
                reference: np.ndarray, error: Optional[np.ndarray]) -> np.ndarray:
        """Direct torque control."""
        # Apply torque limits
        max_torque = self.gains.get('max_torque', 50.0)
        return np.clip(reference, -max_torque, max_torque)


class ImpedanceController(BaseController):
    """Impedance/admittance controller."""
    
    def __init__(self, config: LoopConfig):
        super().__init__(config)
        self.dt = 1.0 / config.frequency
    
    def compute(self, sensor_data: Dict[str, np.ndarray], 
                reference: np.ndarray, error: Optional[np.ndarray]) -> np.ndarray:
        """Compute impedance control."""
        if ('joint_positions' not in sensor_data or 
            'joint_velocities' not in sensor_data or
            'forces' not in sensor_data):
            return np.zeros_like(reference)
        
        position = sensor_data['joint_positions']
        velocity = sensor_data['joint_velocities']
        force = sensor_data['forces']
        
        # Impedance parameters
        stiffness = self.gains.get('stiffness', 100.0)
        damping = self.gains.get('damping', 10.0)
        
        # Compute desired force based on impedance model
        position_error = reference - position
        desired_force = stiffness * position_error - damping * velocity
        
        # Force error compensation
        force_error = desired_force - force
        force_gain = self.gains.get('force_gain', 0.1)
        
        return force_gain * force_error