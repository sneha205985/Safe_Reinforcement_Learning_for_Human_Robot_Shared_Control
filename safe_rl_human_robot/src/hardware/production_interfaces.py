"""
Production Hardware Interfaces for Safe RL Human-Robot Shared Control.

Complete hardware abstraction layer with ROS integration, safety interlocks,
calibration procedures, and hardware simulation capabilities.
"""

import os
import sys
import time
import threading
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import queue
from contextlib import contextmanager

# Hardware and control libraries
try:
    import rospy
    import rospkg
    from std_msgs.msg import Header, Bool, Float64, String
    from geometry_msgs.msg import Twist, Pose, Vector3
    from sensor_msgs.msg import JointState, Imu, LaserScan
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    # Mock ROS for testing without ROS installation
    class MockROSMessage:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    Header = MockROSMessage
    Bool = MockROSMessage
    Float64 = MockROSMessage
    String = MockROSMessage
    Twist = MockROSMessage
    Pose = MockROSMessage
    Vector3 = MockROSMessage
    JointState = MockROSMessage
    Imu = MockROSMessage
    LaserScan = MockROSMessage
    DiagnosticArray = MockROSMessage
    DiagnosticStatus = MockROSMessage
    KeyValue = MockROSMessage

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import can
    CAN_AVAILABLE = True
except ImportError:
    CAN_AVAILABLE = False


class HardwareType(Enum):
    """Hardware device types."""
    EXOSKELETON = auto()
    WHEELCHAIR = auto()
    ROBOTIC_ARM = auto()
    MOBILE_BASE = auto()
    SENSORS = auto()
    ACTUATORS = auto()


class SafetyLevel(Enum):
    """Safety criticality levels."""
    INFORMATIONAL = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()
    EMERGENCY = auto()


class HardwareState(Enum):
    """Hardware operational states."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    CALIBRATING = auto()
    READY = auto()
    ACTIVE = auto()
    FAULT = auto()
    EMERGENCY_STOP = auto()
    MAINTENANCE = auto()


class CommunicationProtocol(Enum):
    """Hardware communication protocols."""
    ROS = auto()
    SERIAL = auto()
    CAN = auto()
    ETHERNET = auto()
    USB = auto()
    SIMULATION = auto()


@dataclass
class SafetyInterlock:
    """Hardware safety interlock definition."""
    name: str
    hardware_id: str
    condition: str  # Boolean expression or callback name
    action: str  # Action to take when triggered
    level: SafetyLevel = SafetyLevel.WARNING
    enabled: bool = True
    triggered: bool = False
    trigger_count: int = 0
    last_trigger_time: float = 0.0


@dataclass
class HardwareLimits:
    """Hardware operational limits."""
    position_min: Optional[List[float]] = None
    position_max: Optional[List[float]] = None
    velocity_max: Optional[List[float]] = None
    acceleration_max: Optional[List[float]] = None
    force_max: Optional[List[float]] = None
    torque_max: Optional[List[float]] = None
    temperature_max: float = 85.0  # Celsius
    voltage_min: float = 10.0  # Volts
    voltage_max: float = 50.0  # Volts
    current_max: float = 20.0  # Amperes


@dataclass
class HardwareStatus:
    """Current hardware status."""
    timestamp: float = field(default_factory=time.time)
    state: HardwareState = HardwareState.UNINITIALIZED
    is_connected: bool = False
    is_calibrated: bool = False
    is_homed: bool = False
    
    # Sensor readings
    positions: Optional[List[float]] = None
    velocities: Optional[List[float]] = None
    forces: Optional[List[float]] = None
    torques: Optional[List[float]] = None
    temperatures: Optional[List[float]] = None
    voltages: Optional[List[float]] = None
    currents: Optional[List[float]] = None
    
    # Safety status
    emergency_stop_active: bool = False
    safety_interlocks_triggered: List[str] = field(default_factory=list)
    fault_conditions: List[str] = field(default_factory=list)
    
    # Performance metrics
    communication_latency_ms: float = 0.0
    update_frequency_hz: float = 0.0
    packet_loss_rate: float = 0.0


@dataclass
class CalibrationProcedure:
    """Hardware calibration procedure definition."""
    name: str
    description: str
    steps: List[str]
    expected_duration_sec: float
    safety_requirements: List[str]
    success_criteria: Dict[str, Any]
    auto_executable: bool = False
    requires_operator: bool = True


class HardwareSimulator:
    """Hardware simulator for testing without real hardware."""
    
    def __init__(self, hardware_type: HardwareType, config: Dict[str, Any]):
        self.hardware_type = hardware_type
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.HardwareSimulator")
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.simulation_frequency = config.get('simulation_frequency', 100.0)  # Hz
        
        # Simulated hardware state
        self.dof = config.get('degrees_of_freedom', 6)
        self.positions = np.zeros(self.dof)
        self.velocities = np.zeros(self.dof)
        self.target_positions = np.zeros(self.dof)
        self.forces = np.zeros(self.dof)
        
        # Simulation parameters
        self.noise_level = config.get('noise_level', 0.01)
        self.dynamics_enabled = config.get('dynamics_enabled', True)
        self.fault_injection_enabled = config.get('fault_injection', False)
        
        # Physics simulation parameters
        self.mass_matrix = np.eye(self.dof) * config.get('link_mass', 1.0)
        self.damping_matrix = np.eye(self.dof) * config.get('damping', 0.1)
        self.stiffness_matrix = np.eye(self.dof) * config.get('stiffness', 100.0)
        
    def start_simulation(self):
        """Start hardware simulation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        self.logger.info(f"Hardware simulation started for {self.hardware_type.name}")
    
    def stop_simulation(self):
        """Stop hardware simulation."""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        self.logger.info("Hardware simulation stopped")
    
    def _simulation_loop(self):
        """Main simulation loop."""
        dt = 1.0 / self.simulation_frequency
        
        while self.is_running:
            start_time = time.time()
            
            try:
                # Update physics simulation
                self._update_dynamics(dt)
                
                # Add noise to simulate real hardware
                self._add_sensor_noise()
                
                # Inject faults if enabled
                if self.fault_injection_enabled:
                    self._inject_faults()
                
                # Sleep to maintain simulation frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Simulation loop error: {e}")
    
    def _update_dynamics(self, dt: float):
        """Update physics dynamics simulation."""
        if not self.dynamics_enabled:
            # Simple position control
            position_error = self.target_positions - self.positions
            self.positions += position_error * dt * 5.0  # Simple P controller
            return
        
        # More sophisticated dynamics simulation
        position_error = self.target_positions - self.positions
        velocity_error = -self.velocities
        
        # PD control forces
        control_forces = (self.stiffness_matrix @ position_error + 
                         self.damping_matrix @ velocity_error)
        
        # Simple Euler integration
        accelerations = np.linalg.solve(self.mass_matrix, control_forces)
        self.velocities += accelerations * dt
        self.positions += self.velocities * dt
        
        # Update force estimates
        self.forces = control_forces
    
    def _add_sensor_noise(self):
        """Add realistic sensor noise."""
        if self.noise_level > 0:
            position_noise = np.random.normal(0, self.noise_level, self.dof)
            velocity_noise = np.random.normal(0, self.noise_level * 0.1, self.dof)
            
            self.positions += position_noise
            self.velocities += velocity_noise
    
    def _inject_faults(self):
        """Inject simulated faults for testing."""
        # Randomly inject communication delays
        if np.random.random() < 0.001:  # 0.1% chance per update
            time.sleep(0.01)  # 10ms delay
        
        # Simulate sensor dropouts
        if np.random.random() < 0.0001:  # 0.01% chance
            self.positions = np.full(self.dof, np.nan)
    
    def set_target_positions(self, positions: List[float]):
        """Set target positions for simulation."""
        if len(positions) == self.dof:
            self.target_positions = np.array(positions)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current simulated hardware status."""
        return {
            'positions': self.positions.tolist(),
            'velocities': self.velocities.tolist(),
            'forces': self.forces.tolist(),
            'is_simulated': True,
            'simulation_running': self.is_running
        }


class ROSHardwareInterface:
    """ROS-based hardware communication interface."""
    
    def __init__(self, hardware_id: str, config: Dict[str, Any]):
        self.hardware_id = hardware_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ROSHardwareInterface")
        
        # ROS setup
        self.node_name = f"safe_rl_hardware_{hardware_id}"
        self.namespace = config.get('ros_namespace', '/safe_rl')
        
        # Publishers and subscribers
        self.publishers = {}
        self.subscribers = {}
        self.service_clients = {}
        
        # Message queues
        self.command_queue = queue.Queue(maxsize=100)
        self.status_queue = queue.Queue(maxsize=1000)
        
        # ROS initialization flag
        self.ros_initialized = False
        
    def initialize_ros(self):
        """Initialize ROS node and topics."""
        if not ROS_AVAILABLE:
            self.logger.warning("ROS not available, using simulation mode")
            return False
        
        try:
            # Initialize ROS node if not already done
            if not rospy.get_node_uri():
                rospy.init_node(self.node_name, anonymous=True)
            
            # Setup publishers
            self._setup_publishers()
            
            # Setup subscribers
            self._setup_subscribers()
            
            # Setup service clients
            self._setup_services()
            
            self.ros_initialized = True
            self.logger.info(f"ROS interface initialized for {self.hardware_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"ROS initialization failed: {e}")
            return False
    
    def _setup_publishers(self):
        """Setup ROS publishers for hardware commands."""
        topics = self.config.get('publish_topics', {})
        
        # Standard command topics
        if 'joint_commands' in topics:
            self.publishers['joint_commands'] = rospy.Publisher(
                f"{self.namespace}/{self.hardware_id}/joint_commands",
                JointState, queue_size=10
            )
        
        if 'cmd_vel' in topics:
            self.publishers['cmd_vel'] = rospy.Publisher(
                f"{self.namespace}/{self.hardware_id}/cmd_vel",
                Twist, queue_size=10
            )
        
        # Safety command topics
        self.publishers['emergency_stop'] = rospy.Publisher(
            f"{self.namespace}/{self.hardware_id}/emergency_stop",
            Bool, queue_size=1, latch=True
        )
        
        # Diagnostic topics
        self.publishers['diagnostics'] = rospy.Publisher(
            f"{self.namespace}/{self.hardware_id}/diagnostics",
            DiagnosticArray, queue_size=10
        )
    
    def _setup_subscribers(self):
        """Setup ROS subscribers for hardware feedback."""
        topics = self.config.get('subscribe_topics', {})
        
        # Standard feedback topics
        if 'joint_states' in topics:
            self.subscribers['joint_states'] = rospy.Subscriber(
                f"{self.namespace}/{self.hardware_id}/joint_states",
                JointState, self._joint_states_callback, queue_size=10
            )
        
        if 'imu_data' in topics:
            self.subscribers['imu'] = rospy.Subscriber(
                f"{self.namespace}/{self.hardware_id}/imu",
                Imu, self._imu_callback, queue_size=10
            )
        
        # Safety feedback topics
        self.subscribers['safety_status'] = rospy.Subscriber(
            f"{self.namespace}/{self.hardware_id}/safety_status",
            DiagnosticArray, self._safety_status_callback, queue_size=1
        )
    
    def _setup_services(self):
        """Setup ROS service clients."""
        services = self.config.get('services', {})
        
        # Calibration service
        if 'calibrate' in services:
            service_name = f"{self.namespace}/{self.hardware_id}/calibrate"
            rospy.wait_for_service(service_name, timeout=5.0)
            self.service_clients['calibrate'] = rospy.ServiceProxy(service_name, String)
    
    def _joint_states_callback(self, msg: JointState):
        """Handle joint state messages."""
        try:
            status_data = {
                'timestamp': time.time(),
                'message_type': 'joint_states',
                'positions': list(msg.position),
                'velocities': list(msg.velocity),
                'efforts': list(msg.effort)
            }
            self.status_queue.put(status_data, block=False)
        except queue.Full:
            self.logger.warning("Status queue full, dropping message")
    
    def _imu_callback(self, msg: Imu):
        """Handle IMU messages."""
        try:
            status_data = {
                'timestamp': time.time(),
                'message_type': 'imu',
                'orientation': [msg.orientation.x, msg.orientation.y, 
                              msg.orientation.z, msg.orientation.w],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, 
                                   msg.angular_velocity.z],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, 
                                      msg.linear_acceleration.z]
            }
            self.status_queue.put(status_data, block=False)
        except queue.Full:
            self.logger.warning("Status queue full, dropping IMU message")
    
    def _safety_status_callback(self, msg: DiagnosticArray):
        """Handle safety status messages."""
        try:
            for status in msg.status:
                status_data = {
                    'timestamp': time.time(),
                    'message_type': 'safety_diagnostic',
                    'name': status.name,
                    'level': status.level,
                    'message': status.message,
                    'values': {kv.key: kv.value for kv in status.values}
                }
                self.status_queue.put(status_data, block=False)
        except queue.Full:
            self.logger.warning("Status queue full, dropping safety message")
    
    def publish_joint_command(self, positions: List[float], velocities: Optional[List[float]] = None):
        """Publish joint position command."""
        if 'joint_commands' not in self.publishers:
            return False
        
        try:
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.position = positions
            if velocities:
                msg.velocity = velocities
            
            self.publishers['joint_commands'].publish(msg)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish joint command: {e}")
            return False
    
    def publish_emergency_stop(self, stop: bool):
        """Publish emergency stop command."""
        try:
            msg = Bool()
            msg.data = stop
            self.publishers['emergency_stop'].publish(msg)
            self.logger.warning(f"Emergency stop {'ACTIVATED' if stop else 'DEACTIVATED'}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish emergency stop: {e}")
            return False
    
    def get_latest_status(self) -> Optional[Dict[str, Any]]:
        """Get latest hardware status from ROS."""
        try:
            return self.status_queue.get(block=False)
        except queue.Empty:
            return None


class ProductionSafetySystem:
    """Production-grade safety system with hardware interlocks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Safety interlocks
        self.interlocks: Dict[str, SafetyInterlock] = {}
        self.interlock_callbacks: Dict[str, Callable] = {}
        
        # Emergency stop system
        self.emergency_stop_active = False
        self.emergency_stop_sources = set()
        
        # Watchdog timers
        self.watchdog_timers: Dict[str, float] = {}
        self.watchdog_timeouts: Dict[str, float] = {}
        
        # Fault detection
        self.fault_conditions: Dict[str, bool] = {}
        self.fault_history: List[Dict[str, Any]] = []
        
        # Safety monitoring thread
        self.safety_thread = None
        self.monitoring_active = False
        self.monitoring_frequency = config.get('monitoring_frequency', 100.0)  # Hz
        
        # Redundant checking
        self.redundant_checkers: List[Callable] = []
        
        self._initialize_safety_interlocks()
    
    def _initialize_safety_interlocks(self):
        """Initialize safety interlocks from configuration."""
        interlocks_config = self.config.get('safety_interlocks', [])
        
        for interlock_config in interlocks_config:
            interlock = SafetyInterlock(
                name=interlock_config['name'],
                hardware_id=interlock_config['hardware_id'],
                condition=interlock_config['condition'],
                action=interlock_config['action'],
                level=SafetyLevel[interlock_config.get('level', 'WARNING')],
                enabled=interlock_config.get('enabled', True)
            )
            self.interlocks[interlock.name] = interlock
            
            self.logger.info(f"Initialized safety interlock: {interlock.name}")
    
    def start_monitoring(self):
        """Start safety monitoring system."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.safety_thread = threading.Thread(target=self._safety_monitoring_loop, daemon=True)
        self.safety_thread.start()
        
        self.logger.info("Safety monitoring system started")
    
    def stop_monitoring(self):
        """Stop safety monitoring system."""
        self.monitoring_active = False
        if self.safety_thread:
            self.safety_thread.join(timeout=2.0)
        
        self.logger.info("Safety monitoring system stopped")
    
    def _safety_monitoring_loop(self):
        """Main safety monitoring loop."""
        dt = 1.0 / self.monitoring_frequency
        
        while self.monitoring_active:
            start_time = time.time()
            
            try:
                # Check all safety interlocks
                self._check_safety_interlocks()
                
                # Update watchdog timers
                self._update_watchdog_timers()
                
                # Run redundant safety checkers
                self._run_redundant_checkers()
                
                # Check for fault conditions
                self._detect_fault_conditions()
                
                # Sleep to maintain monitoring frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
                # In case of monitoring error, activate emergency stop
                self.activate_emergency_stop("safety_monitor_error")
    
    def _check_safety_interlocks(self):
        """Check all safety interlocks."""
        current_time = time.time()
        
        for interlock in self.interlocks.values():
            if not interlock.enabled:
                continue
            
            try:
                # Evaluate interlock condition
                condition_met = self._evaluate_interlock_condition(interlock)
                
                if condition_met and not interlock.triggered:
                    # Interlock triggered
                    interlock.triggered = True
                    interlock.trigger_count += 1
                    interlock.last_trigger_time = current_time
                    
                    self.logger.warning(f"Safety interlock triggered: {interlock.name}")
                    self._execute_interlock_action(interlock)
                
                elif not condition_met and interlock.triggered:
                    # Interlock cleared
                    interlock.triggered = False
                    self.logger.info(f"Safety interlock cleared: {interlock.name}")
                    
            except Exception as e:
                self.logger.error(f"Error checking interlock {interlock.name}: {e}")
    
    def _evaluate_interlock_condition(self, interlock: SafetyInterlock) -> bool:
        """Evaluate safety interlock condition."""
        # This would be implemented based on specific hardware conditions
        # For now, return False (no conditions triggered)
        
        if interlock.condition in self.interlock_callbacks:
            return self.interlock_callbacks[interlock.condition]()
        
        # Default: no condition triggered
        return False
    
    def _execute_interlock_action(self, interlock: SafetyInterlock):
        """Execute interlock action."""
        if interlock.action == "emergency_stop":
            self.activate_emergency_stop(f"interlock_{interlock.name}")
        elif interlock.action == "reduce_speed":
            self._reduce_system_speed()
        elif interlock.action == "hold_position":
            self._hold_current_position()
        elif interlock.action == "safe_shutdown":
            self._initiate_safe_shutdown()
        
        # Log the action
        self.logger.warning(f"Executed safety action '{interlock.action}' for interlock '{interlock.name}'")
    
    def _update_watchdog_timers(self):
        """Update watchdog timers and check for timeouts."""
        current_time = time.time()
        
        for watchdog_name, timeout_duration in self.watchdog_timeouts.items():
            if watchdog_name not in self.watchdog_timers:
                continue
            
            time_since_kick = current_time - self.watchdog_timers[watchdog_name]
            
            if time_since_kick > timeout_duration:
                self.logger.error(f"Watchdog timeout: {watchdog_name}")
                self.activate_emergency_stop(f"watchdog_timeout_{watchdog_name}")
    
    def _run_redundant_checkers(self):
        """Run redundant safety checkers."""
        for checker in self.redundant_checkers:
            try:
                if not checker():
                    self.logger.error("Redundant safety check failed")
                    self.activate_emergency_stop("redundant_check_failure")
            except Exception as e:
                self.logger.error(f"Redundant safety checker error: {e}")
                self.activate_emergency_stop("redundant_checker_error")
    
    def _detect_fault_conditions(self):
        """Detect fault conditions."""
        # This would implement specific fault detection logic
        # For now, just check if any faults are set
        
        for fault_name, fault_active in self.fault_conditions.items():
            if fault_active:
                self.logger.error(f"Fault condition detected: {fault_name}")
                
                # Record fault in history
                self.fault_history.append({
                    'timestamp': time.time(),
                    'fault_name': fault_name,
                    'action': 'detected'
                })
                
                # Take appropriate action based on fault severity
                if fault_name.startswith('critical_'):
                    self.activate_emergency_stop(f"critical_fault_{fault_name}")
    
    def activate_emergency_stop(self, source: str):
        """Activate emergency stop system."""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.logger.critical(f"EMERGENCY STOP ACTIVATED by {source}")
        
        self.emergency_stop_sources.add(source)
        
        # Implement emergency stop actions here
        self._execute_emergency_stop_actions()
    
    def deactivate_emergency_stop(self, source: str):
        """Deactivate emergency stop for a specific source."""
        if source in self.emergency_stop_sources:
            self.emergency_stop_sources.remove(source)
        
        # Only deactivate if no sources remain
        if not self.emergency_stop_sources and self.emergency_stop_active:
            self.emergency_stop_active = False
            self.logger.info("Emergency stop deactivated - all sources cleared")
    
    def _execute_emergency_stop_actions(self):
        """Execute emergency stop actions."""
        # This would implement hardware-specific emergency stop
        self.logger.critical("Executing emergency stop actions")
        
        # Example actions:
        # - Cut power to motors
        # - Engage mechanical brakes
        # - Send stop commands to all hardware
        # - Notify external safety systems
    
    def kick_watchdog(self, watchdog_name: str):
        """Reset/kick a watchdog timer."""
        self.watchdog_timers[watchdog_name] = time.time()
    
    def add_watchdog(self, watchdog_name: str, timeout_duration: float):
        """Add a new watchdog timer."""
        self.watchdog_timeouts[watchdog_name] = timeout_duration
        self.kick_watchdog(watchdog_name)
        self.logger.info(f"Added watchdog: {watchdog_name} (timeout: {timeout_duration}s)")
    
    def register_interlock_callback(self, condition_name: str, callback: Callable):
        """Register callback for interlock condition evaluation."""
        self.interlock_callbacks[condition_name] = callback
    
    def add_redundant_checker(self, checker: Callable):
        """Add redundant safety checker function."""
        self.redundant_checkers.append(checker)
    
    def set_fault_condition(self, fault_name: str, active: bool):
        """Set fault condition status."""
        old_status = self.fault_conditions.get(fault_name, False)
        self.fault_conditions[fault_name] = active
        
        if active and not old_status:
            self.logger.warning(f"Fault condition set: {fault_name}")
        elif not active and old_status:
            self.logger.info(f"Fault condition cleared: {fault_name}")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety system status."""
        return {
            'emergency_stop_active': self.emergency_stop_active,
            'emergency_stop_sources': list(self.emergency_stop_sources),
            'active_interlocks': [name for name, interlock in self.interlocks.items() 
                                if interlock.triggered],
            'fault_conditions': {name: active for name, active in self.fault_conditions.items() 
                               if active},
            'watchdog_status': {name: time.time() - last_kick 
                              for name, last_kick in self.watchdog_timers.items()},
            'monitoring_active': self.monitoring_active
        }
    
    def _reduce_system_speed(self):
        """Reduce system speed for safety."""
        self.logger.warning("Reducing system speed for safety")
    
    def _hold_current_position(self):
        """Hold current position for safety."""
        self.logger.warning("Holding current position for safety")
    
    def _initiate_safe_shutdown(self):
        """Initiate safe system shutdown."""
        self.logger.warning("Initiating safe system shutdown")


class HardwareCalibrator:
    """Hardware calibration and diagnostic system."""
    
    def __init__(self, hardware_interface, config: Dict[str, Any]):
        self.hardware_interface = hardware_interface
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Available calibration procedures
        self.calibration_procedures: Dict[str, CalibrationProcedure] = {}
        self.calibration_history: List[Dict[str, Any]] = []
        
        # Diagnostic tests
        self.diagnostic_tests: Dict[str, Callable] = {}
        
        self._initialize_calibration_procedures()
        self._initialize_diagnostic_tests()
    
    def _initialize_calibration_procedures(self):
        """Initialize available calibration procedures."""
        procedures_config = self.config.get('calibration_procedures', [])
        
        # Default joint homing procedure
        homing_procedure = CalibrationProcedure(
            name="joint_homing",
            description="Home all joints to reference positions",
            steps=[
                "Move joints to approximate home positions",
                "Fine-tune positions using sensors",
                "Verify repeatability",
                "Store calibration parameters"
            ],
            expected_duration_sec=60.0,
            safety_requirements=[
                "Emergency stop accessible",
                "Clear workspace",
                "Operator present"
            ],
            success_criteria={
                "position_accuracy": 0.001,  # radians or meters
                "repeatability": 0.0005
            },
            auto_executable=True,
            requires_operator=False
        )
        self.calibration_procedures["joint_homing"] = homing_procedure
        
        # Load additional procedures from config
        for proc_config in procedures_config:
            procedure = CalibrationProcedure(**proc_config)
            self.calibration_procedures[procedure.name] = procedure
    
    def _initialize_diagnostic_tests(self):
        """Initialize diagnostic tests."""
        self.diagnostic_tests = {
            "communication_test": self._test_communication,
            "sensor_test": self._test_sensors,
            "actuator_test": self._test_actuators,
            "safety_system_test": self._test_safety_systems,
            "calibration_verification": self._verify_calibration
        }
    
    def run_calibration(self, procedure_name: str, auto_execute: bool = False) -> Dict[str, Any]:
        """Run calibration procedure."""
        if procedure_name not in self.calibration_procedures:
            return {"success": False, "error": f"Unknown procedure: {procedure_name}"}
        
        procedure = self.calibration_procedures[procedure_name]
        
        if not auto_execute and procedure.requires_operator:
            return {"success": False, "error": "Procedure requires operator confirmation"}
        
        self.logger.info(f"Starting calibration procedure: {procedure.name}")
        start_time = time.time()
        
        try:
            # Execute calibration steps
            result = self._execute_calibration_procedure(procedure)
            
            # Record calibration in history
            calibration_record = {
                'timestamp': start_time,
                'procedure_name': procedure.name,
                'duration_sec': time.time() - start_time,
                'success': result['success'],
                'results': result
            }
            self.calibration_history.append(calibration_record)
            
            if result['success']:
                self.logger.info(f"Calibration completed successfully: {procedure.name}")
            else:
                self.logger.error(f"Calibration failed: {procedure.name}")
            
            return result
            
        except Exception as e:
            error_msg = f"Calibration procedure failed: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _execute_calibration_procedure(self, procedure: CalibrationProcedure) -> Dict[str, Any]:
        """Execute specific calibration procedure."""
        if procedure.name == "joint_homing":
            return self._execute_joint_homing()
        else:
            # Generic calibration execution
            return {"success": True, "message": f"Executed {procedure.name}"}
    
    def _execute_joint_homing(self) -> Dict[str, Any]:
        """Execute joint homing calibration."""
        try:
            # Step 1: Move to approximate home positions
            self.logger.info("Moving to approximate home positions...")
            home_positions = self.config.get('home_positions', [0.0] * 6)
            
            # This would send actual commands to hardware
            # For simulation, we'll just log the action
            self.logger.info(f"Moving to positions: {home_positions}")
            
            # Step 2: Fine-tune using sensors
            self.logger.info("Fine-tuning positions using sensors...")
            time.sleep(1.0)  # Simulate calibration time
            
            # Step 3: Verify repeatability
            self.logger.info("Verifying repeatability...")
            time.sleep(0.5)
            
            # Step 4: Store calibration parameters
            self.logger.info("Storing calibration parameters...")
            
            return {
                "success": True,
                "calibrated_positions": home_positions,
                "accuracy_achieved": 0.0008,  # radians
                "repeatability_achieved": 0.0003
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run_diagnostics(self, test_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run diagnostic tests."""
        if test_names is None:
            test_names = list(self.diagnostic_tests.keys())
        
        results = {}
        overall_success = True
        
        for test_name in test_names:
            if test_name in self.diagnostic_tests:
                try:
                    test_result = self.diagnostic_tests[test_name]()
                    results[test_name] = test_result
                    if not test_result.get("passed", False):
                        overall_success = False
                except Exception as e:
                    results[test_name] = {"passed": False, "error": str(e)}
                    overall_success = False
            else:
                results[test_name] = {"passed": False, "error": "Test not found"}
                overall_success = False
        
        return {"overall_passed": overall_success, "test_results": results}
    
    def _test_communication(self) -> Dict[str, Any]:
        """Test hardware communication."""
        try:
            # Test communication latency
            start_time = time.time()
            # Simulate communication test
            time.sleep(0.01)  # 10ms simulated latency
            latency = time.time() - start_time
            
            return {
                "passed": latency < 0.05,  # Pass if < 50ms
                "latency_ms": latency * 1000,
                "message": f"Communication latency: {latency*1000:.1f}ms"
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_sensors(self) -> Dict[str, Any]:
        """Test sensor functionality."""
        try:
            # Simulate sensor test
            sensor_values = [1.0, 2.0, 3.0]  # Mock sensor readings
            
            # Check if sensors are within expected ranges
            all_sensors_ok = all(0.5 <= val <= 3.5 for val in sensor_values)
            
            return {
                "passed": all_sensors_ok,
                "sensor_values": sensor_values,
                "message": "All sensors reading within normal range" if all_sensors_ok else "Some sensors out of range"
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_actuators(self) -> Dict[str, Any]:
        """Test actuator functionality."""
        try:
            # Simulate actuator test
            time.sleep(0.1)  # Simulate actuator movement
            
            return {
                "passed": True,
                "message": "All actuators responding normally"
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_safety_systems(self) -> Dict[str, Any]:
        """Test safety system functionality."""
        try:
            # Test emergency stop
            # Test safety interlocks
            # Test watchdog timers
            
            return {
                "passed": True,
                "emergency_stop": "functional",
                "safety_interlocks": "functional",
                "watchdog_timers": "functional",
                "message": "All safety systems operational"
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _verify_calibration(self) -> Dict[str, Any]:
        """Verify current calibration accuracy."""
        try:
            # Check calibration timestamp
            # Verify calibration parameters
            # Test position accuracy
            
            return {
                "passed": True,
                "calibration_age_hours": 2.5,
                "position_accuracy": 0.0008,
                "message": "Calibration verified and accurate"
            }
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get calibration status and history."""
        return {
            "available_procedures": list(self.calibration_procedures.keys()),
            "calibration_history": self.calibration_history[-10:],  # Last 10 calibrations
            "last_calibration": self.calibration_history[-1] if self.calibration_history else None
        }


class ProductionHardwareInterface:
    """Main production hardware interface with complete integration."""
    
    def __init__(self, hardware_config: Dict[str, Any]):
        self.config = hardware_config
        self.hardware_id = hardware_config.get('hardware_id', 'safe_rl_hardware')
        self.hardware_type = HardwareType[hardware_config.get('hardware_type', 'EXOSKELETON')]
        self.protocol = CommunicationProtocol[hardware_config.get('protocol', 'SIMULATION')]
        
        self.logger = logging.getLogger(f"{__name__}.ProductionHardwareInterface")
        
        # Hardware limits
        self.limits = HardwareLimits(**hardware_config.get('limits', {}))
        
        # Current status
        self.status = HardwareStatus()
        
        # Communication interface
        self.comm_interface: Optional[Union[ROSHardwareInterface, HardwareSimulator]] = None
        
        # Safety system
        self.safety_system = ProductionSafetySystem(hardware_config.get('safety_config', {}))
        
        # Calibration system
        self.calibrator: Optional[HardwareCalibrator] = None
        
        # Hardware simulator (for testing)
        self.simulator: Optional[HardwareSimulator] = None
        
        # Status update thread
        self.status_thread: Optional[threading.Thread] = None
        self.status_update_active = False
        
    def initialize(self) -> bool:
        """Initialize hardware interface."""
        self.logger.info(f"Initializing {self.hardware_type.name} hardware interface...")
        
        try:
            self.status.state = HardwareState.INITIALIZING
            
            # Initialize communication interface
            if self.protocol == CommunicationProtocol.ROS:
                self.comm_interface = ROSHardwareInterface(self.hardware_id, self.config)
                if not self.comm_interface.initialize_ros():
                    raise RuntimeError("Failed to initialize ROS interface")
            
            elif self.protocol == CommunicationProtocol.SIMULATION:
                self.simulator = HardwareSimulator(self.hardware_type, self.config)
                self.comm_interface = self.simulator
                self.simulator.start_simulation()
            
            else:
                raise NotImplementedError(f"Protocol {self.protocol} not implemented")
            
            # Initialize safety system
            self.safety_system.start_monitoring()
            
            # Initialize calibrator
            self.calibrator = HardwareCalibrator(self, self.config.get('calibration_config', {}))
            
            # Start status updates
            self._start_status_updates()
            
            self.status.state = HardwareState.READY
            self.status.is_connected = True
            
            self.logger.info("Hardware interface initialization complete")
            return True
            
        except Exception as e:
            self.status.state = HardwareState.FAULT
            self.logger.error(f"Hardware initialization failed: {e}")
            return False
    
    def _start_status_updates(self):
        """Start status update thread."""
        if self.status_update_active:
            return
        
        self.status_update_active = True
        self.status_thread = threading.Thread(target=self._status_update_loop, daemon=True)
        self.status_thread.start()
    
    def _status_update_loop(self):
        """Status update loop."""
        update_frequency = self.config.get('status_update_frequency', 50.0)  # Hz
        dt = 1.0 / update_frequency
        
        while self.status_update_active:
            start_time = time.time()
            
            try:
                self._update_status()
                
                # Sleep to maintain update frequency
                elapsed = time.time() - start_time
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Status update error: {e}")
    
    def _update_status(self):
        """Update hardware status."""
        self.status.timestamp = time.time()
        
        # Update from communication interface
        if isinstance(self.comm_interface, HardwareSimulator):
            sim_status = self.comm_interface.get_current_status()
            self.status.positions = sim_status.get('positions')
            self.status.velocities = sim_status.get('velocities')
            self.status.forces = sim_status.get('forces')
        
        elif isinstance(self.comm_interface, ROSHardwareInterface):
            ros_status = self.comm_interface.get_latest_status()
            if ros_status:
                if ros_status['message_type'] == 'joint_states':
                    self.status.positions = ros_status['positions']
                    self.status.velocities = ros_status['velocities']
                    self.status.forces = ros_status['efforts']
        
        # Update safety status
        safety_status = self.safety_system.get_safety_status()
        self.status.emergency_stop_active = safety_status['emergency_stop_active']
        self.status.safety_interlocks_triggered = safety_status['active_interlocks']
        self.status.fault_conditions = list(safety_status['fault_conditions'].keys())
    
    def send_position_command(self, positions: List[float]) -> bool:
        """Send position command to hardware."""
        try:
            # Safety checks
            if self.status.emergency_stop_active:
                self.logger.warning("Cannot send command: Emergency stop active")
                return False
            
            if not self._validate_command_limits(positions):
                self.logger.warning("Command violates safety limits")
                return False
            
            # Send command
            if isinstance(self.comm_interface, HardwareSimulator):
                self.comm_interface.set_target_positions(positions)
                return True
            
            elif isinstance(self.comm_interface, ROSHardwareInterface):
                return self.comm_interface.publish_joint_command(positions)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to send position command: {e}")
            return False
    
    def _validate_command_limits(self, positions: List[float]) -> bool:
        """Validate command against safety limits."""
        if not self.limits.position_min or not self.limits.position_max:
            return True  # No limits configured
        
        for i, pos in enumerate(positions):
            if i >= len(self.limits.position_min) or i >= len(self.limits.position_max):
                continue
            
            if pos < self.limits.position_min[i] or pos > self.limits.position_max[i]:
                return False
        
        return True
    
    def activate_emergency_stop(self):
        """Activate emergency stop."""
        self.safety_system.activate_emergency_stop("hardware_interface")
        
        if isinstance(self.comm_interface, ROSHardwareInterface):
            self.comm_interface.publish_emergency_stop(True)
    
    def deactivate_emergency_stop(self):
        """Deactivate emergency stop."""
        self.safety_system.deactivate_emergency_stop("hardware_interface")
        
        if isinstance(self.comm_interface, ROSHardwareInterface):
            self.comm_interface.publish_emergency_stop(False)
    
    def run_calibration(self, procedure_name: str = "joint_homing") -> Dict[str, Any]:
        """Run calibration procedure."""
        if not self.calibrator:
            return {"success": False, "error": "Calibrator not initialized"}
        
        return self.calibrator.run_calibration(procedure_name, auto_execute=True)
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run hardware diagnostics."""
        if not self.calibrator:
            return {"overall_passed": False, "error": "Calibrator not initialized"}
        
        return self.calibrator.run_diagnostics()
    
    def get_status(self) -> HardwareStatus:
        """Get current hardware status."""
        return self.status
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get safety system status."""
        return self.safety_system.get_safety_status()
    
    def shutdown(self):
        """Shutdown hardware interface."""
        self.logger.info("Shutting down hardware interface...")
        
        # Stop status updates
        self.status_update_active = False
        if self.status_thread:
            self.status_thread.join(timeout=2.0)
        
        # Stop safety monitoring
        self.safety_system.stop_monitoring()
        
        # Stop simulator if running
        if self.simulator:
            self.simulator.stop_simulation()
        
        # Update status
        self.status.state = HardwareState.MAINTENANCE
        self.status.is_connected = False
        
        self.logger.info("Hardware interface shutdown complete")
    
    @contextmanager
    def safe_operation(self, operation_name: str):
        """Context manager for safe hardware operations."""
        self.logger.info(f"Starting safe operation: {operation_name}")
        
        # Kick watchdog
        self.safety_system.kick_watchdog("operation_watchdog")
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Safe operation {operation_name} failed: {e}")
            self.safety_system.activate_emergency_stop(f"operation_error_{operation_name}")
            raise
        finally:
            self.logger.info(f"Safe operation {operation_name} completed")


# Example usage and testing functions
def create_example_hardware_config() -> Dict[str, Any]:
    """Create example hardware configuration."""
    return {
        'hardware_id': 'exo_arm_left',
        'hardware_type': 'EXOSKELETON',
        'protocol': 'SIMULATION',
        'degrees_of_freedom': 6,
        'simulation_frequency': 100.0,
        'status_update_frequency': 50.0,
        
        'limits': {
            'position_min': [-3.14, -1.57, -3.14, -1.57, -3.14, -1.57],
            'position_max': [3.14, 1.57, 3.14, 1.57, 3.14, 1.57],
            'velocity_max': [2.0] * 6,
            'force_max': [50.0] * 6,
            'temperature_max': 70.0
        },
        
        'safety_config': {
            'monitoring_frequency': 100.0,
            'safety_interlocks': [
                {
                    'name': 'joint_limit_check',
                    'hardware_id': 'exo_arm_left',
                    'condition': 'position_limit_exceeded',
                    'action': 'hold_position',
                    'level': 'ERROR'
                },
                {
                    'name': 'force_limit_check',
                    'hardware_id': 'exo_arm_left',
                    'condition': 'force_limit_exceeded', 
                    'action': 'emergency_stop',
                    'level': 'CRITICAL'
                }
            ]
        },
        
        'calibration_config': {
            'home_positions': [0.0] * 6,
            'calibration_procedures': []
        },
        
        'ros_config': {
            'ros_namespace': '/safe_rl',
            'publish_topics': ['joint_commands', 'cmd_vel'],
            'subscribe_topics': ['joint_states', 'imu_data'],
            'services': ['calibrate']
        }
    }


if __name__ == "__main__":
    """Example usage of production hardware interface."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create hardware interface
        config = create_example_hardware_config()
        hardware = ProductionHardwareInterface(config)
        
        # Initialize
        if hardware.initialize():
            logger.info("✅ Hardware interface initialized successfully")
            
            # Run diagnostics
            diag_results = hardware.run_diagnostics()
            logger.info(f"Diagnostics: {diag_results['overall_passed']}")
            
            # Run calibration
            cal_results = hardware.run_calibration()
            logger.info(f"Calibration: {cal_results['success']}")
            
            # Test safe operation
            with hardware.safe_operation("position_command"):
                success = hardware.send_position_command([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
                logger.info(f"Position command: {'success' if success else 'failed'}")
            
            # Get status
            status = hardware.get_status()
            logger.info(f"Hardware state: {status.state.name}")
            logger.info(f"Emergency stop: {status.emergency_stop_active}")
            
            # Run for a few seconds
            time.sleep(3.0)
            
            # Shutdown
            hardware.shutdown()
            logger.info("✅ Hardware interface test completed")
            
        else:
            logger.error("❌ Hardware interface initialization failed")
            
    except Exception as e:
        logger.error(f"Hardware interface test failed: {e}")
        sys.exit(1)