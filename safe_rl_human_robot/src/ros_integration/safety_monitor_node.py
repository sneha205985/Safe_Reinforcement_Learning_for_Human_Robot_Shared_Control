"""
Independent Safety Monitor ROS Node.

This node provides independent safety monitoring for the Safe RL system,
running at high frequency with hardware-level safety integration and
fail-safe operation modes.

Key Features:
- Independent safety monitoring (2kHz+)
- Hardware-level emergency stops
- Constraint violation detection
- Predictive safety analysis
- Multi-layer safety architecture
- Real-time diagnostics
"""

import rospy
import threading
import numpy as np
from std_msgs.msg import Header, Bool, Float64, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import diagnostic_msgs.msg

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import time

from ..realtime.safety_monitor import RTSafetyMonitor, SafetyConfig, SafetyViolation, ViolationType
from ..hardware.safety_hardware import SafetyHardware
from ..hardware.hardware_interface import HardwareInterface, SafetyLevel
from .ros_messages import SafetyStatus as SafetyStatusMsg, ConstraintViolation, SystemDiagnostics

logger = logging.getLogger(__name__)


@dataclass
class SafetyNodeConfig:
    """Configuration for Safety Monitor ROS node."""
    
    # Node configuration
    node_name: str = "safety_monitor_node"
    namespace: str = "/safe_rl"
    monitoring_frequency: float = 2000.0  # Hz
    
    # Safety configuration
    safety_config: SafetyConfig = SafetyConfig()
    hardware_config_path: str = ""
    
    # ROS topic names
    joint_states_topic: str = "/joint_states"
    cmd_vel_topic: str = "/cmd_vel"
    safety_status_topic: str = "/safe_rl/safety_status"
    constraint_violation_topic: str = "/safe_rl/constraint_violations"
    emergency_stop_topic: str = "/safe_rl/emergency_stop"
    safety_override_topic: str = "/safe_rl/safety_override"
    diagnostics_topic: str = "/diagnostics"
    
    # Safety thresholds
    max_response_time_ms: float = 1.0  # Maximum allowed response time
    violation_timeout_s: float = 0.1   # Time before escalating violations
    
    # Hardware integration
    enable_hardware_emergency_stop: bool = True
    hardware_watchdog_timeout_s: float = 0.05  # 50ms watchdog timeout
    
    # Multi-robot coordination
    robot_id: str = "robot_0"
    coordination_enabled: bool = False
    coordination_topic: str = "/multi_robot/safety_coordination"


class SafetyMonitorNode:
    """
    Independent safety monitor ROS node.
    
    Provides high-frequency, independent safety monitoring with hardware-level
    emergency stop capabilities and real-time constraint checking.
    """
    
    def __init__(self, config: SafetyNodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SafetyMonitorNode")
        
        # Initialize ROS node
        rospy.init_node(config.node_name, anonymous=True)
        self.logger.info(f"Initializing Safety Monitor ROS node: {config.node_name}")
        
        # Node state
        self.is_monitoring = False
        self.is_emergency_stopped = False
        self.safety_override_active = False
        self.last_heartbeat = time.time()
        
        # Safety components
        self.safety_monitor: Optional[RTSafetyMonitor] = None
        self.safety_hardware: Optional[SafetyHardware] = None
        self.hardware_interfaces: Dict[str, HardwareInterface] = {}
        
        # ROS components
        self.publishers = {}
        self.subscribers = {}
        self._setup_ros_interface()
        
        # Safety state tracking
        self.current_safety_level = SafetyLevel.NORMAL
        self.active_violations: Dict[str, SafetyViolation] = {}
        self.violation_history: List[SafetyViolation] = []
        
        # Performance monitoring
        self.monitoring_metrics = {
            'total_checks': 0,
            'violations_detected': 0,
            'emergency_stops_triggered': 0,
            'false_positives': 0,
            'avg_response_time_ms': 0.0,
            'max_response_time_ms': 0.0,
            'monitoring_frequency_hz': 0.0
        }
        
        # Sensor data
        self.current_joint_states = None
        self.last_cmd_vel = None
        
        # Thread management
        self.monitoring_thread = None
        self.diagnostics_thread = None
        self.shutdown_event = threading.Event()
        
        # Thread synchronization
        self.safety_lock = threading.RLock()
        
        self.logger.info("Safety Monitor ROS node initialized")
    
    def initialize(self) -> bool:
        """Initialize safety monitor components."""
        try:
            self.logger.info("Initializing safety monitor components...")
            
            # Initialize hardware interfaces
            if not self._initialize_hardware():
                self.logger.error("Hardware initialization failed")
                return False
            
            # Initialize safety hardware
            if self.config.enable_hardware_emergency_stop:
                if not self._initialize_safety_hardware():
                    self.logger.error("Safety hardware initialization failed")
                    return False
            
            # Initialize RT safety monitor
            if not self._initialize_rt_safety_monitor():
                self.logger.error("RT safety monitor initialization failed")
                return False
            
            self.logger.info("Safety monitor initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Safety monitor initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start safety monitoring."""
        try:
            if not self.is_monitoring:
                self.logger.info("Starting safety monitoring...")
                
                # Start RT safety monitor
                if self.safety_monitor and not self.safety_monitor.start():
                    self.logger.error("Failed to start RT safety monitor")
                    return False
                
                # Start monitoring threads
                self.shutdown_event.clear()
                
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="SafetyMonitoring",
                    daemon=True
                )
                self.monitoring_thread.start()
                
                self.diagnostics_thread = threading.Thread(
                    target=self._diagnostics_loop,
                    name="SafetyDiagnostics",
                    daemon=True
                )
                self.diagnostics_thread.start()
                
                self.is_monitoring = True
                self.logger.info("Safety monitoring started")
                
                # Publish initial status
                self._publish_safety_status()
                
                return True
            else:
                self.logger.warning("Safety monitoring is already active")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start safety monitoring: {e}")
            return False
    
    def stop(self):
        """Stop safety monitoring."""
        try:
            if self.is_monitoring:
                self.logger.info("Stopping safety monitoring...")
                
                self.is_monitoring = False
                self.shutdown_event.set()
                
                # Stop monitoring threads
                for thread in [self.monitoring_thread, self.diagnostics_thread]:
                    if thread and thread.is_alive():
                        thread.join(timeout=2.0)
                
                # Stop RT safety monitor
                if self.safety_monitor:
                    self.safety_monitor.stop()
                
                self.logger.info("Safety monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop safety monitoring: {e}")
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """Trigger emergency stop."""
        try:
            self.logger.critical(f"SAFETY MONITOR EMERGENCY STOP: {reason}")
            
            with self.safety_lock:
                self.is_emergency_stopped = True
                self.monitoring_metrics['emergency_stops_triggered'] += 1
            
            # Trigger RT safety monitor emergency stop
            if self.safety_monitor:
                self.safety_monitor.emergency_stop()
            
            # Activate hardware emergency stop
            if self.safety_hardware:
                self.safety_hardware.activate_emergency_stop()
            
            # Stop all hardware interfaces
            for hw in self.hardware_interfaces.values():
                hw.emergency_stop()
            
            # Publish emergency stop command
            self.publishers['emergency_stop'].publish(Bool(data=True))
            
            # Publish emergency status
            self._publish_emergency_status(reason)
            
        except Exception as e:
            self.logger.critical(f"Emergency stop execution failed: {e}")
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop if conditions are safe."""
        try:
            if self.is_emergency_stopped:
                self.logger.info("Attempting to reset emergency stop...")
                
                # Check if conditions are safe
                if not self._check_safe_to_reset():
                    self.logger.warning("Conditions not safe for emergency stop reset")
                    return False
                
                # Reset hardware emergency stop
                if self.safety_hardware and not self.safety_hardware.reset_emergency_stop():
                    self.logger.error("Hardware emergency stop reset failed")
                    return False
                
                # Reset hardware interfaces
                for hw in self.hardware_interfaces.values():
                    if not hw.reset_hardware_emergency_stop():
                        self.logger.error(f"Hardware {hw.config.device_id} emergency stop reset failed")
                        return False
                
                with self.safety_lock:
                    self.is_emergency_stopped = False
                    self.current_safety_level = SafetyLevel.MINOR_WARNING
                
                # Publish reset status
                self.publishers['emergency_stop'].publish(Bool(data=False))
                self._publish_safety_status()
                
                self.logger.info("Emergency stop reset successfully")
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency stop reset failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown safety monitor node."""
        try:
            self.logger.info("Shutting down safety monitor node...")
            
            # Stop monitoring
            self.stop()
            
            # Shutdown components
            if self.safety_monitor:
                self.safety_monitor.shutdown()
            
            if self.safety_hardware:
                self.safety_hardware.shutdown_hardware()
            
            for hw in self.hardware_interfaces.values():
                hw.shutdown_hardware()
            
            self.logger.info("Safety monitor node shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Safety monitor shutdown failed: {e}")
    
    # ROS callback methods
    
    def _joint_states_callback(self, msg: JointState):
        """Handle joint states updates."""
        try:
            with self.safety_lock:
                self.current_joint_states = msg
                self.last_heartbeat = time.time()
                
        except Exception as e:
            self.logger.error(f"Joint states callback error: {e}")
    
    def _cmd_vel_callback(self, msg: Twist):
        """Handle velocity command monitoring."""
        try:
            with self.safety_lock:
                self.last_cmd_vel = msg
                
                # Check velocity limits
                if self._check_velocity_limits(msg):
                    self._trigger_velocity_violation(msg)
                
        except Exception as e:
            self.logger.error(f"Cmd vel callback error: {e}")
    
    def _safety_override_callback(self, msg: Bool):
        """Handle safety override commands."""
        try:
            with self.safety_lock:
                self.safety_override_active = msg.data
                
            if msg.data:
                self.logger.warning("SAFETY OVERRIDE ACTIVATED - Use with extreme caution!")
            else:
                self.logger.info("Safety override deactivated")
                
            self._publish_safety_status()
            
        except Exception as e:
            self.logger.error(f"Safety override callback error: {e}")
    
    def _emergency_stop_external_callback(self, msg: Bool):
        """Handle external emergency stop commands."""
        try:
            if msg.data and not self.is_emergency_stopped:
                self.emergency_stop("External emergency stop command")
            
        except Exception as e:
            self.logger.error(f"External emergency stop callback error: {e}")
    
    # Private implementation methods
    
    def _setup_ros_interface(self):
        """Setup ROS publishers and subscribers."""
        try:
            # Publishers
            self.publishers = {
                'safety_status': rospy.Publisher(
                    self.config.safety_status_topic,
                    SafetyStatusMsg,
                    queue_size=10
                ),
                'constraint_violations': rospy.Publisher(
                    self.config.constraint_violation_topic,
                    ConstraintViolation,
                    queue_size=10
                ),
                'emergency_stop': rospy.Publisher(
                    self.config.emergency_stop_topic,
                    Bool,
                    queue_size=1
                ),
                'diagnostics': rospy.Publisher(
                    self.config.diagnostics_topic,
                    diagnostic_msgs.msg.DiagnosticArray,
                    queue_size=10
                )
            }
            
            # Subscribers
            self.subscribers = {
                'joint_states': rospy.Subscriber(
                    self.config.joint_states_topic,
                    JointState,
                    self._joint_states_callback,
                    queue_size=10
                ),
                'cmd_vel': rospy.Subscriber(
                    self.config.cmd_vel_topic,
                    Twist,
                    self._cmd_vel_callback,
                    queue_size=10
                ),
                'safety_override': rospy.Subscriber(
                    self.config.safety_override_topic,
                    Bool,
                    self._safety_override_callback,
                    queue_size=1
                ),
                'emergency_stop_external': rospy.Subscriber(
                    f"{self.config.namespace}/emergency_stop_external",
                    Bool,
                    self._emergency_stop_external_callback,
                    queue_size=1
                )
            }
            
            self.logger.info("ROS interface setup complete")
            
        except Exception as e:
            self.logger.error(f"ROS interface setup failed: {e}")
    
    def _initialize_hardware(self) -> bool:
        """Initialize hardware interfaces."""
        try:
            # This would load and initialize hardware configurations
            # For now, this is a placeholder
            self.logger.info("Hardware interfaces initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            return False
    
    def _initialize_safety_hardware(self) -> bool:
        """Initialize safety hardware."""
        try:
            # This would initialize safety hardware components
            # For now, this is a placeholder
            self.logger.info("Safety hardware initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Safety hardware initialization failed: {e}")
            return False
    
    def _initialize_rt_safety_monitor(self) -> bool:
        """Initialize real-time safety monitor."""
        try:
            self.safety_monitor = RTSafetyMonitor(
                self.config.safety_config,
                self.safety_hardware,
                list(self.hardware_interfaces.values())
            )
            
            if not self.safety_monitor.initialize():
                return False
            
            # Add violation callback
            self.safety_monitor.add_violation_callback(self._handle_safety_violation)
            self.safety_monitor.add_emergency_callback(self._handle_emergency_callback)
            
            self.logger.info("RT safety monitor initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"RT safety monitor initialization failed: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main safety monitoring loop."""
        try:
            self.logger.info("Safety monitoring loop started")
            rate = rospy.Rate(self.config.monitoring_frequency)
            
            while not self.shutdown_event.is_set() and not rospy.is_shutdown():
                monitoring_start_time = time.perf_counter()
                
                try:
                    # Check for sensor data timeout
                    current_time = time.time()
                    if (self.current_joint_states is not None and 
                        current_time - self.last_heartbeat > self.config.hardware_watchdog_timeout_s):
                        self.logger.warning("Hardware watchdog timeout")
                        self._trigger_watchdog_violation()
                    
                    # Perform safety checks
                    self._perform_safety_checks()
                    
                    # Update monitoring metrics
                    monitoring_end_time = time.perf_counter()
                    response_time_ms = (monitoring_end_time - monitoring_start_time) * 1000
                    
                    self.monitoring_metrics['total_checks'] += 1
                    self.monitoring_metrics['avg_response_time_ms'] = (
                        (self.monitoring_metrics['avg_response_time_ms'] * 
                         (self.monitoring_metrics['total_checks'] - 1) + response_time_ms) /
                        self.monitoring_metrics['total_checks']
                    )
                    
                    if response_time_ms > self.monitoring_metrics['max_response_time_ms']:
                        self.monitoring_metrics['max_response_time_ms'] = response_time_ms
                    
                    # Check response time threshold
                    if response_time_ms > self.config.max_response_time_ms:
                        self.logger.warning(f"Safety monitoring response time exceeded: {response_time_ms:.2f}ms")
                    
                    # Publish status periodically
                    if self.monitoring_metrics['total_checks'] % 100 == 0:  # Every 100 cycles
                        self._publish_safety_status()
                    
                except Exception as e:
                    self.logger.error(f"Safety monitoring cycle error: {e}")
                
                rate.sleep()
                
        except Exception as e:
            self.logger.error(f"Safety monitoring loop failed: {e}")
        finally:
            self.logger.info("Safety monitoring loop terminated")
    
    def _diagnostics_loop(self):
        """Safety diagnostics publishing loop."""
        try:
            rate = rospy.Rate(10.0)  # 10 Hz diagnostics
            
            while not self.shutdown_event.is_set() and not rospy.is_shutdown():
                try:
                    self._publish_diagnostics()
                except Exception as e:
                    self.logger.error(f"Diagnostics publishing error: {e}")
                
                rate.sleep()
                
        except Exception as e:
            self.logger.error(f"Diagnostics loop failed: {e}")
    
    def _perform_safety_checks(self):
        """Perform comprehensive safety checks."""
        try:
            if self.safety_override_active:
                return  # Skip safety checks if override is active
            
            # Check joint limits
            if self.current_joint_states is not None:
                self._check_joint_limits()
            
            # Check velocity limits
            if self.last_cmd_vel is not None:
                self._check_velocity_limits(self.last_cmd_vel)
            
            # Check hardware status
            self._check_hardware_status()
            
        except Exception as e:
            self.logger.error(f"Safety checks failed: {e}")
    
    def _check_joint_limits(self):
        """Check joint position and velocity limits."""
        try:
            positions = np.array(self.current_joint_states.position)
            velocities = np.array(self.current_joint_states.velocity)
            
            # Check position limits
            for i, (pos, vel) in enumerate(zip(positions, velocities)):
                joint_name = (self.current_joint_states.name[i] 
                             if i < len(self.current_joint_states.name) 
                             else f"joint_{i}")
                
                # Position limit check
                pos_limits = self.config.safety_config.position_limits.get(joint_name)
                if pos_limits and (pos < pos_limits[0] or pos > pos_limits[1]):
                    self._trigger_position_violation(joint_name, pos, pos_limits)
                
                # Velocity limit check
                vel_limit = self.config.safety_config.velocity_limits.get(joint_name)
                if vel_limit and abs(vel) > vel_limit:
                    self._trigger_velocity_violation_joint(joint_name, vel, vel_limit)
                    
        except Exception as e:
            self.logger.error(f"Joint limits check failed: {e}")
    
    def _check_velocity_limits(self, cmd_vel: Twist) -> bool:
        """Check velocity command limits."""
        try:
            linear_vel = np.array([cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.linear.z])
            angular_vel = np.array([cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z])
            
            max_linear = 2.0  # m/s
            max_angular = 2.0  # rad/s
            
            if np.any(np.abs(linear_vel) > max_linear):
                return True
            
            if np.any(np.abs(angular_vel) > max_angular):
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Velocity limits check failed: {e}")
            return False
    
    def _check_hardware_status(self):
        """Check hardware interface status."""
        try:
            for hw_id, hw in self.hardware_interfaces.items():
                safety_status = hw.get_safety_status()
                
                if not safety_status.is_safe_to_operate():
                    self._trigger_hardware_violation(hw_id, safety_status)
                    
        except Exception as e:
            self.logger.error(f"Hardware status check failed: {e}")
    
    def _check_safe_to_reset(self) -> bool:
        """Check if conditions are safe to reset emergency stop."""
        try:
            # Check hardware interfaces
            for hw in self.hardware_interfaces.values():
                if not hw.get_safety_status().is_safe_to_operate():
                    return False
            
            # Check for active violations
            with self.safety_lock:
                if len(self.active_violations) > 0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safe to reset check failed: {e}")
            return False
    
    # Violation handling methods
    
    def _handle_safety_violation(self, violation: SafetyViolation):
        """Handle safety violations from RT monitor."""
        try:
            with self.safety_lock:
                violation_id = f"{violation.violation_type.name}_{int(time.time()*1000)}"
                self.active_violations[violation_id] = violation
                self.violation_history.append(violation)
                self.monitoring_metrics['violations_detected'] += 1
                
                # Update safety level
                if violation.severity < self.current_safety_level:
                    self.current_safety_level = violation.severity
            
            # Publish violation
            self._publish_constraint_violation(violation)
            
            # Take action based on severity
            if violation.severity == SafetyLevel.CRITICAL_FAILURE:
                self.emergency_stop(f"Critical violation: {violation.message}")
                
        except Exception as e:
            self.logger.error(f"Safety violation handling failed: {e}")
    
    def _handle_emergency_callback(self):
        """Handle emergency callbacks from RT monitor."""
        self.emergency_stop("RT Safety Monitor emergency callback")
    
    def _trigger_position_violation(self, joint_name: str, position: float, limits: tuple):
        """Trigger position limit violation."""
        violation = SafetyViolation(
            violation_type=ViolationType.POSITION_LIMIT,
            severity=SafetyLevel.MAJOR_WARNING,
            message=f"Joint {joint_name} position limit exceeded: {position:.3f} not in [{limits[0]:.3f}, {limits[1]:.3f}]",
            timestamp=time.time(),
            recommended_action="GRADUAL_STOP"
        )
        
        self._handle_safety_violation(violation)
    
    def _trigger_velocity_violation(self, cmd_vel: Twist):
        """Trigger velocity command violation."""
        violation = SafetyViolation(
            violation_type=ViolationType.VELOCITY_LIMIT,
            severity=SafetyLevel.MAJOR_WARNING,
            message=f"Velocity command limit exceeded: linear=[{cmd_vel.linear.x:.2f}, {cmd_vel.linear.y:.2f}, {cmd_vel.linear.z:.2f}], angular=[{cmd_vel.angular.x:.2f}, {cmd_vel.angular.y:.2f}, {cmd_vel.angular.z:.2f}]",
            timestamp=time.time(),
            recommended_action="IMMEDIATE_STOP"
        )
        
        self._handle_safety_violation(violation)
    
    def _trigger_velocity_violation_joint(self, joint_name: str, velocity: float, limit: float):
        """Trigger joint velocity violation."""
        violation = SafetyViolation(
            violation_type=ViolationType.VELOCITY_LIMIT,
            severity=SafetyLevel.MAJOR_WARNING,
            message=f"Joint {joint_name} velocity limit exceeded: {velocity:.3f} > {limit:.3f}",
            timestamp=time.time(),
            recommended_action="IMMEDIATE_STOP"
        )
        
        self._handle_safety_violation(violation)
    
    def _trigger_hardware_violation(self, hw_id: str, safety_status):
        """Trigger hardware safety violation."""
        violation = SafetyViolation(
            violation_type=ViolationType.HARDWARE_FAULT,
            severity=safety_status.safety_level,
            message=f"Hardware safety violation: {hw_id} - {safety_status.constraint_violations}",
            timestamp=time.time(),
            recommended_action="EMERGENCY_STOP"
        )
        
        self._handle_safety_violation(violation)
    
    def _trigger_watchdog_violation(self):
        """Trigger watchdog timeout violation."""
        violation = SafetyViolation(
            violation_type=ViolationType.COMMUNICATION_TIMEOUT,
            severity=SafetyLevel.MAJOR_WARNING,
            message="Hardware watchdog timeout - no recent sensor data",
            timestamp=time.time(),
            recommended_action="EMERGENCY_STOP"
        )
        
        self._handle_safety_violation(violation)
    
    # Publishing methods
    
    def _publish_safety_status(self):
        """Publish current safety status."""
        try:
            msg = SafetyStatusMsg()
            msg.header.stamp = rospy.Time.now()
            
            with self.safety_lock:
                msg.is_safe = (self.current_safety_level >= SafetyLevel.MINOR_WARNING and 
                              not self.is_emergency_stopped)
                msg.safety_level = self.current_safety_level.name
                msg.emergency_stop_active = self.is_emergency_stopped
                msg.active_constraints = []  # Fill with active constraints
                msg.violations = [v.message for v in self.active_violations.values()]
                msg.safety_score = float(self.current_safety_level) / 4.0
            
            self.publishers['safety_status'].publish(msg)
            
        except Exception as e:
            self.logger.error(f"Safety status publishing failed: {e}")
    
    def _publish_emergency_status(self, reason: str):
        """Publish emergency status."""
        try:
            msg = SafetyStatusMsg()
            msg.header.stamp = rospy.Time.now()
            
            msg.is_safe = False
            msg.safety_level = "EMERGENCY_STOP"
            msg.emergency_stop_active = True
            msg.violations = [f"EMERGENCY STOP: {reason}"]
            msg.safety_score = 0.0
            
            self.publishers['safety_status'].publish(msg)
            
        except Exception as e:
            self.logger.error(f"Emergency status publishing failed: {e}")
    
    def _publish_constraint_violation(self, violation: SafetyViolation):
        """Publish constraint violation."""
        try:
            msg = ConstraintViolation()
            msg.header.stamp = rospy.Time.now()
            
            msg.constraint_name = violation.violation_type.name
            msg.violation_type = violation.violation_type.name
            msg.severity = violation.severity.name
            msg.description = violation.message
            msg.recommended_action = violation.recommended_action
            
            if violation.predicted_time_to_impact is not None:
                msg.time_to_impact = violation.predicted_time_to_impact
            else:
                msg.time_to_impact = -1.0
            
            self.publishers['constraint_violations'].publish(msg)
            
        except Exception as e:
            self.logger.error(f"Constraint violation publishing failed: {e}")
    
    def _publish_diagnostics(self):
        """Publish safety diagnostics."""
        try:
            diag_array = diagnostic_msgs.msg.DiagnosticArray()
            diag_array.header.stamp = rospy.Time.now()
            
            # Safety monitor status
            status = diagnostic_msgs.msg.DiagnosticStatus()
            status.name = "Safety Monitor"
            status.hardware_id = self.config.robot_id
            
            if self.is_monitoring and not self.is_emergency_stopped:
                status.level = diagnostic_msgs.msg.DiagnosticStatus.OK
                status.message = "Safety monitoring active"
            elif self.is_emergency_stopped:
                status.level = diagnostic_msgs.msg.DiagnosticStatus.ERROR
                status.message = "Emergency stop active"
            else:
                status.level = diagnostic_msgs.msg.DiagnosticStatus.WARN
                status.message = "Safety monitoring inactive"
            
            # Add monitoring metrics
            status.values.extend([
                diagnostic_msgs.msg.KeyValue("Monitoring Frequency (Hz)", 
                                           str(self.config.monitoring_frequency)),
                diagnostic_msgs.msg.KeyValue("Total Checks", 
                                           str(self.monitoring_metrics['total_checks'])),
                diagnostic_msgs.msg.KeyValue("Violations Detected", 
                                           str(self.monitoring_metrics['violations_detected'])),
                diagnostic_msgs.msg.KeyValue("Emergency Stops", 
                                           str(self.monitoring_metrics['emergency_stops_triggered'])),
                diagnostic_msgs.msg.KeyValue("Avg Response Time (ms)", 
                                           f"{self.monitoring_metrics['avg_response_time_ms']:.3f}"),
                diagnostic_msgs.msg.KeyValue("Max Response Time (ms)", 
                                           f"{self.monitoring_metrics['max_response_time_ms']:.3f}")
            ])
            
            diag_array.status.append(status)
            
            self.publishers['diagnostics'].publish(diag_array)
            
        except Exception as e:
            self.logger.error(f"Diagnostics publishing failed: {e}")


def main():
    """Main function for Safety Monitor ROS node."""
    try:
        # Load configuration from ROS parameters
        config = SafetyNodeConfig()
        
        # Override with ROS parameters
        if rospy.has_param('~monitoring_frequency'):
            config.monitoring_frequency = rospy.get_param('~monitoring_frequency')
        if rospy.has_param('~hardware_config_path'):
            config.hardware_config_path = rospy.get_param('~hardware_config_path')
        
        # Create and initialize node
        node = SafetyMonitorNode(config)
        
        if not node.initialize():
            rospy.logerr("Safety monitor node initialization failed")
            return
        
        # Start monitoring
        if not node.start():
            rospy.logerr("Safety monitor node start failed")
            return
        
        rospy.loginfo("Safety monitor node running...")
        
        # Keep node running
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutdown requested by user")
        finally:
            node.shutdown()
            
    except Exception as e:
        rospy.logerr(f"Safety monitor node failed: {e}")


if __name__ == "__main__":
    main()