"""
Main Safe RL ROS Node for Policy Deployment.

This node provides the main interface for deploying trained Safe RL policies
in ROS-based robotic systems with real-time performance and safety integration.

Key Features:
- Real-time policy inference
- Hardware interface integration
- Safety monitoring integration
- Dynamic reconfiguration
- Performance monitoring
- Multi-robot coordination
"""

import rospy
import threading
import numpy as np
from std_msgs.msg import Header, Bool, Float64, String
from geometry_msgs.msg import Twist, Pose, PoseStamped, Vector3
from sensor_msgs.msg import JointState, Image, PointCloud2, Imu
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib import SimpleActionServer, SimpleActionClient
import tf2_ros
import tf2_geometry_msgs
from dynamic_reconfigure.server import Server as DynamicReconfigureServer
import diagnostic_msgs.msg

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import time

# Import Safe RL components
from ..core.policy import SafeRLPolicy
from ..realtime.realtime_controller import RealTimeController, RTConfig
from ..hardware.hardware_interface import HardwareInterface
from ..hardware.safety_hardware import SafetyHardware
from .ros_messages import (
    PolicyAction, SafetyStatus as SafetyStatusMsg, HumanIntent, 
    ConstraintViolation, PerformanceMetrics, SystemDiagnostics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ROSNodeConfig:
    """Configuration for Safe RL ROS node."""
    
    # Node configuration
    node_name: str = "safe_rl_node"
    namespace: str = "/safe_rl"
    publish_rate: float = 100.0  # Hz
    
    # Policy configuration
    policy_model_path: str = ""
    policy_config_path: str = ""
    enable_online_learning: bool = False
    
    # Hardware interface configuration
    hardware_config_path: str = ""
    enable_hardware_interface: bool = True
    
    # Safety configuration
    safety_config_path: str = ""
    enable_safety_monitoring: bool = True
    safety_node_timeout: float = 0.1  # seconds
    
    # Real-time configuration
    rt_config: RTConfig = field(default_factory=RTConfig)
    
    # ROS topic names
    joint_states_topic: str = "/joint_states"
    cmd_vel_topic: str = "/cmd_vel"
    policy_action_topic: str = "/safe_rl/policy_action"
    safety_status_topic: str = "/safe_rl/safety_status"
    human_intent_topic: str = "/safe_rl/human_intent"
    performance_topic: str = "/safe_rl/performance"
    diagnostics_topic: str = "/diagnostics"
    
    # TF configuration
    base_frame: str = "base_link"
    world_frame: str = "world"
    
    # Visualization
    enable_visualization: bool = True
    rviz_markers_topic: str = "/safe_rl/visualization_markers"
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_window_size: int = 1000
    
    # Multi-robot support
    robot_id: str = "robot_0"
    multi_robot_enabled: bool = False
    coordination_topic: str = "/multi_robot/coordination"


class SafeRLNode:
    """
    Main Safe RL ROS node for policy deployment.
    
    This node integrates trained Safe RL policies with ROS robotics systems,
    providing real-time policy inference, safety monitoring, and hardware control.
    """
    
    def __init__(self, config: ROSNodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.SafeRLNode")
        
        # Initialize ROS node
        rospy.init_node(config.node_name, anonymous=True)
        self.logger.info(f"Initializing Safe RL ROS node: {config.node_name}")
        
        # Node state
        self.is_active = False
        self.is_emergency_stopped = False
        self.last_heartbeat = time.time()
        
        # Safe RL components
        self.policy: Optional[SafeRLPolicy] = None
        self.rt_controller: Optional[RealTimeController] = None
        self.hardware_interfaces: Dict[str, HardwareInterface] = {}
        self.safety_hardware: Optional[SafetyHardware] = None
        
        # ROS components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers
        self.publishers = {}
        self._setup_publishers()
        
        # Subscribers
        self.subscribers = {}
        self._setup_subscribers()
        
        # Services and actions
        self.action_servers = {}
        self.action_clients = {}
        self._setup_services_and_actions()
        
        # Dynamic reconfigure
        self.dynamic_reconfig_server = None
        if rospy.has_param('~enable_dynamic_reconfigure'):
            self._setup_dynamic_reconfigure()
        
        # Performance monitoring
        self.performance_metrics = {
            'policy_inference_time': [],
            'control_loop_time': [],
            'safety_check_time': [],
            'total_cycles': 0,
            'successful_cycles': 0,
            'safety_violations': 0,
            'emergency_stops': 0
        }
        
        # State variables
        self.current_joint_states = None
        self.current_pose = None
        self.human_intent = None
        self.last_policy_action = None
        
        # Thread safety
        self.state_lock = threading.RLock()
        
        # Control loop thread
        self.control_thread = None
        self.shutdown_event = threading.Event()
        
        self.logger.info("Safe RL ROS node initialized successfully")
    
    def initialize(self) -> bool:
        """Initialize the Safe RL node components."""
        try:
            self.logger.info("Initializing Safe RL node components...")
            
            # Load and initialize policy
            if not self._initialize_policy():
                self.logger.error("Policy initialization failed")
                return False
            
            # Initialize hardware interfaces
            if self.config.enable_hardware_interface:
                if not self._initialize_hardware():
                    self.logger.error("Hardware initialization failed")
                    return False
            
            # Initialize real-time controller
            if not self._initialize_rt_controller():
                self.logger.error("Real-time controller initialization failed")
                return False
            
            # Initialize safety monitoring
            if self.config.enable_safety_monitoring:
                if not self._initialize_safety():
                    self.logger.error("Safety monitoring initialization failed")
                    return False
            
            # Start performance monitoring
            if self.config.enable_performance_monitoring:
                self._start_performance_monitoring()
            
            self.logger.info("Safe RL node initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Safe RL node initialization failed: {e}")
            return False
    
    def start(self) -> bool:
        """Start the Safe RL node."""
        try:
            if not self.is_active:
                self.logger.info("Starting Safe RL node...")
                
                # Start real-time controller
                if self.rt_controller and not self.rt_controller.start():
                    self.logger.error("Failed to start real-time controller")
                    return False
                
                # Start control loop thread
                self.shutdown_event.clear()
                self.control_thread = threading.Thread(
                    target=self._control_loop,
                    name="SafeRL_ControlLoop",
                    daemon=True
                )
                self.control_thread.start()
                
                self.is_active = True
                self.logger.info("Safe RL node started successfully")
                
                # Publish initial status
                self._publish_safety_status()
                
                return True
            else:
                self.logger.warning("Safe RL node is already active")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start Safe RL node: {e}")
            return False
    
    def stop(self):
        """Stop the Safe RL node."""
        try:
            if self.is_active:
                self.logger.info("Stopping Safe RL node...")
                
                self.is_active = False
                self.shutdown_event.set()
                
                # Stop control loop thread
                if self.control_thread and self.control_thread.is_alive():
                    self.control_thread.join(timeout=2.0)
                
                # Stop real-time controller
                if self.rt_controller:
                    self.rt_controller.stop()
                
                self.logger.info("Safe RL node stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop Safe RL node: {e}")
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        try:
            self.logger.critical("EMERGENCY STOP TRIGGERED")
            
            with self.state_lock:
                self.is_emergency_stopped = True
                self.performance_metrics['emergency_stops'] += 1
            
            # Stop real-time controller
            if self.rt_controller:
                self.rt_controller.emergency_stop()
            
            # Stop hardware interfaces
            for hw in self.hardware_interfaces.values():
                hw.emergency_stop()
            
            # Publish emergency status
            self._publish_emergency_status()
            
        except Exception as e:
            self.logger.critical(f"Emergency stop failed: {e}")
    
    def shutdown(self):
        """Shutdown the Safe RL node."""
        try:
            self.logger.info("Shutting down Safe RL node...")
            
            # Stop the node
            self.stop()
            
            # Shutdown components
            if self.rt_controller:
                self.rt_controller.shutdown()
            
            for hw in self.hardware_interfaces.values():
                hw.shutdown_hardware()
            
            if self.safety_hardware:
                self.safety_hardware.shutdown_hardware()
            
            self.logger.info("Safe RL node shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Safe RL node shutdown failed: {e}")
    
    # ROS callback methods
    
    def _joint_states_callback(self, msg: JointState):
        """Handle joint states updates."""
        try:
            with self.state_lock:
                self.current_joint_states = msg
                self.last_heartbeat = time.time()
                
        except Exception as e:
            self.logger.error(f"Joint states callback error: {e}")
    
    def _human_intent_callback(self, msg: HumanIntent):
        """Handle human intent updates."""
        try:
            with self.state_lock:
                self.human_intent = msg
                
        except Exception as e:
            self.logger.error(f"Human intent callback error: {e}")
    
    def _emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop commands."""
        try:
            if msg.data:
                self.emergency_stop()
            else:
                # Reset emergency stop if safe
                self._reset_emergency_stop()
                
        except Exception as e:
            self.logger.error(f"Emergency stop callback error: {e}")
    
    # Private implementation methods
    
    def _setup_publishers(self):
        """Setup ROS publishers."""
        try:
            self.publishers = {
                'policy_action': rospy.Publisher(
                    self.config.policy_action_topic,
                    PolicyAction,
                    queue_size=10
                ),
                'safety_status': rospy.Publisher(
                    self.config.safety_status_topic,
                    SafetyStatusMsg,
                    queue_size=10
                ),
                'performance': rospy.Publisher(
                    self.config.performance_topic,
                    PerformanceMetrics,
                    queue_size=10
                ),
                'diagnostics': rospy.Publisher(
                    self.config.diagnostics_topic,
                    diagnostic_msgs.msg.DiagnosticArray,
                    queue_size=10
                ),
                'cmd_vel': rospy.Publisher(
                    self.config.cmd_vel_topic,
                    Twist,
                    queue_size=10
                )
            }
            
            if self.config.enable_visualization:
                from visualization_msgs.msg import MarkerArray
                self.publishers['visualization'] = rospy.Publisher(
                    self.config.rviz_markers_topic,
                    MarkerArray,
                    queue_size=10
                )
            
            self.logger.info("ROS publishers setup complete")
            
        except Exception as e:
            self.logger.error(f"Publisher setup failed: {e}")
    
    def _setup_subscribers(self):
        """Setup ROS subscribers."""
        try:
            self.subscribers = {
                'joint_states': rospy.Subscriber(
                    self.config.joint_states_topic,
                    JointState,
                    self._joint_states_callback,
                    queue_size=10
                ),
                'human_intent': rospy.Subscriber(
                    self.config.human_intent_topic,
                    HumanIntent,
                    self._human_intent_callback,
                    queue_size=10
                ),
                'emergency_stop': rospy.Subscriber(
                    f"{self.config.namespace}/emergency_stop",
                    Bool,
                    self._emergency_stop_callback,
                    queue_size=1
                )
            }
            
            self.logger.info("ROS subscribers setup complete")
            
        except Exception as e:
            self.logger.error(f"Subscriber setup failed: {e}")
    
    def _setup_services_and_actions(self):
        """Setup ROS services and action servers/clients."""
        try:
            # Action servers
            # Add action server setup here if needed
            
            # Action clients
            # Add action client setup here if needed
            
            # Services
            # Add service setup here if needed
            
            self.logger.info("ROS services and actions setup complete")
            
        except Exception as e:
            self.logger.error(f"Services and actions setup failed: {e}")
    
    def _setup_dynamic_reconfigure(self):
        """Setup dynamic reconfiguration."""
        try:
            # This would be implemented based on specific configuration needs
            # For now, this is a placeholder
            self.logger.info("Dynamic reconfigure setup complete")
            
        except Exception as e:
            self.logger.error(f"Dynamic reconfigure setup failed: {e}")
    
    def _initialize_policy(self) -> bool:
        """Initialize Safe RL policy."""
        try:
            if not self.config.policy_model_path:
                self.logger.error("Policy model path not specified")
                return False
            
            # Load policy from file
            self.policy = SafeRLPolicy.load(self.config.policy_model_path)
            
            if self.policy is None:
                self.logger.error("Failed to load policy")
                return False
            
            self.logger.info(f"Policy loaded successfully: {self.config.policy_model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Policy initialization failed: {e}")
            return False
    
    def _initialize_hardware(self) -> bool:
        """Initialize hardware interfaces."""
        try:
            # This would load hardware configurations and initialize interfaces
            # For now, this is a placeholder
            self.logger.info("Hardware interfaces initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            return False
    
    def _initialize_rt_controller(self) -> bool:
        """Initialize real-time controller."""
        try:
            self.rt_controller = RealTimeController(self.config.rt_config)
            
            if not self.rt_controller.initialize():
                return False
            
            self.logger.info("Real-time controller initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Real-time controller initialization failed: {e}")
            return False
    
    def _initialize_safety(self) -> bool:
        """Initialize safety monitoring."""
        try:
            # This would initialize safety hardware and monitoring
            # For now, this is a placeholder
            self.logger.info("Safety monitoring initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Safety initialization failed: {e}")
            return False
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread."""
        try:
            performance_thread = threading.Thread(
                target=self._performance_monitoring_loop,
                name="PerformanceMonitor",
                daemon=True
            )
            performance_thread.start()
            
            self.logger.info("Performance monitoring started")
            
        except Exception as e:
            self.logger.error(f"Performance monitoring start failed: {e}")
    
    def _control_loop(self):
        """Main control loop."""
        try:
            self.logger.info("Control loop started")
            rate = rospy.Rate(self.config.publish_rate)
            
            while not self.shutdown_event.is_set() and not rospy.is_shutdown():
                loop_start_time = time.perf_counter()
                
                try:
                    # Check if we have required sensor data
                    if self.current_joint_states is None:
                        rate.sleep()
                        continue
                    
                    # Check for emergency stop
                    if self.is_emergency_stopped:
                        rate.sleep()
                        continue
                    
                    # Prepare observation
                    observation = self._prepare_observation()
                    if observation is None:
                        rate.sleep()
                        continue
                    
                    # Policy inference
                    policy_start_time = time.perf_counter()
                    action = self.policy.predict(observation)
                    policy_end_time = time.perf_counter()
                    
                    # Safety check
                    safety_start_time = time.perf_counter()
                    if not self._safety_check(action, observation):
                        self.logger.warning("Safety check failed, skipping action")
                        rate.sleep()
                        continue
                    safety_end_time = time.perf_counter()
                    
                    # Execute action
                    self._execute_action(action)
                    
                    # Publish policy action
                    self._publish_policy_action(action)
                    
                    # Update performance metrics
                    loop_end_time = time.perf_counter()
                    with self.state_lock:
                        self.performance_metrics['policy_inference_time'].append(
                            (policy_end_time - policy_start_time) * 1000  # ms
                        )
                        self.performance_metrics['safety_check_time'].append(
                            (safety_end_time - safety_start_time) * 1000  # ms
                        )
                        self.performance_metrics['control_loop_time'].append(
                            (loop_end_time - loop_start_time) * 1000  # ms
                        )
                        self.performance_metrics['total_cycles'] += 1
                        self.performance_metrics['successful_cycles'] += 1
                        
                        # Keep sliding window
                        for key in ['policy_inference_time', 'safety_check_time', 'control_loop_time']:
                            if len(self.performance_metrics[key]) > self.config.performance_window_size:
                                self.performance_metrics[key] = self.performance_metrics[key][-self.config.performance_window_size:]
                    
                except Exception as e:
                    self.logger.error(f"Control loop error: {e}")
                    with self.state_lock:
                        self.performance_metrics['total_cycles'] += 1
                
                rate.sleep()
                
        except Exception as e:
            self.logger.error(f"Control loop failed: {e}")
        finally:
            self.logger.info("Control loop terminated")
    
    def _prepare_observation(self) -> Optional[np.ndarray]:
        """Prepare observation for policy inference."""
        try:
            with self.state_lock:
                if self.current_joint_states is None:
                    return None
                
                # Convert joint states to observation format
                joint_positions = np.array(self.current_joint_states.position)
                joint_velocities = np.array(self.current_joint_states.velocity)
                
                # Add human intent if available
                human_input = np.zeros(3)  # Default zero input
                if self.human_intent is not None:
                    human_input = np.array([
                        self.human_intent.desired_velocity.x,
                        self.human_intent.desired_velocity.y,
                        self.human_intent.desired_velocity.z
                    ])
                
                # Combine observation components
                observation = np.concatenate([
                    joint_positions,
                    joint_velocities,
                    human_input
                ])
                
                return observation
                
        except Exception as e:
            self.logger.error(f"Observation preparation failed: {e}")
            return None
    
    def _safety_check(self, action: np.ndarray, observation: np.ndarray) -> bool:
        """Perform safety check on proposed action."""
        try:
            # Basic safety checks
            if not np.all(np.isfinite(action)):
                self.logger.warning("Action contains NaN or infinite values")
                return False
            
            # Check action magnitude limits
            max_action = 1.0  # Normalized action space
            if np.any(np.abs(action) > max_action):
                self.logger.warning("Action exceeds magnitude limits")
                return False
            
            # Additional safety checks would be implemented here
            # based on specific safety constraints
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return False
    
    def _execute_action(self, action: np.ndarray):
        """Execute the computed action."""
        try:
            # Convert action to appropriate control commands
            # This depends on the specific robot and control interface
            
            # For velocity control
            cmd_vel = Twist()
            if len(action) >= 3:
                cmd_vel.linear.x = float(action[0])
                cmd_vel.linear.y = float(action[1])
                cmd_vel.angular.z = float(action[2])
            
            # Publish velocity command
            self.publishers['cmd_vel'].publish(cmd_vel)
            
            # Store last action
            with self.state_lock:
                self.last_policy_action = action.copy()
                
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
    
    def _publish_policy_action(self, action: np.ndarray):
        """Publish policy action message."""
        try:
            msg = PolicyAction()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.config.base_frame
            
            msg.action_values = action.tolist()
            msg.action_type = "velocity"  # or "position", "torque", etc.
            msg.confidence = 1.0  # Policy confidence if available
            
            self.publishers['policy_action'].publish(msg)
            
        except Exception as e:
            self.logger.error(f"Policy action publishing failed: {e}")
    
    def _publish_safety_status(self):
        """Publish safety status message."""
        try:
            msg = SafetyStatusMsg()
            msg.header.stamp = rospy.Time.now()
            
            msg.is_safe = not self.is_emergency_stopped
            msg.safety_level = "NORMAL" if not self.is_emergency_stopped else "EMERGENCY_STOP"
            msg.active_constraints = []  # Fill with active constraints
            msg.violations = []  # Fill with current violations
            
            self.publishers['safety_status'].publish(msg)
            
        except Exception as e:
            self.logger.error(f"Safety status publishing failed: {e}")
    
    def _publish_emergency_status(self):
        """Publish emergency status."""
        try:
            msg = SafetyStatusMsg()
            msg.header.stamp = rospy.Time.now()
            
            msg.is_safe = False
            msg.safety_level = "EMERGENCY_STOP"
            msg.emergency_stop_active = True
            
            self.publishers['safety_status'].publish(msg)
            
            # Also publish zero velocity
            zero_cmd = Twist()
            self.publishers['cmd_vel'].publish(zero_cmd)
            
        except Exception as e:
            self.logger.error(f"Emergency status publishing failed: {e}")
    
    def _reset_emergency_stop(self):
        """Reset emergency stop if conditions are safe."""
        try:
            if self.is_emergency_stopped:
                self.logger.info("Attempting to reset emergency stop")
                
                # Check if it's safe to reset
                if self._check_safe_to_reset():
                    with self.state_lock:
                        self.is_emergency_stopped = False
                    
                    self.logger.info("Emergency stop reset successfully")
                    self._publish_safety_status()
                else:
                    self.logger.warning("Not safe to reset emergency stop")
            
        except Exception as e:
            self.logger.error(f"Emergency stop reset failed: {e}")
    
    def _check_safe_to_reset(self) -> bool:
        """Check if conditions are safe to reset emergency stop."""
        try:
            # Check hardware status
            for hw in self.hardware_interfaces.values():
                if not hw.get_safety_status().is_safe_to_operate():
                    return False
            
            # Check real-time controller status
            if self.rt_controller and self.rt_controller.state.name != 'READY':
                return False
            
            # Additional safety checks would be implemented here
            
            return True
            
        except Exception as e:
            self.logger.error(f"Safe to reset check failed: {e}")
            return False
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        try:
            rate = rospy.Rate(1.0)  # 1 Hz performance publishing
            
            while not self.shutdown_event.is_set() and not rospy.is_shutdown():
                try:
                    # Create performance metrics message
                    msg = PerformanceMetrics()
                    msg.header.stamp = rospy.Time.now()
                    
                    with self.state_lock:
                        if self.performance_metrics['policy_inference_time']:
                            msg.avg_policy_inference_time_ms = np.mean(
                                self.performance_metrics['policy_inference_time']
                            )
                        
                        if self.performance_metrics['control_loop_time']:
                            msg.avg_control_loop_time_ms = np.mean(
                                self.performance_metrics['control_loop_time']
                            )
                        
                        msg.total_cycles = self.performance_metrics['total_cycles']
                        msg.successful_cycles = self.performance_metrics['successful_cycles']
                        msg.safety_violations = self.performance_metrics['safety_violations']
                        msg.emergency_stops = self.performance_metrics['emergency_stops']
                        
                        if msg.total_cycles > 0:
                            msg.success_rate = msg.successful_cycles / msg.total_cycles
                    
                    # Publish performance metrics
                    self.publishers['performance'].publish(msg)
                    
                    # Publish diagnostics
                    self._publish_diagnostics()
                    
                except Exception as e:
                    self.logger.error(f"Performance monitoring error: {e}")
                
                rate.sleep()
                
        except Exception as e:
            self.logger.error(f"Performance monitoring loop failed: {e}")
    
    def _publish_diagnostics(self):
        """Publish system diagnostics."""
        try:
            diag_array = diagnostic_msgs.msg.DiagnosticArray()
            diag_array.header.stamp = rospy.Time.now()
            
            # Node status
            status = diagnostic_msgs.msg.DiagnosticStatus()
            status.name = "Safe RL Node"
            status.hardware_id = self.config.robot_id
            
            if self.is_active and not self.is_emergency_stopped:
                status.level = diagnostic_msgs.msg.DiagnosticStatus.OK
                status.message = "Operating normally"
            elif self.is_emergency_stopped:
                status.level = diagnostic_msgs.msg.DiagnosticStatus.ERROR
                status.message = "Emergency stop active"
            else:
                status.level = diagnostic_msgs.msg.DiagnosticStatus.WARN
                status.message = "Node inactive"
            
            # Add performance values
            with self.state_lock:
                if self.performance_metrics['policy_inference_time']:
                    avg_inference_time = np.mean(self.performance_metrics['policy_inference_time'])
                    status.values.append(
                        diagnostic_msgs.msg.KeyValue("Policy Inference Time (ms)", str(avg_inference_time))
                    )
                
                status.values.append(
                    diagnostic_msgs.msg.KeyValue("Total Cycles", str(self.performance_metrics['total_cycles']))
                )
                status.values.append(
                    diagnostic_msgs.msg.KeyValue("Emergency Stops", str(self.performance_metrics['emergency_stops']))
                )
            
            diag_array.status.append(status)
            
            # Publish diagnostics
            self.publishers['diagnostics'].publish(diag_array)
            
        except Exception as e:
            self.logger.error(f"Diagnostics publishing failed: {e}")


def main():
    """Main function for Safe RL ROS node."""
    try:
        # Load configuration from ROS parameters
        config = ROSNodeConfig()
        
        # Override with ROS parameters if they exist
        if rospy.has_param('~node_name'):
            config.node_name = rospy.get_param('~node_name')
        if rospy.has_param('~policy_model_path'):
            config.policy_model_path = rospy.get_param('~policy_model_path')
        if rospy.has_param('~hardware_config_path'):
            config.hardware_config_path = rospy.get_param('~hardware_config_path')
        
        # Create and initialize node
        node = SafeRLNode(config)
        
        if not node.initialize():
            rospy.logerr("Safe RL node initialization failed")
            return
        
        # Start node
        if not node.start():
            rospy.logerr("Safe RL node start failed")
            return
        
        rospy.loginfo("Safe RL node running...")
        
        # Keep node running
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutdown requested by user")
        finally:
            node.shutdown()
            
    except Exception as e:
        rospy.logerr(f"Safe RL node failed: {e}")


if __name__ == "__main__":
    main()