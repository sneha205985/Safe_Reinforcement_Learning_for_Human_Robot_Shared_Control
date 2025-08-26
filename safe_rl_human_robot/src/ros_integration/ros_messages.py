"""
ROS Message Definitions for Safe RL Human-Robot Shared Control.

This module defines custom ROS messages for Safe RL policy deployment,
safety monitoring, and human-robot interaction.

Custom Messages:
- PolicyAction: Safe policy commands
- SafetyStatus: Real-time safety state
- HumanIntent: Processed human inputs
- ConstraintViolation: Safety violations
- PerformanceMetrics: System performance data
- SystemDiagnostics: Detailed system diagnostics
"""

from std_msgs.msg import Header, Float64, Bool, String, Int32
from geometry_msgs.msg import Vector3, Twist, Pose
from typing import List
import rospy

# Since we can't directly define custom messages in Python,
# we create classes that mirror the ROS message structure
# These would normally be generated from .msg files

class PolicyAction:
    """
    Safe RL policy action message.
    
    Header header
    float64[] action_values
    string action_type
    float64 confidence
    float64 safety_score
    string[] active_constraints
    """
    
    def __init__(self):
        self.header = Header()
        self.action_values = []
        self.action_type = ""
        self.confidence = 0.0
        self.safety_score = 0.0
        self.active_constraints = []


class SafetyStatus:
    """
    Real-time safety status message.
    
    Header header
    bool is_safe
    string safety_level
    bool emergency_stop_active
    string[] active_constraints
    string[] violations
    float64 safety_score
    float64 risk_assessment
    float64 time_to_violation
    """
    
    def __init__(self):
        self.header = Header()
        self.is_safe = True
        self.safety_level = "NORMAL"
        self.emergency_stop_active = False
        self.active_constraints = []
        self.violations = []
        self.safety_score = 1.0
        self.risk_assessment = 0.0
        self.time_to_violation = -1.0


class HumanIntent:
    """
    Processed human input message.
    
    Header header
    geometry_msgs/Vector3 desired_velocity
    geometry_msgs/Vector3 desired_position
    geometry_msgs/Vector3 desired_force
    float64 assistance_level
    string intent_type
    float64 confidence
    bool override_safety
    """
    
    def __init__(self):
        self.header = Header()
        self.desired_velocity = Vector3()
        self.desired_position = Vector3()
        self.desired_force = Vector3()
        self.assistance_level = 0.5
        self.intent_type = "COLLABORATIVE"
        self.confidence = 0.0
        self.override_safety = False


class ConstraintViolation:
    """
    Safety constraint violation message.
    
    Header header
    string constraint_name
    string violation_type
    string severity
    string description
    float64[] sensor_values
    float64[] limit_values
    float64 violation_magnitude
    string recommended_action
    float64 time_to_impact
    """
    
    def __init__(self):
        self.header = Header()
        self.constraint_name = ""
        self.violation_type = ""
        self.severity = ""
        self.description = ""
        self.sensor_values = []
        self.limit_values = []
        self.violation_magnitude = 0.0
        self.recommended_action = ""
        self.time_to_impact = -1.0


class PerformanceMetrics:
    """
    System performance metrics message.
    
    Header header
    float64 avg_policy_inference_time_ms
    float64 max_policy_inference_time_ms
    float64 avg_control_loop_time_ms
    float64 max_control_loop_time_ms
    float64 avg_safety_check_time_ms
    float64 control_frequency_hz
    float64 success_rate
    int32 total_cycles
    int32 successful_cycles
    int32 safety_violations
    int32 emergency_stops
    float64 cpu_usage_percent
    float64 memory_usage_mb
    """
    
    def __init__(self):
        self.header = Header()
        self.avg_policy_inference_time_ms = 0.0
        self.max_policy_inference_time_ms = 0.0
        self.avg_control_loop_time_ms = 0.0
        self.max_control_loop_time_ms = 0.0
        self.avg_safety_check_time_ms = 0.0
        self.control_frequency_hz = 0.0
        self.success_rate = 0.0
        self.total_cycles = 0
        self.successful_cycles = 0
        self.safety_violations = 0
        self.emergency_stops = 0
        self.cpu_usage_percent = 0.0
        self.memory_usage_mb = 0.0


class SystemDiagnostics:
    """
    Detailed system diagnostics message.
    
    Header header
    string node_name
    string node_state
    string hardware_status
    string rt_controller_status
    string safety_monitor_status
    DiagnosticValue[] diagnostic_values
    string[] warnings
    string[] errors
    float64 uptime_seconds
    """
    
    def __init__(self):
        self.header = Header()
        self.node_name = ""
        self.node_state = ""
        self.hardware_status = ""
        self.rt_controller_status = ""
        self.safety_monitor_status = ""
        self.diagnostic_values = []
        self.warnings = []
        self.errors = []
        self.uptime_seconds = 0.0


class DiagnosticValue:
    """
    Individual diagnostic value.
    
    string key
    string value
    string unit
    """
    
    def __init__(self):
        self.key = ""
        self.value = ""
        self.unit = ""


class HardwareStatus:
    """
    Hardware status message.
    
    Header header
    string hardware_id
    string hardware_type
    string connection_status
    bool is_operational
    float64[] joint_positions
    float64[] joint_velocities
    float64[] joint_torques
    float64[] motor_temperatures
    float64[] motor_currents
    string[] fault_codes
    float64 last_update_time
    """
    
    def __init__(self):
        self.header = Header()
        self.hardware_id = ""
        self.hardware_type = ""
        self.connection_status = ""
        self.is_operational = True
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_torques = []
        self.motor_temperatures = []
        self.motor_currents = []
        self.fault_codes = []
        self.last_update_time = 0.0


class MultiRobotCoordination:
    """
    Multi-robot coordination message.
    
    Header header
    string robot_id
    geometry_msgs/Pose robot_pose
    geometry_msgs/Twist robot_velocity
    string current_task
    string coordination_state
    string[] nearby_robots
    float64 priority_level
    bool needs_assistance
    string assistance_type
    """
    
    def __init__(self):
        self.header = Header()
        self.robot_id = ""
        self.robot_pose = Pose()
        self.robot_velocity = Twist()
        self.current_task = ""
        self.coordination_state = ""
        self.nearby_robots = []
        self.priority_level = 0.5
        self.needs_assistance = False
        self.assistance_type = ""


# Message generation templates for creating actual .msg files
MSG_TEMPLATES = {
    "PolicyAction.msg": """
# Safe RL policy action message
Header header

# Action values (normalized)
float64[] action_values

# Type of action (position, velocity, torque, etc.)
string action_type

# Policy confidence in the action
float64 confidence

# Safety score for this action
float64 safety_score

# Currently active safety constraints
string[] active_constraints
""",

    "SafetyStatus.msg": """
# Real-time safety status
Header header

# Overall safety state
bool is_safe

# Safety level (NORMAL, WARNING, CRITICAL, EMERGENCY)
string safety_level

# Emergency stop state
bool emergency_stop_active

# Active safety constraints
string[] active_constraints

# Current safety violations
string[] violations

# Overall safety score (0-1)
float64 safety_score

# Risk assessment score
float64 risk_assessment

# Predicted time to potential violation (seconds, -1 if none predicted)
float64 time_to_violation
""",

    "HumanIntent.msg": """
# Processed human input
Header header

# Desired motion
geometry_msgs/Vector3 desired_velocity
geometry_msgs/Vector3 desired_position
geometry_msgs/Vector3 desired_force

# Level of assistance requested (0-1)
float64 assistance_level

# Type of intent (COLLABORATIVE, OVERRIDE, ASSISTANCE)
string intent_type

# Confidence in intent recognition
float64 confidence

# Safety override flag (use with extreme caution)
bool override_safety
""",

    "ConstraintViolation.msg": """
# Safety constraint violation
Header header

# Name of violated constraint
string constraint_name

# Type of violation (POSITION, VELOCITY, FORCE, etc.)
string violation_type

# Severity level (LOW, MEDIUM, HIGH, CRITICAL)
string severity

# Human-readable description
string description

# Current sensor values that caused violation
float64[] sensor_values

# Limit values that were exceeded
float64[] limit_values

# Magnitude of violation
float64 violation_magnitude

# Recommended action (STOP, SLOW_DOWN, REDIRECT, etc.)
string recommended_action

# Predicted time to impact if no action taken (seconds)
float64 time_to_impact
""",

    "PerformanceMetrics.msg": """
# System performance metrics
Header header

# Policy inference timing
float64 avg_policy_inference_time_ms
float64 max_policy_inference_time_ms

# Control loop timing
float64 avg_control_loop_time_ms
float64 max_control_loop_time_ms

# Safety checking timing
float64 avg_safety_check_time_ms

# Control frequency
float64 control_frequency_hz

# Success metrics
float64 success_rate
int32 total_cycles
int32 successful_cycles
int32 safety_violations
int32 emergency_stops

# System resources
float64 cpu_usage_percent
float64 memory_usage_mb
""",

    "SystemDiagnostics.msg": """
# Detailed system diagnostics
Header header

# Node information
string node_name
string node_state

# Component status
string hardware_status
string rt_controller_status
string safety_monitor_status

# Diagnostic values
DiagnosticValue[] diagnostic_values

# Issues
string[] warnings
string[] errors

# System uptime
float64 uptime_seconds
""",

    "DiagnosticValue.msg": """
# Individual diagnostic value
string key
string value
string unit
""",

    "HardwareStatus.msg": """
# Hardware interface status
Header header

# Hardware identification
string hardware_id
string hardware_type
string connection_status

# Operational state
bool is_operational

# Joint states
float64[] joint_positions
float64[] joint_velocities
float64[] joint_torques

# Motor status
float64[] motor_temperatures
float64[] motor_currents

# Fault information
string[] fault_codes

# Timing
float64 last_update_time
""",

    "MultiRobotCoordination.msg": """
# Multi-robot coordination
Header header

# Robot identification
string robot_id

# Robot state
geometry_msgs/Pose robot_pose
geometry_msgs/Twist robot_velocity

# Task information
string current_task
string coordination_state

# Multi-robot coordination
string[] nearby_robots
float64 priority_level

# Assistance
bool needs_assistance
string assistance_type
"""
}


def generate_msg_files(output_directory: str):
    """
    Generate .msg files for ROS message compilation.
    
    Args:
        output_directory: Directory to write .msg files
    """
    import os
    
    msg_dir = os.path.join(output_directory, "msg")
    os.makedirs(msg_dir, exist_ok=True)
    
    for filename, content in MSG_TEMPLATES.items():
        filepath = os.path.join(msg_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Generated {filepath}")


def generate_cmake_lists(output_directory: str):
    """
    Generate CMakeLists.txt for ROS message compilation.
    
    Args:
        output_directory: Directory to write CMakeLists.txt
    """
    cmake_content = """
cmake_minimum_required(VERSION 3.0.2)
project(safe_rl_msgs)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  PolicyAction.msg
  SafetyStatus.msg
  HumanIntent.msg
  ConstraintViolation.msg
  PerformanceMetrics.msg
  SystemDiagnostics.msg
  DiagnosticValue.msg
  HardwareStatus.msg
  MultiRobotCoordination.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  std_msgs
  geometry_msgs
  sensor_msgs
  message_runtime
)
"""
    
    import os
    filepath = os.path.join(output_directory, "CMakeLists.txt")
    with open(filepath, 'w') as f:
        f.write(cmake_content)
    print(f"Generated {filepath}")


def generate_package_xml(output_directory: str):
    """
    Generate package.xml for ROS package.
    
    Args:
        output_directory: Directory to write package.xml
    """
    package_content = """
<?xml version="1.0"?>
<package format="2">
  <name>safe_rl_msgs</name>
  <version>1.0.0</version>
  <description>Custom ROS messages for Safe RL Human-Robot Shared Control</description>

  <maintainer email="developer@example.com">Safe RL Team</maintainer>
  <license>MIT</license>

  <buildtool_depend>catkin</buildtool_depend>
  
  <build_depend>std_msgs</build_depend>
  <build_depend>geometry_msgs</build_depend>
  <build_depend>sensor_msgs</build_depend>
  <build_depend>message_generation</build_depend>
  
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>message_runtime</exec_depend>

  <export>
  </export>
</package>
"""
    
    import os
    filepath = os.path.join(output_directory, "package.xml")
    with open(filepath, 'w') as f:
        f.write(package_content)
    print(f"Generated {filepath}")


if __name__ == "__main__":
    """Generate ROS message files for compilation."""
    import sys
    import os
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Generating ROS message files in: {output_dir}")
    
    generate_msg_files(output_dir)
    generate_cmake_lists(output_dir)
    generate_package_xml(output_dir)
    
    print("ROS message generation complete!")
    print("To use these messages:")
    print("1. Copy the generated files to a catkin workspace")
    print("2. Run 'catkin_make' to compile the messages")
    print("3. Source the workspace setup file")