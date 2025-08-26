"""
Exoskeleton hardware interface for Safe RL human-robot shared control.

This module provides hardware integration for powered exoskeleton systems,
including motor control, sensor feedback, and safety monitoring.
"""

import numpy as np
import time
import serial
import struct
import threading
from typing import Dict, List, Any, Optional, Tuple
import logging
import can
from dataclasses import dataclass

from .hardware_interface import HardwareInterface, SafetyStatus, SafetyLevel, HardwareConfig
from .sensor_interface import IMUInterface, ForceInterface, EncoderInterface
from .safety_hardware import SafetyHardware

logger = logging.getLogger(__name__)


@dataclass
class ExoskeletonConfig(HardwareConfig):
    """Configuration specific to exoskeleton hardware."""
    num_joints: int = 6
    joint_names: List[str] = None
    max_joint_torques: List[float] = None  # Nm
    max_joint_velocities: List[float] = None  # rad/s
    max_joint_angles: List[Tuple[float, float]] = None  # (min, max) rad
    force_sensor_locations: List[str] = None
    imu_locations: List[str] = None
    encoder_resolution: List[int] = None  # counts per revolution
    gear_ratios: List[float] = None
    torque_constants: List[float] = None  # Nm/A
    motor_can_ids: List[int] = None
    sensor_can_ids: List[int] = None
    can_bus_name: str = "can0"
    can_bitrate: int = 1000000  # 1Mbps
    safety_force_threshold: float = 50.0  # N
    safety_torque_threshold: float = 30.0  # Nm
    emergency_brake_time_ms: int = 50
    
    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(self.num_joints)]
        
        if self.max_joint_torques is None:
            self.max_joint_torques = [40.0] * self.num_joints
        
        if self.max_joint_velocities is None:
            self.max_joint_velocities = [2.0] * self.num_joints  # rad/s
        
        if self.max_joint_angles is None:
            self.max_joint_angles = [(-1.57, 1.57)] * self.num_joints  # ±90°
        
        if self.motor_can_ids is None:
            self.motor_can_ids = list(range(0x100, 0x100 + self.num_joints))
        
        if self.sensor_can_ids is None:
            self.sensor_can_ids = list(range(0x200, 0x200 + self.num_joints))


class ExoskeletonInterface(HardwareInterface):
    """
    Hardware interface for powered exoskeleton systems.
    
    Provides real-time control of joint motors, sensor data acquisition,
    and safety monitoring for exoskeleton devices.
    """
    
    def __init__(self, config: ExoskeletonConfig):
        """
        Initialize exoskeleton interface.
        
        Args:
            config: Exoskeleton-specific configuration
        """
        super().__init__(config)
        self.exo_config = config
        
        # CAN bus interface
        self.can_bus = None
        self.can_lock = threading.Lock()
        
        # Motor control
        self.motor_controllers = {}
        self.joint_positions = np.zeros(self.exo_config.num_joints)
        self.joint_velocities = np.zeros(self.exo_config.num_joints)
        self.joint_torques = np.zeros(self.exo_config.num_joints)
        self.motor_currents = np.zeros(self.exo_config.num_joints)
        self.motor_temperatures = np.zeros(self.exo_config.num_joints)
        
        # Sensor interfaces
        self.force_sensors = {}
        self.imu_sensors = {}
        self.encoders = {}
        
        # Safety monitoring
        self.safety_hardware = SafetyHardware(config)
        self.last_safety_check = time.time()
        self.force_readings = np.zeros(len(config.force_sensor_locations or []))
        
        # Control state
        self.target_positions = np.zeros(self.exo_config.num_joints)
        self.target_velocities = np.zeros(self.exo_config.num_joints)
        self.target_torques = np.zeros(self.exo_config.num_joints)
        self.control_mode = "torque"  # "position", "velocity", "torque"
        
        logger.info(f"Exoskeleton interface initialized for {config.num_joints} joints")
    
    def initialize_hardware(self) -> bool:
        """Initialize exoskeleton hardware components."""
        try:
            logger.info("Initializing exoskeleton hardware...")
            
            # Initialize CAN bus
            if not self._initialize_can_bus():
                return False
            
            # Initialize motor controllers
            if not self._initialize_motors():
                return False
            
            # Initialize sensors
            if not self._initialize_sensors():
                return False
            
            # Initialize safety systems
            if not self.safety_hardware.initialize():
                logger.error("Safety hardware initialization failed")
                return False
            
            # Perform initial health check
            if not self._perform_health_check():
                return False
            
            logger.info("Exoskeleton hardware initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Exoskeleton initialization error: {str(e)}")
            return False
    
    def shutdown_hardware(self):
        """Shutdown exoskeleton hardware."""
        try:
            logger.info("Shutting down exoskeleton hardware...")
            
            # Stop all motors safely
            self._stop_all_motors()
            
            # Shutdown safety systems
            self.safety_hardware.shutdown()
            
            # Close CAN bus
            if self.can_bus:
                self.can_bus.shutdown()
                self.can_bus = None
            
            logger.info("Exoskeleton hardware shutdown complete")
            
        except Exception as e:
            logger.error(f"Exoskeleton shutdown error: {str(e)}")
    
    def send_commands(self, commands: np.ndarray) -> bool:
        """
        Send control commands to exoskeleton motors.
        
        Args:
            commands: Array of joint commands (torques, positions, or velocities)
            
        Returns:
            True if commands sent successfully
        """
        try:
            if len(commands) != self.exo_config.num_joints:
                logger.error(f"Expected {self.exo_config.num_joints} commands, got {len(commands)}")
                return False
            
            # Update target values based on control mode
            if self.control_mode == "torque":
                self.target_torques = commands.copy()
            elif self.control_mode == "position":
                self.target_positions = commands.copy()
            elif self.control_mode == "velocity":
                self.target_velocities = commands.copy()
            
            # Send commands to each motor
            success = True
            for i, joint_name in enumerate(self.exo_config.joint_names):
                if not self._send_motor_command(i, commands[i]):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Command sending error: {str(e)}")
            return False
    
    def read_sensor_data(self) -> Optional[Dict[str, np.ndarray]]:
        """Read all sensor data from exoskeleton."""
        try:
            sensor_data = {}
            
            # Read joint encoder data
            self._update_joint_states()
            sensor_data['joint_positions'] = self.joint_positions.copy()
            sensor_data['joint_velocities'] = self.joint_velocities.copy()
            sensor_data['joint_torques'] = self.joint_torques.copy()
            
            # Read force sensors
            if self.force_sensors:
                force_data = self._read_force_sensors()
                sensor_data['forces'] = force_data
                self.force_readings = force_data
            
            # Read IMU data
            if self.imu_sensors:
                imu_data = self._read_imu_sensors()
                sensor_data.update(imu_data)
            
            # Read motor status
            sensor_data['motor_currents'] = self.motor_currents.copy()
            sensor_data['motor_temperatures'] = self.motor_temperatures.copy()
            
            # Add timestamp
            sensor_data['timestamp'] = time.time()
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Sensor reading error: {str(e)}")
            return None
    
    def execute_emergency_stop(self):
        """Execute emergency stop for exoskeleton."""
        try:
            logger.critical("Executing exoskeleton emergency stop")
            
            # Activate safety hardware emergency stop
            self.safety_hardware.activate_emergency_stop()
            
            # Send emergency brake commands to all motors
            self._emergency_brake_all_motors()
            
            # Reset control targets
            self.target_torques.fill(0.0)
            self.target_velocities.fill(0.0)
            
            logger.critical("Exoskeleton emergency stop complete")
            
        except Exception as e:
            logger.critical(f"Emergency stop execution error: {str(e)}")
    
    def reset_hardware_emergency_stop(self) -> bool:
        """Reset exoskeleton from emergency stop state."""
        try:
            logger.info("Resetting exoskeleton emergency stop")
            
            # Reset safety hardware
            if not self.safety_hardware.reset_emergency_stop():
                return False
            
            # Re-enable motors
            if not self._enable_all_motors():
                return False
            
            # Perform health check
            if not self._perform_health_check():
                return False
            
            logger.info("Exoskeleton emergency stop reset successful")
            return True
            
        except Exception as e:
            logger.error(f"Emergency stop reset error: {str(e)}")
            return False
    
    def get_expected_command_shape(self) -> Tuple[int, ...]:
        """Get expected shape of command array."""
        return (self.exo_config.num_joints,)
    
    def check_constraint_violations(self, sensor_data: Dict[str, np.ndarray]) -> List[str]:
        """Check for safety constraint violations."""
        violations = []
        
        try:
            # Check force constraints
            if 'forces' in sensor_data:
                forces = sensor_data['forces']
                if np.any(np.abs(forces) > self.exo_config.safety_force_threshold):
                    violations.append("excessive_force")
            
            # Check torque constraints
            if 'joint_torques' in sensor_data:
                torques = sensor_data['joint_torques']
                if np.any(np.abs(torques) > self.exo_config.safety_torque_threshold):
                    violations.append("excessive_torque")
            
            # Check joint angle limits
            if 'joint_positions' in sensor_data:
                positions = sensor_data['joint_positions']
                for i, (pos, (min_angle, max_angle)) in enumerate(
                    zip(positions, self.exo_config.max_joint_angles)):
                    if pos < min_angle or pos > max_angle:
                        violations.append(f"joint_{i}_limit_exceeded")
            
            # Check velocity limits
            if 'joint_velocities' in sensor_data:
                velocities = sensor_data['joint_velocities']
                max_vels = np.array(self.exo_config.max_joint_velocities)
                if np.any(np.abs(velocities) > max_vels):
                    violations.append("excessive_velocity")
            
            # Check motor temperature
            if 'motor_temperatures' in sensor_data:
                temps = sensor_data['motor_temperatures']
                if np.any(temps > 80.0):  # 80°C threshold
                    violations.append("motor_overheating")
            
            return violations
            
        except Exception as e:
            logger.error(f"Constraint checking error: {str(e)}")
            return ["constraint_check_error"]
    
    def get_hardware_diagnostics(self) -> Dict[str, Any]:
        """Get exoskeleton-specific diagnostic information."""
        try:
            diagnostics = {
                'joint_states': {
                    'positions': self.joint_positions.tolist(),
                    'velocities': self.joint_velocities.tolist(),
                    'torques': self.joint_torques.tolist()
                },
                'motor_status': {
                    'currents': self.motor_currents.tolist(),
                    'temperatures': self.motor_temperatures.tolist()
                },
                'control_mode': self.control_mode,
                'can_bus_status': self.can_bus.state.name if self.can_bus else "disconnected",
                'safety_hardware_status': self.safety_hardware.get_status(),
                'force_readings': self.force_readings.tolist(),
                'last_safety_check_age_ms': (time.time() - self.last_safety_check) * 1000
            }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Diagnostics collection error: {str(e)}")
            return {}
    
    def calibrate(self) -> bool:
        """Perform exoskeleton calibration procedure."""
        try:
            logger.info("Starting exoskeleton calibration...")
            
            # Home all joints to find zero positions
            if not self._home_all_joints():
                logger.error("Joint homing failed")
                return False
            
            # Calibrate force sensors
            if not self._calibrate_force_sensors():
                logger.error("Force sensor calibration failed")
                return False
            
            # Calibrate IMU sensors
            if not self._calibrate_imu_sensors():
                logger.error("IMU calibration failed")
                return False
            
            # Test safety systems
            if not self._test_safety_systems():
                logger.error("Safety system test failed")
                return False
            
            logger.info("Exoskeleton calibration complete")
            return True
            
        except Exception as e:
            logger.error(f"Calibration error: {str(e)}")
            return False
    
    def set_control_mode(self, mode: str):
        """Set the control mode for the exoskeleton."""
        if mode not in ["position", "velocity", "torque"]:
            raise ValueError(f"Invalid control mode: {mode}")
        
        logger.info(f"Setting control mode to {mode}")
        self.control_mode = mode
        
        # Send mode change commands to motors
        for i in range(self.exo_config.num_joints):
            self._set_motor_control_mode(i, mode)
    
    def get_joint_state(self) -> Dict[str, np.ndarray]:
        """Get current joint state."""
        return {
            'positions': self.joint_positions.copy(),
            'velocities': self.joint_velocities.copy(),
            'torques': self.joint_torques.copy()
        }
    
    # Private methods for hardware-specific operations
    
    def _initialize_can_bus(self) -> bool:
        """Initialize CAN bus communication."""
        try:
            logger.info(f"Initializing CAN bus: {self.exo_config.can_bus_name}")
            
            self.can_bus = can.interface.Bus(
                channel=self.exo_config.can_bus_name,
                bustype='socketcan',
                bitrate=self.exo_config.can_bitrate
            )
            
            # Test CAN communication
            test_msg = can.Message(arbitration_id=0x7DF, data=[0x02, 0x01, 0x0C], is_extended_id=False)
            self.can_bus.send(test_msg)
            
            logger.info("CAN bus initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"CAN bus initialization error: {str(e)}")
            return False
    
    def _initialize_motors(self) -> bool:
        """Initialize motor controllers."""
        try:
            logger.info("Initializing motor controllers...")
            
            for i, motor_id in enumerate(self.exo_config.motor_can_ids):
                # Send initialization command to motor
                init_msg = can.Message(
                    arbitration_id=motor_id,
                    data=[0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                    is_extended_id=False
                )
                
                with self.can_lock:
                    self.can_bus.send(init_msg)
                
                # Wait for motor response
                time.sleep(0.01)
                
                self.motor_controllers[i] = {
                    'can_id': motor_id,
                    'initialized': True,
                    'enabled': False
                }
            
            logger.info(f"Initialized {len(self.motor_controllers)} motor controllers")
            return True
            
        except Exception as e:
            logger.error(f"Motor initialization error: {str(e)}")
            return False
    
    def _initialize_sensors(self) -> bool:
        """Initialize sensor interfaces."""
        try:
            logger.info("Initializing sensors...")
            
            # Initialize force sensors
            if self.exo_config.force_sensor_locations:
                for i, location in enumerate(self.exo_config.force_sensor_locations):
                    force_config = {
                        'sensor_id': f"force_{i}",
                        'location': location,
                        'can_id': self.exo_config.sensor_can_ids[i]
                    }
                    self.force_sensors[location] = ForceInterface(force_config)
            
            # Initialize IMU sensors
            if self.exo_config.imu_locations:
                for location in self.exo_config.imu_locations:
                    imu_config = {
                        'sensor_id': f"imu_{location}",
                        'location': location
                    }
                    self.imu_sensors[location] = IMUInterface(imu_config)
            
            # Initialize encoders
            for i in range(self.exo_config.num_joints):
                encoder_config = {
                    'sensor_id': f"encoder_{i}",
                    'resolution': self.exo_config.encoder_resolution[i] if self.exo_config.encoder_resolution else 4096,
                    'can_id': self.exo_config.sensor_can_ids[i]
                }
                self.encoders[i] = EncoderInterface(encoder_config)
            
            logger.info("Sensor initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Sensor initialization error: {str(e)}")
            return False
    
    def _perform_health_check(self) -> bool:
        """Perform initial health check."""
        try:
            logger.info("Performing hardware health check...")
            
            # Check CAN bus communication
            if not self.can_bus or self.can_bus.state.name != 'ACTIVE':
                logger.error("CAN bus not active")
                return False
            
            # Check motor responses
            for i in range(self.exo_config.num_joints):
                if not self._check_motor_health(i):
                    logger.error(f"Motor {i} health check failed")
                    return False
            
            # Check safety systems
            if not self.safety_hardware.perform_self_test():
                logger.error("Safety system health check failed")
                return False
            
            logger.info("Hardware health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
    
    def _send_motor_command(self, joint_index: int, command_value: float) -> bool:
        """Send command to specific motor."""
        try:
            motor_id = self.exo_config.motor_can_ids[joint_index]
            
            # Convert command to motor units
            if self.control_mode == "torque":
                # Convert Nm to motor current (A)
                current = command_value / self.exo_config.torque_constants[joint_index]
                command_int = int(current * 1000)  # mA
            elif self.control_mode == "position":
                # Convert rad to encoder counts
                counts = command_value * self.exo_config.encoder_resolution[joint_index] / (2 * np.pi)
                command_int = int(counts)
            else:  # velocity
                # Convert rad/s to RPM
                rpm = command_value * 60 / (2 * np.pi)
                command_int = int(rpm * 10)  # 0.1 RPM units
            
            # Create CAN message
            command_bytes = struct.pack('<i', command_int)
            mode_byte = {'torque': 0x01, 'position': 0x02, 'velocity': 0x03}[self.control_mode]
            
            msg = can.Message(
                arbitration_id=motor_id,
                data=[mode_byte] + list(command_bytes) + [0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            with self.can_lock:
                self.can_bus.send(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Motor command error (joint {joint_index}): {str(e)}")
            return False
    
    def _update_joint_states(self):
        """Update joint position, velocity, and torque from sensors."""
        try:
            for i in range(self.exo_config.num_joints):
                # Read encoder data
                encoder_data = self.encoders[i].read()
                if encoder_data:
                    self.joint_positions[i] = encoder_data['position']
                    self.joint_velocities[i] = encoder_data['velocity']
                
                # Read motor torque feedback
                torque_feedback = self._read_motor_torque(i)
                if torque_feedback is not None:
                    self.joint_torques[i] = torque_feedback
                
                # Read motor current and temperature
                motor_status = self._read_motor_status(i)
                if motor_status:
                    self.motor_currents[i] = motor_status['current']
                    self.motor_temperatures[i] = motor_status['temperature']
                    
        except Exception as e:
            logger.error(f"Joint state update error: {str(e)}")
    
    def _read_force_sensors(self) -> np.ndarray:
        """Read data from all force sensors."""
        force_data = np.zeros(len(self.force_sensors))
        
        try:
            for i, (location, sensor) in enumerate(self.force_sensors.items()):
                reading = sensor.read()
                if reading:
                    force_data[i] = reading['force_magnitude']
            
            return force_data
            
        except Exception as e:
            logger.error(f"Force sensor reading error: {str(e)}")
            return force_data
    
    def _read_imu_sensors(self) -> Dict[str, np.ndarray]:
        """Read data from all IMU sensors."""
        imu_data = {}
        
        try:
            for location, sensor in self.imu_sensors.items():
                reading = sensor.read()
                if reading:
                    imu_data[f'imu_{location}_accel'] = reading['acceleration']
                    imu_data[f'imu_{location}_gyro'] = reading['angular_velocity']
                    imu_data[f'imu_{location}_orient'] = reading['orientation']
            
            return imu_data
            
        except Exception as e:
            logger.error(f"IMU sensor reading error: {str(e)}")
            return {}
    
    def _stop_all_motors(self):
        """Safely stop all motors."""
        try:
            logger.info("Stopping all motors...")
            
            # Send zero torque commands
            zero_commands = np.zeros(self.exo_config.num_joints)
            
            old_mode = self.control_mode
            self.set_control_mode("torque")
            
            for _ in range(10):  # Send multiple times for safety
                self.send_commands(zero_commands)
                time.sleep(0.01)
            
            # Disable motors
            for i in range(self.exo_config.num_joints):
                self._disable_motor(i)
            
            self.control_mode = old_mode
            logger.info("All motors stopped")
            
        except Exception as e:
            logger.error(f"Motor stop error: {str(e)}")
    
    def _emergency_brake_all_motors(self):
        """Apply emergency brake to all motors."""
        try:
            # Send emergency stop command to all motors
            for motor_id in self.exo_config.motor_can_ids:
                emergency_msg = can.Message(
                    arbitration_id=motor_id,
                    data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
                    is_extended_id=False
                )
                
                with self.can_lock:
                    self.can_bus.send(emergency_msg)
            
        except Exception as e:
            logger.critical(f"Emergency brake error: {str(e)}")
    
    def _enable_all_motors(self) -> bool:
        """Enable all motors after emergency stop."""
        try:
            success = True
            for i in range(self.exo_config.num_joints):
                if not self._enable_motor(i):
                    success = False
            return success
            
        except Exception as e:
            logger.error(f"Motor enable error: {str(e)}")
            return False
    
    def _enable_motor(self, joint_index: int) -> bool:
        """Enable specific motor."""
        try:
            motor_id = self.exo_config.motor_can_ids[joint_index]
            enable_msg = can.Message(
                arbitration_id=motor_id,
                data=[0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            with self.can_lock:
                self.can_bus.send(enable_msg)
            
            self.motor_controllers[joint_index]['enabled'] = True
            return True
            
        except Exception as e:
            logger.error(f"Motor enable error (joint {joint_index}): {str(e)}")
            return False
    
    def _disable_motor(self, joint_index: int) -> bool:
        """Disable specific motor."""
        try:
            motor_id = self.exo_config.motor_can_ids[joint_index]
            disable_msg = can.Message(
                arbitration_id=motor_id,
                data=[0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            with self.can_lock:
                self.can_bus.send(disable_msg)
            
            self.motor_controllers[joint_index]['enabled'] = False
            return True
            
        except Exception as e:
            logger.error(f"Motor disable error (joint {joint_index}): {str(e)}")
            return False
    
    def _check_motor_health(self, joint_index: int) -> bool:
        """Check health of specific motor."""
        try:
            motor_id = self.exo_config.motor_can_ids[joint_index]
            
            # Send status request
            status_req = can.Message(
                arbitration_id=motor_id,
                data=[0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            with self.can_lock:
                self.can_bus.send(status_req)
                
                # Wait for response
                response = self.can_bus.recv(timeout=0.1)
                
                if response and response.arbitration_id == motor_id + 0x80:
                    # Parse status response
                    status_byte = response.data[0]
                    return (status_byte & 0x01) == 0  # No error flag
            
            return False
            
        except Exception as e:
            logger.error(f"Motor health check error (joint {joint_index}): {str(e)}")
            return False
    
    def _read_motor_torque(self, joint_index: int) -> Optional[float]:
        """Read torque feedback from motor."""
        try:
            # Implementation depends on specific motor controller
            # This is a placeholder for actual torque reading
            return self.target_torques[joint_index]  # Simplified
            
        except Exception as e:
            logger.error(f"Motor torque read error (joint {joint_index}): {str(e)}")
            return None
    
    def _read_motor_status(self, joint_index: int) -> Optional[Dict[str, float]]:
        """Read motor status (current, temperature, etc.)."""
        try:
            # Implementation depends on specific motor controller
            # This is a placeholder for actual status reading
            return {
                'current': abs(self.target_torques[joint_index] * 2.0),  # Simplified
                'temperature': 25.0 + abs(self.target_torques[joint_index]) * 5.0  # Simplified
            }
            
        except Exception as e:
            logger.error(f"Motor status read error (joint {joint_index}): {str(e)}")
            return None
    
    def _set_motor_control_mode(self, joint_index: int, mode: str):
        """Set control mode for specific motor."""
        try:
            motor_id = self.exo_config.motor_can_ids[joint_index]
            mode_byte = {'torque': 0x01, 'position': 0x02, 'velocity': 0x03}[mode]
            
            mode_msg = can.Message(
                arbitration_id=motor_id,
                data=[0x20, mode_byte, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                is_extended_id=False
            )
            
            with self.can_lock:
                self.can_bus.send(mode_msg)
                
        except Exception as e:
            logger.error(f"Motor mode set error (joint {joint_index}): {str(e)}")
    
    def _home_all_joints(self) -> bool:
        """Home all joints to find zero positions."""
        try:
            logger.info("Homing all joints...")
            
            # Implementation depends on specific homing procedure
            # This is a simplified version
            for i in range(self.exo_config.num_joints):
                # Move to mechanical limits and set zero
                if not self._home_joint(i):
                    return False
            
            logger.info("Joint homing complete")
            return True
            
        except Exception as e:
            logger.error(f"Joint homing error: {str(e)}")
            return False
    
    def _home_joint(self, joint_index: int) -> bool:
        """Home specific joint."""
        try:
            # Simplified homing procedure
            # In real implementation, this would involve:
            # 1. Move slowly in one direction until limit switch
            # 2. Back off and approach slowly
            # 3. Set zero position
            
            logger.info(f"Homing joint {joint_index}")
            time.sleep(1.0)  # Simulate homing time
            
            # Reset position to zero
            self.joint_positions[joint_index] = 0.0
            
            return True
            
        except Exception as e:
            logger.error(f"Joint homing error (joint {joint_index}): {str(e)}")
            return False
    
    def _calibrate_force_sensors(self) -> bool:
        """Calibrate force sensors."""
        try:
            logger.info("Calibrating force sensors...")
            
            for location, sensor in self.force_sensors.items():
                if not sensor.calibrate():
                    logger.error(f"Force sensor calibration failed: {location}")
                    return False
            
            logger.info("Force sensor calibration complete")
            return True
            
        except Exception as e:
            logger.error(f"Force sensor calibration error: {str(e)}")
            return False
    
    def _calibrate_imu_sensors(self) -> bool:
        """Calibrate IMU sensors."""
        try:
            logger.info("Calibrating IMU sensors...")
            
            for location, sensor in self.imu_sensors.items():
                if not sensor.calibrate():
                    logger.error(f"IMU calibration failed: {location}")
                    return False
            
            logger.info("IMU calibration complete")
            return True
            
        except Exception as e:
            logger.error(f"IMU calibration error: {str(e)}")
            return False
    
    def _test_safety_systems(self) -> bool:
        """Test safety systems functionality."""
        try:
            logger.info("Testing safety systems...")
            
            # Test emergency stop
            if not self.safety_hardware.test_emergency_stop():
                logger.error("Emergency stop test failed")
                return False
            
            # Test constraint monitoring
            if not self.safety_hardware.test_constraint_monitoring():
                logger.error("Constraint monitoring test failed")
                return False
            
            logger.info("Safety system test complete")
            return True
            
        except Exception as e:
            logger.error(f"Safety system test error: {str(e)}")
            return False