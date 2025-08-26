"""
Multi-Modal Data Collection Module for Human-Robot Shared Control.

This module provides comprehensive data collection from multiple sensor modalities
including EMG, force sensors, kinematics tracking, and eye-tracking with
real-time synchronization, preprocessing, and quality assessment.

Supported Modalities:
- EMG: Surface electromyography from multiple muscle groups
- Force: 6-DOF force/torque sensors and grip force
- Kinematics: Position, velocity, acceleration tracking
- Eye-tracking: Gaze position, pupil diameter, fixations
- Physiological: Heart rate, skin conductance (optional)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import threading
import time
import queue
from abc import ABC, abstractmethod
import logging
from collections import deque
from scipy.signal import butter, filtfilt, welch
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorConfiguration:
    """Configuration for individual sensors."""
    sensor_id: str
    sensor_type: str  # 'emg', 'force', 'kinematics', 'eye_tracking', 'physiological'
    sampling_rate: float  # Hz
    channels: List[str]
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class SensorReading:
    """Individual sensor reading with timestamp and quality metrics."""
    sensor_id: str
    timestamp: float
    data: Union[np.ndarray, Dict[str, float]]
    quality_score: float = 1.0
    processing_latency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalFrame:
    """Synchronized multi-modal data frame."""
    timestamp: float
    frame_id: int
    readings: Dict[str, SensorReading] = field(default_factory=dict)
    synchronization_quality: float = 1.0
    processing_latency: float = 0.0
    
    def get_sensor_data(self, sensor_type: str) -> Optional[Union[np.ndarray, Dict[str, float]]]:
        """Get data from specific sensor type."""
        for sensor_id, reading in self.readings.items():
            if sensor_id.startswith(sensor_type):
                return reading.data
        return None
    
    def is_complete(self, required_sensors: List[str]) -> bool:
        """Check if frame contains all required sensor data."""
        available_types = [sensor_id.split('_')[0] for sensor_id in self.readings.keys()]
        return all(req_sensor in available_types for req_sensor in required_sensors)


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces."""
    
    def __init__(self, config: SensorConfiguration):
        self.config = config
        self.is_connected = False
        self.is_recording = False
        self.data_callback: Optional[Callable] = None
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to sensor hardware."""
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from sensor hardware."""
        pass
    
    @abstractmethod
    def start_recording(self):
        """Start data acquisition."""
        pass
    
    @abstractmethod
    def stop_recording(self):
        """Stop data acquisition."""
        pass
    
    @abstractmethod
    def read_sample(self) -> Optional[SensorReading]:
        """Read a single sample from sensor."""
        pass
    
    @abstractmethod
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate sensor."""
        pass


class EMGInterface(SensorInterface):
    """Interface for EMG sensors."""
    
    def __init__(self, config: SensorConfiguration):
        super().__init__(config)
        self.baseline_values = {}
        self.noise_levels = {}
        
        # EMG processing parameters
        self.filter_params = config.preprocessing_params.get('filter', {
            'highpass_cutoff': 20.0,  # Hz
            'lowpass_cutoff': 450.0,  # Hz
            'notch_freq': 60.0,  # Hz (power line)
            'order': 4
        })
        
        self.envelope_params = config.preprocessing_params.get('envelope', {
            'rectify': True,
            'smoothing_window': 0.05,  # seconds
            'rms_window': 0.1  # seconds
        })
    
    def connect(self) -> bool:
        """Connect to EMG hardware."""
        # Simulated connection - replace with actual hardware interface
        logger.info(f"Connecting to EMG sensor {self.config.sensor_id}")
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from EMG hardware."""
        self.is_connected = False
        self.is_recording = False
        logger.info(f"Disconnected EMG sensor {self.config.sensor_id}")
    
    def start_recording(self):
        """Start EMG recording."""
        if not self.is_connected:
            raise RuntimeError("EMG sensor not connected")
        self.is_recording = True
        logger.info(f"Started EMG recording on {self.config.sensor_id}")
    
    def stop_recording(self):
        """Stop EMG recording."""
        self.is_recording = False
        logger.info(f"Stopped EMG recording on {self.config.sensor_id}")
    
    def read_sample(self) -> Optional[SensorReading]:
        """Read EMG sample."""
        if not self.is_recording:
            return None
        
        # Simulate EMG data - replace with actual hardware reading
        n_channels = len(self.config.channels)
        raw_data = np.random.normal(0, 0.1, n_channels) + \
                  np.random.normal(0, 0.05, n_channels) * np.sin(2 * np.pi * 60 * time.time())
        
        # Add some muscle activation patterns
        activation_pattern = {
            'biceps': 0.3 * np.sin(2 * np.pi * 0.5 * time.time()) + 0.1,
            'triceps': 0.2 * np.cos(2 * np.pi * 0.3 * time.time()) + 0.1,
            'deltoid_anterior': 0.25 * np.sin(2 * np.pi * 0.7 * time.time()) + 0.1,
            'deltoid_posterior': 0.15 * np.cos(2 * np.pi * 0.4 * time.time()) + 0.1
        }
        
        # Apply preprocessing
        processed_data = self._preprocess_emg(raw_data, activation_pattern)
        
        # Quality assessment
        quality_score = self._assess_emg_quality(raw_data, processed_data)
        
        return SensorReading(
            sensor_id=self.config.sensor_id,
            timestamp=time.time(),
            data=processed_data,
            quality_score=quality_score,
            metadata={'raw_data': raw_data}
        )
    
    def _preprocess_emg(self, raw_data: np.ndarray, 
                       activation_pattern: Dict[str, float]) -> Dict[str, float]:
        """Preprocess EMG data."""
        # Band-pass filtering
        nyquist = self.config.sampling_rate / 2
        high_norm = self.filter_params['highpass_cutoff'] / nyquist
        low_norm = self.filter_params['lowpass_cutoff'] / nyquist
        
        # Simulate filtering (in real implementation, maintain filter state)
        filtered_data = raw_data.copy()  # Simplified
        
        # Rectification
        if self.envelope_params['rectify']:
            rectified_data = np.abs(filtered_data)
        else:
            rectified_data = filtered_data
        
        # Extract activation levels for each muscle
        processed_data = {}
        for i, channel in enumerate(self.config.channels):
            if channel in activation_pattern:
                # Combine simulated activation with noise
                muscle_activation = activation_pattern[channel] + \
                                  (rectified_data[i] if i < len(rectified_data) else 0.1)
                processed_data[channel] = max(0.0, min(1.0, muscle_activation))
            else:
                processed_data[channel] = 0.1  # Baseline activation
        
        return processed_data
    
    def _assess_emg_quality(self, raw_data: np.ndarray, 
                          processed_data: Dict[str, float]) -> float:
        """Assess EMG signal quality."""
        # Signal-to-noise ratio
        signal_power = np.mean(np.abs(raw_data))
        noise_power = np.std(raw_data)
        snr = signal_power / (noise_power + 1e-10)
        
        # Saturation check
        saturation_ratio = np.mean(np.abs(raw_data) > 0.95)
        
        # Baseline drift check (simplified)
        baseline_stability = 1.0 - min(1.0, np.std(raw_data) / 0.1)
        
        # Combined quality score
        quality_score = min(1.0, (snr / 10.0) * (1 - saturation_ratio) * baseline_stability)
        
        return max(0.0, quality_score)
    
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate EMG sensor."""
        logger.info(f"Calibrating EMG sensor {self.config.sensor_id}")
        
        if 'baseline_recording' in calibration_data:
            # Establish baseline EMG levels during rest
            baseline_data = calibration_data['baseline_recording']
            self.baseline_values = {channel: np.mean(baseline_data.get(channel, [0.1])) 
                                  for channel in self.config.channels}
        
        if 'mvc_recordings' in calibration_data:
            # Maximum voluntary contractions for normalization
            mvc_data = calibration_data['mvc_recordings']
            for channel in self.config.channels:
                if channel in mvc_data:
                    mvc_value = np.max(mvc_data[channel])
                    self.config.calibration_params[f'{channel}_mvc'] = mvc_value
        
        logger.info("EMG calibration completed")
        return True


class ForceInterface(SensorInterface):
    """Interface for force/torque sensors."""
    
    def __init__(self, config: SensorConfiguration):
        super().__init__(config)
        self.bias_values = np.zeros(6)  # 6-DOF bias
        self.calibration_matrix = np.eye(6)
        
    def connect(self) -> bool:
        """Connect to force sensor."""
        logger.info(f"Connecting to Force sensor {self.config.sensor_id}")
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from force sensor."""
        self.is_connected = False
        self.is_recording = False
        logger.info(f"Disconnected Force sensor {self.config.sensor_id}")
    
    def start_recording(self):
        """Start force recording."""
        if not self.is_connected:
            raise RuntimeError("Force sensor not connected")
        self.is_recording = True
        logger.info(f"Started Force recording on {self.config.sensor_id}")
    
    def stop_recording(self):
        """Stop force recording."""
        self.is_recording = False
        logger.info(f"Stopped Force recording on {self.config.sensor_id}")
    
    def read_sample(self) -> Optional[SensorReading]:
        """Read force/torque sample."""
        if not self.is_recording:
            return None
        
        # Simulate 6-DOF force/torque data
        # Forces: Fx, Fy, Fz (N)
        # Torques: Tx, Ty, Tz (Nm)
        raw_forces = np.array([
            5.0 * np.sin(2 * np.pi * 0.3 * time.time()) + np.random.normal(0, 0.5),  # Fx
            10.0 * np.cos(2 * np.pi * 0.2 * time.time()) + np.random.normal(0, 0.8),  # Fy
            -20.0 + 5.0 * np.sin(2 * np.pi * 0.1 * time.time()) + np.random.normal(0, 1.0),  # Fz
            0.5 * np.sin(2 * np.pi * 0.4 * time.time()) + np.random.normal(0, 0.1),  # Tx
            0.8 * np.cos(2 * np.pi * 0.3 * time.time()) + np.random.normal(0, 0.15),  # Ty
            0.3 * np.sin(2 * np.pi * 0.6 * time.time()) + np.random.normal(0, 0.08)   # Tz
        ])
        
        # Apply calibration
        calibrated_forces = np.dot(self.calibration_matrix, raw_forces - self.bias_values)
        
        # Quality assessment
        quality_score = self._assess_force_quality(raw_forces, calibrated_forces)
        
        return SensorReading(
            sensor_id=self.config.sensor_id,
            timestamp=time.time(),
            data=calibrated_forces,
            quality_score=quality_score,
            metadata={'raw_forces': raw_forces}
        )
    
    def _assess_force_quality(self, raw_forces: np.ndarray, 
                            calibrated_forces: np.ndarray) -> float:
        """Assess force sensor quality."""
        # Check for sensor saturation
        max_range = self.config.quality_thresholds.get('max_force_range', 500.0)  # N
        saturation_ratio = np.sum(np.abs(raw_forces[:3]) > max_range * 0.95) / 3
        
        # Check for excessive noise
        noise_threshold = self.config.quality_thresholds.get('noise_threshold', 2.0)  # N
        noise_levels = np.std(raw_forces[:3])  # Focus on force components
        noise_quality = max(0.0, 1.0 - noise_levels / noise_threshold)
        
        # Check for drift (simplified)
        drift_quality = 1.0  # Placeholder - would need historical data
        
        # Combined quality score
        quality_score = (1 - saturation_ratio) * noise_quality * drift_quality
        
        return max(0.0, quality_score)
    
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate force sensor."""
        logger.info(f"Calibrating Force sensor {self.config.sensor_id}")
        
        if 'bias_measurements' in calibration_data:
            # Zero-load bias estimation
            bias_data = calibration_data['bias_measurements']
            self.bias_values = np.mean(bias_data, axis=0)
        
        if 'calibration_loads' in calibration_data:
            # Known load calibration
            calibration_loads = calibration_data['calibration_loads']
            measured_values = calibration_data['measured_values']
            
            # Compute calibration matrix (simplified linear calibration)
            self.calibration_matrix = np.linalg.lstsq(
                measured_values, calibration_loads, rcond=None)[0].T
        
        logger.info("Force sensor calibration completed")
        return True


class KinematicsInterface(SensorInterface):
    """Interface for kinematic tracking systems."""
    
    def __init__(self, config: SensorConfiguration):
        super().__init__(config)
        self.reference_frame = np.eye(4)  # 4x4 transformation matrix
        self.previous_position = np.zeros(3)
        self.previous_velocity = np.zeros(3)
        self.previous_timestamp = 0.0
        
    def connect(self) -> bool:
        """Connect to motion tracking system."""
        logger.info(f"Connecting to Kinematics sensor {self.config.sensor_id}")
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from motion tracking system."""
        self.is_connected = False
        self.is_recording = False
        logger.info(f"Disconnected Kinematics sensor {self.config.sensor_id}")
    
    def start_recording(self):
        """Start kinematic recording."""
        if not self.is_connected:
            raise RuntimeError("Kinematics sensor not connected")
        self.is_recording = True
        self.previous_timestamp = time.time()
        logger.info(f"Started Kinematics recording on {self.config.sensor_id}")
    
    def stop_recording(self):
        """Stop kinematic recording."""
        self.is_recording = False
        logger.info(f"Stopped Kinematics recording on {self.config.sensor_id}")
    
    def read_sample(self) -> Optional[SensorReading]:
        """Read kinematic data sample."""
        if not self.is_recording:
            return None
        
        current_time = time.time()
        dt = current_time - self.previous_timestamp
        
        # Simulate 3D position tracking (hand/end-effector)
        t = current_time
        position = np.array([
            0.3 + 0.1 * np.sin(2 * np.pi * 0.2 * t),  # X (m)
            0.2 + 0.05 * np.cos(2 * np.pi * 0.3 * t),  # Y (m)
            0.4 + 0.03 * np.sin(2 * np.pi * 0.1 * t)   # Z (m)
        ]) + np.random.normal(0, 0.002, 3)  # Add measurement noise
        
        # Compute velocity and acceleration
        if dt > 0:
            velocity = (position - self.previous_position) / dt
            acceleration = (velocity - self.previous_velocity) / dt
        else:
            velocity = self.previous_velocity
            acceleration = np.zeros(3)
        
        # Simulate orientation (quaternion: w, x, y, z)
        orientation = np.array([
            np.cos(0.1 * t),  # w
            0.1 * np.sin(0.1 * t),  # x
            0.05 * np.cos(0.15 * t),  # y
            0.02 * np.sin(0.2 * t)   # z
        ])
        # Normalize quaternion
        orientation = orientation / np.linalg.norm(orientation)
        
        # Organize kinematic data
        kinematic_data = {
            'position_x': position[0],
            'position_y': position[1], 
            'position_z': position[2],
            'velocity_x': velocity[0],
            'velocity_y': velocity[1],
            'velocity_z': velocity[2],
            'acceleration_x': acceleration[0],
            'acceleration_y': acceleration[1],
            'acceleration_z': acceleration[2],
            'orientation_w': orientation[0],
            'orientation_x': orientation[1],
            'orientation_y': orientation[2],
            'orientation_z': orientation[3]
        }
        
        # Update previous values
        self.previous_position = position.copy()
        self.previous_velocity = velocity.copy()
        self.previous_timestamp = current_time
        
        # Quality assessment
        quality_score = self._assess_kinematic_quality(kinematic_data)
        
        return SensorReading(
            sensor_id=self.config.sensor_id,
            timestamp=current_time,
            data=kinematic_data,
            quality_score=quality_score
        )
    
    def _assess_kinematic_quality(self, kinematic_data: Dict[str, float]) -> float:
        """Assess kinematic tracking quality."""
        # Check for reasonable position values
        position = np.array([kinematic_data['position_x'], 
                           kinematic_data['position_y'], 
                           kinematic_data['position_z']])
        
        workspace_limits = self.config.quality_thresholds.get('workspace_limits', 
                                                            {'min': -1.0, 'max': 1.0})
        in_workspace = np.all((position >= workspace_limits['min']) & 
                             (position <= workspace_limits['max']))
        
        # Check for reasonable velocity values
        velocity = np.array([kinematic_data['velocity_x'],
                           kinematic_data['velocity_y'],
                           kinematic_data['velocity_z']])
        
        velocity_magnitude = np.linalg.norm(velocity)
        max_velocity = self.config.quality_thresholds.get('max_velocity', 5.0)  # m/s
        velocity_reasonable = velocity_magnitude <= max_velocity
        
        # Check orientation quaternion validity
        orientation = np.array([kinematic_data['orientation_w'],
                              kinematic_data['orientation_x'],
                              kinematic_data['orientation_y'],
                              kinematic_data['orientation_z']])
        
        quaternion_valid = abs(np.linalg.norm(orientation) - 1.0) < 0.1
        
        # Combined quality score
        quality_score = float(in_workspace and velocity_reasonable and quaternion_valid)
        
        # Add noise-based quality metric
        if hasattr(self, 'position_history'):
            # Would compute jitter/noise metrics from history
            pass
        
        return quality_score
    
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate kinematic tracking system."""
        logger.info(f"Calibrating Kinematics sensor {self.config.sensor_id}")
        
        if 'reference_points' in calibration_data:
            # Establish coordinate system reference
            reference_points = calibration_data['reference_points']
            # Would compute transformation matrix from known reference points
            logger.info("Reference frame calibrated")
        
        if 'user_workspace' in calibration_data:
            # Define user-specific workspace limits
            workspace = calibration_data['user_workspace']
            self.config.quality_thresholds['workspace_limits'] = workspace
        
        logger.info("Kinematics calibration completed")
        return True


class EyeTrackingInterface(SensorInterface):
    """Interface for eye-tracking systems."""
    
    def __init__(self, config: SensorConfiguration):
        super().__init__(config)
        self.calibration_points = {}
        self.screen_resolution = (1920, 1080)  # Default screen resolution
        
    def connect(self) -> bool:
        """Connect to eye-tracking hardware."""
        logger.info(f"Connecting to Eye-tracking sensor {self.config.sensor_id}")
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from eye-tracking hardware."""
        self.is_connected = False
        self.is_recording = False
        logger.info(f"Disconnected Eye-tracking sensor {self.config.sensor_id}")
    
    def start_recording(self):
        """Start eye-tracking recording."""
        if not self.is_connected:
            raise RuntimeError("Eye-tracking sensor not connected")
        self.is_recording = True
        logger.info(f"Started Eye-tracking recording on {self.config.sensor_id}")
    
    def stop_recording(self):
        """Stop eye-tracking recording."""
        self.is_recording = False
        logger.info(f"Stopped Eye-tracking recording on {self.config.sensor_id}")
    
    def read_sample(self) -> Optional[SensorReading]:
        """Read eye-tracking sample."""
        if not self.is_recording:
            return None
        
        current_time = time.time()
        
        # Simulate eye-tracking data
        # Gaze position in normalized coordinates (0-1)
        gaze_x = 0.5 + 0.2 * np.sin(2 * np.pi * 0.1 * current_time) + np.random.normal(0, 0.02)
        gaze_y = 0.5 + 0.1 * np.cos(2 * np.pi * 0.15 * current_time) + np.random.normal(0, 0.02)
        
        # Clamp to valid range
        gaze_x = max(0, min(1, gaze_x))
        gaze_y = max(0, min(1, gaze_y))
        
        # Pupil diameter (mm)
        pupil_diameter = 4.0 + 0.5 * np.sin(2 * np.pi * 0.05 * current_time) + \
                        np.random.normal(0, 0.1)
        pupil_diameter = max(2.0, min(8.0, pupil_diameter))
        
        # Fixation detection (simplified)
        fixation_duration = np.random.exponential(0.3)  # seconds
        
        # Blink detection
        is_blink = np.random.random() < 0.02  # 2% chance of blink
        
        eye_data = {
            'gaze_x': gaze_x,
            'gaze_y': gaze_y,
            'pupil_diameter': pupil_diameter,
            'fixation_duration': fixation_duration,
            'is_blink': is_blink,
            'tracking_quality': np.random.uniform(0.8, 1.0)  # Tracking confidence
        }
        
        # Quality assessment
        quality_score = self._assess_eye_tracking_quality(eye_data)
        
        return SensorReading(
            sensor_id=self.config.sensor_id,
            timestamp=current_time,
            data=eye_data,
            quality_score=quality_score
        )
    
    def _assess_eye_tracking_quality(self, eye_data: Dict[str, float]) -> float:
        """Assess eye-tracking data quality."""
        # Check gaze position validity
        gaze_valid = (0 <= eye_data['gaze_x'] <= 1) and (0 <= eye_data['gaze_y'] <= 1)
        
        # Check pupil diameter reasonableness
        pupil_valid = 2.0 <= eye_data['pupil_diameter'] <= 8.0
        
        # Use tracking quality from hardware
        tracking_quality = eye_data.get('tracking_quality', 0.5)
        
        # Penalize blinks
        blink_penalty = 0.0 if eye_data.get('is_blink', False) else 1.0
        
        # Combined quality score
        quality_score = (float(gaze_valid and pupil_valid) * 
                        tracking_quality * blink_penalty)
        
        return quality_score
    
    def calibrate(self, calibration_data: Dict[str, Any]) -> bool:
        """Calibrate eye-tracking system."""
        logger.info(f"Calibrating Eye-tracking sensor {self.config.sensor_id}")
        
        if 'calibration_points' in calibration_data:
            # Standard 9-point calibration or similar
            self.calibration_points = calibration_data['calibration_points']
            logger.info(f"Calibrated with {len(self.calibration_points)} points")
        
        if 'screen_resolution' in calibration_data:
            self.screen_resolution = calibration_data['screen_resolution']
        
        logger.info("Eye-tracking calibration completed")
        return True


class DataSynchronizer:
    """Synchronizes multi-modal sensor data streams."""
    
    def __init__(self, target_framerate: float = 100.0, 
                 sync_tolerance: float = 0.01):
        self.target_framerate = target_framerate
        self.sync_tolerance = sync_tolerance  # seconds
        self.frame_period = 1.0 / target_framerate
        
        # Buffer for incoming sensor data
        self.sensor_buffers: Dict[str, deque] = {}
        self.frame_counter = 0
        self.start_time = None
        
        # Interpolation objects for resampling
        self.interpolators: Dict[str, Dict[str, Any]] = {}
        
    def add_sensor_data(self, reading: SensorReading):
        """Add sensor reading to synchronization buffer."""
        sensor_id = reading.sensor_id
        
        if sensor_id not in self.sensor_buffers:
            self.sensor_buffers[sensor_id] = deque(maxlen=1000)  # Circular buffer
        
        self.sensor_buffers[sensor_id].append(reading)
    
    def get_synchronized_frame(self, target_timestamp: Optional[float] = None) -> Optional[MultiModalFrame]:
        """Get synchronized multi-modal frame."""
        if target_timestamp is None:
            if self.start_time is None:
                self.start_time = time.time()
                return None
            target_timestamp = self.start_time + self.frame_counter * self.frame_period
        
        frame = MultiModalFrame(
            timestamp=target_timestamp,
            frame_id=self.frame_counter
        )
        
        sync_errors = []
        
        # Extract data from each sensor buffer
        for sensor_id, buffer in self.sensor_buffers.items():
            if not buffer:
                continue
            
            # Find closest readings in time
            readings = list(buffer)
            timestamps = [r.timestamp for r in readings]
            
            if not timestamps:
                continue
            
            # Find readings bracketing target timestamp
            before_idx = None
            after_idx = None
            
            for i, ts in enumerate(timestamps):
                if ts <= target_timestamp:
                    before_idx = i
                elif ts > target_timestamp and after_idx is None:
                    after_idx = i
                    break
            
            # Interpolate or use closest reading
            if before_idx is not None and after_idx is not None:
                # Interpolate between readings
                reading_before = readings[before_idx]
                reading_after = readings[after_idx]
                
                interpolated_reading = self._interpolate_readings(
                    reading_before, reading_after, target_timestamp)
                
                frame.readings[sensor_id] = interpolated_reading
                
                # Track synchronization error
                sync_error = min(abs(target_timestamp - reading_before.timestamp),
                               abs(target_timestamp - reading_after.timestamp))
                sync_errors.append(sync_error)
                
            elif before_idx is not None:
                # Use most recent reading
                reading = readings[before_idx]
                sync_error = abs(target_timestamp - reading.timestamp)
                
                if sync_error <= self.sync_tolerance:
                    frame.readings[sensor_id] = reading
                    sync_errors.append(sync_error)
            
            elif after_idx is not None:
                # Use next reading if close enough
                reading = readings[after_idx]
                sync_error = abs(target_timestamp - reading.timestamp)
                
                if sync_error <= self.sync_tolerance:
                    frame.readings[sensor_id] = reading
                    sync_errors.append(sync_error)
        
        # Compute synchronization quality
        if sync_errors:
            max_sync_error = max(sync_errors)
            frame.synchronization_quality = max(0.0, 1.0 - max_sync_error / self.sync_tolerance)
        else:
            frame.synchronization_quality = 0.0
        
        self.frame_counter += 1
        
        return frame if frame.readings else None
    
    def _interpolate_readings(self, reading1: SensorReading, reading2: SensorReading,
                            target_timestamp: float) -> SensorReading:
        """Interpolate between two sensor readings."""
        if reading1.sensor_id != reading2.sensor_id:
            raise ValueError("Cannot interpolate readings from different sensors")
        
        # Linear interpolation factor
        dt = reading2.timestamp - reading1.timestamp
        if dt == 0:
            return reading1
        
        alpha = (target_timestamp - reading1.timestamp) / dt
        alpha = max(0, min(1, alpha))
        
        # Interpolate data based on type
        if isinstance(reading1.data, np.ndarray) and isinstance(reading2.data, np.ndarray):
            # Numpy array interpolation
            interpolated_data = (1 - alpha) * reading1.data + alpha * reading2.data
            
        elif isinstance(reading1.data, dict) and isinstance(reading2.data, dict):
            # Dictionary interpolation
            interpolated_data = {}
            for key in reading1.data.keys():
                if key in reading2.data:
                    val1 = reading1.data[key]
                    val2 = reading2.data[key]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        interpolated_data[key] = (1 - alpha) * val1 + alpha * val2
                    else:
                        # Use closest value for non-numeric data
                        interpolated_data[key] = val1 if alpha < 0.5 else val2
                else:
                    interpolated_data[key] = reading1.data[key]
        
        else:
            # Fallback: use closest reading
            interpolated_data = reading1.data if alpha < 0.5 else reading2.data
        
        # Interpolate quality score
        interpolated_quality = (1 - alpha) * reading1.quality_score + alpha * reading2.quality_score
        
        return SensorReading(
            sensor_id=reading1.sensor_id,
            timestamp=target_timestamp,
            data=interpolated_data,
            quality_score=interpolated_quality
        )


class MultiModalDataCollector:
    """Main data collection system coordinating all sensors."""
    
    def __init__(self, sensor_configs: List[SensorConfiguration]):
        self.sensor_configs = {config.sensor_id: config for config in sensor_configs}
        self.sensor_interfaces: Dict[str, SensorInterface] = {}
        self.synchronizer = DataSynchronizer()
        
        # Data collection threads
        self.collection_threads: Dict[str, threading.Thread] = {}
        self.collection_queues: Dict[str, queue.Queue] = {}
        self.stop_collection = threading.Event()
        
        # Data callbacks
        self.frame_callbacks: List[Callable[[MultiModalFrame], None]] = []
        
        # Initialize sensor interfaces
        self._initialize_sensors()
        
        # Collection statistics
        self.stats = {
            'frames_collected': 0,
            'total_runtime': 0.0,
            'average_framerate': 0.0,
            'sensor_quality': {}
        }
        
    def _initialize_sensors(self):
        """Initialize sensor interfaces based on configurations."""
        for sensor_id, config in self.sensor_configs.items():
            if config.sensor_type == 'emg':
                self.sensor_interfaces[sensor_id] = EMGInterface(config)
            elif config.sensor_type == 'force':
                self.sensor_interfaces[sensor_id] = ForceInterface(config)
            elif config.sensor_type == 'kinematics':
                self.sensor_interfaces[sensor_id] = KinematicsInterface(config)
            elif config.sensor_type == 'eye_tracking':
                self.sensor_interfaces[sensor_id] = EyeTrackingInterface(config)
            else:
                logger.warning(f"Unknown sensor type: {config.sensor_type}")
                continue
            
            # Initialize data queue for each sensor
            self.collection_queues[sensor_id] = queue.Queue()
    
    def connect_all_sensors(self) -> bool:
        """Connect to all configured sensors."""
        logger.info("Connecting to all sensors...")
        success = True
        
        for sensor_id, interface in self.sensor_interfaces.items():
            try:
                if not interface.connect():
                    logger.error(f"Failed to connect to sensor {sensor_id}")
                    success = False
            except Exception as e:
                logger.error(f"Error connecting to sensor {sensor_id}: {e}")
                success = False
        
        if success:
            logger.info("All sensors connected successfully")
        return success
    
    def calibrate_sensors(self, calibration_data: Dict[str, Dict[str, Any]]) -> bool:
        """Calibrate all sensors."""
        logger.info("Starting sensor calibration...")
        success = True
        
        for sensor_id, interface in self.sensor_interfaces.items():
            if sensor_id in calibration_data:
                try:
                    if not interface.calibrate(calibration_data[sensor_id]):
                        logger.error(f"Failed to calibrate sensor {sensor_id}")
                        success = False
                except Exception as e:
                    logger.error(f"Error calibrating sensor {sensor_id}: {e}")
                    success = False
        
        return success
    
    def start_collection(self):
        """Start multi-modal data collection."""
        logger.info("Starting multi-modal data collection...")
        
        # Clear stop flag
        self.stop_collection.clear()
        
        # Start recording on all sensors
        for interface in self.sensor_interfaces.values():
            interface.start_recording()
        
        # Start collection thread for each sensor
        for sensor_id in self.sensor_interfaces.keys():
            thread = threading.Thread(
                target=self._sensor_collection_thread,
                args=(sensor_id,),
                daemon=True
            )
            self.collection_threads[sensor_id] = thread
            thread.start()
        
        # Start synchronization thread
        sync_thread = threading.Thread(target=self._synchronization_thread, daemon=True)
        sync_thread.start()
        
        logger.info("Data collection started")
    
    def stop_collection(self):
        """Stop multi-modal data collection."""
        logger.info("Stopping multi-modal data collection...")
        
        # Set stop flag
        self.stop_collection.set()
        
        # Stop recording on all sensors
        for interface in self.sensor_interfaces.values():
            interface.stop_recording()
        
        # Wait for collection threads to finish
        for thread in self.collection_threads.values():
            thread.join(timeout=1.0)
        
        self.collection_threads.clear()
        
        logger.info("Data collection stopped")
    
    def _sensor_collection_thread(self, sensor_id: str):
        """Thread function for collecting data from a single sensor."""
        interface = self.sensor_interfaces[sensor_id]
        config = self.sensor_configs[sensor_id]
        
        # Calculate sleep time for desired sampling rate
        sleep_time = 1.0 / config.sampling_rate
        
        while not self.stop_collection.is_set():
            try:
                # Read sensor sample
                reading = interface.read_sample()
                
                if reading is not None:
                    # Add to synchronizer
                    self.synchronizer.add_sensor_data(reading)
                    
                    # Update quality statistics
                    if sensor_id not in self.stats['sensor_quality']:
                        self.stats['sensor_quality'][sensor_id] = []
                    self.stats['sensor_quality'][sensor_id].append(reading.quality_score)
                    
                    # Keep only recent quality scores
                    if len(self.stats['sensor_quality'][sensor_id]) > 100:
                        self.stats['sensor_quality'][sensor_id].pop(0)
                
                # Sleep to maintain sampling rate
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in sensor collection thread {sensor_id}: {e}")
                time.sleep(0.1)  # Brief pause before retry
    
    def _synchronization_thread(self):
        """Thread function for synchronizing multi-modal data."""
        frame_period = 1.0 / 100.0  # 100 Hz synchronization
        start_time = time.time()
        
        while not self.stop_collection.is_set():
            try:
                # Get synchronized frame
                frame = self.synchronizer.get_synchronized_frame()
                
                if frame is not None:
                    # Call registered callbacks
                    for callback in self.frame_callbacks:
                        try:
                            callback(frame)
                        except Exception as e:
                            logger.error(f"Error in frame callback: {e}")
                    
                    # Update statistics
                    self.stats['frames_collected'] += 1
                    self.stats['total_runtime'] = time.time() - start_time
                    if self.stats['total_runtime'] > 0:
                        self.stats['average_framerate'] = (self.stats['frames_collected'] / 
                                                         self.stats['total_runtime'])
                
                # Sleep until next frame
                time.sleep(frame_period)
                
            except Exception as e:
                logger.error(f"Error in synchronization thread: {e}")
                time.sleep(0.1)
    
    def register_frame_callback(self, callback: Callable[[MultiModalFrame], None]):
        """Register callback for synchronized frames."""
        self.frame_callbacks.append(callback)
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics."""
        stats = self.stats.copy()
        
        # Compute average quality scores
        quality_averages = {}
        for sensor_id, qualities in stats['sensor_quality'].items():
            if qualities:
                quality_averages[sensor_id] = np.mean(qualities)
            else:
                quality_averages[sensor_id] = 0.0
        
        stats['average_sensor_quality'] = quality_averages
        
        return stats
    
    def disconnect_all_sensors(self):
        """Disconnect from all sensors."""
        logger.info("Disconnecting from all sensors...")
        
        for interface in self.sensor_interfaces.values():
            interface.disconnect()
        
        logger.info("All sensors disconnected")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Multi-Modal Data Collection System...")
    
    # Create sensor configurations
    sensor_configs = [
        SensorConfiguration(
            sensor_id="emg_main",
            sensor_type="emg", 
            sampling_rate=1000.0,
            channels=['biceps', 'triceps', 'deltoid_anterior', 'deltoid_posterior'],
            quality_thresholds={'min_snr': 5.0, 'max_saturation': 0.1}
        ),
        SensorConfiguration(
            sensor_id="force_wrist",
            sensor_type="force",
            sampling_rate=1000.0,
            channels=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'],
            quality_thresholds={'max_force_range': 500.0, 'noise_threshold': 2.0}
        ),
        SensorConfiguration(
            sensor_id="kinematics_hand",
            sensor_type="kinematics",
            sampling_rate=120.0,
            channels=['position', 'orientation', 'velocity', 'acceleration'],
            quality_thresholds={'workspace_limits': {'min': -0.8, 'max': 0.8}, 'max_velocity': 3.0}
        ),
        SensorConfiguration(
            sensor_id="eye_tracker",
            sensor_type="eye_tracking",
            sampling_rate=60.0,
            channels=['gaze', 'pupil', 'fixation']
        )
    ]
    
    # Create data collector
    collector = MultiModalDataCollector(sensor_configs)
    
    # Connect sensors
    if collector.connect_all_sensors():
        print("All sensors connected successfully")
        
        # Calibration data (simulated)
        calibration_data = {
            'emg_main': {
                'baseline_recording': {'biceps': [0.1, 0.1], 'triceps': [0.1, 0.1]},
                'mvc_recordings': {'biceps': [0.8, 0.9, 0.85], 'triceps': [0.7, 0.8, 0.75]}
            },
            'force_wrist': {
                'bias_measurements': np.zeros((10, 6)),  # 10 zero-load measurements
            },
            'kinematics_hand': {
                'reference_points': {'origin': [0, 0, 0], 'x_axis': [1, 0, 0]},
                'user_workspace': {'min': -0.5, 'max': 0.5}
            },
            'eye_tracker': {
                'calibration_points': [(0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
                                     (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
                                     (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)]
            }
        }
        
        # Calibrate sensors
        if collector.calibrate_sensors(calibration_data):
            print("Sensor calibration completed")
            
            # Register frame callback
            def frame_callback(frame: MultiModalFrame):
                print(f"Frame {frame.frame_id}: {len(frame.readings)} sensors, "
                      f"sync_quality={frame.synchronization_quality:.3f}")
            
            collector.register_frame_callback(frame_callback)
            
            # Start collection
            collector.start_collection()
            
            # Collect data for 5 seconds
            print("Collecting data for 5 seconds...")
            time.sleep(5.0)
            
            # Stop collection
            collector.stop_collection()
            
            # Show statistics
            stats = collector.get_collection_statistics()
            print(f"\nCollection Statistics:")
            print(f"Frames collected: {stats['frames_collected']}")
            print(f"Average framerate: {stats['average_framerate']:.1f} Hz")
            print(f"Total runtime: {stats['total_runtime']:.1f} seconds")
            print(f"Average sensor quality: {stats['average_sensor_quality']}")
        
        # Disconnect sensors
        collector.disconnect_all_sensors()
    
    print("\nMulti-Modal Data Collection System test completed!")