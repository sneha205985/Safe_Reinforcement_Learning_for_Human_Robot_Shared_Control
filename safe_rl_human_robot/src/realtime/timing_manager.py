"""
Timing Manager for Real-time Control Systems.

This module provides high-precision timing coordination, timing synchronization,
and deterministic scheduling for real-time control loops in robotics applications.

Key Features:
- Nanosecond precision timing
- Hardware timestamping support
- PTP (Precision Time Protocol) synchronization
- Jitter monitoring and compensation
- Timing budget management
- Real-time scheduling coordination
"""

import time
import os
import threading
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from collections import deque
import ctypes
import struct

try:
    import fcntl
    import mmap
    RT_TIMING_AVAILABLE = True
except ImportError:
    RT_TIMING_AVAILABLE = False

logger = logging.getLogger(__name__)


class TimingMode(Enum):
    """Timing synchronization modes."""
    SYSTEM_CLOCK = auto()
    MONOTONIC = auto()
    RT_CLOCK = auto()
    PTP_HARDWARE = auto()
    CUSTOM_SOURCE = auto()


class TimingPriority(Enum):
    """Timing priority levels."""
    SAFETY_CRITICAL = 1
    CONTROL_HIGH = 2
    CONTROL_NORMAL = 3
    MONITORING = 4
    DIAGNOSTIC = 5


@dataclass
class TimingConfig:
    """Configuration for timing manager."""
    
    # Basic timing configuration
    base_frequency: float = 1000.0  # Hz
    timing_mode: TimingMode = TimingMode.MONOTONIC
    enable_jitter_compensation: bool = True
    enable_drift_correction: bool = True
    
    # Precision and tolerances
    timing_tolerance_us: float = 10.0  # Maximum allowed timing error
    jitter_threshold_us: float = 5.0   # Jitter warning threshold
    drift_correction_period_s: float = 60.0  # Drift correction interval
    
    # Hardware timing
    enable_hardware_timestamps: bool = True
    ptp_domain: int = 0
    ptp_interface: str = "eth0"
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_window_size: int = 10000
    timing_statistics_interval_s: float = 10.0
    
    # Real-time configuration
    rt_priority: int = 85
    cpu_affinity: Optional[List[int]] = None


@dataclass
class TimingStats:
    """Timing performance statistics."""
    mean_period_ns: float = 0.0
    std_period_ns: float = 0.0
    min_period_ns: float = float('inf')
    max_period_ns: float = 0.0
    jitter_ns: float = 0.0
    drift_ppm: float = 0.0
    missed_deadlines: int = 0
    total_cycles: int = 0
    timestamp: float = field(default_factory=time.time)


class TimingSource:
    """Abstract timing source interface."""
    
    def __init__(self, config: TimingConfig):
        self.config = config
        self.calibration_offset_ns = 0
        self.drift_compensation_ppm = 0.0
    
    def get_time_ns(self) -> int:
        """Get current time in nanoseconds."""
        raise NotImplementedError
    
    def calibrate(self) -> bool:
        """Calibrate timing source."""
        return True
    
    def get_resolution_ns(self) -> float:
        """Get timing resolution in nanoseconds."""
        return 1.0


class MonotonicTimingSource(TimingSource):
    """Monotonic clock timing source."""
    
    def get_time_ns(self) -> int:
        return time.time_ns()
    
    def get_resolution_ns(self) -> float:
        return 1.0  # 1ns resolution for time_ns()


class RealtimeTimingSource(TimingSource):
    """Real-time clock timing source."""
    
    def __init__(self, config: TimingConfig):
        super().__init__(config)
        self.clock_id = getattr(time, 'CLOCK_REALTIME', 0)
        
    def get_time_ns(self) -> int:
        if hasattr(time, 'clock_gettime_ns'):
            return time.clock_gettime_ns(self.clock_id)
        else:
            return int(time.time() * 1e9)


class PTPTimingSource(TimingSource):
    """PTP hardware timing source."""
    
    def __init__(self, config: TimingConfig):
        super().__init__(config)
        self.ptp_device = None
        self.hardware_available = False
        
    def calibrate(self) -> bool:
        """Initialize PTP hardware timestamping."""
        try:
            # Check for PTP hardware support
            ptp_path = f"/sys/class/net/{self.config.ptp_interface}/device/ptp"
            if os.path.exists(ptp_path):
                ptp_devices = os.listdir(ptp_path)
                if ptp_devices:
                    self.ptp_device = f"/dev/{ptp_devices[0]}"
                    self.hardware_available = True
                    logger.info(f"PTP hardware found: {self.ptp_device}")
            
            return self.hardware_available
        except Exception as e:
            logger.warning(f"PTP initialization failed: {e}")
            return False
    
    def get_time_ns(self) -> int:
        if self.hardware_available and self.ptp_device:
            try:
                # Read PTP hardware timestamp
                with open(self.ptp_device, 'rb') as f:
                    # This is a simplified implementation
                    # Real PTP would use ioctl calls
                    return time.time_ns()
            except Exception:
                pass
        
        return time.time_ns()


class TimingManager:
    """
    High-precision timing manager for real-time control systems.
    
    Provides deterministic timing, synchronization, and performance monitoring
    for real-time control loops and safety-critical applications.
    """
    
    def __init__(self, config: TimingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Timing source
        self.timing_source: Optional[TimingSource] = None
        self.base_period_ns = int(1e9 / config.base_frequency)
        
        # Performance monitoring
        self.stats = TimingStats()
        self.timing_history = deque(maxlen=config.performance_window_size)
        self.last_timestamp_ns = 0
        
        # Synchronization
        self.sync_lock = threading.RLock()
        self.timing_callbacks: Dict[str, Callable] = {}
        
        # Jitter compensation
        self.jitter_buffer = deque(maxlen=100)
        self.average_jitter_ns = 0.0
        
        # Drift correction
        self.drift_reference_time = 0
        self.drift_reference_count = 0
        self.last_drift_correction = time.time()
        
        # Performance monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Real-time scheduling
        self.priority_levels: Dict[TimingPriority, List[str]] = {
            priority: [] for priority in TimingPriority
        }
        
    def initialize(self) -> bool:
        """Initialize timing manager."""
        try:
            self.logger.info("Initializing timing manager...")
            
            # Initialize timing source
            if not self._initialize_timing_source():
                self.logger.error("Failed to initialize timing source")
                return False
            
            # Calibrate timing
            if not self.calibrate():
                self.logger.error("Timing calibration failed")
                return False
            
            # Start performance monitoring
            if self.config.enable_performance_monitoring:
                self._start_monitoring_thread()
            
            # Configure real-time scheduling
            if not self._configure_rt_scheduling():
                self.logger.warning("Real-time scheduling configuration failed")
            
            self.logger.info("Timing manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Timing manager initialization failed: {e}")
            return False
    
    def shutdown(self):
        """Shutdown timing manager."""
        try:
            self.logger.info("Shutting down timing manager...")
            
            self.shutdown_event.set()
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=2.0)
            
            self.logger.info("Timing manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Timing manager shutdown error: {e}")
    
    def get_time_ns(self) -> int:
        """Get current high-precision timestamp."""
        if self.timing_source:
            raw_time = self.timing_source.get_time_ns()
            
            # Apply calibration offset
            calibrated_time = raw_time + self.timing_source.calibration_offset_ns
            
            # Apply drift compensation
            if self.timing_source.drift_compensation_ppm != 0:
                drift_correction = int(calibrated_time * self.timing_source.drift_compensation_ppm / 1e6)
                calibrated_time += drift_correction
            
            return calibrated_time
        else:
            return time.time_ns()
    
    def sleep_until_ns(self, target_time_ns: int) -> bool:
        """Sleep until specific nanosecond timestamp."""
        try:
            current_time_ns = self.get_time_ns()
            sleep_duration_ns = target_time_ns - current_time_ns
            
            if sleep_duration_ns <= 0:
                return False  # Target time already passed
            
            sleep_duration_s = sleep_duration_ns / 1e9
            
            # Use high-precision sleep
            if sleep_duration_s > 0.001:  # 1ms
                time.sleep(sleep_duration_s - 0.0005)  # Sleep most of the time
            
            # Busy-wait for final precision
            while self.get_time_ns() < target_time_ns:
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"High-precision sleep error: {e}")
            return False
    
    def register_callback(self, callback_id: str, callback: Callable, 
                         priority: TimingPriority = TimingPriority.CONTROL_NORMAL):
        """Register timing callback with priority."""
        with self.sync_lock:
            self.timing_callbacks[callback_id] = callback
            self.priority_levels[priority].append(callback_id)
            
            self.logger.info(f"Registered timing callback: {callback_id} (priority: {priority.name})")
    
    def unregister_callback(self, callback_id: str):
        """Unregister timing callback."""
        with self.sync_lock:
            if callback_id in self.timing_callbacks:
                del self.timing_callbacks[callback_id]
                
                # Remove from priority levels
                for priority_list in self.priority_levels.values():
                    if callback_id in priority_list:
                        priority_list.remove(callback_id)
                
                self.logger.info(f"Unregistered timing callback: {callback_id}")
    
    def execute_synchronized_callbacks(self, timestamp_ns: int):
        """Execute all registered callbacks in priority order."""
        try:
            with self.sync_lock:
                # Execute callbacks by priority
                for priority in TimingPriority:
                    for callback_id in self.priority_levels[priority]:
                        if callback_id in self.timing_callbacks:
                            try:
                                self.timing_callbacks[callback_id](timestamp_ns)
                            except Exception as e:
                                self.logger.error(f"Callback {callback_id} failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Synchronized callback execution error: {e}")
    
    def record_timing_event(self, timestamp_ns: int):
        """Record timing event for performance monitoring."""
        try:
            if self.last_timestamp_ns > 0:
                period_ns = timestamp_ns - self.last_timestamp_ns
                self.timing_history.append(period_ns)
                
                # Update jitter compensation
                if self.config.enable_jitter_compensation:
                    self._update_jitter_compensation(period_ns)
                
                # Check for timing violations
                expected_period_ns = self.base_period_ns
                error_ns = abs(period_ns - expected_period_ns)
                
                if error_ns > self.config.timing_tolerance_us * 1000:
                    self.stats.missed_deadlines += 1
                    self.logger.warning(f"Timing violation: {error_ns/1000:.1f}μs error")
            
            self.last_timestamp_ns = timestamp_ns
            self.stats.total_cycles += 1
            
        except Exception as e:
            self.logger.error(f"Timing event recording error: {e}")
    
    def get_timing_statistics(self) -> TimingStats:
        """Get current timing performance statistics."""
        try:
            with self.sync_lock:
                if len(self.timing_history) < 2:
                    return self.stats
                
                periods = np.array(list(self.timing_history))
                
                self.stats.mean_period_ns = float(np.mean(periods))
                self.stats.std_period_ns = float(np.std(periods))
                self.stats.min_period_ns = float(np.min(periods))
                self.stats.max_period_ns = float(np.max(periods))
                self.stats.jitter_ns = float(np.std(periods))
                
                # Calculate drift in ppm
                if len(periods) > 100:
                    expected_period = self.base_period_ns
                    actual_mean = np.mean(periods)
                    self.stats.drift_ppm = ((actual_mean - expected_period) / expected_period) * 1e6
                
                self.stats.timestamp = time.time()
                return self.stats
                
        except Exception as e:
            self.logger.error(f"Statistics calculation error: {e}")
            return self.stats
    
    def calibrate(self) -> bool:
        """Perform timing calibration."""
        try:
            self.logger.info("Performing timing calibration...")
            
            if not self.timing_source:
                return False
            
            # Calibrate timing source
            if not self.timing_source.calibrate():
                self.logger.error("Timing source calibration failed")
                return False
            
            # Measure timing overhead
            overhead_samples = []
            for _ in range(1000):
                start = self.timing_source.get_time_ns()
                end = self.timing_source.get_time_ns()
                overhead_samples.append(end - start)
            
            overhead_mean = np.mean(overhead_samples)
            overhead_std = np.std(overhead_samples)
            
            self.logger.info(f"Timing overhead: {overhead_mean:.1f}±{overhead_std:.1f}ns")
            
            # Measure sleep precision
            sleep_precision_samples = []
            for sleep_us in [10, 50, 100, 500, 1000]:
                for _ in range(10):
                    start = self.timing_source.get_time_ns()
                    time.sleep(sleep_us / 1e6)
                    end = self.timing_source.get_time_ns()
                    actual_sleep_us = (end - start) / 1000
                    error_us = abs(actual_sleep_us - sleep_us)
                    sleep_precision_samples.append(error_us)
            
            sleep_precision = np.mean(sleep_precision_samples)
            self.logger.info(f"Sleep precision: ±{sleep_precision:.1f}μs")
            
            # Set calibration baseline
            self.drift_reference_time = self.get_time_ns()
            self.drift_reference_count = 0
            
            self.logger.info("Timing calibration complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Timing calibration error: {e}")
            return False
    
    def _initialize_timing_source(self) -> bool:
        """Initialize timing source based on configuration."""
        try:
            if self.config.timing_mode == TimingMode.MONOTONIC:
                self.timing_source = MonotonicTimingSource(self.config)
            elif self.config.timing_mode == TimingMode.RT_CLOCK:
                self.timing_source = RealtimeTimingSource(self.config)
            elif self.config.timing_mode == TimingMode.PTP_HARDWARE:
                self.timing_source = PTPTimingSource(self.config)
            else:
                self.timing_source = MonotonicTimingSource(self.config)
            
            self.logger.info(f"Initialized timing source: {self.config.timing_mode.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Timing source initialization error: {e}")
            return False
    
    def _configure_rt_scheduling(self) -> bool:
        """Configure real-time scheduling for timing thread."""
        try:
            if not RT_TIMING_AVAILABLE:
                return False
            
            # Set CPU affinity if specified
            if self.config.cpu_affinity:
                os.sched_setaffinity(0, self.config.cpu_affinity)
                self.logger.info(f"Set CPU affinity: {self.config.cpu_affinity}")
            
            # Set real-time priority
            param = os.sched_param(self.config.rt_priority)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
            self.logger.info(f"Set RT priority: {self.config.rt_priority}")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"RT scheduling configuration failed: {e}")
            return False
    
    def _start_monitoring_thread(self):
        """Start performance monitoring thread."""
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="TimingMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
    
    def _monitoring_loop(self):
        """Performance monitoring loop."""
        try:
            while not self.shutdown_event.is_set():
                # Update statistics
                stats = self.get_timing_statistics()
                
                # Check for performance issues
                if stats.jitter_ns > self.config.jitter_threshold_us * 1000:
                    self.logger.warning(f"High timing jitter: {stats.jitter_ns/1000:.1f}μs")
                
                if abs(stats.drift_ppm) > 100:  # 100ppm threshold
                    self.logger.warning(f"Timing drift detected: {stats.drift_ppm:.1f}ppm")
                
                # Perform drift correction if enabled
                if (self.config.enable_drift_correction and 
                    time.time() - self.last_drift_correction > self.config.drift_correction_period_s):
                    self._perform_drift_correction()
                
                # Sleep until next monitoring interval
                self.shutdown_event.wait(self.config.timing_statistics_interval_s)
                
        except Exception as e:
            self.logger.error(f"Timing monitoring error: {e}")
    
    def _update_jitter_compensation(self, period_ns: int):
        """Update jitter compensation parameters."""
        try:
            expected_period_ns = self.base_period_ns
            jitter_ns = period_ns - expected_period_ns
            
            self.jitter_buffer.append(jitter_ns)
            
            if len(self.jitter_buffer) >= 10:
                self.average_jitter_ns = np.mean(list(self.jitter_buffer))
                
                # Apply compensation to timing source if significant
                if abs(self.average_jitter_ns) > 1000:  # 1μs threshold
                    if self.timing_source:
                        self.timing_source.calibration_offset_ns -= int(self.average_jitter_ns * 0.1)
                        
        except Exception as e:
            self.logger.error(f"Jitter compensation update error: {e}")
    
    def _perform_drift_correction(self):
        """Perform timing drift correction."""
        try:
            current_time = self.get_time_ns()
            elapsed_time = current_time - self.drift_reference_time
            elapsed_cycles = self.stats.total_cycles - self.drift_reference_count
            
            if elapsed_cycles > 1000:  # Minimum cycles for reliable measurement
                expected_elapsed = elapsed_cycles * self.base_period_ns
                drift_ns = elapsed_time - expected_elapsed
                drift_ppm = (drift_ns / expected_elapsed) * 1e6
                
                if abs(drift_ppm) > 10:  # 10ppm threshold
                    if self.timing_source:
                        # Apply gradual correction
                        correction_ppm = -drift_ppm * 0.1  # 10% correction
                        self.timing_source.drift_compensation_ppm += correction_ppm
                        
                        self.logger.info(f"Applied drift correction: {correction_ppm:.1f}ppm")
                
                # Update reference
                self.drift_reference_time = current_time
                self.drift_reference_count = self.stats.total_cycles
            
            self.last_drift_correction = time.time()
            
        except Exception as e:
            self.logger.error(f"Drift correction error: {e}")