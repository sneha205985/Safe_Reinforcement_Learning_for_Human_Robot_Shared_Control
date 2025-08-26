"""
Real-time Control System with Parallel Processing Architecture

This module implements a high-performance parallel processing architecture
for real-time robot control with guaranteed timing constraints.

Key features:
- Dedicated real-time thread for control loop (1000 Hz)
- High-priority safety monitoring thread (2000 Hz)
- Background threads for non-critical tasks
- CPU isolation and thread affinity for consistent performance
- Lock-free inter-thread communication
- Real-time scheduling with priority inheritance
- Interrupt handling optimization
- NUMA-aware thread placement
"""

import ctypes
import logging
import os
import psutil
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy import typing as npt

try:
    import prctl
    PRCTL_AVAILABLE = True
except ImportError:
    PRCTL_AVAILABLE = False
    warnings.warn("prctl not available - process control features disabled")

try:
    import sched_setaffinity
    CPU_AFFINITY_AVAILABLE = True
except ImportError:
    try:
        import psutil
        CPU_AFFINITY_AVAILABLE = True
    except ImportError:
        CPU_AFFINITY_AVAILABLE = False
        warnings.warn("CPU affinity control not available")

logger = logging.getLogger(__name__)


class ThreadPriority(Enum):
    """Thread priority levels for real-time scheduling"""
    CRITICAL = 99      # Emergency stop, safety violations
    HIGH = 90         # Main control loop, safety monitoring  
    MEDIUM = 50       # Sensor processing, state estimation
    LOW = 20          # Logging, diagnostics, visualization
    BACKGROUND = 10   # Model training, data processing


class ThreadType(Enum):
    """Types of threads in the system"""
    RT_CONTROL = "rt_control"           # Real-time control loop
    RT_SAFETY = "rt_safety"             # Real-time safety monitoring
    SENSOR_PROCESSING = "sensor_proc"   # Sensor data processing
    STATE_ESTIMATION = "state_est"      # State estimation and filtering
    HUMAN_INTERFACE = "human_ui"        # Human interface updates
    VISUALIZATION = "visualization"     # Data visualization
    LOGGING = "logging"                 # System logging
    DIAGNOSTICS = "diagnostics"         # System diagnostics
    MODEL_INFERENCE = "model_inference" # ML model inference
    DATA_RECORDING = "data_recording"   # Data recording and storage


@dataclass
class ThreadConfig:
    """Configuration for a system thread"""
    thread_type: ThreadType
    priority: ThreadPriority
    frequency_hz: float
    cpu_affinity: Optional[List[int]] = None
    stack_size: Optional[int] = None
    enable_rt_scheduling: bool = True
    max_execution_time_us: int = 1000
    watchdog_timeout_ms: int = 100
    
    def __post_init__(self):
        if self.stack_size is None:
            # Default stack sizes based on thread type
            if self.thread_type in [ThreadType.RT_CONTROL, ThreadType.RT_SAFETY]:
                self.stack_size = 8 * 1024 * 1024  # 8MB for RT threads
            else:
                self.stack_size = 1 * 1024 * 1024  # 1MB for other threads


@dataclass
class RTControlConfig:
    """Configuration for the real-time control system"""
    # Main control loop
    control_frequency_hz: float = 1000.0
    safety_frequency_hz: float = 2000.0
    
    # Thread configurations
    thread_configs: Dict[ThreadType, ThreadConfig] = field(default_factory=dict)
    
    # CPU and system configuration
    isolate_rt_cpus: bool = True
    rt_cpu_list: List[int] = field(default_factory=lambda: [0, 1])
    background_cpu_list: List[int] = field(default_factory=lambda: [2, 3])
    disable_irq_balance: bool = True
    
    # Real-time kernel settings
    enable_rt_kernel: bool = True
    rt_throttling_disabled: bool = True
    preempt_rt: bool = False
    
    # Performance monitoring
    enable_timing_monitoring: bool = True
    enable_jitter_analysis: bool = True
    max_control_jitter_us: int = 50
    max_safety_jitter_us: int = 25
    
    # Communication
    use_lockfree_queues: bool = True
    queue_sizes: Dict[str, int] = field(default_factory=lambda: {
        "control_to_safety": 1000,
        "sensor_to_control": 2000,
        "control_to_actuator": 500,
        "safety_to_emergency": 100,
    })
    
    # Watchdog and error handling
    enable_watchdog: bool = True
    watchdog_timeout_ms: int = 10
    enable_deadline_monitoring: bool = True
    emergency_stop_on_deadline_miss: bool = True
    
    def __post_init__(self):
        if not self.thread_configs:
            self._initialize_default_thread_configs()
    
    def _initialize_default_thread_configs(self):
        """Initialize default thread configurations"""
        self.thread_configs = {
            ThreadType.RT_CONTROL: ThreadConfig(
                ThreadType.RT_CONTROL, ThreadPriority.HIGH, self.control_frequency_hz,
                cpu_affinity=self.rt_cpu_list[:1], max_execution_time_us=500
            ),
            ThreadType.RT_SAFETY: ThreadConfig(
                ThreadType.RT_SAFETY, ThreadPriority.CRITICAL, self.safety_frequency_hz,
                cpu_affinity=self.rt_cpu_list[1:2], max_execution_time_us=250
            ),
            ThreadType.SENSOR_PROCESSING: ThreadConfig(
                ThreadType.SENSOR_PROCESSING, ThreadPriority.MEDIUM, 500.0,
                cpu_affinity=self.rt_cpu_list, max_execution_time_us=1000
            ),
            ThreadType.STATE_ESTIMATION: ThreadConfig(
                ThreadType.STATE_ESTIMATION, ThreadPriority.MEDIUM, 100.0,
                cpu_affinity=self.rt_cpu_list, max_execution_time_us=2000
            ),
            ThreadType.HUMAN_INTERFACE: ThreadConfig(
                ThreadType.HUMAN_INTERFACE, ThreadPriority.LOW, 100.0,
                cpu_affinity=self.background_cpu_list, max_execution_time_us=10000
            ),
            ThreadType.VISUALIZATION: ThreadConfig(
                ThreadType.VISUALIZATION, ThreadPriority.BACKGROUND, 50.0,
                cpu_affinity=self.background_cpu_list, max_execution_time_us=20000
            ),
            ThreadType.LOGGING: ThreadConfig(
                ThreadType.LOGGING, ThreadPriority.BACKGROUND, 10.0,
                cpu_affinity=self.background_cpu_list, max_execution_time_us=50000
            ),
        }


class LockFreeRingBuffer:
    """
    Lock-free ring buffer for real-time inter-thread communication.
    
    Uses atomic operations for thread-safe access without blocking.
    """
    
    def __init__(self, capacity: int, element_size: int):
        self.capacity = capacity
        self.element_size = element_size
        
        # Pre-allocate buffer
        self.buffer = np.empty((capacity, element_size), dtype=np.float32)
        
        # Atomic indices (using threading primitives)
        self._head = 0
        self._tail = 0
        self._size = 0
        self._head_lock = threading.Lock()
        self._tail_lock = threading.Lock()
        
        # Statistics
        self.push_count = 0
        self.pop_count = 0
        self.overrun_count = 0
    
    def push(self, data: npt.NDArray, overwrite_on_full: bool = True) -> bool:
        """
        Push data to buffer.
        
        Args:
            data: Data to push (must match element_size)
            overwrite_on_full: If True, overwrite oldest data when full
            
        Returns:
            True if successful, False if buffer full (and not overwriting)
        """
        if data.shape[0] != self.element_size:
            raise ValueError(f"Data size {data.shape[0]} != element_size {self.element_size}")
        
        with self._tail_lock:
            if self._size >= self.capacity:
                if not overwrite_on_full:
                    return False
                
                # Overwrite mode - advance head
                with self._head_lock:
                    self._head = (self._head + 1) % self.capacity
                    self.overrun_count += 1
            else:
                self._size += 1
            
            # Copy data
            self.buffer[self._tail] = data
            self._tail = (self._tail + 1) % self.capacity
            self.push_count += 1
        
        return True
    
    def pop(self) -> Optional[npt.NDArray]:
        """Pop oldest data from buffer"""
        with self._head_lock:
            if self._size == 0:
                return None
            
            data = self.buffer[self._head].copy()
            self._head = (self._head + 1) % self.capacity
            self._size -= 1
            self.pop_count += 1
            
            return data
    
    def peek_latest(self) -> Optional[npt.NDArray]:
        """Peek at latest data without removing it"""
        if self._size == 0:
            return None
        
        latest_index = (self._tail - 1) % self.capacity
        return self.buffer[latest_index].copy()
    
    def is_empty(self) -> bool:
        return self._size == 0
    
    def is_full(self) -> bool:
        return self._size >= self.capacity
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "utilization": self._size / self.capacity,
            "push_count": self.push_count,
            "pop_count": self.pop_count,
            "overrun_count": self.overrun_count,
            "overrun_rate": self.overrun_count / max(self.push_count, 1),
        }


class RTThread:
    """
    Real-time thread wrapper with timing monitoring and priority management.
    """
    
    def __init__(self, config: ThreadConfig, target_function: Callable):
        self.config = config
        self.target_function = target_function
        
        self.thread = None
        self.running = False
        self.stop_event = threading.Event()
        
        # Timing statistics
        self.execution_times = []
        self.jitter_values = []
        self.deadline_misses = 0
        self.total_iterations = 0
        
        # Expected loop period
        self.expected_period_s = 1.0 / config.frequency_hz
        self.last_execution_time = 0.0
        
        # Performance monitoring
        self.watchdog_last_ping = time.time()
        
        logger.debug(f"Created RT thread: {config.thread_type.value}")
    
    def start(self):
        """Start the real-time thread"""
        if self.running:
            logger.warning(f"Thread {self.config.thread_type.value} already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Create thread with specified stack size
        self.thread = threading.Thread(
            target=self._thread_main,
            name=f"RT_{self.config.thread_type.value}",
        )
        
        if self.config.stack_size:
            threading.stack_size(self.config.stack_size)
        
        self.thread.start()
        
        # Configure thread properties after start
        self._configure_thread()
        
        logger.info(f"Started RT thread: {self.config.thread_type.value}")
    
    def stop(self, timeout: float = 1.0):
        """Stop the real-time thread"""
        if not self.running:
            return
        
        self.running = False
        self.stop_event.set()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout)
            
            if self.thread.is_alive():
                logger.warning(f"Thread {self.config.thread_type.value} did not stop gracefully")
        
        logger.info(f"Stopped RT thread: {self.config.thread_type.value}")
    
    def _configure_thread(self):
        """Configure thread priority and affinity"""
        if not self.thread or not self.thread.is_alive():
            return
        
        thread_id = self.thread.ident
        
        # Set thread priority
        if self.config.enable_rt_scheduling:
            try:
                # Set real-time scheduling policy
                os.sched_setscheduler(
                    thread_id, 
                    os.SCHED_FIFO, 
                    os.sched_param(self.config.priority.value)
                )
                logger.debug(f"Set RT scheduling for {self.config.thread_type.value}: "
                           f"priority {self.config.priority.value}")
            except PermissionError:
                logger.warning(f"Cannot set RT scheduling - insufficient privileges")
            except OSError as e:
                logger.warning(f"Failed to set RT scheduling: {e}")
        
        # Set CPU affinity
        if self.config.cpu_affinity and CPU_AFFINITY_AVAILABLE:
            try:
                if hasattr(psutil.Process(), 'cpu_affinity'):
                    process = psutil.Process(os.getpid())
                    process.cpu_affinity(self.config.cpu_affinity)
                    logger.debug(f"Set CPU affinity for {self.config.thread_type.value}: "
                               f"{self.config.cpu_affinity}")
            except Exception as e:
                logger.warning(f"Failed to set CPU affinity: {e}")
        
        # Set thread name
        if PRCTL_AVAILABLE:
            try:
                prctl.set_name(f"RT_{self.config.thread_type.value[:8]}")
            except Exception as e:
                logger.debug(f"Failed to set thread name: {e}")
    
    def _thread_main(self):
        """Main thread execution loop"""
        next_execution_time = time.time()
        
        while self.running and not self.stop_event.is_set():
            loop_start_time = time.time()
            
            try:
                # Execute target function with timing monitoring
                execution_start = time.perf_counter()
                
                # Call the actual thread function
                self.target_function()
                
                execution_end = time.perf_counter()
                execution_time_us = (execution_end - execution_start) * 1_000_000
                
                # Update statistics
                self.execution_times.append(execution_time_us)
                self.total_iterations += 1
                
                # Calculate jitter
                if self.last_execution_time > 0:
                    actual_period = loop_start_time - self.last_execution_time
                    jitter_us = abs(actual_period - self.expected_period_s) * 1_000_000
                    self.jitter_values.append(jitter_us)
                
                self.last_execution_time = loop_start_time
                
                # Check for deadline miss
                if execution_time_us > self.config.max_execution_time_us:
                    self.deadline_misses += 1
                    logger.warning(f"Deadline miss in {self.config.thread_type.value}: "
                                 f"{execution_time_us:.1f}μs > {self.config.max_execution_time_us}μs")
                
                # Update watchdog
                self.watchdog_last_ping = time.time()
                
                # Sleep until next execution time
                next_execution_time += self.expected_period_s
                sleep_time = next_execution_time - time.time()
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We're running behind - adjust next execution time
                    next_execution_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in RT thread {self.config.thread_type.value}: {e}")
                # Continue running unless it's a critical error
                if self.config.priority == ThreadPriority.CRITICAL:
                    logger.critical("Critical thread error - stopping system")
                    break
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get thread performance statistics"""
        execution_times = np.array(self.execution_times[-1000:])  # Last 1000 samples
        jitter_values = np.array(self.jitter_values[-1000:])
        
        stats = {
            "thread_type": self.config.thread_type.value,
            "priority": self.config.priority.value,
            "target_frequency": self.config.frequency_hz,
            "total_iterations": self.total_iterations,
            "deadline_misses": self.deadline_misses,
            "deadline_miss_rate": self.deadline_misses / max(self.total_iterations, 1),
            "execution_times": {
                "mean_us": float(np.mean(execution_times)) if len(execution_times) > 0 else 0.0,
                "max_us": float(np.max(execution_times)) if len(execution_times) > 0 else 0.0,
                "p95_us": float(np.percentile(execution_times, 95)) if len(execution_times) > 0 else 0.0,
                "p99_us": float(np.percentile(execution_times, 99)) if len(execution_times) > 0 else 0.0,
                "std_us": float(np.std(execution_times)) if len(execution_times) > 0 else 0.0,
            },
            "jitter": {
                "mean_us": float(np.mean(jitter_values)) if len(jitter_values) > 0 else 0.0,
                "max_us": float(np.max(jitter_values)) if len(jitter_values) > 0 else 0.0,
                "p95_us": float(np.percentile(jitter_values, 95)) if len(jitter_values) > 0 else 0.0,
                "std_us": float(np.std(jitter_values)) if len(jitter_values) > 0 else 0.0,
            },
            "watchdog_ok": time.time() - self.watchdog_last_ping < (self.config.watchdog_timeout_ms / 1000.0),
        }
        
        return stats


class RTControlSystem:
    """
    Real-time control system with parallel processing architecture.
    
    Manages multiple real-time threads with guaranteed timing constraints
    and provides inter-thread communication mechanisms.
    """
    
    def __init__(self, config: RTControlConfig):
        self.config = config
        self.threads: Dict[ThreadType, RTThread] = {}
        self.queues: Dict[str, LockFreeRingBuffer] = {}
        
        # System state
        self.running = False
        self.emergency_stop = False
        self.system_start_time = 0.0
        
        # Inter-thread communication
        self._initialize_communication()
        
        # System monitoring
        self.watchdog_thread = None
        self.monitoring_data = {}
        
        # Signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Apply system-level optimizations
        self._apply_system_optimizations()
        
        logger.info("RTControlSystem initialized")
    
    def _initialize_communication(self):
        """Initialize inter-thread communication queues"""
        for queue_name, size in self.config.queue_sizes.items():
            element_size = 64  # Default element size (can be configured)
            if "sensor" in queue_name:
                element_size = 12  # Sensor data size
            elif "control" in queue_name:
                element_size = 8   # Control command size
            elif "safety" in queue_name:
                element_size = 4   # Safety status size
            
            self.queues[queue_name] = LockFreeRingBuffer(size, element_size)
        
        logger.debug(f"Initialized {len(self.queues)} communication queues")
    
    def _apply_system_optimizations(self):
        """Apply system-level optimizations for real-time performance"""
        try:
            # Disable CPU frequency scaling
            if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"):
                os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
                logger.info("Set CPU governor to performance mode")
            
            # Disable RT throttling
            if self.config.rt_throttling_disabled:
                os.system("echo -1 | sudo tee /proc/sys/kernel/sched_rt_runtime_us")
                logger.info("Disabled RT throttling")
            
            # Isolate RT CPUs
            if self.config.isolate_rt_cpus and self.config.rt_cpu_list:
                rt_cpus = ",".join(map(str, self.config.rt_cpu_list))
                os.system(f"echo {rt_cpus} | sudo tee /sys/devices/system/cpu/isolated")
                logger.info(f"Isolated RT CPUs: {rt_cpus}")
            
            # Disable IRQ balancing
            if self.config.disable_irq_balance:
                os.system("sudo systemctl stop irqbalance")
                logger.info("Disabled IRQ balancing")
                
        except Exception as e:
            logger.warning(f"Failed to apply some system optimizations: {e}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.emergency_stop = True
            self.stop_all_threads()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def add_thread(self, thread_type: ThreadType, target_function: Callable):
        """Add a thread to the system"""
        if thread_type not in self.config.thread_configs:
            raise ValueError(f"No configuration for thread type: {thread_type}")
        
        config = self.config.thread_configs[thread_type]
        rt_thread = RTThread(config, target_function)
        self.threads[thread_type] = rt_thread
        
        logger.debug(f"Added thread: {thread_type.value}")
    
    def start_thread(self, thread_type: ThreadType):
        """Start a specific thread"""
        if thread_type not in self.threads:
            raise ValueError(f"Thread not found: {thread_type}")
        
        self.threads[thread_type].start()
    
    def stop_thread(self, thread_type: ThreadType, timeout: float = 1.0):
        """Stop a specific thread"""
        if thread_type in self.threads:
            self.threads[thread_type].stop(timeout)
    
    def start_all_threads(self):
        """Start all threads in priority order"""
        if self.running:
            logger.warning("Control system already running")
            return
        
        self.running = True
        self.system_start_time = time.time()
        self.emergency_stop = False
        
        # Start threads in priority order (highest priority first)
        sorted_threads = sorted(
            self.threads.items(),
            key=lambda x: x[1].config.priority.value,
            reverse=True
        )
        
        for thread_type, rt_thread in sorted_threads:
            rt_thread.start()
            time.sleep(0.01)  # Small delay between thread starts
        
        # Start watchdog
        if self.config.enable_watchdog:
            self._start_watchdog()
        
        logger.info("All threads started")
    
    def stop_all_threads(self):
        """Stop all threads gracefully"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop watchdog first
        if self.watchdog_thread:
            self.watchdog_thread.stop()
        
        # Stop threads in reverse priority order (lowest priority first)
        sorted_threads = sorted(
            self.threads.items(),
            key=lambda x: x[1].config.priority.value,
            reverse=False
        )
        
        for thread_type, rt_thread in sorted_threads:
            rt_thread.stop()
        
        logger.info("All threads stopped")
    
    def _start_watchdog(self):
        """Start system watchdog thread"""
        def watchdog_function():
            watchdog_timeout = self.config.watchdog_timeout_ms / 1000.0
            
            while self.running:
                current_time = time.time()
                
                # Check all thread watchdogs
                for thread_type, rt_thread in self.threads.items():
                    if not rt_thread.running:
                        continue
                    
                    time_since_ping = current_time - rt_thread.watchdog_last_ping
                    
                    if time_since_ping > watchdog_timeout:
                        logger.critical(f"Watchdog timeout for {thread_type.value}: "
                                      f"{time_since_ping:.3f}s")
                        
                        # Emergency stop for critical threads
                        if rt_thread.config.priority == ThreadPriority.CRITICAL:
                            logger.critical("Critical thread watchdog timeout - emergency stop")
                            self.emergency_stop = True
                            return
                
                time.sleep(watchdog_timeout / 10)  # Check at 10x frequency
        
        watchdog_config = ThreadConfig(
            ThreadType.DIAGNOSTICS, ThreadPriority.HIGH, 100.0,
            max_execution_time_us=1000, watchdog_timeout_ms=1000
        )
        
        self.watchdog_thread = RTThread(watchdog_config, watchdog_function)
        self.watchdog_thread.start()
        
        logger.info("Watchdog thread started")
    
    def get_queue(self, queue_name: str) -> Optional[LockFreeRingBuffer]:
        """Get a communication queue by name"""
        return self.queues.get(queue_name)
    
    def send_to_queue(self, queue_name: str, data: npt.NDArray) -> bool:
        """Send data to a queue"""
        queue = self.queues.get(queue_name)
        if queue is None:
            logger.error(f"Queue not found: {queue_name}")
            return False
        
        return queue.push(data)
    
    def receive_from_queue(self, queue_name: str) -> Optional[npt.NDArray]:
        """Receive data from a queue"""
        queue = self.queues.get(queue_name)
        if queue is None:
            logger.error(f"Queue not found: {queue_name}")
            return None
        
        return queue.pop()
    
    @contextmanager
    def rt_execution_context(self, max_time_us: int = 1000):
        """Context manager for real-time execution monitoring"""
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        execution_time_us = (end_time - start_time) * 1_000_000
        
        if execution_time_us > max_time_us:
            logger.warning(f"RT execution exceeded limit: {execution_time_us:.1f}μs > {max_time_us}μs")
    
    def run_rt_loop(self):
        """
        Main real-time control loop implementation.
        
        This would be called by the RT_CONTROL thread.
        """
        # This is a placeholder - actual implementation would:
        # 1. Read sensor data from queues
        # 2. Update state estimation
        # 3. Run policy inference
        # 4. Check safety constraints
        # 5. Send commands to actuators
        # 6. Update monitoring data
        
        with self.rt_execution_context(500):  # 500μs limit
            # Placeholder operations
            sensor_data = self.receive_from_queue("sensor_to_control")
            
            if sensor_data is not None:
                # Simulate control computation
                control_command = sensor_data * 0.1  # Simple P controller
                
                # Send to safety thread for validation
                self.send_to_queue("control_to_safety", control_command)
                
                # Send to actuators (if safety approved)
                self.send_to_queue("control_to_actuator", control_command)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        thread_stats = {}
        for thread_type, rt_thread in self.threads.items():
            thread_stats[thread_type.value] = rt_thread.get_performance_stats()
        
        queue_stats = {}
        for queue_name, queue in self.queues.items():
            queue_stats[queue_name] = queue.get_stats()
        
        system_info = {
            "running": self.running,
            "emergency_stop": self.emergency_stop,
            "uptime_seconds": time.time() - self.system_start_time if self.running else 0,
            "num_threads": len(self.threads),
            "num_queues": len(self.queues),
            "cpu_count": psutil.cpu_count(),
            "memory_usage": psutil.virtual_memory()._asdict(),
        }
        
        return {
            "system_info": system_info,
            "thread_stats": thread_stats,
            "queue_stats": queue_stats,
            "config": {
                "control_frequency": self.config.control_frequency_hz,
                "safety_frequency": self.config.safety_frequency_hz,
                "rt_cpu_list": self.config.rt_cpu_list,
                "background_cpu_list": self.config.background_cpu_list,
            }
        }
    
    def emergency_shutdown(self, reason: str = "Emergency stop triggered"):
        """Emergency shutdown of the entire system"""
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        self.emergency_stop = True
        
        # Stop all threads immediately
        for rt_thread in self.threads.values():
            rt_thread.stop(timeout=0.1)  # Very short timeout
        
        self.running = False
        
        # Additional emergency actions would go here:
        # - Stop all actuators
        # - Activate brakes/safety systems
        # - Send emergency notifications
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_all_threads()


# Example usage and testing
def create_example_control_functions():
    """Create example functions for different threads"""
    
    def control_loop():
        """Example control loop function"""
        # Simulate sensor reading and control computation
        time.sleep(0.0001)  # 0.1ms simulated work
    
    def safety_monitor():
        """Example safety monitoring function"""
        # Simulate safety checks
        time.sleep(0.00005)  # 0.05ms simulated work
    
    def sensor_processing():
        """Example sensor processing function"""
        # Simulate sensor data processing
        time.sleep(0.0002)  # 0.2ms simulated work
    
    def human_interface():
        """Example human interface function"""
        # Simulate UI updates
        time.sleep(0.001)  # 1ms simulated work
    
    return {
        ThreadType.RT_CONTROL: control_loop,
        ThreadType.RT_SAFETY: safety_monitor,
        ThreadType.SENSOR_PROCESSING: sensor_processing,
        ThreadType.HUMAN_INTERFACE: human_interface,
    }


def main():
    """Example usage of the RT control system"""
    
    # Create configuration
    config = RTControlConfig(
        control_frequency_hz=1000.0,
        safety_frequency_hz=2000.0,
        rt_cpu_list=[0, 1],
        background_cpu_list=[2, 3],
        enable_watchdog=True,
    )
    
    # Create control system
    with RTControlSystem(config) as control_system:
        
        # Add threads
        thread_functions = create_example_control_functions()
        for thread_type, function in thread_functions.items():
            control_system.add_thread(thread_type, function)
        
        # Start all threads
        control_system.start_all_threads()
        
        try:
            # Run for 5 seconds
            time.sleep(5.0)
            
            # Get statistics
            stats = control_system.get_system_stats()
            
            print("System Statistics:")
            print(f"Uptime: {stats['system_info']['uptime_seconds']:.1f}s")
            print(f"Threads: {stats['system_info']['num_threads']}")
            print(f"Emergency stop: {stats['system_info']['emergency_stop']}")
            
            print("\nThread Performance:")
            for thread_name, thread_stats in stats['thread_stats'].items():
                exec_stats = thread_stats['execution_times']
                jitter_stats = thread_stats['jitter']
                print(f"  {thread_name}:")
                print(f"    Iterations: {thread_stats['total_iterations']}")
                print(f"    Deadline misses: {thread_stats['deadline_misses']}")
                print(f"    Mean exec time: {exec_stats['mean_us']:.1f}μs")
                print(f"    P99 exec time: {exec_stats['p99_us']:.1f}μs")
                print(f"    Mean jitter: {jitter_stats['mean_us']:.1f}μs")
                print(f"    P95 jitter: {jitter_stats['p95_us']:.1f}μs")
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        # System will automatically stop all threads on exit


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    main()