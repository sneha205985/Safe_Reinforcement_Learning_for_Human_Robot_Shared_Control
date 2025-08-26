"""
Reproducibility and Profiling Tools for Safe RL Benchmarking.

This module provides comprehensive tools for ensuring reproducible experiments
and detailed performance profiling of Safe RL algorithms.
"""

import numpy as np
import torch
import random
import os
import sys
import time
import psutil
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from contextlib import contextmanager
import pickle
import subprocess
import platform
from datetime import datetime
import threading
import functools
import traceback
import gc

logger = logging.getLogger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducible experiments."""
    # Random seeds
    global_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42
    python_seed: int = 42
    
    # Deterministic settings
    torch_deterministic: bool = True
    torch_benchmark: bool = False
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    
    # Environment settings
    env_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_directory: str = "logs"
    
    # System settings
    num_threads: Optional[int] = None
    gpu_deterministic: bool = True
    
    # Experiment metadata
    experiment_name: str = "safe_rl_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    platform: str
    platform_version: str
    architecture: str
    processor: str
    python_version: str
    numpy_version: str
    torch_version: str
    cuda_version: Optional[str]
    gpu_info: List[Dict[str, Any]]
    cpu_count: int
    memory_total: int
    timestamp: str
    hostname: str
    username: str
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None


@dataclass
class PerformanceMetrics:
    """Performance profiling metrics."""
    # Time metrics
    total_time: float
    cpu_time: float
    wall_time: float
    
    # Memory metrics
    peak_memory_mb: float
    average_memory_mb: float
    memory_efficiency: float
    
    # GPU metrics (if applicable)
    gpu_memory_peak_mb: Optional[float] = None
    gpu_memory_average_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    
    # Algorithm-specific metrics
    forward_pass_time: Optional[float] = None
    backward_pass_time: Optional[float] = None
    inference_time: Optional[float] = None
    
    # System metrics
    cpu_utilization: float = 0.0
    io_operations: int = 0
    network_io: Optional[Dict[str, int]] = None


class ReproducibilityManager:
    """Manages reproducibility settings for experiments."""
    
    def __init__(self, config: ReproducibilityConfig):
        self.config = config
        self.system_info: Optional[SystemInfo] = None
        self.experiment_hash: Optional[str] = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize reproducibility settings
        self._setup_reproducibility()
        
        # Collect system information
        self.system_info = self._collect_system_info()
        
        logger.info("ReproducibilityManager initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create log directory
        if self.config.log_to_file:
            log_dir = Path(self.config.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{self.config.experiment_name}_{timestamp}.log"
            
            # Configure file handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            
            # Configure formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Add handler to root logger
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
            root_logger.setLevel(log_level)
            
            logger.info(f"Logging configured: {log_file}")
    
    def _setup_reproducibility(self):
        """Setup all reproducibility settings."""
        logger.info("Setting up reproducibility environment")
        
        # Python random seed
        random.seed(self.config.python_seed)
        
        # NumPy random seed
        np.random.seed(self.config.numpy_seed)
        
        # PyTorch random seed
        torch.manual_seed(self.config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.torch_seed)
            torch.cuda.manual_seed_all(self.config.torch_seed)
        
        # PyTorch deterministic settings
        if self.config.torch_deterministic:
            torch.backends.cudnn.deterministic = self.config.cudnn_deterministic
            torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
            
            # Set deterministic algorithms (PyTorch 1.8+)
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Thread settings
        if self.config.num_threads:
            torch.set_num_threads(self.config.num_threads)
            os.environ['OMP_NUM_THREADS'] = str(self.config.num_threads)
            os.environ['MKL_NUM_THREADS'] = str(self.config.num_threads)
        
        # Environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = str(self.config.python_seed)
        
        if self.config.gpu_deterministic:
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        logger.info("Reproducibility settings applied")
    
    def _collect_system_info(self) -> SystemInfo:
        """Collect comprehensive system information."""
        logger.info("Collecting system information")
        
        # Basic system info
        system_info = SystemInfo(
            platform=platform.system(),
            platform_version=platform.version(),
            architecture=platform.architecture()[0],
            processor=platform.processor(),
            python_version=sys.version,
            numpy_version=np.__version__,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
            gpu_info=self._get_gpu_info(),
            cpu_count=os.cpu_count(),
            memory_total=psutil.virtual_memory().total,
            timestamp=datetime.now().isoformat(),
            hostname=platform.node(),
            username=os.getenv('USER', 'unknown'),
        )
        
        # Git information (if available)
        git_info = self._get_git_info()
        system_info.git_commit = git_info.get('commit')
        system_info.git_branch = git_info.get('branch')
        system_info.git_dirty = git_info.get('dirty')
        
        return system_info
    
    def _get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information."""
        gpu_info = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'device_id': i,
                    'name': gpu_props.name,
                    'memory_total': gpu_props.total_memory,
                    'multiprocessor_count': gpu_props.multi_processor_count,
                    'compute_capability': (gpu_props.major, gpu_props.minor)
                })
        
        return gpu_info
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information."""
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['commit'] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
            
            # Check if repository is dirty
            result = subprocess.run(['git', 'diff', '--quiet'], 
                                  capture_output=True, timeout=5)
            git_info['dirty'] = result.returncode != 0
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("Could not retrieve Git information")
        
        return git_info
    
    def create_experiment_hash(self, config_dict: Dict[str, Any]) -> str:
        """Create unique hash for experiment configuration."""
        # Combine system info and config
        hash_data = {
            'config': config_dict,
            'system': asdict(self.system_info),
            'reproducibility': asdict(self.config)
        }
        
        # Create hash
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        experiment_hash = hashlib.sha256(hash_string.encode()).hexdigest()[:16]
        
        self.experiment_hash = experiment_hash
        logger.info(f"Experiment hash: {experiment_hash}")
        
        return experiment_hash
    
    def save_experiment_metadata(self, save_path: Path, additional_data: Dict[str, Any] = None):
        """Save complete experiment metadata."""
        metadata = {
            'experiment_name': self.config.experiment_name,
            'description': self.config.description,
            'tags': self.config.tags,
            'experiment_hash': self.experiment_hash,
            'reproducibility_config': asdict(self.config),
            'system_info': asdict(self.system_info),
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            metadata.update(additional_data)
        
        metadata_path = save_path / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Experiment metadata saved: {metadata_path}")
    
    def verify_reproducibility(self, reference_results: Dict[str, Any], 
                             current_results: Dict[str, Any],
                             tolerance: float = 1e-6) -> Dict[str, Any]:
        """Verify reproducibility by comparing results."""
        verification_results = {
            'reproducible': True,
            'differences': {},
            'tolerance': tolerance,
            'timestamp': datetime.now().isoformat()
        }
        
        def compare_values(ref_val, curr_val, key_path):
            if isinstance(ref_val, (int, float)) and isinstance(curr_val, (int, float)):
                diff = abs(ref_val - curr_val)
                if diff > tolerance:
                    verification_results['reproducible'] = False
                    verification_results['differences'][key_path] = {
                        'reference': ref_val,
                        'current': curr_val,
                        'difference': diff
                    }
            elif isinstance(ref_val, dict) and isinstance(curr_val, dict):
                for k in set(ref_val.keys()) | set(curr_val.keys()):
                    if k in ref_val and k in curr_val:
                        compare_values(ref_val[k], curr_val[k], f"{key_path}.{k}")
                    else:
                        verification_results['reproducible'] = False
                        verification_results['differences'][f"{key_path}.{k}"] = {
                            'reference': ref_val.get(k, '<missing>'),
                            'current': curr_val.get(k, '<missing>'),
                            'difference': 'key_mismatch'
                        }
            elif ref_val != curr_val:
                verification_results['reproducible'] = False
                verification_results['differences'][key_path] = {
                    'reference': ref_val,
                    'current': curr_val,
                    'difference': 'value_mismatch'
                }
        
        # Compare all results
        for key in set(reference_results.keys()) | set(current_results.keys()):
            if key in reference_results and key in current_results:
                compare_values(reference_results[key], current_results[key], key)
            else:
                verification_results['reproducible'] = False
                verification_results['differences'][key] = {
                    'reference': reference_results.get(key, '<missing>'),
                    'current': current_results.get(key, '<missing>'),
                    'difference': 'key_mismatch'
                }
        
        logger.info(f"Reproducibility verification: {'PASSED' if verification_results['reproducible'] else 'FAILED'}")
        
        return verification_results


class PerformanceProfiler:
    """Comprehensive performance profiling for Safe RL algorithms."""
    
    def __init__(self):
        self.profiling_data = {}
        self.memory_history = []
        self.gpu_memory_history = []
        self.cpu_history = []
        self.start_time = None
        self.monitoring_thread = None
        self.monitoring_active = False
        
        logger.info("PerformanceProfiler initialized")
    
    @contextmanager
    def profile_algorithm(self, algorithm_name: str, detailed: bool = True):
        """Context manager for profiling algorithm performance."""
        logger.info(f"Starting performance profiling for {algorithm_name}")
        
        # Initialize profiling
        self.start_time = time.time()
        profile_data = {
            'algorithm': algorithm_name,
            'start_time': self.start_time,
            'detailed': detailed
        }
        
        # Start monitoring
        if detailed:
            self._start_monitoring()
        
        # Get initial measurements
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        initial_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
        
        profile_data.update({
            'initial_memory_mb': initial_memory,
            'initial_gpu_memory_mb': initial_gpu_memory
        })
        
        try:
            yield profile_data
        finally:
            # Stop monitoring
            if detailed:
                self._stop_monitoring()
            
            # Final measurements
            end_time = time.time()
            final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
            final_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(
                profile_data, end_time, final_memory, final_gpu_memory
            )
            
            # Store results
            self.profiling_data[algorithm_name] = {
                'profile_data': profile_data,
                'metrics': metrics
            }
            
            logger.info(f"Performance profiling completed for {algorithm_name}")
            logger.info(f"Total time: {metrics.total_time:.2f}s, Peak memory: {metrics.peak_memory_mb:.1f}MB")
    
    def _start_monitoring(self):
        """Start background monitoring of system resources."""
        self.monitoring_active = True
        self.memory_history = []
        self.gpu_memory_history = []
        self.cpu_history = []
        
        def monitor():
            while self.monitoring_active:
                # Memory monitoring
                memory_info = psutil.virtual_memory()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'used_mb': memory_info.used / 1024 / 1024,
                    'percent': memory_info.percent
                })
                
                # GPU monitoring
                if torch.cuda.is_available():
                    gpu_memory = self._get_gpu_memory()
                    gpu_util = self._get_gpu_utilization()
                    self.gpu_memory_history.append({
                        'timestamp': time.time(),
                        'used_mb': gpu_memory,
                        'utilization': gpu_util
                    })
                
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_history.append({
                    'timestamp': time.time(),
                    'percent': cpu_percent
                })
                
                time.sleep(0.1)  # Sample every 100ms
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def _stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available():
            return 0.0
        
        total_memory = 0
        for i in range(torch.cuda.device_count()):
            memory_used = torch.cuda.memory_allocated(i) / 1024 / 1024  # MB
            total_memory += memory_used
        
        return total_memory
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except:
            return 0.0  # Return 0 if pynvml is not available
    
    def _calculate_performance_metrics(self, 
                                     profile_data: Dict[str, Any],
                                     end_time: float,
                                     final_memory: float,
                                     final_gpu_memory: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        total_time = end_time - profile_data['start_time']
        
        # Memory metrics
        if self.memory_history:
            memory_values = [entry['used_mb'] for entry in self.memory_history]
            peak_memory = max(memory_values)
            average_memory = np.mean(memory_values)
            memory_efficiency = (final_memory - profile_data['initial_memory_mb']) / total_time if total_time > 0 else 0
        else:
            peak_memory = max(profile_data['initial_memory_mb'], final_memory)
            average_memory = (profile_data['initial_memory_mb'] + final_memory) / 2
            memory_efficiency = 0
        
        # GPU metrics
        gpu_memory_peak = None
        gpu_memory_average = None
        gpu_utilization = None
        
        if torch.cuda.is_available() and self.gpu_memory_history:
            gpu_memory_values = [entry['used_mb'] for entry in self.gpu_memory_history]
            gpu_util_values = [entry['utilization'] for entry in self.gpu_memory_history]
            
            gpu_memory_peak = max(gpu_memory_values)
            gpu_memory_average = np.mean(gpu_memory_values)
            gpu_utilization = np.mean(gpu_util_values)
        
        # CPU utilization
        cpu_utilization = 0.0
        if self.cpu_history:
            cpu_values = [entry['percent'] for entry in self.cpu_history]
            cpu_utilization = np.mean(cpu_values)
        
        return PerformanceMetrics(
            total_time=total_time,
            cpu_time=total_time,  # Simplified - could use more accurate CPU time
            wall_time=total_time,
            peak_memory_mb=peak_memory,
            average_memory_mb=average_memory,
            memory_efficiency=memory_efficiency,
            gpu_memory_peak_mb=gpu_memory_peak,
            gpu_memory_average_mb=gpu_memory_average,
            gpu_utilization=gpu_utilization,
            cpu_utilization=cpu_utilization
        )
    
    def benchmark_inference_speed(self, 
                                algorithm: Any,
                                test_inputs: List[np.ndarray],
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
        """Benchmark inference speed of algorithm."""
        logger.info(f"Benchmarking inference speed with {num_runs} runs")
        
        # Warmup runs
        for i in range(warmup_runs):
            for test_input in test_inputs:
                algorithm.predict(test_input, deterministic=True)
        
        # Benchmark runs
        inference_times = []
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            
            for test_input in test_inputs:
                algorithm.predict(test_input, deterministic=True)
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) / len(test_inputs))
        
        # Calculate statistics
        inference_stats = {
            'mean_time_ms': np.mean(inference_times) * 1000,
            'std_time_ms': np.std(inference_times) * 1000,
            'min_time_ms': np.min(inference_times) * 1000,
            'max_time_ms': np.max(inference_times) * 1000,
            'median_time_ms': np.median(inference_times) * 1000,
            'throughput_hz': 1.0 / np.mean(inference_times),
            'num_runs': num_runs
        }
        
        logger.info(f"Inference benchmark - Mean: {inference_stats['mean_time_ms']:.2f}ms, "
                   f"Throughput: {inference_stats['throughput_hz']:.1f}Hz")
        
        return inference_stats
    
    def profile_memory_usage(self, 
                           algorithm: Any,
                           training_function: Callable,
                           duration_minutes: int = 5) -> Dict[str, Any]:
        """Profile memory usage during training."""
        logger.info(f"Profiling memory usage for {duration_minutes} minutes")
        
        memory_profile = {
            'timestamps': [],
            'memory_usage_mb': [],
            'gpu_memory_mb': [],
            'cpu_percent': []
        }
        
        start_time = time.time()
        end_time = start_time + duration_minutes * 60
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=training_function, 
            args=(algorithm,), 
            daemon=True
        )
        training_thread.start()
        
        # Monitor memory usage
        try:
            while time.time() < end_time and training_thread.is_alive():
                current_time = time.time()
                
                # System memory
                memory_info = psutil.virtual_memory()
                memory_profile['timestamps'].append(current_time - start_time)
                memory_profile['memory_usage_mb'].append(memory_info.used / 1024 / 1024)
                memory_profile['cpu_percent'].append(psutil.cpu_percent())
                
                # GPU memory
                if torch.cuda.is_available():
                    gpu_memory = self._get_gpu_memory()
                    memory_profile['gpu_memory_mb'].append(gpu_memory)
                else:
                    memory_profile['gpu_memory_mb'].append(0.0)
                
                time.sleep(1)  # Sample every second
        
        except KeyboardInterrupt:
            logger.info("Memory profiling interrupted by user")
        
        # Calculate summary statistics
        memory_stats = {
            'peak_memory_mb': max(memory_profile['memory_usage_mb']),
            'average_memory_mb': np.mean(memory_profile['memory_usage_mb']),
            'memory_std_mb': np.std(memory_profile['memory_usage_mb']),
            'peak_gpu_memory_mb': max(memory_profile['gpu_memory_mb']) if memory_profile['gpu_memory_mb'] else 0,
            'average_cpu_percent': np.mean(memory_profile['cpu_percent']),
            'duration_seconds': memory_profile['timestamps'][-1] if memory_profile['timestamps'] else 0
        }
        
        logger.info(f"Memory profiling completed - Peak: {memory_stats['peak_memory_mb']:.1f}MB, "
                   f"Average: {memory_stats['average_memory_mb']:.1f}MB")
        
        return {
            'profile_data': memory_profile,
            'statistics': memory_stats
        }
    
    def generate_performance_report(self, save_path: Path) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        logger.info("Generating performance report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'algorithms_profiled': list(self.profiling_data.keys()),
            'system_info': {
                'cpu_count': os.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'gpu_available': torch.cuda.is_available(),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            'performance_summary': {}
        }
        
        # Summarize performance for each algorithm
        for alg_name, data in self.profiling_data.items():
            metrics = data['metrics']
            
            report['performance_summary'][alg_name] = {
                'total_time_s': metrics.total_time,
                'peak_memory_mb': metrics.peak_memory_mb,
                'average_memory_mb': metrics.average_memory_mb,
                'memory_efficiency': metrics.memory_efficiency,
                'cpu_utilization_percent': metrics.cpu_utilization,
                'gpu_memory_peak_mb': metrics.gpu_memory_peak_mb,
                'gpu_utilization_percent': metrics.gpu_utilization
            }
        
        # Save report
        report_path = save_path / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved: {report_path}")
        
        return report
    
    def clear_profiling_data(self):
        """Clear all stored profiling data."""
        self.profiling_data.clear()
        self.memory_history.clear()
        self.gpu_memory_history.clear()
        self.cpu_history.clear()
        logger.info("Profiling data cleared")


def reproducible_experiment(config: ReproducibilityConfig):
    """Decorator for ensuring reproducible experiments."""
    def decorator(experiment_func):
        @functools.wraps(experiment_func)
        def wrapper(*args, **kwargs):
            # Setup reproducibility
            repro_manager = ReproducibilityManager(config)
            
            try:
                # Run experiment
                logger.info(f"Starting reproducible experiment: {experiment_func.__name__}")
                result = experiment_func(*args, **kwargs)
                
                # Add metadata to result
                if isinstance(result, dict):
                    result['_reproducibility'] = {
                        'experiment_hash': repro_manager.experiment_hash,
                        'system_info': asdict(repro_manager.system_info),
                        'config': asdict(repro_manager.config)
                    }
                
                return result
                
            except Exception as e:
                logger.error(f"Experiment failed: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
        return wrapper
    return decorator


@contextmanager
def memory_limit(limit_mb: int):
    """Context manager to limit memory usage."""
    try:
        import resource
        
        # Get current limit
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        
        # Set new limit
        new_limit = limit_mb * 1024 * 1024  # Convert to bytes
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        
        logger.info(f"Memory limit set to {limit_mb}MB")
        yield
        
    except ImportError:
        logger.warning("Memory limiting not available on this platform")
        yield
    finally:
        try:
            # Restore original limit
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
            logger.info("Memory limit restored")
        except:
            pass


class ExperimentTracker:
    """Track and manage multiple experiments."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = {}
        self.current_experiment = None
        
        logger.info(f"ExperimentTracker initialized: {self.base_dir}")
    
    def start_experiment(self, 
                        name: str,
                        description: str = "",
                        tags: List[str] = None) -> str:
        """Start tracking a new experiment."""
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"
        experiment_dir = self.base_dir / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment data
        experiment_data = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'directory': experiment_dir,
            'start_time': datetime.now(),
            'status': 'running',
            'results': {},
            'logs': [],
            'artifacts': []
        }
        
        self.experiments[experiment_id] = experiment_data
        self.current_experiment = experiment_id
        
        logger.info(f"Started experiment: {experiment_id}")
        
        return experiment_id
    
    def log_result(self, key: str, value: Any):
        """Log a result for the current experiment."""
        if self.current_experiment:
            self.experiments[self.current_experiment]['results'][key] = value
            logger.info(f"Logged result {key}: {value}")
    
    def log_artifact(self, filepath: Path, description: str = ""):
        """Log an artifact for the current experiment."""
        if self.current_experiment:
            artifact = {
                'path': str(filepath),
                'description': description,
                'timestamp': datetime.now(),
                'size_bytes': filepath.stat().st_size if filepath.exists() else 0
            }
            self.experiments[self.current_experiment]['artifacts'].append(artifact)
            logger.info(f"Logged artifact: {filepath}")
    
    def finish_experiment(self, status: str = 'completed'):
        """Finish the current experiment."""
        if self.current_experiment:
            experiment = self.experiments[self.current_experiment]
            experiment['status'] = status
            experiment['end_time'] = datetime.now()
            experiment['duration'] = experiment['end_time'] - experiment['start_time']
            
            # Save experiment metadata
            metadata_path = experiment['directory'] / 'experiment_info.json'
            with open(metadata_path, 'w') as f:
                json.dump({
                    k: v for k, v in experiment.items() 
                    if k not in ['directory']  # Skip non-serializable items
                }, f, indent=2, default=str)
            
            logger.info(f"Finished experiment: {self.current_experiment}")
            self.current_experiment = None
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        return list(self.experiments.values())
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)
    
    def compare_experiments(self, 
                          experiment_ids: List[str],
                          metrics: List[str]) -> pd.DataFrame:
        """Compare experiments across specified metrics."""
        import pandas as pd
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            if exp_id in self.experiments:
                exp = self.experiments[exp_id]
                row = {'experiment_id': exp_id, 'name': exp['name']}
                
                for metric in metrics:
                    row[metric] = exp['results'].get(metric, None)
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)