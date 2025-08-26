"""
Comprehensive Real-time Performance Benchmarking System

This module provides extensive benchmarking capabilities for validating
real-time performance requirements in safe RL systems.

Key features:
- Deterministic timing validation with statistical analysis
- Stress testing under various load conditions
- Memory fragmentation impact analysis
- Cache miss scenario testing
- Multi-threaded performance validation
- Hardware-specific benchmark suites
- Performance regression detection
- Automated reporting with visualizations
"""

import asyncio
import gc
import logging
import math
import multiprocessing
import os
import platform
import psutil
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy import typing as npt

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available - plotting disabled")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available - advanced analysis disabled")

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks"""
    TIMING_PRECISION = "timing_precision"
    JITTER_ANALYSIS = "jitter_analysis"
    WORST_CASE_TIMING = "worst_case_timing"
    MEMORY_ALLOCATION = "memory_allocation"
    CACHE_PERFORMANCE = "cache_performance"
    CONTEXT_SWITCHING = "context_switching"
    INTERRUPT_LATENCY = "interrupt_latency"
    THROUGHPUT_SCALING = "throughput_scaling"
    STRESS_TEST = "stress_test"
    DETERMINISM_TEST = "determinism_test"


class LoadCondition(Enum):
    """System load conditions for testing"""
    IDLE = "idle"                  # Minimal system load
    LOW = "low"                    # 25% CPU utilization
    MEDIUM = "medium"              # 50% CPU utilization
    HIGH = "high"                  # 75% CPU utilization
    SATURATED = "saturated"        # 95% CPU utilization
    MEMORY_PRESSURE = "mem_pressure"  # High memory usage
    IO_INTENSIVE = "io_intensive"  # Heavy I/O operations


@dataclass
class TimingRequirements:
    """Real-time timing requirements specification"""
    max_execution_time_us: int = 1000      # Maximum execution time
    max_jitter_us: int = 50                # Maximum timing jitter
    deadline_miss_tolerance: float = 0.001  # Max 0.1% deadline misses
    percentile_99_us: int = 1200           # 99th percentile requirement
    percentile_999_us: int = 1500          # 99.9th percentile requirement
    min_frequency_hz: float = 1000.0       # Minimum sustainable frequency
    
    # Test parameters
    test_duration_seconds: int = 60
    warmup_duration_seconds: int = 10
    sample_count: int = 100000


@dataclass
class BenchmarkResult:
    """Individual benchmark test result"""
    benchmark_type: BenchmarkType
    test_name: str
    success: bool
    execution_times_us: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.execution_times_us:
            self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate timing statistics"""
        times = np.array(self.execution_times_us)
        
        self.metadata.update({
            'mean_us': float(np.mean(times)),
            'median_us': float(np.median(times)),
            'std_us': float(np.std(times)),
            'min_us': float(np.min(times)),
            'max_us': float(np.max(times)),
            'p95_us': float(np.percentile(times, 95)),
            'p99_us': float(np.percentile(times, 99)),
            'p999_us': float(np.percentile(times, 99.9)),
            'range_us': float(np.max(times) - np.min(times)),
            'sample_count': len(times),
        })


@dataclass
class SystemLoad:
    """System load generator configuration"""
    cpu_load_percent: float = 0.0
    memory_load_mb: int = 0
    io_operations_per_sec: int = 0
    network_bandwidth_mbps: float = 0.0
    duration_seconds: int = 60
    

class LoadGenerator:
    """Generates various system loads for stress testing"""
    
    def __init__(self):
        self.load_processes = []
        self.load_threads = []
        self.running = False
        
    def start_cpu_load(self, target_percent: float, duration_seconds: int = 0):
        """Generate CPU load"""
        def cpu_load_worker(target_load: float):
            start_time = time.time()
            end_time = start_time + duration_seconds if duration_seconds > 0 else float('inf')
            
            while time.time() < end_time and self.running:
                # Busy wait to consume CPU
                work_time = target_load / 100.0 * 0.01  # 10ms cycle
                sleep_time = 0.01 - work_time
                
                if work_time > 0:
                    work_end = time.time() + work_time
                    while time.time() < work_end:
                        pass  # Busy wait
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        # Start one worker per CPU core
        num_cores = psutil.cpu_count()
        for _ in range(num_cores):
            process = multiprocessing.Process(
                target=cpu_load_worker, 
                args=(target_percent,)
            )
            process.start()
            self.load_processes.append(process)
    
    def start_memory_load(self, target_mb: int):
        """Generate memory pressure"""
        def memory_load_worker():
            # Allocate memory in chunks
            chunk_size = 1024 * 1024  # 1MB chunks
            chunks = []
            
            try:
                for _ in range(target_mb):
                    if not self.running:
                        break
                    chunk = bytearray(chunk_size)
                    # Write to memory to ensure it's actually allocated
                    for i in range(0, chunk_size, 4096):
                        chunk[i] = 0xFF
                    chunks.append(chunk)
                    time.sleep(0.01)  # Small delay to avoid overwhelming system
                
                # Keep memory allocated while running
                while self.running:
                    time.sleep(1)
                    
            except MemoryError:
                logger.warning("Memory allocation failed - system limit reached")
        
        thread = threading.Thread(target=memory_load_worker)
        thread.start()
        self.load_threads.append(thread)
    
    def start_io_load(self, operations_per_sec: int):
        """Generate I/O load"""
        def io_load_worker():
            temp_file = "/tmp/benchmark_io_test"
            data = b"x" * 4096  # 4KB blocks
            
            while self.running:
                try:
                    # Write operations
                    with open(temp_file, "wb") as f:
                        f.write(data)
                        f.flush()
                        os.fsync(f.fileno())
                    
                    # Read operations
                    with open(temp_file, "rb") as f:
                        f.read()
                    
                    # Control rate
                    time.sleep(1.0 / operations_per_sec)
                    
                except Exception as e:
                    logger.debug(f"I/O load error: {e}")
            
            # Cleanup
            try:
                os.remove(temp_file)
            except:
                pass
        
        thread = threading.Thread(target=io_load_worker)
        thread.start()
        self.load_threads.append(thread)
    
    def start_load(self, load_config: SystemLoad):
        """Start combined system load"""
        self.running = True
        
        if load_config.cpu_load_percent > 0:
            self.start_cpu_load(load_config.cpu_load_percent, load_config.duration_seconds)
        
        if load_config.memory_load_mb > 0:
            self.start_memory_load(load_config.memory_load_mb)
        
        if load_config.io_operations_per_sec > 0:
            self.start_io_load(load_config.io_operations_per_sec)
    
    def stop_load(self):
        """Stop all load generation"""
        self.running = False
        
        # Terminate processes
        for process in self.load_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
        
        # Wait for threads
        for thread in self.load_threads:
            thread.join(timeout=5)
        
        self.load_processes.clear()
        self.load_threads.clear()


class RTPerformanceBenchmark:
    """
    Comprehensive real-time performance benchmark system.
    
    Validates timing requirements under various conditions and provides
    detailed analysis of system behavior.
    """
    
    def __init__(self, requirements: TimingRequirements):
        self.requirements = requirements
        self.results = []
        self.load_generator = LoadGenerator()
        
        # System information
        self.system_info = self._collect_system_info()
        
        # Benchmark functions
        self.benchmark_functions = self._initialize_benchmark_functions()
        
        logger.info("RT Performance Benchmark system initialized")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for context"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat(),
        }
    
    def _initialize_benchmark_functions(self) -> Dict[BenchmarkType, Callable]:
        """Initialize benchmark test functions"""
        return {
            BenchmarkType.TIMING_PRECISION: self._benchmark_timing_precision,
            BenchmarkType.JITTER_ANALYSIS: self._benchmark_jitter_analysis,
            BenchmarkType.WORST_CASE_TIMING: self._benchmark_worst_case_timing,
            BenchmarkType.MEMORY_ALLOCATION: self._benchmark_memory_allocation,
            BenchmarkType.CACHE_PERFORMANCE: self._benchmark_cache_performance,
            BenchmarkType.CONTEXT_SWITCHING: self._benchmark_context_switching,
            BenchmarkType.DETERMINISM_TEST: self._benchmark_determinism,
            BenchmarkType.STRESS_TEST: self._benchmark_stress_test,
        }
    
    @contextmanager
    def _timing_context(self):
        """Context manager for high-precision timing"""
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        self._last_timing_us = (end_time - start_time) * 1_000_000
    
    def _benchmark_timing_precision(self) -> BenchmarkResult:
        """Test basic timing precision and accuracy"""
        logger.info("Running timing precision benchmark...")
        
        execution_times = []
        target_work_us = 100  # 100 microseconds of work
        
        # Warmup
        for _ in range(1000):
            with self._timing_context():
                self._simulate_work(target_work_us)
        
        # Actual test
        for _ in range(self.requirements.sample_count):
            with self._timing_context():
                self._simulate_work(target_work_us)
            execution_times.append(self._last_timing_us)
        
        # Validate timing precision
        times = np.array(execution_times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        max_time = np.max(times)
        
        success = (
            mean_time < self.requirements.max_execution_time_us and
            std_time < self.requirements.max_jitter_us and
            max_time < self.requirements.percentile_999_us
        )
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.TIMING_PRECISION,
            test_name="Basic Timing Precision",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'target_work_us': target_work_us,
                'timing_accuracy': abs(mean_time - target_work_us),
            }
        )
    
    def _benchmark_jitter_analysis(self) -> BenchmarkResult:
        """Analyze timing jitter under various conditions"""
        logger.info("Running jitter analysis benchmark...")
        
        execution_times = []
        
        # Test with periodic interrupts and variable load
        for i in range(self.requirements.sample_count):
            
            # Introduce variability every 100 iterations
            if i % 100 == 0:
                # Brief CPU load spike
                self._simulate_work(50)
            
            with self._timing_context():
                self._simulate_work(200)  # 200us target
            
            execution_times.append(self._last_timing_us)
        
        times = np.array(execution_times)
        jitter = np.std(times)
        max_jitter = np.max(times) - np.min(times)
        
        success = (
            jitter < self.requirements.max_jitter_us and
            max_jitter < self.requirements.max_jitter_us * 10
        )
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.JITTER_ANALYSIS,
            test_name="Timing Jitter Analysis",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'jitter_std': float(jitter),
                'max_jitter': float(max_jitter),
                'jitter_coefficient': float(jitter / np.mean(times)),
            }
        )
    
    def _benchmark_worst_case_timing(self) -> BenchmarkResult:
        """Test worst-case execution timing scenarios"""
        logger.info("Running worst-case timing benchmark...")
        
        execution_times = []
        
        # Test scenarios designed to trigger worst-case behavior
        scenarios = [
            ('normal', lambda: self._simulate_work(100)),
            ('cache_miss', lambda: self._simulate_cache_miss_work()),
            ('memory_allocation', lambda: self._simulate_memory_pressure_work()),
            ('context_switch', lambda: self._simulate_context_switch_work()),
            ('gc_trigger', lambda: self._simulate_gc_trigger_work()),
        ]
        
        for scenario_name, work_func in scenarios:
            scenario_times = []
            
            for _ in range(self.requirements.sample_count // len(scenarios)):
                with self._timing_context():
                    work_func()
                scenario_times.append(self._last_timing_us)
            
            execution_times.extend(scenario_times)
        
        times = np.array(execution_times)
        p999_time = np.percentile(times, 99.9)
        max_time = np.max(times)
        
        success = (
            p999_time < self.requirements.percentile_999_us and
            max_time < self.requirements.percentile_999_us * 2
        )
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.WORST_CASE_TIMING,
            test_name="Worst-case Timing Analysis",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'scenarios_tested': len(scenarios),
                'worst_case_time': float(max_time),
            }
        )
    
    def _benchmark_memory_allocation(self) -> BenchmarkResult:
        """Test memory allocation performance impact"""
        logger.info("Running memory allocation benchmark...")
        
        execution_times = []
        
        # Test different allocation patterns
        for _ in range(self.requirements.sample_count):
            with self._timing_context():
                # Simulate typical RT loop with some memory operations
                data = np.zeros(1000, dtype=np.float32)  # Small allocation
                result = np.sum(data * 2.0)  # Some computation
                del data  # Explicit cleanup
            
            execution_times.append(self._last_timing_us)
        
        times = np.array(execution_times)
        mean_time = np.mean(times)
        
        success = mean_time < self.requirements.max_execution_time_us * 2
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.MEMORY_ALLOCATION,
            test_name="Memory Allocation Impact",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'allocation_pattern': 'numpy_arrays_1KB',
            }
        )
    
    def _benchmark_cache_performance(self) -> BenchmarkResult:
        """Test cache performance characteristics"""
        logger.info("Running cache performance benchmark...")
        
        execution_times = []
        
        # Create data larger than L3 cache (typical 8-32MB)
        cache_buster_size = 64 * 1024 * 1024  # 64MB
        cache_buster = np.random.random(cache_buster_size // 8)
        
        # Test with cache-friendly access pattern
        test_data = np.random.random(10000)
        
        for i in range(self.requirements.sample_count):
            with self._timing_context():
                # Cache-friendly sequential access
                result = np.sum(test_data[:1000])
                
                # Occasionally bust cache
                if i % 100 == 0:
                    _ = np.sum(cache_buster[::1000])  # Sparse access
            
            execution_times.append(self._last_timing_us)
        
        times = np.array(execution_times)
        
        success = np.percentile(times, 95) < self.requirements.percentile_99_us
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.CACHE_PERFORMANCE,
            test_name="Cache Performance Analysis",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'cache_buster_size_mb': cache_buster_size / (1024*1024),
                'test_data_size': len(test_data),
            }
        )
    
    def _benchmark_context_switching(self) -> BenchmarkResult:
        """Test context switching overhead"""
        logger.info("Running context switching benchmark...")
        
        execution_times = []
        
        def worker_function():
            """Simple worker that does minimal work"""
            return sum(range(100))
        
        # Test with thread pool to induce context switches
        with ThreadPoolExecutor(max_workers=4) as executor:
            
            for _ in range(self.requirements.sample_count):
                with self._timing_context():
                    # Submit work that may cause context switches
                    future = executor.submit(worker_function)
                    result = future.result()
                
                execution_times.append(self._last_timing_us)
        
        times = np.array(execution_times)
        
        success = np.mean(times) < self.requirements.max_execution_time_us * 5
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.CONTEXT_SWITCHING,
            test_name="Context Switching Overhead",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'thread_pool_workers': 4,
            }
        )
    
    def _benchmark_determinism(self) -> BenchmarkResult:
        """Test determinism across multiple runs"""
        logger.info("Running determinism benchmark...")
        
        # Run the same operation multiple times and check consistency
        num_runs = 10
        run_results = []
        
        for run in range(num_runs):
            run_times = []
            
            # Same operation each run
            for _ in range(1000):
                with self._timing_context():
                    self._simulate_work(500)  # 500us work
                run_times.append(self._last_timing_us)
            
            run_mean = np.mean(run_times)
            run_std = np.std(run_times)
            run_results.append((run_mean, run_std))
        
        # Analyze consistency across runs
        means = [r[0] for r in run_results]
        stds = [r[1] for r in run_results]
        
        mean_variation = np.std(means) / np.mean(means)  # Coefficient of variation
        std_variation = np.std(stds) / np.mean(stds)
        
        # Flatten all times for overall result
        execution_times = []
        for run in range(num_runs):
            run_times = []
            for _ in range(100):  # Reduced for result storage
                with self._timing_context():
                    self._simulate_work(500)
                run_times.append(self._last_timing_us)
            execution_times.extend(run_times)
        
        # Success if variation is low
        success = mean_variation < 0.05 and std_variation < 0.1  # 5% and 10% thresholds
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.DETERMINISM_TEST,
            test_name="Determinism Validation",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'num_runs': num_runs,
                'mean_variation': float(mean_variation),
                'std_variation': float(std_variation),
                'run_means': means,
                'run_stds': stds,
            }
        )
    
    def _benchmark_stress_test(self) -> BenchmarkResult:
        """Stress test under high system load"""
        logger.info("Running stress test benchmark...")
        
        execution_times = []
        
        # Start system load
        load_config = SystemLoad(
            cpu_load_percent=50.0,
            memory_load_mb=1024,  # 1GB
            io_operations_per_sec=100,
            duration_seconds=60
        )
        
        self.load_generator.start_load(load_config)
        
        try:
            # Wait for load to stabilize
            time.sleep(5)
            
            # Run test under load
            for _ in range(self.requirements.sample_count // 2):  # Reduced due to load
                with self._timing_context():
                    self._simulate_work(300)
                execution_times.append(self._last_timing_us)
            
        finally:
            self.load_generator.stop_load()
        
        times = np.array(execution_times)
        p95_time = np.percentile(times, 95)
        
        # More lenient success criteria under load
        success = p95_time < self.requirements.max_execution_time_us * 3
        
        return BenchmarkResult(
            benchmark_type=BenchmarkType.STRESS_TEST,
            test_name="High Load Stress Test",
            success=success,
            execution_times_us=execution_times,
            metadata={
                'cpu_load': load_config.cpu_load_percent,
                'memory_load_mb': load_config.memory_load_mb,
                'io_load_ops_sec': load_config.io_operations_per_sec,
            }
        )
    
    def _simulate_work(self, target_time_us: float):
        """Simulate computational work for specified duration"""
        end_time = time.perf_counter() + (target_time_us / 1_000_000)
        count = 0
        while time.perf_counter() < end_time:
            count += 1
            math.sqrt(count)  # Some computation
    
    def _simulate_cache_miss_work(self):
        """Simulate work that causes cache misses"""
        # Access memory in pattern likely to cause cache misses
        size = 1024 * 1024  # 1MB array
        data = np.random.random(size)
        
        # Stride access pattern to bust cache
        stride = 4096  # Jump by 4KB
        total = 0.0
        for i in range(0, len(data), stride):
            total += data[i]
        
        return total
    
    def _simulate_memory_pressure_work(self):
        """Simulate work under memory pressure"""
        # Allocate and immediately free memory
        temp_data = []
        for _ in range(10):
            temp_data.append(np.zeros(10000))
        
        result = sum(np.sum(arr) for arr in temp_data)
        del temp_data
        return result
    
    def _simulate_context_switch_work(self):
        """Simulate work that may trigger context switches"""
        # Sleep briefly to potentially yield to other threads
        time.sleep(0.00001)  # 10 microseconds
        return sum(range(100))
    
    def _simulate_gc_trigger_work(self):
        """Simulate work that may trigger garbage collection"""
        # Create objects that will need garbage collection
        temp_objects = [[] for _ in range(1000)]
        for obj_list in temp_objects:
            obj_list.extend(range(100))
        
        result = sum(len(obj_list) for obj_list in temp_objects)
        
        # Force garbage collection
        if random.random() < 0.1:  # 10% chance
            gc.collect()
        
        return result
    
    def run_benchmark_suite(self, benchmark_types: Optional[List[BenchmarkType]] = None) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            benchmark_types: Specific benchmarks to run (None for all)
            
        Returns:
            List of benchmark results
        """
        if benchmark_types is None:
            benchmark_types = list(self.benchmark_functions.keys())
        
        logger.info(f"Running {len(benchmark_types)} benchmarks...")
        
        results = []
        start_time = time.time()
        
        for benchmark_type in benchmark_types:
            if benchmark_type in self.benchmark_functions:
                try:
                    logger.info(f"Running {benchmark_type.value}...")
                    result = self.benchmark_functions[benchmark_type]()
                    results.append(result)
                    
                    # Log immediate result
                    status = "PASS" if result.success else "FAIL"
                    logger.info(f"{benchmark_type.value}: {status}")
                    
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_type.value} failed: {e}")
                    # Create failed result
                    failed_result = BenchmarkResult(
                        benchmark_type=benchmark_type,
                        test_name=f"Failed: {benchmark_type.value}",
                        success=False,
                        execution_times_us=[],
                        metadata={'error': str(e)}
                    )
                    results.append(failed_result)
        
        total_time = time.time() - start_time
        logger.info(f"Benchmark suite completed in {total_time:.1f} seconds")
        
        self.results.extend(results)
        return results
    
    def validate_determinism(self, num_iterations: int = 1000) -> Dict[str, Any]:
        """
        Validate system determinism with repeated measurements.
        
        Args:
            num_iterations: Number of iterations to test
            
        Returns:
            Determinism validation results
        """
        logger.info(f"Validating determinism with {num_iterations} iterations...")
        
        # Test consistent timing for identical operations
        operation_times = []
        
        for _ in range(num_iterations):
            with self._timing_context():
                # Consistent operation
                result = sum(math.sqrt(i) for i in range(1000))
            operation_times.append(self._last_timing_us)
        
        times = np.array(operation_times)
        
        # Statistical analysis
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time  # Coefficient of variation
        
        # Check for outliers (>3 standard deviations)
        z_scores = np.abs((times - mean_time) / std_time)
        outliers = np.sum(z_scores > 3)
        outlier_rate = outliers / len(times)
        
        # Check for trends (should be minimal)
        # Split into chunks and compare means
        chunk_size = len(times) // 10
        chunk_means = []
        for i in range(0, len(times), chunk_size):
            chunk = times[i:i+chunk_size]
            if len(chunk) > 10:  # Sufficient sample size
                chunk_means.append(np.mean(chunk))
        
        trend_variation = np.std(chunk_means) / np.mean(chunk_means) if chunk_means else 0
        
        # Determinism criteria
        determinism_score = 1.0 - (cv + outlier_rate + trend_variation)
        is_deterministic = (
            cv < 0.1 and           # Low coefficient of variation
            outlier_rate < 0.01 and  # Less than 1% outliers
            trend_variation < 0.05    # Low trend variation
        )
        
        return {
            'is_deterministic': is_deterministic,
            'determinism_score': max(0.0, determinism_score),
            'coefficient_of_variation': cv,
            'outlier_rate': outlier_rate,
            'trend_variation': trend_variation,
            'mean_time_us': mean_time,
            'std_time_us': std_time,
            'iterations_tested': num_iterations,
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report data dictionary
        """
        if not self.results:
            logger.warning("No benchmark results available for report")
            return {}
        
        logger.info("Generating benchmark report...")
        
        # Overall summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        
        # Collect all timing data
        all_times = []
        for result in self.results:
            all_times.extend(result.execution_times_us)
        
        if all_times:
            all_times = np.array(all_times)
            overall_stats = {
                'mean_us': float(np.mean(all_times)),
                'median_us': float(np.median(all_times)),
                'std_us': float(np.std(all_times)),
                'p95_us': float(np.percentile(all_times, 95)),
                'p99_us': float(np.percentile(all_times, 99)),
                'p999_us': float(np.percentile(all_times, 99.9)),
                'max_us': float(np.max(all_times)),
                'total_samples': len(all_times),
            }
        else:
            overall_stats = {}
        
        # Requirements compliance
        compliance = self._analyze_requirements_compliance(overall_stats)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'requirements': {
                'max_execution_time_us': self.requirements.max_execution_time_us,
                'max_jitter_us': self.requirements.max_jitter_us,
                'percentile_99_us': self.requirements.percentile_99_us,
                'percentile_999_us': self.requirements.percentile_999_us,
            },
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            },
            'overall_performance': overall_stats,
            'requirements_compliance': compliance,
            'individual_results': [
                {
                    'benchmark_type': r.benchmark_type.value,
                    'test_name': r.test_name,
                    'success': r.success,
                    'statistics': r.metadata,
                }
                for r in self.results
            ],
        }
        
        # Save report if requested
        if output_file:
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def _analyze_requirements_compliance(self, overall_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compliance with timing requirements"""
        if not overall_stats:
            return {'compliant': False, 'violations': ['No timing data available']}
        
        violations = []
        
        # Check maximum execution time
        if overall_stats.get('max_us', 0) > self.requirements.max_execution_time_us:
            violations.append(f"Max execution time {overall_stats['max_us']:.1f}μs > {self.requirements.max_execution_time_us}μs")
        
        # Check jitter (using standard deviation as proxy)
        if overall_stats.get('std_us', 0) > self.requirements.max_jitter_us:
            violations.append(f"Jitter {overall_stats['std_us']:.1f}μs > {self.requirements.max_jitter_us}μs")
        
        # Check percentiles
        if overall_stats.get('p99_us', 0) > self.requirements.percentile_99_us:
            violations.append(f"P99 time {overall_stats['p99_us']:.1f}μs > {self.requirements.percentile_99_us}μs")
        
        if overall_stats.get('p999_us', 0) > self.requirements.percentile_999_us:
            violations.append(f"P99.9 time {overall_stats['p999_us']:.1f}μs > {self.requirements.percentile_999_us}μs")
        
        return {
            'compliant': len(violations) == 0,
            'violations': violations,
            'compliance_score': 1.0 - (len(violations) / 4),  # 4 main requirements
        }
    
    def plot_results(self, output_dir: str = "benchmark_plots"):
        """Generate visualization plots for benchmark results"""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting not available - install matplotlib and seaborn")
            return
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Plot timing distributions
        self._plot_timing_distributions(output_dir)
        
        # Plot benchmark comparison
        self._plot_benchmark_comparison(output_dir)
        
        # Plot requirements compliance
        self._plot_requirements_compliance(output_dir)
        
        logger.info(f"Plots saved to {output_dir}")
    
    def _plot_timing_distributions(self, output_dir: str):
        """Plot timing distributions for each benchmark"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for result in self.results[:4]:  # Plot first 4 results
            if not result.execution_times_us or plot_idx >= len(axes):
                continue
            
            ax = axes[plot_idx]
            times = np.array(result.execution_times_us)
            
            # Histogram with density
            ax.hist(times, bins=50, alpha=0.7, density=True, edgecolor='black')
            
            # Add percentile lines
            p95 = np.percentile(times, 95)
            p99 = np.percentile(times, 99)
            
            ax.axvline(p95, color='orange', linestyle='--', label=f'P95: {p95:.1f}μs')
            ax.axvline(p99, color='red', linestyle='--', label=f'P99: {p99:.1f}μs')
            
            # Add requirement line if applicable
            if 'max_execution_time_us' in dir(self.requirements):
                ax.axvline(self.requirements.max_execution_time_us, 
                          color='green', linestyle='-', linewidth=2,
                          label=f'Requirement: {self.requirements.max_execution_time_us}μs')
            
            ax.set_xlabel('Execution Time (μs)')
            ax.set_ylabel('Density')
            ax.set_title(f'{result.test_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/timing_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_benchmark_comparison(self, output_dir: str):
        """Plot comparison of different benchmarks"""
        if not self.results:
            return
        
        # Extract key metrics for comparison
        benchmark_names = []
        mean_times = []
        p99_times = []
        success_indicators = []
        
        for result in self.results:
            if result.execution_times_us:
                benchmark_names.append(result.test_name[:15] + "..." if len(result.test_name) > 15 else result.test_name)
                mean_times.append(result.metadata.get('mean_us', 0))
                p99_times.append(result.metadata.get('p99_us', 0))
                success_indicators.append(1 if result.success else 0)
        
        if not benchmark_names:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean times comparison
        bars1 = ax1.bar(range(len(benchmark_names)), mean_times, 
                       color=['green' if s else 'red' for s in success_indicators])
        ax1.set_xlabel('Benchmark')
        ax1.set_ylabel('Mean Execution Time (μs)')
        ax1.set_title('Mean Execution Times by Benchmark')
        ax1.set_xticks(range(len(benchmark_names)))
        ax1.set_xticklabels(benchmark_names, rotation=45, ha='right')
        
        # Add requirement line
        ax1.axhline(self.requirements.max_execution_time_us, color='orange', 
                   linestyle='--', label=f'Requirement: {self.requirements.max_execution_time_us}μs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P99 times comparison
        bars2 = ax2.bar(range(len(benchmark_names)), p99_times,
                       color=['green' if s else 'red' for s in success_indicators])
        ax2.set_xlabel('Benchmark')
        ax2.set_ylabel('P99 Execution Time (μs)')
        ax2.set_title('P99 Execution Times by Benchmark')
        ax2.set_xticks(range(len(benchmark_names)))
        ax2.set_xticklabels(benchmark_names, rotation=45, ha='right')
        
        ax2.axhline(self.requirements.percentile_99_us, color='orange',
                   linestyle='--', label=f'P99 Requirement: {self.requirements.percentile_99_us}μs')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/benchmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_requirements_compliance(self, output_dir: str):
        """Plot requirements compliance overview"""
        # Create compliance summary
        passed_tests = sum(1 for r in self.results if r.success)
        total_tests = len(self.results)
        
        if total_tests == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Pass/fail pie chart
        labels = ['Passed', 'Failed']
        sizes = [passed_tests, total_tests - passed_tests]
        colors = ['green', 'red']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Test Results Overview\n({passed_tests}/{total_tests} passed)')
        
        # Requirements compliance bar chart
        all_times = []
        for result in self.results:
            all_times.extend(result.execution_times_us)
        
        if all_times:
            times = np.array(all_times)
            
            requirements_data = {
                'Mean Time': (np.mean(times), self.requirements.max_execution_time_us),
                'P99 Time': (np.percentile(times, 99), self.requirements.percentile_99_us),
                'P99.9 Time': (np.percentile(times, 99.9), self.requirements.percentile_999_us),
                'Max Jitter': (np.std(times), self.requirements.max_jitter_us),
            }
            
            metrics = list(requirements_data.keys())
            actual_values = [requirements_data[m][0] for m in metrics]
            required_values = [requirements_data[m][1] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, actual_values, width, label='Actual', alpha=0.8)
            bars2 = ax2.bar(x + width/2, required_values, width, label='Required', alpha=0.8)
            
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Time (μs)')
            ax2.set_title('Requirements Compliance')
            ax2.set_xticks(x)
            ax2.set_xticklabels(metrics, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/requirements_compliance.png", dpi=300, bbox_inches='tight')
        plt.close()


# Example usage and testing
def main():
    """Example usage of the RT performance benchmark system"""
    
    # Define timing requirements
    requirements = TimingRequirements(
        max_execution_time_us=1000,   # 1ms max execution
        max_jitter_us=50,             # 50μs max jitter
        percentile_99_us=1200,        # P99 < 1.2ms
        percentile_999_us=1500,       # P99.9 < 1.5ms
        sample_count=10000,           # 10k samples per test
    )
    
    # Create benchmark system
    benchmark = RTPerformanceBenchmark(requirements)
    
    # Run selected benchmarks (reduced set for example)
    benchmark_types = [
        BenchmarkType.TIMING_PRECISION,
        BenchmarkType.JITTER_ANALYSIS,
        BenchmarkType.MEMORY_ALLOCATION,
        BenchmarkType.DETERMINISM_TEST,
    ]
    
    # Run benchmarks
    results = benchmark.run_benchmark_suite(benchmark_types)
    
    # Test determinism separately
    determinism_results = benchmark.validate_determinism(1000)
    print(f"\nDeterminism Results:")
    print(f"  Deterministic: {determinism_results['is_deterministic']}")
    print(f"  Score: {determinism_results['determinism_score']:.3f}")
    print(f"  Coefficient of Variation: {determinism_results['coefficient_of_variation']:.4f}")
    print(f"  Outlier Rate: {determinism_results['outlier_rate']:.3f}")
    
    # Generate comprehensive report
    report = benchmark.generate_report("benchmark_report.json")
    
    # Print summary
    print(f"\nBenchmark Summary:")
    print(f"  Tests Run: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed_tests']}")
    print(f"  Failed: {report['summary']['failed_tests']}")
    print(f"  Pass Rate: {report['summary']['pass_rate']:.1%}")
    
    if report['overall_performance']:
        perf = report['overall_performance']
        print(f"\nOverall Performance:")
        print(f"  Mean Time: {perf['mean_us']:.1f}μs")
        print(f"  P95 Time: {perf['p95_us']:.1f}μs")
        print(f"  P99 Time: {perf['p99_us']:.1f}μs")
        print(f"  Max Time: {perf['max_us']:.1f}μs")
        print(f"  Jitter (std): {perf['std_us']:.1f}μs")
    
    compliance = report.get('requirements_compliance', {})
    if compliance:
        print(f"\nRequirements Compliance:")
        print(f"  Compliant: {compliance['compliant']}")
        print(f"  Score: {compliance['compliance_score']:.1%}")
        if compliance['violations']:
            print(f"  Violations:")
            for violation in compliance['violations']:
                print(f"    • {violation}")
    
    # Generate plots if available
    try:
        benchmark.plot_results("benchmark_plots")
        print(f"\nPlots saved to: benchmark_plots/")
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run benchmark example
    main()