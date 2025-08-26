"""
Performance monitoring and profiling utilities for CPO algorithm.

This module provides comprehensive performance monitoring including:
- Algorithm execution timing
- Memory usage tracking
- Convergence analysis
- Computational bottleneck identification
- Performance regression detection
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import psutil
import torch
import numpy as np
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from contextlib import contextmanager
import threading
import pickle
import json
from datetime import datetime
import functools
import tracemalloc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for algorithm components."""
    execution_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    memory_usage: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    iteration_times: List[float] = field(default_factory=list)
    convergence_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    gpu_utilization: List[float] = field(default_factory=list)
    cpu_utilization: List[float] = field(default_factory=list)
    
    def add_execution_time(self, component: str, duration: float) -> None:
        """Add execution time for a component."""
        self.execution_times[component].append(duration)
    
    def add_memory_usage(self, component: str, memory_mb: float) -> None:
        """Add memory usage for a component."""
        self.memory_usage[component].append(memory_mb)
    
    def add_convergence_metric(self, metric_name: str, value: float) -> None:
        """Add convergence metric."""
        self.convergence_metrics[metric_name].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        # Execution time summary
        if self.execution_times:
            summary["execution_times"] = {}
            for component, times in self.execution_times.items():
                if times:
                    summary["execution_times"][component] = {
                        "mean": np.mean(times),
                        "std": np.std(times),
                        "min": np.min(times),
                        "max": np.max(times),
                        "total": np.sum(times),
                        "count": len(times)
                    }
        
        # Memory usage summary
        if self.memory_usage:
            summary["memory_usage"] = {}
            for component, memory in self.memory_usage.items():
                if memory:
                    summary["memory_usage"][component] = {
                        "mean": np.mean(memory),
                        "std": np.std(memory),
                        "min": np.min(memory),
                        "max": np.max(memory),
                        "peak": np.max(memory)
                    }
        
        # System utilization
        if self.cpu_utilization:
            summary["cpu_utilization"] = {
                "mean": np.mean(self.cpu_utilization),
                "max": np.max(self.cpu_utilization)
            }
        
        if self.gpu_utilization:
            summary["gpu_utilization"] = {
                "mean": np.mean(self.gpu_utilization),
                "max": np.max(self.gpu_utilization)
            }
        
        # Iteration timing
        if self.iteration_times:
            summary["iteration_timing"] = {
                "mean_iteration_time": np.mean(self.iteration_times),
                "total_time": np.sum(self.iteration_times),
                "iterations_per_second": len(self.iteration_times) / np.sum(self.iteration_times)
            }
        
        return summary


@dataclass
class SystemResourceUsage:
    """System resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_memory_mb: float
    gpu_utilization: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


class PerformanceProfiler:
    """
    Comprehensive performance profiler for CPO algorithm.
    
    Tracks execution times, memory usage, and system resources
    for all major algorithm components.
    """
    
    def __init__(self, 
                 enable_memory_tracking: bool = True,
                 enable_system_monitoring: bool = True,
                 monitoring_interval: float = 1.0,
                 history_size: int = 1000):
        """
        Initialize performance profiler.
        
        Args:
            enable_memory_tracking: Enable detailed memory tracking
            enable_system_monitoring: Enable system resource monitoring
            monitoring_interval: System monitoring interval (seconds)
            history_size: Number of historical measurements to keep
        """
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_system_monitoring = enable_system_monitoring
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Performance metrics
        self.metrics = PerformanceMetrics()
        self.active_timers = {}
        self.memory_snapshots = deque(maxlen=history_size)
        self.resource_usage = deque(maxlen=history_size)
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Memory tracking
        if enable_memory_tracking:
            try:
                tracemalloc.start()
                self.memory_tracking_available = True
                logger.info("Memory tracking enabled")
            except Exception as e:
                self.memory_tracking_available = False
                logger.warning(f"Memory tracking not available: {e}")
        else:
            self.memory_tracking_available = False
        
        # GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            logger.info(f"GPU monitoring enabled: {torch.cuda.get_device_name()}")
        
        # Process info
        self.process = psutil.Process()
        
        logger.info("PerformanceProfiler initialized")
    
    def start_system_monitoring(self) -> None:
        """Start system resource monitoring thread."""
        if self.monitoring_active or not self.enable_system_monitoring:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("System monitoring started")
    
    def stop_system_monitoring(self) -> None:
        """Stop system resource monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """System resource monitoring loop."""
        while self.monitoring_active:
            try:
                # CPU and memory
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                # GPU info
                gpu_memory_mb = 0.0
                gpu_utilization = 0.0
                if self.gpu_available:
                    try:
                        gpu_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                        # GPU utilization requires nvidia-ml-py or similar
                        gpu_utilization = 0.0  # Placeholder
                    except Exception:
                        pass
                
                # Disk I/O
                disk_io = self.process.io_counters()
                
                # Network I/O (system-wide)
                try:
                    net_io = psutil.net_io_counters()
                    network_sent_mb = net_io.bytes_sent / (1024 ** 2)
                    network_recv_mb = net_io.bytes_recv / (1024 ** 2)
                except Exception:
                    network_sent_mb = 0.0
                    network_recv_mb = 0.0
                
                # Create resource usage snapshot
                usage = SystemResourceUsage(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_mb=memory_info.rss / (1024 ** 2),
                    gpu_memory_mb=gpu_memory_mb,
                    gpu_utilization=gpu_utilization,
                    disk_io_read_mb=disk_io.read_bytes / (1024 ** 2),
                    disk_io_write_mb=disk_io.write_bytes / (1024 ** 2),
                    network_sent_mb=network_sent_mb,
                    network_recv_mb=network_recv_mb
                )
                
                self.resource_usage.append(usage)
                self.metrics.cpu_utilization.append(cpu_percent)
                self.metrics.gpu_utilization.append(gpu_utilization)
                
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
            
            time.sleep(self.monitoring_interval)
    
    @contextmanager
    def profile(self, component_name: str):
        """
        Context manager for profiling algorithm components.
        
        Args:
            component_name: Name of the component being profiled
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.metrics.add_execution_time(component_name, duration)
            if memory_delta > 0:  # Only track increases
                self.metrics.add_memory_usage(component_name, memory_delta)
            
            logger.debug(f"{component_name}: {duration:.4f}s, {memory_delta:.2f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if self.memory_tracking_available:
                current, peak = tracemalloc.get_traced_memory()
                return current / (1024 ** 2)
            else:
                return self.process.memory_info().rss / (1024 ** 2)
        except Exception:
            return 0.0
    
    def start_timer(self, timer_name: str) -> None:
        """Start a named timer."""
        self.active_timers[timer_name] = time.time()
    
    def stop_timer(self, timer_name: str) -> float:
        """
        Stop a named timer and return elapsed time.
        
        Args:
            timer_name: Name of the timer
            
        Returns:
            Elapsed time in seconds
        """
        if timer_name not in self.active_timers:
            logger.warning(f"Timer {timer_name} was not started")
            return 0.0
        
        elapsed = time.time() - self.active_timers[timer_name]
        del self.active_timers[timer_name]
        
        self.metrics.add_execution_time(timer_name, elapsed)
        return elapsed
    
    def profile_iteration(self, iteration_func: Callable, *args, **kwargs) -> Any:
        """
        Profile a complete CPO iteration.
        
        Args:
            iteration_func: Function to profile
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
        """
        iteration_start = time.time()
        
        with self.profile("cpo_iteration"):
            result = iteration_func(*args, **kwargs)
        
        iteration_time = time.time() - iteration_start
        self.metrics.iteration_times.append(iteration_time)
        
        return result
    
    def add_convergence_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """Add convergence metrics for analysis."""
        for metric_name, value in metrics_dict.items():
            self.metrics.add_convergence_metric(metric_name, value)
    
    def get_bottlenecks(self, top_k: int = 5) -> List[Tuple[str, Dict[str, float]]]:
        """
        Identify performance bottlenecks.
        
        Args:
            top_k: Number of top bottlenecks to return
            
        Returns:
            List of (component_name, stats) tuples
        """
        bottlenecks = []
        
        for component, times in self.metrics.execution_times.items():
            if times:
                total_time = sum(times)
                mean_time = np.mean(times)
                std_time = np.std(times)
                
                bottlenecks.append((component, {
                    "total_time": total_time,
                    "mean_time": mean_time,
                    "std_time": std_time,
                    "call_count": len(times),
                    "time_per_call": mean_time
                }))
        
        # Sort by total time
        bottlenecks.sort(key=lambda x: x[1]["total_time"], reverse=True)
        
        return bottlenecks[:top_k]
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence behavior from metrics.
        
        Returns:
            Dictionary with convergence analysis
        """
        analysis = {}
        
        for metric_name, values in self.metrics.convergence_metrics.items():
            if len(values) < 2:
                continue
            
            # Trend analysis
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            trend_slope = coeffs[0]
            
            # Variance analysis
            variance = np.var(values)
            
            # Stationarity test (simple)
            if len(values) >= 10:
                first_half_mean = np.mean(values[:len(values)//2])
                second_half_mean = np.mean(values[len(values)//2:])
                stationarity_ratio = abs(second_half_mean - first_half_mean) / (first_half_mean + 1e-8)
            else:
                stationarity_ratio = 0.0
            
            analysis[metric_name] = {
                "trend_slope": trend_slope,
                "variance": variance,
                "final_value": values[-1],
                "initial_value": values[0],
                "improvement": values[-1] - values[0],
                "stationarity_ratio": stationarity_ratio,
                "converged": stationarity_ratio < 0.1 and abs(trend_slope) < 0.01
            }
        
        return analysis
    
    def get_resource_usage_stats(self) -> Dict[str, Any]:
        """Get system resource usage statistics."""
        if not self.resource_usage:
            return {}
        
        cpu_usage = [u.cpu_percent for u in self.resource_usage]
        memory_usage = [u.memory_mb for u in self.resource_usage]
        gpu_memory = [u.gpu_memory_mb for u in self.resource_usage]
        
        return {
            "cpu": {
                "mean": np.mean(cpu_usage),
                "max": np.max(cpu_usage),
                "std": np.std(cpu_usage)
            },
            "memory": {
                "mean": np.mean(memory_usage),
                "max": np.max(memory_usage),
                "min": np.min(memory_usage),
                "peak": np.max(memory_usage)
            },
            "gpu_memory": {
                "mean": np.mean(gpu_memory),
                "max": np.max(gpu_memory),
                "peak": np.max(gpu_memory)
            } if gpu_memory and any(g > 0 for g in gpu_memory) else {},
            "monitoring_duration": (self.resource_usage[-1].timestamp - 
                                  self.resource_usage[0].timestamp) if len(self.resource_usage) > 1 else 0
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "profiler_config": {
                "memory_tracking": self.enable_memory_tracking,
                "system_monitoring": self.enable_system_monitoring,
                "gpu_available": self.gpu_available,
                "history_size": self.history_size
            },
            "summary": self.metrics.get_summary(),
            "bottlenecks": self.get_bottlenecks(),
            "convergence_analysis": self.analyze_convergence(),
            "resource_usage": self.get_resource_usage_stats()
        }
        
        # Add performance insights
        report["insights"] = self._generate_insights(report)
        
        return report
    
    def _generate_insights(self, report: Dict[str, Any]) -> List[str]:
        """Generate performance insights from report data."""
        insights = []
        
        # Execution time insights
        if "execution_times" in report["summary"]:
            exec_times = report["summary"]["execution_times"]
            
            # Find slowest components
            slowest = max(exec_times.items(), key=lambda x: x[1]["mean"], default=None)
            if slowest:
                insights.append(f"Slowest component: {slowest[0]} ({slowest[1]['mean']:.3f}s avg)")
            
            # Check for high variance
            for component, stats in exec_times.items():
                if stats["std"] / stats["mean"] > 0.5:  # High coefficient of variation
                    insights.append(f"High timing variance in {component} (CV: {stats['std']/stats['mean']:.2f})")
        
        # Memory insights
        if "memory_usage" in report["summary"]:
            memory_stats = report["summary"]["memory_usage"]
            
            # Find memory-intensive components
            memory_intensive = max(memory_stats.items(), key=lambda x: x[1]["peak"], default=None)
            if memory_intensive and memory_intensive[1]["peak"] > 100:  # > 100MB
                insights.append(f"Memory intensive: {memory_intensive[0]} (peak: {memory_intensive[1]['peak']:.1f}MB)")
        
        # Resource usage insights
        if "resource_usage" in report:
            resource_stats = report["resource_usage"]
            
            if "cpu" in resource_stats and resource_stats["cpu"]["max"] > 90:
                insights.append(f"High CPU usage detected (peak: {resource_stats['cpu']['max']:.1f}%)")
            
            if "memory" in resource_stats and resource_stats["memory"]["peak"] > 1000:  # > 1GB
                insights.append(f"High memory usage (peak: {resource_stats['memory']['peak']:.1f}MB)")
        
        # Convergence insights
        if "convergence_analysis" in report:
            convergence = report["convergence_analysis"]
            
            for metric, analysis in convergence.items():
                if analysis["converged"]:
                    insights.append(f"{metric} appears to have converged")
                elif abs(analysis["trend_slope"]) > 0.1:
                    trend = "increasing" if analysis["trend_slope"] > 0 else "decreasing"
                    insights.append(f"{metric} shows {trend} trend")
        
        if not insights:
            insights.append("No significant performance issues detected")
        
        return insights
    
    def save_report(self, filepath: str) -> None:
        """Save performance report to file."""
        report = self.generate_performance_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance report saved to {filepath}")
    
    def compare_with_baseline(self, baseline_report_path: str) -> Dict[str, Any]:
        """
        Compare current performance with baseline.
        
        Args:
            baseline_report_path: Path to baseline performance report
            
        Returns:
            Comparison results
        """
        with open(baseline_report_path, 'r') as f:
            baseline = json.load(f)
        
        current = self.generate_performance_report()
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "baseline_timestamp": baseline.get("timestamp", "unknown"),
            "component_comparisons": {},
            "overall_assessment": "unknown"
        }
        
        # Compare execution times
        if ("summary" in baseline and "execution_times" in baseline["summary"] and
            "summary" in current and "execution_times" in current["summary"]):
            
            baseline_times = baseline["summary"]["execution_times"]
            current_times = current["summary"]["execution_times"]
            
            for component in set(baseline_times.keys()) & set(current_times.keys()):
                baseline_mean = baseline_times[component]["mean"]
                current_mean = current_times[component]["mean"]
                
                speedup = baseline_mean / current_mean if current_mean > 0 else 0
                percent_change = ((current_mean - baseline_mean) / baseline_mean) * 100
                
                comparison["component_comparisons"][component] = {
                    "baseline_mean": baseline_mean,
                    "current_mean": current_mean,
                    "speedup": speedup,
                    "percent_change": percent_change,
                    "improved": speedup > 1.0
                }
        
        # Overall assessment
        improvements = sum(1 for comp in comparison["component_comparisons"].values() if comp["improved"])
        total_components = len(comparison["component_comparisons"])
        
        if improvements > total_components * 0.7:
            comparison["overall_assessment"] = "improved"
        elif improvements < total_components * 0.3:
            comparison["overall_assessment"] = "degraded"
        else:
            comparison["overall_assessment"] = "mixed"
        
        return comparison
    
    def reset(self) -> None:
        """Reset all performance metrics."""
        self.metrics = PerformanceMetrics()
        self.active_timers.clear()
        self.memory_snapshots.clear()
        self.resource_usage.clear()
        logger.info("Performance profiler reset")
    
    def __del__(self):
        """Cleanup when profiler is destroyed."""
        self.stop_system_monitoring()
        if self.memory_tracking_available:
            try:
                tracemalloc.stop()
            except Exception:
                pass


def profile_function(component_name: str, profiler: Optional[PerformanceProfiler] = None):
    """
    Decorator for profiling individual functions.
    
    Args:
        component_name: Name of the component being profiled
        profiler: Optional profiler instance (uses global if None)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if profiler is not None:
                with profiler.profile(component_name):
                    return func(*args, **kwargs)
            else:
                # Use basic timing if no profiler provided
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    logger.debug(f"{component_name}: {duration:.4f}s")
        
        return wrapper
    return decorator


class ConvergenceAnalyzer:
    """
    Analyzer for convergence behavior of CPO optimization.
    
    Tracks convergence metrics and provides statistical analysis
    of optimization progress.
    """
    
    def __init__(self, window_size: int = 50):
        """
        Initialize convergence analyzer.
        
        Args:
            window_size: Window size for moving statistics
        """
        self.window_size = window_size
        self.metrics_history = defaultdict(list)
        self.convergence_tests = {}
        
    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """Add metrics for convergence analysis."""
        for name, value in metrics.items():
            self.metrics_history[name].append(value)
            
            # Keep only recent history
            if len(self.metrics_history[name]) > self.window_size * 2:
                self.metrics_history[name] = self.metrics_history[name][-self.window_size:]
    
    def test_convergence(self, metric_name: str, tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Test convergence for a specific metric.
        
        Args:
            metric_name: Name of metric to test
            tolerance: Convergence tolerance
            
        Returns:
            Convergence test results
        """
        if metric_name not in self.metrics_history:
            return {"converged": False, "reason": "no_data"}
        
        values = self.metrics_history[metric_name]
        
        if len(values) < 10:
            return {"converged": False, "reason": "insufficient_data"}
        
        # Recent window for testing
        recent_values = values[-self.window_size:] if len(values) >= self.window_size else values
        
        # Test 1: Small variance in recent window
        variance = np.var(recent_values)
        variance_test = variance < tolerance
        
        # Test 2: Small trend in recent window
        x = np.arange(len(recent_values))
        coeffs = np.polyfit(x, recent_values, 1)
        trend_slope = abs(coeffs[0])
        trend_test = trend_slope < tolerance
        
        # Test 3: Stability test (compare two halves of recent window)
        if len(recent_values) >= 10:
            mid = len(recent_values) // 2
            first_half_mean = np.mean(recent_values[:mid])
            second_half_mean = np.mean(recent_values[mid:])
            stability_ratio = abs(second_half_mean - first_half_mean) / (abs(first_half_mean) + tolerance)
            stability_test = stability_ratio < 0.1
        else:
            stability_test = False
        
        # Overall convergence decision
        converged = variance_test and trend_test and stability_test
        
        return {
            "converged": converged,
            "variance": variance,
            "trend_slope": trend_slope,
            "stability_ratio": stability_ratio if len(recent_values) >= 10 else None,
            "tests": {
                "variance": variance_test,
                "trend": trend_test,
                "stability": stability_test
            },
            "recent_mean": np.mean(recent_values),
            "recent_std": np.std(recent_values)
        }
    
    def analyze_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Analyze convergence for all tracked metrics."""
        results = {}
        
        for metric_name in self.metrics_history.keys():
            results[metric_name] = self.test_convergence(metric_name)
        
        return results
    
    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get summary of convergence status."""
        all_results = self.analyze_all_metrics()
        
        total_metrics = len(all_results)
        converged_metrics = sum(1 for r in all_results.values() if r["converged"])
        
        return {
            "total_metrics": total_metrics,
            "converged_metrics": converged_metrics,
            "convergence_rate": converged_metrics / max(total_metrics, 1),
            "overall_converged": converged_metrics > 0 and converged_metrics == total_metrics,
            "individual_results": all_results
        }