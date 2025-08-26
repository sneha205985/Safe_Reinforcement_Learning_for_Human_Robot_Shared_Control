"""
Real-time Monitoring and Metrics Collection for Safe RL System.

This module provides comprehensive monitoring capabilities including:
- Safety violation tracking
- Performance metrics
- System health monitoring
- Custom business metrics
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import numpy as np
import psutil
import GPUtil
from prometheus_client import (
    Counter, Histogram, Gauge, Info, 
    CollectorRegistry, generate_latest,
    start_http_server, CONTENT_TYPE_LATEST
)
from contextlib import contextmanager
import redis
import json
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a metric value with timestamp and metadata."""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Defines alert rules for metrics."""
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.1", "< 0.95"
    severity: str  # "critical", "warning", "info"
    duration: int = 0  # seconds the condition must be true
    message: str = ""
    labels: Dict[str, str] = field(default_factory=dict)


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(self, name: str, description: str, labels: List[str] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.registry = None
        
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect current metric values."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass


class SafetyViolationRate(BaseMetric):
    """Tracks safety violations per time unit."""
    
    def __init__(self, window_size: int = 3600):  # 1 hour window
        super().__init__(
            "safety_violations_per_hour",
            "Number of safety violations in the past hour",
            ["algorithm", "environment", "severity"]
        )
        self.window_size = window_size
        self.violations = deque()
        self.counter = Counter(
            'safety_violations_total',
            'Total number of safety violations',
            ['algorithm', 'environment', 'severity']
        )
        self.rate_gauge = Gauge(
            'safety_violations_per_hour',
            'Safety violations per hour',
            ['algorithm', 'environment']
        )
        
    def record_violation(self, algorithm: str, environment: str, severity: str = "high"):
        """Record a safety violation."""
        now = datetime.now()
        self.violations.append({
            'timestamp': now,
            'algorithm': algorithm,
            'environment': environment,
            'severity': severity
        })
        
        # Update Prometheus counter
        self.counter.labels(
            algorithm=algorithm,
            environment=environment,
            severity=severity
        ).inc()
        
        # Clean old violations
        self._clean_old_violations()
        
        # Update rate
        self._update_rate()
        
    def _clean_old_violations(self):
        """Remove violations outside the time window."""
        cutoff = datetime.now() - timedelta(seconds=self.window_size)
        while self.violations and self.violations[0]['timestamp'] < cutoff:
            self.violations.popleft()
    
    def _update_rate(self):
        """Update the violations per hour rate."""
        self._clean_old_violations()
        
        # Group by algorithm and environment
        counts = defaultdict(int)
        for violation in self.violations:
            key = (violation['algorithm'], violation['environment'])
            counts[key] += 1
        
        # Update gauges
        for (algorithm, environment), count in counts.items():
            # Convert to hourly rate
            hourly_rate = count * (3600.0 / self.window_size)
            self.rate_gauge.labels(
                algorithm=algorithm,
                environment=environment
            ).set(hourly_rate)
    
    def collect(self) -> Dict[str, Any]:
        """Collect current violation statistics."""
        self._clean_old_violations()
        
        total_violations = len(self.violations)
        if total_violations == 0:
            return {
                'total_violations': 0,
                'hourly_rate': 0.0,
                'by_algorithm': {},
                'by_environment': {},
                'by_severity': {}
            }
        
        # Group violations
        by_algorithm = defaultdict(int)
        by_environment = defaultdict(int)
        by_severity = defaultdict(int)
        
        for violation in self.violations:
            by_algorithm[violation['algorithm']] += 1
            by_environment[violation['environment']] += 1
            by_severity[violation['severity']] += 1
        
        # Calculate hourly rate
        hourly_rate = total_violations * (3600.0 / self.window_size)
        
        return {
            'total_violations': total_violations,
            'hourly_rate': hourly_rate,
            'by_algorithm': dict(by_algorithm),
            'by_environment': dict(by_environment),
            'by_severity': dict(by_severity)
        }
    
    def reset(self):
        """Reset violation history."""
        self.violations.clear()


class LatencyMetric(BaseMetric):
    """Tracks latency percentiles for policy inference."""
    
    def __init__(self, percentile: int = 99, window_size: int = 1000):
        super().__init__(
            f"policy_inference_latency_p{percentile}",
            f"Policy inference latency {percentile}th percentile",
            ["algorithm", "environment"]
        )
        self.percentile = percentile
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        
        # Prometheus histogram for latency tracking
        self.histogram = Histogram(
            'policy_inference_duration_seconds',
            'Time spent on policy inference',
            ['algorithm', 'environment'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Gauge for percentile
        self.percentile_gauge = Gauge(
            f'policy_inference_latency_p{percentile}_seconds',
            f'Policy inference latency {percentile}th percentile',
            ['algorithm', 'environment']
        )
    
    @contextmanager
    def measure(self, algorithm: str, environment: str):
        """Context manager for measuring inference latency."""
        start_time = time.time()
        try:
            yield
        finally:
            latency = time.time() - start_time
            self.record_latency(latency, algorithm, environment)
    
    def record_latency(self, latency: float, algorithm: str, environment: str):
        """Record a latency measurement."""
        self.latencies.append({
            'latency': latency,
            'algorithm': algorithm,
            'environment': environment,
            'timestamp': datetime.now()
        })
        
        # Update Prometheus metrics
        self.histogram.labels(
            algorithm=algorithm,
            environment=environment
        ).observe(latency)
        
        # Update percentile
        self._update_percentile()
    
    def _update_percentile(self):
        """Update percentile calculations."""
        if not self.latencies:
            return
        
        # Group by algorithm and environment
        grouped_latencies = defaultdict(list)
        for entry in self.latencies:
            key = (entry['algorithm'], entry['environment'])
            grouped_latencies[key].append(entry['latency'])
        
        # Calculate percentiles
        for (algorithm, environment), latencies in grouped_latencies.items():
            if latencies:
                percentile_value = np.percentile(latencies, self.percentile)
                self.percentile_gauge.labels(
                    algorithm=algorithm,
                    environment=environment
                ).set(percentile_value)
    
    def collect(self) -> Dict[str, Any]:
        """Collect latency statistics."""
        if not self.latencies:
            return {
                'count': 0,
                'percentile_value': 0.0,
                'mean': 0.0,
                'std': 0.0
            }
        
        latency_values = [entry['latency'] for entry in self.latencies]
        
        return {
            'count': len(latency_values),
            'percentile_value': np.percentile(latency_values, self.percentile),
            'mean': np.mean(latency_values),
            'std': np.std(latency_values),
            'min': np.min(latency_values),
            'max': np.max(latency_values)
        }
    
    def reset(self):
        """Reset latency history."""
        self.latencies.clear()


class ConstraintMetric(BaseMetric):
    """Tracks constraint satisfaction rate."""
    
    def __init__(self, window_size: int = 1000):
        super().__init__(
            "constraint_satisfaction_rate",
            "Rate of constraint satisfaction",
            ["algorithm", "environment", "constraint_type"]
        )
        self.window_size = window_size
        self.evaluations = deque(maxlen=window_size)
        
        # Prometheus metrics
        self.satisfaction_counter = Counter(
            'constraint_evaluations_total',
            'Total constraint evaluations',
            ['algorithm', 'environment', 'constraint_type', 'satisfied']
        )
        
        self.satisfaction_rate = Gauge(
            'constraint_satisfaction_rate',
            'Constraint satisfaction rate',
            ['algorithm', 'environment', 'constraint_type']
        )
    
    def record_evaluation(self, satisfied: bool, algorithm: str, 
                         environment: str, constraint_type: str = "general"):
        """Record a constraint evaluation."""
        self.evaluations.append({
            'satisfied': satisfied,
            'algorithm': algorithm,
            'environment': environment,
            'constraint_type': constraint_type,
            'timestamp': datetime.now()
        })
        
        # Update Prometheus counter
        self.satisfaction_counter.labels(
            algorithm=algorithm,
            environment=environment,
            constraint_type=constraint_type,
            satisfied=str(satisfied)
        ).inc()
        
        # Update satisfaction rate
        self._update_satisfaction_rate()
    
    def _update_satisfaction_rate(self):
        """Update constraint satisfaction rates."""
        if not self.evaluations:
            return
        
        # Group by algorithm, environment, and constraint type
        grouped_evals = defaultdict(list)
        for eval_entry in self.evaluations:
            key = (
                eval_entry['algorithm'],
                eval_entry['environment'],
                eval_entry['constraint_type']
            )
            grouped_evals[key].append(eval_entry['satisfied'])
        
        # Calculate satisfaction rates
        for (algorithm, environment, constraint_type), satisfactions in grouped_evals.items():
            if satisfactions:
                rate = sum(satisfactions) / len(satisfactions)
                self.satisfaction_rate.labels(
                    algorithm=algorithm,
                    environment=environment,
                    constraint_type=constraint_type
                ).set(rate)
    
    def collect(self) -> Dict[str, Any]:
        """Collect constraint satisfaction statistics."""
        if not self.evaluations:
            return {
                'total_evaluations': 0,
                'satisfaction_rate': 0.0,
                'by_algorithm': {},
                'by_environment': {},
                'by_constraint_type': {}
            }
        
        total_evaluations = len(self.evaluations)
        satisfied_count = sum(1 for eval_entry in self.evaluations if eval_entry['satisfied'])
        satisfaction_rate = satisfied_count / total_evaluations
        
        # Group statistics
        by_algorithm = defaultdict(lambda: {'total': 0, 'satisfied': 0})
        by_environment = defaultdict(lambda: {'total': 0, 'satisfied': 0})
        by_constraint_type = defaultdict(lambda: {'total': 0, 'satisfied': 0})
        
        for eval_entry in self.evaluations:
            algorithm = eval_entry['algorithm']
            environment = eval_entry['environment']
            constraint_type = eval_entry['constraint_type']
            satisfied = eval_entry['satisfied']
            
            by_algorithm[algorithm]['total'] += 1
            by_environment[environment]['total'] += 1
            by_constraint_type[constraint_type]['total'] += 1
            
            if satisfied:
                by_algorithm[algorithm]['satisfied'] += 1
                by_environment[environment]['satisfied'] += 1
                by_constraint_type[constraint_type]['satisfied'] += 1
        
        # Calculate rates
        def calculate_rates(grouped_stats):
            rates = {}
            for key, stats in grouped_stats.items():
                rates[key] = stats['satisfied'] / stats['total'] if stats['total'] > 0 else 0.0
            return rates
        
        return {
            'total_evaluations': total_evaluations,
            'satisfaction_rate': satisfaction_rate,
            'by_algorithm': calculate_rates(by_algorithm),
            'by_environment': calculate_rates(by_environment),
            'by_constraint_type': calculate_rates(by_constraint_type)
        }
    
    def reset(self):
        """Reset constraint evaluation history."""
        self.evaluations.clear()


class SatisfactionMetric(BaseMetric):
    """Tracks human satisfaction scores."""
    
    def __init__(self, window_size: int = 100):
        super().__init__(
            "human_satisfaction_score",
            "Human satisfaction score (0-1 scale)",
            ["algorithm", "environment", "user_type"]
        )
        self.window_size = window_size
        self.scores = deque(maxlen=window_size)
        
        # Prometheus metrics
        self.satisfaction_histogram = Histogram(
            'human_satisfaction_score',
            'Human satisfaction score distribution',
            ['algorithm', 'environment', 'user_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.average_satisfaction = Gauge(
            'human_satisfaction_average',
            'Average human satisfaction score',
            ['algorithm', 'environment', 'user_type']
        )
    
    def record_score(self, score: float, algorithm: str, 
                    environment: str, user_type: str = "general"):
        """Record a satisfaction score."""
        if not 0.0 <= score <= 1.0:
            logger.warning(f"Invalid satisfaction score: {score}. Must be between 0 and 1.")
            return
        
        self.scores.append({
            'score': score,
            'algorithm': algorithm,
            'environment': environment,
            'user_type': user_type,
            'timestamp': datetime.now()
        })
        
        # Update Prometheus metrics
        self.satisfaction_histogram.labels(
            algorithm=algorithm,
            environment=environment,
            user_type=user_type
        ).observe(score)
        
        # Update average
        self._update_average()
    
    def _update_average(self):
        """Update average satisfaction scores."""
        if not self.scores:
            return
        
        # Group by algorithm, environment, and user type
        grouped_scores = defaultdict(list)
        for score_entry in self.scores:
            key = (
                score_entry['algorithm'],
                score_entry['environment'],
                score_entry['user_type']
            )
            grouped_scores[key].append(score_entry['score'])
        
        # Calculate averages
        for (algorithm, environment, user_type), scores in grouped_scores.items():
            if scores:
                avg_score = np.mean(scores)
                self.average_satisfaction.labels(
                    algorithm=algorithm,
                    environment=environment,
                    user_type=user_type
                ).set(avg_score)
    
    def collect(self) -> Dict[str, Any]:
        """Collect satisfaction statistics."""
        if not self.scores:
            return {
                'count': 0,
                'average': 0.0,
                'std': 0.0,
                'by_algorithm': {},
                'by_environment': {},
                'by_user_type': {}
            }
        
        score_values = [entry['score'] for entry in self.scores]
        
        # Group statistics
        by_algorithm = defaultdict(list)
        by_environment = defaultdict(list)
        by_user_type = defaultdict(list)
        
        for entry in self.scores:
            by_algorithm[entry['algorithm']].append(entry['score'])
            by_environment[entry['environment']].append(entry['score'])
            by_user_type[entry['user_type']].append(entry['score'])
        
        return {
            'count': len(score_values),
            'average': np.mean(score_values),
            'std': np.std(score_values),
            'min': np.min(score_values),
            'max': np.max(score_values),
            'by_algorithm': {k: np.mean(v) for k, v in by_algorithm.items()},
            'by_environment': {k: np.mean(v) for k, v in by_environment.items()},
            'by_user_type': {k: np.mean(v) for k, v in by_user_type.items()}
        }
    
    def reset(self):
        """Reset satisfaction score history."""
        self.scores.clear()


class UptimeMetric(BaseMetric):
    """Tracks system availability and uptime."""
    
    def __init__(self):
        super().__init__(
            "system_availability",
            "System availability percentage",
            ["service", "instance"]
        )
        self.start_time = datetime.now()
        self.downtime_events = []
        
        # Prometheus metrics
        self.uptime_gauge = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds',
            ['service', 'instance']
        )
        
        self.availability_gauge = Gauge(
            'system_availability_percentage',
            'System availability percentage',
            ['service', 'instance']
        )
    
    def record_downtime(self, service: str, instance: str, 
                       start_time: datetime, end_time: datetime):
        """Record a downtime event."""
        self.downtime_events.append({
            'service': service,
            'instance': instance,
            'start_time': start_time,
            'end_time': end_time,
            'duration': (end_time - start_time).total_seconds()
        })
        
        self._update_availability()
    
    def _update_availability(self):
        """Update availability calculations."""
        now = datetime.now()
        total_time = (now - self.start_time).total_seconds()
        
        if total_time <= 0:
            return
        
        # Group downtime by service and instance
        grouped_downtime = defaultdict(float)
        for event in self.downtime_events:
            key = (event['service'], event['instance'])
            grouped_downtime[key] += event['duration']
        
        # Calculate availability for each service instance
        for (service, instance), total_downtime in grouped_downtime.items():
            availability = ((total_time - total_downtime) / total_time) * 100
            availability = max(0.0, min(100.0, availability))  # Clamp between 0-100
            
            self.availability_gauge.labels(
                service=service,
                instance=instance
            ).set(availability)
            
            self.uptime_gauge.labels(
                service=service,
                instance=instance
            ).set(total_time - total_downtime)
    
    def collect(self) -> Dict[str, Any]:
        """Collect uptime and availability statistics."""
        now = datetime.now()
        total_uptime = (now - self.start_time).total_seconds()
        
        if not self.downtime_events:
            return {
                'uptime_seconds': total_uptime,
                'availability_percentage': 100.0,
                'downtime_events': 0,
                'total_downtime_seconds': 0.0,
                'by_service': {}
            }
        
        total_downtime = sum(event['duration'] for event in self.downtime_events)
        availability = ((total_uptime - total_downtime) / total_uptime) * 100 if total_uptime > 0 else 0.0
        
        # Group by service
        by_service = defaultdict(lambda: {'uptime': 0.0, 'downtime': 0.0, 'events': 0})
        for event in self.downtime_events:
            service = event['service']
            by_service[service]['downtime'] += event['duration']
            by_service[service]['events'] += 1
        
        # Calculate service-specific availability
        for service, stats in by_service.items():
            stats['uptime'] = total_uptime - stats['downtime']
            stats['availability'] = (stats['uptime'] / total_uptime) * 100 if total_uptime > 0 else 0.0
        
        return {
            'uptime_seconds': total_uptime,
            'availability_percentage': availability,
            'downtime_events': len(self.downtime_events),
            'total_downtime_seconds': total_downtime,
            'by_service': dict(by_service)
        }
    
    def reset(self):
        """Reset uptime tracking."""
        self.start_time = datetime.now()
        self.downtime_events.clear()


class HardwareStatusMetric(BaseMetric):
    """Tracks hardware health and resource usage."""
    
    def __init__(self, update_interval: int = 10):
        super().__init__(
            "hardware_health",
            "Hardware health status and resource usage",
            ["component", "instance"]
        )
        self.update_interval = update_interval
        self.last_update = 0
        self.current_stats = {}
        
        # Prometheus metrics
        self.cpu_usage = Gauge(
            'cpu_usage_percentage',
            'CPU usage percentage',
            ['instance']
        )
        
        self.memory_usage = Gauge(
            'memory_usage_percentage',
            'Memory usage percentage',
            ['instance']
        )
        
        self.disk_usage = Gauge(
            'disk_usage_percentage',
            'Disk usage percentage',
            ['instance', 'device']
        )
        
        self.gpu_usage = Gauge(
            'gpu_usage_percentage',
            'GPU usage percentage',
            ['instance', 'gpu_id']
        )
        
        self.gpu_memory = Gauge(
            'gpu_memory_usage_percentage',
            'GPU memory usage percentage',
            ['instance', 'gpu_id']
        )
        
        self.temperature = Gauge(
            'hardware_temperature_celsius',
            'Hardware temperature in Celsius',
            ['instance', 'sensor']
        )
    
    def _should_update(self) -> bool:
        """Check if it's time to update hardware stats."""
        now = time.time()
        if now - self.last_update >= self.update_interval:
            self.last_update = now
            return True
        return False
    
    def _collect_system_stats(self) -> Dict[str, Any]:
        """Collect system hardware statistics."""
        stats = {
            'timestamp': datetime.now(),
            'cpu': {},
            'memory': {},
            'disk': {},
            'gpu': [],
            'temperature': {}
        }
        
        # CPU stats
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            stats['cpu'] = {
                'usage_percent': cpu_percent,
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None
            }
            
            self.cpu_usage.labels(instance="main").set(cpu_percent)
        except Exception as e:
            logger.error(f"Failed to collect CPU stats: {e}")
        
        # Memory stats
        try:
            memory = psutil.virtual_memory()
            stats['memory'] = {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3)
            }
            
            self.memory_usage.labels(instance="main").set(memory.percent)
        except Exception as e:
            logger.error(f"Failed to collect memory stats: {e}")
        
        # Disk stats
        try:
            disk_stats = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    device_name = partition.device.replace('/', '_')
                    
                    disk_stats[partition.device] = {
                        'total_gb': usage.total / (1024**3),
                        'used_gb': usage.used / (1024**3),
                        'usage_percent': (usage.used / usage.total) * 100,
                        'free_gb': usage.free / (1024**3)
                    }
                    
                    self.disk_usage.labels(
                        instance="main",
                        device=device_name
                    ).set((usage.used / usage.total) * 100)
                except PermissionError:
                    continue
            
            stats['disk'] = disk_stats
        except Exception as e:
            logger.error(f"Failed to collect disk stats: {e}")
        
        # GPU stats
        try:
            gpus = GPUtil.getGPUs()
            gpu_stats = []
            for i, gpu in enumerate(gpus):
                gpu_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'usage_percent': gpu.load * 100,
                    'memory_usage_percent': gpu.memoryUtil * 100,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'temperature_c': gpu.temperature
                }
                gpu_stats.append(gpu_info)
                
                # Update Prometheus metrics
                self.gpu_usage.labels(
                    instance="main",
                    gpu_id=str(gpu.id)
                ).set(gpu.load * 100)
                
                self.gpu_memory.labels(
                    instance="main",
                    gpu_id=str(gpu.id)
                ).set(gpu.memoryUtil * 100)
                
                if gpu.temperature:
                    self.temperature.labels(
                        instance="main",
                        sensor=f"gpu_{gpu.id}"
                    ).set(gpu.temperature)
            
            stats['gpu'] = gpu_stats
        except Exception as e:
            logger.warning(f"Failed to collect GPU stats: {e}")
            stats['gpu'] = []
        
        # Temperature stats
        try:
            temperatures = psutil.sensors_temperatures()
            temp_stats = {}
            for name, entries in temperatures.items():
                for entry in entries:
                    sensor_name = f"{name}_{entry.label}" if entry.label else name
                    temp_stats[sensor_name] = {
                        'current_c': entry.current,
                        'high_c': entry.high,
                        'critical_c': entry.critical
                    }
                    
                    self.temperature.labels(
                        instance="main",
                        sensor=sensor_name
                    ).set(entry.current)
            
            stats['temperature'] = temp_stats
        except Exception as e:
            logger.warning(f"Failed to collect temperature stats: {e}")
        
        return stats
    
    def collect(self) -> Dict[str, Any]:
        """Collect current hardware statistics."""
        if self._should_update():
            self.current_stats = self._collect_system_stats()
        
        return self.current_stats
    
    def reset(self):
        """Reset hardware statistics."""
        self.current_stats = {}
        self.last_update = 0


class MetricsCollector:
    """Central metrics collection and management system."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.metrics = {}
        self.alert_rules = []
        self.redis_client = redis_client
        self.registry = CollectorRegistry()
        self.running = False
        self.collection_thread = None
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info("MetricsCollector initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default system metrics."""
        self.metrics = {
            'safety_violations_per_hour': SafetyViolationRate(),
            'policy_inference_latency_p99': LatencyMetric(percentile=99),
            'constraint_satisfaction_rate': ConstraintMetric(),
            'human_satisfaction_score': SatisfactionMetric(),
            'system_availability': UptimeMetric(),
            'hardware_health': HardwareStatusMetric()
        }
    
    def add_metric(self, name: str, metric: BaseMetric):
        """Add a custom metric."""
        self.metrics[name] = metric
        metric.registry = self.registry
        logger.info(f"Added metric: {name}")
    
    def remove_metric(self, name: str):
        """Remove a metric."""
        if name in self.metrics:
            del self.metrics[name]
            logger.info(f"Removed metric: {name}")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics."""
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.collect()
            except Exception as e:
                logger.error(f"Failed to collect metric {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """Get a specific metric."""
        return self.metrics.get(name)
    
    def start_collection(self, interval: int = 30):
        """Start background metrics collection."""
        if self.running:
            logger.warning("Metrics collection already running")
            return
        
        self.running = True
        
        def collection_loop():
            while self.running:
                try:
                    metrics_data = self.collect_all_metrics()
                    
                    # Store in Redis if available
                    if self.redis_client:
                        self._store_metrics_in_redis(metrics_data)
                    
                    # Check alert rules
                    self._check_alert_rules(metrics_data)
                    
                except Exception as e:
                    logger.error(f"Error in metrics collection loop: {e}")
                
                time.sleep(interval)
        
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection with {interval}s interval")
    
    def stop_collection(self):
        """Stop background metrics collection."""
        self.running = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        logger.info("Stopped metrics collection")
    
    def _store_metrics_in_redis(self, metrics_data: Dict[str, Any]):
        """Store metrics data in Redis."""
        try:
            timestamp = datetime.now().isoformat()
            key = f"metrics:{timestamp}"
            
            serialized_data = json.dumps(metrics_data, default=str)
            self.redis_client.setex(key, 3600, serialized_data)  # 1 hour TTL
            
            # Keep only last 1000 entries
            self.redis_client.zremrangebyrank("metrics_timeline", 0, -1000)
            self.redis_client.zadd("metrics_timeline", {key: time.time()})
            
        except Exception as e:
            logger.error(f"Failed to store metrics in Redis: {e}")
    
    def _check_alert_rules(self, metrics_data: Dict[str, Any]):
        """Check alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                metric_data = metrics_data.get(rule.metric_name)
                if metric_data is None:
                    continue
                
                # Extract the value to check
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    value = metric_data['value']
                elif isinstance(metric_data, (int, float)):
                    value = metric_data
                else:
                    continue  # Can't evaluate this metric
                
                # Evaluate condition
                if self._evaluate_condition(value, rule.condition):
                    self._trigger_alert(rule, value, metrics_data)
                    
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate alert condition."""
        try:
            # Simple condition evaluation (can be enhanced)
            if condition.startswith('>'):
                threshold = float(condition[1:].strip())
                return value > threshold
            elif condition.startswith('<'):
                threshold = float(condition[1:].strip())
                return value < threshold
            elif condition.startswith('>='):
                threshold = float(condition[2:].strip())
                return value >= threshold
            elif condition.startswith('<='):
                threshold = float(condition[2:].strip())
                return value <= threshold
            elif condition.startswith('=='):
                threshold = float(condition[2:].strip())
                return value == threshold
            else:
                logger.warning(f"Unknown condition format: {condition}")
                return False
        except Exception as e:
            logger.error(f"Failed to evaluate condition {condition}: {e}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, value: float, context: Dict[str, Any]):
        """Trigger an alert."""
        alert_data = {
            'rule_name': rule.name,
            'metric_name': rule.metric_name,
            'condition': rule.condition,
            'current_value': value,
            'severity': rule.severity,
            'message': rule.message or f"Alert triggered for {rule.metric_name}",
            'timestamp': datetime.now().isoformat(),
            'labels': rule.labels,
            'context': context
        }
        
        # Log the alert
        logger.warning(f"ALERT: {alert_data['message']} (value: {value})")
        
        # Store alert in Redis
        if self.redis_client:
            try:
                alert_key = f"alert:{rule.name}:{int(time.time())}"
                self.redis_client.setex(alert_key, 86400, json.dumps(alert_data, default=str))
                self.redis_client.lpush("alerts", alert_key)
                self.redis_client.ltrim("alerts", 0, 999)  # Keep last 1000 alerts
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")
    
    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def start_prometheus_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        try:
            start_http_server(port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
    
    def reset_all_metrics(self):
        """Reset all metrics."""
        for metric in self.metrics.values():
            try:
                metric.reset()
            except Exception as e:
                logger.error(f"Failed to reset metric: {e}")
        logger.info("All metrics reset")


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector

def initialize_metrics(redis_url: Optional[str] = None) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _global_collector
    
    redis_client = None
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
    
    _global_collector = MetricsCollector(redis_client)
    return _global_collector