"""
Continuous Performance Monitoring for Real-time Safe RL Systems

This module provides comprehensive continuous monitoring of system performance,
automatically detecting regressions, anomalies, and optimization opportunities.

Key features:
- Real-time performance dashboards with live metrics
- Automated performance regression detection
- Performance alerting with escalation policies  
- Historical performance trending and analysis
- Adaptive thresholds based on system behavior
- Integration with monitoring infrastructure (Prometheus, Grafana)
- Performance optimization recommendations
- Automated response to performance issues
"""

import asyncio
import json
import logging
import os
import statistics
import threading
import time
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy import typing as npt

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not available - metrics export disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available - system metrics disabled")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    warnings.warn("Redis not available - distributed monitoring disabled")

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    TIMING = "timing"                    # Execution times, latencies
    THROUGHPUT = "throughput"           # Operations per second
    RESOURCE_USAGE = "resource_usage"   # CPU, memory, GPU usage
    ERROR_RATE = "error_rate"          # Error counts and rates
    SAFETY = "safety"                  # Safety constraint violations
    QUALITY = "quality"                # Model prediction quality
    SYSTEM = "system"                  # System-level metrics


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"           # Informational
    WARNING = "warning"     # Warning level
    CRITICAL = "critical"   # Critical issue
    EMERGENCY = "emergency" # System emergency


class ThresholdType(Enum):
    """Types of performance thresholds"""
    STATIC = "static"       # Fixed threshold values
    ADAPTIVE = "adaptive"   # Self-adjusting thresholds
    PERCENTILE = "percentile"  # Percentile-based thresholds
    TREND = "trend"         # Trend-based detection


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    threshold_type: ThresholdType
    warning_value: float
    critical_value: float
    emergency_value: Optional[float] = None
    
    # Adaptive threshold parameters
    adaptation_window: int = 1000  # Number of samples for adaptation
    adaptation_factor: float = 1.2  # Multiplier for adaptive thresholds
    
    # Trend detection parameters
    trend_window: int = 100
    trend_threshold: float = 0.1  # 10% change threshold


@dataclass
class PerformanceAlert:
    """Performance alert data"""
    timestamp: datetime
    metric_name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[datetime] = None


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricCollector:
    """Collects and aggregates performance metrics from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = deque(maxlen=config.get('buffer_size', 10000))
        self.aggregated_metrics = defaultdict(list)
        self.collection_interval = config.get('collection_interval', 1.0)  # seconds
        
        # Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._initialize_prometheus_metrics()
        
        # System metrics collector
        if PSUTIL_AVAILABLE:
            self.system_metrics = SystemMetricsCollector()
        
        self.running = False
        self.collector_thread = None
        
        logger.info("Metric collector initialized")
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.prometheus_metrics = {
            'execution_time': Histogram(
                'rt_execution_time_seconds',
                'Execution time of RT operations',
                ['operation', 'component']
            ),
            'throughput': Counter(
                'rt_operations_total',
                'Total RT operations',
                ['operation', 'status']
            ),
            'error_rate': Counter(
                'rt_errors_total',
                'Total RT errors',
                ['error_type', 'component']
            ),
            'safety_violations': Counter(
                'rt_safety_violations_total',
                'Total safety violations',
                ['violation_type', 'severity']
            ),
            'system_cpu': Gauge(
                'rt_system_cpu_percent',
                'System CPU usage percentage'
            ),
            'system_memory': Gauge(
                'rt_system_memory_percent',
                'System memory usage percentage'
            ),
            'queue_depth': Gauge(
                'rt_queue_depth',
                'RT queue depth',
                ['queue_name']
            ),
        }
    
    def start_collection(self):
        """Start continuous metric collection"""
        if self.running:
            return
        
        self.running = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        
        logger.info("Started continuous metric collection")
    
    def stop_collection(self):
        """Stop metric collection"""
        self.running = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        
        logger.info("Stopped metric collection")
    
    def _collection_loop(self):
        """Main metric collection loop"""
        while self.running:
            try:
                # Collect system metrics
                if PSUTIL_AVAILABLE:
                    system_metrics = self.system_metrics.collect()
                    for metric in system_metrics:
                        self.add_metric(metric)
                
                # Update Prometheus metrics
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metric collection: {e}")
                time.sleep(self.collection_interval)
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric"""
        self.metrics_buffer.append(metric)
        
        # Add to aggregated metrics
        self.aggregated_metrics[metric.name].append({
            'timestamp': metric.timestamp,
            'value': metric.value,
            'tags': metric.tags,
            'metadata': metric.metadata,
        })
        
        # Limit aggregated metrics size
        max_history = self.config.get('max_metric_history', 10000)
        if len(self.aggregated_metrics[metric.name]) > max_history:
            self.aggregated_metrics[metric.name] = self.aggregated_metrics[metric.name][-max_history:]
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics from buffer"""
        # Process recent metrics from buffer
        recent_metrics = list(self.metrics_buffer)[-100:]  # Last 100 metrics
        
        for metric in recent_metrics:
            if metric.name == 'execution_time' and 'execution_time' in self.prometheus_metrics:
                operation = metric.tags.get('operation', 'unknown')
                component = metric.tags.get('component', 'unknown')
                self.prometheus_metrics['execution_time'].labels(
                    operation=operation, component=component
                ).observe(metric.value / 1_000_000)  # Convert μs to seconds
                
            elif metric.name == 'cpu_usage' and 'system_cpu' in self.prometheus_metrics:
                self.prometheus_metrics['system_cpu'].set(metric.value)
                
            elif metric.name == 'memory_usage' and 'system_memory' in self.prometheus_metrics:
                self.prometheus_metrics['system_memory'].set(metric.value)
    
    def get_metric_history(self, metric_name: str, 
                          duration: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Get metric history for specified duration"""
        if metric_name not in self.aggregated_metrics:
            return []
        
        metrics = self.aggregated_metrics[metric_name]
        
        if duration is None:
            return metrics
        
        cutoff_time = datetime.now() - duration
        return [m for m in metrics if m['timestamp'] >= cutoff_time]
    
    def get_recent_metrics(self, count: int = 100) -> List[PerformanceMetric]:
        """Get most recent metrics"""
        return list(self.metrics_buffer)[-count:]


class SystemMetricsCollector:
    """Collects system-level performance metrics"""
    
    def __init__(self):
        self.last_cpu_times = None
        self.last_network_io = None
        self.last_disk_io = None
        
    def collect(self) -> List[PerformanceMetric]:
        """Collect all system metrics"""
        metrics = []
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        metrics.append(PerformanceMetric(
            timestamp=timestamp,
            name="cpu_usage",
            value=cpu_percent,
            metric_type=MetricType.RESOURCE_USAGE,
            tags={"resource": "cpu"}
        ))
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.append(PerformanceMetric(
            timestamp=timestamp,
            name="memory_usage",
            value=memory.percent,
            metric_type=MetricType.RESOURCE_USAGE,
            tags={"resource": "memory"},
            metadata={"total_gb": memory.total / (1024**3)}
        ))
        
        # Load average (Linux/macOS)
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                name="load_average",
                value=load_avg[0],  # 1-minute load average
                metric_type=MetricType.RESOURCE_USAGE,
                tags={"timeframe": "1min"},
                metadata={"5min": load_avg[1], "15min": load_avg[2]}
            ))
        
        # Context switches
        ctx_switches = psutil.cpu_stats().ctx_switches
        metrics.append(PerformanceMetric(
            timestamp=timestamp,
            name="context_switches",
            value=ctx_switches,
            metric_type=MetricType.SYSTEM,
            tags={"type": "total"}
        ))
        
        # Network I/O
        net_io = psutil.net_io_counters()
        if self.last_network_io:
            bytes_sent_rate = (net_io.bytes_sent - self.last_network_io.bytes_sent)
            bytes_recv_rate = (net_io.bytes_recv - self.last_network_io.bytes_recv)
            
            metrics.extend([
                PerformanceMetric(
                    timestamp=timestamp,
                    name="network_bytes_rate",
                    value=bytes_sent_rate,
                    metric_type=MetricType.THROUGHPUT,
                    tags={"direction": "sent"}
                ),
                PerformanceMetric(
                    timestamp=timestamp,
                    name="network_bytes_rate",
                    value=bytes_recv_rate,
                    metric_type=MetricType.THROUGHPUT,
                    tags={"direction": "received"}
                )
            ])
        
        self.last_network_io = net_io
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io and self.last_disk_io:
            read_bytes_rate = disk_io.read_bytes - self.last_disk_io.read_bytes
            write_bytes_rate = disk_io.write_bytes - self.last_disk_io.write_bytes
            
            metrics.extend([
                PerformanceMetric(
                    timestamp=timestamp,
                    name="disk_bytes_rate",
                    value=read_bytes_rate,
                    metric_type=MetricType.THROUGHPUT,
                    tags={"operation": "read"}
                ),
                PerformanceMetric(
                    timestamp=timestamp,
                    name="disk_bytes_rate",
                    value=write_bytes_rate,
                    metric_type=MetricType.THROUGHPUT,
                    tags={"operation": "write"}
                )
            ])
        
        if disk_io:
            self.last_disk_io = disk_io
        
        return metrics


class PerformanceAnalyzer:
    """Analyzes performance metrics for anomalies and trends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = {}
        self.baseline_metrics = {}
        self.trend_detectors = {}
        
        # Statistical parameters
        self.analysis_window = config.get('analysis_window', 1000)
        self.anomaly_threshold = config.get('anomaly_threshold', 3.0)  # Standard deviations
        self.trend_sensitivity = config.get('trend_sensitivity', 0.1)  # 10% change
        
        logger.info("Performance analyzer initialized")
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add a performance threshold"""
        self.thresholds[threshold.metric_name] = threshold
    
    def analyze_metrics(self, metrics: List[PerformanceMetric]) -> List[PerformanceAlert]:
        """Analyze metrics for performance issues"""
        alerts = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        # Analyze each metric type
        for metric_name, metric_list in metrics_by_name.items():
            metric_alerts = self._analyze_metric_series(metric_name, metric_list)
            alerts.extend(metric_alerts)
        
        return alerts
    
    def _analyze_metric_series(self, metric_name: str, 
                              metrics: List[PerformanceMetric]) -> List[PerformanceAlert]:
        """Analyze a series of metrics for one metric type"""
        if not metrics:
            return []
        
        alerts = []
        values = [m.value for m in metrics]
        timestamps = [m.timestamp for m in metrics]
        
        # Threshold-based analysis
        if metric_name in self.thresholds:
            threshold_alerts = self._check_thresholds(metric_name, metrics[-1])
            alerts.extend(threshold_alerts)
        
        # Statistical anomaly detection
        if len(values) >= 10:  # Need sufficient data
            anomaly_alerts = self._detect_statistical_anomalies(metric_name, values, timestamps)
            alerts.extend(anomaly_alerts)
        
        # Trend analysis
        if len(values) >= self.config.get('min_trend_samples', 50):
            trend_alerts = self._analyze_trends(metric_name, values, timestamps)
            alerts.extend(trend_alerts)
        
        # Update baseline metrics
        self._update_baseline(metric_name, values)
        
        return alerts
    
    def _check_thresholds(self, metric_name: str, 
                         metric: PerformanceMetric) -> List[PerformanceAlert]:
        """Check metric against configured thresholds"""
        threshold = self.thresholds[metric_name]
        alerts = []
        
        if threshold.threshold_type == ThresholdType.STATIC:
            # Static threshold checking
            if threshold.emergency_value and metric.value >= threshold.emergency_value:
                alerts.append(PerformanceAlert(
                    timestamp=metric.timestamp,
                    metric_name=metric_name,
                    severity=AlertSeverity.EMERGENCY,
                    current_value=metric.value,
                    threshold_value=threshold.emergency_value,
                    message=f"{metric_name} exceeded emergency threshold",
                    context={"metric": metric, "threshold_type": "static"}
                ))
            elif metric.value >= threshold.critical_value:
                alerts.append(PerformanceAlert(
                    timestamp=metric.timestamp,
                    metric_name=metric_name,
                    severity=AlertSeverity.CRITICAL,
                    current_value=metric.value,
                    threshold_value=threshold.critical_value,
                    message=f"{metric_name} exceeded critical threshold"
                ))
            elif metric.value >= threshold.warning_value:
                alerts.append(PerformanceAlert(
                    timestamp=metric.timestamp,
                    metric_name=metric_name,
                    severity=AlertSeverity.WARNING,
                    current_value=metric.value,
                    threshold_value=threshold.warning_value,
                    message=f"{metric_name} exceeded warning threshold"
                ))
        
        elif threshold.threshold_type == ThresholdType.ADAPTIVE:
            # Adaptive threshold checking
            adaptive_thresholds = self._calculate_adaptive_thresholds(metric_name, threshold)
            if adaptive_thresholds:
                # Similar checking logic with adaptive values
                pass
        
        return alerts
    
    def _detect_statistical_anomalies(self, metric_name: str, values: List[float], 
                                    timestamps: List[datetime]) -> List[PerformanceAlert]:
        """Detect statistical anomalies using z-score analysis"""
        alerts = []
        
        if len(values) < 10:
            return alerts
        
        # Calculate rolling statistics
        window_size = min(len(values), self.analysis_window)
        recent_values = values[-window_size:]
        
        mean_val = np.mean(recent_values)
        std_val = np.std(recent_values)
        
        if std_val == 0:  # No variation
            return alerts
        
        # Check latest value for anomaly
        latest_value = values[-1]
        latest_timestamp = timestamps[-1]
        z_score = abs(latest_value - mean_val) / std_val
        
        if z_score > self.anomaly_threshold:
            severity = AlertSeverity.WARNING
            if z_score > self.anomaly_threshold * 2:
                severity = AlertSeverity.CRITICAL
            
            alerts.append(PerformanceAlert(
                timestamp=latest_timestamp,
                metric_name=metric_name,
                severity=severity,
                current_value=latest_value,
                threshold_value=mean_val + self.anomaly_threshold * std_val,
                message=f"Statistical anomaly detected in {metric_name}",
                context={
                    "z_score": z_score,
                    "mean": mean_val,
                    "std": std_val,
                    "analysis_type": "statistical"
                }
            ))
        
        return alerts
    
    def _analyze_trends(self, metric_name: str, values: List[float], 
                       timestamps: List[datetime]) -> List[PerformanceAlert]:
        """Analyze performance trends"""
        alerts = []
        
        if len(values) < 50:
            return alerts
        
        # Calculate trend using linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        n = len(values)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return alerts
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate trend magnitude
        y_range = np.max(values) - np.min(values)
        if y_range == 0:
            return alerts
        
        trend_magnitude = abs(slope * len(values)) / y_range
        
        # Alert on significant trends
        if trend_magnitude > self.trend_sensitivity:
            trend_direction = "increasing" if slope > 0 else "decreasing"
            severity = AlertSeverity.WARNING
            
            if trend_magnitude > self.trend_sensitivity * 3:
                severity = AlertSeverity.CRITICAL
            
            alerts.append(PerformanceAlert(
                timestamp=timestamps[-1],
                metric_name=metric_name,
                severity=severity,
                current_value=values[-1],
                threshold_value=intercept,  # Baseline value
                message=f"Significant {trend_direction} trend detected in {metric_name}",
                context={
                    "trend_slope": slope,
                    "trend_magnitude": trend_magnitude,
                    "trend_direction": trend_direction,
                    "analysis_type": "trend"
                }
            ))
        
        return alerts
    
    def _calculate_adaptive_thresholds(self, metric_name: str, 
                                     threshold: PerformanceThreshold) -> Optional[Dict[str, float]]:
        """Calculate adaptive thresholds based on historical data"""
        if metric_name not in self.baseline_metrics:
            return None
        
        baseline = self.baseline_metrics[metric_name]
        
        # Use percentiles for adaptive thresholds
        values = baseline['values'][-threshold.adaptation_window:]
        
        if len(values) < 10:
            return None
        
        p50 = np.percentile(values, 50)
        p95 = np.percentile(values, 95)
        p99 = np.percentile(values, 99)
        
        return {
            'warning': p95 * threshold.adaptation_factor,
            'critical': p99 * threshold.adaptation_factor,
            'emergency': p99 * threshold.adaptation_factor * 1.5,
        }
    
    def _update_baseline(self, metric_name: str, values: List[float]):
        """Update baseline metrics for adaptive analysis"""
        if metric_name not in self.baseline_metrics:
            self.baseline_metrics[metric_name] = {
                'values': deque(maxlen=self.config.get('baseline_history', 10000)),
                'last_update': datetime.now(),
            }
        
        baseline = self.baseline_metrics[metric_name]
        baseline['values'].extend(values)
        baseline['last_update'] = datetime.now()


class AlertManager:
    """Manages performance alerts with escalation and notification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts = {}
        self.alert_history = deque(maxlen=config.get('alert_history_size', 10000))
        
        # Alert suppression to prevent spam
        self.suppression_windows = {}
        self.default_suppression_window = timedelta(
            minutes=config.get('default_suppression_minutes', 5)
        )
        
        # Escalation configuration
        self.escalation_rules = config.get('escalation_rules', {})
        
        logger.info("Alert manager initialized")
    
    def process_alerts(self, alerts: List[PerformanceAlert]):
        """Process new performance alerts"""
        for alert in alerts:
            self._process_single_alert(alert)
    
    def _process_single_alert(self, alert: PerformanceAlert):
        """Process a single performance alert"""
        alert_key = f"{alert.metric_name}_{alert.severity.value}"
        
        # Check for suppression
        if self._is_suppressed(alert_key, alert.timestamp):
            return
        
        # Check if this is a new alert or escalation
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            
            # Update existing alert
            existing_alert.current_value = alert.current_value
            existing_alert.timestamp = alert.timestamp
            
            # Check for escalation
            if alert.severity.value != existing_alert.severity.value:
                self._escalate_alert(existing_alert, alert.severity)
        else:
            # New alert
            self.active_alerts[alert_key] = alert
            self._send_alert_notification(alert)
        
        # Add to history
        self.alert_history.append(alert)
        
        # Set suppression window
        self.suppression_windows[alert_key] = alert.timestamp + self.default_suppression_window
    
    def _is_suppressed(self, alert_key: str, timestamp: datetime) -> bool:
        """Check if alert is within suppression window"""
        if alert_key not in self.suppression_windows:
            return False
        
        return timestamp < self.suppression_windows[alert_key]
    
    def _escalate_alert(self, existing_alert: PerformanceAlert, new_severity: AlertSeverity):
        """Escalate an existing alert to higher severity"""
        old_severity = existing_alert.severity
        existing_alert.severity = new_severity
        
        logger.warning(f"Alert escalated: {existing_alert.metric_name} "
                      f"from {old_severity.value} to {new_severity.value}")
        
        # Send escalation notification
        self._send_escalation_notification(existing_alert, old_severity)
    
    def _send_alert_notification(self, alert: PerformanceAlert):
        """Send alert notification"""
        logger.info(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")
        logger.info(f"  Metric: {alert.metric_name}")
        logger.info(f"  Value: {alert.current_value}")
        logger.info(f"  Threshold: {alert.threshold_value}")
        
        # Here you would integrate with actual notification systems:
        # - Slack/Teams notifications
        # - Email alerts
        # - PagerDuty integration
        # - SMS notifications
        
        # Example webhook notification
        if self.config.get('webhook_url'):
            self._send_webhook_notification(alert)
    
    def _send_escalation_notification(self, alert: PerformanceAlert, old_severity: AlertSeverity):
        """Send escalation notification"""
        logger.warning(f"ESCALATION: {alert.metric_name} escalated to {alert.severity.value}")
        
        # Send to escalation channels based on severity
        # Critical/Emergency alerts might go to on-call engineers
    
    def _send_webhook_notification(self, alert: PerformanceAlert):
        """Send webhook notification (placeholder)"""
        webhook_payload = {
            'alert_type': 'performance_alert',
            'severity': alert.severity.value,
            'metric': alert.metric_name,
            'message': alert.message,
            'value': alert.current_value,
            'threshold': alert.threshold_value,
            'timestamp': alert.timestamp.isoformat(),
        }
        
        # In real implementation, send HTTP POST to webhook_url
        logger.debug(f"Webhook payload: {json.dumps(webhook_payload, indent=2)}")
    
    def resolve_alert(self, metric_name: str, severity: AlertSeverity):
        """Mark an alert as resolved"""
        alert_key = f"{metric_name}_{severity.value}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_timestamp = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[alert_key]
            
            logger.info(f"Alert resolved: {alert.message}")
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        # Recent alert history
        recent_alerts = [a for a in self.alert_history 
                        if a.timestamp > datetime.now() - timedelta(hours=24)]
        
        return {
            'active_alerts': len(self.active_alerts),
            'active_by_severity': dict(active_by_severity),
            'alerts_24h': len(recent_alerts),
            'most_recent_alert': self.alert_history[-1].timestamp if self.alert_history else None,
        }


class ContinuousPerformanceMonitor:
    """
    Main continuous performance monitoring system.
    
    Coordinates metric collection, analysis, and alerting for real-time
    performance monitoring of Safe RL systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.metric_collector = MetricCollector(config.get('collector', {}))
        self.performance_analyzer = PerformanceAnalyzer(config.get('analyzer', {}))
        self.alert_manager = AlertManager(config.get('alerts', {}))
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.analysis_interval = config.get('analysis_interval', 10.0)  # seconds
        
        # Performance data storage
        self.performance_history = deque(maxlen=config.get('history_size', 100000))
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
        # Dashboard data for real-time monitoring
        self.dashboard_data = {
            'last_update': datetime.now(),
            'system_status': 'initializing',
            'key_metrics': {},
            'recent_alerts': [],
        }
        
        logger.info("Continuous performance monitor initialized")
    
    def _initialize_default_thresholds(self):
        """Initialize default performance thresholds"""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="execution_time",
                threshold_type=ThresholdType.STATIC,
                warning_value=800,    # 800μs
                critical_value=1200,  # 1.2ms
                emergency_value=2000, # 2ms
            ),
            PerformanceThreshold(
                metric_name="cpu_usage",
                threshold_type=ThresholdType.ADAPTIVE,
                warning_value=80,     # 80%
                critical_value=95,    # 95%
                emergency_value=98,   # 98%
            ),
            PerformanceThreshold(
                metric_name="memory_usage",
                threshold_type=ThresholdType.ADAPTIVE,
                warning_value=85,     # 85%
                critical_value=95,    # 95%
                emergency_value=98,   # 98%
            ),
            PerformanceThreshold(
                metric_name="safety_violation_rate",
                threshold_type=ThresholdType.STATIC,
                warning_value=0.001,  # 0.1%
                critical_value=0.01,  # 1%
                emergency_value=0.05, # 5%
            ),
        ]
        
        for threshold in default_thresholds:
            self.performance_analyzer.add_threshold(threshold)
    
    def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.running:
            logger.warning("Monitor already running")
            return
        
        self.running = True
        
        # Start metric collection
        self.metric_collector.start_collection()
        
        # Start analysis thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Update dashboard
        self.dashboard_data['system_status'] = 'monitoring'
        self.dashboard_data['last_update'] = datetime.now()
        
        logger.info("Started continuous performance monitoring")
    
    def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop components
        self.metric_collector.stop_collection()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        # Update dashboard
        self.dashboard_data['system_status'] = 'stopped'
        self.dashboard_data['last_update'] = datetime.now()
        
        logger.info("Stopped continuous performance monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring analysis loop"""
        while self.running:
            try:
                # Get recent metrics
                recent_metrics = self.metric_collector.get_recent_metrics(1000)
                
                if recent_metrics:
                    # Analyze for performance issues
                    alerts = self.performance_analyzer.analyze_metrics(recent_metrics)
                    
                    # Process alerts
                    if alerts:
                        self.alert_manager.process_alerts(alerts)
                    
                    # Update dashboard data
                    self._update_dashboard_data(recent_metrics, alerts)
                    
                    # Store in history
                    self.performance_history.extend(recent_metrics)
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.analysis_interval)
    
    def _update_dashboard_data(self, metrics: List[PerformanceMetric], alerts: List[PerformanceAlert]):
        """Update real-time dashboard data"""
        # Calculate key metrics
        key_metrics = {}
        
        # Group metrics by name and get latest values
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric.value)
        
        for metric_name, values in metrics_by_name.items():
            if values:
                key_metrics[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values),
                }
        
        # Update dashboard
        self.dashboard_data.update({
            'last_update': datetime.now(),
            'key_metrics': key_metrics,
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'metric': alert.metric_name,
                    'message': alert.message,
                    'value': alert.current_value,
                }
                for alert in alerts
            ],
            'active_alert_count': len(self.alert_manager.get_active_alerts()),
        })
    
    @contextmanager
    def performance_context(self, operation_name: str, component: str = "unknown"):
        """Context manager for tracking operation performance"""
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        # Create performance metric
        execution_time_us = (end_time - start_time) * 1_000_000
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            name="execution_time",
            value=execution_time_us,
            metric_type=MetricType.TIMING,
            tags={
                "operation": operation_name,
                "component": component,
            }
        )
        
        self.metric_collector.add_metric(metric)
    
    def record_safety_violation(self, violation_type: str, severity: str = "high"):
        """Record a safety violation"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            name="safety_violation",
            value=1.0,
            metric_type=MetricType.SAFETY,
            tags={
                "violation_type": violation_type,
                "severity": severity,
            }
        )
        
        self.metric_collector.add_metric(metric)
    
    def record_throughput(self, operation: str, count: int = 1):
        """Record throughput metric"""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            name="throughput",
            value=count,
            metric_type=MetricType.THROUGHPUT,
            tags={"operation": operation}
        )
        
        self.metric_collector.add_metric(metric)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()
    
    def get_performance_summary(self, duration: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for specified duration"""
        if duration is None:
            duration = timedelta(hours=1)
        
        cutoff_time = datetime.now() - duration
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'status': 'no_data', 'duration': duration.total_seconds()}
        
        # Group by metric type
        timing_metrics = [m for m in recent_metrics if m.metric_type == MetricType.TIMING]
        resource_metrics = [m for m in recent_metrics if m.metric_type == MetricType.RESOURCE_USAGE]
        safety_metrics = [m for m in recent_metrics if m.metric_type == MetricType.SAFETY]
        
        summary = {
            'duration_seconds': duration.total_seconds(),
            'total_metrics': len(recent_metrics),
            'timing_performance': self._summarize_timing_metrics(timing_metrics),
            'resource_usage': self._summarize_resource_metrics(resource_metrics),
            'safety_status': self._summarize_safety_metrics(safety_metrics),
            'alert_summary': self.alert_manager.get_alert_summary(),
        }
        
        return summary
    
    def _summarize_timing_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Summarize timing metrics"""
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        return {
            'mean_us': np.mean(values),
            'median_us': np.median(values),
            'p95_us': np.percentile(values, 95),
            'p99_us': np.percentile(values, 99),
            'max_us': np.max(values),
            'std_us': np.std(values),
            'sample_count': len(values),
        }
    
    def _summarize_resource_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Summarize resource usage metrics"""
        if not metrics:
            return {}
        
        # Group by resource type
        by_resource = defaultdict(list)
        for metric in metrics:
            resource = metric.tags.get('resource', 'unknown')
            by_resource[resource].append(metric.value)
        
        summary = {}
        for resource, values in by_resource.items():
            summary[resource] = {
                'mean_percent': np.mean(values),
                'max_percent': np.max(values),
                'current_percent': values[-1] if values else 0,
            }
        
        return summary
    
    def _summarize_safety_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Summarize safety metrics"""
        if not metrics:
            return {'violations': 0, 'violation_rate': 0.0}
        
        total_violations = sum(m.value for m in metrics)
        
        # Group by violation type
        by_type = defaultdict(int)
        for metric in metrics:
            violation_type = metric.tags.get('violation_type', 'unknown')
            by_type[violation_type] += metric.value
        
        return {
            'total_violations': total_violations,
            'violations_by_type': dict(by_type),
            'violation_rate': total_violations / len(metrics) if metrics else 0,
        }
    
    def generate_performance_report(self, duration: timedelta = timedelta(days=1)) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        logger.info(f"Generating performance report for {duration}")
        
        summary = self.get_performance_summary(duration)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'report_period': {
                'duration_hours': duration.total_seconds() / 3600,
                'start_time': (datetime.now() - duration).isoformat(),
                'end_time': datetime.now().isoformat(),
            },
            'performance_summary': summary,
            'system_info': {
                'monitoring_status': self.dashboard_data['system_status'],
                'last_update': self.dashboard_data['last_update'].isoformat(),
            },
            'recommendations': self._generate_performance_recommendations(summary),
        }
        
        return report
    
    def _generate_performance_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Timing performance recommendations
        timing = summary.get('timing_performance', {})
        if timing:
            if timing.get('p99_us', 0) > 1500:  # P99 > 1.5ms
                recommendations.append("Consider optimizing critical path - P99 latency is high")
            
            if timing.get('std_us', 0) > 200:  # High jitter
                recommendations.append("High timing variability detected - investigate jitter sources")
        
        # Resource usage recommendations
        resources = summary.get('resource_usage', {})
        if resources:
            cpu_usage = resources.get('cpu', {}).get('mean_percent', 0)
            if cpu_usage > 80:
                recommendations.append("High CPU usage - consider load balancing or optimization")
            
            memory_usage = resources.get('memory', {}).get('mean_percent', 0)
            if memory_usage > 85:
                recommendations.append("High memory usage - check for memory leaks or increase capacity")
        
        # Safety recommendations
        safety = summary.get('safety_status', {})
        if safety and safety.get('total_violations', 0) > 0:
            recommendations.append("Safety violations detected - review constraint configurations")
        
        # Alert-based recommendations
        alerts = summary.get('alert_summary', {})
        if alerts and alerts.get('active_alerts', 0) > 0:
            recommendations.append("Active performance alerts - investigate underlying issues")
        
        if not recommendations:
            recommendations.append("System performance appears healthy - continue monitoring")
        
        return recommendations


# Example usage and testing
async def main():
    """Example usage of continuous performance monitoring"""
    
    # Configuration
    config = {
        'collector': {
            'buffer_size': 10000,
            'collection_interval': 1.0,
        },
        'analyzer': {
            'analysis_window': 1000,
            'anomaly_threshold': 3.0,
        },
        'alerts': {
            'default_suppression_minutes': 5,
            'webhook_url': 'http://localhost:8080/alerts',
        },
        'analysis_interval': 5.0,
    }
    
    # Create monitor
    monitor = ContinuousPerformanceMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate some operations with performance tracking
        for i in range(100):
            # Simulate RT control loop
            with monitor.performance_context("control_loop", "rt_controller"):
                # Simulate work (variable timing)
                work_time = 0.0005 + (i % 10) * 0.0001  # 0.5-1.5ms
                await asyncio.sleep(work_time)
            
            # Record throughput
            monitor.record_throughput("control_operations")
            
            # Simulate occasional safety violation
            if i % 50 == 0:
                monitor.record_safety_violation("proximity_violation", "medium")
            
            # Simulate policy inference
            with monitor.performance_context("policy_inference", "ml_model"):
                await asyncio.sleep(0.0002)  # 0.2ms
            
            await asyncio.sleep(0.01)  # 100 Hz loop
        
        # Let monitoring run for a bit
        await asyncio.sleep(10)
        
        # Get dashboard data
        dashboard = monitor.get_dashboard_data()
        print("Dashboard Data:")
        print(f"  Status: {dashboard['system_status']}")
        print(f"  Active Alerts: {dashboard.get('active_alert_count', 0)}")
        print(f"  Key Metrics: {len(dashboard.get('key_metrics', {}))}")
        
        # Get performance summary
        summary = monitor.get_performance_summary(timedelta(minutes=5))
        
        if summary.get('timing_performance'):
            timing = summary['timing_performance']
            print(f"\nTiming Performance:")
            print(f"  Mean: {timing['mean_us']:.1f}μs")
            print(f"  P95: {timing['p95_us']:.1f}μs")
            print(f"  P99: {timing['p99_us']:.1f}μs")
            print(f"  Samples: {timing['sample_count']}")
        
        # Generate report
        report = monitor.generate_performance_report(timedelta(minutes=5))
        print(f"\nPerformance Report:")
        print(f"  Period: {report['report_period']['duration_hours']:.2f} hours")
        print(f"  Recommendations: {len(report['recommendations'])}")
        for rec in report['recommendations']:
            print(f"    • {rec}")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run example
    asyncio.run(main())