"""
Real-time Safety Monitor for CPO Human-Robot Shared Control.

This module implements comprehensive safety monitoring including:
- Real-time constraint violation detection
- Emergency stop mechanisms
- Safety metric computation and logging
- Constraint buffer management
- Predictive safety assessment
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import threading
from queue import Queue, Empty
from enum import Enum
from collections import deque, defaultdict
import json
import warnings

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety level classification."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of constraint violations."""
    COLLISION = "collision"
    FORCE_LIMIT = "force_limit"
    JOINT_LIMIT = "joint_limit"
    VELOCITY_LIMIT = "velocity_limit"
    ACCELERATION_LIMIT = "acceleration_limit"
    WORKSPACE_BOUNDARY = "workspace_boundary"
    HUMAN_COMFORT = "human_comfort"
    SYSTEM_FAILURE = "system_failure"


@dataclass
class SafetyEvent:
    """Safety event record."""
    timestamp: float
    event_type: ViolationType
    severity: SafetyLevel
    violation_value: float
    threshold: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    action_taken: Optional[str] = None


@dataclass
class SafetyMetrics:
    """Safety performance metrics."""
    total_timesteps: int = 0
    total_violations: int = 0
    violations_by_type: Dict[ViolationType, int] = field(default_factory=lambda: defaultdict(int))
    violations_by_severity: Dict[SafetyLevel, int] = field(default_factory=lambda: defaultdict(int))
    mean_time_to_violation: float = 0.0
    mean_violation_magnitude: float = 0.0
    safety_rate: float = 1.0
    consecutive_safe_steps: int = 0
    max_consecutive_violations: int = 0
    emergency_stops: int = 0
    false_alarms: int = 0


class ConstraintBuffer:
    """
    Buffer for managing constraint violation history and trends.
    
    Maintains sliding window of constraint values for trend analysis
    and predictive safety assessment.
    """
    
    def __init__(self, 
                 buffer_size: int = 1000,
                 constraint_names: List[str] = None):
        """
        Initialize constraint buffer.
        
        Args:
            buffer_size: Maximum number of timesteps to store
            constraint_names: Names of constraints to track
        """
        self.buffer_size = buffer_size
        self.constraint_names = constraint_names or []
        
        # Circular buffers for each constraint
        self.constraint_values = {
            name: deque(maxlen=buffer_size) 
            for name in self.constraint_names
        }
        self.timestamps = deque(maxlen=buffer_size)
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        
        # Statistics
        self.current_size = 0
        self.total_added = 0
        
    def add(self, 
            timestamp: float,
            constraint_dict: Dict[str, float],
            state: Optional[torch.Tensor] = None,
            action: Optional[torch.Tensor] = None) -> None:
        """
        Add constraint values to buffer.
        
        Args:
            timestamp: Current timestamp
            constraint_dict: Dictionary mapping constraint names to values
            state: Current state (optional)
            action: Current action (optional)
        """
        self.timestamps.append(timestamp)
        
        # Add constraint values
        for name in self.constraint_names:
            value = constraint_dict.get(name, 0.0)
            self.constraint_values[name].append(value)
        
        # Add state and action if provided
        if state is not None:
            self.states.append(state.clone().detach() if isinstance(state, torch.Tensor) else state)
        if action is not None:
            self.actions.append(action.clone().detach() if isinstance(action, torch.Tensor) else action)
        
        self.current_size = min(self.current_size + 1, self.buffer_size)
        self.total_added += 1
    
    def get_recent_values(self, 
                         constraint_name: str, 
                         window_size: int = 10) -> List[float]:
        """Get recent constraint values."""
        if constraint_name not in self.constraint_values:
            return []
        
        values = list(self.constraint_values[constraint_name])
        return values[-window_size:] if values else []
    
    def compute_trend(self, 
                     constraint_name: str, 
                     window_size: int = 10) -> Dict[str, float]:
        """
        Compute constraint value trend over recent window.
        
        Returns:
            Dictionary with trend statistics
        """
        recent_values = self.get_recent_values(constraint_name, window_size)
        
        if len(recent_values) < 2:
            return {"trend": 0.0, "volatility": 0.0, "direction": "stable"}
        
        # Linear trend
        x = np.arange(len(recent_values))
        coeffs = np.polyfit(x, recent_values, 1)
        trend = coeffs[0]  # Slope
        
        # Volatility (standard deviation)
        volatility = np.std(recent_values)
        
        # Direction classification
        if abs(trend) < volatility * 0.1:
            direction = "stable"
        elif trend > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "trend": trend,
            "volatility": volatility,
            "direction": direction,
            "current_value": recent_values[-1],
            "min_value": min(recent_values),
            "max_value": max(recent_values)
        }
    
    def predict_violation_risk(self, 
                              constraint_name: str,
                              threshold: float,
                              horizon: int = 5) -> Dict[str, Any]:
        """
        Predict risk of constraint violation within horizon.
        
        Args:
            constraint_name: Name of constraint to analyze
            threshold: Violation threshold
            horizon: Prediction horizon (timesteps)
            
        Returns:
            Dictionary with risk assessment
        """
        trend_info = self.compute_trend(constraint_name)
        
        if trend_info["direction"] == "stable":
            # Low risk if stable and not near threshold
            current_value = trend_info["current_value"]
            risk_score = max(0.0, (current_value - threshold + 0.1) / 0.1) if current_value > threshold - 0.1 else 0.0
            predicted_violation_time = float('inf')
        else:
            # Predict based on trend
            trend_rate = trend_info["trend"]
            current_value = trend_info["current_value"]
            
            if trend_rate > 0 and current_value < threshold:
                # Approaching threshold
                time_to_threshold = (threshold - current_value) / trend_rate
                predicted_violation_time = time_to_threshold if time_to_threshold > 0 else 0
                risk_score = max(0.0, min(1.0, 1.0 - time_to_threshold / horizon))
            elif current_value >= threshold:
                # Already violating
                predicted_violation_time = 0
                risk_score = 1.0
            else:
                # Moving away from threshold
                predicted_violation_time = float('inf')
                risk_score = 0.0
        
        return {
            "risk_score": risk_score,
            "predicted_violation_time": predicted_violation_time,
            "current_trend": trend_info,
            "risk_level": self._classify_risk_level(risk_score)
        }
    
    def _classify_risk_level(self, risk_score: float) -> SafetyLevel:
        """Classify risk level based on risk score."""
        if risk_score < 0.1:
            return SafetyLevel.SAFE
        elif risk_score < 0.3:
            return SafetyLevel.CAUTION
        elif risk_score < 0.6:
            return SafetyLevel.WARNING
        elif risk_score < 0.9:
            return SafetyLevel.DANGER
        else:
            return SafetyLevel.CRITICAL
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        stats = {
            "current_size": self.current_size,
            "total_added": self.total_added,
            "buffer_utilization": self.current_size / self.buffer_size
        }
        
        # Per-constraint statistics
        for name in self.constraint_names:
            values = list(self.constraint_values[name])
            if values:
                stats[f"{name}_mean"] = np.mean(values)
                stats[f"{name}_std"] = np.std(values)
                stats[f"{name}_min"] = np.min(values)
                stats[f"{name}_max"] = np.max(values)
        
        return stats


class EmergencyStopController:
    """
    Emergency stop controller for critical safety situations.
    
    Implements multiple layers of emergency stopping based on
    constraint violations and system health.
    """
    
    def __init__(self,
                 emergency_thresholds: Dict[ViolationType, float] = None,
                 enable_predictive_stop: bool = True,
                 stop_response_time: float = 0.1):
        """
        Initialize emergency stop controller.
        
        Args:
            emergency_thresholds: Thresholds triggering emergency stops
            enable_predictive_stop: Enable predictive emergency stopping
            stop_response_time: Maximum response time for emergency stop (seconds)
        """
        self.emergency_thresholds = emergency_thresholds or {
            ViolationType.COLLISION: 0.05,  # 5cm collision threshold
            ViolationType.FORCE_LIMIT: 100.0,  # 100N force limit
            ViolationType.JOINT_LIMIT: 0.1,   # 0.1 rad beyond joint limit
            ViolationType.VELOCITY_LIMIT: 5.0  # 5 rad/s velocity limit
        }
        self.enable_predictive_stop = enable_predictive_stop
        self.stop_response_time = stop_response_time
        
        # Emergency stop state
        self.emergency_active = False
        self.stop_reason = ""
        self.stop_timestamp = 0.0
        self.stop_count = 0
        
        # Recovery state
        self.recovery_time = 5.0  # Seconds to wait before recovery
        self.recovery_checks = []  # List of functions to check before recovery
        
        logger.info("EmergencyStopController initialized")
        for violation_type, threshold in self.emergency_thresholds.items():
            logger.info(f"  {violation_type.value}: {threshold}")
    
    def check_emergency_conditions(self,
                                 constraint_dict: Dict[str, float],
                                 violation_history: List[SafetyEvent],
                                 risk_predictions: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Check if emergency stop should be triggered.
        
        Args:
            constraint_dict: Current constraint values
            violation_history: Recent violation history
            risk_predictions: Risk predictions for each constraint
            
        Returns:
            Tuple of (should_stop, reason)
        """
        # Check immediate critical violations
        for constraint_name, value in constraint_dict.items():
            # Map constraint name to violation type
            violation_type = self._map_constraint_to_violation_type(constraint_name)
            
            if violation_type in self.emergency_thresholds:
                threshold = self.emergency_thresholds[violation_type]
                
                if value > threshold:
                    return True, f"Critical {violation_type.value} violation: {value:.6f} > {threshold:.6f}"
        
        # Check predictive conditions
        if self.enable_predictive_stop:
            for constraint_name, risk_info in risk_predictions.items():
                if risk_info["risk_level"] == SafetyLevel.CRITICAL:
                    predicted_time = risk_info["predicted_violation_time"]
                    if predicted_time < self.stop_response_time * 2:  # Double response time margin
                        return True, f"Predicted critical violation in {constraint_name}: {predicted_time:.3f}s"
        
        # Check consecutive violations
        recent_violations = [event for event in violation_history[-10:] if not event.resolved]
        if len(recent_violations) >= 3:
            return True, f"Multiple unresolved violations: {len(recent_violations)}"
        
        # Check pattern of escalating violations
        if len(violation_history) >= 5:
            recent_severities = [event.severity for event in violation_history[-5:]]
            severity_trend = self._analyze_severity_trend(recent_severities)
            if severity_trend == "escalating":
                return True, "Escalating violation severity pattern detected"
        
        return False, ""
    
    def trigger_emergency_stop(self, reason: str) -> None:
        """
        Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        if not self.emergency_active:
            self.emergency_active = True
            self.stop_reason = reason
            self.stop_timestamp = time.time()
            self.stop_count += 1
            
            logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
            logger.critical(f"Stop count: {self.stop_count}")
            
            # Notify any registered callbacks
            self._notify_emergency_stop(reason)
    
    def check_recovery_conditions(self) -> Tuple[bool, str]:
        """
        Check if system can recover from emergency stop.
        
        Returns:
            Tuple of (can_recover, reason)
        """
        if not self.emergency_active:
            return True, "No emergency stop active"
        
        # Check minimum recovery time
        elapsed_time = time.time() - self.stop_timestamp
        if elapsed_time < self.recovery_time:
            return False, f"Minimum recovery time not met: {elapsed_time:.1f}s < {self.recovery_time:.1f}s"
        
        # Run recovery checks
        for check_func in self.recovery_checks:
            try:
                can_proceed, check_reason = check_func()
                if not can_proceed:
                    return False, f"Recovery check failed: {check_reason}"
            except Exception as e:
                return False, f"Recovery check error: {e}"
        
        return True, "Recovery conditions satisfied"
    
    def attempt_recovery(self) -> bool:
        """
        Attempt to recover from emergency stop.
        
        Returns:
            True if recovery successful
        """
        can_recover, reason = self.check_recovery_conditions()
        
        if can_recover:
            self.emergency_active = False
            self.stop_reason = ""
            recovery_time = time.time() - self.stop_timestamp
            
            logger.info(f"Emergency stop recovery successful after {recovery_time:.1f}s")
            return True
        else:
            logger.warning(f"Emergency stop recovery failed: {reason}")
            return False
    
    def add_recovery_check(self, check_func: Callable[[], Tuple[bool, str]]) -> None:
        """Add custom recovery check function."""
        self.recovery_checks.append(check_func)
    
    def _map_constraint_to_violation_type(self, constraint_name: str) -> ViolationType:
        """Map constraint name to violation type."""
        name_lower = constraint_name.lower()
        
        if "collision" in name_lower:
            return ViolationType.COLLISION
        elif "force" in name_lower:
            return ViolationType.FORCE_LIMIT
        elif "joint" in name_lower:
            return ViolationType.JOINT_LIMIT
        elif "velocity" in name_lower:
            return ViolationType.VELOCITY_LIMIT
        elif "acceleration" in name_lower:
            return ViolationType.ACCELERATION_LIMIT
        elif "workspace" in name_lower:
            return ViolationType.WORKSPACE_BOUNDARY
        elif "comfort" in name_lower:
            return ViolationType.HUMAN_COMFORT
        else:
            return ViolationType.SYSTEM_FAILURE
    
    def _analyze_severity_trend(self, severities: List[SafetyLevel]) -> str:
        """Analyze trend in violation severities."""
        severity_values = {
            SafetyLevel.SAFE: 0,
            SafetyLevel.CAUTION: 1,
            SafetyLevel.WARNING: 2,
            SafetyLevel.DANGER: 3,
            SafetyLevel.CRITICAL: 4
        }
        
        values = [severity_values[s] for s in severities]
        
        if len(values) < 3:
            return "insufficient_data"
        
        # Check if generally increasing
        increasing_count = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        
        if increasing_count >= len(values) - 2:  # Allow one non-increasing step
            return "escalating"
        elif increasing_count <= 1:
            return "de-escalating"
        else:
            return "stable"
    
    def _notify_emergency_stop(self, reason: str) -> None:
        """Notify external systems of emergency stop."""
        # This could be extended to notify external systems
        # For now, just log
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get emergency stop controller status."""
        return {
            "emergency_active": self.emergency_active,
            "stop_reason": self.stop_reason,
            "stop_count": self.stop_count,
            "time_since_stop": time.time() - self.stop_timestamp if self.emergency_active else 0.0,
            "recovery_time_remaining": max(0.0, self.recovery_time - (time.time() - self.stop_timestamp)) if self.emergency_active else 0.0
        }


class SafetyMonitor:
    """
    Comprehensive real-time safety monitoring system.
    
    Monitors constraint violations, tracks safety metrics,
    manages emergency stops, and provides predictive safety assessment.
    """
    
    def __init__(self,
                 constraint_names: List[str] = None,
                 constraint_thresholds: Dict[str, float] = None,
                 buffer_size: int = 1000,
                 enable_emergency_stop: bool = True,
                 enable_predictive_monitoring: bool = True,
                 monitoring_frequency: float = 10.0):  # Hz
        """
        Initialize safety monitor.
        
        Args:
            constraint_names: Names of constraints to monitor
            constraint_thresholds: Violation thresholds for each constraint
            buffer_size: Size of constraint history buffer
            enable_emergency_stop: Enable emergency stop functionality
            enable_predictive_monitoring: Enable predictive safety assessment
            monitoring_frequency: Monitoring frequency in Hz
        """
        self.constraint_names = constraint_names or []
        self.constraint_thresholds = constraint_thresholds or {}
        self.monitoring_frequency = monitoring_frequency
        
        # Initialize components
        self.constraint_buffer = ConstraintBuffer(buffer_size, self.constraint_names)
        
        if enable_emergency_stop:
            self.emergency_controller = EmergencyStopController()
        else:
            self.emergency_controller = None
        
        self.enable_predictive_monitoring = enable_predictive_monitoring
        
        # Safety event tracking
        self.safety_events: List[SafetyEvent] = []
        self.current_violations: Dict[str, SafetyEvent] = {}
        
        # Metrics tracking
        self.metrics = SafetyMetrics()
        self.last_update_time = time.time()
        
        # Real-time monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_queue = Queue()
        
        # Callbacks
        self.violation_callbacks: List[Callable[[SafetyEvent], None]] = []
        self.emergency_callbacks: List[Callable[[str], None]] = []
        
        logger.info(f"SafetyMonitor initialized")
        logger.info(f"  Constraints: {self.constraint_names}")
        logger.info(f"  Emergency stop enabled: {enable_emergency_stop}")
        logger.info(f"  Predictive monitoring: {enable_predictive_monitoring}")
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring thread."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time safety monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Real-time safety monitoring stopped")
    
    def update(self,
               constraint_dict: Dict[str, float],
               state: Optional[torch.Tensor] = None,
               action: Optional[torch.Tensor] = None,
               timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Update safety monitor with current constraint values.
        
        Args:
            constraint_dict: Current constraint values
            state: Current state (optional)
            action: Current action (optional)
            timestamp: Current timestamp (optional, defaults to current time)
            
        Returns:
            Safety assessment results
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Add to constraint buffer
        self.constraint_buffer.add(timestamp, constraint_dict, state, action)
        
        # Update metrics
        self.metrics.total_timesteps += 1
        
        # Check for violations
        violations_detected = self._check_violations(constraint_dict, timestamp)
        
        # Compute risk predictions
        risk_predictions = {}
        if self.enable_predictive_monitoring:
            risk_predictions = self._compute_risk_predictions()
        
        # Check emergency conditions
        emergency_triggered = False
        emergency_reason = ""
        
        if self.emergency_controller and not self.emergency_controller.emergency_active:
            should_stop, reason = self.emergency_controller.check_emergency_conditions(
                constraint_dict, self.safety_events[-10:], risk_predictions
            )
            
            if should_stop:
                self.emergency_controller.trigger_emergency_stop(reason)
                emergency_triggered = True
                emergency_reason = reason
                self.metrics.emergency_stops += 1
        
        # Update safety metrics
        if violations_detected:
            self.metrics.consecutive_safe_steps = 0
        else:
            self.metrics.consecutive_safe_steps += 1
        
        self.metrics.safety_rate = (
            (self.metrics.total_timesteps - self.metrics.total_violations) / 
            max(self.metrics.total_timesteps, 1)
        )
        
        # Compute overall safety level
        overall_safety_level = self._compute_overall_safety_level(
            violations_detected, risk_predictions
        )
        
        # Create assessment result
        assessment = {
            "timestamp": timestamp,
            "constraint_values": constraint_dict.copy(),
            "violations_detected": violations_detected,
            "active_violations": len(self.current_violations),
            "risk_predictions": risk_predictions,
            "overall_safety_level": overall_safety_level,
            "emergency_active": self.emergency_controller.emergency_active if self.emergency_controller else False,
            "emergency_triggered": emergency_triggered,
            "emergency_reason": emergency_reason,
            "metrics": self._get_current_metrics()
        }
        
        self.last_update_time = timestamp
        
        return assessment
    
    def _check_violations(self, 
                         constraint_dict: Dict[str, float], 
                         timestamp: float) -> bool:
        """Check for constraint violations and create events."""
        violations_detected = False
        
        for constraint_name, value in constraint_dict.items():
            threshold = self.constraint_thresholds.get(constraint_name, 0.0)
            
            if value > threshold:  # Violation detected
                violations_detected = True
                
                if constraint_name not in self.current_violations:
                    # New violation
                    violation_type = self._map_constraint_to_violation_type(constraint_name)
                    severity = self._classify_violation_severity(value, threshold)
                    
                    event = SafetyEvent(
                        timestamp=timestamp,
                        event_type=violation_type,
                        severity=severity,
                        violation_value=value,
                        threshold=threshold,
                        context={
                            "constraint_name": constraint_name,
                            "violation_magnitude": value - threshold
                        }
                    )
                    
                    self.safety_events.append(event)
                    self.current_violations[constraint_name] = event
                    
                    # Update metrics
                    self.metrics.total_violations += 1
                    self.metrics.violations_by_type[violation_type] += 1
                    self.metrics.violations_by_severity[severity] += 1
                    
                    # Update running averages
                    self._update_violation_statistics(event)
                    
                    # Trigger callbacks
                    for callback in self.violation_callbacks:
                        try:
                            callback(event)
                        except Exception as e:
                            logger.error(f"Violation callback error: {e}")
                    
                    logger.warning(f"New violation: {constraint_name} = {value:.6f} > {threshold:.6f}")
                
                else:
                    # Update existing violation
                    self.current_violations[constraint_name].violation_value = value
            
            else:  # No violation
                if constraint_name in self.current_violations:
                    # Violation resolved
                    event = self.current_violations[constraint_name]
                    event.resolved = True
                    event.resolution_time = timestamp
                    
                    del self.current_violations[constraint_name]
                    
                    logger.info(f"Violation resolved: {constraint_name}")
        
        return violations_detected
    
    def _compute_risk_predictions(self) -> Dict[str, Dict[str, Any]]:
        """Compute risk predictions for all constraints."""
        predictions = {}
        
        for constraint_name in self.constraint_names:
            threshold = self.constraint_thresholds.get(constraint_name, 0.0)
            
            prediction = self.constraint_buffer.predict_violation_risk(
                constraint_name, threshold, horizon=10
            )
            predictions[constraint_name] = prediction
        
        return predictions
    
    def _compute_overall_safety_level(self, 
                                    violations_detected: bool,
                                    risk_predictions: Dict[str, Dict[str, Any]]) -> SafetyLevel:
        """Compute overall system safety level."""
        if self.emergency_controller and self.emergency_controller.emergency_active:
            return SafetyLevel.CRITICAL
        
        if violations_detected:
            # Find highest severity among current violations
            max_severity = SafetyLevel.SAFE
            severity_order = [SafetyLevel.SAFE, SafetyLevel.CAUTION, SafetyLevel.WARNING, 
                            SafetyLevel.DANGER, SafetyLevel.CRITICAL]
            
            for event in self.current_violations.values():
                if severity_order.index(event.severity) > severity_order.index(max_severity):
                    max_severity = event.severity
            
            return max_severity
        
        # Check risk predictions
        if risk_predictions:
            max_risk_level = SafetyLevel.SAFE
            severity_order = [SafetyLevel.SAFE, SafetyLevel.CAUTION, SafetyLevel.WARNING, 
                            SafetyLevel.DANGER, SafetyLevel.CRITICAL]
            
            for prediction in risk_predictions.values():
                risk_level = prediction["risk_level"]
                if severity_order.index(risk_level) > severity_order.index(max_risk_level):
                    max_risk_level = risk_level
            
            return max_risk_level
        
        return SafetyLevel.SAFE
    
    def _classify_violation_severity(self, value: float, threshold: float) -> SafetyLevel:
        """Classify violation severity based on magnitude."""
        violation_magnitude = value - threshold
        relative_violation = violation_magnitude / max(abs(threshold), 1e-6)
        
        if relative_violation < 0.1:
            return SafetyLevel.CAUTION
        elif relative_violation < 0.5:
            return SafetyLevel.WARNING
        elif relative_violation < 1.0:
            return SafetyLevel.DANGER
        else:
            return SafetyLevel.CRITICAL
    
    def _map_constraint_to_violation_type(self, constraint_name: str) -> ViolationType:
        """Map constraint name to violation type."""
        name_lower = constraint_name.lower()
        
        if "collision" in name_lower:
            return ViolationType.COLLISION
        elif "force" in name_lower:
            return ViolationType.FORCE_LIMIT
        elif "joint" in name_lower:
            return ViolationType.JOINT_LIMIT
        elif "velocity" in name_lower:
            return ViolationType.VELOCITY_LIMIT
        elif "acceleration" in name_lower:
            return ViolationType.ACCELERATION_LIMIT
        elif "workspace" in name_lower:
            return ViolationType.WORKSPACE_BOUNDARY
        elif "comfort" in name_lower:
            return ViolationType.HUMAN_COMFORT
        else:
            return ViolationType.SYSTEM_FAILURE
    
    def _update_violation_statistics(self, event: SafetyEvent) -> None:
        """Update running violation statistics."""
        if self.metrics.total_violations > 1:
            # Update mean time to violation
            time_since_last = event.timestamp - self.last_update_time
            alpha = 0.1  # Smoothing factor
            self.metrics.mean_time_to_violation = (
                (1 - alpha) * self.metrics.mean_time_to_violation + 
                alpha * time_since_last
            )
        
        # Update mean violation magnitude
        alpha = 0.1
        violation_magnitude = event.violation_value - event.threshold
        self.metrics.mean_violation_magnitude = (
            (1 - alpha) * self.metrics.mean_violation_magnitude + 
            alpha * violation_magnitude
        )
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current safety metrics."""
        return {
            "total_timesteps": self.metrics.total_timesteps,
            "total_violations": self.metrics.total_violations,
            "safety_rate": self.metrics.safety_rate,
            "consecutive_safe_steps": self.metrics.consecutive_safe_steps,
            "active_violations": len(self.current_violations),
            "emergency_stops": self.metrics.emergency_stops
        }
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for real-time thread."""
        loop_interval = 1.0 / self.monitoring_frequency
        
        while self.monitoring_active:
            loop_start = time.time()
            
            try:
                # Process any queued monitoring requests
                try:
                    while True:
                        item = self.monitoring_queue.get_nowait()
                        # Process monitoring item if needed
                        self.monitoring_queue.task_done()
                except Empty:
                    pass
                
                # Perform periodic safety checks
                self._periodic_safety_check()
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            # Sleep for remaining time
            elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _periodic_safety_check(self) -> None:
        """Perform periodic safety system health checks."""
        # Check if emergency controller can attempt recovery
        if (self.emergency_controller and 
            self.emergency_controller.emergency_active):
            
            can_recover, reason = self.emergency_controller.check_recovery_conditions()
            if can_recover:
                logger.info(f"Emergency recovery conditions met: {reason}")
        
        # Check for stale data
        current_time = time.time()
        time_since_update = current_time - self.last_update_time
        
        if time_since_update > 1.0:  # More than 1 second since last update
            logger.warning(f"Stale safety data: {time_since_update:.1f}s since last update")
        
        # Check buffer health
        buffer_stats = self.constraint_buffer.get_statistics()
        if buffer_stats["buffer_utilization"] > 0.95:
            logger.warning("Constraint buffer near capacity")
    
    def add_violation_callback(self, callback: Callable[[SafetyEvent], None]) -> None:
        """Add callback function for violation events."""
        self.violation_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback function for emergency stop events."""
        self.emergency_callbacks.append(callback)
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        current_time = time.time()
        
        report = {
            "timestamp": current_time,
            "monitoring_duration": current_time - (self.safety_events[0].timestamp if self.safety_events else current_time),
            "overall_metrics": self._get_current_metrics(),
            "constraint_buffer_stats": self.constraint_buffer.get_statistics(),
            "recent_violations": [
                {
                    "timestamp": event.timestamp,
                    "type": event.event_type.value,
                    "severity": event.severity.value,
                    "value": event.violation_value,
                    "threshold": event.threshold,
                    "resolved": event.resolved
                }
                for event in self.safety_events[-20:]  # Last 20 events
            ],
            "active_violations": list(self.current_violations.keys()),
            "emergency_status": self.emergency_controller.get_status() if self.emergency_controller else None
        }
        
        # Add trend analysis for each constraint
        trends = {}
        for constraint_name in self.constraint_names:
            trend_info = self.constraint_buffer.compute_trend(constraint_name)
            trends[constraint_name] = trend_info
        
        report["constraint_trends"] = trends
        
        return report
    
    def save_safety_log(self, filepath: str) -> None:
        """Save safety events to file."""
        safety_data = {
            "monitoring_session": {
                "start_time": self.safety_events[0].timestamp if self.safety_events else time.time(),
                "end_time": time.time(),
                "total_events": len(self.safety_events)
            },
            "metrics": self.metrics.__dict__,
            "events": [
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "violation_value": event.violation_value,
                    "threshold": event.threshold,
                    "context": event.context,
                    "resolved": event.resolved,
                    "resolution_time": event.resolution_time,
                    "action_taken": event.action_taken
                }
                for event in self.safety_events
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(safety_data, f, indent=2, default=str)
        
        logger.info(f"Safety log saved to {filepath}")
    
    def load_safety_log(self, filepath: str) -> None:
        """Load safety events from file."""
        with open(filepath, 'r') as f:
            safety_data = json.load(f)
        
        # Reconstruct events
        self.safety_events = []
        for event_data in safety_data["events"]:
            event = SafetyEvent(
                timestamp=event_data["timestamp"],
                event_type=ViolationType(event_data["event_type"]),
                severity=SafetyLevel(event_data["severity"]),
                violation_value=event_data["violation_value"],
                threshold=event_data["threshold"],
                context=event_data["context"],
                resolved=event_data["resolved"],
                resolution_time=event_data.get("resolution_time"),
                action_taken=event_data.get("action_taken")
            )
            self.safety_events.append(event)
        
        logger.info(f"Safety log loaded from {filepath}: {len(self.safety_events)} events")
    
    def reset(self) -> None:
        """Reset safety monitor state."""
        self.safety_events.clear()
        self.current_violations.clear()
        self.metrics = SafetyMetrics()
        
        if self.emergency_controller:
            self.emergency_controller.emergency_active = False
            self.emergency_controller.stop_reason = ""
            self.emergency_controller.stop_count = 0
        
        logger.info("Safety monitor reset")
    
    def __del__(self):
        """Cleanup when safety monitor is destroyed."""
        self.stop_monitoring()