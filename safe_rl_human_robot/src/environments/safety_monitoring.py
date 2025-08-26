"""
Advanced Safety Constraint Monitoring Systems for Shared Control.

This module provides comprehensive safety monitoring with predictive capabilities,
real-time constraint evaluation, and adaptive safety measures for human-robot
shared control environments.
"""

import numpy as np
import torch
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import time
import threading
from abc import ABC, abstractmethod

from .shared_control_base import SafetyConstraint, SharedControlState
from .human_robot_env import EnvironmentState

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for constraint monitoring."""
    SAFE = "safe"                    # All constraints satisfied
    WARNING = "warning"              # Approaching constraint violation
    CRITICAL = "critical"            # Constraint violation imminent
    EMERGENCY = "emergency"          # Immediate safety intervention required


class ConstraintType(Enum):
    """Types of safety constraints."""
    COLLISION_AVOIDANCE = "collision_avoidance"
    JOINT_LIMITS = "joint_limits"
    VELOCITY_LIMITS = "velocity_limits"
    FORCE_LIMITS = "force_limits"
    WORKSPACE_BOUNDS = "workspace_bounds"
    SINGULARITY_AVOIDANCE = "singularity_avoidance"
    HUMAN_COMFORT = "human_comfort"
    STABILITY = "stability"
    CUSTOM = "custom"


@dataclass
class SafetyViolation:
    """Information about a safety constraint violation."""
    constraint_name: str
    constraint_type: ConstraintType
    violation_time: float
    severity: float  # 0-1 scale
    constraint_value: float
    threshold: float
    predicted_violation: bool
    recovery_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyStatus:
    """Current safety status of the system."""
    overall_level: SafetyLevel
    active_violations: List[SafetyViolation]
    predicted_violations: List[SafetyViolation]
    safety_margins: Dict[str, float]
    emergency_stop_active: bool
    last_update_time: float
    confidence: float = 1.0


class PredictiveConstraint:
    """Constraint with predictive capability."""
    
    def __init__(self,
                 base_constraint: SafetyConstraint,
                 prediction_horizon: float = 1.0,
                 prediction_samples: int = 10):
        """
        Initialize predictive constraint.
        
        Args:
            base_constraint: Base safety constraint
            prediction_horizon: Time horizon for prediction (seconds)
            prediction_samples: Number of prediction samples
        """
        self.base_constraint = base_constraint
        self.prediction_horizon = prediction_horizon
        self.prediction_samples = prediction_samples
        
        # History for trend analysis
        self.value_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
    def evaluate_predictive(self, 
                           state: EnvironmentState, 
                           action: torch.Tensor,
                           dynamics_model: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate constraint with predictive analysis.
        
        Args:
            state: Current environment state
            action: Proposed action
            dynamics_model: Optional dynamics model for prediction
            
        Returns:
            Comprehensive constraint evaluation
        """
        # Current constraint evaluation
        current_eval = self.base_constraint.evaluate(state, action)
        current_time = time.time()
        
        # Store in history
        self.value_history.append(current_eval['value'])
        self.time_history.append(current_time)
        
        # Predictive analysis
        predicted_violations = []
        
        if len(self.value_history) >= 5:  # Need some history for prediction
            # Simple linear trend prediction
            trend_prediction = self._predict_linear_trend()
            if trend_prediction['violation_predicted']:
                predicted_violations.append({
                    'time_to_violation': trend_prediction['time_to_violation'],
                    'predicted_value': trend_prediction['predicted_value'],
                    'confidence': trend_prediction['confidence']
                })
            
            # Dynamics-based prediction if model available
            if dynamics_model is not None:
                dynamics_prediction = self._predict_with_dynamics(
                    state, action, dynamics_model
                )
                if dynamics_prediction['violation_predicted']:
                    predicted_violations.append(dynamics_prediction)
        
        # Compute safety margins
        margin = current_eval['value'] - self.base_constraint.threshold
        time_to_violation = self._estimate_time_to_violation()
        
        return {
            **current_eval,
            'predicted_violations': predicted_violations,
            'time_to_violation': time_to_violation,
            'trend_derivative': self._compute_trend_derivative(),
            'safety_margin': margin,
            'constraint_name': self.base_constraint.name
        }
    
    def _predict_linear_trend(self) -> Dict[str, Any]:
        """Predict constraint violation using linear trend extrapolation."""
        if len(self.value_history) < 3:
            return {'violation_predicted': False}
        
        # Convert to numpy for easier computation
        values = np.array(list(self.value_history)[-10:])  # Last 10 values
        times = np.array(list(self.time_history)[-10:])
        
        # Linear regression
        if len(times) > 1:
            dt = times[-1] - times[0]
            if dt > 1e-6:  # Avoid division by zero
                # Simple linear fit
                time_normalized = (times - times[0]) / dt
                coeffs = np.polyfit(time_normalized, values, 1)
                slope = coeffs[0] / dt  # Derivative per second
                intercept = coeffs[1]
                
                # Predict when constraint will be violated
                if slope < 0:  # Constraint value decreasing
                    current_value = values[-1]
                    threshold = self.base_constraint.threshold
                    
                    if current_value > threshold:  # Currently safe
                        time_to_violation = (current_value - threshold) / abs(slope)
                        
                        if time_to_violation <= self.prediction_horizon:
                            predicted_value = current_value + slope * time_to_violation
                            confidence = min(1.0, 0.8 * (1.0 - time_to_violation / self.prediction_horizon))
                            
                            return {
                                'violation_predicted': True,
                                'time_to_violation': time_to_violation,
                                'predicted_value': predicted_value,
                                'confidence': confidence,
                                'method': 'linear_trend'
                            }
        
        return {'violation_predicted': False}
    
    def _predict_with_dynamics(self, 
                             state: EnvironmentState,
                             action: torch.Tensor,
                             dynamics_model: Callable) -> Dict[str, Any]:
        """Predict using dynamics model."""
        # This would use a more sophisticated dynamics model
        # For now, return placeholder
        return {'violation_predicted': False, 'method': 'dynamics'}
    
    def _compute_trend_derivative(self) -> float:
        """Compute current trend derivative."""
        if len(self.value_history) < 2:
            return 0.0
        
        recent_values = list(self.value_history)[-5:]  # Last 5 values
        recent_times = list(self.time_history)[-5:]
        
        if len(recent_values) < 2:
            return 0.0
        
        # Simple finite difference
        dt = recent_times[-1] - recent_times[-2]
        if dt > 1e-6:
            return (recent_values[-1] - recent_values[-2]) / dt
        
        return 0.0
    
    def _estimate_time_to_violation(self) -> Optional[float]:
        """Estimate time until constraint violation."""
        if len(self.value_history) < 2:
            return None
        
        current_value = self.value_history[-1]
        threshold = self.base_constraint.threshold
        
        if current_value <= threshold:  # Already violated
            return 0.0
        
        # Use trend to estimate
        derivative = self._compute_trend_derivative()
        
        if derivative < 0:  # Value decreasing toward threshold
            time_to_violation = (current_value - threshold) / abs(derivative)
            return max(0.0, time_to_violation)
        
        return None  # Not approaching violation


class AdaptiveSafetyMonitor:
    """
    Adaptive safety monitoring system with learning capabilities.
    
    Monitors safety constraints, predicts violations, and adapts
    monitoring parameters based on system performance.
    """
    
    def __init__(self,
                 update_frequency: float = 100.0,
                 prediction_horizon: float = 2.0,
                 adaptation_rate: float = 0.01):
        """
        Initialize adaptive safety monitor.
        
        Args:
            update_frequency: Monitoring frequency (Hz)
            prediction_horizon: Prediction time horizon (seconds)
            adaptation_rate: Rate of parameter adaptation
        """
        self.update_frequency = update_frequency
        self.prediction_horizon = prediction_horizon
        self.adaptation_rate = adaptation_rate
        
        # Constraints
        self.predictive_constraints: List[PredictiveConstraint] = []
        self.constraint_weights: Dict[str, float] = {}
        
        # Safety state
        self.current_status = SafetyStatus(
            overall_level=SafetyLevel.SAFE,
            active_violations=[],
            predicted_violations=[],
            safety_margins={},
            emergency_stop_active=False,
            last_update_time=time.time()
        )
        
        # Performance tracking
        self.violation_history = deque(maxlen=1000)
        self.prediction_accuracy = deque(maxlen=100)
        self.false_positive_rate = 0.0
        self.false_negative_rate = 0.0
        
        # Threading for real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.state_lock = threading.Lock()
        
        # Emergency stop mechanism
        self.emergency_callbacks: List[Callable] = []
        
        logger.info(f"AdaptiveSafetyMonitor initialized: {update_frequency}Hz, {prediction_horizon}s horizon")
    
    def add_constraint(self, 
                      constraint: SafetyConstraint,
                      weight: float = 1.0,
                      prediction_samples: int = 10) -> None:
        """
        Add safety constraint to monitoring system.
        
        Args:
            constraint: Safety constraint to monitor
            weight: Weight for constraint importance
            prediction_samples: Number of samples for prediction
        """
        predictive_constraint = PredictiveConstraint(
            constraint, 
            self.prediction_horizon,
            prediction_samples
        )
        
        self.predictive_constraints.append(predictive_constraint)
        self.constraint_weights[constraint.name] = weight
        
        logger.info(f"Added constraint: {constraint.name} (weight: {weight})")
    
    def remove_constraint(self, constraint_name: str) -> bool:
        """Remove constraint from monitoring."""
        for i, pred_constraint in enumerate(self.predictive_constraints):
            if pred_constraint.base_constraint.name == constraint_name:
                del self.predictive_constraints[i]
                if constraint_name in self.constraint_weights:
                    del self.constraint_weights[constraint_name]
                logger.info(f"Removed constraint: {constraint_name}")
                return True
        return False
    
    def evaluate_safety(self, 
                       state: EnvironmentState, 
                       action: torch.Tensor,
                       dynamics_model: Optional[Callable] = None) -> SafetyStatus:
        """
        Comprehensive safety evaluation.
        
        Args:
            state: Current environment state
            action: Proposed action
            dynamics_model: Optional dynamics model for prediction
            
        Returns:
            Current safety status
        """
        current_time = time.time()
        
        with self.state_lock:
            active_violations = []
            predicted_violations = []
            safety_margins = {}
            
            # Evaluate all constraints
            for pred_constraint in self.predictive_constraints:
                try:
                    evaluation = pred_constraint.evaluate_predictive(
                        state, action, dynamics_model
                    )
                    
                    constraint_name = evaluation['constraint_name']
                    safety_margins[constraint_name] = evaluation['safety_margin']
                    
                    # Check for active violations
                    if evaluation['violation']:
                        violation = SafetyViolation(
                            constraint_name=constraint_name,
                            constraint_type=self._infer_constraint_type(constraint_name),
                            violation_time=current_time,
                            severity=self._compute_violation_severity(evaluation),
                            constraint_value=evaluation['value'],
                            threshold=evaluation['threshold'],
                            predicted_violation=False
                        )
                        active_violations.append(violation)
                    
                    # Check for predicted violations
                    for pred_violation in evaluation.get('predicted_violations', []):
                        if pred_violation.get('confidence', 0) > 0.5:
                            violation = SafetyViolation(
                                constraint_name=constraint_name,
                                constraint_type=self._infer_constraint_type(constraint_name),
                                violation_time=current_time + pred_violation['time_to_violation'],
                                severity=pred_violation.get('confidence', 0.5),
                                constraint_value=pred_violation.get('predicted_value', 0),
                                threshold=evaluation['threshold'],
                                predicted_violation=True,
                                metadata=pred_violation
                            )
                            predicted_violations.append(violation)
                
                except Exception as e:
                    logger.error(f"Error evaluating constraint {pred_constraint.base_constraint.name}: {e}")
            
            # Determine overall safety level
            overall_level = self._determine_safety_level(active_violations, predicted_violations)
            
            # Check for emergency stop conditions
            emergency_stop = self._should_trigger_emergency_stop(active_violations, predicted_violations)
            
            # Update safety status
            self.current_status = SafetyStatus(
                overall_level=overall_level,
                active_violations=active_violations,
                predicted_violations=predicted_violations,
                safety_margins=safety_margins,
                emergency_stop_active=emergency_stop,
                last_update_time=current_time,
                confidence=self._compute_overall_confidence()
            )
            
            # Trigger emergency stop if needed
            if emergency_stop and not self.current_status.emergency_stop_active:
                self._trigger_emergency_stop()
        
        # Adapt monitoring parameters
        self._adapt_parameters()
        
        return self.current_status
    
    def _infer_constraint_type(self, constraint_name: str) -> ConstraintType:
        """Infer constraint type from name."""
        name_lower = constraint_name.lower()
        
        if 'collision' in name_lower or 'obstacle' in name_lower:
            return ConstraintType.COLLISION_AVOIDANCE
        elif 'joint' in name_lower and 'limit' in name_lower:
            return ConstraintType.JOINT_LIMITS
        elif 'velocity' in name_lower or 'speed' in name_lower:
            return ConstraintType.VELOCITY_LIMITS
        elif 'force' in name_lower or 'torque' in name_lower:
            return ConstraintType.FORCE_LIMITS
        elif 'workspace' in name_lower or 'boundary' in name_lower:
            return ConstraintType.WORKSPACE_BOUNDS
        elif 'singular' in name_lower:
            return ConstraintType.SINGULARITY_AVOIDANCE
        elif 'comfort' in name_lower or 'human' in name_lower:
            return ConstraintType.HUMAN_COMFORT
        elif 'stability' in name_lower or 'stable' in name_lower:
            return ConstraintType.STABILITY
        else:
            return ConstraintType.CUSTOM
    
    def _compute_violation_severity(self, evaluation: Dict[str, Any]) -> float:
        """Compute violation severity (0-1)."""
        if not evaluation['violation']:
            return 0.0
        
        # Base severity on how much constraint is violated
        margin = evaluation.get('safety_margin', 0)
        penalty = evaluation.get('penalty', 0)
        
        # Normalize penalty to 0-1 range
        if penalty > 0:
            # Use penalty magnitude as severity indicator
            severity = min(1.0, penalty / 100.0)  # Assume max reasonable penalty ~100
        else:
            # Use margin if no penalty available
            severity = min(1.0, max(0.0, -margin / 0.1))  # Assume 0.1 is significant violation
        
        return severity
    
    def _determine_safety_level(self, 
                               active_violations: List[SafetyViolation],
                               predicted_violations: List[SafetyViolation]) -> SafetyLevel:
        """Determine overall safety level."""
        if not active_violations and not predicted_violations:
            return SafetyLevel.SAFE
        
        # Check for emergency level violations
        for violation in active_violations:
            if violation.severity > 0.9 or violation.constraint_type in [
                ConstraintType.COLLISION_AVOIDANCE, ConstraintType.FORCE_LIMITS
            ]:
                return SafetyLevel.EMERGENCY
        
        # Check for critical level
        if len(active_violations) > 2 or any(v.severity > 0.7 for v in active_violations):
            return SafetyLevel.CRITICAL
        
        # Check for warning level
        if active_violations or any(
            v.violation_time - time.time() < 1.0 and v.severity > 0.5 
            for v in predicted_violations
        ):
            return SafetyLevel.WARNING
        
        return SafetyLevel.SAFE
    
    def _should_trigger_emergency_stop(self, 
                                     active_violations: List[SafetyViolation],
                                     predicted_violations: List[SafetyViolation]) -> bool:
        """Determine if emergency stop should be triggered."""
        # Immediate stop for collision or force violations
        for violation in active_violations:
            if (violation.constraint_type in [ConstraintType.COLLISION_AVOIDANCE, ConstraintType.FORCE_LIMITS] and
                violation.severity > 0.8):
                return True
        
        # Stop for imminent critical violations
        for violation in predicted_violations:
            time_to_violation = violation.violation_time - time.time()
            if (time_to_violation < 0.5 and 
                violation.severity > 0.9 and
                violation.constraint_type in [ConstraintType.COLLISION_AVOIDANCE, ConstraintType.FORCE_LIMITS]):
                return True
        
        return False
    
    def _trigger_emergency_stop(self) -> None:
        """Trigger emergency stop procedures."""
        logger.critical("EMERGENCY STOP TRIGGERED")
        
        # Execute all emergency callbacks
        for callback in self.emergency_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")
        
        # Log violation details
        for violation in self.current_status.active_violations:
            logger.critical(f"Active violation: {violation.constraint_name} "
                           f"(severity: {violation.severity:.2f})")
    
    def _compute_overall_confidence(self) -> float:
        """Compute confidence in safety assessment."""
        if not self.predictive_constraints:
            return 1.0
        
        # Base confidence on prediction accuracy and constraint coverage
        base_confidence = 0.8
        
        # Reduce confidence if many predicted violations
        if len(self.current_status.predicted_violations) > 3:
            base_confidence -= 0.2
        
        # Adjust based on prediction accuracy history
        if len(self.prediction_accuracy) > 10:
            accuracy = np.mean(list(self.prediction_accuracy))
            base_confidence = 0.5 * base_confidence + 0.5 * accuracy
        
        return max(0.1, min(1.0, base_confidence))
    
    def _adapt_parameters(self) -> None:
        """Adapt monitoring parameters based on performance."""
        # Simple adaptation: adjust prediction horizon based on accuracy
        if len(self.prediction_accuracy) > 20:
            recent_accuracy = np.mean(list(self.prediction_accuracy)[-20:])
            
            if recent_accuracy > 0.8:
                # Good accuracy - can extend horizon slightly
                self.prediction_horizon = min(5.0, self.prediction_horizon * 1.01)
            elif recent_accuracy < 0.6:
                # Poor accuracy - reduce horizon
                self.prediction_horizon = max(0.5, self.prediction_horizon * 0.99)
    
    def add_emergency_callback(self, callback: Callable) -> None:
        """Add callback to be executed on emergency stop."""
        self.emergency_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start real-time safety monitoring in separate thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        
        logger.info("Safety monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop (runs in separate thread)."""
        dt = 1.0 / self.update_frequency
        
        while self.monitoring_active:
            try:
                # This would integrate with the main control loop
                # For now, just maintain the thread structure
                time.sleep(dt)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        with self.state_lock:
            status = self.current_status
        
        # Performance metrics
        violation_count = len(self.violation_history)
        recent_violations = [v for v in self.violation_history 
                           if time.time() - v.violation_time < 60.0]  # Last minute
        
        # Constraint health
        constraint_health = {}
        for pred_constraint in self.predictive_constraints:
            name = pred_constraint.base_constraint.name
            constraint_health[name] = {
                'active': pred_constraint.base_constraint.active,
                'weight': self.constraint_weights.get(name, 1.0),
                'recent_violations': len([v for v in recent_violations 
                                        if v.constraint_name == name]),
                'trend_derivative': pred_constraint._compute_trend_derivative()
            }
        
        return {
            'timestamp': time.time(),
            'overall_status': {
                'level': status.overall_level.value,
                'confidence': status.confidence,
                'emergency_stop_active': status.emergency_stop_active
            },
            'current_violations': {
                'active_count': len(status.active_violations),
                'predicted_count': len(status.predicted_violations),
                'details': [
                    {
                        'name': v.constraint_name,
                        'type': v.constraint_type.value,
                        'severity': v.severity,
                        'predicted': v.predicted_violation
                    } for v in status.active_violations + status.predicted_violations
                ]
            },
            'safety_margins': status.safety_margins,
            'performance_metrics': {
                'total_violations': violation_count,
                'recent_violations': len(recent_violations),
                'prediction_accuracy': np.mean(list(self.prediction_accuracy)) if self.prediction_accuracy else 0.0,
                'false_positive_rate': self.false_positive_rate,
                'false_negative_rate': self.false_negative_rate
            },
            'constraint_health': constraint_health,
            'monitoring_parameters': {
                'update_frequency': self.update_frequency,
                'prediction_horizon': self.prediction_horizon,
                'active_constraints': len(self.predictive_constraints)
            }
        }