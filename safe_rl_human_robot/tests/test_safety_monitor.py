"""
Tests for safety monitoring system.

Tests real-time constraint violation detection, emergency stop mechanisms,
safety metrics computation, and predictive safety assessment.
"""

import pytest
import torch
import numpy as np
import time
import threading
from typing import Dict, List
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.safety_monitor import (
    SafetyMonitor, SafetyEvent, SafetyMetrics, SafetyLevel, ViolationType,
    ConstraintBuffer, EmergencyStopController
)


class TestSafetyEvent:
    """Test safety event data structure."""
    
    def test_safety_event_creation(self):
        """Test safety event creation."""
        event = SafetyEvent(
            timestamp=time.time(),
            event_type=ViolationType.COLLISION,
            severity=SafetyLevel.WARNING,
            violation_value=0.15,
            threshold=0.1,
            context={"test": "data"}
        )
        
        assert event.event_type == ViolationType.COLLISION
        assert event.severity == SafetyLevel.WARNING
        assert event.violation_value == 0.15
        assert event.threshold == 0.1
        assert not event.resolved
        assert event.resolution_time is None


class TestSafetyMetrics:
    """Test safety metrics data structure."""
    
    def test_safety_metrics_initialization(self):
        """Test safety metrics initialization."""
        metrics = SafetyMetrics()
        
        assert metrics.total_timesteps == 0
        assert metrics.total_violations == 0
        assert len(metrics.violations_by_type) == 0
        assert len(metrics.violations_by_severity) == 0
        assert metrics.safety_rate == 1.0
        assert metrics.consecutive_safe_steps == 0
        assert metrics.emergency_stops == 0


class TestConstraintBuffer:
    """Test constraint buffer for trend analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constraint_names = ["collision", "force_limit", "joint_limit"]
        self.buffer = ConstraintBuffer(
            buffer_size=100,
            constraint_names=self.constraint_names
        )
    
    def test_buffer_initialization(self):
        """Test constraint buffer initialization."""
        assert self.buffer.buffer_size == 100
        assert self.buffer.constraint_names == self.constraint_names
        assert self.buffer.current_size == 0
        assert self.buffer.total_added == 0
        
        for name in self.constraint_names:
            assert name in self.buffer.constraint_values
            assert len(self.buffer.constraint_values[name]) == 0
    
    def test_buffer_data_addition(self):
        """Test adding constraint data to buffer."""
        timestamp = time.time()
        constraint_dict = {
            "collision": 0.05,
            "force_limit": 2.0,
            "joint_limit": -0.1
        }
        state = torch.randn(6)
        action = torch.randn(3)
        
        self.buffer.add(timestamp, constraint_dict, state, action)
        
        assert self.buffer.current_size == 1
        assert self.buffer.total_added == 1
        
        # Check stored values
        assert self.buffer.constraint_values["collision"][0] == 0.05
        assert self.buffer.constraint_values["force_limit"][0] == 2.0
        assert self.buffer.constraint_values["joint_limit"][0] == -0.1
        assert len(self.buffer.timestamps) == 1
        assert len(self.buffer.states) == 1
        assert len(self.buffer.actions) == 1
    
    def test_buffer_overflow_handling(self):
        """Test buffer behavior when capacity is exceeded."""
        # Fill buffer beyond capacity
        for i in range(150):
            timestamp = time.time() + i
            constraint_dict = {"collision": float(i), "force_limit": float(i*2), "joint_limit": float(-i)}
            self.buffer.add(timestamp, constraint_dict)
        
        # Size should be capped at buffer_size
        assert self.buffer.current_size == 100
        assert self.buffer.total_added == 150
        
        # Check that recent data is preserved
        recent_values = list(self.buffer.constraint_values["collision"])
        assert len(recent_values) == 100
        assert recent_values[-1] == 149.0  # Most recent value
    
    def test_recent_values_retrieval(self):
        """Test retrieval of recent constraint values."""
        # Add some data
        for i in range(20):
            constraint_dict = {"collision": float(i), "force_limit": 0.0, "joint_limit": 0.0}
            self.buffer.add(time.time() + i, constraint_dict)
        
        # Get recent values
        recent_collision = self.buffer.get_recent_values("collision", window_size=5)
        
        assert len(recent_collision) == 5
        assert recent_collision == [15.0, 16.0, 17.0, 18.0, 19.0]
        
        # Test with larger window than available data
        large_window = self.buffer.get_recent_values("collision", window_size=30)
        assert len(large_window) == 20  # All available data
    
    def test_trend_computation(self):
        """Test constraint trend analysis."""
        # Add increasing trend data
        for i in range(15):
            constraint_dict = {"collision": float(i * 0.01), "force_limit": 0.0, "joint_limit": 0.0}
            self.buffer.add(time.time() + i, constraint_dict)
        
        trend_info = self.buffer.compute_trend("collision", window_size=10)
        
        assert isinstance(trend_info, dict)
        assert "trend" in trend_info
        assert "volatility" in trend_info
        assert "direction" in trend_info
        assert "current_value" in trend_info
        
        # Should detect increasing trend
        assert trend_info["trend"] > 0
        assert trend_info["direction"] == "increasing"
        assert abs(trend_info["current_value"] - 0.14) < 1e-6
    
    def test_violation_risk_prediction(self):
        """Test predictive violation risk assessment."""
        # Add data with increasing trend toward violation
        threshold = 0.1
        for i in range(20):
            value = 0.05 + i * 0.003  # Gradually increasing toward threshold
            constraint_dict = {"collision": value, "force_limit": 0.0, "joint_limit": 0.0}
            self.buffer.add(time.time() + i, constraint_dict)
        
        risk_info = self.buffer.predict_violation_risk("collision", threshold, horizon=10)
        
        assert isinstance(risk_info, dict)
        assert "risk_score" in risk_info
        assert "predicted_violation_time" in risk_info
        assert "current_trend" in risk_info
        assert "risk_level" in risk_info
        
        # Should predict some risk since trend is increasing toward threshold
        assert risk_info["risk_score"] > 0.0
        assert risk_info["risk_level"] in [SafetyLevel.CAUTION, SafetyLevel.WARNING, SafetyLevel.DANGER]
    
    def test_buffer_statistics(self):
        """Test buffer statistics computation."""
        # Add some varied data
        for i in range(30):
            constraint_dict = {
                "collision": 0.05 + 0.02 * np.sin(i * 0.1),
                "force_limit": 1.0 + 0.5 * np.cos(i * 0.2),
                "joint_limit": -0.1 + 0.05 * np.random.randn()
            }
            self.buffer.add(time.time() + i, constraint_dict)
        
        stats = self.buffer.get_statistics()
        
        assert isinstance(stats, dict)
        assert stats["current_size"] == 30
        assert stats["total_added"] == 30
        assert stats["buffer_utilization"] == 0.3
        
        # Check per-constraint statistics
        for name in self.constraint_names:
            assert f"{name}_mean" in stats
            assert f"{name}_std" in stats
            assert f"{name}_min" in stats
            assert f"{name}_max" in stats


class TestEmergencyStopController:
    """Test emergency stop controller."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.emergency_thresholds = {
            ViolationType.COLLISION: 0.05,
            ViolationType.FORCE_LIMIT: 50.0,
            ViolationType.JOINT_LIMIT: 0.2
        }
        
        self.controller = EmergencyStopController(
            emergency_thresholds=self.emergency_thresholds,
            enable_predictive_stop=True,
            stop_response_time=0.1
        )
    
    def test_controller_initialization(self):
        """Test emergency stop controller initialization."""
        assert self.controller.emergency_thresholds == self.emergency_thresholds
        assert self.controller.enable_predictive_stop is True
        assert self.controller.stop_response_time == 0.1
        assert not self.controller.emergency_active
        assert self.controller.stop_count == 0
    
    def test_emergency_condition_detection(self):
        """Test emergency condition detection."""
        # Test with safe constraint values
        constraint_dict = {"collision_avoidance": 0.02, "force_limits": 30.0}
        violation_history = []
        risk_predictions = {}
        
        should_stop, reason = self.controller.check_emergency_conditions(
            constraint_dict, violation_history, risk_predictions
        )
        
        assert not should_stop
        assert reason == ""
        
        # Test with critical violation
        constraint_dict_critical = {"collision_avoidance": 0.1, "force_limits": 30.0}
        
        should_stop, reason = self.controller.check_emergency_conditions(
            constraint_dict_critical, violation_history, risk_predictions
        )
        
        assert should_stop
        assert "Critical" in reason
        assert "collision" in reason.lower()
    
    def test_emergency_stop_triggering(self):
        """Test emergency stop triggering."""
        assert not self.controller.emergency_active
        
        reason = "Test emergency stop"
        self.controller.trigger_emergency_stop(reason)
        
        assert self.controller.emergency_active
        assert self.controller.stop_reason == reason
        assert self.controller.stop_count == 1
        assert self.controller.stop_timestamp > 0
    
    def test_recovery_condition_checking(self):
        """Test recovery condition checking."""
        # Should be able to recover initially (no emergency)
        can_recover, reason = self.controller.check_recovery_conditions()
        assert can_recover
        assert "No emergency stop active" in reason
        
        # Trigger emergency stop
        self.controller.trigger_emergency_stop("Test")
        
        # Should not be able to recover immediately
        can_recover, reason = self.controller.check_recovery_conditions()
        assert not can_recover
        assert "Minimum recovery time not met" in reason
        
        # Simulate time passage
        self.controller.stop_timestamp = time.time() - 10  # 10 seconds ago
        
        can_recover, reason = self.controller.check_recovery_conditions()
        assert can_recover
        assert "Recovery conditions satisfied" in reason
    
    def test_recovery_attempt(self):
        """Test emergency stop recovery."""
        # Trigger emergency stop
        self.controller.trigger_emergency_stop("Test emergency")
        assert self.controller.emergency_active
        
        # Immediate recovery should fail
        success = self.controller.attempt_recovery()
        assert not success
        assert self.controller.emergency_active
        
        # Simulate time passage
        self.controller.stop_timestamp = time.time() - 10
        
        # Recovery should succeed
        success = self.controller.attempt_recovery()
        assert success
        assert not self.controller.emergency_active
    
    def test_predictive_emergency_stopping(self):
        """Test predictive emergency stopping."""
        constraint_dict = {"collision_avoidance": 0.03}  # Below threshold but trending up
        violation_history = []
        
        # Create risk prediction indicating imminent critical violation
        risk_predictions = {
            "collision_avoidance": {
                "risk_level": SafetyLevel.CRITICAL,
                "predicted_violation_time": 0.05  # Less than response time
            }
        }
        
        should_stop, reason = self.controller.check_emergency_conditions(
            constraint_dict, violation_history, risk_predictions
        )
        
        assert should_stop
        assert "Predicted critical violation" in reason
    
    def test_consecutive_violation_detection(self):
        """Test detection of consecutive violations."""
        constraint_dict = {"collision_avoidance": 0.02}  # Below emergency threshold
        
        # Create multiple unresolved violations
        violation_history = [
            SafetyEvent(time.time()-3, ViolationType.COLLISION, SafetyLevel.WARNING, 0.08, 0.05),
            SafetyEvent(time.time()-2, ViolationType.FORCE_LIMIT, SafetyLevel.WARNING, 45.0, 40.0),
            SafetyEvent(time.time()-1, ViolationType.JOINT_LIMIT, SafetyLevel.WARNING, 0.15, 0.1)
        ]
        
        risk_predictions = {}
        
        should_stop, reason = self.controller.check_emergency_conditions(
            constraint_dict, violation_history, risk_predictions
        )
        
        assert should_stop
        assert "Multiple unresolved violations" in reason
    
    def test_controller_status(self):
        """Test controller status reporting."""
        status = self.controller.get_status()
        
        assert isinstance(status, dict)
        assert "emergency_active" in status
        assert "stop_reason" in status
        assert "stop_count" in status
        assert not status["emergency_active"]
        assert status["stop_count"] == 0
        
        # Trigger emergency and check status
        self.controller.trigger_emergency_stop("Test")
        status_active = self.controller.get_status()
        
        assert status_active["emergency_active"]
        assert status_active["stop_reason"] == "Test"
        assert status_active["stop_count"] == 1


class TestSafetyMonitor:
    """Test comprehensive safety monitoring system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constraint_names = ["collision_avoidance", "force_limits"]
        self.constraint_thresholds = {
            "collision_avoidance": 0.1,
            "force_limits": 5.0
        }
        
        self.monitor = SafetyMonitor(
            constraint_names=self.constraint_names,
            constraint_thresholds=self.constraint_thresholds,
            buffer_size=100,
            enable_emergency_stop=True,
            enable_predictive_monitoring=True,
            monitoring_frequency=10.0
        )
    
    def test_monitor_initialization(self):
        """Test safety monitor initialization."""
        assert self.monitor.constraint_names == self.constraint_names
        assert self.monitor.constraint_thresholds == self.constraint_thresholds
        assert self.monitor.monitoring_frequency == 10.0
        assert self.monitor.emergency_controller is not None
        assert self.monitor.enable_predictive_monitoring is True
        
        assert len(self.monitor.safety_events) == 0
        assert len(self.monitor.current_violations) == 0
    
    def test_monitor_update_with_safe_values(self):
        """Test monitor update with safe constraint values."""
        constraint_dict = {
            "collision_avoidance": 0.05,  # Below threshold
            "force_limits": 3.0           # Below threshold
        }
        
        assessment = self.monitor.update(constraint_dict)
        
        assert isinstance(assessment, dict)
        assert not assessment["violations_detected"]
        assert assessment["active_violations"] == 0
        assert assessment["overall_safety_level"] == SafetyLevel.SAFE
        assert not assessment["emergency_active"]
    
    def test_monitor_update_with_violations(self):
        """Test monitor update with constraint violations."""
        constraint_dict = {
            "collision_avoidance": 0.15,  # Above threshold
            "force_limits": 2.0           # Below threshold
        }
        
        assessment = self.monitor.update(constraint_dict)
        
        assert assessment["violations_detected"]
        assert assessment["active_violations"] == 1
        assert assessment["overall_safety_level"] != SafetyLevel.SAFE
        
        # Check that violation was logged
        assert len(self.monitor.safety_events) == 1
        assert len(self.monitor.current_violations) == 1
        assert "collision_avoidance" in self.monitor.current_violations
    
    def test_violation_resolution(self):
        """Test violation resolution detection."""
        # First, create a violation
        constraint_dict_violation = {"collision_avoidance": 0.15, "force_limits": 2.0}
        self.monitor.update(constraint_dict_violation)
        
        assert len(self.monitor.current_violations) == 1
        
        # Then resolve the violation
        constraint_dict_safe = {"collision_avoidance": 0.05, "force_limits": 2.0}
        assessment = self.monitor.update(constraint_dict_safe)
        
        assert len(self.monitor.current_violations) == 0
        assert not assessment["violations_detected"]
        
        # Check that violation was marked as resolved
        violation_event = self.monitor.safety_events[0]
        assert violation_event.resolved
        assert violation_event.resolution_time is not None
    
    def test_predictive_monitoring(self):
        """Test predictive safety monitoring."""
        # Add trend data that should trigger predictive warnings
        for i in range(10):
            value = 0.07 + i * 0.005  # Gradually increasing toward threshold
            constraint_dict = {"collision_avoidance": value, "force_limits": 2.0}
            assessment = self.monitor.update(constraint_dict)
        
        # Should have risk predictions
        assert "risk_predictions" in assessment
        assert len(assessment["risk_predictions"]) > 0
        
        # Check for increasing risk
        collision_risk = assessment["risk_predictions"].get("collision_avoidance", {})
        if collision_risk:
            assert "risk_score" in collision_risk
            assert collision_risk["risk_score"] >= 0
    
    def test_emergency_stop_integration(self):
        """Test integration with emergency stop controller."""
        # Create critical violation that should trigger emergency stop
        constraint_dict_critical = {"collision_avoidance": 0.2, "force_limits": 10.0}
        
        assessment = self.monitor.update(constraint_dict_critical)
        
        # Emergency stop should be triggered
        assert assessment["emergency_triggered"] or assessment["emergency_active"]
        if assessment["emergency_triggered"]:
            assert len(assessment["emergency_reason"]) > 0
    
    def test_safety_metrics_tracking(self):
        """Test safety metrics computation and tracking."""
        # Update with mixed safe and unsafe values
        test_data = [
            {"collision_avoidance": 0.05, "force_limits": 2.0},  # Safe
            {"collision_avoidance": 0.15, "force_limits": 2.0},  # Violation
            {"collision_avoidance": 0.08, "force_limits": 2.0},  # Safe
            {"collision_avoidance": 0.12, "force_limits": 8.0},  # Two violations
            {"collision_avoidance": 0.03, "force_limits": 1.0},  # Safe
        ]
        
        for constraint_dict in test_data:
            self.monitor.update(constraint_dict)
        
        # Check metrics
        metrics = self.monitor.metrics
        assert metrics.total_timesteps == 5
        assert metrics.total_violations > 0
        assert metrics.safety_rate < 1.0  # Should be less than perfect due to violations
        
        # Check assessment metrics
        final_assessment = self.monitor.update({"collision_avoidance": 0.02, "force_limits": 1.0})
        assessment_metrics = final_assessment["metrics"]
        
        assert "total_timesteps" in assessment_metrics
        assert "total_violations" in assessment_metrics
        assert "safety_rate" in assessment_metrics
    
    def test_violation_callback_system(self):
        """Test violation callback system."""
        callback_called = False
        violation_data = None
        
        def test_callback(event: SafetyEvent):
            nonlocal callback_called, violation_data
            callback_called = True
            violation_data = event
        
        # Add callback
        self.monitor.add_violation_callback(test_callback)
        
        # Create violation
        constraint_dict = {"collision_avoidance": 0.15, "force_limits": 2.0}
        self.monitor.update(constraint_dict)
        
        # Check that callback was called
        assert callback_called
        assert violation_data is not None
        assert violation_data.event_type == ViolationType.COLLISION
        assert violation_data.violation_value == 0.15
    
    def test_safety_report_generation(self):
        """Test comprehensive safety report generation."""
        # Create some history
        test_data = [
            {"collision_avoidance": 0.05, "force_limits": 2.0},
            {"collision_avoidance": 0.15, "force_limits": 2.0},  # Violation
            {"collision_avoidance": 0.08, "force_limits": 2.0},
        ]
        
        for constraint_dict in test_data:
            self.monitor.update(constraint_dict)
        
        # Generate report
        report = self.monitor.get_safety_report()
        
        assert isinstance(report, dict)
        assert "timestamp" in report
        assert "overall_metrics" in report
        assert "constraint_buffer_stats" in report
        assert "recent_violations" in report
        assert "active_violations" in report
        assert "constraint_trends" in report
        
        # Check that violation is included in recent violations
        recent_violations = report["recent_violations"]
        assert len(recent_violations) > 0
        
        # Check constraint trends
        trends = report["constraint_trends"]
        assert len(trends) == len(self.constraint_names)
        for name in self.constraint_names:
            assert name in trends
    
    def test_real_time_monitoring_thread(self):
        """Test real-time monitoring thread functionality."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        assert self.monitor.monitoring_active
        assert self.monitor.monitoring_thread is not None
        assert self.monitor.monitoring_thread.is_alive()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        assert not self.monitor.monitoring_active
        # Thread should terminate
        time.sleep(0.1)
        assert not self.monitor.monitoring_thread.is_alive()
    
    def test_safety_log_save_load(self, tmp_path):
        """Test safety log saving and loading."""
        # Create some safety events
        test_data = [
            {"collision_avoidance": 0.15, "force_limits": 2.0},  # Violation
            {"collision_avoidance": 0.08, "force_limits": 7.0},  # Violation
            {"collision_avoidance": 0.05, "force_limits": 2.0},  # Safe
        ]
        
        for constraint_dict in test_data:
            self.monitor.update(constraint_dict)
        
        # Save safety log
        log_file = tmp_path / "safety_log.json"
        self.monitor.save_safety_log(str(log_file))
        
        assert log_file.exists()
        
        # Create new monitor and load log
        new_monitor = SafetyMonitor(
            constraint_names=self.constraint_names,
            constraint_thresholds=self.constraint_thresholds
        )
        
        assert len(new_monitor.safety_events) == 0  # Initially empty
        
        new_monitor.load_safety_log(str(log_file))
        
        # Should have loaded the events
        assert len(new_monitor.safety_events) == len(self.monitor.safety_events)
        
        # Check event content
        for original, loaded in zip(self.monitor.safety_events, new_monitor.safety_events):
            assert original.event_type == loaded.event_type
            assert original.severity == loaded.severity
            assert abs(original.violation_value - loaded.violation_value) < 1e-6
    
    def test_monitor_reset(self):
        """Test safety monitor reset functionality."""
        # Create some state
        test_data = [
            {"collision_avoidance": 0.15, "force_limits": 2.0},  # Violation
            {"collision_avoidance": 0.08, "force_limits": 2.0},  # Safe
        ]
        
        for constraint_dict in test_data:
            self.monitor.update(constraint_dict)
        
        # Should have some events and metrics
        assert len(self.monitor.safety_events) > 0
        assert self.monitor.metrics.total_timesteps > 0
        
        # Reset monitor
        self.monitor.reset()
        
        # Should be back to initial state
        assert len(self.monitor.safety_events) == 0
        assert len(self.monitor.current_violations) == 0
        assert self.monitor.metrics.total_timesteps == 0
        assert self.monitor.metrics.total_violations == 0


@pytest.mark.parametrize("violation_type,threshold,test_value", [
    (ViolationType.COLLISION, 0.1, 0.15),
    (ViolationType.FORCE_LIMIT, 50.0, 75.0),
    (ViolationType.JOINT_LIMIT, 0.2, 0.3),
])
def test_safety_monitor_with_different_violations(violation_type, threshold, test_value):
    """Test safety monitor with different types of violations."""
    constraint_name = violation_type.value
    
    monitor = SafetyMonitor(
        constraint_names=[constraint_name],
        constraint_thresholds={constraint_name: threshold}
    )
    
    # Test safe value
    safe_dict = {constraint_name: threshold * 0.8}
    assessment_safe = monitor.update(safe_dict)
    assert not assessment_safe["violations_detected"]
    
    # Test violation
    violation_dict = {constraint_name: test_value}
    assessment_violation = monitor.update(violation_dict)
    assert assessment_violation["violations_detected"]
    assert len(monitor.current_violations) == 1


def test_safety_monitor_performance():
    """Test safety monitor performance with high update frequency."""
    monitor = SafetyMonitor(
        constraint_names=["test_constraint"],
        constraint_thresholds={"test_constraint": 1.0},
        buffer_size=1000
    )
    
    num_updates = 1000
    start_time = time.time()
    
    for i in range(num_updates):
        constraint_dict = {"test_constraint": 0.5 + 0.001 * i}
        monitor.update(constraint_dict)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Should be able to handle many updates quickly
    assert duration < 1.0, f"Performance test failed: {duration:.3f}s for {num_updates} updates"
    
    # Check that all updates were processed
    assert monitor.metrics.total_timesteps == num_updates
    
    # Buffer should contain recent data
    assert monitor.constraint_buffer.current_size == min(num_updates, monitor.constraint_buffer.buffer_size)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])