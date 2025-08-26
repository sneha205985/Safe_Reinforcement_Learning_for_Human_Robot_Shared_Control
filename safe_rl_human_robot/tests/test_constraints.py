"""
Unit tests for safety constraint implementations.

Tests constraint evaluation, gradient computation, and violation detection
for collision, joint limit, and force constraints.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.constraints import (
    SafetyConstraint, 
    CollisionConstraint, 
    JointLimitConstraint,
    ForceConstraint,
    ConstraintManager,
    ConstraintViolationError
)
from utils.math_utils import finite_difference_gradient


class TestSafetyConstraint:
    """Test base SafetyConstraint class."""
    
    def test_constraint_base_functionality(self):
        """Test basic constraint functionality."""
        
        # Create a simple test constraint
        class SimpleConstraint(SafetyConstraint):
            def evaluate(self, state, action):
                # g(s,a) = ||a||² - 1 ≤ 0
                return torch.norm(action, dim=1)**2 - 1.0
            
            def gradient(self, state, action):
                state.requires_grad_(True)
                action.requires_grad_(True)
                g = self.evaluate(state, action).sum()
                grad_state = torch.autograd.grad(g, state, create_graph=True)[0]
                grad_action = torch.autograd.grad(g, action, create_graph=True)[0]
                return grad_state, grad_action
        
        constraint = SimpleConstraint("test_constraint")
        
        # Test safe case
        state = torch.randn(5, 4)
        action = torch.tensor([[0.5, 0.5], [0.3, 0.4], [0.1, 0.2], [0.6, 0.3], [0.2, 0.1]])
        
        violations = constraint.is_violated(state, action)
        assert not violations.any(), "Should not have violations for small actions"
        
        # Test violation case
        action_large = torch.tensor([[2.0, 2.0], [1.5, 1.5], [1.2, 1.3], [2.1, 0.5], [1.8, 1.1]])
        violations_large = constraint.is_violated(state, action_large)
        assert violations_large.all(), "Should have violations for large actions"
        
        # Test gradient computation
        grad_state, grad_action = constraint.gradient(state, action)
        assert grad_state.shape == state.shape
        assert grad_action.shape == action.shape
        
        # Test violation logging
        constraint.log_violation(0.5)
        stats = constraint.get_violation_statistics()
        assert stats["count"] == 1
        assert stats["recent"] == 0.5


class TestCollisionConstraint:
    """Test collision avoidance constraint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.constraint = CollisionConstraint(
            min_distance=0.1,
            human_position_idx=(0, 3),
            robot_position_idx=(3, 6)
        )
    
    def test_collision_evaluation(self):
        """Test collision constraint evaluation."""
        batch_size = 10
        
        # Create states with human and robot positions
        states = torch.randn(batch_size, 10)  # 10-dim state
        states[:, 0:3] = torch.tensor([0.0, 0.0, 1.0])  # Human position
        states[:, 3:6] = torch.tensor([0.5, 0.0, 1.0])  # Robot position (safe distance)
        
        # Small actions (safe)
        actions = torch.randn(batch_size, 3) * 0.01
        
        constraint_values = self.constraint.evaluate(states, actions)
        
        # Should be negative (safe) for distant positions
        assert (constraint_values < 0).all(), "Distant positions should be safe"
        
        # Test collision scenario
        states_collision = states.clone()
        states_collision[:, 3:6] = states_collision[:, 0:3] + 0.05  # Very close
        
        constraint_values_collision = self.constraint.evaluate(states_collision, actions)
        
        # Should be positive (violation) for close positions
        assert (constraint_values_collision > 0).all(), "Close positions should violate constraint"
    
    def test_collision_gradients(self):
        """Test collision constraint gradients."""
        batch_size = 3
        state_dim = 10
        action_dim = 6
        
        states = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, action_dim)
        
        # Compute analytical gradients
        grad_state, grad_action = self.constraint.gradient(states, actions)
        
        assert grad_state.shape == states.shape
        assert grad_action.shape == actions.shape
        assert not torch.isnan(grad_state).any()
        assert not torch.isnan(grad_action).any()
        
        # Numerical gradient check for a single sample
        state_single = states[0:1].clone()
        action_single = actions[0:1].clone()
        
        def constraint_func_state(s):
            return self.constraint.evaluate(s, action_single)[0]
        
        def constraint_func_action(a):
            return self.constraint.evaluate(state_single, a)[0]
        
        # Check state gradient
        grad_state_analytical = self.constraint.gradient(state_single, action_single)[0][0]
        grad_state_numerical = finite_difference_gradient(constraint_func_state, state_single[0])
        
        state_grad_error = torch.norm(grad_state_analytical - grad_state_numerical)
        assert state_grad_error < 1e-3, f"State gradient error too large: {state_grad_error}"
        
        # Check action gradient
        grad_action_analytical = self.constraint.gradient(state_single, action_single)[1][0]
        grad_action_numerical = finite_difference_gradient(constraint_func_action, action_single[0])
        
        action_grad_error = torch.norm(grad_action_analytical - grad_action_numerical)
        assert action_grad_error < 1e-3, f"Action gradient error too large: {action_grad_error}"


class TestJointLimitConstraint:
    """Test joint limit constraint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.joint_limits = {
            "joint_1": (-1.5, 1.5),
            "joint_2": (-2.0, 2.0),
            "joint_3": (-1.0, 1.0)
        }
        self.joint_indices = {
            "joint_1": 0,
            "joint_2": 1, 
            "joint_3": 2
        }
        self.constraint = JointLimitConstraint(self.joint_limits, self.joint_indices)
    
    def test_joint_limit_evaluation(self):
        """Test joint limit constraint evaluation."""
        batch_size = 5
        
        # Safe joint positions
        states_safe = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 0.0],  # All joints at center
            [1.0, 1.5, 0.5, 0.0, 0.0],  # Within limits
            [-1.0, -1.5, -0.8, 0.0, 0.0],  # Within limits (negative)
            [1.4, 1.9, 0.9, 0.0, 0.0],  # Close to limits but safe
            [0.5, 0.0, -0.5, 0.0, 0.0]   # Safe positions
        ])
        
        actions_safe = torch.zeros(batch_size, 5)  # No movement
        
        constraint_values = self.constraint.evaluate(states_safe, actions_safe)
        assert (constraint_values <= 0).all(), "Safe positions should not violate constraints"
        
        # Violating joint positions
        states_violation = torch.tensor([
            [2.0, 0.0, 0.0, 0.0, 0.0],   # joint_1 exceeds upper limit
            [0.0, -2.5, 0.0, 0.0, 0.0],  # joint_2 exceeds lower limit
            [0.0, 0.0, 1.5, 0.0, 0.0],   # joint_3 exceeds upper limit
            [-2.0, 2.5, -1.2, 0.0, 0.0], # Multiple violations
            [1.8, 2.2, 1.1, 0.0, 0.0]    # Multiple violations
        ])
        
        constraint_values_violation = self.constraint.evaluate(states_violation, actions_safe)
        assert (constraint_values_violation > 0).all(), "Violating positions should exceed constraints"
    
    def test_joint_limit_with_velocity(self):
        """Test joint limits with predicted motion."""
        batch_size = 3
        
        # States near limits
        states = torch.tensor([
            [1.4, 1.9, 0.9, 0.0, 0.0],  # Close to upper limits
            [-1.4, -1.9, -0.9, 0.0, 0.0],  # Close to lower limits  
            [0.0, 0.0, 0.0, 0.0, 0.0]   # At center
        ])
        
        # Actions that would cause violations
        actions = torch.tensor([
            [2.0, 2.0, 2.0, 0.0, 0.0],   # Would exceed upper limits
            [-2.0, -2.0, -2.0, 0.0, 0.0], # Would exceed lower limits
            [0.0, 0.0, 0.0, 0.0, 0.0]    # No movement
        ])
        
        constraint_values = self.constraint.evaluate(states, actions)
        
        # First two should violate, third should be safe
        assert constraint_values[0] > 0, "Should violate upper limit"
        assert constraint_values[1] > 0, "Should violate lower limit"
        assert constraint_values[2] <= 0, "Should be safe with no movement"


class TestForceConstraint:
    """Test force/torque limit constraint."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.max_force = 10.0
        self.constraint = ForceConstraint(
            max_force=self.max_force,
            force_indices=[0, 1, 2, 3, 4, 5]  # 6-DOF force/torque
        )
    
    def test_force_evaluation(self):
        """Test force constraint evaluation."""
        batch_size = 4
        
        # Safe forces (small magnitude)
        actions_safe = torch.tensor([
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0],  # Small forces
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Single large but safe force
            [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0],  # Moderate forces
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Zero forces
        ])
        
        states = torch.randn(batch_size, 5)  # State doesn't affect force constraint
        
        constraint_values = self.constraint.evaluate(states, actions_safe)
        assert (constraint_values <= 0).all(), "Safe forces should not violate constraint"
        
        # Violating forces (large magnitude)
        actions_violation = torch.tensor([
            [15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Single large force
            [8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Combined forces exceed limit
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0],   # All forces large
            [12.0, 3.0, 2.0, 1.0, 1.0, 1.0, 0.0]   # Mixed large forces
        ])
        
        constraint_values_violation = self.constraint.evaluate(states, actions_violation)
        assert (constraint_values_violation > 0).all(), "Large forces should violate constraint"
    
    def test_force_gradients(self):
        """Test force constraint gradients."""
        batch_size = 2
        states = torch.randn(batch_size, 4)
        actions = torch.randn(batch_size, 7)
        
        grad_state, grad_action = self.constraint.gradient(states, actions)
        
        assert grad_state.shape == states.shape
        assert grad_action.shape == actions.shape
        assert not torch.isnan(grad_state).any()
        assert not torch.isnan(grad_action).any()
        
        # Force gradient should be zero w.r.t. state (force constraint independent of state)
        assert torch.allclose(grad_state, torch.zeros_like(grad_state), atol=1e-6)
        
        # Action gradient should be non-zero for non-zero forces
        if torch.norm(actions) > 1e-6:
            assert torch.norm(grad_action) > 1e-6, "Action gradient should be non-zero for non-zero actions"


class TestConstraintManager:
    """Test constraint manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collision_constraint = CollisionConstraint(min_distance=0.1)
        self.joint_constraint = JointLimitConstraint(
            {"joint_1": (-1.0, 1.0), "joint_2": (-1.0, 1.0)},
            {"joint_1": 0, "joint_2": 1}
        )
        self.force_constraint = ForceConstraint(max_force=5.0)
        
        self.manager = ConstraintManager([
            self.collision_constraint,
            self.joint_constraint, 
            self.force_constraint
        ])
    
    def test_constraint_evaluation(self):
        """Test evaluation of all constraints."""
        batch_size = 3
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6)
        
        results = self.manager.evaluate_all(states, actions)
        
        assert len(results) == 3
        assert "collision_avoidance" in results
        assert "joint_limits" in results
        assert "force_limits" in results
        
        for constraint_id, values in results.items():
            assert values.shape == (batch_size,)
            assert not torch.isnan(values).any()
    
    def test_violation_penalty(self):
        """Test violation penalty computation."""
        batch_size = 2
        states = torch.randn(batch_size, 8)
        
        # Actions that should cause violations
        actions_violation = torch.tensor([
            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # Large forces
            [8.0, 8.0, 8.0, 8.0, 8.0, 8.0]         # Large forces
        ])
        
        penalty = self.manager.compute_violation_penalty(states, actions_violation)
        
        assert penalty.shape == (batch_size,)
        assert (penalty >= 0).all(), "Penalties should be non-negative"
        assert penalty.sum() > 0, "Should have positive penalty for violations"
    
    def test_safety_checking(self):
        """Test safety status checking."""
        batch_size = 2
        states = torch.randn(batch_size, 8)
        
        # Safe actions
        actions_safe = torch.tensor([
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        ])
        
        safety_status = self.manager.check_safety(states, actions_safe)
        
        assert isinstance(safety_status, dict)
        assert len(safety_status) == 3
        
        # Actions causing violations
        actions_unsafe = torch.tensor([
            [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
        ])
        
        safety_status_unsafe = self.manager.check_safety(states, actions_unsafe)
        
        # Should have violations
        violation_count = sum(1 for safe in safety_status_unsafe.values() if not safe)
        assert violation_count > 0, "Should detect violations"
    
    def test_combined_gradients(self):
        """Test combined gradient computation."""
        batch_size = 2
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6)
        
        grad_state, grad_action = self.manager.compute_combined_gradients(states, actions)
        
        assert grad_state.shape == states.shape
        assert grad_action.shape == actions.shape
        assert not torch.isnan(grad_state).any()
        assert not torch.isnan(grad_action).any()
    
    def test_constraint_statistics(self):
        """Test constraint violation statistics."""
        # Log some violations
        self.collision_constraint.log_violation(0.1)
        self.collision_constraint.log_violation(0.05)
        self.force_constraint.log_violation(2.0)
        
        stats = self.manager.get_constraint_statistics()
        
        assert isinstance(stats, dict)
        assert "collision_avoidance" in stats
        assert "force_limits" in stats
        
        collision_stats = stats["collision_avoidance"]
        assert collision_stats["count"] == 2
        assert collision_stats["recent"] == 0.05
        
        force_stats = stats["force_limits"]
        assert force_stats["count"] == 1
        assert force_stats["recent"] == 2.0


@pytest.mark.parametrize("constraint_type", [
    "collision", "joint_limits", "force_limits"
])
def test_constraint_gradient_numerical_check(constraint_type):
    """Numerical gradient check for all constraint types."""
    
    # Set up constraint based on type
    if constraint_type == "collision":
        constraint = CollisionConstraint(min_distance=0.1)
        state_dim = 8
        action_dim = 6
    elif constraint_type == "joint_limits":
        constraint = JointLimitConstraint(
            {"joint_1": (-1.0, 1.0), "joint_2": (-1.0, 1.0)},
            {"joint_1": 0, "joint_2": 1}
        )
        state_dim = 5
        action_dim = 5
    else:  # force_limits
        constraint = ForceConstraint(max_force=5.0)
        state_dim = 4
        action_dim = 6
    
    # Single sample for numerical check
    state = torch.randn(1, state_dim)
    action = torch.randn(1, action_dim)
    
    # Analytical gradients
    grad_state_analytical, grad_action_analytical = constraint.gradient(state, action)
    grad_state_analytical = grad_state_analytical[0]
    grad_action_analytical = grad_action_analytical[0]
    
    # Numerical gradients
    def constraint_func_state(s):
        return constraint.evaluate(s.unsqueeze(0), action)[0]
    
    def constraint_func_action(a):
        return constraint.evaluate(state, a.unsqueeze(0))[0]
    
    grad_state_numerical = finite_difference_gradient(constraint_func_state, state[0])
    grad_action_numerical = finite_difference_gradient(constraint_func_action, action[0])
    
    # Check accuracy
    state_error = torch.norm(grad_state_analytical - grad_state_numerical)
    action_error = torch.norm(grad_action_analytical - grad_action_numerical)
    
    assert state_error < 1e-3, f"State gradient error for {constraint_type}: {state_error}"
    assert action_error < 1e-3, f"Action gradient error for {constraint_type}: {action_error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])