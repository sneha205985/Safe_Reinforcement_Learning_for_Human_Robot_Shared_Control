"""
Unit tests for Lagrangian optimization methods.

Tests Lagrangian formulation, dual variable updates, primal-dual optimization,
and convergence criteria for CPO implementation.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.lagrangian import (
    LagrangianOptimizer, 
    LagrangianState,
    ConvergenceCriteria
)
from core.policy import SafePolicy
from core.constraints import CollisionConstraint, ForceConstraint, ConstraintManager
from utils.math_utils import finite_difference_gradient


class MockPolicy:
    """Mock policy for testing Lagrangian optimizer."""
    
    def __init__(self, state_dim=6, action_dim=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_net = MockPolicyNetwork(state_dim, action_dim)
        
    def compute_policy_gradient(self, states, actions, advantages):
        # Simple quadratic loss for testing
        mean_advantage = advantages.mean()
        return -mean_advantage + 0.1 * torch.norm(actions)**2 / len(actions)
    
    def compute_constraint_gradient(self, states, actions):
        # Simple constraint gradient
        return torch.sum(actions**2) / len(actions)


class MockPolicyNetwork:
    """Mock policy network for testing."""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = [torch.randn(state_dim, action_dim, requires_grad=True)]
        
    def parameters(self):
        return self.params
    
    def get_distribution(self, states):
        from torch.distributions import Normal
        batch_size = states.shape[0]
        mean = torch.zeros(batch_size, self.action_dim)
        std = torch.ones(batch_size, self.action_dim)
        return Normal(mean, std)


class TestConvergenceCriteria:
    """Test convergence criteria for Lagrangian optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.criteria = ConvergenceCriteria(
            gradient_tolerance=1e-4,
            constraint_tolerance=1e-3,
            max_iterations=50
        )
    
    def test_convergence_initialization(self):
        """Test convergence criteria initialization."""
        assert self.criteria.gradient_tolerance == 1e-4
        assert self.criteria.constraint_tolerance == 1e-3
        assert self.criteria.max_iterations == 50
        assert self.criteria.min_iterations == 10
    
    def test_max_iterations_convergence(self):
        """Test convergence by maximum iterations."""
        # Create state that exceeds max iterations
        state = LagrangianState(
            iteration=51,
            primal_variables=torch.zeros(10),
            dual_variables=torch.zeros(2),
            objective_value=1.0,
            constraint_values=torch.tensor([0.1, 0.05]),
            lagrangian_value=0.9,
            kl_divergence=0.01,
            step_size=0.1,
            convergence_error=1e-3
        )
        
        converged, reason = self.criteria.check_convergence(state)
        assert converged
        assert reason == "max_iterations_reached"
    
    def test_gradient_tolerance_convergence(self):
        """Test convergence by gradient tolerance."""
        state = LagrangianState(
            iteration=15,
            primal_variables=torch.zeros(10),
            dual_variables=torch.zeros(2),
            objective_value=1.0,
            constraint_values=torch.tensor([0.1, 0.05]),
            lagrangian_value=0.9,
            kl_divergence=0.01,
            step_size=0.1,
            convergence_error=1e-5  # Below tolerance
        )
        
        converged, reason = self.criteria.check_convergence(state)
        assert converged
        assert reason == "gradient_tolerance_satisfied"
    
    def test_min_iterations_not_reached(self):
        """Test that convergence is not declared before minimum iterations."""
        state = LagrangianState(
            iteration=5,  # Below minimum
            primal_variables=torch.zeros(10),
            dual_variables=torch.zeros(2),
            objective_value=1.0,
            constraint_values=torch.tensor([0.0001, 0.0001]),
            lagrangian_value=0.9,
            kl_divergence=0.001,
            step_size=0.1,
            convergence_error=1e-6  # Very small
        )
        
        converged, reason = self.criteria.check_convergence(state)
        assert not converged
        assert reason == "min_iterations_not_reached"
    
    def test_constraint_and_gradient_convergence(self):
        """Test convergence by both constraint and gradient satisfaction."""
        state = LagrangianState(
            iteration=15,
            primal_variables=torch.zeros(10),
            dual_variables=torch.zeros(2),
            objective_value=1.0,
            constraint_values=torch.tensor([0.0001, 0.0002]),  # Small constraints
            lagrangian_value=0.9,
            kl_divergence=0.01,
            step_size=0.1,
            convergence_error=1e-3  # Larger than gradient tolerance but within 10x
        )
        
        converged, reason = self.criteria.check_convergence(state)
        assert converged
        assert reason == "constraint_and_gradient_tolerance_satisfied"


class TestLagrangianOptimizer:
    """Test Lagrangian optimizer implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create constraints
        collision_constraint = CollisionConstraint(
            min_distance=0.1,
            human_position_idx=(0, 3),
            robot_position_idx=(3, 6)
        )
        force_constraint = ForceConstraint(max_force=5.0)
        
        self.constraint_manager = ConstraintManager([
            collision_constraint,
            force_constraint
        ])
        
        # Create mock policy
        self.policy = MockPolicy(state_dim=8, action_dim=6)
        
        # Create optimizer
        self.optimizer = LagrangianOptimizer(
            policy=self.policy,
            constraint_manager=self.constraint_manager,
            dual_lr=0.1,
            trust_region_size=0.01,
            convergence_criteria=ConvergenceCriteria(max_iterations=10)
        )
    
    def test_optimizer_initialization(self):
        """Test Lagrangian optimizer initialization."""
        assert self.optimizer.policy is self.policy
        assert self.optimizer.constraint_manager is self.constraint_manager
        assert self.optimizer.dual_lr == 0.1
        assert self.optimizer.trust_region_size == 0.01
        
        # Check dual variables initialization
        num_constraints = len(self.constraint_manager.constraints)
        assert self.optimizer.dual_variables.shape == (num_constraints,)
        assert (self.optimizer.dual_variables >= 0).all()
        
        # Check history
        assert len(self.optimizer.optimization_history) == 0
        assert self.optimizer.current_iteration == 0
    
    def test_lagrangian_computation(self):
        """Test Lagrangian L(θ,λ) = J(θ) - λᵀg(θ) computation."""
        batch_size = 16
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6)
        advantages = torch.randn(batch_size)
        
        lagrangian, info = self.optimizer.compute_lagrangian(states, actions, advantages)
        
        # Check return types
        assert isinstance(lagrangian, torch.Tensor)
        assert lagrangian.dim() == 0  # Scalar
        assert isinstance(info, dict)
        
        # Check info contents
        assert "objective" in info
        assert "constraint_values" in info
        assert "constraint_penalty" in info
        assert "dual_variables" in info
        
        # Check shapes
        num_constraints = len(self.constraint_manager.constraints)
        assert info["constraint_values"].shape == (num_constraints,)
        assert info["dual_variables"].shape == (num_constraints,)
        
        # Check that Lagrangian = objective - penalty
        expected_lagrangian = info["objective"] - info["constraint_penalty"]
        assert torch.allclose(lagrangian, expected_lagrangian, atol=1e-6)
    
    def test_dual_variable_update(self):
        """Test dual variable update λ_{k+1} = max(0, λ_k + α_dual * g(θ_k))."""
        num_constraints = len(self.constraint_manager.constraints)
        
        # Set initial dual variables
        self.optimizer.dual_variables = torch.tensor([0.1, 0.2])
        initial_duals = self.optimizer.dual_variables.clone()
        
        # Constraint violations (positive values)
        constraint_values = torch.tensor([0.05, 0.1])
        
        updated_duals = self.optimizer.dual_update(constraint_values)
        
        # Check that duals increased (since constraints are violated)
        assert (updated_duals >= initial_duals).all()
        
        # Check bounds
        assert (updated_duals >= self.optimizer.dual_min).all()
        assert (updated_duals <= self.optimizer.dual_max).all()
        
        # Test with negative constraint values (satisfied constraints)
        constraint_values_negative = torch.tensor([-0.05, -0.1])
        
        # Reset duals
        self.optimizer.dual_variables = torch.tensor([0.5, 0.3])
        initial_duals_2 = self.optimizer.dual_variables.clone()
        
        updated_duals_2 = self.optimizer.dual_update(constraint_values_negative)
        
        # Duals should decrease or stay bounded at minimum
        assert (updated_duals_2 <= initial_duals_2 + 1e-6).all()  # Allow for numerical precision
    
    def test_constraint_gradient_computation(self):
        """Test constraint gradient computation ∇_θ g(θ)."""
        batch_size = 8
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6)
        
        constraint_grads = self.optimizer.compute_constraint_gradients(states, actions)
        
        num_constraints = len(self.constraint_manager.constraints)
        param_dim = sum(p.numel() for p in self.policy.policy_net.parameters())
        
        assert constraint_grads.shape == (num_constraints, param_dim)
        assert torch.isfinite(constraint_grads).all()
    
    def test_optimization_step(self):
        """Test single optimization step."""
        batch_size = 32
        states = torch.randn(batch_size, 8)
        
        # Set up states to avoid extreme constraint violations
        states[:, 0:3] = torch.randn(batch_size, 3) * 0.1  # Small human positions
        states[:, 3:6] = torch.randn(batch_size, 3) * 0.1 + 0.5  # Robot positions away from human
        
        actions = torch.randn(batch_size, 6) * 0.1  # Small actions
        advantages = torch.randn(batch_size)
        
        # Perform optimization step
        initial_iteration = self.optimizer.current_iteration
        state = self.optimizer.optimize_step(states, actions, advantages)
        
        # Check that state is returned
        assert isinstance(state, LagrangianState)
        assert state.iteration == initial_iteration + 1
        
        # Check state contents
        assert torch.isfinite(state.lagrangian_value)
        assert torch.isfinite(state.objective_value)
        assert torch.isfinite(state.constraint_values).all()
        assert torch.isfinite(state.dual_variables).all()
        
        # Check that history is updated
        assert len(self.optimizer.optimization_history) == 1
        assert self.optimizer.optimization_history[0] is state
    
    def test_full_optimization(self):
        """Test complete optimization process."""
        batch_size = 32
        states = torch.randn(batch_size, 8)
        
        # Set up reasonable states
        states[:, 0:3] = 0.0  # Human at origin
        states[:, 3:6] = torch.tensor([1.0, 0.0, 0.0])  # Robot at safe distance
        
        actions = torch.randn(batch_size, 6) * 0.1
        advantages = torch.ones(batch_size)  # Positive advantages
        
        # Run optimization
        converged, reason, history = self.optimizer.optimize(
            states, actions, advantages, max_iterations=5
        )
        
        # Should converge or reach max iterations
        assert isinstance(converged, bool)
        assert isinstance(reason, str)
        assert isinstance(history, list)
        assert len(history) > 0
        assert len(history) <= 5
        
        # Check that dual variables were updated
        assert not torch.allclose(
            self.optimizer.dual_variables,
            torch.zeros_like(self.optimizer.dual_variables),
            atol=1e-6
        )
    
    def test_optimization_statistics(self):
        """Test optimization statistics collection."""
        # Run a few optimization steps
        batch_size = 16
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6) * 0.1
        advantages = torch.randn(batch_size)
        
        for _ in range(3):
            self.optimizer.optimize_step(states, actions, advantages)
        
        stats = self.optimizer.get_optimization_statistics()
        
        assert isinstance(stats, dict)
        assert "total_iterations" in stats
        assert "final_objective" in stats
        assert "final_lagrangian" in stats
        assert "final_constraint_max" in stats
        assert "dual_variable_changes" in stats
        
        assert stats["total_iterations"] == 3
        assert len(stats["dual_variable_changes"]) == 2  # 3-1 changes
    
    def test_optimizer_reset(self):
        """Test optimizer reset functionality."""
        # Run some optimization steps
        batch_size = 8
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6)
        advantages = torch.randn(batch_size)
        
        self.optimizer.optimize_step(states, actions, advantages)
        self.optimizer.optimize_step(states, actions, advantages)
        
        # Check that state has changed
        assert self.optimizer.current_iteration == 2
        assert len(self.optimizer.optimization_history) == 2
        assert not torch.allclose(self.optimizer.dual_variables, torch.zeros_like(self.optimizer.dual_variables))
        
        # Reset optimizer
        self.optimizer.reset()
        
        # Check that everything is reset
        assert self.optimizer.current_iteration == 0
        assert len(self.optimizer.optimization_history) == 0
        assert torch.allclose(self.optimizer.dual_variables, torch.zeros_like(self.optimizer.dual_variables))
    
    def test_save_load_optimizer(self, tmp_path):
        """Test saving and loading optimizer state."""
        # Run some optimization
        batch_size = 8
        states = torch.randn(batch_size, 8)
        actions = torch.randn(batch_size, 6)
        advantages = torch.randn(batch_size)
        
        self.optimizer.optimize_step(states, actions, advantages)
        
        # Save state
        save_path = tmp_path / "optimizer_state.pt"
        self.optimizer.save_state(str(save_path))
        
        assert save_path.exists()
        
        # Modify optimizer
        original_dual_vars = self.optimizer.dual_variables.clone()
        self.optimizer.dual_variables += 1.0
        
        # Load state
        self.optimizer.load_state(str(save_path))
        
        # Check that state was restored
        assert torch.allclose(self.optimizer.dual_variables, original_dual_vars)


class TestLagrangianNumericalAccuracy:
    """Test numerical accuracy of Lagrangian computations."""
    
    def setup_method(self):
        """Set up simple test case for numerical verification."""
        # Single constraint for simplicity
        force_constraint = ForceConstraint(max_force=1.0)
        self.constraint_manager = ConstraintManager([force_constraint])
        
        # Simple policy
        self.policy = MockPolicy(state_dim=4, action_dim=3)
        
        self.optimizer = LagrangianOptimizer(
            policy=self.policy,
            constraint_manager=self.constraint_manager,
            dual_lr=0.01,
            dual_regularization=0.0  # Remove regularization for cleaner tests
        )
    
    def test_dual_update_numerical(self):
        """Test dual update against manual calculation."""
        # Set known dual variables and constraint values
        self.optimizer.dual_variables = torch.tensor([0.5])
        constraint_values = torch.tensor([0.1])  # Violation
        
        # Manual calculation: λ_new = max(0, λ_old + α * g)
        expected_dual = max(0.0, 0.5 + 0.01 * 0.1)
        expected_dual = min(expected_dual, self.optimizer.dual_max)
        expected_dual = max(expected_dual, self.optimizer.dual_min)
        
        updated_dual = self.optimizer.dual_update(constraint_values)
        
        assert torch.allclose(updated_dual, torch.tensor([expected_dual]), atol=1e-6)
    
    def test_lagrangian_formula(self):
        """Test Lagrangian formula L = J - λᵀg."""
        batch_size = 4
        states = torch.randn(batch_size, 4)
        actions = torch.randn(batch_size, 3) * 0.1  # Small actions to avoid extreme values
        advantages = torch.ones(batch_size)
        
        # Set known dual variables
        self.optimizer.dual_variables = torch.tensor([0.3])
        
        # Compute Lagrangian
        lagrangian, info = self.optimizer.compute_lagrangian(states, actions, advantages)
        
        # Manual verification
        expected_lagrangian = info["objective"] - torch.dot(
            self.optimizer.dual_variables, info["constraint_values"]
        )
        
        assert torch.allclose(lagrangian, expected_lagrangian, atol=1e-6)


@pytest.mark.parametrize("dual_lr", [0.01, 0.1, 1.0])
def test_dual_learning_rates(dual_lr):
    """Test optimizer with different dual learning rates."""
    force_constraint = ForceConstraint(max_force=1.0)
    constraint_manager = ConstraintManager([force_constraint])
    policy = MockPolicy()
    
    optimizer = LagrangianOptimizer(
        policy=policy,
        constraint_manager=constraint_manager,
        dual_lr=dual_lr,
        convergence_criteria=ConvergenceCriteria(max_iterations=3)
    )
    
    # Run optimization
    states = torch.randn(8, 6)
    actions = torch.randn(8, 4) * 0.1
    advantages = torch.randn(8)
    
    converged, reason, history = optimizer.optimize(states, actions, advantages)
    
    # Should complete without errors
    assert len(history) > 0
    assert reason in ["max_iterations_reached", "gradient_tolerance_satisfied"]


@pytest.mark.parametrize("trust_region_size", [0.001, 0.01, 0.1])
def test_trust_region_sizes(trust_region_size):
    """Test optimizer with different trust region sizes."""
    force_constraint = ForceConstraint(max_force=2.0)
    constraint_manager = ConstraintManager([force_constraint])
    policy = MockPolicy()
    
    optimizer = LagrangianOptimizer(
        policy=policy,
        constraint_manager=constraint_manager,
        trust_region_size=trust_region_size,
        convergence_criteria=ConvergenceCriteria(max_iterations=2)
    )
    
    states = torch.randn(8, 6)
    actions = torch.randn(8, 4) * 0.1
    advantages = torch.randn(8)
    
    # Should run without errors
    state = optimizer.optimize_step(states, actions, advantages)
    assert isinstance(state, LagrangianState)


def test_constraint_linearization_approximation():
    """Test constraint linearization g(θ) ≈ g(θ_k) + ∇g(θ_k)ᵀ(θ - θ_k)."""
    # This test verifies that the constraint linearization used in CPO
    # is a reasonable approximation for small parameter changes
    
    force_constraint = ForceConstraint(max_force=2.0)
    constraint_manager = ConstraintManager([force_constraint])
    policy = SafePolicy(state_dim=4, action_dim=3, hidden_sizes=[8])
    
    optimizer = LagrangianOptimizer(
        policy=policy,
        constraint_manager=constraint_manager,
        trust_region_size=0.001  # Small trust region for good linearization
    )
    
    states = torch.randn(4, 4)
    actions = torch.randn(4, 3)
    
    # Get current constraint values and gradients
    constraint_grads = optimizer.compute_constraint_gradients(states, actions)
    current_constraints = constraint_manager.evaluate_all(states, actions)
    current_constraint_values = torch.stack([current_constraints[cid].mean() 
                                           for cid in sorted(current_constraints.keys())])
    
    # Small parameter perturbation
    param_change = torch.randn_like(constraint_grads[0]) * 0.001
    
    # Predicted change using linearization
    predicted_change = torch.mv(constraint_grads, param_change)
    
    # The linearization should be reasonable for small changes
    assert torch.isfinite(predicted_change).all()
    assert torch.norm(predicted_change) < 1.0  # Should be small for small param changes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])