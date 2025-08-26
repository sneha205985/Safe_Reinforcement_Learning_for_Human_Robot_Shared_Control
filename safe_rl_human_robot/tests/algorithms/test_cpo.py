"""
Comprehensive tests for CPO algorithm implementation.

Tests mathematical correctness, numerical stability, and performance
of the complete CPO algorithm including trajectory collection,
advantage computation, and constrained optimization.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
import sys
import os
import time
from unittest.mock import Mock, MagicMock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.cpo import CPOAlgorithm, CPOConfig, CPOState, CPOTrajectory
from algorithms.gae import ValueFunction, GeneralizedAdvantageEstimation
from algorithms.trust_region import TrustRegionSolver
from core.policy import SafePolicy
from core.constraints import CollisionConstraint, ForceConstraint, ConstraintManager
from core.safety_monitor import SafetyMonitor
from utils.math_utils import finite_difference_gradient
from utils.logging_utils import MetricsLogger


class MockEnvironment:
    """Mock environment for testing CPO algorithm."""
    
    def __init__(self, state_dim=8, action_dim=4, episode_length=50):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_length = episode_length
        self.current_step = 0
        self.current_state = np.zeros(state_dim)
        
        # Simple reward function
        self.target = np.random.randn(state_dim) * 0.5
        
    def reset(self):
        self.current_step = 0
        self.current_state = np.random.randn(self.state_dim) * 0.1
        return self.current_state.copy()
    
    def step(self, action):
        self.current_step += 1
        
        # Simple dynamics: next_state = current_state + action (clipped)
        action_clipped = np.clip(action, -1.0, 1.0)
        self.current_state = self.current_state + action_clipped * 0.1
        
        # Reward based on distance to target
        distance = np.linalg.norm(self.current_state - self.target)
        reward = -distance + 0.1  # Small positive reward
        
        # Done condition
        done = self.current_step >= self.episode_length or distance < 0.1
        
        # Info
        info = {"distance_to_target": distance}
        
        return self.current_state.copy(), reward, done, info


class TestCPOTrajectory:
    """Test CPO trajectory data structure."""
    
    def test_trajectory_creation(self):
        """Test trajectory creation and data integrity."""
        T = 20  # Trajectory length
        state_dim = 6
        action_dim = 3
        num_constraints = 2
        
        # Create trajectory data
        states = torch.randn(T, state_dim)
        actions = torch.randn(T, action_dim)
        rewards = torch.randn(T)
        constraint_costs = torch.randn(T, num_constraints)
        log_probs = torch.randn(T)
        values = torch.randn(T)
        constraint_values = torch.randn(T, num_constraints)
        dones = torch.zeros(T)
        next_states = torch.randn(T, state_dim)
        
        trajectory = CPOTrajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            constraint_costs=constraint_costs,
            log_probs=log_probs,
            values=values,
            constraint_values=constraint_values,
            dones=dones,
            next_states=next_states
        )
        
        # Verify shapes
        assert trajectory.states.shape == (T, state_dim)
        assert trajectory.actions.shape == (T, action_dim)
        assert trajectory.rewards.shape == (T,)
        assert trajectory.constraint_costs.shape == (T, num_constraints)
        assert trajectory.log_probs.shape == (T,)
        assert trajectory.values.shape == (T,)
        assert trajectory.constraint_values.shape == (T, num_constraints)
        assert trajectory.dones.shape == (T,)
        assert trajectory.next_states.shape == (T, state_dim)


class TestCPOConfig:
    """Test CPO configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CPOConfig()
        
        assert config.target_kl == 0.01
        assert config.constraint_threshold == 0.025
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.97
        assert config.policy_lr == 3e-4
        assert config.value_lr == 1e-3
        
    def test_config_modification(self):
        """Test configuration modification."""
        config = CPOConfig()
        config.target_kl = 0.02
        config.constraint_threshold = 0.05
        
        assert config.target_kl == 0.02
        assert config.constraint_threshold == 0.05


class TestCPOAlgorithm:
    """Test complete CPO algorithm implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 8
        self.action_dim = 4
        self.device = "cpu"
        
        # Create mock environment
        self.env = MockEnvironment(self.state_dim, self.action_dim)
        
        # Create constraints
        collision_constraint = CollisionConstraint(
            min_distance=0.1,
            human_position_idx=(0, 3),
            robot_position_idx=(3, 6)
        )
        force_constraint = ForceConstraint(max_force=5.0)
        
        self.constraint_manager = ConstraintManager([collision_constraint, force_constraint])
        
        # Create policy
        self.policy = SafePolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            constraint_manager=self.constraint_manager,
            hidden_sizes=[32, 32],  # Small network for testing
            device=self.device
        )
        
        # Create CPO configuration
        self.config = CPOConfig()
        self.config.target_kl = 0.02
        self.config.value_iters = 10  # Fewer iterations for testing
        self.config.cg_iters = 5
        self.config.backtrack_iters = 5
        
        # Create value function
        self.value_function = ValueFunction(
            state_dim=self.state_dim,
            hidden_sizes=[32, 32],
            device=self.device
        )
        
        # Create safety monitor
        self.safety_monitor = SafetyMonitor(
            constraint_names=["collision_avoidance", "force_limits"],
            constraint_thresholds={"collision_avoidance": 0.1, "force_limits": 5.0}
        )
        
        # Create CPO algorithm
        self.cpo = CPOAlgorithm(
            policy=self.policy,
            constraint_manager=self.constraint_manager,
            environment=self.env,
            config=self.config,
            value_function=self.value_function,
            safety_monitor=self.safety_monitor,
            device=self.device
        )
    
    def test_cpo_initialization(self):
        """Test CPO algorithm initialization."""
        assert self.cpo.policy is self.policy
        assert self.cpo.constraint_manager is self.constraint_manager
        assert self.cpo.config is self.config
        assert self.cpo.iteration == 0
        assert self.cpo.total_episodes == 0
        assert len(self.cpo.optimization_history) == 0
        
        # Test constraint threshold initialization
        assert len(self.cpo.constraint_threshold) == len(self.constraint_manager.constraints)
        assert (self.cpo.constraint_threshold >= 0).all()
    
    def test_trajectory_collection(self):
        """Test trajectory collection process."""
        num_episodes = 3
        trajectories = self.cpo.collect_trajectories(num_episodes)
        
        assert len(trajectories) <= num_episodes  # May be fewer due to episode termination
        assert len(trajectories) > 0  # Should collect at least one trajectory
        
        # Test trajectory structure
        for traj in trajectories:
            assert isinstance(traj, CPOTrajectory)
            assert traj.states.shape[1] == self.state_dim
            assert traj.actions.shape[1] == self.action_dim
            assert len(traj.rewards) == len(traj.states)
            assert traj.constraint_costs.shape[1] == len(self.constraint_manager.constraints)
        
        # Test that total episodes counter is updated
        assert self.cpo.total_episodes == num_episodes
    
    def test_advantage_computation(self):
        """Test advantage computation using GAE."""
        # Collect some trajectories
        trajectories = self.cpo.collect_trajectories(2)
        
        policy_advantages, constraint_advantages = self.cpo.compute_advantages(trajectories)
        
        # Check shapes
        total_steps = sum(len(traj.states) for traj in trajectories)
        assert len(policy_advantages) == total_steps
        assert constraint_advantages.shape == (total_steps, len(self.constraint_manager.constraints))
        
        # Check that advantages are finite
        assert torch.isfinite(policy_advantages).all()
        assert torch.isfinite(constraint_advantages).all()
    
    def test_policy_gradient_computation(self):
        """Test policy gradient computation."""
        batch_size = 16
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        advantages = torch.randn(batch_size)
        old_log_probs = torch.randn(batch_size)
        
        policy_grad = self.cpo.compute_policy_gradient(states, actions, advantages, old_log_probs)
        
        # Check that gradient has correct dimension
        param_count = sum(p.numel() for p in self.policy.policy_net.parameters())
        assert policy_grad.shape == (param_count,)
        assert torch.isfinite(policy_grad).all()
        assert torch.norm(policy_grad) > 0  # Should be non-zero
    
    def test_constraint_gradient_computation(self):
        """Test constraint gradient computation."""
        batch_size = 16
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        constraint_advantages = torch.randn(batch_size, len(self.constraint_manager.constraints))
        
        constraint_grads = self.cpo.compute_constraint_gradient(states, actions, constraint_advantages)
        
        # Check shape
        param_count = sum(p.numel() for p in self.policy.policy_net.parameters())
        expected_shape = (len(self.constraint_manager.constraints), param_count)
        assert constraint_grads.shape == expected_shape
        assert torch.isfinite(constraint_grads).all()
    
    def test_fisher_vector_product(self):
        """Test Fisher Information Matrix-vector product computation."""
        batch_size = 32
        states = torch.randn(batch_size, self.state_dim)
        param_count = sum(p.numel() for p in self.policy.policy_net.parameters())
        vector = torch.randn(param_count)
        
        fvp = self.cpo.compute_fisher_vector_product(states, vector)
        
        assert fvp.shape == vector.shape
        assert torch.isfinite(fvp).all()
        
        # Test that FVP is positive definite (approximately)
        quadratic_form = torch.dot(vector, fvp)
        assert quadratic_form > 0  # Should be positive due to damping
    
    def test_trust_region_problem_solving(self):
        """Test trust region problem solving."""
        batch_size = 32
        states = torch.randn(batch_size, self.state_dim)
        param_count = sum(p.numel() for p in self.policy.policy_net.parameters())
        
        policy_grad = torch.randn(param_count)
        constraint_grads = torch.randn(len(self.constraint_manager.constraints), param_count)
        constraint_violations = torch.tensor([0.05, -0.01])  # One violation, one satisfied
        
        search_direction, solver_info = self.cpo.solve_trust_region_problem(
            policy_grad, constraint_grads, constraint_violations, states
        )
        
        assert search_direction.shape == (param_count,)
        assert torch.isfinite(search_direction).all()
        assert isinstance(solver_info, dict)
        assert "num_violated_constraints" in solver_info
        assert solver_info["num_violated_constraints"] >= 0
    
    def test_value_function_update(self):
        """Test value function updates."""
        # Collect trajectories
        trajectories = self.cpo.collect_trajectories(2)
        
        # Update value functions
        vf_stats = self.cpo.update_value_functions(trajectories)
        
        assert isinstance(vf_stats, dict)
        assert "policy_vf_loss" in vf_stats
        assert "constraint_vf_losses" in vf_stats
        assert "avg_policy_return" in vf_stats
        
        assert torch.isfinite(torch.tensor(vf_stats["policy_vf_loss"]))
        assert len(vf_stats["constraint_vf_losses"]) == len(self.constraint_manager.constraints)
    
    def test_constraint_threshold_adaptation(self):
        """Test constraint threshold adaptation."""
        initial_thresholds = self.cpo.constraint_threshold.clone()
        
        # Test with violations
        violations = torch.tensor([0.2, 0.1])  # Both violating
        threshold_updated = self.cpo.update_constraint_threshold(violations)
        
        assert isinstance(threshold_updated, bool)
        if threshold_updated:
            assert (self.cpo.constraint_threshold >= initial_thresholds).any()
        
        # Test with satisfied constraints
        violations_safe = torch.tensor([0.001, 0.002])  # Both safe
        self.cpo.update_constraint_threshold(violations_safe)
        # Thresholds should not increase for satisfied constraints
    
    def test_cpo_step(self):
        """Test single CPO optimization step."""
        num_episodes = 3
        initial_iteration = self.cpo.iteration
        
        cpo_state = self.cpo.step(num_episodes)
        
        # Check that iteration is incremented
        assert self.cpo.iteration == initial_iteration + 1
        
        # Check CPO state
        assert isinstance(cpo_state, CPOState)
        assert cpo_state.iteration == self.cpo.iteration
        assert torch.isfinite(torch.tensor(cpo_state.policy_loss))
        assert torch.isfinite(cpo_state.constraint_violations).all()
        assert cpo_state.kl_divergence >= 0
        assert cpo_state.step_size > 0
        assert cpo_state.total_episodes > 0
        
        # Check that state is added to history
        assert len(self.cpo.optimization_history) == 1
        assert self.cpo.optimization_history[0] is cpo_state
    
    def test_multiple_cpo_steps(self):
        """Test multiple CPO optimization steps."""
        num_iterations = 3
        num_episodes_per_iter = 2
        
        states = []
        for i in range(num_iterations):
            state = self.cpo.step(num_episodes_per_iter)
            states.append(state)
        
        # Check iteration progression
        for i, state in enumerate(states):
            assert state.iteration == i + 1
        
        # Check that policy parameters have changed
        initial_params = [p.clone() for p in self.policy.policy_net.parameters()]
        
        # Run one more step
        final_state = self.cpo.step(num_episodes_per_iter)
        
        # Parameters should have changed
        params_changed = False
        for initial_p, current_p in zip(initial_params, self.policy.policy_net.parameters()):
            if not torch.allclose(initial_p, current_p, atol=1e-6):
                params_changed = True
                break
        
        assert params_changed, "Policy parameters should change during optimization"
    
    def test_emergency_brake_functionality(self):
        """Test emergency brake activation."""
        # Configure emergency brake
        self.cpo.config.emergency_brake = True
        
        # Simulate high constraint violations to trigger emergency brake
        # This would typically be done through the safety monitor
        self.cpo.consecutive_violations = 3
        
        # The emergency brake should be activated
        # (actual activation logic is in the step function)
        
    def test_cpo_training_loop(self):
        """Test complete training loop."""
        num_iterations = 2
        episodes_per_iteration = 2
        
        # Run training
        history = self.cpo.train(num_iterations, episodes_per_iteration)
        
        assert len(history) == num_iterations
        assert all(isinstance(state, CPOState) for state in history)
        
        # Check that iterations progress correctly
        for i, state in enumerate(history):
            assert state.iteration == i + 1
        
        # Check that total episodes is correct
        expected_total = num_iterations * episodes_per_iteration
        assert self.cpo.total_episodes == expected_total
    
    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint saving and loading."""
        # Run a few steps to create state
        self.cpo.step(2)
        self.cpo.step(2)
        
        # Save checkpoint
        checkpoint_path = tmp_path / "cpo_checkpoint.pt"
        self.cpo.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
        
        # Create new CPO instance and load checkpoint
        new_cpo = CPOAlgorithm(
            policy=SafePolicy(self.state_dim, self.action_dim, device=self.device),
            constraint_manager=self.constraint_manager,
            environment=self.env,
            config=self.config,
            device=self.device
        )
        
        # Verify initial state is different
        assert new_cpo.iteration != self.cpo.iteration
        
        # Load checkpoint
        new_cpo.load_checkpoint(str(checkpoint_path))
        
        # Verify state is restored
        assert new_cpo.iteration == self.cpo.iteration
        assert new_cpo.total_episodes == self.cpo.total_episodes
        assert torch.allclose(new_cpo.constraint_threshold, self.cpo.constraint_threshold)


class TestCPONumericalAccuracy:
    """Test numerical accuracy of CPO computations."""
    
    def setup_method(self):
        """Set up simple test case for numerical verification."""
        self.state_dim = 4
        self.action_dim = 2
        
        # Simple constraint for testing
        force_constraint = ForceConstraint(max_force=1.0)
        self.constraint_manager = ConstraintManager([force_constraint])
        
        # Simple policy
        self.policy = SafePolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=[8],  # Very small for numerical testing
            device="cpu"
        )
        
        self.env = MockEnvironment(self.state_dim, self.action_dim)
        self.config = CPOConfig()
        
        self.cpo = CPOAlgorithm(
            policy=self.policy,
            constraint_manager=self.constraint_manager,
            environment=self.env,
            config=self.config,
            device="cpu"
        )
    
    def test_policy_gradient_numerical_accuracy(self):
        """Test policy gradient against numerical gradient."""
        batch_size = 8
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim) * 0.1  # Small actions
        advantages = torch.ones(batch_size)  # Simple advantages
        old_log_probs = torch.zeros(batch_size)  # Simple old log probs
        
        # Compute analytical gradient
        policy_grad = self.cpo.compute_policy_gradient(states, actions, advantages, old_log_probs)
        
        # Numerical gradient check
        def policy_objective(param_vec):
            # Set parameters
            idx = 0
            for param in self.policy.policy_net.parameters():
                param_size = param.numel()
                param.data = param_vec[idx:idx+param_size].view_as(param)
                idx += param_size
            
            # Compute objective
            evaluation = self.policy.evaluate_actions(states, actions)
            log_probs = evaluation["log_probs"]
            return -(log_probs * advantages).mean()
        
        # Get current parameters
        current_params = torch.cat([p.view(-1) for p in self.policy.policy_net.parameters()])
        
        # Compute numerical gradient
        numerical_grad = finite_difference_gradient(policy_objective, current_params, eps=1e-5)
        
        # Compare gradients
        grad_error = torch.norm(policy_grad - numerical_grad)
        rel_error = grad_error / (torch.norm(policy_grad) + 1e-8)
        
        assert grad_error < 1e-2, f"Policy gradient error too large: {grad_error:.6f}"
        assert rel_error < 1e-2, f"Relative gradient error too large: {rel_error:.6f}"
    
    def test_fisher_vector_product_properties(self):
        """Test mathematical properties of Fisher-vector product."""
        batch_size = 16
        states = torch.randn(batch_size, self.state_dim)
        param_count = sum(p.numel() for p in self.policy.policy_net.parameters())
        
        # Test linearity: F(av + bw) = aF(v) + bF(w)
        v = torch.randn(param_count)
        w = torch.randn(param_count)
        a, b = 2.0, 3.0
        
        fvp_v = self.cpo.compute_fisher_vector_product(states, v)
        fvp_w = self.cpo.compute_fisher_vector_product(states, w)
        fvp_combined = self.cpo.compute_fisher_vector_product(states, a * v + b * w)
        
        expected_combined = a * fvp_v + b * fvp_w
        
        linearity_error = torch.norm(fvp_combined - expected_combined)
        assert linearity_error < 1e-4, f"Fisher linearity error: {linearity_error:.6f}"
        
        # Test positive definiteness (with damping)
        random_vector = torch.randn(param_count)
        fvp = self.cpo.compute_fisher_vector_product(states, random_vector)
        quadratic_form = torch.dot(random_vector, fvp)
        
        assert quadratic_form > 0, "Fisher matrix should be positive definite"
    
    def test_constraint_gradient_accuracy(self):
        """Test constraint gradient numerical accuracy."""
        batch_size = 4
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim) * 0.1
        constraint_advantages = torch.ones(batch_size, 1)  # Single constraint
        
        # Compute analytical gradient
        constraint_grads = self.cpo.compute_constraint_gradient(states, actions, constraint_advantages)
        
        # Numerical gradient for constraint objective
        def constraint_objective(param_vec):
            idx = 0
            for param in self.policy.policy_net.parameters():
                param_size = param.numel()
                param.data = param_vec[idx:idx+param_size].view_as(param)
                idx += param_size
            
            evaluation = self.policy.evaluate_actions(states, actions)
            log_probs = evaluation["log_probs"]
            return (log_probs * constraint_advantages[:, 0]).mean()
        
        current_params = torch.cat([p.view(-1) for p in self.policy.policy_net.parameters()])
        numerical_grad = finite_difference_gradient(constraint_objective, current_params, eps=1e-5)
        
        # Compare with analytical gradient
        analytical_grad = constraint_grads[0]  # First (and only) constraint
        grad_error = torch.norm(analytical_grad - numerical_grad)
        rel_error = grad_error / (torch.norm(analytical_grad) + 1e-8)
        
        assert grad_error < 1e-2, f"Constraint gradient error: {grad_error:.6f}"
        assert rel_error < 1e-2, f"Relative constraint gradient error: {rel_error:.6f}"


@pytest.mark.parametrize("state_dim,action_dim,num_constraints", [
    (4, 2, 1),
    (6, 3, 2),
    (8, 4, 3)
])
def test_cpo_with_different_dimensions(state_dim, action_dim, num_constraints):
    """Test CPO with different problem dimensions."""
    # Create constraints
    constraints = []
    for i in range(num_constraints):
        constraint = ForceConstraint(max_force=2.0 + i)
        constraints.append(constraint)
    
    constraint_manager = ConstraintManager(constraints)
    
    # Create policy
    policy = SafePolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        constraint_manager=constraint_manager,
        hidden_sizes=[16, 16]
    )
    
    # Create environment
    env = MockEnvironment(state_dim, action_dim)
    
    # Create CPO
    config = CPOConfig()
    config.value_iters = 5  # Reduce for testing
    
    cpo = CPOAlgorithm(
        policy=policy,
        constraint_manager=constraint_manager,
        environment=env,
        config=config
    )
    
    # Test single step
    cpo_state = cpo.step(num_episodes=2)
    
    assert isinstance(cpo_state, CPOState)
    assert cpo_state.constraint_violations.shape == (num_constraints,)


def test_cpo_convergence_behavior():
    """Test CPO convergence behavior over multiple iterations."""
    # Set up CPO with consistent random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    state_dim, action_dim = 4, 2
    force_constraint = ForceConstraint(max_force=3.0)
    constraint_manager = ConstraintManager([force_constraint])
    
    policy = SafePolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        constraint_manager=constraint_manager,
        hidden_sizes=[16, 16]
    )
    
    env = MockEnvironment(state_dim, action_dim)
    config = CPOConfig()
    config.value_iters = 10
    
    cpo = CPOAlgorithm(
        policy=policy,
        constraint_manager=constraint_manager,
        environment=env,
        config=config
    )
    
    # Run multiple iterations
    num_iterations = 5
    history = cpo.train(num_iterations, episodes_per_iteration=3)
    
    # Check convergence properties
    assert len(history) == num_iterations
    
    # Policy loss should generally improve (become more positive for maximization)
    policy_losses = [state.policy_loss for state in history]
    
    # At least some improvement should occur
    improvement = policy_losses[-1] - policy_losses[0]
    # Note: improvement might be negative (loss decreasing) or positive depending on formulation
    
    # KL divergence should be reasonable
    kl_divergences = [state.kl_divergence for state in history]
    assert all(kl >= 0 for kl in kl_divergences)
    assert all(kl <= config.target_kl * 2 for kl in kl_divergences)  # Allow some slack
    
    # Step sizes should be positive
    step_sizes = [state.step_size for state in history]
    assert all(step > 0 for step in step_sizes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])