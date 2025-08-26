"""
Unit tests for safe policy implementation.

Tests policy network, action sampling, gradient computation,
and trust region updates for safe reinforcement learning.
"""

import pytest
import torch
import numpy as np
from torch.distributions import Normal
from typing import Dict, List
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.policy import PolicyNetwork, SafePolicy
from core.constraints import CollisionConstraint, ConstraintManager
from utils.math_utils import compute_kl_divergence, finite_difference_gradient


class TestPolicyNetwork:
    """Test policy network implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 8
        self.action_dim = 4
        self.hidden_sizes = [64, 64]
        self.network = PolicyNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=self.hidden_sizes,
            activation="tanh"
        )
    
    def test_network_initialization(self):
        """Test network initialization and architecture."""
        # Check network structure
        assert len(list(self.network.parameters())) > 0
        
        # Check parameter shapes
        param_count = sum(p.numel() for p in self.network.parameters())
        expected_min_params = self.state_dim * 64 + 64 * 64 + 64 * self.action_dim + self.action_dim
        assert param_count >= expected_min_params
        
        # Check log_std parameter
        assert self.network.log_std.shape == (self.action_dim,)
        
        # Test with different activations
        for activation in ["relu", "elu", "tanh"]:
            net = PolicyNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_sizes=[32],
                activation=activation
            )
            assert net is not None
    
    def test_forward_pass(self):
        """Test forward pass through network."""
        batch_size = 10
        states = torch.randn(batch_size, self.state_dim)
        
        mean, std = self.network.forward(states)
        
        assert mean.shape == (batch_size, self.action_dim)
        assert std.shape == (batch_size, self.action_dim)
        assert (std > 0).all(), "Standard deviation must be positive"
        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()
    
    def test_distribution_creation(self):
        """Test policy distribution creation."""
        batch_size = 5
        states = torch.randn(batch_size, self.state_dim)
        
        dist = self.network.get_distribution(states)
        
        assert isinstance(dist, Normal)
        assert dist.mean.shape == (batch_size, self.action_dim)
        assert dist.stddev.shape == (batch_size, self.action_dim)
        
        # Test sampling
        actions = dist.sample()
        assert actions.shape == (batch_size, self.action_dim)
        
        # Test log probability
        log_probs = dist.log_prob(actions)
        assert log_probs.shape == (batch_size, self.action_dim)
        assert not torch.isnan(log_probs).any()
        
        # Test entropy
        entropy = dist.entropy()
        assert entropy.shape == (batch_size, self.action_dim)
        assert (entropy > 0).all(), "Entropy should be positive for continuous distributions"


class TestSafePolicy:
    """Test safe policy implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.action_dim = 6
        self.device = "cpu"
        
        # Create constraint manager
        collision_constraint = CollisionConstraint(
            min_distance=0.1,
            human_position_idx=(0, 3),
            robot_position_idx=(3, 6)
        )
        self.constraint_manager = ConstraintManager([collision_constraint])
        
        # Create safe policy
        self.policy = SafePolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            constraint_manager=self.constraint_manager,
            hidden_sizes=[64, 64],
            device=self.device
        )
    
    def test_policy_initialization(self):
        """Test policy initialization."""
        assert self.policy.state_dim == self.state_dim
        assert self.policy.action_dim == self.action_dim
        assert self.policy.device == self.device
        assert self.policy.constraint_manager is not None
        
        # Check policy network
        assert isinstance(self.policy.policy_net, PolicyNetwork)
        
        # Check statistics initialization
        assert self.policy.constraint_violations == 0
        assert self.policy.total_actions == 0
        assert len(self.policy.action_history) == 0
    
    def test_action_sampling_safe(self):
        """Test safe action sampling."""
        batch_size = 5
        states = torch.randn(batch_size, self.state_dim)
        
        # Set up states to avoid collisions
        states[:, 0:3] = torch.tensor([0.0, 0.0, 1.0])  # Human position
        states[:, 3:6] = torch.tensor([1.0, 0.0, 1.0])  # Robot position (far from human)
        
        # Sample actions
        actions, info = self.policy.sample_action(states, deterministic=False)
        
        assert actions.shape == (batch_size, self.action_dim)
        assert "log_prob" in info
        assert "entropy" in info
        assert "is_safe" in info
        assert "safety_iterations" in info
        
        # Check info shapes
        assert info["log_prob"].shape == (batch_size,)
        assert info["entropy"].shape == (batch_size,)
        
        # Test deterministic sampling
        actions_det, info_det = self.policy.sample_action(states, deterministic=True)
        assert actions_det.shape == (batch_size, self.action_dim)
        
        # Deterministic actions should be the same across calls
        actions_det2, _ = self.policy.sample_action(states, deterministic=True)
        assert torch.allclose(actions_det, actions_det2, atol=1e-6)
    
    def test_action_evaluation(self):
        """Test action evaluation."""
        batch_size = 8
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        
        evaluation = self.policy.evaluate_actions(states, actions)
        
        assert "log_probs" in evaluation
        assert "entropy" in evaluation
        assert "mean" in evaluation
        assert "std" in evaluation
        
        assert evaluation["log_probs"].shape == (batch_size,)
        assert evaluation["entropy"].shape == (batch_size,)
        assert evaluation["mean"].shape == (batch_size, self.action_dim)
        assert evaluation["std"].shape == (batch_size, self.action_dim)
        
        # Check that log probs are reasonable
        assert not torch.isnan(evaluation["log_probs"]).any()
        assert torch.isfinite(evaluation["log_probs"]).all()
    
    def test_policy_gradient(self):
        """Test policy gradient computation."""
        batch_size = 16
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        advantages = torch.randn(batch_size)
        
        # Test standard policy gradient
        policy_loss = self.policy.compute_policy_gradient(states, actions, advantages)
        
        assert isinstance(policy_loss, torch.Tensor)
        assert policy_loss.dim() == 0  # Scalar loss
        assert torch.isfinite(policy_loss)
        
        # Test with old log probs (PPO-style)
        old_log_probs = torch.randn(batch_size)
        policy_loss_ppo = self.policy.compute_policy_gradient(
            states, actions, advantages, old_log_probs
        )
        
        assert isinstance(policy_loss_ppo, torch.Tensor)
        assert policy_loss_ppo.dim() == 0
        assert torch.isfinite(policy_loss_ppo)
        
        # Losses should be different
        assert not torch.allclose(policy_loss, policy_loss_ppo)
    
    def test_constraint_gradient(self):
        """Test constraint gradient computation."""
        batch_size = 4
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        
        constraint_grad = self.policy.compute_constraint_gradient(states, actions)
        
        assert constraint_grad is not None
        assert isinstance(constraint_grad, torch.Tensor)
        assert constraint_grad.dim() == 0  # Scalar constraint value
        assert torch.isfinite(constraint_grad)
    
    def test_trust_region_update(self):
        """Test trust region policy update."""
        batch_size = 32
        states = torch.randn(batch_size, self.state_dim)
        actions = torch.randn(batch_size, self.action_dim)
        advantages = torch.randn(batch_size)
        
        # Store old parameters
        old_params = [p.clone() for p in self.policy.policy_net.parameters()]
        
        # Perform update
        update_stats = self.policy.update_trust_region_step(
            states, actions, advantages,
            max_kl=0.01,
            damping_coef=0.1
        )
        
        assert isinstance(update_stats, dict)
        assert "kl_divergence" in update_stats
        assert "step_size" in update_stats
        assert "policy_loss" in update_stats
        
        # Check that parameters actually changed
        param_changed = False
        for old_p, new_p in zip(old_params, self.policy.policy_net.parameters()):
            if not torch.allclose(old_p, new_p, atol=1e-6):
                param_changed = True
                break
        
        assert param_changed, "Parameters should have changed after update"
        
        # Check KL constraint
        assert update_stats["kl_divergence"] >= 0
        assert update_stats["step_size"] > 0
    
    def test_policy_statistics(self):
        """Test policy statistics collection."""
        # Sample some actions to populate statistics
        states = torch.randn(10, self.state_dim)
        for _ in range(5):
            self.policy.sample_action(states)
        
        stats = self.policy.get_policy_statistics()
        
        assert isinstance(stats, dict)
        assert "total_actions" in stats
        assert "constraint_violations" in stats
        assert "violation_rate" in stats
        assert "parameter_count" in stats
        
        assert stats["total_actions"] > 0
        assert stats["violation_rate"] >= 0
        assert stats["violation_rate"] <= 1
        assert stats["parameter_count"] > 0
    
    def test_save_load_policy(self, tmp_path):
        """Test policy saving and loading."""
        # Save policy
        save_path = tmp_path / "test_policy.pt"
        self.policy.save(str(save_path))
        
        assert save_path.exists()
        
        # Create new policy and load
        new_policy = SafePolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            device=self.device
        )
        
        new_policy.load(str(save_path))
        
        # Check that parameters match
        for p1, p2 in zip(self.policy.policy_net.parameters(), new_policy.policy_net.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_device_transfer(self):
        """Test moving policy to different device."""
        original_device = self.policy.device
        
        # Move to same device (should work)
        policy_moved = self.policy.to("cpu")
        assert policy_moved.device == "cpu"
        assert policy_moved is self.policy  # Should return self
        
        # Test that network is on correct device
        for param in self.policy.policy_net.parameters():
            assert param.device.type == "cpu"


class TestPolicyGradients:
    """Test policy gradient computations and numerical accuracy."""
    
    def setup_method(self):
        """Set up simple policy for gradient testing."""
        self.policy = SafePolicy(
            state_dim=4,
            action_dim=2,
            hidden_sizes=[8],  # Small network for faster computation
            device="cpu"
        )
    
    def test_policy_gradient_numerical_check(self):
        """Test policy gradient against numerical gradient."""
        # Single state-action pair for precise numerical check
        state = torch.randn(1, 4)
        action = torch.randn(1, 2)
        advantage = torch.tensor([1.0])
        
        # Compute analytical gradient
        self.policy.policy_net.train()
        loss = self.policy.compute_policy_gradient(state, action, advantage)
        loss.backward()
        
        # Collect gradients
        analytical_grads = []
        for param in self.policy.policy_net.parameters():
            if param.grad is not None:
                analytical_grads.append(param.grad.clone().flatten())
        analytical_grad = torch.cat(analytical_grads)
        
        # Clear gradients
        self.policy.policy_net.zero_grad()
        
        # Numerical gradient
        def loss_func(param_vec):
            # Set parameters
            idx = 0
            for param in self.policy.policy_net.parameters():
                param_size = param.numel()
                param.data = param_vec[idx:idx+param_size].view_as(param)
                idx += param_size
            
            return self.policy.compute_policy_gradient(state, action, advantage)
        
        # Get current parameters
        current_params = []
        for param in self.policy.policy_net.parameters():
            current_params.append(param.data.flatten())
        current_param_vec = torch.cat(current_params)
        
        # Compute numerical gradient
        numerical_grad = finite_difference_gradient(loss_func, current_param_vec, eps=1e-5)
        
        # Compare gradients
        grad_error = torch.norm(analytical_grad - numerical_grad)
        rel_error = grad_error / (torch.norm(analytical_grad) + 1e-8)
        
        assert grad_error < 1e-2, f"Gradient error too large: {grad_error:.6f}"
        assert rel_error < 1e-2, f"Relative gradient error too large: {rel_error:.6f}"
    
    def test_kl_divergence_computation(self):
        """Test KL divergence computation between policies."""
        batch_size = 16
        states = torch.randn(batch_size, 4)
        
        # Get old distribution
        old_dist = self.policy.policy_net.get_distribution(states)
        old_mean = old_dist.mean.detach()
        old_std = old_dist.stddev.detach()
        
        # Modify policy slightly
        with torch.no_grad():
            for param in self.policy.policy_net.parameters():
                param += torch.randn_like(param) * 0.01
        
        # Get new distribution
        new_dist = self.policy.policy_net.get_distribution(states)
        new_mean = new_dist.mean
        new_std = new_dist.stddev
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(new_mean, new_std, old_mean, old_std)
        
        assert kl_div.shape == (batch_size,)
        assert (kl_div >= 0).all(), "KL divergence must be non-negative"
        assert torch.isfinite(kl_div).all()
        
        # KL should be small for small parameter changes
        assert torch.mean(kl_div) < 1.0, "KL divergence too large for small changes"


@pytest.mark.parametrize("state_dim,action_dim,hidden_sizes", [
    (5, 3, [32]),
    (8, 4, [64, 32]),
    (12, 6, [128, 64, 32]),
    (3, 2, [16, 16, 16])
])
def test_policy_shapes(state_dim, action_dim, hidden_sizes):
    """Test policy with different dimensions and architectures."""
    policy = SafePolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes
    )
    
    batch_size = 7
    states = torch.randn(batch_size, state_dim)
    
    # Test action sampling
    actions, info = policy.sample_action(states)
    assert actions.shape == (batch_size, action_dim)
    
    # Test evaluation
    evaluation = policy.evaluate_actions(states, actions)
    assert evaluation["log_probs"].shape == (batch_size,)
    assert evaluation["mean"].shape == (batch_size, action_dim)


@pytest.mark.parametrize("activation", ["tanh", "relu", "elu"])
def test_activation_functions(activation):
    """Test different activation functions."""
    policy = SafePolicy(
        state_dim=6,
        action_dim=3,
        hidden_sizes=[32, 32]
    )
    
    # Modify network activation
    policy.policy_net.activation = getattr(torch.nn, activation.upper())() if activation == "relu" else \
                                  torch.nn.ELU() if activation == "elu" else torch.nn.Tanh()
    
    states = torch.randn(4, 6)
    actions, _ = policy.sample_action(states)
    
    assert actions.shape == (4, 3)
    assert torch.isfinite(actions).all()


def test_policy_with_no_constraints():
    """Test policy behavior without constraint manager."""
    policy = SafePolicy(
        state_dim=5,
        action_dim=3,
        constraint_manager=None  # No constraints
    )
    
    states = torch.randn(3, 5)
    actions, info = policy.sample_action(states)
    
    assert actions.shape == (3, 3)
    assert info["is_safe"] is True  # Should be safe when no constraints
    assert info["safety_iterations"] == 0


def test_policy_convergence_behavior():
    """Test policy behavior during training convergence."""
    policy = SafePolicy(state_dim=4, action_dim=2, hidden_sizes=[16])
    
    # Simulate training steps
    states = torch.randn(32, 4)
    actions = torch.randn(32, 2)
    
    losses = []
    for epoch in range(10):
        advantages = torch.randn(32)
        loss = policy.compute_policy_gradient(states, actions, advantages)
        losses.append(loss.item())
        
        # Simple gradient descent step
        loss.backward()
        with torch.no_grad():
            for param in policy.policy_net.parameters():
                if param.grad is not None:
                    param -= 0.01 * param.grad
        policy.policy_net.zero_grad()
    
    # Check that optimization is working (loss should change)
    assert abs(losses[-1] - losses[0]) > 1e-4, "Loss should change during optimization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])