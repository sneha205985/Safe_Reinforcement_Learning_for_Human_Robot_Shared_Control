"""
Tests for Generalized Advantage Estimation (GAE) implementation.

Tests mathematical correctness of GAE computation, value function training,
and advantage buffer management with numerical validation.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.gae import (
    GeneralizedAdvantageEstimation, ValueFunction, AdvantageBuffer,
    GAEConfig, GAEResult, test_gae_implementation
)
from utils.math_utils import finite_difference_gradient


class TestGAEConfig:
    """Test GAE configuration."""
    
    def test_default_config(self):
        """Test default GAE configuration values."""
        config = GAEConfig()
        
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.97
        assert config.normalize_advantages is True
        assert config.use_gae is True
        assert config.value_clipping is True
        assert config.clip_range == 0.2
    
    def test_config_modification(self):
        """Test GAE configuration modification."""
        config = GAEConfig()
        config.gamma = 0.95
        config.gae_lambda = 0.9
        config.normalize_advantages = False
        
        assert config.gamma == 0.95
        assert config.gae_lambda == 0.9
        assert config.normalize_advantages is False


class TestValueFunction:
    """Test value function neural network."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 6
        self.hidden_sizes = [32, 32]
        self.device = "cpu"
        
        self.value_function = ValueFunction(
            state_dim=self.state_dim,
            hidden_sizes=self.hidden_sizes,
            activation="tanh",
            learning_rate=1e-3,
            device=self.device
        )
    
    def test_value_function_initialization(self):
        """Test value function initialization."""
        assert self.value_function.state_dim == self.state_dim
        assert self.value_function.device == self.device
        
        # Check network architecture
        network_layers = list(self.value_function.network.children())
        assert len(network_layers) > 0
        
        # Check that output is single value
        test_input = torch.randn(1, self.state_dim)
        output = self.value_function(test_input)
        assert output.shape == (1, 1)
    
    def test_forward_pass(self):
        """Test value function forward pass."""
        batch_size = 10
        states = torch.randn(batch_size, self.state_dim)
        
        values = self.value_function(states)
        
        assert values.shape == (batch_size, 1)
        assert torch.isfinite(values).all()
    
    def test_prediction(self):
        """Test value function prediction (no gradients)."""
        batch_size = 5
        states = torch.randn(batch_size, self.state_dim)
        
        values = self.value_function.predict(states)
        
        assert values.shape == (batch_size,)
        assert torch.isfinite(values).all()
        assert not values.requires_grad  # Should not require gradients
    
    def test_value_function_update(self):
        """Test value function training update."""
        num_samples = 64
        states = torch.randn(num_samples, self.state_dim)
        targets = torch.randn(num_samples)
        
        initial_loss = float('inf')
        
        # Perform update
        final_loss = self.value_function.update(states, targets, num_epochs=10, batch_size=16)
        
        assert isinstance(final_loss, float)
        assert final_loss >= 0
        assert np.isfinite(final_loss)
        
        # Training should have occurred
        assert self.value_function.update_count == 1
        assert len(self.value_function.training_losses) == 1
    
    def test_value_function_learning(self):
        """Test that value function actually learns."""
        # Create simple dataset: V(s) = sum(s)
        num_samples = 100
        states = torch.randn(num_samples, self.state_dim)
        targets = states.sum(dim=1)  # Simple target function
        
        # Get initial predictions
        initial_predictions = self.value_function.predict(states)
        initial_error = torch.mean((initial_predictions - targets)**2)
        
        # Train value function
        for _ in range(5):
            self.value_function.update(states, targets, num_epochs=20)
        
        # Get final predictions
        final_predictions = self.value_function.predict(states)
        final_error = torch.mean((final_predictions - targets)**2)
        
        # Error should decrease significantly
        assert final_error < initial_error * 0.5, f"Learning failed: {initial_error:.4f} -> {final_error:.4f}"
    
    def test_training_statistics(self):
        """Test value function training statistics."""
        states = torch.randn(32, self.state_dim)
        targets = torch.randn(32)
        
        # Perform multiple updates
        for i in range(3):
            self.value_function.update(states, targets, num_epochs=5)
        
        stats = self.value_function.get_training_stats()
        
        assert isinstance(stats, dict)
        assert stats["updates"] == 3
        assert "avg_loss" in stats
        assert "recent_loss" in stats
        assert "loss_std" in stats
        assert np.isfinite(stats["avg_loss"])


class TestGeneralizedAdvantageEstimation:
    """Test GAE computation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gamma = 0.99
        self.gae_lambda = 0.97
        
        self.gae = GeneralizedAdvantageEstimation(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantages=True,
            use_gae=True
        )
    
    def test_gae_initialization(self):
        """Test GAE initialization."""
        assert self.gae.gamma == self.gamma
        assert self.gae.gae_lambda == self.gae_lambda
        assert self.gae.normalize_advantages is True
        assert self.gae.use_gae is True
        
        assert self.gae.computation_stats["total_calls"] == 0
        assert self.gae.computation_stats["total_timesteps"] == 0
    
    def test_gae_computation_basic(self):
        """Test basic GAE computation."""
        T = 20
        rewards = torch.randn(T) * 0.1 + 0.05  # Small positive rewards
        values = torch.randn(T) * 0.2
        dones = torch.zeros(T)
        dones[10] = 1.0  # Episode boundary
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        
        returns, advantages = self.gae.compute_gae(rewards, values, dones, next_values)
        
        # Check shapes
        assert returns.shape == (T,)
        assert advantages.shape == (T,)
        
        # Check finite values
        assert torch.isfinite(returns).all()
        assert torch.isfinite(advantages).all()
        
        # Check that returns = advantages + values (approximately)
        returns_check = advantages + values
        assert torch.allclose(returns, returns_check, atol=1e-5)
        
        # Check normalization (advantages should have ~zero mean)
        assert abs(advantages.mean().item()) < 0.1
    
    def test_gae_vs_simple_advantages(self):
        """Test GAE vs simple advantage computation."""
        T = 50
        rewards = torch.randn(T) * 0.2
        values = torch.randn(T) * 0.5
        dones = torch.zeros(T)
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        
        # GAE computation
        gae_enabled = GeneralizedAdvantageEstimation(
            gamma=self.gamma, gae_lambda=self.gae_lambda, 
            normalize_advantages=False, use_gae=True
        )
        returns_gae, advantages_gae = gae_enabled.compute_gae(rewards, values, dones, next_values)
        
        # Simple advantage computation
        gae_disabled = GeneralizedAdvantageEstimation(
            gamma=self.gamma, gae_lambda=self.gae_lambda,
            normalize_advantages=False, use_gae=False
        )
        returns_simple, advantages_simple = gae_disabled.compute_gae(rewards, values, dones, next_values)
        
        # Returns should be similar but advantages should differ
        # (GAE provides smoother advantage estimates)
        returns_diff = torch.norm(returns_gae - returns_simple)
        advantages_diff = torch.norm(advantages_gae - advantages_simple)
        
        assert returns_diff < 1e-10  # Returns should be identical
        assert advantages_diff > 1e-3  # Advantages should be different
        
        # GAE advantages should be smoother (lower variance)
        gae_variance = torch.var(advantages_gae)
        simple_variance = torch.var(advantages_simple)
        assert gae_variance <= simple_variance * 1.1  # Allow small tolerance
    
    def test_gae_mathematical_properties(self):
        """Test mathematical properties of GAE."""
        T = 30
        rewards = torch.ones(T) * 0.1  # Constant rewards
        values = torch.zeros(T)  # Zero values
        dones = torch.zeros(T)
        next_values = torch.zeros(T)
        
        returns, advantages = self.gae.compute_gae(rewards, values, dones, next_values)
        
        # With constant rewards and zero values, advantages should follow GAE formula
        # A_t = r + γλA_{t+1} with A_T = r
        expected_advantage_final = rewards[-1]  # Final advantage
        
        # Work backwards to check recursive formula
        expected_advantages = torch.zeros(T)
        expected_advantages[-1] = rewards[-1]
        
        for t in reversed(range(T-1)):
            expected_advantages[t] = rewards[t] + self.gamma * self.gae_lambda * expected_advantages[t+1]
        
        # Remove normalization for comparison
        advantages_unnormalized = advantages * torch.std(expected_advantages) + torch.mean(expected_advantages)
        
        assert torch.allclose(advantages_unnormalized, expected_advantages, atol=1e-4)
    
    def test_gae_with_episode_boundaries(self):
        """Test GAE computation with episode boundaries."""
        T = 40
        rewards = torch.randn(T) * 0.1
        values = torch.randn(T) * 0.2
        dones = torch.zeros(T)
        dones[15] = 1.0  # Episode boundary
        dones[30] = 1.0  # Another episode boundary
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        
        returns, advantages = self.gae.compute_gae(rewards, values, dones, next_values)
        
        # Check that episodes are handled correctly
        # Advantage at episode boundary should not depend on next episode
        assert torch.isfinite(advantages[15])
        assert torch.isfinite(advantages[30])
        
        # Returns should reset at episode boundaries
        assert returns[15] >= rewards[15] - 1.0  # Should not be too negative
        assert returns[30] >= rewards[30] - 1.0
    
    def test_n_step_returns(self):
        """Test n-step return computation."""
        T = 20
        rewards = torch.ones(T) * 0.1  # Constant rewards
        values = torch.zeros(T)
        dones = torch.zeros(T)
        
        n_steps = 5
        n_step_returns = self.gae.compute_n_step_returns(rewards, values, dones, n_steps)
        
        assert n_step_returns.shape == (T,)
        assert torch.isfinite(n_step_returns).all()
        
        # For constant rewards with zero values and no terminations,
        # n-step return should be sum of n rewards
        expected_early_return = sum(self.gamma**k * 0.1 for k in range(n_steps))
        
        # Check first few returns (before edge effects)
        assert abs(n_step_returns[0].item() - expected_early_return) < 1e-5
    
    def test_gae_statistics_tracking(self):
        """Test GAE computation statistics."""
        initial_calls = self.gae.computation_stats["total_calls"]
        
        T = 25
        rewards = torch.randn(T)
        values = torch.randn(T)
        dones = torch.zeros(T)
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        
        self.gae.compute_gae(rewards, values, dones, next_values)
        
        stats = self.gae.get_computation_stats()
        
        assert stats["total_calls"] == initial_calls + 1
        assert stats["total_timesteps"] == T
        assert "avg_advantage" in stats
        assert "avg_return" in stats


class TestAdvantageBuffer:
    """Test advantage buffer for trajectory data management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.capacity = 100
        self.gamma = 0.99
        self.gae_lambda = 0.97
        
        self.buffer = AdvantageBuffer(
            capacity=self.capacity,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        assert self.buffer.capacity == self.capacity
        assert self.buffer.gamma == self.gamma
        assert self.buffer.gae_lambda == self.gae_lambda
        assert self.buffer.size == 0
        assert self.buffer.position == 0
    
    def test_buffer_single_addition(self):
        """Test adding single timestep to buffer."""
        state = torch.randn(4)
        action = torch.randn(2)
        reward = 0.1
        value = 0.5
        done = False
        log_prob = -1.2
        
        self.buffer.add(state, action, reward, value, done, log_prob)
        
        assert self.buffer.size == 1
        assert self.buffer.position == 1
        
        # Check stored data
        assert torch.equal(self.buffer.states[0], state)
        assert torch.equal(self.buffer.actions[0], action)
        assert self.buffer.rewards[0] == reward
    
    def test_buffer_trajectory_addition(self):
        """Test adding complete trajectory to buffer."""
        T = 20
        trajectory_data = {
            "states": torch.randn(T, 4),
            "actions": torch.randn(T, 2),
            "rewards": torch.randn(T),
            "values": torch.randn(T),
            "dones": torch.zeros(T),
            "log_probs": torch.randn(T)
        }
        
        self.buffer.add_trajectory(trajectory_data)
        
        assert self.buffer.size == T
        assert self.buffer.position == T % self.capacity
    
    def test_buffer_overflow_handling(self):
        """Test buffer behavior when capacity is exceeded."""
        # Fill buffer beyond capacity
        for i in range(self.capacity + 10):
            state = torch.tensor([float(i)])
            action = torch.tensor([float(i)])
            
            self.buffer.add(state, action, 0.0, 0.0, False, 0.0)
        
        # Size should be capped at capacity
        assert self.buffer.size == self.capacity
        
        # Should wrap around (circular buffer)
        assert self.buffer.position == 10  # (capacity + 10) % capacity
        
        # Check that old data was overwritten
        assert self.buffer.states[0].item() == self.capacity  # Overwritten data
    
    def test_buffer_advantage_computation(self):
        """Test advantage computation for buffered data."""
        # Add some trajectory data
        T = 30
        for i in range(T):
            state = torch.randn(3)
            action = torch.randn(2)
            reward = 0.1 * (1 - i/T)  # Decreasing rewards
            value = 0.5 + 0.1 * np.sin(i)  # Varying values
            done = (i == T-1)  # Done at end
            log_prob = -1.0
            
            self.buffer.add(state, action, reward, value, done, log_prob)
        
        # Compute advantages
        self.buffer.compute_advantages(next_value=0.0)
        
        # Check that advantages and returns were computed
        assert len(self.buffer.advantages) == T
        assert len(self.buffer.returns) == T
        
        # Check finite values
        assert all(np.isfinite(adv) for adv in self.buffer.advantages)
        assert all(np.isfinite(ret) for ret in self.buffer.returns)
    
    def test_buffer_batch_retrieval(self):
        """Test batch data retrieval from buffer."""
        # Fill buffer with some data
        T = 50
        for i in range(T):
            state = torch.randn(4)
            action = torch.randn(2)
            self.buffer.add(state, action, 0.1, 0.5, False, -1.0)
        
        self.buffer.compute_advantages()
        
        # Get full batch
        batch = self.buffer.get_batch()
        
        assert batch["states"].shape == (T, 4)
        assert batch["actions"].shape == (T, 2)
        assert len(batch["rewards"]) == T
        assert len(batch["advantages"]) == T
        assert len(batch["returns"]) == T
        
        # Get smaller batch
        batch_size = 20
        small_batch = self.buffer.get_batch(batch_size)
        
        assert batch["states"].shape[0] == batch_size
        assert len(batch["advantages"]) == batch_size
    
    def test_buffer_statistics(self):
        """Test buffer statistics."""
        # Empty buffer
        stats = self.buffer.get_stats()
        assert stats["size"] == 0
        assert stats["capacity"] == self.capacity
        
        # Add some data
        for i in range(20):
            state = torch.randn(3)
            action = torch.randn(2)
            self.buffer.add(state, action, 0.1 + 0.01*i, 0.5 + 0.02*i, False, -1.0)
        
        self.buffer.compute_advantages()
        
        stats = self.buffer.get_stats()
        assert stats["size"] == 20
        assert "avg_reward" in stats
        assert "avg_value" in stats
        assert "avg_advantage" in stats
        assert "avg_return" in stats
    
    def test_buffer_clear(self):
        """Test buffer clearing."""
        # Add some data
        for i in range(10):
            state = torch.randn(2)
            action = torch.randn(1)
            self.buffer.add(state, action, 0.0, 0.0, False, 0.0)
        
        assert self.buffer.size == 10
        
        # Clear buffer
        self.buffer.clear()
        
        assert self.buffer.size == 0
        assert self.buffer.position == 0
        assert len(self.buffer.advantages) == 0
        assert len(self.buffer.returns) == 0


class TestGAENumericalAccuracy:
    """Test numerical accuracy of GAE computations."""
    
    def test_gae_formula_verification(self):
        """Test GAE formula against manual computation."""
        gamma = 0.9
        gae_lambda = 0.8
        T = 10
        
        # Simple test case
        rewards = torch.ones(T) * 0.1
        values = torch.zeros(T)
        dones = torch.zeros(T)
        next_values = torch.zeros(T)
        
        gae = GeneralizedAdvantageEstimation(gamma, gae_lambda, normalize_advantages=False)
        returns, advantages = gae.compute_gae(rewards, values, dones, next_values)
        
        # Manual computation of GAE
        td_errors = rewards + gamma * next_values - values  # All should be 0.1
        
        manual_advantages = torch.zeros(T)
        gae_val = 0.0
        
        for t in reversed(range(T)):
            gae_val = td_errors[t] + gamma * gae_lambda * gae_val
            manual_advantages[t] = gae_val
        
        # Compare
        assert torch.allclose(advantages, manual_advantages, atol=1e-6)
    
    def test_gae_convergence_properties(self):
        """Test GAE convergence as λ → 1."""
        gamma = 0.99
        T = 20
        
        rewards = torch.randn(T) * 0.1
        values = torch.randn(T) * 0.2
        dones = torch.zeros(T)
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
        
        # Compare GAE with different λ values
        lambda_values = [0.0, 0.5, 0.9, 0.95, 0.99]
        advantages_list = []
        
        for lam in lambda_values:
            gae = GeneralizedAdvantageEstimation(gamma, lam, normalize_advantages=False)
            _, advantages = gae.compute_gae(rewards, values, dones, next_values)
            advantages_list.append(advantages)
        
        # As λ increases, advantages should approach Monte Carlo estimates
        # (more smoothing with higher λ)
        for i in range(len(lambda_values)-1):
            # Higher λ should give different results
            diff = torch.norm(advantages_list[i+1] - advantages_list[i])
            assert diff > 1e-6, f"λ={lambda_values[i]} vs λ={lambda_values[i+1]} should differ"
    
    def test_gae_discount_factor_consistency(self):
        """Test GAE consistency with discount factor."""
        gae_lambda = 0.95
        T = 15
        
        rewards = torch.ones(T) * 0.2  # Constant rewards
        values = torch.zeros(T)
        dones = torch.zeros(T)
        next_values = torch.zeros(T)
        
        # Test different gamma values
        gamma_values = [0.9, 0.95, 0.99]
        
        for gamma in gamma_values:
            gae = GeneralizedAdvantageEstimation(gamma, gae_lambda, normalize_advantages=False)
            returns, advantages = gae.compute_gae(rewards, values, dones, next_values)
            
            # For constant rewards, returns should follow geometric series
            # R_t = r * (1 - γ^{T-t}) / (1 - γ) for infinite horizon approximation
            
            # Check that returns are reasonable
            assert returns[0] > returns[-1]  # Earlier timesteps should have higher returns
            assert torch.all(returns >= rewards)  # Returns should be at least as large as immediate rewards


def test_gae_integration_with_value_function():
    """Test GAE integration with value function training."""
    state_dim = 4
    T = 50
    
    # Create value function
    value_function = ValueFunction(state_dim, hidden_sizes=[16, 16])
    
    # Create GAE
    gae = GeneralizedAdvantageEstimation(gamma=0.99, gae_lambda=0.97)
    
    # Generate trajectory
    states = torch.randn(T, state_dim)
    rewards = torch.randn(T) * 0.1 + 0.05
    dones = torch.zeros(T)
    dones[25] = 1.0  # Episode boundary
    
    # Get initial value predictions
    with torch.no_grad():
        values = value_function(states).squeeze()
        next_values = torch.cat([values[1:], torch.tensor([0.0])])
    
    # Compute advantages
    returns, advantages = gae.compute_gae(rewards, values, dones, next_values)
    
    # Train value function on returns
    initial_loss = value_function.update(states, returns, num_epochs=20)
    
    # Get improved value predictions
    with torch.no_grad():
        improved_values = value_function(states).squeeze()
        improved_next_values = torch.cat([improved_values[1:], torch.tensor([0.0])])
    
    # Recompute advantages with improved values
    new_returns, new_advantages = gae.compute_gae(rewards, improved_values, dones, improved_next_values)
    
    # Value function should provide better estimates
    # (advantages should have lower variance with better value estimates)
    old_advantage_var = torch.var(advantages)
    new_advantage_var = torch.var(new_advantages)
    
    # Allow for some cases where variance doesn't decrease due to randomness
    assert new_advantage_var <= old_advantage_var * 1.2, "Value function training should improve advantage estimates"


if __name__ == "__main__":
    # Run the built-in test
    test_gae_implementation()
    
    # Run pytest
    pytest.main([__file__, "-v"])