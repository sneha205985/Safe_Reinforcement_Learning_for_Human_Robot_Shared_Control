"""
Generalized Advantage Estimation (GAE) for CPO.

This module implements GAE for computing advantages and returns in reinforcement learning:
- GAE: A^{GAE(γ,λ)}_t = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}^V
- TD Error: δ_t^V = r_t + γV(s_{t+1}) - V(s_t)
- Value function baseline estimation
- Temporal difference learning
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logging_utils import log_execution_time

logger = logging.getLogger(__name__)


@dataclass
class GAEConfig:
    """Configuration for Generalized Advantage Estimation."""
    gamma: float = 0.99              # Discount factor γ
    gae_lambda: float = 0.97         # GAE smoothing parameter λ
    normalize_advantages: bool = True # Normalize advantages to unit variance
    use_gae: bool = True            # Use GAE vs simple advantage estimation
    value_clipping: bool = True     # Clip value function updates
    clip_range: float = 0.2         # Value function clipping range


class GAEResult(NamedTuple):
    """Result from GAE computation."""
    advantages: torch.Tensor    # Computed advantages [T]
    returns: torch.Tensor      # Computed returns [T]
    values: torch.Tensor       # Value function predictions [T]
    td_errors: torch.Tensor    # TD errors δ_t [T]


class ValueFunction(nn.Module):
    """
    Value function V^π(s) for advantage estimation.
    
    Implements a neural network that estimates state values for policy evaluation
    and advantage computation in policy gradient methods.
    """
    
    def __init__(self,
                 state_dim: int,
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = "tanh",
                 learning_rate: float = 1e-3,
                 device: str = "cpu"):
        """
        Initialize value function network.
        
        Args:
            state_dim: Dimension of state space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("tanh", "relu", "elu")
            learning_rate: Learning rate for value function optimization
            device: Device for computations
        """
        super().__init__()
        self.state_dim = state_dim
        self.device = device
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Statistics
        self.training_losses = []
        self.update_count = 0
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"ValueFunction initialized: {state_dim} -> {hidden_sizes} -> 1")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value network.
        
        Args:
            states: State tensor [batch_size, state_dim]
            
        Returns:
            Value predictions [batch_size, 1]
        """
        return self.network(states)
    
    def predict(self, states: torch.Tensor) -> torch.Tensor:
        """
        Predict values for given states.
        
        Args:
            states: State tensor [batch_size, state_dim]
            
        Returns:
            Value predictions [batch_size]
        """
        with torch.no_grad():
            values = self.forward(states).squeeze(-1)
        return values
    
    def update(self, 
               states: torch.Tensor, 
               targets: torch.Tensor,
               num_epochs: int = 80,
               batch_size: int = 256) -> float:
        """
        Update value function using target values.
        
        Args:
            states: State batch [num_samples, state_dim]
            targets: Target values [num_samples]
            num_epochs: Number of training epochs
            batch_size: Mini-batch size for training
            
        Returns:
            Average training loss
        """
        self.train()
        
        num_samples = states.shape[0]
        total_loss = 0.0
        num_batches = 0
        
        with log_execution_time(logger, f"Value function update ({num_epochs} epochs)"):
            for epoch in range(num_epochs):
                # Shuffle data
                indices = torch.randperm(num_samples, device=self.device)
                shuffled_states = states[indices]
                shuffled_targets = targets[indices]
                
                epoch_loss = 0.0
                epoch_batches = 0
                
                # Mini-batch training
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_states = shuffled_states[start_idx:end_idx]
                    batch_targets = shuffled_targets[start_idx:end_idx]
                    
                    # Forward pass
                    predictions = self.forward(batch_states).squeeze(-1)
                    
                    # Compute loss (MSE)
                    loss = F.mse_loss(predictions, batch_targets)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_batches += 1
                
                total_loss += epoch_loss / max(epoch_batches, 1)
                num_batches += 1
                
                # Log progress
                if epoch % 20 == 0:
                    avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
                    logger.debug(f"Value function epoch {epoch}: loss = {avg_epoch_loss:.6f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        self.training_losses.append(avg_loss)
        self.update_count += 1
        
        logger.debug(f"Value function update completed: avg_loss = {avg_loss:.6f}")
        return avg_loss
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get value function training statistics."""
        if not self.training_losses:
            return {"updates": 0, "avg_loss": 0.0, "recent_loss": 0.0}
        
        return {
            "updates": self.update_count,
            "avg_loss": np.mean(self.training_losses),
            "recent_loss": self.training_losses[-1],
            "loss_std": np.std(self.training_losses)
        }


class GeneralizedAdvantageEstimation:
    """
    Generalized Advantage Estimation (GAE) implementation.
    
    Computes advantages using the GAE formula:
    A^{GAE(γ,λ)}_t = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}^V
    
    where δ_t^V = r_t + γV(s_{t+1}) - V(s_t) is the TD error.
    """
    
    def __init__(self,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97,
                 normalize_advantages: bool = True,
                 use_gae: bool = True):
        """
        Initialize GAE estimator.
        
        Args:
            gamma: Discount factor γ
            gae_lambda: GAE smoothing parameter λ
            normalize_advantages: Whether to normalize advantages
            use_gae: Use GAE vs simple advantage estimation
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.use_gae = use_gae
        
        # Statistics
        self.computation_stats = {
            "total_calls": 0,
            "total_timesteps": 0,
            "avg_advantage": 0.0,
            "avg_return": 0.0
        }
        
        logger.info(f"GAE initialized: γ={gamma}, λ={gae_lambda}, normalize={normalize_advantages}")
    
    def compute_gae(self,
                   rewards: torch.Tensor,
                   values: torch.Tensor,
                   dones: torch.Tensor,
                   next_values: torch.Tensor,
                   masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Reward sequence [T]
            values: Value function predictions V(s_t) [T]
            dones: Episode termination flags [T]
            next_values: Next state values V(s_{t+1}) [T]
            masks: Optional mask for valid timesteps [T]
            
        Returns:
            Tuple of (returns, advantages)
        """
        T = len(rewards)
        
        if masks is None:
            masks = torch.ones_like(rewards)
        
        # Convert dones to masks (continue = 1 - done)
        continues = 1.0 - dones.float()
        
        if self.use_gae:
            returns, advantages = self._compute_gae_advantages(
                rewards, values, continues, next_values, masks
            )
        else:
            returns, advantages = self._compute_simple_advantages(
                rewards, values, continues, masks
            )
        
        # Normalize advantages
        if self.normalize_advantages:
            valid_advantages = advantages[masks.bool()]
            if len(valid_advantages) > 1:
                advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)
        
        # Update statistics
        self._update_stats(returns, advantages, masks)
        
        return returns, advantages
    
    def _compute_gae_advantages(self,
                              rewards: torch.Tensor,
                              values: torch.Tensor,
                              continues: torch.Tensor,
                              next_values: torch.Tensor,
                              masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages using the recursive formula.
        
        A^{GAE}_t = δ_t + (γλ)A^{GAE}_{t+1}
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute TD errors
        td_errors = rewards + self.gamma * continues * next_values - values
        
        # Backward pass to compute GAE advantages
        gae = 0.0
        for t in reversed(range(T)):
            if masks[t] > 0:  # Only compute for valid timesteps
                gae = td_errors[t] + self.gamma * self.gae_lambda * continues[t] * gae
                advantages[t] = gae
        
        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + values
        
        return returns, advantages
    
    def _compute_simple_advantages(self,
                                 rewards: torch.Tensor,
                                 values: torch.Tensor,
                                 continues: torch.Tensor,
                                 masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute simple advantages without GAE smoothing.
        
        A_t = R_t - V(s_t) where R_t is the discounted return.
        """
        T = len(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute discounted returns
        running_return = 0.0
        for t in reversed(range(T)):
            if masks[t] > 0:
                running_return = rewards[t] + self.gamma * continues[t] * running_return
                returns[t] = running_return
        
        # Advantages = Returns - Values
        advantages = returns - values
        
        return returns, advantages
    
    def compute_n_step_returns(self,
                              rewards: torch.Tensor,
                              values: torch.Tensor,
                              dones: torch.Tensor,
                              n_steps: int = 5) -> torch.Tensor:
        """
        Compute n-step returns for bootstrapped advantage estimation.
        
        R_t^{(n)} = ∑_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})
        
        Args:
            rewards: Reward sequence [T]
            values: Value predictions [T]
            dones: Episode termination flags [T]
            n_steps: Number of steps for n-step returns
            
        Returns:
            N-step returns [T]
        """
        T = len(rewards)
        n_step_returns = torch.zeros_like(rewards)
        
        for t in range(T):
            return_sum = 0.0
            discount = 1.0
            
            # Sum rewards for n steps or until episode end
            for k in range(n_steps):
                if t + k >= T or (k > 0 and dones[t + k - 1]):
                    break
                
                return_sum += discount * rewards[t + k]
                discount *= self.gamma
            
            # Add bootstrapped value
            bootstrap_idx = min(t + n_steps, T - 1)
            if not dones[bootstrap_idx]:
                return_sum += discount * values[bootstrap_idx]
            
            n_step_returns[t] = return_sum
        
        return n_step_returns
    
    def _update_stats(self, returns: torch.Tensor, advantages: torch.Tensor, masks: torch.Tensor) -> None:
        """Update computation statistics."""
        valid_indices = masks.bool()
        
        if valid_indices.any():
            valid_returns = returns[valid_indices]
            valid_advantages = advantages[valid_indices]
            
            self.computation_stats["total_calls"] += 1
            self.computation_stats["total_timesteps"] += valid_indices.sum().item()
            
            # Running average
            alpha = 0.1  # Smoothing factor
            self.computation_stats["avg_advantage"] = (
                (1 - alpha) * self.computation_stats["avg_advantage"] + 
                alpha * valid_advantages.mean().item()
            )
            self.computation_stats["avg_return"] = (
                (1 - alpha) * self.computation_stats["avg_return"] + 
                alpha * valid_returns.mean().item()
            )
    
    def get_computation_stats(self) -> Dict[str, float]:
        """Get GAE computation statistics."""
        return self.computation_stats.copy()


class AdvantageBuffer:
    """
    Buffer for storing and managing advantage estimation data.
    
    Efficiently stores trajectory data and computes advantages in batches.
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.97):
        """
        Initialize advantage buffer.
        
        Args:
            capacity: Maximum buffer capacity
            gamma: Discount factor
            gae_lambda: GAE smoothing parameter
        """
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Buffer storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
        
        # Computed advantages and returns
        self.advantages = []
        self.returns = []
        
        self.size = 0
        self.position = 0
        
        # GAE estimator
        self.gae = GeneralizedAdvantageEstimation(gamma, gae_lambda)
    
    def add(self,
            state: torch.Tensor,
            action: torch.Tensor,
            reward: float,
            value: float,
            done: bool,
            log_prob: float) -> None:
        """Add single timestep to buffer."""
        if self.size < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.values.append(None)
            self.dones.append(None)
            self.log_probs.append(None)
            self.size += 1
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.values[self.position] = value
        self.dones[self.position] = done
        self.log_probs[self.position] = log_prob
        
        self.position = (self.position + 1) % self.capacity
    
    def add_trajectory(self,
                      trajectory_data: Dict[str, torch.Tensor]) -> None:
        """Add complete trajectory to buffer."""
        trajectory_length = len(trajectory_data["rewards"])
        
        for t in range(trajectory_length):
            self.add(
                state=trajectory_data["states"][t],
                action=trajectory_data["actions"][t],
                reward=trajectory_data["rewards"][t].item(),
                value=trajectory_data["values"][t].item(),
                done=trajectory_data["dones"][t].item() > 0,
                log_prob=trajectory_data["log_probs"][t].item()
            )
    
    def compute_advantages(self, next_value: float = 0.0) -> None:
        """Compute advantages for all stored data."""
        if self.size == 0:
            return
        
        # Convert to tensors
        rewards_tensor = torch.tensor([self.rewards[i] for i in range(self.size)])
        values_tensor = torch.tensor([self.values[i] for i in range(self.size)])
        dones_tensor = torch.tensor([self.dones[i] for i in range(self.size)], dtype=torch.float32)
        
        # Compute next values (shift values by 1, add final next_value)
        next_values = torch.cat([values_tensor[1:], torch.tensor([next_value])])
        
        # Compute GAE
        returns, advantages = self.gae.compute_gae(
            rewards_tensor, values_tensor, dones_tensor, next_values
        )
        
        self.returns = returns.tolist()
        self.advantages = advantages.tolist()
    
    def get_batch(self, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Get batch of data for training."""
        if batch_size is None or batch_size > self.size:
            indices = list(range(self.size))
        else:
            indices = torch.randint(0, self.size, (batch_size,)).tolist()
        
        return {
            "states": torch.stack([self.states[i] for i in indices]),
            "actions": torch.stack([self.actions[i] for i in indices]),
            "rewards": torch.tensor([self.rewards[i] for i in indices]),
            "values": torch.tensor([self.values[i] for i in indices]),
            "dones": torch.tensor([self.dones[i] for i in indices]),
            "log_probs": torch.tensor([self.log_probs[i] for i in indices]),
            "advantages": torch.tensor([self.advantages[i] for i in indices]),
            "returns": torch.tensor([self.returns[i] for i in indices])
        }
    
    def clear(self) -> None:
        """Clear buffer contents."""
        self.size = 0
        self.position = 0
        self.advantages.clear()
        self.returns.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity
    
    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity}
        
        stats = {
            "size": self.size,
            "capacity": self.capacity,
            "avg_reward": np.mean(self.rewards[:self.size]),
            "avg_value": np.mean(self.values[:self.size]),
        }
        
        if self.advantages:
            stats.update({
                "avg_advantage": np.mean(self.advantages[:self.size]),
                "avg_return": np.mean(self.returns[:self.size])
            })
        
        return stats


def test_gae_implementation():
    """Test GAE implementation with synthetic data."""
    logger.info("Testing GAE implementation")
    
    # Create synthetic trajectory
    T = 100
    rewards = torch.randn(T) * 0.5 + 0.1  # Small positive rewards
    values = torch.randn(T) * 0.2 + 1.0   # Values around 1.0
    dones = torch.zeros(T)
    dones[49] = 1.0  # Episode boundary at timestep 49
    next_values = torch.cat([values[1:], torch.tensor([0.0])])
    
    # Initialize GAE
    gae = GeneralizedAdvantageEstimation(gamma=0.99, gae_lambda=0.97)
    
    # Compute advantages
    returns, advantages = gae.compute_gae(rewards, values, dones, next_values)
    
    # Basic checks
    assert returns.shape == (T,)
    assert advantages.shape == (T,)
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()
    
    # Check that advantages have approximately zero mean (after normalization)
    assert abs(advantages.mean().item()) < 0.1
    
    # Check that returns > rewards (due to discounting)
    assert (returns[:49].mean() > rewards[:49].mean()).item()  # First episode
    
    logger.info("GAE test passed")
    logger.info(f"  Average return: {returns.mean().item():.4f}")
    logger.info(f"  Average advantage: {advantages.mean().item():.4f}")
    logger.info(f"  Advantage std: {advantages.std().item():.4f}")


if __name__ == "__main__":
    test_gae_implementation()