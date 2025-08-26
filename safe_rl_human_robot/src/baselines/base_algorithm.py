"""
Base classes for baseline algorithm implementations.

This module provides abstract base classes and common interfaces for
implementing state-of-the-art baseline algorithms for benchmarking.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass, field
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """Base configuration for baseline algorithms."""
    
    # Algorithm identification
    name: str
    algorithm_type: str  # 'safe_rl', 'classical_control'
    
    # Common hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    
    # Safety parameters
    cost_limit: float = 25.0  # Constraint threshold
    safety_weight: float = 1.0
    lagrange_lr: float = 5e-3
    
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = 'relu'
    
    # Training parameters
    gradient_steps: int = 1
    learning_starts: int = 10000
    train_freq: int = 1
    target_update_interval: int = 1
    
    # Device and reproducibility
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: Optional[int] = None
    
    # Logging and monitoring
    verbose: int = 1
    log_interval: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key) for key in self.__dataclass_fields__.keys()
        }


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    
    episode_return: float = 0.0
    episode_cost: float = 0.0
    episode_length: int = 0
    constraint_violation: bool = False
    
    # Algorithm-specific metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    
    # Safety-specific metrics
    cost_value_loss: float = 0.0
    lagrange_multiplier: float = 0.0
    
    # Timing metrics
    step_time: float = 0.0
    update_time: float = 0.0
    
    # Additional metrics
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if key != 'extra_metrics':
                if isinstance(value, (int, float, bool)):
                    result[key] = float(value)
            else:
                result.update(value)
        return result


class BaselineAlgorithm(ABC):
    """Abstract base class for baseline algorithms."""
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds for reproducibility
        if config.seed is not None:
            self._set_seeds(config.seed)
        
        # Training state
        self.total_timesteps = 0
        self.episode_count = 0
        self.training_metrics: List[TrainingMetrics] = []
        
        # Safety monitoring
        self.constraint_violations = 0
        self.total_episodes = 0
        
        # Algorithm-specific initialization
        self._initialize_algorithm()
        
        logger.info(f"Initialized {config.name} algorithm")
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    @abstractmethod
    def _initialize_algorithm(self):
        """Initialize algorithm-specific components."""
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray, 
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        pass
    
    @abstractmethod
    def learn(self, total_timesteps: int, **kwargs) -> 'BaselineAlgorithm':
        """Train the algorithm.
        
        Args:
            total_timesteps: Number of timesteps to train
            **kwargs: Additional training arguments
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update algorithm with a batch of data.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Dictionary of update metrics
        """
        pass
    
    def evaluate(self, env, num_episodes: int = 10, 
                 deterministic: bool = True) -> Dict[str, float]:
        """Evaluate algorithm performance.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_returns = []
        episode_costs = []
        episode_lengths = []
        constraint_violations = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            episode_return = 0.0
            episode_cost = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                
                episode_return += reward
                episode_length += 1
                
                # Extract cost information
                if 'cost' in info:
                    episode_cost += info['cost']
                elif 'constraint_violation' in info:
                    episode_cost += float(info['constraint_violation'])
                
                # Check for constraint violations
                if self._check_constraint_violation(info):
                    constraint_violations += 1
                    break  # Early termination on violation
            
            episode_returns.append(episode_return)
            episode_costs.append(episode_cost)
            episode_lengths.append(episode_length)
        
        return {
            'mean_episode_return': np.mean(episode_returns),
            'std_episode_return': np.std(episode_returns),
            'mean_episode_cost': np.mean(episode_costs),
            'std_episode_cost': np.std(episode_costs),
            'mean_episode_length': np.mean(episode_lengths),
            'constraint_violation_rate': constraint_violations / num_episodes,
            'success_rate': 1.0 - (constraint_violations / num_episodes)
        }
    
    def _check_constraint_violation(self, info: Dict[str, Any]) -> bool:
        """Check if current step violates safety constraints."""
        if 'constraint_violation' in info:
            return bool(info['constraint_violation'])
        elif 'cost' in info:
            return info['cost'] > self.config.cost_limit
        elif 'safety_violation' in info:
            return bool(info['safety_violation'])
        return False
    
    def record_training_metrics(self, metrics: TrainingMetrics):
        """Record training metrics."""
        self.training_metrics.append(metrics)
        
        # Update counters
        if metrics.constraint_violation:
            self.constraint_violations += 1
        self.total_episodes += 1
    
    def get_training_metrics(self) -> List[Dict[str, float]]:
        """Get training metrics as list of dictionaries."""
        return [metrics.to_dict() for metrics in self.training_metrics]
    
    def get_safety_metrics(self) -> Dict[str, float]:
        """Get safety-specific metrics."""
        if self.total_episodes == 0:
            return {'violation_rate': 0.0, 'total_violations': 0, 'total_episodes': 0}
        
        return {
            'violation_rate': self.constraint_violations / self.total_episodes,
            'total_violations': self.constraint_violations,
            'total_episodes': self.total_episodes,
            'safety_success_rate': 1.0 - (self.constraint_violations / self.total_episodes)
        }
    
    def save(self, path: str):
        """Save algorithm state."""
        state = {
            'config': self.config.to_dict(),
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'constraint_violations': self.constraint_violations,
            'total_episodes': self.total_episodes
        }
        
        # Add algorithm-specific state
        state.update(self._get_save_state())
        
        torch.save(state, path)
        logger.info(f"Saved algorithm state to {path}")
    
    def load(self, path: str):
        """Load algorithm state."""
        state = torch.load(path, map_location=self.device)
        
        self.total_timesteps = state['total_timesteps']
        self.episode_count = state['episode_count']
        self.constraint_violations = state['constraint_violations']
        self.total_episodes = state['total_episodes']
        
        # Load algorithm-specific state
        self._load_from_state(state)
        
        logger.info(f"Loaded algorithm state from {path}")
    
    @abstractmethod
    def _get_save_state(self) -> Dict[str, Any]:
        """Get algorithm-specific state for saving."""
        pass
    
    @abstractmethod
    def _load_from_state(self, state: Dict[str, Any]):
        """Load algorithm-specific state."""
        pass
    
    def get_config(self) -> AlgorithmConfig:
        """Get algorithm configuration."""
        return self.config
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information."""
        return {
            'name': self.config.name,
            'algorithm_type': self.config.algorithm_type,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'device': str(self.device),
            'safety_metrics': self.get_safety_metrics()
        }


class NetworkBase(nn.Module):
    """Base neural network class for baseline algorithms."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu'):
        super().__init__()
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class ValueNetwork(NetworkBase):
    """Value function network."""
    
    def __init__(self, state_dim: int, hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu'):
        super().__init__(state_dim, 1, hidden_sizes, activation)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute state value."""
        return super().forward(state).squeeze(-1)


class QNetwork(NetworkBase):
    """Q-function network."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu'):
        super().__init__(state_dim + action_dim, 1, hidden_sizes, activation)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value for state-action pair."""
        x = torch.cat([state, action], dim=-1)
        return super().forward(x).squeeze(-1)


class PolicyNetwork(NetworkBase):
    """Policy network with continuous action output."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu',
                 max_action: float = 1.0):
        # Output both mean and log_std
        super().__init__(state_dim, action_dim * 2, hidden_sizes, activation)
        
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Separate layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_dim)
        
        # Log std bounds
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action mean and log std."""
        # Get features from base network (excluding final layer)
        features = self.network[:-1](state)
        
        # Compute mean and log_std
        mean = self.max_action * torch.tanh(self.mean_layer(features))
        log_std = torch.clamp(self.log_std_layer(features), 
                             self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Apply tanh and scale
        action = torch.tanh(x_t) * self.max_action
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        # Enforce action bounds
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, mean


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros((capacity, action_dim), device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros(capacity, device=device, dtype=torch.bool)
        self.costs = torch.zeros(capacity, device=device)  # For safety
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, cost: float = 0.0):
        """Add transition to buffer."""
        self.states[self.position] = torch.FloatTensor(state).to(self.device)
        self.actions[self.position] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.position] = reward
        self.next_states[self.position] = torch.FloatTensor(next_state).to(self.device)
        self.dones[self.position] = done
        self.costs[self.position] = cost
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'costs': self.costs[indices]
        }
    
    def __len__(self) -> int:
        return self.size


class LagrangeMultiplier:
    """Lagrange multiplier for constrained optimization."""
    
    def __init__(self, initial_value: float = 1.0, lr: float = 5e-3,
                 constraint_limit: float = 25.0):
        self.value = torch.tensor(initial_value, requires_grad=True)
        self.lr = lr
        self.constraint_limit = constraint_limit
        self.optimizer = torch.optim.Adam([self.value], lr=lr)
    
    def update(self, constraint_violation: float):
        """Update Lagrange multiplier based on constraint violation."""
        # Compute Lagrangian gradient: ∇λ L = constraint_violation - constraint_limit
        lagrangian_grad = constraint_violation - self.constraint_limit
        
        # Update using gradient ascent (since we minimize the Lagrangian)
        self.optimizer.zero_grad()
        loss = -self.value * lagrangian_grad  # Negative for gradient ascent
        loss.backward()
        self.optimizer.step()
        
        # Ensure non-negativity
        with torch.no_grad():
            self.value.clamp_(min=0.0)
    
    def get_value(self) -> float:
        """Get current multiplier value."""
        return self.value.item()


# Utility functions for baseline algorithms

def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float):
    """Soft update target network parameters."""
    with torch.no_grad():
        for target_param, source_param in zip(target_net.parameters(), 
                                            source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + 
                                  (1.0 - tau) * target_param.data)


def hard_update(target_net: nn.Module, source_net: nn.Module):
    """Hard update target network parameters."""
    target_net.load_state_dict(source_net.state_dict())


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute gradient norm for monitoring."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def clip_gradients(model: nn.Module, max_norm: float = 1.0):
    """Clip gradients by norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def polyak_update(old_val: torch.Tensor, new_val: torch.Tensor, 
                  polyak: float) -> torch.Tensor:
    """Polyak averaging update."""
    return polyak * old_val + (1 - polyak) * new_val