"""
Safe policy implementation for Constrained Policy Optimization (CPO).

This module implements neural network policies π_θ(a|s) with safety checks,
policy gradient computation, and action sampling for human-robot shared control.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Neural network policy π_θ(a|s) for continuous control.
    
    Implements a feedforward network that outputs mean and log standard deviation
    for a multivariate Gaussian policy distribution.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256],
                 activation: str = "tanh", log_std_init: float = -0.5):
        """
        Initialize policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("tanh", "relu", "elu")
            log_std_init: Initial value for log standard deviation
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_init = log_std_init
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build network layers
        layers = []
        prev_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                self.activation
            ])
            prev_size = hidden_size
            
        # Mean output layer
        layers.append(nn.Linear(prev_size, action_dim))
        self.mean_net = nn.Sequential(*layers)
        
        # Log standard deviation parameters (state-independent)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize network weights using orthogonal initialization."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state: Input state [batch_size, state_dim]
            
        Returns:
            Tuple of (mean, std) where:
                mean: Action mean [batch_size, action_dim]
                std: Action standard deviation [batch_size, action_dim]
        """
        mean = self.mean_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
    
    def get_distribution(self, state: torch.Tensor) -> Normal:
        """
        Get policy distribution π_θ(·|s).
        
        Args:
            state: Input state [batch_size, state_dim]
            
        Returns:
            Multivariate normal distribution
        """
        mean, std = self.forward(state)
        return Normal(mean, std)


class SafePolicy:
    """
    Safe policy implementation with constraint-aware action sampling.
    
    Combines neural network policy with safety constraint checking and
    policy gradient computation for CPO optimization.
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 constraint_manager: Optional[Any] = None,
                 hidden_sizes: List[int] = [256, 256],
                 device: str = "cpu"):
        """
        Initialize safe policy.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            constraint_manager: ConstraintManager for safety checking
            hidden_sizes: Hidden layer sizes for policy network
            device: Device for computations ("cpu" or "cuda")
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.constraint_manager = constraint_manager
        
        # Initialize policy network
        self.policy_net = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim, 
            hidden_sizes=hidden_sizes
        ).to(device)
        
        # Policy statistics
        self.action_history: List[torch.Tensor] = []
        self.constraint_violations = 0
        self.total_actions = 0
        
    def sample_action(self, state: torch.Tensor, deterministic: bool = False,
                     max_safety_iterations: int = 10) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Sample safe action from policy with constraint checking.
        
        Args:
            state: Current state [batch_size, state_dim]
            deterministic: If True, return mean action
            max_safety_iterations: Maximum attempts to find safe action
            
        Returns:
            Tuple of (action, info_dict) where info_dict contains:
                - log_prob: Log probability of sampled action
                - entropy: Policy entropy
                - is_safe: Whether action satisfies constraints
                - safety_iterations: Number of iterations needed
        """
        self.policy_net.eval()
        
        with torch.no_grad():
            dist = self.policy_net.get_distribution(state)
            
            if deterministic:
                action = dist.mean
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                # Iteratively sample until safe action found
                is_safe = False
                safety_iterations = 0
                
                for _ in range(max_safety_iterations):
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    
                    # Check safety if constraint manager available
                    if self.constraint_manager is not None:
                        safety_status = self.constraint_manager.check_safety(state, action)
                        is_safe = all(safety_status.values())
                        
                        if is_safe:
                            break
                        else:
                            safety_iterations += 1
                    else:
                        is_safe = True
                        break
                
                # If no safe action found, use mean action as fallback
                if not is_safe:
                    logger.warning("No safe action found, using mean action")
                    action = dist.mean
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    self.constraint_violations += 1
                    
        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)
        
        # Update statistics
        self.total_actions += state.shape[0]
        self.action_history.append(action.clone())
        
        info = {
            "log_prob": log_prob,
            "entropy": entropy,
            "is_safe": is_safe,
            "safety_iterations": safety_iterations
        }
        
        return action, info
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim]
            
        Returns:
            Dict containing log_probs, entropy, and policy statistics
        """
        self.policy_net.eval()
        
        dist = self.policy_net.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return {
            "log_probs": log_probs,
            "entropy": entropy,
            "mean": dist.mean,
            "std": dist.stddev
        }
    
    def compute_policy_gradient(self, states: torch.Tensor, actions: torch.Tensor,
                              advantages: torch.Tensor, 
                              old_log_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute policy gradient ∇_θ J(θ).
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim] 
            advantages: Advantage estimates [batch_size]
            old_log_probs: Old policy log probabilities for PPO-style updates
            
        Returns:
            Policy loss for backpropagation
        """
        self.policy_net.train()
        
        # Compute current policy probabilities
        evaluation = self.evaluate_actions(states, actions)
        log_probs = evaluation["log_probs"]
        entropy = evaluation["entropy"]
        
        if old_log_probs is not None:
            # PPO-style policy ratio
            ratio = torch.exp(log_probs - old_log_probs)
            policy_loss = -(ratio * advantages).mean()
        else:
            # Standard policy gradient
            policy_loss = -(log_probs * advantages).mean()
        
        # Add entropy regularization
        entropy_coef = 0.01
        policy_loss -= entropy_coef * entropy.mean()
        
        return policy_loss
    
    def compute_constraint_gradient(self, states: torch.Tensor, 
                                  actions: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute constraint gradient for CPO update.
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim]
            
        Returns:
            Constraint gradient with respect to policy parameters
        """
        if self.constraint_manager is None:
            return None
            
        # Enable gradient computation
        states = states.requires_grad_(True)
        actions = actions.requires_grad_(True)
        
        # Sample actions from current policy
        dist = self.policy_net.get_distribution(states)
        sampled_actions = dist.rsample()  # Reparameterized sampling
        
        # Compute constraint violations
        constraint_values = []
        for constraint in self.constraint_manager.constraints:
            g_val = constraint.evaluate(states, sampled_actions)
            constraint_values.append(g_val)
        
        # Combined constraint violation
        total_violation = torch.stack(constraint_values).sum(dim=0).mean()
        
        return total_violation
    
    def update_trust_region_step(self, states: torch.Tensor, actions: torch.Tensor,
                               advantages: torch.Tensor, max_kl: float = 0.01,
                               damping_coef: float = 0.1) -> Dict[str, float]:
        """
        Perform trust region policy update for CPO.
        
        Implements the trust region constraint ||θ - θ_k||² ≤ δ.
        
        Args:
            states: State tensor for policy gradient estimation
            actions: Action tensor
            advantages: Advantage estimates
            max_kl: Maximum KL divergence for trust region
            damping_coef: Damping coefficient for conjugate gradient
            
        Returns:
            Update statistics dictionary
        """
        # Store old policy parameters
        old_params = [p.clone() for p in self.policy_net.parameters()]
        
        # Compute policy gradient
        policy_loss = self.compute_policy_gradient(states, actions, advantages)
        
        # Compute gradients
        policy_grads = torch.autograd.grad(policy_loss, self.policy_net.parameters(), 
                                         create_graph=True)
        policy_grads = torch.cat([g.view(-1) for g in policy_grads])
        
        # Compute Fisher Information Matrix approximation
        old_dist = self.policy_net.get_distribution(states)
        
        def get_kl():
            new_dist = self.policy_net.get_distribution(states)
            return torch.distributions.kl_divergence(old_dist, new_dist).mean()
        
        kl_grad = torch.autograd.grad(get_kl(), self.policy_net.parameters(), 
                                    create_graph=True)
        kl_grad = torch.cat([g.view(-1) for g in kl_grad])
        
        # Conjugate gradient to solve Fisher * x = g
        def Fvp(v):
            # Fisher-vector product
            kl_v = (kl_grad * v).sum()
            grads = torch.autograd.grad(kl_v, self.policy_net.parameters(),
                                      retain_graph=True)
            return torch.cat([g.view(-1) for g in grads]) + damping_coef * v
        
        search_dir = self._conjugate_gradient(Fvp, policy_grads)
        
        # Compute step size using line search
        step_size = self._line_search(states, old_dist, search_dir, max_kl)
        
        # Apply update
        self._apply_update(old_params, search_dir * step_size)
        
        # Compute final KL divergence
        final_kl = get_kl().item()
        
        return {
            "kl_divergence": final_kl,
            "step_size": step_size.item() if isinstance(step_size, torch.Tensor) else step_size,
            "policy_loss": policy_loss.item()
        }
    
    def _conjugate_gradient(self, Avp_func, b: torch.Tensor, max_iter: int = 10,
                           tol: float = 1e-8) -> torch.Tensor:
        """Solve Ax = b using conjugate gradient method."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_old = torch.dot(r, r)
        
        for _ in range(max_iter):
            Ap = Avp_func(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            
            if r_dot_new < tol:
                break
                
            beta = r_dot_new / (r_dot_old + 1e-8)
            p = r + beta * p
            r_dot_old = r_dot_new
            
        return x
    
    def _line_search(self, states: torch.Tensor, old_dist, search_dir: torch.Tensor,
                    max_kl: float, alpha: float = 0.5, max_iter: int = 10) -> float:
        """Perform line search to find appropriate step size."""
        step_size = 1.0
        
        for _ in range(max_iter):
            # Try step
            old_params = [p.clone() for p in self.policy_net.parameters()]
            self._apply_update(old_params, search_dir * step_size)
            
            # Check KL constraint
            new_dist = self.policy_net.get_distribution(states)
            kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            
            if kl <= max_kl:
                break
            else:
                step_size *= alpha
                
        return step_size
    
    def _apply_update(self, old_params: List[torch.Tensor], update: torch.Tensor) -> None:
        """Apply parameter update to policy network."""
        idx = 0
        for p in self.policy_net.parameters():
            param_size = p.numel()
            p.data = old_params[idx // param_size].view_as(p) + \
                    update[idx:idx+param_size].view_as(p)
            idx += param_size
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy performance statistics."""
        violation_rate = self.constraint_violations / max(self.total_actions, 1)
        
        return {
            "total_actions": self.total_actions,
            "constraint_violations": self.constraint_violations,
            "violation_rate": violation_rate,
            "parameter_count": sum(p.numel() for p in self.policy_net.parameters()),
            "recent_action_std": torch.std(torch.cat(self.action_history[-100:])).item() 
                                if len(self.action_history) > 0 else 0.0
        }
    
    def save(self, filepath: str) -> None:
        """Save policy network state."""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "statistics": self.get_policy_statistics()
        }, filepath)
        logger.info(f"Policy saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load policy network state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        logger.info(f"Policy loaded from {filepath}")
    
    def to(self, device: str) -> "SafePolicy":
        """Move policy to device."""
        self.device = device
        self.policy_net.to(device)
        return self