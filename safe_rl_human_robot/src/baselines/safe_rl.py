"""
Advanced Safe Reinforcement Learning baseline implementations.

This module implements state-of-the-art Safe RL algorithms for benchmarking:
- SAC-Lagrangian: Soft Actor-Critic with Lagrangian constraints
- TD3-Constrained: Twin Delayed DDPG with safety constraints
- TRPO-Constrained: Trust Region Policy Optimization with constraints
- PPO-Lagrangian: Proximal Policy Optimization with Lagrangian
- Safe-DDPG: Deep Deterministic Policy Gradient with safety
- RCPO: Reward Constrained Policy Optimization
- CPO Variants: Different trust region solvers for CPO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List, Optional, Union
import copy
import logging
from scipy.optimize import minimize
import warnings

from .base_algorithm import (
    BaselineAlgorithm, AlgorithmConfig, TrainingMetrics,
    NetworkBase, PolicyNetwork, QNetwork, ValueNetwork,
    ReplayBuffer, LagrangeMultiplier, soft_update, hard_update
)

logger = logging.getLogger(__name__)


class SACLagrangian(BaselineAlgorithm):
    """Soft Actor-Critic with Lagrangian constraints for safety."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int,
                 max_action: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize SAC-Lagrangian specific components."""
        # Policy network
        self.policy = PolicyNetwork(
            self.state_dim, self.action_dim, 
            self.config.hidden_sizes, 'relu', self.max_action
        ).to(self.device)
        
        # Q-networks (twin critics)
        self.q1 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.q2 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        
        # Target Q-networks
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        
        # Cost Q-networks for safety
        self.cost_q1 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.cost_q2 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.cost_q1_target = copy.deepcopy(self.cost_q1)
        self.cost_q2_target = copy.deepcopy(self.cost_q2)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=self.config.learning_rate)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.config.learning_rate)
        self.cost_q1_optimizer = torch.optim.Adam(self.cost_q1.parameters(), lr=self.config.learning_rate)
        self.cost_q2_optimizer = torch.optim.Adam(self.cost_q2.parameters(), lr=self.config.learning_rate)
        
        # Lagrange multiplier for safety constraint
        self.lagrange_multiplier = LagrangeMultiplier(
            initial_value=1.0, lr=self.config.lagrange_lr, 
            constraint_limit=self.config.cost_limit
        )
        
        # Entropy temperature (automatic tuning)
        self.target_entropy = -self.action_dim
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.state_dim, self.action_dim, self.device
        )
        
        logger.info("SAC-Lagrangian initialized")
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = mean
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'SACLagrangian':
        """Train SAC-Lagrangian algorithm."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        for timestep in range(total_timesteps):
            # Collect experience
            if timestep < self.config.learning_starts:
                action = env.action_space.sample()  # Random action
            else:
                action, _ = self.predict(observation, deterministic=False)
            
            next_observation, reward, done, info = env.step(action)
            cost = info.get('cost', 0.0)
            
            # Store transition
            self.replay_buffer.add(observation, action, reward, next_observation, done, cost)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Update networks
            if len(self.replay_buffer) >= self.config.batch_size and timestep >= self.config.learning_starts:
                if timestep % self.config.train_freq == 0:
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    update_metrics = self.update(batch)
            
            if done:
                # Record episode metrics
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit
                )
                
                if timestep >= self.config.learning_starts:
                    metrics.lagrange_multiplier = self.lagrange_multiplier.get_value()
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update SAC-Lagrangian networks."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        costs = batch['costs']
        
        # Sample next actions from current policy
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            
            # Target Q-values (twin critics)
            target_q1 = self.q1_target(next_states, next_actions)
            target_q2 = self.q2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Target cost Q-values
            target_cost_q1 = self.cost_q1_target(next_states, next_actions)
            target_cost_q2 = self.cost_q2_target(next_states, next_actions)
            target_cost_q = torch.min(target_cost_q1, target_cost_q2)
            
            # SAC target values with entropy
            alpha = self.log_alpha.exp()
            next_q_values = target_q - alpha * next_log_probs
            next_cost_values = target_cost_q
            
            q_targets = rewards + self.config.gamma * (1 - dones.float()) * next_q_values
            cost_targets = costs + self.config.gamma * (1 - dones.float()) * next_cost_values
        
        # Update Q-networks
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        
        q1_loss = F.mse_loss(current_q1, q_targets)
        q2_loss = F.mse_loss(current_q2, q_targets)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update cost Q-networks
        current_cost_q1 = self.cost_q1(states, actions)
        current_cost_q2 = self.cost_q2(states, actions)
        
        cost_q1_loss = F.mse_loss(current_cost_q1, cost_targets)
        cost_q2_loss = F.mse_loss(current_cost_q2, cost_targets)
        
        self.cost_q1_optimizer.zero_grad()
        cost_q1_loss.backward()
        self.cost_q1_optimizer.step()
        
        self.cost_q2_optimizer.zero_grad()
        cost_q2_loss.backward()
        self.cost_q2_optimizer.step()
        
        # Update policy with Lagrangian
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_values = self.q1(states, new_actions)
        q2_values = self.q2(states, new_actions)
        q_values = torch.min(q1_values, q2_values)
        
        # Cost values for safety constraint
        cost_q1_values = self.cost_q1(states, new_actions)
        cost_q2_values = self.cost_q2(states, new_actions)
        cost_values = torch.min(cost_q1_values, cost_q2_values)
        
        # Policy loss with Lagrangian constraint
        alpha = self.log_alpha.exp()
        lagrange_lambda = self.lagrange_multiplier.value
        
        policy_loss = (alpha * log_probs - q_values + lagrange_lambda * cost_values).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update Lagrange multiplier
        avg_cost = cost_values.mean().item()
        self.lagrange_multiplier.update(avg_cost)
        
        # Soft update target networks
        soft_update(self.q1_target, self.q1, self.config.tau)
        soft_update(self.q2_target, self.q2, self.config.tau)
        soft_update(self.cost_q1_target, self.cost_q1, self.config.tau)
        soft_update(self.cost_q2_target, self.cost_q2, self.config.tau)
        
        return {
            'policy_loss': policy_loss.item(),
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'cost_q1_loss': cost_q1_loss.item(),
            'cost_q2_loss': cost_q2_loss.item(),
            'alpha': alpha.item(),
            'alpha_loss': alpha_loss.item(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value(),
            'average_cost': avg_cost
        }
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get SAC-Lagrangian specific state for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'cost_q1_state_dict': self.cost_q1.state_dict(),
            'cost_q2_state_dict': self.cost_q2.state_dict(),
            'log_alpha': self.log_alpha.item(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value()
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load SAC-Lagrangian specific state."""
        self.policy.load_state_dict(state['policy_state_dict'])
        self.q1.load_state_dict(state['q1_state_dict'])
        self.q2.load_state_dict(state['q2_state_dict'])
        self.cost_q1.load_state_dict(state['cost_q1_state_dict'])
        self.cost_q2.load_state_dict(state['cost_q2_state_dict'])
        
        with torch.no_grad():
            self.log_alpha.fill_(state['log_alpha'])
            self.lagrange_multiplier.value.fill_(state['lagrange_multiplier'])


class TD3Constrained(BaselineAlgorithm):
    """Twin Delayed DDPG with safety constraints."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int,
                 max_action: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # TD3 specific parameters
        self.policy_delay = 2
        self.noise_clip = 0.5
        self.exploration_noise = 0.1
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize TD3-Constrained specific components."""
        # Actor (policy) network
        self.actor = NetworkBase(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Add final tanh activation for bounded actions
        self.actor.network.add_module('tanh', nn.Tanh())
        
        self.actor_target = copy.deepcopy(self.actor)
        
        # Twin critics for Q-values
        self.critic1 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.critic2 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Cost critics for safety
        self.cost_critic1 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.cost_critic2 = QNetwork(self.state_dim, self.action_dim, self.config.hidden_sizes).to(self.device)
        self.cost_critic1_target = copy.deepcopy(self.cost_critic1)
        self.cost_critic2_target = copy.deepcopy(self.cost_critic2)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.learning_rate)
        self.cost_critic1_optimizer = torch.optim.Adam(self.cost_critic1.parameters(), lr=self.config.learning_rate)
        self.cost_critic2_optimizer = torch.optim.Adam(self.cost_critic2.parameters(), lr=self.config.learning_rate)
        
        # Lagrange multiplier for safety
        self.lagrange_multiplier = LagrangeMultiplier(
            initial_value=1.0, lr=self.config.lagrange_lr, 
            constraint_limit=self.config.cost_limit
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.state_dim, self.action_dim, self.device
        )
        
        self.update_count = 0
        
        logger.info("TD3-Constrained initialized")
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state) * self.max_action
            
            if not deterministic:
                noise = torch.randn_like(action) * self.exploration_noise * self.max_action
                action = (action + noise).clamp(-self.max_action, self.max_action)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'TD3Constrained':
        """Train TD3-Constrained algorithm."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        for timestep in range(total_timesteps):
            # Collect experience
            if timestep < self.config.learning_starts:
                action = env.action_space.sample()
            else:
                action, _ = self.predict(observation, deterministic=False)
            
            next_observation, reward, done, info = env.step(action)
            cost = info.get('cost', 0.0)
            
            # Store transition
            self.replay_buffer.add(observation, action, reward, next_observation, done, cost)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Update networks
            if len(self.replay_buffer) >= self.config.batch_size and timestep >= self.config.learning_starts:
                if timestep % self.config.train_freq == 0:
                    batch = self.replay_buffer.sample(self.config.batch_size)
                    update_metrics = self.update(batch)
            
            if done:
                # Record episode metrics
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit
                )
                
                if timestep >= self.config.learning_starts:
                    metrics.lagrange_multiplier = self.lagrange_multiplier.get_value()
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update TD3-Constrained networks."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        costs = batch['costs']
        
        self.update_count += 1
        
        # Update critics
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.noise_clip).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) * self.max_action + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Target Q-values
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Target cost values
            target_cost_q1 = self.cost_critic1_target(next_states, next_actions)
            target_cost_q2 = self.cost_critic2_target(next_states, next_actions)
            target_cost_q = torch.min(target_cost_q1, target_cost_q2)
            
            q_targets = rewards + self.config.gamma * (1 - dones.float()) * target_q
            cost_targets = costs + self.config.gamma * (1 - dones.float()) * target_cost_q
        
        # Update Q-critics
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, q_targets)
        critic2_loss = F.mse_loss(current_q2, q_targets)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update cost critics
        current_cost_q1 = self.cost_critic1(states, actions)
        current_cost_q2 = self.cost_critic2(states, actions)
        
        cost_critic1_loss = F.mse_loss(current_cost_q1, cost_targets)
        cost_critic2_loss = F.mse_loss(current_cost_q2, cost_targets)
        
        self.cost_critic1_optimizer.zero_grad()
        cost_critic1_loss.backward()
        self.cost_critic1_optimizer.step()
        
        self.cost_critic2_optimizer.zero_grad()
        cost_critic2_loss.backward()
        self.cost_critic2_optimizer.step()
        
        actor_loss = torch.tensor(0.0)
        avg_cost = 0.0
        
        # Delayed policy updates
        if self.update_count % self.policy_delay == 0:
            # Policy loss with Lagrangian constraint
            policy_actions = self.actor(states) * self.max_action
            q_values = self.critic1(states, policy_actions)
            cost_values = torch.min(
                self.cost_critic1(states, policy_actions),
                self.cost_critic2(states, policy_actions)
            )
            
            # Actor loss with safety constraint
            lagrange_lambda = self.lagrange_multiplier.value
            actor_loss = -q_values.mean() + lagrange_lambda * cost_values.mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update Lagrange multiplier
            avg_cost = cost_values.mean().item()
            self.lagrange_multiplier.update(avg_cost)
            
            # Soft update target networks
            soft_update(self.actor_target, self.actor, self.config.tau)
            soft_update(self.critic1_target, self.critic1, self.config.tau)
            soft_update(self.critic2_target, self.critic2, self.config.tau)
            soft_update(self.cost_critic1_target, self.cost_critic1, self.config.tau)
            soft_update(self.cost_critic2_target, self.cost_critic2, self.config.tau)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'cost_critic1_loss': cost_critic1_loss.item(),
            'cost_critic2_loss': cost_critic2_loss.item(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value(),
            'average_cost': avg_cost
        }
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get TD3-Constrained specific state for saving."""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'cost_critic1_state_dict': self.cost_critic1.state_dict(),
            'cost_critic2_state_dict': self.cost_critic2.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value(),
            'update_count': self.update_count
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load TD3-Constrained specific state."""
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic1.load_state_dict(state['critic1_state_dict'])
        self.critic2.load_state_dict(state['critic2_state_dict'])
        self.cost_critic1.load_state_dict(state['cost_critic1_state_dict'])
        self.cost_critic2.load_state_dict(state['cost_critic2_state_dict'])
        
        with torch.no_grad():
            self.lagrange_multiplier.value.fill_(state['lagrange_multiplier'])
        
        self.update_count = state.get('update_count', 0)


class TRPOConstrained(BaselineAlgorithm):
    """Trust Region Policy Optimization with safety constraints."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # TRPO specific parameters
        self.kl_threshold = 0.01
        self.damping_coeff = 0.1
        self.cg_iters = 10
        self.backtrack_iters = 10
        self.backtrack_coeff = 0.8
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize TRPO-Constrained specific components."""
        # Policy network (outputs log probabilities)
        self.policy = PolicyNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Value function
        self.value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Cost value function
        self.cost_value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Optimizers (only for value functions, policy updated via TRPO)
        self.value_optimizer = torch.optim.Adam(
            self.value_function.parameters(), lr=self.config.learning_rate
        )
        self.cost_value_optimizer = torch.optim.Adam(
            self.cost_value_function.parameters(), lr=self.config.learning_rate
        )
        
        # Lagrange multiplier for safety
        self.lagrange_multiplier = LagrangeMultiplier(
            initial_value=1.0, lr=self.config.lagrange_lr, 
            constraint_limit=self.config.cost_limit
        )
        
        # Experience storage
        self.experience_buffer = []
        
        logger.info("TRPO-Constrained initialized")
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = mean
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'TRPOConstrained':
        """Train TRPO-Constrained algorithm."""
        observation = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'log_probs': [],
            'values': [],
            'cost_values': []
        }
        
        for timestep in range(total_timesteps):
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _ = self.policy.sample(state_tensor)
                value = self.value_function(state_tensor)
                cost_value = self.cost_value_function(state_tensor)
            
            next_observation, reward, done, info = env.step(action.cpu().numpy()[0])
            cost = info.get('cost', 0.0)
            
            # Store experience
            episode_data['states'].append(observation)
            episode_data['actions'].append(action.cpu().numpy()[0])
            episode_data['rewards'].append(reward)
            episode_data['costs'].append(cost)
            episode_data['log_probs'].append(log_prob.cpu().numpy()[0])
            episode_data['values'].append(value.cpu().numpy()[0])
            episode_data['cost_values'].append(cost_value.cpu().numpy()[0])
            
            if done or len(episode_data['states']) >= 2048:  # Update every 2048 steps
                # Process episode for TRPO update
                self._process_episode(episode_data)
                
                # Reset episode data
                episode_data = {key: [] for key in episode_data.keys()}
                
                if done:
                    observation = env.reset()
                else:
                    observation = next_observation
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def _process_episode(self, episode_data: Dict[str, List]):
        """Process episode data and perform TRPO update."""
        if len(episode_data['states']) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(episode_data['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(episode_data['actions'])).to(self.device)
        rewards = np.array(episode_data['rewards'])
        costs = np.array(episode_data['costs'])
        old_log_probs = torch.FloatTensor(np.array(episode_data['log_probs'])).to(self.device)
        values = np.array(episode_data['values'])
        cost_values = np.array(episode_data['cost_values'])
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, gamma=self.config.gamma, lam=0.95)
        cost_advantages = self._compute_gae(costs, cost_values, gamma=self.config.gamma, lam=0.95)
        
        # Compute returns
        returns = advantages + values
        cost_returns = cost_advantages + cost_values
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        cost_advantages = torch.FloatTensor(cost_advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        cost_returns = torch.FloatTensor(cost_returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # Update value functions
        self._update_value_functions(states, returns, cost_returns)
        
        # TRPO policy update with constraints
        self._trpo_update(states, actions, advantages, cost_advantages, old_log_probs)
        
        # Update Lagrange multiplier
        avg_cost = torch.mean(cost_advantages).item()
        self.lagrange_multiplier.update(avg_cost)
        
        # Record metrics
        episode_return = float(np.sum(rewards))
        episode_cost = float(np.sum(costs))
        
        metrics = TrainingMetrics(
            episode_return=episode_return,
            episode_cost=episode_cost,
            episode_length=len(rewards),
            constraint_violation=episode_cost > self.config.cost_limit,
            lagrange_multiplier=self.lagrange_multiplier.get_value()
        )
        
        self.record_training_metrics(metrics)
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                     gamma: float, lam: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def _update_value_functions(self, states: torch.Tensor, returns: torch.Tensor,
                              cost_returns: torch.Tensor):
        """Update value and cost value functions."""
        # Update value function
        for _ in range(5):  # Multiple updates
            values = self.value_function(states)
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Update cost value function
        for _ in range(5):
            cost_values = self.cost_value_function(states)
            cost_value_loss = F.mse_loss(cost_values, cost_returns)
            
            self.cost_value_optimizer.zero_grad()
            cost_value_loss.backward()
            self.cost_value_optimizer.step()
    
    def _trpo_update(self, states: torch.Tensor, actions: torch.Tensor,
                     advantages: torch.Tensor, cost_advantages: torch.Tensor,
                     old_log_probs: torch.Tensor):
        """Perform TRPO policy update with safety constraints."""
        # Compute policy gradient
        mean, log_std = self.policy(states)
        std = log_std.exp()
        
        # Current log probabilities
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Policy gradient with safety constraint
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Objective with Lagrangian constraint
        lagrange_lambda = self.lagrange_multiplier.value
        policy_gradient = ratio * (advantages - lagrange_lambda * cost_advantages)
        
        # Compute natural gradient using conjugate gradient
        policy_params = list(self.policy.parameters())
        gradient = torch.autograd.grad(policy_gradient.mean(), policy_params, create_graph=True)
        gradient = torch.cat([g.view(-1) for g in gradient])
        
        # Fisher Information Matrix - vector product
        def fvp(v):
            kl = self._compute_kl_divergence(states, old_log_probs)
            kl_gradient = torch.autograd.grad(kl, policy_params, create_graph=True)
            kl_gradient = torch.cat([g.view(-1) for g in kl_gradient])
            
            gradient_product = torch.sum(kl_gradient * v)
            hessian_vector = torch.autograd.grad(gradient_product, policy_params)
            return torch.cat([g.contiguous().view(-1) for g in hessian_vector]) + self.damping_coeff * v
        
        # Conjugate gradient to solve for natural gradient
        step_direction = self._conjugate_gradient(fvp, gradient)
        
        # Compute step size
        quadratic_form = torch.dot(step_direction, fvp(step_direction))
        if quadratic_form > 0:
            max_step_size = torch.sqrt(2 * self.kl_threshold / quadratic_form)
            full_step = max_step_size * step_direction
        else:
            full_step = step_direction
        
        # Line search with backtracking
        self._line_search(policy_params, full_step, states, actions, advantages, 
                         cost_advantages, old_log_probs)
    
    def _compute_kl_divergence(self, states: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between old and new policies."""
        mean, log_std = self.policy(states)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mean, std)
        
        # Sample actions from current policy and compute KL
        actions_sample, _ = dist.rsample(), dist.log_prob
        new_log_probs = dist.log_prob(actions_sample).sum(dim=-1)
        
        kl_div = torch.mean(old_log_probs - new_log_probs)
        return kl_div
    
    def _conjugate_gradient(self, fvp_func, gradient: torch.Tensor, 
                          tolerance: float = 1e-10) -> torch.Tensor:
        """Conjugate gradient solver for natural policy gradient."""
        x = torch.zeros_like(gradient)
        residual = gradient.clone()
        direction = gradient.clone()
        residual_dot = torch.dot(residual, residual)
        
        for i in range(self.cg_iters):
            fvp_direction = fvp_func(direction)
            alpha = residual_dot / (torch.dot(direction, fvp_direction) + 1e-8)
            x += alpha * direction
            residual -= alpha * fvp_direction
            new_residual_dot = torch.dot(residual, residual)
            
            if new_residual_dot < tolerance:
                break
            
            beta = new_residual_dot / residual_dot
            direction = residual + beta * direction
            residual_dot = new_residual_dot
        
        return x
    
    def _line_search(self, policy_params: List[torch.Tensor], full_step: torch.Tensor,
                     states: torch.Tensor, actions: torch.Tensor,
                     advantages: torch.Tensor, cost_advantages: torch.Tensor,
                     old_log_probs: torch.Tensor):
        """Backtracking line search."""
        old_params = torch.cat([p.data.view(-1) for p in policy_params])
        
        for i in range(self.backtrack_iters):
            step_size = self.backtrack_coeff ** i
            new_params = old_params + step_size * full_step
            
            # Set new parameters
            param_idx = 0
            for param in policy_params:
                param_size = param.numel()
                param.data = new_params[param_idx:param_idx + param_size].view_as(param)
                param_idx += param_size
            
            # Check constraints
            kl_div = self._compute_kl_divergence(states, old_log_probs)
            
            if kl_div <= self.kl_threshold:
                break
        
        # If no acceptable step found, revert to old parameters
        if i == self.backtrack_iters - 1:
            param_idx = 0
            for param in policy_params:
                param_size = param.numel()
                param.data = old_params[param_idx:param_idx + param_size].view_as(param)
                param_idx += param_size
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Not used in TRPO - updates happen in _process_episode."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get TRPO-Constrained specific state for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_function.state_dict(),
            'cost_value_state_dict': self.cost_value_function.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value()
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load TRPO-Constrained specific state."""
        self.policy.load_state_dict(state['policy_state_dict'])
        self.value_function.load_state_dict(state['value_state_dict'])
        self.cost_value_function.load_state_dict(state['cost_value_state_dict'])
        
        with torch.no_grad():
            self.lagrange_multiplier.value.fill_(state['lagrange_multiplier'])


# Placeholder classes for other algorithms - these would be implemented similarly

class PPOLagrangian(BaselineAlgorithm):
    """PPO with Lagrangian constraints for safety-constrained policy optimization."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # PPO specific parameters
        self.clip_ratio = 0.2
        self.policy_epochs = 10
        self.value_epochs = 10
        self.batch_size = 64
        self.gae_lambda = 0.95
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize PPO-Lagrangian specific components."""
        # Policy network
        self.policy = PolicyNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Value function
        self.value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Cost value function
        self.cost_value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_function.parameters(), lr=self.config.learning_rate
        )
        self.cost_value_optimizer = torch.optim.Adam(
            self.cost_value_function.parameters(), lr=self.config.learning_rate
        )
        
        # Lagrange multiplier for safety constraint
        self.lagrange_multiplier = LagrangeMultiplier(
            initial_value=1.0, lr=self.config.lagrange_lr,
            constraint_limit=self.config.cost_limit
        )
        
        # Experience storage
        self.experience_buffer = []
        
        logger.info("PPO-Lagrangian initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = mean
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'PPOLagrangian':
        """Train PPO-Lagrangian algorithm."""
        observation = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'log_probs': [],
            'values': [],
            'cost_values': [],
            'dones': []
        }
        
        for timestep in range(total_timesteps):
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _ = self.policy.sample(state_tensor)
                value = self.value_function(state_tensor)
                cost_value = self.cost_value_function(state_tensor)
            
            next_observation, reward, done, info = env.step(action.cpu().numpy()[0])
            cost = info.get('cost', 0.0)
            
            # Store experience
            episode_data['states'].append(observation)
            episode_data['actions'].append(action.cpu().numpy()[0])
            episode_data['rewards'].append(reward)
            episode_data['costs'].append(cost)
            episode_data['log_probs'].append(log_prob.cpu().numpy()[0])
            episode_data['values'].append(value.cpu().numpy()[0])
            episode_data['cost_values'].append(cost_value.cpu().numpy()[0])
            episode_data['dones'].append(done)
            
            if done or len(episode_data['states']) >= 2048:  # Update every 2048 steps
                # Process episode for PPO update
                self._ppo_update(episode_data)
                
                # Reset episode data
                episode_data = {key: [] for key in episode_data.keys()}
                
                if done:
                    observation = env.reset()
                else:
                    observation = next_observation
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def _ppo_update(self, episode_data: Dict[str, List]):
        """Perform PPO update with Lagrangian constraints."""
        if len(episode_data['states']) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(episode_data['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(episode_data['actions'])).to(self.device)
        rewards = np.array(episode_data['rewards'])
        costs = np.array(episode_data['costs'])
        old_log_probs = torch.FloatTensor(np.array(episode_data['log_probs'])).to(self.device)
        values = np.array(episode_data['values'])
        cost_values = np.array(episode_data['cost_values'])
        dones = np.array(episode_data['dones'])
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones, self.config.gamma, self.gae_lambda)
        cost_advantages = self._compute_gae(costs, cost_values, dones, self.config.gamma, self.gae_lambda)
        
        # Compute returns
        returns = advantages + values
        cost_returns = cost_advantages + cost_values
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        cost_advantages = torch.FloatTensor(cost_advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        cost_returns = torch.FloatTensor(cost_returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # PPO epochs
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(self.policy_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_cost_advantages = cost_advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Current policy evaluation
                mean, log_std = self.policy(batch_states)
                std = log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                # Ratio and clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO objective with Lagrangian constraint
                lagrange_lambda = self.lagrange_multiplier.value
                combined_advantages = batch_advantages - lagrange_lambda * batch_cost_advantages
                
                surr1 = ratio * combined_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * combined_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus for exploration
                entropy = dist.entropy().sum(dim=-1).mean()
                policy_loss -= 0.01 * entropy
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
        
        # Update value functions
        for epoch in range(self.value_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                batch_cost_returns = cost_returns[batch_indices]
                
                # Value function loss
                values = self.value_function(batch_states)
                value_loss = F.mse_loss(values, batch_returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
                
                # Cost value function loss
                cost_values = self.cost_value_function(batch_states)
                cost_value_loss = F.mse_loss(cost_values, batch_cost_returns)
                
                self.cost_value_optimizer.zero_grad()
                cost_value_loss.backward()
                self.cost_value_optimizer.step()
        
        # Update Lagrange multiplier
        avg_cost = torch.mean(cost_advantages).item()
        self.lagrange_multiplier.update(avg_cost)
        
        # Record metrics
        episode_return = float(np.sum(rewards))
        episode_cost = float(np.sum(costs))
        
        metrics = TrainingMetrics(
            episode_return=episode_return,
            episode_cost=episode_cost,
            episode_length=len(rewards),
            constraint_violation=episode_cost > self.config.cost_limit,
            lagrange_multiplier=self.lagrange_multiplier.get_value()
        )
        
        self.record_training_metrics(metrics)
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                     gamma: float, lam: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = 0 if dones[t] else values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * lam * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        
        return advantages
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Not used in PPO - updates happen in _ppo_update."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get PPO-Lagrangian specific state for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_function.state_dict(),
            'cost_value_state_dict': self.cost_value_function.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value()
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load PPO-Lagrangian specific state."""
        self.policy.load_state_dict(state['policy_state_dict'])
        self.value_function.load_state_dict(state['value_state_dict'])
        self.cost_value_function.load_state_dict(state['cost_value_state_dict'])
        
        with torch.no_grad():
            self.lagrange_multiplier.value.fill_(state['lagrange_multiplier'])


class SafeDDPG(BaselineAlgorithm):
    """Safe Deep Deterministic Policy Gradient with Lagrangian constraints."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int, max_action: float = 1.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Safe DDPG specific parameters
        self.noise_std = 0.1
        self.noise_clip = 0.5
        self.exploration_noise = 0.2
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize Safe DDPG specific components."""
        # Actor network
        self.actor = PolicyNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes,
            deterministic=True
        ).to(self.device)
        self.actor_target = PolicyNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes,
            deterministic=True
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks (Q-function)
        self.critic = QNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        self.critic_target = QNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Cost critic networks
        self.cost_critic = QNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        self.cost_critic_target = QNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        self.cost_critic_target.load_state_dict(self.cost_critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.learning_rate
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            self.cost_critic.parameters(), lr=self.config.learning_rate
        )
        
        # Lagrange multiplier for safety constraints
        self.lagrange_multiplier = LagrangeMultiplier(
            initial_value=1.0, lr=self.config.lagrange_lr,
            constraint_limit=self.config.cost_limit
        )
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(
            self.config.buffer_size, self.state_dim, self.action_dim
        )
        
        logger.info("Safe-DDPG initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state) * self.max_action
            
            if not deterministic:
                # Add exploration noise
                noise = torch.normal(0, self.exploration_noise, size=action.shape).to(self.device)
                action = (action + noise).clamp(-self.max_action, self.max_action)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'SafeDDPG':
        """Train Safe DDPG algorithm."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        for timestep in range(total_timesteps):
            # Select action with exploration noise
            if timestep < self.config.learning_starts:
                action = env.action_space.sample()
            else:
                action, _ = self.predict(observation, deterministic=False)
            
            # Environment step
            next_observation, reward, done, info = env.step(action)
            cost = info.get('cost', 0.0)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Store transition in replay buffer
            self.replay_buffer.add(observation, action, reward, next_observation, done, cost)
            
            # Update policy
            if timestep >= self.config.learning_starts and len(self.replay_buffer) > self.config.batch_size:
                batch = self.replay_buffer.sample(self.config.batch_size)
                update_metrics = self.update(batch)
            
            if done:
                # Record episode metrics
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit
                )
                
                if timestep >= self.config.learning_starts:
                    metrics.lagrange_multiplier = self.lagrange_multiplier.get_value()
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Safe DDPG networks."""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        costs = batch['costs']
        
        # Target actions with noise
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.noise_std).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) * self.max_action + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Target Q-values
            target_q = self.critic_target(next_states, next_actions)
            target_cost_q = self.cost_critic_target(next_states, next_actions)
            
            q_targets = rewards + self.config.gamma * (1 - dones.float()) * target_q
            cost_targets = costs + self.config.gamma * (1 - dones.float()) * target_cost_q
        
        # Update Q-critic
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update cost critic
        current_cost_q = self.cost_critic(states, actions)
        cost_critic_loss = F.mse_loss(current_cost_q, cost_targets)
        
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()
        
        # Update actor with Lagrangian constraint
        policy_actions = self.actor(states) * self.max_action
        q_values = self.critic(states, policy_actions)
        cost_values = self.cost_critic(states, policy_actions)
        
        # Actor loss with safety constraint
        lagrange_lambda = self.lagrange_multiplier.value
        actor_loss = -q_values.mean() + lagrange_lambda * cost_values.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update Lagrange multiplier
        avg_cost = cost_values.mean().item()
        self.lagrange_multiplier.update(avg_cost)
        
        # Soft update target networks
        soft_update(self.actor_target, self.actor, self.config.tau)
        soft_update(self.critic_target, self.critic, self.config.tau)
        soft_update(self.cost_critic_target, self.cost_critic, self.config.tau)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'cost_critic_loss': cost_critic_loss.item(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value(),
            'average_cost': avg_cost
        }
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get Safe DDPG specific state for saving."""
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'cost_critic_state_dict': self.cost_critic.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier.get_value()
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load Safe DDPG specific state."""
        self.actor.load_state_dict(state['actor_state_dict'])
        self.critic.load_state_dict(state['critic_state_dict'])
        self.cost_critic.load_state_dict(state['cost_critic_state_dict'])
        
        with torch.no_grad():
            self.lagrange_multiplier.value.fill_(state['lagrange_multiplier'])


class RCPO(BaselineAlgorithm):
    """Reward Constrained Policy Optimization with dual formulation."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # RCPO specific parameters
        self.reward_constraint = getattr(config, 'reward_constraint', 0.1)  # Min expected reward
        self.dual_lr = 0.01
        self.cg_iters = 10
        self.damping_coeff = 0.1
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize RCPO specific components."""
        # Policy network
        self.policy = PolicyNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Value function for rewards
        self.reward_value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Value function for costs
        self.cost_value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Optimizers
        self.reward_value_optimizer = torch.optim.Adam(
            self.reward_value_function.parameters(), lr=self.config.learning_rate
        )
        self.cost_value_optimizer = torch.optim.Adam(
            self.cost_value_function.parameters(), lr=self.config.learning_rate
        )
        
        # Dual variables (Lagrange multipliers)
        self.reward_dual = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.cost_dual = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.dual_optimizer = torch.optim.Adam([self.reward_dual, self.cost_dual], lr=self.dual_lr)
        
        # Experience storage
        self.experience_buffer = []
        
        logger.info("RCPO initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = mean
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'RCPO':
        """Train RCPO algorithm."""
        observation = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'log_probs': [],
            'reward_values': [],
            'cost_values': []
        }
        
        for timestep in range(total_timesteps):
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _ = self.policy.sample(state_tensor)
                reward_value = self.reward_value_function(state_tensor)
                cost_value = self.cost_value_function(state_tensor)
            
            next_observation, reward, done, info = env.step(action.cpu().numpy()[0])
            cost = info.get('cost', 0.0)
            
            # Store experience
            episode_data['states'].append(observation)
            episode_data['actions'].append(action.cpu().numpy()[0])
            episode_data['rewards'].append(reward)
            episode_data['costs'].append(cost)
            episode_data['log_probs'].append(log_prob.cpu().numpy()[0])
            episode_data['reward_values'].append(reward_value.cpu().numpy()[0])
            episode_data['cost_values'].append(cost_value.cpu().numpy()[0])
            
            if done or len(episode_data['states']) >= 2048:  # Update every 2048 steps
                # Process episode for RCPO update
                self._rcpo_update(episode_data)
                
                # Reset episode data
                episode_data = {key: [] for key in episode_data.keys()}
                
                if done:
                    observation = env.reset()
                else:
                    observation = next_observation
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def _rcpo_update(self, episode_data: Dict[str, List]):
        """Perform RCPO update with dual formulation."""
        if len(episode_data['states']) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(episode_data['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(episode_data['actions'])).to(self.device)
        rewards = np.array(episode_data['rewards'])
        costs = np.array(episode_data['costs'])
        old_log_probs = torch.FloatTensor(np.array(episode_data['log_probs'])).to(self.device)
        reward_values = np.array(episode_data['reward_values'])
        cost_values = np.array(episode_data['cost_values'])
        
        # Compute advantages using GAE
        reward_advantages = self._compute_gae(rewards, reward_values, gamma=self.config.gamma, lam=0.95)
        cost_advantages = self._compute_gae(costs, cost_values, gamma=self.config.gamma, lam=0.95)
        
        # Compute returns
        reward_returns = reward_advantages + reward_values
        cost_returns = cost_advantages + cost_values
        
        # Convert to tensors
        reward_advantages = torch.FloatTensor(reward_advantages).to(self.device)
        cost_advantages = torch.FloatTensor(cost_advantages).to(self.device)
        reward_returns = torch.FloatTensor(reward_returns).to(self.device)
        cost_returns = torch.FloatTensor(cost_returns).to(self.device)
        
        # Normalize advantages
        reward_advantages = (reward_advantages - reward_advantages.mean()) / (reward_advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # Update value functions
        self._update_value_functions(states, reward_returns, cost_returns)
        
        # RCPO policy update with dual variables
        self._dual_policy_update(states, actions, reward_advantages, cost_advantages, old_log_probs)
        
        # Update dual variables
        self._update_dual_variables(reward_advantages, cost_advantages)
        
        # Record metrics
        episode_return = float(np.sum(rewards))
        episode_cost = float(np.sum(costs))
        
        metrics = TrainingMetrics(
            episode_return=episode_return,
            episode_cost=episode_cost,
            episode_length=len(rewards),
            constraint_violation=episode_cost > self.config.cost_limit,
            reward_dual=self.reward_dual.item(),
            cost_dual=self.cost_dual.item()
        )
        
        self.record_training_metrics(metrics)
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                     gamma: float, lam: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def _update_value_functions(self, states: torch.Tensor, reward_returns: torch.Tensor,
                              cost_returns: torch.Tensor):
        """Update reward and cost value functions."""
        # Update reward value function
        for _ in range(5):
            reward_values = self.reward_value_function(states)
            reward_value_loss = F.mse_loss(reward_values, reward_returns)
            
            self.reward_value_optimizer.zero_grad()
            reward_value_loss.backward()
            self.reward_value_optimizer.step()
        
        # Update cost value function
        for _ in range(5):
            cost_values = self.cost_value_function(states)
            cost_value_loss = F.mse_loss(cost_values, cost_returns)
            
            self.cost_value_optimizer.zero_grad()
            cost_value_loss.backward()
            self.cost_value_optimizer.step()
    
    def _dual_policy_update(self, states: torch.Tensor, actions: torch.Tensor,
                           reward_advantages: torch.Tensor, cost_advantages: torch.Tensor,
                           old_log_probs: torch.Tensor):
        """Update policy using dual formulation."""
        # Current policy evaluation
        mean, log_std = self.policy(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Dual objective (minimize negative reward + constraint penalties)
        dual_objective = -(self.reward_dual * ratio * reward_advantages).mean() + \
                        (self.cost_dual * ratio * cost_advantages).mean()
        
        # Compute natural gradient using conjugate gradient
        policy_params = list(self.policy.parameters())
        gradient = torch.autograd.grad(dual_objective, policy_params, create_graph=True)
        gradient = torch.cat([g.view(-1) for g in gradient])
        
        # Fisher Information Matrix - vector product
        def fvp(v):
            kl = self._compute_kl_divergence(states, old_log_probs)
            kl_gradient = torch.autograd.grad(kl, policy_params, create_graph=True)
            kl_gradient = torch.cat([g.view(-1) for g in kl_gradient])
            
            gradient_product = torch.sum(kl_gradient * v)
            hessian_vector = torch.autograd.grad(gradient_product, policy_params)
            return torch.cat([g.contiguous().view(-1) for g in hessian_vector]) + self.damping_coeff * v
        
        # Conjugate gradient to solve for natural gradient
        step_direction = self._conjugate_gradient(fvp, gradient)
        
        # Apply natural gradient step
        param_idx = 0
        for param in policy_params:
            param_size = param.numel()
            param.data -= step_direction[param_idx:param_idx + param_size].view_as(param) * 0.01
            param_idx += param_size
    
    def _update_dual_variables(self, reward_advantages: torch.Tensor, cost_advantages: torch.Tensor):
        """Update dual variables to satisfy constraints."""
        # Constraint violations
        reward_violation = self.reward_constraint - reward_advantages.mean()
        cost_violation = cost_advantages.mean() - self.config.cost_limit
        
        # Dual variable updates
        dual_loss = -self.reward_dual * reward_violation + self.cost_dual * cost_violation
        
        self.dual_optimizer.zero_grad()
        dual_loss.backward()
        self.dual_optimizer.step()
        
        # Project dual variables to non-negative values
        with torch.no_grad():
            self.reward_dual.clamp_(min=0.0)
            self.cost_dual.clamp_(min=0.0)
    
    def _compute_kl_divergence(self, states: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between old and new policies."""
        mean, log_std = self.policy(states)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mean, std)
        
        # Sample actions from current policy and compute KL
        actions_sample, _ = dist.rsample(), dist.log_prob
        new_log_probs = dist.log_prob(actions_sample).sum(dim=-1)
        
        kl_div = torch.mean(old_log_probs - new_log_probs)
        return kl_div
    
    def _conjugate_gradient(self, fvp_func, gradient: torch.Tensor, 
                          tolerance: float = 1e-10) -> torch.Tensor:
        """Conjugate gradient solver."""
        x = torch.zeros_like(gradient)
        residual = gradient.clone()
        direction = gradient.clone()
        residual_dot = torch.dot(residual, residual)
        
        for i in range(self.cg_iters):
            fvp_direction = fvp_func(direction)
            alpha = residual_dot / (torch.dot(direction, fvp_direction) + 1e-8)
            x += alpha * direction
            residual -= alpha * fvp_direction
            new_residual_dot = torch.dot(residual, residual)
            
            if new_residual_dot < tolerance:
                break
            
            beta = new_residual_dot / residual_dot
            direction = residual + beta * direction
            residual_dot = new_residual_dot
        
        return x
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Not used in RCPO - updates happen in _rcpo_update."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get RCPO specific state for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'reward_value_state_dict': self.reward_value_function.state_dict(),
            'cost_value_state_dict': self.cost_value_function.state_dict(),
            'reward_dual': self.reward_dual.item(),
            'cost_dual': self.cost_dual.item()
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load RCPO specific state."""
        self.policy.load_state_dict(state['policy_state_dict'])
        self.reward_value_function.load_state_dict(state['reward_value_state_dict'])
        self.cost_value_function.load_state_dict(state['cost_value_state_dict'])
        
        with torch.no_grad():
            self.reward_dual.fill_(state['reward_dual'])
            self.cost_dual.fill_(state['cost_dual'])


class CPOVariants(BaselineAlgorithm):
    """CPO with different trust region solvers and variants."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int,
                 solver_type: str = 'conjugate_gradient'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.solver_type = solver_type  # 'conjugate_gradient', 'exact', 'approximate'
        
        # CPO specific parameters based on solver type
        if solver_type == 'conjugate_gradient':
            self.cg_iters = 10
            self.damping_coeff = 0.1
        elif solver_type == 'exact':
            self.use_exact_hessian = True
        elif solver_type == 'approximate':
            self.use_fisher_approximation = True
            
        self.kl_threshold = 0.01
        self.cost_threshold = getattr(config, 'cost_limit', 25.0)
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize CPO variant specific components."""
        # Policy network
        self.policy = PolicyNetwork(
            self.state_dim, self.action_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Value function for rewards
        self.value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Cost value function
        self.cost_value_function = ValueNetwork(
            self.state_dim, self.config.hidden_sizes
        ).to(self.device)
        
        # Optimizers (only for value functions)
        self.value_optimizer = torch.optim.Adam(
            self.value_function.parameters(), lr=self.config.learning_rate
        )
        self.cost_value_optimizer = torch.optim.Adam(
            self.cost_value_function.parameters(), lr=self.config.learning_rate
        )
        
        # Experience storage
        self.experience_buffer = []
        
        logger.info(f"CPO-{self.solver_type} initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Predict action given observation."""
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = mean
            else:
                action, _, _ = self.policy.sample(state)
        
        return action.cpu().numpy()[0], None
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'CPOVariants':
        """Train CPO variant algorithm."""
        observation = env.reset()
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'costs': [],
            'log_probs': [],
            'values': [],
            'cost_values': []
        }
        
        for timestep in range(total_timesteps):
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, _ = self.policy.sample(state_tensor)
                value = self.value_function(state_tensor)
                cost_value = self.cost_value_function(state_tensor)
            
            next_observation, reward, done, info = env.step(action.cpu().numpy()[0])
            cost = info.get('cost', 0.0)
            
            # Store experience
            episode_data['states'].append(observation)
            episode_data['actions'].append(action.cpu().numpy()[0])
            episode_data['rewards'].append(reward)
            episode_data['costs'].append(cost)
            episode_data['log_probs'].append(log_prob.cpu().numpy()[0])
            episode_data['values'].append(value.cpu().numpy()[0])
            episode_data['cost_values'].append(cost_value.cpu().numpy()[0])
            
            if done or len(episode_data['states']) >= 2048:  # Update every 2048 steps
                # Process episode for CPO update
                self._cpo_variant_update(episode_data)
                
                # Reset episode data
                episode_data = {key: [] for key in episode_data.keys()}
                
                if done:
                    observation = env.reset()
                else:
                    observation = next_observation
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def _cpo_variant_update(self, episode_data: Dict[str, List]):
        """Perform CPO update with different solver variants."""
        if len(episode_data['states']) == 0:
            return
        
        # Convert to tensors (similar to other implementations)
        states = torch.FloatTensor(np.array(episode_data['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(episode_data['actions'])).to(self.device)
        rewards = np.array(episode_data['rewards'])
        costs = np.array(episode_data['costs'])
        old_log_probs = torch.FloatTensor(np.array(episode_data['log_probs'])).to(self.device)
        values = np.array(episode_data['values'])
        cost_values = np.array(episode_data['cost_values'])
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, gamma=self.config.gamma, lam=0.95)
        cost_advantages = self._compute_gae(costs, cost_values, gamma=self.config.gamma, lam=0.95)
        
        # Compute returns
        returns = advantages + values
        cost_returns = cost_advantages + cost_values
        
        # Convert to tensors
        advantages = torch.FloatTensor(advantages).to(self.device)
        cost_advantages = torch.FloatTensor(cost_advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        cost_returns = torch.FloatTensor(cost_returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        # Update value functions
        self._update_value_functions(states, returns, cost_returns)
        
        # CPO policy update with variant solver
        if self.solver_type == 'conjugate_gradient':
            self._cpo_conjugate_gradient_update(states, actions, advantages, cost_advantages, old_log_probs)
        elif self.solver_type == 'exact':
            self._cpo_exact_update(states, actions, advantages, cost_advantages, old_log_probs)
        elif self.solver_type == 'approximate':
            self._cpo_approximate_update(states, actions, advantages, cost_advantages, old_log_probs)
        
        # Record metrics
        episode_return = float(np.sum(rewards))
        episode_cost = float(np.sum(costs))
        
        metrics = TrainingMetrics(
            episode_return=episode_return,
            episode_cost=episode_cost,
            episode_length=len(rewards),
            constraint_violation=episode_cost > self.config.cost_limit
        )
        
        self.record_training_metrics(metrics)
    
    def _cpo_conjugate_gradient_update(self, states: torch.Tensor, actions: torch.Tensor,
                                     advantages: torch.Tensor, cost_advantages: torch.Tensor,
                                     old_log_probs: torch.Tensor):
        """CPO update using conjugate gradient solver (original method)."""
        # Current policy evaluation
        mean, log_std = self.policy(states)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Policy gradient
        ratio = torch.exp(new_log_probs - old_log_probs)
        policy_objective = (ratio * advantages).mean()
        cost_constraint = (ratio * cost_advantages).mean()
        
        # Compute gradients
        policy_params = list(self.policy.parameters())
        objective_gradient = torch.autograd.grad(policy_objective, policy_params, create_graph=True)
        objective_gradient = torch.cat([g.view(-1) for g in objective_gradient])
        
        constraint_gradient = torch.autograd.grad(cost_constraint, policy_params, create_graph=True)
        constraint_gradient = torch.cat([g.view(-1) for g in constraint_gradient])
        
        # Fisher Information Matrix vector product
        def fvp(v):
            kl = self._compute_kl_divergence(states)
            kl_gradient = torch.autograd.grad(kl, policy_params, create_graph=True)
            kl_gradient = torch.cat([g.view(-1) for g in kl_gradient])
            
            gradient_product = torch.sum(kl_gradient * v)
            hessian_vector = torch.autograd.grad(gradient_product, policy_params)
            return torch.cat([g.contiguous().view(-1) for g in hessian_vector]) + self.damping_coeff * v
        
        # Solve linear systems using conjugate gradient
        H_inv_g = self._conjugate_gradient(fvp, objective_gradient)
        H_inv_b = self._conjugate_gradient(fvp, constraint_gradient)
        
        # CPO step computation
        approx_g = torch.dot(objective_gradient, H_inv_g)
        c = cost_advantages.mean().item() - self.cost_threshold
        
        if c > 0 and torch.dot(constraint_gradient, H_inv_b) > 0:
            # Feasible step with constraint active
            lam = torch.sqrt(2 * self.kl_threshold / torch.dot(H_inv_b, constraint_gradient))
            nu = torch.max(torch.tensor(0.0), 
                          (torch.dot(objective_gradient, H_inv_b) - torch.sqrt(2 * self.kl_threshold * approx_g)) / 
                          torch.dot(constraint_gradient, H_inv_b))
            step_direction = (1 / (lam + nu + 1e-8)) * (H_inv_g - nu * H_inv_b)
        else:
            # Unconstrained step
            lam = torch.sqrt(2 * self.kl_threshold / approx_g)
            step_direction = (1 / lam) * H_inv_g
        
        # Apply step
        self._apply_policy_update(policy_params, step_direction)
    
    def _cpo_exact_update(self, states: torch.Tensor, actions: torch.Tensor,
                         advantages: torch.Tensor, cost_advantages: torch.Tensor,
                         old_log_probs: torch.Tensor):
        """CPO update using exact Hessian computation."""
        # This is a simplified version - exact Hessian is computationally expensive
        # In practice, you'd compute the exact Hessian of the KL divergence
        self._cpo_conjugate_gradient_update(states, actions, advantages, cost_advantages, old_log_probs)
    
    def _cpo_approximate_update(self, states: torch.Tensor, actions: torch.Tensor,
                               advantages: torch.Tensor, cost_advantages: torch.Tensor,
                               old_log_probs: torch.Tensor):
        """CPO update using Fisher Information approximation."""
        # Use diagonal Fisher Information as approximation
        mean, log_std = self.policy(states)
        std = log_std.exp()
        
        # Diagonal Fisher Information (simplified)
        fisher_diag = 1.0 / (std ** 2 + 1e-8)
        
        # Policy gradients
        ratio = torch.exp(torch.distributions.Normal(mean, std).log_prob(actions).sum(dim=-1) - old_log_probs)
        policy_gradient = torch.autograd.grad((ratio * advantages).mean(), self.policy.parameters())
        constraint_gradient = torch.autograd.grad((ratio * cost_advantages).mean(), self.policy.parameters())
        
        # Apply approximate update (simplified version)
        for param, p_grad, c_grad in zip(self.policy.parameters(), policy_gradient, constraint_gradient):
            if cost_advantages.mean() > self.cost_threshold:
                # With constraint
                param.data += 0.01 * (p_grad - 0.1 * c_grad)
            else:
                # Without constraint
                param.data += 0.01 * p_grad
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, 
                     gamma: float, lam: float) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]
        
        return advantages
    
    def _update_value_functions(self, states: torch.Tensor, returns: torch.Tensor,
                              cost_returns: torch.Tensor):
        """Update value and cost value functions."""
        # Update value function
        for _ in range(5):
            values = self.value_function(states)
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Update cost value function
        for _ in range(5):
            cost_values = self.cost_value_function(states)
            cost_value_loss = F.mse_loss(cost_values, cost_returns)
            
            self.cost_value_optimizer.zero_grad()
            cost_value_loss.backward()
            self.cost_value_optimizer.step()
    
    def _compute_kl_divergence(self, states: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for current policy."""
        mean, log_std = self.policy(states)
        # Simplified KL computation
        kl_div = torch.mean(log_std.exp() ** 2)
        return kl_div
    
    def _conjugate_gradient(self, fvp_func, gradient: torch.Tensor, 
                          tolerance: float = 1e-10) -> torch.Tensor:
        """Conjugate gradient solver."""
        x = torch.zeros_like(gradient)
        residual = gradient.clone()
        direction = gradient.clone()
        residual_dot = torch.dot(residual, residual)
        
        for i in range(self.cg_iters):
            fvp_direction = fvp_func(direction)
            alpha = residual_dot / (torch.dot(direction, fvp_direction) + 1e-8)
            x += alpha * direction
            residual -= alpha * fvp_direction
            new_residual_dot = torch.dot(residual, residual)
            
            if new_residual_dot < tolerance:
                break
            
            beta = new_residual_dot / residual_dot
            direction = residual + beta * direction
            residual_dot = new_residual_dot
        
        return x
    
    def _apply_policy_update(self, policy_params: List[torch.Tensor], step_direction: torch.Tensor):
        """Apply computed step to policy parameters."""
        param_idx = 0
        for param in policy_params:
            param_size = param.numel()
            param.data += step_direction[param_idx:param_idx + param_size].view_as(param)
            param_idx += param_size
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Not used in CPO variants - updates happen in _cpo_variant_update."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get CPO variant specific state for saving."""
        return {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_function.state_dict(),
            'cost_value_state_dict': self.cost_value_function.state_dict(),
            'solver_type': self.solver_type
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load CPO variant specific state."""
        self.policy.load_state_dict(state['policy_state_dict'])
        self.value_function.load_state_dict(state['value_state_dict'])
        self.cost_value_function.load_state_dict(state['cost_value_state_dict'])
        self.solver_type = state.get('solver_type', 'conjugate_gradient')