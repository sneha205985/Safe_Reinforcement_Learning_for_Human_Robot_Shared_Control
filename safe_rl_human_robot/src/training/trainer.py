"""
Robust CPO trainer with comprehensive checkpointing and monitoring.

This module provides a production-ready training system for CPO algorithms
with automatic checkpointing, resume capability, distributed training support,
and comprehensive monitoring and logging.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import pickle
import json
from collections import deque, defaultdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainingConfig
from src.algorithms.cpo import CPOAlgorithm, CPOConfig
from src.environments import (
    ExoskeletonEnvironment, WheelchairEnvironment, 
    ExoskeletonConfig, WheelchairConfig
)
from src.core.safety_monitor import SafetyMonitor
from src.utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)


@dataclass
class TrainingResults:
    """Results from training session."""
    # Training metrics
    total_iterations: int = 0
    total_episodes: int = 0
    total_timesteps: int = 0
    training_time: float = 0.0
    
    # Performance metrics
    best_reward: float = float('-inf')
    best_success_rate: float = 0.0
    final_reward: float = 0.0
    final_success_rate: float = 0.0
    
    # Safety metrics
    total_violations: int = 0
    violation_rate: float = 0.0
    avg_constraint_cost: float = 0.0
    
    # Training dynamics
    policy_gradient_norms: List[float] = field(default_factory=list)
    kl_divergences: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    constraint_losses: List[float] = field(default_factory=list)
    
    # Evaluation history
    eval_rewards: List[float] = field(default_factory=list)
    eval_success_rates: List[float] = field(default_factory=list)
    eval_iterations: List[int] = field(default_factory=list)
    
    # Metadata
    config_hash: str = ""
    environment_name: str = ""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_iterations': self.total_iterations,
            'total_episodes': self.total_episodes,
            'total_timesteps': self.total_timesteps,
            'training_time': self.training_time,
            'best_reward': self.best_reward,
            'best_success_rate': self.best_success_rate,
            'final_reward': self.final_reward,
            'final_success_rate': self.final_success_rate,
            'total_violations': self.total_violations,
            'violation_rate': self.violation_rate,
            'avg_constraint_cost': self.avg_constraint_cost,
            'policy_gradient_norms': self.policy_gradient_norms,
            'kl_divergences': self.kl_divergences,
            'value_losses': self.value_losses,
            'constraint_losses': self.constraint_losses,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': self.eval_success_rates,
            'eval_iterations': self.eval_iterations,
            'config_hash': self.config_hash,
            'environment_name': self.environment_name,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class CPOTrainer:
    """
    Comprehensive trainer for CPO algorithms with robust error handling,
    checkpointing, and monitoring capabilities.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize CPO trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Initialize logging
        self.logger = setup_logger(
            name="CPOTrainer",
            level=getattr(logging, config.log_level.upper()),
            log_file=Path(config.log_dir) / "training.log"
        )
        
        # Initialize training state
        self.iteration = 0
        self.total_timesteps = 0
        self.total_episodes = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.best_reward = float('-inf')
        self.best_success_rate = 0.0
        self.eval_history = deque(maxlen=100)
        self.training_metrics = defaultdict(list)
        
        # Components (initialized in setup)
        self.env = None
        self.eval_env = None
        self.agent = None
        self.safety_monitor = None
        
        # Distributed training
        self.is_distributed = config.distributed.distributed
        self.local_rank = config.distributed.local_rank
        self.world_size = config.distributed.world_size
        
        # Initialize components
        self._setup_environment()
        self._setup_agent()
        self._setup_safety_monitoring()
        
        # Load checkpoint if resuming
        if config.resume_from:
            self.load_checkpoint(config.resume_from)
        
        self.logger.info(f"CPOTrainer initialized: {config.env_name}, "
                        f"device={self.device}, distributed={self.is_distributed}")
    
    def _setup_reproducibility(self):
        """Setup reproducibility settings."""
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if self.config.torch_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        if self.config.cuda_deterministic and torch.cuda.is_available():
            torch.use_deterministic_algorithms(True)
        
        # Set number of threads
        torch.set_num_threads(self.config.num_threads)
    
    def _setup_environment(self):
        """Setup training and evaluation environments."""
        env_config = self.config.env_config.copy()
        
        # Create training environment
        if self.config.env_name.lower() == "exoskeleton":
            exo_config = ExoskeletonConfig(**env_config.get('exoskeleton', {}))
            self.env = ExoskeletonEnvironment(config=exo_config, **env_config)
        elif self.config.env_name.lower() == "wheelchair":
            wheelchair_config = WheelchairConfig(**env_config.get('wheelchair', {}))
            self.env = WheelchairEnvironment(config=wheelchair_config, **env_config)
        else:
            raise ValueError(f"Unknown environment: {self.config.env_name}")
        
        # Create evaluation environment (separate instance)
        self.eval_env = type(self.env)(**env_config) if self.env else None
        
        self.logger.info(f"Environment setup: {self.config.env_name}")
        self.logger.info(f"State space: {self.env.observation_space.shape}")
        self.logger.info(f"Action space: {self.env.action_space.shape}")
    
    def _setup_agent(self):
        """Setup CPO agent with networks."""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create CPO configuration
        cpo_config = CPOConfig(
            policy_lr=self.config.optimization.policy_lr,
            value_lr=self.config.optimization.value_lr,
            constraint_value_lr=self.config.optimization.constraint_value_lr,
            gamma=self.config.optimization.gamma,
            lambda_gae=self.config.optimization.lambda_gae,
            trust_region_radius=self.config.optimization.trust_region_radius,
            constraint_threshold=self.config.optimization.constraint_threshold,
            max_kl_divergence=self.config.optimization.max_kl_divergence,
            cg_iterations=self.config.optimization.cg_iterations,
            line_search_steps=self.config.optimization.line_search_steps,
            value_train_iterations=self.config.optimization.value_train_iterations,
            constraint_value_train_iterations=self.config.optimization.constraint_value_train_iterations
        )
        
        # Create networks
        policy_net = self._create_policy_network(state_dim, action_dim)
        value_net = self._create_value_network(state_dim)
        constraint_value_net = self._create_value_network(state_dim)
        
        # Create CPO agent
        self.agent = CPOAlgorithm(
            policy=policy_net,
            value_network=value_net,
            constraint_value_network=constraint_value_net,
            config=cpo_config
        )
        
        # Move to device and setup distributed training
        self.agent.to(self.device)
        
        if self.is_distributed:
            self._setup_distributed_training()
        
        self.logger.info(f"Agent created: {sum(p.numel() for p in self.agent.parameters())} parameters")
    
    def _create_policy_network(self, state_dim: int, action_dim: int) -> torch.nn.Module:
        """Create policy network based on configuration."""
        net_config = self.config.network
        
        layers = []
        prev_dim = state_dim
        
        # Hidden layers
        for hidden_size in net_config.policy_hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            if net_config.policy_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_size))
            layers.append(self._get_activation(net_config.policy_activation))
            if net_config.policy_dropout > 0:
                layers.append(torch.nn.Dropout(net_config.policy_dropout))
            prev_dim = hidden_size
        
        # Output layer (mean and log_std for Gaussian policy)
        layers.append(torch.nn.Linear(prev_dim, action_dim * 2))
        
        network = torch.nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_network(network, net_config.initialization_scheme)
        
        return PolicyWrapper(network, action_dim)
    
    def _create_value_network(self, state_dim: int) -> torch.nn.Module:
        """Create value network based on configuration."""
        net_config = self.config.network
        
        layers = []
        prev_dim = state_dim
        
        # Use value network configuration
        hidden_sizes = net_config.value_hidden_sizes
        activation = net_config.value_activation
        layer_norm = net_config.value_layer_norm
        dropout = net_config.value_dropout
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_size))
            if layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_size))
            layers.append(self._get_activation(activation))
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev_dim = hidden_size
        
        # Output layer
        layers.append(torch.nn.Linear(prev_dim, 1))
        
        network = torch.nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_network(network, net_config.initialization_scheme)
        
        return network
    
    def _get_activation(self, activation):
        """Get activation function from enum."""
        activation_map = {
            'relu': torch.nn.ReLU(),
            'tanh': torch.nn.Tanh(),
            'elu': torch.nn.ELU(),
            'gelu': torch.nn.GELU(),
            'swish': torch.nn.SiLU(),
            'leaky_relu': torch.nn.LeakyReLU()
        }
        
        if hasattr(activation, 'value'):
            activation = activation.value
        
        return activation_map.get(activation.lower(), torch.nn.ReLU())
    
    def _initialize_network(self, network: torch.nn.Module, scheme: str):
        """Initialize network weights."""
        for module in network.modules():
            if isinstance(module, torch.nn.Linear):
                if scheme == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(module.weight, gain=self.config.network.weight_init_gain)
                elif scheme == "xavier_normal":
                    torch.nn.init.xavier_normal_(module.weight, gain=self.config.network.weight_init_gain)
                elif scheme == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif scheme == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif scheme == "orthogonal":
                    torch.nn.init.orthogonal_(module.weight, gain=self.config.network.weight_init_gain)
                
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, self.config.network.bias_init)
    
    def _setup_safety_monitoring(self):
        """Setup safety monitoring system."""
        # Create safety monitor with environment constraints
        self.safety_monitor = SafetyMonitor()
        
        # Add environment safety constraints to monitor
        if hasattr(self.env, 'safety_constraints'):
            for constraint in self.env.safety_constraints:
                self.safety_monitor.add_constraint(constraint)
        
        self.logger.info(f"Safety monitor setup: {len(self.safety_monitor.constraints)} constraints")
    
    def _setup_distributed_training(self):
        """Setup distributed training."""
        if not self.is_distributed:
            return
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.distributed.backend,
                init_method=self.config.distributed.dist_url,
                world_size=self.world_size,
                rank=self.local_rank,
                timeout=torch.distributed.default_pg_timeout
            )
        
        # Wrap networks with DDP
        self.agent.policy = DDP(
            self.agent.policy,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.distributed.find_unused_parameters
        )
        
        self.agent.value_network = DDP(
            self.agent.value_network,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None
        )
        
        self.agent.constraint_value_network = DDP(
            self.agent.constraint_value_network,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None
        )
        
        self.logger.info(f"Distributed training setup: rank={self.local_rank}, world_size={self.world_size}")
    
    def train(self) -> TrainingResults:
        """
        Main training loop with comprehensive monitoring and checkpointing.
        
        Returns:
            Training results with performance metrics
        """
        self.logger.info("Starting CPO training...")
        
        results = TrainingResults()
        results.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        results.config_hash = str(hash(str(self.config.to_dict())))
        results.environment_name = self.config.env_name
        
        try:
            # Training loop
            while self.iteration < self.config.max_iterations:
                iteration_start_time = time.time()
                
                # Collect rollouts
                rollout_data = self._collect_rollouts()
                
                if rollout_data is None:
                    self.logger.warning(f"Failed to collect rollouts at iteration {self.iteration}")
                    continue
                
                # Train agent
                train_info = self._train_step(rollout_data)
                
                # Update metrics
                self._update_training_metrics(train_info, rollout_data)
                
                # Evaluation
                if (self.iteration + 1) % self.config.evaluation.eval_frequency == 0:
                    eval_metrics = self._evaluate()
                    self._update_evaluation_metrics(eval_metrics, results)
                
                # Checkpointing
                if (self.iteration + 1) % self.config.save_frequency == 0:
                    self.save_checkpoint(self.iteration + 1)
                
                # Logging
                if (self.iteration + 1) % self.config.print_frequency == 0:
                    self._log_training_progress(train_info, rollout_data, iteration_start_time)
                
                self.iteration += 1
            
            # Final evaluation
            final_eval = self._evaluate()
            results.final_reward = final_eval.get('mean_reward', 0.0)
            results.final_success_rate = final_eval.get('success_rate', 0.0)
            
            # Finalize results
            results.total_iterations = self.iteration
            results.total_timesteps = self.total_timesteps
            results.total_episodes = self.total_episodes
            results.training_time = time.time() - self.start_time
            results.best_reward = self.best_reward
            results.best_success_rate = self.best_success_rate
            
            # Save final model
            self.save_checkpoint(self.iteration, is_final=True)
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            # Save current state
            self.save_checkpoint(self.iteration, is_interrupted=True)
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            # Save current state for debugging
            self.save_checkpoint(self.iteration, is_error=True)
            raise
        
        finally:
            results.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Clean up distributed training
            if self.is_distributed and dist.is_initialized():
                dist.destroy_process_group()
        
        self.logger.info("Training completed successfully")
        return results
    
    def _collect_rollouts(self) -> Optional[Dict[str, Any]]:
        """Collect rollouts from environment."""
        try:
            # Initialize rollout storage
            rollout_data = {
                'states': [],
                'actions': [],
                'rewards': [],
                'constraint_costs': [],
                'log_probs': [],
                'values': [],
                'constraint_values': [],
                'dones': [],
                'episode_rewards': [],
                'episode_lengths': [],
                'safety_violations': 0
            }
            
            # Collect rollouts
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episodes_collected = 0
            
            for step in range(self.config.rollout_length):
                # Get action from policy
                state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action, log_prob = self.agent.act(state_tensor)
                    value = self.agent.value_network(state_tensor)
                    constraint_value = self.agent.constraint_value_network(state_tensor)
                
                # Step environment
                next_obs, reward, done, info = self.env.step(action.cpu().numpy()[0])
                
                # Extract constraint cost
                constraint_cost = self._extract_constraint_cost(info)
                
                # Store transition
                rollout_data['states'].append(obs)
                rollout_data['actions'].append(action.cpu().numpy()[0])
                rollout_data['rewards'].append(reward)
                rollout_data['constraint_costs'].append(constraint_cost)
                rollout_data['log_probs'].append(log_prob.cpu().numpy()[0])
                rollout_data['values'].append(value.cpu().numpy()[0])
                rollout_data['constraint_values'].append(constraint_value.cpu().numpy()[0])
                rollout_data['dones'].append(done)
                
                # Track safety violations
                if info.get('constraint_violations', 0) > 0:
                    rollout_data['safety_violations'] += 1
                
                # Update episode tracking
                episode_reward += reward
                episode_length += 1
                self.total_timesteps += 1
                
                if done:
                    # Episode completed
                    rollout_data['episode_rewards'].append(episode_reward)
                    rollout_data['episode_lengths'].append(episode_length)
                    episodes_collected += 1
                    self.total_episodes += 1
                    
                    # Reset environment
                    obs = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs
            
            # Convert to tensors
            for key in ['states', 'actions', 'rewards', 'constraint_costs', 'log_probs', 'values', 'constraint_values']:
                rollout_data[key] = torch.FloatTensor(rollout_data[key]).to(self.device)
            
            rollout_data['dones'] = torch.BoolTensor(rollout_data['dones']).to(self.device)
            rollout_data['episodes_collected'] = episodes_collected
            
            return rollout_data
            
        except Exception as e:
            self.logger.error(f"Error collecting rollouts: {e}", exc_info=True)
            return None
    
    def _extract_constraint_cost(self, info: Dict[str, Any]) -> float:
        """Extract constraint cost from environment info."""
        constraint_cost = 0.0
        
        # Safety penalty from environment
        if 'safety_penalty' in info:
            constraint_cost += abs(info['safety_penalty'])
        
        # Constraint violations
        if 'constraint_violations' in info:
            constraint_cost += info['constraint_violations'] * 10.0
        
        # Environment-specific constraint costs
        if hasattr(self.env, 'get_safety_metrics'):
            try:
                safety_metrics = self.env.get_safety_metrics()
                constraint_cost += safety_metrics.get('violation_rate', 0.0) * 5.0
            except:
                pass  # Ignore errors in safety metrics extraction
        
        return constraint_cost
    
    def _train_step(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one training step with collected rollouts."""
        try:
            # Prepare training data
            states = rollout_data['states']
            actions = rollout_data['actions']
            rewards = rollout_data['rewards']
            constraint_costs = rollout_data['constraint_costs']
            old_log_probs = rollout_data['log_probs']
            
            # Compute advantages using GAE
            advantages = self.agent.gae.compute_advantages(
                rewards, 
                rollout_data['values'][:-1] if len(rollout_data['values']) > len(rewards) else rollout_data['values'],
                rollout_data['values'][1:] if len(rollout_data['values']) > len(rewards) else torch.cat([rollout_data['values'][1:], torch.zeros(1).to(self.device)])
            )
            
            constraint_advantages = self.agent.gae.compute_advantages(
                constraint_costs,
                rollout_data['constraint_values'][:-1] if len(rollout_data['constraint_values']) > len(constraint_costs) else rollout_data['constraint_values'],
                rollout_data['constraint_values'][1:] if len(rollout_data['constraint_values']) > len(constraint_costs) else torch.cat([rollout_data['constraint_values'][1:], torch.zeros(1).to(self.device)])
            )
            
            # Compute returns
            returns = advantages + rollout_data['values'][:len(advantages)]
            constraint_returns = constraint_advantages + rollout_data['constraint_values'][:len(constraint_advantages)]
            
            # Ensure all tensors have the same length
            min_length = min(len(states), len(actions), len(advantages), len(constraint_advantages))
            
            # Train CPO
            train_info = self.agent.train_step(
                states=states[:min_length],
                actions=actions[:min_length],
                rewards=rewards[:min_length],
                constraint_costs=constraint_costs[:min_length],
                old_log_probs=old_log_probs[:min_length],
                advantages=advantages[:min_length],
                constraint_advantages=constraint_advantages[:min_length],
                returns=returns[:min_length],
                constraint_returns=constraint_returns[:min_length]
            )
            
            return train_info
            
        except Exception as e:
            self.logger.error(f"Error in training step: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _update_training_metrics(self, train_info: Dict[str, Any], rollout_data: Dict[str, Any]):
        """Update training metrics."""
        if 'error' in train_info:
            return
        
        # Training dynamics
        if 'policy_gradient_norm' in train_info:
            self.training_metrics['policy_gradient_norms'].append(train_info['policy_gradient_norm'])
        if 'kl_divergence' in train_info:
            self.training_metrics['kl_divergences'].append(train_info['kl_divergence'])
        if 'value_loss' in train_info:
            self.training_metrics['value_losses'].append(train_info['value_loss'])
        if 'constraint_loss' in train_info:
            self.training_metrics['constraint_losses'].append(train_info['constraint_loss'])
        
        # Rollout metrics
        if rollout_data['episode_rewards']:
            mean_reward = np.mean(rollout_data['episode_rewards'])
            self.training_metrics['episode_rewards'].append(mean_reward)
        
        if rollout_data['episode_lengths']:
            mean_length = np.mean(rollout_data['episode_lengths'])
            self.training_metrics['episode_lengths'].append(mean_length)
        
        # Safety metrics
        violation_rate = rollout_data['safety_violations'] / max(1, rollout_data['episodes_collected'])
        self.training_metrics['violation_rates'].append(violation_rate)
    
    def _evaluate(self) -> Dict[str, float]:
        """Perform comprehensive evaluation."""
        if self.eval_env is None:
            return {}
        
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_violations = []
        
        try:
            for episode in range(self.config.evaluation.eval_episodes):
                obs = self.eval_env.reset(seed=self.config.evaluation.eval_env_seeds[episode % len(self.config.evaluation.eval_env_seeds)])
                
                episode_reward = 0
                episode_length = 0
                episode_violations = 0
                done = False
                
                max_steps = self.config.evaluation.eval_max_episode_steps or 1000
                
                while not done and episode_length < max_steps:
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        if self.config.evaluation.eval_deterministic:
                            # Use mean action for deterministic evaluation
                            action_dist = self.agent.policy(state_tensor)
                            action = action_dist.mean
                        else:
                            action, _ = self.agent.act(state_tensor)
                    
                    obs, reward, done, info = self.eval_env.step(action.cpu().numpy()[0])
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    if info.get('constraint_violations', 0) > 0:
                        episode_violations += 1
                
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                eval_successes.append(1.0 if info.get('task_complete', info.get('goal_reached', False)) else 0.0)
                eval_violations.append(episode_violations)
            
            # Compute evaluation metrics
            eval_metrics = {
                'mean_reward': float(np.mean(eval_rewards)),
                'std_reward': float(np.std(eval_rewards)),
                'mean_length': float(np.mean(eval_lengths)),
                'success_rate': float(np.mean(eval_successes)),
                'violation_rate': float(np.mean([v > 0 for v in eval_violations])),
                'mean_violations': float(np.mean(eval_violations))
            }
            
            return eval_metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}", exc_info=True)
            return {}
    
    def _update_evaluation_metrics(self, eval_metrics: Dict[str, float], results: TrainingResults):
        """Update evaluation metrics and track best performance."""
        if not eval_metrics:
            return
        
        # Update results
        results.eval_rewards.append(eval_metrics['mean_reward'])
        results.eval_success_rates.append(eval_metrics['success_rate'])
        results.eval_iterations.append(self.iteration)
        
        # Update best performance
        if eval_metrics['mean_reward'] > self.best_reward:
            self.best_reward = eval_metrics['mean_reward']
            if self.config.evaluation.save_best_model:
                self.save_checkpoint(self.iteration, is_best=True)
        
        if eval_metrics['success_rate'] > self.best_success_rate:
            self.best_success_rate = eval_metrics['success_rate']
        
        # Store in evaluation history
        eval_record = {
            'iteration': self.iteration,
            'timestamp': time.time(),
            **eval_metrics
        }
        self.eval_history.append(eval_record)
    
    def _log_training_progress(self, train_info: Dict[str, Any], rollout_data: Dict[str, Any], iteration_start_time: float):
        """Log training progress."""
        iteration_time = time.time() - iteration_start_time
        
        # Compute metrics
        mean_reward = np.mean(rollout_data['episode_rewards']) if rollout_data['episode_rewards'] else 0.0
        mean_length = np.mean(rollout_data['episode_lengths']) if rollout_data['episode_lengths'] else 0.0
        violation_rate = rollout_data['safety_violations'] / max(1, rollout_data['episodes_collected'])
        
        # Format log message
        log_msg = (
            f"Iter {self.iteration+1:4d}/{self.config.max_iterations}: "
            f"Reward={mean_reward:6.2f}, "
            f"Length={mean_length:5.1f}, "
            f"Violations={violation_rate:5.3f}, "
            f"Time={iteration_time:.2f}s"
        )
        
        # Add training info
        if 'kl_divergence' in train_info:
            log_msg += f", KL={train_info['kl_divergence']:.4f}"
        
        if 'value_loss' in train_info:
            log_msg += f", VLoss={train_info['value_loss']:.4f}"
        
        self.logger.info(log_msg)
    
    def save_checkpoint(self, iteration: int, is_best: bool = False, is_final: bool = False, 
                       is_interrupted: bool = False, is_error: bool = False) -> str:
        """
        Save training checkpoint with comprehensive state.
        
        Args:
            iteration: Current iteration
            is_best: Whether this is the best model so far
            is_final: Whether this is the final checkpoint
            is_interrupted: Whether training was interrupted
            is_error: Whether an error occurred
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint filename
        if is_best:
            checkpoint_name = "best_model.pt"
        elif is_final:
            checkpoint_name = "final_model.pt"
        elif is_interrupted:
            checkpoint_name = f"interrupted_iter_{iteration}.pt"
        elif is_error:
            checkpoint_name = f"error_iter_{iteration}.pt"
        else:
            checkpoint_name = f"checkpoint_iter_{iteration}.pt"
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        try:
            # Prepare checkpoint data
            checkpoint = {
                'iteration': iteration,
                'total_timesteps': self.total_timesteps,
                'total_episodes': self.total_episodes,
                'best_reward': self.best_reward,
                'best_success_rate': self.best_success_rate,
                'training_time': time.time() - self.start_time,
                'config': self.config.to_dict(),
                'training_metrics': dict(self.training_metrics),
                'eval_history': list(self.eval_history),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'random_state': {
                    'torch': torch.get_rng_state(),
                    'numpy': np.random.get_state()
                }
            }
            
            # Add model states
            if hasattr(self.agent, 'state_dict'):
                checkpoint['agent_state_dict'] = self.agent.state_dict()
            
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Also save configuration separately for easy access
            config_path = checkpoint_dir / f"config_{checkpoint_name.replace('.pt', '.json')}"
            self.config.save(config_path)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            return ""
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load training checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                self.logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Restore training state
            self.iteration = checkpoint['iteration']
            self.total_timesteps = checkpoint['total_timesteps']
            self.total_episodes = checkpoint['total_episodes']
            self.best_reward = checkpoint['best_reward']
            self.best_success_rate = checkpoint['best_success_rate']
            
            # Restore training metrics
            self.training_metrics.update(checkpoint.get('training_metrics', {}))
            self.eval_history.extend(checkpoint.get('eval_history', []))
            
            # Restore agent state
            if 'agent_state_dict' in checkpoint and hasattr(self.agent, 'load_state_dict'):
                self.agent.load_state_dict(checkpoint['agent_state_dict'])
            
            # Restore random states
            if 'random_state' in checkpoint:
                torch.set_rng_state(checkpoint['random_state']['torch'])
                np.random.set_state(checkpoint['random_state']['numpy'])
            
            self.logger.info(f"Checkpoint loaded successfully: {checkpoint_path}")
            self.logger.info(f"Resuming from iteration {self.iteration}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return False


class PolicyWrapper(torch.nn.Module):
    """Wrapper for policy network to provide CPO-compatible interface."""
    
    def __init__(self, network: torch.nn.Module, action_dim: int):
        super().__init__()
        self.network = network
        self.action_dim = action_dim
    
    def forward(self, states: torch.Tensor) -> torch.distributions.Distribution:
        """Forward pass returning action distribution."""
        output = self.network(states)
        mean, log_std = output.chunk(2, dim=-1)
        
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        return torch.distributions.Normal(mean, std)
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate actions for training."""
        dist = self.forward(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return {
            'log_probs': log_probs,
            'entropy': entropy,
            'mean': dist.mean,
            'std': dist.stddev
        }