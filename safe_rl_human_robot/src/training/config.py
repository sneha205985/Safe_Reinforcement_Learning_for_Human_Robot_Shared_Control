"""
Training configuration system for CPO with comprehensive parameter management.

This module provides dataclasses for all training-related configurations,
supporting hyperparameter optimization, experiment management, and 
distributed training scenarios.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import torch
import json
import yaml
from enum import Enum


class ActivationFunction(Enum):
    """Available activation functions."""
    RELU = "relu"
    TANH = "tanh"
    ELU = "elu"
    GELU = "gelu"
    SWISH = "swish"
    LEAKY_RELU = "leaky_relu"


class OptimizerType(Enum):
    """Available optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    NATURAL_GRADIENT = "natural_gradient"


class SchedulerType(Enum):
    """Available learning rate scheduler types."""
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    COSINE_WARM_RESTART = "cosine_warm_restart"
    PLATEAU = "plateau"


class ExperimentTrackerType(Enum):
    """Available experiment tracker types."""
    MLFLOW = "mlflow"
    WANDB = "wandb"
    TENSORBOARD = "tensorboard"
    NONE = "none"


@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    # Policy network
    policy_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    policy_activation: ActivationFunction = ActivationFunction.TANH
    policy_output_activation: Optional[ActivationFunction] = None
    policy_layer_norm: bool = False
    policy_dropout: float = 0.0
    
    # Value network
    value_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    value_activation: ActivationFunction = ActivationFunction.TANH
    value_layer_norm: bool = False
    value_dropout: float = 0.0
    
    # Constraint value network
    constraint_value_hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    constraint_value_activation: ActivationFunction = ActivationFunction.TANH
    constraint_value_layer_norm: bool = False
    constraint_value_dropout: float = 0.0
    
    # Network initialization
    initialization_scheme: str = "xavier_uniform"
    weight_init_gain: float = 1.0
    bias_init: float = 0.0
    
    # Advanced options
    spectral_normalization: bool = False
    gradient_clipping: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'policy_hidden_sizes': self.policy_hidden_sizes,
            'policy_activation': self.policy_activation.value,
            'policy_output_activation': self.policy_output_activation.value if self.policy_output_activation else None,
            'policy_layer_norm': self.policy_layer_norm,
            'policy_dropout': self.policy_dropout,
            'value_hidden_sizes': self.value_hidden_sizes,
            'value_activation': self.value_activation.value,
            'value_layer_norm': self.value_layer_norm,
            'value_dropout': self.value_dropout,
            'constraint_value_hidden_sizes': self.constraint_value_hidden_sizes,
            'constraint_value_activation': self.constraint_value_activation.value,
            'constraint_value_layer_norm': self.constraint_value_layer_norm,
            'constraint_value_dropout': self.constraint_value_dropout,
            'initialization_scheme': self.initialization_scheme,
            'weight_init_gain': self.weight_init_gain,
            'bias_init': self.bias_init,
            'spectral_normalization': self.spectral_normalization,
            'gradient_clipping': self.gradient_clipping
        }


@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms."""
    # CPO-specific parameters
    trust_region_radius: float = 0.01
    max_kl_divergence: float = 0.01
    constraint_threshold: float = 0.1
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # Learning rates
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    constraint_value_lr: float = 1e-3
    
    # Optimizer settings
    optimizer_type: OptimizerType = OptimizerType.ADAM
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_momentum: float = 0.9  # For SGD
    optimizer_betas: Tuple[float, float] = (0.9, 0.999)  # For Adam
    
    # Trust region optimization
    cg_iterations: int = 10
    cg_residual_tol: float = 1e-10
    cg_damping: float = 0.01
    line_search_steps: int = 10
    line_search_accept_ratio: float = 0.1
    line_search_backtrack_ratio: float = 0.5
    
    # Value function training
    value_train_iterations: int = 5
    constraint_value_train_iterations: int = 5
    value_batch_size: Optional[int] = None  # Use rollout batch size if None
    
    # Advanced optimization
    natural_gradient: bool = False
    use_gae_returns: bool = True
    normalize_advantages: bool = True
    normalize_rewards: bool = False
    clip_value_function: bool = False
    value_clip_range: float = 0.2
    
    # Regularization
    entropy_coefficient: float = 0.01
    l2_regularization: float = 0.0
    orthogonal_regularization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trust_region_radius': self.trust_region_radius,
            'max_kl_divergence': self.max_kl_divergence,
            'constraint_threshold': self.constraint_threshold,
            'gamma': self.gamma,
            'lambda_gae': self.lambda_gae,
            'policy_lr': self.policy_lr,
            'value_lr': self.value_lr,
            'constraint_value_lr': self.constraint_value_lr,
            'optimizer_type': self.optimizer_type.value,
            'optimizer_eps': self.optimizer_eps,
            'optimizer_weight_decay': self.optimizer_weight_decay,
            'optimizer_momentum': self.optimizer_momentum,
            'optimizer_betas': self.optimizer_betas,
            'cg_iterations': self.cg_iterations,
            'cg_residual_tol': self.cg_residual_tol,
            'cg_damping': self.cg_damping,
            'line_search_steps': self.line_search_steps,
            'line_search_accept_ratio': self.line_search_accept_ratio,
            'line_search_backtrack_ratio': self.line_search_backtrack_ratio,
            'value_train_iterations': self.value_train_iterations,
            'constraint_value_train_iterations': self.constraint_value_train_iterations,
            'value_batch_size': self.value_batch_size,
            'natural_gradient': self.natural_gradient,
            'use_gae_returns': self.use_gae_returns,
            'normalize_advantages': self.normalize_advantages,
            'normalize_rewards': self.normalize_rewards,
            'clip_value_function': self.clip_value_function,
            'value_clip_range': self.value_clip_range,
            'entropy_coefficient': self.entropy_coefficient,
            'l2_regularization': self.l2_regularization,
            'orthogonal_regularization': self.orthogonal_regularization
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation protocols."""
    # Evaluation frequency
    eval_frequency: int = 10  # Every N training iterations
    eval_episodes: int = 10
    eval_deterministic: bool = True
    eval_render: bool = False
    
    # Evaluation environments
    eval_env_seeds: List[int] = field(default_factory=lambda: list(range(42, 52)))
    eval_max_episode_steps: Optional[int] = None
    
    # Safety evaluation
    eval_safety_constraints: bool = True
    eval_constraint_violations: bool = True
    eval_recovery_performance: bool = True
    
    # Performance metrics
    eval_success_threshold: float = 0.8  # Success rate threshold
    eval_efficiency_metrics: bool = True
    eval_human_satisfaction: bool = False  # Requires human subjects
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 1e-4
    early_stopping_metric: str = "eval_reward"  # or "eval_success_rate"
    
    # Checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    save_frequency: int = 50  # Save every N iterations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'eval_frequency': self.eval_frequency,
            'eval_episodes': self.eval_episodes,
            'eval_deterministic': self.eval_deterministic,
            'eval_render': self.eval_render,
            'eval_env_seeds': self.eval_env_seeds,
            'eval_max_episode_steps': self.eval_max_episode_steps,
            'eval_safety_constraints': self.eval_safety_constraints,
            'eval_constraint_violations': self.eval_constraint_violations,
            'eval_recovery_performance': self.eval_recovery_performance,
            'eval_success_threshold': self.eval_success_threshold,
            'eval_efficiency_metrics': self.eval_efficiency_metrics,
            'eval_human_satisfaction': self.eval_human_satisfaction,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold,
            'early_stopping_metric': self.early_stopping_metric,
            'save_best_model': self.save_best_model,
            'save_last_model': self.save_last_model,
            'save_frequency': self.save_frequency
        }


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and management."""
    # Experiment identification
    experiment_name: str = "cpo_safe_rl"
    run_name: Optional[str] = None  # Auto-generated if None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Tracking configuration
    tracker_type: ExperimentTrackerType = ExperimentTrackerType.TENSORBOARD
    log_frequency: int = 1  # Log every N iterations
    log_gradients: bool = False
    log_weights: bool = False
    log_activations: bool = False
    
    # MLflow specific
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None
    
    # Weights & Biases specific
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    
    # Artifact management
    log_model_frequency: int = 50
    log_environment_info: bool = True
    log_hyperparameters: bool = True
    log_system_metrics: bool = True
    
    # Visualization
    log_training_plots: bool = True
    log_evaluation_videos: bool = False
    log_safety_metrics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'tags': self.tags,
            'notes': self.notes,
            'tracker_type': self.tracker_type.value,
            'log_frequency': self.log_frequency,
            'log_gradients': self.log_gradients,
            'log_weights': self.log_weights,
            'log_activations': self.log_activations,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'wandb_group': self.wandb_group,
            'log_model_frequency': self.log_model_frequency,
            'log_environment_info': self.log_environment_info,
            'log_hyperparameters': self.log_hyperparameters,
            'log_system_metrics': self.log_system_metrics,
            'log_training_plots': self.log_training_plots,
            'log_evaluation_videos': self.log_evaluation_videos,
            'log_safety_metrics': self.log_safety_metrics
        }


@dataclass
class SchedulerConfig:
    """Configuration for learning rate and constraint scheduling."""
    # Learning rate scheduling
    lr_scheduler_type: SchedulerType = SchedulerType.CONSTANT
    lr_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Constraint threshold scheduling
    constraint_scheduler_enabled: bool = False
    constraint_scheduler_type: SchedulerType = SchedulerType.CONSTANT
    constraint_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Trust region scheduling
    trust_region_scheduler_enabled: bool = False
    trust_region_scheduler_type: SchedulerType = SchedulerType.CONSTANT
    trust_region_scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Entropy scheduling
    entropy_scheduler_enabled: bool = False
    entropy_scheduler_type: SchedulerType = SchedulerType.LINEAR
    entropy_scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'start_value': 0.01, 'end_value': 0.001, 'decay_steps': 1000
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lr_scheduler_type': self.lr_scheduler_type.value,
            'lr_scheduler_params': self.lr_scheduler_params,
            'constraint_scheduler_enabled': self.constraint_scheduler_enabled,
            'constraint_scheduler_type': self.constraint_scheduler_type.value,
            'constraint_scheduler_params': self.constraint_scheduler_params,
            'trust_region_scheduler_enabled': self.trust_region_scheduler_enabled,
            'trust_region_scheduler_type': self.trust_region_scheduler_type.value,
            'trust_region_scheduler_params': self.trust_region_scheduler_params,
            'entropy_scheduler_enabled': self.entropy_scheduler_enabled,
            'entropy_scheduler_type': self.entropy_scheduler_type.value,
            'entropy_scheduler_params': self.entropy_scheduler_params
        }


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # Distribution setup
    distributed: bool = False
    backend: str = "nccl"  # "nccl", "gloo", "mpi"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Communication
    dist_url: str = "tcp://localhost:23456"
    timeout: int = 1800  # 30 minutes
    
    # Data parallelism
    data_parallel: bool = False
    model_parallel: bool = False
    
    # Gradient synchronization
    sync_batch_norm: bool = False
    find_unused_parameters: bool = False
    
    # Resource allocation
    gpus_per_node: int = 1
    cpus_per_gpu: int = 4
    memory_per_gpu: int = 8  # GB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'distributed': self.distributed,
            'backend': self.backend,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'dist_url': self.dist_url,
            'timeout': self.timeout,
            'data_parallel': self.data_parallel,
            'model_parallel': self.model_parallel,
            'sync_batch_norm': self.sync_batch_norm,
            'find_unused_parameters': self.find_unused_parameters,
            'gpus_per_node': self.gpus_per_node,
            'cpus_per_gpu': self.cpus_per_gpu,
            'memory_per_gpu': self.memory_per_gpu
        }


@dataclass
class TrainingConfig:
    """Master configuration for CPO training."""
    # Basic training parameters
    max_iterations: int = 1000
    rollout_length: int = 2000
    batch_size: int = 64
    mini_batch_size: Optional[int] = None
    
    # Environment configuration
    env_name: str = "exoskeleton"
    env_config: Dict[str, Any] = field(default_factory=dict)
    num_envs: int = 1  # Parallel environments
    
    # Seed and reproducibility
    seed: int = 42
    torch_deterministic: bool = True
    cuda_deterministic: bool = True
    
    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    num_threads: int = 1
    
    # Checkpointing and resuming
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    save_frequency: int = 50
    
    # Logging and monitoring
    log_dir: str = "logs"
    log_level: str = "INFO"
    print_frequency: int = 10
    
    # Component configurations
    network: NetworkConfig = field(default_factory=NetworkConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate configuration
        self._validate_config()
        
        # Set up device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set mini-batch size if not specified
        if self.mini_batch_size is None:
            self.mini_batch_size = min(self.batch_size, 64)
        
        # Create directories
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        if self.rollout_length <= 0:
            raise ValueError("rollout_length must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        
        if not (0 < self.optimization.gamma < 1):
            raise ValueError("gamma must be in (0, 1)")
        
        if not (0 <= self.optimization.lambda_gae <= 1):
            raise ValueError("lambda_gae must be in [0, 1]")
        
        if self.optimization.trust_region_radius <= 0:
            raise ValueError("trust_region_radius must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            'max_iterations': self.max_iterations,
            'rollout_length': self.rollout_length,
            'batch_size': self.batch_size,
            'mini_batch_size': self.mini_batch_size,
            'env_name': self.env_name,
            'env_config': self.env_config,
            'num_envs': self.num_envs,
            'seed': self.seed,
            'torch_deterministic': self.torch_deterministic,
            'cuda_deterministic': self.cuda_deterministic,
            'device': self.device,
            'num_threads': self.num_threads,
            'checkpoint_dir': self.checkpoint_dir,
            'resume_from': self.resume_from,
            'save_frequency': self.save_frequency,
            'log_dir': self.log_dir,
            'log_level': self.log_level,
            'print_frequency': self.print_frequency,
            'network': self.network.to_dict(),
            'optimization': self.optimization.to_dict(),
            'evaluation': self.evaluation.to_dict(),
            'experiment': self.experiment.to_dict(),
            'scheduler': self.scheduler.to_dict(),
            'distributed': self.distributed.to_dict()
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        path = Path(path)
        
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config_dict = json.load(f)
        elif path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Extract component configurations
        network_config = NetworkConfig(**config_dict.pop('network', {}))
        optimization_config = OptimizationConfig(**config_dict.pop('optimization', {}))
        evaluation_config = EvaluationConfig(**config_dict.pop('evaluation', {}))
        experiment_config = ExperimentConfig(**config_dict.pop('experiment', {}))
        scheduler_config = SchedulerConfig(**config_dict.pop('scheduler', {}))
        distributed_config = DistributedConfig(**config_dict.pop('distributed', {}))
        
        # Handle enum conversions
        if 'network' in config_dict:
            network_dict = config_dict['network']
            if 'policy_activation' in network_dict:
                network_dict['policy_activation'] = ActivationFunction(network_dict['policy_activation'])
            if 'policy_output_activation' in network_dict and network_dict['policy_output_activation']:
                network_dict['policy_output_activation'] = ActivationFunction(network_dict['policy_output_activation'])
            if 'value_activation' in network_dict:
                network_dict['value_activation'] = ActivationFunction(network_dict['value_activation'])
            if 'constraint_value_activation' in network_dict:
                network_dict['constraint_value_activation'] = ActivationFunction(network_dict['constraint_value_activation'])
        
        if 'optimization' in config_dict and 'optimizer_type' in config_dict['optimization']:
            config_dict['optimization']['optimizer_type'] = OptimizerType(config_dict['optimization']['optimizer_type'])
        
        if 'experiment' in config_dict and 'tracker_type' in config_dict['experiment']:
            config_dict['experiment']['tracker_type'] = ExperimentTrackerType(config_dict['experiment']['tracker_type'])
        
        # Create configuration
        return cls(
            network=network_config,
            optimization=optimization_config,
            evaluation=evaluation_config,
            experiment=experiment_config,
            scheduler=scheduler_config,
            distributed=distributed_config,
            **config_dict
        )
    
    def get_hyperparameter_dict(self) -> Dict[str, Union[float, int, str]]:
        """Get flattened hyperparameter dictionary for optimization."""
        return {
            # Optimization parameters
            'policy_lr': self.optimization.policy_lr,
            'value_lr': self.optimization.value_lr,
            'trust_region_radius': self.optimization.trust_region_radius,
            'constraint_threshold': self.optimization.constraint_threshold,
            'gamma': self.optimization.gamma,
            'lambda_gae': self.optimization.lambda_gae,
            'entropy_coefficient': self.optimization.entropy_coefficient,
            
            # Network architecture
            'policy_hidden_size': self.network.policy_hidden_sizes[0] if self.network.policy_hidden_sizes else 256,
            'policy_num_layers': len(self.network.policy_hidden_sizes),
            'value_hidden_size': self.network.value_hidden_sizes[0] if self.network.value_hidden_sizes else 256,
            'value_num_layers': len(self.network.value_hidden_sizes),
            
            # Training parameters
            'batch_size': self.batch_size,
            'rollout_length': self.rollout_length,
            
            # Architecture choices
            'policy_activation': self.network.policy_activation.value,
            'value_activation': self.network.value_activation.value,
            'optimizer_type': self.optimization.optimizer_type.value
        }
    
    def update_from_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Update configuration from hyperparameter dictionary."""
        # Update optimization parameters
        if 'policy_lr' in hyperparams:
            self.optimization.policy_lr = hyperparams['policy_lr']
        if 'value_lr' in hyperparams:
            self.optimization.value_lr = hyperparams['value_lr']
        if 'trust_region_radius' in hyperparams:
            self.optimization.trust_region_radius = hyperparams['trust_region_radius']
        if 'constraint_threshold' in hyperparams:
            self.optimization.constraint_threshold = hyperparams['constraint_threshold']
        if 'gamma' in hyperparams:
            self.optimization.gamma = hyperparams['gamma']
        if 'lambda_gae' in hyperparams:
            self.optimization.lambda_gae = hyperparams['lambda_gae']
        if 'entropy_coefficient' in hyperparams:
            self.optimization.entropy_coefficient = hyperparams['entropy_coefficient']
        
        # Update network architecture
        if 'policy_hidden_size' in hyperparams:
            num_layers = hyperparams.get('policy_num_layers', len(self.network.policy_hidden_sizes))
            self.network.policy_hidden_sizes = [hyperparams['policy_hidden_size']] * num_layers
        if 'value_hidden_size' in hyperparams:
            num_layers = hyperparams.get('value_num_layers', len(self.network.value_hidden_sizes))
            self.network.value_hidden_sizes = [hyperparams['value_hidden_size']] * num_layers
        
        # Update training parameters
        if 'batch_size' in hyperparams:
            self.batch_size = hyperparams['batch_size']
        if 'rollout_length' in hyperparams:
            self.rollout_length = hyperparams['rollout_length']
        
        # Update architecture choices
        if 'policy_activation' in hyperparams:
            self.network.policy_activation = ActivationFunction(hyperparams['policy_activation'])
        if 'value_activation' in hyperparams:
            self.network.value_activation = ActivationFunction(hyperparams['value_activation'])
        if 'optimizer_type' in hyperparams:
            self.optimization.optimizer_type = OptimizerType(hyperparams['optimizer_type'])