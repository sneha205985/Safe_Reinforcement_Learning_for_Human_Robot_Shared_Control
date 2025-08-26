"""
Configuration settings for Safe RL Human-Robot Shared Control system.

This module provides centralized configuration management for all components
including hyperparameters, environment settings, and logging configuration.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml
import logging
from enum import Enum


class OptimizationMethod(Enum):
    """Available optimization methods."""
    CPO = "cpo"
    PPO_LAGRANGE = "ppo_lagrange"
    TRPO_CONSTRAINED = "trpo_constrained"


class ConstraintType(Enum):
    """Available constraint types."""
    COLLISION = "collision"
    JOINT_LIMITS = "joint_limits"
    FORCE_LIMITS = "force_limits"
    VELOCITY_LIMITS = "velocity_limits"


@dataclass
class NetworkConfig:
    """Neural network configuration."""
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"
    output_activation: Optional[str] = None
    layer_norm: bool = False
    dropout_rate: float = 0.0
    weight_init: str = "orthogonal"
    bias_init: str = "zeros"


@dataclass
class PolicyConfig:
    """Policy configuration."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    log_std_init: float = -0.5
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    action_scale: float = 1.0
    state_dependent_std: bool = False


@dataclass
class ConstraintConfig:
    """Individual constraint configuration."""
    constraint_type: ConstraintType
    enabled: bool = True
    threshold: float = 0.0
    violation_penalty: float = 1000.0
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CPOConfig:
    """Constrained Policy Optimization configuration."""
    # Lagrangian parameters
    dual_lr: float = 0.1
    dual_regularization: float = 1e-4
    dual_min: float = 1e-8
    dual_max: float = 100.0
    
    # Trust region parameters  
    trust_region_size: float = 0.01
    max_kl_divergence: float = 0.01
    
    # Line search parameters
    line_search_steps: int = 10
    line_search_decay: float = 0.5
    line_search_accept_ratio: float = 0.1
    
    # Conjugate gradient parameters
    cg_iterations: int = 10
    cg_tolerance: float = 1e-8
    cg_damping: float = 1e-4
    
    # Convergence criteria
    gradient_tolerance: float = 1e-4
    constraint_tolerance: float = 1e-3
    max_iterations: int = 100
    min_iterations: int = 10


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    # Robot parameters
    robot_dof: int = 6
    control_frequency: float = 10.0
    max_episode_steps: int = 1000
    
    # Workspace boundaries
    workspace_bounds: List[float] = field(default_factory=lambda: [-1.0, 1.0, 2.0])
    
    # Safety parameters
    safety_distance: float = 0.1
    force_limit: float = 50.0
    velocity_limit: float = 2.0
    
    # Task parameters
    target_reached_threshold: float = 0.05
    success_reward: float = 100.0
    collision_penalty: float = -1000.0
    efficiency_weight: float = 0.1
    
    # Human model parameters
    human_skill_level: float = 0.7
    human_noise_level: float = 0.1
    human_adaptation_rate: float = 0.1
    
    # Assistance mode
    assistance_mode: str = "collaborative"


@dataclass
class TrainingConfig:
    """Training configuration."""
    # General training parameters
    total_timesteps: int = 1000000
    batch_size: int = 256
    mini_batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 3e-4
    
    # Experience collection
    steps_per_rollout: int = 2048
    num_envs: int = 1
    normalize_advantages: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Regularization
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Evaluation
    eval_frequency: int = 10000
    eval_episodes: int = 10
    
    # Checkpointing
    save_frequency: int = 50000
    max_checkpoints: int = 5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    # Logging levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    
    # File logging
    log_dir: str = "logs"
    log_filename: str = "safe_rl_{timestamp}.log"
    max_log_files: int = 10
    max_file_size_mb: int = 100
    
    # Metrics logging
    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "safe-rl-human-robot"
    
    # What to log
    log_gradients: bool = False
    log_parameters: bool = True
    log_constraint_violations: bool = True
    log_episode_stats: bool = True
    
    # Logging frequency
    log_interval: int = 100
    metric_log_interval: int = 10


@dataclass
class SafeRLConfig:
    """Main configuration class for Safe RL system."""
    # Component configurations
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    cpo: CPOConfig = field(default_factory=CPOConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Constraint configurations
    constraints: List[ConstraintConfig] = field(default_factory=lambda: [
        ConstraintConfig(
            constraint_type=ConstraintType.COLLISION,
            threshold=0.1,
            violation_penalty=1000.0,
            parameters={"min_distance": 0.1, "human_position_idx": [0, 2], "robot_position_idx": [3, 5]}
        ),
        ConstraintConfig(
            constraint_type=ConstraintType.FORCE_LIMITS,
            threshold=50.0,
            violation_penalty=500.0,
            parameters={"max_force": 50.0}
        ),
        ConstraintConfig(
            constraint_type=ConstraintType.JOINT_LIMITS,
            threshold=0.0,
            violation_penalty=100.0,
            parameters={"joint_limits": {"joint_1": [-2.0, 2.0], "joint_2": [-2.0, 2.0]}}
        )
    ])
    
    # Optimization method
    optimization_method: OptimizationMethod = OptimizationMethod.CPO
    
    # Device configuration
    device: str = "cpu"
    seed: Optional[int] = None
    deterministic: bool = False
    
    # Experiment metadata
    experiment_name: str = "safe_rl_experiment"
    experiment_description: str = "Safe RL for Human-Robot Shared Control"
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate policy config
        if self.policy.network.activation not in ["tanh", "relu", "elu", "sigmoid"]:
            raise ValueError(f"Invalid activation function: {self.policy.network.activation}")
        
        # Validate CPO config
        if self.cpo.dual_lr <= 0:
            raise ValueError("Dual learning rate must be positive")
        
        if self.cpo.trust_region_size <= 0:
            raise ValueError("Trust region size must be positive")
        
        # Validate environment config
        if self.environment.robot_dof <= 0:
            raise ValueError("Robot DOF must be positive")
        
        if self.environment.safety_distance <= 0:
            raise ValueError("Safety distance must be positive")
        
        # Validate training config
        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.training.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        # Validate constraints
        for constraint in self.constraints:
            if constraint.violation_penalty < 0:
                raise ValueError(f"Violation penalty must be non-negative for {constraint.constraint_type}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SafeRLConfig':
        """Create config from dictionary."""
        # Convert constraint type strings to enums
        if 'constraints' in config_dict:
            for constraint in config_dict['constraints']:
                if isinstance(constraint.get('constraint_type'), str):
                    constraint['constraint_type'] = ConstraintType(constraint['constraint_type'])
        
        # Convert optimization method string to enum
        if 'optimization_method' in config_dict:
            if isinstance(config_dict['optimization_method'], str):
                config_dict['optimization_method'] = OptimizationMethod(config_dict['optimization_method'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'SafeRLConfig':
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'SafeRLConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        
        # Convert dataclass fields
        for field_name, field_obj in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            if isinstance(value, Enum):
                config_dict[field_name] = value.value
            elif hasattr(value, '__dataclass_fields__'):  # Nested dataclass
                config_dict[field_name] = self._dataclass_to_dict(value)
            elif isinstance(value, list) and value and hasattr(value[0], '__dataclass_fields__'):
                config_dict[field_name] = [self._dataclass_to_dict(item) for item in value]
            else:
                config_dict[field_name] = value
        
        return config_dict
    
    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert nested dataclass to dictionary."""
        result = {}
        for field_name, field_obj in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            
            if isinstance(value, Enum):
                result[field_name] = value.value
            elif hasattr(value, '__dataclass_fields__'):
                result[field_name] = self._dataclass_to_dict(value)
            elif isinstance(value, list) and value and hasattr(value[0], '__dataclass_fields__'):
                result[field_name] = [self._dataclass_to_dict(item) for item in value]
            else:
                result[field_name] = value
        
        return result
    
    def to_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update_from_dict(self, update_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary (partial updates)."""
        def update_nested(obj, updates):
            for key, value in updates.items():
                if hasattr(obj, key):
                    current_value = getattr(obj, key)
                    if hasattr(current_value, '__dataclass_fields__') and isinstance(value, dict):
                        # Recursively update nested dataclass
                        update_nested(current_value, value)
                    else:
                        # Direct assignment
                        setattr(obj, key, value)
                else:
                    logging.warning(f"Unknown configuration key: {key}")
        
        update_nested(self, update_dict)
        self._validate_config()
    
    def get_constraint_config(self, constraint_type: ConstraintType) -> Optional[ConstraintConfig]:
        """Get configuration for specific constraint type."""
        for constraint in self.constraints:
            if constraint.constraint_type == constraint_type:
                return constraint
        return None
    
    def is_constraint_enabled(self, constraint_type: ConstraintType) -> bool:
        """Check if specific constraint type is enabled."""
        config = self.get_constraint_config(constraint_type)
        return config.enabled if config else False
    
    def get_enabled_constraints(self) -> List[ConstraintConfig]:
        """Get list of enabled constraints."""
        return [c for c in self.constraints if c.enabled]
    
    def summary(self) -> str:
        """Get configuration summary string."""
        summary_lines = [
            f"Safe RL Configuration Summary",
            f"Experiment: {self.experiment_name} (v{self.version})",
            f"Optimization: {self.optimization_method.value}",
            f"Device: {self.device}",
            f"",
            f"Environment:",
            f"  Robot DOF: {self.environment.robot_dof}",
            f"  Control frequency: {self.environment.control_frequency} Hz",
            f"  Safety distance: {self.environment.safety_distance} m",
            f"  Force limit: {self.environment.force_limit} N",
            f"",
            f"Policy:",
            f"  Network: {self.policy.network.hidden_sizes}",
            f"  Activation: {self.policy.network.activation}",
            f"",
            f"CPO:",
            f"  Dual LR: {self.cpo.dual_lr}",
            f"  Trust region: {self.cpo.trust_region_size}",
            f"  Max KL: {self.cpo.max_kl_divergence}",
            f"",
            f"Training:",
            f"  Total timesteps: {self.training.total_timesteps:,}",
            f"  Batch size: {self.training.batch_size}",
            f"  Learning rate: {self.training.learning_rate}",
            f"",
            f"Constraints ({len(self.get_enabled_constraints())} enabled):"
        ]
        
        for constraint in self.get_enabled_constraints():
            summary_lines.append(f"  - {constraint.constraint_type.value}: penalty={constraint.violation_penalty}")
        
        return "\n".join(summary_lines)