"""
Learning rate and constraint schedulers for Safe RL training.

This module provides advanced scheduling capabilities for learning rates,
constraint thresholds, and other training parameters in CPO algorithms.
"""

import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class SchedulerConfig:
    """Configuration for parameter schedulers."""
    
    scheduler_type: str = "constant"
    initial_value: float = 1e-3
    final_value: Optional[float] = None
    
    # Step scheduler
    step_size: int = 100
    gamma: float = 0.1
    
    # Exponential scheduler
    decay_rate: float = 0.95
    
    # Cosine annealing
    T_max: int = 1000
    eta_min: float = 1e-6
    
    # Linear scheduler
    total_steps: int = 1000
    
    # Adaptive scheduler
    patience: int = 10
    factor: float = 0.5
    threshold: float = 1e-4
    cooldown: int = 0
    min_value: float = 1e-6
    
    # Warmup
    warmup_steps: int = 0
    warmup_start_factor: float = 0.1


class ParameterScheduler(ABC):
    """Abstract base class for parameter schedulers."""
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.step_count = 0
        self.current_value = config.initial_value
        
    @abstractmethod
    def step(self, metric: Optional[float] = None) -> float:
        """Update the parameter value and return it."""
        pass
    
    @abstractmethod
    def get_value(self) -> float:
        """Get the current parameter value."""
        pass
    
    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.step_count = 0
        self.current_value = self.config.initial_value
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'step_count': self.step_count,
            'current_value': self.current_value
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint."""
        self.step_count = state_dict['step_count']
        self.current_value = state_dict['current_value']


class ConstantScheduler(ParameterScheduler):
    """Constant parameter scheduler."""
    
    def step(self, metric: Optional[float] = None) -> float:
        """Return constant value."""
        self.step_count += 1
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value


class LinearScheduler(ParameterScheduler):
    """Linear parameter scheduler."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        if config.final_value is None:
            raise ValueError("LinearScheduler requires final_value")
        
        self.slope = (config.final_value - config.initial_value) / config.total_steps
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update value linearly."""
        self.step_count += 1
        
        if self.step_count <= self.config.total_steps:
            self.current_value = (
                self.config.initial_value + 
                self.slope * min(self.step_count, self.config.total_steps)
            )
        else:
            self.current_value = self.config.final_value
        
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value


class ExponentialScheduler(ParameterScheduler):
    """Exponential decay scheduler."""
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update value exponentially."""
        self.step_count += 1
        self.current_value = self.config.initial_value * (self.config.decay_rate ** self.step_count)
        
        if self.config.final_value is not None:
            self.current_value = max(self.current_value, self.config.final_value)
        
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value


class StepScheduler(ParameterScheduler):
    """Step-wise parameter scheduler."""
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update value in steps."""
        self.step_count += 1
        
        if self.step_count % self.config.step_size == 0:
            self.current_value *= self.config.gamma
        
        if self.config.final_value is not None:
            self.current_value = max(self.current_value, self.config.final_value)
        
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value


class CosineAnnealingScheduler(ParameterScheduler):
    """Cosine annealing scheduler."""
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update value using cosine annealing."""
        self.step_count += 1
        
        # Cosine annealing formula
        self.current_value = (
            self.config.eta_min + 
            (self.config.initial_value - self.config.eta_min) * 
            (1 + math.cos(math.pi * self.step_count / self.config.T_max)) / 2
        )
        
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value


class AdaptiveScheduler(ParameterScheduler):
    """Adaptive scheduler based on metric improvement."""
    
    def __init__(self, config: SchedulerConfig):
        super().__init__(config)
        self.best_metric = None
        self.patience_counter = 0
        self.cooldown_counter = 0
        self.mode = 'min'  # Assume we want to minimize the metric
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update value based on metric improvement."""
        self.step_count += 1
        
        if metric is None:
            return self.current_value
        
        # Cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_value
        
        # Check for improvement
        if self.best_metric is None:
            self.best_metric = metric
        elif metric < self.best_metric - self.config.threshold:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Reduce parameter if no improvement
        if self.patience_counter >= self.config.patience:
            self.current_value *= self.config.factor
            self.current_value = max(self.current_value, self.config.min_value)
            self.patience_counter = 0
            self.cooldown_counter = self.config.cooldown
            
            self.logger.info(f"Adaptive scheduler reduced value to {self.current_value}")
        
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value


class WarmupScheduler(ParameterScheduler):
    """Warmup scheduler that combines warmup with another scheduler."""
    
    def __init__(self, config: SchedulerConfig, base_scheduler: ParameterScheduler):
        super().__init__(config)
        self.base_scheduler = base_scheduler
        self.warmup_factor = config.warmup_start_factor
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update value with warmup."""
        self.step_count += 1
        
        if self.step_count <= self.config.warmup_steps:
            # Warmup phase
            warmup_progress = self.step_count / self.config.warmup_steps
            warmup_value = (
                self.warmup_factor + 
                (1.0 - self.warmup_factor) * warmup_progress
            )
            self.current_value = self.config.initial_value * warmup_value
        else:
            # Use base scheduler
            self.current_value = self.base_scheduler.step(metric)
        
        return self.current_value
    
    def get_value(self) -> float:
        """Get current value."""
        return self.current_value
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state including base scheduler."""
        state = super().state_dict()
        state['base_scheduler'] = self.base_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state including base scheduler."""
        super().load_state_dict(state_dict)
        if 'base_scheduler' in state_dict:
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


class LearningRateScheduler:
    """High-level learning rate scheduler for optimizers."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 config: SchedulerConfig, param_group_idx: int = 0):
        self.optimizer = optimizer
        self.param_group_idx = param_group_idx
        self.scheduler = self._create_scheduler(config)
        self.logger = logging.getLogger(__name__)
        
        # Store initial learning rate
        self.initial_lr = self.optimizer.param_groups[param_group_idx]['lr']
    
    def _create_scheduler(self, config: SchedulerConfig) -> ParameterScheduler:
        """Create appropriate scheduler based on configuration."""
        if config.scheduler_type == "constant":
            return ConstantScheduler(config)
        elif config.scheduler_type == "linear":
            return LinearScheduler(config)
        elif config.scheduler_type == "exponential":
            return ExponentialScheduler(config)
        elif config.scheduler_type == "step":
            return StepScheduler(config)
        elif config.scheduler_type == "cosine":
            return CosineAnnealingScheduler(config)
        elif config.scheduler_type == "adaptive":
            return AdaptiveScheduler(config)
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    def step(self, metric: Optional[float] = None) -> None:
        """Update learning rate."""
        new_lr = self.scheduler.step(metric)
        
        # Update optimizer learning rate
        self.optimizer.param_groups[self.param_group_idx]['lr'] = new_lr
        
        self.logger.debug(f"Learning rate updated to {new_lr}")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[self.param_group_idx]['lr']
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'scheduler': self.scheduler.state_dict(),
            'initial_lr': self.initial_lr
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.initial_lr = state_dict.get('initial_lr', self.initial_lr)


class ConstraintScheduler:
    """Scheduler for constraint-related parameters in CPO."""
    
    def __init__(self, constraint_name: str, config: SchedulerConfig):
        self.constraint_name = constraint_name
        self.scheduler = self._create_scheduler(config)
        self.logger = logging.getLogger(f"{__name__}.{constraint_name}")
    
    def _create_scheduler(self, config: SchedulerConfig) -> ParameterScheduler:
        """Create appropriate scheduler."""
        if config.scheduler_type == "constant":
            return ConstantScheduler(config)
        elif config.scheduler_type == "linear":
            return LinearScheduler(config)
        elif config.scheduler_type == "exponential":
            return ExponentialScheduler(config)
        elif config.scheduler_type == "step":
            return StepScheduler(config)
        elif config.scheduler_type == "cosine":
            return CosineAnnealingScheduler(config)
        elif config.scheduler_type == "adaptive":
            return AdaptiveScheduler(config)
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    def step(self, metric: Optional[float] = None) -> float:
        """Update constraint parameter."""
        new_value = self.scheduler.step(metric)
        self.logger.debug(f"{self.constraint_name} updated to {new_value}")
        return new_value
    
    def get_value(self) -> float:
        """Get current constraint parameter value."""
        return self.scheduler.get_value()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {
            'constraint_name': self.constraint_name,
            'scheduler': self.scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.constraint_name = state_dict['constraint_name']
        self.scheduler.load_state_dict(state_dict['scheduler'])


class SchedulerManager:
    """Manager for multiple schedulers in CPO training."""
    
    def __init__(self):
        self.lr_schedulers: Dict[str, LearningRateScheduler] = {}
        self.constraint_schedulers: Dict[str, ConstraintScheduler] = {}
        self.parameter_schedulers: Dict[str, ParameterScheduler] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_lr_scheduler(self, name: str, optimizer: torch.optim.Optimizer,
                        config: SchedulerConfig, param_group_idx: int = 0) -> None:
        """Add learning rate scheduler."""
        self.lr_schedulers[name] = LearningRateScheduler(
            optimizer, config, param_group_idx
        )
        self.logger.info(f"Added learning rate scheduler: {name}")
    
    def add_constraint_scheduler(self, name: str, config: SchedulerConfig) -> None:
        """Add constraint scheduler."""
        self.constraint_schedulers[name] = ConstraintScheduler(name, config)
        self.logger.info(f"Added constraint scheduler: {name}")
    
    def add_parameter_scheduler(self, name: str, config: SchedulerConfig) -> None:
        """Add general parameter scheduler."""
        self.parameter_schedulers[name] = self._create_scheduler(config)
        self.logger.info(f"Added parameter scheduler: {name}")
    
    def _create_scheduler(self, config: SchedulerConfig) -> ParameterScheduler:
        """Create scheduler from configuration."""
        if config.scheduler_type == "constant":
            return ConstantScheduler(config)
        elif config.scheduler_type == "linear":
            return LinearScheduler(config)
        elif config.scheduler_type == "exponential":
            return ExponentialScheduler(config)
        elif config.scheduler_type == "step":
            return StepScheduler(config)
        elif config.scheduler_type == "cosine":
            return CosineAnnealingScheduler(config)
        elif config.scheduler_type == "adaptive":
            return AdaptiveScheduler(config)
        else:
            raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    def step_all(self, metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Step all schedulers and return current values."""
        current_values = {}
        
        # Update learning rate schedulers
        for name, scheduler in self.lr_schedulers.items():
            metric = metrics.get(name) if metrics else None
            scheduler.step(metric)
            current_values[f"{name}_lr"] = scheduler.get_lr()
        
        # Update constraint schedulers
        for name, scheduler in self.constraint_schedulers.items():
            metric = metrics.get(name) if metrics else None
            value = scheduler.step(metric)
            current_values[name] = value
        
        # Update parameter schedulers
        for name, scheduler in self.parameter_schedulers.items():
            metric = metrics.get(name) if metrics else None
            value = scheduler.step(metric)
            current_values[name] = value
        
        return current_values
    
    def get_values(self) -> Dict[str, float]:
        """Get current values of all schedulers."""
        values = {}
        
        for name, scheduler in self.lr_schedulers.items():
            values[f"{name}_lr"] = scheduler.get_lr()
        
        for name, scheduler in self.constraint_schedulers.items():
            values[name] = scheduler.get_value()
        
        for name, scheduler in self.parameter_schedulers.items():
            values[name] = scheduler.get_value()
        
        return values
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state of all schedulers."""
        return {
            'lr_schedulers': {name: sched.state_dict() 
                             for name, sched in self.lr_schedulers.items()},
            'constraint_schedulers': {name: sched.state_dict() 
                                     for name, sched in self.constraint_schedulers.items()},
            'parameter_schedulers': {name: sched.state_dict() 
                                   for name, sched in self.parameter_schedulers.items()}
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state of all schedulers."""
        if 'lr_schedulers' in state_dict:
            for name, sched_state in state_dict['lr_schedulers'].items():
                if name in self.lr_schedulers:
                    self.lr_schedulers[name].load_state_dict(sched_state)
        
        if 'constraint_schedulers' in state_dict:
            for name, sched_state in state_dict['constraint_schedulers'].items():
                if name in self.constraint_schedulers:
                    self.constraint_schedulers[name].load_state_dict(sched_state)
        
        if 'parameter_schedulers' in state_dict:
            for name, sched_state in state_dict['parameter_schedulers'].items():
                if name in self.parameter_schedulers:
                    self.parameter_schedulers[name].load_state_dict(sched_state)


def create_common_schedulers(policy_optimizer: torch.optim.Optimizer,
                           value_optimizer: torch.optim.Optimizer) -> SchedulerManager:
    """Create commonly used schedulers for CPO training."""
    manager = SchedulerManager()
    
    # Policy learning rate scheduler (cosine annealing)
    policy_lr_config = SchedulerConfig(
        scheduler_type="cosine",
        initial_value=3e-4,
        T_max=1000,
        eta_min=1e-6
    )
    manager.add_lr_scheduler("policy", policy_optimizer, policy_lr_config)
    
    # Value learning rate scheduler (adaptive)
    value_lr_config = SchedulerConfig(
        scheduler_type="adaptive",
        initial_value=3e-4,
        patience=50,
        factor=0.8,
        threshold=1e-3,
        min_value=1e-6
    )
    manager.add_lr_scheduler("value", value_optimizer, value_lr_config)
    
    # Constraint threshold scheduler (linear increase)
    constraint_config = SchedulerConfig(
        scheduler_type="linear",
        initial_value=0.01,
        final_value=0.05,
        total_steps=500
    )
    manager.add_constraint_scheduler("constraint_threshold", constraint_config)
    
    return manager