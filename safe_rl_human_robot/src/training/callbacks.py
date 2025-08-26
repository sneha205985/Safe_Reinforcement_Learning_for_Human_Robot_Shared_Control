"""
Training callbacks and performance monitoring for Safe RL.

This module provides comprehensive callback system for training monitoring,
early stopping, model checkpointing, learning rate scheduling, and safety monitoring.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

from .config import TrainingConfig
from .experiment_tracker import ExperimentMetrics, ExperimentTracker


@dataclass
class CallbackState:
    """State information passed to callbacks."""
    
    iteration: int = 0
    epoch: int = 0
    metrics: ExperimentMetrics = None
    training_info: Dict[str, Any] = None
    rollout_data: Dict[str, torch.Tensor] = None
    
    # Model references
    policy_net: torch.nn.Module = None
    value_net: torch.nn.Module = None
    
    # Trainer reference
    trainer = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'iteration': self.iteration,
            'epoch': self.epoch,
            'metrics': self.metrics.to_dict() if self.metrics else {},
            'training_info': self.training_info or {}
        }


class TrainingCallback(ABC):
    """Abstract base class for training callbacks."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.enabled = True
    
    @abstractmethod
    def on_training_start(self, state: CallbackState) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_training_end(self, state: CallbackState) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_iteration_start(self, state: CallbackState) -> None:
        """Called at the start of each iteration."""
        pass
    
    @abstractmethod
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Called at the end of each iteration. Return True to stop training."""
        pass
    
    @abstractmethod
    def on_evaluation_start(self, state: CallbackState) -> None:
        """Called before evaluation."""
        pass
    
    @abstractmethod
    def on_evaluation_end(self, state: CallbackState) -> None:
        """Called after evaluation."""
        pass
    
    def enable(self) -> None:
        """Enable this callback."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable this callback."""
        self.enabled = False


class EarlyStopping(TrainingCallback):
    """Early stopping callback based on validation metrics."""
    
    def __init__(self, metric_name: str = 'episode_return', 
                 patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'max', restore_best: bool = True):
        super().__init__()
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        # Internal state
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.patience_counter = 0
        self.best_iteration = 0
        self.best_state_dict = None
        self.stopped_early = False
        
        # Validation
        if mode not in ['max', 'min']:
            raise ValueError("Mode must be 'max' or 'min'")
    
    def on_training_start(self, state: CallbackState) -> None:
        """Initialize early stopping state."""
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.patience_counter = 0
        self.best_iteration = 0
        self.best_state_dict = None
        self.stopped_early = False
        
        self.logger.info(f"Early stopping initialized: metric={self.metric_name}, "
                        f"patience={self.patience}, mode={self.mode}")
    
    def on_training_end(self, state: CallbackState) -> None:
        """Restore best model if requested."""
        if self.stopped_early and self.restore_best and self.best_state_dict:
            self.logger.info(f"Restoring best model from iteration {self.best_iteration}")
            state.policy_net.load_state_dict(self.best_state_dict['policy'])
            state.value_net.load_state_dict(self.best_state_dict['value'])
    
    def on_iteration_start(self, state: CallbackState) -> None:
        """No action needed at iteration start."""
        pass
    
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Check early stopping condition."""
        if not state.metrics:
            return False
        
        current_value = getattr(state.metrics, self.metric_name, None)
        if current_value is None:
            self.logger.warning(f"Metric {self.metric_name} not found in metrics")
            return False
        
        # Check if current value is better
        is_better = self._is_better(current_value)
        
        if is_better:
            self.best_value = current_value
            self.best_iteration = state.iteration
            self.patience_counter = 0
            
            # Save best model state
            if self.restore_best:
                self.best_state_dict = {
                    'policy': state.policy_net.state_dict().copy(),
                    'value': state.value_net.state_dict().copy()
                }
            
            self.logger.debug(f"New best {self.metric_name}: {current_value:.4f}")
        else:
            self.patience_counter += 1
            self.logger.debug(f"No improvement for {self.patience_counter} iterations")
        
        # Check if we should stop
        if self.patience_counter >= self.patience:
            self.logger.info(f"Early stopping triggered after {self.patience} iterations "
                           f"without improvement. Best {self.metric_name}: {self.best_value:.4f}")
            self.stopped_early = True
            return True
        
        return False
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """No action needed at evaluation start."""
        pass
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """No action needed at evaluation end."""
        pass
    
    def _is_better(self, current_value: float) -> bool:
        """Check if current value is better than best value."""
        if self.mode == 'max':
            return current_value > self.best_value + self.min_delta
        else:
            return current_value < self.best_value - self.min_delta


class ModelCheckpoint(TrainingCallback):
    """Model checkpointing callback."""
    
    def __init__(self, checkpoint_dir: str, save_frequency: int = 100,
                 save_best: bool = True, metric_name: str = 'episode_return',
                 mode: str = 'max', max_checkpoints: int = 5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.max_checkpoints = max_checkpoints
        
        # Internal state
        self.best_value = float('-inf') if mode == 'max' else float('inf')
        self.checkpoint_files = deque(maxlen=max_checkpoints)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def on_training_start(self, state: CallbackState) -> None:
        """Initialize checkpointing."""
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.logger.info(f"Model checkpointing initialized: dir={self.checkpoint_dir}, "
                        f"frequency={self.save_frequency}")
    
    def on_training_end(self, state: CallbackState) -> None:
        """Save final checkpoint."""
        self._save_checkpoint(state, "final")
    
    def on_iteration_start(self, state: CallbackState) -> None:
        """No action needed at iteration start."""
        pass
    
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Check if we should save checkpoint."""
        # Save periodic checkpoint
        if (state.iteration + 1) % self.save_frequency == 0:
            self._save_checkpoint(state, f"iter_{state.iteration + 1}")
        
        # Save best checkpoint
        if self.save_best and state.metrics:
            current_value = getattr(state.metrics, self.metric_name, None)
            if current_value is not None and self._is_better(current_value):
                self.best_value = current_value
                self._save_checkpoint(state, "best")
                self.logger.info(f"Saved best model with {self.metric_name}: {current_value:.4f}")
        
        return False
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """No action needed at evaluation start."""
        pass
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """No action needed at evaluation end."""
        pass
    
    def _save_checkpoint(self, state: CallbackState, suffix: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}.pth")
        
        try:
            checkpoint = {
                'iteration': state.iteration,
                'policy_net_state_dict': state.policy_net.state_dict(),
                'value_net_state_dict': state.value_net.state_dict(),
                'metrics': state.metrics.to_dict() if state.metrics else {},
                'training_info': state.training_info or {}
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Manage checkpoint files
            if suffix not in ["best", "final"] and self.max_checkpoints > 0:
                self.checkpoint_files.append(checkpoint_path)
                
                # Remove old checkpoints if we exceed max_checkpoints
                if len(self.checkpoint_files) > self.max_checkpoints:
                    old_checkpoint = self.checkpoint_files.popleft()
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
            
            self.logger.debug(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _is_better(self, current_value: float) -> bool:
        """Check if current value is better than best value."""
        if self.mode == 'max':
            return current_value > self.best_value
        else:
            return current_value < self.best_value


class LearningRateScheduler(TrainingCallback):
    """Learning rate scheduling callback."""
    
    def __init__(self, scheduler_type: str = 'step', **scheduler_kwargs):
        super().__init__()
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs
        self.policy_scheduler = None
        self.value_scheduler = None
    
    def on_training_start(self, state: CallbackState) -> None:
        """Initialize learning rate schedulers."""
        if not hasattr(state.trainer, 'policy_optimizer') or not hasattr(state.trainer, 'value_optimizer'):
            self.logger.warning("Optimizers not found in trainer, skipping LR scheduling")
            return
        
        # Create schedulers
        if self.scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.policy_scheduler = StepLR(state.trainer.policy_optimizer, **self.scheduler_kwargs)
            self.value_scheduler = StepLR(state.trainer.value_optimizer, **self.scheduler_kwargs)
            
        elif self.scheduler_type == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            self.policy_scheduler = ExponentialLR(state.trainer.policy_optimizer, **self.scheduler_kwargs)
            self.value_scheduler = ExponentialLR(state.trainer.value_optimizer, **self.scheduler_kwargs)
            
        elif self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.policy_scheduler = CosineAnnealingLR(state.trainer.policy_optimizer, **self.scheduler_kwargs)
            self.value_scheduler = CosineAnnealingLR(state.trainer.value_optimizer, **self.scheduler_kwargs)
            
        elif self.scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.policy_scheduler = ReduceLROnPlateau(state.trainer.policy_optimizer, **self.scheduler_kwargs)
            self.value_scheduler = ReduceLROnPlateau(state.trainer.value_optimizer, **self.scheduler_kwargs)
            
        else:
            self.logger.error(f"Unknown scheduler type: {self.scheduler_type}")
            return
        
        self.logger.info(f"Learning rate scheduler initialized: {self.scheduler_type}")
    
    def on_training_end(self, state: CallbackState) -> None:
        """No action needed at training end."""
        pass
    
    def on_iteration_start(self, state: CallbackState) -> None:
        """No action needed at iteration start."""
        pass
    
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Update learning rates."""
        if self.policy_scheduler is None or self.value_scheduler is None:
            return False
        
        if self.scheduler_type == 'plateau':
            # ReduceLROnPlateau needs a metric
            if state.metrics and hasattr(state.metrics, 'policy_loss'):
                self.policy_scheduler.step(state.metrics.policy_loss)
                self.value_scheduler.step(state.metrics.value_loss)
        else:
            # Other schedulers just need step()
            self.policy_scheduler.step()
            self.value_scheduler.step()
        
        # Log current learning rates
        policy_lr = self.policy_scheduler.get_last_lr()[0]
        value_lr = self.value_scheduler.get_last_lr()[0]
        
        self.logger.debug(f"Updated learning rates - Policy: {policy_lr:.6f}, Value: {value_lr:.6f}")
        
        return False
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """No action needed at evaluation start."""
        pass
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """No action needed at evaluation end."""
        pass


class SafetyMonitorCallback(TrainingCallback):
    """Safety monitoring callback for constraint violations."""
    
    def __init__(self, max_violations: int = 100, violation_threshold: float = 0.1,
                 alert_frequency: int = 10):
        super().__init__()
        self.max_violations = max_violations
        self.violation_threshold = violation_threshold
        self.alert_frequency = alert_frequency
        
        # Internal state
        self.violation_count = 0
        self.violation_history = deque(maxlen=100)
        self.last_alert_iteration = 0
    
    def on_training_start(self, state: CallbackState) -> None:
        """Initialize safety monitoring."""
        self.violation_count = 0
        self.violation_history.clear()
        self.last_alert_iteration = 0
        
        self.logger.info(f"Safety monitoring initialized: max_violations={self.max_violations}, "
                        f"threshold={self.violation_threshold}")
    
    def on_training_end(self, state: CallbackState) -> None:
        """Report final safety statistics."""
        if self.violation_history:
            avg_violations = np.mean(self.violation_history)
            max_violations = max(self.violation_history)
            self.logger.info(f"Safety summary - Total violations: {self.violation_count}, "
                           f"Average per episode: {avg_violations:.2f}, "
                           f"Maximum per episode: {max_violations}")
    
    def on_iteration_start(self, state: CallbackState) -> None:
        """No action needed at iteration start."""
        pass
    
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Monitor safety violations."""
        if not state.metrics:
            return False
        
        current_violations = state.metrics.constraint_violations
        self.violation_count += current_violations
        self.violation_history.append(current_violations)
        
        # Check violation rate
        if len(self.violation_history) >= 10:
            recent_avg = np.mean(list(self.violation_history)[-10:])
            
            # Alert if violation rate is too high
            if (recent_avg > self.violation_threshold and 
                state.iteration - self.last_alert_iteration >= self.alert_frequency):
                
                self.logger.warning(f"High violation rate detected: {recent_avg:.3f} "
                                  f"violations per episode (threshold: {self.violation_threshold})")
                self.last_alert_iteration = state.iteration
        
        # Stop training if violations exceed maximum
        if self.violation_count > self.max_violations:
            self.logger.error(f"Maximum violations exceeded: {self.violation_count} > {self.max_violations}")
            return True
        
        return False
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """No action needed at evaluation start."""
        pass
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """No action needed at evaluation end."""
        pass


class PerformanceMonitorCallback(TrainingCallback):
    """Performance monitoring and visualization callback."""
    
    def __init__(self, plot_frequency: int = 100, save_plots: bool = True,
                 plot_dir: str = "plots"):
        super().__init__()
        self.plot_frequency = plot_frequency
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        # History tracking
        self.metrics_history = []
        self.performance_metrics = [
            'episode_return', 'success_rate', 'constraint_violations',
            'policy_loss', 'value_loss', 'kl_divergence'
        ]
        
        if save_plots:
            os.makedirs(plot_dir, exist_ok=True)
    
    def on_training_start(self, state: CallbackState) -> None:
        """Initialize performance monitoring."""
        self.metrics_history.clear()
        self.logger.info("Performance monitoring initialized")
    
    def on_training_end(self, state: CallbackState) -> None:
        """Generate final performance plots."""
        if self.save_plots and self.metrics_history:
            self._generate_final_plots()
    
    def on_iteration_start(self, state: CallbackState) -> None:
        """No action needed at iteration start."""
        pass
    
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Track metrics and generate plots."""
        if state.metrics:
            # Store metrics with iteration
            metrics_dict = state.metrics.to_dict()
            metrics_dict['iteration'] = state.iteration
            self.metrics_history.append(metrics_dict)
        
        # Generate plots periodically
        if ((state.iteration + 1) % self.plot_frequency == 0 and 
            self.save_plots and len(self.metrics_history) > 10):
            self._generate_plots(state.iteration)
        
        return False
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """No action needed at evaluation start."""
        pass
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """No action needed at evaluation end."""
        pass
    
    def _generate_plots(self, iteration: int) -> None:
        """Generate performance plots."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            iterations = [m['iteration'] for m in self.metrics_history]
            
            for i, metric in enumerate(self.performance_metrics):
                if i < len(axes):
                    values = [m.get(metric, 0) for m in self.metrics_history]
                    axes[i].plot(iterations, values)
                    axes[i].set_title(f'{metric.replace("_", " ").title()}')
                    axes[i].set_xlabel('Iteration')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if self.save_plots:
                plot_path = os.path.join(self.plot_dir, f'performance_iter_{iteration}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to generate plots: {e}")
    
    def _generate_final_plots(self) -> None:
        """Generate comprehensive final plots."""
        self._generate_plots("final")


class CallbackManager:
    """Manager for handling multiple training callbacks."""
    
    def __init__(self, callbacks: List[TrainingCallback] = None):
        self.callbacks = callbacks or []
        self.logger = logging.getLogger(__name__)
        
        # Add default callbacks if none provided
        if not self.callbacks:
            self.add_default_callbacks()
    
    def add_callback(self, callback: TrainingCallback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)
        self.logger.info(f"Added callback: {callback.name}")
    
    def remove_callback(self, callback_name: str) -> None:
        """Remove a callback by name."""
        self.callbacks = [cb for cb in self.callbacks if cb.name != callback_name]
        self.logger.info(f"Removed callback: {callback_name}")
    
    def add_default_callbacks(self) -> None:
        """Add default callbacks for basic functionality."""
        self.add_callback(PerformanceMonitorCallback())
        self.add_callback(SafetyMonitorCallback())
    
    def on_training_start(self, state: CallbackState) -> None:
        """Call on_training_start for all enabled callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_training_start(state)
                except Exception as e:
                    self.logger.error(f"Error in {callback.name}.on_training_start: {e}")
    
    def on_training_end(self, state: CallbackState) -> None:
        """Call on_training_end for all enabled callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_training_end(state)
                except Exception as e:
                    self.logger.error(f"Error in {callback.name}.on_training_end: {e}")
    
    def on_iteration_start(self, state: CallbackState) -> None:
        """Call on_iteration_start for all enabled callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_iteration_start(state)
                except Exception as e:
                    self.logger.error(f"Error in {callback.name}.on_iteration_start: {e}")
    
    def on_iteration_end(self, state: CallbackState) -> bool:
        """Call on_iteration_end for all enabled callbacks. Return True if any callback requests stopping."""
        should_stop = False
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    if callback.on_iteration_end(state):
                        self.logger.info(f"Training stop requested by {callback.name}")
                        should_stop = True
                except Exception as e:
                    self.logger.error(f"Error in {callback.name}.on_iteration_end: {e}")
        
        return should_stop
    
    def on_evaluation_start(self, state: CallbackState) -> None:
        """Call on_evaluation_start for all enabled callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_evaluation_start(state)
                except Exception as e:
                    self.logger.error(f"Error in {callback.name}.on_evaluation_start: {e}")
    
    def on_evaluation_end(self, state: CallbackState) -> None:
        """Call on_evaluation_end for all enabled callbacks."""
        for callback in self.callbacks:
            if callback.enabled:
                try:
                    callback.on_evaluation_end(state)
                except Exception as e:
                    self.logger.error(f"Error in {callback.name}.on_evaluation_end: {e}")