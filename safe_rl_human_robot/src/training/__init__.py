"""
Training pipeline and hyperparameter optimization for Safe RL.

This package provides a complete training infrastructure for CPO algorithms
with hyperparameter optimization, experiment tracking, and distributed training.
"""

from .config import (
    TrainingConfig, NetworkConfig, OptimizationConfig, 
    EvaluationConfig, ExperimentConfig
)
from .trainer import CPOTrainer, TrainingResults
from .hyperopt import HyperparameterOptimizer, OptimizationStudy
from .experiment_tracker import ExperimentTracker, MLflowTracker, WandbTracker
from .callbacks import (
    CallbackManager, EarlyStopping, ModelCheckpoint, 
    LearningRateScheduler, SafetyMonitorCallback
)
from .evaluation import EvaluationManager, EvaluationMetrics
from .schedulers import LearningRateScheduler, ConstraintScheduler
from .distributed import DistributedTrainer

__all__ = [
    # Configuration
    "TrainingConfig", "NetworkConfig", "OptimizationConfig",
    "EvaluationConfig", "ExperimentConfig",
    
    # Training
    "CPOTrainer", "TrainingResults",
    
    # Hyperparameter optimization
    "HyperparameterOptimizer", "OptimizationStudy",
    
    # Experiment tracking
    "ExperimentTracker", "MLflowTracker", "WandbTracker",
    
    # Callbacks and scheduling
    "CallbackManager", "EarlyStopping", "ModelCheckpoint",
    "LearningRateScheduler", "SafetyMonitorCallback",
    "ConstraintScheduler",
    
    # Evaluation
    "EvaluationManager", "EvaluationMetrics",
    
    # Distributed training
    "DistributedTrainer"
]