"""
Experiment tracking and logging for Safe RL training.

This module provides comprehensive experiment tracking capabilities with support
for MLflow, Weights & Biases, and local logging for CPO training experiments.
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import torch

from .config import TrainingConfig, ExperimentConfig


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics and metadata."""
    
    # Training metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    constraint_cost: float = 0.0
    kl_divergence: float = 0.0
    entropy: float = 0.0
    
    # Performance metrics
    episode_return: float = 0.0
    episode_length: float = 0.0
    success_rate: float = 0.0
    completion_time: float = 0.0
    
    # Safety metrics
    constraint_violations: int = 0
    safety_margin: float = 0.0
    violation_severity: float = 0.0
    
    # Human-robot interaction metrics
    human_effort: float = 0.0
    robot_assistance: float = 0.0
    collaboration_efficiency: float = 0.0
    user_comfort: float = 0.0
    
    # Computational metrics
    training_time: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format."""
        return asdict(self)
    
    def update(self, **kwargs) -> None:
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking backends."""
    
    def __init__(self, experiment_name: str, config: ExperimentConfig):
        self.experiment_name = experiment_name
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @abstractmethod
    def start_experiment(self, training_config: TrainingConfig) -> None:
        """Start a new experiment run."""
        pass
        
    @abstractmethod
    def log_metrics(self, metrics: ExperimentMetrics, step: int) -> None:
        """Log metrics for the current step."""
        pass
        
    @abstractmethod
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters for the experiment."""
        pass
        
    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log an artifact (model, plot, etc.)."""
        pass
        
    @abstractmethod
    def log_plot(self, figure, name: str) -> None:
        """Log a matplotlib/plotly figure."""
        pass
        
    @abstractmethod
    def end_experiment(self, status: str = "FINISHED") -> None:
        """End the current experiment run."""
        pass
        
    @abstractmethod
    def get_experiment_id(self) -> str:
        """Get the current experiment ID."""
        pass


class MLflowTracker(ExperimentTracker):
    """MLflow-based experiment tracking."""
    
    def __init__(self, experiment_name: str, config: ExperimentConfig):
        super().__init__(experiment_name, config)
        self.run_id = None
        self.experiment_id = None
        
        try:
            import mlflow
            import mlflow.pytorch
            self.mlflow = mlflow
            self._setup_mlflow()
        except ImportError:
            self.logger.error("MLflow not installed. Run: pip install mlflow")
            raise
    
    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        if self.config.mlflow_tracking_uri:
            self.mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        
        # Get or create experiment
        try:
            experiment = self.mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = self.mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=self.config.artifact_location
                )
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            self.logger.warning(f"MLflow setup warning: {e}")
    
    def start_experiment(self, training_config: TrainingConfig) -> None:
        """Start MLflow run."""
        self.run = self.mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"{self.experiment_name}_{training_config.seed}"
        )
        self.run_id = self.run.info.run_id
        
        # Log training configuration
        config_dict = asdict(training_config)
        self._flatten_and_log_params(config_dict, "config")
        
        # Set tags
        self.mlflow.set_tags({
            "algorithm": "CPO",
            "environment": training_config.environment_name,
            "framework": "PyTorch",
            "phase": "4_training_pipeline"
        })
    
    def _flatten_and_log_params(self, d: Dict[str, Any], prefix: str = "") -> None:
        """Flatten nested dictionary and log as parameters."""
        for key, value in d.items():
            param_name = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_and_log_params(value, param_name)
            elif isinstance(value, (int, float, str, bool)):
                try:
                    self.mlflow.log_param(param_name, value)
                except Exception:
                    # Skip if parameter already exists
                    pass
    
    def log_metrics(self, metrics: ExperimentMetrics, step: int) -> None:
        """Log metrics to MLflow."""
        metrics_dict = metrics.to_dict()
        try:
            self.mlflow.log_metrics(metrics_dict, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        try:
            self.mlflow.log_params(hyperparams)
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log artifact to MLflow."""
        try:
            if artifact_type == "model" and artifact_path.endswith('.pth'):
                # Special handling for PyTorch models
                self.mlflow.pytorch.log_model(
                    torch.load(artifact_path), 
                    "model"
                )
            else:
                self.mlflow.log_artifact(artifact_path)
        except Exception as e:
            self.logger.warning(f"Failed to log artifact: {e}")
    
    def log_plot(self, figure, name: str) -> None:
        """Log plot to MLflow."""
        try:
            self.mlflow.log_figure(figure, name)
        except Exception as e:
            self.logger.warning(f"Failed to log plot: {e}")
    
    def end_experiment(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        if hasattr(self, 'run') and self.run:
            self.mlflow.end_run(status=status)
    
    def get_experiment_id(self) -> str:
        """Get MLflow experiment ID."""
        return self.run_id if self.run_id else "unknown"


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracking."""
    
    def __init__(self, experiment_name: str, config: ExperimentConfig):
        super().__init__(experiment_name, config)
        self.run = None
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            self.logger.error("Weights & Biases not installed. Run: pip install wandb")
            raise
    
    def start_experiment(self, training_config: TrainingConfig) -> None:
        """Start W&B run."""
        self.run = self.wandb.init(
            project=self.config.wandb_project or self.experiment_name,
            entity=self.config.wandb_entity,
            name=f"{self.experiment_name}_{training_config.seed}",
            config=asdict(training_config),
            tags=["CPO", "SafeRL", "HumanRobotSharedControl"],
            notes=f"Training CPO on {training_config.environment_name}"
        )
    
    def log_metrics(self, metrics: ExperimentMetrics, step: int) -> None:
        """Log metrics to W&B."""
        metrics_dict = metrics.to_dict()
        try:
            self.wandb.log(metrics_dict, step=step)
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B."""
        try:
            if self.run:
                self.run.config.update(hyperparams)
        except Exception as e:
            self.logger.warning(f"Failed to log hyperparameters: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log artifact to W&B."""
        try:
            artifact = self.wandb.Artifact(
                name=f"{artifact_type}_{self.run.id}",
                type=artifact_type
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
        except Exception as e:
            self.logger.warning(f"Failed to log artifact: {e}")
    
    def log_plot(self, figure, name: str) -> None:
        """Log plot to W&B."""
        try:
            self.wandb.log({name: self.wandb.Image(figure)})
        except Exception as e:
            self.logger.warning(f"Failed to log plot: {e}")
    
    def end_experiment(self, status: str = "finished") -> None:
        """End W&B run."""
        if self.run:
            self.run.finish(exit_code=0 if status == "finished" else 1)
    
    def get_experiment_id(self) -> str:
        """Get W&B run ID."""
        return self.run.id if self.run else "unknown"


class LocalTracker(ExperimentTracker):
    """Local file-based experiment tracking."""
    
    def __init__(self, experiment_name: str, config: ExperimentConfig):
        super().__init__(experiment_name, config)
        self.experiment_dir = Path(config.artifact_location) / experiment_name
        self.run_dir = None
        self.metrics_log = []
        
    def start_experiment(self, training_config: TrainingConfig) -> None:
        """Start local experiment tracking."""
        # Create unique run directory
        run_name = f"run_{training_config.seed}_{hash(str(training_config)) % 10000}"
        self.run_dir = self.experiment_dir / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training configuration
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(training_config), f, indent=2, default=str)
        
        # Initialize metrics log
        self.metrics_log = []
    
    def log_metrics(self, metrics: ExperimentMetrics, step: int) -> None:
        """Log metrics to local file."""
        metrics_dict = metrics.to_dict()
        metrics_dict['step'] = step
        self.metrics_log.append(metrics_dict)
        
        # Save metrics incrementally
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to local file."""
        hyperparams_path = self.run_dir / "hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2, default=str)
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Copy artifact to local experiment directory."""
        import shutil
        
        artifact_src = Path(artifact_path)
        artifact_dst = self.run_dir / "artifacts" / artifact_type / artifact_src.name
        artifact_dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(artifact_src, artifact_dst)
        except Exception as e:
            self.logger.warning(f"Failed to copy artifact: {e}")
    
    def log_plot(self, figure, name: str) -> None:
        """Save plot to local file."""
        plots_dir = self.run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        try:
            figure.savefig(plots_dir / f"{name}.png", dpi=150, bbox_inches='tight')
            figure.savefig(plots_dir / f"{name}.pdf", bbox_inches='tight')
        except Exception as e:
            self.logger.warning(f"Failed to save plot: {e}")
    
    def end_experiment(self, status: str = "finished") -> None:
        """Finalize local experiment."""
        # Save final status
        status_path = self.run_dir / "status.txt"
        with open(status_path, 'w') as f:
            f.write(f"Status: {status}\n")
    
    def get_experiment_id(self) -> str:
        """Get local experiment ID."""
        return self.run_dir.name if self.run_dir else "unknown"


class CompositeTracker(ExperimentTracker):
    """Composite tracker that can log to multiple backends simultaneously."""
    
    def __init__(self, experiment_name: str, config: ExperimentConfig):
        super().__init__(experiment_name, config)
        self.trackers = []
        
        # Initialize requested trackers
        if config.use_mlflow:
            try:
                self.trackers.append(MLflowTracker(experiment_name, config))
            except ImportError:
                self.logger.warning("MLflow not available, skipping MLflow tracking")
        
        if config.use_wandb:
            try:
                self.trackers.append(WandbTracker(experiment_name, config))
            except ImportError:
                self.logger.warning("W&B not available, skipping W&B tracking")
        
        if config.use_local or not self.trackers:
            # Always include local tracking as fallback
            self.trackers.append(LocalTracker(experiment_name, config))
    
    def start_experiment(self, training_config: TrainingConfig) -> None:
        """Start experiment on all trackers."""
        for tracker in self.trackers:
            try:
                tracker.start_experiment(training_config)
            except Exception as e:
                self.logger.warning(f"Failed to start {type(tracker).__name__}: {e}")
    
    def log_metrics(self, metrics: ExperimentMetrics, step: int) -> None:
        """Log metrics to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_metrics(metrics, step)
            except Exception as e:
                self.logger.warning(f"Failed to log metrics to {type(tracker).__name__}: {e}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_hyperparameters(hyperparams)
            except Exception as e:
                self.logger.warning(f"Failed to log hyperparams to {type(tracker).__name__}: {e}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log artifact to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_artifact(artifact_path, artifact_type)
            except Exception as e:
                self.logger.warning(f"Failed to log artifact to {type(tracker).__name__}: {e}")
    
    def log_plot(self, figure, name: str) -> None:
        """Log plot to all trackers."""
        for tracker in self.trackers:
            try:
                tracker.log_plot(figure, name)
            except Exception as e:
                self.logger.warning(f"Failed to log plot to {type(tracker).__name__}: {e}")
    
    def end_experiment(self, status: str = "finished") -> None:
        """End experiment on all trackers."""
        for tracker in self.trackers:
            try:
                tracker.end_experiment(status)
            except Exception as e:
                self.logger.warning(f"Failed to end {type(tracker).__name__}: {e}")
    
    def get_experiment_id(self) -> str:
        """Get experiment ID from primary tracker."""
        if self.trackers:
            return self.trackers[0].get_experiment_id()
        return "unknown"


class ExperimentManager:
    """High-level experiment management with analysis capabilities."""
    
    def __init__(self, base_experiment_name: str = "safe_rl_cpo"):
        self.base_experiment_name = base_experiment_name
        self.logger = logging.getLogger(__name__)
        
    def create_tracker(self, config: ExperimentConfig, 
                      experiment_suffix: str = "") -> ExperimentTracker:
        """Create appropriate tracker based on configuration."""
        experiment_name = f"{self.base_experiment_name}{experiment_suffix}"
        
        if config.use_mlflow or config.use_wandb or config.use_local:
            return CompositeTracker(experiment_name, config)
        else:
            # Default to local tracking
            config.use_local = True
            return LocalTracker(experiment_name, config)
    
    def compare_experiments(self, experiment_paths: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments and generate analysis."""
        comparison_results = {
            'experiments': [],
            'best_performance': {},
            'safety_analysis': {},
            'convergence_analysis': {}
        }
        
        for exp_path in experiment_paths:
            try:
                exp_data = self._load_experiment_data(exp_path)
                if exp_data:
                    comparison_results['experiments'].append(exp_data)
            except Exception as e:
                self.logger.warning(f"Failed to load experiment {exp_path}: {e}")
        
        if comparison_results['experiments']:
            self._analyze_performance(comparison_results)
            self._analyze_safety(comparison_results)
            self._analyze_convergence(comparison_results)
        
        return comparison_results
    
    def _load_experiment_data(self, exp_path: str) -> Optional[Dict[str, Any]]:
        """Load experiment data from local directory."""
        exp_dir = Path(exp_path)
        if not exp_dir.exists():
            return None
        
        # Load configuration
        config_path = exp_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        
        # Load metrics
        metrics_path = exp_dir / "metrics.json"
        metrics = []
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        
        return {
            'path': str(exp_path),
            'config': config,
            'metrics': metrics
        }
    
    def _analyze_performance(self, comparison_results: Dict[str, Any]) -> None:
        """Analyze performance across experiments."""
        experiments = comparison_results['experiments']
        
        best_return = float('-inf')
        best_success_rate = 0.0
        best_experiment = None
        
        for exp in experiments:
            metrics = exp['metrics']
            if metrics:
                final_return = metrics[-1].get('episode_return', 0)
                final_success_rate = metrics[-1].get('success_rate', 0)
                
                if final_return > best_return:
                    best_return = final_return
                    best_experiment = exp['path']
                
                if final_success_rate > best_success_rate:
                    best_success_rate = final_success_rate
        
        comparison_results['best_performance'] = {
            'best_return': best_return,
            'best_success_rate': best_success_rate,
            'best_experiment': best_experiment
        }
    
    def _analyze_safety(self, comparison_results: Dict[str, Any]) -> None:
        """Analyze safety metrics across experiments."""
        experiments = comparison_results['experiments']
        
        min_violations = float('inf')
        best_safety_margin = 0.0
        safest_experiment = None
        
        for exp in experiments:
            metrics = exp['metrics']
            if metrics:
                total_violations = sum(m.get('constraint_violations', 0) for m in metrics)
                avg_safety_margin = np.mean([m.get('safety_margin', 0) for m in metrics])
                
                if total_violations < min_violations:
                    min_violations = total_violations
                    safest_experiment = exp['path']
                
                if avg_safety_margin > best_safety_margin:
                    best_safety_margin = avg_safety_margin
        
        comparison_results['safety_analysis'] = {
            'min_violations': min_violations if min_violations != float('inf') else 0,
            'best_safety_margin': best_safety_margin,
            'safest_experiment': safest_experiment
        }
    
    def _analyze_convergence(self, comparison_results: Dict[str, Any]) -> None:
        """Analyze convergence properties across experiments."""
        experiments = comparison_results['experiments']
        
        convergence_analysis = {}
        
        for exp in experiments:
            metrics = exp['metrics']
            if len(metrics) > 10:  # Need sufficient data for convergence analysis
                returns = [m.get('episode_return', 0) for m in metrics[-50:]]  # Last 50 episodes
                convergence_variance = np.var(returns) if returns else float('inf')
                
                convergence_analysis[exp['path']] = {
                    'final_variance': convergence_variance,
                    'converged': convergence_variance < 0.1  # Threshold for convergence
                }
        
        comparison_results['convergence_analysis'] = convergence_analysis


def create_experiment_tracker(config: ExperimentConfig, 
                            experiment_name: str = "safe_rl_cpo") -> ExperimentTracker:
    """Factory function to create appropriate experiment tracker."""
    manager = ExperimentManager()
    return manager.create_tracker(config, experiment_name)