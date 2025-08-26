"""
Hyperparameter optimization for CPO using Optuna.

This module provides Bayesian optimization capabilities for CPO hyperparameters
with multi-objective optimization, pruning strategies, and comprehensive 
search space definitions.
"""

import os
import time
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
import json

import numpy as np
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from optuna.study import StudyDirection
import plotly.graph_objects as go
import plotly.express as px

from .config import TrainingConfig, ActivationFunction, OptimizerType
from .trainer import CPOTrainer, TrainingResults

logger = logging.getLogger(__name__)


@dataclass
class OptimizationObjective:
    """Definition of an optimization objective."""
    name: str
    direction: str = "maximize"  # "maximize" or "minimize"
    weight: float = 1.0
    threshold: Optional[float] = None  # Early stopping threshold
    extract_function: Optional[Callable] = None  # Function to extract metric from results


@dataclass
class SearchSpace:
    """Definition of hyperparameter search space."""
    # CPO Parameters
    policy_lr: Tuple[float, float, str] = (1e-5, 1e-2, "log-uniform")
    value_lr: Tuple[float, float, str] = (1e-5, 1e-2, "log-uniform")
    trust_region_radius: Tuple[float, float, str] = (0.001, 0.1, "log-uniform")
    constraint_threshold: Tuple[float, float, str] = (0.01, 0.5, "uniform")
    gamma: Tuple[float, float, str] = (0.95, 0.999, "uniform")
    lambda_gae: Tuple[float, float, str] = (0.9, 0.99, "uniform")
    entropy_coefficient: Tuple[float, float, str] = (0.001, 0.1, "log-uniform")
    
    # Network Architecture
    policy_hidden_size: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    policy_num_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    value_hidden_size: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    value_num_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Activation functions
    policy_activation: List[str] = field(default_factory=lambda: ["relu", "tanh", "elu"])
    value_activation: List[str] = field(default_factory=lambda: ["relu", "tanh", "elu"])
    
    # Training parameters
    batch_size: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    rollout_length: List[int] = field(default_factory=lambda: [1000, 2000, 4000])
    
    # Optimization parameters
    optimizer_type: List[str] = field(default_factory=lambda: ["adam", "adamw"])
    cg_iterations: List[int] = field(default_factory=lambda: [5, 10, 15, 20])
    value_train_iterations: List[int] = field(default_factory=lambda: [3, 5, 10])
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        hyperparams = {}
        
        # CPO parameters
        hyperparams['policy_lr'] = self._suggest_value(trial, 'policy_lr', self.policy_lr)
        hyperparams['value_lr'] = self._suggest_value(trial, 'value_lr', self.value_lr)
        hyperparams['trust_region_radius'] = self._suggest_value(trial, 'trust_region_radius', self.trust_region_radius)
        hyperparams['constraint_threshold'] = self._suggest_value(trial, 'constraint_threshold', self.constraint_threshold)
        hyperparams['gamma'] = self._suggest_value(trial, 'gamma', self.gamma)
        hyperparams['lambda_gae'] = self._suggest_value(trial, 'lambda_gae', self.lambda_gae)
        hyperparams['entropy_coefficient'] = self._suggest_value(trial, 'entropy_coefficient', self.entropy_coefficient)
        
        # Network architecture
        hyperparams['policy_hidden_size'] = trial.suggest_categorical('policy_hidden_size', self.policy_hidden_size)
        hyperparams['policy_num_layers'] = trial.suggest_categorical('policy_num_layers', self.policy_num_layers)
        hyperparams['value_hidden_size'] = trial.suggest_categorical('value_hidden_size', self.value_hidden_size)
        hyperparams['value_num_layers'] = trial.suggest_categorical('value_num_layers', self.value_num_layers)
        
        # Activation functions
        hyperparams['policy_activation'] = trial.suggest_categorical('policy_activation', self.policy_activation)
        hyperparams['value_activation'] = trial.suggest_categorical('value_activation', self.value_activation)
        
        # Training parameters
        hyperparams['batch_size'] = trial.suggest_categorical('batch_size', self.batch_size)
        hyperparams['rollout_length'] = trial.suggest_categorical('rollout_length', self.rollout_length)
        
        # Optimization parameters
        hyperparams['optimizer_type'] = trial.suggest_categorical('optimizer_type', self.optimizer_type)
        hyperparams['cg_iterations'] = trial.suggest_categorical('cg_iterations', self.cg_iterations)
        hyperparams['value_train_iterations'] = trial.suggest_categorical('value_train_iterations', self.value_train_iterations)
        
        return hyperparams
    
    def _suggest_value(self, trial: optuna.Trial, name: str, spec: Tuple[float, float, str]):
        """Suggest value based on distribution type."""
        low, high, distribution = spec
        
        if distribution == "log-uniform":
            return trial.suggest_loguniform(name, low, high)
        elif distribution == "uniform":
            return trial.suggest_uniform(name, low, high)
        elif distribution == "int":
            return trial.suggest_int(name, int(low), int(high))
        else:
            raise ValueError(f"Unknown distribution type: {distribution}")


@dataclass 
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    # Study configuration
    study_name: str = "cpo_optimization"
    storage: Optional[str] = None  # Database URL for study storage
    n_trials: int = 100
    timeout: Optional[float] = None  # Timeout in seconds
    
    # Sampler configuration
    sampler_name: str = "TPE"  # "TPE", "CmaEs", "Random"
    sampler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Pruner configuration
    pruner_name: str = "Median"  # "Median", "SuccessiveHalving", "Hyperband"
    pruner_params: Dict[str, Any] = field(default_factory=dict)
    
    # Multi-objective optimization
    objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective("reward", "maximize", weight=1.0),
        OptimizationObjective("safety", "maximize", weight=0.5)  # 1 - violation_rate
    ])
    
    # Early stopping
    enable_pruning: bool = True
    pruning_warmup_trials: int = 10
    pruning_warmup_steps: int = 50
    
    # Training budget per trial
    max_iterations_per_trial: int = 200
    evaluation_frequency: int = 10
    early_stopping_patience: int = 20
    
    # Parallel optimization
    n_jobs: int = 1
    
    # Results and visualization
    save_results: bool = True
    results_dir: str = "optimization_results"
    plot_optimization_history: bool = True
    plot_parameter_importance: bool = True


class MultiObjectiveStudy:
    """Multi-objective optimization study wrapper."""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
        self.objective_values = []
        
    def add_trial_result(self, trial_results: TrainingResults):
        """Add results from a completed trial."""
        objective_values = []
        
        for obj in self.objectives:
            if obj.name == "reward":
                value = trial_results.best_reward
            elif obj.name == "safety":
                value = 1.0 - trial_results.violation_rate
            elif obj.name == "success_rate":
                value = trial_results.best_success_rate
            elif obj.name == "sample_efficiency":
                # Reward per timestep
                value = trial_results.best_reward / max(trial_results.total_timesteps, 1)
            elif obj.name == "training_time":
                value = -trial_results.training_time  # Minimize time (negative for maximization)
            elif obj.extract_function:
                value = obj.extract_function(trial_results)
            else:
                logger.warning(f"Unknown objective: {obj.name}")
                value = 0.0
            
            # Apply direction
            if obj.direction == "minimize":
                value = -value
            
            objective_values.append(value * obj.weight)
        
        self.objective_values.append(objective_values)
        return objective_values
    
    def compute_composite_score(self, objective_values: List[float]) -> float:
        """Compute composite score from multiple objectives."""
        # Simple weighted sum for now
        # Could be extended to Pareto optimization
        return sum(objective_values)
    
    def get_pareto_front(self) -> List[int]:
        """Get Pareto front trial indices."""
        if not self.objective_values:
            return []
        
        objective_values = np.array(self.objective_values)
        pareto_front = []
        
        for i, candidate in enumerate(objective_values):
            is_pareto = True
            for j, other in enumerate(objective_values):
                if i == j:
                    continue
                
                # Check if 'other' dominates 'candidate'
                if np.all(other >= candidate) and np.any(other > candidate):
                    is_pareto = False
                    break
            
            if is_pareto:
                pareto_front.append(i)
        
        return pareto_front


class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimizer for CPO using Optuna.
    
    Supports multi-objective optimization, advanced pruning strategies,
    and comprehensive result analysis.
    """
    
    def __init__(self, 
                 base_config: TrainingConfig,
                 search_space: SearchSpace,
                 optimization_config: OptimizationConfig):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            base_config: Base training configuration
            search_space: Search space definition
            optimization_config: Optimization configuration
        """
        self.base_config = base_config
        self.search_space = search_space
        self.opt_config = optimization_config
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Create results directory
        self.results_dir = Path(optimization_config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize study
        self.study = None
        self.multi_objective_study = MultiObjectiveStudy(optimization_config.objectives)
        
        # Track best results
        self.best_trial = None
        self.best_score = float('-inf')
        self.all_trial_results = []
        
        self.logger.info(f"HyperparameterOptimizer initialized: {optimization_config.n_trials} trials")
    
    def optimize(self) -> 'OptimizationStudy':
        """
        Run hyperparameter optimization.
        
        Returns:
            Optimization study results
        """
        self.logger.info("Starting hyperparameter optimization...")
        
        # Create Optuna study
        self.study = self._create_study()
        
        try:
            # Run optimization
            self.study.optimize(
                self._objective_function,
                n_trials=self.opt_config.n_trials,
                timeout=self.opt_config.timeout,
                n_jobs=self.opt_config.n_jobs,
                catch=(Exception,)  # Catch all exceptions to continue optimization
            )
            
            # Analyze results
            optimization_study = self._analyze_results()
            
            # Save results
            if self.opt_config.save_results:
                self._save_results(optimization_study)
            
            # Generate visualizations
            if self.opt_config.plot_optimization_history:
                self._plot_optimization_history()
            
            if self.opt_config.plot_parameter_importance:
                self._plot_parameter_importance()
            
            self.logger.info("Hyperparameter optimization completed successfully")
            return optimization_study
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}", exc_info=True)
            raise
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner."""
        # Create sampler
        if self.opt_config.sampler_name.lower() == "tpe":
            sampler = TPESampler(**self.opt_config.sampler_params)
        elif self.opt_config.sampler_name.lower() == "cmaes":
            sampler = CmaEsSampler(**self.opt_config.sampler_params)
        elif self.opt_config.sampler_name.lower() == "random":
            sampler = RandomSampler(**self.opt_config.sampler_params)
        else:
            self.logger.warning(f"Unknown sampler: {self.opt_config.sampler_name}, using TPE")
            sampler = TPESampler()
        
        # Create pruner
        if self.opt_config.enable_pruning:
            if self.opt_config.pruner_name.lower() == "median":
                pruner = MedianPruner(
                    n_startup_trials=self.opt_config.pruning_warmup_trials,
                    n_warmup_steps=self.opt_config.pruning_warmup_steps,
                    **self.opt_config.pruner_params
                )
            elif self.opt_config.pruner_name.lower() == "successivehalving":
                pruner = SuccessiveHalvingPruner(**self.opt_config.pruner_params)
            elif self.opt_config.pruner_name.lower() == "hyperband":
                pruner = HyperbandPruner(**self.opt_config.pruner_params)
            else:
                self.logger.warning(f"Unknown pruner: {self.opt_config.pruner_name}, using Median")
                pruner = MedianPruner(
                    n_startup_trials=self.opt_config.pruning_warmup_trials,
                    n_warmup_steps=self.opt_config.pruning_warmup_steps
                )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=self.opt_config.study_name,
            storage=self.opt_config.storage,
            direction="maximize",  # We'll handle multi-objective manually
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )
        
        return study
    
    def _objective_function(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Objective value to optimize
        """
        try:
            # Suggest hyperparameters
            hyperparams = self.search_space.suggest_hyperparameters(trial)
            
            # Create training configuration with suggested hyperparameters
            trial_config = self._create_trial_config(hyperparams)
            
            # Train model with suggested hyperparameters
            trainer = CPOTrainer(trial_config)
            training_results = trainer.train()
            
            # Store results
            self.all_trial_results.append(training_results)
            
            # Compute multi-objective score
            objective_values = self.multi_objective_study.add_trial_result(training_results)
            composite_score = self.multi_objective_study.compute_composite_score(objective_values)
            
            # Track best trial
            if composite_score > self.best_score:
                self.best_score = composite_score
                self.best_trial = {
                    'trial_number': trial.number,
                    'hyperparams': hyperparams,
                    'score': composite_score,
                    'objective_values': objective_values,
                    'results': training_results
                }
            
            # Report intermediate values for pruning
            self._report_intermediate_values(trial, training_results)
            
            self.logger.info(f"Trial {trial.number}: Score={composite_score:.4f}, "
                           f"Reward={training_results.best_reward:.2f}, "
                           f"Safety={1.0-training_results.violation_rate:.3f}")
            
            return composite_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
            # Return a very low score for failed trials
            return float('-inf')
    
    def _create_trial_config(self, hyperparams: Dict[str, Any]) -> TrainingConfig:
        """Create training configuration for trial."""
        # Copy base configuration
        trial_config = TrainingConfig.from_dict(self.base_config.to_dict())
        
        # Update with hyperparameters
        trial_config.update_from_hyperparameters(hyperparams)
        
        # Set trial-specific settings
        trial_config.max_iterations = self.opt_config.max_iterations_per_trial
        trial_config.evaluation.eval_frequency = self.opt_config.evaluation_frequency
        trial_config.evaluation.early_stopping_patience = self.opt_config.early_stopping_patience
        
        # Disable expensive logging for trials
        trial_config.experiment.log_weights = False
        trial_config.experiment.log_activations = False
        trial_config.experiment.log_evaluation_videos = False
        
        return trial_config
    
    def _report_intermediate_values(self, trial: optuna.Trial, results: TrainingResults):
        """Report intermediate values for pruning."""
        if not self.opt_config.enable_pruning:
            return
        
        # Report evaluation rewards for pruning
        for i, (iteration, reward) in enumerate(zip(results.eval_iterations, results.eval_rewards)):
            trial.report(reward, step=iteration)
            
            # Check if trial should be pruned
            if trial.should_prune():
                self.logger.info(f"Trial {trial.number} pruned at iteration {iteration}")
                raise optuna.TrialPruned()
    
    def _analyze_results(self) -> 'OptimizationStudy':
        """Analyze optimization results."""
        if not self.study.trials:
            raise ValueError("No completed trials found")
        
        # Get completed trials
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            raise ValueError("No successfully completed trials found")
        
        # Find best trial
        best_trial = self.study.best_trial
        
        # Compute statistics
        all_values = [t.value for t in completed_trials if t.value is not None]
        
        statistics = {
            'n_trials': len(self.study.trials),
            'n_completed': len(completed_trials),
            'n_pruned': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            'n_failed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            'best_value': best_trial.value if best_trial else None,
            'mean_value': np.mean(all_values) if all_values else None,
            'std_value': np.std(all_values) if all_values else None,
            'median_value': np.median(all_values) if all_values else None
        }
        
        # Parameter importance
        try:
            param_importance = optuna.importance.get_param_importances(self.study)
        except Exception as e:
            self.logger.warning(f"Failed to compute parameter importance: {e}")
            param_importance = {}
        
        # Pareto front for multi-objective
        pareto_front_indices = self.multi_objective_study.get_pareto_front()
        pareto_front_trials = [completed_trials[i] for i in pareto_front_indices if i < len(completed_trials)]
        
        return OptimizationStudy(
            study=self.study,
            best_trial=best_trial,
            best_hyperparams=self.best_trial['hyperparams'] if self.best_trial else {},
            statistics=statistics,
            parameter_importance=param_importance,
            pareto_front=pareto_front_trials,
            all_results=self.all_trial_results,
            objectives=self.opt_config.objectives
        )
    
    def _save_results(self, optimization_study: 'OptimizationStudy'):
        """Save optimization results."""
        try:
            # Save study object
            study_path = self.results_dir / "study.pkl"
            with open(study_path, 'wb') as f:
                pickle.dump(self.study, f)
            
            # Save detailed results
            results_path = self.results_dir / "optimization_results.json"
            results_data = {
                'study_name': self.opt_config.study_name,
                'n_trials': self.opt_config.n_trials,
                'objectives': [obj.name for obj in self.opt_config.objectives],
                'best_trial': {
                    'number': optimization_study.best_trial.number if optimization_study.best_trial else None,
                    'value': optimization_study.best_trial.value if optimization_study.best_trial else None,
                    'params': optimization_study.best_trial.params if optimization_study.best_trial else {}
                },
                'statistics': optimization_study.statistics,
                'parameter_importance': optimization_study.parameter_importance,
                'search_space': {
                    'policy_lr': self.search_space.policy_lr,
                    'trust_region_radius': self.search_space.trust_region_radius,
                    'constraint_threshold': self.search_space.constraint_threshold,
                    # Add other search space parameters as needed
                }
            }
            
            with open(results_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Save best configuration
            if self.best_trial:
                best_config = self._create_trial_config(self.best_trial['hyperparams'])
                best_config_path = self.results_dir / "best_config.json"
                best_config.save(best_config_path)
            
            self.logger.info(f"Results saved to {self.results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}", exc_info=True)
    
    def _plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            # Get trial values and numbers
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if not completed_trials:
                return
            
            trial_numbers = [t.number for t in completed_trials]
            trial_values = [t.value for t in completed_trials]
            
            # Plot optimization history
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trial_numbers,
                y=trial_values,
                mode='markers+lines',
                name='Trial Values',
                line=dict(color='blue', width=1),
                marker=dict(color='blue', size=6)
            ))
            
            # Add best value line
            best_values = []
            current_best = float('-inf')
            for value in trial_values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)
            
            fig.add_trace(go.Scatter(
                x=trial_numbers,
                y=best_values,
                mode='lines',
                name='Best Value',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Hyperparameter Optimization History',
                xaxis_title='Trial Number',
                yaxis_title='Objective Value',
                showlegend=True,
                width=800,
                height=500
            )
            
            # Save plot
            plot_path = self.results_dir / "optimization_history.html"
            plot(fig, filename=str(plot_path), auto_open=False)
            
            self.logger.info(f"Optimization history plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create optimization history plot: {e}")
    
    def _plot_parameter_importance(self):
        """Plot parameter importance."""
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            param_importance = optuna.importance.get_param_importances(self.study)
            
            if not param_importance:
                return
            
            # Sort parameters by importance
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
            param_names = [p[0] for p in sorted_params]
            importance_values = [p[1] for p in sorted_params]
            
            # Create bar plot
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_values,
                    y=param_names,
                    orientation='h',
                    marker=dict(color='skyblue')
                )
            ])
            
            fig.update_layout(
                title='Parameter Importance',
                xaxis_title='Importance',
                yaxis_title='Parameters',
                showlegend=False,
                width=800,
                height=max(400, len(param_names) * 30)
            )
            
            # Save plot
            plot_path = self.results_dir / "parameter_importance.html"
            plot(fig, filename=str(plot_path), auto_open=False)
            
            self.logger.info(f"Parameter importance plot saved: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create parameter importance plot: {e}")


@dataclass
class OptimizationStudy:
    """Results from hyperparameter optimization study."""
    study: optuna.Study
    best_trial: Optional[optuna.Trial]
    best_hyperparams: Dict[str, Any]
    statistics: Dict[str, Any]
    parameter_importance: Dict[str, float]
    pareto_front: List[optuna.Trial]
    all_results: List[TrainingResults]
    objectives: List[OptimizationObjective]
    
    def get_best_config(self, base_config: TrainingConfig) -> TrainingConfig:
        """Get training configuration with best hyperparameters."""
        if not self.best_hyperparams:
            return base_config
        
        config = TrainingConfig.from_dict(base_config.to_dict())
        config.update_from_hyperparameters(self.best_hyperparams)
        return config
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        
        print(f"Study name: {self.study.study_name}")
        print(f"Total trials: {self.statistics['n_trials']}")
        print(f"Completed trials: {self.statistics['n_completed']}")
        print(f"Pruned trials: {self.statistics['n_pruned']}")
        print(f"Failed trials: {self.statistics['n_failed']}")
        
        if self.best_trial:
            print(f"\nBest trial: #{self.best_trial.number}")
            print(f"Best value: {self.best_trial.value:.4f}")
            
            print("\nBest hyperparameters:")
            for param, value in self.best_hyperparams.items():
                print(f"  {param}: {value}")
        
        if self.parameter_importance:
            print("\nTop parameter importance:")
            sorted_params = sorted(self.parameter_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
            for param, importance in sorted_params[:5]:
                print(f"  {param}: {importance:.3f}")
        
        print("\nObjectives:")
        for obj in self.objectives:
            print(f"  {obj.name}: {obj.direction} (weight: {obj.weight})")
        
        print("="*60)


# Convenience functions
def create_default_search_space() -> SearchSpace:
    """Create default search space for CPO optimization."""
    return SearchSpace()


def create_default_optimization_config(n_trials: int = 100) -> OptimizationConfig:
    """Create default optimization configuration."""
    return OptimizationConfig(n_trials=n_trials)


def optimize_cpo_hyperparameters(base_config: TrainingConfig, 
                                n_trials: int = 100,
                                study_name: str = "cpo_optimization") -> OptimizationStudy:
    """
    Convenience function to optimize CPO hyperparameters.
    
    Args:
        base_config: Base training configuration
        n_trials: Number of optimization trials
        study_name: Name of the optimization study
        
    Returns:
        Optimization study results
    """
    search_space = create_default_search_space()
    opt_config = create_default_optimization_config(n_trials)
    opt_config.study_name = study_name
    
    optimizer = HyperparameterOptimizer(base_config, search_space, opt_config)
    return optimizer.optimize()