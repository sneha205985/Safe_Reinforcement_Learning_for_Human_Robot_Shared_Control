#!/usr/bin/env python3
"""
Hyperparameter optimization script for CPO training.

This script demonstrates how to use Optuna for automated hyperparameter
optimization of CPO algorithms on safe RL environments.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training components
from src.training.config import TrainingConfig, ExperimentConfig
from src.training.hyperopt import (
    HyperparameterOptimizer, OptimizationStudy, SearchSpace
)
from src.training.experiment_tracker import create_experiment_tracker

# Import environments
from src.environments.exoskeleton_env import ExoskeletonEnvironment
from src.environments.wheelchair_env import WheelchairEnvironment


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('hyperopt.log')
        ]
    )


def create_environment(env_type: str, **kwargs) -> object:
    """Create training environment."""
    if env_type == "exoskeleton":
        return ExoskeletonEnvironment(**kwargs)
    elif env_type == "wheelchair":
        return WheelchairEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def create_base_config(args: argparse.Namespace) -> TrainingConfig:
    """Create base training configuration for optimization."""
    return TrainingConfig(
        # Fixed parameters
        max_iterations=args.max_iterations,
        rollout_length=args.rollout_length,
        num_epochs=args.num_epochs,
        environment_name=args.env_type,
        seed=args.seed,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        
        # Parameters to be optimized will be overridden
        batch_size=64,  # Will be optimized
    )


def create_search_space(args: argparse.Namespace) -> SearchSpace:
    """Create search space for hyperparameter optimization."""
    search_space = SearchSpace()
    
    # Custom search space configuration if needed
    if args.custom_search_space:
        # Load custom search space from file or modify defaults
        if args.env_type == "exoskeleton":
            # Exoskeleton-specific search space adjustments
            search_space.policy_lr_range = (1e-5, 1e-2)
            search_space.constraint_threshold_range = (0.001, 0.1)
        elif args.env_type == "wheelchair":
            # Wheelchair-specific search space adjustments
            search_space.policy_lr_range = (1e-5, 5e-3)
            search_space.trust_region_radius_range = (0.005, 0.05)
    
    return search_space


def create_study_config(args: argparse.Namespace) -> dict:
    """Create Optuna study configuration."""
    # Pruner configuration
    if args.pruner == "median":
        pruner = MedianPruner(
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps
        )
    elif args.pruner == "none":
        pruner = None
    else:
        raise ValueError(f"Unknown pruner: {args.pruner}")
    
    # Sampler configuration
    if args.sampler == "tpe":
        sampler = TPESampler(
            n_startup_trials=args.n_startup_trials,
            seed=args.seed
        )
    elif args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")
    
    return {
        'sampler': sampler,
        'pruner': pruner,
        'direction': 'maximize' if args.optimize_metric in ['episode_return', 'success_rate'] else 'minimize'
    }


def objective_function(trial, base_config, environment, search_space, args):
    """Objective function for Optuna optimization."""
    logger = logging.getLogger(__name__)
    
    try:
        # Suggest hyperparameters
        hyperparams = search_space.suggest_hyperparameters(trial)
        
        # Create trial configuration
        trial_config = base_config.copy()
        trial_config.update_from_hyperparameters(hyperparams)
        trial_config.seed = args.seed + trial.number  # Different seed per trial
        
        # Create experiment tracker for this trial
        experiment_config = ExperimentConfig(
            use_local=True,
            artifact_location=os.path.join(args.save_dir, f"trial_{trial.number}")
        )
        
        experiment_tracker = create_experiment_tracker(
            experiment_config,
            f"trial_{trial.number}"
        )
        
        # Create trainer
        from src.training.trainer import CPOTrainer
        trainer = CPOTrainer(
            config=trial_config,
            environment=environment,
            experiment_tracker=experiment_tracker
        )
        
        # Run training with pruning support
        results = trainer.train_with_pruning(trial)
        
        # Extract optimization metric
        if args.optimize_metric == 'episode_return':
            score = results.final_performance.get('episode_return', 0.0)
        elif args.optimize_metric == 'success_rate':
            score = results.final_performance.get('success_rate', 0.0)
        elif args.optimize_metric == 'constraint_violations':
            score = -results.safety_metrics.get('total_violations', 0)  # Minimize violations
        elif args.optimize_metric == 'composite':
            # Multi-objective composite score
            performance = results.final_performance.get('episode_return', 0.0)
            safety = 1.0 - results.safety_metrics.get('violation_rate', 0.0)
            score = args.performance_weight * performance + args.safety_weight * safety
        else:
            raise ValueError(f"Unknown optimization metric: {args.optimize_metric}")
        
        # Log trial results
        trial.set_user_attr('final_performance', results.final_performance)
        trial.set_user_attr('safety_metrics', results.safety_metrics)
        trial.set_user_attr('training_time', results.training_metrics.get('training_time', 0.0))
        
        logger.info(f"Trial {trial.number} completed with score: {score:.4f}")
        
        return score
        
    except optuna.exceptions.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        # Return a very bad score instead of failing completely
        return float('-inf') if args.optimize_metric in ['episode_return', 'success_rate', 'composite'] else float('inf')


def analyze_results(study: optuna.Study, args: argparse.Namespace) -> None:
    """Analyze and visualize optimization results."""
    logger = logging.getLogger(__name__)
    
    # Print best trial
    best_trial = study.best_trial
    logger.info("="*50)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*50)
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best value: {best_trial.value:.4f}")
    
    logger.info("Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")
    
    # Save study
    study_path = os.path.join(args.save_dir, "optimization_study.pkl")
    optuna.storages.save_study(study, study_path)
    logger.info(f"Study saved to {study_path}")
    
    # Generate plots if requested
    if args.generate_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            plots_dir = os.path.join(args.save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Optimization history
            fig = optuna.visualization.matplotlib.plot_optimization_history(study)
            fig.savefig(os.path.join(plots_dir, "optimization_history.png"))
            plt.close(fig)
            
            # Parameter importances
            fig = optuna.visualization.matplotlib.plot_param_importances(study)
            fig.savefig(os.path.join(plots_dir, "param_importances.png"))
            plt.close(fig)
            
            # Parallel coordinate plot
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            fig.savefig(os.path.join(plots_dir, "parallel_coordinate.png"))
            plt.close(fig)
            
            # Contour plot (for 2D parameter space)
            if len(best_trial.params) >= 2:
                param_names = list(best_trial.params.keys())[:2]
                fig = optuna.visualization.matplotlib.plot_contour(
                    study, params=param_names
                )
                fig.savefig(os.path.join(plots_dir, "contour.png"))
                plt.close(fig)
            
            logger.info(f"Plots saved to {plots_dir}")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    # Save best configuration
    best_config = create_base_config(args)
    search_space = create_search_space(args)
    best_hyperparams = search_space.convert_trial_params_to_hyperparams(best_trial.params)
    best_config.update_from_hyperparameters(best_hyperparams)
    
    config_path = os.path.join(args.save_dir, "best_config.json")
    best_config.save_to_file(config_path)
    logger.info(f"Best configuration saved to {config_path}")


def main():
    """Main hyperparameter optimization function."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CPO")
    
    # Environment arguments
    parser.add_argument("--env-type", type=str, default="exoskeleton",
                       choices=["exoskeleton", "wheelchair"],
                       help="Type of environment to optimize on")
    
    # Optimization arguments
    parser.add_argument("--n-trials", type=int, default=100,
                       help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=None,
                       help="Optimization timeout in seconds")
    parser.add_argument("--optimize-metric", type=str, default="composite",
                       choices=["episode_return", "success_rate", "constraint_violations", "composite"],
                       help="Metric to optimize")
    parser.add_argument("--performance-weight", type=float, default=0.7,
                       help="Weight for performance in composite score")
    parser.add_argument("--safety-weight", type=float, default=0.3,
                       help="Weight for safety in composite score")
    
    # Study configuration
    parser.add_argument("--study-name", type=str, default="cpo_optimization",
                       help="Name for the optimization study")
    parser.add_argument("--sampler", type=str, default="tpe",
                       choices=["tpe", "random"],
                       help="Optuna sampler")
    parser.add_argument("--pruner", type=str, default="median",
                       choices=["median", "none"],
                       help="Optuna pruner")
    parser.add_argument("--n-startup-trials", type=int, default=10,
                       help="Number of startup trials for pruning")
    parser.add_argument("--n-warmup-steps", type=int, default=5,
                       help="Number of warmup steps for pruning")
    
    # Training arguments
    parser.add_argument("--max-iterations", type=int, default=200,
                       help="Maximum training iterations per trial")
    parser.add_argument("--rollout-length", type=int, default=1000,
                       help="Length of rollout episodes")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs per iteration")
    
    # Search space arguments
    parser.add_argument("--custom-search-space", action="store_true",
                       help="Use custom search space for environment")
    
    # Output arguments
    parser.add_argument("--save-dir", type=str, default="./hyperopt_results",
                       help="Directory to save results")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Generate optimization plots")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save-frequency", type=int, default=1000,
                       help="Model saving frequency (disabled for optimization)")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting hyperparameter optimization")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        # Create environment
        logger.info(f"Creating {args.env_type} environment")
        if args.env_type == "exoskeleton":
            env_kwargs = {
                'n_dofs': 7,
                'task_type': 'reach',
                'difficulty': 'moderate',
                'human_impairment': 0.3
            }
        elif args.env_type == "wheelchair":
            env_kwargs = {
                'world_size': (10, 10),
                'num_obstacles': 5,
                'difficulty': 'moderate'
            }
        else:
            env_kwargs = {}
        
        environment = create_environment(args.env_type, **env_kwargs)
        
        # Create base configuration and search space
        base_config = create_base_config(args)
        search_space = create_search_space(args)
        
        # Create Optuna study
        study_config = create_study_config(args)
        
        study = optuna.create_study(
            study_name=args.study_name,
            **study_config
        )
        
        logger.info(f"Created study: {args.study_name}")
        logger.info(f"Optimization direction: {study.direction}")
        
        # Define objective function
        def objective(trial):
            return objective_function(trial, base_config, environment, search_space, args)
        
        # Run optimization
        logger.info(f"Starting optimization with {args.n_trials} trials")
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            show_progress_bar=True
        )
        
        # Analyze results
        analyze_results(study, args)
        
        logger.info("Hyperparameter optimization completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        if 'study' in locals() and len(study.trials) > 0:
            logger.info("Analyzing partial results...")
            analyze_results(study, args)
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
    
    logger.info("Optimization script completed")


if __name__ == "__main__":
    main()