#!/usr/bin/env python3
"""
Basic CPO training script for Safe RL.

This script demonstrates how to train a CPO agent on human-robot shared control
environments using the complete training pipeline.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training components
from src.training.config import (
    TrainingConfig, NetworkConfig, OptimizationConfig,
    EvaluationConfig, ExperimentConfig
)
from src.training.trainer import CPOTrainer
from src.training.experiment_tracker import create_experiment_tracker, ExperimentMetrics
from src.training.callbacks import CallbackManager, EarlyStopping, ModelCheckpoint
from src.training.schedulers import create_common_schedulers

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
            logging.FileHandler('training.log')
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


def create_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from command line arguments."""
    
    # Network configuration
    network_config = NetworkConfig(
        policy_hidden_sizes=args.policy_hidden_sizes,
        value_hidden_sizes=args.value_hidden_sizes,
        activation_function=args.activation,
        use_batch_norm=args.use_batch_norm,
        dropout_rate=args.dropout_rate
    )
    
    # Optimization configuration
    optimization_config = OptimizationConfig(
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        trust_region_radius=args.trust_region_radius,
        constraint_threshold=args.constraint_threshold,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda
    )
    
    # Evaluation configuration
    evaluation_config = EvaluationConfig(
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        eval_deterministic=True
    )
    
    # Main training configuration
    training_config = TrainingConfig(
        max_iterations=args.max_iterations,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        environment_name=args.env_type,
        seed=args.seed,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        network=network_config,
        optimization=optimization_config,
        evaluation=evaluation_config
    )
    
    return training_config


def create_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment tracking configuration."""
    return ExperimentConfig(
        use_mlflow=args.use_mlflow,
        use_wandb=args.use_wandb,
        use_local=True,  # Always use local tracking as backup
        artifact_location=args.save_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )


def setup_callbacks(args: argparse.Namespace, save_dir: str) -> CallbackManager:
    """Set up training callbacks."""
    callback_manager = CallbackManager([])
    
    # Early stopping
    if args.early_stopping:
        early_stopping = EarlyStopping(
            metric_name='episode_return',
            patience=args.early_stopping_patience,
            mode='max',
            min_delta=0.01
        )
        callback_manager.add_callback(early_stopping)
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=os.path.join(save_dir, "checkpoints"),
        save_frequency=args.save_frequency,
        save_best=True,
        metric_name='episode_return',
        mode='max',
        max_checkpoints=5
    )
    callback_manager.add_callback(checkpoint_callback)
    
    return callback_manager


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train CPO agent for Safe RL")
    
    # Environment arguments
    parser.add_argument("--env-type", type=str, default="exoskeleton",
                       choices=["exoskeleton", "wheelchair"],
                       help="Type of environment to train on")
    parser.add_argument("--env-config", type=str, default=None,
                       help="Path to environment configuration file")
    
    # Training arguments
    parser.add_argument("--max-iterations", type=int, default=1000,
                       help="Maximum training iterations")
    parser.add_argument("--rollout-length", type=int, default=2000,
                       help="Length of rollout episodes")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs per iteration")
    
    # Network arguments
    parser.add_argument("--policy-hidden-sizes", type=int, nargs="+",
                       default=[256, 256], help="Policy network hidden layer sizes")
    parser.add_argument("--value-hidden-sizes", type=int, nargs="+",
                       default=[256, 256], help="Value network hidden layer sizes")
    parser.add_argument("--activation", type=str, default="tanh",
                       choices=["relu", "tanh", "swish"],
                       help="Activation function")
    parser.add_argument("--use-batch-norm", action="store_true",
                       help="Use batch normalization")
    parser.add_argument("--dropout-rate", type=float, default=0.0,
                       help="Dropout rate")
    
    # Optimization arguments
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                       help="Policy learning rate")
    parser.add_argument("--value-lr", type=float, default=3e-4,
                       help="Value function learning rate")
    parser.add_argument("--trust-region-radius", type=float, default=0.01,
                       help="Trust region radius for CPO")
    parser.add_argument("--constraint-threshold", type=float, default=0.01,
                       help="Constraint violation threshold")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda parameter")
    
    # Evaluation arguments
    parser.add_argument("--eval-frequency", type=int, default=10,
                       help="Evaluation frequency (iterations)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    # Experiment tracking arguments
    parser.add_argument("--use-mlflow", action="store_true",
                       help="Use MLflow for experiment tracking")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb-project", type=str, default="safe-rl-cpo",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="Weights & Biases entity name")
    
    # Callback arguments
    parser.add_argument("--early-stopping", action="store_true",
                       help="Use early stopping")
    parser.add_argument("--early-stopping-patience", type=int, default=50,
                       help="Early stopping patience")
    
    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save-dir", type=str, default="./experiments",
                       help="Directory to save results")
    parser.add_argument("--save-frequency", type=int, default=50,
                       help="Model saving frequency")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting CPO training")
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
        
        # Create configurations
        training_config = create_training_config(args)
        experiment_config = create_experiment_config(args)
        
        # Validate configuration
        training_config.validate()
        logger.info("Training configuration validated")
        
        # Create experiment tracker
        experiment_tracker = create_experiment_tracker(
            experiment_config, 
            f"cpo_{args.env_type}"
        )
        
        # Create callbacks
        callback_manager = setup_callbacks(args, args.save_dir)
        
        # Create trainer
        logger.info("Creating CPO trainer")
        trainer = CPOTrainer(
            config=training_config,
            environment=environment,
            experiment_tracker=experiment_tracker,
            callback_manager=callback_manager
        )
        
        # Set up schedulers
        if hasattr(trainer, 'policy_optimizer') and hasattr(trainer, 'value_optimizer'):
            scheduler_manager = create_common_schedulers(
                trainer.policy_optimizer,
                trainer.value_optimizer
            )
            trainer.scheduler_manager = scheduler_manager
            logger.info("Learning rate schedulers created")
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            start_iteration = trainer.load_checkpoint(args.resume)
            logger.info(f"Resumed from iteration {start_iteration}")
        
        # Start experiment tracking
        experiment_tracker.start_experiment(training_config)
        
        # Run training
        logger.info("Starting training loop")
        results = trainer.train()
        
        # Log final results
        logger.info("Training completed successfully")
        logger.info(f"Total iterations: {results.total_iterations}")
        logger.info(f"Final performance: {results.final_performance}")
        logger.info(f"Safety metrics: {results.safety_metrics}")
        
        # End experiment tracking
        experiment_tracker.end_experiment("FINISHED")
        
        # Save final results
        results_path = os.path.join(args.save_dir, "final_results.json")
        results.save_to_file(results_path)
        logger.info(f"Results saved to {results_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'experiment_tracker' in locals():
            experiment_tracker.end_experiment("INTERRUPTED")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if 'experiment_tracker' in locals():
            experiment_tracker.end_experiment("FAILED")
        raise
    
    logger.info("Training script completed")


if __name__ == "__main__":
    main()