#!/usr/bin/env python3
"""
Distributed training script for CPO.

This script demonstrates how to run distributed CPO training across
multiple GPUs and nodes for improved training efficiency.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import torch.multiprocessing as mp

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import training components
from src.training.config import TrainingConfig, ExperimentConfig
from src.training.distributed import (
    DistributedConfig, DistributedCPOTrainer, DistributedLauncher,
    launch_single_node_training, launch_multi_node_training
)
from src.training.experiment_tracker import create_experiment_tracker

# Import environments
from src.environments.exoskeleton_env import ExoskeletonEnvironment
from src.environments.wheelchair_env import WheelchairEnvironment


def setup_logging(log_level: str = "INFO", rank: int = 0) -> None:
    """Set up logging configuration for distributed training."""
    log_format = f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    # Only main process writes to file
    if rank == 0:
        handlers.append(logging.FileHandler('distributed_training.log'))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
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
    """Create training configuration for distributed training."""
    from src.training.config import NetworkConfig, OptimizationConfig, EvaluationConfig
    
    # Adjust batch size and other parameters for distributed training
    # Total effective batch size will be args.batch_size * world_size
    effective_batch_size = args.batch_size // args.world_size if args.world_size > 1 else args.batch_size
    
    network_config = NetworkConfig(
        policy_hidden_sizes=args.policy_hidden_sizes,
        value_hidden_sizes=args.value_hidden_sizes,
        activation_function=args.activation,
        use_batch_norm=args.use_batch_norm
    )
    
    optimization_config = OptimizationConfig(
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        trust_region_radius=args.trust_region_radius,
        constraint_threshold=args.constraint_threshold,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda
    )
    
    evaluation_config = EvaluationConfig(
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        eval_deterministic=True
    )
    
    return TrainingConfig(
        max_iterations=args.max_iterations,
        rollout_length=args.rollout_length,
        batch_size=effective_batch_size,
        num_epochs=args.num_epochs,
        environment_name=args.env_type,
        seed=args.seed,
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        network=network_config,
        optimization=optimization_config,
        evaluation=evaluation_config
    )


def create_distributed_config(args: argparse.Namespace) -> DistributedConfig:
    """Create distributed training configuration."""
    if args.training_mode == "single_node":
        return DistributedLauncher.setup_single_node_multiprocess(args.gpus_per_node)
    elif args.training_mode == "multi_node":
        return DistributedLauncher.setup_multi_node(
            node_rank=args.node_rank,
            num_nodes=args.num_nodes,
            master_addr=args.master_addr,
            master_port=args.master_port,
            gpus_per_node=args.gpus_per_node
        )
    else:
        # Single process training
        return DistributedConfig(use_distributed=False)


def train_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """Worker function for distributed training."""
    # Set up logging for this worker
    setup_logging(args.log_level, rank)
    logger = logging.getLogger(__name__)
    
    try:
        # Create distributed configuration
        distributed_config = DistributedConfig(
            use_distributed=world_size > 1,
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            backend=args.backend,
            use_zero_optimizer=args.use_zero_optimizer,
            sync_bn=args.sync_bn
        )
        
        # Create environment
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
        
        # Create training configuration
        training_config = create_training_config(args)
        training_config.seed = args.seed + rank  # Different seed per process
        
        # Create experiment tracker (only for rank 0)
        experiment_tracker = None
        if rank == 0:
            experiment_config = ExperimentConfig(
                use_mlflow=args.use_mlflow,
                use_wandb=args.use_wandb,
                use_local=True,
                artifact_location=args.save_dir,
                wandb_project=args.wandb_project
            )
            experiment_tracker = create_experiment_tracker(
                experiment_config,
                f"distributed_cpo_{args.env_type}"
            )
            experiment_tracker.start_experiment(training_config)
        
        # Create distributed trainer
        logger.info(f"Creating distributed trainer (rank {rank}/{world_size})")
        trainer = DistributedCPOTrainer(
            config=training_config,
            environment=environment,
            distributed_config=distributed_config,
            experiment_tracker=experiment_tracker
        )
        
        # Run training
        if rank == 0:
            logger.info("Starting distributed training")
        
        results = trainer.train()
        
        # Log results (only rank 0)
        if rank == 0:
            logger.info("Distributed training completed successfully")
            logger.info(f"Total iterations: {results.total_iterations}")
            logger.info(f"Final performance: {results.final_performance}")
            
            if experiment_tracker:
                experiment_tracker.end_experiment("FINISHED")
            
            # Save results
            results_path = os.path.join(args.save_dir, "distributed_results.json")
            results.save_to_file(results_path)
            logger.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Distributed training failed on rank {rank}: {e}")
        if rank == 0 and 'experiment_tracker' in locals() and experiment_tracker:
            experiment_tracker.end_experiment("FAILED")
        raise


def main():
    """Main distributed training function."""
    parser = argparse.ArgumentParser(description="Distributed CPO training")
    
    # Training mode arguments
    parser.add_argument("--training-mode", type=str, default="single_process",
                       choices=["single_process", "single_node", "multi_node"],
                       help="Distributed training mode")
    
    # Environment arguments
    parser.add_argument("--env-type", type=str, default="exoskeleton",
                       choices=["exoskeleton", "wheelchair"],
                       help="Type of environment to train on")
    
    # Distributed training arguments
    parser.add_argument("--gpus-per-node", type=int, default=None,
                       help="Number of GPUs per node (auto-detect if None)")
    parser.add_argument("--num-nodes", type=int, default=1,
                       help="Number of nodes for multi-node training")
    parser.add_argument("--node-rank", type=int, default=0,
                       help="Rank of current node (0 to num_nodes-1)")
    parser.add_argument("--master-addr", type=str, default="localhost",
                       help="Master node address for multi-node training")
    parser.add_argument("--master-port", type=str, default="12355",
                       help="Master node port for multi-node training")
    parser.add_argument("--backend", type=str, default="nccl",
                       choices=["nccl", "gloo"],
                       help="Distributed backend")
    
    # Distributed optimization arguments
    parser.add_argument("--use-zero-optimizer", action="store_true",
                       help="Use ZeRO optimizer for memory efficiency")
    parser.add_argument("--sync-bn", action="store_true",
                       help="Use synchronized batch normalization")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping value")
    
    # Training arguments
    parser.add_argument("--max-iterations", type=int, default=1000,
                       help="Maximum training iterations")
    parser.add_argument("--rollout-length", type=int, default=2000,
                       help="Length of rollout episodes")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Total batch size across all processes")
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs per iteration")
    parser.add_argument("--world-size", type=int, default=1,
                       help="Total number of processes (computed automatically)")
    
    # Network arguments
    parser.add_argument("--policy-hidden-sizes", type=int, nargs="+",
                       default=[256, 256], help="Policy network hidden sizes")
    parser.add_argument("--value-hidden-sizes", type=int, nargs="+",
                       default=[256, 256], help="Value network hidden sizes")
    parser.add_argument("--activation", type=str, default="tanh",
                       help="Activation function")
    parser.add_argument("--use-batch-norm", action="store_true",
                       help="Use batch normalization")
    
    # Optimization arguments
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                       help="Policy learning rate")
    parser.add_argument("--value-lr", type=float, default=3e-4,
                       help="Value learning rate")
    parser.add_argument("--trust-region-radius", type=float, default=0.01,
                       help="Trust region radius")
    parser.add_argument("--constraint-threshold", type=float, default=0.01,
                       help="Constraint threshold")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                       help="GAE lambda")
    
    # Evaluation arguments
    parser.add_argument("--eval-frequency", type=int, default=10,
                       help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    # Experiment tracking arguments
    parser.add_argument("--use-mlflow", action="store_true",
                       help="Use MLflow for experiment tracking")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="distributed-safe-rl",
                       help="W&B project name")
    
    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save-dir", type=str, default="./distributed_experiments",
                       help="Directory to save results")
    parser.add_argument("--save-frequency", type=int, default=50,
                       help="Model saving frequency")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    parser.add_argument("--resume", type=str, default=None,
                       help="Checkpoint path to resume from")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.training_mode == "multi_node" and args.num_nodes <= 1:
        raise ValueError("Multi-node training requires num_nodes > 1")
    
    if args.training_mode == "multi_node" and not args.master_addr:
        raise ValueError("Multi-node training requires master_addr")
    
    # Auto-detect GPUs per node if not specified
    if args.gpus_per_node is None:
        args.gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Compute world size
    if args.training_mode == "single_process":
        args.world_size = 1
    elif args.training_mode == "single_node":
        args.world_size = args.gpus_per_node
    else:  # multi_node
        args.world_size = args.num_nodes * args.gpus_per_node
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up main process logging
    setup_logging(args.log_level, 0)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting distributed CPO training")
    logger.info(f"Training mode: {args.training_mode}")
    logger.info(f"World size: {args.world_size}")
    logger.info(f"GPUs per node: {args.gpus_per_node}")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        if args.training_mode == "single_process":
            # Single process training
            train_worker(0, 1, args)
            
        elif args.training_mode == "single_node":
            # Single-node multi-GPU training
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available for single-node training")
            
            logger.info(f"Launching single-node training with {args.gpus_per_node} GPUs")
            
            mp.spawn(
                train_worker,
                args=(args.world_size, args),
                nprocs=args.gpus_per_node,
                join=True
            )
            
        elif args.training_mode == "multi_node":
            # Multi-node training
            logger.info(f"Starting multi-node training (node {args.node_rank}/{args.num_nodes})")
            
            # Calculate rank offset for this node
            rank_offset = args.node_rank * args.gpus_per_node
            
            def multi_node_worker(local_rank, world_size, args):
                global_rank = rank_offset + local_rank
                train_worker(global_rank, world_size, args)
            
            mp.spawn(
                multi_node_worker,
                args=(args.world_size, args),
                nprocs=args.gpus_per_node,
                join=True
            )
        
        logger.info("Distributed training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Distributed training failed: {e}")
        raise
    
    logger.info("Distributed training script completed")


if __name__ == "__main__":
    main()