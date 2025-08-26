"""
Distributed training support for Safe RL with CPO.

This module provides comprehensive distributed training capabilities including
multi-GPU training, multi-node training, and gradient synchronization for
scalable CPO algorithm training.
"""

import os
import socket
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import numpy as np

from .config import TrainingConfig
from .trainer import CPOTrainer, TrainingResults


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Multi-GPU settings
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Multi-node settings
    master_addr: str = "localhost"
    master_port: str = "12355"
    node_rank: int = 0
    num_nodes: int = 1
    
    # Training settings
    backend: str = "nccl"  # nccl for GPU, gloo for CPU
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    
    # Optimization settings
    use_zero_optimizer: bool = False
    sync_bn: bool = True
    
    # Communication settings
    bucket_cap_mb: int = 25
    timeout_minutes: int = 30
    
    def __post_init__(self):
        """Validate configuration."""
        if self.use_distributed and self.world_size < 2:
            raise ValueError("World size must be >= 2 for distributed training")
        
        if self.backend == "nccl" and not torch.cuda.is_available():
            logging.warning("NCCL backend requires CUDA, falling back to gloo")
            self.backend = "gloo"


class DistributedManager:
    """Manager for distributed training setup and teardown."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        
    def setup(self) -> None:
        """Initialize distributed training environment."""
        if not self.config.use_distributed:
            return
            
        # Set environment variables
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        
        # Initialize process group
        try:
            dist.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank,
                timeout=torch.distributed.default_pg_timeout * self.config.timeout_minutes
            )
            self.is_initialized = True
            
            if self.is_main_process():
                self.logger.info(f"Initialized distributed training with {self.config.world_size} processes")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up distributed training environment."""
        if self.is_initialized:
            try:
                dist.destroy_process_group()
                self.is_initialized = False
            except Exception as e:
                self.logger.warning(f"Error during distributed cleanup: {e}")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.config.rank == 0
    
    def get_rank(self) -> int:
        """Get current process rank."""
        return self.config.rank
    
    def get_world_size(self) -> int:
        """Get total number of processes."""
        return self.config.world_size


class DistributedSampler:
    """Custom distributed sampler for rollout data."""
    
    def __init__(self, dataset_size: int, world_size: int, rank: int, 
                 shuffle: bool = True, seed: int = 0):
        self.dataset_size = dataset_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Calculate samples per rank
        self.samples_per_rank = dataset_size // world_size
        self.total_samples = self.samples_per_rank * world_size
        
    def __iter__(self):
        """Generate indices for this rank."""
        if self.shuffle:
            # Use seed + epoch for deterministic shuffling across epochs
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        
        # Add extra samples to make data evenly divisible
        if len(indices) < self.total_samples:
            indices += indices[:(self.total_samples - len(indices))]
        
        # Subsample for this rank
        start_idx = self.rank * self.samples_per_rank
        end_idx = start_idx + self.samples_per_rank
        rank_indices = indices[start_idx:end_idx]
        
        return iter(rank_indices)
    
    def __len__(self):
        return self.samples_per_rank
    
    def set_epoch(self, epoch: int):
        """Set current epoch for shuffling."""
        self.epoch = epoch


class DistributedCPOTrainer(CPOTrainer):
    """Distributed version of CPO trainer."""
    
    def __init__(self, config: TrainingConfig, distributed_config: DistributedConfig):
        self.distributed_config = distributed_config
        self.distributed_manager = DistributedManager(distributed_config)
        
        # Initialize distributed environment
        self.distributed_manager.setup()
        
        # Set device for this process
        if torch.cuda.is_available() and distributed_config.use_distributed:
            torch.cuda.set_device(distributed_config.local_rank)
            self.device = torch.device(f'cuda:{distributed_config.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        # Initialize parent class
        super().__init__(config)
        
        # Wrap models with DDP
        self._setup_distributed_models()
        
        # Setup distributed optimizer
        self._setup_distributed_optimizer()
        
    def _setup_distributed_models(self) -> None:
        """Wrap models with DistributedDataParallel."""
        if not self.distributed_config.use_distributed:
            return
            
        # Move models to appropriate device
        self.policy_net = self.policy_net.to(self.device)
        self.value_net = self.value_net.to(self.device)
        
        # Apply synchronized batch normalization if requested
        if self.distributed_config.sync_bn:
            self.policy_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.policy_net)
            self.value_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.value_net)
        
        # Wrap with DDP
        self.policy_net = DDP(
            self.policy_net,
            device_ids=[self.distributed_config.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
            gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
            bucket_cap_mb=self.distributed_config.bucket_cap_mb
        )
        
        self.value_net = DDP(
            self.value_net,
            device_ids=[self.distributed_config.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
            gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
            bucket_cap_mb=self.distributed_config.bucket_cap_mb
        )
    
    def _setup_distributed_optimizer(self) -> None:
        """Setup distributed optimizer with optional ZeRO."""
        if not self.distributed_config.use_distributed:
            return
            
        if self.distributed_config.use_zero_optimizer:
            # Use ZeRO optimizer for memory efficiency
            self.policy_optimizer = ZeroRedundancyOptimizer(
                self.policy_net.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.config.optimization.policy_lr
            )
            
            self.value_optimizer = ZeroRedundancyOptimizer(
                self.value_net.parameters(),
                optimizer_class=torch.optim.Adam,
                lr=self.config.optimization.value_lr
            )
        else:
            # Use standard optimizers
            self.policy_optimizer = torch.optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.optimization.policy_lr
            )
            
            self.value_optimizer = torch.optim.Adam(
                self.value_net.parameters(),
                lr=self.config.optimization.value_lr
            )
    
    def _collect_rollouts(self) -> Dict[str, torch.Tensor]:
        """Collect rollouts with distributed sampling."""
        rollout_data = super()._collect_rollouts()
        
        if not self.distributed_config.use_distributed:
            return rollout_data
        
        # Create distributed sampler for rollout indices
        num_rollouts = rollout_data['states'].shape[0]
        sampler = DistributedSampler(
            num_rollouts, 
            self.distributed_config.world_size,
            self.distributed_config.rank,
            shuffle=True,
            seed=self.config.seed
        )
        
        # Sample subset for this rank
        rank_indices = list(sampler)
        
        # Subsample rollout data
        distributed_data = {}
        for key, value in rollout_data.items():
            distributed_data[key] = value[rank_indices]
        
        return distributed_data
    
    def _synchronize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across all processes."""
        if not self.distributed_config.use_distributed:
            return metrics
        
        synchronized_metrics = {}
        
        for key, value in metrics.items():
            # Convert to tensor for all_reduce
            tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
            
            # Average across all processes
            dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
            tensor_value /= self.distributed_config.world_size
            
            synchronized_metrics[key] = tensor_value.item()
        
        return synchronized_metrics
    
    def _train_step(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Distributed training step with gradient synchronization."""
        train_info = super()._train_step(rollout_data)
        
        # Synchronize metrics across processes
        if self.distributed_config.use_distributed:
            train_info = self._synchronize_metrics(train_info)
        
        return train_info
    
    def save_checkpoint(self, iteration: int) -> None:
        """Save checkpoint from main process only."""
        if self.distributed_manager.is_main_process():
            super().save_checkpoint(iteration)
        
        # Synchronize all processes
        if self.distributed_config.use_distributed:
            dist.barrier()
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint with distributed model handling."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states (handle DDP wrapper)
        if self.distributed_config.use_distributed:
            self.policy_net.module.load_state_dict(checkpoint['policy_net_state_dict'])
            self.value_net.module.load_state_dict(checkpoint['value_net_state_dict'])
        else:
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        
        # Load optimizer states
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        return checkpoint['iteration']
    
    def train(self) -> TrainingResults:
        """Distributed training loop."""
        try:
            results = super().train()
            
            # Gather results from all processes
            if self.distributed_config.use_distributed and self.distributed_manager.is_main_process():
                self.logger.info("Distributed training completed successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Distributed training failed: {e}")
            raise
        finally:
            # Cleanup distributed environment
            self.distributed_manager.cleanup()


class DistributedLauncher:
    """Utility class for launching distributed training."""
    
    @staticmethod
    def launch_multiprocessing(train_fn, config: TrainingConfig, 
                             distributed_config: DistributedConfig,
                             args: tuple = ()) -> None:
        """Launch multi-GPU training using torch.multiprocessing."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for distributed training")
        
        world_size = torch.cuda.device_count()
        distributed_config.world_size = world_size
        
        # Spawn processes
        mp.spawn(
            DistributedLauncher._train_worker,
            args=(train_fn, config, distributed_config, args),
            nprocs=world_size,
            join=True
        )
    
    @staticmethod
    def _train_worker(rank: int, train_fn, config: TrainingConfig,
                     distributed_config: DistributedConfig, args: tuple) -> None:
        """Worker function for distributed training."""
        # Set rank and local rank
        distributed_config.rank = rank
        distributed_config.local_rank = rank
        
        # Call training function
        train_fn(config, distributed_config, *args)
    
    @staticmethod
    def get_free_port() -> str:
        """Find a free port for distributed training."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        return str(port)
    
    @staticmethod
    def setup_single_node_multiprocess(num_gpus: Optional[int] = None) -> DistributedConfig:
        """Setup configuration for single-node multi-GPU training."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        world_size = num_gpus or torch.cuda.device_count()
        
        return DistributedConfig(
            use_distributed=True,
            world_size=world_size,
            master_addr="localhost",
            master_port=DistributedLauncher.get_free_port(),
            backend="nccl"
        )
    
    @staticmethod
    def setup_multi_node(node_rank: int, num_nodes: int, 
                        master_addr: str, master_port: str,
                        gpus_per_node: Optional[int] = None) -> DistributedConfig:
        """Setup configuration for multi-node training."""
        gpus_per_node = gpus_per_node or torch.cuda.device_count()
        world_size = num_nodes * gpus_per_node
        
        return DistributedConfig(
            use_distributed=True,
            world_size=world_size,
            node_rank=node_rank,
            num_nodes=num_nodes,
            master_addr=master_addr,
            master_port=master_port,
            backend="nccl" if torch.cuda.is_available() else "gloo"
        )


def train_distributed_cpo(config: TrainingConfig, 
                         distributed_config: DistributedConfig) -> TrainingResults:
    """Main function for distributed CPO training."""
    trainer = DistributedCPOTrainer(config, distributed_config)
    return trainer.train()


# Example usage functions
def launch_single_node_training(config: TrainingConfig, num_gpus: Optional[int] = None) -> None:
    """Launch single-node multi-GPU training."""
    distributed_config = DistributedLauncher.setup_single_node_multiprocess(num_gpus)
    
    DistributedLauncher.launch_multiprocessing(
        train_distributed_cpo,
        config,
        distributed_config
    )


def launch_multi_node_training(config: TrainingConfig, node_rank: int, 
                              num_nodes: int, master_addr: str, 
                              master_port: str) -> None:
    """Launch multi-node training."""
    distributed_config = DistributedLauncher.setup_multi_node(
        node_rank, num_nodes, master_addr, master_port
    )
    
    # This would typically be called with different node_rank on each machine
    trainer = DistributedCPOTrainer(config, distributed_config)
    trainer.train()