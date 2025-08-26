"""
Enterprise Performance Optimization System for Safe RL Production.

This module provides comprehensive performance optimization including:
- Model quantization and pruning
- Batch inference optimization
- Caching strategies
- Load balancing
- Auto-scaling
- Resource optimization
- Edge deployment optimization
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import script, trace
import onnx
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import pickle
import json
import psutil
import GPUtil
from pathlib import Path
import queue
import hashlib
import struct

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Performance optimization configuration."""
    # Model optimization
    quantization_enabled: bool = True
    quantization_method: str = "dynamic"  # dynamic, static, qat
    pruning_enabled: bool = True
    pruning_sparsity: float = 0.1
    model_compression: bool = True
    
    # Inference optimization
    batch_inference: bool = True
    max_batch_size: int = 32
    batch_timeout_ms: int = 10
    inference_threading: bool = True
    max_inference_threads: int = 4
    
    # Caching
    model_caching: bool = True
    result_caching: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 10000
    
    # Hardware optimization
    gpu_optimization: bool = True
    mixed_precision: bool = True
    tensorrt_optimization: bool = False
    
    # Auto-scaling
    auto_scaling: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    cpu_threshold: float = 70.0
    memory_threshold: float = 80.0
    
    # Edge optimization
    edge_optimization: bool = False
    target_device: str = "cpu"  # cpu, gpu, edge_tpu, jetson
    max_model_size_mb: int = 100


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    throughput_rps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    queue_depth: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'latency_p50': self.latency_p50,
            'latency_p95': self.latency_p95,
            'latency_p99': self.latency_p99,
            'throughput_rps': self.throughput_rps,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory_usage': self.gpu_memory_usage,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate,
            'active_connections': self.active_connections,
            'queue_depth': self.queue_depth
        }


class ModelOptimizer:
    """Model optimization for inference performance."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimized_models = {}
    
    async def optimize_model(self, model: torch.nn.Module, 
                           sample_input: torch.Tensor,
                           model_name: str) -> Dict[str, Any]:
        """Comprehensive model optimization."""
        optimization_results = {
            'original_size': self._get_model_size(model),
            'original_inference_time': 0.0,
            'optimizations_applied': [],
            'final_size': 0.0,
            'final_inference_time': 0.0,
            'speedup': 0.0,
            'size_reduction': 0.0,
            'optimized_model': None
        }
        
        try:
            # Benchmark original model
            original_time = await self._benchmark_inference(model, sample_input)
            optimization_results['original_inference_time'] = original_time
            
            optimized_model = model
            
            # Apply quantization
            if self.config.quantization_enabled:
                quantized_model = await self._apply_quantization(optimized_model, sample_input)
                if quantized_model is not None:
                    optimized_model = quantized_model
                    optimization_results['optimizations_applied'].append('quantization')
            
            # Apply pruning
            if self.config.pruning_enabled:
                pruned_model = await self._apply_pruning(optimized_model)
                if pruned_model is not None:
                    optimized_model = pruned_model
                    optimization_results['optimizations_applied'].append('pruning')
            
            # Apply TorchScript optimization
            scripted_model = await self._apply_torchscript(optimized_model, sample_input)
            if scripted_model is not None:
                optimized_model = scripted_model
                optimization_results['optimizations_applied'].append('torchscript')
            
            # Apply TensorRT optimization if available
            if self.config.tensorrt_optimization and torch.cuda.is_available():
                tensorrt_model = await self._apply_tensorrt(optimized_model, sample_input)
                if tensorrt_model is not None:
                    optimized_model = tensorrt_model
                    optimization_results['optimizations_applied'].append('tensorrt')
            
            # Final benchmarking
            final_time = await self._benchmark_inference(optimized_model, sample_input)
            final_size = self._get_model_size(optimized_model)
            
            optimization_results.update({
                'final_size': final_size,
                'final_inference_time': final_time,
                'speedup': original_time / final_time if final_time > 0 else 0,
                'size_reduction': (optimization_results['original_size'] - final_size) / optimization_results['original_size'],
                'optimized_model': optimized_model
            })
            
            # Cache optimized model
            self.optimized_models[model_name] = optimized_model
            
            logger.info(f"Model {model_name} optimized: {optimization_results['speedup']:.2f}x speedup, "
                       f"{optimization_results['size_reduction']*100:.1f}% size reduction")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            optimization_results['error'] = str(e)
            return optimization_results
    
    async def _apply_quantization(self, model: torch.nn.Module, 
                                sample_input: torch.Tensor) -> Optional[torch.nn.Module]:
        """Apply quantization to model."""
        try:
            model.eval()
            
            if self.config.quantization_method == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
                return quantized_model
            
            elif self.config.quantization_method == "static":
                # Prepare model for static quantization
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                
                # Calibrate with sample data
                with torch.no_grad():
                    model(sample_input)
                
                # Convert to quantized model
                quantized_model = torch.quantization.convert(model, inplace=False)
                return quantized_model
            
            elif self.config.quantization_method == "qat":
                # Quantization Aware Training (simplified)
                model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
                torch.quantization.prepare_qat(model, inplace=True)
                
                # In practice, you would train here
                # For now, just convert directly
                model.eval()
                quantized_model = torch.quantization.convert(model, inplace=False)
                return quantized_model
            
            return None
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return None
    
    async def _apply_pruning(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Apply structured pruning to model."""
        try:
            import torch.nn.utils.prune as prune
            
            # Apply magnitude-based pruning
            for module in model.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=self.config.pruning_sparsity)
                    prune.remove(module, 'weight')
            
            return model
            
        except Exception as e:
            logger.warning(f"Pruning failed: {e}")
            return None
    
    async def _apply_torchscript(self, model: torch.nn.Module, 
                               sample_input: torch.Tensor) -> Optional[torch.jit.ScriptModule]:
        """Apply TorchScript optimization."""
        try:
            model.eval()
            
            # Try tracing first
            try:
                scripted_model = torch.jit.trace(model, sample_input)
                return scripted_model
            except:
                # If tracing fails, try scripting
                scripted_model = torch.jit.script(model)
                return scripted_model
                
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
            return None
    
    async def _apply_tensorrt(self, model: torch.nn.Module, 
                            sample_input: torch.Tensor) -> Optional[Any]:
        """Apply TensorRT optimization."""
        try:
            # This requires torch_tensorrt
            import torch_tensorrt
            
            model.eval()
            
            # Convert to TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(sample_input.shape)],
                enabled_precisions={torch.half} if self.config.mixed_precision else {torch.float}
            )
            
            return trt_model
            
        except ImportError:
            logger.warning("torch_tensorrt not available, skipping TensorRT optimization")
            return None
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return None
    
    async def _benchmark_inference(self, model, sample_input: torch.Tensor, 
                                 num_runs: int = 100) -> float:
        """Benchmark model inference time."""
        try:
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    model(sample_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.perf_counter()
                    model(sample_input)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
            
            return np.mean(times)
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return 0.0
    
    def _get_model_size(self, model) -> int:
        """Get model size in bytes."""
        try:
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return param_size + buffer_size
            else:
                # For non-PyTorch models, estimate size
                return len(pickle.dumps(model))
        except:
            return 0


class BatchInferenceProcessor:
    """Optimized batch inference processing."""
    
    def __init__(self, config: OptimizationConfig, model_cache: Dict[str, Any]):
        self.config = config
        self.model_cache = model_cache
        self.request_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()
        self.result_cache = {}
        self.processing_stats = {
            'requests_processed': 0,
            'batches_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_batch_size': 0.0,
            'avg_processing_time': 0.0
        }
        
        self.is_running = False
        self.worker_tasks = []
    
    async def start(self):
        """Start batch processing workers."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start batch formation worker
        batch_former_task = asyncio.create_task(self._batch_formation_worker())
        self.worker_tasks.append(batch_former_task)
        
        # Start inference workers
        for i in range(self.config.max_inference_threads):
            worker_task = asyncio.create_task(self._inference_worker(f"worker_{i}"))
            self.worker_tasks.append(worker_task)
        
        logger.info("Batch inference processor started")
    
    async def stop(self):
        """Stop batch processing workers."""
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        self.worker_tasks.clear()
        logger.info("Batch inference processor stopped")
    
    async def process_request(self, model_name: str, input_data: np.ndarray, 
                            request_id: str = None) -> Any:
        """Process inference request with batching."""
        if request_id is None:
            request_id = f"req_{int(time.time() * 1000000)}"
        
        # Check cache first
        cache_key = self._generate_cache_key(model_name, input_data)
        if cache_key in self.result_cache:
            self.processing_stats['cache_hits'] += 1
            return self.result_cache[cache_key]
        
        self.processing_stats['cache_misses'] += 1
        
        # Create future for result
        result_future = asyncio.Future()
        
        # Add to request queue
        request = {
            'id': request_id,
            'model_name': model_name,
            'input_data': input_data,
            'cache_key': cache_key,
            'future': result_future,
            'timestamp': time.time()
        }
        
        await self.request_queue.put(request)
        
        # Wait for result
        try:
            result = await asyncio.wait_for(result_future, timeout=30.0)
            
            # Cache result if caching is enabled
            if self.config.result_caching:
                self.result_cache[cache_key] = result
                
                # Simple LRU eviction
                if len(self.result_cache) > self.config.max_cache_size:
                    # Remove oldest entry
                    oldest_key = min(self.result_cache.keys(), 
                                   key=lambda k: getattr(self.result_cache[k], 'timestamp', 0))
                    del self.result_cache[oldest_key]
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            raise
    
    async def _batch_formation_worker(self):
        """Worker that forms batches from individual requests."""
        while self.is_running:
            try:
                batch = []
                batch_start_time = time.time()
                
                # Collect requests for batch
                while (len(batch) < self.config.max_batch_size and 
                       (time.time() - batch_start_time) < (self.config.batch_timeout_ms / 1000)):
                    
                    try:
                        # Try to get request with short timeout
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=0.001
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        # No more requests available
                        if batch:  # If we have some requests, process them
                            break
                        continue
                
                # Process batch if we have requests
                if batch:
                    await self.batch_queue.put(batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch formation error: {e}")
                await asyncio.sleep(0.1)
    
    async def _inference_worker(self, worker_id: str):
        """Worker that processes batches of requests."""
        while self.is_running:
            try:
                # Get batch from queue
                batch = await self.batch_queue.get()
                
                if not batch:
                    continue
                
                # Group by model name
                model_batches = defaultdict(list)
                for request in batch:
                    model_batches[request['model_name']].append(request)
                
                # Process each model's batch
                for model_name, requests in model_batches.items():
                    await self._process_model_batch(worker_id, model_name, requests)
                
                # Update stats
                self.processing_stats['batches_processed'] += 1
                self.processing_stats['requests_processed'] += len(batch)
                self.processing_stats['avg_batch_size'] = (
                    (self.processing_stats['avg_batch_size'] * (self.processing_stats['batches_processed'] - 1) + len(batch)) /
                    self.processing_stats['batches_processed']
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inference worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_model_batch(self, worker_id: str, model_name: str, requests: List[Dict]):
        """Process a batch of requests for a specific model."""
        try:
            start_time = time.time()
            
            # Get model from cache
            if model_name not in self.model_cache:
                # Handle missing model
                error = f"Model {model_name} not found in cache"
                for request in requests:
                    request['future'].set_exception(ValueError(error))
                return
            
            model = self.model_cache[model_name]
            
            # Prepare batch input
            batch_inputs = []
            for request in requests:
                batch_inputs.append(request['input_data'])
            
            # Stack inputs into batch
            try:
                if isinstance(batch_inputs[0], np.ndarray):
                    batch_input = np.stack(batch_inputs)
                    batch_input_tensor = torch.from_numpy(batch_input).float()
                else:
                    batch_input_tensor = torch.stack([torch.from_numpy(inp).float() for inp in batch_inputs])
            except Exception as e:
                logger.error(f"Failed to create batch tensor: {e}")
                for request in requests:
                    request['future'].set_exception(e)
                return
            
            # Run inference
            try:
                with torch.no_grad():
                    batch_outputs = model(batch_input_tensor)
                
                if isinstance(batch_outputs, torch.Tensor):
                    batch_outputs = batch_outputs.numpy()
                
                # Distribute results to requests
                for i, request in enumerate(requests):
                    try:
                        if len(batch_outputs.shape) > 1:
                            result = batch_outputs[i]
                        else:
                            result = batch_outputs[i:i+1]
                        
                        request['future'].set_result(result)
                    except Exception as e:
                        request['future'].set_exception(e)
                
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                for request in requests:
                    request['future'].set_exception(e)
            
            # Update processing time stats
            processing_time = time.time() - start_time
            self.processing_stats['avg_processing_time'] = (
                (self.processing_stats['avg_processing_time'] * (self.processing_stats['batches_processed'] - 1) + processing_time) /
                self.processing_stats['batches_processed']
            )
            
        except Exception as e:
            logger.error(f"Model batch processing error: {e}")
            for request in requests:
                if not request['future'].done():
                    request['future'].set_exception(e)
    
    def _generate_cache_key(self, model_name: str, input_data: np.ndarray) -> str:
        """Generate cache key for input data."""
        try:
            # Create hash of model name and input data
            data_hash = hashlib.md5(input_data.tobytes()).hexdigest()
            return f"{model_name}_{data_hash}"
        except Exception:
            # Fallback to string representation
            return f"{model_name}_{hash(str(input_data))}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        )
        stats['queue_depth'] = self.request_queue.qsize()
        return stats


class CacheManager:
    """Intelligent caching system for models and results."""
    
    def __init__(self, config: OptimizationConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        try:
            # Try local cache first
            if key in self.local_cache:
                entry = self.local_cache[key]
                
                # Check TTL
                if entry['expires_at'] > time.time():
                    self.cache_stats['hits'] += 1
                    entry['access_count'] += 1
                    entry['last_accessed'] = time.time()
                    return entry['value']
                else:
                    # Expired entry
                    del self.local_cache[key]
            
            # Try Redis cache if available
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(f"cache:{key}")
                    if cached_data:
                        value = pickle.loads(cached_data)
                        
                        # Store in local cache too
                        await self._store_local(key, value)
                        
                        self.cache_stats['hits'] += 1
                        return value
                except Exception as e:
                    logger.warning(f"Redis cache get failed: {e}")
            
            # Cache miss
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        try:
            if ttl is None:
                ttl = self.config.cache_ttl_seconds
            
            # Store in local cache
            await self._store_local(key, value, ttl)
            
            # Store in Redis if available
            if self.redis_client:
                try:
                    cached_data = pickle.dumps(value)
                    self.redis_client.setex(f"cache:{key}", ttl, cached_data)
                except Exception as e:
                    logger.warning(f"Redis cache set failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def _store_local(self, key: str, value: Any, ttl: Optional[int] = None):
        """Store item in local cache with LRU eviction."""
        if ttl is None:
            ttl = self.config.cache_ttl_seconds
        
        # Check if we need to evict items
        while len(self.local_cache) >= self.config.max_cache_size:
            await self._evict_lru()
        
        # Store new entry
        entry = {
            'value': value,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'expires_at': time.time() + ttl,
            'access_count': 0,
            'size': len(pickle.dumps(value))
        }
        
        self.local_cache[key] = entry
        self.cache_stats['memory_usage'] += entry['size']
    
    async def _evict_lru(self):
        """Evict least recently used item."""
        if not self.local_cache:
            return
        
        # Find LRU item
        lru_key = min(
            self.local_cache.keys(),
            key=lambda k: self.local_cache[k]['last_accessed']
        )
        
        # Remove LRU item
        entry = self.local_cache[lru_key]
        self.cache_stats['memory_usage'] -= entry['size']
        self.cache_stats['evictions'] += 1
        del self.local_cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache_stats.copy()
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0
        stats['local_cache_size'] = len(self.local_cache)
        return stats
    
    async def clear(self):
        """Clear all caches."""
        self.local_cache.clear()
        self.cache_stats['memory_usage'] = 0
        
        if self.redis_client:
            try:
                # Clear Redis cache keys
                keys = self.redis_client.keys("cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")


class LoadBalancer:
    """Load balancing for model inference requests."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.servers = []
        self.server_stats = {}
        self.current_index = 0
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0
    
    def add_server(self, server_id: str, endpoint: str, weight: int = 1):
        """Add server to load balancer."""
        server = {
            'id': server_id,
            'endpoint': endpoint,
            'weight': weight,
            'healthy': True,
            'current_load': 0,
            'total_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0
        }
        
        self.servers.append(server)
        self.server_stats[server_id] = {
            'requests': 0,
            'errors': 0,
            'total_time': 0.0
        }
        
        logger.info(f"Added server to load balancer: {server_id}")
    
    def remove_server(self, server_id: str):
        """Remove server from load balancer."""
        self.servers = [s for s in self.servers if s['id'] != server_id]
        if server_id in self.server_stats:
            del self.server_stats[server_id]
        
        logger.info(f"Removed server from load balancer: {server_id}")
    
    async def get_server(self, load_balancing_method: str = "weighted_round_robin") -> Optional[Dict]:
        """Get server for request based on load balancing method."""
        healthy_servers = [s for s in self.servers if s['healthy']]
        
        if not healthy_servers:
            return None
        
        if load_balancing_method == "round_robin":
            return await self._round_robin_select(healthy_servers)
        elif load_balancing_method == "weighted_round_robin":
            return await self._weighted_round_robin_select(healthy_servers)
        elif load_balancing_method == "least_connections":
            return await self._least_connections_select(healthy_servers)
        elif load_balancing_method == "response_time":
            return await self._response_time_select(healthy_servers)
        else:
            # Default to round robin
            return await self._round_robin_select(healthy_servers)
    
    async def _round_robin_select(self, servers: List[Dict]) -> Dict:
        """Round robin server selection."""
        if not servers:
            return None
        
        server = servers[self.current_index % len(servers)]
        self.current_index = (self.current_index + 1) % len(servers)
        return server
    
    async def _weighted_round_robin_select(self, servers: List[Dict]) -> Dict:
        """Weighted round robin server selection."""
        if not servers:
            return None
        
        # Simple weighted selection based on server weights
        total_weight = sum(s['weight'] for s in servers)
        if total_weight == 0:
            return servers[0]
        
        # Use current_index to rotate through weighted selections
        weighted_index = self.current_index % total_weight
        current_weight = 0
        
        for server in servers:
            current_weight += server['weight']
            if weighted_index < current_weight:
                self.current_index += 1
                return server
        
        return servers[0]
    
    async def _least_connections_select(self, servers: List[Dict]) -> Dict:
        """Select server with least current connections."""
        return min(servers, key=lambda s: s['current_load'])
    
    async def _response_time_select(self, servers: List[Dict]) -> Dict:
        """Select server with best average response time."""
        return min(servers, key=lambda s: s['avg_response_time'])
    
    async def record_request(self, server_id: str, response_time: float, success: bool):
        """Record request metrics for server."""
        server = next((s for s in self.servers if s['id'] == server_id), None)
        if not server:
            return
        
        # Update server stats
        server['total_requests'] += 1
        if not success:
            server['failed_requests'] += 1
        
        # Update average response time
        if server['total_requests'] > 1:
            server['avg_response_time'] = (
                (server['avg_response_time'] * (server['total_requests'] - 1) + response_time) /
                server['total_requests']
            )
        else:
            server['avg_response_time'] = response_time
        
        # Update load balancer stats
        if server_id in self.server_stats:
            stats = self.server_stats[server_id]
            stats['requests'] += 1
            stats['total_time'] += response_time
            if not success:
                stats['errors'] += 1
    
    async def health_check(self, check_function: Callable[[str], bool] = None):
        """Perform health check on all servers."""
        current_time = time.time()
        if current_time - self.last_health_check < self.health_check_interval:
            return
        
        self.last_health_check = current_time
        
        for server in self.servers:
            try:
                if check_function:
                    healthy = await check_function(server['endpoint'])
                else:
                    # Default health check (simplified)
                    healthy = True
                
                server['healthy'] = healthy
                
                if not healthy:
                    logger.warning(f"Server {server['id']} failed health check")
                
            except Exception as e:
                logger.error(f"Health check failed for server {server['id']}: {e}")
                server['healthy'] = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(s['total_requests'] for s in self.servers)
        healthy_servers = sum(1 for s in self.servers if s['healthy'])
        
        return {
            'total_servers': len(self.servers),
            'healthy_servers': healthy_servers,
            'total_requests': total_requests,
            'server_stats': self.server_stats,
            'current_loads': {s['id']: s['current_load'] for s in self.servers}
        }


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, config: OptimizationConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.redis_client = redis_client
        
        # Initialize components
        self.model_optimizer = ModelOptimizer(config)
        self.cache_manager = CacheManager(config, redis_client)
        self.load_balancer = LoadBalancer(config)
        
        # Model cache
        self.model_cache = {}
        
        # Performance monitoring
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = PerformanceMetrics()
        
        # Batch processor
        self.batch_processor = None
        
        logger.info("PerformanceOptimizer initialized")
    
    async def initialize(self):
        """Initialize performance optimizer."""
        if self.config.batch_inference:
            self.batch_processor = BatchInferenceProcessor(self.config, self.model_cache)
            await self.batch_processor.start()
        
        logger.info("Performance optimizer initialized and started")
    
    async def shutdown(self):
        """Shutdown performance optimizer."""
        if self.batch_processor:
            await self.batch_processor.stop()
        
        logger.info("Performance optimizer shut down")
    
    async def optimize_model(self, model: torch.nn.Module, 
                           sample_input: torch.Tensor,
                           model_name: str) -> Dict[str, Any]:
        """Optimize model for inference."""
        optimization_results = await self.model_optimizer.optimize_model(
            model, sample_input, model_name
        )
        
        # Cache optimized model
        if optimization_results.get('optimized_model') is not None:
            await self.cache_manager.set(
                f"model:{model_name}",
                optimization_results['optimized_model']
            )
            self.model_cache[model_name] = optimization_results['optimized_model']
        
        return optimization_results
    
    async def process_inference_request(self, model_name: str, 
                                      input_data: np.ndarray,
                                      request_id: str = None) -> Any:
        """Process inference request with optimization."""
        if self.batch_processor and self.config.batch_inference:
            return await self.batch_processor.process_request(
                model_name, input_data, request_id
            )
        else:
            return await self._process_single_request(model_name, input_data)
    
    async def _process_single_request(self, model_name: str, input_data: np.ndarray) -> Any:
        """Process single inference request."""
        # Get model from cache
        if model_name not in self.model_cache:
            cached_model = await self.cache_manager.get(f"model:{model_name}")
            if cached_model is not None:
                self.model_cache[model_name] = cached_model
            else:
                raise ValueError(f"Model {model_name} not found")
        
        model = self.model_cache[model_name]
        
        # Convert input to tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data
        
        # Add batch dimension if needed
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
        
        # Convert back to numpy
        if isinstance(output, torch.Tensor):
            return output.numpy()
        
        return output
    
    async def update_metrics(self):
        """Update performance metrics."""
        try:
            # Update system metrics
            self.current_metrics.cpu_usage = psutil.cpu_percent()
            self.current_metrics.memory_usage = psutil.virtual_memory().percent
            
            # GPU metrics if available
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.current_metrics.gpu_usage = gpu.load * 100
                    self.current_metrics.gpu_memory_usage = gpu.memoryUtil * 100
            except:
                pass
            
            # Cache metrics
            if self.cache_manager:
                cache_stats = self.cache_manager.get_stats()
                self.current_metrics.cache_hit_rate = cache_stats.get('hit_rate', 0.0)
            
            # Batch processor metrics
            if self.batch_processor:
                batch_stats = self.batch_processor.get_stats()
                self.current_metrics.queue_depth = batch_stats.get('queue_depth', 0)
                
                # Calculate throughput
                if batch_stats.get('avg_processing_time', 0) > 0:
                    self.current_metrics.throughput_rps = 1.0 / batch_stats['avg_processing_time']
            
            # Store metrics history
            self.metrics_history.append({
                'timestamp': time.time(),
                'metrics': self.current_metrics.to_dict()
            })
            
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'cache_stats': self.cache_manager.get_stats() if self.cache_manager else {},
            'batch_stats': self.batch_processor.get_stats() if self.batch_processor else {},
            'load_balancer_stats': self.load_balancer.get_stats(),
            'optimization_config': {
                'quantization_enabled': self.config.quantization_enabled,
                'pruning_enabled': self.config.pruning_enabled,
                'batch_inference': self.config.batch_inference,
                'model_caching': self.config.model_caching,
                'result_caching': self.config.result_caching
            }
        }
    
    async def auto_scale_check(self) -> Dict[str, Any]:
        """Check if auto-scaling is needed."""
        if not self.config.auto_scaling:
            return {'scaling_needed': False}
        
        current_cpu = self.current_metrics.cpu_usage
        current_memory = self.current_metrics.memory_usage
        
        scale_up = (current_cpu > self.config.cpu_threshold or 
                   current_memory > self.config.memory_threshold)
        
        scale_down = (current_cpu < self.config.cpu_threshold * 0.3 and 
                     current_memory < self.config.memory_threshold * 0.3)
        
        return {
            'scaling_needed': scale_up or scale_down,
            'scale_direction': 'up' if scale_up else 'down' if scale_down else 'none',
            'current_cpu': current_cpu,
            'current_memory': current_memory,
            'cpu_threshold': self.config.cpu_threshold,
            'memory_threshold': self.config.memory_threshold
        }


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = OptimizationConfig(
        quantization_enabled=True,
        batch_inference=True,
        max_batch_size=32,
        model_caching=True,
        result_caching=True,
        auto_scaling=True
    )
    
    async def main():
        optimizer = PerformanceOptimizer(config)
        await optimizer.initialize()
        
        # Example model optimization
        dummy_model = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        sample_input = torch.randn(1, 10)
        
        optimization_results = await optimizer.optimize_model(
            dummy_model, sample_input, "test_model"
        )
        
        print(f"Optimization results: {optimization_results}")
        
        await optimizer.shutdown()
    
    asyncio.run(main())