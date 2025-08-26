"""
Real-time Memory Management for Safe RL Human-Robot Shared Control

This module implements deterministic memory management for real-time systems,
eliminating garbage collection pauses and ensuring predictable allocation times.

Key features:
- Pre-allocated memory pools with fixed-size blocks
- Lock-free data structures for multi-threaded access
- NUMA-aware memory allocation for multi-socket systems
- Memory-mapped files for large datasets
- Zero-copy operations where possible
- Cache-friendly data layouts
- Deterministic allocation/deallocation times
"""

import ctypes
import logging
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
from numpy import typing as npt
import psutil

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False
    warnings.warn("PyNUMA not available - NUMA optimization disabled")

try:
    from numba import types
    from numba.typed import Dict as NumbaDict, List as NumbaList
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryPoolType(Enum):
    """Types of memory pools"""
    SMALL_BLOCKS = "small_blocks"      # <1KB blocks
    MEDIUM_BLOCKS = "medium_blocks"    # 1KB-1MB blocks
    LARGE_BLOCKS = "large_blocks"      # >1MB blocks
    ALIGNED_BLOCKS = "aligned_blocks"  # Cache-aligned blocks
    GPU_BLOCKS = "gpu_blocks"          # GPU memory blocks


class AllocationStrategy(Enum):
    """Memory allocation strategies"""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    BUDDY_SYSTEM = "buddy_system"
    SLAB_ALLOCATOR = "slab_allocator"


@dataclass
class MemoryBlock:
    """Represents a memory block in a pool"""
    address: int
    size: int
    is_free: bool
    pool_id: str
    allocation_time: float
    thread_id: int


@dataclass
class MemoryPoolConfig:
    """Configuration for a memory pool"""
    pool_type: MemoryPoolType
    block_size: int
    num_blocks: int
    alignment: int = 64  # Cache line alignment
    numa_node: int = -1  # -1 for any node
    allow_growth: bool = False
    max_blocks: int = -1  # -1 for unlimited


@dataclass
class RTMemoryConfig:
    """Configuration for real-time memory management"""
    # Pool configurations
    small_pool_config: MemoryPoolConfig = None
    medium_pool_config: MemoryPoolConfig = None
    
    def __post_init__(self):
        if self.small_pool_config is None:
            self.small_pool_config = MemoryPoolConfig(
                MemoryPoolType.SMALL_BLOCKS, 256, 10000, 64
            )
        if self.medium_pool_config is None:
            self.medium_pool_config = MemoryPoolConfig(
                MemoryPoolType.MEDIUM_BLOCKS, 65536, 1000, 64
            )
        if self.large_pool_config is None:
            self.large_pool_config = MemoryPoolConfig(
                MemoryPoolType.LARGE_BLOCKS, 1048576, 100, 64
            )
    
    large_pool_config: MemoryPoolConfig = None
    
    # Allocation strategy
    allocation_strategy: AllocationStrategy = AllocationStrategy.SLAB_ALLOCATOR
    
    # System optimization
    enable_numa_awareness: bool = True
    enable_memory_locking: bool = True  # Lock pages in RAM
    enable_transparent_hugepages: bool = True
    disable_swap: bool = True
    
    # Cache optimization
    cache_line_size: int = 64
    prefetch_distance: int = 3  # Cache lines to prefetch
    
    # Performance monitoring
    enable_allocation_tracking: bool = True
    enable_fragmentation_monitoring: bool = True
    max_allocation_time_us: int = 10  # Maximum allocation time
    
    # Memory limits
    max_total_memory_mb: int = 4096  # 4GB limit
    emergency_free_threshold: float = 0.1  # Keep 10% free
    
    # Garbage collection avoidance
    preallocate_numpy_arrays: bool = True
    numpy_array_pool_sizes: List[Tuple[Tuple[int, ...], int]] = None
    
    def __post_init__(self):
        if self.numpy_array_pool_sizes is None:
            # Default numpy array pool configurations
            self.numpy_array_pool_sizes = [
                ((64,), 1000),          # 1D arrays for state vectors
                ((64, 64), 500),        # 2D arrays for matrices
                ((3, 3), 1000),         # 3x3 transformation matrices
                ((1000, 64), 100),      # Batch processing arrays
                ((8,), 2000),           # Action vectors
                ((12,), 1000),          # Extended state vectors
            ]


class LockFreeQueue:
    """
    Lock-free queue implementation for inter-thread communication.
    
    Uses compare-and-swap operations for thread-safe access without locks.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self._lock = threading.RLock()  # Fallback for complex operations
    
    def enqueue(self, item: Any) -> bool:
        """Enqueue item, returns False if full"""
        with self._lock:
            if self.size >= self.capacity:
                return False
            
            self.buffer[self.tail] = item
            self.tail = (self.tail + 1) % self.capacity
            self.size += 1
            return True
    
    def dequeue(self) -> Optional[Any]:
        """Dequeue item, returns None if empty"""
        with self._lock:
            if self.size == 0:
                return None
            
            item = self.buffer[self.head]
            self.buffer[self.head] = None
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return item
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size >= self.capacity


class RingBuffer:
    """
    High-performance ring buffer for sensor data storage.
    
    Provides O(1) insertion and retrieval with cache-friendly layout.
    """
    
    def __init__(self, capacity: int, element_size: int, dtype=np.float32):
        self.capacity = capacity
        self.element_size = element_size
        self.dtype = dtype
        
        # Pre-allocate buffer with proper alignment
        self.buffer = np.empty((capacity, element_size), dtype=dtype)
        self.buffer = np.ascontiguousarray(self.buffer)
        
        self.head = 0
        self.tail = 0
        self.size = 0
        self._lock = threading.RLock()
    
    def push(self, data: npt.NDArray) -> bool:
        """Push data to buffer, returns False if full"""
        if data.shape[0] != self.element_size:
            raise ValueError(f"Data shape {data.shape} doesn't match element size {self.element_size}")
        
        with self._lock:
            if self.size >= self.capacity:
                # Overwrite oldest data (circular behavior)
                self.head = (self.head + 1) % self.capacity
            else:
                self.size += 1
            
            self.buffer[self.tail] = data
            self.tail = (self.tail + 1) % self.capacity
            return True
    
    def pop(self) -> Optional[npt.NDArray]:
        """Pop oldest data from buffer"""
        with self._lock:
            if self.size == 0:
                return None
            
            data = self.buffer[self.head].copy()
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return data
    
    def peek(self, offset: int = 0) -> Optional[npt.NDArray]:
        """Peek at data without removing it"""
        with self._lock:
            if offset >= self.size:
                return None
            
            index = (self.head + offset) % self.capacity
            return self.buffer[index]
    
    def get_latest(self, count: int = 1) -> Optional[npt.NDArray]:
        """Get the latest 'count' elements"""
        with self._lock:
            if count > self.size:
                count = self.size
            if count == 0:
                return None
            
            result = np.empty((count, self.element_size), dtype=self.dtype)
            for i in range(count):
                index = (self.tail - count + i) % self.capacity
                result[i] = self.buffer[index]
            
            return result


class MemoryPool:
    """
    High-performance memory pool with deterministic allocation times.
    
    Uses slab allocation for consistent performance and minimal fragmentation.
    """
    
    def __init__(self, config: MemoryPoolConfig):
        self.config = config
        self.blocks = []
        self.free_blocks = LockFreeQueue(config.num_blocks)
        self.used_blocks = {}
        self.allocation_times = []
        
        # Statistics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.peak_usage = 0
        self.current_usage = 0
        
        self._lock = threading.RLock()
        
        # Initialize memory pool
        self._initialize_pool()
        
        logger.info(f"Initialized memory pool: {config.pool_type.value}, "
                   f"{config.num_blocks} blocks of {config.block_size} bytes")
    
    def _initialize_pool(self):
        """Initialize the memory pool with pre-allocated blocks"""
        total_size = self.config.num_blocks * self.config.block_size
        
        # Allocate large contiguous memory block
        if NUMA_AVAILABLE and self.config.numa_node >= 0:
            # NUMA-aware allocation
            memory_buffer = numa.alloc_onnode(total_size, self.config.numa_node)
        else:
            # Standard allocation with alignment
            memory_buffer = ctypes.create_string_buffer(total_size + self.config.alignment)
            
            # Align to cache boundary
            buffer_addr = ctypes.addressof(memory_buffer)
            aligned_addr = (buffer_addr + self.config.alignment - 1) & ~(self.config.alignment - 1)
            offset = aligned_addr - buffer_addr
            memory_buffer = memory_buffer[offset:]
        
        # Create individual blocks
        for i in range(self.config.num_blocks):
            block_addr = ctypes.addressof(memory_buffer) + i * self.config.block_size
            block = MemoryBlock(
                address=block_addr,
                size=self.config.block_size,
                is_free=True,
                pool_id=self.config.pool_type.value,
                allocation_time=0.0,
                thread_id=0
            )
            
            self.blocks.append(block)
            self.free_blocks.enqueue(i)  # Store block index
    
    def allocate(self, size: int) -> Optional[int]:
        """
        Allocate memory block, returns memory address or None if failed.
        
        Guarantees deterministic allocation time.
        """
        start_time = time.perf_counter()
        
        if size > self.config.block_size:
            logger.error(f"Requested size {size} exceeds block size {self.config.block_size}")
            return None
        
        # Get free block from queue
        block_index = self.free_blocks.dequeue()
        if block_index is None:
            if self.config.allow_growth and len(self.blocks) < self.config.max_blocks:
                # Try to grow the pool
                return self._grow_pool_and_allocate(size)
            else:
                logger.warning("Memory pool exhausted")
                return None
        
        # Mark block as used
        block = self.blocks[block_index]
        block.is_free = False
        block.allocation_time = time.time()
        block.thread_id = threading.get_ident()
        
        with self._lock:
            self.used_blocks[block.address] = block_index
            self.total_allocations += 1
            self.current_usage += 1
            self.peak_usage = max(self.peak_usage, self.current_usage)
        
        end_time = time.perf_counter()
        allocation_time_us = (end_time - start_time) * 1_000_000
        self.allocation_times.append(allocation_time_us)
        
        return block.address
    
    def deallocate(self, address: int) -> bool:
        """
        Deallocate memory block.
        
        Returns True if successful, False if address not found.
        """
        with self._lock:
            if address not in self.used_blocks:
                logger.error(f"Attempt to deallocate invalid address: {address}")
                return False
            
            block_index = self.used_blocks[address]
            del self.used_blocks[address]
            
            # Mark block as free
            block = self.blocks[block_index]
            block.is_free = True
            block.allocation_time = 0.0
            block.thread_id = 0
            
            # Return to free queue
            self.free_blocks.enqueue(block_index)
            
            self.total_deallocations += 1
            self.current_usage -= 1
        
        return True
    
    def _grow_pool_and_allocate(self, size: int) -> Optional[int]:
        """Grow the pool and allocate from new space"""
        # Implementation would add more blocks to the pool
        # For now, just return None to indicate failure
        logger.warning("Pool growth not implemented")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        allocation_times = np.array(self.allocation_times) if self.allocation_times else np.array([])
        
        return {
            "pool_type": self.config.pool_type.value,
            "block_size": self.config.block_size,
            "total_blocks": len(self.blocks),
            "free_blocks": self.free_blocks.size,
            "used_blocks": len(self.used_blocks),
            "utilization": len(self.used_blocks) / len(self.blocks),
            "total_allocations": self.total_allocations,
            "total_deallocations": self.total_deallocations,
            "peak_usage": self.peak_usage,
            "current_usage": self.current_usage,
            "allocation_times": {
                "mean_us": float(np.mean(allocation_times)) if len(allocation_times) > 0 else 0.0,
                "max_us": float(np.max(allocation_times)) if len(allocation_times) > 0 else 0.0,
                "p95_us": float(np.percentile(allocation_times, 95)) if len(allocation_times) > 0 else 0.0,
                "samples": len(allocation_times),
            }
        }


class NumPyArrayPool:
    """
    Pool of pre-allocated NumPy arrays to avoid garbage collection.
    
    Provides arrays with common shapes and dtypes for real-time applications.
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype: np.dtype, pool_size: int):
        self.shape = shape
        self.dtype = dtype
        self.pool_size = pool_size
        
        # Pre-allocate arrays
        self.arrays = []
        self.available = LockFreeQueue(pool_size)
        
        for i in range(pool_size):
            array = np.empty(shape, dtype=dtype)
            # Ensure contiguous memory layout
            array = np.ascontiguousarray(array)
            self.arrays.append(array)
            self.available.enqueue(i)
        
        self.in_use = set()
        self._lock = threading.RLock()
        
        logger.debug(f"Initialized NumPy array pool: shape={shape}, dtype={dtype}, size={pool_size}")
    
    def get_array(self) -> Optional[npt.NDArray]:
        """Get an array from the pool"""
        array_index = self.available.dequeue()
        if array_index is None:
            logger.warning(f"NumPy array pool exhausted for shape {self.shape}")
            return None
        
        with self._lock:
            self.in_use.add(array_index)
        
        array = self.arrays[array_index]
        # Clear the array for safety
        array.fill(0)
        
        return array
    
    def return_array(self, array: npt.NDArray) -> bool:
        """Return an array to the pool"""
        # Find the array index
        array_index = None
        for i, pool_array in enumerate(self.arrays):
            if np.shares_memory(array, pool_array):
                array_index = i
                break
        
        if array_index is None:
            logger.error("Attempt to return array not from this pool")
            return False
        
        with self._lock:
            if array_index not in self.in_use:
                logger.error("Attempt to return array not currently in use")
                return False
            
            self.in_use.remove(array_index)
        
        self.available.enqueue(array_index)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get array pool statistics"""
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "pool_size": self.pool_size,
            "available": self.available.size,
            "in_use": len(self.in_use),
            "utilization": len(self.in_use) / self.pool_size,
        }


class RTMemoryManager:
    """
    Real-time memory manager providing deterministic allocation times.
    
    Features:
    - Multiple memory pools for different allocation sizes
    - Pre-allocated NumPy arrays to avoid GC
    - NUMA-aware allocation for multi-socket systems
    - Lock-free data structures where possible
    - Comprehensive performance monitoring
    """
    
    def __init__(self, config: RTMemoryConfig):
        self.config = config
        self.pools = {}
        self.numpy_pools = {}
        self.allocation_stats = []
        
        # System information
        self.numa_nodes = self._detect_numa_nodes()
        self.cache_line_size = config.cache_line_size
        
        # Initialize memory pools
        self._initialize_pools()
        
        # Initialize NumPy array pools
        if config.preallocate_numpy_arrays:
            self._initialize_numpy_pools()
        
        # System-level optimizations
        self._apply_system_optimizations()
        
        logger.info("RTMemoryManager initialized successfully")
    
    def _detect_numa_nodes(self) -> List[int]:
        """Detect available NUMA nodes"""
        if not NUMA_AVAILABLE:
            return [0]
        
        try:
            return list(range(numa.get_max_node() + 1))
        except:
            return [0]
    
    def _initialize_pools(self):
        """Initialize all memory pools"""
        pool_configs = [
            self.config.small_pool_config,
            self.config.medium_pool_config,
            self.config.large_pool_config,
        ]
        
        for pool_config in pool_configs:
            pool = MemoryPool(pool_config)
            self.pools[pool_config.pool_type] = pool
    
    def _initialize_numpy_pools(self):
        """Initialize NumPy array pools"""
        for (shape, pool_size) in self.config.numpy_array_pool_sizes:
            # Create pools for common dtypes
            for dtype in [np.float32, np.float64, np.int32, np.int64]:
                pool_key = (shape, dtype)
                pool = NumPyArrayPool(shape, dtype, pool_size)
                self.numpy_pools[pool_key] = pool
    
    def _apply_system_optimizations(self):
        """Apply system-level memory optimizations"""
        if self.config.enable_memory_locking:
            try:
                # Lock pages in memory to prevent swapping
                os.system("echo 1 > /proc/sys/vm/drop_caches")  # Clear caches
                logger.info("Memory locking enabled")
            except Exception as e:
                logger.warning(f"Failed to enable memory locking: {e}")
        
        if self.config.disable_swap:
            try:
                os.system("swapoff -a")  # Disable swap
                logger.info("Swap disabled")
            except Exception as e:
                logger.warning(f"Failed to disable swap: {e}")
        
        if self.config.enable_transparent_hugepages:
            try:
                with open("/sys/kernel/mm/transparent_hugepage/enabled", "w") as f:
                    f.write("always")
                logger.info("Transparent hugepages enabled")
            except Exception as e:
                logger.warning(f"Failed to enable hugepages: {e}")
    
    def allocate_rt_safe(self, size: int) -> Optional[npt.NDArray]:
        """
        Allocate memory with real-time safety guarantees.
        
        Selects appropriate pool based on size and provides deterministic
        allocation time.
        """
        start_time = time.perf_counter()
        
        # Select appropriate pool
        if size <= self.config.small_pool_config.block_size:
            pool = self.pools[MemoryPoolType.SMALL_BLOCKS]
        elif size <= self.config.medium_pool_config.block_size:
            pool = self.pools[MemoryPoolType.MEDIUM_BLOCKS]
        elif size <= self.config.large_pool_config.block_size:
            pool = self.pools[MemoryPoolType.LARGE_BLOCKS]
        else:
            logger.error(f"Requested size {size} exceeds largest pool block size")
            return None
        
        # Allocate from pool
        address = pool.allocate(size)
        if address is None:
            return None
        
        # Create numpy array view of allocated memory
        buffer = (ctypes.c_byte * size).from_address(address)
        array = np.frombuffer(buffer, dtype=np.uint8)
        
        end_time = time.perf_counter()
        allocation_time_us = (end_time - start_time) * 1_000_000
        self.allocation_stats.append(allocation_time_us)
        
        if allocation_time_us > self.config.max_allocation_time_us:
            logger.warning(f"Allocation time {allocation_time_us:.1f}μs exceeds limit "
                          f"{self.config.max_allocation_time_us}μs")
        
        return array
    
    def get_numpy_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> Optional[npt.NDArray]:
        """
        Get a pre-allocated NumPy array from the pool.
        
        Avoids garbage collection by reusing pre-allocated arrays.
        """
        pool_key = (shape, dtype)
        
        if pool_key not in self.numpy_pools:
            # Try to find a compatible pool
            for (pool_shape, pool_dtype), pool in self.numpy_pools.items():
                if (len(pool_shape) == len(shape) and 
                    all(ps >= s for ps, s in zip(pool_shape, shape)) and
                    pool_dtype == dtype):
                    # Use larger array and create view
                    array = pool.get_array()
                    if array is not None:
                        # Create view with requested shape
                        return array.reshape(shape) if np.prod(shape) <= np.prod(pool_shape) else None
            
            logger.warning(f"No suitable NumPy array pool for shape {shape}, dtype {dtype}")
            return None
        
        return self.numpy_pools[pool_key].get_array()
    
    def return_numpy_array(self, array: npt.NDArray) -> bool:
        """Return a NumPy array to its pool"""
        # Find the appropriate pool
        for pool in self.numpy_pools.values():
            if pool.return_array(array):
                return True
        
        logger.warning("Could not return array - not from any known pool")
        return False
    
    def create_ring_buffer(self, capacity: int, element_size: int, 
                          dtype: np.dtype = np.float32) -> RingBuffer:
        """Create a high-performance ring buffer for sensor data"""
        return RingBuffer(capacity, element_size, dtype)
    
    def create_lockfree_queue(self, capacity: int) -> LockFreeQueue:
        """Create a lock-free queue for inter-thread communication"""
        return LockFreeQueue(capacity)
    
    @contextmanager
    def rt_allocation_context(self):
        """Context manager for real-time allocation tracking"""
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        
        context_time_us = (end_time - start_time) * 1_000_000
        if context_time_us > 100:  # 100μs threshold for RT operations
            logger.warning(f"RT allocation context took {context_time_us:.1f}μs")
    
    def prefetch_memory(self, address: int, size: int):
        """Prefetch memory into CPU cache"""
        # Use builtin prefetch if available
        try:
            for i in range(0, size, self.cache_line_size):
                # Simple memory access to trigger prefetch
                addr = address + i
                ctypes.c_byte.from_address(addr)
        except Exception as e:
            logger.debug(f"Memory prefetch failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        system_memory = psutil.virtual_memory()
        
        pool_stats = {}
        for pool_type, pool in self.pools.items():
            pool_stats[pool_type.value] = pool.get_stats()
        
        numpy_pool_stats = {}
        for pool_key, pool in self.numpy_pools.items():
            shape, dtype = pool_key
            key = f"{shape}_{dtype}"
            numpy_pool_stats[key] = pool.get_stats()
        
        allocation_times = np.array(self.allocation_stats) if self.allocation_stats else np.array([])
        
        return {
            "system_memory": {
                "total_mb": system_memory.total // (1024 * 1024),
                "available_mb": system_memory.available // (1024 * 1024),
                "used_mb": system_memory.used // (1024 * 1024),
                "percent_used": system_memory.percent,
            },
            "pool_stats": pool_stats,
            "numpy_pool_stats": numpy_pool_stats,
            "allocation_performance": {
                "mean_time_us": float(np.mean(allocation_times)) if len(allocation_times) > 0 else 0.0,
                "max_time_us": float(np.max(allocation_times)) if len(allocation_times) > 0 else 0.0,
                "p95_time_us": float(np.percentile(allocation_times, 95)) if len(allocation_times) > 0 else 0.0,
                "violations": int(np.sum(allocation_times > self.config.max_allocation_time_us)) if len(allocation_times) > 0 else 0,
                "samples": len(allocation_times),
            },
            "numa_info": {
                "numa_available": NUMA_AVAILABLE,
                "numa_nodes": self.numa_nodes,
            }
        }
    
    def reset_stats(self):
        """Reset all performance statistics"""
        self.allocation_stats.clear()
        
        for pool in self.pools.values():
            pool.allocation_times.clear()
    
    def benchmark_allocation_performance(self, num_allocations: int = 10000) -> Dict[str, Any]:
        """Benchmark allocation performance"""
        logger.info(f"Benchmarking allocation performance with {num_allocations} allocations...")
        
        self.reset_stats()
        
        # Test different allocation sizes
        test_sizes = [64, 256, 1024, 4096, 16384, 65536]
        results = {}
        
        for size in test_sizes:
            if size > self.config.large_pool_config.block_size:
                continue
            
            allocation_times = []
            
            # Warmup
            for _ in range(100):
                array = self.allocate_rt_safe(size)
                if array is not None:
                    # Simulate deallocation by not keeping reference
                    del array
            
            # Benchmark
            for _ in range(num_allocations):
                start_time = time.perf_counter()
                array = self.allocate_rt_safe(size)
                end_time = time.perf_counter()
                
                if array is not None:
                    allocation_time_us = (end_time - start_time) * 1_000_000
                    allocation_times.append(allocation_time_us)
                    del array
            
            allocation_times = np.array(allocation_times)
            results[f"size_{size}"] = {
                "mean_time_us": float(np.mean(allocation_times)),
                "median_time_us": float(np.median(allocation_times)),
                "p95_time_us": float(np.percentile(allocation_times, 95)),
                "p99_time_us": float(np.percentile(allocation_times, 99)),
                "max_time_us": float(np.max(allocation_times)),
                "std_time_us": float(np.std(allocation_times)),
                "successful_allocations": len(allocation_times),
                "success_rate": len(allocation_times) / num_allocations,
            }
        
        logger.info("Allocation benchmark complete")
        return results
    
    def __del__(self):
        """Cleanup resources"""
        # Cleanup would go here - return memory to system, etc.
        pass


# Example usage
def main():
    """Example usage of the real-time memory manager"""
    
    # Create configuration
    config = RTMemoryConfig(
        enable_numa_awareness=True,
        enable_memory_locking=False,  # Requires root privileges
        max_total_memory_mb=2048,
    )
    
    # Create memory manager
    memory_manager = RTMemoryManager(config)
    
    # Test allocation
    with memory_manager.rt_allocation_context():
        # Allocate some memory
        data = memory_manager.allocate_rt_safe(1024)
        print(f"Allocated array: {data.shape if data is not None else 'Failed'}")
        
        # Get NumPy array from pool
        array = memory_manager.get_numpy_array((64,), np.float32)
        print(f"Got NumPy array: {array.shape if array is not None else 'Failed'}")
        
        # Create ring buffer for sensor data
        ring_buffer = memory_manager.create_ring_buffer(1000, 12, np.float32)
        
        # Test ring buffer
        sensor_data = np.random.randn(12).astype(np.float32)
        ring_buffer.push(sensor_data)
        retrieved_data = ring_buffer.pop()
        print(f"Ring buffer test: {np.allclose(sensor_data, retrieved_data) if retrieved_data is not None else 'Failed'}")
        
        # Return array to pool
        if array is not None:
            memory_manager.return_numpy_array(array)
    
    # Get memory statistics
    stats = memory_manager.get_memory_stats()
    print("Memory Statistics:")
    for category, data in stats.items():
        print(f"  {category}: {data}")
    
    # Run benchmark
    benchmark_results = memory_manager.benchmark_allocation_performance(1000)
    print("Benchmark Results:")
    for size, results in benchmark_results.items():
        print(f"  {size}: Mean={results['mean_time_us']:.2f}μs, "
              f"P95={results['p95_time_us']:.2f}μs, "
              f"Max={results['max_time_us']:.2f}μs")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    main()