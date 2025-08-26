"""
Real-time Optimization Integration Module

This module integrates all real-time optimization components into a unified
system that can be easily deployed and configured for production use.

Key features:
- Unified configuration and initialization
- Coordinated startup and shutdown procedures
- Performance validation and tuning
- Health monitoring and diagnostics
- Production deployment utilities
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch

# Import optimization components
from .inference.optimized_policy import OptimizedPolicy, OptimizationConfig
from .safety_rt.fast_constraint_checker import FastConstraintChecker, RTSafetyConfig
from .memory.rt_memory_manager import RTMemoryManager, RTMemoryConfig
from .parallel.rt_control_system import RTControlSystem, RTControlConfig
from .gpu.cuda_optimizer import CUDAOptimizer, GPUConfig
from .system.rt_system_config import RTSystemOptimizer, RTSystemConfig
from .benchmarking.rt_performance_benchmark import RTPerformanceBenchmark, TimingRequirements
from .monitoring.continuous_performance_monitor import ContinuousPerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class RTOptimizationConfig:
    """Unified configuration for all RT optimization components"""
    
    # Timing requirements
    timing_requirements: TimingRequirements = field(default_factory=lambda: TimingRequirements(
        max_execution_time_us=1000,
        max_jitter_us=50,
        min_frequency_hz=1000.0
    ))
    
    # Component configurations
    policy_optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())
    safety_config: RTSafetyConfig = field(default_factory=lambda: RTSafetyConfig())
    memory_config: RTMemoryConfig = field(default_factory=lambda: RTMemoryConfig())
    parallel_config: RTControlConfig = field(default_factory=lambda: RTControlConfig())
    gpu_config: GPUConfig = field(default_factory=lambda: GPUConfig())
    system_config: RTSystemConfig = field(default_factory=lambda: RTSystemConfig())
    
    # Integration settings
    enable_system_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_continuous_monitoring: bool = True
    enable_performance_validation: bool = True
    
    # Deployment settings
    deployment_mode: str = "production"  # development, staging, production
    log_level: str = "INFO"
    metrics_export_enabled: bool = True
    health_check_enabled: bool = True
    
    # Performance tuning
    auto_tune_enabled: bool = True
    benchmark_on_startup: bool = True
    performance_target_percentile: float = 99.0  # Target P99 performance


class RTOptimizedSystem:
    """
    Integrated real-time optimized system for Safe RL.
    
    This class coordinates all optimization components and provides a unified
    interface for deploying high-performance real-time Safe RL systems.
    """
    
    def __init__(self, config: RTOptimizationConfig, base_policy: Optional[torch.nn.Module] = None):
        self.config = config
        self.base_policy = base_policy
        
        # Component instances
        self.optimized_policy: Optional[OptimizedPolicy] = None
        self.constraint_checker: Optional[FastConstraintChecker] = None
        self.memory_manager: Optional[RTMemoryManager] = None
        self.control_system: Optional[RTControlSystem] = None
        self.gpu_optimizer: Optional[CUDAOptimizer] = None
        self.system_optimizer: Optional[RTSystemOptimizer] = None
        self.performance_monitor: Optional[ContinuousPerformanceMonitor] = None
        self.benchmark_system: Optional[RTPerformanceBenchmark] = None
        
        # System state
        self.initialized = False
        self.running = False
        self.performance_validated = False
        self.startup_time = None
        self.shutdown_time = None
        
        # Performance tracking
        self.performance_stats = {}
        self.health_status = {"status": "initializing", "components": {}}
        
        logger.info("RT Optimized System created")
    
    async def initialize(self) -> bool:
        """
        Initialize all optimization components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized:
            logger.warning("System already initialized")
            return True
        
        logger.info("Initializing RT optimization components...")
        startup_start = time.time()
        
        try:
            # 1. Initialize system-level optimizations first
            if self.config.enable_system_optimization:
                await self._initialize_system_optimization()
            
            # 2. Initialize memory management
            await self._initialize_memory_management()
            
            # 3. Initialize GPU optimization (if available)
            if self.config.enable_gpu_acceleration and torch.cuda.is_available():
                await self._initialize_gpu_optimization()
            
            # 4. Initialize optimized policy
            if self.base_policy is not None:
                await self._initialize_policy_optimization()
            
            # 5. Initialize safety constraint checker
            await self._initialize_safety_system()
            
            # 6. Initialize parallel control system
            await self._initialize_control_system()
            
            # 7. Initialize monitoring (if enabled)
            if self.config.enable_continuous_monitoring:
                await self._initialize_monitoring()
            
            # 8. Run performance validation
            if self.config.enable_performance_validation:
                await self._validate_performance()
            
            self.initialized = True
            self.startup_time = time.time() - startup_start
            
            logger.info(f"RT optimization system initialized successfully in {self.startup_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RT optimization system: {e}")
            await self.cleanup()
            return False
    
    async def _initialize_system_optimization(self):
        """Initialize system-level optimizations"""
        logger.info("Initializing system optimization...")
        
        self.system_optimizer = RTSystemOptimizer(self.config.system_config)
        
        # Apply system optimizations (requires root privileges)
        try:
            optimization_results = self.system_optimizer.apply_all_optimizations()
            successful_optimizations = sum(1 for success in optimization_results.values() if success)
            total_optimizations = len(optimization_results)
            
            logger.info(f"Applied {successful_optimizations}/{total_optimizations} system optimizations")
            
            self.health_status["components"]["system_optimizer"] = {
                "status": "healthy",
                "optimizations_applied": successful_optimizations,
                "total_optimizations": total_optimizations
            }
            
        except Exception as e:
            logger.warning(f"System optimization failed (may need root privileges): {e}")
            self.health_status["components"]["system_optimizer"] = {
                "status": "degraded",
                "error": str(e)
            }
    
    async def _initialize_memory_management(self):
        """Initialize memory management"""
        logger.info("Initializing memory management...")
        
        self.memory_manager = RTMemoryManager(self.config.memory_config)
        
        # Run memory benchmark
        benchmark_results = self.memory_manager.benchmark_allocation_performance(1000)
        
        self.health_status["components"]["memory_manager"] = {
            "status": "healthy",
            "benchmark_results": benchmark_results
        }
        
        logger.info("Memory management initialized")
    
    async def _initialize_gpu_optimization(self):
        """Initialize GPU optimization"""
        logger.info("Initializing GPU optimization...")
        
        try:
            self.gpu_optimizer = CUDAOptimizer(self.config.gpu_config)
            
            # Run GPU benchmark
            gpu_stats = self.gpu_optimizer.benchmark_gpu_operations(500)
            
            self.health_status["components"]["gpu_optimizer"] = {
                "status": "healthy",
                "device_info": gpu_stats.get("device_info", {}),
                "benchmark_results": gpu_stats.get("operation_timing", {})
            }
            
            logger.info("GPU optimization initialized")
            
        except Exception as e:
            logger.warning(f"GPU optimization failed: {e}")
            self.gpu_optimizer = None
            self.health_status["components"]["gpu_optimizer"] = {
                "status": "unavailable",
                "error": str(e)
            }
    
    async def _initialize_policy_optimization(self):
        """Initialize optimized policy"""
        if self.base_policy is None:
            logger.warning("No base policy provided - skipping policy optimization")
            return
        
        logger.info("Initializing policy optimization...")
        
        self.optimized_policy = OptimizedPolicy(self.base_policy, self.config.policy_optimization)
        
        # Run policy benchmark
        benchmark_results = self.optimized_policy.benchmark_performance(1000)
        
        self.health_status["components"]["optimized_policy"] = {
            "status": "healthy",
            "performance_stats": benchmark_results.get("timing_stats", {}),
            "model_stats": benchmark_results.get("model_stats", {})
        }
        
        logger.info("Policy optimization initialized")
    
    async def _initialize_safety_system(self):
        """Initialize safety constraint checker"""
        logger.info("Initializing safety system...")
        
        self.constraint_checker = FastConstraintChecker(self.config.safety_config)
        
        # Run safety benchmark
        benchmark_results = self.constraint_checker.benchmark_performance(1000)
        
        self.health_status["components"]["constraint_checker"] = {
            "status": "healthy",
            "performance_stats": benchmark_results.get("timing_stats", {}),
            "constraint_stats": benchmark_results.get("constraint_stats", {})
        }
        
        logger.info("Safety system initialized")
    
    async def _initialize_control_system(self):
        """Initialize parallel control system"""
        logger.info("Initializing control system...")
        
        self.control_system = RTControlSystem(self.config.parallel_config)
        
        # Add RT control thread
        self.control_system.add_thread(
            self.config.parallel_config.thread_configs['RT_CONTROL'].thread_type,
            self._rt_control_loop
        )
        
        # Add safety monitoring thread
        self.control_system.add_thread(
            self.config.parallel_config.thread_configs['RT_SAFETY'].thread_type,
            self._safety_monitoring_loop
        )
        
        self.health_status["components"]["control_system"] = {
            "status": "healthy",
            "thread_count": len(self.control_system.threads),
            "queue_count": len(self.control_system.queues)
        }
        
        logger.info("Control system initialized")
    
    async def _initialize_monitoring(self):
        """Initialize continuous performance monitoring"""
        logger.info("Initializing performance monitoring...")
        
        monitoring_config = {
            'collector': {'collection_interval': 1.0},
            'analyzer': {'analysis_window': 1000},
            'alerts': {'default_suppression_minutes': 5},
            'analysis_interval': 5.0,
        }
        
        self.performance_monitor = ContinuousPerformanceMonitor(monitoring_config)
        self.performance_monitor.start_monitoring()
        
        self.health_status["components"]["performance_monitor"] = {
            "status": "healthy",
            "monitoring_active": True
        }
        
        logger.info("Performance monitoring initialized")
    
    async def _validate_performance(self):
        """Validate system performance against requirements"""
        logger.info("Running performance validation...")
        
        self.benchmark_system = RTPerformanceBenchmark(self.config.timing_requirements)
        
        # Run comprehensive benchmark
        benchmark_results = self.benchmark_system.run_benchmark_suite()
        
        # Check if performance requirements are met
        passed_tests = sum(1 for result in benchmark_results if result.success)
        total_tests = len(benchmark_results)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        self.performance_validated = pass_rate >= 0.8  # 80% pass rate threshold
        
        self.performance_stats = {
            'validation_time': datetime.now().isoformat(),
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'pass_rate': pass_rate,
            'performance_validated': self.performance_validated
        }
        
        if self.performance_validated:
            logger.info(f"Performance validation PASSED ({passed_tests}/{total_tests} tests)")
        else:
            logger.warning(f"Performance validation FAILED ({passed_tests}/{total_tests} tests)")
        
        self.health_status["components"]["benchmark_system"] = {
            "status": "healthy" if self.performance_validated else "degraded",
            "performance_stats": self.performance_stats
        }
    
    def _rt_control_loop(self):
        """Real-time control loop implementation"""
        # This would be the main control loop running at 1000 Hz
        # Placeholder implementation
        
        if self.performance_monitor:
            with self.performance_monitor.performance_context("rt_control_loop", "control_system"):
                # Simulate control work
                if self.optimized_policy and self.constraint_checker:
                    # Get sensor data (simulated)
                    state = np.random.randn(12).astype(np.float32)
                    
                    # Run policy inference
                    state_tensor = torch.from_numpy(state).unsqueeze(0)
                    with torch.no_grad():
                        action = self.optimized_policy.forward_optimized(state_tensor)
                    
                    # Check safety constraints
                    action_np = action.cpu().numpy().flatten()
                    is_safe, safety_score = self.constraint_checker.check_constraints_rt(state, action_np)
                    
                    # Record metrics
                    if self.performance_monitor:
                        self.performance_monitor.record_throughput("control_operations")
                        if not is_safe:
                            self.performance_monitor.record_safety_violation("constraint_violation")
                else:
                    # Minimal work if components not available
                    time.sleep(0.0001)  # 0.1ms
    
    def _safety_monitoring_loop(self):
        """Safety monitoring loop implementation"""
        # This would be the safety monitoring loop running at 2000 Hz
        # Placeholder implementation
        
        if self.performance_monitor:
            with self.performance_monitor.performance_context("safety_monitoring", "safety_system"):
                # Simulate safety monitoring work
                time.sleep(0.00005)  # 0.05ms
    
    async def start(self) -> bool:
        """
        Start the RT optimization system.
        
        Returns:
            True if startup successful, False otherwise
        """
        if not self.initialized:
            logger.error("System not initialized - call initialize() first")
            return False
        
        if self.running:
            logger.warning("System already running")
            return True
        
        logger.info("Starting RT optimization system...")
        
        try:
            # Start control system
            if self.control_system:
                self.control_system.start_all_threads()
            
            self.running = True
            self.health_status["status"] = "running"
            
            logger.info("RT optimization system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RT optimization system: {e}")
            return False
    
    async def stop(self):
        """Stop the RT optimization system"""
        if not self.running:
            return
        
        logger.info("Stopping RT optimization system...")
        shutdown_start = time.time()
        
        # Stop control system
        if self.control_system:
            self.control_system.stop_all_threads()
        
        # Stop monitoring
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        self.running = False
        self.shutdown_time = time.time() - shutdown_start
        self.health_status["status"] = "stopped"
        
        logger.info(f"RT optimization system stopped in {self.shutdown_time:.2f}s")
    
    async def cleanup(self):
        """Cleanup all resources"""
        logger.info("Cleaning up RT optimization system...")
        
        # Stop if running
        if self.running:
            await self.stop()
        
        # Cleanup GPU resources
        if self.gpu_optimizer:
            try:
                del self.gpu_optimizer
                torch.cuda.empty_cache()
            except Exception as e:
                logger.debug(f"GPU cleanup error: {e}")
        
        # Restore system settings if applied
        if self.system_optimizer:
            try:
                self.system_optimizer.restore_original_settings()
            except Exception as e:
                logger.debug(f"System restoration error: {e}")
        
        self.initialized = False
        self.health_status["status"] = "cleaned_up"
        
        logger.info("RT optimization system cleaned up")
    
    @asynccontextmanager
    async def rt_operation_context(self, operation_name: str):
        """Context manager for RT operation performance tracking"""
        if self.performance_monitor:
            async with self.performance_monitor.performance_context(operation_name):
                yield
        else:
            yield
    
    async def execute_rt_inference(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute real-time policy inference with safety checking.
        
        Args:
            state: Current system state
            
        Returns:
            Tuple of (action, metadata)
        """
        if not self.running:
            raise RuntimeError("System not running")
        
        metadata = {
            'timestamp': time.time(),
            'safe': True,
            'safety_score': 1.0,
            'inference_time_us': 0,
            'safety_check_time_us': 0
        }
        
        async with self.rt_operation_context("rt_inference"):
            # Policy inference
            if self.optimized_policy:
                start_time = time.perf_counter()
                
                state_tensor = torch.from_numpy(state).unsqueeze(0)
                with torch.no_grad():
                    action_tensor = self.optimized_policy.forward_optimized(state_tensor)
                    action = action_tensor.cpu().numpy().flatten()
                
                inference_time = time.perf_counter() - start_time
                metadata['inference_time_us'] = inference_time * 1_000_000
            else:
                # Fallback action
                action = np.zeros(8, dtype=np.float32)
            
            # Safety constraint checking
            if self.constraint_checker:
                start_time = time.perf_counter()
                
                is_safe, safety_score = self.constraint_checker.check_constraints_rt(state, action)
                
                safety_time = time.perf_counter() - start_time
                metadata['safety_check_time_us'] = safety_time * 1_000_000
                metadata['safe'] = is_safe
                metadata['safety_score'] = safety_score
                
                # If unsafe, return safe fallback action
                if not is_safe:
                    action = np.zeros_like(action)  # Stop action
                    logger.warning("Unsafe action detected - using fallback")
        
        return action, metadata
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'initialized': self.initialized,
            'running': self.running,
            'performance_validated': self.performance_validated,
            'startup_time_seconds': self.startup_time,
            'health_status': self.health_status,
            'performance_stats': self.performance_stats,
        }
        
        # Add component-specific status
        if self.performance_monitor:
            dashboard_data = self.performance_monitor.get_dashboard_data()
            status['monitoring'] = {
                'system_status': dashboard_data.get('system_status'),
                'active_alerts': dashboard_data.get('active_alert_count', 0),
                'last_update': dashboard_data.get('last_update'),
            }
        
        if self.control_system:
            system_stats = self.control_system.get_system_stats()
            status['control_system'] = {
                'thread_count': system_stats['system_info']['num_threads'],
                'emergency_stop': system_stats['system_info']['emergency_stop'],
                'uptime_seconds': system_stats['system_info']['uptime_seconds'],
            }
        
        return status
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        report = {
            'deployment_timestamp': datetime.now().isoformat(),
            'system_configuration': {
                'deployment_mode': self.config.deployment_mode,
                'timing_requirements': {
                    'max_execution_time_us': self.config.timing_requirements.max_execution_time_us,
                    'max_jitter_us': self.config.timing_requirements.max_jitter_us,
                    'min_frequency_hz': self.config.timing_requirements.min_frequency_hz,
                },
                'optimization_features': {
                    'system_optimization_enabled': self.config.enable_system_optimization,
                    'gpu_acceleration_enabled': self.config.enable_gpu_acceleration,
                    'continuous_monitoring_enabled': self.config.enable_continuous_monitoring,
                }
            },
            'system_status': self.get_system_status(),
        }
        
        # Add performance validation results
        if self.benchmark_system:
            performance_report = self.benchmark_system.generate_report()
            report['performance_validation'] = performance_report
        
        # Add optimization recommendations
        recommendations = []
        
        if not self.performance_validated:
            recommendations.append("Performance validation failed - review timing requirements")
        
        if self.health_status["components"].get("gpu_optimizer", {}).get("status") != "healthy":
            recommendations.append("GPU optimization unavailable - consider CPU-only deployment")
        
        if self.health_status["components"].get("system_optimizer", {}).get("status") != "healthy":
            recommendations.append("System optimization failed - may need root privileges")
        
        report['recommendations'] = recommendations
        
        return report
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


# Factory functions for common deployment scenarios
def create_development_system(base_policy: Optional[torch.nn.Module] = None) -> RTOptimizedSystem:
    """Create RT system optimized for development"""
    config = RTOptimizationConfig(
        deployment_mode="development",
        enable_system_optimization=False,  # Don't require root
        enable_continuous_monitoring=True,
        benchmark_on_startup=True,
        timing_requirements=TimingRequirements(
            max_execution_time_us=2000,  # More lenient for development
            max_jitter_us=100,
            min_frequency_hz=500.0
        )
    )
    
    return RTOptimizedSystem(config, base_policy)


def create_production_system(base_policy: torch.nn.Module) -> RTOptimizedSystem:
    """Create RT system optimized for production"""
    config = RTOptimizationConfig(
        deployment_mode="production",
        enable_system_optimization=True,
        enable_gpu_acceleration=True,
        enable_continuous_monitoring=True,
        enable_performance_validation=True,
        timing_requirements=TimingRequirements(
            max_execution_time_us=1000,
            max_jitter_us=50,
            min_frequency_hz=1000.0
        )
    )
    
    return RTOptimizedSystem(config, base_policy)


def create_edge_system(base_policy: torch.nn.Module) -> RTOptimizedSystem:
    """Create RT system optimized for edge deployment"""
    config = RTOptimizationConfig(
        deployment_mode="production",
        enable_system_optimization=True,
        enable_gpu_acceleration=False,  # May not have GPU on edge
        enable_continuous_monitoring=True,
        timing_requirements=TimingRequirements(
            max_execution_time_us=1500,  # Slightly more lenient for edge
            max_jitter_us=75,
            min_frequency_hz=1000.0
        )
    )
    
    # Configure for resource-constrained environment
    config.memory_config.max_total_memory_mb = 2048  # 2GB limit
    config.policy_optimization.enable_quantization = True
    config.policy_optimization.enable_pruning = True
    
    return RTOptimizedSystem(config, base_policy)


# Example usage
async def main():
    """Example usage of the integrated RT optimization system"""
    
    # Create a simple policy for testing
    base_policy = torch.nn.Sequential(
        torch.nn.Linear(12, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 8),
        torch.nn.Tanh()
    )
    
    # Create development system
    system = create_development_system(base_policy)
    
    try:
        # Initialize and start system
        async with system:
            
            # Run some RT inference operations
            for i in range(100):
                state = np.random.randn(12).astype(np.float32)
                
                action, metadata = await system.execute_rt_inference(state)
                
                if i % 20 == 0:
                    print(f"Inference {i}: "
                          f"time={metadata['inference_time_us']:.1f}μs, "
                          f"safe={metadata['safe']}, "
                          f"safety_score={metadata['safety_score']:.3f}")
                
                await asyncio.sleep(0.001)  # 1000 Hz
            
            # Get system status
            status = system.get_system_status()
            print(f"\nSystem Status:")
            print(f"  Initialized: {status['initialized']}")
            print(f"  Running: {status['running']}")
            print(f"  Performance Validated: {status['performance_validated']}")
            print(f"  Startup Time: {status['startup_time_seconds']:.2f}s")
            
            # Generate deployment report
            report = system.generate_deployment_report()
            print(f"\nDeployment Report Generated:")
            print(f"  Components: {len(status['health_status']['components'])}")
            print(f"  Recommendations: {len(report['recommendations'])}")
            
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run example
    asyncio.run(main())