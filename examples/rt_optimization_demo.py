"""
Real-time Optimization Demo

This script demonstrates the complete real-time optimization system
for Safe RL Human-Robot Shared Control, showing all major features
and performance capabilities.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn

from safe_rl_human_robot.src.optimization.rt_optimizer_integration import (
    create_development_system,
    create_production_system,
    RTOptimizationConfig,
    RTOptimizedSystem
)
from safe_rl_human_robot.src.optimization.benchmarking.rt_performance_benchmark import TimingRequirements

logger = logging.getLogger(__name__)


class SafeRLPolicyNetwork(nn.Module):
    """Example Safe RL policy network"""
    
    def __init__(self, state_dim: int = 12, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """Forward pass returning action probabilities"""
        return self.actor(state)
    
    def get_value(self, state):
        """Get state value from critic"""
        return self.critic(state)


def create_example_policy() -> nn.Module:
    """Create an example policy for demonstration"""
    policy = SafeRLPolicyNetwork(state_dim=12, action_dim=8, hidden_dim=128)
    
    # Initialize with reasonable weights
    for module in policy.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return policy


async def run_basic_demo():
    """Run basic RT optimization demo"""
    print("=" * 60)
    print("BASIC REAL-TIME OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Create example policy
    policy = create_example_policy()
    print(f"Created policy with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Create development system (more permissive for demo)
    system = create_development_system(policy)
    
    print("\nInitializing RT optimization system...")
    start_time = time.time()
    
    success = await system.initialize()
    if not success:
        print("❌ Failed to initialize system")
        return
    
    init_time = time.time() - start_time
    print(f"✅ System initialized in {init_time:.2f} seconds")
    
    # Start system
    print("\nStarting RT system...")
    success = await system.start()
    if not success:
        print("❌ Failed to start system")
        return
    
    print("✅ System started successfully")
    
    try:
        # Run RT inference demonstration
        print(f"\n{'='*40}")
        print("REAL-TIME INFERENCE DEMONSTRATION")
        print(f"{'='*40}")
        
        inference_times = []
        safety_scores = []
        
        num_iterations = 1000
        target_frequency = 1000  # 1000 Hz
        target_period = 1.0 / target_frequency
        
        print(f"Running {num_iterations} RT inference operations at {target_frequency} Hz...")
        
        start_time = time.time()
        
        for i in range(num_iterations):
            iteration_start = time.perf_counter()
            
            # Generate random state (robot + human positions, velocities, etc.)
            state = np.array([
                # Robot position (x, y, z)
                np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(0, 2),
                # Human position (x, y, z)  
                np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(0, 2),
                # Robot velocity (vx, vy, vz)
                np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5),
                # Human velocity (vx, vy, vz)
                np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(0, 0),
            ], dtype=np.float32)
            
            # Execute RT inference
            action, metadata = await system.execute_rt_inference(state)
            
            # Track performance
            total_time = metadata['inference_time_us'] + metadata['safety_check_time_us']
            inference_times.append(total_time)
            safety_scores.append(metadata['safety_score'])
            
            # Print progress
            if i % 200 == 0:
                print(f"  Iteration {i:4d}: {total_time:6.1f}μs "
                      f"(inference: {metadata['inference_time_us']:5.1f}μs, "
                      f"safety: {metadata['safety_check_time_us']:4.1f}μs, "
                      f"safe: {metadata['safe']}, score: {metadata['safety_score']:.3f})")
            
            # Maintain target frequency
            iteration_end = time.perf_counter()
            iteration_time = iteration_end - iteration_start
            sleep_time = target_period - iteration_time
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        total_time = time.time() - start_time
        actual_frequency = num_iterations / total_time
        
        # Analyze performance
        inference_times = np.array(inference_times)
        safety_scores = np.array(safety_scores)
        
        print(f"\n{'='*40}")
        print("PERFORMANCE RESULTS")
        print(f"{'='*40}")
        print(f"Actual frequency: {actual_frequency:.1f} Hz (target: {target_frequency} Hz)")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        print(f"\nTiming Statistics:")
        print(f"  Mean execution time: {np.mean(inference_times):6.1f} μs")
        print(f"  Median execution time: {np.median(inference_times):6.1f} μs")
        print(f"  P95 execution time: {np.percentile(inference_times, 95):6.1f} μs")
        print(f"  P99 execution time: {np.percentile(inference_times, 99):6.1f} μs")
        print(f"  Max execution time: {np.max(inference_times):6.1f} μs")
        print(f"  Jitter (std dev): {np.std(inference_times):6.1f} μs")
        
        print(f"\nSafety Statistics:")
        print(f"  Mean safety score: {np.mean(safety_scores):.4f}")
        print(f"  Min safety score: {np.min(safety_scores):.4f}")
        print(f"  Safety violations: {np.sum(safety_scores < 0.9)} / {len(safety_scores)}")
        
        # Check performance requirements
        timing_req = system.config.timing_requirements
        print(f"\nRequirements Compliance:")
        
        mean_time = np.mean(inference_times)
        p99_time = np.percentile(inference_times, 99)
        jitter = np.std(inference_times)
        
        print(f"  Mean time: {mean_time:6.1f}μs ({'✅' if mean_time < timing_req.max_execution_time_us else '❌'} < {timing_req.max_execution_time_us}μs)")
        print(f"  P99 time: {p99_time:6.1f}μs ({'✅' if p99_time < timing_req.percentile_99_us else '❌'} < {timing_req.percentile_99_us}μs)")
        print(f"  Jitter: {jitter:6.1f}μs ({'✅' if jitter < timing_req.max_jitter_us else '❌'} < {timing_req.max_jitter_us}μs)")
        print(f"  Frequency: {actual_frequency:6.1f}Hz ({'✅' if actual_frequency >= timing_req.min_frequency_hz else '❌'} >= {timing_req.min_frequency_hz}Hz)")
        
    finally:
        # Stop and cleanup
        print(f"\n{'='*40}")
        print("SYSTEM SHUTDOWN")
        print(f"{'='*40}")
        
        await system.stop()
        print("✅ System stopped")
        
        await system.cleanup()
        print("✅ System cleaned up")


async def run_comprehensive_demo():
    """Run comprehensive RT optimization demo with all features"""
    print("=" * 60)
    print("COMPREHENSIVE REAL-TIME OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Create more complex policy
    policy = SafeRLPolicyNetwork(state_dim=12, action_dim=8, hidden_dim=256)
    print(f"Created complex policy with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Create custom configuration
    config = RTOptimizationConfig(
        timing_requirements=TimingRequirements(
            max_execution_time_us=800,    # Stricter requirements
            max_jitter_us=30,
            min_frequency_hz=1000.0,
            sample_count=5000
        ),
        deployment_mode="production",
        enable_system_optimization=False,  # Skip for demo (no root required)
        enable_gpu_acceleration=True,
        enable_continuous_monitoring=True,
        enable_performance_validation=True,
        benchmark_on_startup=True,
    )
    
    system = RTOptimizedSystem(config, policy)
    
    print("\nInitializing comprehensive RT system...")
    success = await system.initialize()
    if not success:
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized with all optimizations")
    
    # Get system status
    status = system.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Performance validated: {status['performance_validated']}")
    print(f"  Components initialized: {len(status['health_status']['components'])}")
    
    for component, health in status['health_status']['components'].items():
        status_icon = "✅" if health['status'] == 'healthy' else "⚠️" if health['status'] == 'degraded' else "❌"
        print(f"    {component}: {status_icon} {health['status']}")
    
    # Start system
    success = await system.start()
    if not success:
        print("❌ Failed to start system")
        return
    
    try:
        # Run stress test
        print(f"\n{'='*50}")
        print("STRESS TEST - HIGH FREQUENCY OPERATION")
        print(f"{'='*50}")
        
        stress_iterations = 2000
        stress_frequency = 1000  # 1000 Hz
        
        print(f"Running {stress_iterations} operations at {stress_frequency} Hz...")
        
        performance_data = []
        violation_count = 0
        
        start_time = time.time()
        
        for i in range(stress_iterations):
            # Create more complex state with potential safety issues
            robot_pos = np.random.uniform(-3, 3, 3)
            human_pos = np.random.uniform(-3, 3, 3)
            
            # Sometimes create close proximity scenarios
            if i % 100 == 0:
                # Force close proximity
                human_pos = robot_pos + np.random.uniform(-0.8, 0.8, 3)
            
            state = np.concatenate([
                robot_pos,
                human_pos,
                np.random.uniform(-1.5, 1.5, 3),  # Robot velocity
                np.random.uniform(-0.3, 0.3, 3),  # Human velocity
            ]).astype(np.float32)
            
            # Execute RT inference
            action, metadata = await system.execute_rt_inference(state)
            
            performance_data.append(metadata)
            
            if not metadata['safe']:
                violation_count += 1
                if violation_count <= 3:  # Show first few violations
                    distance = np.linalg.norm(robot_pos - human_pos)
                    print(f"  ⚠️ Safety violation {violation_count}: distance={distance:.2f}m, score={metadata['safety_score']:.3f}")
            
            # Progress indicator
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                print(f"    {i} operations complete, rate: {rate:.1f} Hz")
            
            # Maintain frequency
            await asyncio.sleep(1.0 / stress_frequency)
        
        total_time = time.time() - start_time
        actual_rate = stress_iterations / total_time
        
        # Analyze stress test results
        timing_data = [d['inference_time_us'] + d['safety_check_time_us'] for d in performance_data]
        safety_data = [d['safety_score'] for d in performance_data]
        
        print(f"\n{'='*50}")
        print("STRESS TEST RESULTS")
        print(f"{'='*50}")
        print(f"Operations completed: {stress_iterations}")
        print(f"Actual rate: {actual_rate:.1f} Hz")
        print(f"Safety violations: {violation_count} ({violation_count/stress_iterations*100:.1f}%)")
        
        timing_array = np.array(timing_data)
        print(f"\nStress Test Timing:")
        print(f"  Mean: {np.mean(timing_array):6.1f} μs")
        print(f"  P95:  {np.percentile(timing_array, 95):6.1f} μs")
        print(f"  P99:  {np.percentile(timing_array, 99):6.1f} μs")
        print(f"  P99.9:{np.percentile(timing_array, 99.9):6.1f} μs")
        print(f"  Max:  {np.max(timing_array):6.1f} μs")
        
        print(f"\nSafety Performance:")
        print(f"  Mean safety score: {np.mean(safety_data):.4f}")
        print(f"  Min safety score:  {np.min(safety_data):.4f}")
        
        # Generate comprehensive report
        print(f"\n{'='*50}")
        print("GENERATING DEPLOYMENT REPORT")
        print(f"{'='*50}")
        
        report = system.generate_deployment_report()
        
        # Save report
        report_file = Path("rt_optimization_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Deployment report saved to: {report_file}")
        print(f"Report summary:")
        print(f"  Deployment mode: {report['system_configuration']['deployment_mode']}")
        print(f"  Performance validated: {report['system_status']['performance_validated']}")
        print(f"  Components: {len(report['system_status']['health_status']['components'])}")
        print(f"  Recommendations: {len(report['recommendations'])}")
        
        if report['recommendations']:
            print(f"  Key recommendations:")
            for rec in report['recommendations'][:3]:  # Show first 3
                print(f"    • {rec}")
        
    finally:
        # Cleanup
        await system.cleanup()
        print("\n✅ Comprehensive demo completed successfully")


async def run_production_simulation():
    """Simulate production deployment scenario"""
    print("=" * 60)
    print("PRODUCTION DEPLOYMENT SIMULATION")
    print("=" * 60)
    
    # Create production-grade policy
    policy = SafeRLPolicyNetwork(state_dim=12, action_dim=8, hidden_dim=512)
    
    # Create production system
    system = create_production_system(policy)
    
    print("Initializing production system (this may take longer)...")
    
    success = await system.initialize()
    if not success:
        print("❌ Production system initialization failed")
        return
    
    # Check if system meets production requirements
    status = system.get_system_status()
    
    if not status['performance_validated']:
        print("⚠️ Warning: System did not pass performance validation")
        print("This system may not be suitable for production deployment")
    else:
        print("✅ Production system validated and ready")
    
    success = await system.start()
    if not success:
        print("❌ Failed to start production system")
        return
    
    try:
        print("\nRunning production simulation...")
        print("Simulating 24/7 operation for 30 seconds...")
        
        # Simulate continuous operation
        start_time = time.time()
        operations = 0
        errors = 0
        
        while time.time() - start_time < 30:  # 30 second simulation
            try:
                # Create realistic state
                state = np.random.randn(12).astype(np.float32)
                
                # Execute inference
                action, metadata = await system.execute_rt_inference(state)
                operations += 1
                
                if not metadata['safe']:
                    errors += 1
                
                # Production frequency (1000 Hz)
                await asyncio.sleep(0.001)
                
            except Exception as e:
                errors += 1
                logger.error(f"Production error: {e}")
        
        duration = time.time() - start_time
        rate = operations / duration
        error_rate = errors / operations if operations > 0 else 0
        
        print(f"\nProduction Simulation Results:")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Operations: {operations}")
        print(f"  Rate: {rate:.1f} Hz")
        print(f"  Errors: {errors} ({error_rate*100:.3f}%)")
        print(f"  Reliability: {(1-error_rate)*100:.3f}%")
        
        if error_rate < 0.001:  # Less than 0.1% error rate
            print("✅ Production system meets reliability requirements")
        else:
            print("⚠️ Warning: Error rate exceeds production thresholds")
    
    finally:
        await system.cleanup()
        print("✅ Production simulation completed")


async def main():
    """Run all demonstrations"""
    print("🚀 SAFE RL REAL-TIME OPTIMIZATION DEMONSTRATION")
    print("This demo showcases comprehensive real-time optimization capabilities\n")
    
    try:
        # Run basic demo
        await run_basic_demo()
        
        print("\n" + "="*80 + "\n")
        
        # Run comprehensive demo
        await run_comprehensive_demo()
        
        print("\n" + "="*80 + "\n")
        
        # Run production simulation
        await run_production_simulation()
        
        print(f"\n{'🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY! 🎉':^80}")
        print(f"{'The real-time optimization system is ready for deployment.':^80}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress some verbose logs
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Run demonstrations
    asyncio.run(main())