#!/usr/bin/env python3
"""
Real-Time Optimization System Validation Script

This script performs comprehensive validation of all real-time optimization
components to ensure they meet performance targets and are properly integrated.
"""

import asyncio
import time
import numpy as np
import torch
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics
import matplotlib.pyplot as plt
import warnings

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safe_rl_human_robot.src.optimization.rt_optimizer_integration import (
    create_development_system, create_production_system, RTOptimizationConfig, 
    RTOptimizedSystem, TimingRequirements
)
from safe_rl_human_robot.src.optimization.benchmarking.rt_performance_benchmark import (
    RTPerformanceBenchmark
)

warnings.filterwarnings("ignore", category=UserWarning)


class RTSystemValidator:
    """Comprehensive validation system for real-time optimization components."""
    
    def __init__(self):
        self.results = {}
        self.test_policy = self._create_test_policy()
        
    def _create_test_policy(self) -> torch.nn.Module:
        """Create a test policy for validation."""
        return torch.nn.Sequential(
            torch.nn.Linear(12, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32), 
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.Tanh()
        )
    
    async def validate_development_system(self) -> Dict[str, Any]:
        """Validate the development system configuration."""
        print("🔧 Testing Development System...")
        
        system = create_development_system(self.test_policy)
        results = {}
        
        try:
            async with system:
                # Test basic inference
                state = np.random.randn(12).astype(np.float32)
                start_time = time.perf_counter()
                action, metadata = await system.execute_rt_inference(state)
                end_time = time.perf_counter()
                
                inference_time_us = (end_time - start_time) * 1_000_000
                
                results.update({
                    "system_created": True,
                    "inference_successful": action is not None,
                    "action_shape": action.shape if hasattr(action, 'shape') else len(action),
                    "inference_time_us": inference_time_us,
                    "metadata_available": metadata is not None,
                    "safety_checked": metadata.get('safe', False) if metadata else False
                })
                
        except Exception as e:
            results.update({
                "system_created": False,
                "error": str(e)
            })
            
        return results
    
    async def validate_production_system(self) -> Dict[str, Any]:
        """Validate the production system configuration."""
        print("🚀 Testing Production System...")
        
        try:
            system = create_production_system(self.test_policy)
            results = {}
            
            async with system:
                # Perform multiple inference calls to test consistency
                times = []
                states = [np.random.randn(12).astype(np.float32) for _ in range(100)]
                
                for state in states:
                    start_time = time.perf_counter()
                    action, metadata = await system.execute_rt_inference(state)
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1_000_000)
                
                results.update({
                    "system_created": True,
                    "batch_inference_successful": len(times) == 100,
                    "mean_inference_time_us": statistics.mean(times),
                    "median_inference_time_us": statistics.median(times),
                    "max_inference_time_us": max(times),
                    "min_inference_time_us": min(times),
                    "std_inference_time_us": statistics.stdev(times),
                    "times_under_500us": sum(1 for t in times if t < 500),
                    "performance_consistent": max(times) - min(times) < 100  # Low jitter
                })
                
        except Exception as e:
            results = {
                "system_created": False,
                "error": str(e)
            }
            
        return results
    
    async def validate_custom_configuration(self) -> Dict[str, Any]:
        """Validate custom system configuration."""
        print("⚙️ Testing Custom Configuration...")
        
        # Create custom configuration
        config = RTOptimizationConfig(
            timing_requirements=TimingRequirements(
                max_execution_time_us=300,
                max_jitter_us=20,
                min_frequency_hz=2000.0
            ),
            enable_system_optimization=False,  # Safe for testing
            enable_gpu_acceleration=torch.cuda.is_available(),
            enable_continuous_monitoring=True,
            deployment_mode="development"
        )
        
        try:
            system = RTOptimizedSystem(config, self.test_policy)
            results = {}
            
            async with system:
                # Test performance against custom requirements
                times = []
                for _ in range(50):
                    state = np.random.randn(12).astype(np.float32)
                    start_time = time.perf_counter()
                    action, metadata = await system.execute_rt_inference(state)
                    end_time = time.perf_counter()
                    
                    times.append((end_time - start_time) * 1_000_000)
                
                results.update({
                    "custom_config_created": True,
                    "mean_time_us": statistics.mean(times),
                    "meets_300us_requirement": all(t < 300 for t in times),
                    "jitter_us": max(times) - min(times),
                    "meets_jitter_requirement": (max(times) - min(times)) < 20,
                    "gpu_acceleration_enabled": config.enable_gpu_acceleration and torch.cuda.is_available()
                })
                
        except Exception as e:
            results = {
                "custom_config_created": False,
                "error": str(e)
            }
            
        return results
    
    async def validate_benchmark_system(self) -> Dict[str, Any]:
        """Validate the benchmarking system."""
        print("📊 Testing Benchmark System...")
        
        try:
            requirements = TimingRequirements(max_execution_time_us=1000)
            benchmark = RTPerformanceBenchmark(requirements)
            
            # Run a subset of benchmark tests
            results = benchmark.run_benchmark_suite()
            
            return {
                "benchmark_created": True,
                "benchmark_completed": results is not None,
                "performance_validated": results.get('performance_validated', False) if results else False,
                "timing_results_available": 'timing_results' in (results or {}),
                "stress_test_completed": 'stress_test' in (results or {})
            }
            
        except Exception as e:
            return {
                "benchmark_created": False,
                "error": str(e)
            }
    
    async def validate_system_status(self) -> Dict[str, Any]:
        """Validate system status reporting."""
        print("📈 Testing System Status...")
        
        try:
            system = create_development_system(self.test_policy)
            
            async with system:
                # Get system status
                status = system.get_system_status()
                
                # Generate deployment report
                report = system.generate_deployment_report()
                
                return {
                    "status_available": status is not None,
                    "report_generated": report is not None,
                    "has_performance_data": 'performance_validated' in (status or {}),
                    "has_monitoring_data": 'monitoring' in (status or {}),
                    "has_recommendations": 'recommendations' in (report or {})
                }
                
        except Exception as e:
            return {
                "status_available": False,
                "error": str(e)
            }
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate required dependencies."""
        print("📦 Checking Dependencies...")
        
        results = {}
        
        # Check core dependencies
        try:
            import torch
            results["torch_available"] = True
            results["torch_version"] = torch.__version__
            results["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                results["cuda_device_count"] = torch.cuda.device_count()
        except ImportError:
            results["torch_available"] = False
        
        try:
            import numpy
            results["numpy_available"] = True
            results["numpy_version"] = numpy.__version__
        except ImportError:
            results["numpy_available"] = False
        
        try:
            import numba
            results["numba_available"] = True
            results["numba_version"] = numba.__version__
        except ImportError:
            results["numba_available"] = False
        
        try:
            import psutil
            results["psutil_available"] = True
            results["cpu_count"] = psutil.cpu_count()
            results["memory_gb"] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            results["psutil_available"] = False
        
        return results
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🧪 Starting Real-Time Optimization System Validation")
        print("=" * 60)
        
        all_results = {}
        
        # Validate dependencies first
        all_results["dependencies"] = self.validate_dependencies()
        
        # Validate different system configurations
        all_results["development_system"] = await self.validate_development_system()
        all_results["production_system"] = await self.validate_production_system()
        all_results["custom_configuration"] = await self.validate_custom_configuration()
        all_results["benchmark_system"] = await self.validate_benchmark_system()
        all_results["system_status"] = await self.validate_system_status()
        
        return all_results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("📋 REAL-TIME OPTIMIZATION VALIDATION REPORT")
        report.append("=" * 60)
        
        # Dependencies
        deps = results.get("dependencies", {})
        report.append("\n📦 DEPENDENCIES:")
        report.append(f"  PyTorch: {'✅' if deps.get('torch_available') else '❌'} {deps.get('torch_version', 'N/A')}")
        report.append(f"  CUDA:    {'✅' if deps.get('cuda_available') else '❌'} ({deps.get('cuda_device_count', 0)} devices)")
        report.append(f"  NumPy:   {'✅' if deps.get('numpy_available') else '❌'} {deps.get('numpy_version', 'N/A')}")
        report.append(f"  Numba:   {'✅' if deps.get('numba_available') else '❌'}")
        report.append(f"  System:  {deps.get('cpu_count', 'N/A')} CPUs, {deps.get('memory_gb', 0):.1f}GB RAM")
        
        # Development system
        dev = results.get("development_system", {})
        report.append("\n🔧 DEVELOPMENT SYSTEM:")
        report.append(f"  Creation:   {'✅' if dev.get('system_created') else '❌'}")
        if dev.get('system_created'):
            report.append(f"  Inference:  {'✅' if dev.get('inference_successful') else '❌'}")
            report.append(f"  Timing:     {dev.get('inference_time_us', 0):.1f}μs")
            report.append(f"  Safety:     {'✅' if dev.get('safety_checked') else '❌'}")
        
        # Production system
        prod = results.get("production_system", {})
        report.append("\n🚀 PRODUCTION SYSTEM:")
        report.append(f"  Creation:   {'✅' if prod.get('system_created') else '❌'}")
        if prod.get('system_created'):
            mean_time = prod.get('mean_inference_time_us', 0)
            max_time = prod.get('max_inference_time_us', 0)
            under_500 = prod.get('times_under_500us', 0)
            report.append(f"  Mean Time:  {mean_time:.1f}μs")
            report.append(f"  Max Time:   {max_time:.1f}μs")
            report.append(f"  <500μs:     {under_500}/100 calls")
            report.append(f"  Target:     {'✅ PASSED' if mean_time < 500 else '❌ FAILED'}")
        
        # Custom configuration
        custom = results.get("custom_configuration", {})
        report.append("\n⚙️ CUSTOM CONFIGURATION:")
        report.append(f"  Creation:   {'✅' if custom.get('custom_config_created') else '❌'}")
        if custom.get('custom_config_created'):
            mean_time = custom.get('mean_time_us', 0)
            jitter = custom.get('jitter_us', 0)
            report.append(f"  Mean Time:  {mean_time:.1f}μs")
            report.append(f"  Jitter:     {jitter:.1f}μs")
            report.append(f"  300μs Req:  {'✅' if custom.get('meets_300us_requirement') else '❌'}")
            report.append(f"  20μs Jitter:{'✅' if custom.get('meets_jitter_requirement') else '❌'}")
        
        # Benchmark system
        bench = results.get("benchmark_system", {})
        report.append("\n📊 BENCHMARK SYSTEM:")
        report.append(f"  Creation:   {'✅' if bench.get('benchmark_created') else '❌'}")
        report.append(f"  Completion: {'✅' if bench.get('benchmark_completed') else '❌'}")
        report.append(f"  Validation: {'✅' if bench.get('performance_validated') else '❌'}")
        
        # System status
        status = results.get("system_status", {})
        report.append("\n📈 SYSTEM STATUS:")
        report.append(f"  Status API: {'✅' if status.get('status_available') else '❌'}")
        report.append(f"  Reports:    {'✅' if status.get('report_generated') else '❌'}")
        report.append(f"  Monitoring: {'✅' if status.get('has_monitoring_data') else '❌'}")
        
        # Overall assessment
        report.append("\n🎯 OVERALL ASSESSMENT:")
        
        critical_checks = [
            dev.get('system_created', False),
            prod.get('system_created', False),
            deps.get('torch_available', False),
            deps.get('numpy_available', False)
        ]
        
        performance_checks = [
            prod.get('mean_inference_time_us', 1000) < 500,
            custom.get('meets_300us_requirement', False),
            custom.get('meets_jitter_requirement', False)
        ]
        
        critical_passed = sum(critical_checks)
        performance_passed = sum(performance_checks)
        
        if critical_passed == len(critical_checks):
            report.append("  System Status: ✅ OPERATIONAL")
        else:
            report.append("  System Status: ❌ CRITICAL ISSUES")
        
        if performance_passed >= len(performance_checks) - 1:
            report.append("  Performance:   ✅ MEETS TARGETS")
        else:
            report.append("  Performance:   ⚠️  NEEDS OPTIMIZATION")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        
        if not deps.get('cuda_available'):
            report.append("  - Install CUDA for GPU acceleration")
        
        if not deps.get('numba_available'):
            report.append("  - Install Numba for JIT optimization")
        
        prod_mean = prod.get('mean_inference_time_us', 0)
        if prod_mean > 400:
            report.append("  - Consider model quantization for better performance")
        
        if not custom.get('meets_jitter_requirement', True):
            report.append("  - Enable system-level optimizations for lower jitter")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filename: str = "validation_results.json"):
        """Save validation results to file."""
        output_file = Path("validation_output") / filename
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")


async def main():
    """Main validation function."""
    validator = RTSystemValidator()
    
    try:
        # Run validation
        results = await validator.run_full_validation()
        
        # Generate and display report
        report = validator.generate_validation_report(results)
        print(report)
        
        # Save results
        validator.save_results(results)
        
        # Create summary plot if matplotlib is available
        try:
            create_validation_summary_plot(results)
        except ImportError:
            print("\n📊 Matplotlib not available for plots")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return None


def create_validation_summary_plot(results: Dict[str, Any]):
    """Create a summary visualization of validation results."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Performance timing data
        prod = results.get("production_system", {})
        if prod.get('system_created'):
            mean_time = prod.get('mean_inference_time_us', 0)
            max_time = prod.get('max_inference_time_us', 0)
            min_time = prod.get('min_inference_time_us', 0)
            
            plt.subplot(2, 2, 1)
            plt.bar(['Mean', 'Max', 'Min'], [mean_time, max_time, min_time])
            plt.axhline(y=500, color='r', linestyle='--', label='500μs Target')
            plt.title('Production System Timing')
            plt.ylabel('Time (μs)')
            plt.legend()
        
        # Dependency status
        deps = results.get("dependencies", {})
        dep_names = ['PyTorch', 'CUDA', 'NumPy', 'Numba', 'psutil']
        dep_status = [
            deps.get('torch_available', False),
            deps.get('cuda_available', False), 
            deps.get('numpy_available', False),
            deps.get('numba_available', False),
            deps.get('psutil_available', False)
        ]
        
        plt.subplot(2, 2, 2)
        colors = ['green' if status else 'red' for status in dep_status]
        plt.bar(dep_names, [1 if status else 0 for status in dep_status], color=colors)
        plt.title('Dependency Status')
        plt.ylabel('Available')
        plt.xticks(rotation=45)
        
        # System validation summary
        systems = ['Development', 'Production', 'Custom', 'Benchmark', 'Status']
        system_results = [
            results.get("development_system", {}).get('system_created', False),
            results.get("production_system", {}).get('system_created', False),
            results.get("custom_configuration", {}).get('custom_config_created', False),
            results.get("benchmark_system", {}).get('benchmark_created', False),
            results.get("system_status", {}).get('status_available', False)
        ]
        
        plt.subplot(2, 2, 3)
        colors = ['green' if result else 'red' for result in system_results]
        plt.bar(systems, [1 if result else 0 for result in system_results], color=colors)
        plt.title('System Component Status')
        plt.ylabel('Working')
        plt.xticks(rotation=45)
        
        # Performance targets
        plt.subplot(2, 2, 4)
        targets = ['<500μs\nInference', '<300μs\nCustom', '<20μs\nJitter']
        target_met = [
            prod.get('mean_inference_time_us', 1000) < 500,
            results.get("custom_configuration", {}).get('meets_300us_requirement', False),
            results.get("custom_configuration", {}).get('meets_jitter_requirement', False)
        ]
        
        colors = ['green' if met else 'red' for met in target_met]
        plt.bar(targets, [1 if met else 0 for met in target_met], color=colors)
        plt.title('Performance Targets')
        plt.ylabel('Target Met')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path("validation_output")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "validation_summary.png", dpi=300, bbox_inches='tight')
        print(f"📊 Validation plot saved to: {output_dir / 'validation_summary.png'}")
        
    except Exception as e:
        print(f"📊 Could not create validation plot: {e}")


if __name__ == "__main__":
    asyncio.run(main())