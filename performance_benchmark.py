#!/usr/bin/env python3
"""
Performance Benchmarking for Safe RL System
"""

import sys
import os
import time
import psutil
import threading
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@dataclass
class PerformanceMetrics:
    """Performance metrics dataclass."""
    startup_time: float
    import_time: float
    memory_mb: float
    cpu_percent: float
    module_availability: Dict[str, bool]
    validation_results: Dict[str, Any]

class PerformanceBenchmark:
    """Comprehensive performance benchmarking."""
    
    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()
        
    def benchmark_startup_performance(self) -> Dict[str, float]:
        """Benchmark system startup performance."""
        print("🚀 Benchmarking Startup Performance...")
        
        # Cold start test
        start_time = time.time()
        
        try:
            import safe_rl_human_robot
            import_time = time.time() - start_time
            
            # Get system status
            status = safe_rl_human_robot.get_system_status()
            total_time = time.time() - start_time
            
            results = {
                'import_time': import_time,
                'total_startup_time': total_time,
                'status_check_time': total_time - import_time
            }
            
            print(f"   📊 Import Time: {import_time:.3f}s")
            print(f"   📊 Total Startup: {total_time:.3f}s")
            print(f"   📊 Status Check: {results['status_check_time']:.3f}s")
            
            # Performance targets
            if import_time < 5.0:
                print(f"   ✅ Import time target MET: {import_time:.3f}s < 5.0s")
            else:
                print(f"   ❌ Import time target MISSED: {import_time:.3f}s >= 5.0s")
                
            return results
            
        except Exception as e:
            print(f"   ❌ Startup benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_memory_usage(self) -> Dict[str, float]:
        """Benchmark memory usage patterns."""
        print("\n💾 Benchmarking Memory Usage...")
        
        try:
            # Initial memory
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            print(f"   📊 Initial Memory: {initial_memory:.2f} MB")
            
            # Import all available modules
            import safe_rl_human_robot
            from safe_rl_human_robot import src
            
            post_import_memory = self.process.memory_info().rss / 1024 / 1024
            import_overhead = post_import_memory - initial_memory
            
            print(f"   📊 Post-Import Memory: {post_import_memory:.2f} MB")
            print(f"   📊 Import Overhead: {import_overhead:.2f} MB")
            
            # Memory efficiency check
            if post_import_memory < 1000:  # Less than 1GB
                print(f"   ✅ Memory usage ACCEPTABLE: {post_import_memory:.2f} MB < 1000 MB")
            else:
                print(f"   ⚠️  Memory usage HIGH: {post_import_memory:.2f} MB >= 1000 MB")
            
            return {
                'initial_memory_mb': initial_memory,
                'post_import_memory_mb': post_import_memory,
                'import_overhead_mb': import_overhead
            }
            
        except Exception as e:
            print(f"   ❌ Memory benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_cpu_performance(self) -> Dict[str, float]:
        """Benchmark CPU usage during operations."""
        print("\n🖥️  Benchmarking CPU Performance...")
        
        try:
            # Monitor CPU usage during system operations
            cpu_samples = []
            
            def monitor_cpu():
                for _ in range(10):  # Monitor for 1 second
                    cpu_samples.append(psutil.cpu_percent(interval=0.1))
            
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # Perform typical operations
            import safe_rl_human_robot
            status = safe_rl_human_robot.get_system_status()
            dependencies = safe_rl_human_robot.check_dependencies()
            
            monitor_thread.join()
            
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
            
            print(f"   📊 Average CPU Usage: {avg_cpu:.2f}%")
            print(f"   📊 Peak CPU Usage: {max_cpu:.2f}%")
            
            # CPU efficiency check
            if avg_cpu < 1.0:  # Less than 1% average
                print(f"   ✅ CPU usage EFFICIENT: {avg_cpu:.2f}% < 1.0%")
            else:
                print(f"   ⚠️  CPU usage ELEVATED: {avg_cpu:.2f}% >= 1.0%")
            
            return {
                'average_cpu_percent': avg_cpu,
                'peak_cpu_percent': max_cpu,
                'cpu_samples': cpu_samples
            }
            
        except Exception as e:
            print(f"   ❌ CPU benchmark failed: {e}")
            return {'error': str(e)}
    
    def benchmark_concurrent_performance(self) -> Dict[str, Any]:
        """Test performance under concurrent load."""
        print("\n⚡ Benchmarking Concurrent Performance...")
        
        try:
            def test_concurrent_import():
                """Test concurrent imports."""
                import safe_rl_human_robot
                return safe_rl_human_robot.get_system_status()
            
            start_time = time.time()
            
            # Run concurrent imports
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(test_concurrent_import) for _ in range(10)]
                results = [future.result() for future in futures]
            
            concurrent_time = time.time() - start_time
            
            print(f"   📊 Concurrent Operations: 10 imports in {concurrent_time:.3f}s")
            print(f"   📊 Average Time per Operation: {concurrent_time/10:.3f}s")
            
            # Concurrent performance check
            if concurrent_time < 10.0:  # 10 operations in less than 10s
                print(f"   ✅ Concurrent performance GOOD: {concurrent_time:.3f}s < 10.0s")
            else:
                print(f"   ⚠️  Concurrent performance SLOW: {concurrent_time:.3f}s >= 10.0s")
            
            return {
                'total_concurrent_time': concurrent_time,
                'average_operation_time': concurrent_time / 10,
                'operations_completed': len(results),
                'success_rate': len([r for r in results if r]) / len(results)
            }
            
        except Exception as e:
            print(f"   ❌ Concurrent benchmark failed: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_benchmark(self) -> PerformanceMetrics:
        """Run all performance benchmarks."""
        print("🎯 Safe RL System Performance Benchmark")
        print("=" * 80)
        
        # Run individual benchmarks
        startup_results = self.benchmark_startup_performance()
        memory_results = self.benchmark_memory_usage()
        cpu_results = self.benchmark_cpu_performance()
        concurrent_results = self.benchmark_concurrent_performance()
        
        # Get system status
        try:
            import safe_rl_human_robot
            status = safe_rl_human_robot.get_system_status()
            dependencies = safe_rl_human_robot.check_dependencies()
        except Exception as e:
            status = {'error': str(e)}
            dependencies = {'error': str(e)}
        
        # Compile metrics
        metrics = PerformanceMetrics(
            startup_time=startup_results.get('total_startup_time', 0),
            import_time=startup_results.get('import_time', 0),
            memory_mb=memory_results.get('post_import_memory_mb', 0),
            cpu_percent=cpu_results.get('average_cpu_percent', 0),
            module_availability=status,
            validation_results={
                'startup': startup_results,
                'memory': memory_results,
                'cpu': cpu_results,
                'concurrent': concurrent_results,
                'dependencies': dependencies
            }
        )
        
        return metrics
    
    def generate_report(self, metrics: PerformanceMetrics):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 80)
        print("📊 PERFORMANCE BENCHMARK REPORT")
        print("=" * 80)
        
        print(f"\n🚀 STARTUP PERFORMANCE:")
        print(f"   • Import Time: {metrics.import_time:.3f}s")
        print(f"   • Total Startup: {metrics.startup_time:.3f}s")
        
        print(f"\n💾 MEMORY PERFORMANCE:")
        print(f"   • Memory Usage: {metrics.memory_mb:.2f} MB")
        
        print(f"\n🖥️  CPU PERFORMANCE:")
        print(f"   • Average CPU Usage: {metrics.cpu_percent:.2f}%")
        
        print(f"\n⚙️  MODULE AVAILABILITY:")
        for key, value in metrics.module_availability.items():
            if key.endswith('_available'):
                status = "✅ Available" if value else "❌ Unavailable"
                print(f"   • {key.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 PERFORMANCE TARGETS:")
        
        # Startup target
        startup_status = "✅ MET" if metrics.startup_time < 5.0 else "❌ MISSED"
        print(f"   • Startup Time < 5.0s: {startup_status} ({metrics.startup_time:.3f}s)")
        
        # Memory target  
        memory_status = "✅ MET" if metrics.memory_mb < 1000 else "❌ MISSED"
        print(f"   • Memory Usage < 1000MB: {memory_status} ({metrics.memory_mb:.2f}MB)")
        
        # CPU target
        cpu_status = "✅ MET" if metrics.cpu_percent < 1.0 else "❌ MISSED"
        print(f"   • CPU Usage < 1.0%: {cpu_status} ({metrics.cpu_percent:.2f}%)")
        
        print(f"\n🏆 OVERALL ASSESSMENT:")
        
        targets_met = [
            metrics.startup_time < 5.0,
            metrics.memory_mb < 1000,
            metrics.cpu_percent < 1.0
        ]
        
        success_rate = sum(targets_met) / len(targets_met) * 100
        
        if success_rate >= 100:
            print("   🟢 EXCELLENT - All performance targets met!")
        elif success_rate >= 67:
            print("   🟡 GOOD - Most performance targets met")
        else:
            print("   🔴 NEEDS IMPROVEMENT - Performance targets missed")
        
        print(f"   📈 Performance Score: {success_rate:.1f}%")
        
        return success_rate

def main():
    """Main benchmark execution."""
    benchmark = PerformanceBenchmark()
    
    try:
        # Run comprehensive benchmark
        metrics = benchmark.run_comprehensive_benchmark()
        
        # Generate report
        score = benchmark.generate_report(metrics)
        
        # Save results
        results_file = 'performance_benchmark_results.json'
        with open(results_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return score >= 67  # Success if 67% or higher
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)