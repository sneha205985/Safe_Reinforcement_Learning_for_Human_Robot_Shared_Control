#!/usr/bin/env python3
"""
Simplified Real-Time Optimization System Validation Script

This script validates the structure and basic functionality of the RT optimization
system without requiring PyTorch or other heavy dependencies.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class RTSystemStructureValidator:
    """Validates the RT optimization system structure and components."""
    
    def __init__(self):
        self.project_root = project_root
        self.results = {}
        
    def validate_file_structure(self) -> Dict[str, Any]:
        """Validate that all required files exist."""
        print("📁 Checking File Structure...")
        
        expected_files = [
            "safe_rl_human_robot/src/optimization/rt_optimizer_integration.py",
            "safe_rl_human_robot/src/optimization/inference/optimized_policy.py", 
            "safe_rl_human_robot/src/optimization/safety_rt/fast_constraint_checker.py",
            "safe_rl_human_robot/src/optimization/memory/rt_memory_manager.py",
            "safe_rl_human_robot/src/optimization/parallel/rt_control_system.py",
            "safe_rl_human_robot/src/optimization/gpu/cuda_optimizer.py",
            "safe_rl_human_robot/src/optimization/system/rt_system_config.py",
            "safe_rl_human_robot/src/optimization/benchmarking/rt_performance_benchmark.py",
            "safe_rl_human_robot/src/optimization/monitoring/continuous_performance_monitor.py",
            "examples/rt_optimization_demo.py",
            "safe_rl_human_robot/src/optimization/README.md"
        ]
        
        results = {}
        for file_path in expected_files:
            full_path = self.project_root / file_path
            exists = full_path.exists()
            results[file_path] = {
                "exists": exists,
                "size_bytes": full_path.stat().st_size if exists else 0
            }
        
        return results
    
    def validate_imports(self) -> Dict[str, Any]:
        """Validate that Python files have proper import structure."""
        print("🔍 Checking Import Structure...")
        
        files_to_check = [
            "safe_rl_human_robot/src/optimization/rt_optimizer_integration.py",
            "safe_rl_human_robot/src/optimization/inference/optimized_policy.py",
            "safe_rl_human_robot/src/optimization/benchmarking/rt_performance_benchmark.py"
        ]
        
        results = {}
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    # Check for key imports and classes
                    has_imports = "import" in content
                    has_classes = "class " in content
                    has_async = "async def" in content
                    has_typing = "typing" in content or "Type" in content
                    
                    results[file_path] = {
                        "readable": True,
                        "has_imports": has_imports,
                        "has_classes": has_classes, 
                        "has_async": has_async,
                        "has_typing": has_typing,
                        "line_count": len(content.split('\n'))
                    }
                    
                except Exception as e:
                    results[file_path] = {
                        "readable": False,
                        "error": str(e)
                    }
            else:
                results[file_path] = {"readable": False, "error": "File not found"}
        
        return results
    
    def validate_dependencies_available(self) -> Dict[str, Any]:
        """Check which dependencies are available."""
        print("📦 Checking Available Dependencies...")
        
        dependencies = {
            "numpy": "numpy",
            "matplotlib": "matplotlib", 
            "scipy": "scipy",
            "pandas": "pandas",
            "torch": "torch",
            "numba": "numba",
            "psutil": "psutil",
            "cupy": "cupy"
        }
        
        results = {}
        for name, module in dependencies.items():
            try:
                __import__(module)
                results[name] = {"available": True, "error": None}
            except ImportError as e:
                results[name] = {"available": False, "error": str(e)}
        
        return results
    
    def validate_configuration_structure(self) -> Dict[str, Any]:
        """Validate configuration files and structure."""
        print("⚙️ Checking Configuration Structure...")
        
        config_indicators = [
            ("TimingRequirements", "safe_rl_human_robot/src/optimization/rt_optimizer_integration.py"),
            ("RTOptimizationConfig", "safe_rl_human_robot/src/optimization/rt_optimizer_integration.py"),
            ("OptimizedPolicy", "safe_rl_human_robot/src/optimization/inference/optimized_policy.py"),
            ("FastConstraintChecker", "safe_rl_human_robot/src/optimization/safety_rt/fast_constraint_checker.py"),
        ]
        
        results = {}
        for class_name, file_path in config_indicators:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    results[class_name] = {
                        "class_defined": f"class {class_name}" in content,
                        "has_init": "__init__" in content,
                        "file_exists": True
                    }
                except Exception as e:
                    results[class_name] = {
                        "class_defined": False,
                        "error": str(e),
                        "file_exists": True
                    }
            else:
                results[class_name] = {
                    "class_defined": False,
                    "file_exists": False,
                    "error": "File not found"
                }
        
        return results
    
    def validate_performance_targets(self) -> Dict[str, Any]:
        """Validate performance target definitions."""
        print("🎯 Checking Performance Targets...")
        
        readme_path = self.project_root / "safe_rl_human_robot/src/optimization/README.md"
        
        if not readme_path.exists():
            return {"readme_exists": False}
        
        try:
            with open(readme_path, 'r') as f:
                content = f.read()
            
            # Check for performance targets mentioned in README
            targets = {
                "policy_inference_500us": "<500μs" in content and "Policy Inference" in content,
                "safety_checking_100us": "<100μs" in content and "Safety Checking" in content,
                "memory_allocation_10us": "<10μs" in content and "Memory Allocation" in content,
                "total_loop_1000us": "<1000μs" in content or "1ms" in content,
                "control_frequency_1000hz": "1000 Hz" in content,
                "jitter_tolerance": "jitter" in content.lower()
            }
            
            return {
                "readme_exists": True,
                "targets_defined": targets,
                "total_targets_found": sum(targets.values())
            }
            
        except Exception as e:
            return {
                "readme_exists": True,
                "error": str(e)
            }
    
    def simulate_basic_performance_test(self) -> Dict[str, Any]:
        """Simulate basic performance characteristics."""
        print("⚡ Simulating Performance Test...")
        
        # Simulate some basic computations that would be in the RT system
        results = {}
        
        # Simulate policy inference timing
        times_policy = []
        for _ in range(100):
            start = time.perf_counter()
            # Simulate neural network forward pass
            x = np.random.randn(12, 64)
            y = np.tanh(np.dot(x, np.random.randn(64, 8)))
            end = time.perf_counter()
            times_policy.append((end - start) * 1_000_000)  # Convert to microseconds
        
        # Simulate constraint checking
        times_safety = []
        for _ in range(100):
            start = time.perf_counter()
            # Simulate constraint evaluation
            state = np.random.randn(12)
            action = np.random.randn(8)
            # Simple constraint: action magnitude < 1.0
            safe = np.linalg.norm(action) < 1.0
            end = time.perf_counter()
            times_safety.append((end - start) * 1_000_000)
        
        # Simulate memory allocation
        times_memory = []
        for _ in range(100):
            start = time.perf_counter()
            # Simulate array allocation
            arr = np.zeros(1000)
            end = time.perf_counter()
            times_memory.append((end - start) * 1_000_000)
        
        results = {
            "policy_inference": {
                "mean_time_us": np.mean(times_policy),
                "max_time_us": np.max(times_policy),
                "min_time_us": np.min(times_policy),
                "std_time_us": np.std(times_policy),
                "meets_500us_target": np.mean(times_policy) < 500
            },
            "safety_checking": {
                "mean_time_us": np.mean(times_safety),
                "max_time_us": np.max(times_safety),
                "meets_100us_target": np.mean(times_safety) < 100
            },
            "memory_allocation": {
                "mean_time_us": np.mean(times_memory),
                "max_time_us": np.max(times_memory),
                "meets_10us_target": np.mean(times_memory) < 10
            }
        }
        
        return results
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete structure validation."""
        print("🧪 Starting RT Optimization System Structure Validation")
        print("=" * 60)
        
        results = {
            "file_structure": self.validate_file_structure(),
            "imports": self.validate_imports(),
            "dependencies": self.validate_dependencies_available(),
            "configuration": self.validate_configuration_structure(),
            "performance_targets": self.validate_performance_targets(),
            "simulated_performance": self.simulate_basic_performance_test()
        }
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate validation report."""
        report = []
        report.append("📋 RT OPTIMIZATION SYSTEM VALIDATION REPORT")
        report.append("=" * 60)
        
        # File structure
        files = results.get("file_structure", {})
        total_files = len(files)
        existing_files = sum(1 for f in files.values() if f.get("exists", False))
        report.append(f"\n📁 FILE STRUCTURE: {existing_files}/{total_files} files exist")
        
        if existing_files < total_files:
            report.append("  Missing files:")
            for path, info in files.items():
                if not info.get("exists", False):
                    report.append(f"    ❌ {path}")
        
        # Dependencies
        deps = results.get("dependencies", {})
        available_deps = sum(1 for d in deps.values() if d.get("available", False))
        total_deps = len(deps)
        report.append(f"\n📦 DEPENDENCIES: {available_deps}/{total_deps} available")
        
        for name, info in deps.items():
            status = "✅" if info.get("available") else "❌"
            report.append(f"  {status} {name}")
        
        # Configuration classes
        config = results.get("configuration", {})
        defined_classes = sum(1 for c in config.values() if c.get("class_defined", False))
        total_classes = len(config)
        report.append(f"\n⚙️ CONFIGURATION: {defined_classes}/{total_classes} classes defined")
        
        # Performance targets
        targets = results.get("performance_targets", {})
        if targets.get("readme_exists"):
            target_count = targets.get("total_targets_found", 0)
            report.append(f"\n🎯 PERFORMANCE TARGETS: {target_count} targets documented")
        
        # Simulated performance
        perf = results.get("simulated_performance", {})
        if perf:
            report.append("\n⚡ SIMULATED PERFORMANCE:")
            
            policy = perf.get("policy_inference", {})
            if policy:
                mean_time = policy.get("mean_time_us", 0)
                meets_target = policy.get("meets_500us_target", False)
                status = "✅" if meets_target else "❌"
                report.append(f"  {status} Policy Inference: {mean_time:.1f}μs (target: <500μs)")
            
            safety = perf.get("safety_checking", {})
            if safety:
                mean_time = safety.get("mean_time_us", 0)
                meets_target = safety.get("meets_100us_target", False)
                status = "✅" if meets_target else "❌"
                report.append(f"  {status} Safety Checking: {mean_time:.1f}μs (target: <100μs)")
            
            memory = perf.get("memory_allocation", {})
            if memory:
                mean_time = memory.get("mean_time_us", 0)
                meets_target = memory.get("meets_10us_target", False)
                status = "✅" if meets_target else "❌"
                report.append(f"  {status} Memory Allocation: {mean_time:.1f}μs (target: <10μs)")
        
        # Overall status
        report.append("\n🎯 OVERALL STATUS:")
        
        structure_ok = existing_files >= total_files * 0.9  # 90% of files exist
        deps_ok = available_deps >= 3  # At least basic deps available
        config_ok = defined_classes >= total_classes * 0.8  # 80% of classes defined
        
        if structure_ok and deps_ok and config_ok:
            report.append("  System Structure: ✅ READY FOR IMPLEMENTATION")
        elif structure_ok and config_ok:
            report.append("  System Structure: ⚠️  NEEDS DEPENDENCY INSTALLATION")
        else:
            report.append("  System Structure: ❌ INCOMPLETE IMPLEMENTATION")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        
        if not deps.get("torch", {}).get("available"):
            report.append("  - Install PyTorch: pip install torch torchvision")
        
        if not deps.get("numba", {}).get("available"):
            report.append("  - Install Numba: pip install numba")
        
        if not deps.get("psutil", {}).get("available"):
            report.append("  - Install psutil: pip install psutil")
        
        if existing_files < total_files:
            report.append("  - Complete missing file implementations")
        
        report.append("  - Run full performance tests after installing dependencies")
        
        return "\n".join(report)
    
    def create_structure_visualization(self, results: Dict[str, Any]):
        """Create visualization of system structure status."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            
            # File structure status
            files = results.get("file_structure", {})
            if files:
                existing = sum(1 for f in files.values() if f.get("exists", False))
                missing = len(files) - existing
                
                ax1.pie([existing, missing], labels=['Existing', 'Missing'], 
                       colors=['green', 'red'], autopct='%1.1f%%')
                ax1.set_title('File Structure Completeness')
            
            # Dependency availability
            deps = results.get("dependencies", {})
            if deps:
                available = [name for name, info in deps.items() if info.get("available")]
                missing = [name for name, info in deps.items() if not info.get("available")]
                
                y_pos = np.arange(len(deps))
                colors = ['green' if deps[name].get("available") else 'red' 
                         for name in deps.keys()]
                
                ax2.barh(y_pos, [1] * len(deps), color=colors)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(list(deps.keys()))
                ax2.set_title('Dependency Availability')
                ax2.set_xlabel('Available')
            
            # Configuration classes
            config = results.get("configuration", {})
            if config:
                defined = sum(1 for c in config.values() if c.get("class_defined", False))
                missing = len(config) - defined
                
                ax3.bar(['Defined', 'Missing'], [defined, missing], 
                       color=['green', 'red'])
                ax3.set_title('Configuration Classes')
                ax3.set_ylabel('Count')
            
            # Performance simulation results
            perf = results.get("simulated_performance", {})
            if perf:
                components = []
                times = []
                targets = []
                
                if 'policy_inference' in perf:
                    components.append('Policy\nInference')
                    times.append(perf['policy_inference'].get('mean_time_us', 0))
                    targets.append(500)
                
                if 'safety_checking' in perf:
                    components.append('Safety\nChecking')  
                    times.append(perf['safety_checking'].get('mean_time_us', 0))
                    targets.append(100)
                
                if 'memory_allocation' in perf:
                    components.append('Memory\nAllocation')
                    times.append(perf['memory_allocation'].get('mean_time_us', 0))
                    targets.append(10)
                
                if components:
                    x_pos = np.arange(len(components))
                    bars = ax4.bar(x_pos, times, color='lightblue', label='Actual')
                    ax4.bar(x_pos, targets, color='red', alpha=0.7, label='Target')
                    ax4.set_xticks(x_pos)
                    ax4.set_xticklabels(components)
                    ax4.set_title('Simulated Performance')
                    ax4.set_ylabel('Time (μs)')
                    ax4.legend()
                    ax4.set_yscale('log')  # Log scale for better visualization
            
            plt.tight_layout()
            
            # Save plot
            output_dir = Path("validation_output")
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / "structure_validation.png", dpi=300, bbox_inches='tight')
            print(f"📊 Structure validation plot saved to: {output_dir / 'structure_validation.png'}")
            
        except Exception as e:
            print(f"📊 Could not create structure visualization: {e}")
    
    def save_results(self, results: Dict[str, Any]):
        """Save validation results to file."""
        output_dir = Path("validation_output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "structure_validation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_dir / 'structure_validation_results.json'}")


def main():
    """Main validation function."""
    validator = RTSystemStructureValidator()
    
    try:
        # Run validation
        results = validator.run_validation()
        
        # Generate and display report
        report = validator.generate_report(results)
        print(report)
        
        # Create visualization
        validator.create_structure_visualization(results)
        
        # Save results
        validator.save_results(results)
        
        return results
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()