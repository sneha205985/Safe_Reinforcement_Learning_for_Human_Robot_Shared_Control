# Real-time Performance Optimization System

This module provides comprehensive real-time optimization capabilities for Safe RL Human-Robot Shared Control systems, ensuring deterministic timing and meeting strict performance requirements for safety-critical applications.

## 🎯 Performance Targets

- **Policy Inference**: <500μs execution time
- **Safety Checking**: <100μs constraint validation  
- **Memory Allocation**: <10μs deterministic allocation
- **Total Loop Time**: <1000μs (1ms) end-to-end
- **Jitter Tolerance**: <50μs timing variability
- **Control Frequency**: 1000 Hz sustained operation
- **Safety Monitoring**: 2000 Hz constraint checking

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RT Optimization System                       │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│ │ Optimized       │ │ Fast Constraint │ │ RT Memory       │    │
│ │ Policy          │ │ Checker         │ │ Manager         │    │
│ │ • Quantization  │ │ • JIT Compiled  │ │ • Pre-allocated │    │
│ │ • Layer Fusion  │ │ • Vectorized    │ │ • Lock-free     │    │
│ │ • TensorRT      │ │ • GPU Accel     │ │ • NUMA-aware    │    │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│ │ RT Control      │ │ GPU Optimizer   │ │ System Config   │    │
│ │ System          │ │ • CUDA Streams  │ │ • RT Kernel     │    │
│ │ • Isolated CPUs │ │ • Memory Pools  │ │ • CPU Isolation │    │
│ │ • RT Threads    │ │ • Custom Kernels│ │ • IRQ Handling  │    │
│ │ • Priority Sched│ │ • Multi-GPU     │ │ • Cache Tuning  │    │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐                        │
│ │ Performance     │ │ Continuous      │                        │
│ │ Benchmark       │ │ Monitor         │                        │
│ │ • Determinism   │ │ • Live Metrics  │                        │
│ │ • Stress Tests  │ │ • Regression    │                        │
│ │ • Validation    │ │ • Alerting      │                        │
│ └─────────────────┘ └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Module Structure

```
src/optimization/
├── rt_optimizer_integration.py      # Main integration module
├── inference/
│   └── optimized_policy.py         # Model optimization & inference
├── safety_rt/
│   └── fast_constraint_checker.py  # Real-time safety validation
├── memory/
│   └── rt_memory_manager.py        # Deterministic memory management
├── parallel/
│   └── rt_control_system.py        # RT thread management
├── gpu/
│   └── cuda_optimizer.py           # GPU acceleration
├── system/
│   └── rt_system_config.py         # System-level optimization
├── benchmarking/
│   └── rt_performance_benchmark.py # Performance validation
└── monitoring/
    └── continuous_performance_monitor.py # Live monitoring
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
import torch
from safe_rl_human_robot.src.optimization.rt_optimizer_integration import create_development_system

# Create your policy model
policy = torch.nn.Sequential(
    torch.nn.Linear(12, 64),
    torch.nn.ReLU(), 
    torch.nn.Linear(64, 8),
    torch.nn.Tanh()
)

async def main():
    # Create optimized system
    system = create_development_system(policy)
    
    async with system:
        # Execute real-time inference
        state = np.random.randn(12).astype(np.float32)
        action, metadata = await system.execute_rt_inference(state)
        
        print(f"Inference time: {metadata['inference_time_us']:.1f}μs")
        print(f"Safe: {metadata['safe']}")

asyncio.run(main())
```

### Production Deployment

```python
from safe_rl_human_robot.src.optimization.rt_optimizer_integration import create_production_system

# Create production system with full optimizations
system = create_production_system(your_trained_policy)

async with system:
    # System automatically applies all optimizations:
    # - System-level RT configuration
    # - GPU acceleration (if available)
    # - Memory optimization
    # - Performance monitoring
    
    # Run high-frequency control loop
    for i in range(10000):
        state = get_sensor_data()
        action, metadata = await system.execute_rt_inference(state)
        send_to_actuators(action)
        await asyncio.sleep(0.001)  # 1000 Hz
```

## 🔧 Configuration

### Custom Configuration

```python
from safe_rl_human_robot.src.optimization.rt_optimizer_integration import (
    RTOptimizationConfig, RTOptimizedSystem, TimingRequirements
)

# Create custom configuration
config = RTOptimizationConfig(
    # Timing requirements
    timing_requirements=TimingRequirements(
        max_execution_time_us=500,
        max_jitter_us=25,
        min_frequency_hz=2000.0
    ),
    
    # Component settings
    enable_system_optimization=True,
    enable_gpu_acceleration=True,
    enable_continuous_monitoring=True,
    
    # Deployment mode
    deployment_mode="production"
)

system = RTOptimizedSystem(config, your_policy)
```

### Environment Variables

```bash
# System optimization (requires root)
export RT_ENABLE_SYSTEM_OPT=true
export RT_CPU_ISOLATION="2,3,4,5"
export RT_GOVERNOR="performance"

# GPU configuration  
export RT_GPU_DEVICE=0
export RT_ENABLE_TENSORRT=true
export RT_GPU_MEMORY_POOL_MB=2048

# Memory configuration
export RT_MEMORY_POOL_MB=4096
export RT_ENABLE_HUGEPAGES=true

# Monitoring
export RT_ENABLE_PROMETHEUS=true
export RT_METRICS_PORT=9090
```

## 📊 Performance Monitoring

### Real-time Dashboard

The system provides comprehensive performance monitoring:

```python
# Get live performance data
status = system.get_system_status()
print(f"Performance validated: {status['performance_validated']}")
print(f"Active alerts: {status['monitoring']['active_alerts']}")

# Generate deployment report
report = system.generate_deployment_report()
with open('deployment_report.json', 'w') as f:
    json.dump(report, f, indent=2)
```

### Metrics Exported

- **Timing Metrics**: Execution times, jitter, deadline misses
- **Throughput**: Operations per second, queue depths
- **Resource Usage**: CPU, memory, GPU utilization  
- **Safety Metrics**: Constraint violations, safety scores
- **System Metrics**: Context switches, interrupts, cache performance

### Integration with Prometheus/Grafana

```python
# Enable Prometheus metrics export
config.metrics_export_enabled = True

# Metrics available at http://localhost:9090/metrics
# - rt_execution_time_seconds
# - rt_operations_total  
# - rt_errors_total
# - rt_safety_violations_total
# - rt_system_cpu_percent
# - rt_system_memory_percent
```

## 🧪 Testing & Validation

### Run Performance Benchmark

```python
from safe_rl_human_robot.src.optimization.benchmarking.rt_performance_benchmark import RTPerformanceBenchmark, TimingRequirements

# Create benchmark system
requirements = TimingRequirements(max_execution_time_us=1000)
benchmark = RTPerformanceBenchmark(requirements)

# Run comprehensive test suite
results = benchmark.run_benchmark_suite()

# Generate report
report = benchmark.generate_report("benchmark_report.json")
benchmark.plot_results("benchmark_plots/")
```

### Stress Testing

```bash
# Run comprehensive demo with stress testing
python examples/rt_optimization_demo.py
```

### Determinism Validation  

```python
# Test timing determinism
determinism_results = benchmark.validate_determinism(10000)
print(f"Deterministic: {determinism_results['is_deterministic']}")
print(f"Coefficient of variation: {determinism_results['coefficient_of_variation']:.4f}")
```

## 🛠️ System Requirements

### Hardware Requirements

**Minimum:**
- CPU: 4 cores, 2.0 GHz
- Memory: 8GB RAM
- Storage: 10GB free space

**Recommended:**
- CPU: 8+ cores, 3.0+ GHz (with RT kernel support)
- Memory: 16GB+ RAM  
- GPU: NVIDIA GPU with 4GB+ VRAM (for GPU acceleration)
- Storage: 50GB+ SSD
- Network: 1Gb/s (for distributed deployments)

### Software Requirements

**Operating System:**
- Ubuntu 20.04+ with RT kernel (recommended)
- CentOS 8+ with RT kernel
- RHEL 8+ with RT kernel
- Any Linux with kernel 5.4+

**Dependencies:**
```bash
# Core dependencies
pip install torch>=1.12 torchvision numpy scipy
pip install psutil numba cupy-cuda11x  # GPU support
pip install prometheus-client redis pandas matplotlib

# System packages (Ubuntu/Debian)
sudo apt-get install linux-lowlatency  # RT kernel
sudo apt-get install numactl cpufrequtils
```

### Permissions

Some optimizations require elevated privileges:

```bash
# Run with capabilities (preferred)
sudo setcap 'cap_sys_nice=eip cap_ipc_lock=eip cap_sys_rawio=eip' python

# Or run as root (for system optimizations)
sudo python your_application.py
```

## 🔄 Deployment Scenarios

### Development Environment

```python
# Minimal setup for development/testing
system = create_development_system(policy)
# - No system optimizations (no root required)
# - Relaxed timing requirements  
# - Full monitoring and debugging
```

### Edge Deployment

```python
# Resource-constrained edge devices
system = create_edge_system(policy)  
# - CPU-only optimization
# - Quantized models
# - Reduced memory footprint
# - Local-only operation
```

### Cloud Deployment  

```python
# Scalable cloud deployment
system = create_production_system(policy)
# - Full GPU acceleration
# - Distributed monitoring
# - Auto-scaling support
# - High availability
```

### Hybrid Deployment

```python
# Edge + cloud coordination
# See deployment/hybrid/ for configuration
```

## 📈 Performance Optimization Tips

### Model Optimization
1. **Enable Quantization**: 2-4x speedup with minimal accuracy loss
2. **Use TensorRT**: Additional 2-3x speedup on NVIDIA GPUs
3. **Apply Pruning**: Reduce model size by 30-70%
4. **Cache Frequent Patterns**: Pre-compute common state-action mappings

### System Optimization
1. **Isolate CPUs**: Dedicate CPUs to RT threads only
2. **Disable Hyperthreading**: Reduces timing variability
3. **Use Performance Governor**: Maintain constant CPU frequency
4. **Enable Hugepages**: Improve memory access patterns

### Memory Optimization  
1. **Pre-allocate Arrays**: Avoid garbage collection in RT loops
2. **Use Memory Pools**: Deterministic allocation times
3. **Pin Memory**: Faster GPU transfers
4. **NUMA Awareness**: Optimize for multi-socket systems

### GPU Optimization
1. **Use CUDA Streams**: Overlap computation and memory transfers
2. **Batch Operations**: Higher GPU utilization
3. **Optimize Memory Layout**: Coalesced memory access
4. **Custom Kernels**: Domain-specific optimizations

## 🐛 Troubleshooting

### Common Issues

**High Latency:**
```python
# Check system configuration
status = system.get_system_status()
if not status['performance_validated']:
    print("Performance validation failed - check requirements")

# Run diagnostics  
report = system.generate_deployment_report()
print("Recommendations:", report['recommendations'])
```

**Memory Issues:**
```bash
# Check memory allocation
free -h
cat /proc/meminfo | grep Huge

# Adjust memory pools
export RT_MEMORY_POOL_MB=8192
```

**GPU Issues:**
```bash
# Check GPU status
nvidia-smi
nvidia-ml-py3 --query

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Timing Issues:**
```bash
# Check RT kernel
uname -a | grep rt
cat /sys/kernel/realtime

# Verify CPU isolation
cat /proc/cmdline | grep isolcpus
cat /sys/devices/system/cpu/isolated
```

### Performance Debugging

```python
# Enable detailed profiling
system.config.enable_profiling = True

# Monitor specific operations
async with system.rt_operation_context("custom_operation"):
    # Your code here
    pass

# Get detailed timing breakdown
perf_summary = system.performance_monitor.get_performance_summary()
print(json.dumps(perf_summary, indent=2))
```

## 📚 API Reference

See individual module documentation:
- [OptimizedPolicy](inference/optimized_policy.py) - Model optimization
- [FastConstraintChecker](safety_rt/fast_constraint_checker.py) - Safety validation
- [RTMemoryManager](memory/rt_memory_manager.py) - Memory management
- [RTControlSystem](parallel/rt_control_system.py) - Thread management
- [CUDAOptimizer](gpu/cuda_optimizer.py) - GPU acceleration
- [RTSystemOptimizer](system/rt_system_config.py) - System optimization
- [RTPerformanceBenchmark](benchmarking/rt_performance_benchmark.py) - Benchmarking
- [ContinuousPerformanceMonitor](monitoring/continuous_performance_monitor.py) - Monitoring

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/rt-optimization`
3. Add tests for new functionality
4. Ensure all benchmarks pass: `python -m pytest tests/optimization/`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For issues and questions:
- Create GitHub issue for bugs/feature requests
- Check existing issues for common problems
- Review troubleshooting section above

## 🏆 Acknowledgments

This optimization system builds upon:
- PyTorch for deep learning inference
- CUDA/CuPy for GPU acceleration  
- Numba for JIT compilation
- Linux RT kernel for real-time capabilities
- Prometheus for metrics collection