"""
Real-time System Configuration and Optimization

This module provides comprehensive system-level optimizations for real-time
performance, including OS kernel tuning, hardware configuration, and 
process isolation.

Key features:
- Real-time kernel configuration and tuning
- CPU isolation and interrupt handling optimization
- Memory management and swap control
- Process priority and affinity management
- Hardware-specific optimizations (SIMD, cache, etc.)
- Performance monitoring and validation
- System health checks and diagnostics
"""

import ctypes
import logging
import os
import platform
import psutil
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

try:
    import numa
    NUMA_AVAILABLE = True
except ImportError:
    NUMA_AVAILABLE = False

try:
    import cpuinfo
    CPUINFO_AVAILABLE = True
except ImportError:
    CPUINFO_AVAILABLE = False

logger = logging.getLogger(__name__)


class RTKernelType(Enum):
    """Types of real-time kernel configurations"""
    PREEMPT_NONE = "none"           # No preemption
    PREEMPT_VOLUNTARY = "voluntary" # Voluntary preemption
    PREEMPT_DESKTOP = "desktop"     # Desktop preemption
    PREEMPT_RT = "rt"              # Real-time preemption patch
    PREEMPT_RT_FULL = "rt_full"    # Full real-time kernel


class CPUGovernor(Enum):
    """CPU frequency scaling governors"""
    PERFORMANCE = "performance"     # Maximum performance
    POWERSAVE = "powersave"        # Power saving
    ONDEMAND = "ondemand"          # Dynamic scaling
    CONSERVATIVE = "conservative"   # Conservative scaling
    USERSPACE = "userspace"        # User-controlled


class IOScheduler(Enum):
    """I/O schedulers for different workloads"""
    NOOP = "noop"                  # No-operation (best for RT)
    DEADLINE = "deadline"          # Deadline scheduler
    CFQ = "cfq"                    # Completely Fair Queuing
    BFQ = "bfq"                    # Budget Fair Queuing
    MQ_DEADLINE = "mq-deadline"    # Multi-queue deadline
    KYBER = "kyber"                # Kyber scheduler


@dataclass
class HardwareInfo:
    """System hardware information"""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_threads: int = 0
    cpu_cache_l1: int = 0
    cpu_cache_l2: int = 0
    cpu_cache_l3: int = 0
    cpu_features: List[str] = field(default_factory=list)
    memory_total_gb: float = 0.0
    memory_speed_mhz: int = 0
    numa_nodes: List[int] = field(default_factory=list)
    gpu_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RTSystemConfig:
    """Configuration for real-time system optimization"""
    
    # Kernel configuration
    target_kernel_type: RTKernelType = RTKernelType.PREEMPT_RT
    kernel_params: Dict[str, Any] = field(default_factory=dict)
    
    # CPU configuration
    cpu_governor: CPUGovernor = CPUGovernor.PERFORMANCE
    isolate_cpus: List[int] = field(default_factory=list)
    rt_cpus: List[int] = field(default_factory=list)
    housekeeping_cpus: List[int] = field(default_factory=list)
    disable_hyperthreading: bool = False
    disable_turbo_boost: bool = True
    
    # Memory configuration
    disable_swap: bool = True
    lock_memory: bool = True
    transparent_hugepages: str = "never"  # always, madvise, never
    memory_compaction: bool = False
    
    # Interrupt handling
    disable_irq_balance: bool = True
    isolate_irqs_from_rt_cpus: bool = True
    rcu_nocbs_cpus: List[int] = field(default_factory=list)
    
    # I/O configuration
    io_scheduler: IOScheduler = IOScheduler.NOOP
    readahead_kb: int = 128
    
    # Network configuration
    disable_network_irq_on_rt_cpus: bool = True
    network_buffer_sizes: Dict[str, int] = field(default_factory=dict)
    
    # Timer configuration
    hpet_enabled: bool = True
    tsc_reliable: bool = True
    clocksource: str = "tsc"
    
    # Security and isolation
    disable_address_space_randomization: bool = True
    disable_core_dumps: bool = True
    
    # Monitoring and debugging
    enable_rt_monitoring: bool = True
    enable_latency_tracing: bool = False
    enable_function_tracing: bool = False
    
    # Process limits
    max_rt_processes: int = 100
    max_locked_memory_mb: int = 2048
    
    def __post_init__(self):
        if not self.isolate_cpus and not self.rt_cpus:
            # Auto-detect CPU configuration
            self._auto_configure_cpus()
        
        if not self.kernel_params:
            self._initialize_default_kernel_params()
        
        if not self.network_buffer_sizes:
            self._initialize_default_network_config()
    
    def _auto_configure_cpus(self):
        """Automatically configure CPU isolation based on system"""
        total_cpus = psutil.cpu_count(logical=True)
        
        if total_cpus >= 8:
            # Use first 2 CPUs for RT, last 2 for housekeeping, isolate middle CPUs
            self.rt_cpus = [0, 1]
            self.housekeeping_cpus = [total_cpus - 2, total_cpus - 1]
            self.isolate_cpus = list(range(2, total_cpus - 2))
        elif total_cpus >= 4:
            # Use first CPU for RT, last CPU for housekeeping
            self.rt_cpus = [0]
            self.housekeeping_cpus = [total_cpus - 1]
            self.isolate_cpus = list(range(1, total_cpus - 1))
        else:
            # Not enough CPUs for isolation
            self.rt_cpus = [0]
            self.housekeeping_cpus = list(range(1, total_cpus))
    
    def _initialize_default_kernel_params(self):
        """Initialize default kernel parameters for RT"""
        self.kernel_params = {
            # CPU isolation
            "isolcpus": ",".join(map(str, self.isolate_cpus)) if self.isolate_cpus else "",
            "nohz_full": ",".join(map(str, self.isolate_cpus)) if self.isolate_cpus else "",
            "rcu_nocbs": ",".join(map(str, self.rcu_nocbs_cpus or self.isolate_cpus)),
            
            # Scheduler
            "preempt": "rt" if self.target_kernel_type == RTKernelType.PREEMPT_RT else "none",
            "processor.max_cstate": "1",
            "intel_idle.max_cstate": "0",
            
            # Memory
            "transparent_hugepage": self.transparent_hugepages,
            "vm.swappiness": "0" if self.disable_swap else "60",
            
            # Timers
            "clocksource": self.clocksource,
            "tsc": "reliable" if self.tsc_reliable else "unstable",
            "hpet": "enable" if self.hpet_enabled else "disable",
            
            # Security
            "kernel.randomize_va_space": "0" if self.disable_address_space_randomization else "2",
            
            # RT specific
            "kernel.sched_rt_runtime_us": "-1",  # No RT throttling
            "kernel.sched_rt_period_us": "1000000",
        }
    
    def _initialize_default_network_config(self):
        """Initialize default network buffer configuration"""
        self.network_buffer_sizes = {
            "net.core.rmem_max": 134217728,      # 128MB
            "net.core.wmem_max": 134217728,      # 128MB
            "net.core.rmem_default": 262144,      # 256KB
            "net.core.wmem_default": 262144,      # 256KB
            "net.core.netdev_max_backlog": 5000,
        }


class SystemHardwareDetector:
    """Detects and analyzes system hardware for optimization"""
    
    def __init__(self):
        self.hardware_info = HardwareInfo()
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect system hardware information"""
        # CPU information
        self.hardware_info.cpu_cores = psutil.cpu_count(logical=False)
        self.hardware_info.cpu_threads = psutil.cpu_count(logical=True)
        
        # Memory information
        memory = psutil.virtual_memory()
        self.hardware_info.memory_total_gb = memory.total / (1024**3)
        
        # CPU features and cache information
        if CPUINFO_AVAILABLE:
            try:
                cpu_info = cpuinfo.get_cpu_info()
                self.hardware_info.cpu_model = cpu_info.get('brand_raw', 'Unknown')
                self.hardware_info.cpu_features = cpu_info.get('flags', [])
                
                # Try to extract cache information
                for key, value in cpu_info.items():
                    if 'l1' in key.lower() and 'cache' in key.lower():
                        self.hardware_info.cpu_cache_l1 = self._parse_cache_size(str(value))
                    elif 'l2' in key.lower() and 'cache' in key.lower():
                        self.hardware_info.cpu_cache_l2 = self._parse_cache_size(str(value))
                    elif 'l3' in key.lower() and 'cache' in key.lower():
                        self.hardware_info.cpu_cache_l3 = self._parse_cache_size(str(value))
                        
            except Exception as e:
                logger.debug(f"CPU info detection failed: {e}")
        
        # NUMA information
        if NUMA_AVAILABLE:
            try:
                self.hardware_info.numa_nodes = list(range(numa.get_max_node() + 1))
            except:
                pass
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                self.hardware_info.gpu_info = {
                    'cuda_available': True,
                    'gpu_count': torch.cuda.device_count(),
                    'gpu_names': [torch.cuda.get_device_name(i) 
                                for i in range(torch.cuda.device_count())],
                }
        except ImportError:
            pass
        
        logger.info(f"Hardware detected: {self.hardware_info.cpu_cores} cores, "
                   f"{self.hardware_info.memory_total_gb:.1f}GB RAM")
    
    def _parse_cache_size(self, cache_str: str) -> int:
        """Parse cache size string to KB"""
        try:
            # Extract number and unit
            match = re.search(r'(\d+)\s*([KMGT]?B?)', cache_str.upper())
            if match:
                size, unit = match.groups()
                size = int(size)
                
                if 'K' in unit:
                    return size
                elif 'M' in unit:
                    return size * 1024
                elif 'G' in unit:
                    return size * 1024 * 1024
                else:
                    return size
        except:
            pass
        return 0
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get hardware-specific optimization recommendations"""
        recommendations = {
            'cpu_optimizations': [],
            'memory_optimizations': [],
            'cache_optimizations': [],
            'numa_optimizations': [],
        }
        
        # CPU recommendations
        if 'avx2' in self.hardware_info.cpu_features:
            recommendations['cpu_optimizations'].append("Enable AVX2 vectorization")
        if 'avx512f' in self.hardware_info.cpu_features:
            recommendations['cpu_optimizations'].append("Enable AVX-512 vectorization")
        if 'tsc_deadline_timer' in self.hardware_info.cpu_features:
            recommendations['cpu_optimizations'].append("Use TSC deadline timer")
        
        # Memory recommendations
        if self.hardware_info.memory_total_gb >= 16:
            recommendations['memory_optimizations'].append("Enable large page support")
        if self.hardware_info.memory_total_gb >= 64:
            recommendations['memory_optimizations'].append("Consider 1GB hugepages")
        
        # Cache recommendations
        if self.hardware_info.cpu_cache_l3 > 0:
            cache_size_mb = self.hardware_info.cpu_cache_l3 // 1024
            recommendations['cache_optimizations'].append(
                f"Optimize for L3 cache size: {cache_size_mb}MB"
            )
        
        # NUMA recommendations
        if len(self.hardware_info.numa_nodes) > 1:
            recommendations['numa_optimizations'].append("Enable NUMA awareness")
            recommendations['numa_optimizations'].append("Pin RT threads to single NUMA node")
        
        return recommendations


class RTSystemOptimizer:
    """
    Real-time system optimizer that applies comprehensive system-level
    optimizations for deterministic performance.
    """
    
    def __init__(self, config: RTSystemConfig):
        self.config = config
        self.hardware_detector = SystemHardwareDetector()
        self.original_settings = {}
        self.applied_optimizations = []
        
        # Check if running with sufficient privileges
        self.is_privileged = os.geteuid() == 0
        if not self.is_privileged:
            logger.warning("Running without root privileges - some optimizations unavailable")
        
        logger.info("RT system optimizer initialized")
    
    def apply_all_optimizations(self) -> Dict[str, bool]:
        """
        Apply all system optimizations.
        
        Returns:
            Dictionary of optimization results
        """
        results = {}
        
        logger.info("Applying real-time system optimizations...")
        
        # CPU optimizations
        results['cpu_governor'] = self._set_cpu_governor()
        results['cpu_isolation'] = self._configure_cpu_isolation()
        results['hyperthreading'] = self._configure_hyperthreading()
        results['turbo_boost'] = self._configure_turbo_boost()
        
        # Memory optimizations
        results['swap'] = self._configure_swap()
        results['transparent_hugepages'] = self._configure_transparent_hugepages()
        results['memory_compaction'] = self._configure_memory_compaction()
        
        # Interrupt optimizations
        results['irq_balance'] = self._configure_irq_balance()
        results['irq_isolation'] = self._configure_irq_isolation()
        
        # I/O optimizations
        results['io_scheduler'] = self._configure_io_scheduler()
        results['readahead'] = self._configure_readahead()
        
        # Timer optimizations
        results['clocksource'] = self._configure_clocksource()
        
        # Process limits
        results['process_limits'] = self._configure_process_limits()
        
        # Network optimizations
        results['network_buffers'] = self._configure_network_buffers()
        
        # Kernel parameters
        results['kernel_params'] = self._apply_kernel_parameters()
        
        # Log results
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Applied {successful}/{total} optimizations successfully")
        
        return results
    
    def _set_cpu_governor(self) -> bool:
        """Set CPU frequency governor"""
        try:
            governor = self.config.cpu_governor.value
            
            # Save current governor
            current_governors = []
            for cpu in range(psutil.cpu_count()):
                gov_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                if os.path.exists(gov_file):
                    with open(gov_file, 'r') as f:
                        current_governors.append(f.read().strip())
            
            self.original_settings['cpu_governors'] = current_governors
            
            # Set new governor
            cmd = f"echo {governor} | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Set CPU governor to {governor}")
                self.applied_optimizations.append(f"cpu_governor_{governor}")
                return True
            else:
                logger.error(f"Failed to set CPU governor: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"CPU governor configuration failed: {e}")
            return False
    
    def _configure_cpu_isolation(self) -> bool:
        """Configure CPU isolation"""
        if not self.config.isolate_cpus:
            return True
        
        try:
            isolated_cpus = ",".join(map(str, self.config.isolate_cpus))
            
            # Set CPU isolation in cpuset
            if os.path.exists("/sys/fs/cgroup/cpuset"):
                # Create isolated cpuset
                isolated_path = "/sys/fs/cgroup/cpuset/isolated"
                os.makedirs(isolated_path, exist_ok=True)
                
                with open(f"{isolated_path}/cpuset.cpus", 'w') as f:
                    f.write(isolated_cpus)
                
                with open(f"{isolated_path}/cpuset.mems", 'w') as f:
                    f.write("0")  # Use first memory node
                
                logger.info(f"Configured CPU isolation: {isolated_cpus}")
                self.applied_optimizations.append(f"cpu_isolation_{isolated_cpus}")
                return True
            else:
                logger.warning("cpuset not available for CPU isolation")
                return False
                
        except Exception as e:
            logger.error(f"CPU isolation configuration failed: {e}")
            return False
    
    def _configure_hyperthreading(self) -> bool:
        """Configure hyperthreading"""
        if not self.config.disable_hyperthreading:
            return True
        
        try:
            # Disable hyperthreading by taking sibling CPUs offline
            siblings_disabled = 0
            
            for cpu in range(psutil.cpu_count()):
                siblings_file = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
                online_file = f"/sys/devices/system/cpu/cpu{cpu}/online"
                
                if os.path.exists(siblings_file) and os.path.exists(online_file):
                    with open(siblings_file, 'r') as f:
                        siblings = f.read().strip().split(',')
                    
                    # If this CPU has siblings and is not the first, disable it
                    if len(siblings) > 1 and str(cpu) != siblings[0]:
                        with open(online_file, 'w') as f:
                            f.write('0')
                        siblings_disabled += 1
            
            if siblings_disabled > 0:
                logger.info(f"Disabled {siblings_disabled} hyperthreading siblings")
                self.applied_optimizations.append("hyperthreading_disabled")
            
            return True
            
        except Exception as e:
            logger.error(f"Hyperthreading configuration failed: {e}")
            return False
    
    def _configure_turbo_boost(self) -> bool:
        """Configure Intel Turbo Boost"""
        if not self.config.disable_turbo_boost:
            return True
        
        try:
            # Disable Intel Turbo Boost
            turbo_file = "/sys/devices/system/cpu/intel_pstate/no_turbo"
            if os.path.exists(turbo_file):
                with open(turbo_file, 'w') as f:
                    f.write('1')
                logger.info("Disabled Intel Turbo Boost")
                self.applied_optimizations.append("turbo_boost_disabled")
                return True
            else:
                # Try alternative method
                cmd = "echo 1 > /sys/devices/system/cpu/cpufreq/boost"
                result = subprocess.run(cmd, shell=True, capture_output=True)
                if result.returncode == 0:
                    logger.info("Disabled CPU boost")
                    self.applied_optimizations.append("cpu_boost_disabled")
                    return True
                else:
                    logger.warning("Could not disable turbo boost - not available")
                    return False
                    
        except Exception as e:
            logger.error(f"Turbo boost configuration failed: {e}")
            return False
    
    def _configure_swap(self) -> bool:
        """Configure swap settings"""
        if not self.config.disable_swap:
            return True
        
        try:
            # Disable swap
            result = subprocess.run("swapoff -a", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Disabled swap")
                self.applied_optimizations.append("swap_disabled")
                
                # Set swappiness to 0
                with open("/proc/sys/vm/swappiness", 'w') as f:
                    f.write('0')
                
                return True
            else:
                logger.error(f"Failed to disable swap: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Swap configuration failed: {e}")
            return False
    
    def _configure_transparent_hugepages(self) -> bool:
        """Configure transparent hugepages"""
        try:
            thp_file = "/sys/kernel/mm/transparent_hugepage/enabled"
            if os.path.exists(thp_file):
                with open(thp_file, 'w') as f:
                    f.write(self.config.transparent_hugepages)
                
                logger.info(f"Set transparent hugepages to {self.config.transparent_hugepages}")
                self.applied_optimizations.append(f"thp_{self.config.transparent_hugepages}")
                return True
            else:
                logger.warning("Transparent hugepages not available")
                return False
                
        except Exception as e:
            logger.error(f"Transparent hugepages configuration failed: {e}")
            return False
    
    def _configure_memory_compaction(self) -> bool:
        """Configure memory compaction"""
        try:
            compaction_value = "0" if not self.config.memory_compaction else "1"
            
            compaction_files = [
                "/proc/sys/vm/compact_memory",
                "/sys/kernel/mm/transparent_hugepage/khugepaged/defrag"
            ]
            
            for file_path in compaction_files:
                if os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write(compaction_value)
            
            logger.info(f"Memory compaction: {'enabled' if self.config.memory_compaction else 'disabled'}")
            self.applied_optimizations.append(f"memory_compaction_{compaction_value}")
            return True
            
        except Exception as e:
            logger.error(f"Memory compaction configuration failed: {e}")
            return False
    
    def _configure_irq_balance(self) -> bool:
        """Configure IRQ balancing"""
        if not self.config.disable_irq_balance:
            return True
        
        try:
            # Stop irqbalance service
            result = subprocess.run("systemctl stop irqbalance", shell=True, capture_output=True)
            
            # Disable irqbalance service
            subprocess.run("systemctl disable irqbalance", shell=True, capture_output=True)
            
            logger.info("Disabled IRQ balancing")
            self.applied_optimizations.append("irq_balance_disabled")
            return True
            
        except Exception as e:
            logger.error(f"IRQ balance configuration failed: {e}")
            return False
    
    def _configure_irq_isolation(self) -> bool:
        """Configure IRQ isolation from RT CPUs"""
        if not self.config.isolate_irqs_from_rt_cpus or not self.config.rt_cpus:
            return True
        
        try:
            # Move all IRQs away from RT CPUs
            rt_cpus = set(self.config.rt_cpus)
            all_cpus = set(range(psutil.cpu_count()))
            non_rt_cpus = all_cpus - rt_cpus
            
            if not non_rt_cpus:
                logger.warning("No non-RT CPUs available for IRQ handling")
                return False
            
            # Create CPU mask for non-RT CPUs
            cpu_mask = 0
            for cpu in non_rt_cpus:
                cpu_mask |= (1 << cpu)
            
            mask_hex = f"{cpu_mask:x}"
            
            # Apply to all IRQs
            irqs_moved = 0
            proc_irq_path = "/proc/irq"
            
            if os.path.exists(proc_irq_path):
                for irq_dir in os.listdir(proc_irq_path):
                    if irq_dir.isdigit():
                        smp_affinity_file = os.path.join(proc_irq_path, irq_dir, "smp_affinity")
                        if os.path.exists(smp_affinity_file):
                            try:
                                with open(smp_affinity_file, 'w') as f:
                                    f.write(mask_hex)
                                irqs_moved += 1
                            except:
                                pass  # Some IRQs cannot be moved
            
            logger.info(f"Moved {irqs_moved} IRQs away from RT CPUs")
            self.applied_optimizations.append(f"irq_isolation_{irqs_moved}")
            return True
            
        except Exception as e:
            logger.error(f"IRQ isolation configuration failed: {e}")
            return False
    
    def _configure_io_scheduler(self) -> bool:
        """Configure I/O scheduler"""
        try:
            scheduler = self.config.io_scheduler.value
            
            # Apply to all block devices
            schedulers_set = 0
            
            for device in os.listdir("/sys/block"):
                scheduler_file = f"/sys/block/{device}/queue/scheduler"
                if os.path.exists(scheduler_file):
                    try:
                        with open(scheduler_file, 'w') as f:
                            f.write(scheduler)
                        schedulers_set += 1
                    except:
                        pass  # Some devices don't support scheduler changes
            
            if schedulers_set > 0:
                logger.info(f"Set I/O scheduler to {scheduler} on {schedulers_set} devices")
                self.applied_optimizations.append(f"io_scheduler_{scheduler}")
            
            return True
            
        except Exception as e:
            logger.error(f"I/O scheduler configuration failed: {e}")
            return False
    
    def _configure_readahead(self) -> bool:
        """Configure readahead settings"""
        try:
            readahead_kb = str(self.config.readahead_kb)
            
            # Apply to all block devices
            devices_configured = 0
            
            for device in os.listdir("/sys/block"):
                readahead_file = f"/sys/block/{device}/queue/read_ahead_kb"
                if os.path.exists(readahead_file):
                    try:
                        with open(readahead_file, 'w') as f:
                            f.write(readahead_kb)
                        devices_configured += 1
                    except:
                        pass
            
            logger.info(f"Set readahead to {readahead_kb}KB on {devices_configured} devices")
            self.applied_optimizations.append(f"readahead_{readahead_kb}KB")
            return True
            
        except Exception as e:
            logger.error(f"Readahead configuration failed: {e}")
            return False
    
    def _configure_clocksource(self) -> bool:
        """Configure system clocksource"""
        try:
            clocksource = self.config.clocksource
            
            clocksource_file = "/sys/devices/system/clocksource/clocksource0/current_clocksource"
            if os.path.exists(clocksource_file):
                with open(clocksource_file, 'w') as f:
                    f.write(clocksource)
                
                logger.info(f"Set clocksource to {clocksource}")
                self.applied_optimizations.append(f"clocksource_{clocksource}")
                return True
            else:
                logger.warning("Clocksource configuration not available")
                return False
                
        except Exception as e:
            logger.error(f"Clocksource configuration failed: {e}")
            return False
    
    def _configure_process_limits(self) -> bool:
        """Configure process limits for RT applications"""
        try:
            import resource
            
            # Set memory lock limit
            if self.config.lock_memory:
                max_locked_memory = self.config.max_locked_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (max_locked_memory, max_locked_memory))
                
                logger.info(f"Set memory lock limit to {self.config.max_locked_memory_mb}MB")
            
            # Set real-time priority limit
            try:
                resource.setrlimit(resource.RLIMIT_RTPRIO, (99, 99))
                logger.info("Set RT priority limit to 99")
            except:
                pass  # May not be available
            
            self.applied_optimizations.append("process_limits")
            return True
            
        except Exception as e:
            logger.error(f"Process limits configuration failed: {e}")
            return False
    
    def _configure_network_buffers(self) -> bool:
        """Configure network buffer sizes"""
        try:
            configured = 0
            
            for param, value in self.config.network_buffer_sizes.items():
                param_file = f"/proc/sys/{param.replace('.', '/')}"
                if os.path.exists(param_file):
                    with open(param_file, 'w') as f:
                        f.write(str(value))
                    configured += 1
            
            if configured > 0:
                logger.info(f"Configured {configured} network parameters")
                self.applied_optimizations.append("network_buffers")
            
            return True
            
        except Exception as e:
            logger.error(f"Network buffer configuration failed: {e}")
            return False
    
    def _apply_kernel_parameters(self) -> bool:
        """Apply kernel parameters"""
        try:
            applied = 0
            
            for param, value in self.config.kernel_params.items():
                if not value:  # Skip empty values
                    continue
                
                # Convert parameter name to sysctl format
                param_path = param.replace('.', '/')
                sysctl_file = f"/proc/sys/{param_path}"
                
                if os.path.exists(sysctl_file):
                    try:
                        with open(sysctl_file, 'w') as f:
                            f.write(str(value))
                        applied += 1
                    except:
                        pass  # Some parameters may be read-only
            
            if applied > 0:
                logger.info(f"Applied {applied} kernel parameters")
                self.applied_optimizations.append("kernel_params")
            
            return True
            
        except Exception as e:
            logger.error(f"Kernel parameter application failed: {e}")
            return False
    
    def validate_rt_performance(self) -> Dict[str, Any]:
        """Validate real-time performance of the system"""
        logger.info("Validating real-time performance...")
        
        validation_results = {
            'timing_tests': {},
            'system_checks': {},
            'recommendations': []
        }
        
        # Test timing precision
        validation_results['timing_tests'] = self._test_timing_precision()
        
        # Check system configuration
        validation_results['system_checks'] = self._check_rt_system_config()
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_rt_recommendations()
        
        return validation_results
    
    def _test_timing_precision(self) -> Dict[str, Any]:
        """Test system timing precision"""
        # Test high-precision sleep
        sleep_times = []
        target_sleep_us = 100  # 100 microseconds
        
        for _ in range(1000):
            start = time.perf_counter()
            time.sleep(target_sleep_us / 1_000_000)
            end = time.perf_counter()
            
            actual_sleep_us = (end - start) * 1_000_000
            sleep_times.append(actual_sleep_us)
        
        sleep_times = np.array(sleep_times)
        
        return {
            'target_sleep_us': target_sleep_us,
            'mean_sleep_us': float(np.mean(sleep_times)),
            'std_sleep_us': float(np.std(sleep_times)),
            'max_jitter_us': float(np.max(sleep_times) - np.min(sleep_times)),
            'p99_sleep_us': float(np.percentile(sleep_times, 99)),
            'samples': len(sleep_times),
        }
    
    def _check_rt_system_config(self) -> Dict[str, Any]:
        """Check real-time system configuration"""
        checks = {}
        
        # Check kernel preemption
        try:
            with open("/boot/config-" + platform.release(), 'r') as f:
                config_content = f.read()
                checks['preempt_rt'] = 'CONFIG_PREEMPT_RT=y' in config_content
        except:
            checks['preempt_rt'] = False
        
        # Check CPU governor
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", 'r') as f:
                governor = f.read().strip()
                checks['performance_governor'] = governor == 'performance'
        except:
            checks['performance_governor'] = False
        
        # Check swap status
        try:
            result = subprocess.run("cat /proc/swaps", shell=True, capture_output=True, text=True)
            checks['swap_disabled'] = len(result.stdout.strip().split('\n')) <= 1
        except:
            checks['swap_disabled'] = False
        
        # Check IRQ balance
        try:
            result = subprocess.run("systemctl is-active irqbalance", shell=True, capture_output=True, text=True)
            checks['irqbalance_disabled'] = result.stdout.strip() != 'active'
        except:
            checks['irqbalance_disabled'] = False
        
        return checks
    
    def _generate_rt_recommendations(self) -> List[str]:
        """Generate recommendations for RT performance improvement"""
        recommendations = []
        
        # Check hardware recommendations
        hw_recommendations = self.hardware_detector.get_optimization_recommendations()
        for category, recs in hw_recommendations.items():
            recommendations.extend(recs)
        
        # Check applied optimizations
        if 'cpu_governor_performance' not in self.applied_optimizations:
            recommendations.append("Set CPU governor to performance mode")
        
        if 'swap_disabled' not in self.applied_optimizations:
            recommendations.append("Disable system swap")
        
        if 'irq_balance_disabled' not in self.applied_optimizations:
            recommendations.append("Disable IRQ balancing")
        
        if not self.config.isolate_cpus:
            recommendations.append("Configure CPU isolation for RT threads")
        
        return recommendations
    
    def restore_original_settings(self):
        """Restore original system settings"""
        logger.info("Restoring original system settings...")
        
        try:
            # Restore CPU governors
            if 'cpu_governors' in self.original_settings:
                governors = self.original_settings['cpu_governors']
                for cpu, governor in enumerate(governors):
                    gov_file = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                    if os.path.exists(gov_file):
                        with open(gov_file, 'w') as f:
                            f.write(governor)
            
            # Re-enable services if they were disabled
            if 'irq_balance_disabled' in self.applied_optimizations:
                subprocess.run("systemctl enable irqbalance", shell=True, capture_output=True)
                subprocess.run("systemctl start irqbalance", shell=True, capture_output=True)
            
            logger.info("System settings restored")
            
        except Exception as e:
            logger.error(f"Failed to restore some settings: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'applied_optimizations': self.applied_optimizations.copy(),
            'hardware_info': self.hardware_detector.hardware_info,
            'config': self.config,
            'is_privileged': self.is_privileged,
        }
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Optionally restore settings on exit
        # self.restore_original_settings()
        pass


# Example usage
def main():
    """Example usage of RT system optimization"""
    
    # Create configuration
    config = RTSystemConfig(
        cpu_governor=CPUGovernor.PERFORMANCE,
        disable_swap=True,
        disable_irq_balance=True,
        io_scheduler=IOScheduler.NOOP,
    )
    
    # Create and use optimizer
    with RTSystemOptimizer(config) as optimizer:
        
        # Apply optimizations
        results = optimizer.apply_all_optimizations()
        
        print("Optimization Results:")
        for optimization, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {optimization}")
        
        # Validate performance
        validation = optimizer.validate_rt_performance()
        
        print(f"\nTiming Test Results:")
        timing = validation['timing_tests']
        print(f"  Target sleep: {timing['target_sleep_us']}μs")
        print(f"  Actual mean: {timing['mean_sleep_us']:.1f}μs")
        print(f"  Jitter (std): {timing['std_sleep_us']:.1f}μs")
        print(f"  Max jitter: {timing['max_jitter_us']:.1f}μs")
        print(f"  P99 sleep: {timing['p99_sleep_us']:.1f}μs")
        
        print(f"\nSystem Configuration Checks:")
        for check, passed in validation['system_checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        if validation['recommendations']:
            print(f"\nRecommendations:")
            for rec in validation['recommendations']:
                print(f"  • {rec}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example (requires root for most optimizations)
    main()