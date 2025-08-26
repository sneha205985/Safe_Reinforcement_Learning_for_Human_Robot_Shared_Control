"""
Real-time Control System for Safe RL Human-Robot Shared Control.

This package provides real-time control capabilities with deterministic timing,
high-frequency control loops, and hardware synchronization for production
robotics applications.

Key Features:
- Real-time control loops with <1ms latency guarantees
- Deterministic timing with priority scheduling
- High-frequency safety monitoring at 2000Hz
- Hardware-in-the-loop synchronization
- Multi-threaded real-time architecture
- Control loop frequency up to 10kHz
- RT-preempt kernel integration
- Lock-free data structures for performance

Components:
- RealTimeController: Main real-time control coordinator
- ControlLoop: High-frequency control loop implementation
- TimingManager: Deterministic timing and scheduling
- RTSafetyMonitor: Real-time safety monitoring
- HardwareSync: Hardware synchronization interface
- RTCommunication: Real-time communication protocols
"""

from .realtime_controller import RealTimeController, RTConfig
from .control_loop import ControlLoop, LoopConfig, LoopState
from .timing_manager import TimingManager, TimingConfig, RTTimer
from .safety_monitor import RTSafetyMonitor, SafetyConfig
from .hardware_sync import HardwareSync, SyncConfig
from .communication import RTCommunication, CommConfig

__all__ = [
    # Core Real-time Components
    "RealTimeController",
    "RTConfig",
    
    # Control Loop
    "ControlLoop",
    "LoopConfig", 
    "LoopState",
    
    # Timing Management
    "TimingManager",
    "TimingConfig",
    "RTTimer",
    
    # Safety Monitoring
    "RTSafetyMonitor",
    "SafetyConfig",
    
    # Hardware Synchronization
    "HardwareSync",
    "SyncConfig",
    
    # Communication
    "RTCommunication",
    "CommConfig"
]