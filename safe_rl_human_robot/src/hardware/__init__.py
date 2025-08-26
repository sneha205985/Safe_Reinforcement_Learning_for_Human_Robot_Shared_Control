"""
Hardware integration package for Safe RL Human-Robot Shared Control.

This package provides hardware abstraction layers, real-time interfaces,
and production-ready deployment capabilities for robotic systems.
"""

try:
    from .hardware_interface import HardwareInterface, SafetyStatus, HardwareConfig
except ImportError as e:
    print(f"Warning: Could not import hardware_interface: {e}")
    HardwareInterface = SafetyStatus = HardwareConfig = None

try:
    from .exoskeleton_interface import ExoskeletonInterface
except ImportError as e:
    print(f"Warning: Could not import exoskeleton_interface: {e}")
    ExoskeletonInterface = None

try:
    from .wheelchair_interface import WheelchairInterface  
except ImportError as e:
    print(f"Warning: Could not import wheelchair_interface: {e}")
    WheelchairInterface = None

try:
    from .sensor_interface import SensorInterface, IMUInterface, ForceInterface, EncoderInterface
except ImportError as e:
    print(f"Warning: Could not import sensor_interface: {e}")
    SensorInterface = IMUInterface = ForceInterface = EncoderInterface = None

try:
    from .safety_hardware import SafetyHardware, EmergencyStop, PhysicalConstraints
except ImportError as e:
    print(f"Warning: Could not import safety_hardware: {e}")
    SafetyHardware = EmergencyStop = PhysicalConstraints = None

# Try to import production interfaces
try:
    from .production_interfaces import (
        ProductionHardwareInterface,
        ProductionSafetySystem,
        HardwareSimulator,
        ROSHardwareInterface,
        HardwareCalibrator
    )
except ImportError as e:
    print(f"Warning: Could not import production_interfaces: {e}")
    ProductionHardwareInterface = ProductionSafetySystem = None
    HardwareSimulator = ROSHardwareInterface = HardwareCalibrator = None

__all__ = [
    # Base Hardware Interface
    "HardwareInterface",
    "SafetyStatus", 
    "HardwareConfig",
    
    # Specific Hardware Implementations
    "ExoskeletonInterface",
    "WheelchairInterface",
    
    # Sensor Interfaces
    "SensorInterface",
    "IMUInterface",
    "ForceInterface", 
    "EncoderInterface",
    
    # Safety Hardware
    "SafetyHardware",
    "EmergencyStop",
    "PhysicalConstraints"
]