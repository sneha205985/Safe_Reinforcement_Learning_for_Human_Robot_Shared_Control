"""
Safe Reinforcement Learning for Human-Robot Shared Control

This package implements Constrained Policy Optimization (CPO) methods for safe
reinforcement learning in human-robot collaborative systems.
"""

import sys
from pathlib import Path

# Add current directory to path for proper imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

__version__ = "1.0.0"
__author__ = "Safe RL Development Team"

# Core module imports with graceful degradation
try:
    from .core.constraints import SafetyConstraint
    from .core.policy import SafePolicy
    from .core.lagrangian import LagrangianOptimizer
    from .environments.human_robot_env import HumanRobotEnv
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core RL modules not fully available: {e}")
    CORE_AVAILABLE = False
    SafetyConstraint = None
    SafePolicy = None
    LagrangianOptimizer = None
    HumanRobotEnv = None

# Integration layer imports
try:
    from . import integration
    INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Integration layer not available: {e}")
    INTEGRATION_AVAILABLE = False
    integration = None

# Hardware interface imports
try:
    from . import hardware
    HARDWARE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Hardware interfaces not available: {e}")
    HARDWARE_AVAILABLE = False
    hardware = None

# Safety system imports
try:
    from .core.safety_monitor import SafetyMonitor
    SAFETY_AVAILABLE = True
    safety = {'SafetyMonitor': SafetyMonitor}
except ImportError as e:
    print(f"Warning: Safety systems not available: {e}")
    SAFETY_AVAILABLE = False
    safety = None

# Module status
MODULE_STATUS = {
    'core_available': CORE_AVAILABLE,
    'integration_available': INTEGRATION_AVAILABLE,
    'hardware_available': HARDWARE_AVAILABLE,
    'safety_available': SAFETY_AVAILABLE,
    'version': __version__
}

def get_module_status():
    """Get current module availability status."""
    return MODULE_STATUS.copy()

__all__ = [
    "SafetyConstraint",
    "SafePolicy", 
    "LagrangianOptimizer",
    "HumanRobotEnv",
    "integration",
    "hardware", 
    "safety",
    "get_module_status",
    "MODULE_STATUS"
]