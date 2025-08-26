"""
Safe RL Human-Robot Shared Control System
=========================================

A comprehensive safe reinforcement learning system for human-robot interaction
with production-grade safety, performance monitoring, and hardware integration.

Key Components:
- Safe RL algorithms with constraint optimization
- Hardware abstraction for multiple robot types  
- Production-ready safety systems
- Configuration management
- Performance monitoring and validation

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Safe RL Development Team"

# Core module imports with graceful degradation
try:
    from . import src
    CORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core modules not fully available: {e}")
    CORE_AVAILABLE = False

# Optional component imports
try:
    from .src import integration
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    from .src import hardware
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

try:
    from .src import safety
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

# System status
SYSTEM_STATUS = {
    'core_available': CORE_AVAILABLE,
    'integration_available': INTEGRATION_AVAILABLE,
    'hardware_available': HARDWARE_AVAILABLE,
    'safety_available': SAFETY_AVAILABLE,
    'version': __version__
}

def get_system_status():
    """Get current system availability status."""
    return SYSTEM_STATUS.copy()

def check_dependencies():
    """Check system dependencies and return status."""
    missing_deps = []
    
    if not CORE_AVAILABLE:
        missing_deps.append("core modules")
    if not INTEGRATION_AVAILABLE:
        missing_deps.append("integration layer")
    if not HARDWARE_AVAILABLE:
        missing_deps.append("hardware interfaces")
    if not SAFETY_AVAILABLE:
        missing_deps.append("safety systems")
    
    return {
        'status': 'ready' if not missing_deps else 'degraded',
        'missing_dependencies': missing_deps,
        'available_components': [k for k, v in SYSTEM_STATUS.items() if v and k.endswith('_available')]
    }

# Export key functions
__all__ = [
    '__version__',
    'get_system_status',
    'check_dependencies',
    'SYSTEM_STATUS'
]