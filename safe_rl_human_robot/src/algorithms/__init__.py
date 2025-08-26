"""
CPO Algorithm implementations for Safe Reinforcement Learning.

This package implements Constrained Policy Optimization (CPO) with trust region methods,
advantage estimation, and safety monitoring for human-robot shared control.
"""

from .cpo import CPOAlgorithm, CPOConfig, CPOState
from .trust_region import TrustRegionSolver, LineSearchResult
from .gae import GeneralizedAdvantageEstimation, ValueFunction
from ..core.safety_monitor import SafetyMonitor

__all__ = [
    "CPOAlgorithm",
    "CPOConfig", 
    "CPOState",
    "TrustRegionSolver",
    "LineSearchResult",
    "GeneralizedAdvantageEstimation",
    "ValueFunction",
    "SafetyMonitor"
]