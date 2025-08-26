"""
State-of-the-art baseline implementations for comprehensive Safe RL benchmarking.

This package provides implementations of advanced Safe RL methods and classical
control baselines for rigorous scientific comparison with CPO.

Components:
- Advanced Safe RL Methods: SAC-Lagrangian, TD3-C, TRPO-C, PPO-Lagrangian, Safe-DDPG, RCPO
- Classical Control: MPC, LQR, PID, Impedance/Admittance control
- Evaluation Framework: Standardized benchmarking with statistical analysis
"""

from .safe_rl import (
    SACLagrangian,
    TD3Constrained,
    TRPOConstrained,
    PPOLagrangian,
    SafeDDPG,
    RCPO,
    CPOVariants
)

from .classical_control import (
    MPCController,
    LQRController,
    PIDController,
    ImpedanceControl,
    AdmittanceControl,
    ControllerConfig
)

from .base_algorithm import BaselineAlgorithm, AlgorithmConfig

__all__ = [
    'SACLagrangian',
    'TD3Constrained',
    'TRPOConstrained', 
    'PPOLagrangian',
    'SafeDDPG',
    'RCPO',
    'CPOVariants',
    'MPCController',
    'LQRController',
    'PIDController',
    'ImpedanceControl',
    'AdmittanceControl',
    'BaselineAlgorithm',
    'AlgorithmConfig',
    'ControllerConfig'
]