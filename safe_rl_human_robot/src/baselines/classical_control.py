"""
Classical Control Baselines for Safe RL Benchmarking.

This module implements state-of-the-art classical control methods for comparison
with Safe RL approaches in human-robot shared control scenarios.

Components:
- Model Predictive Control (MPC) with safety constraints
- Linear Quadratic Regulator (LQR) with safety margins
- PID Controllers with adaptive tuning
- Impedance Control for physical human-robot interaction
- Admittance Control for compliant robot behavior
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from abc import abstractmethod
import scipy.optimize
import scipy.linalg
from scipy.integrate import odeint
import control

from .base_algorithm import BaselineAlgorithm, AlgorithmConfig, TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class ControllerConfig(AlgorithmConfig):
    """Configuration for classical controllers."""
    # Common control parameters
    control_frequency: float = 100.0  # Hz
    prediction_horizon: int = 20  # MPC horizon
    control_horizon: int = 10  # MPC control horizon
    
    # Safety parameters
    velocity_limits: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    acceleration_limits: List[float] = field(default_factory=lambda: [-2.0, 2.0])
    force_limits: List[float] = field(default_factory=lambda: [-10.0, 10.0])
    safety_margin: float = 0.1
    
    # Controller-specific parameters
    mpc_weights: Dict[str, float] = field(default_factory=lambda: {
        'position': 1.0, 'velocity': 0.1, 'acceleration': 0.01, 'jerk': 0.001,
        'control_effort': 0.1, 'safety_violation': 100.0
    })
    
    # PID parameters
    pid_gains: Dict[str, float] = field(default_factory=lambda: {
        'kp': 1.0, 'ki': 0.1, 'kd': 0.01
    })
    
    # Impedance/Admittance parameters  
    impedance_params: Dict[str, float] = field(default_factory=lambda: {
        'mass': 1.0, 'damping': 10.0, 'stiffness': 100.0
    })


class MPCController(BaselineAlgorithm):
    """Model Predictive Control with safety constraints and adaptive planning."""
    
    def __init__(self, config: ControllerConfig, state_dim: int, action_dim: int,
                 system_model: Optional[callable] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.system_model = system_model
        
        # MPC specific parameters
        self.prediction_horizon = config.prediction_horizon
        self.control_horizon = config.control_horizon
        self.mpc_weights = config.mpc_weights
        
        # Safety constraints
        self.velocity_limits = np.array(config.velocity_limits)
        self.acceleration_limits = np.array(config.acceleration_limits)
        self.force_limits = np.array(config.force_limits)
        self.safety_margin = config.safety_margin
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize MPC specific components."""
        # State and control matrices (linear system approximation)
        self.A = np.eye(self.state_dim)  # State transition matrix
        self.B = np.random.randn(self.state_dim, self.action_dim) * 0.1  # Control matrix
        
        # Cost matrices
        self.Q = np.eye(self.state_dim) * self.mpc_weights['position']  # State cost
        self.R = np.eye(self.action_dim) * self.mpc_weights['control_effort']  # Control cost
        self.Qf = self.Q * 10  # Terminal cost
        
        # Reference trajectory storage
        self.reference_trajectory = None
        self.current_reference = np.zeros(self.state_dim)
        
        # MPC optimization history
        self.control_sequence = np.zeros((self.control_horizon, self.action_dim))
        self.predicted_states = np.zeros((self.prediction_horizon + 1, self.state_dim))
        
        # Safety monitoring
        self.constraint_violations = []
        self.safety_costs = []
        
        logger.info("MPC Controller initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Compute MPC control action."""
        current_state = observation.copy()
        
        # Update system identification if available
        if hasattr(self, 'system_model') and self.system_model is not None:
            self._update_system_model(current_state)
        
        # Solve MPC optimization problem
        optimal_control, info = self._solve_mpc(current_state)
        
        # Safety filtering
        safe_control = self._apply_safety_filter(optimal_control, current_state)
        
        # Update control sequence (warm start for next iteration)
        self.control_sequence[:-1] = self.control_sequence[1:]
        self.control_sequence[-1] = safe_control
        
        return safe_control, info
    
    def _solve_mpc(self, current_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Solve the MPC optimization problem."""
        # Decision variables: control sequence
        x0 = self.control_sequence.flatten()
        
        # Define optimization problem
        def objective(u_seq):
            u_seq = u_seq.reshape(self.control_horizon, self.action_dim)
            return self._compute_mpc_cost(current_state, u_seq)
        
        def constraints(u_seq):
            u_seq = u_seq.reshape(self.control_horizon, self.action_dim)
            return self._compute_constraints(current_state, u_seq)
        
        # Define bounds
        bounds = []
        for _ in range(self.control_horizon):
            for i in range(self.action_dim):
                bounds.append((self.force_limits[0], self.force_limits[1]))
        
        # Define constraint dictionary
        constraint_dict = {
            'type': 'ineq',
            'fun': constraints
        }
        
        # Solve optimization
        try:
            result = scipy.optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=[constraint_dict],
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                optimal_sequence = result.x.reshape(self.control_horizon, self.action_dim)
                optimal_control = optimal_sequence[0]
                
                # Store predicted trajectory
                self.predicted_states = self._predict_trajectory(current_state, optimal_sequence)
                
                info = {
                    'optimization_success': True,
                    'cost': result.fun,
                    'iterations': result.nit,
                    'predicted_states': self.predicted_states
                }
            else:
                # Fallback to previous solution or zero control
                optimal_control = self.control_sequence[0] * 0.8  # Reduced control
                info = {
                    'optimization_success': False,
                    'cost': float('inf'),
                    'iterations': 0,
                    'error': result.message
                }
        
        except Exception as e:
            logger.warning(f"MPC optimization failed: {e}")
            optimal_control = np.zeros(self.action_dim)
            info = {
                'optimization_success': False,
                'cost': float('inf'),
                'iterations': 0,
                'error': str(e)
            }
        
        return optimal_control, info
    
    def _compute_mpc_cost(self, initial_state: np.ndarray, control_sequence: np.ndarray) -> float:
        """Compute MPC objective function."""
        total_cost = 0.0
        state = initial_state.copy()
        
        for t in range(self.control_horizon):
            # State cost
            state_error = state - self.current_reference
            state_cost = state_error.T @ self.Q @ state_error
            
            # Control cost
            control_cost = control_sequence[t].T @ self.R @ control_sequence[t]
            
            # Safety cost
            safety_cost = self._compute_safety_cost(state, control_sequence[t])
            
            total_cost += state_cost + control_cost + safety_cost
            
            # Predict next state
            state = self.A @ state + self.B @ control_sequence[t]
        
        # Terminal cost
        terminal_error = state - self.current_reference
        terminal_cost = terminal_error.T @ self.Qf @ terminal_error
        total_cost += terminal_cost
        
        return total_cost
    
    def _compute_safety_cost(self, state: np.ndarray, control: np.ndarray) -> float:
        """Compute safety-related costs."""
        safety_cost = 0.0
        
        # Velocity constraints
        if len(state) > self.state_dim // 2:  # Assume second half is velocities
            velocities = state[self.state_dim // 2:]
            velocity_violations = np.maximum(0, np.abs(velocities) - np.abs(self.velocity_limits))
            safety_cost += self.mpc_weights['safety_violation'] * np.sum(velocity_violations**2)
        
        # Control effort constraints
        force_violations = np.maximum(0, np.abs(control) - np.abs(self.force_limits))
        safety_cost += self.mpc_weights['safety_violation'] * np.sum(force_violations**2)
        
        return safety_cost
    
    def _compute_constraints(self, initial_state: np.ndarray, control_sequence: np.ndarray) -> np.ndarray:
        """Compute constraint violations (should be >= 0 for feasibility)."""
        constraints = []
        state = initial_state.copy()
        
        for t in range(self.control_horizon):
            # Control constraints
            control_lower = control_sequence[t] - self.force_limits[0]
            control_upper = self.force_limits[1] - control_sequence[t]
            constraints.extend(control_lower)
            constraints.extend(control_upper)
            
            # State constraints (velocity limits)
            if len(state) > self.state_dim // 2:
                velocities = state[self.state_dim // 2:]
                vel_lower = velocities - self.velocity_limits[0]
                vel_upper = self.velocity_limits[1] - velocities
                constraints.extend(vel_lower)
                constraints.extend(vel_upper)
            
            # Predict next state
            state = self.A @ state + self.B @ control_sequence[t]
        
        return np.array(constraints)
    
    def _predict_trajectory(self, initial_state: np.ndarray, 
                          control_sequence: np.ndarray) -> np.ndarray:
        """Predict state trajectory given control sequence."""
        trajectory = np.zeros((self.prediction_horizon + 1, self.state_dim))
        trajectory[0] = initial_state
        
        for t in range(self.control_horizon):
            trajectory[t + 1] = self.A @ trajectory[t] + self.B @ control_sequence[t]
        
        # Free evolution for remaining horizon
        for t in range(self.control_horizon, self.prediction_horizon):
            trajectory[t + 1] = self.A @ trajectory[t]
        
        return trajectory
    
    def _apply_safety_filter(self, control: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply safety filtering to control command."""
        safe_control = np.clip(control, self.force_limits[0], self.force_limits[1])
        
        # Additional safety checks
        if len(state) > self.state_dim // 2:
            velocities = state[self.state_dim // 2:]
            
            # Prevent acceleration if already at velocity limit
            for i in range(len(safe_control)):
                if i < len(velocities):
                    if velocities[i] >= self.velocity_limits[1] - self.safety_margin:
                        safe_control[i] = min(safe_control[i], 0)  # No positive acceleration
                    elif velocities[i] <= self.velocity_limits[0] + self.safety_margin:
                        safe_control[i] = max(safe_control[i], 0)  # No negative acceleration
        
        return safe_control
    
    def _update_system_model(self, current_state: np.ndarray):
        """Update system model matrices based on current state (online identification)."""
        # Placeholder for online system identification
        # In practice, this would use recent state-control-next_state data
        pass
    
    def set_reference(self, reference: np.ndarray):
        """Set reference trajectory for tracking."""
        self.current_reference = reference.copy()
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'MPCController':
        """MPC doesn't require learning - it's a model-based method."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        for timestep in range(total_timesteps):
            # Get MPC action
            action, info = self.predict(observation)
            
            # Environment step
            next_observation, reward, done, env_info = env.step(action)
            cost = env_info.get('cost', 0.0)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Update reference if available
            if 'reference' in env_info:
                self.set_reference(env_info['reference'])
            
            # Record constraint violations
            if not info.get('optimization_success', False):
                self.constraint_violations.append(timestep)
            
            self.safety_costs.append(cost)
            
            if done:
                # Record episode metrics
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit,
                    optimization_success_rate=1.0 - len(self.constraint_violations) / max(1, timestep),
                    average_safety_cost=np.mean(self.safety_costs) if self.safety_costs else 0.0
                )
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
                self.constraint_violations = []
                self.safety_costs = []
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """MPC doesn't require batch updates."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get MPC specific state for saving."""
        return {
            'A': self.A,
            'B': self.B,
            'Q': self.Q,
            'R': self.R,
            'Qf': self.Qf,
            'control_sequence': self.control_sequence,
            'current_reference': self.current_reference
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load MPC specific state."""
        self.A = state['A']
        self.B = state['B']
        self.Q = state['Q']
        self.R = state['R']
        self.Qf = state['Qf']
        self.control_sequence = state['control_sequence']
        self.current_reference = state['current_reference']


class LQRController(BaselineAlgorithm):
    """Linear Quadratic Regulator with safety margins and adaptive gains."""
    
    def __init__(self, config: ControllerConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Safety constraints
        self.velocity_limits = np.array(config.velocity_limits)
        self.force_limits = np.array(config.force_limits)
        self.safety_margin = config.safety_margin
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize LQR specific components."""
        # System matrices
        self.A = np.eye(self.state_dim)
        self.B = np.random.randn(self.state_dim, self.action_dim) * 0.1
        
        # Cost matrices
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim) * 0.1
        
        # Compute LQR gain
        self._compute_lqr_gain()
        
        # Reference state
        self.reference_state = np.zeros(self.state_dim)
        
        logger.info("LQR Controller initialized")
    
    def _compute_lqr_gain(self):
        """Compute LQR feedback gain matrix."""
        try:
            # Solve continuous-time algebraic Riccati equation
            P = scipy.linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
            
            # Compute feedback gain
            self.K = np.linalg.inv(self.R) @ self.B.T @ P
            
            logger.info(f"LQR gain computed: K = {self.K}")
            
        except Exception as e:
            logger.warning(f"LQR gain computation failed: {e}")
            # Fallback to simple proportional gain
            self.K = np.eye(self.action_dim, self.state_dim) * 0.1
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Compute LQR control action."""
        state_error = observation - self.reference_state
        
        # Basic LQR control law
        control = -self.K @ state_error
        
        # Apply safety constraints
        safe_control = self._apply_safety_constraints(control, observation)
        
        info = {
            'lqr_control': control,
            'safety_modified': not np.allclose(control, safe_control),
            'state_error_norm': np.linalg.norm(state_error)
        }
        
        return safe_control, info
    
    def _apply_safety_constraints(self, control: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply safety constraints to LQR control."""
        safe_control = np.clip(control, self.force_limits[0], self.force_limits[1])
        
        # Velocity-based safety modification
        if len(state) > self.state_dim // 2:
            velocities = state[self.state_dim // 2:]
            
            for i in range(len(safe_control)):
                if i < len(velocities):
                    # Prevent further acceleration if at velocity limits
                    if velocities[i] >= self.velocity_limits[1] - self.safety_margin:
                        safe_control[i] = min(safe_control[i], -0.1 * velocities[i])
                    elif velocities[i] <= self.velocity_limits[0] + self.safety_margin:
                        safe_control[i] = max(safe_control[i], -0.1 * velocities[i])
        
        return safe_control
    
    def set_reference(self, reference: np.ndarray):
        """Set reference state for tracking."""
        self.reference_state = reference.copy()
    
    def adapt_gains(self, performance_metric: float):
        """Adapt LQR gains based on performance."""
        if performance_metric < 0.5:  # Poor performance
            self.Q *= 1.1  # Increase state penalty
            self.R *= 0.9  # Decrease control penalty
        elif performance_metric > 0.9:  # Good performance
            self.Q *= 0.95  # Decrease state penalty
            self.R *= 1.05  # Increase control penalty
        
        # Recompute gain
        self._compute_lqr_gain()
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'LQRController':
        """LQR learning involves gain adaptation."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        performance_history = []
        
        for timestep in range(total_timesteps):
            # Get LQR action
            action, info = self.predict(observation)
            
            # Environment step
            next_observation, reward, done, env_info = env.step(action)
            cost = env_info.get('cost', 0.0)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Update reference if available
            if 'reference' in env_info:
                self.set_reference(env_info['reference'])
            
            if done:
                # Compute performance metric
                performance = max(0, episode_return) / (episode_length + 1e-6)
                performance_history.append(performance)
                
                # Adapt gains periodically
                if len(performance_history) >= 10:
                    avg_performance = np.mean(performance_history[-10:])
                    self.adapt_gains(avg_performance)
                
                # Record episode metrics
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit,
                    performance_metric=performance
                )
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """LQR doesn't require batch updates."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get LQR specific state for saving."""
        return {
            'A': self.A,
            'B': self.B,
            'Q': self.Q,
            'R': self.R,
            'K': self.K,
            'reference_state': self.reference_state
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load LQR specific state."""
        self.A = state['A']
        self.B = state['B']
        self.Q = state['Q']
        self.R = state['R']
        self.K = state['K']
        self.reference_state = state['reference_state']


class PIDController(BaselineAlgorithm):
    """PID Controller with adaptive tuning and anti-windup."""
    
    def __init__(self, config: ControllerConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # PID gains
        self.kp = config.pid_gains['kp']
        self.ki = config.pid_gains['ki']
        self.kd = config.pid_gains['kd']
        
        # Safety constraints
        self.force_limits = np.array(config.force_limits)
        self.velocity_limits = np.array(config.velocity_limits)
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize PID specific components."""
        # PID state variables
        self.integral_error = np.zeros(self.action_dim)
        self.previous_error = np.zeros(self.action_dim)
        self.dt = 1.0 / self.config.control_frequency
        
        # Anti-windup
        self.integral_max = np.abs(self.force_limits[1]) / (self.ki + 1e-6)
        
        # Reference
        self.reference = np.zeros(self.action_dim)
        
        # Adaptive tuning
        self.performance_history = []
        self.gain_adaptation_rate = 0.01
        
        logger.info("PID Controller initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Compute PID control action."""
        # Extract relevant states for control (assume first action_dim states are controlled)
        controlled_states = observation[:self.action_dim]
        error = self.reference - controlled_states
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -self.integral_max, self.integral_max)
        i_term = self.ki * self.integral_error
        
        # Derivative term
        error_derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * error_derivative
        
        # Combined PID output
        pid_output = p_term + i_term + d_term
        
        # Apply safety constraints
        safe_control = np.clip(pid_output, self.force_limits[0], self.force_limits[1])
        
        # Anti-windup correction
        if not np.allclose(pid_output, safe_control):
            # Reduce integral term if saturated
            self.integral_error *= 0.9
        
        # Update previous error
        self.previous_error = error.copy()
        
        info = {
            'error': error,
            'p_term': p_term,
            'i_term': i_term,
            'd_term': d_term,
            'saturated': not np.allclose(pid_output, safe_control)
        }
        
        return safe_control, info
    
    def set_reference(self, reference: np.ndarray):
        """Set reference signal for PID tracking."""
        self.reference = reference[:self.action_dim]
    
    def reset_integral(self):
        """Reset integral term (useful when reference changes)."""
        self.integral_error = np.zeros(self.action_dim)
    
    def auto_tune(self, error_history: List[np.ndarray], control_history: List[np.ndarray]):
        """Auto-tune PID gains based on performance."""
        if len(error_history) < 10:
            return
        
        # Simple Ziegler-Nichols inspired tuning
        recent_errors = np.array(error_history[-10:])
        error_std = np.std(recent_errors, axis=0)
        error_mean = np.mean(np.abs(recent_errors), axis=0)
        
        # Adjust gains based on error characteristics
        for i in range(self.action_dim):
            if error_std[i] > 0.1:  # High oscillation
                self.kp *= (1 - self.gain_adaptation_rate)
                self.kd *= (1 + self.gain_adaptation_rate)
            elif error_mean[i] > 0.1:  # High steady-state error
                self.ki *= (1 + self.gain_adaptation_rate)
                self.kp *= (1 + self.gain_adaptation_rate * 0.5)
            elif error_mean[i] < 0.01:  # Good performance
                self.ki *= (1 - self.gain_adaptation_rate * 0.5)
        
        # Ensure gains remain positive and bounded
        self.kp = np.clip(self.kp, 0.01, 10.0)
        self.ki = np.clip(self.ki, 0.001, 5.0)  
        self.kd = np.clip(self.kd, 0.0, 2.0)
        
        # Update integral limit
        self.integral_max = np.abs(self.force_limits[1]) / (self.ki + 1e-6)
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'PIDController':
        """PID learning involves auto-tuning."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        error_history = []
        control_history = []
        
        for timestep in range(total_timesteps):
            # Get PID action
            action, info = self.predict(observation)
            
            # Environment step
            next_observation, reward, done, env_info = env.step(action)
            cost = env_info.get('cost', 0.0)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Store history for auto-tuning
            error_history.append(info['error'])
            control_history.append(action)
            
            # Update reference if available
            if 'reference' in env_info:
                self.set_reference(env_info['reference'])
                self.reset_integral()  # Reset integral on reference change
            
            # Auto-tune periodically
            if timestep % 100 == 0 and timestep > 0:
                self.auto_tune(error_history, control_history)
                error_history = error_history[-50:]  # Keep recent history
                control_history = control_history[-50:]
            
            if done:
                # Record episode metrics
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit,
                    kp=float(np.mean(self.kp)) if hasattr(self.kp, '__len__') else float(self.kp),
                    ki=float(np.mean(self.ki)) if hasattr(self.ki, '__len__') else float(self.ki),
                    kd=float(np.mean(self.kd)) if hasattr(self.kd, '__len__') else float(self.kd)
                )
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
                error_history = []
                control_history = []
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """PID doesn't require batch updates."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get PID specific state for saving."""
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral_error': self.integral_error,
            'previous_error': self.previous_error,
            'reference': self.reference
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load PID specific state."""
        self.kp = state['kp']
        self.ki = state['ki']
        self.kd = state['kd']
        self.integral_error = state['integral_error']
        self.previous_error = state['previous_error']
        self.reference = state['reference']


class ImpedanceControl(BaselineAlgorithm):
    """Impedance Control for physical human-robot interaction with adaptive parameters."""
    
    def __init__(self, config: ControllerConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Impedance parameters
        self.mass = config.impedance_params['mass']
        self.damping = config.impedance_params['damping']
        self.stiffness = config.impedance_params['stiffness']
        
        # Safety constraints
        self.force_limits = np.array(config.force_limits)
        self.velocity_limits = np.array(config.velocity_limits)
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize Impedance Control specific components."""
        # Impedance matrices (assume 3D for now, can be generalized)
        self.M = np.eye(3) * self.mass  # Mass matrix
        self.D = np.eye(3) * self.damping  # Damping matrix
        self.K = np.eye(3) * self.stiffness  # Stiffness matrix
        
        # Desired trajectory
        self.desired_position = np.zeros(3)
        self.desired_velocity = np.zeros(3)
        self.desired_acceleration = np.zeros(3)
        
        # External force estimation
        self.external_force = np.zeros(3)
        self.force_filter_alpha = 0.1  # Low-pass filter coefficient
        
        # Adaptive parameters
        self.adaptation_rate = 0.01
        self.interaction_threshold = 0.5  # Force threshold for interaction detection
        
        # Control frequency
        self.dt = 1.0 / self.config.control_frequency
        
        logger.info("Impedance Controller initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Compute impedance control forces."""
        # Extract position and velocity from state (assume structured state)
        current_position = observation[:3]
        current_velocity = observation[3:6] if len(observation) >= 6 else np.zeros(3)
        
        # Compute position and velocity errors
        position_error = self.desired_position - current_position
        velocity_error = self.desired_velocity - current_velocity
        
        # Impedance control law
        # F = M*a_d + D*(v_d - v) + K*(x_d - x) + F_ext
        impedance_force = (
            self.M @ self.desired_acceleration +
            self.D @ velocity_error +
            self.K @ position_error +
            self.external_force
        )
        
        # Apply safety constraints
        safe_force = self._apply_safety_constraints(impedance_force, current_velocity)
        
        # Adaptive parameter adjustment
        self._adapt_impedance_parameters(position_error, velocity_error, self.external_force)
        
        info = {
            'position_error': position_error,
            'velocity_error': velocity_error,
            'external_force': self.external_force,
            'impedance_force': impedance_force,
            'mass': np.diag(self.M),
            'damping': np.diag(self.D),
            'stiffness': np.diag(self.K),
            'interaction_detected': np.linalg.norm(self.external_force) > self.interaction_threshold
        }
        
        return safe_force[:self.action_dim], info
    
    def _apply_safety_constraints(self, force: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Apply safety constraints to impedance force."""
        # Clip forces to limits
        safe_force = np.clip(force, self.force_limits[0], self.force_limits[1])
        
        # Additional velocity-based constraints
        for i in range(len(safe_force)):
            if i < len(velocity):
                # Reduce force if approaching velocity limits
                if velocity[i] > self.velocity_limits[1] * 0.9:
                    safe_force[i] = min(safe_force[i], -0.1 * velocity[i])
                elif velocity[i] < self.velocity_limits[0] * 0.9:
                    safe_force[i] = max(safe_force[i], -0.1 * velocity[i])
        
        return safe_force
    
    def _adapt_impedance_parameters(self, position_error: np.ndarray, 
                                  velocity_error: np.ndarray, external_force: np.ndarray):
        """Adapt impedance parameters based on interaction."""
        force_magnitude = np.linalg.norm(external_force)
        error_magnitude = np.linalg.norm(position_error)
        
        # Adapt stiffness based on position error
        if error_magnitude > 0.1:  # Large position error
            self.K += np.eye(3) * self.adaptation_rate * error_magnitude
        elif error_magnitude < 0.01:  # Small position error
            self.K -= np.eye(3) * self.adaptation_rate * 0.1
        
        # Adapt damping based on external force
        if force_magnitude > self.interaction_threshold:  # Active interaction
            self.D += np.eye(3) * self.adaptation_rate * force_magnitude
        else:  # Free motion
            self.D -= np.eye(3) * self.adaptation_rate * 0.1
        
        # Ensure parameters remain positive and bounded
        self.K = np.clip(self.K, 10.0, 1000.0)
        self.D = np.clip(self.D, 1.0, 100.0)
    
    def set_desired_trajectory(self, position: np.ndarray, velocity: np.ndarray = None, 
                             acceleration: np.ndarray = None):
        """Set desired trajectory for impedance control."""
        self.desired_position = position[:3]
        self.desired_velocity = velocity[:3] if velocity is not None else np.zeros(3)
        self.desired_acceleration = acceleration[:3] if acceleration is not None else np.zeros(3)
    
    def update_external_force(self, measured_force: np.ndarray):
        """Update external force estimate with filtering."""
        # Simple low-pass filter
        self.external_force = (
            self.force_filter_alpha * measured_force[:3] + 
            (1 - self.force_filter_alpha) * self.external_force
        )
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'ImpedanceControl':
        """Impedance control learning involves parameter adaptation."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        interaction_count = 0
        
        for timestep in range(total_timesteps):
            # Get impedance control action
            action, info = self.predict(observation)
            
            # Environment step
            next_observation, reward, done, env_info = env.step(action)
            cost = env_info.get('cost', 0.0)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Update external force if available
            if 'external_force' in env_info:
                self.update_external_force(env_info['external_force'])
            
            # Update desired trajectory if available
            if 'desired_position' in env_info:
                desired_pos = env_info['desired_position']
                desired_vel = env_info.get('desired_velocity', None)
                desired_acc = env_info.get('desired_acceleration', None)
                self.set_desired_trajectory(desired_pos, desired_vel, desired_acc)
            
            # Count interactions
            if info['interaction_detected']:
                interaction_count += 1
            
            if done:
                # Record episode metrics
                interaction_rate = interaction_count / max(1, episode_length)
                
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit,
                    interaction_rate=interaction_rate,
                    avg_stiffness=float(np.mean(np.diag(self.K))),
                    avg_damping=float(np.mean(np.diag(self.D)))
                )
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
                interaction_count = 0
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Impedance control doesn't require batch updates."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get Impedance Control specific state for saving."""
        return {
            'M': self.M,
            'D': self.D,
            'K': self.K,
            'desired_position': self.desired_position,
            'desired_velocity': self.desired_velocity,
            'desired_acceleration': self.desired_acceleration,
            'external_force': self.external_force
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load Impedance Control specific state."""
        self.M = state['M']
        self.D = state['D']
        self.K = state['K']
        self.desired_position = state['desired_position']
        self.desired_velocity = state['desired_velocity']
        self.desired_acceleration = state['desired_acceleration']
        self.external_force = state['external_force']


class AdmittanceControl(BaselineAlgorithm):
    """Admittance Control for compliant robot behavior with force-based adaptation."""
    
    def __init__(self, config: ControllerConfig, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Admittance parameters (inverse of impedance)
        self.mass = config.impedance_params['mass']
        self.damping = config.impedance_params['damping'] 
        self.stiffness = config.impedance_params['stiffness']
        
        # Safety constraints
        self.velocity_limits = np.array(config.velocity_limits)
        self.force_limits = np.array(config.force_limits)
        
        super().__init__(config)
    
    def _initialize_algorithm(self):
        """Initialize Admittance Control specific components."""
        # Admittance matrices (3D space)
        self.M_inv = np.eye(3) / self.mass  # Inverse mass matrix
        self.D_inv = np.eye(3) / self.damping  # Inverse damping matrix
        self.K_inv = np.eye(3) / self.stiffness  # Inverse stiffness matrix
        
        # Reference trajectory
        self.reference_position = np.zeros(3)
        self.reference_velocity = np.zeros(3)
        
        # Admittance model state
        self.admittance_position = np.zeros(3)
        self.admittance_velocity = np.zeros(3)
        self.admittance_acceleration = np.zeros(3)
        
        # External force
        self.external_force = np.zeros(3)
        self.force_filter_alpha = 0.1
        
        # Control frequency
        self.dt = 1.0 / self.config.control_frequency
        
        # Adaptive parameters
        self.adaptation_rate = 0.01
        self.force_threshold = 1.0
        
        logger.info("Admittance Controller initialized")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Compute admittance control commands."""
        # Extract current robot state
        current_position = observation[:3]
        current_velocity = observation[3:6] if len(observation) >= 6 else np.zeros(3)
        
        # Update admittance model based on external forces
        self._update_admittance_model()
        
        # Compute position and velocity references from admittance model
        position_reference = self.admittance_position
        velocity_reference = self.admittance_velocity
        
        # Position control to track admittance model
        # This could be PD control or any position controller
        kp_pos = 100.0  # Position gain
        kd_pos = 20.0   # Velocity gain
        
        position_error = position_reference - current_position
        velocity_error = velocity_reference - current_velocity
        
        # Control force/torque
        control_force = kp_pos * position_error + kd_pos * velocity_error
        
        # Apply safety constraints
        safe_control = self._apply_safety_constraints(control_force, current_velocity)
        
        # Adapt admittance parameters
        self._adapt_admittance_parameters()
        
        info = {
            'external_force': self.external_force,
            'admittance_position': self.admittance_position,
            'admittance_velocity': self.admittance_velocity,
            'position_reference': position_reference,
            'velocity_reference': velocity_reference,
            'position_error': position_error,
            'velocity_error': velocity_error,
            'mass': 1.0 / np.mean(np.diag(self.M_inv)),
            'damping': 1.0 / np.mean(np.diag(self.D_inv)),
            'stiffness': 1.0 / np.mean(np.diag(self.K_inv))
        }
        
        return safe_control[:self.action_dim], info
    
    def _update_admittance_model(self):
        """Update admittance model dynamics."""
        # Admittance equation: M*x_ddot + D*x_dot + K*(x - x_ref) = F_ext
        # Rearranged: x_ddot = M_inv * (F_ext - D*x_dot - K*(x - x_ref))
        
        position_error = self.admittance_position - self.reference_position
        velocity_error = self.admittance_velocity - self.reference_velocity
        
        # Compute admittance acceleration
        self.admittance_acceleration = self.M_inv @ (
            self.external_force - 
            np.linalg.inv(self.D_inv) @ velocity_error - 
            np.linalg.inv(self.K_inv) @ position_error
        )
        
        # Integrate to get velocity and position (Euler integration)
        self.admittance_velocity += self.admittance_acceleration * self.dt
        self.admittance_position += self.admittance_velocity * self.dt
        
        # Apply velocity limits to admittance model
        self.admittance_velocity = np.clip(
            self.admittance_velocity, 
            self.velocity_limits[0], 
            self.velocity_limits[1]
        )
    
    def _apply_safety_constraints(self, control: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Apply safety constraints to admittance control."""
        safe_control = np.clip(control, self.force_limits[0], self.force_limits[1])
        
        # Velocity-based safety
        for i in range(len(safe_control)):
            if i < len(velocity):
                if velocity[i] > self.velocity_limits[1] * 0.9:
                    safe_control[i] = min(safe_control[i], -0.2 * velocity[i])
                elif velocity[i] < self.velocity_limits[0] * 0.9:
                    safe_control[i] = max(safe_control[i], -0.2 * velocity[i])
        
        return safe_control
    
    def _adapt_admittance_parameters(self):
        """Adapt admittance parameters based on force interaction."""
        force_magnitude = np.linalg.norm(self.external_force)
        
        if force_magnitude > self.force_threshold:
            # High force - become more compliant (reduce stiffness, increase admittance)
            self.K_inv += np.eye(3) * self.adaptation_rate * force_magnitude / 1000.0
            self.D_inv += np.eye(3) * self.adaptation_rate * force_magnitude / 100.0
        else:
            # Low force - become stiffer (increase stiffness, reduce admittance)
            self.K_inv -= np.eye(3) * self.adaptation_rate * 0.1 / 1000.0
            self.D_inv -= np.eye(3) * self.adaptation_rate * 0.1 / 100.0
        
        # Ensure parameters remain positive and bounded
        self.M_inv = np.clip(self.M_inv, 0.1, 10.0)
        self.D_inv = np.clip(self.D_inv, 0.01, 1.0)
        self.K_inv = np.clip(self.K_inv, 0.001, 0.1)
    
    def set_reference_trajectory(self, position: np.ndarray, velocity: np.ndarray = None):
        """Set reference trajectory for admittance model."""
        self.reference_position = position[:3]
        self.reference_velocity = velocity[:3] if velocity is not None else np.zeros(3)
    
    def update_external_force(self, measured_force: np.ndarray):
        """Update external force estimate."""
        self.external_force = (
            self.force_filter_alpha * measured_force[:3] + 
            (1 - self.force_filter_alpha) * self.external_force
        )
    
    def reset_admittance_model(self, initial_position: np.ndarray = None):
        """Reset admittance model to initial state."""
        if initial_position is not None:
            self.admittance_position = initial_position[:3]
        else:
            self.admittance_position = self.reference_position.copy()
        
        self.admittance_velocity = np.zeros(3)
        self.admittance_acceleration = np.zeros(3)
    
    def learn(self, total_timesteps: int, env, **kwargs) -> 'AdmittanceControl':
        """Admittance control learning involves parameter adaptation."""
        observation = env.reset()
        episode_return = 0.0
        episode_cost = 0.0
        episode_length = 0
        
        force_interaction_time = 0
        
        for timestep in range(total_timesteps):
            # Get admittance control action
            action, info = self.predict(observation)
            
            # Environment step
            next_observation, reward, done, env_info = env.step(action)
            cost = env_info.get('cost', 0.0)
            
            episode_return += reward
            episode_cost += cost
            episode_length += 1
            
            # Update external force if available
            if 'external_force' in env_info:
                self.update_external_force(env_info['external_force'])
            
            # Update reference trajectory if available
            if 'reference_position' in env_info:
                ref_pos = env_info['reference_position']
                ref_vel = env_info.get('reference_velocity', None)
                self.set_reference_trajectory(ref_pos, ref_vel)
            
            # Track force interaction time
            if np.linalg.norm(self.external_force) > self.force_threshold:
                force_interaction_time += 1
            
            if done:
                # Record episode metrics
                interaction_ratio = force_interaction_time / max(1, episode_length)
                
                metrics = TrainingMetrics(
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    episode_length=episode_length,
                    constraint_violation=episode_cost > self.config.cost_limit,
                    force_interaction_ratio=interaction_ratio,
                    avg_mass=float(1.0 / np.mean(np.diag(self.M_inv))),
                    avg_damping=float(1.0 / np.mean(np.diag(self.D_inv))),
                    avg_stiffness=float(1.0 / np.mean(np.diag(self.K_inv)))
                )
                
                self.record_training_metrics(metrics)
                
                # Reset episode
                observation = env.reset()
                episode_return = 0.0
                episode_cost = 0.0
                episode_length = 0
                force_interaction_time = 0
                
                # Reset admittance model for new episode
                self.reset_admittance_model(observation[:3])
            else:
                observation = next_observation
            
            self.total_timesteps = timestep + 1
        
        return self
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Admittance control doesn't require batch updates."""
        return {}
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get Admittance Control specific state for saving."""
        return {
            'M_inv': self.M_inv,
            'D_inv': self.D_inv,
            'K_inv': self.K_inv,
            'reference_position': self.reference_position,
            'reference_velocity': self.reference_velocity,
            'admittance_position': self.admittance_position,
            'admittance_velocity': self.admittance_velocity,
            'external_force': self.external_force
        }
    
    def _load_from_state(self, state: Dict[str, Any]):
        """Load Admittance Control specific state."""
        self.M_inv = state['M_inv']
        self.D_inv = state['D_inv']
        self.K_inv = state['K_inv']
        self.reference_position = state['reference_position']
        self.reference_velocity = state['reference_velocity']
        self.admittance_position = state['admittance_position']
        self.admittance_velocity = state['admittance_velocity']
        self.external_force = state['external_force']