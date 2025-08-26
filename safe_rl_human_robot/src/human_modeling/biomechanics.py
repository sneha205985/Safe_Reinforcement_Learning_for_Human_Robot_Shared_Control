"""
Biomechanical Human Model with Musculoskeletal Dynamics.

This module implements comprehensive biomechanical modeling including:
- Musculoskeletal dynamics: τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + F_muscle
- Hill-type muscle models with EMG integration
- Fatigue modeling: Force_max(t) = Force_max(0) * exp(-fatigue_rate * t)
- Individual calibration for different users

Mathematical Models:
- Joint dynamics with muscle forces
- Force-length and force-velocity relationships
- Activation dynamics and neural drive
- Fatigue accumulation and recovery
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import time
from collections import deque
import scipy.integrate
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MuscleType(Enum):
    """Types of muscle fibers."""
    SLOW_TWITCH = "slow_twitch"      # Type I - endurance
    FAST_TWITCH_A = "fast_twitch_a"  # Type IIa - power + endurance
    FAST_TWITCH_X = "fast_twitch_x"  # Type IIx - pure power


@dataclass
class MuscleParameters:
    """Parameters for Hill-type muscle model."""
    
    # Architectural parameters
    optimal_fiber_length: float = 0.1  # m
    pennation_angle: float = 0.0  # rad
    tendon_slack_length: float = 0.25  # m
    max_isometric_force: float = 1000.0  # N
    
    # Force-length relationship
    fl_width: float = 1.4  # Width of force-length curve
    fl_c1: float = 0.814  # Shape parameter
    fl_c2: float = 1.055  # Shape parameter
    fl_c3: float = 0.162  # Shape parameter
    fl_c4: float = 0.063  # Shape parameter
    
    # Force-velocity relationship  
    fv_max_velocity: float = 10.0  # optimal_fiber_length/s
    fv_a_f: float = 0.25  # Curvature parameter (concentric)
    fv_b_f: float = 0.25  # Curvature parameter (concentric)
    fv_a_s: float = 0.25  # Curvature parameter (eccentric)
    fv_b_s: float = 0.25  # Curvature parameter (eccentric)
    
    # Activation dynamics
    activation_time_constant: float = 0.010  # s (10ms)
    deactivation_time_constant: float = 0.040  # s (40ms)
    min_activation: float = 0.01  # Minimum activation level
    
    # Fiber type composition
    slow_twitch_ratio: float = 0.5  # Ratio of slow-twitch fibers
    fast_twitch_a_ratio: float = 0.35
    fast_twitch_x_ratio: float = 0.15
    
    # Fatigue parameters
    fatigue_time_constant: float = 30.0  # s
    recovery_time_constant: float = 60.0  # s
    fatigue_factor: float = 1.0  # Fatigue sensitivity


@dataclass
class JointParameters:
    """Parameters for joint dynamics."""
    
    # Inertial parameters
    mass: float = 1.0  # kg
    length: float = 0.3  # m
    center_of_mass: float = 0.15  # m from joint
    moment_of_inertia: float = 0.1  # kg⋅m²
    
    # Damping and stiffness
    joint_damping: float = 0.1  # N⋅m⋅s/rad
    joint_stiffness: float = 0.0  # N⋅m/rad (passive)
    
    # Range of motion
    min_angle: float = -np.pi/2  # rad
    max_angle: float = np.pi/2   # rad
    
    # Moment arms (muscle attachment points)
    moment_arms: List[float] = field(default_factory=lambda: [0.05, -0.05])  # m


class MuscleModel:
    """Hill-type muscle model with activation dynamics and fatigue."""
    
    def __init__(self, name: str, parameters: MuscleParameters):
        self.name = name
        self.params = parameters
        
        # State variables
        self.activation = 0.0
        self.fiber_length = parameters.optimal_fiber_length
        self.fiber_velocity = 0.0
        self.neural_excitation = 0.0
        
        # Fatigue state
        self.fatigue_level = 0.0
        self.recovery_level = 1.0
        
        # History for dynamics
        self.activation_history = deque(maxlen=100)
        self.force_history = deque(maxlen=100)
        
        logger.debug(f"Muscle model created: {name}")
    
    def update_activation(self, neural_drive: float, dt: float) -> float:
        """
        Update muscle activation based on neural drive.
        
        Args:
            neural_drive: Neural excitation signal (0-1)
            dt: Time step (s)
            
        Returns:
            Updated activation level (0-1)
        """
        # Clamp neural drive
        neural_drive = np.clip(neural_drive, 0.0, 1.0)
        self.neural_excitation = neural_drive
        
        # Activation dynamics (first-order)
        if neural_drive > self.activation:
            # Activation
            tau = self.params.activation_time_constant
        else:
            # Deactivation
            tau = self.params.deactivation_time_constant
        
        # Exponential approach to neural drive
        activation_dot = (neural_drive - self.activation) / tau
        self.activation += activation_dot * dt
        
        # Apply minimum activation and bounds
        self.activation = np.clip(self.activation, 
                                self.params.min_activation, 1.0)
        
        self.activation_history.append(self.activation)
        return self.activation
    
    def compute_force_length_relationship(self, normalized_length: float) -> float:
        """
        Compute force-length relationship for muscle fiber.
        
        Args:
            normalized_length: Fiber length / optimal fiber length
            
        Returns:
            Force scaling factor (0-1)
        """
        # Gaussian-like force-length relationship
        b11 = self.params.fl_c1
        b21 = self.params.fl_c2  
        b31 = self.params.fl_c3
        b41 = self.params.fl_c4
        
        exp1 = -((normalized_length - b11) / (b21 * self.params.fl_width))**2
        exp2 = -((normalized_length - b31) / (b41 * self.params.fl_width))**2
        
        fl = np.exp(exp1) + np.exp(exp2)
        return np.clip(fl, 0.0, 1.0)
    
    def compute_force_velocity_relationship(self, normalized_velocity: float) -> float:
        """
        Compute force-velocity relationship for muscle fiber.
        
        Args:
            normalized_velocity: Fiber velocity / max_velocity (positive = concentric)
            
        Returns:
            Force scaling factor (0-1.8 for eccentric)
        """
        if normalized_velocity <= 0:
            # Concentric (shortening)
            a_f = self.params.fv_a_f
            b_f = self.params.fv_b_f
            fv = (1 - normalized_velocity) / (1 + normalized_velocity * b_f / a_f)
        else:
            # Eccentric (lengthening) 
            a_s = self.params.fv_a_s
            b_s = self.params.fv_b_s
            fv = (1 + normalized_velocity * a_s) / (1 - normalized_velocity * b_s)
            fv = np.clip(fv, 0.0, 1.8)  # Limit eccentric enhancement
        
        return np.clip(fv, 0.0, 1.8)
    
    def compute_muscle_force(self, muscle_length: float, 
                           muscle_velocity: float) -> float:
        """
        Compute total muscle force using Hill model.
        
        Args:
            muscle_length: Total muscle-tendon unit length (m)
            muscle_velocity: Muscle-tendon unit velocity (m/s)
            
        Returns:
            Muscle force (N)
        """
        # Compute fiber length and velocity
        cos_alpha = np.cos(self.params.pennation_angle)
        
        # Tendon length (assuming rigid tendon for simplicity)
        tendon_length = self.params.tendon_slack_length
        fiber_length = (muscle_length - tendon_length) / cos_alpha
        fiber_velocity = muscle_velocity / cos_alpha
        
        # Update fiber state
        self.fiber_length = max(0.1 * self.params.optimal_fiber_length, fiber_length)
        self.fiber_velocity = fiber_velocity
        
        # Normalized quantities
        norm_length = self.fiber_length / self.params.optimal_fiber_length
        norm_velocity = self.fiber_velocity / self.params.fv_max_velocity
        
        # Force components
        fl = self.compute_force_length_relationship(norm_length)
        fv = self.compute_force_velocity_relationship(norm_velocity)
        
        # Active force
        active_force = (self.activation * fl * fv * 
                       self.params.max_isometric_force * self.recovery_level)
        
        # Passive force (exponential spring)
        if norm_length > 1.0:
            passive_force = self.params.max_isometric_force * 0.1 * (
                np.exp(3.0 * (norm_length - 1.0)) - 1.0
            )
        else:
            passive_force = 0.0
        
        total_force = active_force + passive_force
        
        # Project force along line of action
        total_force *= cos_alpha
        
        self.force_history.append(total_force)
        return total_force
    
    def update_fatigue(self, dt: float):
        """
        Update muscle fatigue based on activation history.
        
        Args:
            dt: Time step (s)
        """
        # Fatigue accumulation based on activation
        fatigue_rate = self.activation**2 / self.params.fatigue_time_constant
        self.fatigue_level += fatigue_rate * dt * self.params.fatigue_factor
        
        # Recovery during low activation
        if self.activation < 0.1:
            recovery_rate = (1.0 - self.fatigue_level) / self.params.recovery_time_constant
            self.fatigue_level -= recovery_rate * dt
        
        self.fatigue_level = np.clip(self.fatigue_level, 0.0, 1.0)
        
        # Update recovery level (available force capacity)
        self.recovery_level = np.exp(-2.0 * self.fatigue_level)
    
    def get_muscle_state(self) -> Dict[str, float]:
        """Get comprehensive muscle state."""
        return {
            'activation': self.activation,
            'neural_excitation': self.neural_excitation,
            'fiber_length': self.fiber_length,
            'fiber_velocity': self.fiber_velocity,
            'fatigue_level': self.fatigue_level,
            'recovery_level': self.recovery_level,
            'max_force_available': self.params.max_isometric_force * self.recovery_level
        }


class FatigueModel:
    """Advanced fatigue model with different fiber type characteristics."""
    
    def __init__(self, muscle_params: MuscleParameters):
        self.params = muscle_params
        
        # Fiber type specific fatigue
        self.slow_twitch_fatigue = 0.0
        self.fast_twitch_a_fatigue = 0.0  
        self.fast_twitch_x_fatigue = 0.0
        
        # Different fatigue rates for fiber types
        self.slow_twitch_rate = 1.0    # Slow to fatigue
        self.fast_twitch_a_rate = 3.0  # Moderate fatigue
        self.fast_twitch_x_rate = 8.0  # Fast fatigue
        
        # Recovery rates
        self.slow_twitch_recovery = 1.0
        self.fast_twitch_a_recovery = 0.7
        self.fast_twitch_x_recovery = 0.5
    
    def update_fiber_fatigue(self, activation: float, dt: float):
        """Update fatigue for different fiber types."""
        # Recruitment order: slow -> fast IIa -> fast IIx
        slow_activation = min(activation, 1.0)
        fast_a_activation = max(0.0, min(activation - 0.3, 0.7)) / 0.7
        fast_x_activation = max(0.0, activation - 0.8) / 0.2
        
        # Fatigue accumulation
        self.slow_twitch_fatigue += (slow_activation**2 * self.slow_twitch_rate * 
                                   self.params.fatigue_factor * dt)
        self.fast_twitch_a_fatigue += (fast_a_activation**2 * self.fast_twitch_a_rate * 
                                     self.params.fatigue_factor * dt)
        self.fast_twitch_x_fatigue += (fast_x_activation**2 * self.fast_twitch_x_rate * 
                                     self.params.fatigue_factor * dt)
        
        # Recovery during rest
        if activation < 0.1:
            recovery_dt = dt / self.params.recovery_time_constant
            self.slow_twitch_fatigue -= self.slow_twitch_recovery * recovery_dt
            self.fast_twitch_a_fatigue -= self.fast_twitch_a_recovery * recovery_dt
            self.fast_twitch_x_fatigue -= self.fast_twitch_x_recovery * recovery_dt
        
        # Clamp fatigue levels
        self.slow_twitch_fatigue = np.clip(self.slow_twitch_fatigue, 0.0, 1.0)
        self.fast_twitch_a_fatigue = np.clip(self.fast_twitch_a_fatigue, 0.0, 1.0)
        self.fast_twitch_x_fatigue = np.clip(self.fast_twitch_x_fatigue, 0.0, 1.0)
    
    def get_available_force_capacity(self) -> float:
        """Get total available force capacity considering fiber fatigue."""
        # Weighted average based on fiber type composition
        slow_capacity = (1.0 - self.slow_twitch_fatigue) * self.params.slow_twitch_ratio
        fast_a_capacity = (1.0 - self.fast_twitch_a_fatigue) * self.params.fast_twitch_a_ratio
        fast_x_capacity = (1.0 - self.fast_twitch_x_fatigue) * self.params.fast_twitch_x_ratio
        
        return slow_capacity + fast_a_capacity + fast_x_capacity
    
    def get_fatigue_state(self) -> Dict[str, float]:
        """Get detailed fatigue state."""
        return {
            'slow_twitch_fatigue': self.slow_twitch_fatigue,
            'fast_twitch_a_fatigue': self.fast_twitch_a_fatigue,
            'fast_twitch_x_fatigue': self.fast_twitch_x_fatigue,
            'total_capacity': self.get_available_force_capacity(),
            'endurance_capacity': 1.0 - self.slow_twitch_fatigue,
            'power_capacity': 1.0 - self.fast_twitch_x_fatigue
        }


class MusculoskeletalDynamics:
    """Complete musculoskeletal dynamics model for multi-joint system."""
    
    def __init__(self, joint_params: List[JointParameters], 
                 muscle_params: Dict[str, MuscleParameters]):
        """
        Initialize musculoskeletal dynamics.
        
        Args:
            joint_params: List of joint parameters for each DOF
            muscle_params: Dictionary mapping muscle names to parameters
        """
        self.joint_params = joint_params
        self.n_joints = len(joint_params)
        
        # Create muscle models
        self.muscles = {}
        self.muscle_names = list(muscle_params.keys())
        
        for name, params in muscle_params.items():
            self.muscles[name] = MuscleModel(name, params)
        
        # Create fatigue models
        self.fatigue_models = {
            name: FatigueModel(params) 
            for name, params in muscle_params.items()
        }
        
        # Joint state
        self.joint_angles = np.zeros(self.n_joints)
        self.joint_velocities = np.zeros(self.n_joints)
        self.joint_accelerations = np.zeros(self.n_joints)
        
        # Muscle-joint mapping (moment arms)
        self._setup_muscle_moment_arms()
        
        # EMG integration
        self.emg_to_activation_gain = 1.0
        self.emg_delay = 0.030  # 30ms EMG-force delay
        self.emg_buffer = {name: deque(maxlen=10) for name in self.muscle_names}
        
        logger.info(f"Musculoskeletal model initialized: {self.n_joints} joints, "
                   f"{len(self.muscles)} muscles")
    
    def _setup_muscle_moment_arms(self):
        """Setup muscle moment arm matrix."""
        n_muscles = len(self.muscles)
        self.moment_arms = np.zeros((self.n_joints, n_muscles))
        
        # Simplified moment arm assignment (would be more complex in reality)
        for i, muscle_name in enumerate(self.muscle_names):
            # Assign moment arms based on muscle name patterns
            if 'shoulder' in muscle_name.lower():
                if 'flexor' in muscle_name:
                    self.moment_arms[0, i] = 0.05  # Shoulder flexion
                elif 'extensor' in muscle_name:
                    self.moment_arms[0, i] = -0.05  # Shoulder extension
                    
            elif 'elbow' in muscle_name.lower():
                if 'flexor' in muscle_name:
                    self.moment_arms[1, i] = 0.04  # Elbow flexion
                elif 'extensor' in muscle_name:
                    self.moment_arms[1, i] = -0.03  # Elbow extension
                    
            # Add more joints as needed
    
    def integrate_emg_signals(self, emg_signals: Dict[str, float], dt: float):
        """
        Integrate EMG signals to muscle activation with delay.
        
        Args:
            emg_signals: Dictionary of EMG signals (0-1) for each muscle
            dt: Time step (s)
        """
        for muscle_name in self.muscle_names:
            if muscle_name in emg_signals:
                # Add to EMG buffer for delay
                self.emg_buffer[muscle_name].append(emg_signals[muscle_name])
                
                # Get delayed EMG signal
                delay_samples = max(1, int(self.emg_delay / dt))
                if len(self.emg_buffer[muscle_name]) >= delay_samples:
                    delayed_emg = self.emg_buffer[muscle_name][-delay_samples]
                else:
                    delayed_emg = emg_signals[muscle_name]
                
                # Convert EMG to neural drive
                neural_drive = delayed_emg * self.emg_to_activation_gain
                
                # Update muscle activation
                self.muscles[muscle_name].update_activation(neural_drive, dt)
    
    def compute_muscle_forces(self, muscle_lengths: Dict[str, float],
                            muscle_velocities: Dict[str, float]) -> Dict[str, float]:
        """
        Compute forces for all muscles.
        
        Args:
            muscle_lengths: Current length for each muscle (m)
            muscle_velocities: Current velocity for each muscle (m/s)
            
        Returns:
            Dictionary of muscle forces (N)
        """
        muscle_forces = {}
        
        for muscle_name in self.muscle_names:
            if muscle_name in muscle_lengths:
                force = self.muscles[muscle_name].compute_muscle_force(
                    muscle_lengths[muscle_name],
                    muscle_velocities[muscle_name]
                )
                muscle_forces[muscle_name] = force
            else:
                muscle_forces[muscle_name] = 0.0
        
        return muscle_forces
    
    def compute_joint_torques(self, muscle_forces: Dict[str, float]) -> np.ndarray:
        """
        Convert muscle forces to joint torques using moment arms.
        
        Args:
            muscle_forces: Dictionary of muscle forces (N)
            
        Returns:
            Joint torques array (N⋅m)
        """
        force_vector = np.array([
            muscle_forces.get(name, 0.0) for name in self.muscle_names
        ])
        
        # Apply moment arm transformation
        joint_torques = self.moment_arms @ force_vector
        
        return joint_torques
    
    def compute_joint_dynamics(self, joint_torques: np.ndarray,
                             external_torques: np.ndarray = None) -> np.ndarray:
        """
        Compute joint accelerations using equation of motion.
        τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
        
        Args:
            joint_torques: Muscle-generated joint torques (N⋅m)
            external_torques: External torques (N⋅m)
            
        Returns:
            Joint accelerations (rad/s²)
        """
        if external_torques is None:
            external_torques = np.zeros(self.n_joints)
        
        # Compute inertia matrix M(q) - simplified
        M = self._compute_inertia_matrix()
        
        # Compute Coriolis/centripetal matrix C(q,q̇)q̇
        C_qd = self._compute_coriolis_terms()
        
        # Compute gravitational torques G(q)
        G = self._compute_gravity_terms()
        
        # Compute joint damping
        D = self._compute_damping_terms()
        
        # Total applied torques
        total_torques = joint_torques + external_torques
        
        # Solve for accelerations: q̈ = M⁻¹(τ - C(q,q̇)q̇ - G(q) - D⋅q̇)
        try:
            accelerations = np.linalg.solve(
                M, total_torques - C_qd - G - D
            )
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            accelerations = np.zeros(self.n_joints)
            logger.warning("Singular inertia matrix, using zero acceleration")
        
        self.joint_accelerations = accelerations
        return accelerations
    
    def _compute_inertia_matrix(self) -> np.ndarray:
        """Compute joint space inertia matrix."""
        M = np.zeros((self.n_joints, self.n_joints))
        
        # Diagonal approximation (independent joints)
        for i in range(self.n_joints):
            M[i, i] = self.joint_params[i].moment_of_inertia
        
        return M
    
    def _compute_coriolis_terms(self) -> np.ndarray:
        """Compute Coriolis and centripetal terms."""
        # Simplified - would be more complex for coupled joints
        C_qd = np.zeros(self.n_joints)
        
        # Quadratic velocity terms (simplified)
        for i in range(self.n_joints):
            C_qd[i] = 0.1 * self.joint_velocities[i]**2 * np.sign(self.joint_velocities[i])
        
        return C_qd
    
    def _compute_gravity_terms(self) -> np.ndarray:
        """Compute gravitational torques."""
        G = np.zeros(self.n_joints)
        g = 9.81  # m/s²
        
        for i in range(self.n_joints):
            # Simplified gravity model
            mass = self.joint_params[i].mass
            length = self.joint_params[i].center_of_mass
            G[i] = mass * g * length * np.sin(self.joint_angles[i])
        
        return G
    
    def _compute_damping_terms(self) -> np.ndarray:
        """Compute joint damping terms."""
        D = np.zeros(self.n_joints)
        
        for i in range(self.n_joints):
            D[i] = (self.joint_params[i].joint_damping * self.joint_velocities[i])
        
        return D
    
    def update_dynamics(self, dt: float, emg_signals: Dict[str, float] = None,
                       muscle_lengths: Dict[str, float] = None,
                       muscle_velocities: Dict[str, float] = None,
                       external_torques: np.ndarray = None):
        """
        Update complete musculoskeletal dynamics.
        
        Args:
            dt: Time step (s)
            emg_signals: EMG signals for muscles
            muscle_lengths: Current muscle lengths
            muscle_velocities: Current muscle velocities
            external_torques: External applied torques
        """
        # Update EMG to activation
        if emg_signals:
            self.integrate_emg_signals(emg_signals, dt)
        
        # Compute muscle forces
        if muscle_lengths and muscle_velocities:
            muscle_forces = self.compute_muscle_forces(muscle_lengths, muscle_velocities)
        else:
            muscle_forces = {name: 0.0 for name in self.muscle_names}
        
        # Convert to joint torques
        joint_torques = self.compute_joint_torques(muscle_forces)
        
        # Compute joint accelerations
        accelerations = self.compute_joint_dynamics(joint_torques, external_torques)
        
        # Integrate kinematics
        self.joint_velocities += accelerations * dt
        self.joint_angles += self.joint_velocities * dt
        
        # Apply joint limits
        for i in range(self.n_joints):
            if self.joint_angles[i] < self.joint_params[i].min_angle:
                self.joint_angles[i] = self.joint_params[i].min_angle
                self.joint_velocities[i] = max(0, self.joint_velocities[i])
            elif self.joint_angles[i] > self.joint_params[i].max_angle:
                self.joint_angles[i] = self.joint_params[i].max_angle
                self.joint_velocities[i] = min(0, self.joint_velocities[i])
        
        # Update muscle fatigue
        for muscle_name, muscle in self.muscles.items():
            muscle.update_fatigue(dt)
            self.fatigue_models[muscle_name].update_fiber_fatigue(
                muscle.activation, dt
            )
    
    def get_biomechanical_state(self) -> Dict[str, Any]:
        """Get comprehensive biomechanical state."""
        muscle_states = {}
        fatigue_states = {}
        
        for name in self.muscle_names:
            muscle_states[name] = self.muscles[name].get_muscle_state()
            fatigue_states[name] = self.fatigue_models[name].get_fatigue_state()
        
        return {
            'joint_kinematics': {
                'angles': self.joint_angles.copy(),
                'velocities': self.joint_velocities.copy(),
                'accelerations': self.joint_accelerations.copy()
            },
            'muscle_states': muscle_states,
            'fatigue_states': fatigue_states,
            'total_fatigue': np.mean([
                fm.get_available_force_capacity() 
                for fm in self.fatigue_models.values()
            ]),
            'moment_arms': self.moment_arms.copy()
        }


class BiomechanicalModel:
    """
    Complete biomechanical model integrating all components.
    
    Provides high-level interface for biomechanical simulation with
    individual calibration and adaptation capabilities.
    """
    
    def __init__(self, 
                 subject_characteristics: Dict[str, Any] = None,
                 calibration_data: Dict[str, Any] = None):
        """
        Initialize biomechanical model.
        
        Args:
            subject_characteristics: Individual subject characteristics
            calibration_data: Calibration data for personalization
        """
        self.subject_characteristics = subject_characteristics or {}
        self.calibration_data = calibration_data or {}
        
        # Initialize default parameters
        self._initialize_default_parameters()
        
        # Apply individual calibration
        if calibration_data:
            self._apply_calibration(calibration_data)
        
        # Create musculoskeletal system
        self.musculoskeletal = MusculoskeletalDynamics(
            self.joint_parameters, self.muscle_parameters
        )
        
        # State tracking
        self.simulation_time = 0.0
        self.state_history = deque(maxlen=1000)
        
        # EMG processing
        self.emg_processor = EMGProcessor()
        
        logger.info("Biomechanical model initialized with individual calibration")
    
    def _initialize_default_parameters(self):
        """Initialize default biomechanical parameters."""
        # Default joint parameters for 7-DOF arm
        self.joint_parameters = [
            JointParameters(  # Shoulder flexion/extension
                mass=2.0, length=0.3, center_of_mass=0.15,
                moment_of_inertia=0.05, min_angle=-np.pi/2, max_angle=np.pi/2
            ),
            JointParameters(  # Shoulder abduction/adduction
                mass=1.5, length=0.25, center_of_mass=0.12,
                moment_of_inertia=0.03, min_angle=-np.pi/4, max_angle=np.pi
            ),
            JointParameters(  # Shoulder rotation
                mass=1.0, length=0.2, center_of_mass=0.1,
                moment_of_inertia=0.02, min_angle=-np.pi/2, max_angle=np.pi/2
            ),
            JointParameters(  # Elbow flexion/extension
                mass=1.2, length=0.25, center_of_mass=0.12,
                moment_of_inertia=0.025, min_angle=0, max_angle=np.pi*0.8
            ),
            JointParameters(  # Forearm pronation/supination
                mass=0.8, length=0.2, center_of_mass=0.1,
                moment_of_inertia=0.015, min_angle=-np.pi/2, max_angle=np.pi/2
            ),
            JointParameters(  # Wrist flexion/extension
                mass=0.5, length=0.15, center_of_mass=0.075,
                moment_of_inertia=0.008, min_angle=-np.pi/3, max_angle=np.pi/3
            ),
            JointParameters(  # Wrist abduction/adduction
                mass=0.3, length=0.1, center_of_mass=0.05,
                moment_of_inertia=0.005, min_angle=-np.pi/4, max_angle=np.pi/4
            )
        ]
        
        # Default muscle parameters
        self.muscle_parameters = {
            'shoulder_flexor': MuscleParameters(
                max_isometric_force=500, optimal_fiber_length=0.12,
                pennation_angle=np.pi/12, tendon_slack_length=0.15
            ),
            'shoulder_extensor': MuscleParameters(
                max_isometric_force=600, optimal_fiber_length=0.15,
                pennation_angle=np.pi/10, tendon_slack_length=0.18
            ),
            'shoulder_abductor': MuscleParameters(
                max_isometric_force=400, optimal_fiber_length=0.10,
                pennation_angle=np.pi/8, tendon_slack_length=0.12
            ),
            'elbow_flexor': MuscleParameters(
                max_isometric_force=800, optimal_fiber_length=0.08,
                pennation_angle=np.pi/15, tendon_slack_length=0.20
            ),
            'elbow_extensor': MuscleParameters(
                max_isometric_force=900, optimal_fiber_length=0.10,
                pennation_angle=np.pi/12, tendon_slack_length=0.25
            ),
            'wrist_flexor': MuscleParameters(
                max_isometric_force=200, optimal_fiber_length=0.06,
                pennation_angle=0, tendon_slack_length=0.15
            ),
            'wrist_extensor': MuscleParameters(
                max_isometric_force=150, optimal_fiber_length=0.05,
                pennation_angle=0, tendon_slack_length=0.12
            )
        }
    
    def _apply_calibration(self, calibration_data: Dict[str, Any]):
        """Apply individual calibration to model parameters."""
        
        # Anthropometric scaling
        if 'height' in calibration_data and 'mass' in calibration_data:
            height_scale = calibration_data['height'] / 1.75  # Reference height
            mass_scale = calibration_data['mass'] / 70.0      # Reference mass
            
            for joint_params in self.joint_parameters:
                joint_params.length *= height_scale
                joint_params.mass *= mass_scale
                joint_params.moment_of_inertia *= mass_scale * height_scale**2
        
        # Strength scaling
        if 'max_strength' in calibration_data:
            strength_scale = calibration_data['max_strength'] / 500.0  # Reference
            for muscle_params in self.muscle_parameters.values():
                muscle_params.max_isometric_force *= strength_scale
        
        # Age-related modifications
        if 'age' in calibration_data:
            age = calibration_data['age']
            if age > 65:
                # Reduce strength and increase fatigue for elderly
                age_factor = max(0.6, 1.0 - (age - 65) * 0.01)
                for muscle_params in self.muscle_parameters.values():
                    muscle_params.max_isometric_force *= age_factor
                    muscle_params.fatigue_factor *= (2.0 - age_factor)
        
        # Pathology-specific modifications
        if 'pathology' in calibration_data:
            self._apply_pathology_modifications(calibration_data['pathology'])
    
    def _apply_pathology_modifications(self, pathology_data: Dict[str, Any]):
        """Apply pathology-specific modifications."""
        pathology_type = pathology_data.get('type', 'none')
        severity = pathology_data.get('severity', 0.0)  # 0-1
        
        if pathology_type == 'stroke':
            affected_side = pathology_data.get('affected_side', 'left')
            
            # Reduce strength on affected side
            if affected_side == 'left':
                affected_muscles = ['shoulder_flexor', 'elbow_flexor', 'wrist_flexor']
            else:
                affected_muscles = ['shoulder_extensor', 'elbow_extensor', 'wrist_extensor']
            
            for muscle_name in affected_muscles:
                if muscle_name in self.muscle_parameters:
                    reduction_factor = 1.0 - 0.7 * severity
                    self.muscle_parameters[muscle_name].max_isometric_force *= reduction_factor
                    self.muscle_parameters[muscle_name].fatigue_factor *= (1.0 + severity)
        
        elif pathology_type == 'spinal_injury':
            level = pathology_data.get('level', 'C6')  # Cervical level
            
            # Progressive strength loss based on injury level
            if level in ['C5', 'C6']:
                # Partial hand/wrist function loss
                hand_muscles = ['wrist_flexor', 'wrist_extensor']
                for muscle_name in hand_muscles:
                    if muscle_name in self.muscle_parameters:
                        self.muscle_parameters[muscle_name].max_isometric_force *= (1.0 - severity)
    
    def process_emg_input(self, raw_emg: Dict[str, np.ndarray], 
                         dt: float) -> Dict[str, float]:
        """
        Process raw EMG signals to activation levels.
        
        Args:
            raw_emg: Dictionary of raw EMG signals
            dt: Time step
            
        Returns:
            Processed activation signals
        """
        return self.emg_processor.process_emg_signals(raw_emg, dt)
    
    def update(self, dt: float, 
               emg_signals: Dict[str, float] = None,
               external_forces: Dict[str, np.ndarray] = None,
               neural_drive: Dict[str, float] = None):
        """
        Update complete biomechanical model.
        
        Args:
            dt: Time step
            emg_signals: EMG activation signals
            external_forces: External forces applied to segments
            neural_drive: Direct neural drive signals (alternative to EMG)
        """
        # Use neural drive if EMG not available
        if emg_signals is None and neural_drive is not None:
            emg_signals = neural_drive
        
        # Estimate muscle lengths and velocities (simplified)
        muscle_lengths, muscle_velocities = self._estimate_muscle_kinematics()
        
        # Convert external forces to joint torques
        external_torques = self._convert_forces_to_torques(external_forces)
        
        # Update musculoskeletal dynamics
        self.musculoskeletal.update_dynamics(
            dt, emg_signals, muscle_lengths, muscle_velocities, external_torques
        )
        
        # Update simulation time
        self.simulation_time += dt
        
        # Store state history
        state = self.get_state()
        self.state_history.append((self.simulation_time, state))
    
    def _estimate_muscle_kinematics(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Estimate muscle lengths and velocities from joint kinematics."""
        muscle_lengths = {}
        muscle_velocities = {}
        
        # Simplified muscle length estimation
        joint_angles = self.musculoskeletal.joint_angles
        joint_velocities = self.musculoskeletal.joint_velocities
        
        for muscle_name in self.muscle_parameters.keys():
            # Map muscle to primary joint (simplified)
            if 'shoulder' in muscle_name:
                joint_idx = 0 if 'flexor' in muscle_name else 1
            elif 'elbow' in muscle_name:
                joint_idx = 3
            elif 'wrist' in muscle_name:
                joint_idx = 5
            else:
                joint_idx = 0
            
            # Simple length-angle relationship
            optimal_length = self.muscle_parameters[muscle_name].optimal_fiber_length
            tendon_length = self.muscle_parameters[muscle_name].tendon_slack_length
            
            # Simplified: muscle length varies sinusoidally with joint angle
            angle = joint_angles[joint_idx]
            length_variation = 0.02 * np.sin(angle)  # ±2cm variation
            
            muscle_lengths[muscle_name] = optimal_length + tendon_length + length_variation
            muscle_velocities[muscle_name] = 0.02 * np.cos(angle) * joint_velocities[joint_idx]
        
        return muscle_lengths, muscle_velocities
    
    def _convert_forces_to_torques(self, external_forces: Dict[str, np.ndarray] = None) -> np.ndarray:
        """Convert external forces to joint torques."""
        if external_forces is None:
            return np.zeros(self.musculoskeletal.n_joints)
        
        # Simplified force to torque conversion
        external_torques = np.zeros(self.musculoskeletal.n_joints)
        
        # This would involve more complex kinematics in reality
        for segment_name, force in external_forces.items():
            if 'hand' in segment_name.lower():
                # Force at end-effector
                lever_arm = 0.3  # Approximate lever arm
                external_torques[-1] = force[0] * lever_arm  # Simplified
        
        return external_torques
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete biomechanical state."""
        base_state = self.musculoskeletal.get_biomechanical_state()
        
        base_state.update({
            'simulation_time': self.simulation_time,
            'subject_characteristics': self.subject_characteristics,
            'calibrated': bool(self.calibration_data)
        })
        
        return base_state
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self.musculoskeletal.joint_angles.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities."""
        return self.musculoskeletal.joint_velocities.copy()
    
    def get_muscle_activations(self) -> Dict[str, float]:
        """Get current muscle activation levels."""
        return {
            name: muscle.activation 
            for name, muscle in self.musculoskeletal.muscles.items()
        }
    
    def get_fatigue_levels(self) -> Dict[str, float]:
        """Get current fatigue levels for all muscles."""
        return {
            name: muscle.fatigue_level
            for name, muscle in self.musculoskeletal.muscles.items()
        }


class EMGProcessor:
    """EMG signal processing for muscle activation estimation."""
    
    def __init__(self):
        # Filter parameters
        self.high_pass_freq = 20.0  # Hz
        self.low_pass_freq = 500.0  # Hz
        self.notch_freq = 60.0      # Hz (power line)
        self.envelope_freq = 6.0    # Hz
        
        # Processing history
        self.signal_history = {}
        self.envelope_history = {}
    
    def process_emg_signals(self, raw_emg: Dict[str, np.ndarray], 
                           dt: float) -> Dict[str, float]:
        """
        Process raw EMG signals to activation levels.
        
        Args:
            raw_emg: Dictionary of raw EMG signals
            dt: Time step
            
        Returns:
            Processed activation levels (0-1)
        """
        processed_signals = {}
        
        for muscle_name, signal in raw_emg.items():
            # Apply signal processing pipeline
            filtered_signal = self._apply_filters(signal, dt)
            rectified_signal = np.abs(filtered_signal)
            envelope = self._compute_envelope(rectified_signal, dt)
            activation = self._normalize_activation(envelope, muscle_name)
            
            processed_signals[muscle_name] = activation
        
        return processed_signals
    
    def _apply_filters(self, signal: np.ndarray, dt: float) -> np.ndarray:
        """Apply filtering to raw EMG signal."""
        # Simplified filtering (would use proper digital filters)
        
        # High-pass filter (remove DC and low-freq artifacts)
        # Low-pass filter (anti-aliasing)
        # Notch filter (remove power line interference)
        
        # For now, just return signal with some noise reduction
        return signal * 0.95  # Simple scaling
    
    def _compute_envelope(self, rectified_signal: np.ndarray, dt: float) -> float:
        """Compute signal envelope."""
        # Simple moving average envelope
        return np.mean(rectified_signal)
    
    def _normalize_activation(self, envelope: float, muscle_name: str) -> float:
        """Normalize envelope to activation level."""
        # Simple normalization (would use MVC normalization in practice)
        max_envelope = 1000.0  # Assumed max envelope value
        
        activation = envelope / max_envelope
        return np.clip(activation, 0.0, 1.0)


# Example usage and testing
if __name__ == "__main__":
    """Example of biomechanical model usage."""
    
    # Create subject characteristics
    subject_chars = {
        'height': 1.75,  # m
        'mass': 70.0,    # kg
        'age': 35,
        'gender': 'male'
    }
    
    # Create calibration data
    calibration = {
        'max_strength': 600.0,  # N
        'pathology': {
            'type': 'none',
            'severity': 0.0
        }
    }
    
    # Initialize model
    bio_model = BiomechanicalModel(subject_chars, calibration)
    
    print(f"Biomechanical model created for {subject_chars['gender']} "
          f"subject (age {subject_chars['age']})")
    
    # Simulation parameters
    dt = 0.001  # 1ms time step
    simulation_duration = 2.0  # 2 seconds
    steps = int(simulation_duration / dt)
    
    # Simple EMG pattern (sinusoidal activation)
    for step in range(steps):
        t = step * dt
        
        # Generate test EMG signals
        emg_signals = {
            'shoulder_flexor': 0.3 + 0.2 * np.sin(2 * np.pi * 0.5 * t),
            'elbow_flexor': 0.2 + 0.3 * np.sin(2 * np.pi * 0.3 * t),
            'wrist_flexor': 0.1 + 0.1 * np.sin(2 * np.pi * 1.0 * t)
        }
        
        # Update model
        bio_model.update(dt, emg_signals=emg_signals)
        
        # Print state every 100ms
        if step % 100 == 0:
            state = bio_model.get_state()
            total_fatigue = state['total_fatigue']
            joint_angles = bio_model.get_joint_positions()
            
            print(f"Time: {t:.2f}s, Total fatigue: {1-total_fatigue:.3f}, "
                  f"Shoulder angle: {joint_angles[0]:.3f} rad")
    
    print("Biomechanical simulation completed successfully!")