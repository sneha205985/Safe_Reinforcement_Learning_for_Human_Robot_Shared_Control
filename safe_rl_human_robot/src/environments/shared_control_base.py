"""
Base classes for shared control environments with physics simulation and safety constraints.

This module provides the foundational components for realistic human-robot
shared control environments with proper physics, safety monitoring, and
human behavior modeling.
"""

import numpy as np
import torch
import gym
from gym import spaces
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

# Re-export from existing module
from .human_robot_env import (
    AssistanceMode, HumanInput, RobotState, EnvironmentState, 
    HumanModel, SimpleHumanModel
)


@dataclass
class SafetyConstraint:
    """Definition of a safety constraint for shared control."""
    name: str
    constraint_function: callable  # Function that returns constraint value (>0 = safe, ≤0 = violation)
    gradient_function: Optional[callable] = None  # Gradient of constraint
    threshold: float = 0.0  # Safety threshold
    penalty_weight: float = 1.0  # Weight for constraint violation penalty
    active: bool = True
    
    def evaluate(self, state: EnvironmentState, action: torch.Tensor) -> Dict[str, float]:
        """Evaluate constraint and return detailed information."""
        constraint_value = self.constraint_function(state, action)
        violation = constraint_value <= self.threshold
        penalty = max(0, self.threshold - constraint_value) * self.penalty_weight
        
        return {
            'value': constraint_value,
            'threshold': self.threshold, 
            'violation': violation,
            'penalty': penalty,
            'margin': constraint_value - self.threshold
        }


class PhysicsEngine(ABC):
    """Abstract interface for physics engines."""
    
    @abstractmethod
    def step(self, action: torch.Tensor, dt: float) -> Dict[str, Any]:
        """Step physics simulation forward by dt."""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current physics state."""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set physics state."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset physics to initial state.""" 
        pass


class SimplePhysicsEngine(PhysicsEngine):
    """Simple physics engine for basic dynamics without external dependencies."""
    
    def __init__(self, robot_dof: int, mass_matrix: Optional[torch.Tensor] = None,
                 damping: float = 0.1, joint_limits: Optional[torch.Tensor] = None):
        """
        Initialize simple physics engine.
        
        Args:
            robot_dof: Robot degrees of freedom
            mass_matrix: Mass matrix [dof, dof], identity if None
            damping: Damping coefficient
            joint_limits: Joint limits [dof, 2] (min, max), None for unlimited
        """
        self.robot_dof = robot_dof
        self.mass_matrix = mass_matrix if mass_matrix is not None else torch.eye(robot_dof)
        self.damping = damping
        self.joint_limits = joint_limits
        
        # State variables
        self.q = torch.zeros(robot_dof)  # positions
        self.qd = torch.zeros(robot_dof)  # velocities
        self.tau = torch.zeros(robot_dof)  # torques
        
        # Physics constants
        self.gravity_vector = torch.zeros(robot_dof)  # No gravity by default
        
    def step(self, action: torch.Tensor, dt: float) -> Dict[str, Any]:
        """
        Step dynamics forward using Euler integration.
        
        Dynamics: M(q)q̈ + D*q̇ + G(q) = τ
        """
        # Update torques from action
        self.tau = action
        
        # Compute accelerations
        damping_torque = self.damping * self.qd
        net_torque = self.tau - damping_torque - self.gravity_vector
        
        # Solve M * qdd = net_torque
        try:
            qdd = torch.linalg.solve(self.mass_matrix, net_torque)
        except torch.linalg.LinAlgError:
            # Fallback to pseudo-inverse if mass matrix is singular
            qdd = torch.linalg.pinv(self.mass_matrix) @ net_torque
        
        # Integrate
        self.qd += qdd * dt
        self.q += self.qd * dt
        
        # Apply joint limits
        if self.joint_limits is not None:
            # Clamp positions
            self.q = torch.clamp(self.q, self.joint_limits[:, 0], self.joint_limits[:, 1])
            
            # Zero velocity if at limits
            at_lower_limit = torch.abs(self.q - self.joint_limits[:, 0]) < 1e-6
            at_upper_limit = torch.abs(self.q - self.joint_limits[:, 1]) < 1e-6
            limit_mask = at_lower_limit | at_upper_limit
            self.qd[limit_mask] = torch.clamp(self.qd[limit_mask], 0, float('inf')) * (~at_lower_limit[limit_mask]).float() + \
                                 torch.clamp(self.qd[limit_mask], float('-inf'), 0) * (~at_upper_limit[limit_mask]).float()
        
        return {
            'positions': self.q.clone(),
            'velocities': self.qd.clone(), 
            'accelerations': qdd,
            'torques': self.tau.clone()
        }
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state."""
        return {
            'positions': self.q.clone(),
            'velocities': self.qd.clone(),
            'torques': self.tau.clone()
        }
    
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set state."""
        if 'positions' in state:
            self.q = state['positions'].clone()
        if 'velocities' in state:
            self.qd = state['velocities'].clone()
        if 'torques' in state:
            self.tau = state['torques'].clone()
    
    def reset(self) -> None:
        """Reset to zero state."""
        self.q.zero_()
        self.qd.zero_()
        self.tau.zero_()


@dataclass
class SharedControlState:
    """Enhanced state representation for shared control."""
    robot_state: RobotState
    human_input: HumanInput
    environment_info: Dict[str, Any]
    safety_margins: Dict[str, float] = field(default_factory=dict)
    physics_state: Dict[str, torch.Tensor] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass 
class SharedControlAction:
    """Enhanced action representation for shared control."""
    robot_command: torch.Tensor  # Primary robot action
    assistance_level: float = 1.0  # How much to assist (0=passive, 1=full)
    safety_override: bool = False  # Emergency safety override
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedHumanModel(HumanModel):
    """Advanced human behavior model with learning and adaptation."""
    
    def __init__(self, 
                 skill_level: float = 0.7,
                 learning_rate: float = 0.01,
                 fatigue_rate: float = 0.001,
                 tremor_amplitude: float = 0.02,
                 reaction_delay: float = 0.1):
        """
        Initialize advanced human model.
        
        Args:
            skill_level: Base skill level (0-1)
            learning_rate: How quickly human learns from experience
            fatigue_rate: Rate of fatigue accumulation
            tremor_amplitude: Magnitude of hand tremor
            reaction_delay: Reaction delay in seconds
        """
        self.skill_level = skill_level
        self.learning_rate = learning_rate
        self.fatigue_rate = fatigue_rate
        self.tremor_amplitude = tremor_amplitude
        self.reaction_delay = reaction_delay
        
        # State variables
        self.experience = 0.0  # Accumulated experience
        self.fatigue = 0.0  # Current fatigue level
        self.last_action_time = 0.0
        
        # History for delay modeling
        self.action_history = deque(maxlen=10)
        
    def get_intention(self, env_state: EnvironmentState) -> HumanInput:
        """Get human intention with realistic modeling."""
        # Update internal state
        self.experience += self.learning_rate
        self.fatigue += self.fatigue_rate
        
        # Compute effective skill considering experience and fatigue
        effective_skill = min(1.0, self.skill_level + 0.1 * self.experience - 0.2 * self.fatigue)
        
        # Basic intention toward target
        if hasattr(env_state, 'target_position'):
            current_pos = env_state.robot.end_effector_pose[:3]
            target_pos = env_state.target_position
            direction = target_pos - current_pos
            
            if torch.norm(direction) > 1e-6:
                direction = direction / torch.norm(direction)
        else:
            direction = torch.zeros(3)
        
        # Apply skill level and add noise
        intention = direction * effective_skill
        
        # Add tremor (high-frequency noise)
        tremor = torch.normal(0, self.tremor_amplitude, size=intention.shape)
        intention += tremor
        
        # Add reaction delay effect
        current_time = time.time()
        if len(self.action_history) > 0 and (current_time - self.last_action_time) < self.reaction_delay:
            # Use delayed action
            intention = self.action_history[-1] if self.action_history else intention
        
        self.action_history.append(intention.clone())
        self.last_action_time = current_time
        
        # Compute confidence based on multiple factors
        distance_factor = 1.0 / (1.0 + torch.norm(direction).item())
        fatigue_factor = max(0.1, 1.0 - self.fatigue)
        confidence = effective_skill * distance_factor * fatigue_factor
        
        return HumanInput(
            intention=intention,
            confidence=min(1.0, max(0.0, confidence)),
            timestamp=current_time
        )
    
    def adapt_to_assistance(self, robot_action: torch.Tensor, assistance_level: float) -> HumanInput:
        """Model human adaptation to robot assistance."""
        # Reduce effort when robot provides more assistance
        effort_scaling = max(0.1, 1.0 - 0.5 * assistance_level)
        
        # Learn from good robot assistance
        if assistance_level > 0.3:
            self.experience += self.learning_rate * 0.1 * assistance_level
        
        # Adapted intention
        adapted_intention = robot_action[:3] * (1.0 - effort_scaling)
        
        return HumanInput(
            intention=adapted_intention,
            confidence=0.5 + 0.3 * assistance_level,
            timestamp=time.time()
        )


class SharedControlBase(gym.Env, ABC):
    """
    Base class for shared control environments with physics and safety.
    
    Provides common functionality for:
    - Physics simulation
    - Safety constraint monitoring
    - Human-robot interaction modeling
    - Reward computation
    - Visualization interface
    """
    
    def __init__(self,
                 robot_dof: int,
                 action_space_bounds: Tuple[float, float] = (-10.0, 10.0),
                 physics_engine: Optional[PhysicsEngine] = None,
                 human_model: Optional[HumanModel] = None,
                 control_frequency: float = 10.0,
                 max_episode_steps: int = 1000):
        """
        Initialize shared control environment.
        
        Args:
            robot_dof: Robot degrees of freedom
            action_space_bounds: (min, max) bounds for action space
            physics_engine: Physics simulation engine
            human_model: Human behavior model
            control_frequency: Control loop frequency (Hz)
            max_episode_steps: Maximum episode length
        """
        super().__init__()
        
        self.robot_dof = robot_dof
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.max_episode_steps = max_episode_steps
        
        # Physics engine
        self.physics = physics_engine if physics_engine is not None else \
                      SimplePhysicsEngine(robot_dof)
        
        # Human model
        self.human_model = human_model if human_model is not None else \
                          AdvancedHumanModel()
        
        # Action space
        self.action_space = spaces.Box(
            low=action_space_bounds[0],
            high=action_space_bounds[1], 
            shape=(robot_dof,),
            dtype=np.float32
        )
        
        # Observation space - to be defined by subclasses
        self.observation_space = None
        
        # Safety constraints
        self.safety_constraints: List[SafetyConstraint] = []
        
        # State
        self.current_state: Optional[SharedControlState] = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Performance tracking
        self.constraint_violations = 0
        self.safety_interventions = 0
        self.episode_stats = []
        
        logger.info(f"Initialized {self.__class__.__name__}: {robot_dof} DOF")
    
    @abstractmethod
    def _setup_environment_specific(self) -> None:
        """Setup environment-specific parameters. Called during __init__."""
        pass
    
    @abstractmethod
    def _compute_kinematics(self, joint_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute forward kinematics.
        
        Args:
            joint_positions: Joint positions [robot_dof]
            
        Returns:
            Dictionary with kinematic information (end_effector_pose, jacobian, etc.)
        """
        pass
    
    @abstractmethod
    def _get_environment_observation(self) -> torch.Tensor:
        """Get environment-specific observation components."""
        pass
    
    @abstractmethod
    def _compute_environment_reward(self, action: SharedControlAction) -> Tuple[float, Dict[str, float]]:
        """Compute environment-specific reward components.""" 
        pass
    
    @abstractmethod
    def _check_environment_termination(self) -> Tuple[bool, Dict[str, bool]]:
        """Check environment-specific termination conditions."""
        pass
    
    def add_safety_constraint(self, constraint: SafetyConstraint) -> None:
        """Add a safety constraint to the environment."""
        self.safety_constraints.append(constraint)
        logger.info(f"Added safety constraint: {constraint.name}")
    
    def remove_safety_constraint(self, constraint_name: str) -> bool:
        """Remove a safety constraint by name."""
        for i, constraint in enumerate(self.safety_constraints):
            if constraint.name == constraint_name:
                del self.safety_constraints[i]
                logger.info(f"Removed safety constraint: {constraint_name}")
                return True
        return False
    
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Reset physics
        self.physics.reset()
        
        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0
        self.constraint_violations = 0
        self.safety_interventions = 0
        
        # Initialize robot state from physics
        physics_state = self.physics.get_state()
        kinematics = self._compute_kinematics(physics_state['positions'])
        
        robot_state = RobotState(
            position=physics_state['positions'],
            velocity=physics_state['velocities'],
            torque=physics_state['torques'],
            end_effector_pose=kinematics['end_effector_pose'],
            contact_forces=torch.zeros(6)
        )
        
        # Initialize human input
        human_input = HumanInput(
            intention=torch.zeros(3),
            confidence=0.5,
            timestamp=0.0
        )
        
        # Environment-specific initialization
        env_info = self._initialize_environment()
        
        # Create state
        self.current_state = SharedControlState(
            robot_state=robot_state,
            human_input=human_input,
            environment_info=env_info,
            physics_state=physics_state,
            timestamp=0.0
        )
        
        return self._get_observation()
    
    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        # Convert and validate action
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        action = torch.clamp(action, 
                           torch.tensor(self.action_space.low, device=action.device),
                           torch.tensor(self.action_space.high, device=action.device))
        
        # Get human intention
        human_intention = self.human_model.get_intention(
            self._convert_state_for_human_model(self.current_state)
        )
        
        # Create shared control action
        shared_action = self._compute_shared_control_action(action, human_intention)
        
        # Check safety constraints before applying action
        safety_override = self._check_safety_constraints(shared_action)
        if safety_override:
            self.safety_interventions += 1
            shared_action.safety_override = True
            shared_action.robot_command = self._compute_safe_action(shared_action.robot_command)
        
        # Apply action through physics
        physics_result = self.physics.step(shared_action.robot_command, self.dt)
        
        # Update robot state
        kinematics = self._compute_kinematics(physics_result['positions'])
        self.current_state.robot_state = RobotState(
            position=physics_result['positions'],
            velocity=physics_result['velocities'], 
            torque=physics_result['torques'],
            end_effector_pose=kinematics['end_effector_pose'],
            contact_forces=self._compute_contact_forces(shared_action.robot_command)
        )
        
        # Update other state components
        self.current_state.human_input = human_intention
        self.current_state.physics_state = physics_result
        self.current_state.timestamp += self.dt
        self._update_environment_state(shared_action)
        
        # Compute reward
        reward, reward_info = self._compute_total_reward(shared_action)
        self.episode_reward += reward
        
        # Check termination
        done, done_info = self._check_total_termination()
        
        # Update step counter
        self.step_count += 1
        
        # Compile info
        info = {
            **reward_info,
            **done_info,
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
            'constraint_violations': self.constraint_violations,
            'safety_interventions': self.safety_interventions,
            'human_confidence': human_intention.confidence,
            'assistance_level': shared_action.assistance_level,
            'safety_override': shared_action.safety_override
        }
        
        return self._get_observation(), reward, done, info
    
    def _convert_state_for_human_model(self, state: SharedControlState) -> EnvironmentState:
        """Convert SharedControlState to EnvironmentState for human model."""
        return EnvironmentState(
            robot=state.robot_state,
            human_input=state.human_input,
            obstacles=state.environment_info.get('obstacles', []),
            target_position=state.environment_info.get('target_position', torch.zeros(3)),
            time_step=self.step_count,
            assistance_mode=state.environment_info.get('assistance_mode', AssistanceMode.COLLABORATIVE)
        )
    
    def _compute_shared_control_action(self, robot_action: torch.Tensor, 
                                     human_input: HumanInput) -> SharedControlAction:
        """Combine robot action with human input based on assistance strategy."""
        # Default: confidence-based blending
        confidence = human_input.confidence
        assistance_level = 1.0 - confidence
        
        # Convert human intention to robot action space
        human_robot_action = self._map_human_to_robot_action(human_input.intention)
        
        # Blend actions
        blended_action = (confidence * human_robot_action + 
                         (1.0 - confidence) * robot_action)
        
        return SharedControlAction(
            robot_command=blended_action,
            assistance_level=assistance_level
        )
    
    def _map_human_to_robot_action(self, human_intention: torch.Tensor) -> torch.Tensor:
        """Map human intention to robot action space. Override in subclasses."""
        # Default: pad or truncate to match robot DOF
        robot_action = torch.zeros(self.robot_dof, device=human_intention.device)
        min_dim = min(len(human_intention), self.robot_dof)
        robot_action[:min_dim] = human_intention[:min_dim]
        return robot_action
    
    def _check_safety_constraints(self, action: SharedControlAction) -> bool:
        """Check if action violates safety constraints."""
        for constraint in self.safety_constraints:
            if not constraint.active:
                continue
                
            result = constraint.evaluate(
                self._convert_state_for_human_model(self.current_state),
                action.robot_command
            )
            
            if result['violation']:
                self.constraint_violations += 1
                self.current_state.safety_margins[constraint.name] = result['margin']
                return True
        
        return False
    
    def _compute_safe_action(self, unsafe_action: torch.Tensor) -> torch.Tensor:
        """Compute safe alternative to unsafe action."""
        # Simple approach: project onto constraint boundary
        # In practice, this would use more sophisticated methods
        
        safe_action = unsafe_action.clone()
        
        # Reduce action magnitude as simple safety measure
        action_norm = torch.norm(safe_action)
        if action_norm > 0:
            safe_action = safe_action * min(1.0, 5.0 / action_norm)  # Limit to reasonable magnitude
        
        return safe_action
    
    def _compute_contact_forces(self, action: torch.Tensor) -> torch.Tensor:
        """Compute contact forces from action. Override in subclasses.""" 
        # Simple mapping for base implementation
        forces = torch.zeros(6)
        forces[:min(6, len(action))] = action[:min(6, len(action))]
        return forces
    
    def _get_observation(self) -> torch.Tensor:
        """Get full observation vector."""
        # Base observation components
        robot_obs = torch.cat([
            self.current_state.robot_state.position,
            self.current_state.robot_state.velocity,
            self.current_state.robot_state.end_effector_pose
        ])
        
        human_obs = torch.cat([
            self.current_state.human_input.intention,
            torch.tensor([self.current_state.human_input.confidence])
        ])
        
        # Environment-specific observation
        env_obs = self._get_environment_observation()
        
        return torch.cat([robot_obs, human_obs, env_obs])
    
    def _compute_total_reward(self, action: SharedControlAction) -> Tuple[float, Dict[str, float]]:
        """Compute total reward including safety penalties."""
        # Environment-specific reward
        env_reward, env_reward_info = self._compute_environment_reward(action)
        
        # Safety penalties
        safety_penalty = 0.0
        for constraint in self.safety_constraints:
            if not constraint.active:
                continue
                
            result = constraint.evaluate(
                self._convert_state_for_human_model(self.current_state),
                action.robot_command
            )
            safety_penalty += result['penalty']
        
        # Assistance efficiency reward (encourage minimal assistance when possible)
        efficiency_reward = -0.01 * action.assistance_level
        
        # Total reward
        total_reward = env_reward - safety_penalty + efficiency_reward
        
        reward_info = {
            **env_reward_info,
            'safety_penalty': safety_penalty,
            'efficiency_reward': efficiency_reward
        }
        
        return total_reward, reward_info
    
    def _check_total_termination(self) -> Tuple[bool, Dict[str, bool]]:
        """Check all termination conditions."""
        # Environment-specific termination
        env_done, env_done_info = self._check_environment_termination()
        
        # Time limit
        time_limit = self.step_count >= self.max_episode_steps
        
        # Safety termination (too many violations)
        safety_termination = self.constraint_violations > 10
        
        done = env_done or time_limit or safety_termination
        
        done_info = {
            **env_done_info,
            'time_limit': time_limit,
            'safety_termination': safety_termination
        }
        
        return done, done_info
    
    @abstractmethod
    def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize environment-specific components."""
        pass
    
    @abstractmethod
    def _update_environment_state(self, action: SharedControlAction) -> None:
        """Update environment-specific state components."""
        pass
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety-related metrics for the current episode."""
        return {
            'constraint_violations': self.constraint_violations,
            'safety_interventions': self.safety_interventions,
            'violation_rate': self.constraint_violations / max(1, self.step_count),
            'intervention_rate': self.safety_interventions / max(1, self.step_count),
            'safety_margins': self.current_state.safety_margins if self.current_state else {}
        }
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment. Override in subclasses for specific visualization."""
        if mode == "text":
            if self.current_state:
                print(f"Step: {self.step_count}")
                print(f"Robot position: {self.current_state.robot_state.position.numpy()}")
                print(f"End effector: {self.current_state.robot_state.end_effector_pose[:3].numpy()}")
                print(f"Human confidence: {self.current_state.human_input.confidence:.3f}")
                print(f"Constraint violations: {self.constraint_violations}")
                print("---")
        else:
            logger.warning(f"Rendering mode '{mode}' not implemented")
        
        return None