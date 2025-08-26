"""
Standardized Environment Suite for Safe RL Benchmarking.

This module provides standardized environments across different robot platforms,
human interaction models, and safety scenarios for consistent evaluation.
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum, auto
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class RobotPlatform(Enum):
    """Robot platform types for evaluation."""
    MANIPULATOR_7DOF = auto()
    MANIPULATOR_6DOF = auto()
    MOBILE_BASE = auto()
    HUMANOID = auto()
    QUADRUPED = auto()
    AERIAL_VEHICLE = auto()


class HumanModelType(Enum):
    """Human behavior model types."""
    COOPERATIVE = auto()
    ADVERSARIAL = auto()
    INCONSISTENT = auto()
    NOVICE = auto()
    EXPERT = auto()
    ADAPTIVE = auto()


class SafetyScenario(Enum):
    """Safety scenario types."""
    OBSTACLE_AVOIDANCE = auto()
    FORCE_LIMITS = auto()
    WORKSPACE_BOUNDS = auto()
    VELOCITY_CONSTRAINTS = auto()
    COLLISION_PREVENTION = auto()
    HUMAN_SAFETY = auto()


@dataclass
class EnvironmentConfig:
    """Configuration for standardized environments."""
    robot_platform: RobotPlatform
    human_model: HumanModelType
    safety_scenario: SafetyScenario
    
    # Environment parameters
    max_episode_steps: int = 1000
    dt: float = 0.01  # Control timestep
    
    # Safety parameters
    safety_threshold: float = 0.1
    force_limit: float = 10.0
    velocity_limit: float = 1.0
    workspace_bounds: List[Tuple[float, float]] = field(default_factory=lambda: [(-1, 1), (-1, 1), (0, 2)])
    
    # Human model parameters
    human_stiffness: float = 100.0
    human_damping: float = 10.0
    human_reaction_delay: float = 0.1
    cooperation_level: float = 0.8
    
    # Reward parameters
    task_reward_weight: float = 1.0
    safety_penalty_weight: float = 10.0
    efficiency_reward_weight: float = 0.1
    
    # Noise parameters
    observation_noise_std: float = 0.01
    action_noise_std: float = 0.01
    
    # Rendering
    render_mode: Optional[str] = None
    
    # Seed
    seed: Optional[int] = None


class BaseStandardizedEnv(gym.Env):
    """Base class for standardized Safe RL environments."""
    
    def __init__(self, config: EnvironmentConfig):
        self.config = config
        
        # Set random seed
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # Initialize spaces (to be defined by subclasses)
        self.observation_space = None
        self.action_space = None
        
        # Environment state
        self.current_step = 0
        self.episode_return = 0.0
        self.episode_cost = 0.0
        
        # Human model state
        self.human_state = None
        self.human_intent = None
        self.human_force = np.zeros(3)
        
        # Safety monitoring
        self.safety_violations = []
        self.constraint_violations = 0
        
        # Task-specific state (to be defined by subclasses)
        self.task_state = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.episode_return = 0.0
        self.episode_cost = 0.0
        self.constraint_violations = 0
        self.safety_violations = []
        
        # Reset human model
        self._reset_human_model()
        
        # Reset robot state
        observation = self._reset_robot_state()
        
        # Reset task
        self._reset_task()
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Clip and add noise to action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.config.action_noise_std > 0:
            action += np.random.normal(0, self.config.action_noise_std, action.shape)
        
        # Update human model
        self._update_human_model()
        
        # Execute robot action
        next_observation = self._execute_robot_action(action)
        
        # Compute reward and cost
        reward, cost = self._compute_reward_and_cost(action)
        
        # Check termination conditions
        done = self._check_termination()
        
        # Update episode statistics
        self.current_step += 1
        self.episode_return += reward
        self.episode_cost += cost
        
        # Add observation noise
        if self.config.observation_noise_std > 0:
            next_observation += np.random.normal(
                0, self.config.observation_noise_std, next_observation.shape
            )
        
        # Create info dict
        info = self._get_info_dict(action, reward, cost)
        
        return next_observation, reward, done, info
    
    def _reset_human_model(self):
        """Reset human model to initial state."""
        if self.config.human_model == HumanModelType.COOPERATIVE:
            self.human_intent = self._generate_cooperative_intent()
        elif self.config.human_model == HumanModelType.ADVERSARIAL:
            self.human_intent = self._generate_adversarial_intent()
        elif self.config.human_model == HumanModelType.INCONSISTENT:
            self.human_intent = self._generate_inconsistent_intent()
        elif self.config.human_model == HumanModelType.NOVICE:
            self.human_intent = self._generate_novice_intent()
        elif self.config.human_model == HumanModelType.EXPERT:
            self.human_intent = self._generate_expert_intent()
        else:  # ADAPTIVE
            self.human_intent = self._generate_adaptive_intent()
        
        self.human_state = {
            'position': np.zeros(3),
            'velocity': np.zeros(3),
            'force': np.zeros(3),
            'stiffness': self.config.human_stiffness,
            'damping': self.config.human_damping
        }
    
    def _update_human_model(self):
        """Update human model state."""
        # Simulate human dynamics based on model type
        if self.config.human_model == HumanModelType.COOPERATIVE:
            self._update_cooperative_human()
        elif self.config.human_model == HumanModelType.ADVERSARIAL:
            self._update_adversarial_human()
        elif self.config.human_model == HumanModelType.INCONSISTENT:
            self._update_inconsistent_human()
        else:
            self._update_default_human()
        
        # Add reaction delay
        if hasattr(self, '_delayed_human_force'):
            self.human_force = self._delayed_human_force
        
        delay_steps = int(self.config.human_reaction_delay / self.config.dt)
        if not hasattr(self, '_human_force_buffer'):
            self._human_force_buffer = [np.zeros(3)] * delay_steps
        
        self._human_force_buffer.append(self.human_state['force'].copy())
        self._delayed_human_force = self._human_force_buffer.pop(0)
    
    def _generate_cooperative_intent(self) -> Dict[str, Any]:
        """Generate cooperative human intent."""
        return {
            'goal_position': np.array([0.5, 0.0, 1.0]),
            'cooperation_level': self.config.cooperation_level,
            'assistance_weight': 0.8
        }
    
    def _generate_adversarial_intent(self) -> Dict[str, Any]:
        """Generate adversarial human intent.""" 
        return {
            'goal_position': np.array([-0.5, 0.0, 1.0]),  # Opposite direction
            'cooperation_level': -0.5,
            'disturbance_magnitude': 2.0
        }
    
    def _generate_inconsistent_intent(self) -> Dict[str, Any]:
        """Generate inconsistent human intent."""
        return {
            'goal_position': np.random.uniform(-1, 1, 3),
            'cooperation_level': np.random.uniform(-0.5, 1.0),
            'change_frequency': 0.1  # Probability of changing intent each step
        }
    
    def _generate_novice_intent(self) -> Dict[str, Any]:
        """Generate novice human intent."""
        return {
            'goal_position': np.array([0.3, 0.0, 0.8]),
            'cooperation_level': 0.3,  # Lower cooperation due to inexperience
            'uncertainty_level': 0.7,
            'reaction_time_multiplier': 2.0
        }
    
    def _generate_expert_intent(self) -> Dict[str, Any]:
        """Generate expert human intent."""
        return {
            'goal_position': np.array([0.8, 0.0, 1.2]),
            'cooperation_level': 0.95,
            'precision_level': 0.9,
            'anticipation_capability': 0.8
        }
    
    def _generate_adaptive_intent(self) -> Dict[str, Any]:
        """Generate adaptive human intent."""
        return {
            'goal_position': np.array([0.6, 0.0, 1.0]),
            'cooperation_level': 0.7,
            'adaptation_rate': 0.1,
            'learning_enabled': True
        }
    
    def _update_cooperative_human(self):
        """Update cooperative human model."""
        # Human tries to assist robot in reaching shared goal
        goal_error = self.human_intent['goal_position'] - self.human_state['position']
        
        # Cooperative force towards goal
        assistance_force = self.human_intent['assistance_weight'] * goal_error
        
        # Spring-damper model for human arm
        desired_force = (
            self.human_state['stiffness'] * goal_error -
            self.human_state['damping'] * self.human_state['velocity']
        )
        
        self.human_state['force'] = assistance_force + 0.3 * desired_force
    
    def _update_adversarial_human(self):
        """Update adversarial human model."""
        # Human applies disturbances
        disturbance = np.random.normal(0, self.human_intent['disturbance_magnitude'], 3)
        
        # Bias towards opposite goal
        goal_error = self.human_intent['goal_position'] - self.human_state['position']
        opposing_force = -self.human_intent.get('cooperation_level', -0.5) * goal_error
        
        self.human_state['force'] = disturbance + opposing_force
    
    def _update_inconsistent_human(self):
        """Update inconsistent human model."""
        # Randomly change intent
        if np.random.random() < self.human_intent['change_frequency']:
            self.human_intent['goal_position'] = np.random.uniform(-1, 1, 3)
            self.human_intent['cooperation_level'] = np.random.uniform(-0.5, 1.0)
        
        # Apply current intent
        goal_error = self.human_intent['goal_position'] - self.human_state['position']
        cooperation = self.human_intent['cooperation_level']
        
        self.human_state['force'] = cooperation * goal_error
    
    def _update_default_human(self):
        """Default human model update."""
        # Simple spring-damper model
        goal_error = self.human_intent['goal_position'] - self.human_state['position']
        
        self.human_state['force'] = (
            self.human_state['stiffness'] * goal_error -
            self.human_state['damping'] * self.human_state['velocity']
        )
    
    def _check_safety_violations(self, action: np.ndarray) -> List[str]:
        """Check for safety violations."""
        violations = []
        
        if self.config.safety_scenario == SafetyScenario.FORCE_LIMITS:
            if np.linalg.norm(action) > self.config.force_limit:
                violations.append('force_limit_exceeded')
        
        if self.config.safety_scenario == SafetyScenario.VELOCITY_CONSTRAINTS:
            if hasattr(self, 'robot_velocity'):
                if np.linalg.norm(self.robot_velocity) > self.config.velocity_limit:
                    violations.append('velocity_limit_exceeded')
        
        if self.config.safety_scenario == SafetyScenario.WORKSPACE_BOUNDS:
            if hasattr(self, 'robot_position'):
                for i, (low, high) in enumerate(self.config.workspace_bounds):
                    if self.robot_position[i] < low or self.robot_position[i] > high:
                        violations.append('workspace_bounds_violated')
                        break
        
        return violations
    
    def get_human_metrics(self) -> Dict[str, float]:
        """Get human-centric evaluation metrics."""
        # Compute satisfaction based on goal achievement
        if hasattr(self, 'robot_position') and 'goal_position' in self.human_intent:
            goal_distance = np.linalg.norm(
                self.robot_position - self.human_intent['goal_position']
            )
            satisfaction = np.exp(-goal_distance)
        else:
            satisfaction = 0.5
        
        # Compute trust based on safety violations
        violation_rate = len(self.safety_violations) / max(1, self.current_step)
        trust = np.exp(-5 * violation_rate)
        
        # Workload based on human force magnitude
        avg_force = np.mean([np.linalg.norm(self.human_state['force'])])
        workload = np.clip(avg_force / 10.0, 0, 1)
        
        # Naturalness based on consistency
        naturalness = 0.8  # Placeholder
        
        # Collaboration efficiency
        if hasattr(self, 'task_completion_time'):
            efficiency = 1.0 / (1.0 + self.task_completion_time / 100.0)
        else:
            efficiency = 0.7
        
        return {
            'satisfaction': satisfaction,
            'trust': trust,
            'workload': workload,
            'naturalness': naturalness,
            'collaboration_efficiency': efficiency
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return {
            'robot_platform': self.config.robot_platform.name,
            'human_model': self.config.human_model.name,
            'safety_scenario': self.config.safety_scenario.name,
            'max_episode_steps': self.config.max_episode_steps,
            'dt': self.config.dt,
            'safety_threshold': self.config.safety_threshold
        }
    
    # Abstract methods to be implemented by subclasses
    def _reset_robot_state(self) -> np.ndarray:
        """Reset robot state and return initial observation."""
        raise NotImplementedError
    
    def _reset_task(self):
        """Reset task-specific state."""
        raise NotImplementedError
    
    def _execute_robot_action(self, action: np.ndarray) -> np.ndarray:
        """Execute robot action and return next observation."""
        raise NotImplementedError
    
    def _compute_reward_and_cost(self, action: np.ndarray) -> Tuple[float, float]:
        """Compute reward and cost for current state and action."""
        raise NotImplementedError
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        return self.current_step >= self.config.max_episode_steps
    
    def _get_info_dict(self, action: np.ndarray, reward: float, cost: float) -> Dict[str, Any]:
        """Create info dictionary for step return."""
        # Check safety violations
        violations = self._check_safety_violations(action)
        self.safety_violations.extend(violations)
        
        if violations:
            self.constraint_violations += 1
        
        info = {
            'cost': cost,
            'constraint_violation': len(violations) > 0,
            'safety_violations': violations,
            'human_force': self.human_force.copy(),
            'episode_return': self.episode_return,
            'episode_cost': self.episode_cost,
            'current_step': self.current_step
        }
        
        # Add human metrics
        info.update(self.get_human_metrics())
        
        return info


class ManipulatorEnv(BaseStandardizedEnv):
    """Standardized manipulator environment."""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__(config)
        
        # Set up action and observation spaces
        if config.robot_platform == RobotPlatform.MANIPULATOR_7DOF:
            self.dof = 7
        else:  # MANIPULATOR_6DOF
            self.dof = 6
        
        self.action_space = spaces.Box(
            low=-config.force_limit, 
            high=config.force_limit,
            shape=(self.dof,),
            dtype=np.float32
        )
        
        # Observation: joint positions, velocities, human force
        obs_dim = self.dof * 2 + 3  # positions + velocities + human_force
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Robot state
        self.robot_position = np.zeros(self.dof)
        self.robot_velocity = np.zeros(self.dof)
        
        # Task parameters
        self.target_position = np.zeros(self.dof)
    
    def _reset_robot_state(self) -> np.ndarray:
        """Reset robot to initial configuration."""
        self.robot_position = np.random.uniform(-0.1, 0.1, self.dof)
        self.robot_velocity = np.zeros(self.dof)
        
        # Set random target
        self.target_position = np.random.uniform(-1, 1, self.dof)
        
        return self._get_observation()
    
    def _reset_task(self):
        """Reset task-specific state."""
        self.task_state = {
            'target_reached': False,
            'task_completion_time': 0
        }
    
    def _execute_robot_action(self, action: np.ndarray) -> np.ndarray:
        """Execute joint torque action."""
        # Simple robot dynamics: mass-damper system
        mass = 1.0
        damping = 0.1
        
        # Include human force (mapped to joint space)
        human_joint_force = np.random.normal(0, 0.1, self.dof)  # Simplified mapping
        
        # Update robot state
        acceleration = (action + human_joint_force - damping * self.robot_velocity) / mass
        self.robot_velocity += acceleration * self.config.dt
        self.robot_position += self.robot_velocity * self.config.dt
        
        # Update human state (simplified)
        self.human_state['position'] = self.robot_position[:3]  # Map to Cartesian
        self.human_state['velocity'] = self.robot_velocity[:3]
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([
            self.robot_position,
            self.robot_velocity,
            self.human_force
        ])
    
    def _compute_reward_and_cost(self, action: np.ndarray) -> Tuple[float, float]:
        """Compute reward and cost."""
        # Task reward: reaching target
        position_error = np.linalg.norm(self.robot_position - self.target_position)
        task_reward = self.config.task_reward_weight * np.exp(-position_error)
        
        # Efficiency reward: minimize control effort
        efficiency_reward = -self.config.efficiency_reward_weight * np.linalg.norm(action)
        
        # Total reward
        reward = task_reward + efficiency_reward
        
        # Safety cost
        violations = self._check_safety_violations(action)
        safety_cost = self.config.safety_penalty_weight * len(violations)
        
        # Control effort cost
        control_cost = 0.01 * np.linalg.norm(action)
        
        cost = safety_cost + control_cost
        
        return reward, cost


class StandardizedEnvSuite:
    """Suite of standardized environments for evaluation."""
    
    def __init__(self):
        self.env_registry = {
            RobotPlatform.MANIPULATOR_7DOF: ManipulatorEnv,
            RobotPlatform.MANIPULATOR_6DOF: ManipulatorEnv,
            # Add more platforms as needed
        }
        
        logger.info("Standardized Environment Suite initialized")
    
    def create_environment(self, 
                          robot_platform: RobotPlatform,
                          human_model: HumanModelType,
                          safety_scenario: SafetyScenario,
                          seed: Optional[int] = None,
                          **kwargs) -> BaseStandardizedEnv:
        """Create a standardized environment."""
        
        config = EnvironmentConfig(
            robot_platform=robot_platform,
            human_model=human_model,
            safety_scenario=safety_scenario,
            seed=seed,
            **kwargs
        )
        
        if robot_platform in self.env_registry:
            env_class = self.env_registry[robot_platform]
            return env_class(config)
        else:
            raise NotImplementedError(f"Environment for {robot_platform} not implemented")
    
    def list_available_platforms(self) -> List[RobotPlatform]:
        """List available robot platforms."""
        return list(self.env_registry.keys())
    
    def register_environment(self, platform: RobotPlatform, env_class):
        """Register a new environment class."""
        self.env_registry[platform] = env_class
        logger.info(f"Registered environment for {platform}")
    
    def get_default_config(self, 
                          robot_platform: RobotPlatform,
                          human_model: HumanModelType,
                          safety_scenario: SafetyScenario) -> EnvironmentConfig:
        """Get default configuration for environment."""
        return EnvironmentConfig(
            robot_platform=robot_platform,
            human_model=human_model,
            safety_scenario=safety_scenario
        )