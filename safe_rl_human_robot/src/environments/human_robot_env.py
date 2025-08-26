"""
Human-Robot Shared Control Environment for Safe Reinforcement Learning.

This module implements the shared control environment where a robot assists
a human user while maintaining safety constraints. Supports various assistive
devices like exoskeletons and wheelchairs.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
import numpy as np
import torch
import gym
from gym import spaces
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AssistanceMode(Enum):
    """Different modes of human-robot assistance."""
    PASSIVE = "passive"          # Robot provides no assistance
    COLLABORATIVE = "collaborative"  # Shared control between human and robot
    CORRECTIVE = "corrective"    # Robot intervenes only to prevent safety violations
    ADAPTIVE = "adaptive"        # Assistance level adapts based on human performance


@dataclass
class HumanInput:
    """Human input representation."""
    intention: torch.Tensor      # Intended action/direction [action_dim]
    confidence: float           # Confidence in intention (0-1)
    timestamp: float            # Time of input
    muscle_activity: Optional[torch.Tensor] = None  # EMG signals [num_sensors]
    gaze_direction: Optional[torch.Tensor] = None   # Eye tracking [3] (unit vector)


@dataclass
class RobotState:
    """Robot state representation."""
    position: torch.Tensor       # Joint positions [num_joints]
    velocity: torch.Tensor       # Joint velocities [num_joints]
    torque: torch.Tensor         # Joint torques [num_joints]
    end_effector_pose: torch.Tensor  # End-effector pose [6] (position + orientation)
    contact_forces: torch.Tensor # Contact forces [6] (force + torque)


@dataclass
class EnvironmentState:
    """Complete environment state."""
    robot: RobotState
    human_input: HumanInput
    obstacles: List[torch.Tensor]  # List of obstacle positions
    target_position: torch.Tensor  # Goal/target position [3]
    time_step: int
    assistance_mode: AssistanceMode


class HumanModel(ABC):
    """Abstract human behavior model for simulation."""
    
    @abstractmethod
    def get_intention(self, env_state: EnvironmentState) -> HumanInput:
        """Get human intention given current environment state."""
        pass
    
    @abstractmethod
    def adapt_to_assistance(self, robot_action: torch.Tensor, 
                          assistance_level: float) -> HumanInput:
        """Adapt human behavior based on robot assistance."""
        pass


class SimpleHumanModel(HumanModel):
    """Simple human model for basic testing and development."""
    
    def __init__(self, skill_level: float = 0.7, noise_level: float = 0.1,
                 adaptation_rate: float = 0.1):
        """
        Initialize simple human model.
        
        Args:
            skill_level: Human skill level (0-1), affects accuracy
            noise_level: Amount of noise in human actions
            adaptation_rate: How quickly human adapts to robot assistance
        """
        self.skill_level = skill_level
        self.noise_level = noise_level
        self.adaptation_rate = adaptation_rate
        self.learning_progress = 0.0
        
    def get_intention(self, env_state: EnvironmentState) -> HumanInput:
        """
        Generate human intention toward target with noise and skill level.
        """
        # Direction towards target
        current_pos = env_state.robot.end_effector_pose[:3]
        target_pos = env_state.target_position
        direction = target_pos - current_pos
        
        # Normalize and apply skill level
        if torch.norm(direction) > 1e-6:
            direction = direction / torch.norm(direction)
            
        # Add skill-dependent accuracy
        accuracy = self.skill_level + self.learning_progress * 0.3
        intended_direction = direction * accuracy
        
        # Add noise
        noise = torch.normal(0, self.noise_level, size=direction.shape)
        noisy_intention = intended_direction + noise
        
        # Confidence based on skill and distance to target
        distance_to_target = torch.norm(target_pos - current_pos)
        confidence = min(1.0, accuracy * (2.0 / (1.0 + distance_to_target)))
        
        return HumanInput(
            intention=noisy_intention,
            confidence=confidence,
            timestamp=env_state.time_step * 0.1  # Assuming 10Hz control
        )
    
    def adapt_to_assistance(self, robot_action: torch.Tensor, 
                          assistance_level: float) -> HumanInput:
        """
        Simulate human adaptation to robot assistance.
        
        Higher assistance may reduce human effort but also learning.
        """
        # Human tends to rely more on robot when assistance is high
        effort_reduction = assistance_level * 0.5
        
        # But also learns from good assistance
        if assistance_level > 0.3:  # Good assistance threshold
            self.learning_progress += self.adaptation_rate * 0.01
            self.learning_progress = min(self.learning_progress, 0.5)
        
        # Return modified intention (placeholder - would be more complex in reality)
        return HumanInput(
            intention=robot_action * effort_reduction,
            confidence=0.5 + assistance_level * 0.3,
            timestamp=0.0
        )


class HumanRobotEnv(gym.Env):
    """
    OpenAI Gym environment for human-robot shared control.
    
    State space includes robot configuration, human input, and environmental context.
    Action space represents robot assistance commands.
    Reward combines task performance with safety penalties.
    """
    
    def __init__(self,
                 robot_dof: int = 6,
                 max_episode_steps: int = 1000,
                 control_frequency: float = 10.0,
                 human_model: Optional[HumanModel] = None,
                 assistance_mode: AssistanceMode = AssistanceMode.COLLABORATIVE,
                 safety_distance: float = 0.1,
                 force_limit: float = 50.0,
                 workspace_bounds: Tuple[float, float, float] = (-1.0, 1.0, 2.0)):
        """
        Initialize human-robot shared control environment.
        
        Args:
            robot_dof: Degrees of freedom for robot
            max_episode_steps: Maximum steps per episode
            control_frequency: Control loop frequency (Hz)
            human_model: Human behavior model for simulation
            assistance_mode: Mode of robot assistance
            safety_distance: Minimum safe distance to human/obstacles
            force_limit: Maximum allowable force
            workspace_bounds: (x_min, x_max, height) workspace boundaries
        """
        super().__init__()
        
        self.robot_dof = robot_dof
        self.max_episode_steps = max_episode_steps
        self.dt = 1.0 / control_frequency
        self.assistance_mode = assistance_mode
        self.safety_distance = safety_distance
        self.force_limit = force_limit
        self.workspace_bounds = workspace_bounds
        
        # Human model
        self.human_model = human_model or SimpleHumanModel()
        
        # Define action space (robot assistance commands)
        # Actions represent desired assistance forces/torques
        self.action_space = spaces.Box(
            low=-force_limit, 
            high=force_limit, 
            shape=(robot_dof,),
            dtype=np.float32
        )
        
        # Define observation space
        # [robot_pos, robot_vel, human_intention, target_pos, obstacles, contact_forces]
        obs_dim = (robot_dof +           # robot positions
                  robot_dof +           # robot velocities  
                  3 +                   # human intention direction
                  1 +                   # human confidence
                  3 +                   # target position
                  3 +                   # end effector position
                  6 +                   # contact forces
                  4)                    # obstacle info (simplified)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Environment state
        self.current_state: Optional[EnvironmentState] = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Task parameters
        self.target_reached_threshold = 0.05  # meters
        self.success_reward = 100.0
        self.collision_penalty = -1000.0
        self.efficiency_weight = 0.1
        
        # Logging
        self.episode_stats = []
        
        logger.info(f"Initialized HumanRobotEnv: {robot_dof} DOF, mode={assistance_mode.value}")
        
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        super().reset(seed=seed)
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        # Initialize robot state
        robot_state = RobotState(
            position=torch.zeros(self.robot_dof),
            velocity=torch.zeros(self.robot_dof),
            torque=torch.zeros(self.robot_dof),
            end_effector_pose=torch.tensor([0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),  # x,y,z,rx,ry,rz
            contact_forces=torch.zeros(6)
        )
        
        # Random target position within workspace
        target_x = np.random.uniform(self.workspace_bounds[0], self.workspace_bounds[1])
        target_y = np.random.uniform(self.workspace_bounds[0], self.workspace_bounds[1])
        target_z = np.random.uniform(0.5, self.workspace_bounds[2])
        target_position = torch.tensor([target_x, target_y, target_z])
        
        # Initial human input
        human_input = HumanInput(
            intention=torch.zeros(3),
            confidence=0.5,
            timestamp=0.0
        )
        
        # Simple obstacle (single point for now)
        obstacles = [torch.tensor([0.2, 0.2, 1.0])]  # Fixed obstacle position
        
        # Create environment state
        self.current_state = EnvironmentState(
            robot=robot_state,
            human_input=human_input,
            obstacles=obstacles,
            target_position=target_position,
            time_step=0,
            assistance_mode=self.assistance_mode
        )
        
        # Reset counters
        self.step_count = 0
        self.episode_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Robot assistance action [robot_dof]
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
            
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        
        # Clip actions to valid range
        action = torch.clamp(action, -self.force_limit, self.force_limit)
        
        # Get human intention
        human_intention = self.human_model.get_intention(self.current_state)
        
        # Combine human intention with robot assistance
        combined_action = self._combine_human_robot_actions(human_intention.intention, action)
        
        # Apply robot dynamics (simplified)
        self._update_robot_state(combined_action, action)
        
        # Update human input in state
        self.current_state.human_input = human_intention
        self.current_state.time_step += 1
        
        # Compute reward
        reward, reward_info = self._compute_reward(action)
        self.episode_reward += reward
        
        # Check termination conditions
        done, done_info = self._check_done()
        
        # Increment step counter
        self.step_count += 1
        
        # Compile info dict
        info = {
            **reward_info,
            **done_info,
            "step_count": self.step_count,
            "episode_reward": self.episode_reward,
            "human_confidence": human_intention.confidence,
            "assistance_level": torch.norm(action).item() / self.force_limit
        }
        
        return self._get_observation(), reward, done, info
    
    def _combine_human_robot_actions(self, human_intention: torch.Tensor, 
                                   robot_action: torch.Tensor) -> torch.Tensor:
        """
        Combine human intention with robot assistance based on assistance mode.
        
        Args:
            human_intention: Human intended direction [3]
            robot_action: Robot assistance forces [robot_dof]
            
        Returns:
            Combined control command [robot_dof] 
        """
        # Convert human intention to robot space (simplified mapping)
        human_force = torch.zeros(self.robot_dof)
        human_force[:min(3, self.robot_dof)] = human_intention[:min(3, self.robot_dof)]
        
        if self.assistance_mode == AssistanceMode.PASSIVE:
            # Robot provides no assistance
            return human_force
            
        elif self.assistance_mode == AssistanceMode.COLLABORATIVE:
            # Weighted combination based on human confidence
            confidence = self.current_state.human_input.confidence
            human_weight = confidence
            robot_weight = 1.0 - confidence
            return human_weight * human_force + robot_weight * robot_action
            
        elif self.assistance_mode == AssistanceMode.CORRECTIVE:
            # Robot intervenes only if human action would violate safety
            if self._would_violate_safety(human_force):
                return robot_action  # Override with safe action
            else:
                return human_force   # Use human action
                
        elif self.assistance_mode == AssistanceMode.ADAPTIVE:
            # Adapt assistance based on task performance and human skill
            performance_metric = self._compute_performance_metric()
            assistance_level = max(0.1, 1.0 - performance_metric)
            return (1.0 - assistance_level) * human_force + assistance_level * robot_action
            
        else:
            return human_force
    
    def _update_robot_state(self, combined_action: torch.Tensor, robot_action: torch.Tensor) -> None:
        """
        Update robot state using simplified dynamics.
        
        Args:
            combined_action: Combined human-robot command
            robot_action: Pure robot assistance for contact force calculation
        """
        robot = self.current_state.robot
        
        # Simple integration for joint positions and velocities
        # In reality, this would involve complex dynamics and control
        
        # Update joint torques (proportional to combined action)
        robot.torque = combined_action
        
        # Simple velocity update (acceleration proportional to torque)
        mass_matrix = torch.eye(self.robot_dof) * 1.0  # Simplified mass matrix
        joint_acceleration = torch.solve(robot.torque.unsqueeze(1), mass_matrix)[0].squeeze()
        
        # Integrate velocity and position
        robot.velocity += joint_acceleration * self.dt
        robot.position += robot.velocity * self.dt
        
        # Apply joint limits (simplified)
        joint_limits = torch.tensor([[-2.0, 2.0]] * self.robot_dof)  # ±2 rad for all joints
        robot.position = torch.clamp(robot.position, 
                                   joint_limits[:, 0], joint_limits[:, 1])
        
        # Update end-effector position (simplified forward kinematics)
        # In reality, this would use proper kinematics
        robot.end_effector_pose[:3] = torch.tensor([
            0.5 + 0.3 * torch.sin(robot.position[0]) + 0.2 * torch.cos(robot.position[1]),
            0.0 + 0.3 * torch.cos(robot.position[0]) + 0.2 * torch.sin(robot.position[2]),
            1.0 + 0.2 * robot.position[1] + 0.1 * robot.position[2]
        ])
        
        # Update contact forces (based on robot assistance level)
        robot.contact_forces = torch.cat([robot_action[:3], robot_action[3:6]]) \
                              if self.robot_dof >= 6 else torch.cat([robot_action, torch.zeros(6 - self.robot_dof)])
    
    def _get_observation(self) -> torch.Tensor:
        """
        Get current observation vector.
        
        Returns:
            Observation tensor matching observation_space
        """
        robot = self.current_state.robot
        human = self.current_state.human_input
        
        # Obstacle info (distance and direction to nearest obstacle)
        if self.current_state.obstacles:
            nearest_obstacle = self.current_state.obstacles[0]  # Simplified
            obstacle_dist = torch.norm(robot.end_effector_pose[:3] - nearest_obstacle)
            obstacle_dir = (nearest_obstacle - robot.end_effector_pose[:3]) / (obstacle_dist + 1e-6)
            obstacle_info = torch.cat([torch.tensor([obstacle_dist]), obstacle_dir])
        else:
            obstacle_info = torch.zeros(4)
        
        observation = torch.cat([
            robot.position,                    # robot joint positions
            robot.velocity,                    # robot joint velocities
            human.intention,                   # human intention direction
            torch.tensor([human.confidence]),  # human confidence
            self.current_state.target_position, # target position
            robot.end_effector_pose[:3],       # end effector position
            robot.contact_forces,              # contact forces
            obstacle_info                      # obstacle information
        ])
        
        return observation.float()
    
    def _compute_reward(self, robot_action: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward function balancing task performance and safety.
        
        Args:
            robot_action: Robot assistance action
            
        Returns:
            Tuple of (total_reward, reward_components)
        """
        robot = self.current_state.robot
        
        # Task progress reward
        distance_to_target = torch.norm(robot.end_effector_pose[:3] - self.current_state.target_position)
        progress_reward = -distance_to_target.item()  # Closer to target is better
        
        # Efficiency penalty (discourage excessive robot assistance)
        assistance_magnitude = torch.norm(robot_action).item()
        efficiency_penalty = -self.efficiency_weight * assistance_magnitude
        
        # Safety penalty for constraint violations
        safety_penalty = 0.0
        
        # Check collision with obstacles
        for obstacle in self.current_state.obstacles:
            dist_to_obstacle = torch.norm(robot.end_effector_pose[:3] - obstacle)
            if dist_to_obstacle < self.safety_distance:
                safety_penalty += self.collision_penalty * (self.safety_distance - dist_to_obstacle.item())
        
        # Check force limits
        max_force = torch.max(torch.abs(robot.contact_forces))
        if max_force > self.force_limit:
            safety_penalty += -10.0 * (max_force.item() - self.force_limit)
        
        # Success bonus
        success_bonus = 0.0
        if distance_to_target < self.target_reached_threshold:
            success_bonus = self.success_reward
        
        # Smooth human-robot collaboration reward
        human_confidence = self.current_state.human_input.confidence
        collaboration_reward = 0.1 * (1.0 - abs(assistance_magnitude / self.force_limit - (1.0 - human_confidence)))
        
        # Total reward
        total_reward = (progress_reward + efficiency_penalty + safety_penalty + 
                       success_bonus + collaboration_reward)
        
        reward_info = {
            "progress_reward": progress_reward,
            "efficiency_penalty": efficiency_penalty,
            "safety_penalty": safety_penalty,
            "success_bonus": success_bonus,
            "collaboration_reward": collaboration_reward,
            "distance_to_target": distance_to_target.item()
        }
        
        return total_reward, reward_info
    
    def _check_done(self) -> Tuple[bool, Dict[str, bool]]:
        """
        Check if episode should terminate.
        
        Returns:
            Tuple of (done, termination_reasons)
        """
        robot = self.current_state.robot
        
        # Success: reached target
        distance_to_target = torch.norm(robot.end_effector_pose[:3] - self.current_state.target_position)
        success = distance_to_target < self.target_reached_threshold
        
        # Failure: collision with obstacle
        collision = False
        for obstacle in self.current_state.obstacles:
            if torch.norm(robot.end_effector_pose[:3] - obstacle) < self.safety_distance:
                collision = True
                break
        
        # Failure: exceeded force limits
        force_violation = torch.max(torch.abs(robot.contact_forces)) > self.force_limit * 1.5
        
        # Time limit
        time_limit = self.step_count >= self.max_episode_steps
        
        done = success or collision or force_violation or time_limit
        
        done_info = {
            "success": success,
            "collision": collision, 
            "force_violation": force_violation,
            "time_limit": time_limit
        }
        
        return done, done_info
    
    def _would_violate_safety(self, action: torch.Tensor) -> bool:
        """
        Check if given action would violate safety constraints.
        
        Args:
            action: Proposed action
            
        Returns:
            True if action would violate safety
        """
        # Simulate one step ahead with proposed action
        current_pos = self.current_state.robot.end_effector_pose[:3]
        
        # Simple prediction of next position
        predicted_velocity = action[:3] * self.dt  # Simplified dynamics
        predicted_pos = current_pos + predicted_velocity
        
        # Check collision with obstacles
        for obstacle in self.current_state.obstacles:
            if torch.norm(predicted_pos - obstacle) < self.safety_distance:
                return True
        
        # Check force limits
        if torch.max(torch.abs(action)) > self.force_limit:
            return True
            
        return False
    
    def _compute_performance_metric(self) -> float:
        """
        Compute performance metric for adaptive assistance.
        
        Returns:
            Performance score (0=poor, 1=excellent)
        """
        robot = self.current_state.robot
        
        # Distance-based performance
        distance_to_target = torch.norm(robot.end_effector_pose[:3] - self.current_state.target_position)
        distance_performance = max(0.0, 1.0 - distance_to_target.item())
        
        # Smoothness of motion (low acceleration is good)
        motion_smoothness = 1.0 / (1.0 + torch.norm(robot.velocity).item())
        
        # Human confidence factor
        human_confidence = self.current_state.human_input.confidence
        
        # Combined performance metric
        performance = 0.5 * distance_performance + 0.3 * motion_smoothness + 0.2 * human_confidence
        
        return min(1.0, max(0.0, performance))
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ("human", "rgb_array", or "text")
            
        Returns:
            Rendered image array if mode="rgb_array", None otherwise
        """
        if mode == "text":
            robot = self.current_state.robot
            print(f"Step: {self.step_count}")
            print(f"Robot EE pos: {robot.end_effector_pose[:3].numpy()}")
            print(f"Target pos: {self.current_state.target_position.numpy()}")
            print(f"Distance to target: {torch.norm(robot.end_effector_pose[:3] - self.current_state.target_position):.3f}")
            print(f"Human confidence: {self.current_state.human_input.confidence:.3f}")
            print("---")
            
        elif mode in ["human", "rgb_array"]:
            # Placeholder for visual rendering
            # In a full implementation, this would use a graphics library
            logger.warning("Visual rendering not implemented. Use mode='text' for text output.")
            
        return None
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get statistics for the completed episode."""
        if not hasattr(self, 'episode_stats'):
            return {}
            
        return {
            "total_steps": self.step_count,
            "total_reward": self.episode_reward,
            "success_rate": 1.0 if self.step_count > 0 else 0.0,
            "average_human_confidence": getattr(self, '_avg_human_confidence', 0.5),
            "assistance_mode": self.assistance_mode.value
        }
    
    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("HumanRobotEnv closed")
        
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return [seed] if seed is not None else []