"""
Wheelchair Navigation Shared Control Environment.

This module implements a realistic wheelchair navigation environment for mobility
assistance with obstacle avoidance, path planning, and human joystick input modeling.
"""

import numpy as np
import torch
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

from .shared_control_base import (
    SharedControlBase, SafetyConstraint, SharedControlAction, 
    SharedControlState, AdvancedHumanModel
)
from .human_robot_env import AssistanceMode

logger = logging.getLogger(__name__)


@dataclass
class WheelchairConfig:
    """Configuration for wheelchair environment."""
    # Physical parameters
    wheelbase: float = 0.6  # Distance between front and rear axles (m)
    track_width: float = 0.7  # Distance between left and right wheels (m)
    wheel_radius: float = 0.3  # Wheel radius (m)
    max_linear_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 1.5  # rad/s
    max_linear_acceleration: float = 1.0  # m/s^2
    max_angular_acceleration: float = 2.0  # rad/s^2
    
    # Safety parameters
    collision_radius: float = 0.5  # Safety radius around wheelchair (m)
    min_obstacle_distance: float = 0.3  # Minimum distance to obstacles (m)
    max_slope: float = 0.1  # Maximum allowed slope (10 degrees ≈ 0.176 rad)
    
    # Environment parameters
    world_size: Tuple[float, float] = (10.0, 10.0)  # World dimensions (m)
    goal_tolerance: float = 0.2  # Goal reaching tolerance (m)
    path_following_tolerance: float = 0.5  # Path following tolerance (m)
    
    # Joystick parameters
    joystick_max_deflection: float = 1.0  # Maximum joystick deflection
    joystick_deadzone: float = 0.05  # Deadzone radius
    joystick_sensitivity: float = 1.0  # Sensitivity multiplier


@dataclass
class Obstacle:
    """Obstacle representation."""
    position: torch.Tensor  # [x, y] position
    radius: float  # Obstacle radius
    is_static: bool = True  # Whether obstacle moves
    velocity: Optional[torch.Tensor] = None  # [vx, vy] if dynamic


class JoystickModel:
    """Models human joystick input with realistic characteristics."""
    
    def __init__(self, 
                 skill_level: float = 0.8,
                 tremor_frequency: float = 8.0,
                 tremor_amplitude: float = 0.02,
                 fatigue_buildup: float = 0.001,
                 reaction_delay: float = 0.2):
        """
        Initialize joystick model.
        
        Args:
            skill_level: User skill level (0-1)
            tremor_frequency: Tremor frequency (Hz)
            tremor_amplitude: Tremor amplitude
            fatigue_buildup: Rate of fatigue accumulation
            reaction_delay: Reaction delay (seconds)
        """
        self.skill_level = skill_level
        self.tremor_frequency = tremor_frequency
        self.tremor_amplitude = tremor_amplitude
        self.fatigue_buildup = fatigue_buildup
        self.reaction_delay = reaction_delay
        
        # State variables
        self.fatigue_level = 0.0
        self.time_step = 0
        self.last_command = torch.zeros(2)
        
        # Input filtering
        self.input_filter_alpha = 0.3  # Low-pass filter coefficient
        self.filtered_input = torch.zeros(2)
    
    def get_joystick_input(self, intended_direction: torch.Tensor,
                          timestamp: float) -> torch.Tensor:
        """
        Generate realistic joystick input based on intended direction.
        
        Args:
            intended_direction: Intended movement direction [x, y]
            timestamp: Current timestamp
            
        Returns:
            Joystick input [x, y] in range [-1, 1]
        """
        # Apply skill level (accuracy)
        effective_skill = max(0.1, self.skill_level - 0.2 * self.fatigue_level)
        skilled_input = intended_direction * effective_skill
        
        # Add tremor
        tremor_x = self.tremor_amplitude * math.sin(2 * math.pi * self.tremor_frequency * timestamp)
        tremor_y = self.tremor_amplitude * math.cos(2 * math.pi * self.tremor_frequency * timestamp * 1.2)
        tremor = torch.tensor([tremor_x, tremor_y])
        
        # Add random noise
        noise = torch.normal(0, 0.02, size=(2,))
        
        # Combine effects
        raw_input = skilled_input + tremor + noise
        
        # Apply deadzone
        input_magnitude = torch.norm(raw_input)
        if input_magnitude < 0.05:  # Deadzone
            raw_input = torch.zeros(2)
        
        # Low-pass filter for smoothness
        self.filtered_input = (self.input_filter_alpha * raw_input + 
                              (1 - self.input_filter_alpha) * self.filtered_input)
        
        # Clamp to valid range
        joystick_input = torch.clamp(self.filtered_input, -1.0, 1.0)
        
        # Update state
        self.fatigue_level += self.fatigue_buildup
        self.time_step += 1
        self.last_command = joystick_input.clone()
        
        return joystick_input


class PathPlanner:
    """Simple path planner for wheelchair navigation."""
    
    def __init__(self, world_size: Tuple[float, float]):
        self.world_size = world_size
        self.waypoint_spacing = 0.5  # Distance between waypoints
    
    def plan_path(self, start: torch.Tensor, goal: torch.Tensor, 
                 obstacles: List[Obstacle]) -> List[torch.Tensor]:
        """
        Plan path from start to goal avoiding obstacles.
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacles: List of obstacles
            
        Returns:
            List of waypoints including start and goal
        """
        # Simple straight-line path with obstacle avoidance
        path = [start]
        
        direction = goal - start
        distance = torch.norm(direction)
        
        if distance < 1e-6:
            return [start]
        
        unit_direction = direction / distance
        
        # Check for obstacles along direct path
        num_waypoints = int(distance / self.waypoint_spacing) + 1
        for i in range(1, num_waypoints):
            waypoint = start + (i / num_waypoints) * direction
            
            # Check collision with obstacles
            collision = False
            for obstacle in obstacles:
                if torch.norm(waypoint - obstacle.position) < obstacle.radius + 0.3:
                    # Simple avoidance: offset perpendicular to path
                    perpendicular = torch.tensor([-unit_direction[1], unit_direction[0]])
                    offset_distance = obstacle.radius + 0.5
                    
                    # Choose side based on obstacle position
                    to_obstacle = obstacle.position - waypoint
                    if torch.dot(to_obstacle, perpendicular) > 0:
                        waypoint = waypoint + offset_distance * perpendicular
                    else:
                        waypoint = waypoint - offset_distance * perpendicular
                    
                    collision = True
                    break
            
            path.append(waypoint)
        
        path.append(goal)
        return path
    
    def get_next_waypoint(self, current_pos: torch.Tensor, 
                         path: List[torch.Tensor], 
                         lookahead_distance: float = 1.0) -> torch.Tensor:
        """Get next waypoint for path following."""
        if not path:
            return current_pos
        
        # Find closest point on path
        min_distance = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(path):
            distance = torch.norm(waypoint - current_pos)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # Look ahead from closest point
        for i in range(closest_idx, len(path)):
            distance = torch.norm(path[i] - current_pos)
            if distance >= lookahead_distance:
                return path[i]
        
        # Return last waypoint if no lookahead point found
        return path[-1]


class WheelchairEnvironment(SharedControlBase):
    """
    Wheelchair navigation shared control environment.
    
    Models a smart wheelchair with human joystick input and robot assistance
    for navigation tasks with obstacle avoidance and path following.
    """
    
    def __init__(self,
                 config: Optional[WheelchairConfig] = None,
                 task_type: str = "navigation",
                 assistance_mode: AssistanceMode = AssistanceMode.COLLABORATIVE,
                 mobility_impairment_level: float = 0.0,
                 environment_complexity: str = "simple"):
        """
        Initialize wheelchair environment.
        
        Args:
            config: Wheelchair configuration
            task_type: Type of task ("navigation", "path_following", "obstacle_course")
            assistance_mode: Mode of robot assistance
            mobility_impairment_level: Level of mobility impairment (0=healthy, 1=severe)
            environment_complexity: Environment complexity ("simple", "moderate", "complex")
        """
        self.config = config if config is not None else WheelchairConfig()
        self.task_type = task_type
        self.assistance_mode = assistance_mode
        self.mobility_impairment_level = mobility_impairment_level
        self.environment_complexity = environment_complexity
        
        # Wheelchair state
        self.position = torch.zeros(2)  # [x, y]
        self.orientation = torch.tensor(0.0)  # heading angle
        self.linear_velocity = torch.zeros(2)  # [vx, vy]
        self.angular_velocity = torch.tensor(0.0)  # angular velocity
        
        # Joystick model
        joystick_skill = max(0.2, 1.0 - mobility_impairment_level)
        self.joystick = JoystickModel(
            skill_level=joystick_skill,
            tremor_amplitude=0.01 + 0.05 * mobility_impairment_level,
            fatigue_buildup=0.001 * (1.0 + 3.0 * mobility_impairment_level)
        )
        
        # Human model adapted for wheelchair use
        human_model = AdvancedHumanModel(
            skill_level=joystick_skill,
            tremor_amplitude=0.01 + 0.03 * mobility_impairment_level,
            reaction_delay=0.1 + 0.2 * mobility_impairment_level
        )
        
        # Initialize base class (2 DOF: linear and angular velocity)
        super().__init__(
            robot_dof=2,
            action_space_bounds=(-1.0, 1.0),  # Normalized velocity commands
            human_model=human_model,
            control_frequency=20.0,  # 20Hz for wheelchair control
            max_episode_steps=int(30.0 * 20)  # 30 seconds
        )
        
        # Setup observation space
        obs_dim = (2 +      # position [x, y]
                  1 +      # orientation
                  2 +      # linear velocity [vx, vy]
                  1 +      # angular velocity
                  2 +      # joystick input [x, y]
                  1 +      # human confidence
                  2 +      # goal position [x, y]
                  2 +      # next waypoint [x, y]
                  4 +      # nearest obstacles info (dist, angle to 2 closest)
                  4)       # path info (progress, deviation, etc.)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Environment components
        self.obstacles: List[Obstacle] = []
        self.goal_position: Optional[torch.Tensor] = None
        self.planned_path: List[torch.Tensor] = []
        self.path_planner = PathPlanner(self.config.world_size)
        
        # Task state
        self.current_waypoint_idx = 0
        self.total_distance_traveled = 0.0
        self.path_deviations = []
        
        self._setup_safety_constraints()
        self._setup_environment_specific()
        
        logger.info(f"WheelchairEnvironment initialized: task={task_type}, "
                   f"impairment={mobility_impairment_level}, complexity={environment_complexity}")
    
    def _setup_environment_specific(self) -> None:
        """Setup wheelchair-specific parameters."""
        # Generate obstacles based on complexity
        self._generate_obstacles()
    
    def _generate_obstacles(self) -> None:
        """Generate obstacles based on environment complexity."""
        self.obstacles.clear()
        
        world_x, world_y = self.config.world_size
        
        if self.environment_complexity == "simple":
            # Few static obstacles
            num_obstacles = 3
            for _ in range(num_obstacles):
                pos = torch.tensor([
                    torch.rand(1) * (world_x - 2) + 1,
                    torch.rand(1) * (world_y - 2) + 1
                ])
                self.obstacles.append(Obstacle(
                    position=pos,
                    radius=0.3 + torch.rand(1) * 0.2,
                    is_static=True
                ))
        
        elif self.environment_complexity == "moderate":
            # More obstacles, some narrow passages
            num_obstacles = 8
            for _ in range(num_obstacles):
                pos = torch.tensor([
                    torch.rand(1) * (world_x - 2) + 1,
                    torch.rand(1) * (world_y - 2) + 1
                ])
                self.obstacles.append(Obstacle(
                    position=pos,
                    radius=0.2 + torch.rand(1) * 0.3,
                    is_static=True
                ))
        
        elif self.environment_complexity == "complex":
            # Many obstacles, some dynamic
            num_static = 12
            num_dynamic = 3
            
            # Static obstacles
            for _ in range(num_static):
                pos = torch.tensor([
                    torch.rand(1) * (world_x - 2) + 1,
                    torch.rand(1) * (world_y - 2) + 1
                ])
                self.obstacles.append(Obstacle(
                    position=pos,
                    radius=0.15 + torch.rand(1) * 0.35,
                    is_static=True
                ))
            
            # Dynamic obstacles
            for _ in range(num_dynamic):
                pos = torch.tensor([
                    torch.rand(1) * (world_x - 2) + 1,
                    torch.rand(1) * (world_y - 2) + 1
                ])
                velocity = torch.normal(0, 0.3, size=(2,))
                self.obstacles.append(Obstacle(
                    position=pos,
                    radius=0.2 + torch.rand(1) * 0.2,
                    is_static=False,
                    velocity=velocity
                ))
    
    def _setup_safety_constraints(self) -> None:
        """Setup safety constraints for wheelchair."""
        
        # Collision avoidance constraint
        def collision_constraint(state, action):
            min_distance = float('inf')
            wheelchair_pos = self.position
            
            for obstacle in self.obstacles:
                distance = torch.norm(wheelchair_pos - obstacle.position)
                safety_distance = obstacle.radius + self.config.collision_radius
                min_distance = min(min_distance, distance - safety_distance)
            
            return min_distance
        
        self.add_safety_constraint(SafetyConstraint(
            name="collision_avoidance",
            constraint_function=collision_constraint,
            threshold=0.0,
            penalty_weight=1000.0
        ))
        
        # Velocity limits
        def velocity_constraint(state, action):
            linear_vel_magnitude = torch.norm(self.linear_velocity)
            angular_vel_magnitude = abs(self.angular_velocity)
            
            linear_margin = self.config.max_linear_velocity - linear_vel_magnitude
            angular_margin = self.config.max_angular_velocity - angular_vel_magnitude
            
            return min(linear_margin, angular_margin)
        
        self.add_safety_constraint(SafetyConstraint(
            name="velocity_limits",
            constraint_function=velocity_constraint,
            threshold=0.0,
            penalty_weight=50.0
        ))
        
        # Boundary constraint (keep within world)
        def boundary_constraint(state, action):
            world_x, world_y = self.config.world_size
            
            margin_x = min(self.position[0] - 0.5, world_x - self.position[0] - 0.5)
            margin_y = min(self.position[1] - 0.5, world_y - self.position[1] - 0.5)
            
            return min(margin_x, margin_y)
        
        self.add_safety_constraint(SafetyConstraint(
            name="world_boundaries",
            constraint_function=boundary_constraint,
            threshold=0.0,
            penalty_weight=200.0
        ))
    
    def _compute_kinematics(self, joint_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute wheelchair kinematics (position, orientation)."""
        # For wheelchair, joint_positions represents [linear_cmd, angular_cmd]
        # But we need to return wheelchair pose
        return {
            'end_effector_pose': torch.cat([self.position, torch.tensor([self.orientation, 0, 0, 0])]),
            'position': self.position,
            'orientation': self.orientation
        }
    
    def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize environment-specific components."""
        # Initialize wheelchair position
        world_x, world_y = self.config.world_size
        
        # Start position (avoid obstacles)
        max_attempts = 50
        for _ in range(max_attempts):
            start_pos = torch.tensor([
                torch.rand(1) * (world_x - 2) + 1,
                torch.rand(1) * (world_y - 2) + 1
            ])
            
            # Check if start position is collision-free
            collision_free = True
            for obstacle in self.obstacles:
                if torch.norm(start_pos - obstacle.position) < obstacle.radius + self.config.collision_radius:
                    collision_free = False
                    break
            
            if collision_free:
                break
        
        self.position = start_pos
        self.orientation = torch.rand(1) * 2 * math.pi  # Random initial orientation
        self.linear_velocity = torch.zeros(2)
        self.angular_velocity = torch.tensor(0.0)
        
        # Generate goal position
        for _ in range(max_attempts):
            goal_pos = torch.tensor([
                torch.rand(1) * (world_x - 2) + 1,
                torch.rand(1) * (world_y - 2) + 1
            ])
            
            # Ensure goal is far enough from start and collision-free
            if (torch.norm(goal_pos - self.position) > 2.0 and
                all(torch.norm(goal_pos - obs.position) > obs.radius + 0.5 
                    for obs in self.obstacles)):
                break
        
        self.goal_position = goal_pos
        
        # Plan initial path
        self.planned_path = self.path_planner.plan_path(
            self.position, self.goal_position, self.obstacles
        )
        
        # Reset task state
        self.current_waypoint_idx = 0
        self.total_distance_traveled = 0.0
        self.path_deviations.clear()
        
        return {
            'goal_position': self.goal_position,
            'planned_path': self.planned_path,
            'obstacles': self.obstacles,
            'task_type': self.task_type,
            'assistance_mode': self.assistance_mode,
            'wheelchair_state': {
                'position': self.position,
                'orientation': self.orientation,
                'linear_velocity': self.linear_velocity,
                'angular_velocity': self.angular_velocity
            }
        }
    
    def _update_environment_state(self, action: SharedControlAction) -> None:
        """Update wheelchair dynamics and environment state."""
        # Get command velocities (normalized [-1, 1])
        linear_cmd, angular_cmd = action.robot_command
        
        # Convert to actual velocities
        target_linear_vel = linear_cmd * self.config.max_linear_velocity
        target_angular_vel = angular_cmd * self.config.max_angular_velocity
        
        # Apply acceleration limits (simple first-order dynamics)
        dt = self.dt
        max_linear_accel = self.config.max_linear_acceleration
        max_angular_accel = self.config.max_angular_acceleration
        
        # Current velocity in global frame
        current_linear_speed = torch.norm(self.linear_velocity)
        
        # Update linear velocity (in body frame, then transform to global)
        linear_vel_change = target_linear_vel - current_linear_speed
        linear_vel_change = torch.clamp(linear_vel_change, 
                                      -max_linear_accel * dt, max_linear_accel * dt)
        
        new_linear_speed = current_linear_speed + linear_vel_change
        
        # Direction of motion (current heading)
        direction = torch.tensor([torch.cos(self.orientation), torch.sin(self.orientation)])
        self.linear_velocity = new_linear_speed * direction
        
        # Update angular velocity
        angular_vel_change = target_angular_vel - self.angular_velocity
        angular_vel_change = torch.clamp(angular_vel_change,
                                       -max_angular_accel * dt, max_angular_accel * dt)
        self.angular_velocity += angular_vel_change
        
        # Integrate position and orientation
        self.position += self.linear_velocity * dt
        self.orientation += self.angular_velocity * dt
        
        # Normalize orientation
        self.orientation = (self.orientation + math.pi) % (2 * math.pi) - math.pi
        
        # Update dynamic obstacles
        for obstacle in self.obstacles:
            if not obstacle.is_static and obstacle.velocity is not None:
                obstacle.position += obstacle.velocity * dt
                
                # Bounce off walls
                world_x, world_y = self.config.world_size
                if obstacle.position[0] < obstacle.radius or obstacle.position[0] > world_x - obstacle.radius:
                    obstacle.velocity[0] *= -1
                if obstacle.position[1] < obstacle.radius or obstacle.position[1] > world_y - obstacle.radius:
                    obstacle.velocity[1] *= -1
        
        # Update path following metrics
        if self.planned_path:
            current_waypoint = self.path_planner.get_next_waypoint(
                self.position, self.planned_path
            )
            
            # Check if reached current waypoint
            if (self.current_waypoint_idx < len(self.planned_path) and
                torch.norm(self.position - self.planned_path[self.current_waypoint_idx]) < 
                self.config.path_following_tolerance):
                self.current_waypoint_idx += 1
        
        # Track distance traveled
        if hasattr(self, 'prev_position'):
            distance_increment = torch.norm(self.position - self.prev_position)
            self.total_distance_traveled += distance_increment.item()
        self.prev_position = self.position.clone()
        
        # Update environment info
        self.current_state.environment_info.update({
            'wheelchair_state': {
                'position': self.position,
                'orientation': self.orientation,
                'linear_velocity': self.linear_velocity,
                'angular_velocity': self.angular_velocity
            },
            'current_waypoint_idx': self.current_waypoint_idx,
            'total_distance_traveled': self.total_distance_traveled
        })
    
    def _map_human_to_robot_action(self, human_intention: torch.Tensor) -> torch.Tensor:
        """Map human intention to wheelchair control commands."""
        # Get joystick input from human intention
        joystick_input = self.joystick.get_joystick_input(
            human_intention[:2], self.step_count * self.dt
        )
        
        # Map joystick to wheelchair commands
        # Forward/backward from Y-axis, turning from X-axis
        linear_cmd = joystick_input[1]  # Forward/backward
        angular_cmd = -joystick_input[0]  # Left/right (negative for correct turning)
        
        return torch.tensor([linear_cmd, angular_cmd])
    
    def _get_environment_observation(self) -> torch.Tensor:
        """Get wheelchair-specific observations."""
        # Wheelchair state
        wheelchair_obs = torch.cat([
            self.position,
            torch.tensor([self.orientation]),
            self.linear_velocity,
            torch.tensor([self.angular_velocity.item()])
        ])
        
        # Joystick input
        joystick_obs = self.joystick.last_command
        
        # Goal and waypoint information
        if self.goal_position is not None:
            goal_obs = self.goal_position
            
            # Next waypoint
            if self.planned_path and self.current_waypoint_idx < len(self.planned_path):
                waypoint_obs = self.planned_path[self.current_waypoint_idx]
            else:
                waypoint_obs = self.goal_position
        else:
            goal_obs = torch.zeros(2)
            waypoint_obs = torch.zeros(2)
        
        # Obstacle information (2 nearest obstacles)
        obstacle_obs = torch.zeros(4)  # [dist1, angle1, dist2, angle2]
        if self.obstacles:
            # Calculate distances and angles to all obstacles
            obstacle_info = []
            for obstacle in self.obstacles:
                rel_pos = obstacle.position - self.position
                distance = torch.norm(rel_pos)
                angle = torch.atan2(rel_pos[1], rel_pos[0]) - self.orientation
                
                # Normalize angle to [-pi, pi]
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                
                obstacle_info.append((distance.item(), angle.item()))
            
            # Sort by distance and take 2 nearest
            obstacle_info.sort(key=lambda x: x[0])
            for i, (dist, angle) in enumerate(obstacle_info[:2]):
                obstacle_obs[i*2] = dist
                obstacle_obs[i*2 + 1] = angle
        
        # Path following information
        distance_to_goal = torch.norm(self.position - self.goal_position) if self.goal_position is not None else 0
        
        if self.planned_path:
            path_progress = self.current_waypoint_idx / len(self.planned_path)
            # Lateral deviation from path
            if self.current_waypoint_idx > 0 and self.current_waypoint_idx < len(self.planned_path):
                prev_waypoint = self.planned_path[self.current_waypoint_idx - 1]
                next_waypoint = self.planned_path[self.current_waypoint_idx]
                
                # Calculate lateral deviation using cross track error
                path_vector = next_waypoint - prev_waypoint
                to_wheelchair = self.position - prev_waypoint
                
                if torch.norm(path_vector) > 1e-6:
                    path_unit = path_vector / torch.norm(path_vector)
                    along_path = torch.dot(to_wheelchair, path_unit)
                    cross_track = torch.norm(to_wheelchair - along_path * path_unit)
                else:
                    cross_track = 0.0
            else:
                cross_track = 0.0
        else:
            path_progress = 0.0
            cross_track = 0.0
        
        path_obs = torch.tensor([
            distance_to_goal,
            path_progress,
            cross_track,
            self.total_distance_traveled
        ])
        
        return torch.cat([
            wheelchair_obs,    # 6D: pos, orient, vel
            joystick_obs,      # 2D: joystick input
            goal_obs,          # 2D: goal position  
            waypoint_obs,      # 2D: next waypoint
            obstacle_obs,      # 4D: nearest obstacles
            path_obs           # 4D: path info
        ])
    
    def _compute_environment_reward(self, action: SharedControlAction) -> Tuple[float, Dict[str, float]]:
        """Compute wheelchair-specific reward."""
        # Goal reaching reward
        if self.goal_position is not None:
            distance_to_goal = torch.norm(self.position - self.goal_position)
            
            if distance_to_goal < self.config.goal_tolerance:
                goal_reward = 1000.0  # Large reward for reaching goal
            else:
                goal_reward = -distance_to_goal.item()  # Negative distance
        else:
            goal_reward = 0.0
        
        # Path following reward
        path_reward = 0.0
        if self.planned_path and len(self.planned_path) > 1:
            # Reward for making progress along path
            progress_reward = 10.0 * (self.current_waypoint_idx / len(self.planned_path))
            
            # Penalty for deviating from path
            if self.current_waypoint_idx > 0 and self.current_waypoint_idx < len(self.planned_path):
                prev_waypoint = self.planned_path[self.current_waypoint_idx - 1]
                next_waypoint = self.planned_path[self.current_waypoint_idx]
                
                path_vector = next_waypoint - prev_waypoint
                to_wheelchair = self.position - prev_waypoint
                
                if torch.norm(path_vector) > 1e-6:
                    path_unit = path_vector / torch.norm(path_vector)
                    along_path = torch.dot(to_wheelchair, path_unit)
                    cross_track_error = torch.norm(to_wheelchair - along_path * path_unit)
                    deviation_penalty = -2.0 * cross_track_error.item()
                else:
                    deviation_penalty = 0.0
            else:
                deviation_penalty = 0.0
            
            path_reward = progress_reward + deviation_penalty
        
        # Efficiency reward (smooth control)
        linear_cmd, angular_cmd = action.robot_command
        control_effort = torch.norm(action.robot_command).item()
        efficiency_reward = -0.1 * control_effort
        
        # Comfort reward (smooth motion)
        if hasattr(self, 'prev_linear_velocity') and hasattr(self, 'prev_angular_velocity'):
            linear_accel = torch.norm(self.linear_velocity - self.prev_linear_velocity) / self.dt
            angular_accel = abs(self.angular_velocity - self.prev_angular_velocity) / self.dt
            
            comfort_penalty = -0.05 * (linear_accel + angular_accel)
        else:
            comfort_penalty = 0.0
        
        self.prev_linear_velocity = self.linear_velocity.clone()
        self.prev_angular_velocity = self.angular_velocity.clone()
        
        # Assistance appropriateness reward
        if self.mobility_impairment_level > 0:
            # More assistance should be provided for higher impairment
            assistance_appropriateness = 1.0 - abs(action.assistance_level - self.mobility_impairment_level)
            impairment_reward = 0.1 * assistance_appropriateness
        else:
            impairment_reward = 0.0
        
        # Human involvement reward
        human_confidence = self.current_state.human_input.confidence
        involvement_reward = 0.05 * human_confidence
        
        total_reward = (goal_reward + path_reward + efficiency_reward + 
                       comfort_penalty + impairment_reward + involvement_reward)
        
        reward_info = {
            'goal_reward': goal_reward,
            'path_reward': path_reward,
            'efficiency_reward': efficiency_reward,
            'comfort_penalty': comfort_penalty,
            'impairment_reward': impairment_reward,
            'involvement_reward': involvement_reward,
            'distance_to_goal': distance_to_goal.item() if self.goal_position is not None else 0
        }
        
        return total_reward, reward_info
    
    def _check_environment_termination(self) -> Tuple[bool, Dict[str, bool]]:
        """Check wheelchair-specific termination conditions."""
        # Goal reached
        goal_reached = False
        if self.goal_position is not None:
            distance_to_goal = torch.norm(self.position - self.goal_position)
            goal_reached = distance_to_goal < self.config.goal_tolerance
        
        # Collision with obstacles
        collision = False
        for obstacle in self.obstacles:
            if torch.norm(self.position - obstacle.position) < obstacle.radius + self.config.collision_radius:
                collision = True
                break
        
        # Out of bounds
        world_x, world_y = self.config.world_size
        out_of_bounds = (self.position[0] < 0 or self.position[0] > world_x or
                        self.position[1] < 0 or self.position[1] > world_y)
        
        done = goal_reached or collision or out_of_bounds
        
        done_info = {
            'goal_reached': goal_reached,
            'collision': collision,
            'out_of_bounds': out_of_bounds
        }
        
        return done, done_info
    
    def get_navigation_metrics(self) -> Dict[str, Any]:
        """Get navigation-specific performance metrics."""
        if not self.current_state or self.goal_position is None:
            return {}
        
        distance_to_goal = torch.norm(self.position - self.goal_position)
        
        # Path efficiency
        if self.planned_path:
            planned_distance = sum(
                torch.norm(self.planned_path[i+1] - self.planned_path[i])
                for i in range(len(self.planned_path) - 1)
            ).item()
            path_efficiency = planned_distance / max(self.total_distance_traveled, 1e-6)
        else:
            path_efficiency = 0.0
        
        # Success rate
        success = distance_to_goal < self.config.goal_tolerance
        
        metrics = {
            'distance_to_goal': distance_to_goal.item(),
            'total_distance_traveled': self.total_distance_traveled,
            'path_efficiency': path_efficiency,
            'current_waypoint_progress': self.current_waypoint_idx / len(self.planned_path) if self.planned_path else 0,
            'navigation_success': success,
            'mobility_impairment_level': self.mobility_impairment_level,
            'joystick_skill_level': self.joystick.skill_level,
            'fatigue_level': self.joystick.fatigue_level
        }
        
        return metrics
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render wheelchair environment."""
        if mode == "text":
            super().render(mode)
            print(f"Task: {self.task_type}")
            print(f"Position: ({self.position[0]:.2f}, {self.position[1]:.2f})")
            print(f"Orientation: {self.orientation:.2f} rad")
            print(f"Linear velocity: {torch.norm(self.linear_velocity):.2f} m/s")
            print(f"Angular velocity: {self.angular_velocity:.2f} rad/s")
            if self.goal_position is not None:
                distance = torch.norm(self.position - self.goal_position)
                print(f"Distance to goal: {distance:.2f} m")
            print(f"Impairment level: {self.mobility_impairment_level}")
            print(f"Fatigue level: {self.joystick.fatigue_level:.3f}")
        else:
            logger.warning(f"Visual rendering for wheelchair not implemented")
        
        return None