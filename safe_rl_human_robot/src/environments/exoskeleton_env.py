"""
7-DOF Arm Exoskeleton Shared Control Environment.

This module implements a realistic arm exoskeleton environment for rehabilitation
and assistance applications with comprehensive safety constraints and human modeling.
"""

import numpy as np
import torch
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .shared_control_base import (
    SharedControlBase, SafetyConstraint, SharedControlAction, 
    SharedControlState, AdvancedHumanModel
)
from .human_robot_env import AssistanceMode

logger = logging.getLogger(__name__)


@dataclass
class ExoskeletonConfig:
    """Configuration for exoskeleton environment."""
    # Physical parameters
    link_lengths: torch.Tensor = torch.tensor([0.3, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05])  # 7 links
    link_masses: torch.Tensor = torch.tensor([2.0, 1.8, 1.2, 0.8, 0.5, 0.3, 0.2])     # 7 masses (kg)
    joint_damping: torch.Tensor = torch.tensor([0.1, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02])  # Joint damping
    
    # Joint limits (rad)
    joint_limits: torch.Tensor = torch.tensor([
        [-2.8, 2.8],   # Shoulder adduction/abduction
        [-1.5, 1.5],   # Shoulder flexion/extension  
        [-2.5, 2.5],   # Shoulder internal/external rotation
        [0.0, 2.5],    # Elbow flexion (0 to 150 degrees)
        [-1.0, 1.0],   # Forearm pronation/supination
        [-1.5, 1.5],   # Wrist flexion/extension
        [-1.0, 1.0]    # Wrist radial/ulnar deviation
    ])
    
    # Torque limits (Nm)
    torque_limits: torch.Tensor = torch.tensor([25, 20, 15, 12, 8, 5, 3])
    
    # Safety parameters
    max_velocity: float = 2.0  # rad/s
    max_acceleration: float = 10.0  # rad/s^2
    collision_threshold: float = 0.05  # m
    force_threshold: float = 15.0  # N
    
    # Workspace bounds (m)
    workspace_center: torch.Tensor = torch.tensor([0.4, 0.0, 1.2])  # Relative to shoulder
    workspace_radius: float = 0.8
    
    # Task parameters
    target_tolerance: float = 0.02  # m
    time_limit: float = 30.0  # seconds


class ExoskeletonKinematics:
    """Forward and inverse kinematics for 7-DOF arm exoskeleton."""
    
    def __init__(self, config: ExoskeletonConfig):
        self.config = config
        self.link_lengths = config.link_lengths
        
        # DH parameters for 7-DOF arm (simplified anthropomorphic arm)
        # [theta, d, a, alpha] - standard DH convention
        self.dh_params = torch.tensor([
            [0, 0, 0, math.pi/2],           # Shoulder adduction/abduction
            [0, 0, 0, -math.pi/2],          # Shoulder flexion/extension
            [0, config.link_lengths[0], 0, math.pi/2],   # Shoulder rotation
            [0, 0, config.link_lengths[1], 0],           # Elbow flexion
            [0, 0, 0, -math.pi/2],          # Forearm pronation/supination
            [0, config.link_lengths[2], 0, math.pi/2],   # Wrist flexion/extension
            [0, 0, config.link_lengths[3], 0],           # Wrist radial/ulnar deviation
        ])
    
    def forward_kinematics(self, joint_angles: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute forward kinematics for 7-DOF arm.
        
        Args:
            joint_angles: Joint angles [7]
            
        Returns:
            Dictionary with end-effector pose and intermediate frames
        """
        batch_size = joint_angles.shape[0] if joint_angles.dim() > 1 else 1
        if joint_angles.dim() == 1:
            joint_angles = joint_angles.unsqueeze(0)
        
        # Initialize transformation matrix
        T = torch.eye(4, device=joint_angles.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Store all intermediate transformations
        link_transforms = []
        link_positions = []
        
        for i in range(7):
            # DH transformation matrix
            theta = joint_angles[:, i] + self.dh_params[i, 0]
            d = self.dh_params[i, 1]
            a = self.dh_params[i, 2] 
            alpha = self.dh_params[i, 3]
            
            # Individual transformation matrix
            c_theta = torch.cos(theta)
            s_theta = torch.sin(theta)
            c_alpha = torch.cos(torch.tensor(alpha))
            s_alpha = torch.sin(torch.tensor(alpha))
            
            T_i = torch.zeros(batch_size, 4, 4, device=joint_angles.device)
            T_i[:, 0, 0] = c_theta
            T_i[:, 0, 1] = -s_theta * c_alpha
            T_i[:, 0, 2] = s_theta * s_alpha
            T_i[:, 0, 3] = a * c_theta
            T_i[:, 1, 0] = s_theta
            T_i[:, 1, 1] = c_theta * c_alpha
            T_i[:, 1, 2] = -c_theta * s_alpha
            T_i[:, 1, 3] = a * s_theta
            T_i[:, 2, 1] = s_alpha
            T_i[:, 2, 2] = c_alpha
            T_i[:, 2, 3] = d
            T_i[:, 3, 3] = 1
            
            # Accumulate transformation
            T = torch.bmm(T, T_i)
            
            # Store intermediate results
            link_transforms.append(T.clone())
            link_positions.append(T[:, :3, 3].clone())
        
        # Extract end-effector pose
        end_effector_pos = T[:, :3, 3]
        end_effector_rot = T[:, :3, :3]
        
        # Convert rotation matrix to euler angles (ZYX convention)
        euler_angles = self._rotation_matrix_to_euler(end_effector_rot)
        
        # End-effector pose [x, y, z, rx, ry, rz]
        end_effector_pose = torch.cat([end_effector_pos, euler_angles], dim=-1)
        
        # Compute Jacobian
        jacobian = self._compute_jacobian(joint_angles, link_transforms)
        
        if batch_size == 1:
            end_effector_pose = end_effector_pose.squeeze(0)
            jacobian = jacobian.squeeze(0)
            link_positions = [pos.squeeze(0) for pos in link_positions]
        
        return {
            'end_effector_pose': end_effector_pose,
            'end_effector_position': end_effector_pos.squeeze(0) if batch_size == 1 else end_effector_pos,
            'end_effector_rotation': end_effector_rot.squeeze(0) if batch_size == 1 else end_effector_rot,
            'jacobian': jacobian,
            'link_positions': link_positions,
            'transformation_matrix': T.squeeze(0) if batch_size == 1 else T
        }
    
    def _rotation_matrix_to_euler(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to Euler angles (ZYX convention)."""
        batch_size = R.shape[0]
        euler = torch.zeros(batch_size, 3, device=R.device)
        
        # ZYX Euler angles
        euler[:, 0] = torch.atan2(R[:, 2, 1], R[:, 2, 2])  # Roll (X)
        euler[:, 1] = torch.atan2(-R[:, 2, 0], torch.sqrt(R[:, 2, 1]**2 + R[:, 2, 2]**2))  # Pitch (Y)
        euler[:, 2] = torch.atan2(R[:, 1, 0], R[:, 0, 0])  # Yaw (Z)
        
        return euler
    
    def _compute_jacobian(self, joint_angles: torch.Tensor, link_transforms: List[torch.Tensor]) -> torch.Tensor:
        """Compute analytical Jacobian matrix."""
        batch_size = joint_angles.shape[0] if joint_angles.dim() > 1 else 1
        jacobian = torch.zeros(batch_size, 6, 7, device=joint_angles.device)
        
        # End-effector position
        end_pos = link_transforms[-1][:, :3, 3]
        
        for i in range(7):
            if i == 0:
                # Base frame
                z_i = torch.tensor([0, 0, 1], device=joint_angles.device).unsqueeze(0).repeat(batch_size, 1)
                p_i = torch.zeros(batch_size, 3, device=joint_angles.device)
            else:
                # Extract z-axis and position from transformation matrix
                z_i = link_transforms[i-1][:, :3, 2]
                p_i = link_transforms[i-1][:, :3, 3]
            
            # Linear velocity component
            jacobian[:, :3, i] = torch.cross(z_i, end_pos - p_i, dim=1)
            
            # Angular velocity component
            jacobian[:, 3:6, i] = z_i
        
        return jacobian
    
    def inverse_kinematics(self, target_pose: torch.Tensor, 
                          initial_guess: Optional[torch.Tensor] = None,
                          max_iterations: int = 50,
                          tolerance: float = 1e-4) -> torch.Tensor:
        """
        Solve inverse kinematics using Newton-Raphson method.
        
        Args:
            target_pose: Target end-effector pose [6] (position + euler angles)
            initial_guess: Initial joint configuration [7]
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Joint angles [7] that achieve target pose (approximately)
        """
        if initial_guess is None:
            q = torch.zeros(7, device=target_pose.device)
        else:
            q = initial_guess.clone()
        
        for iteration in range(max_iterations):
            # Forward kinematics
            fk_result = self.forward_kinematics(q.unsqueeze(0))
            current_pose = fk_result['end_effector_pose']
            jacobian = fk_result['jacobian']
            
            # Pose error
            pose_error = target_pose - current_pose
            
            # Check convergence
            if torch.norm(pose_error) < tolerance:
                break
            
            # Newton-Raphson update
            try:
                # Use damped least squares for better numerical stability
                damping = 0.01
                JTJ_damped = torch.mm(jacobian.T, jacobian) + damping * torch.eye(7, device=q.device)
                delta_q = torch.linalg.solve(JTJ_damped, torch.mv(jacobian.T, pose_error))
                q = q + delta_q
                
                # Apply joint limits
                q = torch.clamp(q, self.config.joint_limits[:, 0], self.config.joint_limits[:, 1])
                
            except torch.linalg.LinAlgError:
                logger.warning("IK solver: Singular matrix encountered")
                break
        
        return q


class ExoskeletonEnvironment(SharedControlBase):
    """
    7-DOF arm exoskeleton shared control environment.
    
    Models a robotic arm exoskeleton for rehabilitation and assistance tasks
    with realistic biomechanics, safety constraints, and human interaction.
    """
    
    def __init__(self,
                 config: Optional[ExoskeletonConfig] = None,
                 task_type: str = "reach_target",
                 assistance_mode: AssistanceMode = AssistanceMode.COLLABORATIVE,
                 human_impairment_level: float = 0.0,
                 enable_emg: bool = False):
        """
        Initialize exoskeleton environment.
        
        Args:
            config: Exoskeleton configuration
            task_type: Type of task ("reach_target", "tracking", "adl")
            assistance_mode: Mode of robot assistance
            human_impairment_level: Level of human impairment (0=healthy, 1=severe)
            enable_emg: Whether to simulate EMG signals
        """
        self.config = config if config is not None else ExoskeletonConfig()
        self.task_type = task_type
        self.assistance_mode = assistance_mode
        self.human_impairment_level = human_impairment_level
        self.enable_emg = enable_emg
        
        # Initialize kinematics
        self.kinematics = ExoskeletonKinematics(self.config)
        
        # Human model with impairment
        human_skill = max(0.1, 1.0 - human_impairment_level)
        human_model = AdvancedHumanModel(
            skill_level=human_skill,
            tremor_amplitude=0.01 + 0.05 * human_impairment_level,
            fatigue_rate=0.001 * (1.0 + 2.0 * human_impairment_level)
        )
        
        # Initialize base class
        super().__init__(
            robot_dof=7,
            action_space_bounds=(-torch.max(self.config.torque_limits).item(), 
                               torch.max(self.config.torque_limits).item()),
            human_model=human_model,
            control_frequency=100.0,  # High frequency for exoskeleton
            max_episode_steps=int(self.config.time_limit * 100)  # 30 seconds at 100Hz
        )
        
        # Setup observation space
        obs_dim = (7 +      # joint positions
                  7 +      # joint velocities  
                  6 +      # end effector pose
                  3 +      # human intention
                  1 +      # human confidence
                  3 +      # target position
                  7 +      # joint torques
                  4)       # task info (distance, phase, etc.)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Task variables
        self.current_target: Optional[torch.Tensor] = None
        self.task_phase = 0
        self.task_start_time = 0.0
        
        self._setup_safety_constraints()
        self._setup_environment_specific()
        
        logger.info(f"ExoskeletonEnvironment initialized: task={task_type}, "
                   f"impairment={human_impairment_level}, assistance={assistance_mode.value}")
    
    def _setup_environment_specific(self) -> None:
        """Setup exoskeleton-specific parameters."""
        # Already handled in __init__
        pass
    
    def _setup_safety_constraints(self) -> None:
        """Setup safety constraints for exoskeleton."""
        
        # Joint position limits
        def joint_limit_constraint(state, action):
            positions = state.robot.position
            min_margin = torch.min(positions - self.config.joint_limits[:, 0])
            max_margin = torch.min(self.config.joint_limits[:, 1] - positions)
            return torch.min(min_margin, max_margin).item()
        
        self.add_safety_constraint(SafetyConstraint(
            name="joint_limits",
            constraint_function=joint_limit_constraint,
            threshold=0.1,  # 0.1 rad margin
            penalty_weight=100.0
        ))
        
        # Velocity limits
        def velocity_limit_constraint(state, action):
            velocities = state.robot.velocity
            max_vel = torch.max(torch.abs(velocities))
            return self.config.max_velocity - max_vel.item()
        
        self.add_safety_constraint(SafetyConstraint(
            name="velocity_limits",
            constraint_function=velocity_limit_constraint,
            threshold=0.0,
            penalty_weight=50.0
        ))
        
        # Torque limits  
        def torque_limit_constraint(state, action):
            torque_margins = self.config.torque_limits - torch.abs(action)
            return torch.min(torque_margins).item()
        
        self.add_safety_constraint(SafetyConstraint(
            name="torque_limits",
            constraint_function=torque_limit_constraint,
            threshold=0.0,
            penalty_weight=75.0
        ))
        
        # Workspace limits
        def workspace_constraint(state, action):
            ee_pos = state.robot.end_effector_pose[:3]
            distance_from_center = torch.norm(ee_pos - self.config.workspace_center)
            return self.config.workspace_radius - distance_from_center.item()
        
        self.add_safety_constraint(SafetyConstraint(
            name="workspace_limits", 
            constraint_function=workspace_constraint,
            threshold=0.05,  # 5cm margin
            penalty_weight=200.0
        ))
        
        # Singularity avoidance
        def singularity_constraint(state, action):
            # Compute manipulability index
            kinematics = self._compute_kinematics(state.robot.position.unsqueeze(0))
            jacobian = kinematics['jacobian'].squeeze(0)
            manipulability = torch.sqrt(torch.det(torch.mm(jacobian, jacobian.T)))
            return manipulability.item() - 0.01  # Minimum manipulability
        
        self.add_safety_constraint(SafetyConstraint(
            name="singularity_avoidance",
            constraint_function=singularity_constraint,
            threshold=0.0,
            penalty_weight=30.0
        ))
    
    def _compute_kinematics(self, joint_positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute forward kinematics using exoskeleton kinematics."""
        return self.kinematics.forward_kinematics(joint_positions)
    
    def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize environment-specific components."""
        # Generate initial target based on task type
        if self.task_type == "reach_target":
            target = self._generate_reach_target()
        elif self.task_type == "tracking":
            target = self._generate_tracking_target(0)
        else:  # ADL tasks
            target = self._generate_adl_target()
        
        self.current_target = target
        self.task_start_time = 0.0
        self.task_phase = 0
        
        return {
            'target_position': target,
            'task_type': self.task_type,
            'assistance_mode': self.assistance_mode,
            'obstacles': [],  # No obstacles in basic exoskeleton task
            'task_phase': self.task_phase,
            'emg_signals': torch.zeros(8) if self.enable_emg else None
        }
    
    def _generate_reach_target(self) -> torch.Tensor:
        """Generate random reachable target within workspace."""
        # Random point within workspace sphere
        radius = torch.rand(1) * (self.config.workspace_radius - 0.1)
        theta = torch.rand(1) * 2 * math.pi
        phi = torch.acos(2 * torch.rand(1) - 1)  # Uniform on sphere
        
        target = self.config.workspace_center + radius * torch.tensor([
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta), 
            torch.cos(phi)
        ]).squeeze()
        
        return target
    
    def _generate_tracking_target(self, time_step: int) -> torch.Tensor:
        """Generate tracking target for current time step."""
        t = time_step * self.dt
        
        # Figure-8 trajectory
        center = self.config.workspace_center
        radius = 0.2
        
        x = center[0] + radius * torch.sin(2 * t)
        y = center[1] + radius * torch.sin(t)
        z = center[2] + 0.1 * torch.sin(3 * t)
        
        return torch.tensor([x, y, z])
    
    def _generate_adl_target(self) -> torch.Tensor:
        """Generate target for Activities of Daily Living task."""
        # Simple reaching to mouth position (feeding motion)
        shoulder_pos = torch.tensor([0.0, 0.0, 1.4])  # Approximate shoulder height
        mouth_offset = torch.tensor([0.0, -0.15, 0.2])  # Relative to shoulder
        
        return shoulder_pos + mouth_offset
    
    def _update_environment_state(self, action: SharedControlAction) -> None:
        """Update environment-specific state."""
        # Update target for tracking tasks
        if self.task_type == "tracking":
            self.current_target = self._generate_tracking_target(self.step_count)
            self.current_state.environment_info['target_position'] = self.current_target
        
        # Update task phase
        distance_to_target = torch.norm(
            self.current_state.robot_state.end_effector_pose[:3] - self.current_target
        )
        
        if distance_to_target < self.config.target_tolerance:
            if self.task_phase == 0:
                self.task_phase = 1  # Reached target
            elif self.task_type == "reach_target" and self.task_phase == 1:
                # Generate new target
                self.current_target = self._generate_reach_target()
                self.current_state.environment_info['target_position'] = self.current_target
                self.task_phase = 0
        
        self.current_state.environment_info['task_phase'] = self.task_phase
        
        # Simulate EMG signals if enabled
        if self.enable_emg:
            emg = self._simulate_emg_signals(action)
            self.current_state.environment_info['emg_signals'] = emg
    
    def _simulate_emg_signals(self, action: SharedControlAction) -> torch.Tensor:
        """Simulate EMG signals based on human intention and robot assistance."""
        # 8 muscle groups for arm movement
        emg = torch.zeros(8)
        
        # Base EMG from human intention
        human_intention = self.current_state.human_input.intention
        intention_magnitude = torch.norm(human_intention)
        
        # Simulate different muscle activations
        # Simplified model: muscle activation correlates with intended movement
        emg[0] = max(0, human_intention[0]) * 0.8  # Anterior deltoid (forward)
        emg[1] = max(0, -human_intention[0]) * 0.8  # Posterior deltoid (backward)
        emg[2] = max(0, human_intention[1]) * 0.7  # Medial deltoid (up)
        emg[3] = max(0, -human_intention[1]) * 0.7  # Latissimus dorsi (down)
        emg[4] = max(0, human_intention[2]) * 0.6  # Biceps (elbow flex)
        emg[5] = max(0, -human_intention[2]) * 0.6  # Triceps (elbow extend)
        emg[6] = intention_magnitude * 0.3  # Forearm flexors
        emg[7] = intention_magnitude * 0.3  # Forearm extensors
        
        # Reduce EMG when robot provides assistance
        assistance_factor = 1.0 - 0.5 * action.assistance_level
        emg *= assistance_factor
        
        # Add impairment effects
        if self.human_impairment_level > 0:
            # Reduced maximum activation
            max_reduction = 0.8 * self.human_impairment_level
            emg *= (1.0 - max_reduction)
            
            # Add co-contraction (antagonist activation)
            cocontraction = 0.2 * self.human_impairment_level * torch.rand(8)
            emg += cocontraction
        
        # Add noise and clamp
        emg += torch.normal(0, 0.05, size=(8,))
        emg = torch.clamp(emg, 0, 1)
        
        return emg
    
    def _get_environment_observation(self) -> torch.Tensor:
        """Get exoskeleton-specific observations."""
        # Target information
        target_obs = self.current_target
        
        # Task information
        distance_to_target = torch.norm(
            self.current_state.robot_state.end_effector_pose[:3] - self.current_target
        )
        
        task_progress = max(0, 1.0 - distance_to_target / 0.5)  # Progress metric
        
        task_info = torch.tensor([
            distance_to_target,
            task_progress,
            self.task_phase,
            self.step_count * self.dt  # Time elapsed
        ])
        
        # Joint torques
        joint_torques = self.current_state.robot_state.torque
        
        return torch.cat([target_obs, joint_torques, task_info])
    
    def _compute_environment_reward(self, action: SharedControlAction) -> Tuple[float, Dict[str, float]]:
        """Compute exoskeleton-specific reward."""
        robot_state = self.current_state.robot_state
        
        # Task completion reward
        distance_to_target = torch.norm(robot_state.end_effector_pose[:3] - self.current_target)
        
        if self.task_type == "reach_target":
            # Sparse reward for reaching
            task_reward = 100.0 if distance_to_target < self.config.target_tolerance else -distance_to_target
        elif self.task_type == "tracking":
            # Dense reward for tracking
            task_reward = -10.0 * distance_to_target  # Higher penalty for tracking error
        else:  # ADL tasks
            task_reward = 50.0 if distance_to_target < self.config.target_tolerance else -0.5 * distance_to_target
        
        # Smoothness reward (penalize large accelerations)
        if hasattr(self, 'prev_velocity'):
            acceleration = (robot_state.velocity - self.prev_velocity) / self.dt
            smoothness_penalty = -0.1 * torch.norm(acceleration).item()
        else:
            smoothness_penalty = 0.0
        self.prev_velocity = robot_state.velocity.clone()
        
        # Energy efficiency (penalize large torques)
        energy_penalty = -0.01 * torch.norm(robot_state.torque).item()
        
        # Human effort consideration
        human_confidence = self.current_state.human_input.confidence
        collaboration_reward = 0.1 * human_confidence  # Encourage human involvement
        
        # Impairment compensation reward
        if self.human_impairment_level > 0:
            # Reward appropriate assistance for impaired users
            assistance_appropriateness = 1.0 - abs(action.assistance_level - self.human_impairment_level)
            impairment_reward = 0.2 * assistance_appropriateness
        else:
            impairment_reward = 0.0
        
        # EMG-based reward (if enabled)
        emg_reward = 0.0
        if self.enable_emg and 'emg_signals' in self.current_state.environment_info:
            # Encourage efficient muscle activation
            emg_signals = self.current_state.environment_info['emg_signals']
            total_activation = torch.sum(emg_signals)
            emg_reward = -0.05 * total_activation  # Penalize excessive activation
        
        total_reward = (task_reward + smoothness_penalty + energy_penalty + 
                       collaboration_reward + impairment_reward + emg_reward)
        
        reward_info = {
            'task_reward': task_reward,
            'smoothness_penalty': smoothness_penalty,
            'energy_penalty': energy_penalty,
            'collaboration_reward': collaboration_reward,
            'impairment_reward': impairment_reward,
            'emg_reward': emg_reward,
            'distance_to_target': distance_to_target.item()
        }
        
        return total_reward, reward_info
    
    def _check_environment_termination(self) -> Tuple[bool, Dict[str, bool]]:
        """Check exoskeleton-specific termination conditions."""
        robot_state = self.current_state.robot_state
        
        # Task completion
        distance_to_target = torch.norm(robot_state.end_effector_pose[:3] - self.current_target)
        task_complete = False
        
        if self.task_type == "reach_target":
            task_complete = distance_to_target < self.config.target_tolerance
        elif self.task_type == "adl":
            task_complete = (distance_to_target < self.config.target_tolerance and 
                           self.step_count * self.dt > 2.0)  # Hold for 2 seconds
        # Tracking tasks don't have completion condition
        
        # Joint limits violation (hard stop)
        joint_violation = torch.any(
            (robot_state.position < self.config.joint_limits[:, 0] - 0.05) |
            (robot_state.position > self.config.joint_limits[:, 1] + 0.05)
        ).item()
        
        # Excessive forces
        force_violation = torch.max(torch.abs(robot_state.contact_forces)) > self.config.force_threshold
        
        done = task_complete or joint_violation or force_violation
        
        done_info = {
            'task_complete': task_complete,
            'joint_violation': joint_violation,
            'force_violation': force_violation
        }
        
        return done, done_info
    
    def get_task_metrics(self) -> Dict[str, Any]:
        """Get task-specific performance metrics."""
        if not self.current_state:
            return {}
        
        distance = torch.norm(
            self.current_state.robot_state.end_effector_pose[:3] - self.current_target
        )
        
        metrics = {
            'distance_to_target': distance.item(),
            'task_phase': self.task_phase,
            'task_completion_rate': 1.0 if distance < self.config.target_tolerance else 0.0,
            'human_impairment_level': self.human_impairment_level,
            'assistance_effectiveness': self.current_state.human_input.confidence
        }
        
        if self.enable_emg and 'emg_signals' in self.current_state.environment_info:
            emg_signals = self.current_state.environment_info['emg_signals']
            metrics['total_muscle_activation'] = torch.sum(emg_signals).item()
            metrics['muscle_cocontraction'] = self._compute_cocontraction(emg_signals)
        
        return metrics
    
    def _compute_cocontraction(self, emg_signals: torch.Tensor) -> float:
        """Compute muscle co-contraction index."""
        # Pairs of antagonist muscles
        antagonist_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        
        cocontraction = 0.0
        for i, j in antagonist_pairs:
            cocontraction += min(emg_signals[i], emg_signals[j])
        
        return cocontraction / len(antagonist_pairs)
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render exoskeleton environment."""
        if mode == "text":
            super().render(mode)
            if self.current_state:
                print(f"Task: {self.task_type}, Phase: {self.task_phase}")
                print(f"Target: {self.current_target.numpy()}")
                print(f"Joint angles: {self.current_state.robot_state.position.numpy()}")
                print(f"Impairment level: {self.human_impairment_level}")
                if self.enable_emg and 'emg_signals' in self.current_state.environment_info:
                    emg = self.current_state.environment_info['emg_signals']
                    print(f"EMG signals: {emg.numpy()}")
        else:
            logger.warning(f"Visual rendering for exoskeleton not implemented")
        
        return None