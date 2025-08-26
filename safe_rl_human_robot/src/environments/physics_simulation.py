"""
Physics Simulation with PyBullet Integration.

This module provides physics simulation capabilities using PyBullet for realistic
dynamics, collision detection, and force modeling in shared control environments.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import os
import math

from .shared_control_base import PhysicsEngine

logger = logging.getLogger(__name__)

# Optional PyBullet import - graceful fallback if not available
try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    logger.warning("PyBullet not available. Physics simulation will use simplified dynamics.")


@dataclass
class RobotDescription:
    """Robot description for physics simulation."""
    urdf_path: str
    base_position: List[float] = None
    base_orientation: List[float] = None
    joint_names: List[str] = None
    joint_limits: List[Tuple[float, float]] = None
    joint_damping: List[float] = None
    joint_friction: List[float] = None


class PyBulletPhysicsEngine(PhysicsEngine):
    """
    PyBullet-based physics engine for realistic simulation.
    
    Provides high-fidelity physics simulation with proper dynamics,
    collision detection, and contact forces.
    """
    
    def __init__(self, 
                 gui_enabled: bool = False,
                 time_step: float = 1.0/240.0,
                 gravity: Tuple[float, float, float] = (0, 0, -9.81),
                 solver_iterations: int = 50):
        """
        Initialize PyBullet physics engine.
        
        Args:
            gui_enabled: Whether to show PyBullet GUI
            time_step: Physics simulation time step
            gravity: Gravity vector [x, y, z]
            solver_iterations: Number of solver iterations
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError("PyBullet is required for PyBulletPhysicsEngine")
        
        self.gui_enabled = gui_enabled
        self.time_step = time_step
        self.gravity = gravity
        self.solver_iterations = solver_iterations
        
        # Initialize PyBullet
        if gui_enabled:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set physics parameters
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(*gravity, physicsClientId=self.physics_client)
        p.setTimeStep(time_step, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iterations,
            physicsClientId=self.physics_client
        )
        
        # Robot and environment objects
        self.robot_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.obstacle_ids: List[int] = []
        self.joint_indices: List[int] = []
        self.joint_names: List[str] = []
        
        # State tracking
        self.current_joint_positions = torch.zeros(0)
        self.current_joint_velocities = torch.zeros(0)
        self.current_joint_torques = torch.zeros(0)
        
        logger.info(f"PyBulletPhysicsEngine initialized: GUI={gui_enabled}, dt={time_step}")
    
    def load_robot(self, robot_description: RobotDescription) -> bool:
        """
        Load robot from URDF file.
        
        Args:
            robot_description: Robot description with URDF path and parameters
            
        Returns:
            True if robot loaded successfully
        """
        try:
            # Load robot URDF
            base_position = robot_description.base_position or [0, 0, 0]
            base_orientation = robot_description.base_orientation or [0, 0, 0, 1]
            
            self.robot_id = p.loadURDF(
                robot_description.urdf_path,
                basePosition=base_position,
                baseOrientation=base_orientation,
                physicsClientId=self.physics_client
            )
            
            # Get joint information
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
            self.joint_indices = []
            self.joint_names = []
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                joint_name = joint_info[1].decode('utf-8')
                joint_type = joint_info[2]
                
                # Only consider revolute and prismatic joints
                if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    self.joint_indices.append(i)
                    self.joint_names.append(joint_name)
            
            # Set joint properties
            if robot_description.joint_damping:
                for i, joint_idx in enumerate(self.joint_indices):
                    if i < len(robot_description.joint_damping):
                        p.changeDynamics(
                            self.robot_id, joint_idx,
                            jointDamping=robot_description.joint_damping[i],
                            physicsClientId=self.physics_client
                        )
            
            if robot_description.joint_friction:
                for i, joint_idx in enumerate(self.joint_indices):
                    if i < len(robot_description.joint_friction):
                        p.changeDynamics(
                            self.robot_id, joint_idx,
                            jointLateralFriction=robot_description.joint_friction[i],
                            physicsClientId=self.physics_client
                        )
            
            # Initialize state tracking
            num_actuated_joints = len(self.joint_indices)
            self.current_joint_positions = torch.zeros(num_actuated_joints)
            self.current_joint_velocities = torch.zeros(num_actuated_joints)
            self.current_joint_torques = torch.zeros(num_actuated_joints)
            
            logger.info(f"Robot loaded: {num_actuated_joints} actuated joints")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            return False
    
    def load_environment(self, add_ground: bool = True) -> None:
        """
        Load basic environment objects.
        
        Args:
            add_ground: Whether to add ground plane
        """
        if add_ground:
            self.plane_id = p.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client
            )
    
    def add_obstacle(self, 
                    position: Tuple[float, float, float],
                    shape_type: str = "box",
                    size: Tuple[float, float, float] = (0.1, 0.1, 0.1),
                    mass: float = 0.0,
                    color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)) -> int:
        """
        Add obstacle to environment.
        
        Args:
            position: Position [x, y, z]
            shape_type: Shape type ("box", "sphere", "cylinder")
            size: Size parameters (depends on shape)
            mass: Mass (0 for static obstacle)
            color: RGBA color
            
        Returns:
            Obstacle ID
        """
        if shape_type == "box":
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX, 
                halfExtents=size,
                physicsClientId=self.physics_client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=size,
                rgbaColor=color,
                physicsClientId=self.physics_client
            )
        elif shape_type == "sphere":
            radius = size[0]
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=self.physics_client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=radius,
                rgbaColor=color,
                physicsClientId=self.physics_client
            )
        elif shape_type == "cylinder":
            radius, height = size[0], size[1]
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=self.physics_client
            )
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=radius,
                length=height,
                rgbaColor=color,
                physicsClientId=self.physics_client
            )
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        
        obstacle_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self.physics_client
        )
        
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id
    
    def step(self, action: torch.Tensor, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Step physics simulation forward.
        
        Args:
            action: Joint torques/forces
            dt: Time step (uses default if None)
            
        Returns:
            Dictionary with updated state information
        """
        if self.robot_id is None:
            raise RuntimeError("No robot loaded. Call load_robot() first.")
        
        # Convert action to numpy array
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = np.array(action)
        
        # Apply torques to joints
        if len(action_np) == len(self.joint_indices):
            self.current_joint_torques = torch.from_numpy(action_np).float()
            
            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_indices,
                controlMode=p.TORQUE_CONTROL,
                forces=action_np,
                physicsClientId=self.physics_client
            )
        else:
            logger.warning(f"Action dimension {len(action_np)} doesn't match "
                         f"number of joints {len(self.joint_indices)}")
        
        # Step simulation
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # Update state
        joint_states = p.getJointStates(
            self.robot_id, 
            self.joint_indices,
            physicsClientId=self.physics_client
        )
        
        positions = np.array([state[0] for state in joint_states])
        velocities = np.array([state[1] for state in joint_states])
        
        self.current_joint_positions = torch.from_numpy(positions).float()
        self.current_joint_velocities = torch.from_numpy(velocities).float()
        
        # Get end-effector information
        end_effector_info = self._get_end_effector_state()
        
        # Get contact information
        contact_info = self._get_contact_forces()
        
        return {
            'positions': self.current_joint_positions,
            'velocities': self.current_joint_velocities,
            'torques': self.current_joint_torques,
            'end_effector': end_effector_info,
            'contacts': contact_info
        }
    
    def _get_end_effector_state(self) -> Dict[str, torch.Tensor]:
        """Get end-effector pose and velocity."""
        if self.robot_id is None or not self.joint_indices:
            return {}
        
        # Get link state for last link (assumed to be end-effector)
        end_effector_link = max(self.joint_indices)
        link_state = p.getLinkState(
            self.robot_id,
            end_effector_link,
            computeLinkVelocity=1,
            physicsClientId=self.physics_client
        )
        
        position = torch.tensor(link_state[0])
        orientation_quat = torch.tensor(link_state[1])
        
        # Convert quaternion to euler angles
        orientation_euler = torch.tensor(p.getEulerFromQuaternion(orientation_quat))
        
        linear_velocity = torch.tensor(link_state[6]) if len(link_state) > 6 else torch.zeros(3)
        angular_velocity = torch.tensor(link_state[7]) if len(link_state) > 7 else torch.zeros(3)
        
        return {
            'position': position,
            'orientation_quat': orientation_quat,
            'orientation_euler': orientation_euler,
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'pose': torch.cat([position, orientation_euler])
        }
    
    def _get_contact_forces(self) -> Dict[str, Any]:
        """Get contact force information."""
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            physicsClientId=self.physics_client
        )
        
        total_contact_force = torch.zeros(3)
        contact_count = 0
        
        for contact in contact_points:
            # Extract contact force
            normal_force = contact[9]  # Normal force magnitude
            contact_normal = torch.tensor(contact[7])  # Contact normal vector
            
            # Add to total force
            total_contact_force += normal_force * contact_normal
            contact_count += 1
        
        return {
            'total_force': total_contact_force,
            'contact_count': contact_count,
            'contact_points': contact_points
        }
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current physics state."""
        return {
            'positions': self.current_joint_positions.clone(),
            'velocities': self.current_joint_velocities.clone(),
            'torques': self.current_joint_torques.clone()
        }
    
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set physics state."""
        if self.robot_id is None:
            return
        
        # Set joint positions and velocities
        if 'positions' in state:
            positions = state['positions'].detach().cpu().numpy()
            self.current_joint_positions = state['positions'].clone()
            
            for i, joint_idx in enumerate(self.joint_indices):
                if i < len(positions):
                    p.resetJointState(
                        self.robot_id,
                        joint_idx,
                        targetValue=positions[i],
                        targetVelocity=0,
                        physicsClientId=self.physics_client
                    )
        
        if 'velocities' in state:
            velocities = state['velocities'].detach().cpu().numpy()
            self.current_joint_velocities = state['velocities'].clone()
            
            for i, joint_idx in enumerate(self.joint_indices):
                if i < len(velocities):
                    current_pos = p.getJointState(
                        self.robot_id, joint_idx, 
                        physicsClientId=self.physics_client
                    )[0]
                    p.resetJointState(
                        self.robot_id,
                        joint_idx,
                        targetValue=current_pos,
                        targetVelocity=velocities[i],
                        physicsClientId=self.physics_client
                    )
    
    def reset(self) -> None:
        """Reset physics simulation to initial state."""
        if self.robot_id is not None:
            # Reset robot to initial pose
            for joint_idx in self.joint_indices:
                p.resetJointState(
                    self.robot_id,
                    joint_idx,
                    targetValue=0,
                    targetVelocity=0,
                    physicsClientId=self.physics_client
                )
        
        # Reset state tracking
        self.current_joint_positions.zero_()
        self.current_joint_velocities.zero_()
        self.current_joint_torques.zero_()
    
    def check_collisions(self, 
                        body_a: Optional[int] = None, 
                        body_b: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Check for collisions between bodies.
        
        Args:
            body_a: First body ID (robot if None)
            body_b: Second body ID (any if None)
            
        Returns:
            List of collision information dictionaries
        """
        if body_a is None:
            body_a = self.robot_id
        
        if body_a is None:
            return []
        
        if body_b is not None:
            contact_points = p.getContactPoints(
                bodyA=body_a, 
                bodyB=body_b,
                physicsClientId=self.physics_client
            )
        else:
            contact_points = p.getContactPoints(
                bodyA=body_a,
                physicsClientId=self.physics_client
            )
        
        collisions = []
        for contact in contact_points:
            collision_info = {
                'body_a': contact[1],
                'body_b': contact[2],
                'link_a': contact[3],
                'link_b': contact[4],
                'position_a': torch.tensor(contact[5]),
                'position_b': torch.tensor(contact[6]),
                'normal': torch.tensor(contact[7]),
                'distance': contact[8],
                'normal_force': contact[9]
            }
            collisions.append(collision_info)
        
        return collisions
    
    def get_minimum_distance(self, body_b: Optional[int] = None) -> float:
        """
        Get minimum distance between robot and specified body/environment.
        
        Args:
            body_b: Target body ID (any body if None)
            
        Returns:
            Minimum distance
        """
        if self.robot_id is None:
            return float('inf')
        
        if body_b is not None:
            # Distance to specific body
            closest_points = p.getClosestPoints(
                bodyA=self.robot_id,
                bodyB=body_b,
                distance=10.0,  # Maximum distance to check
                physicsClientId=self.physics_client
            )
        else:
            # Distance to any other body
            min_distance = float('inf')
            for obstacle_id in self.obstacle_ids:
                closest_points = p.getClosestPoints(
                    bodyA=self.robot_id,
                    bodyB=obstacle_id,
                    distance=10.0,
                    physicsClientId=self.physics_client
                )
                if closest_points:
                    distance = closest_points[0][8]
                    min_distance = min(min_distance, distance)
            return min_distance
        
        if closest_points:
            return closest_points[0][8]  # Distance
        else:
            return float('inf')
    
    def set_camera_position(self, 
                           target: Tuple[float, float, float],
                           distance: float = 2.0,
                           yaw: float = 0,
                           pitch: float = -30) -> None:
        """Set camera position for visualization."""
        if self.gui_enabled:
            p.resetDebugVisualizerCamera(
                cameraDistance=distance,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=target,
                physicsClientId=self.physics_client
            )
    
    def add_debug_line(self, 
                      start: Tuple[float, float, float],
                      end: Tuple[float, float, float],
                      color: Tuple[float, float, float] = (1, 0, 0),
                      lifetime: float = 0) -> int:
        """Add debug line for visualization."""
        return p.addUserDebugLine(
            start, end, color, lifeTime=lifetime,
            physicsClientId=self.physics_client
        )
    
    def remove_debug_item(self, item_id: int) -> None:
        """Remove debug visualization item."""
        p.removeUserDebugItem(item_id, physicsClientId=self.physics_client)
    
    def close(self) -> None:
        """Clean up physics simulation."""
        if hasattr(self, 'physics_client'):
            p.disconnect(physicsClientId=self.physics_client)
            logger.info("PyBullet physics engine closed")


class FallbackPhysicsEngine(PhysicsEngine):
    """
    Fallback physics engine when PyBullet is not available.
    
    Provides basic dynamics simulation without collision detection.
    """
    
    def __init__(self, robot_dof: int, mass_matrix: Optional[torch.Tensor] = None):
        """
        Initialize fallback physics engine.
        
        Args:
            robot_dof: Robot degrees of freedom
            mass_matrix: Mass matrix (identity if None)
        """
        self.robot_dof = robot_dof
        self.mass_matrix = mass_matrix if mass_matrix is not None else torch.eye(robot_dof)
        
        # State variables
        self.positions = torch.zeros(robot_dof)
        self.velocities = torch.zeros(robot_dof)
        self.torques = torch.zeros(robot_dof)
        
        # Simple dynamics parameters
        self.damping = 0.1
        self.dt = 1.0/240.0
        
        logger.info(f"FallbackPhysicsEngine initialized: {robot_dof} DOF")
    
    def step(self, action: torch.Tensor, dt: Optional[float] = None) -> Dict[str, Any]:
        """Step simulation using simple dynamics."""
        if dt is None:
            dt = self.dt
        
        # Update torques
        self.torques = action
        
        # Simple dynamics: M*qdd + D*qd = tau
        damping_torque = self.damping * self.velocities
        net_torque = self.torques - damping_torque
        
        # Solve for acceleration
        try:
            accelerations = torch.linalg.solve(self.mass_matrix, net_torque)
        except torch.linalg.LinAlgError:
            accelerations = torch.linalg.pinv(self.mass_matrix) @ net_torque
        
        # Integrate
        self.velocities += accelerations * dt
        self.positions += self.velocities * dt
        
        return {
            'positions': self.positions.clone(),
            'velocities': self.velocities.clone(),
            'torques': self.torques.clone(),
            'accelerations': accelerations
        }
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state."""
        return {
            'positions': self.positions.clone(),
            'velocities': self.velocities.clone(),
            'torques': self.torques.clone()
        }
    
    def set_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Set state."""
        if 'positions' in state:
            self.positions = state['positions'].clone()
        if 'velocities' in state:
            self.velocities = state['velocities'].clone()
        if 'torques' in state:
            self.torques = state['torques'].clone()
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.positions.zero_()
        self.velocities.zero_()
        self.torques.zero_()


def create_physics_engine(engine_type: str = "auto", **kwargs) -> PhysicsEngine:
    """
    Factory function to create physics engine.
    
    Args:
        engine_type: Engine type ("pybullet", "fallback", "auto")
        **kwargs: Engine-specific arguments
        
    Returns:
        Physics engine instance
    """
    if engine_type == "auto":
        if PYBULLET_AVAILABLE:
            engine_type = "pybullet"
        else:
            engine_type = "fallback"
    
    if engine_type == "pybullet":
        if not PYBULLET_AVAILABLE:
            logger.warning("PyBullet not available, falling back to simple physics")
            engine_type = "fallback"
        else:
            return PyBulletPhysicsEngine(**kwargs)
    
    if engine_type == "fallback":
        robot_dof = kwargs.get('robot_dof', 7)
        mass_matrix = kwargs.get('mass_matrix', None)
        return FallbackPhysicsEngine(robot_dof, mass_matrix)
    
    raise ValueError(f"Unknown physics engine type: {engine_type}")


# URDF templates for common robots
def create_simple_arm_urdf(num_joints: int = 7, 
                          link_lengths: List[float] = None,
                          output_path: str = "simple_arm.urdf") -> str:
    """
    Create a simple arm URDF for testing.
    
    Args:
        num_joints: Number of joints
        link_lengths: Link lengths (auto-generated if None)
        output_path: Output URDF file path
        
    Returns:
        Path to generated URDF file
    """
    if link_lengths is None:
        link_lengths = [0.3] * num_joints
    
    urdf_content = '<?xml version="1.0"?>\n<robot name="simple_arm">\n'
    
    # Base link
    urdf_content += '''
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
'''
    
    # Generate links and joints
    for i in range(num_joints):
        link_name = f"link_{i+1}"
        joint_name = f"joint_{i+1}"
        parent_link = "base_link" if i == 0 else f"link_{i}"
        
        # Joint
        axis = [1, 0, 0] if i % 3 == 0 else [0, 1, 0] if i % 3 == 1 else [0, 0, 1]
        origin_z = 0.05 if i == 0 else link_lengths[i-1]
        
        urdf_content += f'''
  <joint name="{joint_name}" type="revolute">
    <parent link="{parent_link}"/>
    <child link="{link_name}"/>
    <origin xyz="0 0 {origin_z}" rpy="0 0 0"/>
    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="2"/>
    <dynamics damping="0.1" friction="0.1"/>
  </joint>
'''
        
        # Link
        length = link_lengths[i] if i < len(link_lengths) else 0.2
        urdf_content += f'''
  <link name="{link_name}">
    <visual>
      <origin xyz="0 0 {length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.02" length="{length}"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 {length/2}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.025" length="{length}"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 {length/2}" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.001"/>
    </inertial>
  </link>
'''
    
    urdf_content += '</robot>\n'
    
    # Write to file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(urdf_content)
    
    logger.info(f"Generated URDF: {output_path}")
    return output_path