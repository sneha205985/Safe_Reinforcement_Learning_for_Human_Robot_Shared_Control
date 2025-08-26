"""
Safety constraint implementations for Constrained Policy Optimization (CPO).

This module implements safety constraint functions g(s,a) ≤ 0 for human-robot 
shared control systems, including collision avoidance, joint limits, and force limits.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ConstraintViolationError(Exception):
    """Raised when a safety constraint is violated."""
    pass


class SafetyConstraint(ABC):
    """
    Abstract base class for safety constraints in CPO.
    
    Implements constraint functions g(s,a) ≤ 0 where violations occur when g > 0.
    """
    
    def __init__(self, constraint_id: str, violation_penalty: float = 1000.0):
        """
        Initialize safety constraint.
        
        Args:
            constraint_id: Unique identifier for this constraint
            violation_penalty: Penalty coefficient for constraint violations
        """
        self.constraint_id = constraint_id
        self.violation_penalty = violation_penalty
        self.violation_history: List[float] = []
        
    @abstractmethod
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate constraint function g(s,a).
        
        Args:
            state: Current state tensor [batch_size, state_dim]
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Constraint values [batch_size] where g ≤ 0 is safe
        """
        pass
    
    @abstractmethod
    def gradient(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients ∇_s g(s,a) and ∇_a g(s,a).
        
        Args:
            state: Current state tensor [batch_size, state_dim]  
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Tuple of (grad_state, grad_action) tensors
        """
        pass
    
    def is_violated(self, state: torch.Tensor, action: torch.Tensor, 
                   tolerance: float = 1e-6) -> torch.Tensor:
        """
        Check if constraint is violated (g > tolerance).
        
        Args:
            state: Current state tensor
            action: Action tensor
            tolerance: Tolerance for constraint satisfaction
            
        Returns:
            Boolean tensor indicating violations
        """
        constraint_values = self.evaluate(state, action)
        return constraint_values > tolerance
    
    def log_violation(self, violation_value: float) -> None:
        """Log constraint violation for monitoring."""
        self.violation_history.append(violation_value)
        logger.warning(f"Constraint {self.constraint_id} violated: g = {violation_value:.6f}")
    
    def get_violation_statistics(self) -> Dict[str, float]:
        """Get statistics on constraint violations."""
        if not self.violation_history:
            return {"count": 0, "max": 0.0, "mean": 0.0, "recent": 0.0}
            
        return {
            "count": len(self.violation_history),
            "max": max(self.violation_history),
            "mean": np.mean(self.violation_history),
            "recent": self.violation_history[-1] if self.violation_history else 0.0
        }


class CollisionConstraint(SafetyConstraint):
    """
    Collision avoidance constraint for human-robot interaction.
    
    g(s,a) = d_min - d(s,a) ≤ 0
    where d(s,a) is minimum distance to human/obstacles and d_min is safety threshold.
    """
    
    def __init__(self, min_distance: float = 0.1, human_position_idx: Tuple[int, int] = (0, 2),
                 robot_position_idx: Tuple[int, int] = (3, 5)):
        """
        Initialize collision constraint.
        
        Args:
            min_distance: Minimum safe distance (meters)
            human_position_idx: Slice indices for human position in state
            robot_position_idx: Slice indices for robot position in state  
        """
        super().__init__("collision_avoidance")
        self.min_distance = min_distance
        self.human_pos_idx = human_position_idx
        self.robot_pos_idx = robot_position_idx
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate collision constraint g(s,a) = d_min - d(s,a).
        
        Args:
            state: State tensor [batch_size, state_dim] containing positions
            action: Action tensor [batch_size, action_dim]
            
        Returns:
            Constraint values [batch_size] 
        """
        human_pos = state[:, self.human_pos_idx[0]:self.human_pos_idx[1]]
        robot_pos = state[:, self.robot_pos_idx[0]:self.robot_pos_idx[1]]
        
        # Account for robot motion from action (simplified dynamics)
        dt = 0.1  # Time step
        robot_vel = action[:, :robot_pos.shape[1]]  # Assume action is velocity
        predicted_robot_pos = robot_pos + dt * robot_vel
        
        # Compute distance
        distance = torch.norm(human_pos - predicted_robot_pos, dim=1)
        
        # g(s,a) = d_min - d(s,a) ≤ 0
        constraint_value = self.min_distance - distance
        
        return constraint_value
    
    def gradient(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of collision constraint.
        
        Returns:
            Tuple of (grad_state, grad_action)
        """
        state.requires_grad_(True)
        action.requires_grad_(True)
        
        constraint_val = self.evaluate(state, action).sum()
        
        grad_state = torch.autograd.grad(constraint_val, state, 
                                       create_graph=True, retain_graph=True)[0]
        grad_action = torch.autograd.grad(constraint_val, action,
                                        create_graph=True, retain_graph=True)[0]
        
        return grad_state, grad_action


class JointLimitConstraint(SafetyConstraint):
    """
    Joint angle limit constraint for robotic systems.
    
    g(s,a) = max(q - q_max, q_min - q) ≤ 0
    where q are joint angles, q_min and q_max are joint limits.
    """
    
    def __init__(self, joint_limits: Dict[str, Tuple[float, float]], 
                 joint_indices: Dict[str, int]):
        """
        Initialize joint limit constraint.
        
        Args:
            joint_limits: Dict mapping joint names to (min, max) limits
            joint_indices: Dict mapping joint names to state indices
        """
        super().__init__("joint_limits")
        self.joint_limits = joint_limits
        self.joint_indices = joint_indices
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate joint limit constraint.
        
        Returns maximum violation across all joints.
        """
        batch_size = state.shape[0]
        max_violations = torch.zeros(batch_size, device=state.device)
        
        dt = 0.1  # Time step
        
        for joint_name, (q_min, q_max) in self.joint_limits.items():
            joint_idx = self.joint_indices[joint_name]
            
            # Current joint position
            q = state[:, joint_idx]
            
            # Predicted joint position (assume action is joint velocity)
            if joint_idx < action.shape[1]:
                q_dot = action[:, joint_idx]
                q_pred = q + dt * q_dot
            else:
                q_pred = q
            
            # Constraint violations
            upper_violation = q_pred - q_max
            lower_violation = q_min - q_pred
            
            # Maximum violation for this joint
            joint_violation = torch.max(upper_violation, lower_violation)
            
            # Update maximum across all joints
            max_violations = torch.max(max_violations, joint_violation)
            
        return max_violations
    
    def gradient(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients of joint limit constraint."""
        state.requires_grad_(True)
        action.requires_grad_(True)
        
        constraint_val = self.evaluate(state, action).sum()
        
        grad_state = torch.autograd.grad(constraint_val, state,
                                       create_graph=True, retain_graph=True)[0]
        grad_action = torch.autograd.grad(constraint_val, action,
                                        create_graph=True, retain_graph=True)[0]
        
        return grad_state, grad_action


class ForceConstraint(SafetyConstraint):
    """
    Force/torque limit constraint for safe physical interaction.
    
    g(s,a) = ||F(s,a)|| - F_max ≤ 0
    where F(s,a) is applied force/torque and F_max is maximum safe force.
    """
    
    def __init__(self, max_force: float = 50.0, force_indices: Optional[List[int]] = None):
        """
        Initialize force constraint.
        
        Args:
            max_force: Maximum allowable force magnitude (Newtons)
            force_indices: Indices in action vector corresponding to forces
        """
        super().__init__("force_limits")
        self.max_force = max_force
        self.force_indices = force_indices or list(range(6))  # Default: 6-DOF force/torque
        
    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Evaluate force constraint g(s,a) = ||F|| - F_max.
        """
        # Extract force components from action
        forces = action[:, self.force_indices]
        
        # Compute force magnitude
        force_magnitude = torch.norm(forces, dim=1)
        
        # g(s,a) = ||F|| - F_max ≤ 0
        constraint_value = force_magnitude - self.max_force
        
        return constraint_value
    
    def gradient(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients of force constraint."""
        state.requires_grad_(True)
        action.requires_grad_(True)
        
        constraint_val = self.evaluate(state, action).sum()
        
        grad_state = torch.autograd.grad(constraint_val, state,
                                       create_graph=True, retain_graph=True)[0]
        grad_action = torch.autograd.grad(constraint_val, action,
                                        create_graph=True, retain_graph=True)[0]
        
        return grad_state, grad_action


class ConstraintManager:
    """
    Manages multiple safety constraints for CPO optimization.
    
    Aggregates constraint violations and computes combined gradients.
    """
    
    def __init__(self, constraints: List[SafetyConstraint]):
        """
        Initialize constraint manager.
        
        Args:
            constraints: List of SafetyConstraint instances
        """
        self.constraints = constraints
        self.constraint_weights = torch.ones(len(constraints))
        
    def add_constraint(self, constraint: SafetyConstraint, weight: float = 1.0) -> None:
        """Add new constraint with optional weight."""
        self.constraints.append(constraint)
        self.constraint_weights = torch.cat([self.constraint_weights, torch.tensor([weight])])
        
    def evaluate_all(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate all constraints.
        
        Returns:
            Dict mapping constraint IDs to constraint values
        """
        results = {}
        for constraint in self.constraints:
            results[constraint.constraint_id] = constraint.evaluate(state, action)
        return results
    
    def compute_violation_penalty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted penalty for all constraint violations.
        
        Returns:
            Total violation penalty [batch_size]
        """
        total_penalty = torch.zeros(state.shape[0], device=state.device)
        
        for i, constraint in enumerate(self.constraints):
            violation = constraint.evaluate(state, action)
            # Apply penalty only for violations (g > 0)
            penalty = torch.clamp(violation, min=0.0) * constraint.violation_penalty
            total_penalty += self.constraint_weights[i] * penalty
            
        return total_penalty
    
    def compute_combined_gradients(self, state: torch.Tensor, 
                                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute combined gradients across all constraints.
        
        Returns:
            Tuple of (combined_grad_state, combined_grad_action)
        """
        combined_grad_state = torch.zeros_like(state)
        combined_grad_action = torch.zeros_like(action)
        
        for i, constraint in enumerate(self.constraints):
            grad_state, grad_action = constraint.gradient(state, action)
            
            combined_grad_state += self.constraint_weights[i] * grad_state
            combined_grad_action += self.constraint_weights[i] * grad_action
            
        return combined_grad_state, combined_grad_action
    
    def check_safety(self, state: torch.Tensor, action: torch.Tensor, 
                    tolerance: float = 1e-6) -> Dict[str, bool]:
        """
        Check safety across all constraints.
        
        Returns:
            Dict mapping constraint IDs to safety status (True = safe)
        """
        safety_status = {}
        
        for constraint in self.constraints:
            violations = constraint.is_violated(state, action, tolerance)
            safety_status[constraint.constraint_id] = not violations.any().item()
            
            # Log violations
            if violations.any():
                violation_values = constraint.evaluate(state, action)
                max_violation = violation_values[violations].max().item()
                constraint.log_violation(max_violation)
                
        return safety_status
    
    def get_constraint_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get violation statistics for all constraints."""
        stats = {}
        for constraint in self.constraints:
            stats[constraint.constraint_id] = constraint.get_violation_statistics()
        return stats