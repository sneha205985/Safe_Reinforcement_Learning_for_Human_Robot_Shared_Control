"""
Lagrangian optimization methods for Constrained Policy Optimization (CPO).

This module implements the Lagrangian formulation L(θ,λ) = J(θ) - λᵀg(θ) with
dual variable updates and primal-dual optimization for safe reinforcement learning.
"""

from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
import torch
import torch.optim as optim
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LagrangianState:
    """State of the Lagrangian optimization process."""
    iteration: int
    primal_variables: torch.Tensor  # Policy parameters θ
    dual_variables: torch.Tensor    # Lagrange multipliers λ
    objective_value: float          # J(θ)
    constraint_values: torch.Tensor # g(θ)
    lagrangian_value: float         # L(θ,λ)
    kl_divergence: float           # KL constraint value
    step_size: float               # Current step size
    convergence_error: float       # ||∇L|| for convergence check


class ConvergenceCriteria:
    """Convergence criteria for primal-dual optimization."""
    
    def __init__(self, 
                 gradient_tolerance: float = 1e-4,
                 constraint_tolerance: float = 1e-3,
                 kl_tolerance: float = 1e-2,
                 max_iterations: int = 100,
                 min_iterations: int = 10):
        """
        Initialize convergence criteria.
        
        Args:
            gradient_tolerance: Tolerance for gradient norm ||∇L||
            constraint_tolerance: Tolerance for constraint violations |g(θ)|
            kl_tolerance: Tolerance for KL divergence constraint
            max_iterations: Maximum optimization iterations
            min_iterations: Minimum iterations before convergence check
        """
        self.gradient_tolerance = gradient_tolerance
        self.constraint_tolerance = constraint_tolerance
        self.kl_tolerance = kl_tolerance
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        
    def check_convergence(self, state: LagrangianState) -> Tuple[bool, str]:
        """
        Check if optimization has converged.
        
        Args:
            state: Current optimization state
            
        Returns:
            Tuple of (converged, reason)
        """
        if state.iteration < self.min_iterations:
            return False, "min_iterations_not_reached"
            
        if state.iteration >= self.max_iterations:
            return True, "max_iterations_reached"
            
        if state.convergence_error < self.gradient_tolerance:
            return True, "gradient_tolerance_satisfied"
            
        if torch.max(torch.abs(state.constraint_values)) < self.constraint_tolerance:
            if state.convergence_error < self.gradient_tolerance * 10:
                return True, "constraint_and_gradient_tolerance_satisfied"
                
        if state.kl_divergence > self.kl_tolerance * 2:
            return True, "kl_divergence_too_large"
            
        return False, "not_converged"


class LagrangianOptimizer:
    """
    Lagrangian optimizer for Constrained Policy Optimization.
    
    Implements the primal-dual optimization:
    min_θ max_λ≥0 L(θ,λ) = J(θ) - λᵀg(θ)
    
    where J(θ) is the policy objective and g(θ) are constraint functions.
    """
    
    def __init__(self,
                 policy: Any,
                 constraint_manager: Any,
                 dual_lr: float = 0.1,
                 dual_regularization: float = 1e-4,
                 trust_region_size: float = 0.01,
                 line_search_steps: int = 10,
                 line_search_decay: float = 0.5,
                 convergence_criteria: Optional[ConvergenceCriteria] = None):
        """
        Initialize Lagrangian optimizer.
        
        Args:
            policy: Safe policy instance
            constraint_manager: Constraint manager for safety constraints
            dual_lr: Learning rate for dual variables (λ updates)
            dual_regularization: Regularization for dual variable updates
            trust_region_size: Maximum KL divergence for trust region
            line_search_steps: Number of line search iterations
            line_search_decay: Decay factor for line search
            convergence_criteria: Custom convergence criteria
        """
        self.policy = policy
        self.constraint_manager = constraint_manager
        self.dual_lr = dual_lr
        self.dual_regularization = dual_regularization
        self.trust_region_size = trust_region_size
        self.line_search_steps = line_search_steps
        self.line_search_decay = line_search_decay
        
        # Initialize dual variables (one per constraint)
        num_constraints = len(constraint_manager.constraints)
        self.dual_variables = torch.zeros(num_constraints, requires_grad=False)
        
        # Convergence criteria
        self.convergence_criteria = convergence_criteria or ConvergenceCriteria()
        
        # Optimization history
        self.optimization_history: List[LagrangianState] = []
        self.current_iteration = 0
        
        # Dual variable bounds (λ ≥ 0)
        self.dual_min = 1e-8
        self.dual_max = 100.0
        
        logger.info(f"Initialized Lagrangian optimizer with {num_constraints} constraints")
        
    def compute_lagrangian(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor,
                          advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Lagrangian L(θ,λ) = J(θ) - λᵀg(θ).
        
        Args:
            states: State batch [batch_size, state_dim]
            actions: Action batch [batch_size, action_dim] 
            advantages: Advantage estimates [batch_size]
            
        Returns:
            Tuple of (lagrangian_value, info_dict)
        """
        # Compute policy objective J(θ)
        policy_loss = self.policy.compute_policy_gradient(states, actions, advantages)
        objective = -policy_loss  # Negate because we want to maximize J(θ)
        
        # Compute constraint values g(θ)
        constraint_dict = self.constraint_manager.evaluate_all(states, actions)
        constraint_values = torch.stack([constraint_dict[cid] 
                                       for cid in sorted(constraint_dict.keys())])
        
        # Average over batch
        avg_constraints = constraint_values.mean(dim=1)  # [num_constraints]
        
        # Lagrangian: L(θ,λ) = J(θ) - λᵀg(θ)
        constraint_penalty = torch.dot(self.dual_variables, avg_constraints)
        lagrangian = objective - constraint_penalty
        
        info = {
            "objective": objective,
            "constraint_values": avg_constraints,
            "constraint_penalty": constraint_penalty,
            "dual_variables": self.dual_variables.clone()
        }
        
        return lagrangian, info
        
    def compute_constraint_gradients(self, 
                                   states: torch.Tensor,
                                   actions: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint gradients ∇_θ g(θ) for policy parameters.
        
        Args:
            states: State batch [batch_size, state_dim]
            actions: Action batch [batch_size, action_dim]
            
        Returns:
            Constraint gradient tensor [num_constraints, param_dim]
        """
        constraint_gradients = []
        
        for constraint in self.constraint_manager.constraints:
            # Enable gradient computation for policy parameters
            for param in self.policy.policy_net.parameters():
                param.requires_grad_(True)
            
            # Sample actions from current policy for gradient computation
            dist = self.policy.policy_net.get_distribution(states)
            policy_actions = dist.rsample()  # Reparameterized sampling
            
            # Compute constraint value
            constraint_val = constraint.evaluate(states, policy_actions).mean()
            
            # Compute gradient w.r.t. policy parameters
            grad = torch.autograd.grad(constraint_val, self.policy.policy_net.parameters(),
                                     create_graph=True, retain_graph=True)
            
            # Flatten gradient
            flat_grad = torch.cat([g.view(-1) for g in grad])
            constraint_gradients.append(flat_grad)
            
        return torch.stack(constraint_gradients)
    
    def dual_update(self, constraint_values: torch.Tensor) -> torch.Tensor:
        """
        Update dual variables λ using gradient ascent.
        
        Dual update: λ_{k+1} = max(0, λ_k + α_dual * g(θ_k))
        
        Args:
            constraint_values: Current constraint values g(θ) [num_constraints]
            
        Returns:
            Updated dual variables
        """
        # Gradient ascent on dual variables (maximize w.r.t. λ)
        dual_gradient = constraint_values + self.dual_regularization * self.dual_variables
        
        # Update with projection onto [dual_min, dual_max]
        updated_duals = self.dual_variables + self.dual_lr * dual_gradient
        updated_duals = torch.clamp(updated_duals, self.dual_min, self.dual_max)
        
        # Log dual variable updates
        dual_change = torch.norm(updated_duals - self.dual_variables).item()
        logger.debug(f"Dual update: ||Δλ|| = {dual_change:.6f}")
        
        self.dual_variables = updated_duals
        return updated_duals
    
    def primal_update(self, 
                     states: torch.Tensor,
                     actions: torch.Tensor, 
                     advantages: torch.Tensor) -> Dict[str, float]:
        """
        Update policy parameters θ using constrained optimization.
        
        Implements trust region update with constraint linearization:
        θ_{k+1} = θ_k + α * search_direction
        subject to: ||θ - θ_k||² ≤ δ and g(θ_k) + ∇g(θ_k)ᵀ(θ - θ_k) ≤ 0
        
        Args:
            states: State batch for gradient estimation
            actions: Action batch
            advantages: Advantage estimates
            
        Returns:
            Dictionary with update statistics
        """
        # Compute policy gradient ∇_θ J(θ)  
        policy_loss = self.policy.compute_policy_gradient(states, actions, advantages)
        policy_grad = torch.autograd.grad(policy_loss, self.policy.policy_net.parameters(),
                                        create_graph=True, retain_graph=True)
        policy_grad_flat = torch.cat([g.view(-1) for g in policy_grad])
        
        # Compute constraint gradients ∇_θ g(θ)
        constraint_grads = self.compute_constraint_gradients(states, actions)
        
        # Lagrangian gradient: ∇_θ L = ∇_θ J - λᵀ ∇_θ g
        lagrangian_grad = policy_grad_flat - torch.mv(constraint_grads.T, self.dual_variables)
        
        # Trust region update using conjugate gradient
        old_dist = self.policy.policy_net.get_distribution(states)
        
        def fisher_vector_product(v):
            """Compute Fisher Information Matrix times vector."""
            # KL divergence gradient for Fisher matrix approximation
            kl = self._compute_kl_divergence(states, old_dist)
            kl_grad = torch.autograd.grad(kl, self.policy.policy_net.parameters(),
                                        create_graph=True, retain_graph=True)
            kl_grad_flat = torch.cat([g.view(-1) for g in kl_grad])
            
            # Fisher-vector product
            gvp = torch.sum(kl_grad_flat * v)
            fvp_grad = torch.autograd.grad(gvp, self.policy.policy_net.parameters(),
                                         retain_graph=True)
            return torch.cat([g.view(-1) for g in fvp_grad]) + 1e-4 * v
        
        # Solve Fisher * search_dir = lagrangian_grad using conjugate gradient
        search_direction = self._conjugate_gradient(fisher_vector_product, lagrangian_grad)
        
        # Line search with constraint and trust region checking
        step_size, update_stats = self._constrained_line_search(
            states, actions, advantages, old_dist, search_direction, constraint_grads
        )
        
        # Apply update to policy parameters
        self._apply_parameter_update(search_direction * step_size)
        
        update_stats.update({
            "lagrangian_grad_norm": torch.norm(lagrangian_grad).item(),
            "policy_grad_norm": torch.norm(policy_grad_flat).item(),
            "search_direction_norm": torch.norm(search_direction).item()
        })
        
        return update_stats
    
    def _constrained_line_search(self, 
                               states: torch.Tensor,
                               actions: torch.Tensor,
                               advantages: torch.Tensor,
                               old_dist,
                               search_direction: torch.Tensor,
                               constraint_grads: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """
        Perform line search with trust region and constraint satisfaction.
        
        Args:
            states: State batch
            actions: Action batch  
            advantages: Advantage estimates
            old_dist: Old policy distribution
            search_direction: Search direction from conjugate gradient
            constraint_grads: Constraint gradients [num_constraints, param_dim]
            
        Returns:
            Tuple of (step_size, statistics)
        """
        step_size = 1.0
        old_params = [p.clone() for p in self.policy.policy_net.parameters()]
        
        for i in range(self.line_search_steps):
            # Try this step size
            self._apply_parameter_update(search_direction * step_size, old_params)
            
            # Check trust region constraint (KL divergence)
            new_dist = self.policy.policy_net.get_distribution(states)
            kl_div = torch.distributions.kl_divergence(old_dist, new_dist).mean().item()
            
            if kl_div <= self.trust_region_size:
                # Check constraint linearization
                constraint_dict = self.constraint_manager.evaluate_all(states, actions)
                new_constraints = torch.stack([constraint_dict[cid].mean() 
                                             for cid in sorted(constraint_dict.keys())])
                
                # Linearized constraint approximation
                param_change_flat = search_direction * step_size
                linearized_change = torch.mv(constraint_grads, param_change_flat)
                
                # Check if constraints are satisfied within approximation
                constraint_satisfied = True
                for j, constraint in enumerate(self.constraint_manager.constraints):
                    if new_constraints[j] > self.convergence_criteria.constraint_tolerance:
                        constraint_satisfied = False
                        break
                
                if constraint_satisfied or i == self.line_search_steps - 1:
                    break
            
            # Reduce step size
            step_size *= self.line_search_decay
        
        stats = {
            "step_size": step_size,
            "kl_divergence": kl_div,
            "line_search_iterations": i + 1,
            "final_constraint_max": torch.max(torch.abs(new_constraints)).item()
        }
        
        return step_size, stats
    
    def _conjugate_gradient(self, Avp_func: Callable, b: torch.Tensor, 
                          max_iter: int = 10, tol: float = 1e-8) -> torch.Tensor:
        """
        Solve Ax = b using conjugate gradient method.
        
        Args:
            Avp_func: Function computing A*v product
            b: Right-hand side vector
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            
        Returns:
            Solution vector x
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        r_dot_old = torch.dot(r, r)
        
        for i in range(max_iter):
            Ap = Avp_func(p)
            alpha = r_dot_old / (torch.dot(p, Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            r_dot_new = torch.dot(r, r)
            
            if r_dot_new < tol:
                logger.debug(f"CG converged in {i+1} iterations")
                break
                
            beta = r_dot_new / (r_dot_old + 1e-10)
            p = r + beta * p
            r_dot_old = r_dot_new
            
        return x
    
    def _apply_parameter_update(self, update: torch.Tensor, 
                              old_params: Optional[List[torch.Tensor]] = None) -> None:
        """Apply flat parameter update to policy network."""
        if old_params is None:
            old_params = [p.clone() for p in self.policy.policy_net.parameters()]
            
        idx = 0
        for i, param in enumerate(self.policy.policy_net.parameters()):
            param_size = param.numel()
            if idx < len(update):
                param.data = old_params[i] + update[idx:idx+param_size].view_as(param)
            idx += param_size
    
    def _compute_kl_divergence(self, states: torch.Tensor, old_dist) -> torch.Tensor:
        """Compute KL divergence between old and current policy."""
        new_dist = self.policy.policy_net.get_distribution(states)
        return torch.distributions.kl_divergence(old_dist, new_dist).mean()
    
    def optimize_step(self, 
                     states: torch.Tensor,
                     actions: torch.Tensor, 
                     advantages: torch.Tensor) -> LagrangianState:
        """
        Perform one step of primal-dual optimization.
        
        Args:
            states: State batch [batch_size, state_dim]
            actions: Action batch [batch_size, action_dim]
            advantages: Advantage estimates [batch_size]
            
        Returns:
            Current optimization state
        """
        self.current_iteration += 1
        
        # Compute current Lagrangian and constraints
        lagrangian, info = self.compute_lagrangian(states, actions, advantages)
        constraint_values = info["constraint_values"]
        
        # Dual variable update (maximize w.r.t. λ)
        updated_duals = self.dual_update(constraint_values)
        
        # Primal update (minimize w.r.t. θ)  
        primal_stats = self.primal_update(states, actions, advantages)
        
        # Recompute Lagrangian after updates
        new_lagrangian, new_info = self.compute_lagrangian(states, actions, advantages)
        
        # Create optimization state
        state = LagrangianState(
            iteration=self.current_iteration,
            primal_variables=torch.cat([p.view(-1) for p in self.policy.policy_net.parameters()]),
            dual_variables=updated_duals.clone(),
            objective_value=new_info["objective"].item(),
            constraint_values=new_info["constraint_values"],
            lagrangian_value=new_lagrangian.item(),
            kl_divergence=primal_stats.get("kl_divergence", 0.0),
            step_size=primal_stats.get("step_size", 0.0),
            convergence_error=primal_stats.get("lagrangian_grad_norm", float('inf'))
        )
        
        # Store in history
        self.optimization_history.append(state)
        
        # Log progress
        logger.info(
            f"Iteration {self.current_iteration}: "
            f"L={state.lagrangian_value:.4f}, "
            f"J={state.objective_value:.4f}, "
            f"max|g|={torch.max(torch.abs(state.constraint_values)):.4f}, "
            f"||∇L||={state.convergence_error:.6f}"
        )
        
        return state
    
    def optimize(self, 
                states: torch.Tensor,
                actions: torch.Tensor,
                advantages: torch.Tensor,
                max_iterations: Optional[int] = None) -> Tuple[bool, str, List[LagrangianState]]:
        """
        Run full primal-dual optimization until convergence.
        
        Args:
            states: State batch for optimization
            actions: Action batch 
            advantages: Advantage estimates
            max_iterations: Override default max iterations
            
        Returns:
            Tuple of (converged, reason, optimization_history)
        """
        if max_iterations is not None:
            self.convergence_criteria.max_iterations = max_iterations
            
        logger.info(f"Starting Lagrangian optimization with {len(states)} samples")
        
        converged = False
        reason = "not_started"
        
        while not converged:
            # Perform optimization step
            state = self.optimize_step(states, actions, advantages)
            
            # Check convergence
            converged, reason = self.convergence_criteria.check_convergence(state)
            
            if converged:
                logger.info(f"Optimization converged: {reason}")
                break
                
        final_stats = self.get_optimization_statistics()
        logger.info(f"Final statistics: {final_stats}")
        
        return converged, reason, self.optimization_history
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.optimization_history:
            return {}
            
        history = self.optimization_history
        
        return {
            "total_iterations": len(history),
            "final_objective": history[-1].objective_value,
            "final_lagrangian": history[-1].lagrangian_value,
            "final_constraint_max": torch.max(torch.abs(history[-1].constraint_values)).item(),
            "final_dual_variables": history[-1].dual_variables.tolist(),
            "convergence_error": history[-1].convergence_error,
            "objective_improvement": history[-1].objective_value - history[0].objective_value if len(history) > 1 else 0,
            "dual_variable_changes": [
                torch.norm(history[i].dual_variables - history[i-1].dual_variables).item() 
                for i in range(1, len(history))
            ][-5:] if len(history) > 1 else []  # Last 5 changes
        }
    
    def reset(self) -> None:
        """Reset optimizer state."""
        self.dual_variables.zero_()
        self.optimization_history.clear()
        self.current_iteration = 0
        logger.info("Lagrangian optimizer reset")
        
    def save_state(self, filepath: str) -> None:
        """Save optimizer state to file."""
        state = {
            "dual_variables": self.dual_variables,
            "current_iteration": self.current_iteration,
            "optimization_history": self.optimization_history,
            "hyperparameters": {
                "dual_lr": self.dual_lr,
                "trust_region_size": self.trust_region_size,
                "dual_regularization": self.dual_regularization
            }
        }
        torch.save(state, filepath)
        logger.info(f"Lagrangian optimizer state saved to {filepath}")
        
    def load_state(self, filepath: str) -> None:
        """Load optimizer state from file."""
        state = torch.load(filepath, map_location="cpu")
        self.dual_variables = state["dual_variables"]
        self.current_iteration = state["current_iteration"] 
        self.optimization_history = state["optimization_history"]
        logger.info(f"Lagrangian optimizer state loaded from {filepath}")