"""
Trust Region Optimization Methods for CPO.

This module implements trust region constrained optimization including:
- Conjugate gradient solver for Hessian-vector products
- Line search with backtracking
- KL divergence computation and constraints
- Fisher Information Matrix approximation
"""

from typing import Dict, List, Optional, Tuple, Union, Callable, NamedTuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
import logging
from dataclasses import dataclass
import scipy.optimize
from abc import ABC, abstractmethod

from ..utils.math_utils import conjugate_gradient, line_search_backtrack
from ..utils.logging_utils import log_execution_time

logger = logging.getLogger(__name__)


class LineSearchResult(NamedTuple):
    """Result from line search optimization."""
    step_size: float
    iterations: int
    kl_divergence: float
    policy_improvement: float
    constraint_violations: torch.Tensor
    success: bool
    termination_reason: str = ""


@dataclass
class TrustRegionConfig:
    """Configuration for trust region optimization."""
    target_kl: float = 0.01              # Target KL divergence δ
    damping: float = 1e-4                # Damping coefficient for numerical stability
    cg_iters: int = 10                   # Conjugate gradient iterations
    cg_tolerance: float = 1e-8           # CG convergence tolerance
    backtrack_iters: int = 10            # Maximum line search iterations
    backtrack_ratio: float = 0.8         # Line search decay factor
    accept_ratio: float = 0.1            # Minimum improvement ratio
    max_constraint_violation: float = 0.1 # Maximum allowed constraint violation
    use_natural_gradient: bool = True     # Use natural policy gradient
    adaptive_kl_penalty: bool = True      # Adapt KL penalty based on violations


class TrustRegionSolver:
    """
    Trust region constrained optimization solver for CPO.
    
    Solves constrained optimization problems of the form:
    minimize: -g^T s
    subject to: 1/2 s^T H s ≤ δ (trust region constraint)
               b_i^T s ≤ c_i (linear inequality constraints)
    """
    
    def __init__(self, 
                 target_kl: float = 0.01,
                 damping: float = 1e-4,
                 cg_iters: int = 10,
                 backtrack_iters: int = 10,
                 accept_ratio: float = 0.1):
        """
        Initialize trust region solver.
        
        Args:
            target_kl: Target KL divergence (trust region size)
            damping: Damping coefficient for numerical stability
            cg_iters: Conjugate gradient iterations
            backtrack_iters: Line search iterations
            accept_ratio: Minimum acceptable improvement ratio
        """
        self.target_kl = target_kl
        self.damping = damping
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.accept_ratio = accept_ratio
        
        # Solver statistics
        self.total_cg_iterations = 0
        self.total_line_search_iterations = 0
        self.convergence_failures = 0
        
        logger.info(f"TrustRegionSolver initialized with target_kl={target_kl}")
    
    def solve_unconstrained(self, 
                          gradient: torch.Tensor,
                          hessian_vector_product: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Solve unconstrained trust region problem.
        
        minimize: -g^T s
        subject to: 1/2 s^T H s ≤ δ
        
        Args:
            gradient: Policy gradient g [param_dim]
            hessian_vector_product: Function computing H*v
            
        Returns:
            Tuple of (search_direction, solver_info)
        """
        with log_execution_time(logger, f"Unconstrained trust region solve"):
            # Solve H*s = g using conjugate gradient
            search_direction, cg_info = conjugate_gradient(
                hessian_vector_product, 
                gradient,
                max_iter=self.cg_iters,
                tolerance=1e-8
            )
            
            # Check trust region constraint: 1/2 s^T H s ≤ δ
            quadratic_form = 0.5 * torch.dot(search_direction, hessian_vector_product(search_direction))
            
            if quadratic_form > self.target_kl:
                # Scale down to satisfy trust region
                scale_factor = torch.sqrt(2 * self.target_kl / quadratic_form)
                search_direction = search_direction * scale_factor
                trust_region_active = True
                logger.debug(f"Trust region active: scaled by {scale_factor:.6f}")
            else:
                trust_region_active = False
            
            # Update statistics
            self.total_cg_iterations += cg_info["iterations"]
            
            solver_info = {
                "method": "unconstrained",
                "cg_iterations": cg_info["iterations"],
                "cg_converged": cg_info["converged"],
                "trust_region_active": trust_region_active,
                "quadratic_form": quadratic_form.item(),
                "gradient_norm": torch.norm(gradient).item(),
                "search_direction_norm": torch.norm(search_direction).item()
            }
            
            return search_direction, solver_info
    
    def solve_constrained(self,
                         gradient: torch.Tensor,
                         constraint_gradients: torch.Tensor,
                         constraint_violations: torch.Tensor,
                         hessian_vector_product: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Solve constrained trust region problem using dual methods.
        
        minimize: -g^T s
        subject to: 1/2 s^T H s ≤ δ
                   B^T s ≤ c (constraint violations)
        
        Args:
            gradient: Policy gradient g [param_dim]
            constraint_gradients: Constraint gradients B [num_constraints, param_dim]
            constraint_violations: Constraint violation levels c [num_constraints]
            hessian_vector_product: Function computing H*v
            
        Returns:
            Tuple of (search_direction, solver_info)
        """
        with log_execution_time(logger, f"Constrained trust region solve"):
            num_constraints = constraint_gradients.shape[0]
            param_dim = gradient.shape[0]
            
            # Check if any constraints are violated
            violated_mask = constraint_violations > 1e-6
            
            if not violated_mask.any():
                # No violated constraints - solve unconstrained problem
                return self.solve_unconstrained(gradient, hessian_vector_product)
            
            # Active constraint gradients and violations
            active_constraints = constraint_gradients[violated_mask]
            active_violations = constraint_violations[violated_mask]
            num_active = active_constraints.shape[0]
            
            logger.debug(f"Solving with {num_active} active constraints")
            
            # Dual formulation: solve for Lagrange multipliers
            try:
                search_direction, dual_vars, solver_info = self._solve_dual_problem(
                    gradient, active_constraints, active_violations,
                    hessian_vector_product
                )
            except Exception as e:
                logger.warning(f"Dual solver failed: {e}, falling back to projection method")
                search_direction, solver_info = self._solve_projection_method(
                    gradient, active_constraints, hessian_vector_product
                )
                dual_vars = torch.zeros(num_active)
            
            solver_info.update({
                "method": "constrained",
                "num_active_constraints": num_active,
                "dual_variables": dual_vars.tolist(),
                "max_constraint_violation": constraint_violations.max().item()
            })
            
            return search_direction, solver_info
    
    def _solve_dual_problem(self,
                           gradient: torch.Tensor,
                           constraint_gradients: torch.Tensor,
                           constraint_violations: torch.Tensor,
                           hessian_vector_product: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Solve constrained problem using dual optimization.
        
        The dual problem is:
        max_λ≥0,ν  -1/2 (g - B^T λ)^T H^{-1} (g - B^T λ) - λ^T c - ν δ
        subject to: 1/2 (g - B^T λ)^T H^{-1} (g - B^T λ) ≤ ν δ
        
        Args:
            gradient: Policy gradient [param_dim]
            constraint_gradients: Active constraint gradients [num_active, param_dim]
            constraint_violations: Active constraint violations [num_active]
            hessian_vector_product: Function computing H*v
            
        Returns:
            Tuple of (search_direction, dual_variables, solver_info)
        """
        num_active = constraint_gradients.shape[0]
        
        def solve_primal_given_dual(dual_vars):
            """Solve primal problem for given dual variables."""
            # Compute modified gradient: g_modified = g - B^T λ
            modified_grad = gradient - torch.mv(constraint_gradients.T, dual_vars)
            
            # Solve H * s = g_modified
            search_dir, cg_info = conjugate_gradient(
                hessian_vector_product,
                modified_grad,
                max_iter=self.cg_iters,
                tolerance=1e-8
            )
            
            # Project to trust region if needed
            quadratic_form = 0.5 * torch.dot(search_dir, hessian_vector_product(search_dir))
            
            if quadratic_form > self.target_kl:
                scale_factor = torch.sqrt(2 * self.target_kl / quadratic_form)
                search_dir = search_dir * scale_factor
            
            return search_dir, cg_info
        
        def dual_objective(dual_vars_np):
            """Dual objective function for scipy optimization."""
            dual_vars = torch.from_numpy(dual_vars_np).float()
            
            # Solve primal problem
            search_dir, _ = solve_primal_given_dual(dual_vars)
            
            # Compute dual objective
            obj = -torch.dot(gradient, search_dir) - torch.dot(dual_vars, constraint_violations)
            return -obj.item()  # Negative because scipy minimizes
        
        def dual_gradient(dual_vars_np):
            """Gradient of dual objective."""
            dual_vars = torch.from_numpy(dual_vars_np).float()
            search_dir, _ = solve_primal_given_dual(dual_vars)
            
            # Dual gradient: -B * s - c
            grad = -torch.mv(constraint_gradients, search_dir) - constraint_violations
            return -grad.numpy()  # Negative because scipy minimizes
        
        # Solve dual problem using scipy
        initial_dual = torch.ones(num_active) * 0.1
        bounds = [(0, 100) for _ in range(num_active)]  # λ ≥ 0
        
        try:
            result = scipy.optimize.minimize(
                dual_objective,
                initial_dual.numpy(),
                method='L-BFGS-B',
                jac=dual_gradient,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-9}
            )
            
            optimal_dual = torch.from_numpy(result.x).float()
            success = result.success
            
        except Exception as e:
            logger.warning(f"Scipy optimization failed: {e}")
            optimal_dual = initial_dual
            success = False
        
        # Compute final search direction
        search_direction, cg_info = solve_primal_given_dual(optimal_dual)
        
        # Verify constraints
        constraint_products = torch.mv(constraint_gradients, search_direction)
        constraint_satisfied = torch.all(constraint_products <= constraint_violations + 1e-6)
        
        solver_info = {
            "dual_optimization_success": success,
            "cg_iterations": cg_info["iterations"],
            "constraint_satisfied": constraint_satisfied.item(),
            "dual_objective_value": -dual_objective(optimal_dual.numpy()),
            "constraint_products": constraint_products.tolist()
        }
        
        return search_direction, optimal_dual, solver_info
    
    def _solve_projection_method(self,
                               gradient: torch.Tensor,
                               constraint_gradients: torch.Tensor,
                               hessian_vector_product: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fallback projection method for constrained optimization.
        
        Uses projected gradient approach when dual method fails.
        """
        # Start with unconstrained solution
        search_direction, unconstrained_info = self.solve_unconstrained(gradient, hessian_vector_product)
        
        # Project onto constraint feasible region
        max_iterations = 10
        for iteration in range(max_iterations):
            # Check constraint violations
            constraint_products = torch.mv(constraint_gradients, search_direction)
            violations = constraint_products > 1e-6
            
            if not violations.any():
                break  # All constraints satisfied
            
            # Project onto violated constraints
            for i, violated in enumerate(violations):
                if violated:
                    # Project: s = s - (b_i^T s / ||b_i||^2) * b_i
                    constraint_grad = constraint_gradients[i]
                    projection_coeff = torch.dot(constraint_grad, search_direction) / torch.dot(constraint_grad, constraint_grad)
                    search_direction = search_direction - projection_coeff * constraint_grad
        
        solver_info = {
            "projection_iterations": iteration + 1,
            "final_constraint_violations": constraint_products.tolist()
        }
        
        return search_direction, solver_info


class AdaptiveKLPenalty:
    """
    Adaptive KL penalty for trust region methods.
    
    Automatically adjusts KL penalty coefficient based on constraint violations
    and policy performance.
    """
    
    def __init__(self, 
                 initial_penalty: float = 1.0,
                 min_penalty: float = 0.1,
                 max_penalty: float = 10.0,
                 adaptation_rate: float = 1.1):
        """
        Initialize adaptive KL penalty.
        
        Args:
            initial_penalty: Initial penalty coefficient
            min_penalty: Minimum penalty value
            max_penalty: Maximum penalty value  
            adaptation_rate: Rate of penalty adaptation
        """
        self.penalty = initial_penalty
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.adaptation_rate = adaptation_rate
        
        # History for adaptation
        self.violation_history = []
        self.kl_history = []
        
    def update_penalty(self, 
                      constraint_violations: torch.Tensor,
                      kl_divergence: float,
                      target_kl: float) -> float:
        """
        Update penalty based on constraint violations and KL divergence.
        
        Args:
            constraint_violations: Current constraint violation levels
            kl_divergence: Current KL divergence
            target_kl: Target KL divergence
            
        Returns:
            Updated penalty coefficient
        """
        self.violation_history.append(constraint_violations.max().item())
        self.kl_history.append(kl_divergence)
        
        # Keep recent history
        if len(self.violation_history) > 10:
            self.violation_history = self.violation_history[-10:]
            self.kl_history = self.kl_history[-10:]
        
        # Adaptation logic
        recent_violations = np.mean(self.violation_history[-3:])
        recent_kl = np.mean(self.kl_history[-3:])
        
        if recent_violations > 0.05:  # High violations
            if recent_kl < target_kl * 0.5:  # Low KL usage
                # Increase penalty to be more conservative
                self.penalty = min(self.penalty * self.adaptation_rate, self.max_penalty)
            elif recent_kl > target_kl * 1.5:  # High KL
                # Decrease penalty to allow more exploration
                self.penalty = max(self.penalty / self.adaptation_rate, self.min_penalty)
        elif recent_violations < 0.01:  # Low violations
            if recent_kl < target_kl * 0.3:  # Very low KL usage
                # Decrease penalty to encourage more exploration
                self.penalty = max(self.penalty / self.adaptation_rate, self.min_penalty)
        
        return self.penalty


class LineSearch:
    """
    Backtracking line search with constraint and KL checks.
    """
    
    def __init__(self,
                 max_iterations: int = 10,
                 backtrack_ratio: float = 0.8,
                 accept_ratio: float = 0.1,
                 target_kl: float = 0.01):
        """
        Initialize line search.
        
        Args:
            max_iterations: Maximum backtracking iterations
            backtrack_ratio: Step size reduction factor
            accept_ratio: Minimum improvement acceptance ratio
            target_kl: Target KL divergence constraint
        """
        self.max_iterations = max_iterations
        self.backtrack_ratio = backtrack_ratio
        self.accept_ratio = accept_ratio
        self.target_kl = target_kl
        
    def search(self,
               policy_eval_func: Callable[[float], Dict[str, float]],
               search_direction: torch.Tensor,
               initial_step_size: float = 1.0) -> LineSearchResult:
        """
        Perform backtracking line search.
        
        Args:
            policy_eval_func: Function that evaluates policy at given step size
                             Returns dict with 'policy_loss', 'kl_div', 'constraint_violations'
            search_direction: Search direction from trust region solver
            initial_step_size: Initial step size to try
            
        Returns:
            Line search result
        """
        step_size = initial_step_size
        
        # Baseline evaluation
        baseline = policy_eval_func(0.0)
        baseline_loss = baseline['policy_loss']
        
        for iteration in range(self.max_iterations):
            # Evaluate at current step size
            result = policy_eval_func(step_size)
            
            policy_loss = result['policy_loss']
            kl_div = result['kl_div']
            constraint_violations = result['constraint_violations']
            
            # Check acceptance criteria
            improvement = baseline_loss - policy_loss
            expected_improvement = -step_size * torch.norm(search_direction)**2 * self.accept_ratio
            
            kl_ok = kl_div <= self.target_kl * 1.2  # Allow some slack
            improvement_ok = improvement >= expected_improvement
            constraint_ok = torch.max(constraint_violations) <= torch.max(baseline['constraint_violations']) + 0.01
            
            if kl_ok and improvement_ok and constraint_ok:
                return LineSearchResult(
                    step_size=step_size,
                    iterations=iteration + 1,
                    kl_divergence=kl_div,
                    policy_improvement=improvement,
                    constraint_violations=constraint_violations,
                    success=True,
                    termination_reason="accepted"
                )
            
            # Log why step was rejected
            reasons = []
            if not kl_ok:
                reasons.append(f"KL too high ({kl_div:.6f} > {self.target_kl * 1.2:.6f})")
            if not improvement_ok:
                reasons.append(f"Insufficient improvement ({improvement:.6f} < {expected_improvement:.6f})")
            if not constraint_ok:
                reasons.append("Constraint violations increased")
            
            logger.debug(f"Line search iteration {iteration}: step_size={step_size:.6f}, rejected: {', '.join(reasons)}")
            
            # Reduce step size
            step_size *= self.backtrack_ratio
        
        # Return failed result
        final_result = policy_eval_func(step_size)
        return LineSearchResult(
            step_size=step_size,
            iterations=self.max_iterations,
            kl_divergence=final_result['kl_div'],
            policy_improvement=baseline_loss - final_result['policy_loss'],
            constraint_violations=final_result['constraint_violations'],
            success=False,
            termination_reason="max_iterations_reached"
        )


class FisherInformationEstimator:
    """
    Estimates Fisher Information Matrix and computes matrix-vector products.
    """
    
    def __init__(self, 
                 damping: float = 1e-4,
                 use_empirical_fisher: bool = False):
        """
        Initialize Fisher Information estimator.
        
        Args:
            damping: Damping coefficient for numerical stability
            use_empirical_fisher: Use empirical Fisher (based on actual gradients)
                                rather than expected Fisher (based on score function)
        """
        self.damping = damping
        self.use_empirical_fisher = use_empirical_fisher
        
    def compute_fvp(self,
                    policy,
                    states: torch.Tensor,
                    vector: torch.Tensor) -> torch.Tensor:
        """
        Compute Fisher Information Matrix-vector product F*v.
        
        Args:
            policy: Policy network
            states: State batch for computing Fisher matrix
            vector: Vector to multiply [param_dim]
            
        Returns:
            Fisher-vector product [param_dim]
        """
        if self.use_empirical_fisher:
            return self._compute_empirical_fvp(policy, states, vector)
        else:
            return self._compute_expected_fvp(policy, states, vector)
    
    def _compute_expected_fvp(self,
                             policy,
                             states: torch.Tensor,
                             vector: torch.Tensor) -> torch.Tensor:
        """
        Compute expected Fisher-vector product using score function.
        
        F = E[∇log π(a|s) ∇log π(a|s)^T]
        """
        # Sample actions from current policy
        with torch.no_grad():
            dist = policy.get_distribution(states)
            actions = dist.sample()
        
        # Compute log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1).mean()
        
        # First-order gradient of log probability
        log_prob_grad = torch.autograd.grad(
            log_probs, policy.parameters(),
            create_graph=True, retain_graph=True
        )
        log_prob_grad_flat = torch.cat([g.view(-1) for g in log_prob_grad])
        
        # Gradient-vector product
        gvp = torch.dot(log_prob_grad_flat, vector)
        
        # Hessian-vector product (second-order)
        hvp = torch.autograd.grad(
            gvp, policy.parameters(),
            retain_graph=True
        )
        hvp_flat = torch.cat([h.view(-1) for h in hvp])
        
        # Add damping
        return hvp_flat + self.damping * vector
    
    def _compute_empirical_fvp(self,
                              policy,
                              states: torch.Tensor,
                              vector: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical Fisher-vector product using actual gradients.
        """
        # This requires storing gradients from actual policy updates
        # For now, fallback to expected Fisher
        logger.warning("Empirical Fisher not fully implemented, using expected Fisher")
        return self._compute_expected_fvp(policy, states, vector)
    
    def estimate_fisher_diagonal(self,
                               policy,
                               states: torch.Tensor,
                               num_samples: int = 100) -> torch.Tensor:
        """
        Estimate diagonal of Fisher Information Matrix.
        
        Args:
            policy: Policy network
            states: State batch
            num_samples: Number of samples for estimation
            
        Returns:
            Diagonal elements of Fisher matrix [param_dim]
        """
        param_count = sum(p.numel() for p in policy.parameters())
        fisher_diagonal = torch.zeros(param_count)
        
        for _ in range(num_samples):
            # Sample random vector
            random_vector = torch.randn(param_count)
            
            # Compute F*v
            fvp = self.compute_fvp(policy, states, random_vector)
            
            # Accumulate diagonal estimate
            fisher_diagonal += random_vector * fvp
        
        fisher_diagonal /= num_samples
        return torch.abs(fisher_diagonal)  # Take absolute value for diagonal elements
    
    def condition_number_estimate(self,
                                policy,
                                states: torch.Tensor,
                                num_vectors: int = 10) -> float:
        """
        Estimate condition number of Fisher Information Matrix.
        
        Args:
            policy: Policy network
            states: State batch
            num_vectors: Number of random vectors for estimation
            
        Returns:
            Estimated condition number
        """
        param_count = sum(p.numel() for p in policy.parameters())
        eigenvalue_estimates = []
        
        for _ in range(num_vectors):
            # Random vector
            v = torch.randn(param_count)
            v = v / torch.norm(v)
            
            # Power iteration step
            fv = self.compute_fvp(policy, states, v)
            eigenvalue = torch.dot(v, fv).item()
            eigenvalue_estimates.append(max(eigenvalue, 1e-8))  # Avoid division by zero
        
        max_eigenvalue = max(eigenvalue_estimates)
        min_eigenvalue = min(eigenvalue_estimates)
        
        condition_number = max_eigenvalue / min_eigenvalue
        return condition_number