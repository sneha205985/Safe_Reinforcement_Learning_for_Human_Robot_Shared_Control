"""
Mathematical utility functions for Safe RL optimization.

This module provides mathematical operations commonly used in constrained
optimization, gradient computation, and policy evaluation.
"""

from typing import Callable, Optional, Tuple, Union, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
import scipy.optimize
from scipy import linalg
import logging

logger = logging.getLogger(__name__)


def finite_difference_gradient(func: Callable[[torch.Tensor], torch.Tensor],
                             x: torch.Tensor,
                             eps: float = 1e-6,
                             method: str = "central") -> torch.Tensor:
    """
    Compute gradient using finite differences for validation.
    
    Args:
        func: Function to differentiate f: R^n -> R
        x: Point at which to evaluate gradient [n]
        eps: Step size for finite differences
        method: "forward", "backward", or "central"
        
    Returns:
        Gradient vector [n]
    """
    x = x.clone().detach()
    grad = torch.zeros_like(x)
    
    if method == "forward":
        f_x = func(x)
        for i in range(len(x)):
            x_plus = x.clone()
            x_plus[i] += eps
            f_x_plus = func(x_plus)
            grad[i] = (f_x_plus - f_x) / eps
            
    elif method == "backward":
        f_x = func(x)
        for i in range(len(x)):
            x_minus = x.clone()
            x_minus[i] -= eps
            f_x_minus = func(x_minus)
            grad[i] = (f_x - f_x_minus) / eps
            
    elif method == "central":
        for i in range(len(x)):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[i] += eps
            x_minus[i] -= eps
            
            f_x_plus = func(x_plus)
            f_x_minus = func(x_minus)
            grad[i] = (f_x_plus - f_x_minus) / (2 * eps)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return grad


def compute_kl_divergence(p_mean: torch.Tensor, p_std: torch.Tensor,
                         q_mean: torch.Tensor, q_std: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between two multivariate normal distributions.
    
    KL(P||Q) where P ~ N(p_mean, diag(p_std²)) and Q ~ N(q_mean, diag(q_std²))
    
    Args:
        p_mean: Mean of distribution P [batch_size, dim]
        p_std: Standard deviation of distribution P [batch_size, dim]  
        q_mean: Mean of distribution Q [batch_size, dim]
        q_std: Standard deviation of distribution Q [batch_size, dim]
        
    Returns:
        KL divergence [batch_size]
    """
    p_dist = Normal(p_mean, p_std)
    q_dist = Normal(q_mean, q_std)
    return kl_divergence(p_dist, q_dist).sum(dim=-1)


def conjugate_gradient(Ax_func: Callable[[torch.Tensor], torch.Tensor],
                      b: torch.Tensor,
                      x0: Optional[torch.Tensor] = None,
                      max_iter: int = 100,
                      tolerance: float = 1e-8,
                      residual_tolerance: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Solve linear system Ax = b using conjugate gradient method.
    
    Args:
        Ax_func: Function that computes matrix-vector product Ax
        b: Right-hand side vector [n]
        x0: Initial guess [n] (zero if None)
        max_iter: Maximum iterations
        tolerance: Convergence tolerance for x
        residual_tolerance: Convergence tolerance for residual
        
    Returns:
        Tuple of (solution, info_dict)
    """
    n = len(b)
    x = x0.clone() if x0 is not None else torch.zeros_like(b)
    
    # Initial residual
    r = b - Ax_func(x)
    p = r.clone()
    r_dot_old = torch.dot(r, r)
    
    info = {"iterations": 0, "residual_norm": [], "converged": False}
    
    for iteration in range(max_iter):
        # Matrix-vector product
        Ap = Ax_func(p)
        
        # Step size
        alpha = r_dot_old / (torch.dot(p, Ap) + 1e-10)
        
        # Update solution
        x_new = x + alpha * p
        
        # Update residual
        r = r - alpha * Ap
        r_dot_new = torch.dot(r, r)
        
        # Store residual norm
        residual_norm = torch.sqrt(r_dot_new).item()
        info["residual_norm"].append(residual_norm)
        
        # Check convergence
        solution_change = torch.norm(x_new - x).item()
        if solution_change < tolerance or residual_norm < residual_tolerance:
            info["converged"] = True
            break
        
        # Update for next iteration
        beta = r_dot_new / (r_dot_old + 1e-10)
        p = r + beta * p
        r_dot_old = r_dot_new
        x = x_new
    
    info["iterations"] = iteration + 1
    info["final_residual"] = info["residual_norm"][-1] if info["residual_norm"] else float('inf')
    
    if not info["converged"]:
        logger.warning(f"CG did not converge after {max_iter} iterations. "
                      f"Final residual: {info['final_residual']:.2e}")
    
    return x, info


def line_search_backtrack(func: Callable[[torch.Tensor], torch.Tensor],
                         grad: torch.Tensor,
                         x: torch.Tensor,
                         direction: torch.Tensor,
                         initial_step: float = 1.0,
                         c1: float = 1e-4,
                         rho: float = 0.5,
                         max_iter: int = 20) -> Tuple[float, int]:
    """
    Backtracking line search with Armijo condition.
    
    Args:
        func: Objective function f: R^n -> R
        grad: Gradient at current point [n]
        x: Current point [n]
        direction: Search direction [n]
        initial_step: Initial step size
        c1: Armijo condition parameter
        rho: Step size reduction factor
        max_iter: Maximum line search iterations
        
    Returns:
        Tuple of (step_size, iterations)
    """
    step_size = initial_step
    f_x = func(x)
    grad_dot_dir = torch.dot(grad, direction).item()
    
    # Armijo condition: f(x + α*d) ≤ f(x) + c1*α*∇f(x)ᵀd
    for iteration in range(max_iter):
        x_new = x + step_size * direction
        f_x_new = func(x_new)
        
        if f_x_new <= f_x + c1 * step_size * grad_dot_dir:
            break
            
        step_size *= rho
    
    return step_size, iteration + 1


def compute_fisher_vector_product(policy_func: Callable[[torch.Tensor], torch.distributions.Distribution],
                                states: torch.Tensor,
                                vector: torch.Tensor,
                                damping: float = 1e-4) -> torch.Tensor:
    """
    Compute Fisher Information Matrix-vector product efficiently.
    
    FIM is the covariance matrix of the policy gradients:
    F = E[∇log π(a|s) ∇log π(a|s)ᵀ]
    
    Args:
        policy_func: Function that returns policy distribution given states
        states: State batch [batch_size, state_dim]
        vector: Vector to multiply with FIM [param_dim]
        damping: Damping coefficient for numerical stability
        
    Returns:
        FIM-vector product [param_dim]
    """
    # Get policy distribution
    dist = policy_func(states)
    
    # Sample actions and compute log probabilities
    actions = dist.sample()
    log_probs = dist.log_prob(actions).sum(dim=-1).mean()
    
    # First-order gradient
    grad = torch.autograd.grad(log_probs, policy_func.parameters(), create_graph=True)
    grad_flat = torch.cat([g.view(-1) for g in grad])
    
    # Gradient-vector product
    gvp = torch.dot(grad_flat, vector)
    
    # Second-order gradient (Hessian-vector product)
    hvp = torch.autograd.grad(gvp, policy_func.parameters(), retain_graph=True)
    hvp_flat = torch.cat([h.view(-1) for h in hvp])
    
    # Add damping for numerical stability
    return hvp_flat + damping * vector


def compute_natural_gradient(policy_func: Callable[[torch.Tensor], torch.distributions.Distribution],
                           states: torch.Tensor,
                           policy_gradient: torch.Tensor,
                           cg_iterations: int = 10,
                           damping: float = 1e-4) -> torch.Tensor:
    """
    Compute natural policy gradient using conjugate gradient.
    
    Natural gradient: F⁻¹ g where F is Fisher Information Matrix and g is policy gradient.
    
    Args:
        policy_func: Function returning policy distribution
        states: State batch [batch_size, state_dim]
        policy_gradient: Policy gradient vector [param_dim]
        cg_iterations: CG iterations for solving F⁻¹g
        damping: Damping coefficient
        
    Returns:
        Natural gradient vector [param_dim]
    """
    def fvp_func(v):
        return compute_fisher_vector_product(policy_func, states, v, damping)
    
    natural_grad, info = conjugate_gradient(
        fvp_func, policy_gradient,
        max_iter=cg_iterations,
        tolerance=1e-8
    )
    
    return natural_grad


def project_onto_simplex(v: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Project vector onto probability simplex.
    
    Projects v onto {x : x ≥ 0, sum(x) = radius}
    
    Args:
        v: Input vector [n]
        radius: Simplex radius (sum constraint)
        
    Returns:
        Projected vector [n]
    """
    n = len(v)
    
    # Sort in descending order
    u, _ = torch.sort(v, descending=True)
    
    # Find threshold
    cumsum = torch.cumsum(u, dim=0)
    k = torch.arange(1, n + 1, dtype=v.dtype, device=v.device)
    
    # Compute threshold candidates
    threshold_candidates = (cumsum - radius) / k
    
    # Find largest k such that u[k] > threshold[k]
    valid = u > threshold_candidates
    if valid.any():
        k_max = valid.nonzero()[-1].item()
        threshold = threshold_candidates[k_max]
    else:
        threshold = (cumsum[-1] - radius) / n
    
    # Project
    return torch.clamp(v - threshold, min=0.0)


def compute_quadratic_approximation(func: Callable[[torch.Tensor], torch.Tensor],
                                  x: torch.Tensor,
                                  direction: torch.Tensor,
                                  step_size: float = 1e-3) -> Tuple[float, float, float]:
    """
    Compute quadratic approximation of function along direction.
    
    f(x + α*d) ≈ a + b*α + c*α²
    
    Args:
        func: Function to approximate
        x: Current point [n]
        direction: Direction vector [n]  
        step_size: Step size for finite differences
        
    Returns:
        Tuple of (a, b, c) quadratic coefficients
    """
    # Evaluate at three points
    f0 = func(x).item()
    f1 = func(x + step_size * direction).item()
    f2 = func(x + 2 * step_size * direction).item()
    
    # Solve for quadratic coefficients
    # f(α) = a + b*α + c*α²
    # f(0) = a
    # f(h) = a + b*h + c*h²
    # f(2h) = a + b*2h + c*4h²
    
    a = f0
    b = (f1 - f0) / step_size
    c = (f2 - 2*f1 + f0) / (2 * step_size**2)
    
    return a, b, c


def trust_region_subproblem_solver(grad: torch.Tensor,
                                  hessian_func: Callable[[torch.Tensor], torch.Tensor],
                                  radius: float,
                                  max_iter: int = 100,
                                  tolerance: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Solve trust region subproblem using conjugate gradient method.
    
    min_p (gᵀp + 0.5 * pᵀHp) subject to ||p|| ≤ radius
    
    Args:
        grad: Gradient vector [n]
        hessian_func: Function computing Hessian-vector product
        radius: Trust region radius
        max_iter: Maximum CG iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (solution, info)
    """
    n = len(grad)
    p = torch.zeros_like(grad)
    r = grad.clone()
    d = -r.clone()
    
    info = {
        "iterations": 0,
        "boundary_solution": False,
        "negative_curvature": False,
        "converged": False
    }
    
    for iteration in range(max_iter):
        # Hessian-vector product
        Hd = hessian_func(d)
        dHd = torch.dot(d, Hd).item()
        
        # Check for negative curvature
        if dHd <= 0:
            # Find boundary solution
            a = torch.dot(d, d).item()
            b = 2 * torch.dot(p, d).item()
            c = torch.dot(p, p).item() - radius**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                tau1 = (-b + np.sqrt(discriminant)) / (2*a)
                tau2 = (-b - np.sqrt(discriminant)) / (2*a)
                
                # Choose tau that gives lower objective
                p1 = p + tau1 * d
                p2 = p + tau2 * d
                
                obj1 = torch.dot(grad, p1) + 0.5 * torch.dot(p1, hessian_func(p1))
                obj2 = torch.dot(grad, p2) + 0.5 * torch.dot(p2, hessian_func(p2))
                
                p = p1 if obj1 < obj2 else p2
            
            info["negative_curvature"] = True
            break
        
        # Standard CG step
        r_dot_old = torch.dot(r, r)
        alpha = r_dot_old / dHd
        
        # Check if step would violate trust region
        p_new = p + alpha * d
        if torch.norm(p_new) > radius:
            # Find boundary intersection
            a = torch.dot(d, d).item()
            b = 2 * torch.dot(p, d).item()
            c = torch.dot(p, p).item() - radius**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                tau = (-b + np.sqrt(discriminant)) / (2*a)
                p = p + tau * d
            
            info["boundary_solution"] = True
            break
        
        # Update solution and residual
        p = p_new
        r = r + alpha * Hd
        r_dot_new = torch.dot(r, r)
        
        # Check convergence
        if torch.sqrt(r_dot_new) < tolerance:
            info["converged"] = True
            break
        
        # Update search direction
        beta = r_dot_new / r_dot_old
        d = -r + beta * d
    
    info["iterations"] = iteration + 1
    info["final_norm"] = torch.norm(p).item()
    info["radius_ratio"] = info["final_norm"] / radius
    
    return p, info


def compute_constraint_linearization(constraint_func: Callable[[torch.Tensor], torch.Tensor],
                                   x: torch.Tensor,
                                   direction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute linearization of constraint: g(x + d) ≈ g(x) + ∇g(x)ᵀd
    
    Args:
        constraint_func: Constraint function g: R^n -> R^m
        x: Current point [n]
        direction: Direction vector [n]
        
    Returns:
        Tuple of (constraint_value, predicted_change)
    """
    x = x.requires_grad_(True)
    
    # Evaluate constraint at current point
    g_x = constraint_func(x)
    
    # Compute constraint gradient
    grad_g = torch.autograd.grad(
        outputs=g_x.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Predicted change in constraint
    predicted_change = torch.dot(grad_g, direction)
    
    return g_x, predicted_change


def safe_cholesky_decomposition(A: torch.Tensor, regularization: float = 1e-6) -> torch.Tensor:
    """
    Compute Cholesky decomposition with regularization for numerical stability.
    
    Args:
        A: Symmetric positive definite matrix [n, n]
        regularization: Regularization parameter
        
    Returns:
        Lower triangular Cholesky factor [n, n]
    """
    # Add regularization to diagonal
    A_reg = A + regularization * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    
    try:
        L = torch.cholesky(A_reg)
        return L
    except RuntimeError:
        # Fallback: use eigendecomposition
        eigenvals, eigenvecs = torch.symeig(A_reg, eigenvectors=True)
        eigenvals = torch.clamp(eigenvals, min=regularization)
        A_regularized = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
        return torch.cholesky(A_regularized)


def compute_policy_hessian(policy_func: Callable[[torch.Tensor], torch.distributions.Distribution],
                          states: torch.Tensor,
                          actions: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian of policy log probability.
    
    Args:
        policy_func: Policy function
        states: State batch [batch_size, state_dim]
        actions: Action batch [batch_size, action_dim]
        
    Returns:
        Hessian matrix [param_dim, param_dim]
    """
    # Get policy distribution and log probability
    dist = policy_func(states)
    log_prob = dist.log_prob(actions).sum()
    
    # First-order gradients
    first_grads = torch.autograd.grad(log_prob, policy_func.parameters(), create_graph=True)
    first_grads_flat = torch.cat([g.view(-1) for g in first_grads])
    
    # Compute Hessian
    hessian_rows = []
    for i in range(len(first_grads_flat)):
        grad_grad = torch.autograd.grad(
            first_grads_flat[i], policy_func.parameters(), retain_graph=True
        )
        hessian_row = torch.cat([gg.view(-1) for gg in grad_grad])
        hessian_rows.append(hessian_row)
    
    hessian = torch.stack(hessian_rows)
    return hessian


def numerical_gradient_check(func: Callable[[torch.Tensor], torch.Tensor],
                           grad_func: Callable[[torch.Tensor], torch.Tensor],
                           x: torch.Tensor,
                           eps: float = 1e-6,
                           tolerance: float = 1e-4) -> Dict[str, Any]:
    """
    Check analytical gradient against numerical gradient.
    
    Args:
        func: Function f: R^n -> R
        grad_func: Gradient function ∇f: R^n -> R^n
        x: Point to check [n]
        eps: Finite difference step size
        tolerance: Tolerance for gradient check
        
    Returns:
        Dictionary with check results
    """
    # Analytical gradient
    analytical_grad = grad_func(x).detach()
    
    # Numerical gradient
    numerical_grad = finite_difference_gradient(func, x, eps, method="central")
    
    # Compute errors
    abs_error = torch.norm(analytical_grad - numerical_grad)
    rel_error = abs_error / (torch.norm(analytical_grad) + 1e-8)
    
    # Element-wise comparison
    element_errors = torch.abs(analytical_grad - numerical_grad)
    max_element_error = torch.max(element_errors)
    
    results = {
        "passed": abs_error < tolerance and rel_error < tolerance,
        "absolute_error": abs_error.item(),
        "relative_error": rel_error.item(),
        "max_element_error": max_element_error.item(),
        "tolerance": tolerance,
        "analytical_grad": analytical_grad,
        "numerical_grad": numerical_grad
    }
    
    return results