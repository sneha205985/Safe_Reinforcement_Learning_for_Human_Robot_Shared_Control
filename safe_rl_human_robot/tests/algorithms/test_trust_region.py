"""
Tests for trust region optimization methods.

Tests mathematical correctness of conjugate gradient solver, line search,
KL divergence computation, and Fisher Information Matrix operations.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Callable
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from algorithms.trust_region import (
    TrustRegionSolver, LineSearchResult, AdaptiveKLPenalty,
    LineSearch, FisherInformationEstimator, TrustRegionConfig
)
from core.policy import SafePolicy
from utils.math_utils import finite_difference_gradient, conjugate_gradient


class TestTrustRegionConfig:
    """Test trust region configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrustRegionConfig()
        
        assert config.target_kl == 0.01
        assert config.damping == 1e-4
        assert config.cg_iters == 10
        assert config.backtrack_iters == 10
        assert config.accept_ratio == 0.1
    
    def test_config_modification(self):
        """Test configuration modification."""
        config = TrustRegionConfig()
        config.target_kl = 0.02
        config.damping = 1e-3
        
        assert config.target_kl == 0.02
        assert config.damping == 1e-3


class TestTrustRegionSolver:
    """Test trust region constrained optimization solver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = TrustRegionSolver(
            target_kl=0.01,
            damping=1e-4,
            cg_iters=10,
            backtrack_iters=5,
            accept_ratio=0.1
        )
        
        # Create a simple quadratic Hessian for testing
        self.param_dim = 10
        self.hessian_matrix = torch.eye(self.param_dim) + 0.1 * torch.randn(self.param_dim, self.param_dim)
        self.hessian_matrix = self.hessian_matrix @ self.hessian_matrix.T + self.solver.damping * torch.eye(self.param_dim)
        
    def _quadratic_hessian_vector_product(self, vector: torch.Tensor) -> torch.Tensor:
        """Simple quadratic Hessian-vector product for testing."""
        return torch.mv(self.hessian_matrix, vector)
    
    def test_solver_initialization(self):
        """Test solver initialization."""
        assert self.solver.target_kl == 0.01
        assert self.solver.damping == 1e-4
        assert self.solver.cg_iters == 10
        assert self.solver.total_cg_iterations == 0
        assert self.solver.total_line_search_iterations == 0
    
    def test_unconstrained_trust_region(self):
        """Test unconstrained trust region problem solving."""
        # Create gradient
        gradient = torch.randn(self.param_dim)
        
        # Solve trust region problem
        search_direction, solver_info = self.solver.solve_unconstrained(
            gradient, self._quadratic_hessian_vector_product
        )
        
        # Check solution properties
        assert search_direction.shape == gradient.shape
        assert torch.isfinite(search_direction).all()
        
        # Check solver info
        assert isinstance(solver_info, dict)
        assert "method" in solver_info
        assert solver_info["method"] == "unconstrained"
        assert "cg_iterations" in solver_info
        assert "trust_region_active" in solver_info
        assert "quadratic_form" in solver_info
        
        # Check trust region constraint
        quadratic_form = 0.5 * torch.dot(search_direction, self._quadratic_hessian_vector_product(search_direction))
        assert quadratic_form <= self.solver.target_kl + 1e-6  # Allow small numerical error
        
        # For well-conditioned problems, should satisfy optimality conditions
        if not solver_info["trust_region_active"]:
            # Should satisfy H*s = g (approximately)
            residual = self._quadratic_hessian_vector_product(search_direction) - gradient
            assert torch.norm(residual) < 1e-3
    
    def test_constrained_trust_region(self):
        """Test constrained trust region problem solving."""
        gradient = torch.randn(self.param_dim)
        
        # Create constraint gradients (2 constraints)
        constraint_gradients = torch.randn(2, self.param_dim)
        constraint_gradients = constraint_gradients / torch.norm(constraint_gradients, dim=1, keepdim=True)  # Normalize
        
        # Create constraint violations (one violated, one satisfied)
        constraint_violations = torch.tensor([0.1, -0.05])
        
        search_direction, solver_info = self.solver.solve_constrained(
            gradient, constraint_gradients, constraint_violations,
            self._quadratic_hessian_vector_product
        )
        
        assert search_direction.shape == gradient.shape
        assert torch.isfinite(search_direction).all()
        
        # Check solver info
        assert solver_info["method"] == "constrained"
        assert "num_active_constraints" in solver_info
        assert solver_info["num_active_constraints"] >= 0
        
        # Check constraint satisfaction (approximately)
        constraint_products = torch.mv(constraint_gradients, search_direction)
        active_violations = constraint_violations[constraint_violations > 1e-6]
        active_products = constraint_products[constraint_violations > 1e-6]
        
        # Active constraints should be approximately satisfied
        for product, violation in zip(active_products, active_violations):
            assert product <= violation + 1e-3  # Allow small violation
    
    def test_dual_problem_solving(self):
        """Test dual problem solving for constrained optimization."""
        gradient = torch.randn(self.param_dim)
        constraint_gradients = torch.randn(1, self.param_dim)  # Single constraint
        constraint_violations = torch.tensor([0.1])  # Violated
        
        # Test dual problem solving
        search_direction, dual_vars, solver_info = self.solver._solve_dual_problem(
            gradient, constraint_gradients, constraint_violations,
            self._quadratic_hessian_vector_product
        )
        
        assert search_direction.shape == gradient.shape
        assert dual_vars.shape == (1,)
        assert dual_vars >= 0  # Dual variables should be non-negative
        
        # Check solver info
        assert isinstance(solver_info, dict)
        assert "dual_optimization_success" in solver_info
    
    def test_projection_method_fallback(self):
        """Test projection method as fallback for dual solver."""
        gradient = torch.randn(self.param_dim)
        constraint_gradients = torch.randn(2, self.param_dim)
        
        search_direction, solver_info = self.solver._solve_projection_method(
            gradient, constraint_gradients, self._quadratic_hessian_vector_product
        )
        
        assert search_direction.shape == gradient.shape
        assert torch.isfinite(search_direction).all()
        assert isinstance(solver_info, dict)
        assert "projection_iterations" in solver_info
    
    def test_trust_region_scaling(self):
        """Test trust region scaling behavior."""
        # Large gradient that should trigger trust region scaling
        large_gradient = torch.randn(self.param_dim) * 10
        
        search_direction, solver_info = self.solver.solve_unconstrained(
            large_gradient, self._quadratic_hessian_vector_product
        )
        
        # Should be scaled to satisfy trust region
        quadratic_form = 0.5 * torch.dot(search_direction, self._quadratic_hessian_vector_product(search_direction))
        assert quadratic_form <= self.solver.target_kl + 1e-6
        
        # Should be marked as trust region active
        assert solver_info["trust_region_active"]
    
    def test_solver_statistics_tracking(self):
        """Test solver statistics tracking."""
        initial_cg_iters = self.solver.total_cg_iterations
        
        gradient = torch.randn(self.param_dim)
        self.solver.solve_unconstrained(gradient, self._quadratic_hessian_vector_product)
        
        # Statistics should be updated
        assert self.solver.total_cg_iterations > initial_cg_iters


class TestAdaptiveKLPenalty:
    """Test adaptive KL penalty mechanism."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adaptive_penalty = AdaptiveKLPenalty(
            initial_penalty=1.0,
            min_penalty=0.1,
            max_penalty=10.0,
            adaptation_rate=1.1
        )
    
    def test_penalty_initialization(self):
        """Test penalty initialization."""
        assert self.adaptive_penalty.penalty == 1.0
        assert self.adaptive_penalty.min_penalty == 0.1
        assert self.adaptive_penalty.max_penalty == 10.0
        assert len(self.adaptive_penalty.violation_history) == 0
        assert len(self.adaptive_penalty.kl_history) == 0
    
    def test_penalty_increase_on_violations(self):
        """Test penalty increase when violations are high."""
        target_kl = 0.01
        
        # Simulate high violations with low KL usage
        for _ in range(5):
            constraint_violations = torch.tensor([0.1, 0.2])  # High violations
            kl_divergence = 0.002  # Low KL usage
            
            updated_penalty = self.adaptive_penalty.update_penalty(
                constraint_violations, kl_divergence, target_kl
            )
        
        # Penalty should increase
        assert updated_penalty > 1.0
        assert updated_penalty <= self.adaptive_penalty.max_penalty
    
    def test_penalty_decrease_on_safe_behavior(self):
        """Test penalty decrease when violations are low."""
        target_kl = 0.01
        
        # First increase penalty
        for _ in range(3):
            self.adaptive_penalty.update_penalty(torch.tensor([0.1]), 0.002, target_kl)
        
        initial_penalty = self.adaptive_penalty.penalty
        
        # Then simulate safe behavior with low KL usage
        for _ in range(5):
            constraint_violations = torch.tensor([0.005])  # Low violations
            kl_divergence = 0.001  # Very low KL usage
            
            updated_penalty = self.adaptive_penalty.update_penalty(
                constraint_violations, kl_divergence, target_kl
            )
        
        # Penalty should decrease
        assert updated_penalty < initial_penalty
        assert updated_penalty >= self.adaptive_penalty.min_penalty
    
    def test_penalty_bounds_enforcement(self):
        """Test that penalty stays within bounds."""
        target_kl = 0.01
        
        # Try to push penalty very high
        for _ in range(20):
            constraint_violations = torch.tensor([1.0])  # Very high violations
            kl_divergence = 0.001  # Low KL
            
            penalty = self.adaptive_penalty.update_penalty(
                constraint_violations, kl_divergence, target_kl
            )
        
        assert penalty <= self.adaptive_penalty.max_penalty
        
        # Try to push penalty very low
        self.adaptive_penalty.penalty = self.adaptive_penalty.min_penalty
        
        for _ in range(20):
            constraint_violations = torch.tensor([0.0])  # No violations
            kl_divergence = 0.001  # Low KL
            
            penalty = self.adaptive_penalty.update_penalty(
                constraint_violations, kl_divergence, target_kl
            )
        
        assert penalty >= self.adaptive_penalty.min_penalty


class TestLineSearch:
    """Test backtracking line search implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.line_search = LineSearch(
            max_iterations=10,
            backtrack_ratio=0.8,
            accept_ratio=0.1,
            target_kl=0.01
        )
        
        # Create a simple quadratic objective for testing
        self.param_dim = 5
        self.optimal_point = torch.randn(self.param_dim)
        
    def _quadratic_objective(self, step_size: float, search_direction: torch.Tensor) -> Dict[str, float]:
        """Simple quadratic objective for line search testing."""
        current_point = torch.zeros(self.param_dim)  # Starting at origin
        new_point = current_point + step_size * search_direction
        
        # Quadratic objective: f(x) = ||x - x_opt||^2
        objective_value = torch.norm(new_point - self.optimal_point)**2
        
        # Simple KL divergence model
        kl_div = step_size**2 * torch.norm(search_direction)**2 * 0.01
        
        # No constraint violations for this test
        constraint_violations = torch.tensor([0.0])
        
        return {
            "policy_loss": objective_value.item(),
            "kl_div": kl_div.item(),
            "constraint_violations": constraint_violations
        }
    
    def test_line_search_initialization(self):
        """Test line search initialization."""
        assert self.line_search.max_iterations == 10
        assert self.line_search.backtrack_ratio == 0.8
        assert self.line_search.accept_ratio == 0.1
        assert self.line_search.target_kl == 0.01
    
    def test_successful_line_search(self):
        """Test successful line search convergence."""
        # Search direction toward optimal point
        search_direction = self.optimal_point / torch.norm(self.optimal_point)
        
        # Create policy evaluation function
        def policy_eval(step_size):
            return self._quadratic_objective(step_size, search_direction)
        
        # Perform line search
        result = self.line_search.search(policy_eval, search_direction, initial_step_size=1.0)
        
        assert isinstance(result, LineSearchResult)
        assert result.success
        assert result.step_size > 0
        assert result.kl_divergence >= 0
        assert result.kl_divergence <= self.line_search.target_kl * 1.2  # Allow some slack
        assert result.policy_improvement > 0  # Should improve objective
    
    def test_line_search_with_kl_constraint_violation(self):
        """Test line search when KL constraint is violated."""
        # Large search direction that will violate KL constraint
        search_direction = torch.randn(self.param_dim) * 10
        
        def policy_eval(step_size):
            result = self._quadratic_objective(step_size, search_direction)
            # Make KL divergence large
            result["kl_div"] = step_size * 1.0  # Large KL
            return result
        
        result = self.line_search.search(policy_eval, search_direction, initial_step_size=1.0)
        
        # Should still succeed but with small step size
        assert result.step_size < 1.0
        assert result.iterations > 1  # Should require backtracking
    
    def test_line_search_failure(self):
        """Test line search failure when no acceptable step is found."""
        # Create pathological objective that never improves
        def bad_policy_eval(step_size):
            return {
                "policy_loss": 1.0 + step_size,  # Gets worse with any step
                "kl_div": step_size * 0.005,     # Reasonable KL
                "constraint_violations": torch.tensor([0.0])
            }
        
        search_direction = torch.randn(self.param_dim)
        result = self.line_search.search(bad_policy_eval, search_direction)
        
        assert not result.success
        assert result.iterations == self.line_search.max_iterations
        assert result.termination_reason == "max_iterations_reached"


class TestFisherInformationEstimator:
    """Test Fisher Information Matrix estimation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state_dim = 4
        self.action_dim = 2
        
        # Create simple policy for testing
        self.policy = SafePolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=[8, 8],
            device="cpu"
        )
        
        self.fisher_estimator = FisherInformationEstimator(
            damping=1e-4,
            use_empirical_fisher=False
        )
        
        self.states = torch.randn(16, self.state_dim)
        self.param_count = sum(p.numel() for p in self.policy.policy_net.parameters())
    
    def test_fisher_estimator_initialization(self):
        """Test Fisher estimator initialization."""
        assert self.fisher_estimator.damping == 1e-4
        assert not self.fisher_estimator.use_empirical_fisher
    
    def test_fisher_vector_product_computation(self):
        """Test Fisher-vector product computation."""
        vector = torch.randn(self.param_count)
        
        fvp = self.fisher_estimator.compute_fvp(self.policy.policy_net, self.states, vector)
        
        assert fvp.shape == vector.shape
        assert torch.isfinite(fvp).all()
        
        # Should be positive definite due to damping
        quadratic_form = torch.dot(vector, fvp)
        assert quadratic_form > 0
    
    def test_fisher_linearity(self):
        """Test linearity of Fisher-vector product."""
        v1 = torch.randn(self.param_count)
        v2 = torch.randn(self.param_count)
        a, b = 2.0, 3.0
        
        fvp1 = self.fisher_estimator.compute_fvp(self.policy.policy_net, self.states, v1)
        fvp2 = self.fisher_estimator.compute_fvp(self.policy.policy_net, self.states, v2)
        fvp_combined = self.fisher_estimator.compute_fvp(self.policy.policy_net, self.states, a * v1 + b * v2)
        
        expected = a * fvp1 + b * fvp2
        linearity_error = torch.norm(fvp_combined - expected)
        
        assert linearity_error < 1e-3, f"Fisher linearity error: {linearity_error:.6f}"
    
    def test_fisher_diagonal_estimation(self):
        """Test diagonal Fisher matrix estimation."""
        diagonal = self.fisher_estimator.estimate_fisher_diagonal(
            self.policy.policy_net, self.states, num_samples=10
        )
        
        assert diagonal.shape == (self.param_count,)
        assert torch.isfinite(diagonal).all()
        assert (diagonal >= 0).all()  # Diagonal elements should be non-negative
    
    def test_condition_number_estimation(self):
        """Test Fisher matrix condition number estimation."""
        condition_number = self.fisher_estimator.condition_number_estimate(
            self.policy.policy_net, self.states, num_vectors=5
        )
        
        assert isinstance(condition_number, float)
        assert condition_number >= 1.0  # Condition number should be >= 1
        assert np.isfinite(condition_number)


class TestTrustRegionNumericalAccuracy:
    """Test numerical accuracy of trust region methods."""
    
    def setup_method(self):
        """Set up numerical accuracy tests."""
        self.param_dim = 6
        
        # Create well-conditioned positive definite matrix
        A = torch.randn(self.param_dim, self.param_dim)
        self.hessian = A @ A.T + 0.1 * torch.eye(self.param_dim)
        
        self.solver = TrustRegionSolver(target_kl=1.0, damping=1e-6)  # Large trust region, small damping
    
    def _hessian_vector_product(self, v: torch.Tensor) -> torch.Tensor:
        """Exact Hessian-vector product."""
        return torch.mv(self.hessian, v)
    
    def test_conjugate_gradient_accuracy(self):
        """Test conjugate gradient solver accuracy."""
        b = torch.randn(self.param_dim)
        
        # Solve using CG
        x_cg, cg_info = conjugate_gradient(self._hessian_vector_product, b, max_iter=self.param_dim)
        
        # Solve exactly
        x_exact = torch.solve(b.unsqueeze(1), self.hessian)[0].squeeze()
        
        # Compare solutions
        error = torch.norm(x_cg - x_exact)
        assert error < 1e-4, f"CG solution error: {error:.6f}"
        
        # Check residual
        residual = self._hessian_vector_product(x_cg) - b
        residual_norm = torch.norm(residual)
        assert residual_norm < 1e-6, f"CG residual error: {residual_norm:.6f}"
    
    def test_trust_region_kkt_conditions(self):
        """Test KKT conditions for trust region solution."""
        gradient = torch.randn(self.param_dim) * 0.1  # Small gradient
        
        # Solve trust region problem
        search_direction, solver_info = self.solver.solve_unconstrained(
            gradient, self._hessian_vector_product
        )
        
        if not solver_info["trust_region_active"]:
            # Should satisfy H*s = g
            residual = self._hessian_vector_product(search_direction) - gradient
            assert torch.norm(residual) < 1e-4
        else:
            # Should satisfy trust region constraint
            quadratic_form = 0.5 * torch.dot(search_direction, self._hessian_vector_product(search_direction))
            assert abs(quadratic_form - self.solver.target_kl) < 1e-4
    
    def test_constrained_trust_region_feasibility(self):
        """Test feasibility of constrained trust region solutions."""
        gradient = torch.randn(self.param_dim)
        
        # Create orthogonal constraints
        constraint_gradients = torch.randn(2, self.param_dim)
        constraint_gradients, _ = torch.qr(constraint_gradients.T)
        constraint_gradients = constraint_gradients.T[:2]
        
        constraint_violations = torch.tensor([0.1, 0.05])  # Both violated
        
        search_direction, solver_info = self.solver.solve_constrained(
            gradient, constraint_gradients, constraint_violations, self._hessian_vector_product
        )
        
        # Check constraint satisfaction
        constraint_products = torch.mv(constraint_gradients, search_direction)
        
        for i, (product, violation) in enumerate(zip(constraint_products, constraint_violations)):
            if violation > 1e-6:  # Active constraint
                assert product <= violation + 1e-3, f"Constraint {i} violation: {product:.6f} > {violation:.6f}"


@pytest.mark.parametrize("target_kl,damping", [
    (0.001, 1e-5),
    (0.01, 1e-4),
    (0.1, 1e-3)
])
def test_trust_region_solver_robustness(target_kl, damping):
    """Test trust region solver with different parameter settings."""
    solver = TrustRegionSolver(target_kl=target_kl, damping=damping)
    
    param_dim = 8
    gradient = torch.randn(param_dim)
    
    # Simple quadratic Hessian
    A = torch.randn(param_dim, param_dim)
    hessian = A @ A.T + damping * torch.eye(param_dim)
    
    def hvp(v):
        return torch.mv(hessian, v)
    
    # Should not crash and should return reasonable solution
    search_direction, solver_info = solver.solve_unconstrained(gradient, hvp)
    
    assert torch.isfinite(search_direction).all()
    assert solver_info["cg_iterations"] >= 0
    
    # Check trust region constraint
    quadratic_form = 0.5 * torch.dot(search_direction, hvp(search_direction))
    assert quadratic_form <= target_kl + 1e-6


def test_trust_region_edge_cases():
    """Test trust region solver edge cases."""
    solver = TrustRegionSolver(target_kl=0.01)
    param_dim = 4
    
    def identity_hvp(v):
        return v + solver.damping * v
    
    # Test with zero gradient
    zero_gradient = torch.zeros(param_dim)
    search_direction, solver_info = solver.solve_unconstrained(zero_gradient, identity_hvp)
    
    assert torch.norm(search_direction) < 1e-6  # Should be near zero
    assert solver_info["cg_converged"]
    
    # Test with very large gradient
    large_gradient = torch.ones(param_dim) * 100
    search_direction, solver_info = solver.solve_unconstrained(large_gradient, identity_hvp)
    
    # Should be scaled by trust region
    assert solver_info["trust_region_active"]
    quadratic_form = 0.5 * torch.dot(search_direction, identity_hvp(search_direction))
    assert quadratic_form <= solver.target_kl + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])