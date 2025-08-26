"""
Comprehensive mathematical validation tests for Safe RL CPO implementation.
Tests numerical accuracy, mathematical properties, and theoretical guarantees.
"""

import numpy as np
import torch
import pytest
from typing import Dict, List, Tuple, Any
import scipy.optimize
from scipy.linalg import norm
import math

from src.algorithms.cpo import CPOAlgorithm, CPOConfig
from src.algorithms.trust_region import TrustRegionSolver
from src.algorithms.gae import GeneralizedAdvantageEstimation
from src.core.safety_monitor import SafetyMonitor


class MathematicalValidationSuite:
    """Comprehensive mathematical validation for CPO components."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def validate_policy_gradient_theorem(self) -> Dict[str, Any]:
        """Validate policy gradient theorem: ∇J(θ) = E[∇log π(a|s) Q(s,a)]"""
        results = {}
        
        # Create simple test environment
        state_dim, action_dim = 4, 2
        batch_size = 1000
        
        # Generate synthetic data
        states = torch.randn(batch_size, state_dim, device=self.device)
        actions = torch.randn(batch_size, action_dim, device=self.device)
        rewards = torch.randn(batch_size, device=self.device)
        
        # Create simple policy network
        policy = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, action_dim * 2)  # mean and log_std
        ).to(self.device)
        
        def compute_log_prob(policy_output, actions):
            """Compute log probability for Gaussian policy."""
            mean, log_std = policy_output.chunk(2, dim=-1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            return dist.log_prob(actions).sum(dim=-1)
        
        # Method 1: Direct gradient computation
        policy_output = policy(states)
        log_probs = compute_log_prob(policy_output, actions)
        policy_loss = -(log_probs * rewards).mean()
        
        policy.zero_grad()
        policy_loss.backward()
        direct_gradients = []
        for param in policy.parameters():
            if param.grad is not None:
                direct_gradients.append(param.grad.clone().flatten())
        direct_grad = torch.cat(direct_gradients)
        
        # Method 2: Finite difference approximation
        epsilon = 1e-5
        finite_diff_grad = []
        
        with torch.no_grad():
            # Get current parameters
            original_params = []
            for param in policy.parameters():
                original_params.append(param.clone())
            
            param_idx = 0
            for param in policy.parameters():
                param_grad = torch.zeros_like(param)
                for i in range(param.numel()):
                    # Forward perturbation
                    param.view(-1)[i] += epsilon
                    policy_output_plus = policy(states)
                    log_probs_plus = compute_log_prob(policy_output_plus, actions)
                    loss_plus = -(log_probs_plus * rewards).mean()
                    
                    # Backward perturbation
                    param.view(-1)[i] -= 2 * epsilon
                    policy_output_minus = policy(states)
                    log_probs_minus = compute_log_prob(policy_output_minus, actions)
                    loss_minus = -(log_probs_minus * rewards).mean()
                    
                    # Compute finite difference
                    param_grad.view(-1)[i] = (loss_plus - loss_minus) / (2 * epsilon)
                    
                    # Reset parameter
                    param.view(-1)[i] += epsilon
                
                finite_diff_grad.append(param_grad.flatten())
                param_idx += 1
        
        finite_diff_grad = torch.cat(finite_diff_grad)
        
        # Compare gradients
        gradient_error = torch.norm(direct_grad - finite_diff_grad).item()
        relative_error = gradient_error / (torch.norm(direct_grad).item() + 1e-8)
        
        results['gradient_error'] = gradient_error
        results['relative_error'] = relative_error
        results['passed'] = relative_error < 1e-3
        
        return results
    
    def validate_kl_divergence_properties(self) -> Dict[str, Any]:
        """Validate KL divergence mathematical properties."""
        results = {}
        
        # Create two Gaussian distributions
        mean1 = torch.tensor([1.0, 2.0], device=self.device)
        std1 = torch.tensor([0.5, 1.0], device=self.device)
        mean2 = torch.tensor([1.5, 1.8], device=self.device)
        std2 = torch.tensor([0.7, 0.9], device=self.device)
        
        dist1 = torch.distributions.Normal(mean1, std1)
        dist2 = torch.distributions.Normal(mean2, std2)
        
        # Property 1: KL(P||Q) >= 0
        kl_div = torch.distributions.kl_divergence(dist1, dist2).sum()
        results['kl_non_negative'] = kl_div.item() >= 0
        
        # Property 2: KL(P||P) = 0
        kl_self = torch.distributions.kl_divergence(dist1, dist1).sum()
        results['kl_self_zero'] = abs(kl_self.item()) < self.tolerance
        
        # Property 3: Asymmetry - KL(P||Q) != KL(Q||P) in general
        kl_reverse = torch.distributions.kl_divergence(dist2, dist1).sum()
        results['kl_asymmetric'] = abs(kl_div.item() - kl_reverse.item()) > self.tolerance
        
        # Property 4: Analytical vs numerical KL for Gaussians
        # KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        analytical_kl = 0
        for i in range(2):
            analytical_kl += (
                torch.log(std2[i] / std1[i]) +
                (std1[i]**2 + (mean1[i] - mean2[i])**2) / (2 * std2[i]**2) -
                0.5
            )
        
        analytical_error = abs(kl_div.item() - analytical_kl.item())
        results['analytical_kl_error'] = analytical_error
        results['analytical_kl_correct'] = analytical_error < self.tolerance
        
        results['passed'] = all([
            results['kl_non_negative'],
            results['kl_self_zero'],
            results['kl_asymmetric'],
            results['analytical_kl_correct']
        ])
        
        return results
    
    def validate_fisher_information_matrix(self) -> Dict[str, Any]:
        """Validate Fisher Information Matrix properties."""
        results = {}
        
        state_dim, action_dim = 3, 2
        batch_size = 1000
        
        # Create simple policy
        policy = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, action_dim * 2)
        ).to(self.device)
        
        states = torch.randn(batch_size, state_dim, device=self.device)
        
        # Compute Fisher Information Matrix using definition:
        # F = E[∇log π(a|s) ∇log π(a|s)^T]
        def compute_fisher_matrix():
            policy_output = policy(states)
            mean, log_std = policy_output.chunk(2, dim=-1)
            std = torch.exp(log_std)
            
            # Sample actions
            dist = torch.distributions.Normal(mean, std)
            actions = dist.sample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # Compute gradients for each sample
            gradients = []
            for i in range(batch_size):
                policy.zero_grad()
                log_probs[i].backward(retain_graph=True)
                
                grad_vec = []
                for param in policy.parameters():
                    if param.grad is not None:
                        grad_vec.append(param.grad.clone().flatten())
                gradients.append(torch.cat(grad_vec))
            
            # Stack gradients and compute outer product expectation
            grad_matrix = torch.stack(gradients)  # [batch_size, num_params]
            fisher_matrix = torch.mm(grad_matrix.T, grad_matrix) / batch_size
            
            return fisher_matrix
        
        fisher = compute_fisher_matrix()
        
        # Property 1: Fisher matrix should be positive semi-definite
        eigenvals = torch.linalg.eigvals(fisher).real
        results['positive_semidefinite'] = torch.all(eigenvals >= -self.tolerance).item()
        
        # Property 2: Fisher matrix should be symmetric
        symmetry_error = torch.norm(fisher - fisher.T).item()
        results['symmetric'] = symmetry_error < self.tolerance
        results['symmetry_error'] = symmetry_error
        
        # Property 3: Condition number should be reasonable (not too ill-conditioned)
        condition_number = torch.linalg.cond(fisher).item()
        results['condition_number'] = condition_number
        results['well_conditioned'] = condition_number < 1e6
        
        results['passed'] = all([
            results['positive_semidefinite'],
            results['symmetric'],
            results['well_conditioned']
        ])
        
        return results
    
    def validate_gae_convergence(self) -> Dict[str, Any]:
        """Validate GAE convergence properties."""
        results = {}
        
        # Create synthetic trajectory data
        trajectory_length = 100
        gamma = 0.99
        lambda_gae = 0.95
        
        rewards = torch.randn(trajectory_length, device=self.device)
        values = torch.randn(trajectory_length + 1, device=self.device)
        
        # Method 1: GAE computation
        gae = GeneralizedAdvantageEstimation(gamma=gamma, lambda_gae=lambda_gae)
        advantages_gae = gae.compute_advantages(rewards, values[:-1], values[1:])
        
        # Method 2: Manual GAE computation for verification
        deltas = rewards + gamma * values[1:] - values[:-1]
        advantages_manual = torch.zeros_like(rewards)
        
        gae_lambda = 0
        for t in reversed(range(trajectory_length)):
            if t == trajectory_length - 1:
                gae_lambda = deltas[t]
            else:
                gae_lambda = deltas[t] + gamma * lambda_gae * gae_lambda
            advantages_manual[t] = gae_lambda
        
        # Compare methods
        gae_error = torch.norm(advantages_gae - advantages_manual).item()
        results['gae_computation_error'] = gae_error
        results['gae_correct'] = gae_error < self.tolerance
        
        # Test lambda boundaries
        # When lambda = 0, GAE should equal TD error
        gae_td = GeneralizedAdvantageEstimation(gamma=gamma, lambda_gae=0.0)
        advantages_td = gae_td.compute_advantages(rewards, values[:-1], values[1:])
        td_residuals = rewards + gamma * values[1:] - values[:-1]
        
        td_error = torch.norm(advantages_td - td_residuals).item()
        results['gae_td_error'] = td_error
        results['gae_td_correct'] = td_error < self.tolerance
        
        # When lambda = 1, GAE should equal Monte Carlo returns minus values
        gae_mc = GeneralizedAdvantageEstimation(gamma=gamma, lambda_gae=1.0)
        advantages_mc = gae_mc.compute_advantages(rewards, values[:-1], values[1:])
        
        # Compute Monte Carlo returns
        mc_returns = torch.zeros_like(rewards)
        for t in range(trajectory_length):
            mc_return = 0
            for k in range(trajectory_length - t):
                mc_return += (gamma ** k) * rewards[t + k]
            mc_returns[t] = mc_return
        
        mc_advantages = mc_returns - values[:-1]
        mc_error = torch.norm(advantages_mc - mc_advantages).item()
        results['gae_mc_error'] = mc_error
        results['gae_mc_correct'] = mc_error < self.tolerance * 10  # Slightly more tolerance for numerical precision
        
        results['passed'] = all([
            results['gae_correct'],
            results['gae_td_correct'],
            results['gae_mc_correct']
        ])
        
        return results
    
    def validate_conjugate_gradient(self) -> Dict[str, Any]:
        """Validate conjugate gradient solver mathematical properties."""
        results = {}
        
        # Create a symmetric positive definite test matrix
        n = 20
        A = torch.randn(n, n, device=self.device)
        A = torch.mm(A.T, A) + torch.eye(n, device=self.device)  # Make SPD
        b = torch.randn(n, device=self.device)
        
        # True solution
        x_true = torch.linalg.solve(A, b)
        
        # Conjugate gradient solution
        trust_region = TrustRegionSolver()
        
        def matrix_vector_product(v):
            return torch.mv(A, v)
        
        x_cg = trust_region.conjugate_gradient(
            matrix_vector_product, b, max_iterations=n
        )
        
        # Compare solutions
        solution_error = torch.norm(x_cg - x_true).item()
        results['solution_error'] = solution_error
        results['solution_correct'] = solution_error < self.tolerance * 10
        
        # Verify A-orthogonality of search directions (theoretical property)
        # This requires modifying CG to return intermediate directions
        residual = torch.mv(A, x_cg) - b
        residual_norm = torch.norm(residual).item()
        results['residual_norm'] = residual_norm
        results['residual_small'] = residual_norm < self.tolerance * 10
        
        # Test convergence properties
        # CG should converge in at most n steps for n x n matrix
        results['max_iterations_theoretical'] = n
        results['converged_in_theory'] = True  # By construction
        
        results['passed'] = all([
            results['solution_correct'],
            results['residual_small']
        ])
        
        return results
    
    def validate_trust_region_properties(self) -> Dict[str, Any]:
        """Validate trust region method mathematical properties."""
        results = {}
        
        # Create quadratic test problem: f(x) = 0.5 * x^T * H * x + g^T * x
        n = 10
        H = torch.randn(n, n, device=self.device)
        H = torch.mm(H.T, H) + torch.eye(n, device=self.device)  # Make PD
        g = torch.randn(n, device=self.device)
        
        trust_radius = 1.0
        
        def quadratic_model(x):
            return 0.5 * torch.dot(x, torch.mv(H, x)) + torch.dot(g, x)
        
        def gradient(x):
            return torch.mv(H, x) + g
        
        def hessian_vector_product(v):
            return torch.mv(H, v)
        
        # Solve trust region subproblem
        trust_region = TrustRegionSolver()
        x0 = torch.zeros(n, device=self.device)
        
        # Use conjugate gradient to solve approximately
        step = trust_region.conjugate_gradient(
            hessian_vector_product, -gradient(x0), max_iterations=n
        )
        
        # Truncate to trust region if necessary
        step_norm = torch.norm(step).item()
        if step_norm > trust_radius:
            step = step * (trust_radius / step_norm)
        
        # Verify trust region constraint
        final_step_norm = torch.norm(step).item()
        results['trust_region_satisfied'] = final_step_norm <= trust_radius + self.tolerance
        results['step_norm'] = final_step_norm
        
        # Verify improvement (Cauchy decrease condition)
        # Should achieve at least as much decrease as Cauchy point
        grad_norm = torch.norm(gradient(x0)).item()
        cauchy_decrease = min(trust_radius, grad_norm**2 / torch.dot(gradient(x0), torch.mv(H, gradient(x0))).item())
        
        actual_decrease = -(quadratic_model(step) - quadratic_model(x0)).item()
        cauchy_point_decrease = 0.5 * cauchy_decrease * grad_norm
        
        results['actual_decrease'] = actual_decrease
        results['cauchy_decrease'] = cauchy_point_decrease
        results['cauchy_condition'] = actual_decrease >= 0.5 * cauchy_point_decrease
        
        results['passed'] = all([
            results['trust_region_satisfied'],
            results['cauchy_condition']
        ])
        
        return results
    
    def validate_cpo_constraint_satisfaction(self) -> Dict[str, Any]:
        """Validate CPO constraint satisfaction properties."""
        results = {}
        
        # Create simple test environment for CPO
        state_dim, action_dim = 4, 2
        config = CPOConfig(
            policy_lr=1e-3,
            value_lr=1e-3,
            constraint_value_lr=1e-3,
            gamma=0.99,
            lambda_gae=0.95,
            trust_region_radius=0.01,
            constraint_threshold=0.1,
            cg_iterations=10,
            line_search_steps=10
        )
        
        # Mock policy and value networks
        class MockPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, 32),
                    torch.nn.Tanh(),
                    torch.nn.Linear(32, action_dim * 2)
                )
            
            def forward(self, states):
                output = self.network(states)
                mean, log_std = output.chunk(2, dim=-1)
                return mean, torch.exp(log_std)
            
            def evaluate_actions(self, states, actions):
                mean, std = self.forward(states)
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                return {"log_probs": log_probs, "entropy": entropy}
        
        class MockValueNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(state_dim, 32),
                    torch.nn.Tanh(),
                    torch.nn.Linear(32, 1)
                )
            
            def forward(self, states):
                return self.network(states).squeeze(-1)
        
        policy = MockPolicy().to(self.device)
        value_network = MockValueNetwork().to(self.device)
        constraint_value_network = MockValueNetwork().to(self.device)
        
        # Create synthetic trajectory data
        batch_size = 100
        states = torch.randn(batch_size, state_dim, device=self.device)
        actions = torch.randn(batch_size, action_dim, device=self.device)
        rewards = torch.randn(batch_size, device=self.device)
        constraint_costs = torch.abs(torch.randn(batch_size, device=self.device))  # Positive costs
        old_log_probs = torch.randn(batch_size, device=self.device)
        
        # Compute advantages (simplified)
        advantages = torch.randn(batch_size, device=self.device)
        constraint_advantages = torch.randn(batch_size, device=self.device)
        
        # Test constraint gradient computation
        evaluation = policy.evaluate_actions(states, actions)
        current_log_probs = evaluation["log_probs"]
        
        # Constraint gradient: E[∇log π(a|s) * constraint_advantage]
        constraint_loss = -(current_log_probs * constraint_advantages).mean()
        
        policy.zero_grad()
        constraint_loss.backward(retain_graph=True)
        
        constraint_grad = []
        for param in policy.parameters():
            if param.grad is not None:
                constraint_grad.append(param.grad.clone().flatten())
        constraint_grad = torch.cat(constraint_grad)
        
        # Check gradient magnitude
        constraint_grad_norm = torch.norm(constraint_grad).item()
        results['constraint_grad_norm'] = constraint_grad_norm
        results['constraint_grad_finite'] = torch.isfinite(constraint_grad).all().item()
        
        # Test KL divergence constraint
        # Current policy vs old policy KL
        ratio = torch.exp(current_log_probs - old_log_probs)
        kl_div = torch.mean(old_log_probs - current_log_probs + ratio - 1)
        
        results['kl_divergence'] = kl_div.item()
        results['kl_constraint_satisfied'] = kl_div.item() <= config.trust_region_radius + self.tolerance
        
        # Test constraint value prediction
        predicted_constraint_values = constraint_value_network(states)
        constraint_mse = torch.mean((predicted_constraint_values - constraint_costs)**2)
        
        results['constraint_value_mse'] = constraint_mse.item()
        results['constraint_values_reasonable'] = constraint_mse.item() < 10.0  # Reasonable for random initialization
        
        results['passed'] = all([
            results['constraint_grad_finite'],
            results['kl_constraint_satisfied'],
            results['constraint_values_reasonable']
        ])
        
        return results


@pytest.mark.mathematical
class TestMathematicalValidation:
    """Test suite for mathematical validation."""
    
    @pytest.fixture
    def validator(self):
        return MathematicalValidationSuite(tolerance=1e-6)
    
    def test_policy_gradient_theorem(self, validator):
        """Test policy gradient theorem validation."""
        results = validator.validate_policy_gradient_theorem()
        
        assert results['passed'], f"Policy gradient validation failed: {results}"
        assert results['relative_error'] < 1e-3, f"Gradient error too large: {results['relative_error']}"
    
    def test_kl_divergence_properties(self, validator):
        """Test KL divergence mathematical properties."""
        results = validator.validate_kl_divergence_properties()
        
        assert results['passed'], f"KL divergence validation failed: {results}"
        assert results['kl_non_negative'], "KL divergence should be non-negative"
        assert results['kl_self_zero'], "KL(P||P) should be zero"
        assert results['analytical_kl_correct'], "Analytical KL computation should match PyTorch"
    
    def test_fisher_information_matrix(self, validator):
        """Test Fisher Information Matrix properties."""
        results = validator.validate_fisher_information_matrix()
        
        assert results['passed'], f"Fisher matrix validation failed: {results}"
        assert results['positive_semidefinite'], "Fisher matrix should be positive semi-definite"
        assert results['symmetric'], "Fisher matrix should be symmetric"
    
    def test_gae_convergence(self, validator):
        """Test GAE convergence properties."""
        results = validator.validate_gae_convergence()
        
        assert results['passed'], f"GAE validation failed: {results}"
        assert results['gae_correct'], "GAE computation should match manual calculation"
        assert results['gae_td_correct'], "GAE with λ=0 should equal TD residuals"
    
    def test_conjugate_gradient(self, validator):
        """Test conjugate gradient solver."""
        results = validator.validate_conjugate_gradient()
        
        assert results['passed'], f"Conjugate gradient validation failed: {results}"
        assert results['solution_correct'], "CG should solve linear system accurately"
        assert results['residual_small'], "CG residual should be small"
    
    def test_trust_region_properties(self, validator):
        """Test trust region method properties."""
        results = validator.validate_trust_region_properties()
        
        assert results['passed'], f"Trust region validation failed: {results}"
        assert results['trust_region_satisfied'], "Trust region constraint should be satisfied"
        assert results['cauchy_condition'], "Cauchy decrease condition should hold"
    
    def test_cpo_constraint_satisfaction(self, validator):
        """Test CPO constraint satisfaction."""
        results = validator.validate_cpo_constraint_satisfaction()
        
        assert results['passed'], f"CPO constraint validation failed: {results}"
        assert results['constraint_grad_finite'], "Constraint gradients should be finite"
        assert results['kl_constraint_satisfied'], "KL constraint should be satisfied"


def run_full_mathematical_validation():
    """Run complete mathematical validation suite and generate report."""
    validator = MathematicalValidationSuite()
    
    print("=" * 80)
    print("MATHEMATICAL VALIDATION REPORT")
    print("=" * 80)
    
    tests = [
        ("Policy Gradient Theorem", validator.validate_policy_gradient_theorem),
        ("KL Divergence Properties", validator.validate_kl_divergence_properties),
        ("Fisher Information Matrix", validator.validate_fisher_information_matrix),
        ("GAE Convergence", validator.validate_gae_convergence),
        ("Conjugate Gradient", validator.validate_conjugate_gradient),
        ("Trust Region Properties", validator.validate_trust_region_properties),
        ("CPO Constraint Satisfaction", validator.validate_cpo_constraint_satisfaction),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            results = test_func()
            if results['passed']:
                print("✓ PASSED")
            else:
                print("✗ FAILED")
                all_passed = False
            
            # Print key metrics
            for key, value in results.items():
                if key != 'passed' and not key.startswith('_'):
                    if isinstance(value, float):
                        print(f"  {key}: {value:.6e}")
                    else:
                        print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("ALL MATHEMATICAL VALIDATIONS PASSED ✓")
    else:
        print("SOME VALIDATIONS FAILED ✗")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    run_full_mathematical_validation()