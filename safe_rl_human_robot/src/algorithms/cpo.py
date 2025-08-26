"""
Complete Constrained Policy Optimization (CPO) Algorithm Implementation.

This module implements the full CPO algorithm with mathematical rigor:
- Objective: J(θ) = E[∑ γᵗ r(sₜ,aₜ)]
- Constraints: Jᶜ(θ) = E[∑ γᵗ c(sₜ,aₜ)] ≤ d
- Update: θ_{k+1} = arg max_θ J(θ) s.t. Jᶜ(θ) ≤ d, D_KL(π_old, π) ≤ δ
"""

from typing import Dict, List, Optional, Tuple, Union, NamedTuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from dataclasses import dataclass, field
import logging
import time
from abc import ABC, abstractmethod

from .trust_region import TrustRegionSolver, LineSearchResult
from .gae import GeneralizedAdvantageEstimation, ValueFunction
from ..core.policy import SafePolicy
from ..core.constraints import ConstraintManager
from ..core.safety_monitor import SafetyMonitor
from ..utils.math_utils import compute_kl_divergence
from ..utils.logging_utils import MetricsLogger

logger = logging.getLogger(__name__)


class CPOTrajectory(NamedTuple):
    """Container for trajectory data."""
    states: torch.Tensor          # [T, state_dim]
    actions: torch.Tensor         # [T, action_dim]
    rewards: torch.Tensor         # [T]
    constraint_costs: torch.Tensor # [T, num_constraints]
    log_probs: torch.Tensor       # [T]
    values: torch.Tensor          # [T]
    constraint_values: torch.Tensor # [T, num_constraints]
    dones: torch.Tensor           # [T]
    next_states: torch.Tensor     # [T, state_dim]


@dataclass
class CPOConfig:
    """Configuration for CPO algorithm."""
    # Core CPO parameters
    target_kl: float = 0.01              # Trust region size δ
    constraint_threshold: float = 0.025   # Safety threshold d
    damping_coeff: float = 1e-4          # Damping for conjugate gradient
    cg_iters: int = 10                   # Conjugate gradient iterations
    
    # Line search parameters
    backtrack_iters: int = 10            # Line search iterations
    backtrack_ratio: float = 0.8         # Line search decay factor
    accept_ratio: float = 0.1            # Minimum improvement ratio
    
    # GAE parameters
    gamma: float = 0.99                  # Discount factor γ
    gae_lambda: float = 0.97             # GAE smoothing λ
    
    # Training parameters
    policy_lr: float = 3e-4              # Policy learning rate
    value_lr: float = 1e-3               # Value function learning rate
    value_iters: int = 80                # Value function training iterations
    
    # Safety parameters
    constraint_penalty_coeff: float = 1.0  # Constraint penalty coefficient
    safety_margin: float = 0.1             # Safety margin for constraints
    emergency_brake: bool = True           # Enable emergency braking
    
    # Logging parameters
    log_frequency: int = 10              # Log every N iterations
    save_frequency: int = 100            # Save model every N iterations


@dataclass
class CPOState:
    """State of CPO optimization process."""
    iteration: int
    policy_loss: float
    constraint_violations: torch.Tensor
    kl_divergence: float
    step_size: float
    line_search_steps: int
    constraint_threshold_updated: bool
    safety_violations: int
    total_episodes: int
    average_return: float
    average_constraint_cost: torch.Tensor
    convergence_metrics: Dict[str, float] = field(default_factory=dict)


class CPOAlgorithm:
    """
    Complete Constrained Policy Optimization implementation.
    
    Implements the CPO algorithm as described in Achiam et al. (2017) with
    mathematical rigor and numerical stability guarantees.
    """
    
    def __init__(self,
                 policy: SafePolicy,
                 constraint_manager: ConstraintManager,
                 environment,
                 config: CPOConfig = None,
                 value_function: Optional[ValueFunction] = None,
                 safety_monitor: Optional[SafetyMonitor] = None,
                 metrics_logger: Optional[MetricsLogger] = None,
                 device: str = "cpu"):
        """
        Initialize CPO algorithm.
        
        Args:
            policy: Safe policy with constraint-aware sampling
            constraint_manager: Safety constraint manager
            environment: Training environment
            config: CPO configuration parameters
            value_function: Value function for advantage estimation
            safety_monitor: Real-time safety monitoring
            metrics_logger: Metrics and experiment tracking
            device: Computation device
        """
        self.policy = policy
        self.constraint_manager = constraint_manager
        self.env = environment
        self.config = config or CPOConfig()
        self.device = device
        
        # Initialize value function
        if value_function is None:
            self.value_function = ValueFunction(
                state_dim=policy.state_dim,
                hidden_sizes=[256, 256],
                learning_rate=self.config.value_lr,
                device=device
            )
        else:
            self.value_function = value_function
            
        # Initialize constraint value functions (one per constraint)
        self.constraint_value_functions = nn.ModuleList([
            ValueFunction(
                state_dim=policy.state_dim,
                hidden_sizes=[128, 128],
                learning_rate=self.config.value_lr,
                device=device
            )
            for _ in constraint_manager.constraints
        ])
        
        # Initialize components
        self.gae = GeneralizedAdvantageEstimation(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )
        
        self.trust_region_solver = TrustRegionSolver(
            target_kl=self.config.target_kl,
            damping=self.config.damping_coeff,
            cg_iters=self.config.cg_iters,
            backtrack_iters=self.config.backtrack_iters,
            accept_ratio=self.config.accept_ratio
        )
        
        self.safety_monitor = safety_monitor or SafetyMonitor()
        self.metrics_logger = metrics_logger
        
        # CPO state tracking
        self.iteration = 0
        self.total_episodes = 0
        self.optimization_history: List[CPOState] = []
        
        # Constraint threshold adaptation
        self.constraint_threshold = torch.tensor(
            [self.config.constraint_threshold] * len(constraint_manager.constraints),
            device=device
        )
        
        # Emergency brake state
        self.emergency_brake_active = False
        self.consecutive_violations = 0
        
        logger.info(f"CPO Algorithm initialized with {len(constraint_manager.constraints)} constraints")
        logger.info(f"Trust region size: {self.config.target_kl}")
        logger.info(f"Constraint threshold: {self.config.constraint_threshold}")
    
    def collect_trajectories(self, num_episodes: int = 100) -> List[CPOTrajectory]:
        """
        Collect trajectories using current policy.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            List of trajectory data
        """
        trajectories = []
        
        for episode in range(num_episodes):
            states, actions, rewards, constraint_costs = [], [], [], []
            log_probs, values, constraint_values = [], [], []
            dones, next_states = [], []
            
            state = self.env.reset()
            done = False
            episode_length = 0
            
            while not done and episode_length < self.config.max_episode_length:
                state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
                
                # Sample action from policy
                with torch.no_grad():
                    action, action_info = self.policy.sample_action(state_tensor)
                    value = self.value_function(state_tensor)
                    
                    # Evaluate constraint values
                    constraint_vals = []
                    for i, constraint_vf in enumerate(self.constraint_value_functions):
                        constraint_vals.append(constraint_vf(state_tensor))
                    constraint_values_tensor = torch.stack(constraint_vals, dim=1)
                
                # Execute action in environment
                next_state, reward, done, info = self.env.step(action.squeeze().cpu().numpy())
                
                # Compute constraint costs
                constraint_dict = self.constraint_manager.evaluate_all(state_tensor, action)
                constraint_cost = torch.stack([
                    torch.clamp(constraint_dict[cid], min=0.0)  # Only violations count as cost
                    for cid in sorted(constraint_dict.keys())
                ], dim=1)
                
                # Store trajectory data
                states.append(state_tensor)
                actions.append(action)
                rewards.append(torch.tensor([reward], device=self.device))
                constraint_costs.append(constraint_cost)
                log_probs.append(action_info["log_prob"])
                values.append(value)
                constraint_values.append(constraint_values_tensor)
                dones.append(torch.tensor([done], device=self.device))
                next_states.append(torch.from_numpy(next_state).float().to(self.device).unsqueeze(0))
                
                # Safety monitoring
                if constraint_cost.sum() > 0:
                    self.safety_monitor.log_violation(
                        f"episode_{episode}_step_{episode_length}",
                        constraint_cost.sum().item(),
                        {
                            "state": state.tolist(),
                            "action": action.squeeze().tolist(),
                            "constraint_values": [constraint_dict[cid].item() for cid in sorted(constraint_dict.keys())]
                        }
                    )
                
                state = next_state
                episode_length += 1
            
            # Create trajectory
            if len(states) > 0:  # Ensure we have data
                trajectory = CPOTrajectory(
                    states=torch.cat(states, dim=0),
                    actions=torch.cat(actions, dim=0),
                    rewards=torch.cat(rewards, dim=0),
                    constraint_costs=torch.cat(constraint_costs, dim=0),
                    log_probs=torch.cat(log_probs, dim=0),
                    values=torch.cat(values, dim=0),
                    constraint_values=torch.cat(constraint_values, dim=0),
                    dones=torch.cat(dones, dim=0),
                    next_states=torch.cat(next_states, dim=0)
                )
                trajectories.append(trajectory)
            
            self.total_episodes += 1
        
        logger.info(f"Collected {len(trajectories)} trajectories with total steps: {sum(len(t.states) for t in trajectories)}")
        return trajectories
    
    def compute_advantages(self, trajectories: List[CPOTrajectory]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy and constraint advantages using GAE.
        
        Args:
            trajectories: List of trajectory data
            
        Returns:
            Tuple of (policy_advantages, constraint_advantages)
        """
        policy_advantages = []
        constraint_advantages = []
        
        for trajectory in trajectories:
            # Compute returns and advantages for policy objective
            policy_returns, policy_advs = self.gae.compute_gae(
                rewards=trajectory.rewards,
                values=trajectory.values,
                dones=trajectory.dones,
                next_values=self.value_function(trajectory.next_states).squeeze()
            )
            
            policy_advantages.append(policy_advs)
            
            # Compute constraint advantages for each constraint
            traj_constraint_advs = []
            for i in range(trajectory.constraint_costs.shape[1]):
                constraint_returns, constraint_advs = self.gae.compute_gae(
                    rewards=trajectory.constraint_costs[:, i],
                    values=trajectory.constraint_values[:, i],
                    dones=trajectory.dones,
                    next_values=self.constraint_value_functions[i](trajectory.next_states).squeeze()
                )
                traj_constraint_advs.append(constraint_advs)
            
            constraint_advantages.append(torch.stack(traj_constraint_advs, dim=1))
        
        return torch.cat(policy_advantages), torch.cat(constraint_advantages)
    
    def compute_policy_gradient(self, states: torch.Tensor, actions: torch.Tensor,
                               advantages: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute natural policy gradient ∇_θ J(θ).
        
        Args:
            states: State batch [batch_size, state_dim]
            actions: Action batch [batch_size, action_dim]
            advantages: Policy advantages [batch_size]
            old_log_probs: Old policy log probabilities [batch_size]
            
        Returns:
            Policy gradient vector [param_dim]
        """
        # Compute current log probabilities
        evaluation = self.policy.evaluate_actions(states, actions)
        log_probs = evaluation["log_probs"]
        
        # Policy gradient: E[∇_θ log π_θ(a|s) A^π(s,a)]
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute gradient
        policy_grad = torch.autograd.grad(
            policy_loss, self.policy.policy_net.parameters(),
            create_graph=True, retain_graph=True
        )
        
        # Flatten gradient
        policy_grad_flat = torch.cat([g.view(-1) for g in policy_grad])
        
        return policy_grad_flat
    
    def compute_constraint_gradient(self, states: torch.Tensor, actions: torch.Tensor,
                                  constraint_advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute constraint gradient ∇_θ Jᶜ(θ).
        
        Args:
            states: State batch [batch_size, state_dim]
            actions: Action batch [batch_size, action_dim]
            constraint_advantages: Constraint advantages [batch_size, num_constraints]
            
        Returns:
            Constraint gradient matrix [num_constraints, param_dim]
        """
        constraint_grads = []
        
        for i in range(constraint_advantages.shape[1]):
            # Compute log probabilities for constraint i
            evaluation = self.policy.evaluate_actions(states, actions)
            log_probs = evaluation["log_probs"]
            
            # Constraint gradient: E[∇_θ log π_θ(a|s) A^c(s,a)]
            constraint_loss = (log_probs * constraint_advantages[:, i]).mean()
            
            # Compute gradient
            constraint_grad = torch.autograd.grad(
                constraint_loss, self.policy.policy_net.parameters(),
                create_graph=True, retain_graph=True
            )
            
            # Flatten gradient
            constraint_grad_flat = torch.cat([g.view(-1) for g in constraint_grad])
            constraint_grads.append(constraint_grad_flat)
        
        return torch.stack(constraint_grads)
    
    def compute_fisher_vector_product(self, states: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """
        Compute Fisher Information Matrix-vector product H*v.
        
        Args:
            states: State batch for computing KL divergence
            vector: Vector to multiply [param_dim]
            
        Returns:
            Fisher-vector product [param_dim]
        """
        # Get old distribution
        with torch.no_grad():
            old_dist = self.policy.policy_net.get_distribution(states)
            old_means = old_dist.mean
            old_stds = old_dist.stddev
        
        # Get current distribution
        current_dist = self.policy.policy_net.get_distribution(states)
        
        # KL divergence
        kl_div = compute_kl_divergence(
            current_dist.mean, current_dist.stddev,
            old_means, old_stds
        ).mean()
        
        # First-order gradient of KL
        kl_grad = torch.autograd.grad(
            kl_div, self.policy.policy_net.parameters(),
            create_graph=True, retain_graph=True
        )
        kl_grad_flat = torch.cat([g.view(-1) for g in kl_grad])
        
        # Gradient-vector product
        gvp = torch.dot(kl_grad_flat, vector)
        
        # Second-order gradient (Hessian-vector product)
        hvp = torch.autograd.grad(
            gvp, self.policy.policy_net.parameters(),
            retain_graph=True
        )
        hvp_flat = torch.cat([g.view(-1) for g in hvp])
        
        # Add damping for numerical stability
        return hvp_flat + self.config.damping_coeff * vector
    
    def solve_trust_region_problem(self, 
                                 policy_grad: torch.Tensor,
                                 constraint_grads: torch.Tensor,
                                 constraint_violations: torch.Tensor,
                                 states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Solve constrained trust region optimization problem.
        
        minimize: -g^T s
        subject to: 1/2 s^T H s ≤ δ
                   b_i^T s ≤ c_i  (for violated constraints)
        
        Args:
            policy_grad: Policy gradient g [param_dim]
            constraint_grads: Constraint gradients B [num_constraints, param_dim]  
            constraint_violations: Constraint violation levels [num_constraints]
            states: State batch for Fisher matrix computation
            
        Returns:
            Tuple of (search_direction, solver_info)
        """
        def fisher_vector_product(v):
            return self.compute_fisher_vector_product(states, v)
        
        # Identify violated constraints
        violated_constraints = constraint_violations > self.config.safety_margin
        
        if not violated_constraints.any():
            # No constraints violated - standard trust region problem
            search_direction, info = self.trust_region_solver.solve_unconstrained(
                policy_grad, fisher_vector_product
            )
        else:
            # Constrained trust region problem
            active_constraint_grads = constraint_grads[violated_constraints]
            active_violations = constraint_violations[violated_constraints]
            
            search_direction, info = self.trust_region_solver.solve_constrained(
                policy_grad, active_constraint_grads, active_violations,
                fisher_vector_product
            )
        
        info["num_violated_constraints"] = violated_constraints.sum().item()
        info["max_constraint_violation"] = constraint_violations.max().item()
        
        return search_direction, info
    
    def line_search(self, 
                   search_direction: torch.Tensor,
                   states: torch.Tensor,
                   actions: torch.Tensor,
                   advantages: torch.Tensor,
                   constraint_advantages: torch.Tensor,
                   old_log_probs: torch.Tensor) -> LineSearchResult:
        """
        Perform line search with constraint and KL checks.
        
        Args:
            search_direction: Search direction from trust region solver
            states: State batch
            actions: Action batch
            advantages: Policy advantages
            constraint_advantages: Constraint advantages
            old_log_probs: Old policy log probabilities
            
        Returns:
            Line search result with step size and statistics
        """
        # Store old parameters
        old_params = [p.clone() for p in self.policy.policy_net.parameters()]
        
        # Compute baseline values
        with torch.no_grad():
            old_dist = self.policy.policy_net.get_distribution(states)
            old_policy_loss = -(old_log_probs * advantages).mean().item()
        
        # Evaluate current constraint violations
        old_constraint_violations = torch.zeros(len(self.constraint_manager.constraints), device=self.device)
        for i, constraint in enumerate(self.constraint_manager.constraints):
            with torch.no_grad():
                violations = constraint.evaluate(states, actions)
                old_constraint_violations[i] = violations.mean()
        
        # Line search
        step_size = 1.0
        line_search_steps = 0
        
        for step in range(self.config.backtrack_iters):
            # Apply step
            self._apply_parameter_update(old_params, search_direction * step_size)
            
            # Evaluate new policy
            evaluation = self.policy.evaluate_actions(states, actions)
            new_log_probs = evaluation["log_probs"]
            new_policy_loss = -(new_log_probs * advantages).mean().item()
            
            # Check KL constraint
            new_dist = self.policy.policy_net.get_distribution(states)
            kl_div = kl_divergence(new_dist, old_dist).mean().item()
            
            # Check constraint violations
            new_constraint_violations = torch.zeros(len(self.constraint_manager.constraints), device=self.device)
            for i, constraint in enumerate(self.constraint_manager.constraints):
                violations = constraint.evaluate(states, actions)
                new_constraint_violations[i] = violations.mean()
            
            # Line search conditions
            improvement = old_policy_loss - new_policy_loss
            expected_improvement = -torch.dot(self.compute_policy_gradient(states, actions, advantages, old_log_probs), 
                                           search_direction * step_size).item()
            
            kl_ok = kl_div <= self.config.target_kl * 1.5  # Allow some slack
            constraint_ok = torch.all(new_constraint_violations <= old_constraint_violations + self.config.safety_margin)
            improvement_ok = improvement >= self.config.accept_ratio * expected_improvement
            
            if kl_ok and constraint_ok and improvement_ok:
                break
            
            # Reduce step size
            step_size *= self.config.backtrack_ratio
            line_search_steps += 1
        
        # Create result
        result = LineSearchResult(
            step_size=step_size,
            iterations=line_search_steps + 1,
            kl_divergence=kl_div,
            policy_improvement=improvement,
            constraint_violations=new_constraint_violations,
            success=line_search_steps < self.config.backtrack_iters
        )
        
        return result
    
    def _apply_parameter_update(self, old_params: List[torch.Tensor], update: torch.Tensor) -> None:
        """Apply flat parameter update to policy network."""
        idx = 0
        for i, param in enumerate(self.policy.policy_net.parameters()):
            param_size = param.numel()
            param.data = old_params[i] + update[idx:idx+param_size].view_as(param)
            idx += param_size
    
    def update_value_functions(self, trajectories: List[CPOTrajectory]) -> Dict[str, float]:
        """
        Update value functions using collected trajectories.
        
        Args:
            trajectories: Collected trajectory data
            
        Returns:
            Value function training statistics
        """
        # Prepare data
        all_states = torch.cat([traj.states for traj in trajectories])
        all_rewards = torch.cat([traj.rewards for traj in trajectories])
        all_constraint_costs = torch.cat([traj.constraint_costs for traj in trajectories])
        all_dones = torch.cat([traj.dones for traj in trajectories])
        
        # Compute returns
        policy_returns = []
        constraint_returns = []
        
        for trajectory in trajectories:
            # Policy returns
            returns = torch.zeros_like(trajectory.rewards)
            running_return = 0.0
            for t in reversed(range(len(trajectory.rewards))):
                running_return = trajectory.rewards[t] + self.config.gamma * running_return * (1 - trajectory.dones[t])
                returns[t] = running_return
            policy_returns.append(returns)
            
            # Constraint returns
            traj_constraint_returns = []
            for i in range(trajectory.constraint_costs.shape[1]):
                returns = torch.zeros_like(trajectory.constraint_costs[:, i])
                running_return = 0.0
                for t in reversed(range(len(trajectory.constraint_costs))):
                    running_return = trajectory.constraint_costs[t, i] + self.config.gamma * running_return * (1 - trajectory.dones[t])
                    returns[t] = running_return
                traj_constraint_returns.append(returns)
            constraint_returns.append(torch.stack(traj_constraint_returns, dim=1))
        
        all_policy_returns = torch.cat(policy_returns)
        all_constraint_returns = torch.cat(constraint_returns)
        
        # Update policy value function
        policy_vf_loss = self.value_function.update(all_states, all_policy_returns, self.config.value_iters)
        
        # Update constraint value functions
        constraint_vf_losses = []
        for i, constraint_vf in enumerate(self.constraint_value_functions):
            loss = constraint_vf.update(all_states, all_constraint_returns[:, i], self.config.value_iters)
            constraint_vf_losses.append(loss)
        
        return {
            "policy_vf_loss": policy_vf_loss,
            "constraint_vf_losses": constraint_vf_losses,
            "avg_policy_return": all_policy_returns.mean().item(),
            "avg_constraint_costs": all_constraint_returns.mean(dim=0).tolist()
        }
    
    def update_constraint_threshold(self, constraint_violations: torch.Tensor) -> bool:
        """
        Adapt constraint threshold based on violations.
        
        Args:
            constraint_violations: Current constraint violation levels
            
        Returns:
            Whether threshold was updated
        """
        threshold_updated = False
        
        # Increase threshold if consistently violating
        for i, violation in enumerate(constraint_violations):
            if violation > self.constraint_threshold[i]:
                old_threshold = self.constraint_threshold[i].item()
                self.constraint_threshold[i] = min(
                    violation * 1.1,  # 10% increase
                    self.config.constraint_threshold * 2.0  # Max 2x original
                )
                
                if self.constraint_threshold[i] > old_threshold:
                    threshold_updated = True
                    logger.warning(f"Constraint {i} threshold updated: {old_threshold:.6f} → {self.constraint_threshold[i]:.6f}")
        
        return threshold_updated
    
    def step(self, num_episodes: int = 100) -> CPOState:
        """
        Perform one CPO optimization step.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            CPO optimization state
        """
        self.iteration += 1
        step_start_time = time.time()
        
        logger.info(f"CPO Step {self.iteration}: Collecting {num_episodes} episodes")
        
        # 1. Collect trajectories using current policy
        trajectories = self.collect_trajectories(num_episodes)
        
        if not trajectories:
            raise RuntimeError("No trajectories collected")
        
        # 2. Update value functions
        vf_stats = self.update_value_functions(trajectories)
        
        # 3. Compute advantages
        policy_advantages, constraint_advantages = self.compute_advantages(trajectories)
        
        # Prepare batch data
        all_states = torch.cat([traj.states for traj in trajectories])
        all_actions = torch.cat([traj.actions for traj in trajectories])
        all_log_probs = torch.cat([traj.log_probs for traj in trajectories])
        
        # 4. Compute gradients
        policy_grad = self.compute_policy_gradient(
            all_states, all_actions, policy_advantages, all_log_probs
        )
        
        constraint_grads = self.compute_constraint_gradient(
            all_states, all_actions, constraint_advantages
        )
        
        # 5. Evaluate current constraint violations
        current_violations = torch.zeros(len(self.constraint_manager.constraints), device=self.device)
        for i, constraint in enumerate(self.constraint_manager.constraints):
            violations = constraint.evaluate(all_states, all_actions)
            current_violations[i] = violations.mean()
        
        # 6. Solve trust region problem
        search_direction, tr_info = self.solve_trust_region_problem(
            policy_grad, constraint_grads, current_violations, all_states
        )
        
        # 7. Line search
        line_search_result = self.line_search(
            search_direction, all_states, all_actions,
            policy_advantages, constraint_advantages, all_log_probs
        )
        
        # 8. Update constraint threshold if needed
        threshold_updated = self.update_constraint_threshold(line_search_result.constraint_violations)
        
        # 9. Safety monitoring
        safety_violations = (line_search_result.constraint_violations > self.constraint_threshold).sum().item()
        
        if safety_violations > 0 and self.config.emergency_brake:
            self.consecutive_violations += 1
            if self.consecutive_violations >= 3:
                self.emergency_brake_active = True
                logger.critical("Emergency brake activated due to consecutive safety violations")
        else:
            self.consecutive_violations = 0
            self.emergency_brake_active = False
        
        # 10. Compute performance metrics
        total_steps = sum(len(traj.states) for traj in trajectories)
        average_return = sum(traj.rewards.sum().item() for traj in trajectories) / len(trajectories)
        
        # Create CPO state
        cpo_state = CPOState(
            iteration=self.iteration,
            policy_loss=-(policy_advantages.mean().item()),
            constraint_violations=line_search_result.constraint_violations,
            kl_divergence=line_search_result.kl_divergence,
            step_size=line_search_result.step_size,
            line_search_steps=line_search_result.iterations,
            constraint_threshold_updated=threshold_updated,
            safety_violations=safety_violations,
            total_episodes=self.total_episodes,
            average_return=average_return,
            average_constraint_cost=constraint_advantages.mean(dim=0),
            convergence_metrics={
                "policy_grad_norm": torch.norm(policy_grad).item(),
                "search_direction_norm": torch.norm(search_direction).item(),
                "trust_region_radius": self.config.target_kl,
                "vf_loss": vf_stats["policy_vf_loss"],
                "step_time": time.time() - step_start_time
            }
        )
        
        # Store in history
        self.optimization_history.append(cpo_state)
        
        # Logging
        if self.iteration % self.config.log_frequency == 0:
            self._log_cpo_state(cpo_state, tr_info, vf_stats)
        
        # Metrics logging
        if self.metrics_logger:
            self._log_metrics(cpo_state, tr_info, vf_stats)
        
        logger.info(f"CPO Step {self.iteration} completed in {cpo_state.convergence_metrics['step_time']:.2f}s")
        logger.info(f"  Average return: {cpo_state.average_return:.3f}")
        logger.info(f"  KL divergence: {cpo_state.kl_divergence:.6f}")
        logger.info(f"  Max constraint violation: {cpo_state.constraint_violations.max().item():.6f}")
        logger.info(f"  Step size: {cpo_state.step_size:.6f}")
        
        return cpo_state
    
    def _log_cpo_state(self, state: CPOState, tr_info: Dict, vf_stats: Dict) -> None:
        """Log CPO state information."""
        logger.info(f"=== CPO Iteration {state.iteration} ===")
        logger.info(f"Episodes: {state.total_episodes}")
        logger.info(f"Average return: {state.average_return:.4f}")
        logger.info(f"Policy loss: {state.policy_loss:.6f}")
        logger.info(f"KL divergence: {state.kl_divergence:.6f} (target: {self.config.target_kl:.6f})")
        logger.info(f"Step size: {state.step_size:.6f}")
        logger.info(f"Line search steps: {state.line_search_steps}")
        
        # Constraint information
        for i, violation in enumerate(state.constraint_violations):
            constraint_name = self.constraint_manager.constraints[i].constraint_id
            logger.info(f"Constraint {constraint_name}: {violation:.6f} (threshold: {self.constraint_threshold[i]:.6f})")
        
        # Trust region solver info
        logger.info(f"Trust region info: {tr_info}")
        
        # Value function stats
        logger.info(f"Value function loss: {vf_stats['policy_vf_loss']:.6f}")
        
        if state.safety_violations > 0:
            logger.warning(f"Safety violations: {state.safety_violations}")
        
        if self.emergency_brake_active:
            logger.critical("EMERGENCY BRAKE ACTIVE")
        
        logger.info("=" * 40)
    
    def _log_metrics(self, state: CPOState, tr_info: Dict, vf_stats: Dict) -> None:
        """Log metrics to tracking system."""
        metrics = {
            "cpo/iteration": state.iteration,
            "cpo/average_return": state.average_return,
            "cpo/policy_loss": state.policy_loss,
            "cpo/kl_divergence": state.kl_divergence,
            "cpo/step_size": state.step_size,
            "cpo/line_search_steps": state.line_search_steps,
            "cpo/safety_violations": state.safety_violations,
            "cpo/emergency_brake": int(self.emergency_brake_active),
            "cpo/step_time": state.convergence_metrics["step_time"],
            "cpo/policy_grad_norm": state.convergence_metrics["policy_grad_norm"],
            "cpo/value_function_loss": vf_stats["policy_vf_loss"]
        }
        
        # Constraint metrics
        for i, violation in enumerate(state.constraint_violations):
            constraint_name = self.constraint_manager.constraints[i].constraint_id
            metrics[f"constraints/{constraint_name}"] = violation.item()
            metrics[f"constraints/{constraint_name}_threshold"] = self.constraint_threshold[i].item()
        
        # Trust region metrics
        for key, value in tr_info.items():
            metrics[f"trust_region/{key}"] = value
        
        self.metrics_logger.log_scalars(metrics, step=state.iteration)
    
    def train(self, num_iterations: int, episodes_per_iteration: int = 100) -> List[CPOState]:
        """
        Train policy using CPO algorithm.
        
        Args:
            num_iterations: Number of CPO iterations
            episodes_per_iteration: Episodes to collect per iteration
            
        Returns:
            Training history
        """
        logger.info(f"Starting CPO training for {num_iterations} iterations")
        logger.info(f"Episodes per iteration: {episodes_per_iteration}")
        
        training_start_time = time.time()
        
        for iteration in range(num_iterations):
            if self.emergency_brake_active:
                logger.critical("Training halted due to emergency brake")
                break
            
            try:
                cpo_state = self.step(episodes_per_iteration)
                
                # Save checkpoints
                if iteration % self.config.save_frequency == 0:
                    self.save_checkpoint(f"cpo_checkpoint_iter_{iteration}.pt")
                
            except Exception as e:
                logger.error(f"Error in CPO iteration {iteration}: {e}")
                if self.config.emergency_brake:
                    logger.critical("Emergency brake activated due to training error")
                    self.emergency_brake_active = True
                    break
                else:
                    raise e
        
        training_time = time.time() - training_start_time
        logger.info(f"CPO training completed in {training_time:.2f} seconds")
        logger.info(f"Total episodes: {self.total_episodes}")
        logger.info(f"Final average return: {self.optimization_history[-1].average_return:.4f}")
        
        return self.optimization_history
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save CPO training checkpoint."""
        checkpoint = {
            "iteration": self.iteration,
            "total_episodes": self.total_episodes,
            "policy_state_dict": self.policy.policy_net.state_dict(),
            "value_function_state_dict": self.value_function.state_dict(),
            "constraint_value_functions_state_dict": [cvf.state_dict() for cvf in self.constraint_value_functions],
            "constraint_threshold": self.constraint_threshold,
            "emergency_brake_active": self.emergency_brake_active,
            "consecutive_violations": self.consecutive_violations,
            "optimization_history": self.optimization_history,
            "config": self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"CPO checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load CPO training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.iteration = checkpoint["iteration"]
        self.total_episodes = checkpoint["total_episodes"]
        self.policy.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_function.load_state_dict(checkpoint["value_function_state_dict"])
        
        for i, state_dict in enumerate(checkpoint["constraint_value_functions_state_dict"]):
            self.constraint_value_functions[i].load_state_dict(state_dict)
        
        self.constraint_threshold = checkpoint["constraint_threshold"]
        self.emergency_brake_active = checkpoint["emergency_brake_active"]
        self.consecutive_violations = checkpoint["consecutive_violations"]
        self.optimization_history = checkpoint["optimization_history"]
        
        logger.info(f"CPO checkpoint loaded from {filepath}")
        logger.info(f"Resumed from iteration {self.iteration}")