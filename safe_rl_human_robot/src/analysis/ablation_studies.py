"""
Ablation Studies Framework for Safe RL Component Analysis.

This module provides comprehensive ablation study capabilities to analyze the
contribution of different algorithmic components to overall performance.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import itertools
from copy import deepcopy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..baselines.base_algorithm import BaselineAlgorithm, AlgorithmConfig
from ..evaluation.evaluation_suite import EvaluationSuite, EvaluationConfig, BenchmarkResult
from ..evaluation.environments import StandardizedEnvSuite, RobotPlatform, HumanModelType, SafetyScenario
from ..evaluation.statistics import StatisticalAnalyzer
from ..evaluation.visualization import ResultVisualizer

logger = logging.getLogger(__name__)


@dataclass
class AblationComponent:
    """Definition of a component for ablation study."""
    name: str
    description: str
    default_enabled: bool
    parameter_name: Optional[str] = None
    parameter_values: Optional[List[Any]] = None
    modification_func: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    # Study parameters
    num_seeds: int = 3
    num_evaluation_episodes: int = 50
    max_episode_steps: int = 500
    
    # Environment settings
    robot_platform: RobotPlatform = RobotPlatform.MANIPULATOR_7DOF
    human_model: HumanModelType = HumanModelType.COOPERATIVE
    safety_scenario: SafetyScenario = SafetyScenario.FORCE_LIMITS
    
    # Analysis settings
    significance_level: float = 0.05
    effect_size_threshold: float = 0.3
    
    # Computational resources
    max_workers: int = 2
    
    # Output settings
    results_dir: str = "ablation_results"
    generate_plots: bool = True
    create_report: bool = True
    
    # Reproducibility
    base_seed: int = 12345


@dataclass
class AblationResult:
    """Results from an ablation study."""
    component_combination: Dict[str, bool]
    performance_metrics: Dict[str, float]
    training_time: float
    seed: int
    configuration: Dict[str, Any]


class ComponentRegistry:
    """Registry of ablation study components for different algorithms."""
    
    def __init__(self):
        self.components = {
            'CPO': self._register_cpo_components(),
            'SAC_Lagrangian': self._register_sac_components(),
            'PPO_Lagrangian': self._register_ppo_components(),
            'TD3_Constrained': self._register_td3_components(),
            'TRPO_Constrained': self._register_trpo_components(),
        }
    
    def _register_cpo_components(self) -> List[AblationComponent]:
        """Register CPO-specific ablation components."""
        return [
            AblationComponent(
                name="trust_region",
                description="Trust region constraint for policy updates",
                default_enabled=True,
                parameter_name="use_trust_region",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="conjugate_gradient",
                description="Conjugate gradient solver for natural gradient",
                default_enabled=True,
                parameter_name="use_conjugate_gradient", 
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="line_search",
                description="Backtracking line search for step size",
                default_enabled=True,
                parameter_name="use_line_search",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="advantage_normalization",
                description="Advantage normalization",
                default_enabled=True,
                parameter_name="normalize_advantages",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="gae",
                description="Generalized Advantage Estimation",
                default_enabled=True,
                parameter_name="use_gae",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="cost_value_function",
                description="Separate cost value function",
                default_enabled=True,
                parameter_name="use_cost_value_function",
                parameter_values=[True, False]
            )
        ]
    
    def _register_sac_components(self) -> List[AblationComponent]:
        """Register SAC-specific ablation components."""
        return [
            AblationComponent(
                name="twin_critics",
                description="Twin critic networks (TD3-style)",
                default_enabled=True,
                parameter_name="use_twin_critics",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="target_networks",
                description="Target networks for stability",
                default_enabled=True,
                parameter_name="use_target_networks",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="entropy_regularization",
                description="Entropy regularization for exploration",
                default_enabled=True,
                parameter_name="use_entropy_regularization",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="lagrangian_constraint",
                description="Lagrangian multiplier for safety constraints",
                default_enabled=True,
                parameter_name="use_lagrangian_constraint",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="replay_buffer",
                description="Experience replay buffer",
                default_enabled=True,
                parameter_name="use_replay_buffer",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="cost_critics",
                description="Separate critics for cost estimation",
                default_enabled=True,
                parameter_name="use_cost_critics",
                parameter_values=[True, False]
            )
        ]
    
    def _register_ppo_components(self) -> List[AblationComponent]:
        """Register PPO-specific ablation components."""
        return [
            AblationComponent(
                name="clipped_objective",
                description="Clipped surrogate objective",
                default_enabled=True,
                parameter_name="use_clipped_objective",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="value_clipping",
                description="Value function clipping",
                default_enabled=True,
                parameter_name="use_value_clipping",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="gae",
                description="Generalized Advantage Estimation",
                default_enabled=True,
                parameter_name="use_gae",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="multiple_epochs",
                description="Multiple optimization epochs per batch",
                default_enabled=True,
                parameter_name="use_multiple_epochs",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="entropy_bonus",
                description="Entropy bonus for exploration",
                default_enabled=True,
                parameter_name="use_entropy_bonus",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="lagrangian_constraint",
                description="Lagrangian constraint for safety",
                default_enabled=True,
                parameter_name="use_lagrangian_constraint",
                parameter_values=[True, False]
            )
        ]
    
    def _register_td3_components(self) -> List[AblationComponent]:
        """Register TD3-specific ablation components."""
        return [
            AblationComponent(
                name="twin_critics",
                description="Twin critic networks",
                default_enabled=True,
                parameter_name="use_twin_critics",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="delayed_policy_updates",
                description="Delayed policy updates",
                default_enabled=True,
                parameter_name="use_delayed_updates",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="target_smoothing",
                description="Target policy smoothing",
                default_enabled=True,
                parameter_name="use_target_smoothing",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="exploration_noise",
                description="Action noise for exploration",
                default_enabled=True,
                parameter_name="use_exploration_noise",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="lagrangian_constraint",
                description="Lagrangian safety constraint",
                default_enabled=True,
                parameter_name="use_lagrangian_constraint",
                parameter_values=[True, False]
            )
        ]
    
    def _register_trpo_components(self) -> List[AblationComponent]:
        """Register TRPO-specific ablation components."""
        return [
            AblationComponent(
                name="kl_constraint",
                description="KL divergence constraint",
                default_enabled=True,
                parameter_name="use_kl_constraint",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="conjugate_gradient",
                description="Conjugate gradient for natural gradient",
                default_enabled=True,
                parameter_name="use_conjugate_gradient",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="line_search",
                description="Backtracking line search",
                default_enabled=True,
                parameter_name="use_line_search",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="gae",
                description="Generalized Advantage Estimation",
                default_enabled=True,
                parameter_name="use_gae",
                parameter_values=[True, False]
            ),
            AblationComponent(
                name="safety_constraint",
                description="Safety constraint in optimization",
                default_enabled=True,
                parameter_name="use_safety_constraint",
                parameter_values=[True, False]
            )
        ]
    
    def get_components(self, algorithm_name: str) -> List[AblationComponent]:
        """Get ablation components for a specific algorithm."""
        return self.components.get(algorithm_name, [])
    
    def register_custom_component(self, algorithm_name: str, component: AblationComponent):
        """Register a custom ablation component."""
        if algorithm_name not in self.components:
            self.components[algorithm_name] = []
        self.components[algorithm_name].append(component)


class AblationStudy:
    """Main class for conducting ablation studies."""
    
    def __init__(self, config: AblationConfig):
        self.config = config
        self.component_registry = ComponentRegistry()
        self.env_suite = StandardizedEnvSuite()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer()
        
        # Results storage
        self.results: List[AblationResult] = []
        self.component_importance: Dict[str, float] = {}
        self.interaction_effects: Dict[Tuple[str, str], float] = {}
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Ablation Study framework initialized")
    
    def conduct_comprehensive_ablation(self, 
                                     algorithm_class: type, 
                                     base_config: AlgorithmConfig,
                                     algorithm_name: str,
                                     training_timesteps: int = 50000) -> Dict[str, Any]:
        """Conduct comprehensive ablation study for an algorithm."""
        logger.info(f"Starting comprehensive ablation study for {algorithm_name}")
        
        # Get ablation components for this algorithm
        components = self.component_registry.get_components(algorithm_name)
        
        if not components:
            logger.warning(f"No ablation components registered for {algorithm_name}")
            return {}
        
        # Generate all valid component combinations
        combinations = self._generate_component_combinations(components)
        logger.info(f"Generated {len(combinations)} component combinations")
        
        # Run ablation experiments
        results = self._run_ablation_experiments(
            algorithm_class, base_config, combinations, training_timesteps
        )
        
        # Analyze results
        analysis = self._analyze_ablation_results(results, components)
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_ablation_visualizations(results, components, algorithm_name)
        
        # Create report
        if self.config.create_report:
            report = self._create_ablation_report(analysis, algorithm_name)
        else:
            report = {}
        
        return {
            'results': results,
            'analysis': analysis,
            'report': report,
            'components': components
        }
    
    def conduct_targeted_ablation(self,
                                algorithm_class: type,
                                base_config: AlgorithmConfig,
                                target_components: List[str],
                                algorithm_name: str,
                                training_timesteps: int = 50000) -> Dict[str, Any]:
        """Conduct targeted ablation study for specific components."""
        logger.info(f"Starting targeted ablation study for {target_components}")
        
        # Get only target components
        all_components = self.component_registry.get_components(algorithm_name)
        components = [c for c in all_components if c.name in target_components]
        
        if not components:
            logger.warning(f"No target components found: {target_components}")
            return {}
        
        # Generate combinations for target components only
        combinations = self._generate_component_combinations(components, exhaustive=True)
        
        # Run experiments
        results = self._run_ablation_experiments(
            algorithm_class, base_config, combinations, training_timesteps
        )
        
        # Analyze results
        analysis = self._analyze_ablation_results(results, components)
        
        return {
            'results': results,
            'analysis': analysis,
            'components': components
        }
    
    def analyze_component_interactions(self,
                                     algorithm_class: type,
                                     base_config: AlgorithmConfig, 
                                     component_pairs: List[Tuple[str, str]],
                                     algorithm_name: str,
                                     training_timesteps: int = 50000) -> Dict[str, Any]:
        """Analyze interactions between specific component pairs."""
        logger.info(f"Analyzing component interactions: {component_pairs}")
        
        interaction_results = {}
        all_components = self.component_registry.get_components(algorithm_name)
        
        for comp_a, comp_b in component_pairs:
            # Find the components
            component_a = next((c for c in all_components if c.name == comp_a), None)
            component_b = next((c for c in all_components if c.name == comp_b), None)
            
            if not component_a or not component_b:
                logger.warning(f"Components not found: {comp_a}, {comp_b}")
                continue
            
            # Generate 2x2 factorial design
            combinations = [
                {comp_a: False, comp_b: False},
                {comp_a: True, comp_b: False}, 
                {comp_a: False, comp_b: True},
                {comp_a: True, comp_b: True}
            ]
            
            # Run experiments
            results = self._run_ablation_experiments(
                algorithm_class, base_config, combinations, training_timesteps
            )
            
            # Analyze interaction effect
            interaction_effect = self._compute_interaction_effect(results, comp_a, comp_b)
            
            interaction_results[f"{comp_a}_{comp_b}"] = {
                'results': results,
                'interaction_effect': interaction_effect,
                'significant': abs(interaction_effect) > self.config.effect_size_threshold
            }
        
        return interaction_results
    
    def _generate_component_combinations(self, 
                                       components: List[AblationComponent],
                                       exhaustive: bool = False,
                                       max_combinations: int = 32) -> List[Dict[str, bool]]:
        """Generate valid component combinations for ablation study."""
        if exhaustive or len(components) <= 5:
            # Full factorial design for small number of components
            combinations = []
            for r in range(len(components) + 1):
                for combo in itertools.combinations(components, r):
                    enabled_components = {c.name: c in combo for c in components}
                    if self._is_valid_combination(enabled_components, components):
                        combinations.append(enabled_components)
        else:
            # Use fractional factorial or random sampling for large component sets
            combinations = self._generate_fractional_factorial(components, max_combinations)
        
        # Always include the full configuration (all components enabled)
        full_config = {c.name: True for c in components}
        if full_config not in combinations:
            combinations.append(full_config)
        
        # Always include minimal configuration (only essential components)
        minimal_config = {c.name: c.default_enabled for c in components}
        if minimal_config not in combinations:
            combinations.append(minimal_config)
        
        return combinations
    
    def _generate_fractional_factorial(self, 
                                     components: List[AblationComponent], 
                                     max_combinations: int) -> List[Dict[str, bool]]:
        """Generate fractional factorial design for large component sets."""
        combinations = []
        
        # Add single component ablations (remove one component at a time)
        full_config = {c.name: True for c in components}
        combinations.append(full_config)
        
        for component in components:
            config = full_config.copy()
            config[component.name] = False
            if self._is_valid_combination(config, components):
                combinations.append(config)
        
        # Add random combinations
        np.random.seed(self.config.base_seed)
        while len(combinations) < max_combinations:
            config = {}
            for component in components:
                # Higher probability of enabling important components
                prob = 0.7 if component.default_enabled else 0.3
                config[component.name] = np.random.random() < prob
            
            if self._is_valid_combination(config, components) and config not in combinations:
                combinations.append(config)
        
        return combinations[:max_combinations]
    
    def _is_valid_combination(self, 
                            combination: Dict[str, bool], 
                            components: List[AblationComponent]) -> bool:
        """Check if a component combination is valid."""
        component_map = {c.name: c for c in components}
        
        for comp_name, enabled in combination.items():
            if not enabled:
                continue
                
            component = component_map.get(comp_name)
            if not component:
                continue
            
            # Check dependencies
            for dep in component.dependencies:
                if dep in combination and not combination[dep]:
                    return False
            
            # Check exclusions
            for exc in component.exclusions:
                if exc in combination and combination[exc]:
                    return False
        
        return True
    
    def _run_ablation_experiments(self,
                                algorithm_class: type,
                                base_config: AlgorithmConfig,
                                combinations: List[Dict[str, bool]],
                                training_timesteps: int) -> List[AblationResult]:
        """Run ablation experiments for all combinations."""
        results = []
        
        # Create environment
        env = self.env_suite.create_environment(
            robot_platform=self.config.robot_platform,
            human_model=self.config.human_model,
            safety_scenario=self.config.safety_scenario,
            seed=self.config.base_seed
        )
        
        total_experiments = len(combinations) * self.config.num_seeds
        logger.info(f"Running {total_experiments} ablation experiments")
        
        for i, combination in enumerate(combinations):
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {combination}")
            
            for seed in range(self.config.num_seeds):
                result = self._run_single_ablation(
                    algorithm_class, base_config, combination, env, 
                    training_timesteps, self.config.base_seed + seed
                )
                results.append(result)
        
        return results
    
    def _run_single_ablation(self,
                           algorithm_class: type,
                           base_config: AlgorithmConfig,
                           combination: Dict[str, bool],
                           env,
                           training_timesteps: int,
                           seed: int) -> AblationResult:
        """Run a single ablation experiment."""
        import time
        
        # Create modified config
        modified_config = deepcopy(base_config)
        
        # Apply component modifications
        for comp_name, enabled in combination.items():
            if hasattr(modified_config, comp_name):
                setattr(modified_config, comp_name, enabled)
        
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        env_seed = seed + 1000
        
        # Initialize algorithm with modified config
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        algorithm = algorithm_class(modified_config, state_dim, action_dim)
        
        # Train the algorithm
        start_time = time.time()
        algorithm.learn(training_timesteps, env)
        training_time = time.time() - start_time
        
        # Evaluate the trained algorithm
        performance_metrics = self._evaluate_algorithm(algorithm, env, seed + 2000)
        
        return AblationResult(
            component_combination=combination.copy(),
            performance_metrics=performance_metrics,
            training_time=training_time,
            seed=seed,
            configuration=modified_config.__dict__ if hasattr(modified_config, '__dict__') else {}
        )
    
    def _evaluate_algorithm(self, algorithm: BaselineAlgorithm, env, eval_seed: int) -> Dict[str, float]:
        """Evaluate algorithm and return performance metrics."""
        np.random.seed(eval_seed)
        
        episode_rewards = []
        episode_costs = []
        safety_violations = []
        
        for episode in range(self.config.num_evaluation_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_cost = 0
            violations = 0
            
            for step in range(self.config.max_episode_steps):
                action, _ = algorithm.predict(obs, deterministic=True)
                next_obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_cost += info.get('cost', 0)
                
                if info.get('constraint_violation', False):
                    violations += 1
                
                if done:
                    break
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            safety_violations.append(violations)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs),
            'violation_rate': np.mean(safety_violations),
            'constraint_satisfaction': np.mean([c <= 25.0 for c in episode_costs])  # Assuming cost limit of 25
        }
    
    def _analyze_ablation_results(self, 
                                results: List[AblationResult],
                                components: List[AblationComponent]) -> Dict[str, Any]:
        """Analyze ablation study results."""
        analysis = {
            'component_importance': {},
            'performance_summary': {},
            'statistical_tests': {},
            'best_configurations': {}
        }
        
        # Calculate component importance
        analysis['component_importance'] = self._calculate_component_importance(results, components)
        
        # Performance summary
        analysis['performance_summary'] = self._summarize_performance(results)
        
        # Statistical significance tests
        analysis['statistical_tests'] = self._perform_statistical_tests(results, components)
        
        # Identify best configurations
        analysis['best_configurations'] = self._identify_best_configurations(results)
        
        return analysis
    
    def _calculate_component_importance(self, 
                                      results: List[AblationResult],
                                      components: List[AblationComponent]) -> Dict[str, Dict[str, float]]:
        """Calculate importance of each component."""
        importance = {}
        
        for component in components:
            comp_name = component.name
            
            # Separate results with and without this component
            with_component = [r for r in results if r.component_combination.get(comp_name, False)]
            without_component = [r for r in results if not r.component_combination.get(comp_name, True)]
            
            if with_component and without_component:
                # Calculate effect sizes for different metrics
                importance[comp_name] = {}
                
                for metric in ['mean_reward', 'mean_cost', 'violation_rate', 'constraint_satisfaction']:
                    with_values = [r.performance_metrics[metric] for r in with_component if metric in r.performance_metrics]
                    without_values = [r.performance_metrics[metric] for r in without_component if metric in r.performance_metrics]
                    
                    if with_values and without_values:
                        # Cohen's d effect size
                        mean_diff = np.mean(with_values) - np.mean(without_values)
                        pooled_std = np.sqrt((np.var(with_values, ddof=1) + np.var(without_values, ddof=1)) / 2)
                        
                        if pooled_std > 0:
                            cohens_d = mean_diff / pooled_std
                            importance[comp_name][f'{metric}_effect_size'] = cohens_d
                        else:
                            importance[comp_name][f'{metric}_effect_size'] = 0.0
                        
                        importance[comp_name][f'{metric}_with'] = np.mean(with_values)
                        importance[comp_name][f'{metric}_without'] = np.mean(without_values)
        
        return importance
    
    def _summarize_performance(self, results: List[AblationResult]) -> Dict[str, Any]:
        """Summarize overall performance across configurations."""
        if not results:
            return {}
        
        summary = {}
        
        # Overall statistics
        all_rewards = [r.performance_metrics.get('mean_reward', 0) for r in results]
        all_costs = [r.performance_metrics.get('mean_cost', 0) for r in results]
        all_violations = [r.performance_metrics.get('violation_rate', 0) for r in results]
        
        summary['overall'] = {
            'reward_mean': np.mean(all_rewards),
            'reward_std': np.std(all_rewards),
            'reward_range': (np.min(all_rewards), np.max(all_rewards)),
            'cost_mean': np.mean(all_costs),
            'cost_std': np.std(all_costs),
            'violation_mean': np.mean(all_violations),
            'violation_std': np.std(all_violations)
        }
        
        # Configuration diversity
        unique_configs = set()
        for result in results:
            config_tuple = tuple(sorted(result.component_combination.items()))
            unique_configs.add(config_tuple)
        
        summary['diversity'] = {
            'total_results': len(results),
            'unique_configurations': len(unique_configs),
            'seeds_per_config': len(results) / len(unique_configs) if unique_configs else 1
        }
        
        return summary
    
    def _perform_statistical_tests(self, 
                                 results: List[AblationResult],
                                 components: List[AblationComponent]) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}
        
        for component in components:
            comp_name = component.name
            
            # Group results by component state
            with_comp = [r for r in results if r.component_combination.get(comp_name, False)]
            without_comp = [r for r in results if not r.component_combination.get(comp_name, True)]
            
            if len(with_comp) >= 3 and len(without_comp) >= 3:
                tests[comp_name] = {}
                
                for metric in ['mean_reward', 'mean_cost', 'violation_rate']:
                    with_values = [r.performance_metrics.get(metric, 0) for r in with_comp]
                    without_values = [r.performance_metrics.get(metric, 0) for r in without_comp]
                    
                    if with_values and without_values:
                        # Mann-Whitney U test
                        test_result = self.statistical_analyzer.hypothesis_test.mann_whitney_u(
                            with_values, without_values, self.config.significance_level
                        )
                        
                        tests[comp_name][metric] = {
                            'statistic': test_result.statistic,
                            'p_value': test_result.p_value,
                            'significant': test_result.significant,
                            'effect_size': test_result.effect_size,
                            'interpretation': test_result.interpretation
                        }
        
        return tests
    
    def _identify_best_configurations(self, results: List[AblationResult]) -> Dict[str, Any]:
        """Identify best performing configurations."""
        if not results:
            return {}
        
        best_configs = {}
        
        # Best by reward
        best_reward = max(results, key=lambda r: r.performance_metrics.get('mean_reward', -float('inf')))
        best_configs['best_reward'] = {
            'configuration': best_reward.component_combination,
            'performance': best_reward.performance_metrics,
            'training_time': best_reward.training_time
        }
        
        # Best by safety (lowest violation rate)
        best_safety = min(results, key=lambda r: r.performance_metrics.get('violation_rate', float('inf')))
        best_configs['best_safety'] = {
            'configuration': best_safety.component_combination,
            'performance': best_safety.performance_metrics,
            'training_time': best_safety.training_time
        }
        
        # Best overall (balanced reward and safety)
        def balanced_score(result):
            reward = result.performance_metrics.get('mean_reward', 0)
            violations = result.performance_metrics.get('violation_rate', float('inf'))
            # Higher reward is better, lower violations are better
            return reward - 10 * violations
        
        best_balanced = max(results, key=balanced_score)
        best_configs['best_balanced'] = {
            'configuration': best_balanced.component_combination,
            'performance': best_balanced.performance_metrics,
            'training_time': best_balanced.training_time,
            'balanced_score': balanced_score(best_balanced)
        }
        
        return best_configs
    
    def _compute_interaction_effect(self, 
                                  results: List[AblationResult],
                                  comp_a: str, 
                                  comp_b: str) -> float:
        """Compute interaction effect between two components."""
        # Find results for 2x2 factorial design
        both_off = [r for r in results if not r.component_combination.get(comp_a, True) 
                   and not r.component_combination.get(comp_b, True)]
        a_on_b_off = [r for r in results if r.component_combination.get(comp_a, False) 
                     and not r.component_combination.get(comp_b, True)]
        a_off_b_on = [r for r in results if not r.component_combination.get(comp_a, True) 
                     and r.component_combination.get(comp_b, False)]
        both_on = [r for r in results if r.component_combination.get(comp_a, False) 
                  and r.component_combination.get(comp_b, False)]
        
        if not all([both_off, a_on_b_off, a_off_b_on, both_on]):
            return 0.0
        
        # Calculate mean rewards for each condition
        reward_00 = np.mean([r.performance_metrics.get('mean_reward', 0) for r in both_off])
        reward_10 = np.mean([r.performance_metrics.get('mean_reward', 0) for r in a_on_b_off])  
        reward_01 = np.mean([r.performance_metrics.get('mean_reward', 0) for r in a_off_b_on])
        reward_11 = np.mean([r.performance_metrics.get('mean_reward', 0) for r in both_on])
        
        # Interaction effect = (reward_11 - reward_10) - (reward_01 - reward_00)
        interaction_effect = (reward_11 - reward_10) - (reward_01 - reward_00)
        
        return interaction_effect
    
    def _generate_ablation_visualizations(self, 
                                        results: List[AblationResult],
                                        components: List[AblationComponent],
                                        algorithm_name: str):
        """Generate visualizations for ablation study results."""
        # Component importance plot
        self._plot_component_importance(results, components, algorithm_name)
        
        # Performance distribution plot
        self._plot_performance_distribution(results, algorithm_name)
        
        # Configuration comparison plot
        self._plot_configuration_comparison(results, algorithm_name)
    
    def _plot_component_importance(self, 
                                 results: List[AblationResult],
                                 components: List[AblationComponent], 
                                 algorithm_name: str):
        """Plot component importance analysis."""
        import matplotlib.pyplot as plt
        
        component_effects = {}
        
        for component in components:
            comp_name = component.name
            with_comp = [r.performance_metrics.get('mean_reward', 0) 
                        for r in results if r.component_combination.get(comp_name, False)]
            without_comp = [r.performance_metrics.get('mean_reward', 0) 
                           for r in results if not r.component_combination.get(comp_name, True)]
            
            if with_comp and without_comp:
                effect = np.mean(with_comp) - np.mean(without_comp)
                component_effects[comp_name] = effect
        
        if component_effects:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            components_sorted = sorted(component_effects.items(), key=lambda x: abs(x[1]), reverse=True)
            names, effects = zip(*components_sorted)
            
            colors = ['green' if e > 0 else 'red' for e in effects]
            bars = ax.bar(names, effects, color=colors, alpha=0.7)
            
            ax.set_xlabel('Components')
            ax.set_ylabel('Effect on Mean Reward')
            ax.set_title(f'{algorithm_name} - Component Importance Analysis')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Rotate labels if needed
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            save_path = self.results_dir / f'{algorithm_name}_component_importance.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Component importance plot saved to {save_path}")
    
    def _plot_performance_distribution(self, results: List[AblationResult], algorithm_name: str):
        """Plot performance distribution across configurations."""
        import matplotlib.pyplot as plt
        
        rewards = [r.performance_metrics.get('mean_reward', 0) for r in results]
        violations = [r.performance_metrics.get('violation_rate', 0) for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reward distribution
        ax1.hist(rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Mean Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{algorithm_name} - Reward Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Safety violation distribution
        ax2.hist(violations, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Violation Rate') 
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{algorithm_name} - Safety Violation Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.results_dir / f'{algorithm_name}_performance_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Performance distribution plot saved to {save_path}")
    
    def _plot_configuration_comparison(self, results: List[AblationResult], algorithm_name: str):
        """Plot comparison of top configurations."""
        import matplotlib.pyplot as plt
        
        # Group results by configuration and take mean
        config_performance = {}
        for result in results:
            config_key = tuple(sorted(result.component_combination.items()))
            if config_key not in config_performance:
                config_performance[config_key] = []
            config_performance[config_key].append(result.performance_metrics.get('mean_reward', 0))
        
        # Calculate mean performance for each configuration
        config_means = {k: np.mean(v) for k, v in config_performance.items()}
        
        # Select top 10 configurations
        top_configs = sorted(config_means.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_configs:
            fig, ax = plt.subplots(figsize=(15, 8))
            
            config_labels = []
            performances = []
            
            for i, (config, performance) in enumerate(top_configs):
                # Create readable label from configuration
                enabled_components = [comp for comp, enabled in config if enabled]
                label = f"Config {i+1}"
                config_labels.append(label)
                performances.append(performance)
            
            bars = ax.bar(config_labels, performances, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Configuration')
            ax.set_ylabel('Mean Reward')
            ax.set_title(f'{algorithm_name} - Top 10 Configuration Comparison')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            save_path = self.results_dir / f'{algorithm_name}_top_configurations.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Configuration comparison plot saved to {save_path}")
    
    def _create_ablation_report(self, analysis: Dict[str, Any], algorithm_name: str) -> Dict[str, Any]:
        """Create comprehensive ablation study report."""
        report = {
            'algorithm': algorithm_name,
            'study_configuration': self.config.__dict__,
            'executive_summary': self._generate_executive_summary(analysis),
            'detailed_analysis': analysis,
            'recommendations': self._generate_recommendations(analysis),
            'methodology': self._describe_methodology()
        }
        
        # Save report as JSON
        report_path = self.results_dir / f'{algorithm_name}_ablation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Ablation study report saved to {report_path}")
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate executive summary of ablation study."""
        summary = []
        
        # Component importance
        if 'component_importance' in analysis:
            importance = analysis['component_importance']
            most_important = max(importance.keys(), 
                               key=lambda k: abs(importance[k].get('mean_reward_effect_size', 0))
                               ) if importance else None
            
            if most_important:
                effect_size = importance[most_important].get('mean_reward_effect_size', 0)
                summary.append(f"Most important component: {most_important} (effect size: {effect_size:.3f})")
        
        # Best configurations
        if 'best_configurations' in analysis:
            best_configs = analysis['best_configurations']
            if 'best_balanced' in best_configs:
                summary.append("Best balanced configuration identified with optimal performance-safety trade-off")
        
        # Statistical significance
        if 'statistical_tests' in analysis:
            significant_components = []
            for comp, tests in analysis['statistical_tests'].items():
                if any(test.get('significant', False) for test in tests.values()):
                    significant_components.append(comp)
            
            if significant_components:
                summary.append(f"Statistically significant components: {', '.join(significant_components)}")
        
        return summary
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations from ablation study."""
        recommendations = []
        
        # Component usage recommendations
        if 'component_importance' in analysis:
            for comp, stats in analysis['component_importance'].items():
                effect_size = stats.get('mean_reward_effect_size', 0)
                if abs(effect_size) > self.config.effect_size_threshold:
                    if effect_size > 0:
                        recommendations.append(f"Recommend enabling {comp} (positive effect: +{effect_size:.3f})")
                    else:
                        recommendations.append(f"Consider disabling {comp} (negative effect: {effect_size:.3f})")
        
        # Configuration recommendations
        if 'best_configurations' in analysis:
            best_configs = analysis['best_configurations']
            if 'best_balanced' in best_configs:
                config = best_configs['best_balanced']['configuration']
                enabled = [k for k, v in config.items() if v]
                recommendations.append(f"Recommended component set: {', '.join(enabled)}")
        
        return recommendations
    
    def _describe_methodology(self) -> Dict[str, Any]:
        """Describe the ablation study methodology."""
        return {
            'design': 'Systematic component ablation study',
            'evaluation_metrics': ['mean_reward', 'mean_cost', 'violation_rate', 'constraint_satisfaction'],
            'statistical_tests': 'Mann-Whitney U test with effect size calculation',
            'significance_level': self.config.significance_level,
            'effect_size_threshold': self.config.effect_size_threshold,
            'seeds_per_configuration': self.config.num_seeds,
            'evaluation_episodes': self.config.num_evaluation_episodes
        }