"""
Cross-Domain Evaluation System for Safe RL Generalization Analysis.

This module provides comprehensive cross-domain evaluation capabilities to assess
generalization across different robot platforms, human behaviors, and environmental conditions.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from ..baselines.base_algorithm import BaselineAlgorithm, AlgorithmConfig
from .environments import StandardizedEnvSuite, RobotPlatform, HumanModelType, SafetyScenario, EnvironmentConfig
from .evaluation_suite import EvaluationSuite, EvaluationConfig, BenchmarkResult
from .statistics import StatisticalAnalyzer
from .visualization import ResultVisualizer
from .metrics import MetricAggregator

logger = logging.getLogger(__name__)


@dataclass
class DomainTransferConfig:
    """Configuration for domain transfer evaluation."""
    # Source domain (training)
    source_platforms: List[RobotPlatform] = field(default_factory=lambda: [RobotPlatform.MANIPULATOR_7DOF])
    source_human_models: List[HumanModelType] = field(default_factory=lambda: [HumanModelType.COOPERATIVE])
    source_safety_scenarios: List[SafetyScenario] = field(default_factory=lambda: [SafetyScenario.FORCE_LIMITS])
    
    # Target domains (evaluation)
    target_platforms: List[RobotPlatform] = field(default_factory=lambda: [
        RobotPlatform.MANIPULATOR_6DOF, RobotPlatform.MOBILE_BASE
    ])
    target_human_models: List[HumanModelType] = field(default_factory=lambda: [
        HumanModelType.ADVERSARIAL, HumanModelType.INCONSISTENT
    ])
    target_safety_scenarios: List[SafetyScenario] = field(default_factory=lambda: [
        SafetyScenario.OBSTACLE_AVOIDANCE, SafetyScenario.WORKSPACE_BOUNDS
    ])
    
    # Evaluation parameters
    num_seeds: int = 5
    num_evaluation_episodes: int = 100
    training_timesteps: int = 500000
    
    # Domain adaptation settings
    enable_fine_tuning: bool = True
    fine_tuning_timesteps: int = 50000
    adaptation_learning_rate: float = 1e-4
    
    # Analysis settings
    compute_domain_distance: bool = True
    analyze_failure_modes: bool = True
    
    # Output settings
    results_dir: str = "cross_domain_results"
    save_intermediate: bool = True


@dataclass
class DomainCharacteristics:
    """Characteristics of a specific domain."""
    platform: RobotPlatform
    human_model: HumanModelType
    safety_scenario: SafetyScenario
    
    # Computed characteristics
    state_dim: int = 0
    action_dim: int = 0
    dynamics_complexity: float = 0.0
    safety_criticality: float = 0.0
    human_predictability: float = 0.0
    task_difficulty: float = 0.0


@dataclass
class TransferResult:
    """Results from domain transfer evaluation."""
    source_domain: DomainCharacteristics
    target_domain: DomainCharacteristics
    algorithm_name: str
    
    # Performance metrics
    source_performance: Dict[str, float]
    target_performance: Dict[str, float]
    transfer_ratio: float
    generalization_gap: float
    
    # Adaptation results (if enabled)
    adapted_performance: Optional[Dict[str, float]] = None
    adaptation_improvement: Optional[float] = None
    
    # Analysis metrics
    domain_distance: Optional[float] = None
    failure_modes: List[str] = field(default_factory=list)
    
    # Raw data
    source_episodes: List[Dict[str, Any]] = field(default_factory=list)
    target_episodes: List[Dict[str, Any]] = field(default_factory=list)


class DomainAnalyzer:
    """Analyze characteristics and relationships between domains."""
    
    def __init__(self):
        self.env_suite = StandardizedEnvSuite()
        self.metric_aggregator = MetricAggregator()
    
    def characterize_domain(self, 
                          platform: RobotPlatform,
                          human_model: HumanModelType,
                          safety_scenario: SafetyScenario,
                          seed: int = 42) -> DomainCharacteristics:
        """Analyze characteristics of a specific domain."""
        
        # Create environment to extract characteristics
        env = self.env_suite.create_environment(
            robot_platform=platform,
            human_model=human_model,
            safety_scenario=safety_scenario,
            seed=seed
        )
        
        characteristics = DomainCharacteristics(
            platform=platform,
            human_model=human_model,
            safety_scenario=safety_scenario
        )
        
        # Basic dimensionality
        characteristics.state_dim = env.observation_space.shape[0]
        characteristics.action_dim = env.action_space.shape[0]
        
        # Analyze dynamics complexity
        characteristics.dynamics_complexity = self._estimate_dynamics_complexity(env)
        
        # Analyze safety criticality
        characteristics.safety_criticality = self._estimate_safety_criticality(safety_scenario)
        
        # Analyze human predictability
        characteristics.human_predictability = self._estimate_human_predictability(human_model)
        
        # Estimate task difficulty
        characteristics.task_difficulty = self._estimate_task_difficulty(env)
        
        return characteristics
    
    def compute_domain_distance(self, 
                               domain_a: DomainCharacteristics,
                               domain_b: DomainCharacteristics) -> float:
        """Compute distance between two domains."""
        
        # Normalize characteristics to [0, 1] scale
        features_a = np.array([
            domain_a.state_dim / 20.0,  # Normalize assuming max 20 dimensions
            domain_a.action_dim / 10.0,  # Normalize assuming max 10 dimensions
            domain_a.dynamics_complexity,
            domain_a.safety_criticality,
            domain_a.human_predictability,
            domain_a.task_difficulty
        ])
        
        features_b = np.array([
            domain_b.state_dim / 20.0,
            domain_b.action_dim / 10.0,
            domain_b.dynamics_complexity,
            domain_b.safety_criticality,
            domain_b.human_predictability,
            domain_b.task_difficulty
        ])
        
        # Euclidean distance with feature weights
        weights = np.array([0.2, 0.2, 0.25, 0.15, 0.1, 0.1])  # Higher weight for dynamics and safety
        distance = np.sqrt(np.sum(weights * (features_a - features_b) ** 2))
        
        return distance
    
    def _estimate_dynamics_complexity(self, env) -> float:
        """Estimate complexity of environment dynamics."""
        complexity_scores = {
            RobotPlatform.MANIPULATOR_7DOF: 0.9,
            RobotPlatform.MANIPULATOR_6DOF: 0.8,
            RobotPlatform.MOBILE_BASE: 0.5,
            RobotPlatform.HUMANOID: 1.0,
            RobotPlatform.QUADRUPED: 0.7,
            RobotPlatform.AERIAL_VEHICLE: 0.6
        }
        
        if hasattr(env, 'config') and hasattr(env.config, 'robot_platform'):
            return complexity_scores.get(env.config.robot_platform, 0.5)
        return 0.5
    
    def _estimate_safety_criticality(self, safety_scenario: SafetyScenario) -> float:
        """Estimate safety criticality of scenario."""
        criticality_scores = {
            SafetyScenario.HUMAN_SAFETY: 1.0,
            SafetyScenario.COLLISION_PREVENTION: 0.9,
            SafetyScenario.FORCE_LIMITS: 0.7,
            SafetyScenario.WORKSPACE_BOUNDS: 0.6,
            SafetyScenario.VELOCITY_CONSTRAINTS: 0.5,
            SafetyScenario.OBSTACLE_AVOIDANCE: 0.8
        }
        
        return criticality_scores.get(safety_scenario, 0.5)
    
    def _estimate_human_predictability(self, human_model: HumanModelType) -> float:
        """Estimate predictability of human behavior."""
        predictability_scores = {
            HumanModelType.EXPERT: 0.9,
            HumanModelType.COOPERATIVE: 0.8,
            HumanModelType.ADAPTIVE: 0.6,
            HumanModelType.NOVICE: 0.5,
            HumanModelType.INCONSISTENT: 0.2,
            HumanModelType.ADVERSARIAL: 0.1
        }
        
        return predictability_scores.get(human_model, 0.5)
    
    def _estimate_task_difficulty(self, env) -> float:
        """Estimate overall task difficulty."""
        # Simple heuristic based on observation and action space sizes
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Normalize by typical dimensions and combine
        difficulty = (obs_dim / 20.0 + action_dim / 10.0) / 2.0
        return min(1.0, difficulty)


class CrossDomainEvaluator:
    """Main class for cross-domain evaluation."""
    
    def __init__(self, config: DomainTransferConfig):
        self.config = config
        self.domain_analyzer = DomainAnalyzer()
        self.env_suite = StandardizedEnvSuite()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer()
        
        # Results storage
        self.transfer_results: List[TransferResult] = []
        self.domain_characteristics: Dict[str, DomainCharacteristics] = {}
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Cross-Domain Evaluator initialized")
    
    def evaluate_cross_domain_transfer(self,
                                     algorithms: Dict[str, BaselineAlgorithm]) -> Dict[str, Any]:
        """Evaluate cross-domain transfer for all algorithms."""
        logger.info("Starting comprehensive cross-domain evaluation")
        
        # Step 1: Characterize all domains
        self._characterize_all_domains()
        
        # Step 2: Train algorithms on source domains
        source_models = self._train_on_source_domains(algorithms)
        
        # Step 3: Evaluate on target domains
        transfer_results = self._evaluate_on_target_domains(source_models)
        
        # Step 4: Perform domain adaptation (if enabled)
        if self.config.enable_fine_tuning:
            adapted_results = self._perform_domain_adaptation(source_models)
            transfer_results.extend(adapted_results)
        
        self.transfer_results = transfer_results
        
        # Step 5: Analyze results
        analysis = self._analyze_transfer_results()
        
        # Step 6: Generate visualizations
        self._generate_cross_domain_visualizations()
        
        # Step 7: Create comprehensive report
        report = self._create_cross_domain_report(analysis)
        
        return {
            'transfer_results': transfer_results,
            'domain_characteristics': self.domain_characteristics,
            'analysis': analysis,
            'report': report
        }
    
    def _characterize_all_domains(self):
        """Characterize all source and target domains."""
        logger.info("Characterizing domains...")
        
        all_combinations = []
        
        # Source domains
        for platform in self.config.source_platforms:
            for human_model in self.config.source_human_models:
                for safety_scenario in self.config.source_safety_scenarios:
                    all_combinations.append((platform, human_model, safety_scenario, 'source'))
        
        # Target domains
        for platform in self.config.target_platforms:
            for human_model in self.config.target_human_models:
                for safety_scenario in self.config.target_safety_scenarios:
                    all_combinations.append((platform, human_model, safety_scenario, 'target'))
        
        for platform, human_model, safety_scenario, domain_type in all_combinations:
            domain_key = f"{platform.name}_{human_model.name}_{safety_scenario.name}"
            
            characteristics = self.domain_analyzer.characterize_domain(
                platform, human_model, safety_scenario
            )
            
            self.domain_characteristics[domain_key] = characteristics
            logger.info(f"Characterized {domain_type} domain: {domain_key}")
    
    def _train_on_source_domains(self, 
                               algorithms: Dict[str, BaselineAlgorithm]) -> Dict[str, Dict[str, BaselineAlgorithm]]:
        """Train algorithms on all source domains."""
        logger.info("Training algorithms on source domains...")
        
        source_models = {}
        
        for alg_name, algorithm in algorithms.items():
            source_models[alg_name] = {}
            
            for platform in self.config.source_platforms:
                for human_model in self.config.source_human_models:
                    for safety_scenario in self.config.source_safety_scenarios:
                        domain_key = f"{platform.name}_{human_model.name}_{safety_scenario.name}"
                        
                        logger.info(f"Training {alg_name} on source domain: {domain_key}")
                        
                        # Create environment
                        env = self.env_suite.create_environment(
                            robot_platform=platform,
                            human_model=human_model,
                            safety_scenario=safety_scenario,
                            seed=self.config.base_seed
                        )
                        
                        # Train algorithm
                        trained_algorithm = self._train_algorithm(algorithm, env, self.config.training_timesteps)
                        source_models[alg_name][domain_key] = trained_algorithm
        
        # Save trained models
        if self.config.save_intermediate:
            models_path = self.results_dir / "trained_source_models.pkl"
            with open(models_path, 'wb') as f:
                pickle.dump(source_models, f)
            logger.info(f"Saved trained source models to {models_path}")
        
        return source_models
    
    def _evaluate_on_target_domains(self, 
                                  source_models: Dict[str, Dict[str, BaselineAlgorithm]]) -> List[TransferResult]:
        """Evaluate trained models on target domains."""
        logger.info("Evaluating on target domains...")
        
        transfer_results = []
        
        for alg_name, models_dict in source_models.items():
            for source_domain_key, trained_model in models_dict.items():
                source_characteristics = self.domain_characteristics[source_domain_key]
                
                # Evaluate on all target domains
                for platform in self.config.target_platforms:
                    for human_model in self.config.target_human_models:
                        for safety_scenario in self.config.target_safety_scenarios:
                            target_domain_key = f"{platform.name}_{human_model.name}_{safety_scenario.name}"
                            target_characteristics = self.domain_characteristics[target_domain_key]
                            
                            logger.info(f"Evaluating {alg_name} transfer: {source_domain_key} -> {target_domain_key}")
                            
                            # Create target environment
                            target_env = self.env_suite.create_environment(
                                robot_platform=platform,
                                human_model=human_model,
                                safety_scenario=safety_scenario,
                                seed=self.config.base_seed + 1000
                            )
                            
                            # Evaluate performance
                            target_performance = self._evaluate_algorithm_performance(trained_model, target_env)
                            
                            # Get source performance (for comparison)
                            source_env = self.env_suite.create_environment(
                                robot_platform=source_characteristics.platform,
                                human_model=source_characteristics.human_model,
                                safety_scenario=source_characteristics.safety_scenario,
                                seed=self.config.base_seed + 2000
                            )
                            source_performance = self._evaluate_algorithm_performance(trained_model, source_env)
                            
                            # Compute transfer metrics
                            transfer_ratio = self._compute_transfer_ratio(source_performance, target_performance)
                            generalization_gap = self._compute_generalization_gap(source_performance, target_performance)
                            
                            # Compute domain distance
                            domain_distance = None
                            if self.config.compute_domain_distance:
                                domain_distance = self.domain_analyzer.compute_domain_distance(
                                    source_characteristics, target_characteristics
                                )
                            
                            # Analyze failure modes
                            failure_modes = []
                            if self.config.analyze_failure_modes:
                                failure_modes = self._analyze_failure_modes(target_performance)
                            
                            # Create transfer result
                            result = TransferResult(
                                source_domain=source_characteristics,
                                target_domain=target_characteristics,
                                algorithm_name=alg_name,
                                source_performance=source_performance,
                                target_performance=target_performance,
                                transfer_ratio=transfer_ratio,
                                generalization_gap=generalization_gap,
                                domain_distance=domain_distance,
                                failure_modes=failure_modes
                            )
                            
                            transfer_results.append(result)
        
        return transfer_results
    
    def _perform_domain_adaptation(self, 
                                 source_models: Dict[str, Dict[str, BaselineAlgorithm]]) -> List[TransferResult]:
        """Perform domain adaptation through fine-tuning."""
        logger.info("Performing domain adaptation...")
        
        adapted_results = []
        
        for alg_name, models_dict in source_models.items():
            for source_domain_key, trained_model in models_dict.items():
                source_characteristics = self.domain_characteristics[source_domain_key]
                
                # Adapt to each target domain
                for platform in self.config.target_platforms:
                    for human_model in self.config.target_human_models:
                        for safety_scenario in self.config.target_safety_scenarios:
                            target_domain_key = f"{platform.name}_{human_model.name}_{safety_scenario.name}"
                            target_characteristics = self.domain_characteristics[target_domain_key]
                            
                            logger.info(f"Adapting {alg_name}: {source_domain_key} -> {target_domain_key}")
                            
                            # Create target environment
                            target_env = self.env_suite.create_environment(
                                robot_platform=platform,
                                human_model=human_model,
                                safety_scenario=safety_scenario,
                                seed=self.config.base_seed + 3000
                            )
                            
                            # Fine-tune on target domain
                            adapted_model = self._fine_tune_algorithm(
                                trained_model, target_env, self.config.fine_tuning_timesteps
                            )
                            
                            # Evaluate adapted performance
                            adapted_performance = self._evaluate_algorithm_performance(adapted_model, target_env)
                            
                            # Get original transfer performance for comparison
                            original_result = next(
                                (r for r in self.transfer_results if 
                                 r.algorithm_name == alg_name and 
                                 r.source_domain.platform == source_characteristics.platform and
                                 r.target_domain.platform == target_characteristics.platform), 
                                None
                            )
                            
                            if original_result:
                                # Compute adaptation improvement
                                adaptation_improvement = self._compute_adaptation_improvement(
                                    original_result.target_performance, adapted_performance
                                )
                                
                                # Create adapted result
                                adapted_result = TransferResult(
                                    source_domain=source_characteristics,
                                    target_domain=target_characteristics,
                                    algorithm_name=f"{alg_name}_adapted",
                                    source_performance=original_result.source_performance,
                                    target_performance=original_result.target_performance,
                                    transfer_ratio=original_result.transfer_ratio,
                                    generalization_gap=original_result.generalization_gap,
                                    adapted_performance=adapted_performance,
                                    adaptation_improvement=adaptation_improvement,
                                    domain_distance=original_result.domain_distance,
                                    failure_modes=self._analyze_failure_modes(adapted_performance)
                                )
                                
                                adapted_results.append(adapted_result)
        
        return adapted_results
    
    def _train_algorithm(self, algorithm: BaselineAlgorithm, env, timesteps: int) -> BaselineAlgorithm:
        """Train algorithm on environment."""
        # Create a copy of the algorithm for training
        import copy
        trained_algorithm = copy.deepcopy(algorithm)
        
        # Train the algorithm
        trained_algorithm.learn(timesteps, env)
        
        return trained_algorithm
    
    def _fine_tune_algorithm(self, algorithm: BaselineAlgorithm, env, timesteps: int) -> BaselineAlgorithm:
        """Fine-tune algorithm on target domain."""
        import copy
        adapted_algorithm = copy.deepcopy(algorithm)
        
        # Reduce learning rate for fine-tuning
        if hasattr(adapted_algorithm, 'config'):
            original_lr = adapted_algorithm.config.learning_rate
            adapted_algorithm.config.learning_rate = self.config.adaptation_learning_rate
        
        # Fine-tune
        adapted_algorithm.learn(timesteps, env)
        
        # Restore original learning rate
        if hasattr(adapted_algorithm, 'config'):
            adapted_algorithm.config.learning_rate = original_lr
        
        return adapted_algorithm
    
    def _evaluate_algorithm_performance(self, algorithm: BaselineAlgorithm, env) -> Dict[str, float]:
        """Evaluate algorithm performance on environment."""
        episode_rewards = []
        episode_costs = []
        safety_violations = []
        human_metrics = []
        
        for episode in range(self.config.num_evaluation_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_cost = 0
            violations = 0
            
            for step in range(1000):  # Max episode length
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
            
            # Collect human metrics if available
            if hasattr(env, 'get_human_metrics'):
                human_metrics.append(env.get_human_metrics())
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_cost': np.mean(episode_costs),
            'violation_rate': np.mean(safety_violations),
            'constraint_satisfaction': np.mean([c <= 25.0 for c in episode_costs]),
            'success_rate': np.mean([r > 0 for r in episode_rewards]),
            'human_satisfaction': np.mean([h.get('satisfaction', 0.5) for h in human_metrics]) if human_metrics else 0.5
        }
    
    def _compute_transfer_ratio(self, source_perf: Dict[str, float], target_perf: Dict[str, float]) -> float:
        """Compute transfer ratio (target_performance / source_performance)."""
        source_reward = source_perf.get('mean_reward', 1e-6)
        target_reward = target_perf.get('mean_reward', 0)
        
        if abs(source_reward) < 1e-6:
            return 1.0 if abs(target_reward) < 1e-6 else 0.0
        
        return target_reward / source_reward
    
    def _compute_generalization_gap(self, source_perf: Dict[str, float], target_perf: Dict[str, float]) -> float:
        """Compute generalization gap (source_performance - target_performance)."""
        source_reward = source_perf.get('mean_reward', 0)
        target_reward = target_perf.get('mean_reward', 0)
        
        return source_reward - target_reward
    
    def _compute_adaptation_improvement(self, original_perf: Dict[str, float], adapted_perf: Dict[str, float]) -> float:
        """Compute improvement from domain adaptation."""
        original_reward = original_perf.get('mean_reward', 0)
        adapted_reward = adapted_perf.get('mean_reward', 0)
        
        return adapted_reward - original_reward
    
    def _analyze_failure_modes(self, performance: Dict[str, float]) -> List[str]:
        """Analyze failure modes from performance metrics."""
        failure_modes = []
        
        if performance.get('mean_reward', 0) < -100:
            failure_modes.append('catastrophic_failure')
        
        if performance.get('violation_rate', 0) > 0.5:
            failure_modes.append('safety_violations')
        
        if performance.get('constraint_satisfaction', 1) < 0.5:
            failure_modes.append('constraint_violations')
        
        if performance.get('success_rate', 1) < 0.3:
            failure_modes.append('task_failure')
        
        if performance.get('human_satisfaction', 0.5) < 0.3:
            failure_modes.append('human_dissatisfaction')
        
        return failure_modes
    
    def _analyze_transfer_results(self) -> Dict[str, Any]:
        """Analyze cross-domain transfer results."""
        analysis = {
            'transfer_summary': self._summarize_transfer_performance(),
            'domain_analysis': self._analyze_domain_relationships(),
            'algorithm_comparison': self._compare_algorithm_transfer(),
            'adaptation_analysis': self._analyze_adaptation_effectiveness(),
            'failure_analysis': self._analyze_failure_patterns()
        }
        
        return analysis
    
    def _summarize_transfer_performance(self) -> Dict[str, Any]:
        """Summarize overall transfer performance."""
        if not self.transfer_results:
            return {}
        
        transfer_ratios = [r.transfer_ratio for r in self.transfer_results if r.transfer_ratio is not None]
        generalization_gaps = [r.generalization_gap for r in self.transfer_results if r.generalization_gap is not None]
        
        return {
            'mean_transfer_ratio': np.mean(transfer_ratios) if transfer_ratios else 0,
            'std_transfer_ratio': np.std(transfer_ratios) if transfer_ratios else 0,
            'mean_generalization_gap': np.mean(generalization_gaps) if generalization_gaps else 0,
            'std_generalization_gap': np.std(generalization_gaps) if generalization_gaps else 0,
            'positive_transfer_rate': np.mean([r > 1.0 for r in transfer_ratios]) if transfer_ratios else 0,
            'negative_transfer_rate': np.mean([r < 0.8 for r in transfer_ratios]) if transfer_ratios else 0
        }
    
    def _analyze_domain_relationships(self) -> Dict[str, Any]:
        """Analyze relationships between domains."""
        relationships = {
            'domain_distances': {},
            'transfer_difficulty': {},
            'best_source_domains': {},
            'hardest_target_domains': {}
        }
        
        # Domain distance analysis
        for result in self.transfer_results:
            if result.domain_distance is not None:
                source_key = f"{result.source_domain.platform.name}_{result.source_domain.human_model.name}"
                target_key = f"{result.target_domain.platform.name}_{result.target_domain.human_model.name}"
                pair_key = f"{source_key}->{target_key}"
                
                relationships['domain_distances'][pair_key] = result.domain_distance
                relationships['transfer_difficulty'][pair_key] = result.generalization_gap
        
        return relationships
    
    def _compare_algorithm_transfer(self) -> Dict[str, Any]:
        """Compare transfer performance across algorithms."""
        algorithm_performance = {}
        
        for result in self.transfer_results:
            if result.algorithm_name not in algorithm_performance:
                algorithm_performance[result.algorithm_name] = {
                    'transfer_ratios': [],
                    'generalization_gaps': [],
                    'success_rates': []
                }
            
            algorithm_performance[result.algorithm_name]['transfer_ratios'].append(result.transfer_ratio)
            algorithm_performance[result.algorithm_name]['generalization_gaps'].append(result.generalization_gap)
            algorithm_performance[result.algorithm_name]['success_rates'].append(
                result.target_performance.get('success_rate', 0)
            )
        
        # Compute summary statistics
        comparison = {}
        for alg_name, metrics in algorithm_performance.items():
            comparison[alg_name] = {
                'mean_transfer_ratio': np.mean(metrics['transfer_ratios']),
                'mean_generalization_gap': np.mean(metrics['generalization_gaps']),
                'mean_success_rate': np.mean(metrics['success_rates']),
                'transfer_consistency': 1.0 / (np.std(metrics['transfer_ratios']) + 1e-6)
            }
        
        return comparison
    
    def _analyze_adaptation_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of domain adaptation."""
        adapted_results = [r for r in self.transfer_results if r.adapted_performance is not None]
        
        if not adapted_results:
            return {'message': 'No adaptation results available'}
        
        improvements = [r.adaptation_improvement for r in adapted_results if r.adaptation_improvement is not None]
        
        return {
            'mean_improvement': np.mean(improvements) if improvements else 0,
            'std_improvement': np.std(improvements) if improvements else 0,
            'positive_adaptation_rate': np.mean([i > 0 for i in improvements]) if improvements else 0,
            'significant_improvement_rate': np.mean([i > 10 for i in improvements]) if improvements else 0
        }
    
    def _analyze_failure_patterns(self) -> Dict[str, Any]:
        """Analyze common failure patterns."""
        all_failure_modes = []
        for result in self.transfer_results:
            all_failure_modes.extend(result.failure_modes)
        
        if not all_failure_modes:
            return {'message': 'No failure modes detected'}
        
        # Count failure mode frequencies
        from collections import Counter
        failure_counts = Counter(all_failure_modes)
        
        return {
            'most_common_failures': failure_counts.most_common(5),
            'total_failures': len(all_failure_modes),
            'unique_failure_types': len(failure_counts)
        }
    
    def _generate_cross_domain_visualizations(self):
        """Generate visualizations for cross-domain evaluation."""
        logger.info("Generating cross-domain visualizations...")
        
        # Transfer matrix heatmap
        self._plot_transfer_matrix()
        
        # Domain distance vs transfer performance
        self._plot_domain_distance_analysis()
        
        # Algorithm comparison across domains
        self._plot_algorithm_transfer_comparison()
        
        # Adaptation effectiveness
        if any(r.adapted_performance for r in self.transfer_results):
            self._plot_adaptation_analysis()
    
    def _plot_transfer_matrix(self):
        """Plot transfer performance matrix."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create transfer matrix
        source_domains = set()
        target_domains = set()
        
        for result in self.transfer_results:
            source_key = f"{result.source_domain.platform.name}_{result.source_domain.human_model.name}"
            target_key = f"{result.target_domain.platform.name}_{result.target_domain.human_model.name}"
            source_domains.add(source_key)
            target_domains.add(target_key)
        
        source_domains = sorted(source_domains)
        target_domains = sorted(target_domains)
        
        # Initialize matrix
        transfer_matrix = np.zeros((len(source_domains), len(target_domains)))
        
        for result in self.transfer_results:
            source_key = f"{result.source_domain.platform.name}_{result.source_domain.human_model.name}"
            target_key = f"{result.target_domain.platform.name}_{result.target_domain.human_model.name}"
            
            if source_key in source_domains and target_key in target_domains:
                i = source_domains.index(source_key)
                j = target_domains.index(target_key)
                transfer_matrix[i, j] = result.transfer_ratio
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(transfer_matrix, 
                   xticklabels=target_domains, 
                   yticklabels=source_domains,
                   annot=True, 
                   fmt='.2f', 
                   cmap='RdYlGn',
                   center=1.0,
                   ax=ax)
        
        ax.set_title('Cross-Domain Transfer Performance Matrix')
        ax.set_xlabel('Target Domains')
        ax.set_ylabel('Source Domains')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = self.results_dir / 'transfer_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Transfer matrix plot saved to {save_path}")
    
    def _plot_domain_distance_analysis(self):
        """Plot domain distance vs transfer performance analysis."""
        import matplotlib.pyplot as plt
        
        distances = []
        transfer_ratios = []
        
        for result in self.transfer_results:
            if result.domain_distance is not None:
                distances.append(result.domain_distance)
                transfer_ratios.append(result.transfer_ratio)
        
        if not distances:
            logger.warning("No domain distance data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(distances, transfer_ratios, alpha=0.6, s=50)
        
        # Add trend line
        if len(distances) > 1:
            z = np.polyfit(distances, transfer_ratios, 1)
            p = np.poly1d(z)
            ax.plot(distances, p(distances), "r--", alpha=0.8)
        
        ax.set_xlabel('Domain Distance')
        ax.set_ylabel('Transfer Ratio')
        ax.set_title('Domain Distance vs Transfer Performance')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, label='No Transfer Effect')
        ax.legend()
        
        plt.tight_layout()
        
        save_path = self.results_dir / 'domain_distance_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Domain distance analysis plot saved to {save_path}")
    
    def _plot_algorithm_transfer_comparison(self):
        """Plot algorithm comparison across domains."""
        import matplotlib.pyplot as plt
        
        # Group results by algorithm
        algorithm_data = {}
        for result in self.transfer_results:
            if result.algorithm_name not in algorithm_data:
                algorithm_data[result.algorithm_name] = []
            algorithm_data[result.algorithm_name].append(result.transfer_ratio)
        
        if not algorithm_data:
            logger.warning("No algorithm data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = list(algorithm_data.keys())
        transfer_data = [algorithm_data[alg] for alg in algorithms]
        
        box_plot = ax.boxplot(transfer_data, labels=algorithms, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Transfer Ratio')
        ax.set_title('Algorithm Transfer Performance Comparison')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Transfer Effect')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = self.results_dir / 'algorithm_transfer_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Algorithm transfer comparison plot saved to {save_path}")
    
    def _plot_adaptation_analysis(self):
        """Plot adaptation effectiveness analysis."""
        import matplotlib.pyplot as plt
        
        adapted_results = [r for r in self.transfer_results if r.adapted_performance is not None]
        
        if not adapted_results:
            logger.warning("No adaptation data available for plotting")
            return
        
        original_performance = [r.target_performance['mean_reward'] for r in adapted_results]
        adapted_performance = [r.adapted_performance['mean_reward'] for r in adapted_results]
        improvements = [r.adaptation_improvement for r in adapted_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Before/after comparison
        ax1.scatter(original_performance, adapted_performance, alpha=0.6)
        
        # Add diagonal line (no improvement)
        min_val = min(min(original_performance), min(adapted_performance))
        max_val = max(max(original_performance), max(adapted_performance))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='No Improvement')
        
        ax1.set_xlabel('Original Performance')
        ax1.set_ylabel('Adapted Performance')
        ax1.set_title('Domain Adaptation: Before vs After')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement distribution
        ax2.hist(improvements, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Improvement')
        ax2.set_xlabel('Performance Improvement')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Adaptation Improvements')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.results_dir / 'adaptation_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Adaptation analysis plot saved to {save_path}")
    
    def _create_cross_domain_report(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive cross-domain evaluation report."""
        report = {
            'executive_summary': self._generate_cross_domain_summary(analysis),
            'methodology': self._describe_cross_domain_methodology(),
            'results': {
                'transfer_performance': analysis.get('transfer_summary', {}),
                'domain_analysis': analysis.get('domain_analysis', {}),
                'algorithm_comparison': analysis.get('algorithm_comparison', {}),
                'adaptation_effectiveness': analysis.get('adaptation_analysis', {}),
                'failure_analysis': analysis.get('failure_analysis', {})
            },
            'recommendations': self._generate_cross_domain_recommendations(analysis),
            'limitations': self._describe_limitations()
        }
        
        # Save report
        report_path = self.results_dir / 'cross_domain_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cross-domain evaluation report saved to {report_path}")
        
        return report
    
    def _generate_cross_domain_summary(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate executive summary for cross-domain evaluation."""
        summary = []
        
        transfer_summary = analysis.get('transfer_summary', {})
        mean_transfer_ratio = transfer_summary.get('mean_transfer_ratio', 0)
        
        if mean_transfer_ratio > 1.1:
            summary.append(f"Positive transfer observed: average {mean_transfer_ratio:.2f}x performance retention")
        elif mean_transfer_ratio > 0.8:
            summary.append(f"Moderate transfer: average {mean_transfer_ratio:.2f}x performance retention")
        else:
            summary.append(f"Negative transfer detected: average {mean_transfer_ratio:.2f}x performance retention")
        
        # Algorithm comparison
        algorithm_comparison = analysis.get('algorithm_comparison', {})
        if algorithm_comparison:
            best_algorithm = max(algorithm_comparison.keys(), 
                               key=lambda k: algorithm_comparison[k].get('mean_transfer_ratio', 0))
            summary.append(f"Best transferring algorithm: {best_algorithm}")
        
        # Adaptation effectiveness
        adaptation_analysis = analysis.get('adaptation_analysis', {})
        if 'mean_improvement' in adaptation_analysis:
            improvement = adaptation_analysis['mean_improvement']
            if improvement > 5:
                summary.append(f"Domain adaptation effective: +{improvement:.1f} average improvement")
            else:
                summary.append("Domain adaptation showed limited effectiveness")
        
        return summary
    
    def _describe_cross_domain_methodology(self) -> Dict[str, Any]:
        """Describe cross-domain evaluation methodology."""
        return {
            'approach': 'Systematic cross-domain transfer evaluation',
            'source_domains': len(self.config.source_platforms) * len(self.config.source_human_models) * len(self.config.source_safety_scenarios),
            'target_domains': len(self.config.target_platforms) * len(self.config.target_human_models) * len(self.config.target_safety_scenarios),
            'training_timesteps': self.config.training_timesteps,
            'evaluation_episodes': self.config.num_evaluation_episodes,
            'adaptation_enabled': self.config.enable_fine_tuning,
            'domain_distance_metric': 'Weighted Euclidean distance over domain characteristics'
        }
    
    def _generate_cross_domain_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for cross-domain deployment."""
        recommendations = []
        
        # Transfer performance recommendations
        transfer_summary = analysis.get('transfer_summary', {})
        if transfer_summary.get('negative_transfer_rate', 0) > 0.3:
            recommendations.append("High negative transfer rate detected - consider domain-specific training")
        
        # Algorithm recommendations
        algorithm_comparison = analysis.get('algorithm_comparison', {})
        if algorithm_comparison:
            most_consistent = max(algorithm_comparison.keys(), 
                                key=lambda k: algorithm_comparison[k].get('transfer_consistency', 0))
            recommendations.append(f"For consistent cross-domain performance, use {most_consistent}")
        
        # Adaptation recommendations
        adaptation_analysis = analysis.get('adaptation_analysis', {})
        if adaptation_analysis.get('positive_adaptation_rate', 0) > 0.7:
            recommendations.append("Domain adaptation is effective - recommend fine-tuning for new domains")
        
        return recommendations
    
    def _describe_limitations(self) -> List[str]:
        """Describe limitations of cross-domain evaluation."""
        return [
            "Simulated environments may not capture all real-world domain variations",
            "Limited number of target domains evaluated",
            "Human behavior models are simplified approximations",
            "Domain distance metric is heuristic-based",
            "Transfer evaluation limited to immediate performance measures"
        ]
    
    def save_results(self, filename: str = "cross_domain_results.pkl"):
        """Save all cross-domain evaluation results."""
        results_data = {
            'config': self.config,
            'transfer_results': self.transfer_results,
            'domain_characteristics': self.domain_characteristics
        }
        
        save_path = self.results_dir / filename
        with open(save_path, 'wb') as f:
            pickle.dump(results_data, f)
        
        logger.info(f"Cross-domain results saved to {save_path}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load previously saved cross-domain results."""
        load_path = Path(filename)
        with open(load_path, 'rb') as f:
            results_data = pickle.load(f)
        
        self.config = results_data['config']
        self.transfer_results = results_data['transfer_results']
        self.domain_characteristics = results_data['domain_characteristics']
        
        logger.info(f"Cross-domain results loaded from {load_path}")
        return results_data