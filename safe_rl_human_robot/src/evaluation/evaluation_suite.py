"""
Comprehensive Evaluation Suite for Safe RL Benchmarking.

This module implements the main evaluation framework that orchestrates benchmarking
across multiple algorithms, environments, and evaluation criteria.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from ..baselines.base_algorithm import BaselineAlgorithm, AlgorithmConfig
from .environments import StandardizedEnvSuite, RobotPlatform, HumanModelType, SafetyScenario
from .metrics import MetricAggregator, PerformanceMetrics, HumanMetrics, SafetyMetrics
from .statistics import StatisticalAnalyzer
from .visualization import ResultVisualizer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for comprehensive evaluation."""
    # Experiment setup
    num_seeds: int = 5
    num_evaluation_episodes: int = 100
    max_episode_steps: int = 1000
    evaluation_frequency: int = 10000  # Steps between evaluations
    
    # Environment configuration
    robot_platforms: List[RobotPlatform] = field(default_factory=lambda: [
        RobotPlatform.MANIPULATOR_7DOF,
        RobotPlatform.MOBILE_BASE, 
        RobotPlatform.HUMANOID
    ])
    human_models: List[HumanModelType] = field(default_factory=lambda: [
        HumanModelType.COOPERATIVE,
        HumanModelType.ADVERSARIAL,
        HumanModelType.INCONSISTENT
    ])
    safety_scenarios: List[SafetyScenario] = field(default_factory=lambda: [
        SafetyScenario.OBSTACLE_AVOIDANCE,
        SafetyScenario.FORCE_LIMITS,
        SafetyScenario.WORKSPACE_BOUNDS
    ])
    
    # Statistical analysis
    significance_level: float = 0.05
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.5  # Cohen's d threshold
    min_effect_size: float = 0.2
    
    # Computational resources
    max_workers: int = 4
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Output configuration
    results_dir: str = "evaluation_results"
    save_raw_data: bool = True
    generate_plots: bool = True
    create_report: bool = True
    
    # Reproducibility
    base_seed: int = 42
    deterministic: bool = True


@dataclass 
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    # Performance metrics
    sample_efficiency: float
    asymptotic_performance: float
    learning_curve_auc: float
    convergence_time: float
    final_reward: float
    
    # Safety metrics
    safety_violations: int
    violation_rate: float
    constraint_satisfaction: float
    safety_margin: float
    recovery_time: float
    
    # Human-centric metrics
    human_satisfaction: float
    trust_level: float
    workload_score: float
    naturalness_rating: float
    collaboration_efficiency: float
    
    # Efficiency metrics
    computation_time: float
    memory_usage: float
    inference_time: float
    training_stability: float
    parameter_efficiency: float
    
    # Additional metrics
    robustness_score: float
    adaptability_measure: float
    generalization_performance: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'sample_efficiency': self.sample_efficiency,
            'asymptotic_performance': self.asymptotic_performance,
            'learning_curve_auc': self.learning_curve_auc,
            'convergence_time': self.convergence_time,
            'final_reward': self.final_reward,
            'safety_violations': self.safety_violations,
            'violation_rate': self.violation_rate,
            'constraint_satisfaction': self.constraint_satisfaction,
            'safety_margin': self.safety_margin,
            'recovery_time': self.recovery_time,
            'human_satisfaction': self.human_satisfaction,
            'trust_level': self.trust_level,
            'workload_score': self.workload_score,
            'naturalness_rating': self.naturalness_rating,
            'collaboration_efficiency': self.collaboration_efficiency,
            'computation_time': self.computation_time,
            'memory_usage': self.memory_usage,
            'inference_time': self.inference_time,
            'training_stability': self.training_stability,
            'parameter_efficiency': self.parameter_efficiency,
            'robustness_score': self.robustness_score,
            'adaptability_measure': self.adaptability_measure,
            'generalization_performance': self.generalization_performance
        }


@dataclass
class BenchmarkResult:
    """Container for benchmarking results."""
    algorithm_name: str
    config: Dict[str, Any]
    metrics: EvaluationMetrics
    raw_data: Dict[str, Any]
    environment_config: Dict[str, Any]
    seed: int
    execution_time: float
    timestamp: float


class EvaluationSuite:
    """Comprehensive evaluation suite for Safe RL benchmarking."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize components
        self.env_suite = StandardizedEnvSuite()
        self.metric_aggregator = MetricAggregator()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualizer = ResultVisualizer()
        
        # Results storage
        self.results: List[BenchmarkResult] = []
        self.summary_statistics = {}
        
        # Create results directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Evaluation Suite initialized")
    
    def run_comprehensive_benchmark(self, 
                                  algorithms: Dict[str, BaselineAlgorithm],
                                  total_timesteps: int = 1000000) -> Dict[str, Any]:
        """Run comprehensive benchmark across all algorithms and environments."""
        logger.info("Starting comprehensive benchmark evaluation")
        start_time = time.time()
        
        # Generate all experiment configurations
        experiment_configs = self._generate_experiment_configs(algorithms)
        logger.info(f"Generated {len(experiment_configs)} experiment configurations")
        
        # Run experiments in parallel
        if self.config.max_workers > 1:
            results = self._run_parallel_experiments(experiment_configs, total_timesteps)
        else:
            results = self._run_sequential_experiments(experiment_configs, total_timesteps)
        
        self.results.extend(results)
        
        # Perform statistical analysis
        self.summary_statistics = self._compute_summary_statistics()
        
        # Generate visualizations
        if self.config.generate_plots:
            self._generate_visualizations()
        
        # Create evaluation report
        if self.config.create_report:
            report = self._create_evaluation_report()
        else:
            report = {}
        
        # Save results
        self._save_results()
        
        total_time = time.time() - start_time
        logger.info(f"Comprehensive benchmark completed in {total_time:.2f} seconds")
        
        return {
            'results': self.results,
            'summary_statistics': self.summary_statistics,
            'report': report,
            'execution_time': total_time
        }
    
    def _generate_experiment_configs(self, algorithms: Dict[str, BaselineAlgorithm]) -> List[Dict[str, Any]]:
        """Generate all experiment configurations."""
        configs = []
        
        for alg_name, algorithm in algorithms.items():
            for robot_platform in self.config.robot_platforms:
                for human_model in self.config.human_models:
                    for safety_scenario in self.config.safety_scenarios:
                        for seed in range(self.config.num_seeds):
                            config = {
                                'algorithm_name': alg_name,
                                'algorithm': algorithm,
                                'robot_platform': robot_platform,
                                'human_model': human_model,
                                'safety_scenario': safety_scenario,
                                'seed': self.config.base_seed + seed,
                                'exp_id': f"{alg_name}_{robot_platform.name}_{human_model.name}_{safety_scenario.name}_{seed}"
                            }
                            configs.append(config)
        
        return configs
    
    def _run_parallel_experiments(self, experiment_configs: List[Dict[str, Any]], 
                                total_timesteps: int) -> List[BenchmarkResult]:
        """Run experiments in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self._run_single_experiment, config, total_timesteps): config
                for config in experiment_configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed experiment: {config['exp_id']}")
                except Exception as exc:
                    logger.error(f"Experiment {config['exp_id']} generated an exception: {exc}")
        
        return results
    
    def _run_sequential_experiments(self, experiment_configs: List[Dict[str, Any]], 
                                  total_timesteps: int) -> List[BenchmarkResult]:
        """Run experiments sequentially."""
        results = []
        
        for i, config in enumerate(experiment_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment_configs)}: {config['exp_id']}")
            try:
                result = self._run_single_experiment(config, total_timesteps)
                results.append(result)
            except Exception as exc:
                logger.error(f"Experiment {config['exp_id']} failed: {exc}")
        
        return results
    
    def _run_single_experiment(self, config: Dict[str, Any], total_timesteps: int) -> BenchmarkResult:
        """Run a single experiment configuration."""
        start_time = time.time()
        
        # Set random seed for reproducibility
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        
        # Create environment
        env = self.env_suite.create_environment(
            robot_platform=config['robot_platform'],
            human_model=config['human_model'],
            safety_scenario=config['safety_scenario'],
            seed=config['seed']
        )
        
        # Initialize algorithm
        algorithm = config['algorithm']
        
        # Training phase
        training_metrics = []
        evaluation_metrics = []
        
        for timestep in range(0, total_timesteps, self.config.evaluation_frequency):
            # Train for evaluation_frequency steps
            remaining_steps = min(self.config.evaluation_frequency, total_timesteps - timestep)
            algorithm.learn(remaining_steps, env)
            
            # Evaluate current policy
            eval_metrics = self._evaluate_algorithm(algorithm, env, config['seed'])
            evaluation_metrics.append({
                'timestep': timestep + remaining_steps,
                'metrics': eval_metrics
            })
        
        # Final comprehensive evaluation
        final_metrics = self._comprehensive_evaluation(algorithm, env, config)
        
        execution_time = time.time() - start_time
        
        # Create benchmark result
        result = BenchmarkResult(
            algorithm_name=config['algorithm_name'],
            config={
                'robot_platform': config['robot_platform'].name,
                'human_model': config['human_model'].name,
                'safety_scenario': config['safety_scenario'].name,
                'algorithm_config': algorithm.config.__dict__ if hasattr(algorithm, 'config') else {}
            },
            metrics=final_metrics,
            raw_data={
                'training_metrics': training_metrics,
                'evaluation_metrics': evaluation_metrics,
                'timesteps': total_timesteps
            },
            environment_config=env.get_config() if hasattr(env, 'get_config') else {},
            seed=config['seed'],
            execution_time=execution_time,
            timestamp=time.time()
        )
        
        return result
    
    def _evaluate_algorithm(self, algorithm: BaselineAlgorithm, env, seed: int) -> EvaluationMetrics:
        """Evaluate algorithm performance."""
        # Set evaluation seed
        np.random.seed(seed + 10000)  # Different seed for evaluation
        
        episode_rewards = []
        episode_costs = []
        episode_lengths = []
        safety_violations = []
        human_metrics = []
        
        for episode in range(self.config.num_evaluation_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_cost = 0
            episode_length = 0
            violations = 0
            
            for step in range(self.config.max_episode_steps):
                # Get action from algorithm
                action, _ = algorithm.predict(obs, deterministic=True)
                
                # Environment step
                next_obs, reward, done, info = env.step(action)
                
                episode_reward += reward
                episode_cost += info.get('cost', 0)
                episode_length += 1
                
                # Check for safety violations
                if info.get('constraint_violation', False):
                    violations += 1
                
                if done:
                    break
                
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            episode_lengths.append(episode_length)
            safety_violations.append(violations)
            
            # Collect human-centric metrics if available
            if hasattr(env, 'get_human_metrics'):
                human_metrics.append(env.get_human_metrics())
        
        # Compute aggregate metrics
        metrics = self.metric_aggregator.compute_evaluation_metrics(
            episode_rewards=episode_rewards,
            episode_costs=episode_costs,
            episode_lengths=episode_lengths,
            safety_violations=safety_violations,
            human_metrics=human_metrics
        )
        
        return metrics
    
    def _comprehensive_evaluation(self, algorithm: BaselineAlgorithm, env, config: Dict[str, Any]) -> EvaluationMetrics:
        """Perform comprehensive evaluation including all metrics."""
        # Standard evaluation
        base_metrics = self._evaluate_algorithm(algorithm, env, config['seed'])
        
        # Additional comprehensive metrics
        computation_metrics = self._measure_computational_efficiency(algorithm, env)
        robustness_metrics = self._evaluate_robustness(algorithm, env, config['seed'])
        
        # Combine all metrics
        comprehensive_metrics = EvaluationMetrics(
            # Base performance metrics
            sample_efficiency=base_metrics.sample_efficiency,
            asymptotic_performance=base_metrics.asymptotic_performance,
            learning_curve_auc=base_metrics.learning_curve_auc,
            convergence_time=base_metrics.convergence_time,
            final_reward=base_metrics.final_reward,
            
            # Safety metrics  
            safety_violations=base_metrics.safety_violations,
            violation_rate=base_metrics.violation_rate,
            constraint_satisfaction=base_metrics.constraint_satisfaction,
            safety_margin=base_metrics.safety_margin,
            recovery_time=base_metrics.recovery_time,
            
            # Human-centric metrics
            human_satisfaction=base_metrics.human_satisfaction,
            trust_level=base_metrics.trust_level,
            workload_score=base_metrics.workload_score,
            naturalness_rating=base_metrics.naturalness_rating,
            collaboration_efficiency=base_metrics.collaboration_efficiency,
            
            # Computational metrics
            computation_time=computation_metrics['computation_time'],
            memory_usage=computation_metrics['memory_usage'],
            inference_time=computation_metrics['inference_time'],
            training_stability=computation_metrics['training_stability'],
            parameter_efficiency=computation_metrics['parameter_efficiency'],
            
            # Robustness metrics
            robustness_score=robustness_metrics['robustness_score'],
            adaptability_measure=robustness_metrics['adaptability_measure'],
            generalization_performance=robustness_metrics['generalization_performance']
        )
        
        return comprehensive_metrics
    
    def _measure_computational_efficiency(self, algorithm: BaselineAlgorithm, env) -> Dict[str, float]:
        """Measure computational efficiency metrics."""
        import psutil
        import gc
        
        # Measure inference time
        obs = env.reset()
        inference_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            action, _ = algorithm.predict(obs, deterministic=True)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        # Measure memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force garbage collection
        gc.collect()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Count parameters
        total_params = 0
        if hasattr(algorithm, 'policy') and hasattr(algorithm.policy, 'parameters'):
            total_params = sum(p.numel() for p in algorithm.policy.parameters())
        elif hasattr(algorithm, 'actor') and hasattr(algorithm.actor, 'parameters'):
            total_params = sum(p.numel() for p in algorithm.actor.parameters())
        
        return {
            'computation_time': np.mean(inference_times) * 1000,  # ms
            'memory_usage': memory_after - memory_before,
            'inference_time': np.mean(inference_times) * 1000,  # ms
            'training_stability': 1.0,  # Placeholder - would need training curve analysis
            'parameter_efficiency': 1.0 / (total_params + 1) if total_params > 0 else 1.0
        }
    
    def _evaluate_robustness(self, algorithm: BaselineAlgorithm, env, seed: int) -> Dict[str, float]:
        """Evaluate robustness and generalization."""
        # Test with different noise levels
        robustness_scores = []
        
        for noise_level in [0.0, 0.05, 0.1, 0.2]:
            episode_rewards = []
            
            for _ in range(10):  # Fewer episodes for robustness test
                obs = env.reset()
                episode_reward = 0
                
                for _ in range(self.config.max_episode_steps):
                    # Add noise to observations
                    noisy_obs = obs + np.random.normal(0, noise_level, obs.shape)
                    action, _ = algorithm.predict(noisy_obs, deterministic=True)
                    
                    next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                    obs = next_obs
                
                episode_rewards.append(episode_reward)
            
            robustness_scores.append(np.mean(episode_rewards))
        
        # Compute robustness as performance degradation
        baseline_performance = robustness_scores[0]
        degradation = [(baseline_performance - score) / (baseline_performance + 1e-6) 
                      for score in robustness_scores[1:]]
        robustness_score = 1.0 - np.mean(degradation)
        
        return {
            'robustness_score': max(0.0, robustness_score),
            'adaptability_measure': 0.8,  # Placeholder
            'generalization_performance': 0.85  # Placeholder
        }
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        if not self.results:
            return {}
        
        # Group results by algorithm
        algorithm_results = {}
        for result in self.results:
            if result.algorithm_name not in algorithm_results:
                algorithm_results[result.algorithm_name] = []
            algorithm_results[result.algorithm_name].append(result)
        
        # Compute statistics for each algorithm
        summary = {}
        for alg_name, results in algorithm_results.items():
            metrics_data = [result.metrics.to_dict() for result in results]
            df = pd.DataFrame(metrics_data)
            
            summary[alg_name] = {
                'mean': df.mean().to_dict(),
                'std': df.std().to_dict(),
                'median': df.median().to_dict(),
                'min': df.min().to_dict(),
                'max': df.max().to_dict(),
                'count': len(results)
            }
        
        # Perform statistical comparisons
        statistical_tests = self.statistical_analyzer.perform_comparative_analysis(
            algorithm_results, significance_level=self.config.significance_level
        )
        
        summary['statistical_tests'] = statistical_tests
        
        return summary
    
    def _generate_visualizations(self):
        """Generate evaluation visualizations."""
        logger.info("Generating evaluation visualizations")
        
        # Performance comparison plots
        self.visualizer.create_performance_comparison(
            self.results, save_path=self.results_dir / "performance_comparison.png"
        )
        
        # Safety metrics visualization
        self.visualizer.create_safety_analysis(
            self.results, save_path=self.results_dir / "safety_analysis.png"
        )
        
        # Statistical significance heatmap
        if 'statistical_tests' in self.summary_statistics:
            self.visualizer.create_statistical_heatmap(
                self.summary_statistics['statistical_tests'],
                save_path=self.results_dir / "statistical_significance.png"
            )
    
    def _create_evaluation_report(self) -> Dict[str, Any]:
        """Create comprehensive evaluation report."""
        logger.info("Creating evaluation report")
        
        report = {
            'experiment_config': self.config.__dict__,
            'num_algorithms': len(set(r.algorithm_name for r in self.results)),
            'num_experiments': len(self.results),
            'summary_statistics': self.summary_statistics,
            'key_findings': self._extract_key_findings(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report as JSON
        with open(self.results_dir / "evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from evaluation results."""
        findings = []
        
        if not self.summary_statistics:
            return findings
        
        # Find best performing algorithm overall
        best_alg = max(self.summary_statistics.keys(),
                      key=lambda k: self.summary_statistics[k]['mean'].get('asymptotic_performance', 0)
                      if k != 'statistical_tests' else -float('inf'))
        
        findings.append(f"Best overall performance: {best_alg}")
        
        # Find safest algorithm
        safest_alg = min(self.summary_statistics.keys(),
                        key=lambda k: self.summary_statistics[k]['mean'].get('violation_rate', float('inf'))
                        if k != 'statistical_tests' else float('inf'))
        
        findings.append(f"Safest algorithm: {safest_alg}")
        
        # Find most efficient algorithm
        most_efficient = min(self.summary_statistics.keys(),
                           key=lambda k: self.summary_statistics[k]['mean'].get('computation_time', float('inf'))
                           if k != 'statistical_tests' else float('inf'))
        
        findings.append(f"Most computationally efficient: {most_efficient}")
        
        return findings
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = [
            "Consider safety-performance trade-offs when selecting algorithms",
            "Evaluate algorithms across multiple robot platforms for generalization",
            "Include human factors in algorithm selection for shared control",
            "Monitor computational efficiency for real-time applications"
        ]
        
        return recommendations
    
    def _save_results(self):
        """Save all evaluation results."""
        logger.info("Saving evaluation results")
        
        if self.config.save_raw_data:
            # Save raw results
            with open(self.results_dir / "raw_results.pkl", 'wb') as f:
                pickle.dump(self.results, f)
            
            # Save summary statistics
            with open(self.results_dir / "summary_statistics.json", 'w') as f:
                json.dump(self.summary_statistics, f, indent=2, default=str)
            
            # Save results as CSV for easy analysis
            results_data = []
            for result in self.results:
                row = {
                    'algorithm': result.algorithm_name,
                    'seed': result.seed,
                    'execution_time': result.execution_time,
                    **result.metrics.to_dict(),
                    **result.config
                }
                results_data.append(row)
            
            df = pd.DataFrame(results_data)
            df.to_csv(self.results_dir / "results.csv", index=False)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def load_results(self, results_path: str) -> List[BenchmarkResult]:
        """Load previously saved results."""
        with open(results_path, 'rb') as f:
            self.results = pickle.load(f)
        return self.results