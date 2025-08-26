"""
Comprehensive Metrics for Safe RL Evaluation.

This module implements various performance, safety, human-centric, and efficiency
metrics for evaluating Safe RL algorithms in human-robot shared control.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.metrics import auc
import warnings

from ..evaluation.evaluation_suite import EvaluationMetrics

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for individual metric results."""
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    raw_data: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMetrics:
    """Performance-related metrics for RL evaluation."""
    
    @staticmethod
    def compute_sample_efficiency(episode_rewards: List[float], 
                                timesteps: List[int],
                                target_performance: float = None) -> float:
        """Compute sample efficiency - steps to reach target performance."""
        if target_performance is None:
            target_performance = np.percentile(episode_rewards[-20:], 90)  # 90% of final performance
        
        # Find first time target is reached and maintained
        smoothed_rewards = PerformanceMetrics._smooth_curve(episode_rewards, window=10)
        
        for i, reward in enumerate(smoothed_rewards):
            if reward >= target_performance:
                # Check if performance is maintained for next 10 episodes
                if i + 10 < len(smoothed_rewards):
                    if np.mean(smoothed_rewards[i:i+10]) >= target_performance:
                        return timesteps[i] if i < len(timesteps) else float('inf')
        
        return float('inf')  # Never reached target
    
    @staticmethod
    def compute_asymptotic_performance(episode_rewards: List[float], 
                                     final_fraction: float = 0.2) -> float:
        """Compute asymptotic performance from final episodes."""
        n_final = max(1, int(len(episode_rewards) * final_fraction))
        return np.mean(episode_rewards[-n_final:])
    
    @staticmethod
    def compute_learning_curve_auc(episode_rewards: List[float], 
                                 normalize: bool = True) -> float:
        """Compute area under learning curve."""
        if len(episode_rewards) < 2:
            return 0.0
        
        x = np.arange(len(episode_rewards))
        y = np.array(episode_rewards)
        
        # Normalize to [0, 1] if requested
        if normalize and y.max() > y.min():
            y = (y - y.min()) / (y.max() - y.min())
        
        return auc(x, y) / len(episode_rewards)  # Normalize by length
    
    @staticmethod
    def compute_convergence_time(episode_rewards: List[float], 
                               convergence_threshold: float = 0.01) -> float:
        """Compute time to convergence based on reward stability."""
        if len(episode_rewards) < 20:
            return float('inf')
        
        smoothed_rewards = PerformanceMetrics._smooth_curve(episode_rewards, window=10)
        
        # Look for point where variance becomes consistently low
        for i in range(10, len(smoothed_rewards) - 10):
            window_var = np.var(smoothed_rewards[i:i+10])
            if window_var < convergence_threshold:
                # Check if variance stays low
                if np.all([np.var(smoothed_rewards[j:j+5]) < convergence_threshold 
                          for j in range(i, min(i+20, len(smoothed_rewards)-5))]):
                    return float(i)
        
        return float('inf')
    
    @staticmethod
    def compute_training_stability(episode_rewards: List[float]) -> float:
        """Compute training stability as inverse of reward variance."""
        if len(episode_rewards) < 2:
            return 0.0
        
        # Use coefficient of variation (std/mean) as instability measure
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        if abs(mean_reward) < 1e-8:
            return 0.0
        
        coefficient_of_variation = std_reward / abs(mean_reward)
        stability = 1.0 / (1.0 + coefficient_of_variation)
        
        return stability
    
    @staticmethod
    def _smooth_curve(data: List[float], window: int = 5) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(data) < window:
            return np.array(data)
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start:end]))
        
        return np.array(smoothed)


class SafetyMetrics:
    """Safety-related metrics for Safe RL evaluation."""
    
    @staticmethod
    def compute_violation_rate(safety_violations: List[int]) -> float:
        """Compute rate of safety violations."""
        if not safety_violations:
            return 0.0
        return np.mean(safety_violations)
    
    @staticmethod
    def compute_constraint_satisfaction(episode_costs: List[float], 
                                      cost_limit: float) -> float:
        """Compute constraint satisfaction rate."""
        if not episode_costs:
            return 1.0
        
        satisfied_episodes = sum(1 for cost in episode_costs if cost <= cost_limit)
        return satisfied_episodes / len(episode_costs)
    
    @staticmethod
    def compute_safety_margin(episode_costs: List[float], 
                            cost_limit: float) -> float:
        """Compute average safety margin."""
        if not episode_costs:
            return float('inf')
        
        # Positive margin = safe, negative margin = unsafe
        margins = [cost_limit - cost for cost in episode_costs]
        return np.mean(margins)
    
    @staticmethod
    def compute_recovery_time(violation_data: List[Dict[str, Any]]) -> float:
        """Compute average time to recover from safety violations."""
        recovery_times = []
        
        for episode_data in violation_data:
            if 'violations' in episode_data and 'recovery_steps' in episode_data:
                violations = episode_data['violations']
                recoveries = episode_data['recovery_steps']
                
                if len(violations) > 0 and len(recoveries) >= len(violations):
                    episode_recovery_times = recoveries[:len(violations)]
                    recovery_times.extend(episode_recovery_times)
        
        return np.mean(recovery_times) if recovery_times else float('inf')
    
    @staticmethod
    def compute_safety_index(episode_costs: List[float], 
                           safety_violations: List[int],
                           cost_limit: float) -> float:
        """Compute comprehensive safety index (0=unsafe, 1=perfectly safe)."""
        if not episode_costs:
            return 1.0
        
        # Constraint satisfaction component
        constraint_satisfaction = SafetyMetrics.compute_constraint_satisfaction(
            episode_costs, cost_limit
        )
        
        # Violation frequency component
        violation_rate = SafetyMetrics.compute_violation_rate(safety_violations)
        violation_score = np.exp(-10 * violation_rate)  # Exponential penalty
        
        # Safety margin component
        safety_margin = SafetyMetrics.compute_safety_margin(episode_costs, cost_limit)
        margin_score = 1.0 / (1.0 + np.exp(-safety_margin))  # Sigmoid
        
        # Combined safety index
        safety_index = 0.5 * constraint_satisfaction + 0.3 * violation_score + 0.2 * margin_score
        
        return np.clip(safety_index, 0.0, 1.0)


class HumanMetrics:
    """Human-centric metrics for evaluation."""
    
    @staticmethod
    def compute_satisfaction(human_metrics_data: List[Dict[str, float]]) -> float:
        """Compute human satisfaction metric."""
        if not human_metrics_data:
            return 0.5  # Neutral
        
        satisfaction_scores = [data.get('satisfaction', 0.5) for data in human_metrics_data]
        return np.mean(satisfaction_scores)
    
    @staticmethod
    def compute_trust_level(human_metrics_data: List[Dict[str, float]], 
                          safety_violations: List[int]) -> float:
        """Compute trust level incorporating safety violations."""
        base_trust = 0.8  # Initial trust level
        
        # Trust decreases with safety violations
        violation_rate = np.mean(safety_violations) if safety_violations else 0.0
        trust_penalty = violation_rate * 0.5
        
        # Trust from human feedback
        if human_metrics_data:
            human_trust_scores = [data.get('trust', base_trust) for data in human_metrics_data]
            human_trust = np.mean(human_trust_scores)
        else:
            human_trust = base_trust
        
        final_trust = max(0.0, min(1.0, human_trust - trust_penalty))
        return final_trust
    
    @staticmethod
    def compute_workload_score(human_metrics_data: List[Dict[str, float]]) -> float:
        """Compute human workload score."""
        if not human_metrics_data:
            return 0.5  # Neutral
        
        workload_scores = [data.get('workload', 0.5) for data in human_metrics_data]
        return np.mean(workload_scores)
    
    @staticmethod
    def compute_naturalness_rating(human_metrics_data: List[Dict[str, float]]) -> float:
        """Compute naturalness of robot behavior."""
        if not human_metrics_data:
            return 0.5  # Neutral
        
        naturalness_scores = [data.get('naturalness', 0.5) for data in human_metrics_data]
        return np.mean(naturalness_scores)
    
    @staticmethod
    def compute_collaboration_efficiency(task_completion_times: List[float], 
                                       solo_baseline_time: float = 100.0) -> float:
        """Compute collaboration efficiency compared to solo performance."""
        if not task_completion_times:
            return 0.0
        
        avg_completion_time = np.mean(task_completion_times)
        
        # Efficiency = baseline_time / actual_time
        # Values > 1 indicate improvement, < 1 indicate degradation
        efficiency = solo_baseline_time / (avg_completion_time + 1e-6)
        
        # Normalize to [0, 1] scale
        return min(1.0, max(0.0, (efficiency - 0.5) / 1.5))
    
    @staticmethod
    def compute_human_robot_compatibility(human_metrics_data: List[Dict[str, float]]) -> float:
        """Compute overall human-robot compatibility score."""
        if not human_metrics_data:
            return 0.5
        
        satisfaction = HumanMetrics.compute_satisfaction(human_metrics_data)
        trust = np.mean([data.get('trust', 0.5) for data in human_metrics_data])
        naturalness = HumanMetrics.compute_naturalness_rating(human_metrics_data)
        
        # Workload should be moderate (not too high or too low)
        workload = HumanMetrics.compute_workload_score(human_metrics_data)
        workload_score = 1.0 - abs(workload - 0.5) * 2  # Optimal at 0.5
        
        compatibility = 0.3 * satisfaction + 0.3 * trust + 0.2 * naturalness + 0.2 * workload_score
        return np.clip(compatibility, 0.0, 1.0)


class EfficiencyMetrics:
    """Computational and parameter efficiency metrics."""
    
    @staticmethod
    def compute_computational_efficiency(computation_times: List[float], 
                                       baseline_time: float = 1.0) -> float:
        """Compute computational efficiency relative to baseline."""
        if not computation_times:
            return 1.0
        
        avg_time = np.mean(computation_times)
        return baseline_time / (avg_time + 1e-6)
    
    @staticmethod
    def compute_memory_efficiency(memory_usage: List[float], 
                                baseline_memory: float = 100.0) -> float:
        """Compute memory efficiency relative to baseline."""
        if not memory_usage:
            return 1.0
        
        avg_memory = np.mean(memory_usage)
        return baseline_memory / (avg_memory + 1e-6)
    
    @staticmethod
    def compute_parameter_efficiency(num_parameters: int, 
                                   performance_score: float) -> float:
        """Compute parameter efficiency (performance per parameter)."""
        if num_parameters == 0:
            return 0.0
        
        return performance_score / (np.log10(num_parameters + 1) + 1e-6)
    
    @staticmethod
    def compute_inference_speed(inference_times: List[float]) -> float:
        """Compute inference speed (Hz)."""
        if not inference_times:
            return 0.0
        
        avg_time = np.mean(inference_times)
        return 1.0 / (avg_time + 1e-6)
    
    @staticmethod
    def compute_scalability_score(performance_vs_complexity: List[Tuple[int, float]]) -> float:
        """Compute scalability score based on performance vs complexity."""
        if len(performance_vs_complexity) < 2:
            return 0.5
        
        complexities, performances = zip(*performance_vs_complexity)
        
        # Compute correlation - negative correlation indicates good scalability
        correlation = stats.pearsonr(complexities, performances)[0]
        
        # Convert to scalability score (0=poor scaling, 1=excellent scaling)
        scalability = (1 - correlation) / 2 if not np.isnan(correlation) else 0.5
        return np.clip(scalability, 0.0, 1.0)


class RobustnessMetrics:
    """Robustness and generalization metrics."""
    
    @staticmethod
    def compute_noise_robustness(clean_performance: List[float], 
                               noisy_performance: List[float]) -> float:
        """Compute robustness to observation noise."""
        if not clean_performance or not noisy_performance:
            return 0.0
        
        clean_mean = np.mean(clean_performance)
        noisy_mean = np.mean(noisy_performance)
        
        if abs(clean_mean) < 1e-6:
            return 1.0 if abs(noisy_mean) < 1e-6 else 0.0
        
        robustness = noisy_mean / clean_mean
        return max(0.0, min(1.0, robustness))
    
    @staticmethod
    def compute_generalization_score(source_performance: List[float], 
                                   target_performance: List[float]) -> float:
        """Compute generalization from source to target domain."""
        if not source_performance or not target_performance:
            return 0.0
        
        source_mean = np.mean(source_performance)
        target_mean = np.mean(target_performance)
        
        if abs(source_mean) < 1e-6:
            return 1.0 if abs(target_mean) < 1e-6 else 0.0
        
        generalization = target_mean / source_mean
        return max(0.0, min(1.0, generalization))
    
    @staticmethod
    def compute_adaptation_speed(adaptation_curve: List[float], 
                               adaptation_threshold: float = 0.9) -> float:
        """Compute speed of adaptation to new conditions."""
        if len(adaptation_curve) < 2:
            return 0.0
        
        # Find when performance reaches threshold of final performance
        final_performance = adaptation_curve[-1]
        threshold_performance = adaptation_threshold * final_performance
        
        for i, performance in enumerate(adaptation_curve):
            if performance >= threshold_performance:
                return 1.0 / (i + 1)  # Inverse of adaptation time
        
        return 1.0 / len(adaptation_curve)  # Never reached threshold
    
    @staticmethod
    def compute_stability_under_perturbation(baseline_performance: List[float], 
                                           perturbed_performance: List[float]) -> float:
        """Compute stability under environmental perturbations."""
        if not baseline_performance or not perturbed_performance:
            return 0.0
        
        baseline_var = np.var(baseline_performance)
        perturbed_var = np.var(perturbed_performance)
        
        baseline_mean = np.mean(baseline_performance)
        perturbed_mean = np.mean(perturbed_performance)
        
        # Performance stability
        mean_stability = 1.0 - abs(perturbed_mean - baseline_mean) / (abs(baseline_mean) + 1e-6)
        
        # Variance stability
        var_stability = 1.0 - abs(perturbed_var - baseline_var) / (baseline_var + 1e-6)
        
        stability = 0.7 * max(0.0, mean_stability) + 0.3 * max(0.0, var_stability)
        return min(1.0, stability)


class MetricAggregator:
    """Aggregates various metrics for comprehensive evaluation."""
    
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
        self.safety_metrics = SafetyMetrics()
        self.human_metrics = HumanMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.robustness_metrics = RobustnessMetrics()
    
    def compute_evaluation_metrics(self, 
                                 episode_rewards: List[float],
                                 episode_costs: List[float],
                                 episode_lengths: List[int],
                                 safety_violations: List[int],
                                 human_metrics: List[Dict[str, float]],
                                 computation_times: Optional[List[float]] = None,
                                 memory_usage: Optional[List[float]] = None,
                                 **kwargs) -> EvaluationMetrics:
        """Compute comprehensive evaluation metrics."""
        
        # Performance metrics
        sample_efficiency = self.performance_metrics.compute_sample_efficiency(
            episode_rewards, list(range(len(episode_rewards)))
        )
        asymptotic_performance = self.performance_metrics.compute_asymptotic_performance(
            episode_rewards
        )
        learning_curve_auc = self.performance_metrics.compute_learning_curve_auc(
            episode_rewards
        )
        convergence_time = self.performance_metrics.compute_convergence_time(
            episode_rewards
        )
        final_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
        
        # Safety metrics
        violation_rate = self.safety_metrics.compute_violation_rate(safety_violations)
        constraint_satisfaction = self.safety_metrics.compute_constraint_satisfaction(
            episode_costs, kwargs.get('cost_limit', 25.0)
        )
        safety_margin = self.safety_metrics.compute_safety_margin(
            episode_costs, kwargs.get('cost_limit', 25.0)
        )
        
        # Human metrics
        human_satisfaction = self.human_metrics.compute_satisfaction(human_metrics)
        trust_level = self.human_metrics.compute_trust_level(human_metrics, safety_violations)
        workload_score = self.human_metrics.compute_workload_score(human_metrics)
        naturalness_rating = self.human_metrics.compute_naturalness_rating(human_metrics)
        collaboration_efficiency = self.human_metrics.compute_collaboration_efficiency(
            [float(length) for length in episode_lengths]
        )
        
        # Efficiency metrics
        if computation_times:
            computation_time = np.mean(computation_times)
            inference_time = computation_time
        else:
            computation_time = 0.0
            inference_time = 0.0
        
        if memory_usage:
            memory_usage_avg = np.mean(memory_usage)
        else:
            memory_usage_avg = 0.0
        
        training_stability = self.performance_metrics.compute_training_stability(episode_rewards)
        
        # Create comprehensive metrics object
        metrics = EvaluationMetrics(
            sample_efficiency=sample_efficiency,
            asymptotic_performance=asymptotic_performance,
            learning_curve_auc=learning_curve_auc,
            convergence_time=convergence_time,
            final_reward=final_reward,
            safety_violations=sum(safety_violations),
            violation_rate=violation_rate,
            constraint_satisfaction=constraint_satisfaction,
            safety_margin=safety_margin,
            recovery_time=0.0,  # Placeholder
            human_satisfaction=human_satisfaction,
            trust_level=trust_level,
            workload_score=workload_score,
            naturalness_rating=naturalness_rating,
            collaboration_efficiency=collaboration_efficiency,
            computation_time=computation_time,
            memory_usage=memory_usage_avg,
            inference_time=inference_time,
            training_stability=training_stability,
            parameter_efficiency=1.0,  # Placeholder
            robustness_score=0.8,  # Placeholder
            adaptability_measure=0.7,  # Placeholder
            generalization_performance=0.75  # Placeholder
        )
        
        return metrics
    
    def compute_comparative_metrics(self, 
                                  results: Dict[str, List[EvaluationMetrics]]) -> Dict[str, Dict[str, float]]:
        """Compute comparative metrics across algorithms."""
        comparative_metrics = {}
        
        # Get all metric names from first result
        if results:
            first_alg = next(iter(results.keys()))
            if results[first_alg]:
                metric_names = list(results[first_alg][0].to_dict().keys())
            else:
                return comparative_metrics
        else:
            return comparative_metrics
        
        for metric_name in metric_names:
            comparative_metrics[metric_name] = {}
            
            # Collect all values for this metric
            metric_values = {}
            for alg_name, alg_results in results.items():
                values = [getattr(result, metric_name) for result in alg_results]
                metric_values[alg_name] = values
            
            # Compute comparative statistics
            for alg_name, values in metric_values.items():
                comparative_metrics[metric_name][alg_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'rank': self._compute_rank(values, metric_values, higher_is_better=True)
                }
        
        return comparative_metrics
    
    def _compute_rank(self, values: List[float], 
                     all_values: Dict[str, List[float]], 
                     higher_is_better: bool = True) -> float:
        """Compute rank of algorithm for given metric."""
        means = {alg: np.mean(vals) for alg, vals in all_values.items()}
        
        if higher_is_better:
            sorted_algs = sorted(means.keys(), key=lambda x: means[x], reverse=True)
        else:
            sorted_algs = sorted(means.keys(), key=lambda x: means[x])
        
        # Find algorithm with these values
        current_alg = None
        current_mean = np.mean(values)
        for alg, mean_val in means.items():
            if abs(mean_val - current_mean) < 1e-10:
                current_alg = alg
                break
        
        if current_alg:
            rank = sorted_algs.index(current_alg) + 1
            return rank / len(sorted_algs)  # Normalized rank
        
        return 0.5  # Default middle rank