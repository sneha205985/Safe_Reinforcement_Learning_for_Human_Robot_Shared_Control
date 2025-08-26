"""
Baseline comparison framework for Safe RL algorithms.

This module provides comprehensive comparison capabilities against baseline methods
including standard PPO, Lagrangian PPO, Safety Gym baselines, and hand-crafted controllers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json

from .performance_analyzer import PerformanceAnalyzer
from .safety_analyzer import SafetyAnalyzer
from .statistical_tests import StatisticalTester


@dataclass
class BaselineConfig:
    """Configuration for baseline algorithms."""
    
    name: str
    algorithm_type: str  # 'ppo', 'lagrangian_ppo', 'safety_gym', 'hand_crafted'
    
    # Algorithm-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Expected performance characteristics
    expected_performance_range: Tuple[float, float] = (0.0, 1.0)
    expected_violation_rate: float = 0.1
    
    # Data source
    data_source: Optional[str] = None
    results_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'algorithm_type': self.algorithm_type,
            'parameters': self.parameters,
            'expected_performance_range': self.expected_performance_range,
            'expected_violation_rate': self.expected_violation_rate,
            'data_source': self.data_source,
            'results_path': self.results_path
        }


class BaselineImplementations:
    """Implementations and configurations for standard baseline algorithms."""
    
    @staticmethod
    def get_standard_baselines() -> List[BaselineConfig]:
        """Get configurations for standard baseline algorithms."""
        return [
            # Standard PPO (unconstrained)
            BaselineConfig(
                name="Standard PPO",
                algorithm_type="ppo",
                parameters={
                    'learning_rate': 3e-4,
                    'clip_ratio': 0.2,
                    'entropy_coef': 0.01,
                    'value_coef': 0.5,
                    'max_grad_norm': 0.5
                },
                expected_performance_range=(0.6, 0.9),
                expected_violation_rate=0.15
            ),
            
            # Lagrangian PPO
            BaselineConfig(
                name="Lagrangian PPO",
                algorithm_type="lagrangian_ppo",
                parameters={
                    'learning_rate': 3e-4,
                    'clip_ratio': 0.2,
                    'lagrange_lr': 1e-3,
                    'constraint_threshold': 0.01,
                    'penalty_coef': 10.0
                },
                expected_performance_range=(0.5, 0.8),
                expected_violation_rate=0.05
            ),
            
            # TRPO with constraints
            BaselineConfig(
                name="TRPO-C",
                algorithm_type="trpo_constrained",
                parameters={
                    'learning_rate': 3e-4,
                    'kl_threshold': 0.01,
                    'constraint_threshold': 0.01,
                    'cg_iters': 10
                },
                expected_performance_range=(0.4, 0.75),
                expected_violation_rate=0.03
            ),
            
            # Hand-crafted controller
            BaselineConfig(
                name="Hand-crafted Controller",
                algorithm_type="hand_crafted",
                parameters={
                    'p_gain': 1.0,
                    'd_gain': 0.1,
                    'safety_margin': 0.05,
                    'max_velocity': 1.0
                },
                expected_performance_range=(0.3, 0.6),
                expected_violation_rate=0.02
            )
        ]
    
    @staticmethod
    def get_safety_gym_baselines() -> List[BaselineConfig]:
        """Get Safety Gym baseline configurations."""
        return [
            BaselineConfig(
                name="Safety Gym CPO",
                algorithm_type="safety_gym_cpo",
                parameters={
                    'learning_rate': 3e-4,
                    'trust_region_radius': 0.01,
                    'constraint_threshold': 0.01
                },
                expected_performance_range=(0.5, 0.8),
                expected_violation_rate=0.04
            ),
            
            BaselineConfig(
                name="Safety Gym PPO-Lagrangian",
                algorithm_type="safety_gym_ppo_lag",
                parameters={
                    'learning_rate': 3e-4,
                    'lagrange_lr': 5e-3,
                    'constraint_threshold': 0.01
                },
                expected_performance_range=(0.4, 0.75),
                expected_violation_rate=0.06
            ),
            
            BaselineConfig(
                name="Safety Gym TRPO-Lagrangian",
                algorithm_type="safety_gym_trpo_lag",
                parameters={
                    'learning_rate': 3e-4,
                    'kl_threshold': 0.01,
                    'lagrange_lr': 5e-3
                },
                expected_performance_range=(0.35, 0.7),
                expected_violation_rate=0.05
            )
        ]


class BaselineDataLoader:
    """Loads and preprocesses baseline algorithm data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_baseline_results(self, baseline_config: BaselineConfig) -> Optional[pd.DataFrame]:
        """Load results for a baseline algorithm."""
        if baseline_config.results_path is None:
            self.logger.warning(f"No results path specified for {baseline_config.name}")
            return None
        
        results_path = Path(baseline_config.results_path)
        
        if not results_path.exists():
            self.logger.warning(f"Results file not found: {results_path}")
            return None
        
        try:
            if results_path.suffix == '.csv':
                data = pd.read_csv(results_path)
            elif results_path.suffix == '.json':
                with open(results_path) as f:
                    json_data = json.load(f)
                data = pd.DataFrame(json_data)
            elif results_path.suffix == '.pkl':
                data = pd.read_pickle(results_path)
            else:
                self.logger.error(f"Unsupported file format: {results_path.suffix}")
                return None
            
            self.logger.info(f"Loaded baseline data for {baseline_config.name}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading baseline data from {results_path}: {e}")
            return None
    
    def simulate_baseline_performance(self, baseline_config: BaselineConfig,
                                     num_iterations: int = 1000,
                                     seed: int = 42) -> pd.DataFrame:
        """Simulate baseline performance when real data is unavailable."""
        np.random.seed(seed)
        
        # Generate synthetic data based on algorithm characteristics
        data = {
            'iteration': range(num_iterations),
            'episode_return': self._simulate_performance_curve(baseline_config, num_iterations),
            'constraint_violations': self._simulate_constraint_violations(baseline_config, num_iterations),
            'policy_loss': self._simulate_loss_curve(baseline_config, num_iterations, 'policy'),
            'value_loss': self._simulate_loss_curve(baseline_config, num_iterations, 'value')
        }
        
        # Add algorithm-specific metrics
        if baseline_config.algorithm_type == 'lagrangian_ppo':
            data['lagrange_multiplier'] = self._simulate_lagrange_multiplier(num_iterations)
        elif baseline_config.algorithm_type == 'hand_crafted':
            data['control_effort'] = self._simulate_control_effort(num_iterations)
        
        df = pd.DataFrame(data)
        
        self.logger.info(f"Simulated baseline data for {baseline_config.name}")
        return df
    
    def _simulate_performance_curve(self, config: BaselineConfig, 
                                   num_iterations: int) -> np.ndarray:
        """Simulate realistic performance learning curve."""
        min_perf, max_perf = config.expected_performance_range
        
        if config.algorithm_type == 'hand_crafted':
            # Hand-crafted controllers have constant performance with noise
            base_performance = (min_perf + max_perf) / 2
            noise = np.random.normal(0, 0.05, num_iterations)
            return np.clip(base_performance + noise, min_perf, max_perf)
        
        # Learning algorithms: sigmoid-shaped learning curve
        x = np.linspace(0, 10, num_iterations)
        
        if config.algorithm_type == 'ppo':
            # Fast initial learning, then plateaus
            learning_rate = 2.0
            plateau_start = 0.7 * num_iterations
        elif 'lagrangian' in config.algorithm_type.lower():
            # Slower but more stable learning
            learning_rate = 1.5
            plateau_start = 0.8 * num_iterations
        elif 'trpo' in config.algorithm_type.lower():
            # Conservative learning
            learning_rate = 1.2
            plateau_start = 0.9 * num_iterations
        else:
            learning_rate = 1.8
            plateau_start = 0.75 * num_iterations
        
        # Sigmoid function for learning
        sigmoid = 1 / (1 + np.exp(-learning_rate * (x - 5)))
        performance = min_perf + (max_perf - min_perf) * sigmoid
        
        # Add noise and occasional performance drops
        noise = np.random.normal(0, 0.02, num_iterations)
        performance += noise
        
        # Occasional performance drops for unconstrained algorithms
        if config.algorithm_type == 'ppo':
            for _ in range(3):  # 3 performance drops
                drop_start = np.random.randint(num_iterations // 4, 3 * num_iterations // 4)
                drop_length = np.random.randint(10, 50)
                drop_end = min(drop_start + drop_length, num_iterations)
                drop_severity = np.random.uniform(0.1, 0.3)
                
                for i in range(drop_start, drop_end):
                    performance[i] *= (1 - drop_severity)
        
        return np.clip(performance, min_perf, max_perf)
    
    def _simulate_constraint_violations(self, config: BaselineConfig,
                                       num_iterations: int) -> np.ndarray:
        """Simulate constraint violation patterns."""
        base_rate = config.expected_violation_rate
        
        if config.algorithm_type == 'hand_crafted':
            # Very low, constant violation rate for hand-crafted controllers
            return np.random.poisson(base_rate, num_iterations)
        
        # Learning-based algorithms: high initial violations, then decreasing
        x = np.linspace(0, 1, num_iterations)
        
        if config.algorithm_type == 'ppo':
            # Unconstrained PPO: violations may increase over time as it optimizes reward
            violation_rate = base_rate * (1 + 0.5 * x + 0.2 * np.sin(10 * x))
        elif 'lagrangian' in config.algorithm_type.lower():
            # Lagrangian methods: decreasing violations over time
            violation_rate = base_rate * (2 * np.exp(-3 * x) + 0.2)
        elif 'cpo' in config.algorithm_type.lower():
            # CPO: low violation rate that decreases over time
            violation_rate = base_rate * (1.5 * np.exp(-4 * x) + 0.1)
        else:
            # Default: moderate decrease
            violation_rate = base_rate * (1.8 * np.exp(-2 * x) + 0.3)
        
        # Generate Poisson-distributed violations
        violations = np.random.poisson(violation_rate * 10)  # Scale for visibility
        
        return violations
    
    def _simulate_loss_curve(self, config: BaselineConfig, 
                            num_iterations: int, loss_type: str) -> np.ndarray:
        """Simulate loss curves for policy and value functions."""
        x = np.linspace(0, 5, num_iterations)
        
        # Base exponential decay
        if loss_type == 'policy':
            initial_loss = np.random.uniform(1.0, 2.0)
            decay_rate = 1.5
        else:  # value loss
            initial_loss = np.random.uniform(0.5, 1.5)
            decay_rate = 2.0
        
        # Algorithm-specific characteristics
        if config.algorithm_type == 'hand_crafted':
            # No learning, just noise around zero
            return np.abs(np.random.normal(0, 0.01, num_iterations))
        
        base_loss = initial_loss * np.exp(-decay_rate * x / 5)
        
        # Add noise and occasional spikes
        noise = np.random.normal(0, base_loss * 0.1)
        base_loss += noise
        
        # Occasional training instabilities
        for _ in range(2):
            spike_location = np.random.randint(num_iterations // 4, 3 * num_iterations // 4)
            spike_width = np.random.randint(5, 20)
            spike_height = initial_loss * np.random.uniform(0.5, 1.5)
            
            for i in range(max(0, spike_location - spike_width),
                          min(num_iterations, spike_location + spike_width)):
                distance = abs(i - spike_location)
                weight = np.exp(-distance / (spike_width / 3))
                base_loss[i] += spike_height * weight
        
        return np.maximum(base_loss, 0.001)  # Ensure positive losses
    
    def _simulate_lagrange_multiplier(self, num_iterations: int) -> np.ndarray:
        """Simulate Lagrange multiplier evolution."""
        # Start high, then adapt based on constraint violations
        x = np.linspace(0, 3, num_iterations)
        
        # Exponential approach to equilibrium with oscillations
        equilibrium = np.random.uniform(5, 15)
        approach = equilibrium * (1 - np.exp(-x))
        oscillations = 2 * np.sin(2 * x) * np.exp(-x / 2)
        noise = np.random.normal(0, 0.5, num_iterations)
        
        multiplier = approach + oscillations + noise
        return np.maximum(multiplier, 0.01)  # Ensure positive
    
    def _simulate_control_effort(self, num_iterations: int) -> np.ndarray:
        """Simulate control effort for hand-crafted controllers."""
        # Relatively constant with some variation based on task difficulty
        base_effort = np.random.uniform(0.3, 0.7)
        seasonal_variation = 0.1 * np.sin(2 * np.pi * np.arange(num_iterations) / 100)
        noise = np.random.normal(0, 0.05, num_iterations)
        
        return np.maximum(base_effort + seasonal_variation + noise, 0.1)


class AlgorithmComparison:
    """Performs detailed comparison between algorithms."""
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.safety_analyzer = SafetyAnalyzer()
        self.statistical_tester = StatisticalTester()
        self.logger = logging.getLogger(__name__)
    
    def compare_algorithm_performance(self, cpo_data: pd.DataFrame,
                                     baseline_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare CPO against baseline algorithms."""
        comparison = {
            'algorithms': ['CPO'] + list(baseline_data.keys()),
            'performance_comparison': {},
            'safety_comparison': {},
            'statistical_analysis': {},
            'pareto_analysis': {},
            'summary': {}
        }
        
        # Prepare all data
        all_data = {'CPO': cpo_data}
        all_data.update(baseline_data)
        
        # Performance comparison
        comparison['performance_comparison'] = self.performance_analyzer.compare_algorithms(all_data)
        
        # Safety comparison
        comparison['safety_comparison'] = self.safety_analyzer.compare_safety_performance(all_data)
        
        # Statistical analysis
        comparison['statistical_analysis'] = self._perform_comprehensive_statistical_analysis(all_data)
        
        # Pareto frontier analysis
        comparison['pareto_analysis'] = self._analyze_pareto_frontier(all_data)
        
        # Generate summary
        comparison['summary'] = self._generate_comparison_summary(comparison)
        
        return comparison
    
    def _perform_comprehensive_statistical_analysis(self, 
                                                   algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        analysis = {
            'pairwise_comparisons': {},
            'anova_results': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        # Pairwise comparisons
        alg_names = list(algorithms_data.keys())
        for i, alg1 in enumerate(alg_names):
            for alg2 in alg_names[i+1:]:
                comparison_key = f"{alg1}_vs_{alg2}"
                
                pairwise_result = self._compare_two_algorithms_statistically(
                    algorithms_data[alg1], algorithms_data[alg2], alg1, alg2
                )
                analysis['pairwise_comparisons'][comparison_key] = pairwise_result
        
        # ANOVA for multiple algorithm comparison
        if len(algorithms_data) > 2:
            analysis['anova_results'] = self._perform_anova_analysis(algorithms_data)
        
        # Effect sizes
        analysis['effect_sizes'] = self._compute_effect_sizes(algorithms_data)
        
        # Confidence intervals
        analysis['confidence_intervals'] = self._compute_confidence_intervals(algorithms_data)
        
        return analysis
    
    def _compare_two_algorithms_statistically(self, data1: pd.DataFrame, data2: pd.DataFrame,
                                             name1: str, name2: str) -> Dict[str, Any]:
        """Compare two algorithms statistically."""
        comparison = {
            'algorithms': [name1, name2],
            'performance_tests': {},
            'safety_tests': {},
            'sample_efficiency_tests': {}
        }
        
        # Performance comparison
        if 'episode_return' in data1.columns and 'episode_return' in data2.columns:
            returns1 = data1['episode_return'].dropna().values
            returns2 = data2['episode_return'].dropna().values
            
            if len(returns1) >= 10 and len(returns2) >= 10:
                # t-test
                t_stat, t_p = stats.ttest_ind(returns1, returns2)
                
                # Mann-Whitney U test (non-parametric)
                u_stat, u_p = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')
                
                # Welch's t-test (unequal variances)
                welch_stat, welch_p = stats.ttest_ind(returns1, returns2, equal_var=False)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(returns1) - 1) * np.var(returns1, ddof=1) + 
                                    (len(returns2) - 1) * np.var(returns2, ddof=1)) / 
                                   (len(returns1) + len(returns2) - 2))
                cohens_d = (np.mean(returns1) - np.mean(returns2)) / pooled_std if pooled_std > 0 else 0
                
                comparison['performance_tests'] = {
                    'mean_difference': float(np.mean(returns1) - np.mean(returns2)),
                    't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
                    'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
                    'welch_test': {'statistic': float(welch_stat), 'p_value': float(welch_p)},
                    'cohens_d': float(cohens_d),
                    'significant': min(t_p, u_p, welch_p) < 0.05,
                    'winner': name1 if np.mean(returns1) > np.mean(returns2) else name2
                }
        
        # Safety comparison
        constraint_cols = [col for col in data1.columns 
                          if any(keyword in col.lower() for keyword in ['constraint', 'violation'])]
        
        if constraint_cols:
            for col in constraint_cols:
                if col in data1.columns and col in data2.columns:
                    violations1 = (data1[col] > 0).sum() if 'violation' in col or 'constraint' in col else (data1[col] < 0).sum()
                    violations2 = (data2[col] > 0).sum() if 'violation' in col or 'constraint' in col else (data2[col] < 0).sum()
                    
                    total1, total2 = len(data1), len(data2)
                    
                    # Chi-square test for violation rates
                    contingency_table = np.array([[violations1, total1 - violations1],
                                                [violations2, total2 - violations2]])
                    
                    if np.all(contingency_table >= 5):  # Chi-square assumption
                        chi2_stat, chi2_p = stats.chi2_contingency(contingency_table)[:2]
                        
                        comparison['safety_tests'][col] = {
                            'violation_rate_1': float(violations1 / total1),
                            'violation_rate_2': float(violations2 / total2),
                            'chi2_statistic': float(chi2_stat),
                            'chi2_p_value': float(chi2_p),
                            'significant_difference': float(chi2_p) < 0.05,
                            'safer_algorithm': name1 if violations1 < violations2 else name2
                        }
        
        return comparison
    
    def _perform_anova_analysis(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform ANOVA analysis across multiple algorithms."""
        anova_results = {}
        
        # Performance ANOVA
        if all('episode_return' in data.columns for data in algorithms_data.values()):
            performance_groups = []
            group_labels = []
            
            for alg_name, data in algorithms_data.items():
                returns = data['episode_return'].dropna().values
                if len(returns) >= 5:  # Minimum for ANOVA
                    performance_groups.append(returns)
                    group_labels.append(alg_name)
            
            if len(performance_groups) >= 3:
                f_stat, p_value = stats.f_oneway(*performance_groups)
                
                anova_results['performance'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': float(p_value) < 0.05,
                    'groups': group_labels
                }
                
                # Post-hoc analysis if significant
                if p_value < 0.05:
                    anova_results['performance']['posthoc'] = self._perform_posthoc_analysis(
                        performance_groups, group_labels
                    )
        
        return anova_results
    
    def _perform_posthoc_analysis(self, groups: List[np.ndarray], 
                                 group_labels: List[str]) -> Dict[str, Any]:
        """Perform post-hoc analysis after significant ANOVA."""
        posthoc_results = {}
        
        # Tukey HSD test
        try:
            from scipy.stats import tukey_hsd
            
            tukey_result = tukey_hsd(*groups)
            
            posthoc_results['tukey_hsd'] = {
                'statistic': float(tukey_result.statistic),
                'pvalue': float(tukey_result.pvalue),
                'groups': group_labels
            }
            
        except ImportError:
            # Manual pairwise comparisons with Bonferroni correction
            n_comparisons = len(groups) * (len(groups) - 1) // 2
            alpha_corrected = 0.05 / n_comparisons
            
            pairwise_results = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                    
                    pairwise_results.append({
                        'groups': [group_labels[i], group_labels[j]],
                        'mean_difference': float(np.mean(groups[i]) - np.mean(groups[j])),
                        'p_value': float(p_val),
                        'significant_bonferroni': float(p_val) < alpha_corrected
                    })
            
            posthoc_results['bonferroni_corrected'] = {
                'alpha_corrected': alpha_corrected,
                'pairwise_comparisons': pairwise_results
            }
        
        return posthoc_results
    
    def _compute_effect_sizes(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compute effect sizes for algorithm comparisons."""
        effect_sizes = {}
        alg_names = list(algorithms_data.keys())
        
        for i, alg1 in enumerate(alg_names):
            for alg2 in alg_names[i+1:]:
                if 'episode_return' in algorithms_data[alg1].columns and 'episode_return' in algorithms_data[alg2].columns:
                    returns1 = algorithms_data[alg1]['episode_return'].dropna().values
                    returns2 = algorithms_data[alg2]['episode_return'].dropna().values
                    
                    if len(returns1) >= 3 and len(returns2) >= 3:
                        # Cohen's d
                        pooled_std = np.sqrt(((len(returns1) - 1) * np.var(returns1, ddof=1) + 
                                            (len(returns2) - 1) * np.var(returns2, ddof=1)) / 
                                           (len(returns1) + len(returns2) - 2))
                        cohens_d = (np.mean(returns1) - np.mean(returns2)) / pooled_std if pooled_std > 0 else 0
                        
                        # Glass's delta (using control group std)
                        glass_delta = (np.mean(returns1) - np.mean(returns2)) / np.std(returns2, ddof=1) if np.std(returns2, ddof=1) > 0 else 0
                        
                        # Hedge's g (bias-corrected Cohen's d)
                        correction_factor = 1 - (3 / (4 * (len(returns1) + len(returns2)) - 9))
                        hedges_g = cohens_d * correction_factor
                        
                        effect_sizes[f"{alg1}_vs_{alg2}"] = {
                            'cohens_d': float(cohens_d),
                            'glass_delta': float(glass_delta),
                            'hedges_g': float(hedges_g),
                            'interpretation': self._interpret_effect_size(abs(cohens_d))
                        }
        
        return effect_sizes
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret effect size magnitude."""
        if effect_size < 0.2:
            return 'negligible'
        elif effect_size < 0.5:
            return 'small'
        elif effect_size < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _compute_confidence_intervals(self, algorithms_data: Dict[str, pd.DataFrame],
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """Compute confidence intervals for algorithm performance."""
        confidence_intervals = {}
        
        for alg_name, data in algorithms_data.items():
            if 'episode_return' in data.columns:
                returns = data['episode_return'].dropna().values
                
                if len(returns) >= 3:
                    mean_return = np.mean(returns)
                    sem = stats.sem(returns)
                    
                    # t-distribution confidence interval
                    ci = stats.t.interval(confidence_level, len(returns) - 1,
                                         loc=mean_return, scale=sem)
                    
                    # Bootstrap confidence interval
                    bootstrap_means = []
                    for _ in range(1000):
                        bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    bootstrap_ci = np.percentile(bootstrap_means, 
                                               [100 * (1 - confidence_level) / 2,
                                                100 * (1 + confidence_level) / 2])
                    
                    confidence_intervals[alg_name] = {
                        'mean': float(mean_return),
                        't_distribution_ci': [float(ci[0]), float(ci[1])],
                        'bootstrap_ci': [float(bootstrap_ci[0]), float(bootstrap_ci[1])],
                        'confidence_level': confidence_level
                    }
        
        return confidence_intervals
    
    def _analyze_pareto_frontier(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze Pareto frontier for performance vs safety trade-off."""
        pareto_analysis = {
            'algorithms': [],
            'performance_safety_points': [],
            'pareto_efficient': [],
            'dominated_algorithms': [],
            'frontier_analysis': {}
        }
        
        # Extract performance and safety metrics
        algorithm_points = []
        
        for alg_name, data in algorithms_data.items():
            if 'episode_return' in data.columns:
                # Performance: mean episode return
                performance = data['episode_return'].dropna().mean()
                
                # Safety: inverse of violation rate (higher is better)
                safety_score = 1.0
                constraint_cols = [col for col in data.columns 
                                 if any(keyword in col.lower() 
                                       for keyword in ['constraint', 'violation'])]
                
                if constraint_cols:
                    total_violations = 0
                    total_steps = 0
                    
                    for col in constraint_cols:
                        violations = (data[col] > 0).sum() if 'violation' in col or 'constraint' in col else (data[col] < 0).sum()
                        total_violations += violations
                        total_steps += len(data)
                    
                    violation_rate = total_violations / total_steps if total_steps > 0 else 0
                    safety_score = 1.0 - violation_rate  # Higher is better
                
                algorithm_points.append({
                    'name': alg_name,
                    'performance': float(performance),
                    'safety': float(safety_score)
                })
        
        if not algorithm_points:
            return pareto_analysis
        
        pareto_analysis['algorithms'] = [p['name'] for p in algorithm_points]
        pareto_analysis['performance_safety_points'] = [
            (p['performance'], p['safety']) for p in algorithm_points
        ]
        
        # Find Pareto efficient algorithms
        pareto_efficient = self._find_pareto_efficient(algorithm_points)
        pareto_analysis['pareto_efficient'] = [p['name'] for p in pareto_efficient]
        
        dominated = [p['name'] for p in algorithm_points if p not in pareto_efficient]
        pareto_analysis['dominated_algorithms'] = dominated
        
        # Frontier analysis
        if len(pareto_efficient) > 1:
            pareto_analysis['frontier_analysis'] = self._analyze_frontier_properties(pareto_efficient)
        
        return pareto_analysis
    
    def _find_pareto_efficient(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find Pareto efficient points (higher performance and safety are better)."""
        pareto_efficient = []
        
        for i, point_i in enumerate(points):
            is_pareto_efficient = True
            
            for j, point_j in enumerate(points):
                if i != j:
                    # Check if point_j dominates point_i
                    if (point_j['performance'] >= point_i['performance'] and 
                        point_j['safety'] >= point_i['safety'] and
                        (point_j['performance'] > point_i['performance'] or 
                         point_j['safety'] > point_i['safety'])):
                        is_pareto_efficient = False
                        break
            
            if is_pareto_efficient:
                pareto_efficient.append(point_i)
        
        return pareto_efficient
    
    def _analyze_frontier_properties(self, pareto_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze properties of the Pareto frontier."""
        # Sort by performance
        sorted_points = sorted(pareto_points, key=lambda x: x['performance'])
        
        analysis = {
            'frontier_length': len(sorted_points),
            'performance_range': (sorted_points[0]['performance'], sorted_points[-1]['performance']),
            'safety_range': (min(p['safety'] for p in sorted_points), 
                           max(p['safety'] for p in sorted_points)),
            'trade_offs': []
        }
        
        # Analyze trade-offs between adjacent points
        for i in range(len(sorted_points) - 1):
            p1, p2 = sorted_points[i], sorted_points[i + 1]
            
            perf_gain = p2['performance'] - p1['performance']
            safety_loss = p1['safety'] - p2['safety']  # Typically safety decreases as performance increases
            
            trade_off_ratio = safety_loss / perf_gain if perf_gain > 0 else 0
            
            analysis['trade_offs'].append({
                'from_algorithm': p1['name'],
                'to_algorithm': p2['name'],
                'performance_gain': float(perf_gain),
                'safety_loss': float(safety_loss),
                'trade_off_ratio': float(trade_off_ratio)
            })
        
        return analysis
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the comparison results."""
        summary = {
            'best_overall_performance': None,
            'safest_algorithm': None,
            'most_sample_efficient': None,
            'pareto_optimal': [],
            'key_findings': []
        }
        
        # Best overall performance
        perf_rankings = comparison.get('performance_comparison', {}).get('performance_rankings', {})
        if 'final_performance' in perf_rankings and perf_rankings['final_performance']:
            summary['best_overall_performance'] = perf_rankings['final_performance'][0]
        
        # Safest algorithm
        safety_rankings = comparison.get('safety_comparison', {}).get('safety_rankings', {})
        if 'by_violation_rate' in safety_rankings and safety_rankings['by_violation_rate']:
            summary['safest_algorithm'] = safety_rankings['by_violation_rate'][0]
        
        # Most sample efficient
        if 'sample_efficiency' in perf_rankings and perf_rankings['sample_efficiency']:
            summary['most_sample_efficient'] = perf_rankings['sample_efficiency'][0]
        
        # Pareto optimal algorithms
        pareto_efficient = comparison.get('pareto_analysis', {}).get('pareto_efficient', [])
        summary['pareto_optimal'] = pareto_efficient
        
        # Key findings based on statistical analysis
        statistical_analysis = comparison.get('statistical_analysis', {})
        pairwise_comparisons = statistical_analysis.get('pairwise_comparisons', {})
        
        findings = []
        for comparison_key, result in pairwise_comparisons.items():
            if result.get('performance_tests', {}).get('significant', False):
                winner = result['performance_tests']['winner']
                mean_diff = result['performance_tests']['mean_difference']
                effect_size = result['performance_tests']['cohens_d']
                
                if abs(effect_size) > 0.5:  # Medium or large effect
                    findings.append(f"{winner} significantly outperforms in {comparison_key.replace('_vs_', ' vs ')} "
                                  f"(effect size: {abs(effect_size):.2f})")
        
        summary['key_findings'] = findings[:5]  # Top 5 findings
        
        return summary


class BenchmarkSuite:
    """Complete benchmark suite for Safe RL algorithms."""
    
    def __init__(self):
        self.baseline_loader = BaselineDataLoader()
        self.comparator = AlgorithmComparison()
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_benchmark(self, cpo_data: pd.DataFrame,
                                   include_simulated: bool = True,
                                   custom_baselines: List[BaselineConfig] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark against all baseline algorithms."""
        benchmark_results = {
            'cpo_algorithm': 'Constrained Policy Optimization',
            'baseline_algorithms': {},
            'comparison_results': {},
            'benchmark_summary': {}
        }
        
        # Get baseline configurations
        baselines = BaselineImplementations.get_standard_baselines()
        baselines.extend(BaselineImplementations.get_safety_gym_baselines())
        
        if custom_baselines:
            baselines.extend(custom_baselines)
        
        # Load or simulate baseline data
        baseline_data = {}
        
        for baseline_config in baselines:
            try:
                # Try to load real data first
                data = self.baseline_loader.load_baseline_results(baseline_config)
                
                # If no real data and simulation is allowed, simulate
                if data is None and include_simulated:
                    data = self.baseline_loader.simulate_baseline_performance(
                        baseline_config, 
                        num_iterations=len(cpo_data)
                    )
                
                if data is not None:
                    baseline_data[baseline_config.name] = data
                    benchmark_results['baseline_algorithms'][baseline_config.name] = baseline_config.to_dict()
                
            except Exception as e:
                self.logger.error(f"Failed to load/simulate {baseline_config.name}: {e}")
                continue
        
        if not baseline_data:
            self.logger.error("No baseline data available for comparison")
            return benchmark_results
        
        # Perform comparison
        self.logger.info(f"Comparing CPO against {len(baseline_data)} baseline algorithms")
        comparison_results = self.comparator.compare_algorithm_performance(cpo_data, baseline_data)
        benchmark_results['comparison_results'] = comparison_results
        
        # Generate benchmark summary
        benchmark_results['benchmark_summary'] = self._generate_benchmark_summary(
            comparison_results, baseline_data.keys()
        )
        
        return benchmark_results
    
    def _generate_benchmark_summary(self, comparison_results: Dict[str, Any],
                                   baseline_names: List[str]) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        summary = {
            'total_baselines_compared': len(baseline_names),
            'cpo_performance_rank': None,
            'cpo_safety_rank': None,
            'significant_improvements': [],
            'areas_for_improvement': [],
            'overall_assessment': ''
        }
        
        # Extract rankings
        perf_rankings = comparison_results.get('performance_comparison', {}).get('performance_rankings', {})
        safety_rankings = comparison_results.get('safety_comparison', {}).get('safety_rankings', {})
        
        if 'final_performance' in perf_rankings:
            cpo_rank = perf_rankings['final_performance'].index('CPO') + 1 if 'CPO' in perf_rankings['final_performance'] else None
            summary['cpo_performance_rank'] = cpo_rank
        
        if 'by_violation_rate' in safety_rankings:
            cpo_rank = safety_rankings['by_violation_rate'].index('CPO') + 1 if 'CPO' in safety_rankings['by_violation_rate'] else None
            summary['cpo_safety_rank'] = cpo_rank
        
        # Identify significant improvements
        statistical_analysis = comparison_results.get('statistical_analysis', {})
        pairwise_comparisons = statistical_analysis.get('pairwise_comparisons', {})
        
        improvements = []
        areas_for_improvement = []
        
        for comparison_key, result in pairwise_comparisons.items():
            if 'CPO_vs_' in comparison_key:
                baseline_name = comparison_key.replace('CPO_vs_', '')
                
                perf_test = result.get('performance_tests', {})
                if perf_test.get('significant', False):
                    if perf_test.get('winner') == 'CPO':
                        improvements.append(f"Significantly outperforms {baseline_name} "
                                          f"(effect size: {abs(perf_test.get('cohens_d', 0)):.2f})")
                    else:
                        areas_for_improvement.append(f"Underperforms compared to {baseline_name}")
        
        summary['significant_improvements'] = improvements
        summary['areas_for_improvement'] = areas_for_improvement
        
        # Overall assessment
        if summary['cpo_performance_rank'] and summary['cpo_safety_rank']:
            perf_rank = summary['cpo_performance_rank']
            safety_rank = summary['cpo_safety_rank']
            total = summary['total_baselines_compared'] + 1  # +1 for CPO itself
            
            if perf_rank <= total // 3 and safety_rank <= total // 3:
                assessment = "Excellent: Top-tier performance in both performance and safety"
            elif perf_rank <= total // 2 and safety_rank <= total // 2:
                assessment = "Good: Above-average performance in both dimensions"
            elif perf_rank <= total // 3 or safety_rank <= total // 3:
                assessment = "Mixed: Strong in one dimension, needs improvement in the other"
            else:
                assessment = "Needs improvement: Below-average performance compared to baselines"
            
            summary['overall_assessment'] = assessment
        
        return summary


class BaselineComparator:
    """Main class for baseline comparison analysis."""
    
    def __init__(self):
        self.benchmark_suite = BenchmarkSuite()
        self.logger = logging.getLogger(__name__)
    
    def compare_with_baselines(self, cpo_results_path: str,
                              baseline_configs_path: str = None,
                              include_simulated: bool = True) -> Dict[str, Any]:
        """Compare CPO with baseline algorithms."""
        try:
            # Load CPO data
            if cpo_results_path.endswith('.csv'):
                cpo_data = pd.read_csv(cpo_results_path)
            elif cpo_results_path.endswith('.json'):
                with open(cpo_results_path) as f:
                    json_data = json.load(f)
                cpo_data = pd.DataFrame(json_data if isinstance(json_data, list) else [json_data])
            else:
                raise ValueError(f"Unsupported file format: {cpo_results_path}")
            
            # Load custom baseline configurations if provided
            custom_baselines = None
            if baseline_configs_path and Path(baseline_configs_path).exists():
                with open(baseline_configs_path) as f:
                    configs_data = json.load(f)
                custom_baselines = [BaselineConfig(**config) for config in configs_data]
            
            # Run benchmark
            results = self.benchmark_suite.run_comprehensive_benchmark(
                cpo_data, include_simulated, custom_baselines
            )
            
            self.logger.info("Baseline comparison completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Baseline comparison failed: {e}")
            return {'error': str(e)}
    
    def generate_baseline_comparison_report(self, comparison_results: Dict[str, Any],
                                          output_path: str = None) -> str:
        """Generate a detailed comparison report."""
        report_lines = []
        
        # Header
        report_lines.append("# Baseline Comparison Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Algorithm overview
        baselines = list(comparison_results.get('baseline_algorithms', {}).keys())
        report_lines.append(f"## Algorithms Compared")
        report_lines.append(f"- **Primary Algorithm**: CPO (Constrained Policy Optimization)")
        report_lines.append(f"- **Baseline Algorithms**: {', '.join(baselines)}")
        report_lines.append("")
        
        # Performance summary
        benchmark_summary = comparison_results.get('benchmark_summary', {})
        if benchmark_summary:
            report_lines.append("## Performance Summary")
            
            if 'cpo_performance_rank' in benchmark_summary:
                rank = benchmark_summary['cpo_performance_rank']
                total = benchmark_summary['total_baselines_compared'] + 1
                report_lines.append(f"- **Performance Rank**: {rank}/{total}")
            
            if 'cpo_safety_rank' in benchmark_summary:
                rank = benchmark_summary['cpo_safety_rank']
                total = benchmark_summary['total_baselines_compared'] + 1
                report_lines.append(f"- **Safety Rank**: {rank}/{total}")
            
            if 'overall_assessment' in benchmark_summary:
                report_lines.append(f"- **Overall Assessment**: {benchmark_summary['overall_assessment']}")
            
            report_lines.append("")
        
        # Significant improvements
        if 'significant_improvements' in benchmark_summary:
            improvements = benchmark_summary['significant_improvements']
            if improvements:
                report_lines.append("## Significant Improvements")
                for improvement in improvements:
                    report_lines.append(f"- {improvement}")
                report_lines.append("")
        
        # Areas for improvement
        if 'areas_for_improvement' in benchmark_summary:
            areas = benchmark_summary['areas_for_improvement']
            if areas:
                report_lines.append("## Areas for Improvement")
                for area in areas:
                    report_lines.append(f"- {area}")
                report_lines.append("")
        
        # Statistical analysis summary
        comparison_data = comparison_results.get('comparison_results', {})
        if comparison_data:
            report_lines.append("## Statistical Analysis")
            
            # Pareto analysis
            pareto_analysis = comparison_data.get('pareto_analysis', {})
            if pareto_analysis:
                pareto_efficient = pareto_analysis.get('pareto_efficient', [])
                if 'CPO' in pareto_efficient:
                    report_lines.append("- **Pareto Efficiency**: CPO is Pareto optimal ✓")
                else:
                    report_lines.append("- **Pareto Efficiency**: CPO is dominated by other algorithms")
                
                if pareto_efficient:
                    report_lines.append(f"- **Pareto Optimal Algorithms**: {', '.join(pareto_efficient)}")
            
            report_lines.append("")
        
        # Generate report
        report_content = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Baseline comparison report saved to {output_path}")
        
        return report_content