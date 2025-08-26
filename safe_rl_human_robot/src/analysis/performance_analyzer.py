"""
Performance analysis for Safe RL training results.

This module provides comprehensive performance analysis including learning curves,
sample efficiency, convergence analysis, and statistical significance testing.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import auc
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class PerformanceMetrics:
    """Container for performance analysis results."""
    
    # Learning efficiency
    area_under_curve: float = 0.0
    sample_efficiency_50: int = 0  # Iterations to reach 50% of max performance
    sample_efficiency_90: int = 0  # Iterations to reach 90% of max performance
    
    # Convergence analysis
    converged: bool = False
    convergence_iteration: int = 0
    final_performance: float = 0.0
    performance_stability: float = 0.0
    
    # Statistical measures
    mean_return: float = 0.0
    std_return: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Trend analysis
    learning_rate: float = 0.0  # Slope of performance improvement
    r_squared: float = 0.0      # Fit quality of linear trend
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field, value in self.__dict__.items():
            if isinstance(value, tuple):
                result[field] = list(value)
            else:
                result[field] = value
        return result


class LearningCurveAnalyzer:
    """Analyzes learning curves with confidence intervals."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def analyze_learning_curve(self, data: pd.DataFrame, 
                              value_col: str = 'episode_return',
                              time_col: str = 'iteration') -> Dict[str, Any]:
        """Analyze learning curve with statistical measures."""
        if value_col not in data.columns or time_col not in data.columns:
            raise ValueError(f"Columns {value_col} or {time_col} not found in data")
        
        # Extract and clean data
        df_clean = data[[time_col, value_col]].dropna()
        if len(df_clean) < 3:
            raise ValueError("Insufficient data points for analysis")
        
        x = df_clean[time_col].values
        y = df_clean[value_col].values
        
        # Sort by time
        sort_idx = np.argsort(x)
        x, y = x[sort_idx], y[sort_idx]
        
        # Basic statistics
        analysis = {
            'raw_data': {'x': x.tolist(), 'y': y.tolist()},
            'statistics': self._compute_basic_stats(y),
            'trend_analysis': self._analyze_trend(x, y),
            'smoothed_curve': self._compute_smoothed_curve(x, y),
            'confidence_intervals': self._compute_confidence_intervals(x, y),
            'performance_milestones': self._compute_milestones(x, y)
        }
        
        return analysis
    
    def _compute_basic_stats(self, y: np.ndarray) -> Dict[str, float]:
        """Compute basic statistical measures."""
        return {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'median': float(np.median(y)),
            'q25': float(np.percentile(y, 25)),
            'q75': float(np.percentile(y, 75)),
            'final_value': float(y[-1]),
            'initial_value': float(y[0]),
            'total_improvement': float(y[-1] - y[0])
        }
    
    def _analyze_trend(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Analyze trend using linear regression."""
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_error': float(std_err),
            'significant': float(p_value) < 0.05
        }
    
    def _compute_smoothed_curve(self, x: np.ndarray, y: np.ndarray, 
                               window_size: int = None) -> Dict[str, Any]:
        """Compute smoothed learning curve."""
        if window_size is None:
            window_size = max(3, len(y) // 20)  # 5% of data points
        
        # Moving average
        if len(y) >= window_size:
            smoothed = pd.Series(y).rolling(window=window_size, center=True).mean()
            smoothed_y = smoothed.fillna(method='bfill').fillna(method='ffill').values
        else:
            smoothed_y = y.copy()
        
        # Polynomial fit for extrapolation
        if len(x) >= 3:
            poly_degree = min(3, len(x) - 1)
            poly_coeffs = np.polyfit(x, y, poly_degree)
            poly_fit = np.polyval(poly_coeffs, x)
        else:
            poly_fit = y.copy()
        
        return {
            'x': x.tolist(),
            'moving_average': smoothed_y.tolist(),
            'polynomial_fit': poly_fit.tolist(),
            'window_size': window_size,
            'polynomial_degree': poly_degree if len(x) >= 3 else 1
        }
    
    def _compute_confidence_intervals(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute confidence intervals for learning curve."""
        if len(y) < 3:
            return {'insufficient_data': True}
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        alpha = 1 - self.confidence_level
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(y, size=len(y), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        # Rolling confidence intervals
        window_size = max(5, len(y) // 10)
        rolling_ci_lower = []
        rolling_ci_upper = []
        
        for i in range(len(y)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(y), i + window_size // 2 + 1)
            window_data = y[start_idx:end_idx]
            
            if len(window_data) >= 3:
                sem = stats.sem(window_data)
                ci = stats.t.interval(self.confidence_level, len(window_data) - 1,
                                     loc=np.mean(window_data), scale=sem)
                rolling_ci_lower.append(ci[0])
                rolling_ci_upper.append(ci[1])
            else:
                rolling_ci_lower.append(y[i])
                rolling_ci_upper.append(y[i])
        
        return {
            'overall_ci_lower': float(ci_lower),
            'overall_ci_upper': float(ci_upper),
            'rolling_ci_lower': rolling_ci_lower,
            'rolling_ci_upper': rolling_ci_upper,
            'confidence_level': self.confidence_level
        }
    
    def _compute_milestones(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Compute performance milestones."""
        max_performance = np.max(y)
        min_performance = np.min(y)
        performance_range = max_performance - min_performance
        
        milestones = {}
        thresholds = [0.25, 0.5, 0.75, 0.9, 0.95]
        
        for threshold in thresholds:
            target_performance = min_performance + threshold * performance_range
            reaching_indices = np.where(y >= target_performance)[0]
            
            if len(reaching_indices) > 0:
                milestone_iteration = x[reaching_indices[0]]
                milestones[f'threshold_{int(threshold*100)}%'] = {
                    'iteration': int(milestone_iteration),
                    'performance': float(target_performance),
                    'achieved': True
                }
            else:
                milestones[f'threshold_{int(threshold*100)}%'] = {
                    'achieved': False
                }
        
        return milestones


class SampleEfficiencyAnalyzer:
    """Analyzes sample efficiency across different algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compute_sample_efficiency(self, learning_curves: Dict[str, Dict[str, Any]],
                                 efficiency_thresholds: List[float] = None) -> Dict[str, Any]:
        """Compute sample efficiency metrics for multiple algorithms."""
        if efficiency_thresholds is None:
            efficiency_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        
        efficiency_analysis = {
            'algorithms': list(learning_curves.keys()),
            'thresholds': efficiency_thresholds,
            'efficiency_metrics': {},
            'area_under_curve': {},
            'relative_efficiency': {}
        }
        
        # Compute metrics for each algorithm
        for alg_name, curve_data in learning_curves.items():
            if 'raw_data' not in curve_data:
                continue
            
            x = np.array(curve_data['raw_data']['x'])
            y = np.array(curve_data['raw_data']['y'])
            
            # Sample efficiency for each threshold
            max_perf = np.max(y)
            min_perf = np.min(y)
            perf_range = max_perf - min_perf
            
            efficiency_metrics = {}
            for threshold in efficiency_thresholds:
                target_perf = min_perf + threshold * perf_range
                reaching_idx = np.where(y >= target_perf)[0]
                
                if len(reaching_idx) > 0:
                    efficiency_metrics[f'samples_to_{int(threshold*100)}%'] = int(x[reaching_idx[0]])
                else:
                    efficiency_metrics[f'samples_to_{int(threshold*100)}%'] = None
            
            efficiency_analysis['efficiency_metrics'][alg_name] = efficiency_metrics
            
            # Area under curve (normalized by max x-value for fair comparison)
            if len(x) > 1 and x[-1] > x[0]:
                normalized_auc = auc(x, y) / (x[-1] * max_perf) if max_perf > 0 else 0
                efficiency_analysis['area_under_curve'][alg_name] = float(normalized_auc)
        
        # Compute relative efficiency (compared to best performer)
        efficiency_analysis['relative_efficiency'] = self._compute_relative_efficiency(
            efficiency_analysis['efficiency_metrics']
        )
        
        return efficiency_analysis
    
    def _compute_relative_efficiency(self, efficiency_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compute relative sample efficiency compared to best performer."""
        relative_efficiency = {}
        
        # Find best performance for each threshold
        thresholds = set()
        for metrics in efficiency_metrics.values():
            thresholds.update(metrics.keys())
        
        for threshold in thresholds:
            # Get all sample counts for this threshold
            sample_counts = {}
            for alg_name, metrics in efficiency_metrics.items():
                if threshold in metrics and metrics[threshold] is not None:
                    sample_counts[alg_name] = metrics[threshold]
            
            if not sample_counts:
                continue
            
            # Find minimum (best) sample count
            best_samples = min(sample_counts.values())
            
            # Compute relative efficiency
            for alg_name in efficiency_metrics.keys():
                if alg_name not in relative_efficiency:
                    relative_efficiency[alg_name] = {}
                
                if alg_name in sample_counts:
                    relative_efficiency[alg_name][threshold] = float(best_samples / sample_counts[alg_name])
                else:
                    relative_efficiency[alg_name][threshold] = 0.0
        
        return relative_efficiency


class PerformanceAnalyzer:
    """Main performance analysis class."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.curve_analyzer = LearningCurveAnalyzer(confidence_level)
        self.efficiency_analyzer = SampleEfficiencyAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze_training_performance(self, data: pd.DataFrame,
                                   algorithm_name: str = "CPO") -> Dict[str, Any]:
        """Comprehensive analysis of training performance."""
        performance_analysis = {
            'algorithm': algorithm_name,
            'learning_curve_analysis': {},
            'convergence_analysis': {},
            'stability_analysis': {},
            'efficiency_analysis': {},
            'statistical_summary': {}
        }
        
        try:
            # Learning curve analysis
            if 'episode_return' in data.columns:
                performance_analysis['learning_curve_analysis'] = self.curve_analyzer.analyze_learning_curve(
                    data, 'episode_return', 'iteration'
                )
            
            # Convergence analysis
            performance_analysis['convergence_analysis'] = self._analyze_convergence(data)
            
            # Stability analysis
            performance_analysis['stability_analysis'] = self._analyze_stability(data)
            
            # Efficiency analysis
            performance_analysis['efficiency_analysis'] = self._analyze_efficiency(data)
            
            # Statistical summary
            performance_analysis['statistical_summary'] = self._compute_statistical_summary(data)
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            performance_analysis['error'] = str(e)
        
        return performance_analysis
    
    def compare_algorithms(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Compare performance across multiple algorithms."""
        comparison_analysis = {
            'algorithms': list(algorithms_data.keys()),
            'individual_analyses': {},
            'sample_efficiency_comparison': {},
            'statistical_comparisons': {},
            'performance_rankings': {}
        }
        
        # Analyze each algorithm individually
        learning_curves = {}
        for alg_name, data in algorithms_data.items():
            analysis = self.analyze_training_performance(data, alg_name)
            comparison_analysis['individual_analyses'][alg_name] = analysis
            learning_curves[alg_name] = analysis.get('learning_curve_analysis', {})
        
        # Sample efficiency comparison
        if learning_curves:
            comparison_analysis['sample_efficiency_comparison'] = self.efficiency_analyzer.compute_sample_efficiency(
                learning_curves
            )
        
        # Statistical comparisons
        comparison_analysis['statistical_comparisons'] = self._perform_statistical_comparisons(algorithms_data)
        
        # Performance rankings
        comparison_analysis['performance_rankings'] = self._rank_algorithms(algorithms_data)
        
        return comparison_analysis
    
    def _analyze_convergence(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze convergence properties."""
        if 'episode_return' not in data.columns:
            return {'no_return_data': True}
        
        returns = data['episode_return'].dropna().values
        iterations = data.get('iteration', range(len(returns))).values[:len(returns)]
        
        convergence_analysis = {
            'converged': False,
            'convergence_iteration': None,
            'convergence_value': None,
            'convergence_criterion': None
        }
        
        if len(returns) < 10:
            convergence_analysis['insufficient_data'] = True
            return convergence_analysis
        
        # Method 1: Variance-based convergence
        window_size = max(10, len(returns) // 10)
        convergence_threshold = 0.01  # Relative change threshold
        
        for i in range(window_size, len(returns)):
            recent_window = returns[i-window_size:i]
            recent_mean = np.mean(recent_window)
            recent_std = np.std(recent_window)
            
            # Check if coefficient of variation is small enough
            cv = recent_std / abs(recent_mean) if abs(recent_mean) > 1e-8 else float('inf')
            
            if cv < convergence_threshold:
                convergence_analysis.update({
                    'converged': True,
                    'convergence_iteration': int(iterations[i]),
                    'convergence_value': float(recent_mean),
                    'convergence_criterion': 'variance_based',
                    'coefficient_of_variation': float(cv)
                })
                break
        
        # Method 2: Trend-based convergence
        if not convergence_analysis['converged'] and len(returns) >= 20:
            # Check if slope in recent data is near zero
            recent_fraction = 0.3  # Last 30% of data
            recent_start = int(len(returns) * (1 - recent_fraction))
            recent_returns = returns[recent_start:]
            recent_iterations = iterations[recent_start:len(recent_returns) + recent_start]
            
            if len(recent_returns) >= 3:
                slope, _, r_squared, p_value, _ = stats.linregress(recent_iterations, recent_returns)
                
                # Consider converged if slope is not significantly different from zero
                if p_value > 0.05 or abs(slope) < 0.001:
                    convergence_analysis.update({
                        'converged': True,
                        'convergence_iteration': int(recent_iterations[0]),
                        'convergence_value': float(np.mean(recent_returns)),
                        'convergence_criterion': 'trend_based',
                        'final_slope': float(slope),
                        'slope_p_value': float(p_value)
                    })
        
        return convergence_analysis
    
    def _analyze_stability(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance stability."""
        stability_analysis = {}
        
        if 'episode_return' not in data.columns:
            return {'no_return_data': True}
        
        returns = data['episode_return'].dropna().values
        
        if len(returns) < 10:
            return {'insufficient_data': True}
        
        # Overall stability
        overall_cv = np.std(returns) / abs(np.mean(returns)) if abs(np.mean(returns)) > 1e-8 else float('inf')
        
        # Rolling stability
        window_size = max(10, len(returns) // 5)
        rolling_cv = []
        
        for i in range(window_size, len(returns) + 1):
            window_data = returns[i-window_size:i]
            window_mean = np.mean(window_data)
            window_cv = np.std(window_data) / abs(window_mean) if abs(window_mean) > 1e-8 else float('inf')
            rolling_cv.append(window_cv)
        
        # Stability trend
        if len(rolling_cv) >= 3:
            stability_trend_slope, _, stability_r_squared, stability_p_value, _ = stats.linregress(
                range(len(rolling_cv)), rolling_cv
            )
            
            stability_analysis.update({
                'overall_coefficient_of_variation': float(overall_cv),
                'rolling_cv_mean': float(np.mean(rolling_cv)),
                'rolling_cv_std': float(np.std(rolling_cv)),
                'final_cv': float(rolling_cv[-1]),
                'stability_trend_slope': float(stability_trend_slope),
                'stability_trend_r_squared': float(stability_r_squared),
                'stability_trend_p_value': float(stability_p_value),
                'stability_improving': float(stability_trend_slope) < -0.001 and stability_p_value < 0.05,
                'rolling_cv_values': rolling_cv
            })
        else:
            stability_analysis['overall_coefficient_of_variation'] = float(overall_cv)
        
        return stability_analysis
    
    def _analyze_efficiency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze learning efficiency."""
        efficiency_analysis = {}
        
        if 'episode_return' not in data.columns:
            return {'no_return_data': True}
        
        returns = data['episode_return'].dropna().values
        iterations = data.get('iteration', range(len(returns))).values[:len(returns)]
        
        if len(returns) < 3:
            return {'insufficient_data': True}
        
        # Area under learning curve
        if len(iterations) > 1:
            auc_value = auc(iterations, returns)
            normalized_auc = auc_value / (iterations[-1] * np.max(returns)) if np.max(returns) > 0 else 0
            efficiency_analysis['area_under_curve'] = float(auc_value)
            efficiency_analysis['normalized_auc'] = float(normalized_auc)
        
        # Sample efficiency milestones
        max_return = np.max(returns)
        min_return = np.min(returns)
        return_range = max_return - min_return
        
        milestones = {}
        for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
            target_return = min_return + threshold * return_range
            reaching_indices = np.where(returns >= target_return)[0]
            
            if len(reaching_indices) > 0:
                milestone_iteration = iterations[reaching_indices[0]]
                milestones[f'samples_to_{int(threshold*100)}%'] = int(milestone_iteration)
        
        efficiency_analysis['sample_efficiency_milestones'] = milestones
        
        # Learning rate (improvement per iteration)
        if len(returns) > 1:
            total_improvement = returns[-1] - returns[0]
            total_iterations = iterations[-1] - iterations[0] if len(iterations) > 1 else len(returns)
            learning_rate = total_improvement / total_iterations if total_iterations > 0 else 0
            efficiency_analysis['learning_rate'] = float(learning_rate)
        
        return efficiency_analysis
    
    def _compute_statistical_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistical summary."""
        summary = {}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            col_stats = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'skewness': float(stats.skew(col_data)),
                'kurtosis': float(stats.kurtosis(col_data))
            }
            
            # Confidence interval
            if len(col_data) > 1:
                ci = stats.t.interval(
                    self.confidence_level,
                    len(col_data) - 1,
                    loc=col_data.mean(),
                    scale=stats.sem(col_data)
                )
                col_stats['confidence_interval'] = [float(ci[0]), float(ci[1])]
            
            summary[col] = col_stats
        
        return summary
    
    def _perform_statistical_comparisons(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform statistical comparisons between algorithms."""
        comparisons = {}
        alg_names = list(algorithms_data.keys())
        
        for i, alg1 in enumerate(alg_names):
            for alg2 in alg_names[i+1:]:
                comparison_key = f"{alg1}_vs_{alg2}"
                
                # Compare final performance
                if 'episode_return' in algorithms_data[alg1].columns and 'episode_return' in algorithms_data[alg2].columns:
                    returns1 = algorithms_data[alg1]['episode_return'].dropna().values
                    returns2 = algorithms_data[alg2]['episode_return'].dropna().values
                    
                    if len(returns1) >= 3 and len(returns2) >= 3:
                        # t-test
                        t_stat, t_p_value = stats.ttest_ind(returns1, returns2)
                        
                        # Mann-Whitney U test
                        u_stat, u_p_value = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(returns1) - 1) * np.var(returns1) + 
                                            (len(returns2) - 1) * np.var(returns2)) / 
                                           (len(returns1) + len(returns2) - 2))
                        cohens_d = (np.mean(returns1) - np.mean(returns2)) / pooled_std if pooled_std > 0 else 0
                        
                        comparisons[comparison_key] = {
                            't_test': {'statistic': float(t_stat), 'p_value': float(t_p_value)},
                            'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p_value)},
                            'effect_size': float(cohens_d),
                            'significant': float(min(t_p_value, u_p_value)) < 0.05,
                            'mean_difference': float(np.mean(returns1) - np.mean(returns2))
                        }
        
        return comparisons
    
    def _rank_algorithms(self, algorithms_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Rank algorithms by various performance metrics."""
        rankings = {}
        
        # Rank by final performance
        final_performances = {}
        for alg_name, data in algorithms_data.items():
            if 'episode_return' in data.columns:
                returns = data['episode_return'].dropna().values
                if len(returns) > 0:
                    final_performances[alg_name] = np.mean(returns[-10:])  # Average of last 10
        
        if final_performances:
            sorted_algs = sorted(final_performances.items(), key=lambda x: x[1], reverse=True)
            rankings['final_performance'] = [alg for alg, _ in sorted_algs]
        
        # Rank by sample efficiency (time to 90% of max performance)
        sample_efficiency = {}
        for alg_name, data in algorithms_data.items():
            if 'episode_return' in data.columns:
                returns = data['episode_return'].dropna().values
                iterations = data.get('iteration', range(len(returns))).values[:len(returns)]
                
                if len(returns) > 10:
                    max_return = np.max(returns)
                    target_return = 0.9 * max_return
                    reaching_indices = np.where(returns >= target_return)[0]
                    
                    if len(reaching_indices) > 0:
                        sample_efficiency[alg_name] = iterations[reaching_indices[0]]
        
        if sample_efficiency:
            sorted_algs = sorted(sample_efficiency.items(), key=lambda x: x[1])  # Lower is better
            rankings['sample_efficiency'] = [alg for alg, _ in sorted_algs]
        
        return rankings