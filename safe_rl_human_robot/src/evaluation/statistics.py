"""
Statistical Analysis Tools for Safe RL Benchmarking.

This module provides comprehensive statistical analysis including hypothesis testing,
effect size calculations, confidence intervals, and multiple comparison corrections.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.stats import mannwhitneyu, friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.contingency_tables import mcnemar
import warnings

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTestResult:
    """Result of a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    power: float
    alpha: float
    interpretation: str


@dataclass
class ComparisonResult:
    """Result of algorithm comparison."""
    algorithm_a: str
    algorithm_b: str
    metric: str
    test_result: HypothesisTestResult
    practical_significance: bool
    recommendation: str


class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    @staticmethod
    def glass_delta(group1: List[float], group2: List[float]) -> float:
        """Calculate Glass's delta effect size."""
        if len(group2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std2 = np.std(group2, ddof=1)
        
        if std2 == 0:
            return 0.0
        
        return (mean1 - mean2) / std2
    
    @staticmethod
    def hedges_g(group1: List[float], group2: List[float]) -> float:
        """Calculate Hedges' g effect size (bias-corrected Cohen's d)."""
        cohens_d = EffectSizeCalculator.cohens_d(group1, group2)
        n1, n2 = len(group1), len(group2)
        
        # Bias correction factor
        j = 1 - (3 / (4 * (n1 + n2 - 2) - 1))
        
        return cohens_d * j
    
    @staticmethod
    def rank_biserial_correlation(group1: List[float], group2: List[float]) -> float:
        """Calculate rank-biserial correlation (for Mann-Whitney U test)."""
        if not group1 or not group2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        
        try:
            u_statistic, _ = mannwhitneyu(group1, group2, alternative='two-sided')
            # Convert to rank-biserial correlation
            r = 1 - (2 * u_statistic) / (n1 * n2)
            return r
        except Exception:
            return 0.0
    
    @staticmethod
    def common_language_effect_size(group1: List[float], group2: List[float]) -> float:
        """Calculate Common Language Effect Size (probability of superiority)."""
        if not group1 or not group2:
            return 0.5
        
        count = 0
        total = 0
        
        for val1 in group1:
            for val2 in group2:
                if val1 > val2:
                    count += 1
                total += 1
        
        return count / total if total > 0 else 0.5
    
    @staticmethod
    def interpret_effect_size(effect_size: float, measure: str = 'cohens_d') -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        
        if measure.lower() in ['cohens_d', 'hedges_g']:
            if abs_effect < 0.2:
                return "negligible"
            elif abs_effect < 0.5:
                return "small"
            elif abs_effect < 0.8:
                return "medium"
            else:
                return "large"
        
        elif measure.lower() == 'rank_biserial':
            if abs_effect < 0.1:
                return "negligible"
            elif abs_effect < 0.3:
                return "small"
            elif abs_effect < 0.5:
                return "medium"
            else:
                return "large"
        
        elif measure.lower() == 'common_language':
            # CLES is probability, so different interpretation
            if abs_effect - 0.5 < 0.06:  # Close to 0.5
                return "negligible"
            elif abs_effect - 0.5 < 0.14:
                return "small"
            elif abs_effect - 0.5 < 0.21:
                return "medium"
            else:
                return "large"
        
        return "unknown"


class ConfidenceIntervalEstimator:
    """Estimate confidence intervals using various methods."""
    
    @staticmethod
    def bootstrap_ci(data: List[float], 
                    confidence_level: float = 0.95,
                    n_bootstrap: int = 10000,
                    statistic_func: callable = np.mean) -> Tuple[float, float]:
        """Bootstrap confidence interval."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        data = np.array(data)
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return (lower, upper)
    
    @staticmethod
    def t_confidence_interval(data: List[float], 
                            confidence_level: float = 0.95) -> Tuple[float, float]:
        """Student's t confidence interval for mean."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        data = np.array(data)
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        
        alpha = 1 - confidence_level
        df = len(data) - 1
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        margin = t_critical * sem
        return (mean - margin, mean + margin)
    
    @staticmethod
    def wilcoxon_ci(data: List[float], 
                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """Wilcoxon confidence interval for median."""
        if len(data) < 6:  # Minimum sample size for Wilcoxon
            return (0.0, 0.0)
        
        data = np.array(data)
        n = len(data)
        alpha = 1 - confidence_level
        
        # Calculate critical value for Wilcoxon signed-rank test
        try:
            # Use normal approximation for large samples
            if n > 20:
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                w_critical = n * (n + 1) / 4 - z_alpha * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
            else:
                # For small samples, use exact critical values (simplified)
                w_critical = stats.wilcoxon(data - np.median(data)).statistic
            
            # Compute confidence interval bounds
            sorted_data = np.sort(data)
            k = int(w_critical)
            
            if k >= 0 and k < len(sorted_data):
                lower = sorted_data[k]
                upper = sorted_data[-(k+1)] if k < len(sorted_data) - 1 else sorted_data[-1]
            else:
                lower = np.min(data)
                upper = np.max(data)
            
            return (lower, upper)
        
        except Exception:
            # Fallback to bootstrap
            return ConfidenceIntervalEstimator.bootstrap_ci(
                data, confidence_level, statistic_func=np.median
            )


class HypothesisTest:
    """Perform various hypothesis tests."""
    
    @staticmethod
    def mann_whitney_u(group1: List[float], 
                      group2: List[float],
                      alpha: float = 0.05,
                      alternative: str = 'two-sided') -> HypothesisTestResult:
        """Mann-Whitney U test (non-parametric)."""
        if len(group1) < 3 or len(group2) < 3:
            return HypothesisTestResult(
                test_name="Mann-Whitney U",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                power=0.0,
                alpha=alpha,
                interpretation="Insufficient data"
            )
        
        try:
            statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
            
            # Calculate effect size (rank-biserial correlation)
            effect_size = EffectSizeCalculator.rank_biserial_correlation(group1, group2)
            
            # Bootstrap confidence interval for effect size
            ci = ConfidenceIntervalEstimator.bootstrap_ci(
                group1 + group2, 
                confidence_level=1-alpha,
                statistic_func=lambda x: EffectSizeCalculator.rank_biserial_correlation(
                    x[:len(group1)], x[len(group1):]
                )
            )
            
            # Power calculation (approximate)
            power = HypothesisTest._calculate_power_mann_whitney(
                len(group1), len(group2), effect_size, alpha
            )
            
            significant = p_value < alpha
            
            interpretation = f"Effect size is {EffectSizeCalculator.interpret_effect_size(effect_size, 'rank_biserial')}"
            if significant:
                interpretation += f", statistically significant (p={p_value:.4f})"
            else:
                interpretation += f", not statistically significant (p={p_value:.4f})"
            
            return HypothesisTestResult(
                test_name="Mann-Whitney U",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=effect_size,
                confidence_interval=ci,
                significant=significant,
                power=power,
                alpha=alpha,
                interpretation=interpretation
            )
        
        except Exception as e:
            logger.warning(f"Mann-Whitney U test failed: {e}")
            return HypothesisTestResult(
                test_name="Mann-Whitney U",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                power=0.0,
                alpha=alpha,
                interpretation="Test failed"
            )
    
    @staticmethod
    def welch_t_test(group1: List[float], 
                    group2: List[float],
                    alpha: float = 0.05,
                    alternative: str = 'two-sided') -> HypothesisTestResult:
        """Welch's t-test (assumes unequal variances)."""
        if len(group1) < 2 or len(group2) < 2:
            return HypothesisTestResult(
                test_name="Welch's t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                power=0.0,
                alpha=alpha,
                interpretation="Insufficient data"
            )
        
        try:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False, 
                                               alternative=alternative)
            
            # Calculate effect size (Cohen's d)
            effect_size = EffectSizeCalculator.cohens_d(group1, group2)
            
            # Confidence interval for Cohen's d
            ci = ConfidenceIntervalEstimator.bootstrap_ci(
                group1 + group2,
                confidence_level=1-alpha,
                statistic_func=lambda x: EffectSizeCalculator.cohens_d(
                    x[:len(group1)], x[len(group1):]
                )
            )
            
            # Power calculation
            power = HypothesisTest._calculate_power_t_test(
                len(group1), len(group2), effect_size, alpha
            )
            
            significant = p_value < alpha
            
            interpretation = f"Effect size is {EffectSizeCalculator.interpret_effect_size(effect_size, 'cohens_d')}"
            if significant:
                interpretation += f", statistically significant (p={p_value:.4f})"
            else:
                interpretation += f", not statistically significant (p={p_value:.4f})"
            
            return HypothesisTestResult(
                test_name="Welch's t-test",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=effect_size,
                confidence_interval=ci,
                significant=significant,
                power=power,
                alpha=alpha,
                interpretation=interpretation
            )
        
        except Exception as e:
            logger.warning(f"Welch's t-test failed: {e}")
            return HypothesisTestResult(
                test_name="Welch's t-test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                power=0.0,
                alpha=alpha,
                interpretation="Test failed"
            )
    
    @staticmethod
    def friedman_test(groups: Dict[str, List[float]],
                     alpha: float = 0.05) -> HypothesisTestResult:
        """Friedman test for multiple related groups."""
        if len(groups) < 3:
            return HypothesisTestResult(
                test_name="Friedman test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                power=0.0,
                alpha=alpha,
                interpretation="Need at least 3 groups"
            )
        
        try:
            # Ensure all groups have same length
            group_values = list(groups.values())
            min_length = min(len(group) for group in group_values)
            
            if min_length < 2:
                return HypothesisTestResult(
                    test_name="Friedman test",
                    statistic=0.0,
                    p_value=1.0,
                    effect_size=0.0,
                    confidence_interval=(0.0, 0.0),
                    significant=False,
                    power=0.0,
                    alpha=alpha,
                    interpretation="Insufficient data in groups"
                )
            
            # Truncate all groups to same length
            truncated_groups = [group[:min_length] for group in group_values]
            
            statistic, p_value = friedmanchisquare(*truncated_groups)
            
            # Effect size: Kendall's W (coefficient of concordance)
            effect_size = HypothesisTest._calculate_kendalls_w(truncated_groups)
            
            significant = p_value < alpha
            
            interpretation = f"Effect size (Kendall's W) = {effect_size:.3f}"
            if significant:
                interpretation += f", statistically significant (p={p_value:.4f})"
            else:
                interpretation += f", not statistically significant (p={p_value:.4f})"
            
            return HypothesisTestResult(
                test_name="Friedman test",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=effect_size,
                confidence_interval=(0.0, 0.0),  # Not applicable
                significant=significant,
                power=0.0,  # Complex to calculate
                alpha=alpha,
                interpretation=interpretation
            )
        
        except Exception as e:
            logger.warning(f"Friedman test failed: {e}")
            return HypothesisTestResult(
                test_name="Friedman test",
                statistic=0.0,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                significant=False,
                power=0.0,
                alpha=alpha,
                interpretation="Test failed"
            )
    
    @staticmethod
    def _calculate_power_mann_whitney(n1: int, n2: int, effect_size: float, alpha: float) -> float:
        """Approximate power calculation for Mann-Whitney U test."""
        # Simplified power calculation based on normal approximation
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = abs(effect_size) * np.sqrt(n_harmonic / 3) - z_alpha
        power = stats.norm.cdf(z_beta)
        return max(0.0, min(1.0, power))
    
    @staticmethod
    def _calculate_power_t_test(n1: int, n2: int, effect_size: float, alpha: float) -> float:
        """Approximate power calculation for t-test."""
        n_harmonic = 2 * n1 * n2 / (n1 + n2)
        ncp = abs(effect_size) * np.sqrt(n_harmonic / 2)  # Non-centrality parameter
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        # Power = 1 - beta = P(|t| > t_critical | H1 is true)
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
        return max(0.0, min(1.0, power))
    
    @staticmethod
    def _calculate_kendalls_w(groups: List[List[float]]) -> float:
        """Calculate Kendall's W (coefficient of concordance)."""
        try:
            k = len(groups)  # Number of groups
            n = len(groups[0])  # Number of subjects
            
            # Create rank matrix
            rank_matrix = np.zeros((n, k))
            for j, group in enumerate(groups):
                rank_matrix[:, j] = stats.rankdata(group)
            
            # Sum of ranks for each subject
            rank_sums = np.sum(rank_matrix, axis=1)
            
            # Calculate W
            s = np.sum((rank_sums - np.mean(rank_sums)) ** 2)
            w = 12 * s / (k ** 2 * (n ** 3 - n))
            
            return max(0.0, min(1.0, w))
        
        except Exception:
            return 0.0


class StatisticalAnalyzer:
    """Main class for comprehensive statistical analysis."""
    
    def __init__(self):
        self.effect_size_calc = EffectSizeCalculator()
        self.ci_estimator = ConfidenceIntervalEstimator()
        self.hypothesis_test = HypothesisTest()
    
    def perform_comparative_analysis(self, 
                                   algorithm_results: Dict[str, List],
                                   significance_level: float = 0.05,
                                   correction_method: str = 'holm') -> Dict[str, Any]:
        """Perform comprehensive comparative analysis."""
        
        # Extract metric data
        metric_data = self._extract_metric_data(algorithm_results)
        
        # Perform pairwise comparisons
        pairwise_results = self._perform_pairwise_comparisons(
            metric_data, significance_level
        )
        
        # Apply multiple comparison correction
        corrected_results = self._apply_multiple_comparison_correction(
            pairwise_results, correction_method, significance_level
        )
        
        # Perform omnibus tests
        omnibus_results = self._perform_omnibus_tests(
            metric_data, significance_level
        )
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(metric_data)
        
        # Generate practical significance assessments
        practical_significance = self._assess_practical_significance(
            corrected_results, significance_level
        )
        
        return {
            'pairwise_comparisons': corrected_results,
            'omnibus_tests': omnibus_results,
            'summary_statistics': summary_stats,
            'practical_significance': practical_significance,
            'correction_method': correction_method,
            'alpha': significance_level
        }
    
    def _extract_metric_data(self, algorithm_results: Dict[str, List]) -> Dict[str, Dict[str, List[float]]]:
        """Extract metric data organized by metric and algorithm."""
        metric_data = {}
        
        # Get all available metrics from first algorithm
        if algorithm_results:
            first_alg = next(iter(algorithm_results.keys()))
            if algorithm_results[first_alg]:
                first_result = algorithm_results[first_alg][0]
                if hasattr(first_result, 'metrics'):
                    metric_names = list(first_result.metrics.to_dict().keys())
                else:
                    return metric_data
            else:
                return metric_data
        else:
            return metric_data
        
        # Extract data for each metric
        for metric_name in metric_names:
            metric_data[metric_name] = {}
            
            for alg_name, results in algorithm_results.items():
                values = []
                for result in results:
                    if hasattr(result, 'metrics'):
                        metric_dict = result.metrics.to_dict()
                        if metric_name in metric_dict:
                            values.append(metric_dict[metric_name])
                
                if values:  # Only include if we have data
                    metric_data[metric_name][alg_name] = values
        
        return metric_data
    
    def _perform_pairwise_comparisons(self, 
                                    metric_data: Dict[str, Dict[str, List[float]]], 
                                    alpha: float) -> Dict[str, List[ComparisonResult]]:
        """Perform all pairwise comparisons."""
        pairwise_results = {}
        
        for metric_name, alg_data in metric_data.items():
            pairwise_results[metric_name] = []
            algorithms = list(alg_data.keys())
            
            # Perform all pairwise comparisons
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    alg_a, alg_b = algorithms[i], algorithms[j]
                    data_a, data_b = alg_data[alg_a], alg_data[alg_b]
                    
                    # Choose appropriate test based on data characteristics
                    test_result = self._choose_and_run_test(data_a, data_b, alpha)
                    
                    # Assess practical significance
                    practical_sig = self._is_practically_significant(
                        test_result.effect_size, metric_name
                    )
                    
                    # Generate recommendation
                    recommendation = self._generate_recommendation(
                        test_result, practical_sig, alg_a, alg_b
                    )
                    
                    comparison_result = ComparisonResult(
                        algorithm_a=alg_a,
                        algorithm_b=alg_b,
                        metric=metric_name,
                        test_result=test_result,
                        practical_significance=practical_sig,
                        recommendation=recommendation
                    )
                    
                    pairwise_results[metric_name].append(comparison_result)
        
        return pairwise_results
    
    def _choose_and_run_test(self, data_a: List[float], data_b: List[float], alpha: float) -> HypothesisTestResult:
        """Choose and run appropriate statistical test."""
        # Check sample sizes
        if len(data_a) < 10 or len(data_b) < 10:
            # Small samples: use non-parametric test
            return self.hypothesis_test.mann_whitney_u(data_a, data_b, alpha)
        
        # Check normality (simplified check using skewness and kurtosis)
        try:
            skew_a = stats.skew(data_a)
            skew_b = stats.skew(data_b)
            kurt_a = stats.kurtosis(data_a)
            kurt_b = stats.kurtosis(data_b)
            
            # If highly skewed or extreme kurtosis, use non-parametric
            if abs(skew_a) > 2 or abs(skew_b) > 2 or abs(kurt_a) > 7 or abs(kurt_b) > 7:
                return self.hypothesis_test.mann_whitney_u(data_a, data_b, alpha)
        except Exception:
            pass
        
        # Check equal variances
        try:
            _, levene_p = stats.levene(data_a, data_b)
            if levene_p < 0.05:  # Unequal variances
                return self.hypothesis_test.welch_t_test(data_a, data_b, alpha)
            else:
                # Equal variances, use standard t-test
                return self.hypothesis_test.welch_t_test(data_a, data_b, alpha)  # Welch is more robust
        except Exception:
            # Fallback to Mann-Whitney
            return self.hypothesis_test.mann_whitney_u(data_a, data_b, alpha)
    
    def _apply_multiple_comparison_correction(self, 
                                           pairwise_results: Dict[str, List[ComparisonResult]],
                                           method: str,
                                           alpha: float) -> Dict[str, List[ComparisonResult]]:
        """Apply multiple comparison correction."""
        corrected_results = {}
        
        for metric_name, comparisons in pairwise_results.items():
            # Extract p-values
            p_values = [comp.test_result.p_value for comp in comparisons]
            
            if not p_values:
                corrected_results[metric_name] = comparisons
                continue
            
            try:
                # Apply correction
                rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                    p_values, alpha=alpha, method=method
                )
                
                # Update comparison results
                corrected_comparisons = []
                for i, comparison in enumerate(comparisons):
                    # Create new test result with corrected p-value
                    corrected_test = HypothesisTestResult(
                        test_name=comparison.test_result.test_name + f" ({method} corrected)",
                        statistic=comparison.test_result.statistic,
                        p_value=p_corrected[i],
                        effect_size=comparison.test_result.effect_size,
                        confidence_interval=comparison.test_result.confidence_interval,
                        significant=rejected[i],
                        power=comparison.test_result.power,
                        alpha=alpha,
                        interpretation=comparison.test_result.interpretation
                    )
                    
                    corrected_comparison = ComparisonResult(
                        algorithm_a=comparison.algorithm_a,
                        algorithm_b=comparison.algorithm_b,
                        metric=comparison.metric,
                        test_result=corrected_test,
                        practical_significance=comparison.practical_significance,
                        recommendation=self._generate_recommendation(
                            corrected_test, comparison.practical_significance,
                            comparison.algorithm_a, comparison.algorithm_b
                        )
                    )
                    corrected_comparisons.append(corrected_comparison)
                
                corrected_results[metric_name] = corrected_comparisons
            
            except Exception as e:
                logger.warning(f"Multiple comparison correction failed for {metric_name}: {e}")
                corrected_results[metric_name] = comparisons
        
        return corrected_results
    
    def _perform_omnibus_tests(self, 
                             metric_data: Dict[str, Dict[str, List[float]]], 
                             alpha: float) -> Dict[str, HypothesisTestResult]:
        """Perform omnibus tests for overall differences."""
        omnibus_results = {}
        
        for metric_name, alg_data in metric_data.items():
            if len(alg_data) >= 3:  # Need at least 3 groups for omnibus test
                omnibus_results[metric_name] = self.hypothesis_test.friedman_test(
                    alg_data, alpha
                )
        
        return omnibus_results
    
    def _calculate_summary_statistics(self, 
                                    metric_data: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
        """Calculate summary statistics for each metric and algorithm."""
        summary_stats = {}
        
        for metric_name, alg_data in metric_data.items():
            summary_stats[metric_name] = {}
            
            for alg_name, values in alg_data.items():
                if values:
                    stats_dict = {
                        'mean': np.mean(values),
                        'std': np.std(values, ddof=1),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'q25': np.percentile(values, 25),
                        'q75': np.percentile(values, 75),
                        'count': len(values),
                        'skewness': stats.skew(values),
                        'kurtosis': stats.kurtosis(values),
                        'confidence_interval': self.ci_estimator.t_confidence_interval(values)
                    }
                    summary_stats[metric_name][alg_name] = stats_dict
        
        return summary_stats
    
    def _assess_practical_significance(self, 
                                     pairwise_results: Dict[str, List[ComparisonResult]], 
                                     alpha: float) -> Dict[str, Any]:
        """Assess practical significance across all comparisons."""
        practical_sig_summary = {
            'metrics_with_practical_differences': [],
            'effect_size_distributions': {},
            'recommendations_summary': {}
        }
        
        for metric_name, comparisons in pairwise_results.items():
            effect_sizes = [comp.test_result.effect_size for comp in comparisons]
            practical_comparisons = [comp for comp in comparisons if comp.practical_significance]
            
            if practical_comparisons:
                practical_sig_summary['metrics_with_practical_differences'].append(metric_name)
            
            practical_sig_summary['effect_size_distributions'][metric_name] = {
                'mean_effect_size': np.mean(np.abs(effect_sizes)) if effect_sizes else 0.0,
                'max_effect_size': np.max(np.abs(effect_sizes)) if effect_sizes else 0.0,
                'practical_comparisons': len(practical_comparisons),
                'total_comparisons': len(comparisons)
            }
        
        return practical_sig_summary
    
    def _is_practically_significant(self, effect_size: float, metric_name: str) -> bool:
        """Determine if effect size represents practical significance."""
        abs_effect = abs(effect_size)
        
        # Different thresholds for different types of metrics
        if 'safety' in metric_name.lower() or 'violation' in metric_name.lower():
            return abs_effect > 0.3  # Lower threshold for safety metrics
        elif 'performance' in metric_name.lower() or 'reward' in metric_name.lower():
            return abs_effect > 0.5  # Standard Cohen's medium effect
        elif 'efficiency' in metric_name.lower() or 'time' in metric_name.lower():
            return abs_effect > 0.5
        else:
            return abs_effect > 0.5  # Default threshold
    
    def _generate_recommendation(self, 
                               test_result: HypothesisTestResult,
                               practical_significance: bool, 
                               alg_a: str, 
                               alg_b: str) -> str:
        """Generate recommendation based on statistical and practical significance."""
        if test_result.significant and practical_significance:
            if test_result.effect_size > 0:
                return f"Recommend {alg_a} over {alg_b}: statistically and practically significant advantage"
            else:
                return f"Recommend {alg_b} over {alg_a}: statistically and practically significant advantage"
        elif test_result.significant and not practical_significance:
            return f"Statistically significant difference between {alg_a} and {alg_b}, but effect size is small"
        elif not test_result.significant and practical_significance:
            return f"Large effect size between {alg_a} and {alg_b}, but not statistically significant (may need more data)"
        else:
            return f"No meaningful difference between {alg_a} and {alg_b}"