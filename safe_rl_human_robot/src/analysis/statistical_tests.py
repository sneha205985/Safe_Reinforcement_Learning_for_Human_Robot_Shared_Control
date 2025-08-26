"""
Statistical testing framework for Safe RL analysis.

This module provides comprehensive statistical testing capabilities including
Mann-Whitney U tests, Kolmogorov-Smirnov tests, bootstrap confidence intervals,
and multiple comparison corrections for Safe RL performance analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import warnings
from itertools import combinations


@dataclass
class TestResult:
    """Container for statistical test results."""
    
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: str = ""
    significant: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'critical_value': self.critical_value,
            'effect_size': self.effect_size,
            'confidence_interval': list(self.confidence_interval) if self.confidence_interval else None,
            'interpretation': self.interpretation,
            'significant': self.significant
        }


@dataclass
class ComparisonResult:
    """Container for algorithm comparison results."""
    
    algorithm1: str
    algorithm2: str
    metric: str
    tests: List[TestResult]
    winner: Optional[str] = None
    confidence: float = 0.0
    practical_significance: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm1': self.algorithm1,
            'algorithm2': self.algorithm2,
            'metric': self.metric,
            'tests': [test.to_dict() for test in self.tests],
            'winner': self.winner,
            'confidence': self.confidence,
            'practical_significance': self.practical_significance
        }


class StatisticalTester:
    """Main class for statistical testing."""
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)
    
    def mann_whitney_u_test(self, sample1: np.ndarray, sample2: np.ndarray,
                           alternative: str = 'two-sided') -> TestResult:
        """Perform Mann-Whitney U test (non-parametric comparison)."""
        if len(sample1) < 3 or len(sample2) < 3:
            return TestResult(
                test_name="Mann-Whitney U",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient sample size"
            )
        
        try:
            statistic, p_value = stats.mannwhitneyu(
                sample1, sample2, alternative=alternative
            )
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(sample1), len(sample2)
            u1 = statistic
            u2 = n1 * n2 - u1
            
            # Rank-biserial correlation
            r = 1 - (2 * min(u1, u2)) / (n1 * n2)
            
            # Interpretation
            if alternative == 'two-sided':
                if p_value < self.alpha:
                    if np.median(sample1) > np.median(sample2):
                        interpretation = f"Sample 1 significantly higher (p={p_value:.4f})"
                    else:
                        interpretation = f"Sample 2 significantly higher (p={p_value:.4f})"
                else:
                    interpretation = f"No significant difference (p={p_value:.4f})"
            else:
                interpretation = f"p-value: {p_value:.4f}"
            
            return TestResult(
                test_name="Mann-Whitney U",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(r),
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"Mann-Whitney U test failed: {e}")
            return TestResult(
                test_name="Mann-Whitney U",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
    
    def kolmogorov_smirnov_test(self, sample1: np.ndarray, sample2: np.ndarray) -> TestResult:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        if len(sample1) < 3 or len(sample2) < 3:
            return TestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient sample size"
            )
        
        try:
            statistic, p_value = stats.ks_2samp(sample1, sample2)
            
            # Calculate effect size (D statistic is already a measure of effect size)
            effect_size = statistic
            
            # Interpretation
            if p_value < self.alpha:
                interpretation = f"Distributions significantly different (D={statistic:.4f}, p={p_value:.4f})"
            else:
                interpretation = f"Distributions not significantly different (D={statistic:.4f}, p={p_value:.4f})"
            
            return TestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"Kolmogorov-Smirnov test failed: {e}")
            return TestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
    
    def t_test(self, sample1: np.ndarray, sample2: np.ndarray,
              equal_var: bool = True) -> TestResult:
        """Perform independent samples t-test."""
        if len(sample1) < 3 or len(sample2) < 3:
            return TestResult(
                test_name="Independent t-test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient sample size"
            )
        
        try:
            statistic, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
            
            # Calculate Cohen's d
            n1, n2 = len(sample1), len(sample2)
            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            
            if equal_var:
                # Pooled standard deviation
                pooled_std = np.sqrt(((n1 - 1) * np.var(sample1, ddof=1) + 
                                    (n2 - 1) * np.var(sample2, ddof=1)) / (n1 + n2 - 2))
            else:
                # Use average of standard deviations
                pooled_std = np.sqrt((np.var(sample1, ddof=1) + np.var(sample2, ddof=1)) / 2)
            
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for mean difference
            sem = pooled_std * np.sqrt(1/n1 + 1/n2)
            df = n1 + n2 - 2
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
            margin_error = t_critical * sem
            mean_diff = mean1 - mean2
            ci = (mean_diff - margin_error, mean_diff + margin_error)
            
            # Interpretation
            test_name = "Welch's t-test" if not equal_var else "Student's t-test"
            if p_value < self.alpha:
                if mean1 > mean2:
                    interpretation = f"Sample 1 significantly higher (t={statistic:.4f}, p={p_value:.4f})"
                else:
                    interpretation = f"Sample 2 significantly higher (t={statistic:.4f}, p={p_value:.4f})"
            else:
                interpretation = f"No significant difference (t={statistic:.4f}, p={p_value:.4f})"
            
            return TestResult(
                test_name=test_name,
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(cohens_d),
                confidence_interval=ci,
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"t-test failed: {e}")
            return TestResult(
                test_name="Independent t-test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
    
    def bootstrap_test(self, sample1: np.ndarray, sample2: np.ndarray,
                      n_bootstrap: int = 10000,
                      statistic_func: callable = np.mean) -> TestResult:
        """Perform bootstrap hypothesis test."""
        if len(sample1) < 3 or len(sample2) < 3:
            return TestResult(
                test_name="Bootstrap test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient sample size"
            )
        
        try:
            # Observed difference
            observed_diff = statistic_func(sample1) - statistic_func(sample2)
            
            # Combined sample for null hypothesis
            combined = np.concatenate([sample1, sample2])
            n1, n2 = len(sample1), len(sample2)
            
            # Bootstrap under null hypothesis
            bootstrap_diffs = []
            np.random.seed(42)  # For reproducibility
            
            for _ in range(n_bootstrap):
                # Resample from combined distribution
                resampled = np.random.choice(combined, size=n1 + n2, replace=True)
                boot_sample1 = resampled[:n1]
                boot_sample2 = resampled[n1:]
                
                boot_diff = statistic_func(boot_sample1) - statistic_func(boot_sample2)
                bootstrap_diffs.append(boot_diff)
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            
            # Calculate p-value (two-tailed)
            p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
            
            # Confidence interval for the difference
            alpha_level = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_diffs, 100 * alpha_level / 2)
            ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha_level / 2))
            
            # Effect size (standardized difference)
            effect_size = observed_diff / np.std(combined) if np.std(combined) > 0 else 0
            
            # Interpretation
            if p_value < self.alpha:
                if observed_diff > 0:
                    interpretation = f"Sample 1 significantly higher (bootstrap p={p_value:.4f})"
                else:
                    interpretation = f"Sample 2 significantly higher (bootstrap p={p_value:.4f})"
            else:
                interpretation = f"No significant difference (bootstrap p={p_value:.4f})"
            
            return TestResult(
                test_name="Bootstrap test",
                statistic=float(observed_diff),
                p_value=float(p_value),
                effect_size=float(effect_size),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"Bootstrap test failed: {e}")
            return TestResult(
                test_name="Bootstrap test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
    
    def permutation_test(self, sample1: np.ndarray, sample2: np.ndarray,
                        n_permutations: int = 10000,
                        statistic_func: callable = None) -> TestResult:
        """Perform permutation test."""
        if statistic_func is None:
            statistic_func = lambda x1, x2: np.mean(x1) - np.mean(x2)
        
        if len(sample1) < 3 or len(sample2) < 3:
            return TestResult(
                test_name="Permutation test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient sample size"
            )
        
        try:
            # Observed test statistic
            observed_statistic = statistic_func(sample1, sample2)
            
            # Combined data
            combined = np.concatenate([sample1, sample2])
            n1 = len(sample1)
            
            # Permutation distribution
            permutation_statistics = []
            np.random.seed(42)  # For reproducibility
            
            for _ in range(n_permutations):
                # Randomly permute the combined data
                permuted = np.random.permutation(combined)
                perm_sample1 = permuted[:n1]
                perm_sample2 = permuted[n1:]
                
                perm_statistic = statistic_func(perm_sample1, perm_sample2)
                permutation_statistics.append(perm_statistic)
            
            permutation_statistics = np.array(permutation_statistics)
            
            # Calculate p-value (two-tailed)
            p_value = np.mean(np.abs(permutation_statistics) >= np.abs(observed_statistic))
            
            # Effect size
            effect_size = observed_statistic / np.std(permutation_statistics) if np.std(permutation_statistics) > 0 else 0
            
            # Interpretation
            if p_value < self.alpha:
                interpretation = f"Significant difference (permutation p={p_value:.4f})"
            else:
                interpretation = f"No significant difference (permutation p={p_value:.4f})"
            
            return TestResult(
                test_name="Permutation test",
                statistic=float(observed_statistic),
                p_value=float(p_value),
                effect_size=float(effect_size),
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"Permutation test failed: {e}")
            return TestResult(
                test_name="Permutation test",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
    
    def wilcoxon_signed_rank_test(self, sample1: np.ndarray, 
                                 sample2: np.ndarray) -> TestResult:
        """Perform Wilcoxon signed-rank test for paired samples."""
        if len(sample1) != len(sample2):
            return TestResult(
                test_name="Wilcoxon signed-rank",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Samples must have equal length for paired test"
            )
        
        if len(sample1) < 6:
            return TestResult(
                test_name="Wilcoxon signed-rank",
                statistic=np.nan,
                p_value=np.nan,
                interpretation="Insufficient sample size (need >= 6)"
            )
        
        try:
            statistic, p_value = stats.wilcoxon(sample1, sample2, alternative='two-sided')
            
            # Effect size (rank-biserial correlation for paired samples)
            differences = sample1 - sample2
            n = len(differences)
            
            # Count positive and negative differences
            n_pos = np.sum(differences > 0)
            n_neg = np.sum(differences < 0)
            
            # Rank-biserial correlation
            r = (n_pos - n_neg) / n
            
            # Interpretation
            if p_value < self.alpha:
                median_diff = np.median(differences)
                if median_diff > 0:
                    interpretation = f"Sample 1 significantly higher (W={statistic}, p={p_value:.4f})"
                else:
                    interpretation = f"Sample 2 significantly higher (W={statistic}, p={p_value:.4f})"
            else:
                interpretation = f"No significant difference (W={statistic}, p={p_value:.4f})"
            
            return TestResult(
                test_name="Wilcoxon signed-rank",
                statistic=float(statistic),
                p_value=float(p_value),
                effect_size=float(r),
                interpretation=interpretation,
                significant=p_value < self.alpha
            )
            
        except Exception as e:
            self.logger.error(f"Wilcoxon signed-rank test failed: {e}")
            return TestResult(
                test_name="Wilcoxon signed-rank",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )


class MultipleComparisonCorrection:
    """Handles multiple comparison corrections."""
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple comparisons."""
        n_comparisons = len(p_values)
        corrected_alpha = alpha / n_comparisons
        
        adjusted_p_values = [min(p * n_comparisons, 1.0) for p in p_values]
        significant = [p < corrected_alpha for p in p_values]
        
        return {
            'method': 'Bonferroni',
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'original_p_values': p_values,
            'adjusted_p_values': adjusted_p_values,
            'significant': significant,
            'n_comparisons': n_comparisons,
            'family_wise_error_rate': corrected_alpha
        }
    
    @staticmethod
    def holm_bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Apply Holm-Bonferroni step-down correction."""
        n_comparisons = len(p_values)
        
        # Sort p-values with their original indices
        sorted_p_with_idx = sorted(enumerate(p_values), key=lambda x: x[1])
        
        significant = [False] * n_comparisons
        adjusted_p_values = [0.0] * n_comparisons
        
        for i, (original_idx, p_value) in enumerate(sorted_p_with_idx):
            adjusted_alpha = alpha / (n_comparisons - i)
            adjusted_p = min(p_value * (n_comparisons - i), 1.0)
            adjusted_p_values[original_idx] = adjusted_p
            
            if p_value <= adjusted_alpha:
                significant[original_idx] = True
            else:
                # Stop testing (step-down procedure)
                break
        
        return {
            'method': 'Holm-Bonferroni',
            'original_alpha': alpha,
            'original_p_values': p_values,
            'adjusted_p_values': adjusted_p_values,
            'significant': significant,
            'n_comparisons': n_comparisons
        }
    
    @staticmethod
    def benjamini_hochberg_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """Apply Benjamini-Hochberg FDR correction."""
        n_comparisons = len(p_values)
        
        # Sort p-values with their original indices
        sorted_p_with_idx = sorted(enumerate(p_values), key=lambda x: x[1])
        
        significant = [False] * n_comparisons
        adjusted_p_values = [0.0] * n_comparisons
        
        # Find the largest i such that p_i <= (i/m) * alpha
        last_significant = -1
        for i, (original_idx, p_value) in enumerate(sorted_p_with_idx):
            threshold = ((i + 1) / n_comparisons) * alpha
            if p_value <= threshold:
                last_significant = i
        
        # Mark significant tests
        for i in range(last_significant + 1):
            original_idx = sorted_p_with_idx[i][0]
            significant[original_idx] = True
        
        # Calculate adjusted p-values
        for i, (original_idx, p_value) in enumerate(sorted_p_with_idx):
            adjusted_p = min(p_value * n_comparisons / (i + 1), 1.0)
            adjusted_p_values[original_idx] = adjusted_p
        
        return {
            'method': 'Benjamini-Hochberg',
            'original_alpha': alpha,
            'original_p_values': p_values,
            'adjusted_p_values': adjusted_p_values,
            'significant': significant,
            'n_comparisons': n_comparisons,
            'false_discovery_rate': alpha
        }


class PerformanceComparator:
    """Specialized class for comparing algorithm performance."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.tester = StatisticalTester(alpha=alpha)
        self.logger = logging.getLogger(__name__)
    
    def compare_two_algorithms(self, algorithm1_data: np.ndarray, 
                              algorithm2_data: np.ndarray,
                              algorithm1_name: str = "Algorithm 1",
                              algorithm2_name: str = "Algorithm 2",
                              metric_name: str = "Performance") -> ComparisonResult:
        """Comprehensive comparison between two algorithms."""
        tests = []
        
        # Parametric tests
        t_test_result = self.tester.t_test(algorithm1_data, algorithm2_data, equal_var=False)
        tests.append(t_test_result)
        
        # Non-parametric tests
        mann_whitney_result = self.tester.mann_whitney_u_test(algorithm1_data, algorithm2_data)
        tests.append(mann_whitney_result)
        
        # Distribution comparison
        ks_result = self.tester.kolmogorov_smirnov_test(algorithm1_data, algorithm2_data)
        tests.append(ks_result)
        
        # Bootstrap test
        bootstrap_result = self.tester.bootstrap_test(algorithm1_data, algorithm2_data)
        tests.append(bootstrap_result)
        
        # Determine winner and confidence
        winner = None
        confidence = 0.0
        
        significant_tests = [test for test in tests if test.significant]
        if significant_tests:
            # Check consistency across tests
            mean1, mean2 = np.mean(algorithm1_data), np.mean(algorithm2_data)
            winner = algorithm1_name if mean1 > mean2 else algorithm2_name
            confidence = 1.0 - np.mean([test.p_value for test in significant_tests])
        
        # Check for practical significance (effect size)
        effect_sizes = [test.effect_size for test in tests if test.effect_size is not None]
        practical_significance = any(abs(effect) > 0.5 for effect in effect_sizes if not np.isnan(effect))
        
        return ComparisonResult(
            algorithm1=algorithm1_name,
            algorithm2=algorithm2_name,
            metric=metric_name,
            tests=tests,
            winner=winner,
            confidence=confidence,
            practical_significance=practical_significance
        )
    
    def compare_multiple_algorithms(self, algorithms_data: Dict[str, np.ndarray],
                                   metric_name: str = "Performance") -> Dict[str, Any]:
        """Compare multiple algorithms with multiple comparison correction."""
        algorithm_names = list(algorithms_data.keys())
        n_algorithms = len(algorithm_names)
        
        if n_algorithms < 2:
            raise ValueError("Need at least 2 algorithms for comparison")
        
        # Pairwise comparisons
        pairwise_comparisons = {}
        p_values = []
        comparison_pairs = []
        
        for i, alg1 in enumerate(algorithm_names):
            for alg2 in algorithm_names[i+1:]:
                comparison_key = f"{alg1}_vs_{alg2}"
                
                comparison_result = self.compare_two_algorithms(
                    algorithms_data[alg1], algorithms_data[alg2], alg1, alg2, metric_name
                )
                
                pairwise_comparisons[comparison_key] = comparison_result
                
                # Extract p-values for multiple comparison correction
                mann_whitney_test = next((test for test in comparison_result.tests 
                                        if test.test_name == "Mann-Whitney U"), None)
                if mann_whitney_test:
                    p_values.append(mann_whitney_test.p_value)
                    comparison_pairs.append(comparison_key)
        
        # Multiple comparison corrections
        corrections = {}
        if p_values:
            corrections['bonferroni'] = MultipleComparisonCorrection.bonferroni_correction(p_values, self.alpha)
            corrections['holm_bonferroni'] = MultipleComparisonCorrection.holm_bonferroni_correction(p_values, self.alpha)
            corrections['benjamini_hochberg'] = MultipleComparisonCorrection.benjamini_hochberg_correction(p_values, self.alpha)
        
        # ANOVA if applicable
        anova_result = None
        if n_algorithms > 2:
            try:
                algorithm_groups = [algorithms_data[name] for name in algorithm_names]
                f_stat, p_value = stats.f_oneway(*algorithm_groups)
                
                anova_result = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value),
                    'significant': p_value < self.alpha,
                    'interpretation': f"Overall difference significant (F={f_stat:.4f}, p={p_value:.4f})" if p_value < self.alpha else f"No overall difference (F={f_stat:.4f}, p={p_value:.4f})"
                }
            except Exception as e:
                self.logger.error(f"ANOVA failed: {e}")
        
        # Rankings
        rankings = self._compute_rankings(algorithms_data)
        
        return {
            'metric_name': metric_name,
            'algorithms': algorithm_names,
            'pairwise_comparisons': {k: v.to_dict() for k, v in pairwise_comparisons.items()},
            'multiple_comparison_corrections': corrections,
            'anova': anova_result,
            'rankings': rankings,
            'summary': self._generate_multiple_comparison_summary(
                pairwise_comparisons, corrections, rankings
            )
        }
    
    def _compute_rankings(self, algorithms_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Compute algorithm rankings based on performance."""
        # Mean performance ranking
        mean_performances = {name: np.mean(data) for name, data in algorithms_data.items()}
        mean_ranking = sorted(mean_performances.items(), key=lambda x: x[1], reverse=True)
        
        # Median performance ranking
        median_performances = {name: np.median(data) for name, data in algorithms_data.items()}
        median_ranking = sorted(median_performances.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'by_mean': [name for name, _ in mean_ranking],
            'by_median': [name for name, _ in median_ranking],
            'mean_values': mean_performances,
            'median_values': median_performances
        }
    
    def _generate_multiple_comparison_summary(self, pairwise_comparisons: Dict[str, ComparisonResult],
                                            corrections: Dict[str, Any],
                                            rankings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of multiple comparison analysis."""
        summary = {
            'total_comparisons': len(pairwise_comparisons),
            'significant_comparisons_uncorrected': 0,
            'significant_comparisons_corrected': {},
            'best_algorithm': None,
            'clear_winner': False,
            'key_findings': []
        }
        
        # Count significant comparisons
        significant_uncorrected = sum(1 for comp in pairwise_comparisons.values() 
                                    if any(test.significant for test in comp.tests))
        summary['significant_comparisons_uncorrected'] = significant_uncorrected
        
        # Count significant comparisons after correction
        for correction_name, correction_result in corrections.items():
            significant_corrected = sum(correction_result['significant'])
            summary['significant_comparisons_corrected'][correction_name] = significant_corrected
        
        # Best algorithm
        if 'by_mean' in rankings and rankings['by_mean']:
            summary['best_algorithm'] = rankings['by_mean'][0]
        
        # Check for clear winner (algorithm that wins most comparisons)
        winner_counts = {}
        for comp in pairwise_comparisons.values():
            if comp.winner:
                winner_counts[comp.winner] = winner_counts.get(comp.winner, 0) + 1
        
        if winner_counts:
            most_wins = max(winner_counts.values())
            algorithms_with_most_wins = [alg for alg, count in winner_counts.items() if count == most_wins]
            
            if len(algorithms_with_most_wins) == 1:
                summary['clear_winner'] = True
                summary['best_algorithm'] = algorithms_with_most_wins[0]
        
        # Key findings
        findings = []
        
        if significant_uncorrected > 0:
            findings.append(f"{significant_uncorrected} significant pairwise differences found")
        
        for correction_name, count in summary['significant_comparisons_corrected'].items():
            if count > 0:
                findings.append(f"{count} differences remain significant after {correction_name.replace('_', ' ').title()} correction")
        
        if summary['clear_winner']:
            findings.append(f"{summary['best_algorithm']} appears to be the best performing algorithm")
        
        summary['key_findings'] = findings
        
        return summary


class SignificanceTest:
    """Utility class for various significance tests."""
    
    @staticmethod
    def effect_size_interpretation(cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    @staticmethod
    def power_analysis(effect_size: float, alpha: float = 0.05, 
                      power: float = 0.8) -> Dict[str, Any]:
        """Calculate required sample size for given power."""
        try:
            import statsmodels.stats.power as smp
            
            # Power analysis for t-test
            sample_size = smp.ttest_power(effect_size=effect_size, 
                                        alpha=alpha, 
                                        power=power,
                                        alternative='two-sided')
            
            return {
                'required_sample_size_per_group': int(np.ceil(sample_size)),
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'interpretation': f"Need {int(np.ceil(sample_size))} samples per group to detect effect size {effect_size} with {power*100}% power"
            }
            
        except ImportError:
            # Approximate calculation without statsmodels
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = stats.norm.ppf(power)
            
            n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
            return {
                'required_sample_size_per_group': int(np.ceil(n_per_group)),
                'effect_size': effect_size,
                'alpha': alpha,
                'power': power,
                'interpretation': f"Approximate: Need {int(np.ceil(n_per_group))} samples per group",
                'note': "Install statsmodels for more accurate power analysis"
            }
    
    @staticmethod
    def confidence_interval_bootstrap(data: np.ndarray, 
                                    statistic_func: callable = np.mean,
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if len(data) < 3:
            return (np.nan, np.nan)
        
        bootstrap_statistics = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_statistics.append(statistic_func(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_lower = np.percentile(bootstrap_statistics, lower_percentile)
        ci_upper = np.percentile(bootstrap_statistics, upper_percentile)
        
        return (float(ci_lower), float(ci_upper))
    
    @staticmethod
    def equivalence_test(sample1: np.ndarray, sample2: np.ndarray,
                        epsilon: float = 0.5) -> Dict[str, Any]:
        """Test for statistical equivalence (TOST - Two One-Sided Tests)."""
        if len(sample1) < 3 or len(sample2) < 3:
            return {'error': 'Insufficient sample size'}
        
        try:
            mean1, mean2 = np.mean(sample1), np.mean(sample2)
            n1, n2 = len(sample1), len(sample2)
            
            # Pooled standard error
            s1, s2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
            pooled_se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
            
            # Degrees of freedom (Welch's approximation)
            df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
            
            # Two one-sided tests
            t1 = (mean1 - mean2 + epsilon) / pooled_se  # Test if difference > -epsilon
            t2 = (mean1 - mean2 - epsilon) / pooled_se  # Test if difference < epsilon
            
            p1 = stats.t.cdf(t1, df)   # Upper tail for first test
            p2 = 1 - stats.t.cdf(t2, df)  # Lower tail for second test
            
            # TOST p-value is maximum of the two p-values
            tost_p_value = max(p1, p2)
            
            equivalent = tost_p_value < 0.05  # Significant means equivalent
            
            return {
                'equivalent': equivalent,
                'tost_p_value': float(tost_p_value),
                'epsilon': epsilon,
                'mean_difference': float(mean1 - mean2),
                'interpretation': f"Groups are {'equivalent' if equivalent else 'not equivalent'} within ±{epsilon} units"
            }
            
        except Exception as e:
            return {'error': f'Equivalence test failed: {str(e)}'}
    
    @staticmethod
    def normality_test_battery(data: np.ndarray) -> Dict[str, TestResult]:
        """Run multiple normality tests."""
        if len(data) < 8:
            return {'error': 'Insufficient sample size for normality tests'}
        
        tests = {}
        
        # Shapiro-Wilk test
        try:
            stat, p_value = stats.shapiro(data[:5000])  # Limit for performance
            tests['shapiro_wilk'] = TestResult(
                test_name="Shapiro-Wilk",
                statistic=float(stat),
                p_value=float(p_value),
                significant=p_value < 0.05,
                interpretation="Data is normal" if p_value >= 0.05 else "Data is not normal"
            )
        except Exception as e:
            tests['shapiro_wilk'] = TestResult(
                test_name="Shapiro-Wilk",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
        
        # Anderson-Darling test
        try:
            result = stats.anderson(data, dist='norm')
            critical_value = result.critical_values[2]  # 5% significance level
            significant = result.statistic > critical_value
            
            tests['anderson_darling'] = TestResult(
                test_name="Anderson-Darling",
                statistic=float(result.statistic),
                p_value=np.nan,  # Anderson-Darling doesn't provide exact p-value
                critical_value=float(critical_value),
                significant=significant,
                interpretation="Data is normal" if not significant else "Data is not normal"
            )
        except Exception as e:
            tests['anderson_darling'] = TestResult(
                test_name="Anderson-Darling",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
        
        # Kolmogorov-Smirnov test against normal distribution
        try:
            stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
            tests['kolmogorov_smirnov'] = TestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=float(stat),
                p_value=float(p_value),
                significant=p_value < 0.05,
                interpretation="Data is normal" if p_value >= 0.05 else "Data is not normal"
            )
        except Exception as e:
            tests['kolmogorov_smirnov'] = TestResult(
                test_name="Kolmogorov-Smirnov",
                statistic=np.nan,
                p_value=np.nan,
                interpretation=f"Test failed: {str(e)}"
            )
        
        return tests