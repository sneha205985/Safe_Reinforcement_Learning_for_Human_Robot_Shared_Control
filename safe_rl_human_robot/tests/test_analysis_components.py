"""
Unit tests for analysis components in Phase 5.

This module tests individual analysis components including performance analyzers,
safety analyzers, baseline comparisons, and statistical testing.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

# Import analysis components
from src.analysis.performance_analyzer import (
    PerformanceAnalyzer, LearningCurveAnalyzer, SampleEfficiencyAnalyzer
)
from src.analysis.safety_analyzer import (
    SafetyAnalyzer, ConstraintAnalyzer, RiskAnalyzer
)
from src.analysis.baseline_comparison import (
    BaselineComparator, AlgorithmComparison, BenchmarkSuite
)
from src.analysis.statistical_tests import StatisticalTester


class TestPerformanceAnalyzer:
    """Test performance analyzer components."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_episodes = 1000
        
        # Simulate learning curve
        iterations = np.arange(n_episodes)
        base_performance = 100 * (1 - np.exp(-iterations / 200))
        noise = np.random.normal(0, 5, n_episodes)
        
        return pd.DataFrame({
            'iteration': iterations,
            'episode_return': base_performance + noise,
            'episode_length': np.random.poisson(200, n_episodes),
            'success_rate': np.minimum(1.0, iterations / n_episodes + np.random.normal(0, 0.1, n_episodes)),
            'timesteps': np.cumsum(np.random.poisson(200, n_episodes))
        })
    
    def test_learning_curve_analyzer_initialization(self):
        """Test LearningCurveAnalyzer initialization."""
        analyzer = LearningCurveAnalyzer()
        assert analyzer.smoothing_window >= 1
        assert analyzer.plateau_threshold > 0
        assert analyzer.trend_window >= 1
    
    def test_learning_curve_analysis(self, sample_training_data):
        """Test learning curve analysis."""
        analyzer = LearningCurveAnalyzer()
        result = analyzer.analyze_learning_curve(
            sample_training_data, 
            value_col='episode_return',
            time_col='iteration'
        )
        
        # Verify required fields
        required_fields = ['learning_rate', 'trend', 'plateau_detected', 'smoothed_curve', 'trend_line']
        for field in required_fields:
            assert field in result
        
        # Verify data types and ranges
        assert isinstance(result['learning_rate'], float)
        assert result['learning_rate'] >= 0
        assert result['trend'] in ['increasing', 'decreasing', 'stable']
        assert isinstance(result['plateau_detected'], bool)
        assert isinstance(result['smoothed_curve'], (list, np.ndarray))
    
    def test_learning_curve_trend_detection(self):
        """Test trend detection in learning curves."""
        analyzer = LearningCurveAnalyzer()
        
        # Test increasing trend
        increasing_data = pd.DataFrame({
            'iteration': range(100),
            'episode_return': np.arange(100) * 0.5 + np.random.normal(0, 0.1, 100)
        })
        result = analyzer.analyze_learning_curve(increasing_data)
        assert result['trend'] == 'increasing'
        
        # Test decreasing trend
        decreasing_data = pd.DataFrame({
            'iteration': range(100),
            'episode_return': 100 - np.arange(100) * 0.5 + np.random.normal(0, 0.1, 100)
        })
        result = analyzer.analyze_learning_curve(decreasing_data)
        assert result['trend'] == 'decreasing'
    
    def test_plateau_detection(self):
        """Test plateau detection in learning curves."""
        analyzer = LearningCurveAnalyzer(plateau_threshold=0.01)
        
        # Create data with plateau
        plateau_data = pd.DataFrame({
            'iteration': range(200),
            'episode_return': np.concatenate([
                np.arange(100) * 0.5,  # Learning phase
                100 + np.random.normal(0, 0.5, 100)  # Plateau phase
            ])
        })
        
        result = analyzer.analyze_learning_curve(plateau_data)
        assert result['plateau_detected'] == True
        assert 'plateau_start' in result
        assert result['plateau_start'] > 50  # Should detect plateau in second half
    
    def test_sample_efficiency_analyzer(self, sample_training_data):
        """Test sample efficiency analysis."""
        analyzer = SampleEfficiencyAnalyzer()
        
        target_performance = sample_training_data['episode_return'].quantile(0.8)
        result = analyzer.analyze_sample_efficiency(
            sample_training_data,
            target_performance=target_performance,
            timesteps_col='timesteps'
        )
        
        # Verify required fields
        required_fields = ['target_achieved', 'samples_to_target', 'efficiency_score', 'learning_efficiency']
        for field in required_fields:
            assert field in result
        
        # Verify data types
        assert isinstance(result['target_achieved'], bool)
        assert isinstance(result['efficiency_score'], float)
        assert isinstance(result['learning_efficiency'], float)
        
        if result['target_achieved']:
            assert isinstance(result['samples_to_target'], int)
            assert result['samples_to_target'] > 0
    
    def test_performance_analyzer_integration(self, sample_training_data):
        """Test PerformanceAnalyzer integration."""
        analyzer = PerformanceAnalyzer()
        
        # Test learning curve analysis
        learning_result = analyzer.analyze_learning_curve(sample_training_data)
        assert 'learning_rate' in learning_result
        
        # Test sample efficiency analysis
        efficiency_result = analyzer.analyze_sample_efficiency(sample_training_data)
        assert 'efficiency_score' in efficiency_result
        
        # Test comprehensive analysis
        comprehensive_result = analyzer.analyze_training_performance(sample_training_data)
        assert 'learning_curve_analysis' in comprehensive_result
        assert 'sample_efficiency_analysis' in comprehensive_result
        assert 'performance_summary' in comprehensive_result


class TestSafetyAnalyzer:
    """Test safety analyzer components."""
    
    @pytest.fixture
    def sample_safety_data(self):
        """Generate sample safety data."""
        np.random.seed(42)
        n_episodes = 1000
        
        return pd.DataFrame({
            'iteration': range(n_episodes),
            'constraint_violation': np.maximum(0, np.random.normal(0.02, 0.01, n_episodes)),
            'safety_score': np.minimum(1.0, np.maximum(0.0, np.random.beta(4, 1, n_episodes))),
            'constraint_1': np.random.normal(0, 0.1, n_episodes),
            'constraint_2': np.random.normal(0, 0.05, n_episodes),
            'constraint_3': np.random.normal(0, 0.02, n_episodes),
            'risk_level': np.random.choice(['low', 'medium', 'high'], n_episodes, p=[0.7, 0.25, 0.05])
        })
    
    def test_constraint_analyzer_initialization(self):
        """Test ConstraintAnalyzer initialization."""
        analyzer = ConstraintAnalyzer()
        assert analyzer.violation_threshold >= 0
        assert analyzer.analysis_window >= 1
    
    def test_constraint_violation_analysis(self, sample_safety_data):
        """Test constraint violation analysis."""
        analyzer = ConstraintAnalyzer()
        
        constraint_columns = ['constraint_1', 'constraint_2', 'constraint_3']
        result = analyzer.analyze_constraint_violations(
            sample_safety_data, 
            constraint_columns=constraint_columns
        )
        
        # Verify required fields
        required_fields = ['total_violations', 'violation_rate', 'constraint_analysis', 'temporal_analysis']
        for field in required_fields:
            assert field in result
        
        # Verify data types and ranges
        assert isinstance(result['total_violations'], int)
        assert result['total_violations'] >= 0
        assert isinstance(result['violation_rate'], float)
        assert 0 <= result['violation_rate'] <= 1
        
        # Verify per-constraint analysis
        constraint_analysis = result['constraint_analysis']
        for constraint in constraint_columns:
            assert constraint in constraint_analysis
            assert 'violations' in constraint_analysis[constraint]
            assert 'rate' in constraint_analysis[constraint]
    
    def test_constraint_severity_analysis(self, sample_safety_data):
        """Test constraint severity analysis."""
        analyzer = ConstraintAnalyzer()
        
        # Add violations with different severities
        sample_safety_data['severe_violation'] = sample_safety_data['constraint_1'] > 0.15
        sample_safety_data['moderate_violation'] = (sample_safety_data['constraint_1'] > 0.05) & (sample_safety_data['constraint_1'] <= 0.15)
        
        result = analyzer.analyze_constraint_violations(sample_safety_data)
        
        # Verify severity analysis is included
        assert 'severity_analysis' in result or 'violation_distribution' in result
    
    def test_risk_analyzer(self, sample_safety_data):
        """Test RiskAnalyzer."""
        analyzer = RiskAnalyzer()
        
        result = analyzer.analyze_risk_patterns(
            sample_safety_data,
            safety_columns=['safety_score', 'constraint_violation']
        )
        
        # Verify required fields
        required_fields = ['risk_distribution', 'risk_trends', 'high_risk_episodes']
        for field in required_fields:
            assert field in result
        
        # Verify risk distribution
        risk_dist = result['risk_distribution']
        assert 'low' in risk_dist
        assert 'medium' in risk_dist  
        assert 'high' in risk_dist
        
        # Verify risk trends
        risk_trends = result['risk_trends']
        assert isinstance(risk_trends, dict)
    
    def test_safety_margin_analysis(self, sample_safety_data):
        """Test safety margin analysis."""
        analyzer = SafetyAnalyzer()
        
        result = analyzer.analyze_safety_margins(
            sample_safety_data,
            safety_score_col='safety_score',
            constraint_cols=['constraint_1', 'constraint_2']
        )
        
        # Verify safety margin results
        assert 'average_margin' in result
        assert 'minimum_margin' in result
        assert 'margin_distribution' in result
        
        assert isinstance(result['average_margin'], float)
        assert isinstance(result['minimum_margin'], float)
    
    def test_safety_analyzer_integration(self, sample_safety_data):
        """Test SafetyAnalyzer integration."""
        analyzer = SafetyAnalyzer()
        
        # Test constraint analysis
        constraint_result = analyzer.analyze_constraint_violations(sample_safety_data)
        assert 'violation_rate' in constraint_result
        
        # Test risk analysis  
        risk_result = analyzer.analyze_risk_patterns(sample_safety_data)
        assert 'risk_distribution' in risk_result
        
        # Test comprehensive safety analysis
        comprehensive_result = analyzer.analyze_safety_performance(sample_safety_data)
        assert 'constraint_analysis' in comprehensive_result
        assert 'risk_analysis' in comprehensive_result
        assert 'safety_summary' in comprehensive_result


class TestBaselineComparison:
    """Test baseline comparison components."""
    
    @pytest.fixture
    def multiple_algorithm_data(self):
        """Generate data for multiple algorithms."""
        np.random.seed(42)
        algorithms = ['CPO', 'PPO', 'Lagrangian-PPO', 'TRPO']
        data = {}
        
        for i, alg in enumerate(algorithms):
            n_episodes = 500
            base_performance = 80 + i * 10 + np.random.uniform(-10, 10)
            
            data[alg] = pd.DataFrame({
                'iteration': range(n_episodes),
                'episode_return': base_performance + np.cumsum(np.random.normal(0.02, 1, n_episodes)),
                'episode_length': np.random.poisson(200, n_episodes),
                'success_rate': np.minimum(1.0, np.maximum(0.0, 
                    (np.arange(n_episodes) / n_episodes) + np.random.normal(0, 0.1, n_episodes))),
                'constraint_violation': np.maximum(0, np.random.normal(0.03 - i * 0.005, 0.01, n_episodes)),
                'safety_score': np.minimum(1.0, np.random.beta(3 + i, 2, n_episodes))
            })
        
        return data
    
    def test_algorithm_comparison_initialization(self):
        """Test AlgorithmComparison initialization."""
        comparison = AlgorithmComparison()
        assert comparison.metrics is not None
        assert isinstance(comparison.metrics, list)
    
    def test_performance_comparison(self, multiple_algorithm_data):
        """Test performance comparison between algorithms."""
        comparison = AlgorithmComparison()
        
        result = comparison.compare_performance(multiple_algorithm_data)
        
        # Verify required fields
        assert 'performance_metrics' in result
        assert 'rankings' in result
        assert 'statistical_significance' in result
        
        # Verify performance metrics for each algorithm
        perf_metrics = result['performance_metrics']
        for alg in multiple_algorithm_data.keys():
            assert alg in perf_metrics
            assert 'mean_return' in perf_metrics[alg]
            assert 'final_return' in perf_metrics[alg]
            assert 'best_return' in perf_metrics[alg]
    
    def test_safety_comparison(self, multiple_algorithm_data):
        """Test safety comparison between algorithms."""
        comparison = AlgorithmComparison()
        
        result = comparison.compare_safety(multiple_algorithm_data)
        
        # Verify safety comparison results
        assert 'safety_metrics' in result
        assert 'violation_rates' in result
        assert 'safety_rankings' in result
        
        # Verify safety metrics for each algorithm
        safety_metrics = result['safety_metrics']
        for alg in multiple_algorithm_data.keys():
            assert alg in safety_metrics
            assert 'avg_safety_score' in safety_metrics[alg]
            assert 'violation_rate' in safety_metrics[alg]
    
    def test_baseline_comparator(self, multiple_algorithm_data):
        """Test BaselineComparator."""
        comparator = BaselineComparator()
        
        result = comparator.compare_algorithms(
            multiple_algorithm_data,
            baseline_algorithms=['PPO', 'TRPO']
        )
        
        # Verify comprehensive comparison results
        assert 'performance_comparison' in result
        assert 'safety_comparison' in result
        assert 'statistical_tests' in result
        assert 'rankings' in result
        
        # Verify statistical tests
        stat_tests = result['statistical_tests']
        assert isinstance(stat_tests, dict)
    
    def test_benchmark_suite(self, multiple_algorithm_data):
        """Test BenchmarkSuite."""
        benchmark = BenchmarkSuite()
        
        # Add algorithms to benchmark
        for alg_name, data in multiple_algorithm_data.items():
            benchmark.add_algorithm(alg_name, data)
        
        # Run benchmark
        result = benchmark.run_benchmark()
        
        # Verify benchmark results
        assert 'overall_rankings' in result
        assert 'performance_scores' in result
        assert 'safety_scores' in result
        assert 'composite_scores' in result
        
        # Verify all algorithms are included
        for alg in multiple_algorithm_data.keys():
            assert alg in result['overall_rankings']
    
    def test_pareto_frontier_analysis(self, multiple_algorithm_data):
        """Test Pareto frontier analysis."""
        comparator = BaselineComparator()
        
        result = comparator.analyze_pareto_frontier(
            multiple_algorithm_data,
            performance_metric='episode_return',
            safety_metric='safety_score'
        )
        
        # Verify Pareto frontier results
        assert 'pareto_points' in result
        assert 'dominated_algorithms' in result
        assert 'pareto_efficiency_scores' in result
        
        # Verify Pareto points
        pareto_points = result['pareto_points']
        assert isinstance(pareto_points, list)
        
        for point in pareto_points:
            assert 'algorithm' in point
            assert 'performance' in point
            assert 'safety' in point


class TestStatisticalTests:
    """Test statistical testing components."""
    
    @pytest.fixture
    def sample_distributions(self):
        """Generate sample distributions for testing."""
        np.random.seed(42)
        
        # Create different distributions
        data = {
            'normal_1': np.random.normal(100, 15, 1000),
            'normal_2': np.random.normal(105, 15, 1000),  # Slightly different mean
            'normal_3': np.random.normal(100, 20, 1000),  # Different variance
            'skewed': np.random.gamma(2, 50, 1000),       # Different distribution shape
            'bimodal': np.concatenate([
                np.random.normal(90, 5, 500),
                np.random.normal(110, 5, 500)
            ])
        }
        
        return data
    
    def test_statistical_tester_initialization(self):
        """Test StatisticalTester initialization."""
        tester = StatisticalTester()
        assert tester.alpha > 0 and tester.alpha < 1
        assert tester.n_bootstrap_samples > 0
    
    def test_mann_whitney_u_test(self, sample_distributions):
        """Test Mann-Whitney U test."""
        tester = StatisticalTester()
        
        # Test with similar distributions
        result_similar = tester.mann_whitney_u_test(
            sample_distributions['normal_1'],
            sample_distributions['normal_2']
        )
        
        assert hasattr(result_similar, 'statistic')
        assert hasattr(result_similar, 'p_value')
        assert hasattr(result_similar, 'effect_size')
        assert 0 <= result_similar.p_value <= 1
        
        # Test with very different distributions
        result_different = tester.mann_whitney_u_test(
            sample_distributions['normal_1'],
            sample_distributions['skewed']
        )
        
        # Should have low p-value for different distributions
        assert result_different.p_value < result_similar.p_value
    
    def test_kolmogorov_smirnov_test(self, sample_distributions):
        """Test Kolmogorov-Smirnov test."""
        tester = StatisticalTester()
        
        # Test with same distribution
        result_same = tester.kolmogorov_smirnov_test(
            sample_distributions['normal_1'],
            sample_distributions['normal_1']
        )
        
        assert hasattr(result_same, 'statistic')
        assert hasattr(result_same, 'p_value')
        assert result_same.p_value > 0.05  # Should not reject null hypothesis
        
        # Test with different distributions
        result_different = tester.kolmogorov_smirnov_test(
            sample_distributions['normal_1'],
            sample_distributions['bimodal']
        )
        
        assert result_different.p_value < 0.05  # Should reject null hypothesis
    
    def test_bootstrap_confidence_interval(self, sample_distributions):
        """Test bootstrap confidence interval."""
        tester = StatisticalTester()
        
        result = tester.bootstrap_confidence_interval(
            sample_distributions['normal_1'],
            confidence_level=0.95
        )
        
        assert hasattr(result, 'lower')
        assert hasattr(result, 'upper')
        assert hasattr(result, 'mean')
        assert result.lower < result.mean < result.upper
        
        # Test with different confidence level
        result_99 = tester.bootstrap_confidence_interval(
            sample_distributions['normal_1'],
            confidence_level=0.99
        )
        
        # 99% CI should be wider than 95% CI
        assert (result_99.upper - result_99.lower) > (result.upper - result.lower)
    
    def test_permutation_test(self, sample_distributions):
        """Test permutation test."""
        tester = StatisticalTester()
        
        result = tester.permutation_test(
            sample_distributions['normal_1'],
            sample_distributions['normal_2'],
            n_permutations=1000
        )
        
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'observed_difference')
        assert hasattr(result, 'permutation_differences')
        assert 0 <= result.p_value <= 1
        
        # Verify permutation differences
        assert len(result.permutation_differences) == 1000
    
    def test_multiple_comparisons_correction(self, sample_distributions):
        """Test multiple comparisons correction."""
        tester = StatisticalTester()
        
        # Generate multiple p-values
        p_values = []
        algorithms = list(sample_distributions.keys())
        
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                result = tester.mann_whitney_u_test(
                    sample_distributions[algorithms[i]],
                    sample_distributions[algorithms[j]]
                )
                p_values.append(result.p_value)
        
        # Test Bonferroni correction
        bonferroni_result = tester.bonferroni_correction(p_values)
        assert len(bonferroni_result.corrected_p_values) == len(p_values)
        assert all(cp >= p for cp, p in zip(bonferroni_result.corrected_p_values, p_values))
        
        # Test Benjamini-Hochberg correction
        bh_result = tester.benjamini_hochberg_correction(p_values)
        assert len(bh_result.corrected_p_values) == len(p_values)
    
    def test_effect_size_calculations(self, sample_distributions):
        """Test effect size calculations."""
        tester = StatisticalTester()
        
        # Test Cohen's d
        cohens_d = tester.cohens_d(
            sample_distributions['normal_1'],
            sample_distributions['normal_2']
        )
        
        assert isinstance(cohens_d, float)
        
        # Test with identical distributions
        cohens_d_same = tester.cohens_d(
            sample_distributions['normal_1'],
            sample_distributions['normal_1']
        )
        
        assert abs(cohens_d_same) < 0.1  # Should be close to 0
        
        # Test Cliff's delta
        cliffs_delta = tester.cliffs_delta(
            sample_distributions['normal_1'],
            sample_distributions['skewed']
        )
        
        assert isinstance(cliffs_delta, float)
        assert -1 <= cliffs_delta <= 1


class TestIntegrationBetweenComponents:
    """Test integration between analysis components."""
    
    def test_performance_to_comparison_pipeline(self):
        """Test pipeline from performance analysis to comparison."""
        # Generate data for multiple algorithms
        np.random.seed(42)
        algorithms = ['Algorithm_A', 'Algorithm_B']
        algorithm_data = {}
        
        for alg in algorithms:
            data = pd.DataFrame({
                'iteration': range(100),
                'episode_return': np.random.normal(100, 10, 100),
                'episode_length': np.random.poisson(200, 100)
            })
            algorithm_data[alg] = data
        
        # Run performance analysis
        perf_analyzer = PerformanceAnalyzer()
        performance_results = {}
        
        for alg, data in algorithm_data.items():
            performance_results[alg] = perf_analyzer.analyze_learning_curve(data)
        
        # Run comparison analysis using performance results
        comparator = BaselineComparator()
        comparison_result = comparator.compare_algorithms(algorithm_data)
        
        # Verify integration
        assert 'performance_comparison' in comparison_result
        assert len(comparison_result['performance_comparison']) == len(algorithms)
    
    def test_safety_to_statistical_testing_pipeline(self):
        """Test pipeline from safety analysis to statistical testing."""
        # Generate safety data
        np.random.seed(42)
        safety_data_a = pd.DataFrame({
            'constraint_violation': np.random.exponential(0.01, 1000),
            'safety_score': np.random.beta(5, 1, 1000)
        })
        
        safety_data_b = pd.DataFrame({
            'constraint_violation': np.random.exponential(0.02, 1000),
            'safety_score': np.random.beta(4, 1, 1000)
        })
        
        # Run safety analysis
        safety_analyzer = SafetyAnalyzer()
        
        safety_result_a = safety_analyzer.analyze_constraint_violations(safety_data_a)
        safety_result_b = safety_analyzer.analyze_constraint_violations(safety_data_b)
        
        # Extract violation rates for statistical testing
        violations_a = safety_data_a['constraint_violation'].values
        violations_b = safety_data_b['constraint_violation'].values
        
        # Run statistical test
        tester = StatisticalTester()
        stat_result = tester.mann_whitney_u_test(violations_a, violations_b)
        
        # Verify integration
        assert stat_result.p_value < 0.05  # Should detect difference
        
        # Verify safety analysis results inform statistical interpretation
        violation_rate_a = safety_result_a['violation_rate']
        violation_rate_b = safety_result_b['violation_rate']
        assert violation_rate_a != violation_rate_b


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])