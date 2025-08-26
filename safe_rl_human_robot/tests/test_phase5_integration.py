"""
Comprehensive integration tests for Phase 5: Results Analysis & Visualization.

This test suite validates the complete Phase 5 functionality including analysis,
visualization, dashboard, and reporting components.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
import threading
import time

# Import Phase 5 components
from src.analysis.performance_analyzer import PerformanceAnalyzer, LearningCurveAnalyzer
from src.analysis.safety_analyzer import SafetyAnalyzer, ConstraintAnalyzer
from src.analysis.baseline_comparison import BaselineComparator
from src.analysis.statistical_tests import StatisticalTester
from src.visualization.training_plots import TrainingPlotter
from src.visualization.safety_plots import SafetyPlotter
from src.visualization.comparison_plots import ComparisonPlotter
from src.visualization.dashboard import DashboardManager, TrainingDashboard
from src.reporting.automated_reports import (
    PerformanceReportGenerator,
    SafetyReportGenerator,
    generate_all_reports
)


class TestPhase5Integration:
    """Integration tests for Phase 5 components."""
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp(prefix="phase5_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture(scope="class")
    def sample_training_data(self):
        """Generate sample training data for testing."""
        np.random.seed(42)
        
        algorithms = ['CPO', 'PPO', 'Lagrangian-PPO']
        training_data = {}
        
        for alg in algorithms:
            n_episodes = 500
            base_return = 100 + np.random.uniform(-20, 20)
            
            # Simulate learning curve with trend
            iterations = np.arange(n_episodes)
            base_curve = base_return * (1 - np.exp(-iterations / 100))
            noise = np.random.normal(0, 5, n_episodes)
            episode_returns = base_curve + noise
            
            data = pd.DataFrame({
                'iteration': iterations,
                'episode_return': episode_returns,
                'episode_length': np.random.poisson(200, n_episodes),
                'policy_loss': np.random.exponential(0.1, n_episodes),
                'value_loss': np.random.exponential(0.05, n_episodes),
                'kl_divergence': np.random.exponential(0.01, n_episodes),
                'constraint_violation': np.maximum(0, np.random.normal(0.02, 0.01, n_episodes)),
                'safety_score': np.minimum(1.0, np.random.beta(4, 1, n_episodes))
            })
            
            training_data[alg] = data
        
        return training_data
    
    @pytest.fixture(scope="class") 
    def sample_safety_data(self):
        """Generate sample safety data for testing."""
        np.random.seed(42)
        
        algorithms = ['CPO', 'PPO', 'Lagrangian-PPO']
        safety_data = {}
        
        for alg in algorithms:
            n_episodes = 500
            
            # Simulate different safety profiles
            if 'CPO' in alg:
                violation_rate = 0.01
                safety_base = 0.9
            elif 'Lagrangian' in alg:
                violation_rate = 0.02
                safety_base = 0.85
            else:  # PPO
                violation_rate = 0.05
                safety_base = 0.75
            
            data = pd.DataFrame({
                'iteration': range(n_episodes),
                'constraint_violation': np.random.exponential(violation_rate, n_episodes),
                'safety_score': np.minimum(1.0, np.random.beta(
                    safety_base * 10, (1 - safety_base) * 10, n_episodes)),
                'constraint_1': np.random.normal(0, 0.1, n_episodes),
                'constraint_2': np.random.normal(0, 0.05, n_episodes), 
                'constraint_3': np.random.normal(0, 0.02, n_episodes),
                'risk_level': np.random.choice(['low', 'medium', 'high'], 
                                             n_episodes, p=[0.7, 0.25, 0.05])
            })
            
            safety_data[alg] = data
        
        return safety_data


class TestAnalysisComponents:
    """Test analysis component integration."""
    
    def test_performance_analysis_pipeline(self, sample_training_data, temp_dir):
        """Test complete performance analysis pipeline."""
        analyzer = PerformanceAnalyzer()
        
        results = {}
        for alg_name, data in sample_training_data.items():
            # Test learning curve analysis
            learning_result = analyzer.analyze_learning_curve(data)
            assert 'learning_rate' in learning_result
            assert 'trend' in learning_result
            assert 'plateau_detected' in learning_result
            
            # Test sample efficiency analysis
            efficiency_result = analyzer.analyze_sample_efficiency(data)
            assert 'efficiency_score' in efficiency_result
            assert 'learning_efficiency' in efficiency_result
            
            results[alg_name] = {
                'learning_curve': learning_result,
                'sample_efficiency': efficiency_result
            }
        
        # Verify results are reasonable
        assert len(results) == 3
        for alg_result in results.values():
            assert alg_result['learning_curve']['learning_rate'] >= 0
            assert alg_result['sample_efficiency']['efficiency_score'] >= 0
    
    def test_safety_analysis_pipeline(self, sample_safety_data, temp_dir):
        """Test complete safety analysis pipeline."""
        analyzer = SafetyAnalyzer()
        
        results = {}
        for alg_name, data in sample_safety_data.items():
            # Test constraint violation analysis
            constraint_result = analyzer.analyze_constraint_violations(data)
            assert 'total_violations' in constraint_result
            assert 'violation_rate' in constraint_result
            assert 'constraint_analysis' in constraint_result
            
            # Test risk analysis
            risk_result = analyzer.analyze_risk_patterns(data)
            assert 'risk_distribution' in risk_result
            assert 'risk_trends' in risk_result
            
            results[alg_name] = {
                'constraints': constraint_result,
                'risk': risk_result
            }
        
        # Verify safety analysis results
        assert len(results) == 3
        # CPO should have lowest violation rate
        cpo_violations = results['CPO']['constraints']['violation_rate']
        ppo_violations = results['PPO']['constraints']['violation_rate']
        assert cpo_violations <= ppo_violations
    
    def test_baseline_comparison_pipeline(self, sample_training_data, temp_dir):
        """Test baseline comparison pipeline."""
        comparator = BaselineComparator()
        
        # Compare algorithms
        comparison_result = comparator.compare_algorithms(sample_training_data)
        
        assert 'performance_comparison' in comparison_result
        assert 'statistical_significance' in comparison_result
        assert 'rankings' in comparison_result
        
        # Verify comparison results
        performance_comp = comparison_result['performance_comparison']
        assert len(performance_comp) == 3
        
        for alg in sample_training_data.keys():
            assert alg in performance_comp
            assert 'mean_performance' in performance_comp[alg]
            assert 'final_performance' in performance_comp[alg]
    
    def test_statistical_testing_pipeline(self, sample_training_data, temp_dir):
        """Test statistical testing pipeline."""
        tester = StatisticalTester()
        
        algorithms = list(sample_training_data.keys())
        data1 = sample_training_data[algorithms[0]]['episode_return'].values
        data2 = sample_training_data[algorithms[1]]['episode_return'].values
        
        # Test Mann-Whitney U test
        mw_result = tester.mann_whitney_u_test(data1, data2)
        assert hasattr(mw_result, 'p_value')
        assert hasattr(mw_result, 'effect_size')
        assert 0 <= mw_result.p_value <= 1
        
        # Test Kolmogorov-Smirnov test
        ks_result = tester.kolmogorov_smirnov_test(data1, data2)
        assert hasattr(ks_result, 'p_value')
        assert 0 <= ks_result.p_value <= 1
        
        # Test bootstrap confidence intervals
        ci_result = tester.bootstrap_confidence_interval(data1)
        assert hasattr(ci_result, 'lower')
        assert hasattr(ci_result, 'upper')
        assert ci_result.lower <= ci_result.upper


class TestVisualizationComponents:
    """Test visualization component integration."""
    
    def test_training_plots_generation(self, sample_training_data, temp_dir):
        """Test training plots generation."""
        plotter = TrainingPlotter()
        
        # Test learning curves plot
        fig = plotter.create_learning_curves_plot(sample_training_data)
        assert fig is not None
        
        # Save plot to verify it works
        save_path = temp_dir / "test_learning_curves.png"
        fig.savefig(save_path)
        assert save_path.exists()
        plt.close(fig)
        
        # Test convergence analysis plot
        single_data = list(sample_training_data.values())[0]
        fig = plotter.create_convergence_analysis_plot(single_data)
        assert fig is not None
        plt.close(fig)
        
        # Test training dashboard
        fig = plotter.create_training_dashboard(single_data)
        assert fig is not None
        plt.close(fig)
    
    def test_safety_plots_generation(self, sample_safety_data, temp_dir):
        """Test safety plots generation."""
        plotter = SafetyPlotter()
        
        # Test safety dashboard
        single_data = list(sample_safety_data.values())[0]
        fig = plotter.create_safety_dashboard(single_data)
        assert fig is not None
        
        # Save plot to verify it works
        save_path = temp_dir / "test_safety_dashboard.png"
        fig.savefig(save_path)
        assert save_path.exists()
        plt.close(fig)
        
        # Test constraint violation plots
        constraint_plotter = plotter.constraint_plotter
        violation_data = {
            'constraint_1': single_data['constraint_1'].values,
            'constraint_2': single_data['constraint_2'].values,
            'constraint_3': single_data['constraint_3'].values
        }
        
        fig = constraint_plotter.create_violation_frequency_plot(violation_data)
        assert fig is not None
        plt.close(fig)
    
    def test_comparison_plots_generation(self, sample_training_data, temp_dir):
        """Test comparison plots generation."""
        plotter = ComparisonPlotter()
        
        # Test performance comparison plot
        fig = plotter.create_performance_comparison(sample_training_data)
        assert fig is not None
        
        # Save plot to verify it works
        save_path = temp_dir / "test_performance_comparison.png"
        fig.savefig(save_path)
        assert save_path.exists()
        plt.close(fig)
        
        # Test statistical comparison
        statistical_data = {
            'significance_matrix': [[1, 0.05, 0.01], [0.05, 1, 0.1], [0.01, 0.1, 1]],
            'algorithms': list(sample_training_data.keys()),
            'performance_distributions': {
                alg: data['episode_return'].values 
                for alg, data in sample_training_data.items()
            }
        }
        
        fig = plotter.create_statistical_comparison(statistical_data)
        assert fig is not None
        plt.close(fig)


class TestDashboardComponents:
    """Test dashboard component integration."""
    
    def test_dashboard_manager_initialization(self, temp_dir):
        """Test dashboard manager initialization."""
        manager = DashboardManager(data_dir=temp_dir / "dashboard_data")
        
        # Verify database initialization
        assert manager.db_path.exists()
        assert manager.data_dir.exists()
        assert isinstance(manager.data_queue, type(manager.data_queue))
    
    def test_dashboard_data_operations(self, temp_dir):
        """Test dashboard data operations."""
        manager = DashboardManager(data_dir=temp_dir / "dashboard_data")
        
        # Test adding data points
        training_point = {
            'type': 'training',
            'experiment_id': 'test_exp',
            'iteration': 1,
            'episode_return': 100.0,
            'policy_loss': 0.1,
            'constraint_violation': 0.01
        }
        
        manager.add_data_point(training_point)
        assert not manager.data_queue.empty()
        
        # Process the data point
        manager._process_data_update(training_point)
        
        # Retrieve data
        recent_data = manager.get_recent_data('test_exp', 'training', limit=10)
        assert len(recent_data) >= 0  # Should not error
    
    @pytest.mark.slow
    def test_training_dashboard_creation(self, temp_dir):
        """Test training dashboard creation (without running server)."""
        manager = DashboardManager(data_dir=temp_dir / "dashboard_data")
        dashboard = TrainingDashboard(manager, "test_experiment")
        
        # Test dashboard app creation
        app = dashboard.create_app()
        assert app is not None
        assert dashboard.app is not None
        
        # Test layout creation
        layout = dashboard._create_layout()
        assert layout is not None


class TestReportingComponents:
    """Test reporting component integration."""
    
    def test_performance_report_generation(self, sample_training_data, temp_dir):
        """Test performance report generation."""
        generator = PerformanceReportGenerator(output_dir=temp_dir)
        
        # Generate performance report
        report_path = generator.generate_performance_report(
            sample_training_data,
            experiment_config={'test': 'config'},
            report_title="Test Performance Report"
        )
        
        # Verify report was created
        assert Path(report_path).exists()
        assert Path(report_path).suffix == '.html'
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Test Performance Report' in content
            assert 'CPO' in content
            assert 'Performance Comparison' in content
    
    def test_safety_report_generation(self, sample_safety_data, temp_dir):
        """Test safety report generation."""
        generator = SafetyReportGenerator(output_dir=temp_dir)
        
        # Generate safety report
        report_path = generator.generate_safety_report(
            sample_safety_data,
            experiment_config={'test': 'config'},
            report_title="Test Safety Report"
        )
        
        # Verify report was created
        assert Path(report_path).exists()
        assert Path(report_path).suffix == '.html'
        
        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Test Safety Report' in content
            assert 'Safety Executive Summary' in content
    
    def test_batch_report_generation(self, sample_training_data, sample_safety_data, temp_dir):
        """Test batch report generation."""
        report_paths = generate_all_reports(
            training_data=sample_training_data,
            safety_data=sample_safety_data,
            experiment_config={'environment': 'test'},
            output_dir=temp_dir
        )
        
        # Verify all report types were generated
        assert 'performance' in report_paths
        assert 'safety' in report_paths
        assert 'comparison' in report_paths
        assert 'executive' in report_paths
        
        # Verify files exist
        for report_type, path in report_paths.items():
            assert Path(path).exists()
            assert Path(path).suffix == '.html'


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_complete_analysis_workflow(self, sample_training_data, sample_safety_data, temp_dir):
        """Test complete analysis workflow from data to reports."""
        
        # Step 1: Perform all analyses
        perf_analyzer = PerformanceAnalyzer()
        safety_analyzer = SafetyAnalyzer()
        comparator = BaselineComparator()
        
        analysis_results = {}
        
        # Analyze each algorithm
        for alg_name, train_data in sample_training_data.items():
            safety_data = sample_safety_data[alg_name]
            
            # Performance analysis
            learning_analysis = perf_analyzer.analyze_learning_curve(train_data)
            efficiency_analysis = perf_analyzer.analyze_sample_efficiency(train_data)
            
            # Safety analysis
            constraint_analysis = safety_analyzer.analyze_constraint_violations(safety_data)
            risk_analysis = safety_analyzer.analyze_risk_patterns(safety_data)
            
            analysis_results[alg_name] = {
                'performance': {
                    'learning_curve': learning_analysis,
                    'sample_efficiency': efficiency_analysis
                },
                'safety': {
                    'constraints': constraint_analysis,
                    'risk': risk_analysis
                }
            }
        
        # Step 2: Generate visualizations
        training_plotter = TrainingPlotter()
        safety_plotter = SafetyPlotter()
        comparison_plotter = ComparisonPlotter()
        
        viz_paths = []
        
        # Training visualizations
        fig = training_plotter.create_learning_curves_plot(sample_training_data)
        path = temp_dir / "learning_curves.png"
        fig.savefig(path)
        viz_paths.append(path)
        plt.close(fig)
        
        # Safety visualizations
        single_safety_data = list(sample_safety_data.values())[0]
        fig = safety_plotter.create_safety_dashboard(single_safety_data)
        path = temp_dir / "safety_dashboard.png"
        fig.savefig(path)
        viz_paths.append(path)
        plt.close(fig)
        
        # Comparison visualizations
        fig = comparison_plotter.create_performance_comparison(sample_training_data)
        path = temp_dir / "performance_comparison.png"
        fig.savefig(path)
        viz_paths.append(path)
        plt.close(fig)
        
        # Step 3: Generate reports
        report_paths = generate_all_reports(
            training_data=sample_training_data,
            safety_data=sample_safety_data,
            experiment_config={
                'environment': 'TestEnvironment',
                'algorithms': list(sample_training_data.keys()),
                'total_timesteps': 500000
            },
            output_dir=temp_dir
        )
        
        # Verify complete workflow results
        assert len(analysis_results) == 3
        assert len(viz_paths) == 3
        assert len(report_paths) == 4
        
        # Verify all files were created
        for path in viz_paths:
            assert path.exists()
        
        for path in report_paths.values():
            assert Path(path).exists()
        
        # Verify analysis results structure
        for alg_result in analysis_results.values():
            assert 'performance' in alg_result
            assert 'safety' in alg_result
            assert 'learning_curve' in alg_result['performance']
            assert 'constraints' in alg_result['safety']
    
    def test_data_pipeline_integration(self, sample_training_data, temp_dir):
        """Test integration between data processing and visualization."""
        
        # Process data through analysis pipeline
        analyzer = PerformanceAnalyzer()
        processed_results = {}
        
        for alg_name, data in sample_training_data.items():
            # Analyze learning curve
            learning_result = analyzer.analyze_learning_curve(data)
            
            # Extract processed data for visualization
            processed_results[alg_name] = {
                'raw_data': data,
                'learning_analysis': learning_result,
                'smoothed_curve': learning_result.get('smoothed_curve', []),
                'trend_line': learning_result.get('trend_line', [])
            }
        
        # Verify data flows correctly to visualization
        plotter = TrainingPlotter()
        
        # Test that processed data can be visualized
        for alg_name, results in processed_results.items():
            raw_data = results['raw_data']
            
            # Create visualization with processed data
            fig = plotter.create_learning_curves_plot({alg_name: raw_data})
            assert fig is not None
            
            # Verify plot has expected components
            axes = fig.get_axes()
            assert len(axes) > 0
            
            # Verify data is plotted
            for ax in axes:
                lines = ax.get_lines()
                assert len(lines) > 0
            
            plt.close(fig)
    
    @pytest.mark.slow
    def test_real_time_data_simulation(self, temp_dir):
        """Test real-time data processing simulation."""
        
        # Initialize dashboard manager
        manager = DashboardManager(data_dir=temp_dir / "dashboard_data")
        
        # Simulate real-time data generation
        def generate_data():
            for i in range(10):
                data_point = {
                    'type': 'training',
                    'experiment_id': 'real_time_test',
                    'iteration': i,
                    'episode_return': 100 + i * 2 + np.random.normal(0, 1),
                    'episode_length': np.random.poisson(200),
                    'policy_loss': np.random.exponential(0.1),
                    'constraint_violation': max(0, np.random.normal(0.02, 0.01)),
                    'safety_score': min(1, np.random.beta(4, 1))
                }
                
                manager.add_data_point(data_point)
                time.sleep(0.1)  # Simulate real-time delay
        
        # Start data generation in background
        data_thread = threading.Thread(target=generate_data, daemon=True)
        data_thread.start()
        
        # Wait for some data to be generated
        time.sleep(0.5)
        
        # Process queued data
        while not manager.data_queue.empty():
            data_update = manager.data_queue.get_nowait()
            manager._process_data_update(data_update)
        
        # Verify data was stored correctly
        recent_data = manager.get_recent_data('real_time_test', 'training', limit=20)
        assert len(recent_data) > 0
        
        data_thread.join(timeout=1)


class TestPerformanceAndScalability:
    """Test performance and scalability of Phase 5 components."""
    
    def test_large_dataset_handling(self, temp_dir):
        """Test handling of large datasets."""
        
        # Generate large dataset
        n_episodes = 10000
        large_data = pd.DataFrame({
            'iteration': range(n_episodes),
            'episode_return': np.cumsum(np.random.normal(0.01, 0.1, n_episodes)),
            'episode_length': np.random.poisson(200, n_episodes),
            'policy_loss': np.random.exponential(0.1, n_episodes),
            'constraint_violation': np.maximum(0, np.random.normal(0.02, 0.01, n_episodes)),
            'safety_score': np.minimum(1.0, np.random.beta(4, 1, n_episodes))
        })
        
        # Test analysis performance
        analyzer = PerformanceAnalyzer()
        start_time = time.time()
        result = analyzer.analyze_learning_curve(large_data)
        analysis_time = time.time() - start_time
        
        # Verify analysis completes in reasonable time (< 5 seconds)
        assert analysis_time < 5.0
        assert 'learning_rate' in result
        
        # Test visualization performance
        plotter = TrainingPlotter()
        start_time = time.time()
        fig = plotter.create_learning_curves_plot({'Large Dataset': large_data})
        viz_time = time.time() - start_time
        
        # Verify visualization completes in reasonable time (< 10 seconds)
        assert viz_time < 10.0
        assert fig is not None
        plt.close(fig)
    
    def test_memory_efficiency(self, temp_dir):
        """Test memory efficiency with multiple datasets."""
        
        # Create multiple datasets
        datasets = {}
        for i in range(5):
            datasets[f'algorithm_{i}'] = pd.DataFrame({
                'iteration': range(1000),
                'episode_return': np.random.normal(100, 10, 1000),
                'episode_length': np.random.poisson(200, 1000),
                'constraint_violation': np.random.exponential(0.02, 1000)
            })
        
        # Test batch processing
        analyzer = PerformanceAnalyzer()
        results = {}
        
        for alg_name, data in datasets.items():
            results[alg_name] = analyzer.analyze_learning_curve(data)
            
            # Clear data reference to test memory management
            data = None
        
        # Verify all analyses completed successfully
        assert len(results) == 5
        for result in results.values():
            assert 'learning_rate' in result


# Test utilities and fixtures
@pytest.fixture(scope="session")
def phase5_test_config():
    """Configuration for Phase 5 tests."""
    return {
        'test_data_size': 500,
        'test_algorithms': ['CPO', 'PPO', 'Lagrangian-PPO'],
        'visualization_formats': ['png', 'svg'],
        'report_formats': ['html', 'json'],
        'timeout_seconds': 30
    }


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])