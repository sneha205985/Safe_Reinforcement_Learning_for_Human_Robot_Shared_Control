"""
Unit tests for visualization components in Phase 5.

This module tests visualization components including training plots,
safety plots, comparison plots, and dashboard functionality.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import threading
import time

# Import visualization components
from src.visualization.training_plots import (
    TrainingPlotter, LearningCurvePlotter, ConvergencePlotter
)
from src.visualization.safety_plots import (
    SafetyPlotter, ConstraintViolationPlotter, RiskVisualization
)
from src.visualization.comparison_plots import (
    ComparisonPlotter, BaselinePlotter, ParetoFrontierPlotter
)
from src.visualization.dashboard import (
    DashboardManager, TrainingDashboard, SafetyDashboard
)


class TestTrainingPlots:
    """Test training visualization components."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        algorithms = ['CPO', 'PPO', 'Lagrangian-PPO']
        data = {}
        
        for alg in algorithms:
            n_episodes = 500
            base_return = 100 + np.random.uniform(-20, 20)
            
            data[alg] = pd.DataFrame({
                'iteration': range(n_episodes),
                'episode_return': base_return + np.cumsum(np.random.normal(0.02, 1, n_episodes)),
                'episode_length': np.random.poisson(200, n_episodes),
                'policy_loss': np.random.exponential(0.1, n_episodes),
                'value_loss': np.random.exponential(0.05, n_episodes),
                'kl_divergence': np.random.exponential(0.01, n_episodes),
                'lagrange_multiplier': np.maximum(0, np.random.normal(0.1, 0.05, n_episodes))
            })
        
        return data
    
    def test_learning_curve_plotter_initialization(self):
        """Test LearningCurvePlotter initialization."""
        plotter = LearningCurvePlotter()
        assert plotter.figsize is not None
        assert plotter.dpi > 0
        assert isinstance(plotter.colors, list)
    
    def test_learning_curves_plot_creation(self, sample_training_data):
        """Test learning curves plot creation."""
        plotter = LearningCurvePlotter()
        
        fig = plotter.create_learning_curves_plot(
            sample_training_data,
            metrics=['episode_return', 'episode_length']
        )
        
        # Verify figure is created
        assert isinstance(fig, Figure)
        
        # Verify axes are created
        axes = fig.get_axes()
        assert len(axes) >= 2  # Should have at least 2 subplots
        
        # Verify each algorithm is plotted
        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) >= len(sample_training_data)  # Each algorithm should have a line
        
        # Cleanup
        plt.close(fig)
    
    def test_learning_curves_with_confidence_intervals(self, sample_training_data):
        """Test learning curves with confidence intervals."""
        plotter = LearningCurvePlotter()
        
        fig = plotter.create_learning_curves_with_confidence(
            sample_training_data,
            confidence_level=0.95
        )
        
        assert isinstance(fig, Figure)
        
        # Verify confidence intervals are plotted
        axes = fig.get_axes()
        for ax in axes:
            # Should have both lines and fills for confidence intervals
            lines = ax.get_lines()
            collections = ax.collections  # Fill_between creates collections
            assert len(lines) > 0
            assert len(collections) > 0  # Confidence intervals
        
        plt.close(fig)
    
    def test_convergence_plotter(self, sample_training_data):
        """Test convergence analysis plotting."""
        plotter = ConvergencePlotter()
        
        # Test with single algorithm
        single_data = list(sample_training_data.values())[0]
        
        fig = plotter.create_convergence_analysis_plot(
            single_data,
            value_col='episode_return'
        )
        
        assert isinstance(fig, Figure)
        
        # Should have multiple subplots for convergence analysis
        axes = fig.get_axes()
        assert len(axes) >= 2
        
        plt.close(fig)
    
    def test_policy_gradient_plots(self, sample_training_data):
        """Test policy gradient visualization."""
        plotter = TrainingPlotter()
        
        single_data = list(sample_training_data.values())[0]
        
        fig = plotter.create_policy_gradient_plot(single_data)
        
        assert isinstance(fig, Figure)
        
        # Verify policy gradient metrics are plotted
        axes = fig.get_axes()
        assert len(axes) > 0
        
        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) > 0
        
        plt.close(fig)
    
    def test_training_dashboard_creation(self, sample_training_data):
        """Test training dashboard creation."""
        plotter = TrainingPlotter()
        
        single_data = list(sample_training_data.values())[0]
        
        fig = plotter.create_training_dashboard(single_data)
        
        assert isinstance(fig, Figure)
        
        # Dashboard should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 4  # Comprehensive dashboard
        
        plt.close(fig)
    
    def test_interactive_training_plot(self, sample_training_data):
        """Test interactive training plot creation."""
        plotter = TrainingPlotter()
        
        fig = plotter.create_interactive_training_plot(sample_training_data)
        
        # Should return Plotly figure
        assert isinstance(fig, go.Figure)
        
        # Verify traces are added
        assert len(fig.data) > 0
        
        # Verify layout is set
        assert fig.layout.title is not None
    
    def test_plot_saving(self, sample_training_data):
        """Test plot saving functionality."""
        plotter = TrainingPlotter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_plot.png"
            
            fig = plotter.create_learning_curves_plot(
                sample_training_data,
                save_path=str(save_path)
            )
            
            # Verify file is saved
            assert save_path.exists()
            
            plt.close(fig)


class TestSafetyPlots:
    """Test safety visualization components."""
    
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
            'risk_level': np.random.choice(['low', 'medium', 'high'], n_episodes, p=[0.7, 0.25, 0.05]),
            'safety_margin': np.random.uniform(0, 1, n_episodes)
        })
    
    def test_constraint_violation_plotter(self, sample_safety_data):
        """Test constraint violation plotting."""
        plotter = ConstraintViolationPlotter()
        
        # Test violation frequency plot
        violation_data = {
            'constraint_1': sample_safety_data['constraint_1'].values,
            'constraint_2': sample_safety_data['constraint_2'].values,
            'constraint_3': sample_safety_data['constraint_3'].values
        }
        
        fig = plotter.create_violation_frequency_plot(violation_data)
        
        assert isinstance(fig, Figure)
        
        # Should have histogram for each constraint
        axes = fig.get_axes()
        assert len(axes) >= len(violation_data)
        
        plt.close(fig)
    
    def test_violation_timeline_plot(self, sample_safety_data):
        """Test violation timeline plotting."""
        plotter = ConstraintViolationPlotter()
        
        fig = plotter.create_violation_timeline_plot(
            sample_safety_data,
            violation_col='constraint_violation'
        )
        
        assert isinstance(fig, Figure)
        
        # Verify timeline plot
        axes = fig.get_axes()
        assert len(axes) > 0
        
        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) > 0  # Should have violation timeline
        
        plt.close(fig)
    
    def test_risk_visualization(self, sample_safety_data):
        """Test risk visualization components."""
        risk_viz = RiskVisualization()
        
        # Test safety margin distribution
        fig = risk_viz.create_safety_margin_distribution(
            sample_safety_data['safety_margin'].values
        )
        
        assert isinstance(fig, Figure)
        
        # Should show distribution
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)
    
    def test_risk_heatmap_creation(self, sample_safety_data):
        """Test risk heatmap creation."""
        risk_viz = RiskVisualization()
        
        # Create risk matrix
        risk_matrix = np.random.rand(5, 5)
        
        fig = risk_viz.create_risk_heatmap(
            risk_matrix,
            row_labels=['Risk1', 'Risk2', 'Risk3', 'Risk4', 'Risk5'],
            col_labels=['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5']
        )
        
        assert isinstance(fig, Figure)
        
        # Verify heatmap
        axes = fig.get_axes()
        assert len(axes) > 0
        
        # Should have image plot (heatmap)
        for ax in axes:
            images = ax.get_images()
            assert len(images) > 0
        
        plt.close(fig)
    
    def test_safety_dashboard_creation(self, sample_safety_data):
        """Test comprehensive safety dashboard."""
        plotter = SafetyPlotter()
        
        fig = plotter.create_safety_dashboard(sample_safety_data)
        
        assert isinstance(fig, Figure)
        
        # Dashboard should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 6  # Comprehensive safety dashboard
        
        # Verify all subplots have content
        for ax in axes:
            # Each subplot should have some visual elements
            has_content = (len(ax.get_lines()) > 0 or 
                          len(ax.collections) > 0 or 
                          len(ax.patches) > 0 or
                          len(ax.get_images()) > 0)
            assert has_content
        
        plt.close(fig)
    
    def test_interactive_safety_plot(self, sample_safety_data):
        """Test interactive safety plots."""
        plotter = SafetyPlotter()
        
        fig = plotter.create_interactive_safety_dashboard(sample_safety_data)
        
        # Should return Plotly figure
        assert isinstance(fig, go.Figure)
        
        # Verify traces are added
        assert len(fig.data) > 0
        
        # Verify layout
        assert fig.layout.title is not None
    
    def test_failure_mode_analysis(self, sample_safety_data):
        """Test failure mode analysis visualization."""
        plotter = SafetyPlotter()
        
        # Add failure mode data
        sample_safety_data['failure_mode'] = np.random.choice(
            ['collision', 'boundary', 'speed', 'none'], 
            len(sample_safety_data),
            p=[0.05, 0.03, 0.02, 0.9]
        )
        
        fig = plotter.create_failure_mode_analysis(sample_safety_data)
        
        assert isinstance(fig, Figure)
        
        # Should show failure mode analysis
        axes = fig.get_axes()
        assert len(axes) > 0
        
        plt.close(fig)


class TestComparisonPlots:
    """Test comparison visualization components."""
    
    @pytest.fixture
    def comparison_data(self):
        """Generate data for algorithm comparison."""
        np.random.seed(42)
        algorithms = ['CPO', 'PPO', 'Lagrangian-PPO', 'TRPO']
        data = {}
        
        for i, alg in enumerate(algorithms):
            n_episodes = 300
            base_performance = 80 + i * 5 + np.random.uniform(-5, 5)
            
            data[alg] = pd.DataFrame({
                'iteration': range(n_episodes),
                'episode_return': base_performance + np.cumsum(np.random.normal(0.02, 0.5, n_episodes)),
                'episode_length': np.random.poisson(200, n_episodes),
                'success_rate': np.minimum(1.0, np.maximum(0.0, 
                    (np.arange(n_episodes) / n_episodes) + np.random.normal(0, 0.05, n_episodes))),
                'sample_efficiency': np.random.exponential(0.8, n_episodes)
            })
        
        return data
    
    def test_comparison_plotter_initialization(self):
        """Test ComparisonPlotter initialization."""
        plotter = ComparisonPlotter()
        assert plotter.figsize is not None
        assert plotter.dpi > 0
        assert isinstance(plotter.colors, list)
    
    def test_performance_comparison_plot(self, comparison_data):
        """Test performance comparison visualization."""
        plotter = ComparisonPlotter()
        
        fig = plotter.create_performance_comparison(
            comparison_data,
            metrics=['episode_return', 'episode_length', 'success_rate', 'sample_efficiency']
        )
        
        assert isinstance(fig, Figure)
        
        # Should have subplot for each metric
        axes = fig.get_axes()
        assert len(axes) >= 4
        
        # Each subplot should have lines for each algorithm
        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) >= len(comparison_data)
        
        plt.close(fig)
    
    def test_algorithm_radar_chart(self, comparison_data):
        """Test algorithm radar chart creation."""
        plotter = ComparisonPlotter()
        
        # Prepare radar chart data
        algorithm_scores = {}
        for alg, data in comparison_data.items():
            algorithm_scores[alg] = {
                'Performance': np.random.uniform(0.5, 1.0),
                'Safety': np.random.uniform(0.4, 0.9),
                'Efficiency': np.random.uniform(0.3, 0.8),
                'Stability': np.random.uniform(0.6, 0.95)
            }
        
        fig = plotter.create_algorithm_radar_chart(algorithm_scores)
        
        # Should return Plotly figure
        assert isinstance(fig, go.Figure)
        
        # Verify traces for each algorithm
        assert len(fig.data) == len(algorithm_scores)
        
        # Verify radar chart layout
        assert fig.layout.polar is not None
    
    def test_statistical_comparison_plot(self, comparison_data):
        """Test statistical comparison visualization."""
        plotter = ComparisonPlotter()
        
        # Prepare statistical comparison data
        algorithms = list(comparison_data.keys())
        n_algs = len(algorithms)
        
        comparison_results = {
            'performance_distributions': {
                alg: data['episode_return'].values 
                for alg, data in comparison_data.items()
            },
            'significance_matrix': np.random.rand(n_algs, n_algs),
            'algorithms': algorithms,
            'effect_sizes': {alg: np.random.uniform(0.1, 0.8) for alg in algorithms},
            'confidence_intervals': {
                alg: {
                    'mean': np.random.uniform(80, 120),
                    'lower': np.random.uniform(70, 90),
                    'upper': np.random.uniform(90, 130)
                }
                for alg in algorithms
            }
        }
        
        fig = plotter.create_statistical_comparison(comparison_results)
        
        assert isinstance(fig, Figure)
        
        # Should have multiple subplots for different statistical views
        axes = fig.get_axes()
        assert len(axes) >= 3
        
        plt.close(fig)
    
    def test_baseline_plotter(self, comparison_data):
        """Test baseline-specific plotting."""
        baseline_plotter = BaselinePlotter()
        
        # Prepare baseline results
        baseline_results = {
            'performance_safety_data': {
                alg: {
                    'performance': np.random.uniform(0.5, 1.0),
                    'safety': np.random.uniform(0.4, 0.9)
                }
                for alg in comparison_data.keys()
            },
            'sample_efficiency': {alg: np.random.uniform(0.3, 1.0) for alg in comparison_data.keys()},
            'training_times': {alg: np.random.uniform(1, 10) for alg in comparison_data.keys()},
            'violation_rates': {alg: np.random.uniform(0.01, 0.1) for alg in comparison_data.keys()},
            'success_rates': {alg: np.random.uniform(0.6, 0.95) for alg in comparison_data.keys()},
            'overall_rankings': {alg: np.random.uniform(0.5, 1.0) for alg in comparison_data.keys()}
        }
        
        fig = baseline_plotter.create_baseline_performance_plot(
            baseline_results,
            target_algorithm='CPO'
        )
        
        assert isinstance(fig, Figure)
        
        # Should have multiple subplots for comprehensive baseline comparison
        axes = fig.get_axes()
        assert len(axes) >= 4
        
        plt.close(fig)
    
    def test_pareto_frontier_plotter(self, comparison_data):
        """Test Pareto frontier visualization."""
        pareto_plotter = ParetoFrontierPlotter()
        
        # Prepare Pareto data
        pareto_data = {
            alg: {
                'performance': np.random.uniform(0.3, 0.9),
                'safety': np.random.uniform(0.4, 0.85)
            }
            for alg in comparison_data.keys()
        }
        
        fig = pareto_plotter.create_pareto_frontier_plot(pareto_data)
        
        assert isinstance(fig, Figure)
        
        # Should have subplots for Pareto analysis
        axes = fig.get_axes()
        assert len(axes) >= 2
        
        # First subplot should be scatter plot with Pareto frontier
        scatter_ax = axes[0]
        lines = scatter_ax.get_lines()
        collections = scatter_ax.collections
        
        # Should have scatter points and potentially Pareto frontier line
        assert len(collections) > 0 or len(lines) > 0
        
        plt.close(fig)
    
    def test_interactive_pareto_plot(self, comparison_data):
        """Test interactive Pareto frontier plot."""
        pareto_plotter = ParetoFrontierPlotter()
        
        # Prepare Pareto data
        pareto_data = {
            alg: {
                'performance': np.random.uniform(0.3, 0.9),
                'safety': np.random.uniform(0.4, 0.85)
            }
            for alg in comparison_data.keys()
        }
        
        fig = pareto_plotter.create_interactive_pareto_plot(pareto_data)
        
        # Should return Plotly figure
        assert isinstance(fig, go.Figure)
        
        # Should have scatter traces for each algorithm
        assert len(fig.data) >= len(pareto_data)
        
        # Verify layout
        assert fig.layout.xaxis.title is not None
        assert fig.layout.yaxis.title is not None


class TestDashboardComponents:
    """Test dashboard functionality."""
    
    def test_dashboard_manager_initialization(self):
        """Test DashboardManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            
            # Verify initialization
            assert manager.data_dir.exists()
            assert manager.db_path.exists()
            assert manager.data_queue is not None
            assert not manager.is_running
    
    def test_dashboard_data_operations(self):
        """Test dashboard data operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            
            # Test adding data point
            data_point = {
                'type': 'training',
                'experiment_id': 'test_exp',
                'iteration': 1,
                'episode_return': 100.0,
                'episode_length': 200,
                'policy_loss': 0.1,
                'value_loss': 0.05,
                'kl_divergence': 0.01,
                'constraint_violation': 0.02,
                'safety_score': 0.85
            }
            
            manager.add_data_point(data_point)
            
            # Verify data was added to queue
            assert not manager.data_queue.empty()
            
            # Process data point
            manager._process_data_update(data_point)
            
            # Test data retrieval
            recent_data = manager.get_recent_data('test_exp', 'training', limit=10)
            assert isinstance(recent_data, pd.DataFrame)
    
    def test_training_dashboard_creation(self):
        """Test TrainingDashboard creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            dashboard = TrainingDashboard(manager, "test_experiment")
            
            # Test dashboard creation
            app = dashboard.create_app()
            assert app is not None
            assert dashboard.app is not None
            
            # Test layout creation
            layout = dashboard._create_layout()
            assert layout is not None
    
    def test_safety_dashboard_creation(self):
        """Test SafetyDashboard creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            dashboard = SafetyDashboard(manager, "test_experiment")
            
            # Test dashboard creation
            app = dashboard.create_app()
            assert app is not None
            assert dashboard.app is not None
            
            # Test layout creation
            layout = dashboard._create_safety_layout()
            assert layout is not None
    
    @pytest.mark.slow
    def test_real_time_data_updates(self):
        """Test real-time data update functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            
            # Start real-time updates
            manager.start_real_time_updates(update_interval=0.1)
            assert manager.is_running
            
            # Add some data points
            for i in range(5):
                data_point = {
                    'type': 'training',
                    'experiment_id': 'real_time_test',
                    'iteration': i,
                    'episode_return': 100 + i,
                    'constraint_violation': 0.01 * i
                }
                manager.add_data_point(data_point)
            
            # Wait for updates to process
            time.sleep(0.5)
            
            # Stop updates
            manager.stop_real_time_updates()
            assert not manager.is_running
            
            # Verify data was processed
            recent_data = manager.get_recent_data('real_time_test', 'training', limit=10)
            assert len(recent_data) > 0


class TestPlotIntegration:
    """Test integration between different plotting components."""
    
    @pytest.fixture
    def integrated_data(self):
        """Generate integrated dataset for testing."""
        np.random.seed(42)
        
        algorithms = ['CPO', 'PPO']
        training_data = {}
        safety_data = {}
        
        for alg in algorithms:
            n_episodes = 200
            
            # Training data
            training_data[alg] = pd.DataFrame({
                'iteration': range(n_episodes),
                'episode_return': 100 + np.cumsum(np.random.normal(0.1, 1, n_episodes)),
                'episode_length': np.random.poisson(200, n_episodes),
                'policy_loss': np.random.exponential(0.1, n_episodes)
            })
            
            # Safety data
            safety_data[alg] = pd.DataFrame({
                'iteration': range(n_episodes),
                'constraint_violation': np.maximum(0, np.random.normal(0.02, 0.01, n_episodes)),
                'safety_score': np.minimum(1.0, np.random.beta(4, 1, n_episodes))
            })
        
        return {'training': training_data, 'safety': safety_data}
    
    def test_training_to_comparison_workflow(self, integrated_data):
        """Test workflow from training plots to comparison plots."""
        training_data = integrated_data['training']
        
        # Create training plots
        training_plotter = TrainingPlotter()
        training_fig = training_plotter.create_learning_curves_plot(training_data)
        
        assert isinstance(training_fig, Figure)
        
        # Create comparison plots using same data
        comparison_plotter = ComparisonPlotter()
        comparison_fig = comparison_plotter.create_performance_comparison(training_data)
        
        assert isinstance(comparison_fig, Figure)
        
        # Verify both plots have appropriate content
        training_axes = training_fig.get_axes()
        comparison_axes = comparison_fig.get_axes()
        
        assert len(training_axes) > 0
        assert len(comparison_axes) > 0
        
        plt.close(training_fig)
        plt.close(comparison_fig)
    
    def test_safety_to_dashboard_workflow(self, integrated_data):
        """Test workflow from safety plots to dashboard integration."""
        safety_data = integrated_data['safety']
        
        # Create safety plots
        safety_plotter = SafetyPlotter()
        
        single_safety_data = list(safety_data.values())[0]
        safety_fig = safety_plotter.create_safety_dashboard(single_safety_data)
        
        assert isinstance(safety_fig, Figure)
        
        # Test dashboard integration
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DashboardManager(data_dir=temp_dir)
            
            # Add safety data to dashboard
            for i, row in single_safety_data.iterrows():
                if i >= 10:  # Limit for testing
                    break
                    
                safety_event = {
                    'type': 'safety_event',
                    'experiment_id': 'test_exp',
                    'event_type': 'constraint_violation',
                    'severity': 'medium' if row['constraint_violation'] > 0.02 else 'low',
                    'constraint_name': 'test_constraint',
                    'violation_value': row['constraint_violation']
                }
                manager.add_data_point(safety_event)
            
            # Verify data integration
            recent_events = manager.get_recent_data('test_exp', 'safety', limit=20)
            assert isinstance(recent_events, pd.DataFrame)
        
        plt.close(safety_fig)
    
    def test_end_to_end_visualization_pipeline(self, integrated_data):
        """Test complete visualization pipeline."""
        training_data = integrated_data['training']
        safety_data = integrated_data['safety']
        
        # Step 1: Individual component plots
        training_plotter = TrainingPlotter()
        safety_plotter = SafetyPlotter()
        comparison_plotter = ComparisonPlotter()
        
        plots_created = []
        
        # Training plots
        training_fig = training_plotter.create_learning_curves_plot(training_data)
        plots_created.append(('training', training_fig))
        
        # Safety plots
        single_safety = list(safety_data.values())[0]
        safety_fig = safety_plotter.create_safety_dashboard(single_safety)
        plots_created.append(('safety', safety_fig))
        
        # Comparison plots
        comparison_fig = comparison_plotter.create_performance_comparison(training_data)
        plots_created.append(('comparison', comparison_fig))
        
        # Step 2: Verify all plots were created successfully
        assert len(plots_created) == 3
        
        for plot_type, fig in plots_created:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) > 0
        
        # Step 3: Test saving all plots
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for i, (plot_type, fig) in enumerate(plots_created):
                save_path = temp_path / f"{plot_type}_plot.png"
                fig.savefig(save_path)
                assert save_path.exists()
        
        # Cleanup
        for _, fig in plots_created:
            plt.close(fig)


class TestPlotCustomization:
    """Test plot customization and styling."""
    
    def test_custom_styling(self):
        """Test custom plot styling options."""
        # Test custom colors
        custom_colors = ['red', 'blue', 'green']
        plotter = TrainingPlotter(colors=custom_colors)
        
        assert plotter.colors == custom_colors
    
    def test_custom_figure_size(self):
        """Test custom figure size settings."""
        custom_figsize = (15, 10)
        custom_dpi = 150
        
        plotter = TrainingPlotter(figsize=custom_figsize, dpi=custom_dpi)
        
        assert plotter.figsize == custom_figsize
        assert plotter.dpi == custom_dpi
    
    def test_plot_format_options(self):
        """Test different plot format options."""
        np.random.seed(42)
        data = {'test_alg': pd.DataFrame({
            'iteration': range(100),
            'episode_return': np.random.normal(100, 10, 100)
        })}
        
        plotter = TrainingPlotter()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test different formats
            formats = ['png', 'pdf', 'svg']
            
            for fmt in formats:
                save_path = temp_path / f"test_plot.{fmt}"
                fig = plotter.create_learning_curves_plot(data, save_path=str(save_path))
                
                assert save_path.exists()
                plt.close(fig)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])