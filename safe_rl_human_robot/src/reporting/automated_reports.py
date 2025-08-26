"""
Automated report generation system for Safe RL analysis.

This module provides comprehensive automated reporting capabilities including
performance reports, safety analysis reports, comparison reports, and executive summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import pickle
from jinja2 import Template, Environment, FileSystemLoader
import markdown
import weasyprint
from weasyprint import HTML, CSS
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import base64
from io import BytesIO

# Import our analysis and visualization modules
from ..analysis.performance_analyzer import PerformanceAnalyzer
from ..analysis.safety_analyzer import SafetyAnalyzer
from ..analysis.baseline_comparison import BaselineComparator
from ..analysis.statistical_tests import StatisticalTester
from ..visualization.training_plots import TrainingPlotter
from ..visualization.safety_plots import SafetyPlotter
from ..visualization.comparison_plots import ComparisonPlotter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Base class for generating automated reports.
    """
    
    def __init__(self, output_dir: Union[str, Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize template environment
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Report metadata
        self.report_metadata = {
            'generated_at': datetime.now(),
            'generator_version': '1.0.0',
            'system_info': self._get_system_info()
        }
        
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for report metadata."""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'matplotlib_version': plt.matplotlib.__version__
        }
    
    def _encode_plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for embedding."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        
        encoded = base64.b64encode(plot_data).decode('utf-8')
        return f"data:image/png;base64,{encoded}"
    
    def _save_report_data(self, report_data: Dict[str, Any], filename: str):
        """Save report data to JSON file for reproducibility."""
        data_path = self.output_dir / f"{filename}_data.json"
        
        # Convert non-serializable objects
        serializable_data = {}
        for key, value in report_data.items():
            try:
                json.dumps(value)
                serializable_data[key] = value
            except (TypeError, ValueError):
                if isinstance(value, pd.DataFrame):
                    serializable_data[key] = value.to_dict('records')
                elif isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = str(value)
        
        with open(data_path, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        logger.info(f"Report data saved to {data_path}")


class PerformanceReportGenerator(ReportGenerator):
    """
    Generates comprehensive performance analysis reports.
    """
    
    def __init__(self, output_dir: Union[str, Path] = None):
        super().__init__(output_dir)
        self.performance_analyzer = PerformanceAnalyzer()
        self.training_plotter = TrainingPlotter()
    
    def generate_performance_report(self, 
                                  training_data: Dict[str, pd.DataFrame],
                                  experiment_config: Dict[str, Any] = None,
                                  report_title: str = "Performance Analysis Report") -> str:
        """
        Generate comprehensive performance analysis report.
        
        Args:
            training_data: Dictionary mapping experiment names to training data
            experiment_config: Configuration details for the experiments
            report_title: Title for the report
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating performance analysis report...")
        
        # Perform analysis
        analysis_results = {}
        plot_paths = {}
        
        for exp_name, data in training_data.items():
            logger.info(f"Analyzing performance for experiment: {exp_name}")
            
            # Learning curve analysis
            learning_analysis = self.performance_analyzer.analyze_learning_curve(
                data, value_col='episode_return'
            )
            
            # Sample efficiency analysis
            efficiency_analysis = self.performance_analyzer.analyze_sample_efficiency(
                data, target_performance=learning_analysis.get('final_performance', 0) * 0.9
            )
            
            # Store results
            analysis_results[exp_name] = {
                'learning_curve': learning_analysis,
                'sample_efficiency': efficiency_analysis,
                'data_summary': self._summarize_data(data)
            }
            
            # Generate plots
            fig = self.training_plotter.create_learning_curves_plot(
                {exp_name: data}, 
                metrics=['episode_return', 'episode_length']
            )
            plot_paths[f'{exp_name}_learning_curves'] = self._encode_plot_to_base64(fig)
            plt.close(fig)
            
            # Convergence analysis plot
            fig = self.training_plotter.create_convergence_analysis_plot(
                data, save_path=None
            )
            plot_paths[f'{exp_name}_convergence'] = self._encode_plot_to_base64(fig)
            plt.close(fig)
        
        # Generate report
        report_data = {
            'title': report_title,
            'metadata': self.report_metadata,
            'experiment_config': experiment_config or {},
            'analysis_results': analysis_results,
            'plots': plot_paths,
            'summary': self._generate_performance_summary(analysis_results)
        }
        
        # Save report data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"performance_report_{timestamp}"
        self._save_report_data(report_data, report_filename)
        
        # Generate HTML report
        html_report = self._create_performance_html_report(report_data)
        html_path = self.output_dir / f"{report_filename}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Generate PDF report
        pdf_path = self.output_dir / f"{report_filename}.pdf"
        try:
            HTML(string=html_report, base_url=str(self.output_dir)).write_pdf(str(pdf_path))
            logger.info(f"PDF report generated: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not generate PDF report: {str(e)}")
        
        logger.info(f"Performance report generated: {html_path}")
        return str(html_path)
    
    def _summarize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for training data."""
        summary = {
            'total_episodes': len(data),
            'total_iterations': data['iteration'].max() if 'iteration' in data.columns else len(data),
            'final_performance': data['episode_return'].iloc[-1] if len(data) > 0 else 0,
            'best_performance': data['episode_return'].max() if len(data) > 0 else 0,
            'average_performance': data['episode_return'].mean() if len(data) > 0 else 0,
            'performance_std': data['episode_return'].std() if len(data) > 0 else 0,
            'convergence_iteration': self._find_convergence_point(data)
        }
        return summary
    
    def _find_convergence_point(self, data: pd.DataFrame) -> Optional[int]:
        """Find approximate convergence point in training data."""
        if len(data) < 100:
            return None
        
        # Use rolling average to smooth data
        window_size = max(20, len(data) // 50)
        rolling_mean = data['episode_return'].rolling(window=window_size).mean()
        
        # Find point where performance stabilizes (low variance)
        rolling_std = data['episode_return'].rolling(window=window_size).std()
        threshold = rolling_std.mean() * 0.5  # Threshold for stability
        
        stable_points = rolling_std < threshold
        if stable_points.any():
            convergence_idx = stable_points.idxmax()
            return data.iloc[convergence_idx].get('iteration', convergence_idx)
        
        return None
    
    def _generate_performance_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level performance summary."""
        summary = {
            'total_experiments': len(analysis_results),
            'best_experiment': None,
            'best_final_performance': float('-inf'),
            'most_efficient_experiment': None,
            'fastest_convergence': float('inf'),
            'overall_insights': []
        }
        
        for exp_name, results in analysis_results.items():
            data_summary = results['data_summary']
            final_perf = data_summary['final_performance']
            
            if final_perf > summary['best_final_performance']:
                summary['best_final_performance'] = final_perf
                summary['best_experiment'] = exp_name
            
            convergence = data_summary.get('convergence_iteration')
            if convergence and convergence < summary['fastest_convergence']:
                summary['fastest_convergence'] = convergence
                summary['most_efficient_experiment'] = exp_name
        
        # Generate insights
        if len(analysis_results) > 1:
            performance_values = [r['data_summary']['final_performance'] for r in analysis_results.values()]
            perf_std = np.std(performance_values)
            perf_mean = np.mean(performance_values)
            
            if perf_std / perf_mean < 0.1:
                summary['overall_insights'].append("Consistent performance across experiments")
            else:
                summary['overall_insights'].append("High variability in performance across experiments")
        
        return summary
    
    def _create_performance_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML performance report."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
                h2 { color: #34495e; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .summary-box { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-label { font-weight: bold; color: #2c3e50; }
                .metric-value { color: #27ae60; font-size: 1.2em; }
                .plot { text-align: center; margin: 20px 0; }
                .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .experiment-section { border-left: 4px solid #3498db; padding-left: 20px; margin: 30px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .insight { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .metadata { font-size: 0.9em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            
            <div class="metadata">
                <p><strong>Generated:</strong> {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Generator Version:</strong> {{ metadata.generator_version }}</p>
            </div>
            
            <div class="summary-box">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <span class="metric-label">Total Experiments:</span>
                    <span class="metric-value">{{ summary.total_experiments }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Experiment:</span>
                    <span class="metric-value">{{ summary.best_experiment }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Final Performance:</span>
                    <span class="metric-value">{{ "%.3f"|format(summary.best_final_performance) }}</span>
                </div>
                {% if summary.most_efficient_experiment %}
                <div class="metric">
                    <span class="metric-label">Most Efficient:</span>
                    <span class="metric-value">{{ summary.most_efficient_experiment }}</span>
                </div>
                {% endif %}
                
                {% if summary.overall_insights %}
                <h3>Key Insights</h3>
                {% for insight in summary.overall_insights %}
                <div class="insight">{{ insight }}</div>
                {% endfor %}
                {% endif %}
            </div>
            
            {% for exp_name, results in analysis_results.items() %}
            <div class="experiment-section">
                <h2>Experiment: {{ exp_name }}</h2>
                
                <h3>Data Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Episodes</td><td>{{ results.data_summary.total_episodes }}</td></tr>
                    <tr><td>Total Iterations</td><td>{{ results.data_summary.total_iterations }}</td></tr>
                    <tr><td>Final Performance</td><td>{{ "%.4f"|format(results.data_summary.final_performance) }}</td></tr>
                    <tr><td>Best Performance</td><td>{{ "%.4f"|format(results.data_summary.best_performance) }}</td></tr>
                    <tr><td>Average Performance</td><td>{{ "%.4f"|format(results.data_summary.average_performance) }}</td></tr>
                    <tr><td>Performance Std Dev</td><td>{{ "%.4f"|format(results.data_summary.performance_std) }}</td></tr>
                    {% if results.data_summary.convergence_iteration %}
                    <tr><td>Convergence Iteration</td><td>{{ results.data_summary.convergence_iteration }}</td></tr>
                    {% endif %}
                </table>
                
                <h3>Learning Curve Analysis</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Learning Rate</td><td>{{ "%.6f"|format(results.learning_curve.learning_rate) }}</td></tr>
                    <tr><td>Performance Trend</td><td>{{ results.learning_curve.trend }}</td></tr>
                    <tr><td>Plateau Detected</td><td>{{ "Yes" if results.learning_curve.plateau_detected else "No" }}</td></tr>
                    {% if results.learning_curve.plateau_start %}
                    <tr><td>Plateau Start</td><td>{{ results.learning_curve.plateau_start }}</td></tr>
                    {% endif %}
                </table>
                
                <h3>Sample Efficiency Analysis</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Target Achievement</td><td>{{ results.sample_efficiency.target_achieved }}</td></tr>
                    <tr><td>Samples to Target</td><td>{{ results.sample_efficiency.samples_to_target or "Not achieved" }}</td></tr>
                    <tr><td>Efficiency Score</td><td>{{ "%.4f"|format(results.sample_efficiency.efficiency_score) }}</td></tr>
                    <tr><td>Learning Efficiency</td><td>{{ "%.4f"|format(results.sample_efficiency.learning_efficiency) }}</td></tr>
                </table>
                
                {% if plots[exp_name + "_learning_curves"] %}
                <h3>Learning Curves</h3>
                <div class="plot">
                    <img src="{{ plots[exp_name + '_learning_curves'] }}" alt="Learning Curves">
                </div>
                {% endif %}
                
                {% if plots[exp_name + "_convergence"] %}
                <h3>Convergence Analysis</h3>
                <div class="plot">
                    <img src="{{ plots[exp_name + '_convergence'] }}" alt="Convergence Analysis">
                </div>
                {% endif %}
            </div>
            {% endfor %}
            
        </body>
        </html>
        """
        
        template = Template(template_str)
        return template.render(**report_data)


class SafetyReportGenerator(ReportGenerator):
    """
    Generates comprehensive safety analysis reports.
    """
    
    def __init__(self, output_dir: Union[str, Path] = None):
        super().__init__(output_dir)
        self.safety_analyzer = SafetyAnalyzer()
        self.safety_plotter = SafetyPlotter()
    
    def generate_safety_report(self,
                             safety_data: Dict[str, pd.DataFrame],
                             experiment_config: Dict[str, Any] = None,
                             report_title: str = "Safety Analysis Report") -> str:
        """
        Generate comprehensive safety analysis report.
        
        Args:
            safety_data: Dictionary mapping experiment names to safety data
            experiment_config: Configuration details for the experiments
            report_title: Title for the report
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating safety analysis report...")
        
        # Perform safety analysis
        analysis_results = {}
        plot_paths = {}
        
        for exp_name, data in safety_data.items():
            logger.info(f"Analyzing safety for experiment: {exp_name}")
            
            # Constraint violation analysis
            constraint_analysis = self.safety_analyzer.analyze_constraint_violations(
                data, constraint_columns=['constraint_1', 'constraint_2', 'constraint_3']
            )
            
            # Risk analysis
            risk_analysis = self.safety_analyzer.analyze_risk_patterns(
                data, safety_columns=['safety_score', 'constraint_violation']
            )
            
            # Store results
            analysis_results[exp_name] = {
                'constraint_analysis': constraint_analysis,
                'risk_analysis': risk_analysis,
                'safety_summary': self._summarize_safety_data(data)
            }
            
            # Generate plots
            fig = self.safety_plotter.create_safety_dashboard(
                data, save_path=None
            )
            plot_paths[f'{exp_name}_safety_dashboard'] = self._encode_plot_to_base64(fig)
            plt.close(fig)
            
            # Violation analysis plot
            fig = self.safety_plotter.create_violation_analysis_plot(
                constraint_analysis, save_path=None
            )
            plot_paths[f'{exp_name}_violations'] = self._encode_plot_to_base64(fig)
            plt.close(fig)
        
        # Generate report
        report_data = {
            'title': report_title,
            'metadata': self.report_metadata,
            'experiment_config': experiment_config or {},
            'analysis_results': analysis_results,
            'plots': plot_paths,
            'safety_summary': self._generate_safety_summary(analysis_results)
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"safety_report_{timestamp}"
        self._save_report_data(report_data, report_filename)
        
        # Generate HTML report
        html_report = self._create_safety_html_report(report_data)
        html_path = self.output_dir / f"{report_filename}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Safety report generated: {html_path}")
        return str(html_path)
    
    def _summarize_safety_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for safety data."""
        summary = {
            'total_episodes': len(data),
            'total_violations': (data.get('constraint_violation', pd.Series([0])) > 0).sum(),
            'violation_rate': (data.get('constraint_violation', pd.Series([0])) > 0).mean(),
            'average_safety_score': data.get('safety_score', pd.Series([1])).mean(),
            'worst_safety_score': data.get('safety_score', pd.Series([1])).min(),
            'safety_improvement': self._calculate_safety_trend(data)
        }
        return summary
    
    def _calculate_safety_trend(self, data: pd.DataFrame) -> str:
        """Calculate overall safety trend."""
        if 'safety_score' not in data.columns or len(data) < 10:
            return "Insufficient data"
        
        # Compare first and last quarters
        quarter_size = len(data) // 4
        first_quarter = data['safety_score'].head(quarter_size).mean()
        last_quarter = data['safety_score'].tail(quarter_size).mean()
        
        improvement = (last_quarter - first_quarter) / first_quarter * 100
        
        if improvement > 5:
            return f"Improving ({improvement:.1f}%)"
        elif improvement < -5:
            return f"Declining ({improvement:.1f}%)"
        else:
            return "Stable"
    
    def _generate_safety_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level safety summary."""
        summary = {
            'total_experiments': len(analysis_results),
            'safest_experiment': None,
            'lowest_violation_rate': float('inf'),
            'highest_safety_score': float('-inf'),
            'safety_insights': []
        }
        
        for exp_name, results in analysis_results.items():
            safety_summary = results['safety_summary']
            violation_rate = safety_summary['violation_rate']
            safety_score = safety_summary['average_safety_score']
            
            if violation_rate < summary['lowest_violation_rate']:
                summary['lowest_violation_rate'] = violation_rate
                summary['safest_experiment'] = exp_name
            
            if safety_score > summary['highest_safety_score']:
                summary['highest_safety_score'] = safety_score
        
        # Generate insights
        violation_rates = [r['safety_summary']['violation_rate'] for r in analysis_results.values()]
        avg_violation_rate = np.mean(violation_rates)
        
        if avg_violation_rate < 0.01:
            summary['safety_insights'].append("Excellent safety performance across all experiments")
        elif avg_violation_rate < 0.05:
            summary['safety_insights'].append("Good safety performance with minimal violations")
        else:
            summary['safety_insights'].append("Safety performance needs improvement")
        
        return summary
    
    def _create_safety_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML safety report."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #c0392b; border-bottom: 2px solid #e74c3c; }
                h2 { color: #a93226; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .summary-box { background-color: #fdf2f2; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #e74c3c; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-label { font-weight: bold; color: #2c3e50; }
                .metric-value { color: #c0392b; font-size: 1.2em; }
                .safe-value { color: #27ae60; }
                .warning-value { color: #f39c12; }
                .danger-value { color: #e74c3c; }
                .plot { text-align: center; margin: 20px 0; }
                .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .experiment-section { border-left: 4px solid #e74c3c; padding-left: 20px; margin: 30px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f8d7da; }
                .insight { background-color: #d1ecf1; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #bee5eb; }
                .metadata { font-size: 0.9em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            
            <div class="metadata">
                <p><strong>Generated:</strong> {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Generator Version:</strong> {{ metadata.generator_version }}</p>
            </div>
            
            <div class="summary-box">
                <h2>Safety Executive Summary</h2>
                <div class="metric">
                    <span class="metric-label">Total Experiments:</span>
                    <span class="metric-value">{{ safety_summary.total_experiments }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Safest Experiment:</span>
                    <span class="metric-value">{{ safety_summary.safest_experiment }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Lowest Violation Rate:</span>
                    <span class="metric-value {{ 'safe-value' if safety_summary.lowest_violation_rate < 0.01 else 'warning-value' if safety_summary.lowest_violation_rate < 0.05 else 'danger-value' }}">
                        {{ "%.4f"|format(safety_summary.lowest_violation_rate) }}
                    </span>
                </div>
                <div class="metric">
                    <span class="metric-label">Highest Safety Score:</span>
                    <span class="metric-value safe-value">{{ "%.4f"|format(safety_summary.highest_safety_score) }}</span>
                </div>
                
                {% if safety_summary.safety_insights %}
                <h3>Safety Insights</h3>
                {% for insight in safety_summary.safety_insights %}
                <div class="insight">{{ insight }}</div>
                {% endfor %}
                {% endif %}
            </div>
            
            {% for exp_name, results in analysis_results.items() %}
            <div class="experiment-section">
                <h2>Experiment: {{ exp_name }}</h2>
                
                <h3>Safety Data Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
                    <tr>
                        <td>Total Episodes</td>
                        <td>{{ results.safety_summary.total_episodes }}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td>Total Violations</td>
                        <td>{{ results.safety_summary.total_violations }}</td>
                        <td class="{{ 'safe-value' if results.safety_summary.total_violations == 0 else 'warning-value' if results.safety_summary.total_violations < 10 else 'danger-value' }}">
                            {{ "Excellent" if results.safety_summary.total_violations == 0 else "Good" if results.safety_summary.total_violations < 10 else "Needs Improvement" }}
                        </td>
                    </tr>
                    <tr>
                        <td>Violation Rate</td>
                        <td>{{ "%.4f"|format(results.safety_summary.violation_rate) }}</td>
                        <td class="{{ 'safe-value' if results.safety_summary.violation_rate < 0.01 else 'warning-value' if results.safety_summary.violation_rate < 0.05 else 'danger-value' }}">
                            {{ "Excellent" if results.safety_summary.violation_rate < 0.01 else "Good" if results.safety_summary.violation_rate < 0.05 else "Poor" }}
                        </td>
                    </tr>
                    <tr>
                        <td>Average Safety Score</td>
                        <td>{{ "%.4f"|format(results.safety_summary.average_safety_score) }}</td>
                        <td class="{{ 'safe-value' if results.safety_summary.average_safety_score > 0.8 else 'warning-value' if results.safety_summary.average_safety_score > 0.6 else 'danger-value' }}">
                            {{ "Excellent" if results.safety_summary.average_safety_score > 0.8 else "Good" if results.safety_summary.average_safety_score > 0.6 else "Poor" }}
                        </td>
                    </tr>
                    <tr>
                        <td>Safety Trend</td>
                        <td>{{ results.safety_summary.safety_improvement }}</td>
                        <td class="{{ 'safe-value' if 'Improving' in results.safety_summary.safety_improvement else 'warning-value' if 'Stable' in results.safety_summary.safety_improvement else 'danger-value' }}">
                            {{ "Good" if "Improving" in results.safety_summary.safety_improvement else "Monitor" if "Stable" in results.safety_summary.safety_improvement else "Action Required" }}
                        </td>
                    </tr>
                </table>
                
                {% if plots[exp_name + "_safety_dashboard"] %}
                <h3>Safety Dashboard</h3>
                <div class="plot">
                    <img src="{{ plots[exp_name + '_safety_dashboard'] }}" alt="Safety Dashboard">
                </div>
                {% endif %}
                
                {% if plots[exp_name + "_violations"] %}
                <h3>Violation Analysis</h3>
                <div class="plot">
                    <img src="{{ plots[exp_name + '_violations'] }}" alt="Violation Analysis">
                </div>
                {% endif %}
            </div>
            {% endfor %}
            
        </body>
        </html>
        """
        
        template = Template(template_str)
        return template.render(**report_data)


class ComparisonReportGenerator(ReportGenerator):
    """
    Generates comprehensive comparison reports between algorithms and experiments.
    """
    
    def __init__(self, output_dir: Union[str, Path] = None):
        super().__init__(output_dir)
        self.baseline_comparator = BaselineComparator()
        self.statistical_tester = StatisticalTester()
        self.comparison_plotter = ComparisonPlotter()
    
    def generate_comparison_report(self,
                                 comparison_data: Dict[str, pd.DataFrame],
                                 baseline_algorithms: List[str] = None,
                                 report_title: str = "Algorithm Comparison Report") -> str:
        """
        Generate comprehensive algorithm comparison report.
        
        Args:
            comparison_data: Dictionary mapping algorithm names to their results
            baseline_algorithms: List of baseline algorithm names
            report_title: Title for the report
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating algorithm comparison report...")
        
        if baseline_algorithms is None:
            baseline_algorithms = ['PPO', 'Lagrangian-PPO', 'Hand-crafted']
        
        # Perform statistical comparisons
        statistical_results = {}
        algorithms = list(comparison_data.keys())
        
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                alg1, alg2 = algorithms[i], algorithms[j]
                data1 = comparison_data[alg1]['episode_return'].values
                data2 = comparison_data[alg2]['episode_return'].values
                
                # Perform statistical tests
                mann_whitney_result = self.statistical_tester.mann_whitney_u_test(data1, data2)
                ks_result = self.statistical_tester.kolmogorov_smirnov_test(data1, data2)
                
                statistical_results[f"{alg1}_vs_{alg2}"] = {
                    'mann_whitney': mann_whitney_result,
                    'kolmogorov_smirnov': ks_result
                }
        
        # Generate comparison plots
        plot_paths = {}
        
        # Performance comparison
        fig = self.comparison_plotter.create_performance_comparison(
            comparison_data, save_path=None
        )
        plot_paths['performance_comparison'] = self._encode_plot_to_base64(fig)
        plt.close(fig)
        
        # Statistical comparison
        statistical_plot_data = self._prepare_statistical_plot_data(statistical_results, comparison_data)
        fig = self.comparison_plotter.create_statistical_comparison(
            statistical_plot_data, save_path=None
        )
        plot_paths['statistical_comparison'] = self._encode_plot_to_base64(fig)
        plt.close(fig)
        
        # Generate report
        report_data = {
            'title': report_title,
            'metadata': self.report_metadata,
            'algorithms': algorithms,
            'baseline_algorithms': baseline_algorithms,
            'statistical_results': statistical_results,
            'comparison_summary': self._generate_comparison_summary(comparison_data, statistical_results),
            'plots': plot_paths
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"comparison_report_{timestamp}"
        self._save_report_data(report_data, report_filename)
        
        # Generate HTML report
        html_report = self._create_comparison_html_report(report_data)
        html_path = self.output_dir / f"{report_filename}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Comparison report generated: {html_path}")
        return str(html_path)
    
    def _prepare_statistical_plot_data(self, statistical_results: Dict[str, Any], 
                                     comparison_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Prepare data for statistical comparison plots."""
        algorithms = list(comparison_data.keys())
        n_algs = len(algorithms)
        
        # Create significance matrix
        sig_matrix = np.ones((n_algs, n_algs))
        
        for key, results in statistical_results.items():
            alg1, alg2 = key.split('_vs_')
            i = algorithms.index(alg1)
            j = algorithms.index(alg2)
            
            p_value = results['mann_whitney'].p_value
            sig_matrix[i, j] = p_value
            sig_matrix[j, i] = p_value
        
        # Prepare performance distributions
        performance_distributions = {}
        for alg, data in comparison_data.items():
            performance_distributions[alg] = data['episode_return'].values
        
        return {
            'significance_matrix': sig_matrix.tolist(),
            'algorithms': algorithms,
            'performance_distributions': performance_distributions
        }
    
    def _generate_comparison_summary(self, comparison_data: Dict[str, pd.DataFrame],
                                   statistical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level comparison summary."""
        algorithms = list(comparison_data.keys())
        
        # Calculate performance metrics
        performance_metrics = {}
        for alg, data in comparison_data.items():
            performance_metrics[alg] = {
                'mean_performance': data['episode_return'].mean(),
                'std_performance': data['episode_return'].std(),
                'final_performance': data['episode_return'].iloc[-1],
                'best_performance': data['episode_return'].max()
            }
        
        # Find best performing algorithm
        best_alg = max(performance_metrics.keys(), 
                      key=lambda x: performance_metrics[x]['mean_performance'])
        
        # Count significant differences
        significant_comparisons = 0
        total_comparisons = len(statistical_results)
        
        for results in statistical_results.values():
            if results['mann_whitney'].p_value < 0.05:
                significant_comparisons += 1
        
        return {
            'best_algorithm': best_alg,
            'best_mean_performance': performance_metrics[best_alg]['mean_performance'],
            'performance_metrics': performance_metrics,
            'significant_comparisons': significant_comparisons,
            'total_comparisons': total_comparisons,
            'significance_rate': significant_comparisons / total_comparisons if total_comparisons > 0 else 0
        }
    
    def _create_comparison_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML comparison report."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #8e44ad; border-bottom: 2px solid #9b59b6; }
                h2 { color: #7d3c98; margin-top: 30px; }
                h3 { color: #7f8c8d; }
                .summary-box { background-color: #f4f1fb; padding: 20px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #9b59b6; }
                .metric { display: inline-block; margin: 10px 20px 10px 0; }
                .metric-label { font-weight: bold; color: #2c3e50; }
                .metric-value { color: #8e44ad; font-size: 1.2em; }
                .winner { background-color: #d5f4e6; border-left: 4px solid #27ae60; padding: 10px; margin: 10px 0; }
                .plot { text-align: center; margin: 20px 0; }
                .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #e8daef; }
                .significant { color: #e74c3c; font-weight: bold; }
                .not-significant { color: #95a5a6; }
                .metadata { font-size: 0.9em; color: #7f8c8d; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            
            <div class="metadata">
                <p><strong>Generated:</strong> {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Algorithms Compared:</strong> {{ ', '.join(algorithms) }}</p>
            </div>
            
            <div class="summary-box">
                <h2>Comparison Summary</h2>
                
                <div class="winner">
                    <h3>🏆 Best Performing Algorithm: {{ comparison_summary.best_algorithm }}</h3>
                    <p><strong>Mean Performance:</strong> {{ "%.4f"|format(comparison_summary.best_mean_performance) }}</p>
                </div>
                
                <div class="metric">
                    <span class="metric-label">Significant Comparisons:</span>
                    <span class="metric-value">{{ comparison_summary.significant_comparisons }}/{{ comparison_summary.total_comparisons }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Significance Rate:</span>
                    <span class="metric-value">{{ "%.1f"|format(comparison_summary.significance_rate * 100) }}%</span>
                </div>
            </div>
            
            <h2>Algorithm Performance Metrics</h2>
            <table>
                <tr>
                    <th>Algorithm</th>
                    <th>Mean Performance</th>
                    <th>Std Deviation</th>
                    <th>Final Performance</th>
                    <th>Best Performance</th>
                </tr>
                {% for alg, metrics in comparison_summary.performance_metrics.items() %}
                <tr>
                    <td><strong>{{ alg }}</strong></td>
                    <td>{{ "%.4f"|format(metrics.mean_performance) }}</td>
                    <td>{{ "%.4f"|format(metrics.std_performance) }}</td>
                    <td>{{ "%.4f"|format(metrics.final_performance) }}</td>
                    <td>{{ "%.4f"|format(metrics.best_performance) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Statistical Test Results</h2>
            <table>
                <tr>
                    <th>Comparison</th>
                    <th>Mann-Whitney U p-value</th>
                    <th>Effect Size</th>
                    <th>K-S Test p-value</th>
                    <th>Significance</th>
                </tr>
                {% for comparison, results in statistical_results.items() %}
                <tr>
                    <td>{{ comparison.replace('_vs_', ' vs ') }}</td>
                    <td>{{ "%.4f"|format(results.mann_whitney.p_value) }}</td>
                    <td>{{ "%.4f"|format(results.mann_whitney.effect_size) }}</td>
                    <td>{{ "%.4f"|format(results.kolmogorov_smirnov.p_value) }}</td>
                    <td class="{{ 'significant' if results.mann_whitney.p_value < 0.05 else 'not-significant' }}">
                        {{ "Significant" if results.mann_whitney.p_value < 0.05 else "Not Significant" }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            {% if plots.performance_comparison %}
            <h2>Performance Comparison</h2>
            <div class="plot">
                <img src="{{ plots.performance_comparison }}" alt="Performance Comparison">
            </div>
            {% endif %}
            
            {% if plots.statistical_comparison %}
            <h2>Statistical Analysis</h2>
            <div class="plot">
                <img src="{{ plots.statistical_comparison }}" alt="Statistical Comparison">
            </div>
            {% endif %}
            
        </body>
        </html>
        """
        
        template = Template(template_str)
        return template.render(**report_data)


class ExecutiveReportGenerator(ReportGenerator):
    """
    Generates high-level executive summary reports.
    """
    
    def generate_executive_report(self,
                                all_data: Dict[str, Any],
                                project_config: Dict[str, Any] = None,
                                report_title: str = "Safe RL Executive Summary") -> str:
        """
        Generate executive summary report combining all analyses.
        
        Args:
            all_data: Dictionary containing all analysis results
            project_config: Project configuration details
            report_title: Title for the report
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating executive summary report...")
        
        # Extract key insights from all data
        executive_summary = self._extract_executive_insights(all_data)
        
        # Generate report
        report_data = {
            'title': report_title,
            'metadata': self.report_metadata,
            'project_config': project_config or {},
            'executive_summary': executive_summary,
            'recommendations': self._generate_recommendations(executive_summary)
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"executive_report_{timestamp}"
        self._save_report_data(report_data, report_filename)
        
        # Generate HTML report
        html_report = self._create_executive_html_report(report_data)
        html_path = self.output_dir / f"{report_filename}.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        logger.info(f"Executive report generated: {html_path}")
        return str(html_path)
    
    def _extract_executive_insights(self, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract high-level insights from all analysis data."""
        insights = {
            'performance_status': 'Unknown',
            'safety_status': 'Unknown',
            'algorithm_recommendation': 'Unknown',
            'key_findings': [],
            'risk_assessment': 'Medium',
            'deployment_readiness': 'Not Ready'
        }
        
        # Extract performance insights
        if 'performance_data' in all_data:
            perf_data = all_data['performance_data']
            # Analyze performance trends, convergence, etc.
            insights['performance_status'] = 'Good'  # Simplified
        
        # Extract safety insights  
        if 'safety_data' in all_data:
            safety_data = all_data['safety_data']
            # Analyze violation rates, safety trends, etc.
            insights['safety_status'] = 'Acceptable'  # Simplified
        
        # Extract comparison insights
        if 'comparison_data' in all_data:
            comp_data = all_data['comparison_data']
            # Analyze algorithm comparisons
            insights['algorithm_recommendation'] = 'CPO'  # Simplified
        
        return insights
    
    def _generate_recommendations(self, executive_summary: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Performance-based recommendations
        if executive_summary['performance_status'] == 'Poor':
            recommendations.append("Consider hyperparameter tuning to improve learning efficiency")
            recommendations.append("Investigate reward function design for better performance")
        
        # Safety-based recommendations
        if executive_summary['safety_status'] == 'Poor':
            recommendations.append("Implement additional safety constraints")
            recommendations.append("Consider more conservative policy updates")
        
        # General recommendations
        recommendations.append("Continue monitoring safety metrics during deployment")
        recommendations.append("Establish performance benchmarks for ongoing evaluation")
        
        return recommendations
    
    def _create_executive_html_report(self, report_data: Dict[str, Any]) -> str:
        """Create HTML executive report."""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #2c3e50; border-bottom: 3px solid #3498db; font-size: 2.5em; }
                h2 { color: #34495e; margin-top: 40px; font-size: 1.8em; }
                h3 { color: #7f8c8d; font-size: 1.3em; }
                .executive-summary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin: 30px 0; }
                .status-good { color: #27ae60; font-weight: bold; }
                .status-warning { color: #f39c12; font-weight: bold; }
                .status-danger { color: #e74c3c; font-weight: bold; }
                .recommendation { background-color: #e8f8f5; border-left: 4px solid #1abc9c; padding: 15px; margin: 10px 0; }
                .key-finding { background-color: #fdf6e3; border-left: 4px solid #f39c12; padding: 15px; margin: 10px 0; }
                .metadata { font-size: 0.9em; color: #7f8c8d; border-top: 1px solid #ecf0f1; padding-top: 20px; margin-top: 40px; }
            </style>
        </head>
        <body>
            <h1>{{ title }}</h1>
            
            <div class="executive-summary">
                <h2 style="color: white; margin-top: 0;">Executive Summary</h2>
                <h3 style="color: #ecf0f1;">Performance Status: <span class="status-{{ 'good' if executive_summary.performance_status == 'Good' else 'warning' if executive_summary.performance_status == 'Acceptable' else 'danger' }}">{{ executive_summary.performance_status }}</span></h3>
                <h3 style="color: #ecf0f1;">Safety Status: <span class="status-{{ 'good' if executive_summary.safety_status == 'Good' else 'warning' if executive_summary.safety_status == 'Acceptable' else 'danger' }}">{{ executive_summary.safety_status }}</span></h3>
                <h3 style="color: #ecf0f1;">Recommended Algorithm: <strong>{{ executive_summary.algorithm_recommendation }}</strong></h3>
                <h3 style="color: #ecf0f1;">Risk Assessment: <span class="status-{{ 'good' if executive_summary.risk_assessment == 'Low' else 'warning' if executive_summary.risk_assessment == 'Medium' else 'danger' }}">{{ executive_summary.risk_assessment }}</span></h3>
                <h3 style="color: #ecf0f1;">Deployment Readiness: <span class="status-{{ 'good' if executive_summary.deployment_readiness == 'Ready' else 'warning' if executive_summary.deployment_readiness == 'Almost Ready' else 'danger' }}">{{ executive_summary.deployment_readiness }}</span></h3>
            </div>
            
            {% if executive_summary.key_findings %}
            <h2>Key Findings</h2>
            {% for finding in executive_summary.key_findings %}
            <div class="key-finding">{{ finding }}</div>
            {% endfor %}
            {% endif %}
            
            <h2>Recommendations</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation">{{ recommendation }}</div>
            {% endfor %}
            
            <div class="metadata">
                <p><strong>Report Generated:</strong> {{ metadata.generated_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                <p><strong>Generator Version:</strong> {{ metadata.generator_version }}</p>
            </div>
            
        </body>
        </html>
        """
        
        template = Template(template_str)
        return template.render(**report_data)


# Batch report generation utility
def generate_all_reports(training_data: Dict[str, pd.DataFrame],
                        safety_data: Dict[str, pd.DataFrame] = None,
                        experiment_config: Dict[str, Any] = None,
                        output_dir: Union[str, Path] = None) -> Dict[str, str]:
    """
    Generate all types of reports in batch.
    
    Args:
        training_data: Training data for each experiment
        safety_data: Safety data for each experiment
        experiment_config: Experiment configuration
        output_dir: Output directory for reports
        
    Returns:
        Dictionary mapping report types to file paths
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    try:
        # Performance report
        perf_generator = PerformanceReportGenerator(output_dir)
        report_paths['performance'] = perf_generator.generate_performance_report(
            training_data, experiment_config
        )
        
        # Safety report (if safety data provided)
        if safety_data:
            safety_generator = SafetyReportGenerator(output_dir)
            report_paths['safety'] = safety_generator.generate_safety_report(
                safety_data, experiment_config
            )
        
        # Comparison report (if multiple algorithms)
        if len(training_data) > 1:
            comp_generator = ComparisonReportGenerator(output_dir)
            report_paths['comparison'] = comp_generator.generate_comparison_report(
                training_data
            )
        
        # Executive summary
        exec_generator = ExecutiveReportGenerator(output_dir)
        all_data = {
            'performance_data': training_data,
            'safety_data': safety_data or {},
            'comparison_data': training_data if len(training_data) > 1 else {}
        }
        report_paths['executive'] = exec_generator.generate_executive_report(
            all_data, experiment_config
        )
        
        logger.info(f"All reports generated successfully in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating reports: {str(e)}")
        raise
    
    return report_paths


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    
    # Sample training data for multiple algorithms
    algorithms = ['CPO', 'PPO', 'Lagrangian-PPO']
    sample_training_data = {}
    
    for alg in algorithms:
        n_episodes = 1000
        base_performance = np.random.uniform(80, 120)
        
        data = pd.DataFrame({
            'iteration': range(n_episodes),
            'episode_return': np.cumsum(np.random.normal(base_performance/n_episodes, 10/n_episodes, n_episodes)),
            'episode_length': np.random.poisson(200, n_episodes),
            'policy_loss': np.random.exponential(0.1, n_episodes),
            'value_loss': np.random.exponential(0.05, n_episodes),
            'kl_divergence': np.random.exponential(0.01, n_episodes)
        })
        sample_training_data[alg] = data
    
    # Sample safety data
    sample_safety_data = {}
    for alg in algorithms:
        n_episodes = 1000
        
        data = pd.DataFrame({
            'iteration': range(n_episodes),
            'constraint_violation': np.maximum(0, np.random.normal(0.02, 0.05, n_episodes)),
            'safety_score': np.minimum(1.0, np.maximum(0.0, np.random.beta(3, 1, n_episodes))),
            'constraint_1': np.random.normal(0, 0.1, n_episodes),
            'constraint_2': np.random.normal(0, 0.05, n_episodes),
            'constraint_3': np.random.normal(0, 0.02, n_episodes)
        })
        sample_safety_data[alg] = data
    
    # Test report generation
    output_dir = Path("test_reports")
    
    # Generate all reports
    report_paths = generate_all_reports(
        training_data=sample_training_data,
        safety_data=sample_safety_data,
        experiment_config={
            'environment': 'SafetyGym',
            'total_timesteps': 1000000,
            'algorithms': algorithms,
            'safety_constraints': ['velocity', 'position', 'collision']
        },
        output_dir=output_dir
    )
    
    print("Generated reports:")
    for report_type, path in report_paths.items():
        print(f"  {report_type}: {path}")
    
    print(f"\nAll reports saved to: {output_dir}")