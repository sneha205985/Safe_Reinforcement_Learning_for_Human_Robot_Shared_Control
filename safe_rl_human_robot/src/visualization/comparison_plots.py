"""
Comparison and benchmarking visualization plots for Safe RL analysis.

This module provides comprehensive comparison visualization capabilities including
algorithm performance comparisons, baseline benchmarking, and Pareto frontier analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path

# Configure matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComparisonPlotter:
    """
    Creates comprehensive comparison plots between different algorithms and configurations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def create_performance_comparison(self, 
                                    comparison_data: Dict[str, pd.DataFrame],
                                    metrics: List[str] = None,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive performance comparison plots.
        
        Args:
            comparison_data: Dict mapping algorithm names to their performance data
            metrics: List of metrics to compare ['episode_return', 'success_rate', etc.]
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = ['episode_return', 'episode_length', 'success_rate', 'sample_efficiency']
            
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
        axes = axes.flatten()
        
        algorithms = list(comparison_data.keys())
        
        for idx, metric in enumerate(metrics[:4]):  # Limit to 4 metrics for 2x2 layout
            ax = axes[idx]
            
            # Learning curves with confidence intervals
            for alg_idx, (alg_name, data) in enumerate(comparison_data.items()):
                if metric not in data.columns:
                    logger.warning(f"Metric {metric} not found in {alg_name} data")
                    continue
                    
                # Calculate rolling statistics
                window = max(1, len(data) // 20)  # Adaptive window size
                rolling_mean = data[metric].rolling(window=window).mean()
                rolling_std = data[metric].rolling(window=window).std()
                
                iterations = range(len(rolling_mean))
                color = self.colors[alg_idx % len(self.colors)]
                
                # Plot mean line
                ax.plot(iterations, rolling_mean, 
                       color=color, linewidth=2, label=alg_name, alpha=0.8)
                
                # Plot confidence interval
                ax.fill_between(iterations, 
                              rolling_mean - rolling_std,
                              rolling_mean + rolling_std,
                              color=color, alpha=0.2)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Performance comparison plot saved to {save_path}")
            
        return fig
    
    def create_algorithm_radar_chart(self,
                                   algorithm_scores: Dict[str, Dict[str, float]],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create radar chart comparing algorithms across multiple dimensions.
        
        Args:
            algorithm_scores: Dict mapping algorithm names to metric scores
            save_path: Path to save the interactive plot
            
        Returns:
            Plotly figure object
        """
        # Extract metrics from first algorithm
        metrics = list(next(iter(algorithm_scores.values())).keys())
        
        fig = go.Figure()
        
        for alg_name, scores in algorithm_scores.items():
            fig.add_trace(go.Scatterpolar(
                r=list(scores.values()),
                theta=metrics,
                fill='toself',
                name=alg_name,
                line_color=self.colors[len(fig.data) % len(self.colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]  # Assuming normalized scores
                )
            ),
            showlegend=True,
            title="Algorithm Performance Radar Chart",
            font=dict(size=12)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Radar chart saved to {save_path}")
            
        return fig
    
    def create_statistical_comparison(self,
                                    comparison_results: Dict[str, Any],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create statistical comparison visualization with significance tests.
        
        Args:
            comparison_results: Results from statistical comparison analysis
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        
        # Box plots for performance distribution
        ax1 = axes[0, 0]
        if 'performance_distributions' in comparison_results:
            performance_data = comparison_results['performance_distributions']
            box_data = [data for data in performance_data.values()]
            box_labels = list(performance_data.keys())
            
            bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], self.colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax1.set_title('Performance Distribution Comparison')
        ax1.set_ylabel('Episode Return')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Statistical significance matrix
        ax2 = axes[0, 1]
        if 'significance_matrix' in comparison_results:
            sig_matrix = comparison_results['significance_matrix']
            im = ax2.imshow(sig_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.05)
            
            # Add text annotations
            for i in range(len(sig_matrix)):
                for j in range(len(sig_matrix[0])):
                    text = ax2.text(j, i, f'{sig_matrix[i][j]:.3f}',
                                   ha="center", va="center", color="black")
            
            ax2.set_title('Statistical Significance (p-values)')
            algorithms = comparison_results.get('algorithms', [f'Alg{i}' for i in range(len(sig_matrix))])
            ax2.set_xticks(range(len(algorithms)))
            ax2.set_yticks(range(len(algorithms)))
            ax2.set_xticklabels(algorithms, rotation=45)
            ax2.set_yticklabels(algorithms)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('p-value')
        
        # Effect sizes
        ax3 = axes[1, 0]
        if 'effect_sizes' in comparison_results:
            effect_data = comparison_results['effect_sizes']
            algorithms = list(effect_data.keys())
            effect_values = list(effect_data.values())
            
            bars = ax3.bar(algorithms, effect_values, 
                          color=self.colors[:len(algorithms)], alpha=0.7)
            ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect')
            ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium effect')
            ax3.axhline(y=0.8, color='darkred', linestyle='--', alpha=0.7, label='Large effect')
            
            ax3.set_title('Effect Sizes (Cohen\'s d)')
            ax3.set_ylabel('Effect Size')
            ax3.legend()
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Confidence intervals
        ax4 = axes[1, 1]
        if 'confidence_intervals' in comparison_results:
            ci_data = comparison_results['confidence_intervals']
            algorithms = list(ci_data.keys())
            means = [ci['mean'] for ci in ci_data.values()]
            lower = [ci['lower'] for ci in ci_data.values()]
            upper = [ci['upper'] for ci in ci_data.values()]
            
            y_pos = np.arange(len(algorithms))
            
            ax4.barh(y_pos, means, xerr=[np.array(means) - np.array(lower),
                                        np.array(upper) - np.array(means)],
                    capsize=5, color=self.colors[:len(algorithms)], alpha=0.7)
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(algorithms)
            ax4.set_xlabel('Performance (95% CI)')
            ax4.set_title('Performance Confidence Intervals')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Statistical comparison plot saved to {save_path}")
            
        return fig


class BaselinePlotter:
    """
    Specialized plotter for baseline algorithm comparisons.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
    def create_baseline_performance_plot(self,
                                       baseline_results: Dict[str, Any],
                                       target_algorithm: str = 'CPO',
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive baseline comparison plot.
        
        Args:
            baseline_results: Results from baseline comparison analysis
            target_algorithm: Name of the target algorithm to highlight
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=self.dpi)
        
        # Performance vs Safety scatter plot
        ax1 = axes[0, 0]
        if 'performance_safety_data' in baseline_results:
            data = baseline_results['performance_safety_data']
            
            for alg_name, metrics in data.items():
                performance = metrics.get('performance', 0)
                safety = metrics.get('safety', 0)
                
                marker = 'o' if alg_name != target_algorithm else '*'
                size = 100 if alg_name != target_algorithm else 200
                
                ax1.scatter(performance, safety, label=alg_name, 
                           marker=marker, s=size, alpha=0.7)
            
            ax1.set_xlabel('Performance Score')
            ax1.set_ylabel('Safety Score')
            ax1.set_title('Performance vs Safety Trade-off')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Sample efficiency comparison
        ax2 = axes[0, 1]
        if 'sample_efficiency' in baseline_results:
            efficiency_data = baseline_results['sample_efficiency']
            algorithms = list(efficiency_data.keys())
            efficiency_values = list(efficiency_data.values())
            
            bars = ax2.bar(algorithms, efficiency_values, alpha=0.7)
            
            # Highlight target algorithm
            for i, alg in enumerate(algorithms):
                if alg == target_algorithm:
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.9)
            
            ax2.set_title('Sample Efficiency Comparison')
            ax2.set_ylabel('Sample Efficiency Score')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # Training time comparison
        ax3 = axes[0, 2]
        if 'training_times' in baseline_results:
            time_data = baseline_results['training_times']
            algorithms = list(time_data.keys())
            times = list(time_data.values())
            
            bars = ax3.bar(algorithms, times, alpha=0.7)
            
            # Highlight target algorithm
            for i, alg in enumerate(algorithms):
                if alg == target_algorithm:
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.9)
            
            ax3.set_title('Training Time Comparison')
            ax3.set_ylabel('Training Time (hours)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # Constraint violation rates
        ax4 = axes[1, 0]
        if 'violation_rates' in baseline_results:
            violation_data = baseline_results['violation_rates']
            algorithms = list(violation_data.keys())
            rates = list(violation_data.values())
            
            bars = ax4.bar(algorithms, rates, alpha=0.7, color='red')
            
            # Highlight target algorithm
            for i, alg in enumerate(algorithms):
                if alg == target_algorithm:
                    bars[i].set_color('darkred')
                    bars[i].set_alpha(0.9)
            
            ax4.set_title('Constraint Violation Rates')
            ax4.set_ylabel('Violation Rate (%)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        # Success rate comparison
        ax5 = axes[1, 1]
        if 'success_rates' in baseline_results:
            success_data = baseline_results['success_rates']
            algorithms = list(success_data.keys())
            rates = list(success_data.values())
            
            bars = ax5.bar(algorithms, rates, alpha=0.7, color='green')
            
            # Highlight target algorithm
            for i, alg in enumerate(algorithms):
                if alg == target_algorithm:
                    bars[i].set_color('darkgreen')
                    bars[i].set_alpha(0.9)
            
            ax5.set_title('Success Rate Comparison')
            ax5.set_ylabel('Success Rate (%)')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        
        # Overall ranking
        ax6 = axes[1, 2]
        if 'overall_rankings' in baseline_results:
            ranking_data = baseline_results['overall_rankings']
            algorithms = list(ranking_data.keys())
            scores = list(ranking_data.values())
            
            bars = ax6.barh(algorithms, scores, alpha=0.7)
            
            # Highlight target algorithm
            for i, alg in enumerate(algorithms):
                if alg == target_algorithm:
                    bars[i].set_color('gold')
                    bars[i].set_alpha(0.9)
            
            ax6.set_title('Overall Performance Ranking')
            ax6.set_xlabel('Composite Score')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Baseline comparison plot saved to {save_path}")
            
        return fig
    
    def create_environment_comparison(self,
                                    environment_results: Dict[str, Dict[str, Any]],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison across different environments.
        
        Args:
            environment_results: Results for each environment
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        environments = list(environment_results.keys())
        n_envs = len(environments)
        
        fig, axes = plt.subplots(2, max(2, n_envs), figsize=(5*n_envs, 10), dpi=self.dpi)
        if n_envs == 1:
            axes = axes.reshape(-1, 1)
        
        for env_idx, (env_name, results) in enumerate(environment_results.items()):
            # Performance comparison for this environment
            ax1 = axes[0, env_idx % axes.shape[1]]
            
            if 'algorithm_performance' in results:
                perf_data = results['algorithm_performance']
                algorithms = list(perf_data.keys())
                performance = list(perf_data.values())
                
                bars = ax1.bar(algorithms, performance, alpha=0.7)
                ax1.set_title(f'{env_name}\nPerformance Comparison')
                ax1.set_ylabel('Episode Return')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
            
            # Safety comparison for this environment
            ax2 = axes[1, env_idx % axes.shape[1]]
            
            if 'algorithm_safety' in results:
                safety_data = results['algorithm_safety']
                algorithms = list(safety_data.keys())
                safety_scores = list(safety_data.values())
                
                bars = ax2.bar(algorithms, safety_scores, alpha=0.7, color='red')
                ax2.set_title(f'{env_name}\nSafety Comparison')
                ax2.set_ylabel('Safety Score')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
        
        # Remove extra subplots if needed
        for i in range(n_envs, axes.shape[1]):
            axes[0, i].remove()
            axes[1, i].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Environment comparison plot saved to {save_path}")
            
        return fig


class ParetoFrontierPlotter:
    """
    Specialized plotter for Pareto frontier analysis in performance vs safety trade-offs.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
    def create_pareto_frontier_plot(self,
                                  pareto_data: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create Pareto frontier visualization.
        
        Args:
            pareto_data: Data containing algorithm performance and safety scores
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=self.dpi)
        
        # Extract data
        algorithms = list(pareto_data.keys())
        performance_scores = [data.get('performance', 0) for data in pareto_data.values()]
        safety_scores = [data.get('safety', 0) for data in pareto_data.values()]
        
        # Main Pareto frontier plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, (alg, perf, safety) in enumerate(zip(algorithms, performance_scores, safety_scores)):
            marker_size = 150 if 'CPO' in alg else 100
            marker = '*' if 'CPO' in alg else 'o'
            
            ax1.scatter(perf, safety, c=[colors[i]], s=marker_size, 
                       marker=marker, alpha=0.8, label=alg, edgecolors='black', linewidth=1)
        
        # Calculate and plot Pareto frontier
        pareto_points = self._calculate_pareto_frontier(performance_scores, safety_scores)
        if len(pareto_points) > 1:
            pareto_perf = [performance_scores[i] for i in pareto_points]
            pareto_safety = [safety_scores[i] for i in pareto_points]
            
            # Sort points for proper line connection
            sorted_points = sorted(zip(pareto_perf, pareto_safety))
            pareto_perf_sorted, pareto_safety_sorted = zip(*sorted_points)
            
            ax1.plot(pareto_perf_sorted, pareto_safety_sorted, 'r--', 
                    linewidth=2, alpha=0.7, label='Pareto Frontier')
        
        ax1.set_xlabel('Performance Score', fontsize=12)
        ax1.set_ylabel('Safety Score', fontsize=12)
        ax1.set_title('Performance vs Safety Trade-off\n(Pareto Frontier Analysis)', fontsize=14)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal lines for trade-off reference
        max_perf, max_safety = max(performance_scores), max(safety_scores)
        min_perf, min_safety = min(performance_scores), min(safety_scores)
        
        x_range = np.linspace(min_perf, max_perf, 100)
        ax1.plot(x_range, min_safety + (max_safety - min_safety) * 
                (max_perf - x_range) / (max_perf - min_perf), 
                'gray', linestyle=':', alpha=0.5, label='Trade-off Line')
        
        # Pareto efficiency ranking
        efficiency_scores = []
        for i, (perf, safety) in enumerate(zip(performance_scores, safety_scores)):
            # Distance to ideal point (max performance, max safety)
            ideal_distance = np.sqrt((max_perf - perf)**2 + (max_safety - safety)**2)
            # Distance to worst point (min performance, min safety)  
            worst_distance = np.sqrt((max_perf - min_perf)**2 + (max_safety - min_safety)**2)
            # Efficiency as inverse of normalized distance to ideal
            efficiency = 1 - (ideal_distance / worst_distance) if worst_distance > 0 else 1
            efficiency_scores.append(efficiency)
        
        # Efficiency bar chart
        sorted_indices = sorted(range(len(algorithms)), key=lambda i: efficiency_scores[i], reverse=True)
        sorted_algorithms = [algorithms[i] for i in sorted_indices]
        sorted_efficiency = [efficiency_scores[i] for i in sorted_indices]
        
        bars = ax2.bar(range(len(sorted_algorithms)), sorted_efficiency, 
                      color=[colors[sorted_indices[i]] for i in range(len(sorted_algorithms))],
                      alpha=0.8)
        
        # Highlight CPO algorithms
        for i, alg in enumerate(sorted_algorithms):
            if 'CPO' in alg:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(3)
        
        ax2.set_xlabel('Algorithms', fontsize=12)
        ax2.set_ylabel('Pareto Efficiency Score', fontsize=12)
        ax2.set_title('Algorithm Pareto Efficiency Ranking', fontsize=14)
        ax2.set_xticks(range(len(sorted_algorithms)))
        ax2.set_xticklabels(sorted_algorithms, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add efficiency score annotations
        for i, score in enumerate(sorted_efficiency):
            ax2.text(i, score + 0.01, f'{score:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Pareto frontier plot saved to {save_path}")
            
        return fig
    
    def _calculate_pareto_frontier(self, performance_scores: List[float], 
                                 safety_scores: List[float]) -> List[int]:
        """
        Calculate Pareto frontier points.
        
        Args:
            performance_scores: Performance values for each algorithm
            safety_scores: Safety values for each algorithm
            
        Returns:
            Indices of points on the Pareto frontier
        """
        pareto_points = []
        n_points = len(performance_scores)
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Point j dominates point i if j is better in both dimensions
                    if (performance_scores[j] >= performance_scores[i] and 
                        safety_scores[j] >= safety_scores[i] and
                        (performance_scores[j] > performance_scores[i] or 
                         safety_scores[j] > safety_scores[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_points.append(i)
        
        return pareto_points
    
    def create_interactive_pareto_plot(self,
                                     pareto_data: Dict[str, Any],
                                     save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Pareto frontier plot using Plotly.
        
        Args:
            pareto_data: Data containing algorithm performance and safety scores
            save_path: Path to save the interactive plot
            
        Returns:
            Plotly figure object
        """
        algorithms = list(pareto_data.keys())
        performance_scores = [data.get('performance', 0) for data in pareto_data.values()]
        safety_scores = [data.get('safety', 0) for data in pareto_data.values()]
        
        # Create scatter plot
        fig = go.Figure()
        
        for i, alg in enumerate(algorithms):
            marker_size = 15 if 'CPO' in alg else 10
            marker_symbol = 'star' if 'CPO' in alg else 'circle'
            
            fig.add_trace(go.Scatter(
                x=[performance_scores[i]],
                y=[safety_scores[i]],
                mode='markers',
                name=alg,
                marker=dict(
                    size=marker_size,
                    symbol=marker_symbol,
                    line=dict(width=2, color='black')
                ),
                hovertemplate=f'<b>{alg}</b><br>' +
                             'Performance: %{x:.3f}<br>' +
                             'Safety: %{y:.3f}<br>' +
                             '<extra></extra>'
            ))
        
        # Add Pareto frontier
        pareto_points = self._calculate_pareto_frontier(performance_scores, safety_scores)
        if len(pareto_points) > 1:
            pareto_perf = [performance_scores[i] for i in pareto_points]
            pareto_safety = [safety_scores[i] for i in pareto_points]
            
            # Sort points for proper line connection
            sorted_points = sorted(zip(pareto_perf, pareto_safety))
            pareto_perf_sorted, pareto_safety_sorted = zip(*sorted_points)
            
            fig.add_trace(go.Scatter(
                x=pareto_perf_sorted,
                y=pareto_safety_sorted,
                mode='lines',
                name='Pareto Frontier',
                line=dict(color='red', width=3, dash='dash'),
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Interactive Performance vs Safety Trade-off<br><sub>Pareto Frontier Analysis</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Performance Score',
            yaxis_title='Safety Score',
            hovermode='closest',
            showlegend=True,
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive Pareto plot saved to {save_path}")
            
        return fig


def save_all_comparison_plots(comparison_data: Dict[str, Any],
                            output_dir: Union[str, Path],
                            formats: List[str] = ['png', 'svg', 'html']) -> Dict[str, str]:
    """
    Save all comparison plots to specified directory.
    
    Args:
        comparison_data: Comprehensive comparison analysis results
        output_dir: Directory to save plots
        formats: List of formats to save ['png', 'svg', 'html']
        
    Returns:
        Dictionary mapping plot names to saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Initialize plotters
    comparison_plotter = ComparisonPlotter()
    baseline_plotter = BaselinePlotter()
    pareto_plotter = ParetoFrontierPlotter()
    
    try:
        # Performance comparison plots
        if 'algorithm_data' in comparison_data:
            for fmt in formats:
                if fmt == 'html':
                    continue
                filename = f'performance_comparison.{fmt}'
                filepath = output_dir / filename
                fig = comparison_plotter.create_performance_comparison(
                    comparison_data['algorithm_data'], 
                    save_path=str(filepath)
                )
                saved_files[f'performance_comparison_{fmt}'] = str(filepath)
                plt.close(fig)
        
        # Radar chart (HTML only)
        if 'radar_data' in comparison_data and 'html' in formats:
            filename = 'algorithm_radar_chart.html'
            filepath = output_dir / filename
            fig = comparison_plotter.create_algorithm_radar_chart(
                comparison_data['radar_data'],
                save_path=str(filepath)
            )
            saved_files['radar_chart'] = str(filepath)
        
        # Statistical comparison
        if 'statistical_results' in comparison_data:
            for fmt in formats:
                if fmt == 'html':
                    continue
                filename = f'statistical_comparison.{fmt}'
                filepath = output_dir / filename
                fig = comparison_plotter.create_statistical_comparison(
                    comparison_data['statistical_results'],
                    save_path=str(filepath)
                )
                saved_files[f'statistical_comparison_{fmt}'] = str(filepath)
                plt.close(fig)
        
        # Baseline comparison
        if 'baseline_results' in comparison_data:
            for fmt in formats:
                if fmt == 'html':
                    continue
                filename = f'baseline_comparison.{fmt}'
                filepath = output_dir / filename
                fig = baseline_plotter.create_baseline_performance_plot(
                    comparison_data['baseline_results'],
                    save_path=str(filepath)
                )
                saved_files[f'baseline_comparison_{fmt}'] = str(filepath)
                plt.close(fig)
        
        # Environment comparison
        if 'environment_results' in comparison_data:
            for fmt in formats:
                if fmt == 'html':
                    continue
                filename = f'environment_comparison.{fmt}'
                filepath = output_dir / filename
                fig = baseline_plotter.create_environment_comparison(
                    comparison_data['environment_results'],
                    save_path=str(filepath)
                )
                saved_files[f'environment_comparison_{fmt}'] = str(filepath)
                plt.close(fig)
        
        # Pareto frontier analysis
        if 'pareto_data' in comparison_data:
            # Static plot
            for fmt in formats:
                if fmt == 'html':
                    continue
                filename = f'pareto_frontier.{fmt}'
                filepath = output_dir / filename
                fig = pareto_plotter.create_pareto_frontier_plot(
                    comparison_data['pareto_data'],
                    save_path=str(filepath)
                )
                saved_files[f'pareto_frontier_{fmt}'] = str(filepath)
                plt.close(fig)
            
            # Interactive plot
            if 'html' in formats:
                filename = 'pareto_frontier_interactive.html'
                filepath = output_dir / filename
                fig = pareto_plotter.create_interactive_pareto_plot(
                    comparison_data['pareto_data'],
                    save_path=str(filepath)
                )
                saved_files['pareto_frontier_interactive'] = str(filepath)
        
        logger.info(f"All comparison plots saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error saving comparison plots: {str(e)}")
        raise
    
    return saved_files


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    
    # Sample algorithm comparison data
    algorithms = ['CPO', 'PPO', 'Lagrangian-PPO', 'TRPO', 'Hand-crafted']
    sample_comparison_data = {}
    
    for alg in algorithms:
        n_episodes = 1000
        base_performance = np.random.uniform(0.5, 0.9)
        noise_level = 0.1
        
        data = pd.DataFrame({
            'episode_return': np.cumsum(np.random.normal(base_performance/n_episodes, noise_level/n_episodes, n_episodes)),
            'episode_length': np.random.poisson(100, n_episodes),
            'success_rate': np.minimum(1.0, np.maximum(0.0, 
                                     np.cumsum(np.random.normal(0.001, 0.0001, n_episodes)))),
            'sample_efficiency': np.random.exponential(0.8, n_episodes)
        })
        sample_comparison_data[alg] = data
    
    # Test comparison plots
    output_dir = Path("test_comparison_plots")
    output_dir.mkdir(exist_ok=True)
    
    # Test basic comparison
    plotter = ComparisonPlotter()
    fig = plotter.create_performance_comparison(
        sample_comparison_data, 
        save_path=str(output_dir / "test_performance_comparison.png")
    )
    plt.close(fig)
    
    # Test radar chart
    radar_data = {
        alg: {
            'Performance': np.random.uniform(0.5, 1.0),
            'Safety': np.random.uniform(0.4, 0.9),
            'Sample Efficiency': np.random.uniform(0.3, 0.8),
            'Stability': np.random.uniform(0.6, 0.95)
        }
        for alg in algorithms
    }
    
    fig = plotter.create_algorithm_radar_chart(
        radar_data,
        save_path=str(output_dir / "test_radar_chart.html")
    )
    
    # Test Pareto frontier
    pareto_plotter = ParetoFrontierPlotter()
    pareto_data = {
        alg: {
            'performance': np.random.uniform(0.3, 0.9),
            'safety': np.random.uniform(0.4, 0.85)
        }
        for alg in algorithms
    }
    
    fig = pareto_plotter.create_pareto_frontier_plot(
        pareto_data,
        save_path=str(output_dir / "test_pareto_frontier.png")
    )
    plt.close(fig)
    
    print("All test plots generated successfully!")