"""
Visualization Tools for Safe RL Evaluation Results.

This module provides comprehensive visualization capabilities for evaluation results,
including performance comparisons, statistical significance plots, and publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import gaussian_kde
import warnings

logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class ResultVisualizer:
    """Main class for visualizing evaluation results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", 10)
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.dpi': dpi,
            'savefig.dpi': dpi,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def create_performance_comparison(self, results: List, save_path: Optional[Path] = None):
        """Create comprehensive performance comparison visualization."""
        # Organize data
        data = self._organize_results_data(results)
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Performance Comparison', fontsize=16, y=0.98)
        
        # 1. Asymptotic Performance Box Plot
        self._plot_metric_boxplot(data, 'asymptotic_performance', axes[0, 0], 
                                'Asymptotic Performance', 'Performance Score')
        
        # 2. Sample Efficiency
        self._plot_metric_boxplot(data, 'sample_efficiency', axes[0, 1],
                                'Sample Efficiency', 'Steps to Target')
        
        # 3. Safety Violation Rate
        self._plot_metric_boxplot(data, 'violation_rate', axes[0, 2],
                                'Safety Violation Rate', 'Violation Rate')
        
        # 4. Performance vs Safety Scatter
        self._plot_performance_safety_scatter(data, axes[1, 0])
        
        # 5. Radar Chart for Multiple Metrics
        self._plot_radar_chart(data, axes[1, 1])
        
        # 6. Training Stability
        self._plot_metric_boxplot(data, 'training_stability', axes[1, 2],
                                'Training Stability', 'Stability Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Performance comparison saved to {save_path}")
        
        plt.show()
    
    def create_safety_analysis(self, results: List, save_path: Optional[Path] = None):
        """Create safety-focused analysis visualization."""
        data = self._organize_results_data(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Safety Analysis', fontsize=16, y=0.98)
        
        # 1. Safety Violation Distribution
        self._plot_safety_distribution(data, axes[0, 0])
        
        # 2. Constraint Satisfaction
        self._plot_constraint_satisfaction(data, axes[0, 1])
        
        # 3. Safety Margin Analysis
        self._plot_safety_margins(data, axes[1, 0])
        
        # 4. Safety vs Performance Trade-off
        self._plot_safety_performance_tradeoff(data, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Safety analysis saved to {save_path}")
        
        plt.show()
    
    def create_statistical_heatmap(self, statistical_results: Dict[str, Any], 
                                 save_path: Optional[Path] = None):
        """Create statistical significance heatmap."""
        if 'pairwise_comparisons' not in statistical_results:
            logger.warning("No pairwise comparison data found")
            return
        
        # Create heatmap for each metric
        metrics = list(statistical_results['pairwise_comparisons'].keys())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            logger.warning("No metrics found for heatmap")
            return
        
        # Calculate subplot layout
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle('Statistical Significance Analysis', fontsize=16, y=0.98)
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                self._plot_significance_heatmap(
                    statistical_results['pairwise_comparisons'][metric], 
                    axes[i], metric
                )
        
        # Hide unused subplots
        for j in range(n_metrics, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Statistical heatmap saved to {save_path}")
        
        plt.show()
    
    def create_learning_curves(self, results: List, save_path: Optional[Path] = None):
        """Create learning curve visualization."""
        # This would require time-series data from training
        # Placeholder implementation
        fig, ax = plt.subplots(figsize=self.figsize)
        
        algorithms = set(result.algorithm_name for result in results)
        
        for alg in algorithms:
            # Placeholder learning curve
            x = np.arange(100)
            y = np.random.cumsum(np.random.randn(100)) + x * 0.1
            ax.plot(x, y, label=alg, linewidth=2)
        
        ax.set_xlabel('Training Episodes')
        ax.set_ylabel('Cumulative Reward')
        ax.set_title('Learning Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Learning curves saved to {save_path}")
        
        plt.show()
    
    def create_human_metrics_analysis(self, results: List, save_path: Optional[Path] = None):
        """Create human-centric metrics visualization."""
        data = self._organize_results_data(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Human-Centric Metrics Analysis', fontsize=16, y=0.98)
        
        # 1. Human Satisfaction
        self._plot_metric_boxplot(data, 'human_satisfaction', axes[0, 0],
                                'Human Satisfaction', 'Satisfaction Score')
        
        # 2. Trust Level
        self._plot_metric_boxplot(data, 'trust_level', axes[0, 1],
                                'Trust Level', 'Trust Score')
        
        # 3. Workload Score
        self._plot_metric_boxplot(data, 'workload_score', axes[1, 0],
                                'Workload Score', 'Workload Level')
        
        # 4. Collaboration Efficiency
        self._plot_metric_boxplot(data, 'collaboration_efficiency', axes[1, 1],
                                'Collaboration Efficiency', 'Efficiency Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Human metrics analysis saved to {save_path}")
        
        plt.show()
    
    def _organize_results_data(self, results: List) -> pd.DataFrame:
        """Organize results into a pandas DataFrame for easier plotting."""
        data_rows = []
        
        for result in results:
            if hasattr(result, 'metrics') and hasattr(result, 'algorithm_name'):
                row = {
                    'algorithm': result.algorithm_name,
                    'seed': getattr(result, 'seed', 0),
                    **result.metrics.to_dict()
                }
                
                # Add configuration info if available
                if hasattr(result, 'config'):
                    row.update({f'config_{k}': v for k, v in result.config.items()})
                
                data_rows.append(row)
        
        return pd.DataFrame(data_rows) if data_rows else pd.DataFrame()
    
    def _plot_metric_boxplot(self, data: pd.DataFrame, metric: str, ax, title: str, ylabel: str):
        """Plot boxplot for a specific metric."""
        if data.empty or metric not in data.columns:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(title)
            return
        
        sns.boxplot(data=data, x='algorithm', y=metric, ax=ax, palette=self.colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Algorithm')
        
        # Rotate x-axis labels if needed
        if len(data['algorithm'].unique()) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        # Add mean markers
        for i, alg in enumerate(data['algorithm'].unique()):
            alg_data = data[data['algorithm'] == alg][metric]
            if not alg_data.empty:
                mean_val = alg_data.mean()
                ax.scatter(i, mean_val, color='red', s=50, marker='D', zorder=10)
    
    def _plot_performance_safety_scatter(self, data: pd.DataFrame, ax):
        """Plot performance vs safety scatter plot."""
        if data.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Performance vs Safety')
            return
        
        algorithms = data['algorithm'].unique()
        colors = self.colors[:len(algorithms)]
        
        for i, alg in enumerate(algorithms):
            alg_data = data[data['algorithm'] == alg]
            if not alg_data.empty and 'asymptotic_performance' in alg_data.columns and 'violation_rate' in alg_data.columns:
                ax.scatter(alg_data['violation_rate'], alg_data['asymptotic_performance'], 
                         c=[colors[i]], label=alg, alpha=0.7, s=60)
        
        ax.set_xlabel('Safety Violation Rate')
        ax.set_ylabel('Asymptotic Performance')
        ax.set_title('Performance vs Safety Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_radar_chart(self, data: pd.DataFrame, ax):
        """Plot radar chart for multiple metrics comparison."""
        if data.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Multi-Metric Radar Chart')
            return
        
        # Select key metrics for radar chart
        radar_metrics = [
            'asymptotic_performance', 'training_stability', 'constraint_satisfaction',
            'human_satisfaction', 'collaboration_efficiency'
        ]
        
        available_metrics = [m for m in radar_metrics if m in data.columns]
        
        if not available_metrics:
            ax.text(0.5, 0.5, 'No radar metrics available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Multi-Metric Radar Chart')
            return
        
        # Calculate mean values for each algorithm
        algorithms = data['algorithm'].unique()[:5]  # Limit to 5 algorithms for clarity
        
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), available_metrics)
        
        for i, alg in enumerate(algorithms):
            alg_data = data[data['algorithm'] == alg]
            if not alg_data.empty:
                values = []
                for metric in available_metrics:
                    if metric in alg_data.columns:
                        # Normalize to [0, 1] scale
                        metric_values = alg_data[metric].dropna()
                        if not metric_values.empty:
                            normalized_val = (metric_values.mean() - data[metric].min()) / \
                                           (data[metric].max() - data[metric].min() + 1e-8)
                            values.append(max(0, min(1, normalized_val)))
                        else:
                            values.append(0)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=alg, 
                       color=self.colors[i % len(self.colors)])
                ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance Profile')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    def _plot_safety_distribution(self, data: pd.DataFrame, ax):
        """Plot safety violation distribution."""
        if data.empty or 'safety_violations' not in data.columns:
            ax.text(0.5, 0.5, 'No safety data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Safety Violation Distribution')
            return
        
        algorithms = data['algorithm'].unique()
        
        for i, alg in enumerate(algorithms):
            alg_data = data[data['algorithm'] == alg]['safety_violations'].dropna()
            if not alg_data.empty:
                ax.hist(alg_data, alpha=0.6, label=alg, bins=20, 
                       color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Number of Safety Violations')
        ax.set_ylabel('Frequency')
        ax.set_title('Safety Violation Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_constraint_satisfaction(self, data: pd.DataFrame, ax):
        """Plot constraint satisfaction rates."""
        if data.empty or 'constraint_satisfaction' not in data.columns:
            ax.text(0.5, 0.5, 'No constraint data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Constraint Satisfaction Rates')
            return
        
        alg_satisfaction = data.groupby('algorithm')['constraint_satisfaction'].mean()
        
        bars = ax.bar(alg_satisfaction.index, alg_satisfaction.values, 
                     color=self.colors[:len(alg_satisfaction)])
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Constraint Satisfaction Rate')
        ax.set_title('Constraint Satisfaction Rates')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom')
        
        if len(alg_satisfaction) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_safety_margins(self, data: pd.DataFrame, ax):
        """Plot safety margin analysis."""
        if data.empty or 'safety_margin' not in data.columns:
            ax.text(0.5, 0.5, 'No safety margin data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Safety Margins')
            return
        
        sns.violinplot(data=data, x='algorithm', y='safety_margin', ax=ax, 
                      palette=self.colors, inner='box')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Safety Margin')
        ax.set_title('Safety Margin Distributions')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Safety Threshold')
        ax.legend()
        
        if len(data['algorithm'].unique()) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_safety_performance_tradeoff(self, data: pd.DataFrame, ax):
        """Plot Pareto frontier for safety vs performance."""
        if data.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Safety vs Performance Trade-off')
            return
        
        algorithms = data['algorithm'].unique()
        
        for i, alg in enumerate(algorithms):
            alg_data = data[data['algorithm'] == alg]
            if not alg_data.empty and 'constraint_satisfaction' in alg_data.columns and 'asymptotic_performance' in alg_data.columns:
                mean_safety = alg_data['constraint_satisfaction'].mean()
                mean_performance = alg_data['asymptotic_performance'].mean()
                
                ax.scatter(mean_safety, mean_performance, 
                         s=100, label=alg, color=self.colors[i % len(self.colors)])
                
                # Add algorithm name annotation
                ax.annotate(alg, (mean_safety, mean_performance), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Constraint Satisfaction Rate')
        ax.set_ylabel('Asymptotic Performance')
        ax.set_title('Safety vs Performance Trade-off')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_significance_heatmap(self, comparisons: List, ax, metric_name: str):
        """Plot statistical significance heatmap for a metric."""
        if not comparisons:
            ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{metric_name} - Statistical Significance')
            return
        
        # Extract algorithm names
        algorithms = sorted(set([comp.algorithm_a for comp in comparisons] + 
                               [comp.algorithm_b for comp in comparisons]))
        
        # Create significance matrix
        n_algs = len(algorithms)
        sig_matrix = np.zeros((n_algs, n_algs))
        p_value_matrix = np.ones((n_algs, n_algs))
        
        for comp in comparisons:
            i = algorithms.index(comp.algorithm_a)
            j = algorithms.index(comp.algorithm_b)
            
            # Fill both triangles of the matrix
            sig_val = 1 if comp.test_result.significant else 0
            sig_matrix[i, j] = sig_val
            sig_matrix[j, i] = sig_val
            
            p_val = comp.test_result.p_value
            p_value_matrix[i, j] = p_val
            p_value_matrix[j, i] = p_val
        
        # Create heatmap
        im = ax.imshow(sig_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(n_algs):
            for j in range(n_algs):
                if i != j:
                    p_val = p_value_matrix[i, j]
                    significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    ax.text(j, i, significance, ha='center', va='center', fontweight='bold')
        
        ax.set_xticks(range(n_algs))
        ax.set_yticks(range(n_algs))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.set_yticklabels(algorithms)
        ax.set_title(f'{metric_name}\nStatistical Significance')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Significant (1) vs Not Significant (0)')


class PerformanceProfiler:
    """Create performance profiles for algorithm comparison."""
    
    def __init__(self):
        self.visualizer = ResultVisualizer()
    
    def create_performance_profile(self, results: Dict[str, List], 
                                 metrics: List[str],
                                 save_path: Optional[Path] = None):
        """Create Dolan-More performance profile."""
        # This is a simplified version of performance profiles
        fig, ax = plt.subplots(figsize=(12, 8))
        
        algorithms = list(results.keys())
        
        for metric in metrics:
            # Extract metric values for all algorithms
            metric_data = {}
            for alg, alg_results in results.items():
                values = []
                for result in alg_results:
                    if hasattr(result, 'metrics'):
                        metric_dict = result.metrics.to_dict()
                        if metric in metric_dict:
                            values.append(metric_dict[metric])
                
                if values:
                    metric_data[alg] = np.mean(values)
            
            if metric_data:
                # Calculate performance ratios
                best_performance = min(metric_data.values()) if 'time' in metric.lower() or 'violation' in metric.lower() else max(metric_data.values())
                
                performance_ratios = {}
                for alg, perf in metric_data.items():
                    if 'time' in metric.lower() or 'violation' in metric.lower():
                        ratio = perf / best_performance if best_performance > 0 else 1
                    else:
                        ratio = best_performance / perf if perf > 0 else float('inf')
                    performance_ratios[alg] = ratio
                
                # Plot performance profile
                tau_values = np.logspace(0, 2, 100)  # Performance ratios from 1 to 100
                
                for i, alg in enumerate(algorithms):
                    if alg in performance_ratios:
                        rho_values = []
                        for tau in tau_values:
                            # Fraction of problems solved within factor tau of the best
                            rho = 1 if performance_ratios[alg] <= tau else 0
                            rho_values.append(rho)
                        
                        ax.semilogx(tau_values, rho_values, label=f'{alg} ({metric})', 
                                   linewidth=2, marker='o', markersize=3)
        
        ax.set_xlabel('Performance Ratio (τ)')
        ax.set_ylabel('Fraction of Problems Solved')
        ax.set_title('Performance Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 1.1)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            logger.info(f"Performance profile saved to {save_path}")
        
        plt.show()


class PublicationPlots:
    """Generate publication-quality plots for papers."""
    
    def __init__(self):
        # Configure for publication
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times'],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'text.usetex': False,  # Set to True if LaTeX is available
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'lines.markersize': 6
        })
    
    def create_main_results_figure(self, results: List, save_path: Optional[Path] = None):
        """Create the main results figure for publication."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        data = ResultVisualizer()._organize_results_data(results)
        
        # Performance comparison
        if not data.empty and 'asymptotic_performance' in data.columns:
            sns.barplot(data=data, x='algorithm', y='asymptotic_performance', 
                       ax=axes[0], palette='viridis', ci=95)
            axes[0].set_title('(a) Performance Comparison')
            axes[0].set_ylabel('Asymptotic Performance')
            axes[0].set_xlabel('Algorithm')
            axes[0].tick_params(axis='x', rotation=45)
        
        # Safety analysis
        if not data.empty and 'violation_rate' in data.columns:
            sns.barplot(data=data, x='algorithm', y='violation_rate', 
                       ax=axes[1], palette='plasma', ci=95)
            axes[1].set_title('(b) Safety Analysis')
            axes[1].set_ylabel('Violation Rate')
            axes[1].set_xlabel('Algorithm')
            axes[1].tick_params(axis='x', rotation=45)
        
        # Efficiency comparison
        if not data.empty and 'sample_efficiency' in data.columns:
            # Convert to log scale for better visualization
            data_copy = data.copy()
            data_copy['log_sample_efficiency'] = np.log10(data_copy['sample_efficiency'] + 1)
            
            sns.barplot(data=data_copy, x='algorithm', y='log_sample_efficiency', 
                       ax=axes[2], palette='cividis', ci=95)
            axes[2].set_title('(c) Sample Efficiency')
            axes[2].set_ylabel('Log₁₀(Sample Efficiency)')
            axes[2].set_xlabel('Algorithm')
            axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', facecolor='white', 
                       format='pdf', dpi=300)
            logger.info(f"Main results figure saved to {save_path}")
        
        plt.show()
    
    def create_supplementary_analysis(self, statistical_results: Dict[str, Any], 
                                    save_path: Optional[Path] = None):
        """Create supplementary statistical analysis figure."""
        # This would create detailed statistical analysis plots
        # Implementation depends on the specific statistical results structure
        pass