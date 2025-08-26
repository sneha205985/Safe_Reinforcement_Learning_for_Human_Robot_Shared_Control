"""
Training visualization plots for Safe RL.

This module provides comprehensive visualization capabilities for training results
including learning curves, convergence plots, policy gradient magnitudes, and more.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PlottingConfig:
    """Configuration for plot styling and parameters."""
    
    def __init__(self):
        self.figure_size = (12, 8)
        self.dpi = 150
        self.font_size = 12
        self.title_size = 16
        self.legend_size = 10
        self.line_width = 2.0
        self.confidence_alpha = 0.2
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#28B463', 
            'accent': '#F39C12',
            'danger': '#E74C3C',
            'warning': '#F1C40F',
            'success': '#27AE60',
            'info': '#3498DB'
        }
        
        # Set matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': self.figure_size,
            'figure.dpi': self.dpi,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'legend.fontsize': self.legend_size,
            'lines.linewidth': self.line_width
        })


class LearningCurvePlotter:
    """Creates learning curve visualizations."""
    
    def __init__(self, config: PlottingConfig = None):
        self.config = config or PlottingConfig()
        self.logger = logging.getLogger(__name__)
    
    def plot_learning_curve_with_confidence(self, data: pd.DataFrame,
                                          value_col: str = 'episode_return',
                                          time_col: str = 'iteration',
                                          confidence_intervals: Dict[str, List] = None,
                                          title: str = "Learning Curve") -> plt.Figure:
        """Plot learning curve with confidence intervals."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        if value_col not in data.columns or time_col not in data.columns:
            ax.text(0.5, 0.5, f"Missing columns: {value_col} or {time_col}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        x = data[time_col].values
        y = data[value_col].values
        
        # Plot main learning curve
        ax.plot(x, y, color=self.config.colors['primary'], 
               linewidth=self.config.line_width, label='Training Performance', alpha=0.8)
        
        # Add confidence intervals if provided
        if confidence_intervals and 'rolling_ci_lower' in confidence_intervals:
            ci_lower = confidence_intervals['rolling_ci_lower']
            ci_upper = confidence_intervals['rolling_ci_upper']
            
            if len(ci_lower) == len(x) and len(ci_upper) == len(x):
                ax.fill_between(x, ci_lower, ci_upper, 
                              color=self.config.colors['primary'], 
                              alpha=self.config.confidence_alpha,
                              label=f'{int(confidence_intervals.get("confidence_level", 0.95)*100)}% CI')
        
        # Add smoothed curve if available
        if f'{value_col}_smoothed' in data.columns:
            y_smooth = data[f'{value_col}_smoothed'].dropna().values
            x_smooth = data[time_col].iloc[:len(y_smooth)].values
            ax.plot(x_smooth, y_smooth, color=self.config.colors['secondary'],
                   linewidth=self.config.line_width, linestyle='--', 
                   label='Smoothed', alpha=0.9)
        
        ax.set_xlabel(time_col.title())
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add trend annotation
        if len(x) > 10:
            from scipy import stats
            slope, _, r_squared, p_value, _ = stats.linregress(x, y)
            trend_text = f'Trend: {"↗" if slope > 0 else "↘"} (R²={r_squared:.3f})'
            if p_value < 0.05:
                trend_text += '*'
            ax.text(0.02, 0.98, trend_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_multiple_learning_curves(self, algorithms_data: Dict[str, pd.DataFrame],
                                     value_col: str = 'episode_return',
                                     time_col: str = 'iteration',
                                     title: str = "Learning Curves Comparison") -> plt.Figure:
        """Plot learning curves for multiple algorithms."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        colors = list(self.config.colors.values())
        
        for i, (alg_name, data) in enumerate(algorithms_data.items()):
            if value_col not in data.columns or time_col not in data.columns:
                continue
            
            x = data[time_col].values
            y = data[value_col].values
            
            color = colors[i % len(colors)]
            ax.plot(x, y, color=color, linewidth=self.config.line_width, 
                   label=alg_name, alpha=0.8)
            
            # Add final performance annotation
            if len(y) > 0:
                final_perf = y[-1]
                ax.annotate(f'{final_perf:.2f}', 
                           xy=(x[-1], y[-1]), 
                           xytext=(5, 5), 
                           textcoords='offset points',
                           color=color,
                           fontweight='bold')
        
        ax.set_xlabel(time_col.title())
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curve_with_milestones(self, data: pd.DataFrame,
                                          milestones: Dict[str, Dict],
                                          value_col: str = 'episode_return',
                                          time_col: str = 'iteration',
                                          title: str = "Learning Curve with Milestones") -> plt.Figure:
        """Plot learning curve with performance milestones marked."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = data[time_col].values
        y = data[value_col].values
        
        # Main learning curve
        ax.plot(x, y, color=self.config.colors['primary'], 
               linewidth=self.config.line_width, label='Performance')
        
        # Mark milestones
        milestone_colors = [self.config.colors['success'], self.config.colors['warning'], 
                          self.config.colors['accent'], self.config.colors['danger']]
        
        for i, (milestone_name, milestone_data) in enumerate(milestones.items()):
            if milestone_data.get('achieved', False):
                milestone_iter = milestone_data['iteration']
                milestone_perf = milestone_data['performance']
                color = milestone_colors[i % len(milestone_colors)]
                
                # Vertical line at milestone
                ax.axvline(x=milestone_iter, color=color, linestyle='--', alpha=0.7)
                
                # Marker at milestone
                ax.plot(milestone_iter, milestone_perf, 'o', 
                       color=color, markersize=8, label=f'{milestone_name} achieved')
                
                # Annotation
                ax.annotate(milestone_name.replace('_', ' ').title(), 
                           xy=(milestone_iter, milestone_perf),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel(time_col.title())
        ax.set_ylabel(value_col.replace('_', ' ').title())
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ConvergencePlotter:
    """Creates convergence analysis visualizations."""
    
    def __init__(self, config: PlottingConfig = None):
        self.config = config or PlottingConfig()
        self.logger = logging.getLogger(__name__)
    
    def plot_policy_gradient_magnitudes(self, data: pd.DataFrame,
                                       grad_col: str = 'policy_gradient_norm',
                                       time_col: str = 'iteration',
                                       title: str = "Policy Gradient Magnitudes") -> plt.Figure:
        """Plot policy gradient magnitudes over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figure_size[0], 
                                                      self.config.figure_size[1] * 1.2))
        
        if grad_col not in data.columns:
            ax1.text(0.5, 0.5, f"Column '{grad_col}' not found", 
                    ha='center', va='center', transform=ax1.transAxes)
            return fig
        
        x = data[time_col].values
        y = data[grad_col].values
        
        # Main gradient magnitude plot
        ax1.plot(x, y, color=self.config.colors['primary'], linewidth=self.config.line_width)
        ax1.set_ylabel('Gradient Magnitude')
        ax1.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Log scale version
        ax2.semilogy(x, y, color=self.config.colors['secondary'], linewidth=self.config.line_width)
        ax2.set_xlabel(time_col.title())
        ax2.set_ylabel('Log Gradient Magnitude')
        ax2.set_title('Log Scale', fontsize=self.config.title_size-2)
        ax2.grid(True, alpha=0.3)
        
        # Add convergence threshold line if gradient is decreasing
        if len(y) > 10:
            convergence_threshold = np.percentile(y[-50:], 10)  # 10th percentile of last 50 values
            ax1.axhline(y=convergence_threshold, color=self.config.colors['danger'], 
                       linestyle='--', alpha=0.7, label=f'Convergence threshold: {convergence_threshold:.4f}')
            ax2.axhline(y=convergence_threshold, color=self.config.colors['danger'], 
                       linestyle='--', alpha=0.7)
            ax1.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_kl_divergence(self, data: pd.DataFrame,
                          kl_col: str = 'kl_divergence',
                          time_col: str = 'iteration',
                          target_kl: float = 0.01,
                          title: str = "KL Divergence Between Policy Updates") -> plt.Figure:
        """Plot KL divergence between policy updates."""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        if kl_col not in data.columns:
            ax.text(0.5, 0.5, f"Column '{kl_col}' not found", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        x = data[time_col].values
        y = data[kl_col].values
        
        # Main KL divergence plot
        ax.plot(x, y, color=self.config.colors['primary'], 
               linewidth=self.config.line_width, label='KL Divergence')
        
        # Target KL line
        ax.axhline(y=target_kl, color=self.config.colors['danger'], 
                  linestyle='--', alpha=0.8, label=f'Target KL: {target_kl}')
        
        # Highlight violations
        violations = y > target_kl
        if np.any(violations):
            ax.scatter(x[violations], y[violations], 
                      color=self.config.colors['danger'], s=30, alpha=0.7, 
                      label=f'KL violations: {np.sum(violations)}')
        
        # Add moving average
        if len(y) > 10:
            window_size = max(5, len(y) // 20)
            moving_avg = pd.Series(y).rolling(window=window_size, center=True).mean()
            ax.plot(x, moving_avg, color=self.config.colors['secondary'], 
                   linewidth=self.config.line_width, linestyle=':', 
                   label=f'Moving Average (window={window_size})')
        
        ax.set_xlabel(time_col.title())
        ax.set_ylabel('KL Divergence')
        ax.set_title(title, fontsize=self.config.title_size, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_value_function_accuracy(self, data: pd.DataFrame,
                                   value_loss_col: str = 'value_loss',
                                   returns_col: str = 'episode_return',
                                   time_col: str = 'iteration',
                                   title: str = "Value Function Accuracy") -> plt.Figure:
        """Plot value function accuracy metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.5, 
                                                      self.config.figure_size[1]))
        
        x = data[time_col].values
        
        # Value loss over time
        if value_loss_col in data.columns:
            value_loss = data[value_loss_col].values
            ax1.plot(x, value_loss, color=self.config.colors['primary'], 
                    linewidth=self.config.line_width, label='Value Loss')
            ax1.set_yscale('log')
            ax1.set_xlabel(time_col.title())
            ax1.set_ylabel('Value Loss (log scale)')
            ax1.set_title('Value Function Loss', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Value loss vs returns correlation
        if value_loss_col in data.columns and returns_col in data.columns:
            returns = data[returns_col].values
            value_loss = data[value_loss_col].values
            
            # Remove NaN values for correlation
            mask = ~(np.isnan(returns) | np.isnan(value_loss))
            returns_clean = returns[mask]
            value_loss_clean = value_loss[mask]
            
            if len(returns_clean) > 10:
                ax2.scatter(value_loss_clean, returns_clean, 
                           color=self.config.colors['accent'], alpha=0.6, s=20)
                
                # Add trend line
                z = np.polyfit(value_loss_clean, returns_clean, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(value_loss_clean), max(value_loss_clean), 100)
                ax2.plot(x_trend, p(x_trend), 
                        color=self.config.colors['danger'], linestyle='--', 
                        linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
                
                # Calculate correlation
                correlation = np.corrcoef(value_loss_clean, returns_clean)[0, 1]
                ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                ax2.set_xlabel('Value Loss')
                ax2.set_ylabel('Episode Return')
                ax2.set_title('Value Loss vs Returns', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
        
        plt.suptitle(title, fontsize=self.config.title_size, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def plot_lagrange_multiplier_evolution(self, data: pd.DataFrame,
                                         lambda_col: str = 'lagrange_multiplier',
                                         constraint_cost_col: str = 'constraint_cost',
                                         time_col: str = 'iteration',
                                         title: str = "Lagrange Multiplier Evolution") -> plt.Figure:
        """Plot evolution of Lagrange multipliers in constrained algorithms."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config.figure_size[0], 
                                                      self.config.figure_size[1] * 1.2))
        
        x = data[time_col].values
        
        # Lagrange multiplier evolution
        if lambda_col in data.columns:
            lambda_vals = data[lambda_col].values
            ax1.plot(x, lambda_vals, color=self.config.colors['primary'], 
                    linewidth=self.config.line_width, label='λ (Lagrange Multiplier)')
            ax1.set_ylabel('Lagrange Multiplier')
            ax1.set_title('Multiplier Evolution', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Constraint cost over time
        if constraint_cost_col in data.columns:
            constraint_costs = data[constraint_cost_col].values
            ax2.plot(x, constraint_costs, color=self.config.colors['danger'], 
                    linewidth=self.config.line_width, label='Constraint Cost')
            
            # Add constraint threshold line
            threshold = 0.01  # Typical constraint threshold
            ax2.axhline(y=threshold, color=self.config.colors['warning'], 
                       linestyle='--', alpha=0.8, label=f'Threshold: {threshold}')
            
            ax2.set_xlabel(time_col.title())
            ax2.set_ylabel('Constraint Cost')
            ax2.set_title('Constraint Satisfaction', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Show relationship between multiplier and constraint cost
        if lambda_col in data.columns and constraint_cost_col in data.columns:
            # Add text box with correlation
            lambda_vals = data[lambda_col].dropna().values
            constraint_costs = data[constraint_cost_col].dropna().values
            min_len = min(len(lambda_vals), len(constraint_costs))
            
            if min_len > 10:
                correlation = np.corrcoef(lambda_vals[:min_len], constraint_costs[:min_len])[0, 1]
                ax1.text(0.02, 0.98, f'λ-Cost Correlation: {correlation:.3f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(title, fontsize=self.config.title_size, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


class TrainingPlotter:
    """Main class for training visualization."""
    
    def __init__(self, config: PlottingConfig = None):
        self.config = config or PlottingConfig()
        self.learning_plotter = LearningCurvePlotter(config)
        self.convergence_plotter = ConvergencePlotter(config)
        self.logger = logging.getLogger(__name__)
    
    def create_training_overview(self, data: pd.DataFrame,
                               analysis_results: Dict[str, Any] = None) -> plt.Figure:
        """Create comprehensive training overview dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main learning curve (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'episode_return' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['episode_return'].values
            ax1.plot(x, y, color=self.config.colors['primary'], linewidth=2)
            ax1.set_title('Learning Curve', fontweight='bold')
            ax1.set_ylabel('Episode Return')
            ax1.grid(True, alpha=0.3)
        
        # 2. Success rate (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        if 'success_rate' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['success_rate'].values
            ax2.plot(x, y, color=self.config.colors['success'], linewidth=2)
            ax2.set_title('Success Rate', fontweight='bold')
            ax2.set_ylabel('Success Rate')
            ax2.grid(True, alpha=0.3)
        
        # 3. Policy loss (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'policy_loss' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['policy_loss'].values
            ax3.semilogy(x, y, color=self.config.colors['accent'], linewidth=2)
            ax3.set_title('Policy Loss', fontweight='bold')
            ax3.set_ylabel('Loss (log)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Value loss (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if 'value_loss' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['value_loss'].values
            ax4.semilogy(x, y, color=self.config.colors['info'], linewidth=2)
            ax4.set_title('Value Loss', fontweight='bold')
            ax4.set_ylabel('Loss (log)')
            ax4.grid(True, alpha=0.3)
        
        # 5. Constraint violations (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if 'constraint_violations' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['constraint_violations'].values
            ax5.bar(x[::max(1, len(x)//50)], y[::max(1, len(x)//50)], 
                   color=self.config.colors['danger'], alpha=0.7, width=max(1, len(x)//100))
            ax5.set_title('Constraint Violations', fontweight='bold')
            ax5.set_ylabel('Violations')
            ax5.grid(True, alpha=0.3)
        
        # 6. KL Divergence (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        if 'kl_divergence' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['kl_divergence'].values
            ax6.semilogy(x, y, color=self.config.colors['warning'], linewidth=2)
            ax6.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Target KL')
            ax6.set_title('KL Divergence', fontweight='bold')
            ax6.set_ylabel('KL (log)')
            ax6.set_xlabel('Iteration')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        # 7. Entropy (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        if 'entropy' in data.columns:
            x = data['iteration'].values if 'iteration' in data.columns else range(len(data))
            y = data['entropy'].values
            ax7.plot(x, y, color=self.config.colors['secondary'], linewidth=2)
            ax7.set_title('Policy Entropy', fontweight='bold')
            ax7.set_ylabel('Entropy')
            ax7.set_xlabel('Iteration')
            ax7.grid(True, alpha=0.3)
        
        # 8. Training summary (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Add summary statistics
        summary_text = "Training Summary\n" + "="*15 + "\n"
        
        if analysis_results and 'convergence_analysis' in analysis_results:
            conv_analysis = analysis_results['convergence_analysis']
            if 'episode_return' in conv_analysis:
                converged = conv_analysis['episode_return'].get('converged', False)
                summary_text += f"Converged: {'✓' if converged else '✗'}\n"
        
        if 'episode_return' in data.columns:
            final_perf = data['episode_return'].iloc[-1] if len(data) > 0 else 0
            max_perf = data['episode_return'].max() if len(data) > 0 else 0
            summary_text += f"Final Return: {final_perf:.2f}\n"
            summary_text += f"Max Return: {max_perf:.2f}\n"
        
        if 'constraint_violations' in data.columns:
            total_violations = data['constraint_violations'].sum() if len(data) > 0 else 0
            violation_rate = total_violations / len(data) if len(data) > 0 else 0
            summary_text += f"Total Violations: {int(total_violations)}\n"
            summary_text += f"Violation Rate: {violation_rate:.3f}\n"
        
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle('Training Overview Dashboard', fontsize=18, fontweight='bold')
        return fig
    
    def create_interactive_training_plot(self, data: pd.DataFrame,
                                       metrics: List[str] = None) -> go.Figure:
        """Create interactive training plot using Plotly."""
        if metrics is None:
            metrics = ['episode_return', 'policy_loss', 'value_loss', 'constraint_violations']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in data.columns]
        
        if not available_metrics:
            # Create empty plot with error message
            fig = go.Figure()
            fig.add_annotation(text="No metrics available for plotting", 
                             x=0.5, y=0.5, showarrow=False, font_size=16)
            return fig
        
        # Create subplots
        n_metrics = len(available_metrics)
        rows = (n_metrics + 1) // 2
        
        subplot_titles = [m.replace('_', ' ').title() for m in available_metrics]
        fig = make_subplots(rows=rows, cols=2, 
                           subplot_titles=subplot_titles,
                           vertical_spacing=0.08)
        
        x = data['iteration'] if 'iteration' in data.columns else list(range(len(data)))
        
        colors = ['#2E86C1', '#E74C3C', '#F39C12', '#27AE60', '#8E44AD', '#17A2B8']
        
        for i, metric in enumerate(available_metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            y = data[metric].values
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(x=x, y=y, name=metric.replace('_', ' ').title(),
                          line=dict(color=color, width=2),
                          hovertemplate=f'<b>{metric}</b><br>Iteration: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'),
                row=row, col=col
            )
            
            # Add trend line for main metrics
            if metric in ['episode_return', 'success_rate'] and len(x) > 10:
                z = np.polyfit(x, y, 1)
                trend_line = np.poly1d(z)(x)
                fig.add_trace(
                    go.Scatter(x=x, y=trend_line, 
                             name=f'{metric} trend',
                             line=dict(color=color, dash='dash', width=1),
                             opacity=0.7,
                             showlegend=False,
                             hovertemplate=f'<b>Trend</b><br>Slope: {z[0]:.6f}<extra></extra>'),
                    row=row, col=col
                )
        
        # Update layout
        fig.update_layout(
            title=dict(text="Interactive Training Dashboard", 
                       font=dict(size=20)),
            showlegend=False,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Update y-axes for log scale where appropriate
        for i, metric in enumerate(available_metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            if 'loss' in metric.lower():
                fig.update_yaxes(type="log", row=row, col=col)
        
        return fig
    
    def save_all_training_plots(self, data: pd.DataFrame,
                              analysis_results: Dict[str, Any] = None,
                              output_dir: str = "training_plots",
                              formats: List[str] = ['png', 'pdf']) -> Dict[str, str]:
        """Save all training plots to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # Training overview
            fig_overview = self.create_training_overview(data, analysis_results)
            for fmt in formats:
                filename = f"training_overview.{fmt}"
                filepath = output_path / filename
                fig_overview.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                saved_files[f'overview_{fmt}'] = str(filepath)
            plt.close(fig_overview)
            
            # Learning curve with confidence intervals
            if analysis_results and 'learning_curve_analysis' in analysis_results:
                ci_data = analysis_results['learning_curve_analysis'].get('confidence_intervals', {})
                fig_learning = self.learning_plotter.plot_learning_curve_with_confidence(
                    data, confidence_intervals=ci_data
                )
                for fmt in formats:
                    filename = f"learning_curve.{fmt}"
                    filepath = output_path / filename
                    fig_learning.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    saved_files[f'learning_curve_{fmt}'] = str(filepath)
                plt.close(fig_learning)
            
            # Policy gradient magnitudes
            if 'policy_gradient_norm' in data.columns:
                fig_gradients = self.convergence_plotter.plot_policy_gradient_magnitudes(data)
                for fmt in formats:
                    filename = f"gradient_magnitudes.{fmt}"
                    filepath = output_path / filename
                    fig_gradients.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    saved_files[f'gradients_{fmt}'] = str(filepath)
                plt.close(fig_gradients)
            
            # KL divergence
            if 'kl_divergence' in data.columns:
                fig_kl = self.convergence_plotter.plot_kl_divergence(data)
                for fmt in formats:
                    filename = f"kl_divergence.{fmt}"
                    filepath = output_path / filename
                    fig_kl.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    saved_files[f'kl_divergence_{fmt}'] = str(filepath)
                plt.close(fig_kl)
            
            # Value function accuracy
            fig_value = self.convergence_plotter.plot_value_function_accuracy(data)
            for fmt in formats:
                filename = f"value_function.{fmt}"
                filepath = output_path / filename
                fig_value.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                saved_files[f'value_function_{fmt}'] = str(filepath)
            plt.close(fig_value)
            
            # Interactive plot (HTML only)
            fig_interactive = self.create_interactive_training_plot(data)
            html_filepath = output_path / "interactive_training.html"
            fig_interactive.write_html(str(html_filepath))
            saved_files['interactive'] = str(html_filepath)
            
            self.logger.info(f"Saved {len(saved_files)} training plot files to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving training plots: {e}")
        
        return saved_files