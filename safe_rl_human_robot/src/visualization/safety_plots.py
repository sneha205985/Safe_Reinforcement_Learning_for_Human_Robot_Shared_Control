"""
Safety visualization plots for Safe RL.

This module provides comprehensive visualization capabilities for safety analysis
including constraint violations, risk assessment, failure modes, and safety margins.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


class SafetyPlottingConfig:
    """Configuration for safety plot styling."""
    
    def __init__(self):
        self.figure_size = (12, 8)
        self.dpi = 150
        self.colors = {
            'safe': '#27AE60',      # Green
            'warning': '#F39C12',    # Orange  
            'danger': '#E74C3C',     # Red
            'critical': '#8E44AD',   # Purple
            'neutral': '#34495E',    # Dark gray
            'background': '#ECF0F1'  # Light gray
        }
        self.violation_colormap = 'Reds'
        self.safety_colormap = 'RdYlGn_r'


class ConstraintViolationPlotter:
    """Creates constraint violation visualizations."""
    
    def __init__(self, config: SafetyPlottingConfig = None):
        self.config = config or SafetyPlottingConfig()
        self.logger = logging.getLogger(__name__)
    
    def plot_violation_frequency_histogram(self, data: pd.DataFrame,
                                         constraint_columns: List[str] = None,
                                         title: str = "Constraint Violation Frequency") -> plt.Figure:
        """Plot histogram of constraint violation frequencies."""
        if constraint_columns is None:
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost'])]
        
        if not constraint_columns:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'No constraint columns found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        n_constraints = len(constraint_columns)
        fig, axes = plt.subplots((n_constraints + 1) // 2, 2, 
                               figsize=(self.config.figure_size[0], 
                                       self.config.figure_size[1] * ((n_constraints + 1) // 2)))
        
        if n_constraints == 1:
            axes = [axes]
        elif n_constraints <= 2:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(constraint_columns):
            ax = axes[i] if n_constraints > 1 else axes[0]
            
            constraint_values = data[col].dropna().values
            
            # Determine if values represent violations (positive) or margins (negative for violations)
            if 'margin' in col.lower():
                violations = constraint_values[constraint_values < 0]
                violation_magnitudes = np.abs(violations)
            else:
                violations = constraint_values[constraint_values > 0]
                violation_magnitudes = violations
            
            if len(violation_magnitudes) > 0:
                # Create histogram
                n_bins = min(50, max(10, len(violation_magnitudes) // 10))
                counts, bins, patches = ax.hist(violation_magnitudes, bins=n_bins, 
                                              color=self.config.colors['danger'], 
                                              alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Color bars based on severity
                for j, (patch, count) in enumerate(zip(patches, counts)):
                    if count > np.max(counts) * 0.7:  # High frequency
                        patch.set_facecolor(self.config.colors['critical'])
                    elif count > np.max(counts) * 0.3:  # Medium frequency
                        patch.set_facecolor(self.config.colors['danger'])
                    else:  # Low frequency
                        patch.set_facecolor(self.config.colors['warning'])
                
                # Add statistics
                mean_violation = np.mean(violation_magnitudes)
                max_violation = np.max(violation_magnitudes)
                
                stats_text = f'Mean: {mean_violation:.4f}\nMax: {max_violation:.4f}\nCount: {len(violations)}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            else:
                ax.text(0.5, 0.5, 'No violations detected', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
            
            ax.set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Violation Magnitude')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        if n_constraints % 2 == 1 and n_constraints > 1:
            axes[-1].set_visible(False)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_violation_timeline(self, data: pd.DataFrame,
                              constraint_columns: List[str] = None,
                              time_col: str = 'iteration',
                              title: str = "Constraint Violations Over Time") -> plt.Figure:
        """Plot constraint violations over time."""
        if constraint_columns is None:
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost'])]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        if not constraint_columns:
            ax.text(0.5, 0.5, 'No constraint columns found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        x = data[time_col].values if time_col in data.columns else range(len(data))
        colors = [self.config.colors['danger'], self.config.colors['critical'], 
                 self.config.colors['warning'], '#FF69B4', '#8A2BE2']
        
        violation_detected = False
        
        for i, col in enumerate(constraint_columns):
            if col not in data.columns:
                continue
                
            constraint_values = data[col].values
            color = colors[i % len(colors)]
            
            # Determine violations
            if 'margin' in col.lower():
                violations_mask = constraint_values < 0
                violation_values = np.abs(constraint_values)
            else:
                violations_mask = constraint_values > 0
                violation_values = constraint_values
            
            if np.any(violations_mask):
                violation_detected = True
                
                # Plot violation timeline
                violation_times = x[violations_mask]
                violation_magnitudes = violation_values[violations_mask]
                
                # Scatter plot for individual violations
                ax.scatter(violation_times, violation_magnitudes, 
                          color=color, alpha=0.7, s=30, label=f'{col} violations')
                
                # Add trend line if enough violations
                if len(violation_times) > 5:
                    z = np.polyfit(violation_times, violation_magnitudes, 1)
                    trend_line = np.poly1d(z)(violation_times)
                    ax.plot(violation_times, trend_line, color=color, 
                           linestyle='--', alpha=0.8, linewidth=1)
        
        if not violation_detected:
            ax.text(0.5, 0.5, 'No violations detected in the selected constraints', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor=self.config.colors['safe'], alpha=0.3))
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax.set_xlabel(time_col.title())
        ax.set_ylabel('Violation Magnitude')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_violation_heatmap(self, data: pd.DataFrame,
                             constraint_columns: List[str] = None,
                             time_col: str = 'iteration',
                             window_size: int = 50,
                             title: str = "Constraint Violation Heatmap") -> plt.Figure:
        """Plot heatmap of constraint violations over time windows."""
        if constraint_columns is None:
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost'])]
        
        if not constraint_columns:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'No constraint columns found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create time windows
        n_windows = max(1, len(data) // window_size)
        violation_matrix = np.zeros((len(constraint_columns), n_windows))
        
        for i, col in enumerate(constraint_columns):
            if col not in data.columns:
                continue
                
            constraint_values = data[col].values
            
            for w in range(n_windows):
                start_idx = w * window_size
                end_idx = min((w + 1) * window_size, len(constraint_values))
                window_data = constraint_values[start_idx:end_idx]
                
                # Count violations in this window
                if 'margin' in col.lower():
                    violations = np.sum(window_data < 0)
                else:
                    violations = np.sum(window_data > 0)
                
                violation_matrix[i, w] = violations / len(window_data) * 100  # Percentage
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        im = ax.imshow(violation_matrix, cmap=self.config.violation_colormap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(n_windows))
        ax.set_xticklabels([f'{i*window_size}-{(i+1)*window_size}' for i in range(n_windows)], 
                          rotation=45, ha='right')
        ax.set_yticks(range(len(constraint_columns)))
        ax.set_yticklabels([col.replace('_', ' ').title() for col in constraint_columns])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Violation Rate (%)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(constraint_columns)):
            for j in range(n_windows):
                text = f'{violation_matrix[i, j]:.1f}%'
                color = 'white' if violation_matrix[i, j] > 50 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Time Windows')
        ax.set_ylabel('Constraints')
        
        plt.tight_layout()
        return fig
    
    def plot_constraint_activation_patterns(self, data: pd.DataFrame,
                                          constraint_columns: List[str] = None,
                                          title: str = "Constraint Activation Patterns") -> plt.Figure:
        """Plot co-occurrence patterns of constraint activations."""
        if constraint_columns is None:
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost'])]
        
        if len(constraint_columns) < 2:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Need at least 2 constraints for pattern analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Create binary violation matrix
        violation_matrix = np.zeros((len(data), len(constraint_columns)))
        
        for i, col in enumerate(constraint_columns):
            if col not in data.columns:
                continue
            
            constraint_values = data[col].values
            if 'margin' in col.lower():
                violations = constraint_values < 0
            else:
                violations = constraint_values > 0
            
            violation_matrix[:, i] = violations.astype(int)
        
        # Compute co-occurrence matrix
        co_occurrence = np.zeros((len(constraint_columns), len(constraint_columns)))
        
        for i in range(len(constraint_columns)):
            for j in range(len(constraint_columns)):
                if i != j:
                    # Count simultaneous violations
                    co_violations = np.sum((violation_matrix[:, i] == 1) & (violation_matrix[:, j] == 1))
                    total_i_violations = np.sum(violation_matrix[:, i])
                    
                    if total_i_violations > 0:
                        co_occurrence[i, j] = co_violations / total_i_violations
                else:
                    co_occurrence[i, j] = 1.0  # Self-correlation
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        mask = np.zeros_like(co_occurrence, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True  # Mask upper triangle
        
        sns.heatmap(co_occurrence, mask=mask, annot=True, cmap='Blues', 
                   xticklabels=[col.replace('_', ' ').title() for col in constraint_columns],
                   yticklabels=[col.replace('_', ' ').title() for col in constraint_columns],
                   ax=ax, cbar_kws={'label': 'Co-occurrence Probability'})
        
        ax.set_title(title, fontweight='bold')
        plt.tight_layout()
        return fig


class RiskVisualization:
    """Creates risk assessment visualizations."""
    
    def __init__(self, config: SafetyPlottingConfig = None):
        self.config = config or SafetyPlottingConfig()
        self.logger = logging.getLogger(__name__)
    
    def plot_safety_margin_distribution(self, data: pd.DataFrame,
                                       margin_columns: List[str] = None,
                                       title: str = "Safety Margin Distribution") -> plt.Figure:
        """Plot distribution of safety margins."""
        if margin_columns is None:
            margin_columns = [col for col in data.columns 
                            if 'margin' in col.lower() or 'distance' in col.lower()]
        
        if not margin_columns:
            # Try to infer from constraint columns
            constraint_columns = [col for col in data.columns 
                                if any(keyword in col.lower() 
                                      for keyword in ['constraint', 'violation', 'cost'])]
            if constraint_columns:
                margin_columns = constraint_columns  # Will be converted to margins
            else:
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                ax.text(0.5, 0.5, 'No margin or constraint columns found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return fig
        
        n_margins = len(margin_columns)
        fig, axes = plt.subplots(1, min(3, n_margins), 
                               figsize=(self.config.figure_size[0], self.config.figure_size[1] * 0.8))
        
        if n_margins == 1:
            axes = [axes]
        
        for i, col in enumerate(margin_columns[:3]):  # Limit to first 3
            ax = axes[i] if n_margins > 1 else axes[0]
            
            margin_values = data[col].dropna().values
            
            # Convert constraint costs to margins if needed
            if 'constraint' in col.lower() or 'cost' in col.lower():
                margin_values = -margin_values  # Negative cost = positive margin
            
            # Create histogram
            ax.hist(margin_values, bins=50, color=self.config.colors['neutral'], 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color regions by safety level
            x_min, x_max = ax.get_xlim()
            
            # Danger zone (negative margins)
            if np.any(margin_values < 0):
                danger_x = np.linspace(min(x_min, np.min(margin_values)), 0, 100)
                ax.axvspan(min(x_min, np.min(margin_values)), 0, 
                          color=self.config.colors['danger'], alpha=0.2, label='Danger Zone')
            
            # Warning zone (small positive margins)
            warning_threshold = np.percentile(margin_values, 25) if len(margin_values) > 4 else 0.1
            if warning_threshold > 0:
                ax.axvspan(0, warning_threshold, 
                          color=self.config.colors['warning'], alpha=0.2, label='Warning Zone')
            
            # Safe zone
            ax.axvspan(warning_threshold, x_max, 
                      color=self.config.colors['safe'], alpha=0.2, label='Safe Zone')
            
            # Add vertical line at zero
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Safety Boundary')
            
            # Statistics
            mean_margin = np.mean(margin_values)
            std_margin = np.std(margin_values)
            min_margin = np.min(margin_values)
            
            stats_text = f'Mean: {mean_margin:.4f}\nStd: {std_margin:.4f}\nMin: {min_margin:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
            ax.set_xlabel('Safety Margin')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Only show legend for first subplot
                ax.legend(loc='upper right')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_risk_heatmap_state_space(self, data: pd.DataFrame,
                                     state_columns: List[str],
                                     risk_column: str,
                                     grid_size: int = 50,
                                     title: str = "Risk Heatmap in State Space") -> plt.Figure:
        """Plot risk heatmap in 2D state space."""
        if len(state_columns) < 2:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Need at least 2 state dimensions', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Use first two state dimensions
        state1_col, state2_col = state_columns[0], state_columns[1]
        
        if not all(col in data.columns for col in [state1_col, state2_col, risk_column]):
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, f'Missing columns: {state1_col}, {state2_col}, or {risk_column}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Extract data
        clean_data = data[[state1_col, state2_col, risk_column]].dropna()
        
        if len(clean_data) < 10:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        x = clean_data[state1_col].values
        y = clean_data[state2_col].values
        risk = clean_data[risk_column].values
        
        # Convert constraint costs to risk if needed
        if 'constraint' in risk_column.lower() or 'cost' in risk_column.lower():
            risk = np.maximum(risk, 0)  # Only positive values represent risk
        elif 'margin' in risk_column.lower():
            risk = np.maximum(-risk, 0)  # Negative margins represent risk
        
        # Create grid
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolate risk values onto grid
        from scipy.interpolate import griddata
        
        Zi = griddata((x, y), risk, (Xi, Yi), method='cubic', fill_value=0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Risk heatmap
        im = ax.contourf(Xi, Yi, Zi, levels=20, cmap=self.config.safety_colormap, extend='max')
        
        # Scatter plot of actual data points
        scatter = ax.scatter(x, y, c=risk, cmap=self.config.safety_colormap, 
                           s=20, alpha=0.6, edgecolors='black', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Risk Level', rotation=270, labelpad=20)
        
        # Add contour lines for high risk regions
        high_risk_threshold = np.percentile(risk, 90)
        if high_risk_threshold > 0:
            contour_lines = ax.contour(Xi, Yi, Zi, levels=[high_risk_threshold], 
                                     colors='red', linewidths=2, linestyles='--')
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='High Risk')
        
        ax.set_xlabel(state1_col.replace('_', ' ').title())
        ax.set_ylabel(state2_col.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_failure_mode_clustering(self, data: pd.DataFrame,
                                   constraint_columns: List[str],
                                   title: str = "Failure Mode Clustering") -> plt.Figure:
        """Plot clustering of failure modes."""
        if len(constraint_columns) < 2:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Need at least 2 constraints for clustering', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        # Extract violation data
        violation_data = []
        violation_indices = []
        
        for i, row in data.iterrows():
            violations = []
            has_violation = False
            
            for col in constraint_columns:
                if col in data.columns:
                    value = row[col]
                    if pd.isna(value):
                        violations.append(0)
                    elif 'margin' in col.lower():
                        violations.append(1 if value < 0 else 0)
                    else:
                        violations.append(1 if value > 0 else 0)
                    
                    if violations[-1] == 1:
                        has_violation = True
            
            if has_violation:
                violation_data.append(violations)
                violation_indices.append(i)
        
        if len(violation_data) < 5:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Insufficient violation data for clustering', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig
        
        violation_array = np.array(violation_data)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=3)
        cluster_labels = clustering.fit_predict(violation_array)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Visualize clusters
        if violation_array.shape[1] >= 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.config.figure_size[0] * 1.5, 
                                                         self.config.figure_size[1]))
            
            # 2D scatter plot of first two constraints
            unique_labels = set(cluster_labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise
                    col = [0, 0, 0, 1]
                
                class_member_mask = (cluster_labels == k)
                xy = violation_array[class_member_mask]
                
                if k == -1:
                    ax1.scatter(xy[:, 0], xy[:, 1], c=[col], marker='x', s=50, alpha=0.6, label='Noise')
                else:
                    ax1.scatter(xy[:, 0], xy[:, 1], c=[col], s=50, alpha=0.8, label=f'Cluster {k}')
            
            ax1.set_xlabel(constraint_columns[0].replace('_', ' ').title())
            ax1.set_ylabel(constraint_columns[1].replace('_', ' ').title())
            ax1.set_title('Failure Mode Clusters (2D Projection)', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Cluster characteristics heatmap
            cluster_centers = []
            cluster_names = []
            
            for k in unique_labels:
                if k != -1:  # Skip noise
                    cluster_mask = cluster_labels == k
                    cluster_data = violation_array[cluster_mask]
                    center = np.mean(cluster_data, axis=0)
                    cluster_centers.append(center)
                    cluster_names.append(f'Cluster {k}')
            
            if cluster_centers:
                cluster_matrix = np.array(cluster_centers)
                
                im = ax2.imshow(cluster_matrix.T, cmap='Reds', aspect='auto')
                ax2.set_xticks(range(len(cluster_names)))
                ax2.set_xticklabels(cluster_names, rotation=45, ha='right')
                ax2.set_yticks(range(len(constraint_columns)))
                ax2.set_yticklabels([col.replace('_', ' ').title() for col in constraint_columns])
                ax2.set_title('Cluster Violation Patterns', fontweight='bold')
                
                # Add text annotations
                for i in range(len(constraint_columns)):
                    for j in range(len(cluster_names)):
                        text = f'{cluster_matrix[j, i]:.2f}'
                        ax2.text(j, i, text, ha='center', va='center', 
                               color='white' if cluster_matrix[j, i] > 0.5 else 'black')
                
                plt.colorbar(im, ax=ax2, label='Violation Probability')
        
        else:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Single constraint - no 2D visualization available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
        
        # Add summary text
        summary_text = f'Clusters found: {n_clusters}\nNoise points: {n_noise}\nTotal violations: {len(violation_data)}'
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class SafetyPlotter:
    """Main class for safety visualization."""
    
    def __init__(self, config: SafetyPlottingConfig = None):
        self.config = config or SafetyPlottingConfig()
        self.violation_plotter = ConstraintViolationPlotter(config)
        self.risk_plotter = RiskVisualization(config)
        self.logger = logging.getLogger(__name__)
    
    def create_safety_dashboard(self, data: pd.DataFrame,
                              safety_analysis: Dict[str, Any] = None) -> plt.Figure:
        """Create comprehensive safety analysis dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        constraint_columns = [col for col in data.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['constraint', 'violation', 'cost', 'margin'])]
        
        if not constraint_columns:
            fig.text(0.5, 0.5, 'No safety-related columns found in data', 
                    ha='center', va='center', fontsize=16)
            return fig
        
        # 1. Overall violation rate over time (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_overall_violation_rate(data, constraint_columns, ax1)
        
        # 2. Safety score over time (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_safety_score(data, constraint_columns, ax2)
        
        # 3. Constraint violation frequencies (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_violation_frequencies_summary(data, constraint_columns, ax3)
        
        # 4. Safety margin distribution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_safety_margins_summary(data, constraint_columns, ax4)
        
        # 5. Risk level over time (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_risk_level_trend(data, constraint_columns, ax5)
        
        # 6. Violation co-occurrence (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_violation_correlation_summary(data, constraint_columns, ax6)
        
        # 7. Time to violation recovery (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_recovery_time_analysis(data, constraint_columns, ax7)
        
        # 8. Safety summary statistics (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_safety_summary_stats(data, constraint_columns, safety_analysis, ax8)
        
        fig.suptitle('Safety Analysis Dashboard', fontsize=18, fontweight='bold')
        return fig
    
    def _plot_overall_violation_rate(self, data: pd.DataFrame, 
                                   constraint_columns: List[str], ax):
        """Plot overall violation rate over time."""
        time_col = 'iteration' if 'iteration' in data.columns else None
        x = data[time_col].values if time_col else range(len(data))
        
        # Compute overall violation rate in windows
        window_size = max(10, len(data) // 50)
        violation_rates = []
        window_centers = []
        
        for i in range(0, len(data) - window_size + 1, window_size // 2):
            window_data = data.iloc[i:i+window_size]
            total_violations = 0
            total_possible = 0
            
            for col in constraint_columns:
                if col in window_data.columns:
                    col_data = window_data[col].dropna()
                    if 'margin' in col.lower():
                        violations = (col_data < 0).sum()
                    else:
                        violations = (col_data > 0).sum()
                    
                    total_violations += violations
                    total_possible += len(col_data)
            
            if total_possible > 0:
                violation_rate = total_violations / total_possible
                violation_rates.append(violation_rate)
                window_centers.append(i + window_size // 2)
        
        if violation_rates:
            ax.plot(window_centers, violation_rates, color=self.config.colors['danger'], 
                   linewidth=2, label='Violation Rate')
            ax.fill_between(window_centers, violation_rates, alpha=0.3, 
                          color=self.config.colors['danger'])
            
            # Add safety threshold line
            safe_threshold = 0.05  # 5% violation rate threshold
            ax.axhline(y=safe_threshold, color=self.config.colors['warning'], 
                      linestyle='--', label=f'Safety Threshold ({safe_threshold*100}%)')
        
        ax.set_title('Overall Violation Rate', fontweight='bold')
        ax.set_ylabel('Violation Rate')
        ax.set_xlabel('Iteration' if time_col else 'Step')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_safety_score(self, data: pd.DataFrame, constraint_columns: List[str], ax):
        """Plot safety score over time."""
        time_col = 'iteration' if 'iteration' in data.columns else None
        x = data[time_col].values if time_col else range(len(data))
        
        # Compute safety score (1 - violation_rate)
        window_size = max(10, len(data) // 50)
        safety_scores = []
        window_centers = []
        
        for i in range(0, len(data) - window_size + 1, window_size // 2):
            window_data = data.iloc[i:i+window_size]
            total_violations = 0
            total_possible = 0
            
            for col in constraint_columns:
                if col in window_data.columns:
                    col_data = window_data[col].dropna()
                    if 'margin' in col.lower():
                        violations = (col_data < 0).sum()
                    else:
                        violations = (col_data > 0).sum()
                    
                    total_violations += violations
                    total_possible += len(col_data)
            
            if total_possible > 0:
                safety_score = 1.0 - (total_violations / total_possible)
                safety_scores.append(max(0, safety_score))  # Ensure non-negative
                window_centers.append(i + window_size // 2)
        
        if safety_scores:
            ax.plot(window_centers, safety_scores, color=self.config.colors['safe'], 
                   linewidth=2, label='Safety Score')
            ax.fill_between(window_centers, safety_scores, alpha=0.3, 
                          color=self.config.colors['safe'])
        
        ax.set_title('Safety Score', fontweight='bold')
        ax.set_ylabel('Safety Score')
        ax.set_xlabel('Iteration' if time_col else 'Step')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
    
    def _plot_violation_frequencies_summary(self, data: pd.DataFrame, 
                                          constraint_columns: List[str], ax):
        """Plot summary of violation frequencies."""
        violation_counts = []
        constraint_names = []
        
        for col in constraint_columns[:5]:  # Limit to first 5 constraints
            if col in data.columns:
                col_data = data[col].dropna()
                if 'margin' in col.lower():
                    violations = (col_data < 0).sum()
                else:
                    violations = (col_data > 0).sum()
                
                violation_counts.append(violations)
                constraint_names.append(col.replace('_', ' ').title()[:15])  # Truncate long names
        
        if violation_counts:
            bars = ax.bar(range(len(constraint_names)), violation_counts, 
                         color=self.config.colors['danger'], alpha=0.7)
            
            # Color bars by frequency
            max_violations = max(violation_counts) if violation_counts else 1
            for bar, count in zip(bars, violation_counts):
                if count > max_violations * 0.7:
                    bar.set_color(self.config.colors['critical'])
                elif count > max_violations * 0.3:
                    bar.set_color(self.config.colors['danger'])
                else:
                    bar.set_color(self.config.colors['warning'])
            
            ax.set_xticks(range(len(constraint_names)))
            ax.set_xticklabels(constraint_names, rotation=45, ha='right')
        
        ax.set_title('Violation Frequencies', fontweight='bold')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
    
    def _plot_safety_margins_summary(self, data: pd.DataFrame, 
                                   constraint_columns: List[str], ax):
        """Plot summary of safety margins."""
        # Find or create margin data
        margin_data = []
        
        for col in constraint_columns:
            if col in data.columns:
                col_data = data[col].dropna().values
                
                if 'margin' in col.lower():
                    margin_data.extend(col_data)
                else:
                    # Convert constraint costs to margins
                    margin_data.extend(-col_data)
        
        if margin_data:
            # Create histogram
            ax.hist(margin_data, bins=30, color=self.config.colors['neutral'], 
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Add safety zones
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Safety Boundary')
            
            # Color regions
            x_min, x_max = ax.get_xlim()
            if x_min < 0:
                ax.axvspan(x_min, 0, color=self.config.colors['danger'], alpha=0.2, label='Danger')
            
            warning_threshold = np.percentile(margin_data, 25) if len(margin_data) > 4 else 0.1
            if warning_threshold > 0:
                ax.axvspan(0, warning_threshold, color=self.config.colors['warning'], 
                          alpha=0.2, label='Warning')
            
            ax.axvspan(max(0, warning_threshold), x_max, color=self.config.colors['safe'], 
                      alpha=0.2, label='Safe')
        
        ax.set_title('Safety Margins', fontweight='bold')
        ax.set_xlabel('Margin')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_level_trend(self, data: pd.DataFrame, constraint_columns: List[str], ax):
        """Plot risk level trend over time."""
        time_col = 'iteration' if 'iteration' in data.columns else None
        x = data[time_col].values if time_col else range(len(data))
        
        # Compute aggregate risk level
        risk_levels = []
        
        for i, row in data.iterrows():
            total_risk = 0
            
            for col in constraint_columns:
                if col in data.columns and not pd.isna(row[col]):
                    if 'margin' in col.lower():
                        risk = max(0, -row[col])  # Negative margin = positive risk
                    else:
                        risk = max(0, row[col])   # Positive constraint cost = positive risk
                    
                    total_risk += risk
            
            risk_levels.append(total_risk)
        
        if risk_levels:
            # Smooth the risk levels
            window_size = max(5, len(risk_levels) // 20)
            if len(risk_levels) >= window_size:
                risk_smooth = pd.Series(risk_levels).rolling(window=window_size, center=True).mean()
                risk_smooth = risk_smooth.fillna(method='bfill').fillna(method='ffill')
            else:
                risk_smooth = risk_levels
            
            ax.plot(x, risk_levels, color=self.config.colors['danger'], 
                   alpha=0.5, linewidth=1, label='Risk Level')
            ax.plot(x, risk_smooth, color=self.config.colors['critical'], 
                   linewidth=2, label='Smoothed Risk')
            
            # Add risk threshold
            high_risk_threshold = np.percentile(risk_levels, 90)
            ax.axhline(y=high_risk_threshold, color=self.config.colors['warning'], 
                      linestyle='--', alpha=0.8, label='High Risk Threshold')
        
        ax.set_title('Risk Level Trend', fontweight='bold')
        ax.set_ylabel('Risk Level')
        ax.set_xlabel('Iteration' if time_col else 'Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_violation_correlation_summary(self, data: pd.DataFrame, 
                                          constraint_columns: List[str], ax):
        """Plot correlation between constraint violations."""
        if len(constraint_columns) < 2:
            ax.text(0.5, 0.5, 'Need ≥2 constraints\nfor correlation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            return
        
        # Create violation matrix
        violation_matrix = np.zeros((len(data), len(constraint_columns)))
        
        for i, col in enumerate(constraint_columns):
            if col in data.columns:
                col_data = data[col].values
                if 'margin' in col.lower():
                    violations = (col_data < 0).astype(int)
                else:
                    violations = (col_data > 0).astype(int)
                violation_matrix[:, i] = violations
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(violation_matrix.T)
        
        # Plot correlation heatmap
        im = ax.imshow(corr_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # Add text annotations
        n_constraints = min(len(constraint_columns), 5)  # Limit display
        for i in range(n_constraints):
            for j in range(n_constraints):
                text = f'{corr_matrix[i, j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                       fontsize=8)
        
        constraint_names_short = [col.replace('_', ' ')[:10] for col in constraint_columns[:n_constraints]]
        ax.set_xticks(range(n_constraints))
        ax.set_yticks(range(n_constraints))
        ax.set_xticklabels(constraint_names_short, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(constraint_names_short, fontsize=8)
        ax.set_title('Violation Correlations', fontweight='bold')
    
    def _plot_recovery_time_analysis(self, data: pd.DataFrame, 
                                   constraint_columns: List[str], ax):
        """Plot recovery time analysis."""
        # Find violation episodes and recovery times
        recovery_times = []
        
        for col in constraint_columns[:3]:  # Analyze first 3 constraints
            if col not in data.columns:
                continue
                
            col_data = data[col].values
            if 'margin' in col.lower():
                violations = col_data < 0
            else:
                violations = col_data > 0
            
            # Find violation episodes
            in_violation = False
            violation_start = 0
            
            for i, is_violation in enumerate(violations):
                if is_violation and not in_violation:
                    # Start of violation
                    in_violation = True
                    violation_start = i
                elif not is_violation and in_violation:
                    # End of violation - compute recovery time
                    recovery_time = i - violation_start
                    recovery_times.append(recovery_time)
                    in_violation = False
        
        if recovery_times:
            ax.hist(recovery_times, bins=min(20, len(recovery_times)), 
                   color=self.config.colors['warning'], alpha=0.7, 
                   edgecolor='black', linewidth=0.5)
            
            mean_recovery = np.mean(recovery_times)
            ax.axvline(x=mean_recovery, color=self.config.colors['danger'], 
                      linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_recovery:.1f} steps')
            
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No violation\nepisodes found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        ax.set_title('Recovery Time Analysis', fontweight='bold')
        ax.set_xlabel('Recovery Time (steps)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    def _plot_safety_summary_stats(self, data: pd.DataFrame, 
                                 constraint_columns: List[str],
                                 safety_analysis: Dict[str, Any], ax):
        """Plot safety summary statistics."""
        ax.axis('off')
        
        # Compute summary statistics
        total_steps = len(data)
        total_violations = 0
        total_constraints = len(constraint_columns)
        
        for col in constraint_columns:
            if col in data.columns:
                col_data = data[col].dropna()
                if 'margin' in col.lower():
                    violations = (col_data < 0).sum()
                else:
                    violations = (col_data > 0).sum()
                total_violations += violations
        
        violation_rate = total_violations / (total_steps * total_constraints) if total_steps > 0 else 0
        safety_score = 1.0 - violation_rate
        
        # Create summary text
        summary_text = "Safety Summary\n" + "="*20 + "\n\n"
        summary_text += f"Total Steps: {total_steps:,}\n"
        summary_text += f"Constraints Monitored: {total_constraints}\n"
        summary_text += f"Total Violations: {total_violations:,}\n"
        summary_text += f"Violation Rate: {violation_rate:.3%}\n"
        summary_text += f"Safety Score: {safety_score:.3f}\n\n"
        
        # Safety classification
        if safety_score >= 0.95:
            safety_class = "EXCELLENT ✓"
            color = self.config.colors['safe']
        elif safety_score >= 0.90:
            safety_class = "GOOD"
            color = self.config.colors['safe']
        elif safety_score >= 0.80:
            safety_class = "ACCEPTABLE"
            color = self.config.colors['warning']
        elif safety_score >= 0.70:
            safety_class = "POOR"
            color = self.config.colors['danger']
        else:
            safety_class = "CRITICAL ⚠"
            color = self.config.colors['critical']
        
        summary_text += f"Safety Rating: {safety_class}"
        
        # Add analysis results if available
        if safety_analysis and 'overall_safety_metrics' in safety_analysis:
            overall_metrics = safety_analysis['overall_safety_metrics']
            if 'safety_budget_utilization' in overall_metrics:
                budget_util = overall_metrics['safety_budget_utilization']
                summary_text += f"\n\nBudget Utilization: {budget_util:.1%}"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    
    def save_all_safety_plots(self, data: pd.DataFrame,
                            safety_analysis: Dict[str, Any] = None,
                            output_dir: str = "safety_plots",
                            formats: List[str] = ['png', 'pdf']) -> Dict[str, str]:
        """Save all safety plots to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        constraint_columns = [col for col in data.columns 
                            if any(keyword in col.lower() 
                                  for keyword in ['constraint', 'violation', 'cost', 'margin'])]
        
        try:
            # Safety dashboard
            fig_dashboard = self.create_safety_dashboard(data, safety_analysis)
            for fmt in formats:
                filename = f"safety_dashboard.{fmt}"
                filepath = output_path / filename
                fig_dashboard.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                saved_files[f'dashboard_{fmt}'] = str(filepath)
            plt.close(fig_dashboard)
            
            # Violation frequency histogram
            fig_violations = self.violation_plotter.plot_violation_frequency_histogram(data, constraint_columns)
            for fmt in formats:
                filename = f"violation_frequencies.{fmt}"
                filepath = output_path / filename
                fig_violations.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                saved_files[f'violations_{fmt}'] = str(filepath)
            plt.close(fig_violations)
            
            # Violation timeline
            fig_timeline = self.violation_plotter.plot_violation_timeline(data, constraint_columns)
            for fmt in formats:
                filename = f"violation_timeline.{fmt}"
                filepath = output_path / filename
                fig_timeline.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                saved_files[f'timeline_{fmt}'] = str(filepath)
            plt.close(fig_timeline)
            
            # Safety margin distribution
            fig_margins = self.risk_plotter.plot_safety_margin_distribution(data)
            for fmt in formats:
                filename = f"safety_margins.{fmt}"
                filepath = output_path / filename
                fig_margins.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                saved_files[f'margins_{fmt}'] = str(filepath)
            plt.close(fig_margins)
            
            # Violation heatmap
            if len(constraint_columns) > 1:
                fig_heatmap = self.violation_plotter.plot_violation_heatmap(data, constraint_columns)
                for fmt in formats:
                    filename = f"violation_heatmap.{fmt}"
                    filepath = output_path / filename
                    fig_heatmap.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                    saved_files[f'heatmap_{fmt}'] = str(filepath)
                plt.close(fig_heatmap)
            
            self.logger.info(f"Saved {len(saved_files)} safety plot files to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving safety plots: {e}")
        
        return saved_files