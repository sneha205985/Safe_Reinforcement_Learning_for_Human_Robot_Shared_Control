"""
Visualization and Rendering Components for Shared Control Environments.

This module provides comprehensive visualization capabilities for human-robot
shared control environments, including real-time safety monitoring, trajectory
visualization, and performance metrics display.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, Arrow, Wedge
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import time
import threading
from collections import deque
from abc import ABC, abstractmethod

from .safety_monitoring import SafetyStatus, SafetyLevel
from .human_robot_env import EnvironmentState

logger = logging.getLogger(__name__)

# Optional imports for advanced visualization
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from mayavi import mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    # General settings
    update_rate: float = 30.0  # FPS
    window_size: Tuple[int, int] = (1200, 800)
    background_color: str = "white"
    
    # Safety visualization
    show_safety_zones: bool = True
    show_constraint_margins: bool = True
    show_predicted_violations: bool = True
    safety_color_scheme: Dict[str, str] = None
    
    # Trajectory visualization  
    trajectory_history_length: int = 1000
    show_planned_path: bool = True
    show_human_intent: bool = True
    
    # Performance metrics
    show_real_time_metrics: bool = True
    metric_history_length: int = 500
    
    def __post_init__(self):
        if self.safety_color_scheme is None:
            self.safety_color_scheme = {
                'safe': '#2ECC71',      # Green
                'warning': '#F39C12',    # Orange  
                'critical': '#E74C3C',   # Red
                'emergency': '#8E44AD'   # Purple
            }


class BaseVisualizer(ABC):
    """Base class for environment visualizers."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.active = False
        self.data_history = deque(maxlen=config.trajectory_history_length)
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize visualization components."""
        pass
    
    @abstractmethod
    def update(self, 
              env_state: EnvironmentState,
              safety_status: SafetyStatus,
              metrics: Dict[str, Any]) -> None:
        """Update visualization with new data."""
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render current frame."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up visualization resources."""
        pass


class MatplotlibVisualizer(BaseVisualizer):
    """Matplotlib-based visualizer for 2D environments and plots."""
    
    def __init__(self, config: VisualizationConfig, env_type: str = "wheelchair"):
        super().__init__(config)
        self.env_type = env_type
        
        # Matplotlib components
        self.fig = None
        self.axes = None
        self.animation = None
        
        # Data containers
        self.robot_positions = deque(maxlen=config.trajectory_history_length)
        self.safety_history = deque(maxlen=config.metric_history_length)
        self.performance_history = deque(maxlen=config.metric_history_length)
        
        # Plot elements
        self.robot_artist = None
        self.trajectory_artist = None
        self.safety_artists = []
        self.metric_artists = []
        
        logger.info(f"MatplotlibVisualizer initialized for {env_type} environment")
    
    def initialize(self) -> bool:
        """Initialize matplotlib visualization."""
        try:
            # Set up the figure based on environment type
            if self.env_type == "wheelchair":
                self._initialize_wheelchair_visualization()
            elif self.env_type == "exoskeleton":
                self._initialize_exoskeleton_visualization()
            else:
                self._initialize_generic_visualization()
            
            plt.ion()  # Interactive mode
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize matplotlib visualization: {e}")
            return False
    
    def _initialize_wheelchair_visualization(self) -> None:
        """Initialize visualization for wheelchair environment."""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main environment view (2D top-down)
        self.ax_main = self.fig.add_subplot(gs[:2, :2])
        self.ax_main.set_title("Wheelchair Navigation Environment", fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel("X Position (m)")
        self.ax_main.set_ylabel("Y Position (m)")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Safety status panel
        self.ax_safety = self.fig.add_subplot(gs[0, 2])
        self.ax_safety.set_title("Safety Status", fontsize=12, fontweight='bold')
        self.ax_safety.axis('off')
        
        # Performance metrics
        self.ax_metrics = self.fig.add_subplot(gs[0, 3])
        self.ax_metrics.set_title("Performance Metrics", fontsize=12)
        
        # Safety margins over time
        self.ax_safety_history = self.fig.add_subplot(gs[1, 2:])
        self.ax_safety_history.set_title("Safety Margins", fontsize=12)
        self.ax_safety_history.set_xlabel("Time (s)")
        self.ax_safety_history.set_ylabel("Safety Margin")
        
        # Human-robot interaction metrics
        self.ax_interaction = self.fig.add_subplot(gs[2, :2])
        self.ax_interaction.set_title("Human-Robot Interaction", fontsize=12)
        self.ax_interaction.set_xlabel("Time (s)")
        self.ax_interaction.set_ylabel("Assistance Level")
        
        # Control effort
        self.ax_control = self.fig.add_subplot(gs[2, 2:])
        self.ax_control.set_title("Control Effort", fontsize=12)
        self.ax_control.set_xlabel("Time (s)")
        self.ax_control.set_ylabel("Command Magnitude")
        
        self.axes = [self.ax_main, self.ax_safety, self.ax_metrics, 
                    self.ax_safety_history, self.ax_interaction, self.ax_control]
    
    def _initialize_exoskeleton_visualization(self) -> None:
        """Initialize visualization for exoskeleton environment."""
        self.fig = plt.figure(figsize=(16, 12))
        
        gs = self.fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
        
        # 3D arm visualization (simplified 2D projection)
        self.ax_main = self.fig.add_subplot(gs[:2, :2])
        self.ax_main.set_title("Exoskeleton Arm Configuration", fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel("X Position (m)")
        self.ax_main.set_ylabel("Z Position (m)")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Joint angles display
        self.ax_joints = self.fig.add_subplot(gs[0, 2:])
        self.ax_joints.set_title("Joint Angles", fontsize=12)
        self.ax_joints.set_xlabel("Joint Index")
        self.ax_joints.set_ylabel("Angle (rad)")
        
        # EMG signals (if available)
        self.ax_emg = self.fig.add_subplot(gs[1, 2:])
        self.ax_emg.set_title("Muscle Activity (EMG)", fontsize=12)
        self.ax_emg.set_xlabel("Muscle Group")
        self.ax_emg.set_ylabel("Activation Level")
        
        # Safety status
        self.ax_safety = self.fig.add_subplot(gs[2, 0])
        self.ax_safety.set_title("Safety Status", fontsize=12, fontweight='bold')
        self.ax_safety.axis('off')
        
        # Task progress
        self.ax_task = self.fig.add_subplot(gs[2, 1])
        self.ax_task.set_title("Task Progress", fontsize=12)
        
        # Force/torque levels
        self.ax_forces = self.fig.add_subplot(gs[2, 2:])
        self.ax_forces.set_title("Joint Torques", fontsize=12)
        self.ax_forces.set_xlabel("Joint Index")
        self.ax_forces.set_ylabel("Torque (Nm)")
        
        # Performance over time
        self.ax_performance = self.fig.add_subplot(gs[3, :])
        self.ax_performance.set_title("Performance Metrics Over Time", fontsize=12)
        self.ax_performance.set_xlabel("Time (s)")
        
        self.axes = [self.ax_main, self.ax_joints, self.ax_emg, self.ax_safety, 
                    self.ax_task, self.ax_forces, self.ax_performance]
    
    def _initialize_generic_visualization(self) -> None:
        """Initialize generic visualization layout."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Shared Control Environment", fontsize=16, fontweight='bold')
        
        # Main environment view
        self.ax_main = self.axes[0, 0]
        self.ax_main.set_title("Environment State")
        
        # Safety status
        self.ax_safety = self.axes[0, 1] 
        self.ax_safety.set_title("Safety Status")
        
        # Performance metrics
        self.ax_metrics = self.axes[1, 0]
        self.ax_metrics.set_title("Performance Metrics")
        
        # Additional info
        self.ax_info = self.axes[1, 1]
        self.ax_info.set_title("System Information")
    
    def update(self, 
              env_state: EnvironmentState,
              safety_status: SafetyStatus,
              metrics: Dict[str, Any]) -> None:
        """Update visualization with new data."""
        current_time = time.time()
        
        # Store data in history
        self.data_history.append({
            'time': current_time,
            'env_state': env_state,
            'safety_status': safety_status,
            'metrics': metrics
        })
        
        # Update based on environment type
        if self.env_type == "wheelchair":
            self._update_wheelchair_visualization(env_state, safety_status, metrics)
        elif self.env_type == "exoskeleton":
            self._update_exoskeleton_visualization(env_state, safety_status, metrics)
        else:
            self._update_generic_visualization(env_state, safety_status, metrics)
    
    def _update_wheelchair_visualization(self, 
                                       env_state: EnvironmentState,
                                       safety_status: SafetyStatus,
                                       metrics: Dict[str, Any]) -> None:
        """Update wheelchair-specific visualization."""
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Re-initialize titles and labels
        self._initialize_wheelchair_visualization()
        
        # Get wheelchair position from metrics or environment state
        wheelchair_pos = metrics.get('wheelchair_state', {}).get('position', [0, 0])
        wheelchair_orientation = metrics.get('wheelchair_state', {}).get('orientation', 0)
        
        # Main environment view
        self.ax_main.set_xlim(-1, 11)
        self.ax_main.set_ylim(-1, 11)
        
        # Draw obstacles
        if hasattr(env_state, 'obstacles'):
            for obstacle in env_state.obstacles:
                if hasattr(obstacle, 'position') and hasattr(obstacle, 'radius'):
                    circle = Circle(obstacle.position[:2], obstacle.radius, 
                                  color='red', alpha=0.7, label='Obstacle')
                    self.ax_main.add_patch(circle)
        
        # Draw wheelchair
        wheelchair_size = 0.3
        wheelchair_rect = Rectangle(
            (wheelchair_pos[0] - wheelchair_size/2, wheelchair_pos[1] - wheelchair_size/2),
            wheelchair_size, wheelchair_size,
            angle=np.degrees(wheelchair_orientation),
            color='blue', alpha=0.8
        )
        self.ax_main.add_patch(wheelchair_rect)
        
        # Draw trajectory
        if len(self.data_history) > 1:
            positions = []
            for data_point in list(self.data_history)[-50:]:  # Last 50 points
                pos = data_point['metrics'].get('wheelchair_state', {}).get('position', [0, 0])
                positions.append(pos)
            
            if positions:
                positions = np.array(positions)
                self.ax_main.plot(positions[:, 0], positions[:, 1], 
                                'b--', alpha=0.6, linewidth=2, label='Trajectory')
        
        # Draw goal
        goal_pos = metrics.get('goal_position', [5, 5])
        self.ax_main.scatter(goal_pos[0], goal_pos[1], 
                           marker='*', s=200, color='green', label='Goal')
        
        # Draw planned path
        planned_path = metrics.get('planned_path', [])
        if planned_path:
            path_array = np.array([p[:2] for p in planned_path])
            self.ax_main.plot(path_array[:, 0], path_array[:, 1], 
                            'g:', alpha=0.5, linewidth=1, label='Planned Path')
        
        self.ax_main.legend()
        
        # Safety status
        self._draw_safety_status(self.ax_safety, safety_status)
        
        # Performance metrics
        self._draw_performance_metrics(self.ax_metrics, metrics)
        
        # Safety margins over time
        self._draw_safety_history(self.ax_safety_history)
        
        # Human-robot interaction
        self._draw_interaction_metrics(self.ax_interaction)
        
        # Control effort
        self._draw_control_effort(self.ax_control, metrics)
    
    def _update_exoskeleton_visualization(self, 
                                        env_state: EnvironmentState,
                                        safety_status: SafetyStatus,
                                        metrics: Dict[str, Any]) -> None:
        """Update exoskeleton-specific visualization."""
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Re-initialize
        self._initialize_exoskeleton_visualization()
        
        # Draw arm configuration
        self._draw_arm_configuration(self.ax_main, env_state)
        
        # Joint angles
        joint_positions = env_state.robot.position.numpy()
        self.ax_joints.bar(range(len(joint_positions)), joint_positions, 
                          color='skyblue', alpha=0.7)
        self.ax_joints.set_ylim(-3.14, 3.14)
        
        # EMG signals (if available)
        if hasattr(env_state.human_input, 'muscle_activity') and env_state.human_input.muscle_activity is not None:
            emg_signals = env_state.human_input.muscle_activity.numpy()
            muscle_names = ['Deltoid A', 'Deltoid P', 'Deltoid M', 'Latissimus', 
                           'Biceps', 'Triceps', 'Forearm F', 'Forearm E']
            self.ax_emg.bar(range(len(emg_signals)), emg_signals, 
                           color='orange', alpha=0.7)
            self.ax_emg.set_xticks(range(len(muscle_names)))
            self.ax_emg.set_xticklabels(muscle_names, rotation=45)
            self.ax_emg.set_ylim(0, 1)
        
        # Safety status
        self._draw_safety_status(self.ax_safety, safety_status)
        
        # Task progress
        distance_to_target = metrics.get('distance_to_target', 1.0)
        progress = max(0, 1.0 - distance_to_target / 2.0)  # Assuming 2m max distance
        self.ax_task.pie([progress, 1-progress], labels=['Complete', 'Remaining'], 
                        colors=['green', 'lightgray'], startangle=90)
        
        # Joint torques
        joint_torques = env_state.robot.torque.numpy()
        colors = ['red' if abs(t) > 20 else 'blue' for t in joint_torques]
        self.ax_forces.bar(range(len(joint_torques)), joint_torques, color=colors, alpha=0.7)
        
        # Performance over time
        self._draw_performance_history(self.ax_performance)
    
    def _update_generic_visualization(self, 
                                    env_state: EnvironmentState,
                                    safety_status: SafetyStatus,
                                    metrics: Dict[str, Any]) -> None:
        """Update generic visualization."""
        # Clear axes
        for ax_row in self.axes:
            for ax in ax_row:
                ax.clear()
        
        # Safety status
        self._draw_safety_status(self.axes[0, 1], safety_status)
        
        # Performance metrics  
        self._draw_performance_metrics(self.axes[1, 0], metrics)
    
    def _draw_arm_configuration(self, ax, env_state: EnvironmentState) -> None:
        """Draw arm configuration for exoskeleton."""
        # Simplified 2D arm drawing
        joint_positions = env_state.robot.position.numpy()
        
        # Link lengths (simplified)
        link_lengths = [0.3, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        
        # Compute joint locations
        x, y = [0], [0]  # Base position
        current_angle = 0
        
        for i, (angle, length) in enumerate(zip(joint_positions, link_lengths)):
            if i < 3:  # First 3 joints affect 2D position significantly
                current_angle += angle
                next_x = x[-1] + length * np.cos(current_angle)
                next_y = y[-1] + length * np.sin(current_angle)
                x.append(next_x)
                y.append(next_y)
        
        # Draw links
        ax.plot(x, y, 'b-', linewidth=3, marker='o', markersize=6)
        
        # Draw target if available
        if hasattr(env_state, 'target_position'):
            target = env_state.target_position.numpy()
            ax.scatter(target[0], target[2], marker='*', s=200, color='red', label='Target')
        
        # Draw end-effector
        ax.scatter(x[-1], y[-1], marker='s', s=100, color='green', label='End-Effector')
        
        ax.legend()
        ax.set_xlim(-1, 1.5)
        ax.set_ylim(-0.5, 1.5)
    
    def _draw_safety_status(self, ax, safety_status: SafetyStatus) -> None:
        """Draw safety status panel."""
        ax.text(0.1, 0.8, f"Safety Level: {safety_status.overall_level.value.upper()}", 
                fontsize=14, fontweight='bold',
                color=self.config.safety_color_scheme.get(safety_status.overall_level.value, 'black'),
                transform=ax.transAxes)
        
        ax.text(0.1, 0.6, f"Active Violations: {len(safety_status.active_violations)}", 
                fontsize=12, transform=ax.transAxes)
        
        ax.text(0.1, 0.4, f"Predicted Violations: {len(safety_status.predicted_violations)}", 
                fontsize=12, transform=ax.transAxes)
        
        ax.text(0.1, 0.2, f"Emergency Stop: {'ACTIVE' if safety_status.emergency_stop_active else 'OFF'}", 
                fontsize=12, 
                color='red' if safety_status.emergency_stop_active else 'green',
                fontweight='bold' if safety_status.emergency_stop_active else 'normal',
                transform=ax.transAxes)
    
    def _draw_performance_metrics(self, ax, metrics: Dict[str, Any]) -> None:
        """Draw key performance metrics."""
        metric_names = []
        metric_values = []
        
        # Extract key metrics
        if 'distance_to_goal' in metrics:
            metric_names.append('Dist to Goal')
            metric_values.append(metrics['distance_to_goal'])
        
        if 'human_confidence' in metrics:
            metric_names.append('Human Conf')
            metric_values.append(metrics['human_confidence'])
        
        if 'assistance_level' in metrics:
            metric_names.append('Assistance')
            metric_values.append(metrics['assistance_level'])
        
        if metric_names:
            colors = ['green' if v < 0.5 else 'orange' if v < 0.8 else 'red' for v in metric_values]
            bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
    
    def _draw_safety_history(self, ax) -> None:
        """Draw safety margins over time."""
        if len(self.data_history) < 2:
            return
        
        times = []
        safety_levels = []
        
        for data_point in list(self.data_history)[-100:]:  # Last 100 points
            times.append(data_point['time'])
            
            # Convert safety level to numeric
            level = data_point['safety_status'].overall_level
            level_map = {
                SafetyLevel.SAFE: 1.0,
                SafetyLevel.WARNING: 0.6,
                SafetyLevel.CRITICAL: 0.3,
                SafetyLevel.EMERGENCY: 0.0
            }
            safety_levels.append(level_map.get(level, 0.5))
        
        if times:
            # Normalize times
            times = np.array(times) - times[0]
            ax.plot(times, safety_levels, 'b-', linewidth=2)
            ax.fill_between(times, 0, safety_levels, alpha=0.3)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel('Safety Level')
    
    def _draw_interaction_metrics(self, ax) -> None:
        """Draw human-robot interaction metrics."""
        if len(self.data_history) < 2:
            return
        
        times = []
        human_confidence = []
        assistance_levels = []
        
        for data_point in list(self.data_history)[-100:]:
            times.append(data_point['time'])
            human_confidence.append(data_point['metrics'].get('human_confidence', 0.5))
            assistance_levels.append(data_point['metrics'].get('assistance_level', 0.5))
        
        if times:
            times = np.array(times) - times[0]
            ax.plot(times, human_confidence, 'g-', label='Human Confidence', linewidth=2)
            ax.plot(times, assistance_levels, 'b-', label='Robot Assistance', linewidth=2)
            ax.set_ylim(0, 1)
            ax.legend()
    
    def _draw_control_effort(self, ax, metrics: Dict[str, Any]) -> None:
        """Draw control effort metrics."""
        # This would show command magnitudes over time
        if len(self.data_history) < 2:
            return
        
        times = []
        control_efforts = []
        
        for data_point in list(self.data_history)[-100:]:
            times.append(data_point['time'])
            # Extract control effort from metrics
            effort = data_point['metrics'].get('control_effort', 0)
            control_efforts.append(effort)
        
        if times:
            times = np.array(times) - times[0]
            ax.plot(times, control_efforts, 'r-', linewidth=2)
            ax.set_ylabel('Control Effort')
    
    def _draw_performance_history(self, ax) -> None:
        """Draw performance metrics history."""
        if len(self.data_history) < 2:
            return
        
        times = []
        distances = []
        rewards = []
        
        for data_point in list(self.data_history)[-200:]:
            times.append(data_point['time'])
            distances.append(data_point['metrics'].get('distance_to_target', 1.0))
            rewards.append(data_point['metrics'].get('episode_reward', 0))
        
        if times:
            times = np.array(times) - times[0]
            
            # Dual y-axis plot
            ax2 = ax.twinx()
            
            line1 = ax.plot(times, distances, 'b-', label='Distance to Target', linewidth=2)
            line2 = ax2.plot(times, rewards, 'g-', label='Cumulative Reward', linewidth=2)
            
            ax.set_ylabel('Distance (m)', color='b')
            ax2.set_ylabel('Reward', color='g')
            
            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
    
    def render(self) -> None:
        """Render current frame."""
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Small pause to allow updates
    
    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        if self.fig:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
    
    def close(self) -> None:
        """Clean up matplotlib resources."""
        if self.fig:
            plt.close(self.fig)
        plt.ioff()
        logger.info("MatplotlibVisualizer closed")


class PlotlyVisualizer(BaseVisualizer):
    """Plotly-based visualizer for interactive web-based visualization."""
    
    def __init__(self, config: VisualizationConfig, env_type: str = "wheelchair"):
        super().__init__(config)
        self.env_type = env_type
        
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for PlotlyVisualizer")
        
        self.fig = None
        logger.info(f"PlotlyVisualizer initialized for {env_type} environment")
    
    def initialize(self) -> bool:
        """Initialize plotly visualization."""
        try:
            # Create subplots based on environment type
            if self.env_type == "wheelchair":
                self.fig = sp.make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Environment', 'Safety Status', 
                                  'Performance Metrics', 'Control History'),
                    specs=[[{"type": "scatter"}, {"type": "indicator"}],
                           [{"type": "scatter"}, {"type": "scatter"}]]
                )
            else:
                self.fig = sp.make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Arm Configuration', 'Joint Angles',
                                  'Safety Metrics', 'Performance'),
                    specs=[[{"type": "scatter"}, {"type": "bar"}],
                           [{"type": "scatter"}, {"type": "scatter"}]]
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plotly visualization: {e}")
            return False
    
    def update(self, 
              env_state: EnvironmentState,
              safety_status: SafetyStatus,
              metrics: Dict[str, Any]) -> None:
        """Update plotly visualization."""
        # Store data
        self.data_history.append({
            'time': time.time(),
            'env_state': env_state,
            'safety_status': safety_status,
            'metrics': metrics
        })
        
        # Update plots based on environment type
        if self.env_type == "wheelchair":
            self._update_wheelchair_plotly(env_state, safety_status, metrics)
        else:
            self._update_exoskeleton_plotly(env_state, safety_status, metrics)
    
    def _update_wheelchair_plotly(self, env_state, safety_status, metrics):
        """Update wheelchair visualization in plotly."""
        # Clear existing traces
        self.fig.data = []
        
        # Environment view
        wheelchair_pos = metrics.get('wheelchair_state', {}).get('position', [0, 0])
        
        # Add wheelchair position
        self.fig.add_trace(
            go.Scatter(x=[wheelchair_pos[0]], y=[wheelchair_pos[1]], 
                      mode='markers', marker=dict(size=15, color='blue'),
                      name='Wheelchair'),
            row=1, col=1
        )
        
        # Add obstacles
        if hasattr(env_state, 'obstacles'):
            for i, obstacle in enumerate(env_state.obstacles):
                if hasattr(obstacle, 'position'):
                    self.fig.add_trace(
                        go.Scatter(x=[obstacle.position[0]], y=[obstacle.position[1]],
                                  mode='markers', marker=dict(size=20, color='red'),
                                  name=f'Obstacle {i+1}'),
                        row=1, col=1
                    )
        
        # Safety indicator
        safety_color = self.config.safety_color_scheme.get(safety_status.overall_level.value, 'gray')
        self.fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=len(safety_status.active_violations),
                title={'text': "Safety Violations"},
                gauge={'bar': {'color': safety_color}},
            ),
            row=1, col=2
        )
    
    def _update_exoskeleton_plotly(self, env_state, safety_status, metrics):
        """Update exoskeleton visualization in plotly."""
        # Implementation for exoskeleton-specific plotly visualization
        pass
    
    def render(self) -> None:
        """Render plotly figure."""
        if self.fig:
            plot(self.fig, filename='shared_control_viz.html', auto_open=False)
    
    def close(self) -> None:
        """Clean up plotly resources."""
        logger.info("PlotlyVisualizer closed")


class VisualizationManager:
    """Manages multiple visualizers and coordinates updates."""
    
    def __init__(self, 
                 config: VisualizationConfig,
                 env_type: str = "wheelchair",
                 visualizer_types: List[str] = ["matplotlib"]):
        """
        Initialize visualization manager.
        
        Args:
            config: Visualization configuration
            env_type: Environment type
            visualizer_types: List of visualizer types to use
        """
        self.config = config
        self.env_type = env_type
        self.visualizers = []
        
        # Initialize requested visualizers
        for viz_type in visualizer_types:
            try:
                if viz_type == "matplotlib":
                    visualizer = MatplotlibVisualizer(config, env_type)
                elif viz_type == "plotly" and PLOTLY_AVAILABLE:
                    visualizer = PlotlyVisualizer(config, env_type)
                else:
                    logger.warning(f"Visualizer type '{viz_type}' not available")
                    continue
                
                if visualizer.initialize():
                    self.visualizers.append(visualizer)
                    logger.info(f"Added {viz_type} visualizer")
                
            except Exception as e:
                logger.error(f"Failed to initialize {viz_type} visualizer: {e}")
        
        # Update control
        self.last_update_time = 0
        self.update_interval = 1.0 / config.update_rate
        
        logger.info(f"VisualizationManager initialized with {len(self.visualizers)} visualizers")
    
    def update(self, 
              env_state: EnvironmentState,
              safety_status: SafetyStatus,
              metrics: Dict[str, Any]) -> None:
        """Update all visualizers."""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < self.update_interval:
            return
        
        # Update all visualizers
        for visualizer in self.visualizers:
            try:
                visualizer.update(env_state, safety_status, metrics)
            except Exception as e:
                logger.error(f"Error updating visualizer: {e}")
        
        self.last_update_time = current_time
    
    def render(self) -> None:
        """Render all visualizers."""
        for visualizer in self.visualizers:
            try:
                visualizer.render()
            except Exception as e:
                logger.error(f"Error rendering visualizer: {e}")
    
    def save_frame(self, filename_prefix: str = "frame") -> None:
        """Save current frame from all visualizers."""
        for i, visualizer in enumerate(self.visualizers):
            try:
                if hasattr(visualizer, 'save_frame'):
                    filename = f"{filename_prefix}_{i}_{type(visualizer).__name__}.png"
                    visualizer.save_frame(filename)
            except Exception as e:
                logger.error(f"Error saving frame: {e}")
    
    def close(self) -> None:
        """Close all visualizers."""
        for visualizer in self.visualizers:
            try:
                visualizer.close()
            except Exception as e:
                logger.error(f"Error closing visualizer: {e}")
        
        self.visualizers.clear()
        logger.info("VisualizationManager closed")


def create_visualization_config(env_type: str = "wheelchair", 
                               advanced_features: bool = True) -> VisualizationConfig:
    """
    Create visualization configuration for specific environment type.
    
    Args:
        env_type: Environment type
        advanced_features: Whether to enable advanced visualization features
        
    Returns:
        Visualization configuration
    """
    config = VisualizationConfig()
    
    if env_type == "wheelchair":
        config.window_size = (1400, 900)
        config.show_safety_zones = True
        config.show_planned_path = True
    elif env_type == "exoskeleton":
        config.window_size = (1600, 1200)
        config.show_safety_zones = True
        config.show_human_intent = True
    
    if advanced_features:
        config.show_predicted_violations = True
        config.show_real_time_metrics = True
        config.metric_history_length = 1000
    else:
        config.show_predicted_violations = False
        config.metric_history_length = 100
    
    return config