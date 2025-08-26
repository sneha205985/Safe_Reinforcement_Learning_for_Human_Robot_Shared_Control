"""
Evaluation protocols and metrics for Safe RL training.

This module provides comprehensive evaluation capabilities for trained CPO models
including performance metrics, safety analysis, and statistical testing.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from ..algorithms.cpo import CPOAgent
from ..environments.shared_control_base import SharedControlEnvironment
from .experiment_tracker import ExperimentMetrics


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for Safe RL."""
    
    # Performance metrics
    mean_return: float = 0.0
    std_return: float = 0.0
    median_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0
    
    # Success metrics
    success_rate: float = 0.0
    completion_rate: float = 0.0
    mean_episode_length: float = 0.0
    std_episode_length: float = 0.0
    
    # Safety metrics
    constraint_violation_rate: float = 0.0
    mean_constraint_cost: float = 0.0
    max_constraint_violation: float = 0.0
    safety_margin: float = 0.0
    
    # Human-robot interaction metrics
    mean_human_effort: float = 0.0
    mean_robot_assistance: float = 0.0
    collaboration_efficiency: float = 0.0
    user_comfort_score: float = 0.0
    
    # Statistical confidence
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    
    # Environment-specific metrics
    environment_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.environment_metrics is None:
            self.environment_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result['confidence_interval_95'] = list(self.confidence_interval_95)
        return result


class EvaluationProtocol:
    """Base class for evaluation protocols."""
    
    def __init__(self, name: str, num_episodes: int = 100, 
                 deterministic: bool = True):
        self.name = name
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def evaluate(self, agent: CPOAgent, environment: SharedControlEnvironment,
                 render: bool = False) -> EvaluationMetrics:
        """Run evaluation protocol."""
        self.logger.info(f"Starting {self.name} evaluation with {self.num_episodes} episodes")
        
        # Collect episode data
        episode_data = self._run_episodes(agent, environment, render)
        
        # Compute metrics
        metrics = self._compute_metrics(episode_data)
        
        self.logger.info(f"Evaluation completed - Mean return: {metrics.mean_return:.3f}, "
                        f"Success rate: {metrics.success_rate:.3f}")
        
        return metrics
    
    def _run_episodes(self, agent: CPOAgent, environment: SharedControlEnvironment,
                      render: bool = False) -> List[Dict[str, Any]]:
        """Run evaluation episodes and collect data."""
        episode_data = []
        
        for episode in range(self.num_episodes):
            if episode % 10 == 0:
                self.logger.debug(f"Episode {episode}/{self.num_episodes}")
            
            # Reset environment
            state = environment.reset()
            episode_return = 0.0
            episode_length = 0
            constraint_violations = 0
            constraint_costs = []
            human_effort = []
            robot_assistance = []
            
            done = False
            while not done:
                # Get action from agent
                if self.deterministic:
                    action = agent.select_action(state, deterministic=True)
                else:
                    action = agent.select_action(state, deterministic=False)
                
                # Take step in environment
                next_state, reward, done, info = environment.step(action)
                
                # Collect metrics
                episode_return += reward
                episode_length += 1
                
                # Safety metrics
                if 'constraint_cost' in info:
                    constraint_costs.append(info['constraint_cost'])
                    if info['constraint_cost'] > 0:
                        constraint_violations += 1
                
                # Human-robot interaction metrics
                if 'human_effort' in info:
                    human_effort.append(info['human_effort'])
                if 'robot_assistance' in info:
                    robot_assistance.append(info['robot_assistance'])
                
                state = next_state
                
                if render and episode < 5:  # Only render first few episodes
                    environment.render()
            
            # Store episode data
            episode_info = {
                'return': episode_return,
                'length': episode_length,
                'success': info.get('success', False),
                'constraint_violations': constraint_violations,
                'constraint_costs': constraint_costs,
                'human_effort': human_effort,
                'robot_assistance': robot_assistance,
                'info': info
            }
            
            episode_data.append(episode_info)
        
        return episode_data
    
    def _compute_metrics(self, episode_data: List[Dict[str, Any]]) -> EvaluationMetrics:
        """Compute evaluation metrics from episode data."""
        returns = [ep['return'] for ep in episode_data]
        lengths = [ep['length'] for ep in episode_data]
        successes = [ep['success'] for ep in episode_data]
        
        # Performance metrics
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        median_return = float(np.median(returns))
        min_return = float(np.min(returns))
        max_return = float(np.max(returns))
        
        # Success metrics
        success_rate = float(np.mean(successes))
        completion_rate = success_rate  # Assuming success = completion
        mean_episode_length = float(np.mean(lengths))
        std_episode_length = float(np.std(lengths))
        
        # Safety metrics
        total_violations = sum(ep['constraint_violations'] for ep in episode_data)
        total_steps = sum(lengths)
        constraint_violation_rate = total_violations / total_steps if total_steps > 0 else 0.0
        
        all_constraint_costs = []
        for ep in episode_data:
            all_constraint_costs.extend(ep['constraint_costs'])
        
        mean_constraint_cost = float(np.mean(all_constraint_costs)) if all_constraint_costs else 0.0
        max_constraint_violation = float(np.max(all_constraint_costs)) if all_constraint_costs else 0.0
        safety_margin = 1.0 - constraint_violation_rate  # Simple safety margin
        
        # Human-robot interaction metrics
        all_human_effort = []
        all_robot_assistance = []
        
        for ep in episode_data:
            all_human_effort.extend(ep['human_effort'])
            all_robot_assistance.extend(ep['robot_assistance'])
        
        mean_human_effort = float(np.mean(all_human_effort)) if all_human_effort else 0.0
        mean_robot_assistance = float(np.mean(all_robot_assistance)) if all_robot_assistance else 0.0
        
        # Collaboration efficiency (lower human effort + higher success = better)
        collaboration_efficiency = success_rate * (1.0 - mean_human_effort) if mean_human_effort <= 1.0 else 0.0
        user_comfort_score = 1.0 - mean_constraint_cost  # Inverse of constraint cost
        
        # Statistical confidence (95% confidence interval for mean return)
        if len(returns) > 1:
            sem = stats.sem(returns)
            confidence_interval = stats.t.interval(0.95, len(returns)-1, loc=mean_return, scale=sem)
        else:
            confidence_interval = (mean_return, mean_return)
        
        # Environment-specific metrics
        environment_metrics = self._compute_environment_specific_metrics(episode_data)
        
        return EvaluationMetrics(
            mean_return=mean_return,
            std_return=std_return,
            median_return=median_return,
            min_return=min_return,
            max_return=max_return,
            success_rate=success_rate,
            completion_rate=completion_rate,
            mean_episode_length=mean_episode_length,
            std_episode_length=std_episode_length,
            constraint_violation_rate=constraint_violation_rate,
            mean_constraint_cost=mean_constraint_cost,
            max_constraint_violation=max_constraint_violation,
            safety_margin=safety_margin,
            mean_human_effort=mean_human_effort,
            mean_robot_assistance=mean_robot_assistance,
            collaboration_efficiency=collaboration_efficiency,
            user_comfort_score=user_comfort_score,
            confidence_interval_95=confidence_interval,
            sample_size=len(episode_data),
            environment_metrics=environment_metrics
        )
    
    def _compute_environment_specific_metrics(self, episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute environment-specific metrics. Override in subclasses."""
        return {}


class StandardEvaluation(EvaluationProtocol):
    """Standard evaluation protocol for general performance assessment."""
    
    def __init__(self, num_episodes: int = 100):
        super().__init__("Standard", num_episodes, deterministic=True)


class StochasticEvaluation(EvaluationProtocol):
    """Stochastic evaluation to assess policy robustness."""
    
    def __init__(self, num_episodes: int = 200):
        super().__init__("Stochastic", num_episodes, deterministic=False)


class SafetyEvaluation(EvaluationProtocol):
    """Focused safety evaluation with challenging scenarios."""
    
    def __init__(self, num_episodes: int = 150):
        super().__init__("Safety", num_episodes, deterministic=True)
        self.challenge_levels = ['normal', 'challenging', 'extreme']
    
    def evaluate(self, agent: CPOAgent, environment: SharedControlEnvironment,
                 render: bool = False) -> Dict[str, EvaluationMetrics]:
        """Evaluate across different challenge levels."""
        results = {}
        
        for level in self.challenge_levels:
            self.logger.info(f"Evaluating safety at {level} challenge level")
            
            # Configure environment for challenge level
            self._configure_challenge_level(environment, level)
            
            # Run evaluation
            metrics = super().evaluate(agent, environment, render)
            results[level] = metrics
        
        return results
    
    def _configure_challenge_level(self, environment: SharedControlEnvironment, 
                                   level: str) -> None:
        """Configure environment for different challenge levels."""
        if hasattr(environment, 'set_safety_challenge_level'):
            environment.set_safety_challenge_level(level)
        elif hasattr(environment, 'difficulty'):
            if level == 'normal':
                environment.difficulty = 0.3
            elif level == 'challenging':
                environment.difficulty = 0.7
            elif level == 'extreme':
                environment.difficulty = 1.0


class HumanFactorsEvaluation(EvaluationProtocol):
    """Evaluation focused on human-robot interaction aspects."""
    
    def __init__(self, num_episodes: int = 100):
        super().__init__("HumanFactors", num_episodes, deterministic=True)
        self.impairment_levels = [0.0, 0.3, 0.6, 0.9]  # No impairment to severe
    
    def evaluate(self, agent: CPOAgent, environment: SharedControlEnvironment,
                 render: bool = False) -> Dict[str, EvaluationMetrics]:
        """Evaluate across different human impairment levels."""
        results = {}
        
        for impairment in self.impairment_levels:
            level_name = f"impairment_{impairment:.1f}"
            self.logger.info(f"Evaluating with impairment level {impairment}")
            
            # Configure human model impairment
            if hasattr(environment, 'human_model'):
                original_impairment = environment.human_model.impairment_severity
                environment.human_model.set_impairment_severity(impairment)
            
            # Run evaluation
            metrics = super().evaluate(agent, environment, render)
            results[level_name] = metrics
            
            # Restore original impairment
            if hasattr(environment, 'human_model'):
                environment.human_model.set_impairment_severity(original_impairment)
        
        return results


class EvaluationManager:
    """High-level manager for running multiple evaluation protocols."""
    
    def __init__(self, save_results: bool = True, results_dir: str = "evaluation_results"):
        self.save_results = save_results
        self.results_dir = results_dir
        self.logger = logging.getLogger(__name__)
        
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # Available evaluation protocols
        self.protocols = {
            'standard': StandardEvaluation(),
            'stochastic': StochasticEvaluation(),
            'safety': SafetyEvaluation(),
            'human_factors': HumanFactorsEvaluation()
        }
    
    def evaluate_model(self, agent: CPOAgent, environment: SharedControlEnvironment,
                      protocols: List[str] = None, render: bool = False) -> Dict[str, Any]:
        """Run comprehensive evaluation with selected protocols."""
        if protocols is None:
            protocols = ['standard', 'stochastic']
        
        self.logger.info(f"Running evaluation with protocols: {protocols}")
        
        results = {
            'agent_info': self._get_agent_info(agent),
            'environment_info': self._get_environment_info(environment),
            'protocols': {}
        }
        
        for protocol_name in protocols:
            if protocol_name not in self.protocols:
                self.logger.warning(f"Unknown protocol: {protocol_name}")
                continue
            
            try:
                protocol = self.protocols[protocol_name]
                protocol_results = protocol.evaluate(agent, environment, render)
                results['protocols'][protocol_name] = protocol_results
                
            except Exception as e:
                self.logger.error(f"Error in {protocol_name} evaluation: {e}")
                continue
        
        # Generate summary statistics
        results['summary'] = self._generate_summary(results['protocols'])
        
        # Save results if requested
        if self.save_results:
            self._save_results(results)
        
        return results
    
    def compare_models(self, agents: List[CPOAgent], agent_names: List[str],
                      environment: SharedControlEnvironment,
                      protocols: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models using the same evaluation protocols."""
        if len(agents) != len(agent_names):
            raise ValueError("Number of agents must match number of names")
        
        self.logger.info(f"Comparing {len(agents)} models")
        
        comparison_results = {
            'agents': agent_names,
            'environment_info': self._get_environment_info(environment),
            'individual_results': {},
            'comparison': {}
        }
        
        # Evaluate each agent
        for i, (agent, name) in enumerate(zip(agents, agent_names)):
            self.logger.info(f"Evaluating agent {i+1}/{len(agents)}: {name}")
            
            agent_results = self.evaluate_model(
                agent, environment, protocols, render=False
            )
            comparison_results['individual_results'][name] = agent_results
        
        # Generate comparison statistics
        comparison_results['comparison'] = self._generate_comparison(
            comparison_results['individual_results']
        )
        
        # Generate comparison plots
        if self.save_results:
            self._generate_comparison_plots(comparison_results)
        
        return comparison_results
    
    def _get_agent_info(self, agent: CPOAgent) -> Dict[str, Any]:
        """Extract agent information for reporting."""
        return {
            'type': type(agent).__name__,
            'parameters': getattr(agent, 'get_parameters', lambda: {})()
        }
    
    def _get_environment_info(self, environment: SharedControlEnvironment) -> Dict[str, Any]:
        """Extract environment information for reporting."""
        return {
            'type': type(environment).__name__,
            'observation_space': str(environment.observation_space),
            'action_space': str(environment.action_space)
        }
    
    def _generate_summary(self, protocol_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics across protocols."""
        summary = {
            'overall_performance': {},
            'safety_analysis': {},
            'human_factors': {}
        }
        
        # Collect metrics across protocols
        all_returns = []
        all_success_rates = []
        all_violation_rates = []
        
        for protocol_name, results in protocol_results.items():
            if isinstance(results, EvaluationMetrics):
                all_returns.append(results.mean_return)
                all_success_rates.append(results.success_rate)
                all_violation_rates.append(results.constraint_violation_rate)
            elif isinstance(results, dict):
                # Handle multi-level results (e.g., safety evaluation)
                for level_results in results.values():
                    all_returns.append(level_results.mean_return)
                    all_success_rates.append(level_results.success_rate)
                    all_violation_rates.append(level_results.constraint_violation_rate)
        
        # Overall performance
        if all_returns:
            summary['overall_performance'] = {
                'mean_return': float(np.mean(all_returns)),
                'std_return': float(np.std(all_returns)),
                'mean_success_rate': float(np.mean(all_success_rates)),
                'std_success_rate': float(np.std(all_success_rates))
            }
        
        # Safety analysis
        if all_violation_rates:
            summary['safety_analysis'] = {
                'mean_violation_rate': float(np.mean(all_violation_rates)),
                'max_violation_rate': float(np.max(all_violation_rates)),
                'safety_score': float(1.0 - np.mean(all_violation_rates))
            }
        
        return summary
    
    def _generate_comparison(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison statistics between agents."""
        comparison = {}
        
        # Extract metrics for comparison
        agent_metrics = {}
        for agent_name, results in individual_results.items():
            metrics = {}
            protocols = results['protocols']
            
            for protocol_name, protocol_results in protocols.items():
                if isinstance(protocol_results, EvaluationMetrics):
                    metrics[f"{protocol_name}_return"] = protocol_results.mean_return
                    metrics[f"{protocol_name}_success"] = protocol_results.success_rate
                    metrics[f"{protocol_name}_violations"] = protocol_results.constraint_violation_rate
            
            agent_metrics[agent_name] = metrics
        
        # Statistical comparisons
        if len(agent_metrics) >= 2:
            comparison['rankings'] = self._compute_rankings(agent_metrics)
            comparison['statistical_tests'] = self._perform_statistical_tests(individual_results)
        
        return comparison
    
    def _compute_rankings(self, agent_metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[str]]:
        """Compute rankings for different metrics."""
        rankings = {}
        
        if not agent_metrics:
            return rankings
        
        # Get all metric names
        all_metrics = set()
        for metrics in agent_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Rank agents for each metric
        for metric in all_metrics:
            agent_scores = [(name, metrics.get(metric, 0)) for name, metrics in agent_metrics.items()]
            
            # Sort by score (descending for positive metrics, ascending for violations)
            reverse = not ('violation' in metric.lower())
            agent_scores.sort(key=lambda x: x[1], reverse=reverse)
            
            rankings[metric] = [name for name, score in agent_scores]
        
        return rankings
    
    def _perform_statistical_tests(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical significance tests between agents."""
        # This would require access to raw episode data for proper statistical testing
        # For now, return placeholder
        return {"note": "Statistical tests require raw episode data"}
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to file."""
        import json
        import time
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            # Convert results to JSON-serializable format
            json_results = self._convert_to_json_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    def _convert_to_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, EvaluationMetrics):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_comparison_plots(self, comparison_results: Dict[str, Any]) -> None:
        """Generate comparison plots for multiple agents."""
        try:
            # Extract data for plotting
            agent_names = comparison_results['agents']
            individual_results = comparison_results['individual_results']
            
            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            metrics_to_plot = [
                ('mean_return', 'Mean Return'),
                ('success_rate', 'Success Rate'),
                ('constraint_violation_rate', 'Violation Rate'),
                ('collaboration_efficiency', 'Collaboration Efficiency')
            ]
            
            for i, (metric, title) in enumerate(metrics_to_plot):
                if i < len(axes):
                    self._plot_metric_comparison(
                        axes[i], agent_names, individual_results, metric, title
                    )
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.results_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Comparison plots saved to {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate comparison plots: {e}")
    
    def _plot_metric_comparison(self, ax, agent_names: List[str], 
                               individual_results: Dict[str, Any],
                               metric: str, title: str) -> None:
        """Plot comparison for a specific metric."""
        values = []
        labels = []
        
        for agent_name in agent_names:
            protocols = individual_results[agent_name]['protocols']
            
            # Try to get metric from standard protocol first
            if 'standard' in protocols:
                protocol_results = protocols['standard']
                if isinstance(protocol_results, EvaluationMetrics):
                    value = getattr(protocol_results, metric, 0)
                    values.append(value)
                    labels.append(agent_name)
        
        if values:
            bars = ax.bar(labels, values)
            ax.set_title(title)
            ax.set_ylabel(title)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                   transform=ax.transAxes)