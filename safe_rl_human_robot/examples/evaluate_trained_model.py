#!/usr/bin/env python3
"""
Model evaluation script for trained CPO agents.

This script demonstrates how to evaluate trained CPO models using
comprehensive evaluation protocols and generate detailed analysis reports.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import torch
import numpy as np
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation components
from src.training.evaluation import (
    EvaluationManager, StandardEvaluation, StochasticEvaluation,
    SafetyEvaluation, HumanFactorsEvaluation
)
from src.training.config import TrainingConfig

# Import environments and algorithms
from src.environments.exoskeleton_env import ExoskeletonEnvironment
from src.environments.wheelchair_env import WheelchairEnvironment
from src.algorithms.cpo import CPOAgent


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def create_environment(env_type: str, **kwargs) -> object:
    """Create evaluation environment."""
    if env_type == "exoskeleton":
        return ExoskeletonEnvironment(**kwargs)
    elif env_type == "wheelchair":
        return WheelchairEnvironment(**kwargs)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def load_trained_model(model_path: str, config_path: str = None) -> tuple:
    """Load trained CPO model and configuration."""
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path) as f:
            config_dict = json.load(f)
        config = TrainingConfig.from_dict(config_dict)
    elif 'config' in checkpoint:
        logger.info("Using configuration from checkpoint")
        config = TrainingConfig.from_dict(checkpoint['config'])
    else:
        logger.warning("No configuration found, using defaults")
        config = TrainingConfig()
    
    return checkpoint, config


def create_agent_from_checkpoint(checkpoint: dict, config: TrainingConfig, environment) -> CPOAgent:
    """Create CPO agent from checkpoint."""
    logger = logging.getLogger(__name__)
    
    # Create agent
    agent = CPOAgent(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        config=config
    )
    
    # Load model states
    if 'policy_net_state_dict' in checkpoint:
        agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        logger.info("Loaded policy network state")
    
    if 'value_net_state_dict' in checkpoint:
        agent.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        logger.info("Loaded value network state")
    
    # Set to evaluation mode
    agent.policy_net.eval()
    agent.value_net.eval()
    
    logger.info("Agent created and loaded successfully")
    return agent


def run_evaluation_protocols(agent: CPOAgent, environment, args: argparse.Namespace) -> dict:
    """Run all requested evaluation protocols."""
    logger = logging.getLogger(__name__)
    
    # Create evaluation manager
    eval_manager = EvaluationManager(
        save_results=True,
        results_dir=args.results_dir
    )
    
    # Determine which protocols to run
    if args.protocols == "all":
        protocols = ["standard", "stochastic", "safety", "human_factors"]
    else:
        protocols = args.protocols.split(",")
    
    logger.info(f"Running evaluation protocols: {protocols}")
    
    # Run evaluation
    results = eval_manager.evaluate_model(
        agent=agent,
        environment=environment,
        protocols=protocols,
        render=args.render
    )
    
    return results


def run_model_comparison(model_paths: list, model_names: list, environment, args: argparse.Namespace) -> dict:
    """Run comparison between multiple models."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Comparing {len(model_paths)} models")
    
    # Load all models
    agents = []
    for i, model_path in enumerate(model_paths):
        config_path = None
        if args.config_paths and i < len(args.config_paths):
            config_path = args.config_paths[i]
        
        checkpoint, config = load_trained_model(model_path, config_path)
        agent = create_agent_from_checkpoint(checkpoint, config, environment)
        agents.append(agent)
    
    # Create evaluation manager
    eval_manager = EvaluationManager(
        save_results=True,
        results_dir=args.results_dir
    )
    
    # Determine protocols
    if args.protocols == "all":
        protocols = ["standard", "stochastic"]  # Use fewer protocols for comparison
    else:
        protocols = args.protocols.split(",")
    
    # Run comparison
    comparison_results = eval_manager.compare_models(
        agents=agents,
        agent_names=model_names,
        environment=environment,
        protocols=protocols
    )
    
    return comparison_results


def generate_detailed_report(results: dict, args: argparse.Namespace) -> None:
    """Generate detailed evaluation report."""
    logger = logging.getLogger(__name__)
    
    report_path = os.path.join(args.results_dir, "evaluation_report.md")
    
    logger.info(f"Generating detailed report: {report_path}")
    
    with open(report_path, 'w') as f:
        f.write("# CPO Model Evaluation Report\n\n")
        
        # Environment information
        if 'environment_info' in results:
            f.write("## Environment Information\n\n")
            env_info = results['environment_info']
            f.write(f"- **Environment Type**: {env_info.get('type', 'Unknown')}\n")
            f.write(f"- **Observation Space**: {env_info.get('observation_space', 'Unknown')}\n")
            f.write(f"- **Action Space**: {env_info.get('action_space', 'Unknown')}\n\n")
        
        # Agent information
        if 'agent_info' in results:
            f.write("## Agent Information\n\n")
            agent_info = results['agent_info']
            f.write(f"- **Agent Type**: {agent_info.get('type', 'Unknown')}\n")
            if 'parameters' in agent_info:
                f.write("- **Parameters**:\n")
                for key, value in agent_info['parameters'].items():
                    f.write(f"  - {key}: {value}\n")
            f.write("\n")
        
        # Protocol results
        if 'protocols' in results:
            f.write("## Evaluation Results\n\n")
            
            for protocol_name, protocol_results in results['protocols'].items():
                f.write(f"### {protocol_name.title()} Evaluation\n\n")
                
                if hasattr(protocol_results, 'to_dict'):
                    metrics = protocol_results.to_dict()
                elif isinstance(protocol_results, dict):
                    # Handle multi-level results (e.g., safety evaluation)
                    f.write("Results by challenge level:\n\n")
                    for level, level_results in protocol_results.items():
                        f.write(f"#### {level.title()} Level\n\n")
                        if hasattr(level_results, 'to_dict'):
                            metrics = level_results.to_dict()
                            _write_metrics_table(f, metrics)
                        f.write("\n")
                    continue
                else:
                    continue
                
                _write_metrics_table(f, metrics)
                f.write("\n")
        
        # Summary
        if 'summary' in results:
            f.write("## Summary\n\n")
            summary = results['summary']
            
            if 'overall_performance' in summary:
                f.write("### Overall Performance\n\n")
                perf = summary['overall_performance']
                f.write(f"- **Mean Return**: {perf.get('mean_return', 0):.3f} ± {perf.get('std_return', 0):.3f}\n")
                f.write(f"- **Mean Success Rate**: {perf.get('mean_success_rate', 0):.3f} ± {perf.get('std_success_rate', 0):.3f}\n\n")
            
            if 'safety_analysis' in summary:
                f.write("### Safety Analysis\n\n")
                safety = summary['safety_analysis']
                f.write(f"- **Mean Violation Rate**: {safety.get('mean_violation_rate', 0):.3f}\n")
                f.write(f"- **Max Violation Rate**: {safety.get('max_violation_rate', 0):.3f}\n")
                f.write(f"- **Safety Score**: {safety.get('safety_score', 0):.3f}\n\n")
        
        # Comparison results (if applicable)
        if 'comparison' in results:
            f.write("## Model Comparison\n\n")
            comparison = results['comparison']
            
            if 'rankings' in comparison:
                f.write("### Rankings by Metric\n\n")
                for metric, ranking in comparison['rankings'].items():
                    f.write(f"**{metric}**: {' > '.join(ranking)}\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated using evaluation arguments: {vars(args)}*\n")
    
    logger.info("Detailed report generated successfully")


def _write_metrics_table(f, metrics: dict) -> None:
    """Write metrics as a markdown table."""
    f.write("| Metric | Value |\n")
    f.write("|--------|-------|\n")
    
    # Group metrics by category
    categories = {
        'Performance': ['mean_return', 'success_rate', 'completion_rate', 'mean_episode_length'],
        'Safety': ['constraint_violation_rate', 'mean_constraint_cost', 'safety_margin'],
        'Human Factors': ['mean_human_effort', 'collaboration_efficiency', 'user_comfort_score'],
        'Statistics': ['sample_size', 'confidence_interval_95']
    }
    
    for category, metric_names in categories.items():
        category_written = False
        for metric_name in metric_names:
            if metric_name in metrics:
                if not category_written:
                    f.write(f"| **{category}** |  |\n")
                    category_written = True
                
                value = metrics[metric_name]
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    f.write(f"| {metric_name} | [{value[0]:.3f}, {value[1]:.3f}] |\n")
                elif isinstance(value, float):
                    f.write(f"| {metric_name} | {value:.3f} |\n")
                elif isinstance(value, int):
                    f.write(f"| {metric_name} | {value} |\n")
                else:
                    f.write(f"| {metric_name} | {value} |\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained CPO models")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--config-path", type=str, default=None,
                       help="Path to training configuration file")
    
    # Multiple model comparison
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare multiple models")
    parser.add_argument("--model-paths", type=str, nargs="+",
                       help="Paths to multiple model checkpoints for comparison")
    parser.add_argument("--model-names", type=str, nargs="+",
                       help="Names for models in comparison")
    parser.add_argument("--config-paths", type=str, nargs="*",
                       help="Configuration paths for multiple models")
    
    # Environment arguments
    parser.add_argument("--env-type", type=str, default="exoskeleton",
                       choices=["exoskeleton", "wheelchair"],
                       help="Type of environment for evaluation")
    parser.add_argument("--env-difficulty", type=str, default="moderate",
                       choices=["easy", "moderate", "hard"],
                       help="Environment difficulty level")
    parser.add_argument("--human-impairment", type=float, default=0.3,
                       help="Human impairment level (0.0-1.0)")
    
    # Evaluation arguments
    parser.add_argument("--protocols", type=str, default="standard,safety",
                       help="Comma-separated list of evaluation protocols or 'all'")
    parser.add_argument("--num-episodes", type=int, default=100,
                       help="Number of episodes per evaluation")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation episodes")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic policy evaluation")
    
    # Output arguments
    parser.add_argument("--results-dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate detailed markdown report")
    parser.add_argument("--save-videos", action="store_true",
                       help="Save evaluation videos")
    
    # Miscellaneous arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for evaluation")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device for model evaluation")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    logger.info("Starting model evaluation")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create environment
        logger.info(f"Creating {args.env_type} environment")
        if args.env_type == "exoskeleton":
            env_kwargs = {
                'n_dofs': 7,
                'task_type': 'reach',
                'difficulty': args.env_difficulty,
                'human_impairment': args.human_impairment
            }
        elif args.env_type == "wheelchair":
            difficulty_map = {'easy': 3, 'moderate': 7, 'hard': 12}
            env_kwargs = {
                'world_size': (10, 10),
                'num_obstacles': difficulty_map[args.env_difficulty],
                'difficulty': args.env_difficulty
            }
        else:
            env_kwargs = {}
        
        environment = create_environment(args.env_type, **env_kwargs)
        
        if args.compare_models:
            # Model comparison mode
            if not args.model_paths or len(args.model_paths) < 2:
                raise ValueError("At least 2 model paths required for comparison")
            
            model_names = args.model_names or [f"Model_{i+1}" for i in range(len(args.model_paths))]
            
            if len(model_names) != len(args.model_paths):
                raise ValueError("Number of model names must match number of model paths")
            
            logger.info(f"Comparing models: {model_names}")
            
            results = run_model_comparison(
                model_paths=args.model_paths,
                model_names=model_names,
                environment=environment,
                args=args
            )
            
        else:
            # Single model evaluation
            logger.info(f"Evaluating model: {args.model_path}")
            
            # Load model
            checkpoint, config = load_trained_model(args.model_path, args.config_path)
            agent = create_agent_from_checkpoint(checkpoint, config, environment)
            
            # Run evaluation
            results = run_evaluation_protocols(agent, environment, args)
        
        # Generate report if requested
        if args.generate_report:
            generate_detailed_report(results, args)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 50)
        
        if 'summary' in results:
            summary = results['summary']
            if 'overall_performance' in summary:
                perf = summary['overall_performance']
                logger.info(f"Mean Return: {perf.get('mean_return', 0):.3f}")
                logger.info(f"Success Rate: {perf.get('mean_success_rate', 0):.3f}")
            
            if 'safety_analysis' in summary:
                safety = summary['safety_analysis']
                logger.info(f"Safety Score: {safety.get('safety_score', 0):.3f}")
                logger.info(f"Violation Rate: {safety.get('mean_violation_rate', 0):.3f}")
        
        logger.info(f"Detailed results saved to: {args.results_dir}")
        logger.info("Evaluation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    logger.info("Evaluation script completed")


if __name__ == "__main__":
    main()