# API Reference

## Safe RL Baselines

### safe_rl_human_robot.src.baselines.safe_rl

#### SACLagrangian

```python
class SACLagrangian(BaseSafeRLAlgorithm):
    """
    Soft Actor-Critic with Lagrangian constraint optimization.
    
    Implements SAC with automatic constraint handling through Lagrangian multipliers.
    Suitable for continuous control problems with safety constraints.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 constraint_threshold: float = 0.1,
                 lagrange_lr: float = 1e-3):
        """
        Initialize SAC-Lagrangian algorithm.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            hidden_dim: Hidden layer dimension for neural networks
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic networks
            constraint_threshold: Maximum allowed constraint violation
            lagrange_lr: Learning rate for Lagrange multiplier
        """
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Generate action for given state."""
        
    def learn(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Update algorithm parameters."""
```

#### TD3Constrained

```python
class TD3Constrained(BaseSafeRLAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient with safety constraints.
    
    Extends TD3 with constraint handling for safe reinforcement learning.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int, 
                 hidden_dim: int = 256,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 3e-4,
                 constraint_threshold: float = 0.1,
                 policy_delay: int = 2,
                 target_noise: float = 0.2):
        """Initialize TD3-Constrained algorithm."""
```

## Evaluation Framework

### safe_rl_human_robot.src.evaluation.evaluation_suite

#### EvaluationSuite

```python
class EvaluationSuite:
    """
    Main evaluation orchestrator for benchmarking Safe RL algorithms.
    
    Provides standardized evaluation across environments, metrics, and 
    statistical analysis methods.
    """
    
    def __init__(self):
        """Initialize evaluation suite with default configuration."""
    
    def evaluate_algorithm(self, 
                          algorithm: Any,
                          environment_config: Dict[str, Any],
                          num_episodes: int = 100,
                          num_seeds: int = 5) -> Dict[str, Any]:
        """
        Evaluate algorithm performance in specified environment.
        
        Args:
            algorithm: Safe RL algorithm instance
            environment_config: Environment configuration dictionary
            num_episodes: Number of evaluation episodes
            num_seeds: Number of random seeds for evaluation
            
        Returns:
            Dictionary containing evaluation metrics and statistics
        """
    
    def compare_algorithms(self, 
                          algorithms: List[Any],
                          environment_configs: List[Dict[str, Any]],
                          num_episodes: int = 100,
                          num_seeds: int = 10) -> Dict[str, Any]:
        """
        Compare multiple algorithms across environments.
        
        Returns comprehensive comparison with statistical analysis.
        """
```

### safe_rl_human_robot.src.evaluation.metrics

#### MetricsCalculator

```python
class MetricsCalculator:
    """
    Comprehensive metrics calculation for Safe RL evaluation.
    
    Computes performance, safety, human-centric, and efficiency metrics.
    """
    
    def calculate_performance_metrics(self, 
                                    trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance-related metrics.
        
        Args:
            trajectories: List of episode trajectories
            
        Returns:
            Dictionary of performance metrics:
            - sample_efficiency: Learning speed measure
            - asymptotic_performance: Final performance level
            - success_rate: Task completion rate
            - convergence_speed: Episodes to convergence
        """
    
    def calculate_safety_metrics(self, 
                                trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate safety-related metrics.
        
        Returns:
            Dictionary of safety metrics:
            - safety_violation_rate: Frequency of violations
            - constraint_satisfaction_rate: Safe episode percentage
            - risk_score: Violations per timestep
            - recovery_capability: Return to safe states
        """
    
    def calculate_human_metrics(self, 
                               trajectories: List[Dict[str, Any]],
                               human_feedback: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate human-centric metrics.
        
        Returns:
            Dictionary of human metrics:
            - human_satisfaction: Subjective preference scores
            - predictability_score: Behavioral consistency
            - trust_score: Reliability assessment
            - workload_score: Cognitive/physical demands
        """
    
    def register_custom_metric(self, name: str, metric_function: Callable):
        """Register custom metric calculation function."""
```

### safe_rl_human_robot.src.evaluation.statistics

#### StatisticalAnalysis

```python
class StatisticalAnalysis:
    """
    Statistical analysis tools for Safe RL benchmarking.
    
    Provides hypothesis testing, effect size calculation, and 
    multiple comparison corrections.
    """
    
    def mann_whitney_test(self, 
                         group1: List[float], 
                         group2: List[float]) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U test for two groups.
        
        Args:
            group1: First group of measurements
            group2: Second group of measurements
            
        Returns:
            Dictionary containing:
            - statistic: U statistic
            - p_value: Two-tailed p-value
            - effect_size: Rank-biserial correlation
            - interpretation: Statistical interpretation
        """
    
    def friedman_test(self, 
                     groups: List[List[float]], 
                     group_names: List[str]) -> Dict[str, Any]:
        """
        Perform Friedman test for multiple related groups.
        
        Returns:
            Dictionary with test statistics and post-hoc analysis
        """
    
    def calculate_effect_size(self, 
                            group1: List[float], 
                            group2: List[float],
                            method: str = 'cohens_d') -> float:
        """
        Calculate effect size between two groups.
        
        Args:
            group1: First group measurements
            group2: Second group measurements
            method: Effect size method ('cohens_d', 'hedges_g', 'rank_biserial')
            
        Returns:
            Effect size value
        """
    
    def multiple_comparison_correction(self, 
                                     p_values: List[float],
                                     method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple comparison correction.
        
        Args:
            p_values: List of p-values from multiple tests
            method: Correction method ('bonferroni', 'fdr_bh')
            
        Returns:
            Dictionary with corrected p-values and significance flags
        """
    
    def bootstrap_confidence_interval(self, 
                                    data: List[float],
                                    statistic_function: Callable = np.mean,
                                    confidence_level: float = 0.95,
                                    num_bootstrap: int = 10000) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
```

## Visualization Tools

### safe_rl_human_robot.src.evaluation.visualization

#### VisualizationTools

```python
class VisualizationTools:
    """
    Publication-quality visualization tools for Safe RL results.
    
    Generates standardized plots for performance comparison, safety analysis,
    and statistical reporting.
    """
    
    def plot_algorithm_comparison(self, 
                                 results: Dict[str, Any],
                                 metrics: List[str],
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create algorithm performance comparison plot.
        
        Args:
            results: Evaluation results dictionary
            metrics: List of metrics to plot
            save_path: Optional file path to save figure
            
        Returns:
            Matplotlib figure object
        """
    
    def plot_safety_analysis(self, 
                           results: Dict[str, Any],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Create safety analysis visualization.
        
        Shows violation rates, constraint satisfaction, and risk scores.
        """
    
    def plot_statistical_heatmap(self, 
                                statistical_results: Dict[str, Any],
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create statistical significance heatmap.
        
        Visualizes p-values and effect sizes between algorithm pairs.
        """
    
    def plot_human_factors_radar(self, 
                               results: Dict[str, Any],
                               algorithms: List[str],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create radar chart for human-centric metrics.
        
        Shows satisfaction, trust, predictability, and workload scores.
        """
```

## Ablation Studies

### safe_rl_human_robot.src.analysis.ablation_studies

#### AblationStudies

```python
class AblationStudies:
    """
    Comprehensive ablation study framework for Safe RL algorithms.
    
    Systematically evaluates component importance and interactions.
    """
    
    def run_ablation_study(self,
                          study_name: str,
                          base_algorithm: str,
                          components: List[str],
                          environments: List[str],
                          num_runs: int = 5) -> Dict[str, Any]:
        """
        Run comprehensive ablation study.
        
        Args:
            study_name: Name identifier for the study
            base_algorithm: Base algorithm class name
            components: List of component names to ablate
            environments: List of environment names for testing
            num_runs: Number of runs per configuration
            
        Returns:
            Dictionary containing:
            - component_importance: Individual component contributions
            - interaction_effects: Component interaction analysis
            - statistical_significance: Statistical test results
            - recommendations: Algorithm design recommendations
        """
    
    def analyze_component_importance(self, 
                                   results: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze individual component importance.
        
        Uses ANOVA and effect size calculations to rank components.
        """
    
    def detect_interaction_effects(self, 
                                 results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect interaction effects between components.
        
        Returns significant two-way and three-way interactions.
        """
```

## Cross-Domain Evaluation

### safe_rl_human_robot.src.evaluation.cross_domain

#### CrossDomainEvaluator

```python
class CrossDomainEvaluator:
    """
    Cross-domain evaluation and transfer learning analysis.
    
    Evaluates algorithm generalization across different robot platforms
    and human interaction scenarios.
    """
    
    def evaluate_transfer_performance(self,
                                    source_domain: str,
                                    target_domains: List[str],
                                    algorithms: List[str],
                                    num_adaptation_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate transfer learning performance.
        
        Args:
            source_domain: Source domain for pre-training
            target_domains: List of target domains for transfer
            algorithms: Algorithms to evaluate
            num_adaptation_episodes: Episodes for domain adaptation
            
        Returns:
            Dictionary containing:
            - transfer_success_rates: Adaptation success per domain pair
            - performance_degradation: Performance loss during transfer
            - adaptation_speed: Episodes required for successful transfer
            - failure_modes: Analysis of transfer failures
        """
    
    def characterize_domain_similarity(self, 
                                     domain1: str, 
                                     domain2: str) -> Dict[str, float]:
        """
        Characterize similarity between domains.
        
        Returns:
            Dictionary of similarity metrics:
            - morphological_similarity: Kinematic/dynamic similarity
            - task_complexity_difference: Complexity gap
            - interaction_similarity: Human interaction patterns
        """
    
    def analyze_generalization_capability(self, 
                                        results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze algorithm generalization capabilities.
        
        Identifies algorithms with best cross-domain performance.
        """
```

## Reproducibility Tools

### safe_rl_human_robot.src.utils.reproducibility

#### ReproducibilityManager

```python
class ReproducibilityManager:
    """
    Manages reproducibility settings for experiments.
    
    Ensures deterministic experiments with complete environment logging.
    """
    
    def __init__(self, config: ReproducibilityConfig):
        """
        Initialize reproducibility manager.
        
        Args:
            config: Reproducibility configuration object
        """
    
    def create_experiment_hash(self, config_dict: Dict[str, Any]) -> str:
        """
        Create unique hash for experiment configuration.
        
        Combines system info, algorithm config, and reproducibility settings
        to create a unique experiment identifier.
        """
    
    def verify_reproducibility(self, 
                             reference_results: Dict[str, Any],
                             current_results: Dict[str, Any],
                             tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Verify reproducibility by comparing results.
        
        Args:
            reference_results: Previously obtained results
            current_results: Current experiment results
            tolerance: Numerical tolerance for comparisons
            
        Returns:
            Dictionary containing:
            - reproducible: Boolean indicating reproducibility
            - differences: Detailed differences if any
            - tolerance: Used tolerance value
        """
```

#### PerformanceProfiler

```python
class PerformanceProfiler:
    """
    Comprehensive performance profiling for Safe RL algorithms.
    
    Monitors CPU, memory, GPU usage, and algorithm-specific metrics.
    """
    
    def profile_algorithm(self, algorithm_name: str) -> ContextManager:
        """
        Context manager for profiling algorithm performance.
        
        Usage:
            with profiler.profile_algorithm("SACLagrangian") as profile:
                # Run algorithm
                pass
        """
    
    def benchmark_inference_speed(self, 
                                algorithm: Any,
                                test_inputs: List[np.ndarray],
                                num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark algorithm inference speed.
        
        Returns:
            Dictionary of timing statistics:
            - mean_time_ms: Average inference time
            - std_time_ms: Standard deviation
            - throughput_hz: Inference throughput
        """
    
    def profile_memory_usage(self, 
                           algorithm: Any,
                           training_function: Callable,
                           duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Profile memory usage during training.
        
        Returns detailed memory usage statistics and timeline.
        """
```

## Publication Experiments

### safe_rl_human_robot.src.experiments.publication_experiments

#### PublicationExperimentRunner

```python
class PublicationExperimentRunner:
    """
    Orchestrates comprehensive experiments for publication-quality results.
    
    Manages the complete experimental pipeline from algorithm evaluation
    to publication material generation.
    """
    
    def __init__(self, config: ExperimentConfig, output_dir: Path):
        """
        Initialize publication experiment runner.
        
        Args:
            config: Experiment configuration
            output_dir: Directory for saving results
        """
    
    def run_comprehensive_experiments(self) -> Dict[str, Any]:
        """
        Run complete experimental suite for publication.
        
        Executes:
        1. Main benchmarking experiments
        2. Statistical analysis
        3. Ablation studies
        4. Cross-domain evaluation
        5. Publication material generation
        
        Returns:
            Dictionary containing all results and paths to generated materials
        """
    
    def generate_publication_materials(self) -> Dict[str, Any]:
        """
        Generate publication-quality figures and tables.
        
        Returns:
            Dictionary of generated materials:
            - figures: List of figure file paths
            - tables: List of table file paths
            - latex_report: LaTeX report template path
        """
```

#### ExperimentConfig

```python
@dataclass
class ExperimentConfig:
    """Configuration for publication experiments."""
    
    experiment_name: str = "safe_rl_benchmark"
    algorithms: List[str] = field(default_factory=lambda: ["SACLagrangian", "TD3Constrained"])
    environments: List[str] = field(default_factory=lambda: ["manipulator_7dof", "mobile_base"])
    human_behaviors: List[str] = field(default_factory=lambda: ["cooperative", "adversarial"])
    num_seeds: int = 10
    training_episodes: int = 1000
    evaluation_episodes: int = 100
    significance_level: float = 0.05
    effect_size_threshold: float = 0.5
```

## Usage Examples

### Basic Algorithm Evaluation

```python
from safe_rl_human_robot.src.baselines.safe_rl import SACLagrangian
from safe_rl_human_robot.src.evaluation.evaluation_suite import EvaluationSuite

# Initialize algorithm
algorithm = SACLagrangian(state_dim=10, action_dim=4)

# Configure evaluation
env_config = {
    'environment_type': 'manipulator_7dof',
    'human_behavior': 'cooperative',
    'safety_constraints': True
}

# Run evaluation
evaluator = EvaluationSuite()
results = evaluator.evaluate_algorithm(algorithm, env_config)

print(f"Performance: {results['performance_metrics']}")
print(f"Safety: {results['safety_metrics']}")
```

### Statistical Comparison

```python
from safe_rl_human_robot.src.evaluation.statistics import StatisticalAnalysis

stats = StatisticalAnalysis()

# Compare two algorithms
alg1_results = [0.8, 0.85, 0.82, 0.88, 0.79]
alg2_results = [0.75, 0.78, 0.76, 0.80, 0.74]

comparison = stats.mann_whitney_test(alg1_results, alg2_results)
print(f"p-value: {comparison['p_value']}")
print(f"Effect size: {comparison['effect_size']}")
```

### Publication Experiment

```python
from safe_rl_human_robot.src.experiments.publication_experiments import (
    PublicationExperimentRunner, ExperimentConfig
)

config = ExperimentConfig(
    experiment_name="comprehensive_benchmark",
    algorithms=["SACLagrangian", "TD3Constrained", "MPCController"],
    environments=["manipulator_7dof", "mobile_base", "collaborative_assembly"],
    num_seeds=10
)

runner = PublicationExperimentRunner(config, Path("experiment_outputs"))
results = runner.run_comprehensive_experiments()

print(f"Report saved to: {results['report_path']}")
```

## Error Handling

All API functions include comprehensive error handling and logging:

- **Input Validation**: Parameter type and range checking
- **Graceful Degradation**: Fallback behaviors for missing components
- **Detailed Logging**: Comprehensive error and warning messages
- **Exception Handling**: Proper exception propagation with context

## Performance Considerations

- **Memory Management**: Automatic cleanup of large data structures
- **Parallel Processing**: Multi-core utilization where applicable
- **GPU Support**: Automatic CUDA detection and utilization
- **Scalability**: Efficient handling of large-scale experiments