# Safe Reinforcement Learning for Human-Robot Shared Control

## Comprehensive State-of-the-art Benchmarking Framework

This repository implements a comprehensive benchmarking framework for Safe Reinforcement Learning (Safe RL) algorithms in human-robot shared control scenarios. The framework provides rigorous evaluation of state-of-the-art methods with statistical analysis, cross-domain evaluation, and publication-quality results.

## 🎯 Overview

The framework evaluates both advanced Safe RL algorithms and classical control baselines across multiple robot platforms and human interaction scenarios. It includes comprehensive statistical analysis, ablation studies, cross-domain evaluation, and reproducibility tools suitable for top-tier conference submissions (ICRA, RSS, NeurIPS, ICML).

## 🏗️ Architecture

```
safe_rl_human_robot/
├── src/
│   ├── baselines/
│   │   ├── safe_rl.py              # Advanced Safe RL algorithms
│   │   └── classical_control.py    # Classical control baselines
│   ├── evaluation/
│   │   ├── evaluation_suite.py     # Main evaluation orchestrator
│   │   ├── environments.py         # Standardized environment suite
│   │   ├── metrics.py             # Comprehensive metrics calculation
│   │   ├── statistics.py          # Statistical analysis tools
│   │   ├── visualization.py       # Publication-quality plots
│   │   └── cross_domain.py        # Cross-domain evaluation
│   ├── analysis/
│   │   └── ablation_studies.py    # Ablation study framework
│   ├── utils/
│   │   └── reproducibility.py     # Reproducibility and profiling
│   └── experiments/
│       └── publication_experiments.py # Publication experiment runner
├── docs/                          # Comprehensive documentation
├── examples/                      # Usage examples and tutorials
└── tests/                        # Unit and integration tests
```

## 🤖 Implemented Algorithms

### Safe Reinforcement Learning Methods

1. **SAC-Lagrangian**: Soft Actor-Critic with Lagrangian constraint optimization
2. **TD3-Constrained**: Twin Delayed Deep Deterministic Policy Gradient with safety constraints
3. **TRPO-Constrained**: Trust Region Policy Optimization with constraint handling
4. **PPO-Lagrangian**: Proximal Policy Optimization with Lagrangian multipliers
5. **Safe-DDPG**: Deep Deterministic Policy Gradient with safety guarantees
6. **RCPO**: Reward Constrained Policy Optimization
7. **CPO Variants**: Constrained Policy Optimization with different solvers

### Classical Control Baselines

1. **Model Predictive Control (MPC)**: Optimization-based control with safety constraints
2. **Linear Quadratic Regulator (LQR)**: Optimal linear control with adaptive gains
3. **PID Controller**: Proportional-Integral-Derivative control with auto-tuning
4. **Impedance Control**: Physical human-robot interaction control
5. **Admittance Control**: Compliant robot behavior control

## 🌍 Evaluation Environments

- **7-DOF Manipulator**: Industrial robot arm scenarios
- **6-DOF Manipulator**: Collaborative manipulation tasks
- **Mobile Base**: Navigation and path-following
- **Humanoid Robot**: Full-body control scenarios
- **Dual-Arm System**: Bimanual manipulation
- **Collaborative Assembly**: Human-robot teamwork
- **Force Interaction**: Physical collaboration tasks

### Human Behavior Models

- **Cooperative**: Helpful and predictable human partner
- **Adversarial**: Challenging and unpredictable behavior
- **Inconsistent**: Variable cooperation levels
- **Novice**: Learning human partner
- **Expert**: Skilled human collaborator

## 📊 Evaluation Metrics

### Performance Metrics
- Sample efficiency
- Asymptotic performance  
- Task success rate
- Convergence speed
- Computational efficiency

### Safety Metrics
- Safety violation rate
- Constraint satisfaction
- Risk assessment scores
- Recovery capability
- Robustness analysis

### Human-Centric Metrics
- Human satisfaction scores
- Predictability measures
- Trust and acceptance
- Workload assessment
- Collaboration quality

## 🔬 Statistical Analysis

The framework includes comprehensive statistical analysis:

- **Hypothesis Testing**: Mann-Whitney U, Welch's t-test, Friedman test
- **Effect Sizes**: Cohen's d, Hedges' g, rank-biserial correlation
- **Multiple Comparisons**: Bonferroni correction, FDR control
- **Bootstrap Methods**: Confidence intervals and significance testing
- **Power Analysis**: Sample size determination and validation

## 🧪 Experimental Features

### Ablation Studies
- Component importance analysis
- Factorial design experiments
- Interaction effect detection
- Automated report generation

### Cross-Domain Evaluation
- Domain transfer analysis
- Generalization assessment
- Adaptation capability testing
- Failure mode analysis

### Reproducibility Tools
- Deterministic experiment settings
- System information logging
- Performance profiling
- Experiment tracking and management

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control.git
cd Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control
pip install -r requirements.txt
```

### Basic Usage

```python
from safe_rl_human_robot.src.experiments.publication_experiments import (
    PublicationExperimentRunner, ExperimentConfig
)
from pathlib import Path

# Configure experiment
config = ExperimentConfig(
    experiment_name="my_safe_rl_benchmark",
    algorithms=["SACLagrangian", "TD3Constrained", "MPCController"],
    environments=["manipulator_7dof", "mobile_base"],
    human_behaviors=["cooperative", "adversarial"],
    num_seeds=5,
    training_episodes=1000,
    evaluation_episodes=100
)

# Run comprehensive experiments
runner = PublicationExperimentRunner(config, Path("outputs"))
results = runner.run_comprehensive_experiments()

print(f"Results available at: {results['report_path']}")
```

### Individual Component Usage

```python
# Use specific algorithms
from safe_rl_human_robot.src.baselines.safe_rl import SACLagrangian
from safe_rl_human_robot.src.baselines.classical_control import MPCController

# Safe RL algorithm
safe_rl_agent = SACLagrangian(
    state_dim=10, action_dim=4,
    constraint_threshold=0.1
)

# Classical control baseline
mpc_controller = MPCController(
    prediction_horizon=10,
    state_dim=10, action_dim=4
)

# Evaluation framework
from safe_rl_human_robot.src.evaluation.evaluation_suite import EvaluationSuite

evaluator = EvaluationSuite()
results = evaluator.evaluate_algorithm(safe_rl_agent, "manipulator_7dof")
```

## 📈 Generating Publication Materials

The framework automatically generates publication-quality materials:

```python
# Run publication experiments
results = runner.run_comprehensive_experiments()

# Generated materials include:
# - Performance comparison figures (PDF, 300 DPI)
# - Statistical significance heatmaps
# - Safety analysis visualizations
# - Ablation study results
# - Cross-domain transfer plots
# - Comprehensive data tables (CSV)
# - LaTeX report template
# - Supplementary materials
```

Generated files:
- `publication_materials/algorithm_performance_comparison.pdf`
- `publication_materials/safety_analysis.pdf`
- `publication_materials/statistical_significance.pdf`
- `publication_materials/main_results_table.csv`
- `final_report/publication_report.tex`

## 🔧 Configuration Options

### Algorithm Configuration

```python
# Safe RL algorithms with hyperparameters
algorithms = {
    "SACLagrangian": {
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "constraint_threshold": 0.1,
        "lagrange_lr": 1e-3
    },
    "TD3Constrained": {
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "constraint_threshold": 0.1,
        "policy_delay": 2
    }
}
```

### Environment Configuration

```python
# Environment settings
environments = {
    "manipulator_7dof": {
        "max_episode_steps": 1000,
        "safety_constraints": True,
        "human_interaction": True
    },
    "mobile_base": {
        "max_episode_steps": 2000,
        "navigation_constraints": True,
        "obstacle_avoidance": True
    }
}
```

### Statistical Analysis Settings

```python
# Statistical configuration
statistical_config = {
    "significance_level": 0.05,
    "effect_size_threshold": 0.5,
    "bootstrap_samples": 10000,
    "multiple_comparison_method": "bonferroni"
}
```

## 📊 Results Interpretation

### Performance Rankings
Results include algorithm rankings across different metrics with statistical significance testing.

### Effect Sizes
- **Small**: |d| < 0.2 (minimal practical difference)
- **Medium**: 0.2 ≤ |d| < 0.5 (moderate difference)
- **Large**: 0.5 ≤ |d| < 0.8 (substantial difference)
- **Very Large**: |d| ≥ 0.8 (major difference)

### Safety Analysis
- **Violation Rate**: Lower is better
- **Constraint Satisfaction**: Higher is better (0-1 scale)
- **Risk Score**: Violations per timestep (lower is better)

### Human Factors
- **Satisfaction**: Human preference scores (0-1 scale)
- **Trust**: Based on safety and predictability
- **Workload**: Cognitive/physical demands (lower is better)

## 🧪 Advanced Features

### Custom Metrics

```python
from safe_rl_human_robot.src.evaluation.metrics import MetricsCalculator

# Add custom metrics
def custom_metric(trajectory_data):
    # Your custom calculation
    return metric_value

metrics_calc = MetricsCalculator()
metrics_calc.register_custom_metric("my_metric", custom_metric)
```

### Custom Environments

```python
from safe_rl_human_robot.src.evaluation.environments import BaseEnvironment

class MyCustomEnvironment(BaseEnvironment):
    def __init__(self, config):
        super().__init__(config)
        # Your environment implementation
    
    def step(self, action):
        # Environment dynamics
        return next_state, reward, done, info
```

### Reproducibility Management

```python
from safe_rl_human_robot.src.utils.reproducibility import (
    ReproducibilityManager, ReproducibilityConfig, reproducible_experiment
)

# Configure reproducibility
repro_config = ReproducibilityConfig(
    global_seed=42,
    torch_deterministic=True,
    log_level="INFO"
)

# Use as decorator
@reproducible_experiment(repro_config)
def my_experiment():
    # Your experiment code
    return results
```

## 📚 Documentation

Comprehensive documentation is available:

- **API Reference**: Complete function and class documentation
- **User Guide**: Step-by-step usage instructions  
- **Developer Guide**: Contributing and extending the framework
- **Algorithm Details**: Mathematical formulations and implementation notes
- **Benchmarking Guide**: Best practices for reproducible evaluation
- **Publication Guide**: How to use results in academic papers

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_algorithms.py  # Algorithm tests
python -m pytest tests/test_evaluation.py  # Evaluation tests  
python -m pytest tests/test_statistics.py # Statistical tests
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guide:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@software{safe_rl_human_robot_benchmark,
  title={Safe Reinforcement Learning for Human-Robot Shared Control: A Comprehensive Benchmarking Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gym for environment interfaces
- Stable Baselines3 for RL algorithm inspiration  
- PyTorch for deep learning framework
- SciPy for statistical analysis tools
- Matplotlib/Seaborn for visualization

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/repo/discussions)
- **Email**: your.email@institution.edu

## 🗺️ Roadmap

### Current Version (1.0)
- ✅ Advanced Safe RL algorithms
- ✅ Classical control baselines  
- ✅ Comprehensive evaluation framework
- ✅ Statistical analysis tools
- ✅ Publication-quality results

### Upcoming Features (1.1)
- 🔄 Real robot integration
- 🔄 Online human studies
- 🔄 Interactive visualization dashboard
- 🔄 Hyperparameter optimization
- 🔄 Multi-objective optimization

### Future Plans (2.0)
- 🔮 Federated learning support
- 🔮 Lifelong learning evaluation
- 🔮 Sim-to-real transfer analysis
- 🔮 Human preference learning
- 🔮 Explainable AI integration

---

**Note**: This is a research framework designed for scientific evaluation and comparison of Safe RL algorithms. For production deployments, additional safety validation and testing are required.

For more information, visit our [documentation](docs/) or check out our [examples](examples/).
