"""
Comprehensive Evaluation Framework for Safe RL Benchmarking.

This package provides a complete evaluation system for comparing Safe RL approaches
with state-of-the-art baselines across multiple metrics and environments.

Components:
- Standardized Environment Suite with various robot platforms
- Performance Metrics: Sample efficiency, asymptotic performance, safety violations
- Human-Centric Metrics: Satisfaction, trust, workload, naturalness
- Statistical Analysis: Hypothesis testing, effect size calculations, confidence intervals
- Visualization: Publication-quality plots, charts, and performance profiles
- Reproducibility: Fixed seeds, logging, experiment tracking
"""

from .evaluation_suite import (
    EvaluationSuite,
    EvaluationConfig,
    EvaluationMetrics,
    BenchmarkResult
)

from .environments import (
    StandardizedEnvSuite,
    RobotPlatform,
    HumanModelType,
    SafetyScenario
)

from .metrics import (
    PerformanceMetrics,
    HumanMetrics, 
    SafetyMetrics,
    EfficiencyMetrics,
    MetricAggregator
)

from .statistics import (
    StatisticalAnalyzer,
    EffectSizeCalculator,
    ConfidenceIntervalEstimator,
    HypothesisTest
)

from .visualization import (
    ResultVisualizer,
    PerformanceProfiler,
    PublicationPlots
)

__all__ = [
    'EvaluationSuite',
    'EvaluationConfig', 
    'EvaluationMetrics',
    'BenchmarkResult',
    'StandardizedEnvSuite',
    'RobotPlatform',
    'HumanModelType',
    'SafetyScenario',
    'PerformanceMetrics',
    'HumanMetrics',
    'SafetyMetrics', 
    'EfficiencyMetrics',
    'MetricAggregator',
    'StatisticalAnalyzer',
    'EffectSizeCalculator',
    'ConfidenceIntervalEstimator',
    'HypothesisTest',
    'ResultVisualizer',
    'PerformanceProfiler',
    'PublicationPlots'
]