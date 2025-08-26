"""
Validation Framework for Human Subject Studies.

This module provides comprehensive tools for designing, conducting, and analyzing
human subject studies to validate the human modeling components. Includes
experimental protocols, statistical analysis, and compliance with research ethics.

Key Features:
- IRB-compliant experimental protocols
- Statistical power analysis and sample size calculation
- Multi-factorial experimental design
- Real-time data validation and quality control
- Advanced statistical analysis (ANOVA, regression, ML validation)
- Automated report generation
- Cross-validation and generalization testing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import logging
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Statistical packages
try:
    from statsmodels.stats.power import ttest_power, anova_power
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudyType(Enum):
    """Types of validation studies."""
    BIOMECHANICAL_VALIDATION = "biomechanical_validation"
    INTENT_RECOGNITION_VALIDATION = "intent_recognition_validation"
    ADAPTATION_VALIDATION = "adaptation_validation"
    VARIABILITY_VALIDATION = "variability_validation"
    SYSTEM_INTEGRATION = "system_integration"
    USABILITY_STUDY = "usability_study"
    SAFETY_VALIDATION = "safety_validation"


class ExperimentalCondition(Enum):
    """Experimental conditions for testing."""
    BASELINE = "baseline"
    NO_ASSISTANCE = "no_assistance"
    FULL_ASSISTANCE = "full_assistance"
    ADAPTIVE_ASSISTANCE = "adaptive_assistance"
    DEGRADED_SENSORS = "degraded_sensors"
    HIGH_WORKLOAD = "high_workload"
    FATIGUE_CONDITION = "fatigue_condition"


@dataclass
class ParticipantProfile:
    """Individual participant profile."""
    participant_id: str
    age: int
    gender: str
    handedness: str
    experience_robotics: int  # years
    experience_task: int  # years
    physical_limitations: List[str] = field(default_factory=list)
    cognitive_assessment_scores: Dict[str, float] = field(default_factory=dict)
    anthropometric_data: Dict[str, float] = field(default_factory=dict)
    consent_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    exclusion_criteria_met: bool = False
    
    def meets_inclusion_criteria(self, study_requirements: Dict[str, Any]) -> bool:
        """Check if participant meets inclusion criteria."""
        # Age requirements
        if 'age_range' in study_requirements:
            age_min, age_max = study_requirements['age_range']
            if not (age_min <= self.age <= age_max):
                return False
        
        # Experience requirements
        if 'min_robotics_experience' in study_requirements:
            if self.experience_robotics < study_requirements['min_robotics_experience']:
                return False
        
        # Physical limitations check
        excluded_conditions = study_requirements.get('excluded_physical_conditions', [])
        if any(condition in self.physical_limitations for condition in excluded_conditions):
            return False
        
        return not self.exclusion_criteria_met


@dataclass
class ExperimentalTrial:
    """Single experimental trial data."""
    trial_id: str
    participant_id: str
    condition: ExperimentalCondition
    task_parameters: Dict[str, Any]
    start_time: float
    end_time: float
    
    # Performance metrics
    completion_time: float = 0.0
    success_rate: float = 0.0
    path_efficiency: float = 0.0
    force_smoothness: float = 0.0
    movement_accuracy: float = 0.0
    
    # Physiological data
    emg_data: Dict[str, np.ndarray] = field(default_factory=dict)
    force_data: np.ndarray = field(default_factory=lambda: np.array([]))
    kinematic_data: Dict[str, np.ndarray] = field(default_factory=dict)
    eye_tracking_data: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Subjective measures
    workload_score: float = 0.0  # NASA-TLX
    usability_score: float = 0.0  # SUS
    trust_score: float = 0.0
    perceived_safety: float = 0.0
    
    # Model predictions vs actual
    predicted_intent: Dict[str, float] = field(default_factory=dict)
    actual_intent: str = ""
    predicted_assistance_need: float = 0.0
    actual_assistance_provided: float = 0.0
    
    # Quality metrics
    data_quality_score: float = 1.0
    missing_data_percentage: float = 0.0
    
    def compute_performance_score(self) -> float:
        """Compute overall performance score."""
        metrics = [
            self.success_rate,
            self.path_efficiency,
            self.force_smoothness,
            self.movement_accuracy,
            1.0 - min(1.0, self.completion_time / 60.0)  # Normalize by 60 seconds
        ]
        
        return np.mean([m for m in metrics if m > 0])


@dataclass
class StudyProtocol:
    """Experimental study protocol."""
    study_id: str
    study_type: StudyType
    title: str
    description: str
    
    # Participant requirements
    target_sample_size: int
    inclusion_criteria: Dict[str, Any]
    exclusion_criteria: List[str]
    
    # Experimental design
    conditions: List[ExperimentalCondition]
    trials_per_condition: int
    randomization_scheme: str = "block_randomization"
    counterbalancing: bool = True
    
    # Tasks and measures
    tasks: List[Dict[str, Any]] = field(default_factory=list)
    primary_outcomes: List[str] = field(default_factory=list)
    secondary_outcomes: List[str] = field(default_factory=list)
    
    # Timing
    session_duration_minutes: int = 60
    break_intervals_minutes: List[int] = field(default_factory=lambda: [15, 30, 45])
    
    # Safety and ethics
    irb_approval_number: str = ""
    risk_assessment: Dict[str, str] = field(default_factory=dict)
    safety_protocols: List[str] = field(default_factory=list)
    
    # Data collection specifications
    sensor_configurations: Dict[str, Any] = field(default_factory=dict)
    data_quality_requirements: Dict[str, float] = field(default_factory=dict)


class PowerAnalysis:
    """Statistical power analysis for study design."""
    
    @staticmethod
    def calculate_sample_size(effect_size: float, 
                            power: float = 0.8, 
                            alpha: float = 0.05,
                            test_type: str = "two_sample_ttest") -> int:
        """Calculate required sample size."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available - using approximation")
            # Cohen's rule of thumb for two-sample t-test
            if test_type == "two_sample_ttest":
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_beta = stats.norm.ppf(power)
                n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
                return int(np.ceil(n))
        
        try:
            if test_type == "two_sample_ttest":
                sample_size = ttest_power(effect_size, power=power, alpha=alpha)
                return int(np.ceil(sample_size))
            elif test_type == "anova":
                # For ANOVA, need to specify number of groups
                n_groups = 3  # Default assumption
                sample_size = anova_power(effect_size, n_groups, power=power, alpha=alpha)
                return int(np.ceil(sample_size))
        except Exception as e:
            logger.error(f"Power analysis error: {e}")
        
        # Fallback calculation
        return max(30, int(np.ceil(16 * (2 / effect_size) ** 2)))
    
    @staticmethod
    def estimate_effect_size(pilot_data: Dict[str, List[float]]) -> float:
        """Estimate effect size from pilot data."""
        if len(pilot_data) < 2:
            return 0.5  # Medium effect size default
        
        groups = list(pilot_data.values())
        if len(groups) >= 2:
            group1, group2 = groups[0], groups[1]
            
            # Cohen's d
            pooled_std = np.sqrt((np.var(group1) + np.var(group2)) / 2)
            if pooled_std > 0:
                effect_size = abs(np.mean(group1) - np.mean(group2)) / pooled_std
                return effect_size
        
        return 0.5


class ExperimentalDesign:
    """Generate experimental designs and randomization."""
    
    def __init__(self, protocol: StudyProtocol):
        self.protocol = protocol
        self.randomization_seed = 42
        
    def generate_trial_sequence(self, participant_id: str) -> List[Dict[str, Any]]:
        """Generate randomized trial sequence for participant."""
        np.random.seed(hash(participant_id) % 2**32)  # Deterministic per participant
        
        trials = []
        trial_counter = 0
        
        if self.protocol.randomization_scheme == "complete_randomization":
            # Completely randomized design
            all_conditions = []
            for condition in self.protocol.conditions:
                all_conditions.extend([condition] * self.protocol.trials_per_condition)
            
            np.random.shuffle(all_conditions)
            
            for i, condition in enumerate(all_conditions):
                trials.append({
                    'trial_id': f"{participant_id}_trial_{i:03d}",
                    'condition': condition,
                    'block': i // len(self.protocol.conditions),
                    'order': i
                })
        
        elif self.protocol.randomization_scheme == "block_randomization":
            # Randomized block design
            n_blocks = self.protocol.trials_per_condition
            
            for block in range(n_blocks):
                # Randomize condition order within each block
                conditions = self.protocol.conditions.copy()
                np.random.shuffle(conditions)
                
                for order, condition in enumerate(conditions):
                    trials.append({
                        'trial_id': f"{participant_id}_block{block}_trial_{order:02d}",
                        'condition': condition,
                        'block': block,
                        'order': order
                    })
        
        elif self.protocol.randomization_scheme == "latin_square":
            # Latin square design (for counterbalancing)
            n_conditions = len(self.protocol.conditions)
            n_participants_per_square = n_conditions
            
            # Generate Latin square
            latin_square = self._generate_latin_square(n_conditions)
            participant_row = hash(participant_id) % n_participants_per_square
            
            for trial_num in range(self.protocol.trials_per_condition):
                for order, condition_idx in enumerate(latin_square[participant_row]):
                    condition = self.protocol.conditions[condition_idx]
                    trials.append({
                        'trial_id': f"{participant_id}_latin_{trial_num}_{order:02d}",
                        'condition': condition,
                        'block': trial_num,
                        'order': order
                    })
        
        return trials
    
    def _generate_latin_square(self, n: int) -> List[List[int]]:
        """Generate Latin square for counterbalancing."""
        square = []
        for i in range(n):
            row = [(i + j) % n for j in range(n)]
            square.append(row)
        return square
    
    def assign_task_parameters(self, condition: ExperimentalCondition) -> Dict[str, Any]:
        """Assign task parameters based on condition."""
        base_params = {
            'target_size': 0.02,  # meters
            'movement_distance': 0.3,  # meters
            'time_limit': 30.0,  # seconds
            'force_threshold': 10.0,  # Newtons
            'precision_requirement': 0.005  # meters
        }
        
        if condition == ExperimentalCondition.HIGH_WORKLOAD:
            base_params.update({
                'target_size': 0.01,  # Smaller targets
                'time_limit': 15.0,  # Less time
                'secondary_task': True
            })
        
        elif condition == ExperimentalCondition.DEGRADED_SENSORS:
            base_params.update({
                'sensor_noise_level': 0.3,
                'sensor_dropout_rate': 0.1
            })
        
        elif condition == ExperimentalCondition.FATIGUE_CONDITION:
            base_params.update({
                'continuous_duration': 600.0,  # 10 minutes continuous
                'force_threshold': 20.0  # Higher force requirement
            })
        
        return base_params


class DataValidator:
    """Validate experimental data quality in real-time."""
    
    def __init__(self, quality_thresholds: Dict[str, float]):
        self.quality_thresholds = quality_thresholds
        self.validation_history = []
        
    def validate_trial_data(self, trial: ExperimentalTrial) -> Dict[str, Any]:
        """Validate data quality for a single trial."""
        validation_results = {
            'overall_quality': 1.0,
            'issues': [],
            'warnings': [],
            'valid': True
        }
        
        # EMG data validation
        emg_quality = self._validate_emg_data(trial.emg_data)
        validation_results['emg_quality'] = emg_quality
        
        # Force data validation
        force_quality = self._validate_force_data(trial.force_data)
        validation_results['force_quality'] = force_quality
        
        # Kinematic data validation
        kinematic_quality = self._validate_kinematic_data(trial.kinematic_data)
        validation_results['kinematic_quality'] = kinematic_quality
        
        # Performance metrics validation
        performance_quality = self._validate_performance_metrics(trial)
        validation_results['performance_quality'] = performance_quality
        
        # Overall quality score
        quality_scores = [emg_quality, force_quality, kinematic_quality, performance_quality]
        validation_results['overall_quality'] = np.mean([q for q in quality_scores if q > 0])
        
        # Determine if trial is valid
        min_quality = self.quality_thresholds.get('min_overall_quality', 0.7)
        validation_results['valid'] = validation_results['overall_quality'] >= min_quality
        
        # Update trial quality metrics
        trial.data_quality_score = validation_results['overall_quality']
        
        return validation_results
    
    def _validate_emg_data(self, emg_data: Dict[str, np.ndarray]) -> float:
        """Validate EMG data quality."""
        if not emg_data:
            return 0.0
        
        quality_scores = []
        
        for channel, data in emg_data.items():
            if len(data) == 0:
                quality_scores.append(0.0)
                continue
            
            # Check for saturation
            saturation_ratio = np.mean(np.abs(data) > 0.95 * np.max(np.abs(data)))
            saturation_quality = 1.0 - saturation_ratio
            
            # Check signal-to-noise ratio
            signal_power = np.var(data)
            if signal_power > 0:
                # Estimate noise from high-frequency components
                from scipy.signal import butter, filtfilt
                nyquist = 500  # Assume 1000 Hz sampling
                high_freq = butter(4, 100/nyquist, btype='high')
                noise_estimate = np.var(filtfilt(high_freq[0], high_freq[1], data))
                snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
                snr_quality = min(1.0, max(0.0, (snr - 10) / 20))  # 10-30 dB range
            else:
                snr_quality = 0.0
            
            # Check for missing data
            missing_ratio = np.mean(np.isnan(data) | np.isinf(data))
            completeness_quality = 1.0 - missing_ratio
            
            channel_quality = np.mean([saturation_quality, snr_quality, completeness_quality])
            quality_scores.append(channel_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _validate_force_data(self, force_data: np.ndarray) -> float:
        """Validate force sensor data quality."""
        if len(force_data) == 0:
            return 0.0
        
        # Check for reasonable force magnitudes
        force_magnitude = np.linalg.norm(force_data.reshape(-1, 6)[:, :3], axis=1)
        reasonable_forces = np.logical_and(force_magnitude >= 0, force_magnitude <= 500)
        magnitude_quality = np.mean(reasonable_forces)
        
        # Check for sensor drift
        if len(force_magnitude) > 100:
            drift_trend = np.polyfit(np.arange(len(force_magnitude)), force_magnitude, 1)[0]
            drift_quality = max(0.0, 1.0 - abs(drift_trend) / 10.0)  # 10 N/s max drift
        else:
            drift_quality = 1.0
        
        # Check data completeness
        missing_ratio = np.mean(np.isnan(force_data) | np.isinf(force_data))
        completeness_quality = 1.0 - missing_ratio
        
        return np.mean([magnitude_quality, drift_quality, completeness_quality])
    
    def _validate_kinematic_data(self, kinematic_data: Dict[str, np.ndarray]) -> float:
        """Validate kinematic tracking data quality."""
        if not kinematic_data:
            return 0.0
        
        quality_scores = []
        
        # Validate position data
        if 'position' in kinematic_data:
            position_data = kinematic_data['position']
            if len(position_data) > 0:
                # Check workspace bounds
                in_workspace = np.all(np.abs(position_data) <= 1.0, axis=1)  # ±1m workspace
                workspace_quality = np.mean(in_workspace)
                
                # Check for tracking dropouts
                position_jumps = np.linalg.norm(np.diff(position_data, axis=0), axis=1)
                max_reasonable_jump = 0.1  # 10cm per sample max
                no_dropouts = np.mean(position_jumps <= max_reasonable_jump)
                
                quality_scores.append(np.mean([workspace_quality, no_dropouts]))
        
        # Validate velocity data
        if 'velocity' in kinematic_data:
            velocity_data = kinematic_data['velocity']
            if len(velocity_data) > 0:
                velocity_magnitude = np.linalg.norm(velocity_data, axis=1)
                reasonable_velocities = velocity_magnitude <= 5.0  # 5 m/s max
                velocity_quality = np.mean(reasonable_velocities)
                quality_scores.append(velocity_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _validate_performance_metrics(self, trial: ExperimentalTrial) -> float:
        """Validate performance metrics."""
        quality_scores = []
        
        # Check completion time reasonableness
        if 0 < trial.completion_time <= trial.task_parameters.get('time_limit', 60):
            quality_scores.append(1.0)
        else:
            quality_scores.append(0.5)
        
        # Check success rate bounds
        if 0 <= trial.success_rate <= 1:
            quality_scores.append(1.0)
        else:
            quality_scores.append(0.0)
        
        # Check other metrics are in reasonable ranges
        bounded_metrics = [
            ('path_efficiency', 0, 2),
            ('force_smoothness', 0, 1),
            ('movement_accuracy', 0, 1),
            ('workload_score', 1, 100),  # NASA-TLX scale
            ('usability_score', 0, 100),  # SUS scale
            ('trust_score', 0, 10),
            ('perceived_safety', 0, 10)
        ]
        
        for metric_name, min_val, max_val in bounded_metrics:
            metric_value = getattr(trial, metric_name, 0)
            if min_val <= metric_value <= max_val:
                quality_scores.append(1.0)
            else:
                quality_scores.append(0.5)
        
        return np.mean(quality_scores)


class StatisticalAnalyzer:
    """Advanced statistical analysis for validation studies."""
    
    def __init__(self):
        self.analysis_cache = {}
        
    def analyze_study_results(self, trials: List[ExperimentalTrial]) -> Dict[str, Any]:
        """Comprehensive statistical analysis of study results."""
        logger.info(f"Analyzing {len(trials)} trials...")
        
        # Convert to DataFrame for analysis
        df = self._trials_to_dataframe(trials)
        
        analysis_results = {
            'descriptive_statistics': self._descriptive_analysis(df),
            'inferential_statistics': self._inferential_analysis(df),
            'model_validation': self._model_validation_analysis(df),
            'effect_sizes': self._effect_size_analysis(df),
            'correlation_analysis': self._correlation_analysis(df),
            'regression_analysis': self._regression_analysis(df)
        }
        
        return analysis_results
    
    def _trials_to_dataframe(self, trials: List[ExperimentalTrial]) -> pd.DataFrame:
        """Convert trials to pandas DataFrame."""
        data = []
        
        for trial in trials:
            row = {
                'trial_id': trial.trial_id,
                'participant_id': trial.participant_id,
                'condition': trial.condition.value,
                'completion_time': trial.completion_time,
                'success_rate': trial.success_rate,
                'path_efficiency': trial.path_efficiency,
                'force_smoothness': trial.force_smoothness,
                'movement_accuracy': trial.movement_accuracy,
                'workload_score': trial.workload_score,
                'usability_score': trial.usability_score,
                'trust_score': trial.trust_score,
                'perceived_safety': trial.perceived_safety,
                'performance_score': trial.compute_performance_score(),
                'data_quality': trial.data_quality_score,
                'predicted_assistance_need': trial.predicted_assistance_need,
                'actual_assistance_provided': trial.actual_assistance_provided
            }
            
            # Add intent prediction accuracy
            if trial.predicted_intent and trial.actual_intent:
                predicted_class = max(trial.predicted_intent.keys(), 
                                    key=lambda k: trial.predicted_intent[k])
                row['intent_prediction_correct'] = (predicted_class == trial.actual_intent)
                row['intent_prediction_confidence'] = max(trial.predicted_intent.values())
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Descriptive statistical analysis."""
        results = {}
        
        # Overall descriptive statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results['overall_stats'] = df[numeric_cols].describe().to_dict()
        
        # Condition-wise statistics
        results['condition_stats'] = {}
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            results['condition_stats'][condition] = condition_data[numeric_cols].describe().to_dict()
        
        # Participant-wise statistics (for repeated measures analysis)
        results['participant_stats'] = {}
        for participant in df['participant_id'].unique():
            participant_data = df[df['participant_id'] == participant]
            results['participant_stats'][participant] = participant_data[numeric_cols].mean().to_dict()
        
        return results
    
    def _inferential_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inferential statistical tests."""
        results = {}
        
        # Primary outcome variables
        outcome_vars = ['completion_time', 'success_rate', 'performance_score', 
                       'workload_score', 'usability_score']
        
        for outcome in outcome_vars:
            if outcome not in df.columns:
                continue
            
            # One-way ANOVA across conditions
            condition_groups = [df[df['condition'] == cond][outcome].dropna() 
                              for cond in df['condition'].unique()]
            
            if len(condition_groups) >= 2 and all(len(group) > 1 for group in condition_groups):
                f_stat, p_value = stats.f_oneway(*condition_groups)
                results[f'{outcome}_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                
                # Post-hoc tests if significant
                if p_value < 0.05 and STATSMODELS_AVAILABLE:
                    try:
                        # Tukey's HSD for multiple comparisons
                        tukey_results = pairwise_tukeyhsd(df[outcome].dropna(), 
                                                        df[df[outcome].notna()]['condition'])
                        results[f'{outcome}_posthoc'] = {
                            'method': 'Tukey HSD',
                            'summary': str(tukey_results)
                        }
                    except Exception as e:
                        logger.warning(f"Post-hoc test failed for {outcome}: {e}")
        
        # Paired comparisons for key conditions
        baseline_condition = 'baseline'
        if baseline_condition in df['condition'].values:
            for condition in df['condition'].unique():
                if condition != baseline_condition:
                    for outcome in outcome_vars:
                        if outcome not in df.columns:
                            continue
                        
                        baseline_data = df[df['condition'] == baseline_condition][outcome].dropna()
                        condition_data = df[df['condition'] == condition][outcome].dropna()
                        
                        if len(baseline_data) > 0 and len(condition_data) > 0:
                            t_stat, p_value = stats.ttest_ind(baseline_data, condition_data)
                            results[f'{outcome}_{condition}_vs_baseline'] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'effect_direction': 'positive' if t_stat > 0 else 'negative'
                            }
        
        return results
    
    def _model_validation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analysis of model prediction accuracy."""
        results = {}
        
        # Intent prediction accuracy
        if 'intent_prediction_correct' in df.columns:
            accuracy = df['intent_prediction_correct'].mean()
            confidence_interval = stats.binom.interval(0.95, len(df), accuracy)
            
            results['intent_accuracy'] = {
                'accuracy': accuracy,
                'confidence_interval_95': [ci/len(df) for ci in confidence_interval],
                'n_predictions': len(df)
            }
            
            # Accuracy by condition
            results['intent_accuracy_by_condition'] = {}
            for condition in df['condition'].unique():
                condition_data = df[df['condition'] == condition]
                if 'intent_prediction_correct' in condition_data.columns:
                    cond_accuracy = condition_data['intent_prediction_correct'].mean()
                    results['intent_accuracy_by_condition'][condition] = cond_accuracy
        
        # Assistance need prediction accuracy
        if 'predicted_assistance_need' in df.columns and 'actual_assistance_provided' in df.columns:
            predicted = df['predicted_assistance_need'].dropna()
            actual = df['actual_assistance_provided'].dropna()
            
            if len(predicted) > 0 and len(actual) > 0:
                # Correlation between predicted and actual
                correlation, p_value = stats.pearsonr(predicted, actual)
                
                # Mean absolute error
                mae = np.mean(np.abs(predicted - actual))
                
                # Root mean square error
                rmse = np.sqrt(np.mean((predicted - actual) ** 2))
                
                results['assistance_prediction'] = {
                    'correlation': correlation,
                    'correlation_p_value': p_value,
                    'mean_absolute_error': mae,
                    'rmse': rmse,
                    'r_squared': correlation ** 2
                }
        
        return results
    
    def _effect_size_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate effect sizes for comparisons."""
        results = {}
        
        outcome_vars = ['completion_time', 'success_rate', 'performance_score', 
                       'workload_score', 'usability_score']
        
        # Effect sizes compared to baseline
        baseline_condition = 'baseline'
        if baseline_condition in df['condition'].values:
            baseline_data = df[df['condition'] == baseline_condition]
            
            for condition in df['condition'].unique():
                if condition != baseline_condition:
                    condition_data = df[df['condition'] == condition]
                    
                    results[f'{condition}_vs_baseline'] = {}
                    
                    for outcome in outcome_vars:
                        if outcome not in df.columns:
                            continue
                        
                        baseline_vals = baseline_data[outcome].dropna()
                        condition_vals = condition_data[outcome].dropna()
                        
                        if len(baseline_vals) > 0 and len(condition_vals) > 0:
                            # Cohen's d
                            pooled_std = np.sqrt((baseline_vals.var() + condition_vals.var()) / 2)
                            if pooled_std > 0:
                                cohens_d = (condition_vals.mean() - baseline_vals.mean()) / pooled_std
                                
                                # Effect size interpretation
                                if abs(cohens_d) < 0.2:
                                    magnitude = 'negligible'
                                elif abs(cohens_d) < 0.5:
                                    magnitude = 'small'
                                elif abs(cohens_d) < 0.8:
                                    magnitude = 'medium'
                                else:
                                    magnitude = 'large'
                                
                                results[f'{condition}_vs_baseline'][outcome] = {
                                    'cohens_d': cohens_d,
                                    'magnitude': magnitude,
                                    'direction': 'improvement' if cohens_d > 0 else 'decline'
                                }
        
        return results
    
    def _correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Correlation analysis between variables."""
        results = {}
        
        # Select relevant numeric columns
        numeric_cols = ['completion_time', 'success_rate', 'path_efficiency', 
                       'force_smoothness', 'movement_accuracy', 'workload_score',
                       'usability_score', 'trust_score', 'perceived_safety',
                       'performance_score', 'data_quality']
        
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            correlation_matrix = df[available_cols].corr()
            results['correlation_matrix'] = correlation_matrix.to_dict()
            
            # Significant correlations (p < 0.05)
            results['significant_correlations'] = {}
            
            for i, var1 in enumerate(available_cols):
                for var2 in available_cols[i+1:]:
                    if var1 in df.columns and var2 in df.columns:
                        valid_data = df[[var1, var2]].dropna()
                        if len(valid_data) > 2:
                            corr, p_value = stats.pearsonr(valid_data[var1], valid_data[var2])
                            
                            if p_value < 0.05:
                                results['significant_correlations'][f'{var1}_vs_{var2}'] = {
                                    'correlation': corr,
                                    'p_value': p_value,
                                    'strength': self._interpret_correlation_strength(abs(corr))
                                }
        
        return results
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr < 0.1:
            return 'negligible'
        elif abs_corr < 0.3:
            return 'weak'
        elif abs_corr < 0.5:
            return 'moderate'
        elif abs_corr < 0.7:
            return 'strong'
        else:
            return 'very_strong'
    
    def _regression_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Regression analysis for predictive modeling."""
        results = {}
        
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available - skipping regression analysis")
            return results
        
        # Multiple regression for performance prediction
        if all(col in df.columns for col in ['completion_time', 'success_rate', 
                                           'path_efficiency', 'performance_score']):
            try:
                # Predict performance score from other metrics
                model_formula = 'performance_score ~ completion_time + success_rate + path_efficiency'
                model = ols(model_formula, data=df).fit()
                
                results['performance_regression'] = {
                    'r_squared': model.rsquared,
                    'adjusted_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue,
                    'coefficients': model.params.to_dict(),
                    'p_values': model.pvalues.to_dict(),
                    'summary': str(model.summary())
                }
                
            except Exception as e:
                logger.warning(f"Regression analysis failed: {e}")
        
        return results


class ValidationFramework:
    """Main validation framework coordinating all components."""
    
    def __init__(self):
        self.protocols: Dict[str, StudyProtocol] = {}
        self.participants: Dict[str, ParticipantProfile] = {}
        self.trials: List[ExperimentalTrial] = []
        self.data_validator = DataValidator({'min_overall_quality': 0.7})
        self.statistical_analyzer = StatisticalAnalyzer()
        self.experimental_designer = None
        
    def create_study_protocol(self, study_type: StudyType, 
                            custom_params: Optional[Dict[str, Any]] = None) -> StudyProtocol:
        """Create standardized study protocol."""
        protocol_templates = {
            StudyType.BIOMECHANICAL_VALIDATION: {
                'title': 'Biomechanical Model Validation Study',
                'target_sample_size': 30,
                'conditions': [ExperimentalCondition.BASELINE, 
                             ExperimentalCondition.NO_ASSISTANCE,
                             ExperimentalCondition.FULL_ASSISTANCE],
                'trials_per_condition': 10,
                'primary_outcomes': ['emg_prediction_accuracy', 'force_prediction_accuracy'],
                'session_duration_minutes': 90
            },
            StudyType.INTENT_RECOGNITION_VALIDATION: {
                'title': 'Intent Recognition System Validation',
                'target_sample_size': 40,
                'conditions': [ExperimentalCondition.BASELINE,
                             ExperimentalCondition.HIGH_WORKLOAD,
                             ExperimentalCondition.DEGRADED_SENSORS],
                'trials_per_condition': 15,
                'primary_outcomes': ['intent_classification_accuracy', 'response_time'],
                'session_duration_minutes': 60
            },
            StudyType.ADAPTATION_VALIDATION: {
                'title': 'Adaptive Model Validation Study',
                'target_sample_size': 25,
                'conditions': [ExperimentalCondition.BASELINE,
                             ExperimentalCondition.ADAPTIVE_ASSISTANCE],
                'trials_per_condition': 20,
                'primary_outcomes': ['learning_curve', 'assistance_optimization'],
                'session_duration_minutes': 120
            },
            StudyType.USABILITY_STUDY: {
                'title': 'System Usability and User Experience Study',
                'target_sample_size': 20,
                'conditions': [ExperimentalCondition.NO_ASSISTANCE,
                             ExperimentalCondition.FULL_ASSISTANCE,
                             ExperimentalCondition.ADAPTIVE_ASSISTANCE],
                'trials_per_condition': 8,
                'primary_outcomes': ['usability_score', 'trust_score', 'workload_score'],
                'session_duration_minutes': 90
            }
        }
        
        base_template = protocol_templates.get(study_type, {})
        
        # Apply custom parameters
        if custom_params:
            base_template.update(custom_params)
        
        # Calculate sample size with power analysis
        if 'effect_size' in base_template:
            power_analysis = PowerAnalysis()
            calculated_n = power_analysis.calculate_sample_size(
                base_template['effect_size'],
                power=base_template.get('power', 0.8),
                alpha=base_template.get('alpha', 0.05)
            )
            base_template['target_sample_size'] = max(base_template['target_sample_size'], calculated_n)
        
        protocol = StudyProtocol(
            study_id=f"study_{study_type.value}_{datetime.datetime.now().strftime('%Y%m%d')}",
            study_type=study_type,
            inclusion_criteria={
                'age_range': (18, 65),
                'min_robotics_experience': 0,
                'excluded_physical_conditions': ['severe_motor_impairment']
            },
            exclusion_criteria=['pregnancy', 'cardiac_pacemaker', 'severe_visual_impairment'],
            randomization_scheme='block_randomization',
            **base_template
        )
        
        self.protocols[protocol.study_id] = protocol
        self.experimental_designer = ExperimentalDesign(protocol)
        
        logger.info(f"Created study protocol: {protocol.title}")
        return protocol
    
    def recruit_participant(self, participant_data: Dict[str, Any]) -> ParticipantProfile:
        """Process participant recruitment and screening."""
        participant = ParticipantProfile(**participant_data)
        
        # Check inclusion/exclusion criteria
        if self.protocols:
            # Use first available protocol for screening
            protocol = list(self.protocols.values())[0]
            
            if not participant.meets_inclusion_criteria(protocol.inclusion_criteria):
                participant.exclusion_criteria_met = True
                logger.warning(f"Participant {participant.participant_id} excluded due to criteria")
        
        self.participants[participant.participant_id] = participant
        logger.info(f"Recruited participant: {participant.participant_id}")
        
        return participant
    
    def generate_session_plan(self, participant_id: str, protocol_id: str) -> List[Dict[str, Any]]:
        """Generate experimental session plan for participant."""
        if protocol_id not in self.protocols:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not found")
        
        protocol = self.protocols[protocol_id]
        self.experimental_designer = ExperimentalDesign(protocol)
        
        # Generate trial sequence
        trial_sequence = self.experimental_designer.generate_trial_sequence(participant_id)
        
        # Add task parameters to each trial
        for trial_info in trial_sequence:
            condition = trial_info['condition']
            trial_info['task_parameters'] = self.experimental_designer.assign_task_parameters(condition)
        
        logger.info(f"Generated session plan for {participant_id}: {len(trial_sequence)} trials")
        return trial_sequence
    
    def record_trial(self, trial: ExperimentalTrial) -> Dict[str, Any]:
        """Record and validate experimental trial."""
        # Validate data quality
        validation_results = self.data_validator.validate_trial_data(trial)
        
        # Store trial
        self.trials.append(trial)
        
        logger.info(f"Recorded trial {trial.trial_id}, quality: {validation_results['overall_quality']:.3f}")
        
        return validation_results
    
    def analyze_results(self, study_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensive analysis of study results."""
        # Filter trials if specified
        trials_to_analyze = self.trials
        
        if study_filter:
            if 'participant_ids' in study_filter:
                trials_to_analyze = [t for t in trials_to_analyze 
                                   if t.participant_id in study_filter['participant_ids']]
            
            if 'conditions' in study_filter:
                trials_to_analyze = [t for t in trials_to_analyze 
                                   if t.condition in study_filter['conditions']]
            
            if 'min_quality' in study_filter:
                trials_to_analyze = [t for t in trials_to_analyze 
                                   if t.data_quality_score >= study_filter['min_quality']]
        
        logger.info(f"Analyzing {len(trials_to_analyze)} trials...")
        
        # Run statistical analysis
        analysis_results = self.statistical_analyzer.analyze_study_results(trials_to_analyze)
        
        # Add study metadata
        analysis_results['study_metadata'] = {
            'total_trials': len(trials_to_analyze),
            'unique_participants': len(set(t.participant_id for t in trials_to_analyze)),
            'conditions_tested': list(set(t.condition.value for t in trials_to_analyze)),
            'analysis_timestamp': datetime.datetime.now().isoformat()
        }
        
        return analysis_results
    
    def generate_report(self, analysis_results: Dict[str, Any], 
                       report_type: str = 'full') -> str:
        """Generate comprehensive study report."""
        report_sections = []
        
        # Title and metadata
        metadata = analysis_results.get('study_metadata', {})
        report_sections.append(f"# Validation Study Report")
        report_sections.append(f"Generated: {metadata.get('analysis_timestamp', 'Unknown')}")
        report_sections.append(f"Total Trials: {metadata.get('total_trials', 0)}")
        report_sections.append(f"Participants: {metadata.get('unique_participants', 0)}")
        report_sections.append(f"Conditions: {', '.join(metadata.get('conditions_tested', []))}\n")
        
        # Descriptive statistics
        if 'descriptive_statistics' in analysis_results:
            report_sections.append("## Descriptive Statistics")
            desc_stats = analysis_results['descriptive_statistics']
            
            if 'overall_stats' in desc_stats:
                report_sections.append("### Overall Performance Metrics")
                for metric, stats in desc_stats['overall_stats'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        report_sections.append(
                            f"- {metric}: M = {stats['mean']:.3f}, "
                            f"SD = {stats['std']:.3f}, Range = [{stats['min']:.3f}, {stats['max']:.3f}]"
                        )
        
        # Inferential statistics
        if 'inferential_statistics' in analysis_results:
            report_sections.append("\n## Statistical Tests")
            inf_stats = analysis_results['inferential_statistics']
            
            for test_name, test_results in inf_stats.items():
                if isinstance(test_results, dict) and 'p_value' in test_results:
                    significance = "significant" if test_results['p_value'] < 0.05 else "not significant"
                    report_sections.append(
                        f"- {test_name}: F = {test_results.get('f_statistic', 'N/A'):.3f}, "
                        f"p = {test_results['p_value']:.3f} ({significance})"
                    )
        
        # Model validation results
        if 'model_validation' in analysis_results:
            report_sections.append("\n## Model Validation")
            model_val = analysis_results['model_validation']
            
            if 'intent_accuracy' in model_val:
                intent_acc = model_val['intent_accuracy']
                report_sections.append(
                    f"- Intent Recognition Accuracy: {intent_acc['accuracy']:.3f} "
                    f"(95% CI: {intent_acc['confidence_interval_95']})"
                )
            
            if 'assistance_prediction' in model_val:
                assist_pred = model_val['assistance_prediction']
                report_sections.append(
                    f"- Assistance Need Prediction: r = {assist_pred['correlation']:.3f}, "
                    f"RMSE = {assist_pred['rmse']:.3f}"
                )
        
        # Effect sizes
        if 'effect_sizes' in analysis_results:
            report_sections.append("\n## Effect Sizes")
            effect_sizes = analysis_results['effect_sizes']
            
            for comparison, effects in effect_sizes.items():
                report_sections.append(f"### {comparison}")
                for outcome, effect in effects.items():
                    if isinstance(effect, dict) and 'cohens_d' in effect:
                        report_sections.append(
                            f"- {outcome}: d = {effect['cohens_d']:.3f} ({effect['magnitude']} effect)"
                        )
        
        # Conclusions and recommendations
        report_sections.append("\n## Conclusions and Recommendations")
        report_sections.append("(Based on automated analysis - requires expert interpretation)")
        
        # Generate recommendations based on results
        recommendations = self._generate_recommendations(analysis_results)
        for rec in recommendations:
            report_sections.append(f"- {rec}")
        
        return "\n".join(report_sections)
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check model validation results
        if 'model_validation' in analysis_results:
            model_val = analysis_results['model_validation']
            
            if 'intent_accuracy' in model_val:
                accuracy = model_val['intent_accuracy']['accuracy']
                if accuracy < 0.8:
                    recommendations.append(
                        f"Intent recognition accuracy ({accuracy:.3f}) below target (0.8). "
                        "Consider improving feature extraction or model training."
                    )
                elif accuracy >= 0.9:
                    recommendations.append(
                        f"Excellent intent recognition accuracy ({accuracy:.3f}). "
                        "Model ready for deployment."
                    )
            
            if 'assistance_prediction' in model_val:
                correlation = model_val['assistance_prediction']['correlation']
                if correlation < 0.6:
                    recommendations.append(
                        f"Assistance prediction correlation ({correlation:.3f}) needs improvement. "
                        "Consider incorporating additional user state indicators."
                    )
        
        # Check effect sizes for interventions
        if 'effect_sizes' in analysis_results:
            large_effects = []
            for comparison, effects in analysis_results['effect_sizes'].items():
                for outcome, effect in effects.items():
                    if isinstance(effect, dict) and effect.get('magnitude') == 'large':
                        large_effects.append(f"{comparison} on {outcome}")
            
            if large_effects:
                recommendations.append(
                    f"Large effect sizes found for: {', '.join(large_effects)}. "
                    "These interventions show strong practical significance."
                )
        
        # Data quality recommendations
        metadata = analysis_results.get('study_metadata', {})
        n_trials = metadata.get('total_trials', 0)
        n_participants = metadata.get('unique_participants', 0)
        
        if n_participants < 20:
            recommendations.append(
                f"Small sample size (n={n_participants}). Consider recruiting additional participants "
                "to improve statistical power and generalizability."
            )
        
        if not recommendations:
            recommendations.append("No specific recommendations generated. Manual review of results recommended.")
        
        return recommendations


# Example usage and testing
if __name__ == "__main__":
    print("Testing Validation Framework...")
    
    # Create validation framework
    framework = ValidationFramework()
    
    # Create study protocol
    protocol = framework.create_study_protocol(
        StudyType.INTENT_RECOGNITION_VALIDATION,
        custom_params={'effect_size': 0.6, 'target_sample_size': 15}
    )
    
    print(f"Created protocol: {protocol.title}")
    print(f"Target sample size: {protocol.target_sample_size}")
    print(f"Conditions: {[c.value for c in protocol.conditions]}")
    
    # Simulate participant recruitment
    for i in range(10):
        participant_data = {
            'participant_id': f'P{i:03d}',
            'age': np.random.randint(20, 60),
            'gender': np.random.choice(['M', 'F']),
            'handedness': np.random.choice(['right', 'left'], p=[0.9, 0.1]),
            'experience_robotics': np.random.randint(0, 10),
            'experience_task': np.random.randint(0, 15)
        }
        framework.recruit_participant(participant_data)
    
    print(f"Recruited {len(framework.participants)} participants")
    
    # Generate and simulate experimental trials
    for participant_id in list(framework.participants.keys())[:5]:  # First 5 participants
        session_plan = framework.generate_session_plan(participant_id, protocol.study_id)
        
        # Simulate trials for this participant
        for i, trial_info in enumerate(session_plan[:6]):  # Limit to 6 trials per participant
            # Create simulated trial data
            trial = ExperimentalTrial(
                trial_id=trial_info['trial_id'],
                participant_id=participant_id,
                condition=trial_info['condition'],
                task_parameters=trial_info['task_parameters'],
                start_time=time.time() - 100 + i * 10,
                end_time=time.time() - 90 + i * 10,
                
                # Simulated performance metrics
                completion_time=np.random.uniform(15, 45),
                success_rate=np.random.uniform(0.6, 0.95),
                path_efficiency=np.random.uniform(0.7, 0.95),
                force_smoothness=np.random.uniform(0.6, 0.9),
                movement_accuracy=np.random.uniform(0.7, 0.95),
                
                # Simulated subjective measures
                workload_score=np.random.uniform(20, 80),
                usability_score=np.random.uniform(60, 90),
                trust_score=np.random.uniform(6, 9),
                perceived_safety=np.random.uniform(7, 10),
                
                # Simulated model predictions
                predicted_intent={'precision_task': 0.7, 'power_task': 0.2, 'rest': 0.1},
                actual_intent='precision_task',
                predicted_assistance_need=np.random.uniform(0.2, 0.8),
                actual_assistance_provided=np.random.uniform(0.1, 0.9)
            )
            
            # Add simulated sensor data
            trial.emg_data = {
                'biceps': np.random.normal(0.3, 0.1, 1000),
                'triceps': np.random.normal(0.2, 0.1, 1000)
            }
            trial.force_data = np.random.normal(0, 10, (1000, 6))
            
            # Record trial
            framework.record_trial(trial)
    
    print(f"Recorded {len(framework.trials)} experimental trials")
    
    # Analyze results
    analysis_results = framework.analyze_results()
    
    print("\nAnalysis Results Summary:")
    print(f"Trials analyzed: {analysis_results['study_metadata']['total_trials']}")
    print(f"Participants: {analysis_results['study_metadata']['unique_participants']}")
    
    # Check for significant results
    if 'inferential_statistics' in analysis_results:
        significant_tests = [test for test, result in analysis_results['inferential_statistics'].items()
                           if isinstance(result, dict) and result.get('p_value', 1.0) < 0.05]
        print(f"Significant statistical tests: {len(significant_tests)}")
    
    # Model validation results
    if 'model_validation' in analysis_results:
        model_val = analysis_results['model_validation']
        if 'intent_accuracy' in model_val:
            print(f"Intent recognition accuracy: {model_val['intent_accuracy']['accuracy']:.3f}")
        if 'assistance_prediction' in model_val:
            print(f"Assistance prediction correlation: {model_val['assistance_prediction']['correlation']:.3f}")
    
    # Generate report
    report = framework.generate_report(analysis_results)
    print("\n" + "="*50)
    print(report)
    
    print("\nValidation Framework test completed!")