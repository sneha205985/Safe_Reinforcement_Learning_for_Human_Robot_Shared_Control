"""
Adaptive Human Model for Online Parameter Estimation and Personalization.

This module implements adaptive modeling that learns and updates human behavior
parameters in real-time, including skill assessment, personalization engines,
and online parameter estimation using Bayesian methods.

Mathematical Framework:
- Bayesian parameter updating: P(θ|data) ∝ P(data|θ) * P(θ)
- Recursive least squares for parameter estimation
- Skill assessment based on performance metrics
- Individual calibration and anthropometric scaling
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HumanParameters:
    """Human model parameters that can be adapted online."""
    # Biomechanical parameters
    muscle_strength: Dict[str, float] = field(default_factory=lambda: {
        'biceps': 1.0, 'triceps': 1.0, 'deltoid_anterior': 1.0, 'deltoid_posterior': 1.0
    })
    muscle_fatigue_rates: Dict[str, float] = field(default_factory=lambda: {
        'biceps': 0.1, 'triceps': 0.1, 'deltoid_anterior': 0.12, 'deltoid_posterior': 0.12
    })
    
    # Anthropometric scaling factors
    limb_lengths: Dict[str, float] = field(default_factory=lambda: {
        'upper_arm': 1.0, 'forearm': 1.0, 'hand': 1.0
    })
    limb_masses: Dict[str, float] = field(default_factory=lambda: {
        'upper_arm': 1.0, 'forearm': 1.0, 'hand': 1.0
    })
    
    # Control and intent parameters
    reaction_time: float = 0.2  # seconds
    movement_smoothness: float = 1.0  # dimensionless
    precision_preference: float = 0.5  # 0=power, 1=precision
    assistance_tolerance: float = 0.5  # willingness to accept robot help
    
    # Learning and adaptation rates
    learning_rate: float = 0.01
    forgetting_rate: float = 0.001
    
    # Skill levels (0-1 scale)
    motor_skill: float = 0.5
    task_familiarity: float = 0.5
    spatial_awareness: float = 0.5
    
    # Uncertainty/confidence in parameters
    parameter_uncertainty: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert parameters to vector for optimization."""
        vector = []
        
        # Muscle parameters
        vector.extend(list(self.muscle_strength.values()))
        vector.extend(list(self.muscle_fatigue_rates.values()))
        
        # Anthropometric parameters
        vector.extend(list(self.limb_lengths.values()))
        vector.extend(list(self.limb_masses.values()))
        
        # Control parameters
        vector.extend([
            self.reaction_time,
            self.movement_smoothness,
            self.precision_preference,
            self.assistance_tolerance,
            self.learning_rate,
            self.forgetting_rate
        ])
        
        # Skill parameters
        vector.extend([
            self.motor_skill,
            self.task_familiarity,
            self.spatial_awareness
        ])
        
        return np.array(vector)
    
    def from_vector(self, vector: np.ndarray):
        """Update parameters from vector."""
        idx = 0
        
        # Muscle strength
        for muscle in self.muscle_strength.keys():
            self.muscle_strength[muscle] = max(0.1, vector[idx])
            idx += 1
        
        # Muscle fatigue rates
        for muscle in self.muscle_fatigue_rates.keys():
            self.muscle_fatigue_rates[muscle] = max(0.01, vector[idx])
            idx += 1
        
        # Limb lengths
        for limb in self.limb_lengths.keys():
            self.limb_lengths[limb] = max(0.5, vector[idx])
            idx += 1
        
        # Limb masses
        for limb in self.limb_masses.keys():
            self.limb_masses[limb] = max(0.5, vector[idx])
            idx += 1
        
        # Control parameters
        self.reaction_time = max(0.05, min(1.0, vector[idx]))
        idx += 1
        self.movement_smoothness = max(0.1, min(2.0, vector[idx]))
        idx += 1
        self.precision_preference = max(0.0, min(1.0, vector[idx]))
        idx += 1
        self.assistance_tolerance = max(0.0, min(1.0, vector[idx]))
        idx += 1
        self.learning_rate = max(0.001, min(0.1, vector[idx]))
        idx += 1
        self.forgetting_rate = max(0.0, min(0.01, vector[idx]))
        idx += 1
        
        # Skill parameters
        self.motor_skill = max(0.0, min(1.0, vector[idx]))
        idx += 1
        self.task_familiarity = max(0.0, min(1.0, vector[idx]))
        idx += 1
        self.spatial_awareness = max(0.0, min(1.0, vector[idx]))
        idx += 1


@dataclass
class PerformanceMetrics:
    """Performance metrics for skill assessment."""
    task_completion_time: float = 0.0
    path_efficiency: float = 0.0  # 0-1, higher is better
    movement_smoothness: float = 0.0  # 0-1, higher is better
    precision_error: float = 0.0  # lower is better
    force_consistency: float = 0.0  # 0-1, higher is better
    success_rate: float = 0.0  # 0-1, higher is better
    cognitive_load: float = 0.0  # 0-1, lower is better
    adaptation_speed: float = 0.0  # 0-1, higher is better
    timestamp: float = 0.0
    
    def compute_overall_score(self) -> float:
        """Compute weighted overall performance score."""
        positive_metrics = [
            self.path_efficiency,
            self.movement_smoothness,
            self.force_consistency,
            self.success_rate,
            self.adaptation_speed
        ]
        
        negative_metrics = [
            self.precision_error,
            self.cognitive_load
        ]
        
        # Normalize completion time (assuming 10s is average)
        normalized_time = max(0, 1.0 - self.task_completion_time / 10.0)
        
        # Weighted average
        weights = [0.15, 0.15, 0.10, 0.20, 0.10, 0.15, 0.10, 0.05]
        scores = positive_metrics + [1.0 - err for err in negative_metrics] + [normalized_time]
        
        return np.average(scores, weights=weights)


class OnlineParameterEstimation:
    """Online parameter estimation using recursive methods."""
    
    def __init__(self, initial_parameters: HumanParameters, 
                 forgetting_factor: float = 0.95):
        self.parameters = initial_parameters
        self.forgetting_factor = forgetting_factor
        
        # Recursive Least Squares parameters
        n_params = len(self.parameters.to_vector())
        self.P = np.eye(n_params) * 100  # Covariance matrix
        self.theta = self.parameters.to_vector()  # Parameter estimates
        
        # Kalman filter for parameter tracking
        self.Q = np.eye(n_params) * 0.01  # Process noise
        self.R = 1.0  # Measurement noise
        
        # History for convergence analysis
        self.parameter_history: List[np.ndarray] = []
        self.residual_history: List[float] = []
        
    def update_rls(self, feature_vector: np.ndarray, measurement: float):
        """Update parameters using Recursive Least Squares."""
        # RLS update equations
        # K = P * φ / (λ + φ^T * P * φ)
        # θ = θ + K * (y - φ^T * θ)
        # P = (P - K * φ^T * P) / λ
        
        phi = feature_vector.reshape(-1, 1)
        
        # Innovation
        innovation = measurement - np.dot(phi.T, self.theta)
        
        # Kalman gain
        S = np.dot(np.dot(phi.T, self.P), phi) + self.R
        K = np.dot(self.P, phi) / S
        
        # Update parameters
        self.theta += K.flatten() * innovation
        
        # Update covariance
        I_KH = np.eye(len(self.theta)) - np.outer(K, phi.T)
        self.P = np.dot(I_KH, self.P) / self.forgetting_factor
        
        # Update parameter object
        self.parameters.from_vector(self.theta)
        
        # Store history
        self.parameter_history.append(self.theta.copy())
        self.residual_history.append(float(innovation))
        
        # Limit history length
        if len(self.parameter_history) > 1000:
            self.parameter_history.pop(0)
            self.residual_history.pop(0)
    
    def update_bayesian(self, observation: Dict[str, Any], 
                       likelihood_params: Dict[str, Any]):
        """Bayesian parameter update using observation likelihood."""
        # Simplified Bayesian update for key parameters
        
        if 'muscle_activation' in observation:
            muscle_data = observation['muscle_activation']
            
            # Update muscle strength estimates
            for muscle, activation in muscle_data.items():
                if muscle in self.parameters.muscle_strength:
                    prior_strength = self.parameters.muscle_strength[muscle]
                    
                    # Simple Bayesian update (assuming Gaussian)
                    # P(θ|data) ∝ P(data|θ) * P(θ)
                    likelihood_mean = likelihood_params.get(f'{muscle}_strength_mean', activation)
                    likelihood_std = likelihood_params.get(f'{muscle}_strength_std', 0.1)
                    prior_std = 0.2
                    
                    # Posterior parameters (conjugate Gaussian)
                    posterior_precision = (1/prior_std**2) + (1/likelihood_std**2)
                    posterior_std = 1 / np.sqrt(posterior_precision)
                    posterior_mean = ((prior_strength/prior_std**2) + 
                                    (likelihood_mean/likelihood_std**2)) * posterior_std**2
                    
                    # Update with learning rate
                    self.parameters.muscle_strength[muscle] = (
                        (1 - self.parameters.learning_rate) * prior_strength +
                        self.parameters.learning_rate * posterior_mean
                    )
        
        if 'reaction_time' in observation:
            # Update reaction time estimate
            observed_rt = observation['reaction_time']
            current_rt = self.parameters.reaction_time
            
            # Exponential moving average
            self.parameters.reaction_time = (
                (1 - self.parameters.learning_rate) * current_rt +
                self.parameters.learning_rate * observed_rt
            )
    
    def get_parameter_uncertainty(self) -> Dict[str, float]:
        """Get uncertainty estimates for parameters."""
        # Extract diagonal of covariance matrix
        uncertainties = np.diag(self.P)
        
        uncertainty_dict = {}
        param_names = [
            'biceps_strength', 'triceps_strength', 'deltoid_anterior_strength', 'deltoid_posterior_strength',
            'biceps_fatigue', 'triceps_fatigue', 'deltoid_anterior_fatigue', 'deltoid_posterior_fatigue',
            'upper_arm_length', 'forearm_length', 'hand_length',
            'upper_arm_mass', 'forearm_mass', 'hand_mass',
            'reaction_time', 'movement_smoothness', 'precision_preference', 'assistance_tolerance',
            'learning_rate', 'forgetting_rate',
            'motor_skill', 'task_familiarity', 'spatial_awareness'
        ]
        
        for i, param_name in enumerate(param_names):
            if i < len(uncertainties):
                uncertainty_dict[param_name] = float(uncertainties[i])
        
        return uncertainty_dict
    
    def is_converged(self, window_size: int = 50, threshold: float = 1e-4) -> bool:
        """Check if parameter estimation has converged."""
        if len(self.parameter_history) < window_size:
            return False
        
        recent_params = self.parameter_history[-window_size:]
        param_changes = [np.linalg.norm(recent_params[i] - recent_params[i-1]) 
                        for i in range(1, len(recent_params))]
        
        return np.mean(param_changes) < threshold


class SkillAssessment:
    """Skill assessment and proficiency evaluation."""
    
    def __init__(self):
        self.performance_history: List[PerformanceMetrics] = []
        self.skill_trends: Dict[str, List[float]] = {
            'motor_skill': [],
            'task_familiarity': [],
            'spatial_awareness': []
        }
        
        # Skill thresholds
        self.skill_thresholds = {
            'novice': 0.3,
            'intermediate': 0.6,
            'expert': 0.8
        }
        
    def assess_performance(self, performance_metrics: PerformanceMetrics) -> Dict[str, float]:
        """Assess performance and update skill estimates."""
        self.performance_history.append(performance_metrics)
        
        # Motor skill assessment
        motor_components = [
            performance_metrics.movement_smoothness,
            performance_metrics.force_consistency,
            1.0 - performance_metrics.precision_error  # Convert to positive metric
        ]
        motor_skill = np.mean(motor_components)
        
        # Task familiarity assessment
        familiarity_components = [
            performance_metrics.success_rate,
            1.0 / max(0.1, performance_metrics.task_completion_time / 5.0),  # Normalized time
            performance_metrics.adaptation_speed
        ]
        task_familiarity = np.mean([min(1.0, c) for c in familiarity_components])
        
        # Spatial awareness assessment
        spatial_components = [
            performance_metrics.path_efficiency,
            1.0 - performance_metrics.cognitive_load
        ]
        spatial_awareness = np.mean(spatial_components)
        
        # Update trends
        self.skill_trends['motor_skill'].append(motor_skill)
        self.skill_trends['task_familiarity'].append(task_familiarity)
        self.skill_trends['spatial_awareness'].append(spatial_awareness)
        
        # Limit history
        for skill in self.skill_trends:
            if len(self.skill_trends[skill]) > 100:
                self.skill_trends[skill].pop(0)
        
        return {
            'motor_skill': motor_skill,
            'task_familiarity': task_familiarity,
            'spatial_awareness': spatial_awareness
        }
    
    def get_skill_level(self, skill_type: str) -> str:
        """Get qualitative skill level."""
        if skill_type not in self.skill_trends or not self.skill_trends[skill_type]:
            return 'unknown'
        
        current_skill = self.skill_trends[skill_type][-1]
        
        if current_skill < self.skill_thresholds['novice']:
            return 'novice'
        elif current_skill < self.skill_thresholds['intermediate']:
            return 'intermediate'
        else:
            return 'expert'
    
    def get_learning_rate(self, skill_type: str, window_size: int = 20) -> float:
        """Estimate learning rate from skill progression."""
        if (skill_type not in self.skill_trends or 
            len(self.skill_trends[skill_type]) < window_size):
            return 0.0
        
        recent_skills = self.skill_trends[skill_type][-window_size:]
        
        # Linear regression to estimate slope
        x = np.arange(len(recent_skills))
        coeffs = np.polyfit(x, recent_skills, 1)
        learning_rate = coeffs[0]  # Slope
        
        return max(0.0, learning_rate)  # Only positive learning
    
    def predict_performance(self, future_trials: int = 10) -> Dict[str, float]:
        """Predict future performance based on learning trends."""
        predictions = {}
        
        for skill_type, history in self.skill_trends.items():
            if len(history) < 10:
                predictions[skill_type] = history[-1] if history else 0.5
                continue
            
            # Fit learning curve (exponential approach to asymptote)
            x = np.arange(len(history))
            y = np.array(history)
            
            try:
                # Fit: y = a * (1 - exp(-b * x)) + c
                def learning_curve(x, a, b, c):
                    return a * (1 - np.exp(-b * x)) + c
                
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(learning_curve, x, y, 
                                  bounds=([0, 0, 0], [1, 1, 1]),
                                  maxfev=1000)
                
                future_x = len(history) + future_trials
                prediction = learning_curve(future_x, *popt)
                predictions[skill_type] = min(1.0, max(0.0, prediction))
                
            except Exception:
                # Fallback: linear extrapolation
                if len(history) >= 2:
                    slope = (history[-1] - history[-5]) / 5 if len(history) >= 5 else 0
                    prediction = history[-1] + slope * future_trials
                    predictions[skill_type] = min(1.0, max(0.0, prediction))
                else:
                    predictions[skill_type] = history[-1]
        
        return predictions


class PersonalizationEngine:
    """Personalization engine for individual adaptation."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_profile: Dict[str, Any] = {}
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Individual difference categories
        self.user_categories = {
            'physical_ability': 'average',  # low, average, high
            'cognitive_style': 'analytical',  # analytical, intuitive, mixed
            'risk_tolerance': 'moderate',  # conservative, moderate, aggressive
            'technology_comfort': 'intermediate'  # novice, intermediate, expert
        }
        
        # Personalized parameters
        self.personal_preferences = {
            'assistance_level': 0.5,
            'feedback_frequency': 0.7,
            'challenge_level': 0.5,
            'interaction_style': 'collaborative'  # passive, collaborative, directive
        }
        
    def build_user_profile(self, initial_assessment: Dict[str, Any]):
        """Build initial user profile from assessment."""
        self.user_profile = {
            'demographics': initial_assessment.get('demographics', {}),
            'physical_characteristics': initial_assessment.get('physical', {}),
            'cognitive_preferences': initial_assessment.get('cognitive', {}),
            'experience_level': initial_assessment.get('experience', {}),
            'goals': initial_assessment.get('goals', []),
            'constraints': initial_assessment.get('constraints', [])
        }
        
        # Update categories based on assessment
        if 'physical' in initial_assessment:
            strength = initial_assessment['physical'].get('strength_percentile', 50)
            if strength < 25:
                self.user_categories['physical_ability'] = 'low'
            elif strength > 75:
                self.user_categories['physical_ability'] = 'high'
        
        if 'cognitive' in initial_assessment:
            style = initial_assessment['cognitive'].get('decision_style', 'analytical')
            self.user_categories['cognitive_style'] = style
        
        logger.info(f"User profile built for {self.user_id}: {self.user_categories}")
    
    def personalize_parameters(self, base_parameters: HumanParameters) -> HumanParameters:
        """Personalize parameters based on user profile."""
        personalized = HumanParameters()
        
        # Copy base parameters
        for key, value in base_parameters.__dict__.items():
            if isinstance(value, dict):
                setattr(personalized, key, value.copy())
            else:
                setattr(personalized, key, value)
        
        # Apply physical ability scaling
        if self.user_categories['physical_ability'] == 'low':
            for muscle in personalized.muscle_strength:
                personalized.muscle_strength[muscle] *= 0.7
        elif self.user_categories['physical_ability'] == 'high':
            for muscle in personalized.muscle_strength:
                personalized.muscle_strength[muscle] *= 1.3
        
        # Apply cognitive style preferences
        if self.user_categories['cognitive_style'] == 'analytical':
            personalized.precision_preference = 0.7
            personalized.reaction_time *= 1.1  # More deliberate
        elif self.user_categories['cognitive_style'] == 'intuitive':
            personalized.precision_preference = 0.3
            personalized.reaction_time *= 0.9  # Faster reactions
        
        # Apply risk tolerance
        if self.user_categories['risk_tolerance'] == 'conservative':
            personalized.assistance_tolerance = 0.8  # Welcomes help
        elif self.user_categories['risk_tolerance'] == 'aggressive':
            personalized.assistance_tolerance = 0.2  # Prefers independence
        
        # Apply technology comfort
        if self.user_categories['technology_comfort'] == 'novice':
            personalized.learning_rate *= 0.5  # Slower adaptation
        elif self.user_categories['technology_comfort'] == 'expert':
            personalized.learning_rate *= 1.5  # Faster adaptation
        
        return personalized
    
    def adapt_assistance_strategy(self, performance_metrics: PerformanceMetrics, 
                                current_parameters: HumanParameters) -> Dict[str, float]:
        """Adapt assistance strategy based on performance."""
        adaptation_factors = {}
        
        # Analyze performance trends
        recent_performance = performance_metrics.compute_overall_score()
        
        # Adjust assistance level
        if recent_performance < 0.4:  # Struggling
            adaptation_factors['assistance_level'] = min(1.0, 
                self.personal_preferences['assistance_level'] + 0.2)
        elif recent_performance > 0.8:  # Performing well
            adaptation_factors['assistance_level'] = max(0.0, 
                self.personal_preferences['assistance_level'] - 0.1)
        else:
            adaptation_factors['assistance_level'] = self.personal_preferences['assistance_level']
        
        # Adjust challenge level
        if recent_performance > 0.7 and performance_metrics.cognitive_load < 0.3:
            adaptation_factors['challenge_level'] = min(1.0, 
                self.personal_preferences['challenge_level'] + 0.1)
        elif recent_performance < 0.3 or performance_metrics.cognitive_load > 0.8:
            adaptation_factors['challenge_level'] = max(0.0, 
                self.personal_preferences['challenge_level'] - 0.15)
        else:
            adaptation_factors['challenge_level'] = self.personal_preferences['challenge_level']
        
        # Adjust feedback frequency
        if performance_metrics.adaptation_speed < 0.3:  # Slow learning
            adaptation_factors['feedback_frequency'] = min(1.0, 
                self.personal_preferences['feedback_frequency'] + 0.2)
        else:
            adaptation_factors['feedback_frequency'] = self.personal_preferences['feedback_frequency']
        
        # Store adaptation decision
        self.adaptation_history.append({
            'timestamp': performance_metrics.timestamp,
            'performance_score': recent_performance,
            'adaptations': adaptation_factors.copy()
        })
        
        # Update preferences with moving average
        alpha = 0.3  # Learning rate for preference updates
        for key, new_value in adaptation_factors.items():
            if key in self.personal_preferences:
                self.personal_preferences[key] = (
                    (1 - alpha) * self.personal_preferences[key] + alpha * new_value
                )
        
        return adaptation_factors
    
    def get_personalization_summary(self) -> Dict[str, Any]:
        """Get summary of personalization status."""
        return {
            'user_id': self.user_id,
            'user_categories': self.user_categories,
            'current_preferences': self.personal_preferences,
            'adaptation_count': len(self.adaptation_history),
            'last_adaptation': self.adaptation_history[-1] if self.adaptation_history else None
        }


class AdaptiveHumanModel:
    """Complete adaptive human model integrating all components."""
    
    def __init__(self, user_id: str, initial_parameters: Optional[HumanParameters] = None):
        self.user_id = user_id
        
        # Initialize components
        self.base_parameters = initial_parameters or HumanParameters()
        self.parameter_estimator = OnlineParameterEstimation(self.base_parameters)
        self.skill_assessor = SkillAssessment()
        self.personalization_engine = PersonalizationEngine(user_id)
        
        # Current state
        self.current_parameters = self.base_parameters
        self.is_initialized = False
        
        # Adaptation control
        self.adaptation_enabled = True
        self.update_frequency = 10  # Update every 10 observations
        self.observation_count = 0
        
        logger.info(f"Adaptive Human Model initialized for user {user_id}")
    
    def initialize_user(self, initial_assessment: Dict[str, Any], 
                       calibration_data: Optional[List[Dict[str, Any]]] = None):
        """Initialize user profile and calibrate model."""
        # Build user profile
        self.personalization_engine.build_user_profile(initial_assessment)
        
        # Personalize base parameters
        self.current_parameters = self.personalization_engine.personalize_parameters(
            self.base_parameters)
        
        # Calibrate with initial data if available
        if calibration_data:
            for data_point in calibration_data:
                self.update_parameters(data_point)
        
        self.is_initialized = True
        logger.info(f"User {self.user_id} initialized and calibrated")
    
    def update_parameters(self, observation: Dict[str, Any]):
        """Update model parameters based on new observation."""
        if not self.adaptation_enabled:
            return
        
        self.observation_count += 1
        
        # Extract performance metrics if available
        if 'performance' in observation:
            perf_data = observation['performance']
            performance_metrics = PerformanceMetrics(
                task_completion_time=perf_data.get('completion_time', 0.0),
                path_efficiency=perf_data.get('path_efficiency', 0.5),
                movement_smoothness=perf_data.get('smoothness', 0.5),
                precision_error=perf_data.get('precision_error', 0.5),
                force_consistency=perf_data.get('force_consistency', 0.5),
                success_rate=perf_data.get('success_rate', 0.5),
                cognitive_load=perf_data.get('cognitive_load', 0.5),
                adaptation_speed=perf_data.get('adaptation_speed', 0.5),
                timestamp=observation.get('timestamp', 0.0)
            )
            
            # Skill assessment
            skill_scores = self.skill_assessor.assess_performance(performance_metrics)
            
            # Update skill parameters
            self.current_parameters.motor_skill = skill_scores['motor_skill']
            self.current_parameters.task_familiarity = skill_scores['task_familiarity']
            self.current_parameters.spatial_awareness = skill_scores['spatial_awareness']
            
            # Personalized adaptation
            if self.observation_count % self.update_frequency == 0:
                adaptation_factors = self.personalization_engine.adapt_assistance_strategy(
                    performance_metrics, self.current_parameters)
                
                # Apply adaptations to parameters
                self.current_parameters.assistance_tolerance = adaptation_factors.get(
                    'assistance_level', self.current_parameters.assistance_tolerance)
        
        # Bayesian parameter update
        likelihood_params = observation.get('likelihood_params', {})
        self.parameter_estimator.update_bayesian(observation, likelihood_params)
        
        # Update current parameters from estimator
        self.current_parameters = self.parameter_estimator.parameters
        
        # Feature vector update for RLS (if measurement available)
        if 'feature_vector' in observation and 'measurement' in observation:
            feature_vec = observation['feature_vector']
            measurement = observation['measurement']
            self.parameter_estimator.update_rls(feature_vec, measurement)
    
    def predict_behavior(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict human behavior given context."""
        predictions = {}
        
        # Predict reaction time
        base_rt = self.current_parameters.reaction_time
        complexity_factor = context.get('task_complexity', 1.0)
        fatigue_factor = context.get('fatigue_level', 0.0)
        
        predicted_rt = base_rt * complexity_factor * (1 + fatigue_factor)
        predictions['reaction_time'] = predicted_rt
        
        # Predict force output
        max_forces = {}
        for muscle, strength in self.current_parameters.muscle_strength.items():
            base_force = strength * 100  # Assume 100N baseline
            fatigue = context.get('current_fatigue', {}).get(muscle, 0.0)
            fatigue_rate = self.current_parameters.muscle_fatigue_rates[muscle]
            
            current_force = base_force * np.exp(-fatigue_rate * fatigue)
            max_forces[muscle] = current_force
        
        predictions['max_muscle_forces'] = max_forces
        
        # Predict assistance need
        task_difficulty = context.get('task_difficulty', 0.5)
        user_confidence = 1.0 - context.get('uncertainty_level', 0.3)
        
        assistance_need = (task_difficulty * (1 - self.current_parameters.motor_skill) * 
                         (1 - user_confidence))
        assistance_need *= (1 - self.current_parameters.assistance_tolerance)
        
        predictions['assistance_need'] = min(1.0, max(0.0, assistance_need))
        
        # Predict performance
        future_performance = self.skill_assessor.predict_performance()
        predictions['expected_performance'] = future_performance
        
        return predictions
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and convergence."""
        status = {
            'user_id': self.user_id,
            'is_initialized': self.is_initialized,
            'observation_count': self.observation_count,
            'parameters_converged': self.parameter_estimator.is_converged(),
            'parameter_uncertainty': self.parameter_estimator.get_parameter_uncertainty(),
            'current_skill_levels': {
                skill: self.skill_assessor.get_skill_level(skill) 
                for skill in ['motor_skill', 'task_familiarity', 'spatial_awareness']
            },
            'learning_rates': {
                skill: self.skill_assessor.get_learning_rate(skill) 
                for skill in ['motor_skill', 'task_familiarity', 'spatial_awareness']
            },
            'personalization_summary': self.personalization_engine.get_personalization_summary()
        }
        
        return status
    
    def save_model_state(self) -> Dict[str, Any]:
        """Save complete model state for persistence."""
        return {
            'user_id': self.user_id,
            'current_parameters': self.current_parameters.__dict__,
            'parameter_history': self.parameter_estimator.parameter_history[-100:],  # Last 100
            'skill_trends': self.skill_assessor.skill_trends,
            'user_profile': self.personalization_engine.user_profile,
            'personal_preferences': self.personalization_engine.personal_preferences,
            'adaptation_history': self.personalization_engine.adaptation_history[-50:],  # Last 50
            'observation_count': self.observation_count
        }
    
    def load_model_state(self, state: Dict[str, Any]):
        """Load model state from saved data."""
        self.user_id = state['user_id']
        self.observation_count = state.get('observation_count', 0)
        
        # Load parameters
        params_dict = state['current_parameters']
        for key, value in params_dict.items():
            if hasattr(self.current_parameters, key):
                setattr(self.current_parameters, key, value)
        
        # Load skill trends
        self.skill_assessor.skill_trends = state.get('skill_trends', {})
        
        # Load personalization data
        self.personalization_engine.user_profile = state.get('user_profile', {})
        self.personalization_engine.personal_preferences = state.get('personal_preferences', {})
        self.personalization_engine.adaptation_history = state.get('adaptation_history', [])
        
        self.is_initialized = True
        logger.info(f"Model state loaded for user {self.user_id}")


# Example usage and testing
if __name__ == "__main__":
    # Test the adaptive human model
    print("Testing Adaptive Human Model...")
    
    # Create model for test user
    user_model = AdaptiveHumanModel("test_user_001")
    
    # Initialize with mock assessment
    initial_assessment = {
        'demographics': {'age': 25, 'gender': 'M'},
        'physical': {'strength_percentile': 60, 'flexibility': 0.7},
        'cognitive': {'decision_style': 'analytical'},
        'experience': {'robotics_experience': 0.3, 'task_specific': 0.2}
    }
    
    user_model.initialize_user(initial_assessment)
    
    # Simulate learning over time
    print("\nSimulating learning progression...")
    
    for trial in range(20):
        # Simulate performance improvement over time
        base_performance = 0.3 + 0.5 * (1 - np.exp(-trial * 0.2))  # Learning curve
        noise = np.random.normal(0, 0.1)
        
        observation = {
            'timestamp': trial * 30.0,  # 30 seconds per trial
            'performance': {
                'completion_time': 10.0 - base_performance * 5.0 + noise,
                'path_efficiency': base_performance + noise * 0.5,
                'smoothness': base_performance + noise * 0.3,
                'precision_error': (1 - base_performance) * 0.5 + abs(noise) * 0.2,
                'force_consistency': base_performance + noise * 0.2,
                'success_rate': min(1.0, base_performance + 0.1),
                'cognitive_load': (1 - base_performance) * 0.7,
                'adaptation_speed': base_performance
            },
            'muscle_activation': {
                'biceps': 0.3 + np.random.normal(0, 0.1),
                'triceps': 0.25 + np.random.normal(0, 0.1)
            },
            'reaction_time': 0.2 + np.random.normal(0, 0.05)
        }
        
        user_model.update_parameters(observation)
        
        if trial % 5 == 0:
            status = user_model.get_adaptation_status()
            print(f"Trial {trial}: Motor Skill = {status['current_skill_levels']['motor_skill']}, "
                  f"Converged = {status['parameters_converged']}")
    
    # Test behavior prediction
    print("\nTesting behavior prediction...")
    context = {
        'task_complexity': 1.2,
        'task_difficulty': 0.7,
        'fatigue_level': 0.3,
        'uncertainty_level': 0.4
    }
    
    predictions = user_model.predict_behavior(context)
    print(f"Predicted reaction time: {predictions['reaction_time']:.3f}s")
    print(f"Predicted assistance need: {predictions['assistance_need']:.3f}")
    print(f"Expected motor skill improvement: {predictions['expected_performance']['motor_skill']:.3f}")
    
    # Show final status
    final_status = user_model.get_adaptation_status()
    print(f"\nFinal Status:")
    print(f"Observations processed: {final_status['observation_count']}")
    print(f"Parameters converged: {final_status['parameters_converged']}")
    print(f"Skill levels: {final_status['current_skill_levels']}")
    
    print("\nAdaptive Human Model implementation completed!")