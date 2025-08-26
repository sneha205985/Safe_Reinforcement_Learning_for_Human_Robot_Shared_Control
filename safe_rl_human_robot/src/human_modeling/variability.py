"""
Human Variability Modeling for Population-Level Simulation and Edge Case Generation.

This module implements comprehensive human variability modeling including
population-level parameter distributions, individual difference modeling,
and edge case generation for robust testing of human-robot systems.

Mathematical Framework:
- Population distributions: P(θ) with anthropometric and biomechanical parameters
- Individual differences: Gaussian mixture models and clustering
- Edge case sampling: Tail distributions and adversarial scenarios
- Anthropometric scaling: Geometric and mass property relationships
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PopulationParameters:
    """Population-level parameter distributions."""
    
    # Anthropometric distributions (mean, std, min, max)
    height: Tuple[float, float, float, float] = (1.70, 0.12, 1.45, 2.10)  # meters
    weight: Tuple[float, float, float, float] = (70.0, 15.0, 45.0, 120.0)  # kg
    arm_span: Tuple[float, float, float, float] = (1.68, 0.13, 1.40, 2.05)  # meters
    
    # Limb length ratios (relative to height)
    upper_arm_ratio: Tuple[float, float] = (0.186, 0.015)  # (mean, std)
    forearm_ratio: Tuple[float, float] = (0.146, 0.012)
    hand_ratio: Tuple[float, float] = (0.108, 0.008)
    
    # Strength distributions (percentiles)
    grip_strength: Tuple[float, float, float, float] = (35.0, 12.0, 15.0, 70.0)  # kg
    shoulder_strength: Tuple[float, float, float, float] = (45.0, 15.0, 20.0, 80.0)  # kg
    
    # Biomechanical parameters
    muscle_fiber_type_ratio: Tuple[float, float] = (0.5, 0.15)  # Type I fiber ratio
    tendon_stiffness: Tuple[float, float] = (1.0, 0.3)  # relative to baseline
    joint_flexibility: Tuple[float, float] = (1.0, 0.2)  # relative to baseline
    
    # Cognitive and motor parameters
    reaction_time: Tuple[float, float, float, float] = (0.25, 0.08, 0.15, 0.50)  # seconds
    movement_variability: Tuple[float, float] = (1.0, 0.3)  # relative to baseline
    learning_rate: Tuple[float, float] = (0.05, 0.02)
    
    # Age-related parameters
    age_strength_decline: float = 0.008  # per year after 30
    age_reaction_time_increase: float = 0.002  # per year after 20
    age_flexibility_decline: float = 0.005  # per year after 25


@dataclass
class IndividualProfile:
    """Individual human profile with specific characteristics."""
    user_id: str = ""
    
    # Demographics
    age: int = 25
    gender: str = "M"  # M, F, Other
    height: float = 1.70  # meters
    weight: float = 70.0  # kg
    
    # Physical characteristics
    limb_lengths: Dict[str, float] = field(default_factory=lambda: {
        'upper_arm': 0.32, 'forearm': 0.25, 'hand': 0.18
    })
    limb_masses: Dict[str, float] = field(default_factory=lambda: {
        'upper_arm': 2.0, 'forearm': 1.2, 'hand': 0.6
    })
    
    # Strength and capabilities
    muscle_strengths: Dict[str, float] = field(default_factory=lambda: {
        'biceps': 1.0, 'triceps': 1.0, 'deltoid': 1.0, 'forearm': 1.0
    })
    
    # Motor characteristics
    reaction_time: float = 0.25
    movement_smoothness: float = 1.0
    precision_capability: float = 1.0
    endurance: float = 1.0
    
    # Cognitive characteristics
    learning_rate: float = 0.05
    attention_span: float = 1.0
    spatial_ability: float = 1.0
    risk_tolerance: float = 0.5
    
    # Special conditions
    dominant_hand: str = "right"
    vision_correction: bool = False
    motor_impairments: List[str] = field(default_factory=list)
    cognitive_load_sensitivity: float = 1.0
    
    # Variability characteristics
    consistency_level: float = 1.0  # 0-2, higher = more consistent
    adaptability: float = 1.0  # 0-2, higher = more adaptable
    
    def compute_anthropometric_scaling(self) -> Dict[str, float]:
        """Compute anthropometric scaling factors."""
        # Height-based scaling
        height_factor = self.height / 1.70  # Relative to average height
        
        # Gender-based adjustments
        if self.gender == "F":
            strength_factor = 0.75  # Average female strength relative to male
            mass_factor = 0.85
        else:
            strength_factor = 1.0
            mass_factor = 1.0
        
        # Age-based adjustments
        if self.age > 30:
            age_strength_factor = 1.0 - 0.008 * (self.age - 30)
        else:
            age_strength_factor = 1.0
        
        if self.age > 20:
            age_reaction_factor = 1.0 + 0.002 * (self.age - 20)
        else:
            age_reaction_factor = 1.0
        
        scaling_factors = {
            'length_scaling': height_factor,
            'mass_scaling': mass_factor * (height_factor ** 3),
            'strength_scaling': strength_factor * age_strength_factor,
            'reaction_time_scaling': age_reaction_factor,
            'flexibility_scaling': 1.0 - 0.005 * max(0, self.age - 25)
        }
        
        return scaling_factors


class PopulationSimulator:
    """Simulates diverse human population for testing and validation."""
    
    def __init__(self, population_params: Optional[PopulationParameters] = None):
        self.population_params = population_params or PopulationParameters()
        self.generated_profiles: List[IndividualProfile] = []
        
        # Population clusters for individual differences
        self.n_clusters = 5
        self.population_clusters: Optional[GaussianMixture] = None
        
        # Edge case definitions
        self.edge_case_types = [
            'high_strength_low_precision',
            'low_strength_high_precision', 
            'high_variability',
            'slow_reaction_time',
            'fast_reaction_time',
            'extreme_anthropometrics',
            'motor_impairments',
            'cognitive_limitations'
        ]
        
    def generate_population_sample(self, n_individuals: int = 1000, 
                                 include_edge_cases: bool = True,
                                 edge_case_fraction: float = 0.1) -> List[IndividualProfile]:
        """Generate a diverse population sample."""
        logger.info(f"Generating population sample of {n_individuals} individuals...")
        
        profiles = []
        
        # Determine number of edge cases
        if include_edge_cases:
            n_edge_cases = int(n_individuals * edge_case_fraction)
            n_normal = n_individuals - n_edge_cases
        else:
            n_normal = n_individuals
            n_edge_cases = 0
        
        # Generate normal population
        for i in range(n_normal):
            profile = self._generate_individual_profile(f"user_{i:04d}")
            profiles.append(profile)
        
        # Generate edge cases
        for i in range(n_edge_cases):
            edge_type = np.random.choice(self.edge_case_types)
            profile = self._generate_edge_case_profile(f"edge_{i:04d}", edge_type)
            profiles.append(profile)
        
        self.generated_profiles.extend(profiles)
        logger.info(f"Generated {len(profiles)} individual profiles")
        
        return profiles
    
    def _generate_individual_profile(self, user_id: str) -> IndividualProfile:
        """Generate a single individual profile from population distributions."""
        profile = IndividualProfile(user_id=user_id)
        
        # Demographics
        profile.age = max(18, min(80, int(np.random.normal(35, 15))))
        profile.gender = np.random.choice(['M', 'F'], p=[0.5, 0.5])
        
        # Anthropometrics
        height_params = self.population_params.height
        profile.height = np.clip(np.random.normal(height_params[0], height_params[1]),
                               height_params[2], height_params[3])
        
        weight_params = self.population_params.weight
        # Correlate weight with height (BMI distribution)
        bmi = np.clip(np.random.normal(23, 4), 18, 35)
        profile.weight = bmi * (profile.height ** 2)
        
        # Limb lengths (correlated with height)
        upper_arm_params = self.population_params.upper_arm_ratio
        profile.limb_lengths['upper_arm'] = (profile.height * 
            np.clip(np.random.normal(upper_arm_params[0], upper_arm_params[1]), 
                   0.15, 0.22))
        
        forearm_params = self.population_params.forearm_ratio
        profile.limb_lengths['forearm'] = (profile.height * 
            np.clip(np.random.normal(forearm_params[0], forearm_params[1]), 
                   0.12, 0.18))
        
        hand_params = self.population_params.hand_ratio
        profile.limb_lengths['hand'] = (profile.height * 
            np.clip(np.random.normal(hand_params[0], hand_params[1]), 
                   0.09, 0.13))
        
        # Limb masses (allometric scaling with height and weight)
        height_factor = profile.height / 1.70
        weight_factor = profile.weight / 70.0
        
        profile.limb_masses = {
            'upper_arm': 2.0 * weight_factor * (height_factor ** 2),
            'forearm': 1.2 * weight_factor * (height_factor ** 2), 
            'hand': 0.6 * weight_factor * (height_factor ** 2)
        }
        
        # Strength capabilities
        grip_params = self.population_params.grip_strength
        base_grip = np.clip(np.random.normal(grip_params[0], grip_params[1]),
                           grip_params[2], grip_params[3])
        
        # Scale strength by gender and anthropometrics
        if profile.gender == 'F':
            strength_factor = 0.75
        else:
            strength_factor = 1.0
        
        strength_factor *= (profile.weight / 70.0) ** 0.67  # Allometric scaling
        
        profile.muscle_strengths = {
            'biceps': strength_factor * np.clip(np.random.lognormal(0, 0.3), 0.5, 2.0),
            'triceps': strength_factor * np.clip(np.random.lognormal(0, 0.3), 0.5, 2.0),
            'deltoid': strength_factor * np.clip(np.random.lognormal(0, 0.3), 0.5, 2.0),
            'forearm': base_grip / 35.0  # Normalize to grip strength
        }
        
        # Motor characteristics
        rt_params = self.population_params.reaction_time
        profile.reaction_time = np.clip(np.random.normal(rt_params[0], rt_params[1]),
                                      rt_params[2], rt_params[3])
        
        # Age adjustments
        if profile.age > 20:
            profile.reaction_time *= (1 + self.population_params.age_reaction_time_increase * 
                                    (profile.age - 20))
        
        if profile.age > 30:
            strength_decline = (1 - self.population_params.age_strength_decline * 
                              (profile.age - 30))
            for muscle in profile.muscle_strengths:
                profile.muscle_strengths[muscle] *= strength_decline
        
        # Motor skills and cognitive characteristics
        profile.movement_smoothness = np.clip(np.random.normal(1.0, 0.2), 0.5, 1.5)
        profile.precision_capability = np.clip(np.random.normal(1.0, 0.25), 0.3, 1.8)
        profile.endurance = np.clip(np.random.normal(1.0, 0.3), 0.4, 1.8)
        
        profile.learning_rate = np.clip(np.random.normal(0.05, 0.02), 0.01, 0.15)
        profile.attention_span = np.clip(np.random.normal(1.0, 0.3), 0.3, 2.0)
        profile.spatial_ability = np.clip(np.random.normal(1.0, 0.25), 0.4, 1.8)
        profile.risk_tolerance = np.clip(np.random.beta(2, 2), 0.0, 1.0)
        
        # Individual variability characteristics
        profile.consistency_level = np.clip(np.random.gamma(4, 0.25), 0.2, 2.0)
        profile.adaptability = np.clip(np.random.gamma(4, 0.25), 0.2, 2.0)
        profile.cognitive_load_sensitivity = np.clip(np.random.normal(1.0, 0.3), 0.3, 2.0)
        
        # Handedness
        profile.dominant_hand = np.random.choice(['right', 'left'], p=[0.9, 0.1])
        
        return profile
    
    def _generate_edge_case_profile(self, user_id: str, edge_type: str) -> IndividualProfile:
        """Generate edge case profile with specific characteristics."""
        # Start with normal profile
        profile = self._generate_individual_profile(user_id)
        
        if edge_type == 'high_strength_low_precision':
            # Very strong but imprecise
            for muscle in profile.muscle_strengths:
                profile.muscle_strengths[muscle] *= 2.0
            profile.precision_capability *= 0.3
            profile.movement_smoothness *= 0.5
            
        elif edge_type == 'low_strength_high_precision':
            # Weak but very precise
            for muscle in profile.muscle_strengths:
                profile.muscle_strengths[muscle] *= 0.3
            profile.precision_capability *= 2.0
            profile.movement_smoothness *= 1.5
            
        elif edge_type == 'high_variability':
            # Highly inconsistent performance
            profile.consistency_level *= 0.2
            profile.reaction_time *= np.random.uniform(0.5, 2.0)
            profile.cognitive_load_sensitivity *= 2.0
            
        elif edge_type == 'slow_reaction_time':
            # Very slow reactions
            profile.reaction_time *= 3.0
            profile.learning_rate *= 0.5
            profile.adaptability *= 0.6
            
        elif edge_type == 'fast_reaction_time':
            # Very fast reactions but potentially impulsive
            profile.reaction_time *= 0.3
            profile.risk_tolerance *= 1.5
            profile.precision_capability *= 0.8
            
        elif edge_type == 'extreme_anthropometrics':
            # Extreme body dimensions
            if np.random.random() < 0.5:
                # Very tall and heavy
                profile.height *= 1.4
                profile.weight *= 1.8
            else:
                # Very short and light
                profile.height *= 0.7
                profile.weight *= 0.6
            
            # Recalculate derived properties
            height_factor = profile.height / 1.70
            weight_factor = profile.weight / 70.0
            
            for limb in profile.limb_lengths:
                profile.limb_lengths[limb] *= height_factor
            
            for limb in profile.limb_masses:
                profile.limb_masses[limb] *= weight_factor
                
        elif edge_type == 'motor_impairments':
            # Simulated motor impairments
            impairment = np.random.choice(['tremor', 'weakness', 'spasticity', 'ataxia'])
            profile.motor_impairments.append(impairment)
            
            if impairment == 'tremor':
                profile.precision_capability *= 0.4
                profile.movement_smoothness *= 0.3
            elif impairment == 'weakness':
                for muscle in profile.muscle_strengths:
                    profile.muscle_strengths[muscle] *= 0.4
            elif impairment == 'spasticity':
                profile.movement_smoothness *= 0.5
                profile.reaction_time *= 1.5
            elif impairment == 'ataxia':
                profile.precision_capability *= 0.3
                profile.consistency_level *= 0.2
                
        elif edge_type == 'cognitive_limitations':
            # Cognitive processing limitations
            profile.learning_rate *= 0.3
            profile.attention_span *= 0.4
            profile.spatial_ability *= 0.5
            profile.cognitive_load_sensitivity *= 2.5
            profile.adaptability *= 0.3
        
        return profile
    
    def cluster_population(self, profiles: Optional[List[IndividualProfile]] = None) -> Dict[int, List[IndividualProfile]]:
        """Cluster population into groups with similar characteristics."""
        if profiles is None:
            profiles = self.generated_profiles
            
        if not profiles:
            logger.warning("No profiles available for clustering")
            return {}
        
        # Extract features for clustering
        features = []
        for profile in profiles:
            feature_vector = [
                profile.height,
                profile.weight,
                profile.reaction_time,
                profile.movement_smoothness,
                profile.precision_capability,
                profile.endurance,
                profile.learning_rate,
                profile.spatial_ability,
                profile.risk_tolerance,
                profile.consistency_level,
                profile.adaptability,
                np.mean(list(profile.muscle_strengths.values()))
            ]
            features.append(feature_vector)
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Fit Gaussian Mixture Model
        self.population_clusters = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            random_state=42
        )
        
        cluster_labels = self.population_clusters.fit_predict(features_normalized)
        
        # Group profiles by cluster
        clusters = {i: [] for i in range(self.n_clusters)}
        for profile, label in zip(profiles, cluster_labels):
            clusters[label].append(profile)
        
        logger.info(f"Population clustered into {self.n_clusters} groups:")
        for cluster_id, cluster_profiles in clusters.items():
            logger.info(f"  Cluster {cluster_id}: {len(cluster_profiles)} individuals")
        
        return clusters
    
    def get_population_statistics(self, profiles: Optional[List[IndividualProfile]] = None) -> Dict[str, Any]:
        """Compute population statistics."""
        if profiles is None:
            profiles = self.generated_profiles
            
        if not profiles:
            return {}
        
        stats = {}
        
        # Anthropometric statistics
        heights = [p.height for p in profiles]
        weights = [p.weight for p in profiles]
        
        stats['anthropometrics'] = {
            'height': {'mean': np.mean(heights), 'std': np.std(heights),
                      'min': np.min(heights), 'max': np.max(heights)},
            'weight': {'mean': np.mean(weights), 'std': np.std(weights),
                      'min': np.min(weights), 'max': np.max(weights)}
        }
        
        # Performance characteristics
        reaction_times = [p.reaction_time for p in profiles]
        precisions = [p.precision_capability for p in profiles]
        strengths = [np.mean(list(p.muscle_strengths.values())) for p in profiles]
        
        stats['performance'] = {
            'reaction_time': {'mean': np.mean(reaction_times), 'std': np.std(reaction_times)},
            'precision': {'mean': np.mean(precisions), 'std': np.std(precisions)},
            'strength': {'mean': np.mean(strengths), 'std': np.std(strengths)}
        }
        
        # Demographics
        ages = [p.age for p in profiles]
        gender_counts = {'M': 0, 'F': 0}
        hand_counts = {'right': 0, 'left': 0}
        
        for profile in profiles:
            gender_counts[profile.gender] += 1
            hand_counts[profile.dominant_hand] += 1
        
        stats['demographics'] = {
            'age': {'mean': np.mean(ages), 'std': np.std(ages)},
            'gender_distribution': {k: v/len(profiles) for k, v in gender_counts.items()},
            'handedness_distribution': {k: v/len(profiles) for k, v in hand_counts.items()}
        }
        
        # Edge case statistics
        edge_cases = sum(1 for p in profiles if p.motor_impairments or 
                        p.consistency_level < 0.5 or p.adaptability < 0.5)
        stats['edge_cases'] = {
            'count': edge_cases,
            'fraction': edge_cases / len(profiles)
        }
        
        return stats


class IndividualDifferenceModel:
    """Models individual differences using clustering and statistical methods."""
    
    def __init__(self):
        self.difference_dimensions = [
            'physical_capability',
            'cognitive_style',
            'motor_skill',
            'adaptability',
            'risk_preference'
        ]
        
        self.dimension_models: Dict[str, Any] = {}
        self.individual_clusters: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        
    def fit_difference_model(self, profiles: List[IndividualProfile]):
        """Fit statistical models for individual differences."""
        logger.info("Fitting individual difference models...")
        
        # Extract features for each dimension
        features = self._extract_difference_features(profiles)
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features)
        
        # Fit clustering model
        self.individual_clusters = GaussianMixture(
            n_components=6,  # 6 personality clusters
            covariance_type='full',
            random_state=42
        )
        self.individual_clusters.fit(features_normalized)
        
        # Fit dimension-specific models
        for i, dimension in enumerate(self.difference_dimensions):
            dimension_data = features_normalized[:, i*3:(i+1)*3]  # 3 features per dimension
            
            # Fit Gaussian Mixture for each dimension
            gmm = GaussianMixture(n_components=3, random_state=42)
            gmm.fit(dimension_data)
            self.dimension_models[dimension] = gmm
        
        logger.info("Individual difference models fitted successfully")
    
    def _extract_difference_features(self, profiles: List[IndividualProfile]) -> np.ndarray:
        """Extract features representing individual differences."""
        features = []
        
        for profile in profiles:
            # Physical capability features
            physical_features = [
                np.mean(list(profile.muscle_strengths.values())),
                profile.endurance,
                profile.precision_capability
            ]
            
            # Cognitive style features
            cognitive_features = [
                profile.learning_rate,
                profile.attention_span,
                profile.spatial_ability
            ]
            
            # Motor skill features
            motor_features = [
                1.0 / profile.reaction_time,  # Speed
                profile.movement_smoothness,
                profile.consistency_level
            ]
            
            # Adaptability features
            adaptability_features = [
                profile.adaptability,
                1.0 / profile.cognitive_load_sensitivity,  # Inverse for higher = better
                profile.learning_rate
            ]
            
            # Risk preference features
            risk_features = [
                profile.risk_tolerance,
                profile.precision_capability,  # Precision-seeking vs risk-taking
                1.0 - profile.consistency_level  # Variability as risk indicator
            ]
            
            # Combine all features
            individual_features = (physical_features + cognitive_features + 
                                 motor_features + adaptability_features + risk_features)
            features.append(individual_features)
        
        return np.array(features)
    
    def predict_individual_type(self, profile: IndividualProfile) -> Dict[str, Any]:
        """Predict individual type and characteristics."""
        if self.individual_clusters is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        features = self._extract_difference_features([profile])
        features_normalized = self.scaler.transform(features)
        
        # Predict cluster
        cluster_probs = self.individual_clusters.predict_proba(features_normalized)[0]
        cluster_id = np.argmax(cluster_probs)
        
        # Predict dimension scores
        dimension_scores = {}
        for i, dimension in enumerate(self.difference_dimensions):
            dimension_data = features_normalized[:, i*3:(i+1)*3]
            scores = self.dimension_models[dimension].predict_proba(dimension_data)[0]
            dimension_scores[dimension] = {
                'low': scores[0],
                'medium': scores[1],
                'high': scores[2]
            }
        
        return {
            'cluster_id': cluster_id,
            'cluster_confidence': cluster_probs[cluster_id],
            'dimension_scores': dimension_scores,
            'personality_type': self._interpret_cluster(cluster_id)
        }
    
    def _interpret_cluster(self, cluster_id: int) -> str:
        """Interpret cluster ID as personality type."""
        cluster_types = [
            'analytical_perfectionist',
            'adaptive_collaborator', 
            'confident_performer',
            'cautious_learner',
            'intuitive_explorer',
            'steady_operator'
        ]
        
        return cluster_types[cluster_id % len(cluster_types)]


class EdgeCaseGenerator:
    """Generates edge cases and stress test scenarios for robust evaluation."""
    
    def __init__(self):
        self.edge_case_categories = {
            'physical_extremes': ['very_weak', 'very_strong', 'extreme_anthropometrics'],
            'motor_challenges': ['high_tremor', 'slow_reactions', 'inconsistent_performance'],
            'cognitive_challenges': ['low_attention', 'poor_spatial_ability', 'slow_learning'],
            'interaction_challenges': ['high_risk_tolerance', 'assistance_rejection', 'over_reliance'],
            'medical_conditions': ['motor_impairments', 'visual_impairments', 'fatigue_conditions']
        }
        
        self.stress_test_scenarios = [
            'degraded_sensor_input',
            'high_cognitive_load',
            'time_pressure',
            'environmental_disturbances',
            'robot_failures',
            'unexpected_task_changes'
        ]
        
    def generate_edge_case_profiles(self, n_profiles: int = 100, 
                                   categories: Optional[List[str]] = None) -> List[IndividualProfile]:
        """Generate edge case profiles for testing."""
        if categories is None:
            categories = list(self.edge_case_categories.keys())
        
        profiles = []
        
        for i in range(n_profiles):
            category = np.random.choice(categories)
            edge_type = np.random.choice(self.edge_case_categories[category])
            
            profile = self._create_edge_case_profile(f"edge_{category}_{i:03d}", 
                                                   category, edge_type)
            profiles.append(profile)
        
        logger.info(f"Generated {len(profiles)} edge case profiles")
        return profiles
    
    def _create_edge_case_profile(self, user_id: str, category: str, 
                                edge_type: str) -> IndividualProfile:
        """Create specific edge case profile."""
        # Start with baseline profile
        profile = IndividualProfile(user_id=user_id)
        
        if category == 'physical_extremes':
            if edge_type == 'very_weak':
                for muscle in profile.muscle_strengths:
                    profile.muscle_strengths[muscle] = 0.2  # 20% of normal
                profile.endurance = 0.3
                
            elif edge_type == 'very_strong':
                for muscle in profile.muscle_strengths:
                    profile.muscle_strengths[muscle] = 3.0  # 300% of normal
                profile.endurance = 2.0
                
            elif edge_type == 'extreme_anthropometrics':
                if np.random.random() < 0.5:
                    profile.height = 2.0  # Very tall
                    profile.weight = 110.0
                else:
                    profile.height = 1.5  # Very short
                    profile.weight = 50.0
                    
        elif category == 'motor_challenges':
            if edge_type == 'high_tremor':
                profile.movement_smoothness = 0.2
                profile.precision_capability = 0.3
                profile.consistency_level = 0.1
                profile.motor_impairments.append('tremor')
                
            elif edge_type == 'slow_reactions':
                profile.reaction_time = 0.8  # Very slow
                profile.adaptability = 0.3
                
            elif edge_type == 'inconsistent_performance':
                profile.consistency_level = 0.1
                profile.cognitive_load_sensitivity = 3.0
                
        elif category == 'cognitive_challenges':
            if edge_type == 'low_attention':
                profile.attention_span = 0.2
                profile.cognitive_load_sensitivity = 2.5
                
            elif edge_type == 'poor_spatial_ability':
                profile.spatial_ability = 0.2
                profile.precision_capability = 0.4
                
            elif edge_type == 'slow_learning':
                profile.learning_rate = 0.005  # Very slow learning
                profile.adaptability = 0.2
                
        elif category == 'interaction_challenges':
            if edge_type == 'high_risk_tolerance':
                profile.risk_tolerance = 0.95
                profile.precision_capability = 0.6  # Less careful
                
            elif edge_type == 'assistance_rejection':
                profile.risk_tolerance = 0.9
                profile.adaptability = 0.3  # Resistant to change
                
            elif edge_type == 'over_reliance':
                profile.risk_tolerance = 0.1
                profile.learning_rate = 0.01  # Doesn't learn independence
                
        elif category == 'medical_conditions':
            condition = np.random.choice(['stroke_recovery', 'arthritis', 
                                        'muscular_dystrophy', 'parkinson'])
            profile.motor_impairments.append(condition)
            
            if condition == 'stroke_recovery':
                # Hemiparesis simulation
                side = np.random.choice(['left', 'right'])
                profile.muscle_strengths['biceps'] = 0.3 if side == 'left' else 1.0
                profile.muscle_strengths['triceps'] = 0.3 if side == 'left' else 1.0
                profile.movement_smoothness = 0.4
                
            elif condition == 'arthritis':
                profile.movement_smoothness = 0.5
                profile.precision_capability = 0.6
                profile.endurance = 0.4
                
            elif condition == 'parkinson':
                profile.movement_smoothness = 0.3
                profile.reaction_time = 0.6
                profile.consistency_level = 0.2
                profile.motor_impairments.append('tremor')
        
        return profile
    
    def create_stress_test_scenarios(self, base_profiles: List[IndividualProfile],
                                   n_scenarios: int = 50) -> List[Dict[str, Any]]:
        """Create stress test scenarios combining profiles with challenging conditions."""
        scenarios = []
        
        for i in range(n_scenarios):
            base_profile = np.random.choice(base_profiles)
            stress_type = np.random.choice(self.stress_test_scenarios)
            
            scenario = {
                'scenario_id': f"stress_test_{i:03d}",
                'base_profile': base_profile,
                'stress_type': stress_type,
                'conditions': self._generate_stress_conditions(stress_type),
                'expected_challenges': self._predict_challenges(base_profile, stress_type)
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_stress_conditions(self, stress_type: str) -> Dict[str, Any]:
        """Generate specific conditions for stress test."""
        if stress_type == 'degraded_sensor_input':
            return {
                'emg_noise_level': np.random.uniform(0.3, 0.8),
                'force_sensor_dropout': np.random.uniform(0.1, 0.4),
                'kinematic_delay': np.random.uniform(0.05, 0.2)
            }
            
        elif stress_type == 'high_cognitive_load':
            return {
                'secondary_task_load': np.random.uniform(0.5, 0.9),
                'time_pressure': True,
                'distraction_level': np.random.uniform(0.3, 0.7)
            }
            
        elif stress_type == 'time_pressure':
            return {
                'time_limit_factor': np.random.uniform(0.5, 0.8),
                'performance_feedback': True,
                'urgency_cues': True
            }
            
        elif stress_type == 'environmental_disturbances':
            return {
                'vibration_level': np.random.uniform(0.2, 0.6),
                'noise_level': np.random.uniform(0.3, 0.7),
                'lighting_variation': np.random.uniform(0.2, 0.5)
            }
            
        elif stress_type == 'robot_failures':
            return {
                'actuator_degradation': np.random.uniform(0.2, 0.6),
                'communication_delays': np.random.uniform(0.05, 0.15),
                'intermittent_failures': True
            }
            
        else:  # unexpected_task_changes
            return {
                'task_complexity_increase': np.random.uniform(0.3, 0.8),
                'goal_changes': np.random.randint(2, 5),
                'adaptation_time': np.random.uniform(5.0, 15.0)
            }
    
    def _predict_challenges(self, profile: IndividualProfile, 
                          stress_type: str) -> List[str]:
        """Predict likely challenges for profile under stress."""
        challenges = []
        
        if stress_type == 'high_cognitive_load':
            if profile.attention_span < 0.5:
                challenges.append('attention_breakdown')
            if profile.cognitive_load_sensitivity > 1.5:
                challenges.append('performance_degradation')
                
        elif stress_type == 'time_pressure':
            if profile.reaction_time > 0.4:
                challenges.append('insufficient_response_time')
            if profile.precision_capability < 0.5:
                challenges.append('accuracy_loss_under_pressure')
                
        elif stress_type == 'robot_failures':
            if profile.adaptability < 0.5:
                challenges.append('poor_failure_adaptation')
            if profile.risk_tolerance < 0.3:
                challenges.append('excessive_caution')
        
        # Add profile-specific vulnerabilities
        if profile.motor_impairments:
            challenges.append('motor_limitation_amplification')
        if profile.consistency_level < 0.5:
            challenges.append('increased_variability')
        
        return challenges


class HumanVariabilityModel:
    """Main class integrating all variability modeling components."""
    
    def __init__(self, population_params: Optional[PopulationParameters] = None):
        self.population_params = population_params or PopulationParameters()
        self.population_simulator = PopulationSimulator(self.population_params)
        self.difference_model = IndividualDifferenceModel()
        self.edge_case_generator = EdgeCaseGenerator()
        
        self.population_sample: List[IndividualProfile] = []
        self.population_clusters: Dict[int, List[IndividualProfile]] = {}
        self.is_trained = False
        
        logger.info("Human Variability Model initialized")
    
    def generate_population_dataset(self, n_individuals: int = 1000,
                                  edge_case_fraction: float = 0.1) -> Dict[str, Any]:
        """Generate comprehensive population dataset."""
        logger.info(f"Generating population dataset with {n_individuals} individuals...")
        
        # Generate population sample
        self.population_sample = self.population_simulator.generate_population_sample(
            n_individuals, include_edge_cases=True, edge_case_fraction=edge_case_fraction)
        
        # Cluster population
        self.population_clusters = self.population_simulator.cluster_population(
            self.population_sample)
        
        # Fit individual difference models
        self.difference_model.fit_difference_model(self.population_sample)
        
        # Generate additional edge cases
        edge_cases = self.edge_case_generator.generate_edge_case_profiles(
            n_profiles=int(n_individuals * 0.05))  # Additional 5% edge cases
        
        # Get population statistics
        all_profiles = self.population_sample + edge_cases
        statistics = self.population_simulator.get_population_statistics(all_profiles)
        
        self.is_trained = True
        
        dataset = {
            'population_sample': self.population_sample,
            'edge_cases': edge_cases,
            'clusters': self.population_clusters,
            'statistics': statistics,
            'total_size': len(all_profiles)
        }
        
        logger.info(f"Population dataset generated successfully")
        logger.info(f"Total profiles: {dataset['total_size']}")
        logger.info(f"Edge cases: {len(edge_cases)}")
        logger.info(f"Clusters: {len(self.population_clusters)}")
        
        return dataset
    
    def sample_representative_individuals(self, n_samples: int = 10, 
                                        strategy: str = 'cluster_based') -> List[IndividualProfile]:
        """Sample representative individuals for testing."""
        if not self.population_sample:
            raise ValueError("Population must be generated before sampling")
        
        if strategy == 'cluster_based':
            # Sample from each cluster proportionally
            samples = []
            for cluster_id, cluster_profiles in self.population_clusters.items():
                n_cluster_samples = max(1, int(n_samples * len(cluster_profiles) / 
                                              len(self.population_sample)))
                cluster_samples = np.random.choice(cluster_profiles, 
                                                 min(n_cluster_samples, len(cluster_profiles)), 
                                                 replace=False)
                samples.extend(cluster_samples)
            
            return samples[:n_samples]
            
        elif strategy == 'random':
            return list(np.random.choice(self.population_sample, n_samples, replace=False))
            
        elif strategy == 'extreme_cases':
            # Sample from distribution tails
            samples = []
            
            # Extreme heights
            heights = [p.height for p in self.population_sample]
            height_percentiles = np.percentile(heights, [5, 95])
            extreme_height_profiles = [p for p in self.population_sample 
                                     if p.height <= height_percentiles[0] or 
                                        p.height >= height_percentiles[1]]
            
            # Extreme reaction times
            reaction_times = [p.reaction_time for p in self.population_sample]
            rt_percentiles = np.percentile(reaction_times, [5, 95])
            extreme_rt_profiles = [p for p in self.population_sample 
                                 if p.reaction_time <= rt_percentiles[0] or 
                                    p.reaction_time >= rt_percentiles[1]]
            
            # Combine and sample
            extreme_profiles = list(set(extreme_height_profiles + extreme_rt_profiles))
            n_extreme = min(n_samples, len(extreme_profiles))
            
            if extreme_profiles:
                samples.extend(np.random.choice(extreme_profiles, n_extreme, replace=False))
            
            return samples
        
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def predict_individual_behavior(self, profile: IndividualProfile,
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict individual behavior given profile and context."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get individual type prediction
        individual_type = self.difference_model.predict_individual_type(profile)
        
        # Baseline predictions based on profile
        scaling_factors = profile.compute_anthropometric_scaling()
        
        # Predict performance under context
        task_difficulty = context.get('task_difficulty', 0.5)
        stress_level = context.get('stress_level', 0.0)
        
        # Adjust predictions based on individual characteristics
        predicted_reaction_time = (profile.reaction_time * 
                                 scaling_factors['reaction_time_scaling'] *
                                 (1 + stress_level * profile.cognitive_load_sensitivity))
        
        predicted_precision = (profile.precision_capability * 
                             (1 - task_difficulty * 0.3) *
                             (1 - stress_level * 0.5))
        
        predicted_strength = {muscle: strength * scaling_factors['strength_scaling']
                            for muscle, strength in profile.muscle_strengths.items()}
        
        predicted_endurance_factor = (profile.endurance * 
                                    (1 - stress_level * 0.4))
        
        # Learning and adaptation predictions
        predicted_learning_rate = (profile.learning_rate * 
                                 profile.adaptability *
                                 (1 - stress_level * 0.3))
        
        return {
            'individual_type': individual_type,
            'predicted_reaction_time': predicted_reaction_time,
            'predicted_precision': predicted_precision,
            'predicted_strength': predicted_strength,
            'predicted_endurance_factor': predicted_endurance_factor,
            'predicted_learning_rate': predicted_learning_rate,
            'risk_assessment': {
                'high_variability_risk': profile.consistency_level < 0.5,
                'cognitive_overload_risk': profile.cognitive_load_sensitivity > 1.5,
                'adaptation_difficulty': profile.adaptability < 0.5
            }
        }
    
    def generate_test_battery(self, test_coverage: str = 'comprehensive') -> Dict[str, Any]:
        """Generate comprehensive test battery for system evaluation."""
        test_battery = {
            'population_coverage_tests': [],
            'edge_case_tests': [],
            'stress_tests': [],
            'interaction_tests': []
        }
        
        if test_coverage in ['basic', 'comprehensive']:
            # Representative population tests
            representative_profiles = self.sample_representative_individuals(
                n_samples=20, strategy='cluster_based')
            test_battery['population_coverage_tests'] = representative_profiles
            
            # Basic edge cases
            basic_edge_cases = self.edge_case_generator.generate_edge_case_profiles(
                n_profiles=10, categories=['physical_extremes', 'motor_challenges'])
            test_battery['edge_case_tests'] = basic_edge_cases
        
        if test_coverage == 'comprehensive':
            # Comprehensive edge cases
            all_edge_cases = self.edge_case_generator.generate_edge_case_profiles(
                n_profiles=50)
            test_battery['edge_case_tests'].extend(all_edge_cases)
            
            # Stress test scenarios
            stress_scenarios = self.edge_case_generator.create_stress_test_scenarios(
                representative_profiles, n_scenarios=25)
            test_battery['stress_tests'] = stress_scenarios
            
            # Interaction pattern tests
            interaction_profiles = self.sample_representative_individuals(
                n_samples=15, strategy='extreme_cases')
            test_battery['interaction_tests'] = interaction_profiles
        
        return test_battery


# Example usage and testing
if __name__ == "__main__":
    print("Testing Human Variability Model...")
    
    # Create variability model
    variability_model = HumanVariabilityModel()
    
    # Generate population dataset
    dataset = variability_model.generate_population_dataset(n_individuals=500)
    
    print(f"\nPopulation Dataset Summary:")
    print(f"Total individuals: {dataset['total_size']}")
    print(f"Regular profiles: {len(dataset['population_sample'])}")
    print(f"Edge cases: {len(dataset['edge_cases'])}")
    print(f"Population clusters: {len(dataset['clusters'])}")
    
    # Show population statistics
    stats = dataset['statistics']
    print(f"\nAnthropometric Statistics:")
    print(f"Height: {stats['anthropometrics']['height']['mean']:.2f} ± {stats['anthropometrics']['height']['std']:.2f} m")
    print(f"Weight: {stats['anthropometrics']['weight']['mean']:.1f} ± {stats['anthropometrics']['weight']['std']:.1f} kg")
    
    print(f"\nPerformance Statistics:")
    print(f"Reaction time: {stats['performance']['reaction_time']['mean']:.3f} ± {stats['performance']['reaction_time']['std']:.3f} s")
    print(f"Precision capability: {stats['performance']['precision']['mean']:.2f} ± {stats['performance']['precision']['std']:.2f}")
    
    # Sample representative individuals
    representative_sample = variability_model.sample_representative_individuals(
        n_samples=5, strategy='cluster_based')
    
    print(f"\nRepresentative Sample:")
    for i, profile in enumerate(representative_sample):
        print(f"Individual {i+1}: Age {profile.age}, Height {profile.height:.2f}m, "
              f"RT {profile.reaction_time:.3f}s, Precision {profile.precision_capability:.2f}")
    
    # Test behavior prediction
    test_profile = representative_sample[0]
    test_context = {
        'task_difficulty': 0.7,
        'stress_level': 0.3
    }
    
    behavior_prediction = variability_model.predict_individual_behavior(
        test_profile, test_context)
    
    print(f"\nBehavior Prediction for {test_profile.user_id}:")
    print(f"Individual type: {behavior_prediction['individual_type']['personality_type']}")
    print(f"Predicted reaction time: {behavior_prediction['predicted_reaction_time']:.3f}s")
    print(f"Predicted precision: {behavior_prediction['predicted_precision']:.2f}")
    
    # Generate test battery
    test_battery = variability_model.generate_test_battery(test_coverage='comprehensive')
    
    print(f"\nTest Battery Generated:")
    print(f"Population coverage tests: {len(test_battery['population_coverage_tests'])}")
    print(f"Edge case tests: {len(test_battery['edge_case_tests'])}")
    print(f"Stress tests: {len(test_battery['stress_tests'])}")
    print(f"Interaction tests: {len(test_battery['interaction_tests'])}")
    
    print("\nHuman Variability Model implementation completed!")