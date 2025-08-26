"""
Intent Recognition System for Human-Robot Shared Control.

This module implements multi-modal intent recognition using EMG, force, kinematics,
and eye-tracking data with probabilistic methods and uncertainty quantification.

Mathematical Framework:
- Intent estimation: P(intent|observations) using Bayesian inference
- Temporal dynamics: Hidden Markov Models for intent sequences
- Multi-modal fusion: Weighted likelihood combination
- Uncertainty quantification: Entropy-based confidence measures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Enumeration of possible human intents in shared control."""
    PRECISION_TASK = "precision_task"
    POWER_TASK = "power_task"
    REST = "rest"
    CORRECTION = "correction"
    EXPLORATION = "exploration"
    ASSISTANCE_REQUEST = "assistance_request"
    OVERRIDE_REQUEST = "override_request"


@dataclass
class ObservationData:
    """Multi-modal observation data structure."""
    emg_signals: Dict[str, float] = field(default_factory=dict)
    force_data: np.ndarray = field(default_factory=lambda: np.zeros(6))  # 6-DOF force/torque
    kinematics: Dict[str, float] = field(default_factory=dict)  # position, velocity, acceleration
    eye_tracking: Dict[str, float] = field(default_factory=dict)  # gaze position, pupil diameter
    timestamp: float = 0.0
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert observation to feature vector for ML processing."""
        features = []
        
        # EMG features (8 muscles)
        emg_muscles = ['biceps', 'triceps', 'deltoid_anterior', 'deltoid_posterior',
                      'pectoralis', 'latissimus', 'forearm_flexor', 'forearm_extensor']
        for muscle in emg_muscles:
            features.append(self.emg_signals.get(muscle, 0.0))
        
        # Force/torque features (6-DOF)
        features.extend(self.force_data.flatten())
        
        # Kinematic features
        kin_features = ['position_x', 'position_y', 'position_z', 
                       'velocity_x', 'velocity_y', 'velocity_z',
                       'acceleration_x', 'acceleration_y', 'acceleration_z']
        for feature in kin_features:
            features.append(self.kinematics.get(feature, 0.0))
        
        # Eye-tracking features
        eye_features = ['gaze_x', 'gaze_y', 'pupil_diameter', 'fixation_duration']
        for feature in eye_features:
            features.append(self.eye_tracking.get(feature, 0.0))
        
        return np.array(features)


@dataclass
class IntentPosterior:
    """Posterior distribution over intents."""
    probabilities: Dict[IntentType, float] = field(default_factory=dict)
    confidence: float = 0.0
    entropy: float = 0.0
    most_likely: IntentType = IntentType.REST
    
    def update_metrics(self):
        """Update confidence and entropy metrics."""
        if not self.probabilities:
            return
        
        probs = np.array(list(self.probabilities.values()))
        
        # Confidence as maximum probability
        self.confidence = np.max(probs)
        
        # Entropy as uncertainty measure
        # H = -Σ p_i * log(p_i)
        probs_safe = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)
        self.entropy = -np.sum(probs_safe * np.log(probs_safe))
        
        # Most likely intent
        max_idx = np.argmax(probs)
        self.most_likely = list(self.probabilities.keys())[max_idx]


class MultiModalFusion:
    """Multi-modal sensor fusion for intent recognition."""
    
    def __init__(self, modality_weights: Optional[Dict[str, float]] = None):
        self.modality_weights = modality_weights or {
            'emg': 0.3,
            'force': 0.3,
            'kinematics': 0.25,
            'eye_tracking': 0.15
        }
        
        # Normalize weights
        total_weight = sum(self.modality_weights.values())
        self.modality_weights = {k: v/total_weight for k, v in self.modality_weights.items()}
        
        # Gaussian Mixture Models for each modality
        self.modality_models = {}
        self.feature_scalers = {}
        self.is_trained = False
        
    def train_modality_models(self, training_data: List[Tuple[ObservationData, IntentType]]):
        """Train Gaussian Mixture Models for each modality."""
        logger.info("Training multi-modal intent recognition models...")
        
        # Organize data by modality and intent
        modality_data = {
            'emg': {intent: [] for intent in IntentType},
            'force': {intent: [] for intent in IntentType},
            'kinematics': {intent: [] for intent in IntentType},
            'eye_tracking': {intent: [] for intent in IntentType}
        }
        
        for obs, intent in training_data:
            # EMG features
            emg_features = [obs.emg_signals.get(muscle, 0.0) for muscle in 
                           ['biceps', 'triceps', 'deltoid_anterior', 'deltoid_posterior']]
            modality_data['emg'][intent].append(emg_features)
            
            # Force features
            force_features = obs.force_data.flatten()
            modality_data['force'][intent].append(force_features)
            
            # Kinematic features
            kin_features = [obs.kinematics.get(f, 0.0) for f in 
                           ['velocity_x', 'velocity_y', 'velocity_z']]
            modality_data['kinematics'][intent].append(kin_features)
            
            # Eye-tracking features
            eye_features = [obs.eye_tracking.get(f, 0.0) for f in 
                           ['gaze_x', 'gaze_y', 'pupil_diameter']]
            modality_data['eye_tracking'][intent].append(eye_features)
        
        # Train GMM for each modality
        for modality in self.modality_weights.keys():
            self.modality_models[modality] = {}
            self.feature_scalers[modality] = StandardScaler()
            
            # Combine all data for scaling
            all_data = []
            for intent in IntentType:
                if modality_data[modality][intent]:
                    all_data.extend(modality_data[modality][intent])
            
            if all_data:
                all_data = np.array(all_data)
                self.feature_scalers[modality].fit(all_data)
                
                # Train GMM for each intent
                for intent in IntentType:
                    if modality_data[modality][intent]:
                        intent_data = np.array(modality_data[modality][intent])
                        scaled_data = self.feature_scalers[modality].transform(intent_data)
                        
                        # Fit Gaussian Mixture Model
                        gmm = GaussianMixture(n_components=min(3, len(intent_data)), 
                                            covariance_type='full', random_state=42)
                        gmm.fit(scaled_data)
                        self.modality_models[modality][intent] = gmm
        
        self.is_trained = True
        logger.info("Multi-modal models trained successfully")
    
    def compute_likelihood(self, observation: ObservationData) -> Dict[IntentType, Dict[str, float]]:
        """Compute likelihood P(observation|intent) for each modality."""
        if not self.is_trained:
            raise ValueError("Models must be trained before computing likelihoods")
        
        likelihoods = {intent: {} for intent in IntentType}
        
        # EMG likelihood
        emg_features = np.array([[observation.emg_signals.get(muscle, 0.0) for muscle in 
                                ['biceps', 'triceps', 'deltoid_anterior', 'deltoid_posterior']]])
        if 'emg' in self.feature_scalers:
            emg_scaled = self.feature_scalers['emg'].transform(emg_features)
            for intent in IntentType:
                if intent in self.modality_models['emg']:
                    likelihoods[intent]['emg'] = np.exp(
                        self.modality_models['emg'][intent].score(emg_scaled)
                    )
        
        # Force likelihood
        force_features = observation.force_data.reshape(1, -1)
        if 'force' in self.feature_scalers:
            force_scaled = self.feature_scalers['force'].transform(force_features)
            for intent in IntentType:
                if intent in self.modality_models['force']:
                    likelihoods[intent]['force'] = np.exp(
                        self.modality_models['force'][intent].score(force_scaled)
                    )
        
        # Kinematic likelihood
        kin_features = np.array([[observation.kinematics.get(f, 0.0) for f in 
                                ['velocity_x', 'velocity_y', 'velocity_z']]])
        if 'kinematics' in self.feature_scalers:
            kin_scaled = self.feature_scalers['kinematics'].transform(kin_features)
            for intent in IntentType:
                if intent in self.modality_models['kinematics']:
                    likelihoods[intent]['kinematics'] = np.exp(
                        self.modality_models['kinematics'][intent].score(kin_scaled)
                    )
        
        # Eye-tracking likelihood
        eye_features = np.array([[observation.eye_tracking.get(f, 0.0) for f in 
                                ['gaze_x', 'gaze_y', 'pupil_diameter']]])
        if 'eye_tracking' in self.feature_scalers:
            eye_scaled = self.feature_scalers['eye_tracking'].transform(eye_features)
            for intent in IntentType:
                if intent in self.modality_models['eye_tracking']:
                    likelihoods[intent]['eye_tracking'] = np.exp(
                        self.modality_models['eye_tracking'][intent].score(eye_scaled)
                    )
        
        return likelihoods
    
    def fuse_modalities(self, modality_likelihoods: Dict[IntentType, Dict[str, float]]) -> Dict[IntentType, float]:
        """Fuse multi-modal likelihoods using weighted combination."""
        fused_likelihoods = {intent: 0.0 for intent in IntentType}
        
        for intent in IntentType:
            for modality, weight in self.modality_weights.items():
                if modality in modality_likelihoods[intent]:
                    fused_likelihoods[intent] += weight * modality_likelihoods[intent][modality]
        
        # Normalize to ensure valid probabilities
        total_likelihood = sum(fused_likelihoods.values())
        if total_likelihood > 0:
            fused_likelihoods = {intent: likelihood/total_likelihood 
                               for intent, likelihood in fused_likelihoods.items()}
        
        return fused_likelihoods


class HiddenMarkovIntentModel:
    """Hidden Markov Model for temporal intent dynamics."""
    
    def __init__(self, intents: List[IntentType]):
        self.intents = intents
        self.n_states = len(intents)
        
        # Initialize transition matrix (A[i,j] = P(intent_t = j | intent_{t-1} = i))
        # Default: slight preference for staying in same state
        self.transition_matrix = np.ones((self.n_states, self.n_states)) * 0.1
        np.fill_diagonal(self.transition_matrix, 0.7)
        
        # Normalize rows
        for i in range(self.n_states):
            self.transition_matrix[i] /= np.sum(self.transition_matrix[i])
        
        # Initial state distribution (uniform)
        self.initial_distribution = np.ones(self.n_states) / self.n_states
        
        # Current state belief
        self.current_belief = self.initial_distribution.copy()
        
    def set_transition_probabilities(self, transitions: Dict[Tuple[IntentType, IntentType], float]):
        """Set custom transition probabilities."""
        for (from_intent, to_intent), prob in transitions.items():
            from_idx = self.intents.index(from_intent)
            to_idx = self.intents.index(to_intent)
            self.transition_matrix[from_idx, to_idx] = prob
        
        # Normalize rows
        for i in range(self.n_states):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
    
    def forward_step(self, observation_likelihoods: Dict[IntentType, float]) -> Dict[IntentType, float]:
        """Forward step of HMM inference: P(intent_t | observations_{1:t})."""
        # Convert likelihoods to array
        likelihood_vector = np.array([observation_likelihoods.get(intent, 1e-10) 
                                    for intent in self.intents])
        
        # Prediction step: P(intent_t | observations_{1:t-1})
        predicted_belief = np.dot(self.current_belief, self.transition_matrix)
        
        # Update step: P(intent_t | observations_{1:t})
        updated_belief = predicted_belief * likelihood_vector
        
        # Normalize
        total_belief = np.sum(updated_belief)
        if total_belief > 0:
            updated_belief /= total_belief
        else:
            updated_belief = self.initial_distribution.copy()
        
        self.current_belief = updated_belief
        
        # Convert back to dictionary
        posterior = {intent: updated_belief[i] for i, intent in enumerate(self.intents)}
        return posterior
    
    def reset(self):
        """Reset HMM to initial state."""
        self.current_belief = self.initial_distribution.copy()
    
    def get_most_likely_sequence(self, observations: List[Dict[IntentType, float]], 
                                sequence_length: int = 10) -> List[IntentType]:
        """Viterbi algorithm to find most likely intent sequence."""
        if not observations:
            return []
        
        n_obs = min(len(observations), sequence_length)
        
        # Initialize Viterbi tables
        viterbi_prob = np.zeros((n_obs, self.n_states))
        viterbi_path = np.zeros((n_obs, self.n_states), dtype=int)
        
        # Initialize first time step
        obs_0 = np.array([observations[0].get(intent, 1e-10) for intent in self.intents])
        viterbi_prob[0] = self.initial_distribution * obs_0
        
        # Forward pass
        for t in range(1, n_obs):
            obs_t = np.array([observations[t].get(intent, 1e-10) for intent in self.intents])
            
            for j in range(self.n_states):
                # Find most probable previous state
                trans_probs = viterbi_prob[t-1] * self.transition_matrix[:, j]
                viterbi_path[t, j] = np.argmax(trans_probs)
                viterbi_prob[t, j] = np.max(trans_probs) * obs_t[j]
        
        # Backward pass - find most likely sequence
        sequence = [0] * n_obs
        sequence[-1] = np.argmax(viterbi_prob[-1])
        
        for t in range(n_obs - 2, -1, -1):
            sequence[t] = viterbi_path[t + 1, sequence[t + 1]]
        
        # Convert indices to intents
        return [self.intents[idx] for idx in sequence]


class UncertaintyQuantification:
    """Quantifies uncertainty in intent predictions."""
    
    @staticmethod
    def compute_entropy(probabilities: Dict[IntentType, float]) -> float:
        """Compute Shannon entropy: H = -Σ p_i * log(p_i)."""
        probs = np.array(list(probabilities.values()))
        probs_safe = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs_safe * np.log(probs_safe))
    
    @staticmethod
    def compute_confidence(probabilities: Dict[IntentType, float]) -> float:
        """Compute confidence as maximum probability."""
        return max(probabilities.values()) if probabilities else 0.0
    
    @staticmethod
    def compute_certainty_margin(probabilities: Dict[IntentType, float]) -> float:
        """Compute margin between top two probabilities."""
        sorted_probs = sorted(probabilities.values(), reverse=True)
        if len(sorted_probs) >= 2:
            return sorted_probs[0] - sorted_probs[1]
        return sorted_probs[0] if sorted_probs else 0.0
    
    @staticmethod
    def is_prediction_reliable(probabilities: Dict[IntentType, float], 
                             confidence_threshold: float = 0.6,
                             entropy_threshold: float = 1.0) -> bool:
        """Determine if prediction is reliable based on uncertainty metrics."""
        confidence = UncertaintyQuantification.compute_confidence(probabilities)
        entropy = UncertaintyQuantification.compute_entropy(probabilities)
        
        return confidence > confidence_threshold and entropy < entropy_threshold


class IntentRecognitionSystem:
    """Complete intent recognition system with multi-modal fusion and temporal modeling."""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.intents = list(IntentType)
        self.multimodal_fusion = MultiModalFusion()
        self.hmm_model = HiddenMarkovIntentModel(self.intents)
        self.uncertainty_quantifier = UncertaintyQuantification()
        self.confidence_threshold = confidence_threshold
        
        # History for sequence analysis
        self.observation_history: List[Dict[IntentType, float]] = []
        self.intent_history: List[IntentPosterior] = []
        self.max_history_length = 50
        
        # Performance monitoring
        self.prediction_counts = {intent: 0 for intent in IntentType}
        self.total_predictions = 0
        
        self.is_trained = False
        
    def train(self, training_data: List[Tuple[ObservationData, IntentType]],
              transition_data: Optional[Dict[Tuple[IntentType, IntentType], float]] = None):
        """Train the complete intent recognition system."""
        logger.info("Training Intent Recognition System...")
        
        # Train multi-modal fusion models
        self.multimodal_fusion.train_modality_models(training_data)
        
        # Set custom transition probabilities if provided
        if transition_data:
            self.hmm_model.set_transition_probabilities(transition_data)
        else:
            # Set reasonable defaults based on human behavior
            default_transitions = {
                (IntentType.REST, IntentType.PRECISION_TASK): 0.3,
                (IntentType.REST, IntentType.POWER_TASK): 0.2,
                (IntentType.PRECISION_TASK, IntentType.REST): 0.2,
                (IntentType.PRECISION_TASK, IntentType.CORRECTION): 0.3,
                (IntentType.POWER_TASK, IntentType.REST): 0.3,
                (IntentType.POWER_TASK, IntentType.PRECISION_TASK): 0.2,
                (IntentType.CORRECTION, IntentType.PRECISION_TASK): 0.4,
                (IntentType.EXPLORATION, IntentType.PRECISION_TASK): 0.3,
                (IntentType.ASSISTANCE_REQUEST, IntentType.REST): 0.5,
                (IntentType.OVERRIDE_REQUEST, IntentType.POWER_TASK): 0.4
            }
            self.hmm_model.set_transition_probabilities(default_transitions)
        
        self.is_trained = True
        logger.info("Intent Recognition System training completed")
    
    def predict_intent(self, observation: ObservationData) -> IntentPosterior:
        """Predict human intent from multi-modal observation."""
        if not self.is_trained:
            raise ValueError("System must be trained before making predictions")
        
        # Compute multi-modal likelihoods
        modality_likelihoods = self.multimodal_fusion.compute_likelihood(observation)
        
        # Fuse modalities
        fused_likelihoods = self.multimodal_fusion.fuse_modalities(modality_likelihoods)
        
        # Apply temporal modeling with HMM
        temporal_posterior = self.hmm_model.forward_step(fused_likelihoods)
        
        # Create intent posterior
        posterior = IntentPosterior(probabilities=temporal_posterior)
        posterior.update_metrics()
        
        # Update history
        self.observation_history.append(fused_likelihoods)
        self.intent_history.append(posterior)
        
        # Maintain history length
        if len(self.observation_history) > self.max_history_length:
            self.observation_history.pop(0)
        if len(self.intent_history) > self.max_history_length:
            self.intent_history.pop(0)
        
        # Update statistics
        self.prediction_counts[posterior.most_likely] += 1
        self.total_predictions += 1
        
        return posterior
    
    def get_intent_sequence(self, sequence_length: int = 10) -> List[IntentType]:
        """Get most likely intent sequence using Viterbi algorithm."""
        if len(self.observation_history) < sequence_length:
            sequence_length = len(self.observation_history)
        
        recent_observations = self.observation_history[-sequence_length:]
        return self.hmm_model.get_most_likely_sequence(recent_observations, sequence_length)
    
    def is_intent_stable(self, window_size: int = 5) -> bool:
        """Check if intent predictions are stable over recent window."""
        if len(self.intent_history) < window_size:
            return False
        
        recent_intents = [posterior.most_likely for posterior in self.intent_history[-window_size:]]
        return len(set(recent_intents)) == 1  # All same intent
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.total_predictions == 0:
            return {"total_predictions": 0}
        
        stats = {
            "total_predictions": self.total_predictions,
            "intent_distribution": {intent.value: count/self.total_predictions 
                                  for intent, count in self.prediction_counts.items()},
            "average_confidence": np.mean([p.confidence for p in self.intent_history[-100:]]) 
                                if self.intent_history else 0.0,
            "average_entropy": np.mean([p.entropy for p in self.intent_history[-100:]]) 
                             if self.intent_history else 0.0
        }
        
        return stats
    
    def reset(self):
        """Reset the intent recognition system."""
        self.hmm_model.reset()
        self.observation_history.clear()
        self.intent_history.clear()
        self.prediction_counts = {intent: 0 for intent in IntentType}
        self.total_predictions = 0


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic training data
    def generate_synthetic_data(n_samples: int = 1000) -> List[Tuple[ObservationData, IntentType]]:
        """Generate synthetic multi-modal data for testing."""
        training_data = []
        
        for _ in range(n_samples):
            intent = np.random.choice(list(IntentType))
            
            # Generate intent-dependent features
            if intent == IntentType.PRECISION_TASK:
                emg_base = {'biceps': 0.3, 'triceps': 0.25, 'deltoid_anterior': 0.4}
                force_mag = 20.0
                velocity_mag = 0.5
            elif intent == IntentType.POWER_TASK:
                emg_base = {'biceps': 0.7, 'triceps': 0.6, 'deltoid_anterior': 0.8}
                force_mag = 100.0
                velocity_mag = 2.0
            else:  # REST or other
                emg_base = {'biceps': 0.1, 'triceps': 0.1, 'deltoid_anterior': 0.1}
                force_mag = 5.0
                velocity_mag = 0.1
            
            # Add noise
            emg_signals = {muscle: max(0, val + np.random.normal(0, 0.1)) 
                          for muscle, val in emg_base.items()}
            
            obs = ObservationData(
                emg_signals=emg_signals,
                force_data=np.random.normal(0, force_mag, 6),
                kinematics={
                    'velocity_x': np.random.normal(0, velocity_mag),
                    'velocity_y': np.random.normal(0, velocity_mag),
                    'velocity_z': np.random.normal(0, velocity_mag)
                },
                eye_tracking={
                    'gaze_x': np.random.uniform(-1, 1),
                    'gaze_y': np.random.uniform(-1, 1),
                    'pupil_diameter': np.random.normal(4.0, 0.5)
                }
            )
            
            training_data.append((obs, intent))
        
        return training_data
    
    # Test the system
    print("Testing Intent Recognition System...")
    
    # Create and train system
    intent_system = IntentRecognitionSystem()
    training_data = generate_synthetic_data(500)
    intent_system.train(training_data)
    
    # Test predictions
    test_data = generate_synthetic_data(50)
    
    print("\nTesting predictions...")
    correct_predictions = 0
    
    for obs, true_intent in test_data[:10]:
        predicted_posterior = intent_system.predict_intent(obs)
        predicted_intent = predicted_posterior.most_likely
        
        print(f"True: {true_intent.value}, Predicted: {predicted_intent.value}, "
              f"Confidence: {predicted_posterior.confidence:.3f}, "
              f"Entropy: {predicted_posterior.entropy:.3f}")
        
        if predicted_intent == true_intent:
            correct_predictions += 1
    
    accuracy = correct_predictions / 10
    print(f"\nAccuracy on test sample: {accuracy:.2f}")
    
    # Show system statistics
    stats = intent_system.get_prediction_statistics()
    print(f"\nSystem Statistics:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    print(f"Average entropy: {stats['average_entropy']:.3f}")
    
    print("\nIntent Recognition System implementation completed!")