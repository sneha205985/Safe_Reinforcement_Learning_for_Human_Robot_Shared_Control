"""
Advanced Human Behavior and Intent Models for Shared Control.

This module provides sophisticated models of human behavior, intent recognition,
biomechanical modeling, and adaptation mechanisms for realistic shared control simulation.
"""

import numpy as np
import torch
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import time

from .human_robot_env import HumanModel, HumanInput, EnvironmentState

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of human intent in shared control."""
    REACHING = "reaching"           # Moving toward target
    EXPLORING = "exploring"         # Exploring environment
    RESTING = "resting"            # Minimal movement/rest
    CORRECTING = "correcting"      # Correcting robot action
    AVOIDING = "avoiding"          # Avoiding obstacles
    FOLLOWING = "following"        # Following path/trajectory
    UNCERTAIN = "uncertain"        # Unclear intent


class MotorImpairment(Enum):
    """Types of motor impairments affecting control."""
    NONE = "none"
    TREMOR = "tremor"              # Involuntary oscillations
    SPASTICITY = "spasticity"      # Increased muscle tone
    WEAKNESS = "weakness"          # Reduced strength
    ATAXIA = "ataxia"             # Poor coordination
    HEMIPLEGIA = "hemiplegia"     # One-sided weakness
    FATIGUE = "fatigue"           # Easy exhaustion


@dataclass
class BiomechanicalParameters:
    """Human biomechanical parameters for realistic modeling."""
    # Strength parameters
    max_force_capacity: torch.Tensor = field(default_factory=lambda: torch.ones(7) * 50.0)  # N
    strength_asymmetry: float = 0.0  # Difference between left/right sides
    
    # Range of motion
    joint_rom: torch.Tensor = field(default_factory=lambda: torch.ones(7) * math.pi)  # rad
    rom_limitations: torch.Tensor = field(default_factory=lambda: torch.zeros(7))  # Fraction of ROM lost
    
    # Motor control
    precision: float = 0.9  # Movement precision (0-1)
    reaction_time: float = 0.15  # Reaction time (seconds)
    movement_smoothness: float = 0.8  # Movement smoothness (0-1)
    
    # Fatigue characteristics
    fatigue_rate: float = 0.001  # Rate of fatigue accumulation
    recovery_rate: float = 0.0005  # Rate of fatigue recovery during rest
    fatigue_threshold: float = 0.7  # Fatigue level that significantly impacts performance
    
    # Impairment-specific parameters
    tremor_frequency: float = 5.0  # Hz
    tremor_amplitude: float = 0.02  # Amplitude scaling
    spasticity_level: float = 0.0  # Spasticity level (0-1)
    coordination_deficit: float = 0.0  # Coordination deficit (0-1)


class BiomechanicalModel:
    """Models human biomechanical characteristics and limitations."""
    
    def __init__(self, 
                 parameters: Optional[BiomechanicalParameters] = None,
                 impairment_type: MotorImpairment = MotorImpairment.NONE,
                 severity: float = 0.0):
        """
        Initialize biomechanical model.
        
        Args:
            parameters: Biomechanical parameters
            impairment_type: Type of motor impairment
            severity: Impairment severity (0-1)
        """
        self.parameters = parameters if parameters is not None else BiomechanicalParameters()
        self.impairment_type = impairment_type
        self.severity = severity
        
        # State variables
        self.current_fatigue = 0.0
        self.muscle_activation = torch.zeros(8)  # 8 muscle groups
        self.joint_stiffness = torch.ones(7) * 0.1  # Joint stiffness
        
        # History for temporal modeling
        self.activation_history = deque(maxlen=100)
        self.fatigue_history = deque(maxlen=1000)
        
        # Apply impairment effects to parameters
        self._apply_impairment_effects()
    
    def _apply_impairment_effects(self) -> None:
        """Apply impairment-specific modifications to parameters."""
        if self.impairment_type == MotorImpairment.TREMOR:
            self.parameters.tremor_amplitude *= (1.0 + 2.0 * self.severity)
            self.parameters.precision *= (1.0 - 0.3 * self.severity)
            
        elif self.impairment_type == MotorImpairment.SPASTICITY:
            self.parameters.spasticity_level = 0.2 + 0.6 * self.severity
            self.joint_stiffness *= (1.0 + 3.0 * self.severity)
            self.parameters.rom_limitations += 0.3 * self.severity
            
        elif self.impairment_type == MotorImpairment.WEAKNESS:
            self.parameters.max_force_capacity *= (1.0 - 0.7 * self.severity)
            self.parameters.fatigue_rate *= (1.0 + 4.0 * self.severity)
            
        elif self.impairment_type == MotorImpairment.ATAXIA:
            self.parameters.coordination_deficit = 0.1 + 0.8 * self.severity
            self.parameters.movement_smoothness *= (1.0 - 0.6 * self.severity)
            self.parameters.precision *= (1.0 - 0.5 * self.severity)
            
        elif self.impairment_type == MotorImpairment.HEMIPLEGIA:
            # Affect one side more than the other
            self.parameters.strength_asymmetry = 0.3 + 0.6 * self.severity
            affected_side = slice(0, 4) if torch.rand(1) > 0.5 else slice(3, 7)
            self.parameters.max_force_capacity[affected_side] *= (1.0 - 0.8 * self.severity)
            
        elif self.impairment_type == MotorImpairment.FATIGUE:
            self.parameters.fatigue_rate *= (1.0 + 8.0 * self.severity)
            self.parameters.recovery_rate *= (1.0 - 0.5 * self.severity)
            self.parameters.fatigue_threshold *= (1.0 - 0.4 * self.severity)
    
    def compute_muscle_activation(self, intended_force: torch.Tensor,
                                 joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute muscle activation patterns based on intended force.
        
        Args:
            intended_force: Intended force/torque [7]
            joint_angles: Current joint angles [7]
            
        Returns:
            Muscle activation levels [8] (0-1)
        """
        # Simplified muscle model: map joint torques to muscle activations
        # In reality, this would use detailed musculoskeletal models
        
        # Basic agonist-antagonist pairs for 7-DOF arm
        muscle_mapping = torch.tensor([
            [1, -0.5, 0, 0, 0, 0, 0],      # Shoulder flexors
            [-1, 0.5, 0, 0, 0, 0, 0],      # Shoulder extensors
            [0, 1, -0.3, 0, 0, 0, 0],      # Shoulder abductors
            [0, -1, 0.3, 0, 0, 0, 0],      # Shoulder adductors
            [0, 0, 0, 1, 0, 0, 0],         # Elbow flexors
            [0, 0, 0, -1, 0, 0, 0],        # Elbow extensors
            [0, 0, 0, 0, 1, 1, 1],         # Forearm/wrist flexors
            [0, 0, 0, 0, -1, -1, -1],      # Forearm/wrist extensors
        ])
        
        # Base activation from intended force
        base_activation = torch.abs(torch.mm(muscle_mapping, intended_force.unsqueeze(1))).squeeze()
        base_activation = torch.clamp(base_activation, 0, 1)
        
        # Apply biomechanical constraints
        
        # 1. Strength limitations
        max_activation = self.parameters.max_force_capacity[:8] / 100.0  # Normalize to 0-1
        constrained_activation = torch.min(base_activation, max_activation)
        
        # 2. Fatigue effects
        fatigue_factor = max(0.1, 1.0 - self.current_fatigue / self.parameters.fatigue_threshold)
        constrained_activation *= fatigue_factor
        
        # 3. Spasticity effects (increased co-contraction)
        if self.parameters.spasticity_level > 0:
            # Add co-contraction between antagonist pairs
            for i in range(0, 8, 2):
                cocontraction = self.parameters.spasticity_level * 0.3
                constrained_activation[i] += cocontraction * constrained_activation[i+1]
                constrained_activation[i+1] += cocontraction * constrained_activation[i]
        
        # 4. Coordination deficits (random activation errors)
        if self.parameters.coordination_deficit > 0:
            noise_level = self.parameters.coordination_deficit * 0.2
            coordination_noise = torch.normal(0, noise_level, size=(8,))
            constrained_activation += coordination_noise
        
        # 5. Apply precision limitations
        if self.parameters.precision < 1.0:
            precision_noise = (1.0 - self.parameters.precision) * 0.1
            precision_error = torch.normal(0, precision_noise, size=(8,))
            constrained_activation += precision_error
        
        # Final clamping and normalization
        final_activation = torch.clamp(constrained_activation, 0, 1)
        
        # Update fatigue based on activation
        activation_magnitude = torch.sum(final_activation).item()
        self.current_fatigue += self.parameters.fatigue_rate * activation_magnitude
        
        # Recovery during low activation
        if activation_magnitude < 0.1:
            self.current_fatigue -= self.parameters.recovery_rate
            self.current_fatigue = max(0, self.current_fatigue)
        
        # Store in history
        self.activation_history.append(final_activation.clone())
        self.fatigue_history.append(self.current_fatigue)
        self.muscle_activation = final_activation
        
        return final_activation
    
    def add_tremor(self, base_movement: torch.Tensor, timestamp: float) -> torch.Tensor:
        """Add tremor to base movement."""
        if self.parameters.tremor_amplitude <= 0:
            return base_movement
        
        # Multi-frequency tremor model
        tremor = torch.zeros_like(base_movement)
        
        # Primary tremor frequency
        primary_tremor = self.parameters.tremor_amplitude * torch.sin(
            2 * math.pi * self.parameters.tremor_frequency * timestamp
        )
        
        # Secondary harmonics
        secondary_tremor = 0.3 * self.parameters.tremor_amplitude * torch.sin(
            2 * math.pi * self.parameters.tremor_frequency * 1.7 * timestamp
        )
        
        # Apply to movement
        for i in range(len(base_movement)):
            phase_offset = i * 0.5  # Different phases for different DOF
            tremor[i] = primary_tremor * math.cos(phase_offset) + secondary_tremor * math.sin(phase_offset)
        
        # Scale by severity and current fatigue (tremor often worsens with fatigue)
        fatigue_scaling = 1.0 + 0.5 * (self.current_fatigue / self.parameters.fatigue_threshold)
        tremor *= self.severity * fatigue_scaling
        
        return base_movement + tremor
    
    def get_biomechanical_state(self) -> Dict[str, Any]:
        """Get current biomechanical state information."""
        return {
            'muscle_activation': self.muscle_activation.clone(),
            'fatigue_level': self.current_fatigue,
            'joint_stiffness': self.joint_stiffness.clone(),
            'effective_strength': self.parameters.max_force_capacity * (
                1.0 - 0.3 * self.current_fatigue / self.parameters.fatigue_threshold
            ),
            'movement_quality': {
                'precision': self.parameters.precision * (1.0 - 0.2 * self.current_fatigue),
                'smoothness': self.parameters.movement_smoothness,
                'coordination': 1.0 - self.parameters.coordination_deficit
            },
            'impairment_info': {
                'type': self.impairment_type.value,
                'severity': self.severity
            }
        }


class IntentRecognizer:
    """Recognizes human intent from movement patterns and context."""
    
    def __init__(self, history_length: int = 20):
        """
        Initialize intent recognizer.
        
        Args:
            history_length: Length of movement history to consider
        """
        self.history_length = history_length
        self.movement_history = deque(maxlen=history_length)
        self.intent_history = deque(maxlen=100)
        
        # Intent classification parameters
        self.intent_confidence_threshold = 0.6
        self.velocity_threshold = 0.01  # m/s
        self.directional_consistency_threshold = 0.8
        
    def recognize_intent(self, 
                        current_movement: torch.Tensor,
                        environment_context: Dict[str, Any],
                        timestamp: float) -> Tuple[IntentType, float]:
        """
        Recognize human intent from movement and context.
        
        Args:
            current_movement: Current movement vector
            environment_context: Environmental context information
            timestamp: Current timestamp
            
        Returns:
            Tuple of (intent_type, confidence)
        """
        # Store movement in history
        self.movement_history.append({
            'movement': current_movement.clone(),
            'timestamp': timestamp,
            'context': environment_context.copy()
        })
        
        if len(self.movement_history) < 3:
            return IntentType.UNCERTAIN, 0.0
        
        # Analyze movement patterns
        velocities = torch.stack([entry['movement'] for entry in list(self.movement_history)[-10:]])
        avg_velocity = torch.mean(velocities, dim=0)
        velocity_magnitude = torch.norm(avg_velocity)
        
        # Check for different intent patterns
        intent_scores = {}
        
        # 1. Resting intent (low movement)
        if velocity_magnitude < self.velocity_threshold:
            intent_scores[IntentType.RESTING] = 0.9
        
        # 2. Reaching intent (consistent direction toward target)
        if 'target_position' in environment_context and 'current_position' in environment_context:
            target_direction = environment_context['target_position'] - environment_context['current_position']
            if torch.norm(target_direction) > 1e-6:
                target_direction = target_direction / torch.norm(target_direction)
                
                if velocity_magnitude > self.velocity_threshold:
                    movement_direction = avg_velocity / velocity_magnitude
                    directional_alignment = torch.dot(movement_direction[:2], target_direction[:2]).item()
                    
                    if directional_alignment > self.directional_consistency_threshold:
                        intent_scores[IntentType.REACHING] = min(0.95, directional_alignment)
        
        # 3. Avoiding intent (movement away from obstacles)
        if 'obstacles' in environment_context and 'current_position' in environment_context:
            obstacle_avoidance_score = self._compute_avoidance_intent(
                environment_context['current_position'],
                environment_context['obstacles'],
                avg_velocity
            )
            if obstacle_avoidance_score > 0.5:
                intent_scores[IntentType.AVOIDING] = obstacle_avoidance_score
        
        # 4. Exploring intent (varying directions, moderate speed)
        direction_changes = self._compute_direction_variability(velocities)
        if (velocity_magnitude > 0.02 and 
            velocity_magnitude < 0.2 and 
            direction_changes > 0.5):
            intent_scores[IntentType.EXPLORING] = 0.6 + 0.3 * direction_changes
        
        # 5. Following intent (movement along path)
        if 'planned_path' in environment_context:
            path_following_score = self._compute_path_following_intent(
                environment_context['current_position'],
                environment_context['planned_path'],
                avg_velocity
            )
            if path_following_score > 0.4:
                intent_scores[IntentType.FOLLOWING] = path_following_score
        
        # 6. Correcting intent (movement opposite to recent robot actions)
        if 'recent_robot_action' in environment_context:
            correction_score = self._compute_correction_intent(
                current_movement,
                environment_context['recent_robot_action']
            )
            if correction_score > 0.6:
                intent_scores[IntentType.CORRECTING] = correction_score
        
        # Select most likely intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            intent_type, confidence = best_intent
            
            # Apply confidence threshold
            if confidence > self.intent_confidence_threshold:
                self.intent_history.append((intent_type, confidence, timestamp))
                return intent_type, confidence
        
        # Default to uncertain
        return IntentType.UNCERTAIN, 0.3
    
    def _compute_avoidance_intent(self, position: torch.Tensor, 
                                obstacles: List[Any], 
                                velocity: torch.Tensor) -> float:
        """Compute obstacle avoidance intent score."""
        if not obstacles or torch.norm(velocity) < 1e-6:
            return 0.0
        
        # Find nearest obstacle
        nearest_distance = float('inf')
        nearest_obstacle = None
        
        for obstacle in obstacles:
            if hasattr(obstacle, 'position'):
                distance = torch.norm(position[:2] - obstacle.position[:2])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_obstacle = obstacle
        
        if nearest_obstacle is None or nearest_distance > 2.0:
            return 0.0
        
        # Check if movement is away from obstacle
        to_obstacle = nearest_obstacle.position[:2] - position[:2]
        movement_direction = velocity[:2] / torch.norm(velocity[:2])
        
        # Negative dot product means moving away
        avoidance_alignment = -torch.dot(movement_direction, to_obstacle / torch.norm(to_obstacle))
        
        # Scale by proximity (closer obstacles get higher scores)
        proximity_factor = max(0, 1.0 - nearest_distance / 2.0)
        
        return max(0, avoidance_alignment.item()) * proximity_factor
    
    def _compute_direction_variability(self, velocities: torch.Tensor) -> float:
        """Compute direction variability for exploration detection."""
        if velocities.shape[0] < 3:
            return 0.0
        
        # Normalize velocity vectors
        velocity_norms = torch.norm(velocities, dim=1)
        valid_velocities = velocities[velocity_norms > 1e-6]
        
        if len(valid_velocities) < 2:
            return 0.0
        
        normalized_velocities = valid_velocities / torch.norm(valid_velocities, dim=1, keepdim=True)
        
        # Compute pairwise angular differences
        angular_differences = []
        for i in range(len(normalized_velocities) - 1):
            dot_product = torch.dot(normalized_velocities[i][:2], normalized_velocities[i+1][:2])
            angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            angular_differences.append(angle.item())
        
        # Variability is high when there are significant direction changes
        mean_angular_difference = np.mean(angular_differences)
        return min(1.0, mean_angular_difference / (math.pi / 4))  # Normalize by 45 degrees
    
    def _compute_path_following_intent(self, position: torch.Tensor,
                                     path: List[torch.Tensor],
                                     velocity: torch.Tensor) -> float:
        """Compute path following intent score."""
        if not path or len(path) < 2 or torch.norm(velocity) < 1e-6:
            return 0.0
        
        # Find closest path segment
        min_distance = float('inf')
        best_segment_idx = 0
        
        for i in range(len(path) - 1):
            segment_start = path[i][:2]
            segment_end = path[i+1][:2]
            
            # Distance from point to line segment
            distance = self._point_to_segment_distance(position[:2], segment_start, segment_end)
            if distance < min_distance:
                min_distance = distance
                best_segment_idx = i
        
        if min_distance > 1.0:  # Too far from path
            return 0.0
        
        # Check alignment with path direction
        segment_start = path[best_segment_idx][:2]
        segment_end = path[best_segment_idx + 1][:2]
        path_direction = segment_end - segment_start
        
        if torch.norm(path_direction) < 1e-6:
            return 0.0
        
        path_direction = path_direction / torch.norm(path_direction)
        movement_direction = velocity[:2] / torch.norm(velocity[:2])
        
        alignment = torch.dot(movement_direction, path_direction).item()
        proximity_factor = max(0, 1.0 - min_distance / 1.0)
        
        return max(0, alignment) * proximity_factor
    
    def _compute_correction_intent(self, human_movement: torch.Tensor,
                                 robot_action: torch.Tensor) -> float:
        """Compute correction intent score."""
        if torch.norm(human_movement) < 1e-6 or torch.norm(robot_action) < 1e-6:
            return 0.0
        
        # Check if human movement opposes robot action
        human_normalized = human_movement / torch.norm(human_movement)
        robot_normalized = robot_action / torch.norm(robot_action)
        
        # Negative dot product indicates opposition
        opposition_score = -torch.dot(human_normalized[:2], robot_normalized[:2]).item()
        
        return max(0, opposition_score)
    
    def _point_to_segment_distance(self, point: torch.Tensor, 
                                 segment_start: torch.Tensor, 
                                 segment_end: torch.Tensor) -> float:
        """Compute distance from point to line segment."""
        segment_vector = segment_end - segment_start
        segment_length_squared = torch.sum(segment_vector ** 2)
        
        if segment_length_squared < 1e-12:
            return torch.norm(point - segment_start).item()
        
        # Project point onto line
        t = torch.sum((point - segment_start) * segment_vector) / segment_length_squared
        t = torch.clamp(t, 0, 1)  # Clamp to segment endpoints
        
        projection = segment_start + t * segment_vector
        return torch.norm(point - projection).item()
    
    def get_intent_history(self) -> List[Tuple[IntentType, float, float]]:
        """Get recent intent recognition history."""
        return list(self.intent_history)


class AdvancedHumanModel(HumanModel):
    """
    Advanced human model with biomechanical modeling, intent recognition,
    and sophisticated adaptation mechanisms.
    """
    
    def __init__(self,
                 skill_level: float = 0.8,
                 impairment_type: MotorImpairment = MotorImpairment.NONE,
                 impairment_severity: float = 0.0,
                 biomechanical_params: Optional[BiomechanicalParameters] = None,
                 learning_rate: float = 0.01,
                 adaptation_speed: float = 0.1):
        """
        Initialize advanced human model.
        
        Args:
            skill_level: Base skill level (0-1)
            impairment_type: Type of motor impairment
            impairment_severity: Severity of impairment (0-1)
            biomechanical_params: Biomechanical parameters
            learning_rate: Learning rate for skill adaptation
            adaptation_speed: Speed of adaptation to assistance
        """
        self.skill_level = skill_level
        self.impairment_type = impairment_type
        self.impairment_severity = impairment_severity
        self.learning_rate = learning_rate
        self.adaptation_speed = adaptation_speed
        
        # Initialize biomechanical model
        self.biomechanics = BiomechanicalModel(
            biomechanical_params, impairment_type, impairment_severity
        )
        
        # Initialize intent recognizer
        self.intent_recognizer = IntentRecognizer()
        
        # Adaptation state
        self.experience_level = 0.0
        self.adaptation_state = {
            'trust_in_robot': 0.5,  # Initial trust level
            'preferred_assistance_level': 0.5,
            'learning_progress': 0.0
        }
        
        # Performance tracking
        self.task_success_rate = deque(maxlen=50)
        self.effort_history = deque(maxlen=100)
        
        logger.info(f"AdvancedHumanModel initialized: skill={skill_level}, "
                   f"impairment={impairment_type.value}({impairment_severity})")
    
    def get_intention(self, env_state: EnvironmentState) -> HumanInput:
        """
        Get human intention using advanced modeling.
        
        Args:
            env_state: Current environment state
            
        Returns:
            Human input with intention and metadata
        """
        timestamp = time.time()
        
        # Determine base intention based on task context
        base_intention = self._compute_base_intention(env_state)
        
        # Recognize intent type
        environment_context = self._extract_environment_context(env_state)
        intent_type, intent_confidence = self.intent_recognizer.recognize_intent(
            base_intention, environment_context, timestamp
        )
        
        # Apply biomechanical constraints and impairments
        constrained_intention = self._apply_biomechanical_constraints(
            base_intention, env_state.robot.position, timestamp
        )
        
        # Compute confidence based on multiple factors
        confidence = self._compute_confidence(
            constrained_intention, intent_confidence, env_state
        )
        
        # Add noise and variability
        final_intention = self._add_human_variability(
            constrained_intention, timestamp
        )
        
        # Update internal state
        self._update_internal_state(final_intention, env_state, timestamp)
        
        # Generate additional human state information
        muscle_activity = self.biomechanics.compute_muscle_activation(
            final_intention, env_state.robot.position
        )
        
        # Simulate gaze direction (simplified)
        gaze_direction = self._compute_gaze_direction(env_state)
        
        return HumanInput(
            intention=final_intention,
            confidence=confidence,
            timestamp=timestamp,
            muscle_activity=muscle_activity,
            gaze_direction=gaze_direction
        )
    
    def _compute_base_intention(self, env_state: EnvironmentState) -> torch.Tensor:
        """Compute base intention before applying constraints."""
        # Basic intention toward target
        if hasattr(env_state, 'target_position'):
            current_pos = env_state.robot.end_effector_pose[:3]
            target_pos = env_state.target_position
            
            direction_to_target = target_pos - current_pos
            distance_to_target = torch.norm(direction_to_target)
            
            if distance_to_target > 1e-6:
                # Normalize and scale by skill
                base_direction = direction_to_target / distance_to_target
                
                # Apply skill-based accuracy
                effective_skill = self._compute_effective_skill()
                intention_magnitude = min(1.0, distance_to_target) * effective_skill
                
                base_intention = base_direction * intention_magnitude
            else:
                base_intention = torch.zeros(3)
        else:
            # No clear target - use exploratory movement
            base_intention = torch.normal(0, 0.1, size=(3,))
        
        return base_intention
    
    def _extract_environment_context(self, env_state: EnvironmentState) -> Dict[str, Any]:
        """Extract relevant context for intent recognition."""
        context = {
            'current_position': env_state.robot.end_effector_pose[:3],
            'obstacles': getattr(env_state, 'obstacles', []),
            'assistance_mode': env_state.assistance_mode
        }
        
        if hasattr(env_state, 'target_position'):
            context['target_position'] = env_state.target_position
        
        return context
    
    def _apply_biomechanical_constraints(self, 
                                       intention: torch.Tensor,
                                       joint_positions: torch.Tensor,
                                       timestamp: float) -> torch.Tensor:
        """Apply biomechanical constraints and impairments."""
        # Add tremor if present
        tremor_intention = self.biomechanics.add_tremor(intention, timestamp)
        
        # Apply strength limitations
        biomech_state = self.biomechanics.get_biomechanical_state()
        strength_factor = torch.mean(biomech_state['effective_strength'][:3]) / 50.0  # Normalize
        strength_limited = tremor_intention * min(1.0, strength_factor)
        
        # Apply precision limitations
        precision = biomech_state['movement_quality']['precision']
        if precision < 1.0:
            precision_noise = (1.0 - precision) * 0.05
            noise = torch.normal(0, precision_noise, size=intention.shape)
            strength_limited += noise
        
        # Apply coordination deficits
        coordination = biomech_state['movement_quality']['coordination']
        if coordination < 1.0:
            coordination_error = (1.0 - coordination) * 0.1
            error = torch.normal(0, coordination_error, size=intention.shape)
            strength_limited += error
        
        return strength_limited
    
    def _compute_effective_skill(self) -> float:
        """Compute current effective skill considering all factors."""
        base_skill = self.skill_level
        
        # Experience bonus
        experience_bonus = min(0.2, self.experience_level * 0.1)
        
        # Fatigue penalty
        fatigue_penalty = 0.3 * (self.biomechanics.current_fatigue / 
                               self.biomechanics.parameters.fatigue_threshold)
        
        # Impairment penalty
        impairment_penalty = 0.4 * self.impairment_severity
        
        effective_skill = base_skill + experience_bonus - fatigue_penalty - impairment_penalty
        return max(0.1, min(1.0, effective_skill))
    
    def _compute_confidence(self, 
                          intention: torch.Tensor,
                          intent_confidence: float,
                          env_state: EnvironmentState) -> float:
        """Compute human confidence in their intention."""
        # Base confidence from skill level
        skill_confidence = self._compute_effective_skill()
        
        # Intent recognition confidence
        intent_conf = intent_confidence
        
        # Task complexity factor
        if hasattr(env_state, 'target_position'):
            distance_factor = 1.0 / (1.0 + torch.norm(
                env_state.robot.end_effector_pose[:3] - env_state.target_position
            ).item())
        else:
            distance_factor = 0.5
        
        # Obstacle complexity
        obstacle_factor = 1.0
        if hasattr(env_state, 'obstacles') and env_state.obstacles:
            # More obstacles reduce confidence
            obstacle_factor = max(0.3, 1.0 - len(env_state.obstacles) * 0.1)
        
        # Trust in robot assistance
        trust_factor = 0.5 + 0.5 * self.adaptation_state['trust_in_robot']
        
        # Fatigue effects
        fatigue_factor = max(0.2, 1.0 - 0.5 * (
            self.biomechanics.current_fatigue / 
            self.biomechanics.parameters.fatigue_threshold
        ))
        
        total_confidence = (
            0.3 * skill_confidence +
            0.2 * intent_conf +
            0.2 * distance_factor +
            0.1 * obstacle_factor +
            0.1 * trust_factor +
            0.1 * fatigue_factor
        )
        
        return max(0.0, min(1.0, total_confidence))
    
    def _add_human_variability(self, intention: torch.Tensor, timestamp: float) -> torch.Tensor:
        """Add natural human movement variability."""
        # Base noise level depends on skill and fatigue
        skill_factor = self._compute_effective_skill()
        noise_level = 0.02 * (1.0 - skill_factor)
        
        # Add random noise
        noise = torch.normal(0, noise_level, size=intention.shape)
        
        # Add low-frequency drift (attention variations)
        drift_freq = 0.1  # Hz
        drift_amplitude = 0.01
        drift = drift_amplitude * math.sin(2 * math.pi * drift_freq * timestamp)
        
        return intention + noise + drift
    
    def _compute_gaze_direction(self, env_state: EnvironmentState) -> torch.Tensor:
        """Compute gaze direction based on intent and context."""
        if hasattr(env_state, 'target_position'):
            # Primarily look at target
            gaze_direction = env_state.target_position - env_state.robot.end_effector_pose[:3]
            
            # Add some gaze variability and saccades
            variability = torch.normal(0, 0.1, size=(3,))
            gaze_direction = gaze_direction + variability
            
            # Normalize
            if torch.norm(gaze_direction) > 1e-6:
                gaze_direction = gaze_direction / torch.norm(gaze_direction)
        else:
            # Random gaze direction
            gaze_direction = torch.normal(0, 1, size=(3,))
            gaze_direction = gaze_direction / torch.norm(gaze_direction)
        
        return gaze_direction
    
    def _update_internal_state(self, 
                             intention: torch.Tensor,
                             env_state: EnvironmentState,
                             timestamp: float) -> None:
        """Update internal model state."""
        # Update experience
        self.experience_level += self.learning_rate * 0.001
        
        # Track effort
        effort = torch.norm(intention).item()
        self.effort_history.append(effort)
        
        # Update adaptation state based on recent performance
        # (This would be more sophisticated in a full implementation)
        if len(self.task_success_rate) > 10:
            recent_success = np.mean(list(self.task_success_rate)[-10:])
            if recent_success > 0.8:
                self.adaptation_state['trust_in_robot'] = min(1.0, 
                    self.adaptation_state['trust_in_robot'] + 0.01)
            elif recent_success < 0.3:
                self.adaptation_state['trust_in_robot'] = max(0.0,
                    self.adaptation_state['trust_in_robot'] - 0.02)
    
    def adapt_to_assistance(self, 
                          robot_action: torch.Tensor, 
                          assistance_level: float) -> HumanInput:
        """
        Advanced adaptation to robot assistance.
        
        Args:
            robot_action: Robot's assistance action
            assistance_level: Level of assistance provided
            
        Returns:
            Adapted human input
        """
        # Compute adaptation based on assistance quality and human preferences
        assistance_quality = self._evaluate_assistance_quality(robot_action)
        
        # Update trust based on assistance quality
        trust_update = (assistance_quality - 0.5) * self.adaptation_speed * 0.1
        self.adaptation_state['trust_in_robot'] += trust_update
        self.adaptation_state['trust_in_robot'] = max(0.0, min(1.0, 
            self.adaptation_state['trust_in_robot']))
        
        # Adjust preferred assistance level
        if assistance_quality > 0.7:
            # Good assistance - slightly increase preference
            self.adaptation_state['preferred_assistance_level'] += 0.01
        elif assistance_quality < 0.3:
            # Poor assistance - decrease preference
            self.adaptation_state['preferred_assistance_level'] -= 0.02
        
        self.adaptation_state['preferred_assistance_level'] = max(0.0, min(1.0,
            self.adaptation_state['preferred_assistance_level']))
        
        # Compute adapted intention
        effort_reduction = assistance_level * self.adaptation_state['trust_in_robot']
        adapted_intention = robot_action[:3] * (1.0 - effort_reduction)
        
        # Add learning component
        learning_bonus = self.adaptation_state['learning_progress'] * 0.1
        adapted_confidence = 0.5 + 0.3 * assistance_level + learning_bonus
        
        return HumanInput(
            intention=adapted_intention,
            confidence=min(1.0, adapted_confidence),
            timestamp=time.time()
        )
    
    def _evaluate_assistance_quality(self, robot_action: torch.Tensor) -> float:
        """Evaluate quality of robot assistance."""
        # Simplified quality assessment based on action magnitude and smoothness
        action_magnitude = torch.norm(robot_action)
        
        # Moderate assistance levels are generally preferred
        magnitude_quality = 1.0 - abs(action_magnitude - 0.5)
        
        # Smoothness quality (would require action history in full implementation)
        smoothness_quality = 0.7  # Placeholder
        
        return 0.6 * magnitude_quality + 0.4 * smoothness_quality
    
    def get_human_state(self) -> Dict[str, Any]:
        """Get comprehensive human state information."""
        biomech_state = self.biomechanics.get_biomechanical_state()
        intent_history = self.intent_recognizer.get_intent_history()
        
        return {
            'skill_level': self.skill_level,
            'effective_skill': self._compute_effective_skill(),
            'experience_level': self.experience_level,
            'biomechanical_state': biomech_state,
            'adaptation_state': self.adaptation_state.copy(),
            'intent_history': intent_history[-10:] if intent_history else [],
            'performance_metrics': {
                'recent_success_rate': np.mean(list(self.task_success_rate)[-10:]) if self.task_success_rate else 0.5,
                'average_effort': np.mean(list(self.effort_history)[-20:]) if self.effort_history else 0.0
            },
            'impairment_info': {
                'type': self.impairment_type.value,
                'severity': self.impairment_severity
            }
        }