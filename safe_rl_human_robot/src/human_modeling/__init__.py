"""
Advanced Human Behavior Modeling for Safe RL Human-Robot Shared Control.

This module provides comprehensive human modeling including biomechanical dynamics,
multi-modal intent recognition, adaptive learning, and individual variability
modeling for realistic human-robot interaction.

Components:
- Biomechanics: Musculoskeletal dynamics and fatigue modeling
- IntentRecognition: Multi-modal intent fusion and uncertainty quantification
- AdaptiveModel: Online parameter estimation and personalization
- Variability: Population-level and individual difference modeling
- DataCollection: Multi-modal sensor integration (EMG, force, kinematics, eye-tracking)
- RealTimeProcessing: Online processing pipeline with <10ms latency
- Validation: Human subject studies integration and validation framework
- Privacy: Data anonymization, encryption, and GDPR compliance
"""

from .biomechanics import (
    BiomechanicalModel,
    MuscleModel,
    FatigueModel,
    MusculoskeletalDynamics
)

from .intent_recognition import (
    IntentRecognitionSystem,
    MultiModalFusion,
    HiddenMarkovIntentModel,
    UncertaintyQuantification
)

from .adaptive_model import (
    AdaptiveHumanModel,
    OnlineParameterEstimation,
    SkillAssessment,
    PersonalizationEngine
)

from .variability import (
    HumanVariabilityModel,
    PopulationSimulator,
    IndividualDifferenceModel,
    EdgeCaseGenerator
)

__all__ = [
    'BiomechanicalModel',
    'MuscleModel', 
    'FatigueModel',
    'MusculoskeletalDynamics',
    'IntentRecognitionSystem',
    'MultiModalFusion',
    'HiddenMarkovIntentModel',
    'UncertaintyQuantification',
    'AdaptiveHumanModel',
    'OnlineParameterEstimation',
    'SkillAssessment',
    'PersonalizationEngine',
    'HumanVariabilityModel',
    'PopulationSimulator',
    'IndividualDifferenceModel',
    'EdgeCaseGenerator'
]