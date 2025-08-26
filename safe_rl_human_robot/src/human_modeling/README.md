# Advanced Human Behavior Modeling System

A comprehensive human behavior modeling system for Safe Reinforcement Learning in Human-Robot Shared Control applications. This system provides realistic and human-centric modeling capabilities with real-time processing, privacy compliance, and extensive validation frameworks.

## 🚀 Features

### Core Modeling Components
- **Biomechanical Human Model**: Hill-type muscle models with EMG integration and fatigue dynamics
- **Intent Recognition System**: Multi-modal intent recognition with Hidden Markov Models and uncertainty quantification  
- **Adaptive Human Model**: Online parameter estimation with personalization and skill assessment
- **Human Variability Modeling**: Population-level simulation and edge case generation for robust testing

### Data Processing & Collection
- **Multi-Modal Data Collection**: Synchronized EMG, force, kinematics, and eye-tracking data acquisition
- **Real-Time Processing Pipeline**: <10ms latency processing with deterministic timing guarantees
- **Data Quality Validation**: Real-time quality assessment and signal validation

### Validation & Compliance
- **Validation Framework**: IRB-compliant experimental protocols with statistical power analysis
- **Privacy & Ethics System**: GDPR compliance with data anonymization and audit trails
- **Comprehensive Testing**: Integration tests and performance benchmarks

## 📋 Requirements

### Core Dependencies
```python
numpy >= 1.19.0
scipy >= 1.7.0
scikit-learn >= 0.24.0
pandas >= 1.3.0
matplotlib >= 3.3.0
```

### Optional Dependencies
```python
# For enhanced performance
numba >= 0.53.0
cupy >= 9.0.0  # GPU acceleration

# For statistical analysis
statsmodels >= 0.12.0
pingouin >= 0.3.0

# For encryption and privacy
cryptography >= 3.4.0

# For system monitoring
psutil >= 5.8.0
```

## 🏗️ Architecture Overview

```
Human Modeling System
├── biomechanics.py           # Biomechanical dynamics and muscle models
├── intent_recognition.py     # Multi-modal intent recognition
├── adaptive_model.py         # Online adaptation and personalization
├── variability.py           # Population modeling and edge cases
├── data_collection.py       # Multi-modal sensor integration
├── realtime_processing.py   # High-performance processing pipeline
├── validation_framework.py  # Human subject study validation
├── privacy_ethics.py        # Privacy compliance and data protection
└── integration_tests.py     # Comprehensive test suite
```

## 🚀 Quick Start

### Basic Usage

```python
from human_modeling import (
    BiomechanicalModel, 
    IntentRecognitionSystem,
    AdaptiveHumanModel,
    MultiModalDataCollector
)

# Initialize biomechanical model
bio_model = BiomechanicalModel()

# Update with EMG signals
emg_signals = {
    'biceps': 0.6,
    'triceps': 0.3,
    'deltoid_anterior': 0.4
}
bio_model.update(dt=0.01, emg_signals=emg_signals)

# Get muscle forces and joint torques
forces = bio_model.get_muscle_forces()
torques = bio_model.get_joint_torques()
```

### Intent Recognition

```python
# Initialize and train intent recognition
intent_system = IntentRecognitionSystem()

# Train with labeled data (EMG, force, kinematics, eye-tracking)
training_data = [(observation, intent_label), ...]
intent_system.train(training_data)

# Predict intent from new observation
observation = ObservationData(
    emg_signals={'biceps': 0.5, 'triceps': 0.3},
    force_data=np.array([10, 5, -20, 0.5, 0.8, 0.3]),
    kinematics={'velocity_x': 1.2, 'velocity_y': 0.8}
)

intent_posterior = intent_system.predict_intent(observation)
print(f"Predicted intent: {intent_posterior.most_likely}")
print(f"Confidence: {intent_posterior.confidence:.3f}")
```

### Real-Time Processing

```python
# Initialize real-time processor
processor = RealTimeProcessor(target_latency_ms=10.0)

# Set up human models
models = {
    'biomechanical': bio_model,
    'intent_recognition': intent_system
}
processor.set_human_models(models)

# Start processing pipeline
processor.start()

# Submit sensor data for processing
sensor_data = {
    'emg_main': emg_signals,
    'force_wrist': force_data,
    'kinematics_hand': kinematic_data
}

task = ProcessingTask(
    task_id="sensor_update_001",
    priority=ProcessingPriority.HIGH,
    data=sensor_data
)

processor.submit_task(task)

# Get processed results
results = processor.get_results(timeout=0.01)  # 10ms timeout
```

### Adaptive Modeling

```python
# Initialize adaptive model for specific user
user_model = AdaptiveHumanModel("user_001")

# Initialize with assessment data
initial_assessment = {
    'demographics': {'age': 25, 'gender': 'M'},
    'physical': {'strength_percentile': 60},
    'cognitive': {'decision_style': 'analytical'}
}
user_model.initialize_user(initial_assessment)

# Update with performance observations
observation = {
    'performance': {
        'completion_time': 18.5,
        'success_rate': 0.85,
        'path_efficiency': 0.78
    },
    'muscle_activation': emg_signals
}
user_model.update_parameters(observation)

# Predict behavior in new context
context = {
    'task_complexity': 1.2,
    'stress_level': 0.3
}
predictions = user_model.predict_behavior(context)
```

## 📊 Data Collection System

### Multi-Modal Sensor Setup

```python
# Configure sensors
sensor_configs = [
    SensorConfiguration(
        sensor_id="emg_main",
        sensor_type="emg",
        sampling_rate=1000.0,
        channels=['biceps', 'triceps', 'deltoid_anterior', 'deltoid_posterior']
    ),
    SensorConfiguration(
        sensor_id="force_wrist",
        sensor_type="force", 
        sampling_rate=1000.0,
        channels=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    ),
    SensorConfiguration(
        sensor_id="kinematics_hand",
        sensor_type="kinematics",
        sampling_rate=120.0,
        channels=['position', 'velocity', 'acceleration', 'orientation']
    )
]

# Initialize data collector
collector = MultiModalDataCollector(sensor_configs)

# Connect and calibrate sensors
collector.connect_all_sensors()
collector.calibrate_sensors(calibration_data)

# Start synchronized data collection
collector.start_collection()

# Register callback for synchronized frames
def process_frame(frame):
    print(f"Frame {frame.frame_id}: {len(frame.readings)} sensors")
    # Process multi-modal data
    
collector.register_frame_callback(process_frame)
```

## 🧪 Validation Framework

### Experimental Study Setup

```python
# Create validation framework
framework = ValidationFramework()

# Define study protocol
protocol = framework.create_study_protocol(
    study_type=StudyType.INTENT_RECOGNITION_VALIDATION,
    custom_params={
        'target_sample_size': 30,
        'effect_size': 0.6,
        'power': 0.8
    }
)

# Recruit participants
participant_data = {
    'participant_id': 'P001',
    'age': 28,
    'gender': 'F',
    'experience_robotics': 3
}
participant = framework.recruit_participant(participant_data)

# Generate experimental session
session_plan = framework.generate_session_plan('P001', protocol.study_id)

# Record experimental trials
for trial_info in session_plan:
    # Conduct experiment and record data
    trial = ExperimentalTrial(
        trial_id=trial_info['trial_id'],
        participant_id='P001',
        condition=trial_info['condition'],
        # ... performance metrics ...
    )
    framework.record_trial(trial)

# Analyze results
analysis_results = framework.analyze_results()
report = framework.generate_report(analysis_results)
print(report)
```

## 🔐 Privacy & Ethics Compliance

### GDPR Compliance Setup

```python
# Initialize privacy manager
privacy_config = {
    'consent_db_path': 'consent_records.db',
    'audit_log_path': 'privacy_audit.log',
    'retention_policies': {
        DataCategory.BIOMETRIC: timedelta(days=2555),  # 7 years
        DataCategory.BEHAVIORAL: timedelta(days=1825)  # 5 years
    }
}
privacy_manager = PrivacyComplianceManager(privacy_config)

# Record participant consent
consent = ConsentRecord(
    participant_id='P001',
    consent_type=ConsentType.BIOMETRIC_DATA,
    granted=True,
    timestamp=datetime.now(),
    consent_version='v1.0'
)
privacy_manager.consent_manager.record_consent(consent)

# Register data processing
processing_id = privacy_manager.register_data_processing(
    participant_id='P001',
    data_categories=[DataCategory.BIOMETRIC, DataCategory.BEHAVIORAL],
    purposes=[ProcessingPurpose.RESEARCH],
    legal_basis='consent'
)

# Handle data subject requests
response = privacy_manager.process_data_subject_request(
    participant_id='P001',
    request_type=DataSubjectRight.ACCESS,
    user_id='researcher_001'
)
```

### Data Anonymization

```python
# Anonymize dataset for publication
dataset = [
    {
        'name': 'John Doe',
        'age': 25,
        'performance_score': 0.85,
        'reaction_time': 0.23
    },
    # ... more records ...
]

# Different anonymization levels
anonymized_basic = privacy_manager.anonymize_dataset(dataset, 'basic')
anonymized_standard = privacy_manager.anonymize_dataset(dataset, 'standard')  
anonymized_high = privacy_manager.anonymize_dataset(dataset, 'high')  # With differential privacy
```

## 🎯 Population Modeling & Edge Cases

### Generate Diverse Population

```python
# Initialize variability model
variability_model = HumanVariabilityModel()

# Generate population dataset
dataset = variability_model.generate_population_dataset(
    n_individuals=1000,
    edge_case_fraction=0.1
)

print(f"Generated {dataset['total_size']} individuals")
print(f"Population clusters: {len(dataset['clusters'])}")
print(f"Edge cases: {len(dataset['edge_cases'])}")

# Sample representative individuals for testing
test_individuals = variability_model.sample_representative_individuals(
    n_samples=20,
    strategy='cluster_based'
)

# Generate comprehensive test battery
test_battery = variability_model.generate_test_battery(
    test_coverage='comprehensive'
)
```

### Individual Behavior Prediction

```python
# Predict behavior for specific individual
individual_profile = dataset['population_sample'][0]
context = {
    'task_difficulty': 0.7,
    'stress_level': 0.3,
    'fatigue_level': 0.2
}

prediction = variability_model.predict_individual_behavior(
    individual_profile, context
)

print(f"Predicted reaction time: {prediction['predicted_reaction_time']:.3f}s")
print(f"Assistance need: {prediction['assistance_need']:.2f}")
print(f"Risk factors: {prediction['risk_assessment']}")
```

## 📈 Performance & Monitoring

### System Performance Metrics

```python
# Get real-time processing metrics
metrics = processor.get_performance_metrics()

print(f"Average latency: {metrics.average_latency*1000:.1f}ms")
print(f"Throughput: {metrics.throughput:.1f} tasks/sec") 
print(f"Deadline miss rate: {metrics.deadline_miss_rate:.2%}")
print(f"CPU usage: {metrics.cpu_usage:.1f}%")
print(f"Queue depths: {metrics.queue_depths}")

# Collection statistics
stats = collector.get_collection_statistics()
print(f"Frames collected: {stats['frames_collected']}")
print(f"Average framerate: {stats['average_framerate']:.1f} Hz")
print(f"Sensor quality: {stats['average_sensor_quality']}")
```

### Model Performance Validation

```python
# Biomechanical model validation
bio_predictions = []
ground_truth = []

for trial in validation_trials:
    predicted_force = bio_model.predict_muscle_force(trial.emg_data)
    bio_predictions.append(predicted_force)
    ground_truth.append(trial.measured_force)

# Calculate validation metrics
correlation = np.corrcoef(bio_predictions, ground_truth)[0,1]
rmse = np.sqrt(np.mean((np.array(bio_predictions) - np.array(ground_truth))**2))
print(f"Biomechanical model - Correlation: {correlation:.3f}, RMSE: {rmse:.3f}")

# Intent recognition validation  
intent_accuracy = intent_system.get_prediction_statistics()
print(f"Intent recognition accuracy: {intent_accuracy['accuracy']:.3f}")
print(f"Average confidence: {intent_accuracy['avg_confidence']:.3f}")
```

## 🧪 Testing

### Run Integration Tests

```bash
# Run full integration test suite
python -m human_modeling.integration_tests

# Run specific test categories
python -m unittest human_modeling.integration_tests.TestBiomechanicalIntegration
python -m unittest human_modeling.integration_tests.TestRealTimeProcessingIntegration
python -m unittest human_modeling.integration_tests.TestEndToEndIntegration
```

### Performance Benchmarks

```python
# Run performance benchmarks
from human_modeling.integration_tests import run_performance_benchmarks

run_performance_benchmarks()
```

Expected output:
```
=== Performance Benchmarks ===
Biomechanical model: 0.045s for 1000 updates (22222.2 Hz)
Intent recognition: 0.123s for 1000 predictions (8130.1 Hz)  
Data anonymization: 0.000150s per record (6666 records/s)
```

## ⚙️ Configuration

### System Configuration

```python
# Real-time processing configuration
processor_config = {
    'target_latency_ms': 10.0,
    'num_worker_threads': 4,
    'memory_pool_size_mb': 100,
    'enable_gpu_acceleration': True
}

# Privacy compliance configuration  
privacy_config = {
    'consent_db_path': 'data/consent_records.db',
    'audit_log_path': 'logs/privacy_audit.log',
    'encryption_enabled': True,
    'retention_policies': {
        DataCategory.PERSONAL_IDENTIFIABLE: timedelta(days=1095),
        DataCategory.BIOMETRIC: timedelta(days=2555),
        DataCategory.BEHAVIORAL: timedelta(days=1825)
    }
}

# Validation framework configuration
validation_config = {
    'irb_approval_required': True,
    'min_sample_size': 20,
    'alpha_level': 0.05,
    'power_requirement': 0.8,
    'effect_size_estimates': {
        'intent_recognition': 0.6,
        'biomechanical_validation': 0.5,
        'adaptation_improvement': 0.7
    }
}
```

## 📚 Mathematical Framework

### Biomechanical Dynamics

The system implements comprehensive musculoskeletal dynamics:

**Hill-type Muscle Model:**
```
F_muscle = F_max * f_l(l_m) * f_v(v_m) * a(t)
```

Where:
- `F_max`: Maximum isometric force
- `f_l(l_m)`: Force-length relationship
- `f_v(v_m)`: Force-velocity relationship  
- `a(t)`: Neural activation from EMG

**Joint Dynamics:**
```
τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + F_muscle
```

**Fatigue Model:**
```
Force_max(t) = Force_max(0) * exp(-fatigue_rate * t)
```

### Intent Recognition

**Bayesian Intent Inference:**
```
P(I_t | O_{1:t}) = Σ P(I_t | I_{t-1}) * P(O_t | I_t) * P(I_{t-1} | O_{1:t-1})
```

**Uncertainty Quantification:**
```
H(I_t) = -Σ P(I_t = i) * log P(I_t = i)
```

### Adaptive Modeling

**Recursive Parameter Estimation:**
```
θ_t = θ_{t-1} + K_t * (y_t - H_t * θ_{t-1})
K_t = P_{t-1} * H_t^T / (R + H_t * P_{t-1} * H_t^T)
```

### Privacy Protection

**Differential Privacy:**
```
f(D) + Lap(Δf/ε)
```

Where `ε` is the privacy budget and `Δf` is the global sensitivity.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`python -m human_modeling.integration_tests`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new functionality
- Update integration tests for system changes
- Ensure privacy compliance for any data handling
- Validate performance benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions, bug reports, or feature requests:

- Create an issue on GitHub
- Contact the development team
- Check the [documentation](docs/) for detailed guides

## 🙏 Acknowledgments

- Built for Safe Reinforcement Learning research
- Incorporates validated biomechanical models
- Designed with human-centered AI principles
- Compliant with research ethics standards

## 📋 Citation

If you use this system in your research, please cite:

```bibtex
@software{human_behavior_modeling,
  title={Advanced Human Behavior Modeling for Safe RL Human-Robot Shared Control},
  author={Safe RL Research Team},
  year={2024},
  url={https://github.com/safe-rl/human-robot-shared-control}
}
```