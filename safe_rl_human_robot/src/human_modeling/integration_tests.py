"""
Integration Tests for Advanced Human Behavior Modeling System.

This module provides comprehensive integration tests for all components of the
human modeling system, including end-to-end workflows, performance validation,
and stress testing.

Test Categories:
- Component integration tests
- End-to-end workflow tests  
- Performance and latency tests
- Data quality and validation tests
- Privacy and ethics compliance tests
- Error handling and edge cases
"""

import unittest
import numpy as np
import time
import threading
import tempfile
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import logging

# Import all human modeling components
from . import biomechanics
from . import intent_recognition
from . import adaptive_model  
from . import variability
from . import data_collection
from . import realtime_processing
from . import validation_framework
from . import privacy_ethics

# Suppress warnings during testing
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing
logger = logging.getLogger(__name__)


class TestBiomechanicalIntegration(unittest.TestCase):
    """Test biomechanical model integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.muscle_model = biomechanics.MuscleModel(
            muscle_type="biceps",
            max_force=300.0,
            optimal_length=0.25,
            pennation_angle=0.0
        )
        
        self.fatigue_model = biomechanics.FatigueModel()
        self.dynamics_model = biomechanics.MusculoskeletalDynamics()
        self.bio_model = biomechanics.BiomechanicalModel()
    
    def test_muscle_force_computation(self):
        """Test muscle force computation with EMG input."""
        muscle_length = 0.25
        muscle_velocity = 0.0
        activation = 0.5
        
        force = self.muscle_model.compute_muscle_force(muscle_length, muscle_velocity, activation)
        
        # Force should be positive and reasonable
        self.assertGreater(force, 0)
        self.assertLess(force, 300.0)  # Should not exceed max force
        
        # Force should scale with activation
        force_low = self.muscle_model.compute_muscle_force(muscle_length, muscle_velocity, 0.1)
        force_high = self.muscle_model.compute_muscle_force(muscle_length, muscle_velocity, 0.9)
        self.assertLess(force_low, force_high)
    
    def test_fatigue_accumulation(self):
        """Test fatigue accumulation over time."""
        initial_force = 100.0
        
        # Simulate continuous activation
        for t in np.linspace(0, 60, 100):  # 1 minute
            self.fatigue_model.update_fatigue(0.8, 0.01)  # High activation
        
        current_capacity = self.fatigue_model.get_current_capacity()
        
        # Capacity should decrease due to fatigue
        self.assertLess(current_capacity, 1.0)
        self.assertGreater(current_capacity, 0.1)  # Should not go to zero
    
    def test_full_biomechanical_integration(self):
        """Test full biomechanical model integration."""
        # Setup EMG data
        emg_signals = {
            'biceps': 0.6,
            'triceps': 0.3,
            'deltoid_anterior': 0.4,
            'deltoid_posterior': 0.2
        }
        
        # Update model
        self.bio_model.update(dt=0.01, emg_signals=emg_signals)
        
        # Check that model state is updated
        muscle_forces = self.bio_model.get_muscle_forces()
        joint_torques = self.bio_model.get_joint_torques()
        
        self.assertIsInstance(muscle_forces, dict)
        self.assertIsInstance(joint_torques, dict)
        self.assertGreater(len(muscle_forces), 0)
        self.assertGreater(len(joint_torques), 0)
    
    def test_biomechanical_prediction_accuracy(self):
        """Test prediction accuracy with synthetic data."""
        # Generate synthetic EMG with known muscle activation
        true_activation = 0.7
        emg_with_noise = true_activation + np.random.normal(0, 0.1)
        
        emg_signals = {'biceps': max(0, min(1, emg_with_noise))}
        
        self.bio_model.update(dt=0.01, emg_signals=emg_signals)
        forces = self.bio_model.get_muscle_forces()
        
        # Predicted force should correlate with true activation
        predicted_activation = forces.get('biceps', 0) / 300.0  # Normalize by max force
        
        # Allow for some error due to noise and model approximations
        self.assertLess(abs(predicted_activation - true_activation), 0.3)


class TestIntentRecognitionIntegration(unittest.TestCase):
    """Test intent recognition system integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.intent_system = intent_recognition.IntentRecognitionSystem()
        
        # Create mock training data
        self.training_data = []
        for _ in range(100):
            intent = np.random.choice(list(intent_recognition.IntentType))
            obs = self._create_mock_observation(intent)
            self.training_data.append((obs, intent))
    
    def _create_mock_observation(self, intent: intent_recognition.IntentType):
        """Create mock observation data for given intent."""
        obs = intent_recognition.ObservationData()
        
        if intent == intent_recognition.IntentType.PRECISION_TASK:
            obs.emg_signals = {'biceps': 0.3, 'triceps': 0.2}
            obs.force_data = np.random.normal(5, 2, 6)
            obs.kinematics = {'velocity_x': 0.1, 'velocity_y': 0.1, 'velocity_z': 0.05}
        elif intent == intent_recognition.IntentType.POWER_TASK:
            obs.emg_signals = {'biceps': 0.8, 'triceps': 0.7}
            obs.force_data = np.random.normal(50, 10, 6)
            obs.kinematics = {'velocity_x': 1.5, 'velocity_y': 1.0, 'velocity_z': 0.8}
        else:  # REST
            obs.emg_signals = {'biceps': 0.1, 'triceps': 0.1}
            obs.force_data = np.random.normal(0, 1, 6)
            obs.kinematics = {'velocity_x': 0.01, 'velocity_y': 0.01, 'velocity_z': 0.01}
        
        return obs
    
    def test_intent_recognition_training(self):
        """Test intent recognition system training."""
        # Training should complete without errors
        self.intent_system.train(self.training_data)
        self.assertTrue(self.intent_system.is_trained)
    
    def test_intent_prediction_accuracy(self):
        """Test intent prediction accuracy."""
        # Train system
        self.intent_system.train(self.training_data)
        
        # Test predictions
        correct_predictions = 0
        total_predictions = 0
        
        test_data = []
        for _ in range(20):
            intent = np.random.choice(list(intent_recognition.IntentType))
            obs = self._create_mock_observation(intent)
            test_data.append((obs, intent))
        
        for obs, true_intent in test_data:
            prediction = self.intent_system.predict_intent(obs)
            predicted_intent = prediction.most_likely
            
            if predicted_intent == true_intent:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        
        # Should achieve reasonable accuracy on synthetic data
        self.assertGreater(accuracy, 0.4)  # At least better than random
    
    def test_temporal_intent_tracking(self):
        """Test temporal intent tracking with HMM."""
        self.intent_system.train(self.training_data)
        
        # Create sequence of observations for same intent
        intent = intent_recognition.IntentType.PRECISION_TASK
        observations = [self._create_mock_observation(intent) for _ in range(5)]
        
        predictions = []
        for obs in observations:
            pred = self.intent_system.predict_intent(obs)
            predictions.append(pred)
        
        # Later predictions should be more confident due to temporal consistency
        confidences = [pred.confidence for pred in predictions]
        
        # Confidence should generally increase or remain stable
        self.assertGreaterEqual(confidences[-1], confidences[0])


class TestAdaptiveModelIntegration(unittest.TestCase):
    """Test adaptive human model integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.user_id = "test_user_001"
        self.adaptive_model = adaptive_model.AdaptiveHumanModel(self.user_id)
        
        # Mock initial assessment
        self.initial_assessment = {
            'demographics': {'age': 25, 'gender': 'M'},
            'physical': {'strength_percentile': 60},
            'cognitive': {'decision_style': 'analytical'}
        }
    
    def test_user_initialization(self):
        """Test user initialization and calibration."""
        self.adaptive_model.initialize_user(self.initial_assessment)
        self.assertTrue(self.adaptive_model.is_initialized)
        
        # Parameters should be personalized
        params = self.adaptive_model.current_parameters
        self.assertIsInstance(params, adaptive_model.HumanParameters)
    
    def test_parameter_adaptation(self):
        """Test online parameter adaptation."""
        self.adaptive_model.initialize_user(self.initial_assessment)
        
        initial_skill = self.adaptive_model.current_parameters.motor_skill
        
        # Simulate good performance over time
        for trial in range(10):
            observation = {
                'timestamp': time.time(),
                'performance': {
                    'completion_time': 15.0 - trial * 0.5,  # Improving times
                    'success_rate': 0.5 + trial * 0.04,     # Improving success
                    'path_efficiency': 0.6 + trial * 0.03,
                    'smoothness': 0.5 + trial * 0.02
                }
            }
            
            self.adaptive_model.update_parameters(observation)
        
        final_skill = self.adaptive_model.current_parameters.motor_skill
        
        # Motor skill should improve with good performance
        self.assertGreater(final_skill, initial_skill)
    
    def test_behavior_prediction(self):
        """Test behavior prediction."""
        self.adaptive_model.initialize_user(self.initial_assessment)
        
        context = {
            'task_complexity': 1.2,
            'task_difficulty': 0.7,
            'fatigue_level': 0.3
        }
        
        predictions = self.adaptive_model.predict_behavior(context)
        
        # Should return reasonable predictions
        self.assertIn('reaction_time', predictions)
        self.assertIn('assistance_need', predictions)
        self.assertIn('expected_performance', predictions)
        
        self.assertGreater(predictions['reaction_time'], 0.1)
        self.assertLess(predictions['reaction_time'], 2.0)
        self.assertGreaterEqual(predictions['assistance_need'], 0.0)
        self.assertLessEqual(predictions['assistance_need'], 1.0)


class TestVariabilityModelIntegration(unittest.TestCase):
    """Test human variability model integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.variability_model = variability.HumanVariabilityModel()
    
    def test_population_generation(self):
        """Test population dataset generation."""
        dataset = self.variability_model.generate_population_dataset(n_individuals=50)
        
        self.assertIn('population_sample', dataset)
        self.assertIn('edge_cases', dataset)
        self.assertIn('statistics', dataset)
        
        # Should have requested number of individuals
        self.assertEqual(len(dataset['population_sample']), 50)
        
        # Should have some edge cases
        self.assertGreater(len(dataset['edge_cases']), 0)
    
    def test_individual_behavior_prediction(self):
        """Test individual behavior prediction."""
        # Generate population first
        dataset = self.variability_model.generate_population_dataset(n_individuals=20)
        
        # Test prediction for a sample individual
        sample_profile = dataset['population_sample'][0]
        context = {'task_difficulty': 0.5, 'stress_level': 0.2}
        
        prediction = self.variability_model.predict_individual_behavior(sample_profile, context)
        
        self.assertIn('predicted_reaction_time', prediction)
        self.assertIn('predicted_precision', prediction)
        self.assertIn('risk_assessment', prediction)
        
        # Values should be reasonable
        self.assertGreater(prediction['predicted_reaction_time'], 0.05)
        self.assertLess(prediction['predicted_reaction_time'], 2.0)
    
    def test_edge_case_generation(self):
        """Test edge case generation."""
        dataset = self.variability_model.generate_population_dataset(n_individuals=30)
        edge_cases = dataset['edge_cases']
        
        # Should have various edge case types
        edge_case_types = set()
        for profile in edge_cases:
            if profile.motor_impairments:
                edge_case_types.add('motor_impairments')
            if profile.consistency_level < 0.5:
                edge_case_types.add('high_variability')
        
        self.assertGreater(len(edge_case_types), 0)


class TestDataCollectionIntegration(unittest.TestCase):
    """Test data collection system integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        # Create sensor configurations
        self.sensor_configs = [
            data_collection.SensorConfiguration(
                sensor_id="emg_test",
                sensor_type="emg",
                sampling_rate=1000.0,
                channels=['biceps', 'triceps']
            ),
            data_collection.SensorConfiguration(
                sensor_id="force_test",
                sensor_type="force",
                sampling_rate=1000.0,
                channels=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
            )
        ]
        
        self.collector = data_collection.MultiModalDataCollector(self.sensor_configs)
    
    def test_sensor_connection(self):
        """Test sensor connection."""
        success = self.collector.connect_all_sensors()
        self.assertTrue(success)
        
        # All sensors should be connected
        for interface in self.collector.sensor_interfaces.values():
            self.assertTrue(interface.is_connected)
    
    def test_data_synchronization(self):
        """Test multi-modal data synchronization."""
        self.collector.connect_all_sensors()
        
        # Create synchronizer
        synchronizer = data_collection.DataSynchronizer(target_framerate=100.0)
        
        collected_frames = []
        
        def frame_callback(frame):
            collected_frames.append(frame)
        
        # Add some mock sensor readings
        for i in range(10):
            timestamp = time.time() + i * 0.01
            
            # EMG reading
            emg_reading = data_collection.SensorReading(
                sensor_id="emg_test",
                timestamp=timestamp,
                data={'biceps': 0.5, 'triceps': 0.3}
            )
            synchronizer.add_sensor_data(emg_reading)
            
            # Force reading
            force_reading = data_collection.SensorReading(
                sensor_id="force_test",
                timestamp=timestamp + 0.001,  # Small time offset
                data=np.array([10, 5, -20, 0.5, 0.8, 0.3])
            )
            synchronizer.add_sensor_data(force_reading)
        
        # Get synchronized frames
        for i in range(5):
            frame = synchronizer.get_synchronized_frame()
            if frame:
                collected_frames.append(frame)
        
        self.assertGreater(len(collected_frames), 0)
        
        # Check frame completeness
        for frame in collected_frames:
            if frame.readings:
                self.assertGreater(frame.synchronization_quality, 0.0)


class TestRealTimeProcessingIntegration(unittest.TestCase):
    """Test real-time processing pipeline integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.processor = realtime_processing.RealTimeProcessor(
            target_latency_ms=10.0,
            num_worker_threads=2
        )
    
    def test_pipeline_startup_shutdown(self):
        """Test pipeline startup and shutdown."""
        # Start processor
        self.processor.start()
        self.assertTrue(self.processor.running)
        
        # Should have worker threads
        self.assertGreater(len(self.processor.worker_threads), 0)
        
        # Stop processor
        self.processor.stop()
        self.assertFalse(self.processor.running)
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing pipeline."""
        self.processor.start()
        
        try:
            results = []
            
            def result_callback(result):
                results.append(result)
            
            self.processor.register_result_callback(result_callback)
            
            # Submit test tasks
            for i in range(5):
                sensor_data = {
                    'emg_test': {'biceps': 0.5, 'triceps': 0.3},
                    'force_test': np.array([10, 5, -20, 0.5, 0.8, 0.3])
                }
                
                task = realtime_processing.ProcessingTask(
                    task_id=f"test_task_{i}",
                    priority=realtime_processing.ProcessingPriority.HIGH,
                    stage=realtime_processing.ProcessingStage.PREPROCESSING,
                    data=sensor_data
                )
                
                self.processor.submit_task(task)
            
            # Wait for processing
            time.sleep(1.0)
            
            # Should have some results
            self.assertGreater(len(results), 0)
            
        finally:
            self.processor.stop()
    
    def test_latency_requirements(self):
        """Test that latency requirements are met."""
        self.processor.start()
        
        try:
            processing_times = []
            
            def timing_callback(result):
                processing_times.append(result.processing_time)
            
            self.processor.register_result_callback(timing_callback)
            
            # Submit tasks and measure latency
            start_time = time.time()
            
            for i in range(10):
                task = realtime_processing.ProcessingTask(
                    task_id=f"latency_test_{i}",
                    priority=realtime_processing.ProcessingPriority.HIGH,
                    stage=realtime_processing.ProcessingStage.PREPROCESSING,
                    data={'test': i}
                )
                
                self.processor.submit_task(task)
            
            time.sleep(0.5)  # Wait for processing
            
            if processing_times:
                avg_latency = np.mean(processing_times)
                max_latency = np.max(processing_times)
                
                # Should meet latency requirements (allowing some margin)
                self.assertLess(avg_latency, 0.020)  # 20ms average
                self.assertLess(max_latency, 0.050)   # 50ms max
        
        finally:
            self.processor.stop()


class TestValidationFrameworkIntegration(unittest.TestCase):
    """Test validation framework integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        self.framework = validation_framework.ValidationFramework()
    
    def test_study_protocol_creation(self):
        """Test study protocol creation."""
        protocol = self.framework.create_study_protocol(
            study_type=validation_framework.StudyType.INTENT_RECOGNITION_VALIDATION,
            custom_params={'target_sample_size': 20}
        )
        
        self.assertIsInstance(protocol, validation_framework.StudyProtocol)
        self.assertEqual(protocol.target_sample_size, 20)
        self.assertIn(protocol.study_id, self.framework.protocols)
    
    def test_participant_recruitment(self):
        """Test participant recruitment and screening."""
        participant_data = {
            'participant_id': 'TEST_P001',
            'age': 25,
            'gender': 'M',
            'handedness': 'right',
            'experience_robotics': 2
        }
        
        participant = self.framework.recruit_participant(participant_data)
        
        self.assertIn('TEST_P001', self.framework.participants)
        self.assertEqual(participant.age, 25)
    
    def test_experimental_design(self):
        """Test experimental design generation."""
        # Create protocol first
        protocol = self.framework.create_study_protocol(
            validation_framework.StudyType.INTENT_RECOGNITION_VALIDATION
        )
        
        # Recruit participant
        self.framework.recruit_participant({
            'participant_id': 'TEST_P001',
            'age': 25,
            'gender': 'M',
            'handedness': 'right'
        })
        
        # Generate session plan
        session_plan = self.framework.generate_session_plan('TEST_P001', protocol.study_id)
        
        self.assertIsInstance(session_plan, list)
        self.assertGreater(len(session_plan), 0)
        
        # Check trial structure
        for trial_info in session_plan:
            self.assertIn('trial_id', trial_info)
            self.assertIn('condition', trial_info)
            self.assertIn('task_parameters', trial_info)
    
    def test_statistical_analysis(self):
        """Test statistical analysis capabilities."""
        # Create mock experimental data
        trials = []
        
        for i in range(20):
            trial = validation_framework.ExperimentalTrial(
                trial_id=f"trial_{i:03d}",
                participant_id="TEST_P001",
                condition=validation_framework.ExperimentalCondition.BASELINE,
                task_parameters={'target_size': 0.02},
                start_time=time.time() - 100,
                end_time=time.time() - 50,
                completion_time=np.random.normal(25, 5),
                success_rate=np.random.uniform(0.7, 0.95),
                path_efficiency=np.random.uniform(0.6, 0.9)
            )
            trials.append(trial)
        
        # Record trials
        for trial in trials:
            self.framework.record_trial(trial)
        
        # Run analysis
        analysis_results = self.framework.analyze_results()
        
        self.assertIn('descriptive_statistics', analysis_results)
        self.assertIn('study_metadata', analysis_results)
        self.assertGreater(analysis_results['study_metadata']['total_trials'], 0)


class TestPrivacyEthicsIntegration(unittest.TestCase):
    """Test privacy and ethics compliance integration."""
    
    def setUp(self):
        """Setup test fixtures."""
        # Use temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        
        config = {
            'consent_db_path': os.path.join(self.temp_dir, 'test_consent.db'),
            'audit_log_path': os.path.join(self.temp_dir, 'test_audit.log')
        }
        
        self.privacy_manager = privacy_ethics.PrivacyComplianceManager(config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_consent_management_workflow(self):
        """Test complete consent management workflow."""
        participant_id = "TEST_P001"
        
        # Record consent
        consent = privacy_ethics.ConsentRecord(
            participant_id=participant_id,
            consent_type=privacy_ethics.ConsentType.RESEARCH_PARTICIPATION,
            granted=True,
            timestamp=time.time(),
            consent_version="v1.0"
        )
        
        self.privacy_manager.consent_manager.record_consent(consent)
        
        # Check consent
        has_consent = self.privacy_manager.consent_manager.check_consent(
            participant_id, privacy_ethics.ConsentType.RESEARCH_PARTICIPATION
        )
        self.assertTrue(has_consent)
        
        # Withdraw consent
        self.privacy_manager.consent_manager.withdraw_consent(
            participant_id, privacy_ethics.ConsentType.RESEARCH_PARTICIPATION
        )
        
        # Should no longer have consent
        has_consent = self.privacy_manager.consent_manager.check_consent(
            participant_id, privacy_ethics.ConsentType.RESEARCH_PARTICIPATION
        )
        self.assertFalse(has_consent)
    
    def test_data_processing_registration(self):
        """Test data processing registration and compliance."""
        participant_id = "TEST_P001"
        
        # Register consent first
        consent = privacy_ethics.ConsentRecord(
            participant_id=participant_id,
            consent_type=privacy_ethics.ConsentType.DATA_COLLECTION,
            granted=True,
            timestamp=time.time(),
            consent_version="v1.0"
        )
        self.privacy_manager.consent_manager.record_consent(consent)
        
        # Register data processing
        processing_id = self.privacy_manager.register_data_processing(
            participant_id=participant_id,
            data_categories=[privacy_ethics.DataCategory.BEHAVIORAL],
            purposes=[privacy_ethics.ProcessingPurpose.RESEARCH],
            legal_basis="consent"
        )
        
        self.assertIsInstance(processing_id, str)
        self.assertIn(processing_id, self.privacy_manager.processing_records)
    
    def test_data_subject_rights(self):
        """Test data subject rights processing."""
        participant_id = "TEST_P001"
        
        # Setup consent and processing
        consent = privacy_ethics.ConsentRecord(
            participant_id=participant_id,
            consent_type=privacy_ethics.ConsentType.DATA_COLLECTION,
            granted=True,
            timestamp=time.time(),
            consent_version="v1.0"
        )
        self.privacy_manager.consent_manager.record_consent(consent)
        
        # Process access request
        response = self.privacy_manager.process_data_subject_request(
            participant_id=participant_id,
            request_type=privacy_ethics.DataSubjectRight.ACCESS,
            user_id="admin"
        )
        
        self.assertEqual(response['status'], 'completed')
        self.assertIn('data', response)
    
    def test_data_anonymization(self):
        """Test data anonymization capabilities."""
        sample_data = [
            {
                'name': 'John Doe',
                'email': 'john@example.com',
                'age': 25,
                'performance_score': 0.85
            },
            {
                'name': 'Jane Smith',
                'email': 'jane@example.com', 
                'age': 28,
                'performance_score': 0.92
            }
        ]
        
        anonymized = self.privacy_manager.anonymize_dataset(sample_data, 'standard')
        
        # Should not contain original identifiers
        for record in anonymized:
            self.assertNotIn('name', record)
            self.assertNotIn('email', record)
        
        # Should still contain non-identifying data
        self.assertEqual(len(anonymized), len(sample_data))


class TestEndToEndIntegration(unittest.TestCase):
    """Test complete end-to-end system integration."""
    
    def setUp(self):
        """Setup complete system."""
        # Initialize all components
        self.bio_model = biomechanics.BiomechanicalModel()
        self.intent_system = intent_recognition.IntentRecognitionSystem()
        self.adaptive_model = adaptive_model.AdaptiveHumanModel("integration_test_user")
        
        # Initialize real-time processor
        self.processor = realtime_processing.RealTimeProcessor(
            target_latency_ms=10.0, num_worker_threads=2
        )
        
        # Initialize privacy manager
        self.temp_dir = tempfile.mkdtemp()
        privacy_config = {
            'consent_db_path': os.path.join(self.temp_dir, 'test_consent.db'),
            'audit_log_path': os.path.join(self.temp_dir, 'test_audit.log')
        }
        self.privacy_manager = privacy_ethics.PrivacyComplianceManager(privacy_config)
    
    def tearDown(self):
        """Clean up resources."""
        self.processor.stop()
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Setup privacy compliance
        participant_id = "INTEGRATION_TEST_001"
        
        # Record consent
        consent = privacy_ethics.ConsentRecord(
            participant_id=participant_id,
            consent_type=privacy_ethics.ConsentType.DATA_COLLECTION,
            granted=True,
            timestamp=time.time(),
            consent_version="v1.0"
        )
        self.privacy_manager.consent_manager.record_consent(consent)
        
        # Register data processing
        processing_id = self.privacy_manager.register_data_processing(
            participant_id=participant_id,
            data_categories=[privacy_ethics.DataCategory.BIOMETRIC],
            purposes=[privacy_ethics.ProcessingPurpose.RESEARCH]
        )
        
        # Initialize adaptive model
        self.adaptive_model.initialize_user({
            'demographics': {'age': 30, 'gender': 'M'},
            'physical': {'strength_percentile': 70}
        })
        
        # Start real-time processing
        self.processor.start()
        
        try:
            results = []
            
            def collect_results(result):
                results.append(result)
            
            self.processor.register_result_callback(collect_results)
            
            # Simulate data collection and processing
            for i in range(5):
                # Simulate sensor data
                sensor_data = {
                    'emg_main': {
                        'biceps': 0.4 + 0.1 * np.sin(i),
                        'triceps': 0.3 + 0.1 * np.cos(i)
                    },
                    'force_wrist': np.random.normal(10, 3, 6),
                    'kinematics_hand': {
                        'position_x': 0.3 + 0.1 * i / 10,
                        'velocity_x': 0.1,
                        'velocity_y': 0.05
                    }
                }
                
                # Submit for real-time processing
                task = realtime_processing.ProcessingTask(
                    task_id=f"integration_task_{i}",
                    priority=realtime_processing.ProcessingPriority.HIGH,
                    stage=realtime_processing.ProcessingStage.PREPROCESSING,
                    data=sensor_data
                )
                
                self.processor.submit_task(task)
                
                # Update models
                self.bio_model.update(dt=0.01, emg_signals=sensor_data['emg_main'])
                
                # Update adaptive model with mock performance
                observation = {
                    'performance': {
                        'completion_time': 20 + i,
                        'success_rate': 0.8 + 0.02 * i,
                        'path_efficiency': 0.7 + 0.03 * i
                    },
                    'muscle_activation': sensor_data['emg_main']
                }
                self.adaptive_model.update_parameters(observation)
            
            # Wait for processing
            time.sleep(0.5)
            
            # Verify results
            self.assertGreater(len(results), 0)
            
            # Check that models have been updated
            muscle_forces = self.bio_model.get_muscle_forces()
            self.assertIsInstance(muscle_forces, dict)
            self.assertGreater(len(muscle_forces), 0)
            
            adaptation_status = self.adaptive_model.get_adaptation_status()
            self.assertIn('observation_count', adaptation_status)
            self.assertGreater(adaptation_status['observation_count'], 0)
            
        finally:
            self.processor.stop()
    
    def test_system_performance_under_load(self):
        """Test system performance under high load."""
        self.processor.start()
        
        try:
            start_time = time.time()
            submitted_tasks = 0
            completed_tasks = 0
            
            def count_completions(result):
                nonlocal completed_tasks
                completed_tasks += 1
            
            self.processor.register_result_callback(count_completions)
            
            # Submit many tasks rapidly
            for i in range(50):
                sensor_data = {
                    'emg_main': {'biceps': np.random.uniform(0.1, 0.8)},
                    'force_wrist': np.random.normal(0, 10, 6)
                }
                
                task = realtime_processing.ProcessingTask(
                    task_id=f"load_test_{i}",
                    priority=realtime_processing.ProcessingPriority.HIGH,
                    stage=realtime_processing.ProcessingStage.PREPROCESSING,
                    data=sensor_data
                )
                
                if self.processor.submit_task(task):
                    submitted_tasks += 1
                
                time.sleep(0.001)  # 1ms between submissions
            
            # Wait for processing
            time.sleep(2.0)
            
            # Check performance
            processing_time = time.time() - start_time
            throughput = completed_tasks / processing_time
            
            # Should handle reasonable throughput
            self.assertGreater(throughput, 10)  # At least 10 tasks/second
            self.assertGreater(completed_tasks / submitted_tasks, 0.8)  # 80% completion rate
            
            # Check latency metrics
            metrics = self.processor.get_performance_metrics()
            if metrics.average_latency > 0:
                self.assertLess(metrics.average_latency, 0.05)  # < 50ms average
            
        finally:
            self.processor.stop()
    
    def test_error_recovery_and_resilience(self):
        """Test system error recovery and resilience."""
        self.processor.start()
        
        try:
            results = []
            errors = []
            
            def handle_result(result):
                if hasattr(result, 'error') and result.error:
                    errors.append(result)
                else:
                    results.append(result)
            
            self.processor.register_result_callback(handle_result)
            
            # Submit mix of valid and invalid tasks
            for i in range(10):
                if i % 3 == 0:
                    # Invalid data to test error handling
                    sensor_data = {
                        'invalid_sensor': None,
                        'malformed_data': 'not_a_dict'
                    }
                else:
                    # Valid data
                    sensor_data = {
                        'emg_main': {'biceps': 0.5},
                        'force_wrist': np.random.normal(0, 5, 6)
                    }
                
                task = realtime_processing.ProcessingTask(
                    task_id=f"resilience_test_{i}",
                    priority=realtime_processing.ProcessingPriority.HIGH,
                    stage=realtime_processing.ProcessingStage.PREPROCESSING,
                    data=sensor_data
                )
                
                self.processor.submit_task(task)
            
            time.sleep(1.0)
            
            # System should continue processing despite errors
            self.assertGreater(len(results), 0)  # Some valid results
            
            # System should still be responsive
            metrics = self.processor.get_performance_metrics()
            self.assertIsInstance(metrics, realtime_processing.PerformanceMetrics)
        
        finally:
            self.processor.stop()


def run_performance_benchmarks():
    """Run performance benchmarks for the system."""
    print("\n=== Performance Benchmarks ===")
    
    # Biomechanical model performance
    bio_model = biomechanics.BiomechanicalModel()
    
    start_time = time.time()
    for _ in range(1000):
        emg_signals = {'biceps': 0.5, 'triceps': 0.3}
        bio_model.update(dt=0.01, emg_signals=emg_signals)
    
    bio_time = time.time() - start_time
    print(f"Biomechanical model: {bio_time:.3f}s for 1000 updates ({1000/bio_time:.1f} Hz)")
    
    # Intent recognition performance
    intent_system = intent_recognition.IntentRecognitionSystem()
    
    # Mock training data
    training_data = []
    for _ in range(100):
        obs = intent_recognition.ObservationData()
        obs.emg_signals = {'biceps': np.random.uniform(0.1, 0.8)}
        training_data.append((obs, intent_recognition.IntentType.PRECISION_TASK))
    
    intent_system.train(training_data)
    
    start_time = time.time()
    for _ in range(1000):
        obs = intent_recognition.ObservationData()
        obs.emg_signals = {'biceps': np.random.uniform(0.1, 0.8)}
        intent_system.predict_intent(obs)
    
    intent_time = time.time() - start_time
    print(f"Intent recognition: {intent_time:.3f}s for 1000 predictions ({1000/intent_time:.1f} Hz)")
    
    # Data anonymization performance
    anonymizer = privacy_ethics.DataAnonymizer()
    
    sample_data = [
        {'name': f'User_{i}', 'age': 20 + i % 40, 'score': np.random.uniform(0, 1)}
        for i in range(1000)
    ]
    
    start_time = time.time()
    anonymized = anonymizer.anonymize_identifiers(sample_data[0])
    anon_time = time.time() - start_time
    
    print(f"Data anonymization: {anon_time:.6f}s per record ({1/anon_time:.0f} records/s)")


if __name__ == '__main__':
    # Run performance benchmarks first
    run_performance_benchmarks()
    
    # Run all tests
    print("\n=== Running Integration Tests ===")
    unittest.main(verbosity=2, exit=False)
    
    print("\nAll integration tests completed!")