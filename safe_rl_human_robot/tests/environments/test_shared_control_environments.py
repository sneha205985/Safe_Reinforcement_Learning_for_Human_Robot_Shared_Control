"""
Comprehensive tests for shared control environments.

This module tests all environment components including physics simulation,
human models, safety constraints, and visualization systems.
"""

import pytest
import numpy as np
import torch
import time
from typing import Dict, Any, List
import tempfile
import os

from src.environments.shared_control_base import (
    SharedControlBase, SafetyConstraint, SharedControlAction, 
    SharedControlState, SimplePhysicsEngine
)
from src.environments.exoskeleton_env import ExoskeletonEnvironment, ExoskeletonConfig
from src.environments.wheelchair_env import WheelchairEnvironment, WheelchairConfig
from src.environments.human_models import (
    AdvancedHumanModel, BiomechanicalModel, IntentRecognizer,
    MotorImpairment, IntentType
)
from src.environments.physics_simulation import (
    create_physics_engine, create_simple_arm_urdf
)
from src.environments.safety_monitoring import (
    AdaptiveSafetyMonitor, PredictiveConstraint, SafetyLevel
)
from src.environments.visualization import (
    VisualizationManager, create_visualization_config
)
from src.environments.human_robot_env import AssistanceMode


class TestSharedControlBase:
    """Test base shared control functionality."""
    
    @pytest.fixture
    def simple_physics_engine(self):
        """Create simple physics engine for testing."""
        return SimplePhysicsEngine(robot_dof=7)
    
    @pytest.fixture
    def safety_constraint(self):
        """Create test safety constraint."""
        def constraint_func(state, action):
            return 1.0 - torch.norm(action).item()  # Simple constraint
        
        return SafetyConstraint(
            name="test_constraint",
            constraint_function=constraint_func,
            threshold=0.0,
            penalty_weight=1.0
        )
    
    def test_physics_engine_creation(self):
        """Test physics engine factory function."""
        # Test fallback engine
        engine = create_physics_engine("fallback", robot_dof=7)
        assert engine is not None
        assert hasattr(engine, 'step')
        assert hasattr(engine, 'get_state')
        
        # Test auto selection
        engine_auto = create_physics_engine("auto")
        assert engine_auto is not None
    
    def test_simple_physics_dynamics(self, simple_physics_engine):
        """Test simple physics engine dynamics."""
        engine = simple_physics_engine
        
        # Test initial state
        initial_state = engine.get_state()
        assert torch.allclose(initial_state['positions'], torch.zeros(7))
        assert torch.allclose(initial_state['velocities'], torch.zeros(7))
        
        # Apply action and step
        action = torch.ones(7) * 0.1
        result = engine.step(action)
        
        assert 'positions' in result
        assert 'velocities' in result
        assert 'accelerations' in result
        
        # Check that state changed
        new_state = engine.get_state()
        assert not torch.allclose(new_state['velocities'], torch.zeros(7))
    
    def test_safety_constraint_evaluation(self, safety_constraint):
        """Test safety constraint evaluation."""
        from src.environments.human_robot_env import EnvironmentState, RobotState, HumanInput
        
        # Create test state
        robot_state = RobotState(
            position=torch.zeros(7),
            velocity=torch.zeros(7),
            torque=torch.zeros(7),
            end_effector_pose=torch.zeros(6),
            contact_forces=torch.zeros(6)
        )
        
        human_input = HumanInput(
            intention=torch.zeros(3),
            confidence=0.5,
            timestamp=0.0
        )
        
        env_state = EnvironmentState(
            robot=robot_state,
            human_input=human_input,
            obstacles=[],
            target_position=torch.ones(3),
            time_step=0,
            assistance_mode=AssistanceMode.COLLABORATIVE
        )
        
        # Test safe action
        safe_action = torch.ones(7) * 0.1
        result = safety_constraint.evaluate(env_state, safe_action)
        
        assert 'value' in result
        assert 'violation' in result
        assert 'penalty' in result
        
        # Test unsafe action
        unsafe_action = torch.ones(7) * 2.0
        unsafe_result = safety_constraint.evaluate(env_state, unsafe_action)
        assert unsafe_result['violation']
        assert unsafe_result['penalty'] > 0


class TestExoskeletonEnvironment:
    """Test exoskeleton environment functionality."""
    
    @pytest.fixture
    def exoskeleton_config(self):
        """Create test exoskeleton configuration."""
        return ExoskeletonConfig()
    
    @pytest.fixture
    def exoskeleton_env(self, exoskeleton_config):
        """Create exoskeleton environment for testing."""
        return ExoskeletonEnvironment(
            config=exoskeleton_config,
            task_type="reach_target",
            human_impairment_level=0.0
        )
    
    def test_environment_initialization(self, exoskeleton_env):
        """Test environment initialization."""
        assert exoskeleton_env.config is not None
        assert exoskeleton_env.robot_dof == 7
        assert exoskeleton_env.action_space is not None
        assert exoskeleton_env.observation_space is not None
        assert len(exoskeleton_env.safety_constraints) > 0
    
    def test_environment_reset(self, exoskeleton_env):
        """Test environment reset functionality."""
        obs = exoskeleton_env.reset()
        
        assert obs is not None
        assert obs.shape[0] == exoskeleton_env.observation_space.shape[0]
        assert exoskeleton_env.current_state is not None
        assert exoskeleton_env.step_count == 0
    
    def test_environment_step(self, exoskeleton_env):
        """Test environment step functionality."""
        exoskeleton_env.reset()
        
        # Test valid action
        action = torch.zeros(7)  # Safe zero action
        obs, reward, done, info = exoskeleton_env.step(action)
        
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert exoskeleton_env.step_count == 1
        
        # Check info contains expected keys
        expected_keys = ['step_count', 'episode_reward', 'human_confidence', 
                        'assistance_level', 'safety_override']
        for key in expected_keys:
            assert key in info
    
    def test_kinematics_computation(self, exoskeleton_env):
        """Test forward kinematics computation."""
        joint_angles = torch.zeros(7)
        kinematics = exoskeleton_env._compute_kinematics(joint_angles.unsqueeze(0))
        
        assert 'end_effector_pose' in kinematics
        assert 'jacobian' in kinematics
        assert kinematics['end_effector_pose'].shape[-1] == 6  # [x,y,z,rx,ry,rz]
        assert kinematics['jacobian'].shape == (6, 7)  # 6D pose, 7 joints
    
    def test_safety_constraints(self, exoskeleton_env):
        """Test safety constraints are properly enforced."""
        exoskeleton_env.reset()
        
        # Test constraint violation with large action
        large_action = torch.ones(7) * 100.0  # Very large torques
        obs, reward, done, info = exoskeleton_env.step(large_action)
        
        # Should have safety violations
        assert info.get('constraint_violations', 0) > 0 or info.get('safety_override', False)
    
    def test_human_model_integration(self, exoskeleton_env):
        """Test human model integration."""
        exoskeleton_env.reset()
        
        # Get human intention
        human_input = exoskeleton_env.human_model.get_intention(
            exoskeleton_env._convert_state_for_human_model(exoskeleton_env.current_state)
        )
        
        assert human_input is not None
        assert human_input.intention.shape[0] == 3
        assert 0 <= human_input.confidence <= 1
        assert hasattr(human_input, 'muscle_activity')
    
    def test_task_metrics(self, exoskeleton_env):
        """Test task-specific metrics."""
        exoskeleton_env.reset()
        
        metrics = exoskeleton_env.get_task_metrics()
        
        assert 'distance_to_target' in metrics
        assert 'task_completion_rate' in metrics
        assert 'human_impairment_level' in metrics
        
        # Test with EMG enabled
        exo_with_emg = ExoskeletonEnvironment(enable_emg=True)
        exo_with_emg.reset()
        emg_metrics = exo_with_emg.get_task_metrics()
        
        if 'emg_signals' in exo_with_emg.current_state.environment_info:
            assert 'total_muscle_activation' in emg_metrics
            assert 'muscle_cocontraction' in emg_metrics


class TestWheelchairEnvironment:
    """Test wheelchair environment functionality."""
    
    @pytest.fixture
    def wheelchair_config(self):
        """Create test wheelchair configuration."""
        return WheelchairConfig()
    
    @pytest.fixture
    def wheelchair_env(self, wheelchair_config):
        """Create wheelchair environment for testing."""
        return WheelchairEnvironment(
            config=wheelchair_config,
            task_type="navigation",
            mobility_impairment_level=0.0,
            environment_complexity="simple"
        )
    
    def test_environment_initialization(self, wheelchair_env):
        """Test wheelchair environment initialization."""
        assert wheelchair_env.config is not None
        assert wheelchair_env.robot_dof == 2  # Linear and angular velocity
        assert wheelchair_env.action_space is not None
        assert wheelchair_env.observation_space is not None
        assert len(wheelchair_env.obstacles) > 0  # Should have some obstacles
    
    def test_environment_reset(self, wheelchair_env):
        """Test environment reset."""
        obs = wheelchair_env.reset()
        
        assert obs is not None
        assert obs.shape[0] == wheelchair_env.observation_space.shape[0]
        assert wheelchair_env.goal_position is not None
        assert len(wheelchair_env.planned_path) > 0
        assert wheelchair_env.step_count == 0
    
    def test_wheelchair_dynamics(self, wheelchair_env):
        """Test wheelchair dynamics."""
        wheelchair_env.reset()
        initial_position = wheelchair_env.position.clone()
        
        # Apply forward motion
        action = torch.tensor([0.5, 0.0])  # Forward, no turning
        obs, reward, done, info = wheelchair_env.step(action)
        
        # Position should have changed
        assert not torch.allclose(wheelchair_env.position, initial_position)
        assert torch.norm(wheelchair_env.linear_velocity) > 0
    
    def test_obstacle_generation(self, wheelchair_env):
        """Test obstacle generation for different complexities."""
        # Test simple complexity
        simple_env = WheelchairEnvironment(environment_complexity="simple")
        simple_env.reset()
        simple_obstacles = len(simple_env.obstacles)
        
        # Test complex complexity
        complex_env = WheelchairEnvironment(environment_complexity="complex")
        complex_env.reset()
        complex_obstacles = len(complex_env.obstacles)
        
        assert complex_obstacles > simple_obstacles
        
        # Check for dynamic obstacles in complex environment
        dynamic_obstacles = [obs for obs in complex_env.obstacles if not obs.is_static]
        assert len(dynamic_obstacles) > 0
    
    def test_path_planning(self, wheelchair_env):
        """Test path planning functionality."""
        wheelchair_env.reset()
        
        start = torch.tensor([1.0, 1.0])
        goal = torch.tensor([8.0, 8.0])
        
        path = wheelchair_env.path_planner.plan_path(start, goal, wheelchair_env.obstacles)
        
        assert len(path) >= 2  # At least start and goal
        assert torch.allclose(path[0], start, atol=0.1)
        assert torch.allclose(path[-1], goal, atol=0.1)
    
    def test_joystick_model(self, wheelchair_env):
        """Test joystick input modeling."""
        intended_direction = torch.tensor([1.0, 0.5])
        timestamp = 0.0
        
        joystick_input = wheelchair_env.joystick.get_joystick_input(
            intended_direction, timestamp
        )
        
        assert joystick_input.shape == (2,)
        assert torch.all(torch.abs(joystick_input) <= 1.0)  # Within valid range
    
    def test_navigation_metrics(self, wheelchair_env):
        """Test navigation performance metrics."""
        wheelchair_env.reset()
        wheelchair_env.step(torch.zeros(2))  # One step to generate metrics
        
        metrics = wheelchair_env.get_navigation_metrics()
        
        assert 'distance_to_goal' in metrics
        assert 'total_distance_traveled' in metrics
        assert 'path_efficiency' in metrics
        assert 'navigation_success' in metrics
        assert 'mobility_impairment_level' in metrics


class TestHumanModels:
    """Test advanced human behavior models."""
    
    @pytest.fixture
    def advanced_human_model(self):
        """Create advanced human model for testing."""
        return AdvancedHumanModel(
            skill_level=0.8,
            impairment_type=MotorImpairment.TREMOR,
            impairment_severity=0.3
        )
    
    @pytest.fixture
    def biomechanical_model(self):
        """Create biomechanical model for testing."""
        return BiomechanicalModel(
            impairment_type=MotorImpairment.SPASTICITY,
            severity=0.2
        )
    
    @pytest.fixture
    def intent_recognizer(self):
        """Create intent recognizer for testing."""
        return IntentRecognizer()
    
    def test_advanced_human_model_initialization(self, advanced_human_model):
        """Test advanced human model initialization."""
        assert advanced_human_model.skill_level == 0.8
        assert advanced_human_model.impairment_type == MotorImpairment.TREMOR
        assert advanced_human_model.impairment_severity == 0.3
        assert advanced_human_model.biomechanics is not None
        assert advanced_human_model.intent_recognizer is not None
    
    def test_human_intention_generation(self, advanced_human_model):
        """Test human intention generation."""
        from src.environments.human_robot_env import EnvironmentState, RobotState, HumanInput
        
        # Create test environment state
        robot_state = RobotState(
            position=torch.zeros(7),
            velocity=torch.zeros(7),
            torque=torch.zeros(7),
            end_effector_pose=torch.tensor([0.5, 0.0, 1.0, 0.0, 0.0, 0.0]),
            contact_forces=torch.zeros(6)
        )
        
        env_state = EnvironmentState(
            robot=robot_state,
            human_input=HumanInput(torch.zeros(3), 0.5, 0.0),
            obstacles=[],
            target_position=torch.tensor([1.0, 0.0, 1.0]),
            time_step=0,
            assistance_mode=AssistanceMode.COLLABORATIVE
        )
        
        human_input = advanced_human_model.get_intention(env_state)
        
        assert human_input.intention.shape == (3,)
        assert 0 <= human_input.confidence <= 1
        assert human_input.muscle_activity is not None
        assert human_input.muscle_activity.shape == (8,)
        assert human_input.gaze_direction is not None
    
    def test_biomechanical_constraints(self, biomechanical_model):
        """Test biomechanical constraint application."""
        intended_force = torch.ones(7) * 10.0  # 10N force
        joint_angles = torch.zeros(7)
        
        muscle_activation = biomechanical_model.compute_muscle_activation(
            intended_force, joint_angles
        )
        
        assert muscle_activation.shape == (8,)
        assert torch.all(muscle_activation >= 0)
        assert torch.all(muscle_activation <= 1)
        
        # Test fatigue accumulation
        initial_fatigue = biomechanical_model.current_fatigue
        
        # Apply high activation repeatedly
        for _ in range(100):
            biomechanical_model.compute_muscle_activation(
                torch.ones(7) * 50.0, joint_angles
            )
        
        assert biomechanical_model.current_fatigue > initial_fatigue
    
    def test_intent_recognition(self, intent_recognizer):
        """Test intent recognition functionality."""
        # Test reaching intent
        movement = torch.tensor([1.0, 0.0, 0.0])  # Forward movement
        context = {
            'current_position': torch.tensor([0.0, 0.0, 1.0]),
            'target_position': torch.tensor([1.0, 0.0, 1.0])
        }
        
        intent_type, confidence = intent_recognizer.recognize_intent(
            movement, context, time.time()
        )
        
        assert isinstance(intent_type, IntentType)
        assert 0 <= confidence <= 1
        
        # Add more movements to build history
        for i in range(10):
            intent_recognizer.recognize_intent(
                movement + torch.normal(0, 0.1, (3,)), 
                context, 
                time.time() + i * 0.1
            )
        
        # Should have better confidence with history
        intent_type_2, confidence_2 = intent_recognizer.recognize_intent(
            movement, context, time.time() + 1.0
        )
        
        # Intent should be consistent
        if intent_type != IntentType.UNCERTAIN:
            assert intent_type == intent_type_2 or confidence_2 > 0.3
    
    def test_human_adaptation(self, advanced_human_model):
        """Test human adaptation to robot assistance."""
        robot_action = torch.tensor([0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        assistance_level = 0.7
        
        initial_trust = advanced_human_model.adaptation_state['trust_in_robot']
        
        # Simulate good assistance
        for _ in range(50):
            advanced_human_model.adapt_to_assistance(robot_action, assistance_level)
        
        # Trust should change based on assistance quality
        final_trust = advanced_human_model.adaptation_state['trust_in_robot']
        # Trust may increase or decrease depending on assistance evaluation
        assert abs(final_trust - initial_trust) > 0.01  # Some change occurred
    
    def test_human_state_reporting(self, advanced_human_model):
        """Test comprehensive human state reporting."""
        state = advanced_human_model.get_human_state()
        
        expected_keys = [
            'skill_level', 'effective_skill', 'experience_level',
            'biomechanical_state', 'adaptation_state', 'intent_history',
            'performance_metrics', 'impairment_info'
        ]
        
        for key in expected_keys:
            assert key in state
        
        # Check biomechanical state structure
        biomech_state = state['biomechanical_state']
        assert 'muscle_activation' in biomech_state
        assert 'fatigue_level' in biomech_state
        assert 'movement_quality' in biomech_state


class TestSafetyMonitoring:
    """Test safety monitoring systems."""
    
    @pytest.fixture
    def safety_monitor(self):
        """Create safety monitor for testing."""
        return AdaptiveSafetyMonitor(
            update_frequency=10.0,
            prediction_horizon=1.0
        )
    
    @pytest.fixture
    def test_constraint(self):
        """Create test constraint for safety monitoring."""
        def constraint_func(state, action):
            # Simple constraint: limit action magnitude
            return 1.0 - torch.norm(action).item()
        
        return SafetyConstraint(
            name="action_magnitude",
            constraint_function=constraint_func,
            threshold=0.0,
            penalty_weight=10.0
        )
    
    def test_safety_monitor_initialization(self, safety_monitor):
        """Test safety monitor initialization."""
        assert safety_monitor.update_frequency == 10.0
        assert safety_monitor.prediction_horizon == 1.0
        assert len(safety_monitor.predictive_constraints) == 0
        assert safety_monitor.current_status.overall_level == SafetyLevel.SAFE
    
    def test_constraint_addition(self, safety_monitor, test_constraint):
        """Test adding constraints to safety monitor."""
        safety_monitor.add_constraint(test_constraint, weight=1.5)
        
        assert len(safety_monitor.predictive_constraints) == 1
        assert safety_monitor.constraint_weights['action_magnitude'] == 1.5
        
        # Test removal
        success = safety_monitor.remove_constraint('action_magnitude')
        assert success
        assert len(safety_monitor.predictive_constraints) == 0
    
    def test_safety_evaluation(self, safety_monitor, test_constraint):
        """Test safety evaluation with constraints."""
        from src.environments.human_robot_env import EnvironmentState, RobotState, HumanInput
        
        safety_monitor.add_constraint(test_constraint)
        
        # Create test state
        robot_state = RobotState(
            position=torch.zeros(7),
            velocity=torch.zeros(7),
            torque=torch.zeros(7),
            end_effector_pose=torch.zeros(6),
            contact_forces=torch.zeros(6)
        )
        
        env_state = EnvironmentState(
            robot=robot_state,
            human_input=HumanInput(torch.zeros(3), 0.5, 0.0),
            obstacles=[],
            target_position=torch.ones(3),
            time_step=0,
            assistance_mode=AssistanceMode.COLLABORATIVE
        )
        
        # Test safe action
        safe_action = torch.ones(7) * 0.1
        status = safety_monitor.evaluate_safety(env_state, safe_action)
        
        assert status.overall_level == SafetyLevel.SAFE
        assert len(status.active_violations) == 0
        
        # Test unsafe action
        unsafe_action = torch.ones(7) * 2.0
        unsafe_status = safety_monitor.evaluate_safety(env_state, unsafe_action)
        
        assert unsafe_status.overall_level != SafetyLevel.SAFE
        assert len(unsafe_status.active_violations) > 0
    
    def test_predictive_constraints(self, test_constraint):
        """Test predictive constraint functionality."""
        pred_constraint = PredictiveConstraint(test_constraint, prediction_horizon=1.0)
        
        from src.environments.human_robot_env import EnvironmentState, RobotState, HumanInput
        
        env_state = EnvironmentState(
            robot=RobotState(
                position=torch.zeros(7),
                velocity=torch.zeros(7),
                torque=torch.zeros(7),
                end_effector_pose=torch.zeros(6),
                contact_forces=torch.zeros(6)
            ),
            human_input=HumanInput(torch.zeros(3), 0.5, 0.0),
            obstacles=[],
            target_position=torch.ones(3),
            time_step=0,
            assistance_mode=AssistanceMode.COLLABORATIVE
        )
        
        # Build some history first
        for i in range(10):
            action = torch.ones(7) * (0.8 + i * 0.1)  # Increasing violation
            pred_constraint.evaluate_predictive(env_state, action)
            time.sleep(0.01)  # Small delay to build temporal history
        
        # Final evaluation should show trend toward violation
        final_eval = pred_constraint.evaluate_predictive(env_state, torch.ones(7) * 1.5)
        
        assert 'predicted_violations' in final_eval
        assert 'trend_derivative' in final_eval
        assert 'time_to_violation' in final_eval
    
    def test_emergency_stop_mechanism(self, safety_monitor, test_constraint):
        """Test emergency stop triggering."""
        # Add emergency callback
        emergency_triggered = {'value': False}
        
        def emergency_callback():
            emergency_triggered['value'] = True
        
        safety_monitor.add_emergency_callback(emergency_callback)
        
        # Add constraint that will trigger emergency
        def emergency_constraint_func(state, action):
            return -10.0  # Always violated severely
        
        emergency_constraint = SafetyConstraint(
            name="emergency_test",
            constraint_function=emergency_constraint_func,
            threshold=0.0,
            penalty_weight=100.0
        )
        
        safety_monitor.add_constraint(emergency_constraint)
        
        # Create test state
        from src.environments.human_robot_env import EnvironmentState, RobotState, HumanInput
        
        env_state = EnvironmentState(
            robot=RobotState(
                position=torch.zeros(7),
                velocity=torch.zeros(7), 
                torque=torch.zeros(7),
                end_effector_pose=torch.zeros(6),
                contact_forces=torch.zeros(6)
            ),
            human_input=HumanInput(torch.zeros(3), 0.5, 0.0),
            obstacles=[],
            target_position=torch.ones(3),
            time_step=0,
            assistance_mode=AssistanceMode.COLLABORATIVE
        )
        
        # Evaluate with any action (constraint always violated)
        status = safety_monitor.evaluate_safety(env_state, torch.zeros(7))
        
        # Emergency should be triggered
        assert emergency_triggered['value']
        assert status.emergency_stop_active
    
    def test_safety_report_generation(self, safety_monitor, test_constraint):
        """Test safety report generation."""
        safety_monitor.add_constraint(test_constraint)
        
        report = safety_monitor.get_safety_report()
        
        expected_keys = [
            'timestamp', 'overall_status', 'current_violations',
            'safety_margins', 'performance_metrics', 'constraint_health',
            'monitoring_parameters'
        ]
        
        for key in expected_keys:
            assert key in report
        
        # Check report structure
        assert 'level' in report['overall_status']
        assert 'confidence' in report['overall_status']
        assert 'active_count' in report['current_violations']
        assert 'predicted_count' in report['current_violations']


class TestPhysicsSimulation:
    """Test physics simulation components."""
    
    def test_urdf_generation(self):
        """Test URDF generation for simple arms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = os.path.join(temp_dir, "test_arm.urdf")
            
            generated_path = create_simple_arm_urdf(
                num_joints=7,
                link_lengths=[0.3, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
                output_path=urdf_path
            )
            
            assert os.path.exists(generated_path)
            
            # Check URDF content
            with open(generated_path, 'r') as f:
                content = f.read()
                assert '<robot name="simple_arm">' in content
                assert 'joint_1' in content
                assert 'link_1' in content
    
    def test_physics_engine_fallback(self):
        """Test fallback physics engine functionality."""
        from src.environments.physics_simulation import FallbackPhysicsEngine
        
        engine = FallbackPhysicsEngine(robot_dof=5)
        
        # Test initial state
        state = engine.get_state()
        assert state['positions'].shape == (5,)
        assert torch.allclose(state['positions'], torch.zeros(5))
        
        # Test dynamics
        action = torch.ones(5) * 0.5
        result = engine.step(action)
        
        assert 'positions' in result
        assert 'velocities' in result
        assert 'accelerations' in result
        
        # State should have changed
        new_state = engine.get_state()
        assert not torch.allclose(new_state['velocities'], torch.zeros(5))
        
        # Test reset
        engine.reset()
        reset_state = engine.get_state()
        assert torch.allclose(reset_state['positions'], torch.zeros(5))
        assert torch.allclose(reset_state['velocities'], torch.zeros(5))


class TestVisualization:
    """Test visualization components."""
    
    def test_visualization_config_creation(self):
        """Test visualization configuration creation."""
        wheelchair_config = create_visualization_config("wheelchair", advanced_features=True)
        
        assert wheelchair_config.window_size == (1400, 900)
        assert wheelchair_config.show_safety_zones
        assert wheelchair_config.show_planned_path
        assert wheelchair_config.show_predicted_violations
        
        exo_config = create_visualization_config("exoskeleton", advanced_features=False)
        
        assert exo_config.window_size == (1600, 1200)
        assert exo_config.show_safety_zones
        assert exo_config.show_human_intent
        assert not exo_config.show_predicted_violations
    
    def test_visualization_manager_initialization(self):
        """Test visualization manager initialization."""
        config = create_visualization_config("wheelchair")
        
        # Test with matplotlib only (should always be available)
        viz_manager = VisualizationManager(
            config=config,
            env_type="wheelchair",
            visualizer_types=["matplotlib"]
        )
        
        # Should have at least attempted to create visualizers
        # Note: May fail in headless environments, but should not crash
        assert hasattr(viz_manager, 'visualizers')
        assert hasattr(viz_manager, 'config')
        assert viz_manager.env_type == "wheelchair"
        
        # Test cleanup
        viz_manager.close()


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""
    
    def test_exoskeleton_full_pipeline(self):
        """Test complete exoskeleton pipeline."""
        # Create environment with advanced features
        env = ExoskeletonEnvironment(
            task_type="reach_target",
            human_impairment_level=0.2,
            enable_emg=True
        )
        
        # Reset environment
        obs = env.reset()
        assert obs is not None
        
        # Run several steps
        for _ in range(10):
            # Random action within limits
            action = torch.normal(0, 0.1, size=(7,))
            obs, reward, done, info = env.step(action)
            
            # Check all components are working
            assert obs is not None
            assert isinstance(reward, float)
            assert isinstance(info, dict)
            assert 'human_confidence' in info
            assert 'safety_override' in info
            
            if done:
                break
        
        # Test task metrics
        metrics = env.get_task_metrics()
        assert 'distance_to_target' in metrics
        assert 'human_impairment_level' in metrics
        
        # Test safety metrics
        safety_metrics = env.get_safety_metrics()
        assert 'constraint_violations' in safety_metrics
        assert 'safety_interventions' in safety_metrics
    
    def test_wheelchair_full_pipeline(self):
        """Test complete wheelchair pipeline."""
        env = WheelchairEnvironment(
            task_type="navigation",
            mobility_impairment_level=0.3,
            environment_complexity="moderate"
        )
        
        # Reset and run
        obs = env.reset()
        assert obs is not None
        
        total_reward = 0
        for step in range(20):
            # Simple forward motion with occasional turning
            if step % 5 == 0:
                action = torch.tensor([0.0, 0.3])  # Turn
            else:
                action = torch.tensor([0.5, 0.0])  # Forward
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Verify wheelchair dynamics
            assert env.position is not None
            assert env.orientation is not None
            
            if done:
                break
        
        # Test navigation metrics
        nav_metrics = env.get_navigation_metrics()
        assert 'distance_to_goal' in nav_metrics
        assert 'path_efficiency' in nav_metrics
        assert 'navigation_success' in nav_metrics
    
    def test_safety_monitoring_integration(self):
        """Test safety monitoring with environment."""
        env = ExoskeletonEnvironment()
        safety_monitor = AdaptiveSafetyMonitor()
        
        # Add environment's constraints to monitor
        for constraint in env.safety_constraints:
            safety_monitor.add_constraint(constraint)
        
        env.reset()
        
        # Simulate monitoring during environment steps
        for _ in range(10):
            action = torch.normal(0, 0.5, size=(7,))
            
            # Evaluate safety before step
            env_state_for_monitor = env._convert_state_for_human_model(env.current_state)
            safety_status = safety_monitor.evaluate_safety(env_state_for_monitor, action)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Verify safety monitoring
            assert safety_status is not None
            assert hasattr(safety_status, 'overall_level')
            assert hasattr(safety_status, 'active_violations')
            
            if safety_status.emergency_stop_active:
                print(f"Emergency stop triggered at step {_}")
                break
        
        # Generate safety report
        report = safety_monitor.get_safety_report()
        assert 'overall_status' in report
        assert 'constraint_health' in report


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance tests to ensure environments run efficiently."""
    
    def test_exoskeleton_performance(self):
        """Test exoskeleton environment performance."""
        env = ExoskeletonEnvironment()
        env.reset()
        
        start_time = time.time()
        num_steps = 1000
        
        for _ in range(num_steps):
            action = torch.normal(0, 0.1, size=(7,))
            env.step(action)
        
        elapsed_time = time.time() - start_time
        steps_per_second = num_steps / elapsed_time
        
        print(f"Exoskeleton performance: {steps_per_second:.1f} steps/second")
        assert steps_per_second > 100  # Should achieve at least 100 steps/second
    
    def test_wheelchair_performance(self):
        """Test wheelchair environment performance."""
        env = WheelchairEnvironment(environment_complexity="complex")
        env.reset()
        
        start_time = time.time()
        num_steps = 1000
        
        for _ in range(num_steps):
            action = torch.normal(0, 0.1, size=(2,))
            env.step(action)
        
        elapsed_time = time.time() - start_time
        steps_per_second = num_steps / elapsed_time
        
        print(f"Wheelchair performance: {steps_per_second:.1f} steps/second")
        assert steps_per_second > 200  # Should be faster than exoskeleton
    
    def test_safety_monitoring_performance(self):
        """Test safety monitoring performance."""
        safety_monitor = AdaptiveSafetyMonitor()
        
        # Add multiple constraints
        for i in range(5):
            def constraint_func(state, action):
                return 1.0 - torch.norm(action).item() - i * 0.1
            
            constraint = SafetyConstraint(
                name=f"test_constraint_{i}",
                constraint_function=constraint_func,
                threshold=0.0
            )
            safety_monitor.add_constraint(constraint)
        
        # Create test state
        from src.environments.human_robot_env import EnvironmentState, RobotState, HumanInput
        
        env_state = EnvironmentState(
            robot=RobotState(
                position=torch.zeros(7),
                velocity=torch.zeros(7),
                torque=torch.zeros(7),
                end_effector_pose=torch.zeros(6),
                contact_forces=torch.zeros(6)
            ),
            human_input=HumanInput(torch.zeros(3), 0.5, 0.0),
            obstacles=[],
            target_position=torch.ones(3),
            time_step=0,
            assistance_mode=AssistanceMode.COLLABORATIVE
        )
        
        start_time = time.time()
        num_evaluations = 1000
        
        for _ in range(num_evaluations):
            action = torch.normal(0, 0.1, size=(7,))
            safety_monitor.evaluate_safety(env_state, action)
        
        elapsed_time = time.time() - start_time
        evaluations_per_second = num_evaluations / elapsed_time
        
        print(f"Safety monitoring performance: {evaluations_per_second:.1f} evaluations/second")
        assert evaluations_per_second > 500  # Should be very fast


if __name__ == "__main__":
    pytest.main([__file__])