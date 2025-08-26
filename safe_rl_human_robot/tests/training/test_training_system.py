"""
Comprehensive tests for the training system.

Tests all components: configuration, training, hyperparameter optimization,
experiment tracking, distributed training, callbacks, evaluation, and schedulers.
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json

# Import components to test
from safe_rl_human_robot.src.training.config import (
    TrainingConfig, NetworkConfig, OptimizationConfig,
    EvaluationConfig, ExperimentConfig
)
from safe_rl_human_robot.src.training.trainer import CPOTrainer, TrainingResults
from safe_rl_human_robot.src.training.hyperopt import (
    HyperparameterOptimizer, OptimizationStudy, SearchSpace
)
from safe_rl_human_robot.src.training.experiment_tracker import (
    ExperimentMetrics, LocalTracker, CompositeTracker, ExperimentManager
)
from safe_rl_human_robot.src.training.callbacks import (
    CallbackState, EarlyStopping, ModelCheckpoint, LearningRateScheduler,
    SafetyMonitorCallback, CallbackManager
)
from safe_rl_human_robot.src.training.evaluation import (
    EvaluationMetrics, StandardEvaluation, SafetyEvaluation, EvaluationManager
)
from safe_rl_human_robot.src.training.schedulers import (
    SchedulerConfig, LinearScheduler, ExponentialScheduler,
    SchedulerManager, create_common_schedulers
)
from safe_rl_human_robot.src.training.distributed import (
    DistributedConfig, DistributedManager, DistributedLauncher
)


class TestTrainingConfig:
    """Test training configuration classes."""
    
    def test_training_config_creation(self):
        """Test basic training configuration creation."""
        config = TrainingConfig()
        
        assert config.max_iterations == 1000
        assert config.rollout_length == 2000
        assert config.batch_size == 64
        assert isinstance(config.network, NetworkConfig)
        assert isinstance(config.optimization, OptimizationConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        config = TrainingConfig(
            max_iterations=500,
            rollout_length=1000,
            batch_size=32
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict['max_iterations'] == 500
        assert 'network' in config_dict
        
        # Test from_dict
        restored_config = TrainingConfig.from_dict(config_dict)
        assert restored_config.max_iterations == 500
        assert restored_config.rollout_length == 1000
    
    def test_hyperparameter_extraction(self):
        """Test hyperparameter dictionary extraction."""
        config = TrainingConfig()
        hyperparams = config.get_hyperparameter_dict()
        
        expected_keys = [
            'policy_lr', 'value_lr', 'trust_region_radius',
            'constraint_threshold', 'gamma', 'gae_lambda'
        ]
        
        for key in expected_keys:
            assert key in hyperparams
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = TrainingConfig(max_iterations=100, rollout_length=500)
        config.validate()
        
        # Invalid config should raise
        with pytest.raises(ValueError):
            invalid_config = TrainingConfig(max_iterations=-1)
            invalid_config.validate()


class TestCPOTrainer:
    """Test CPO trainer functionality."""
    
    @pytest.fixture
    def mock_environment(self):
        """Create a mock environment."""
        env = Mock()
        env.observation_space.shape = (10,)
        env.action_space.shape = (4,)
        env.reset.return_value = np.zeros(10)
        env.step.return_value = (np.zeros(10), 1.0, False, {})
        env.get_safety_constraints.return_value = {'collision': 0.1}
        return env
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TrainingConfig(
            max_iterations=5,
            rollout_length=100,
            batch_size=32,
            save_frequency=2
        )
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_trainer_initialization(self, config, mock_environment, temp_dir):
        """Test trainer initialization."""
        config.save_dir = temp_dir
        
        with patch('safe_rl_human_robot.src.algorithms.cpo.CPOAgent') as MockAgent:
            trainer = CPOTrainer(config, mock_environment)
            
            assert trainer.config == config
            assert trainer.environment == mock_environment
            assert trainer.iteration == 0
    
    def test_rollout_collection(self, config, mock_environment, temp_dir):
        """Test rollout data collection."""
        config.save_dir = temp_dir
        config.rollout_length = 10
        
        with patch('safe_rl_human_robot.src.algorithms.cpo.CPOAgent') as MockAgent:
            mock_agent = Mock()
            mock_agent.select_action.return_value = np.array([0.5, 0.5, 0.5, 0.5])
            MockAgent.return_value = mock_agent
            
            trainer = CPOTrainer(config, mock_environment)
            rollout_data = trainer._collect_rollouts()
            
            assert 'states' in rollout_data
            assert 'actions' in rollout_data
            assert 'rewards' in rollout_data
            assert rollout_data['states'].shape[0] > 0
    
    def test_checkpoint_saving_loading(self, config, mock_environment, temp_dir):
        """Test checkpoint saving and loading."""
        config.save_dir = temp_dir
        
        with patch('safe_rl_human_robot.src.algorithms.cpo.CPOAgent') as MockAgent:
            trainer = CPOTrainer(config, mock_environment)
            
            # Mock the models
            trainer.policy_net = Mock()
            trainer.value_net = Mock()
            trainer.policy_net.state_dict.return_value = {'param': torch.tensor([1.0])}
            trainer.value_net.state_dict.return_value = {'param': torch.tensor([2.0])}
            
            # Save checkpoint
            trainer.save_checkpoint(10)
            
            # Check file exists
            checkpoint_path = os.path.join(temp_dir, "checkpoint_10.pth")
            assert os.path.exists(checkpoint_path)
            
            # Load checkpoint
            iteration = trainer.load_checkpoint(checkpoint_path)
            assert iteration == 10


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality."""
    
    def test_search_space_creation(self):
        """Test search space creation."""
        search_space = SearchSpace()
        
        # Mock trial object
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 64
        mock_trial.suggest_categorical.return_value = 'tanh'
        
        hyperparams = search_space.suggest_hyperparameters(mock_trial)
        
        assert 'policy_lr' in hyperparams
        assert 'value_lr' in hyperparams
        assert 'batch_size' in hyperparams
        assert 'activation_function' in hyperparams
    
    @patch('optuna.create_study')
    def test_optimizer_creation(self, mock_create_study):
        """Test hyperparameter optimizer creation."""
        mock_study = Mock()
        mock_create_study.return_value = mock_study
        
        config = TrainingConfig()
        search_space = SearchSpace()
        
        optimizer = HyperparameterOptimizer(
            base_config=config,
            search_space=search_space,
            n_trials=10,
            study_name="test_study"
        )
        
        assert optimizer.base_config == config
        assert optimizer.search_space == search_space
        assert optimizer.n_trials == 10
    
    def test_multi_objective_study(self):
        """Test multi-objective optimization study."""
        study = OptimizationStudy()
        
        # Add mock trial results
        trial_results = TrainingResults(
            final_performance={'episode_return': 100.0, 'success_rate': 0.8},
            safety_metrics={'constraint_violations': 5, 'safety_margin': 0.9},
            training_metrics={'training_time': 3600.0}
        )
        
        objectives = study.add_trial_result(trial_results)
        
        assert 'performance' in objectives
        assert 'safety' in objectives
        assert 'efficiency' in objectives
        
        composite_score = study.compute_composite_score(objectives)
        assert isinstance(composite_score, float)


class TestExperimentTracking:
    """Test experiment tracking functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_experiment_metrics(self):
        """Test experiment metrics creation and conversion."""
        metrics = ExperimentMetrics(
            policy_loss=0.1,
            value_loss=0.2,
            episode_return=100.0,
            success_rate=0.8,
            constraint_violations=2
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['policy_loss'] == 0.1
        assert metrics_dict['episode_return'] == 100.0
        assert metrics_dict['constraint_violations'] == 2
    
    def test_local_tracker(self, temp_dir):
        """Test local experiment tracker."""
        config = ExperimentConfig(
            artifact_location=temp_dir,
            use_local=True
        )
        
        tracker = LocalTracker("test_experiment", config)
        
        # Test experiment start
        training_config = TrainingConfig()
        tracker.start_experiment(training_config)
        
        assert tracker.run_dir is not None
        assert tracker.run_dir.exists()
        
        # Test metrics logging
        metrics = ExperimentMetrics(episode_return=100.0, success_rate=0.8)
        tracker.log_metrics(metrics, step=1)
        
        # Check metrics file
        metrics_file = tracker.run_dir / "metrics.json"
        assert metrics_file.exists()
        
        with open(metrics_file) as f:
            logged_metrics = json.load(f)
        
        assert len(logged_metrics) == 1
        assert logged_metrics[0]['episode_return'] == 100.0
    
    def test_experiment_manager(self, temp_dir):
        """Test experiment manager functionality."""
        manager = ExperimentManager()
        
        config = ExperimentConfig(
            artifact_location=temp_dir,
            use_local=True
        )
        
        tracker = manager.create_tracker(config, "_test")
        assert isinstance(tracker, CompositeTracker)


class TestCallbacks:
    """Test training callbacks."""
    
    @pytest.fixture
    def mock_state(self):
        """Create mock callback state."""
        state = CallbackState()
        state.iteration = 10
        state.metrics = ExperimentMetrics(episode_return=100.0, success_rate=0.8)
        state.policy_net = Mock()
        state.value_net = Mock()
        return state
    
    def test_early_stopping(self, mock_state):
        """Test early stopping callback."""
        early_stopping = EarlyStopping(
            metric_name='episode_return',
            patience=3,
            mode='max'
        )
        
        early_stopping.on_training_start(mock_state)
        
        # Should not stop with improving metric
        mock_state.metrics.episode_return = 105.0
        should_stop = early_stopping.on_iteration_end(mock_state)
        assert not should_stop
        
        # Should stop after patience exhausted
        for _ in range(4):
            mock_state.metrics.episode_return = 90.0  # Worse performance
            should_stop = early_stopping.on_iteration_end(mock_state)
        
        assert should_stop
    
    def test_model_checkpoint(self, mock_state):
        """Test model checkpointing callback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = ModelCheckpoint(
                checkpoint_dir=temp_dir,
                save_frequency=5,
                save_best=True,
                metric_name='episode_return'
            )
            
            # Mock model state dicts
            mock_state.policy_net.state_dict.return_value = {'param': torch.tensor([1.0])}
            mock_state.value_net.state_dict.return_value = {'param': torch.tensor([2.0])}
            
            checkpoint.on_training_start(mock_state)
            
            # Should save at frequency
            mock_state.iteration = 4  # 5th iteration (0-indexed)
            checkpoint.on_iteration_end(mock_state)
            
            checkpoint_file = os.path.join(temp_dir, "checkpoint_iter_5.pth")
            assert os.path.exists(checkpoint_file)
    
    def test_safety_monitor(self, mock_state):
        """Test safety monitoring callback."""
        safety_monitor = SafetyMonitorCallback(
            max_violations=10,
            violation_threshold=0.5
        )
        
        safety_monitor.on_training_start(mock_state)
        
        # Normal operation
        mock_state.metrics.constraint_violations = 1
        should_stop = safety_monitor.on_iteration_end(mock_state)
        assert not should_stop
        
        # Too many violations
        safety_monitor.violation_count = 15
        should_stop = safety_monitor.on_iteration_end(mock_state)
        assert should_stop
    
    def test_callback_manager(self, mock_state):
        """Test callback manager."""
        manager = CallbackManager()
        
        # Add callbacks
        early_stopping = EarlyStopping(patience=3)
        manager.add_callback(early_stopping)
        
        # Test coordinated callbacks
        manager.on_training_start(mock_state)
        should_stop = manager.on_iteration_end(mock_state)
        
        assert not should_stop  # Should not stop immediately
        assert len(manager.callbacks) > 0


class TestEvaluation:
    """Test evaluation protocols and metrics."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock CPO agent."""
        agent = Mock()
        agent.select_action.return_value = np.array([0.5, 0.5, 0.5, 0.5])
        return agent
    
    @pytest.fixture
    def mock_environment(self):
        """Create mock environment."""
        env = Mock()
        env.reset.return_value = np.zeros(10)
        env.step.return_value = (
            np.zeros(10),  # next_state
            1.0,           # reward
            False,         # done
            {'success': True, 'constraint_cost': 0.0, 'human_effort': 0.5}  # info
        )
        return env
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics creation."""
        metrics = EvaluationMetrics(
            mean_return=100.0,
            success_rate=0.8,
            constraint_violation_rate=0.1,
            mean_human_effort=0.3
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['mean_return'] == 100.0
        assert metrics_dict['success_rate'] == 0.8
        assert metrics_dict['constraint_violation_rate'] == 0.1
    
    def test_standard_evaluation(self, mock_agent, mock_environment):
        """Test standard evaluation protocol."""
        evaluation = StandardEvaluation(num_episodes=5)
        
        # Mock environment to terminate episodes
        episode_count = 0
        def mock_step(*args):
            nonlocal episode_count
            episode_count += 1
            done = episode_count % 10 == 0  # End episode every 10 steps
            if done:
                episode_count = 0
            return (
                np.zeros(10),
                1.0,
                done,
                {'success': True, 'constraint_cost': 0.0}
            )
        
        mock_environment.step = mock_step
        
        metrics = evaluation.evaluate(mock_agent, mock_environment)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert metrics.sample_size == 5
        assert metrics.mean_return > 0
    
    def test_evaluation_manager(self, mock_agent, mock_environment):
        """Test evaluation manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = EvaluationManager(save_results=True, results_dir=temp_dir)
            
            # Mock environment behavior
            episode_count = 0
            def mock_step(*args):
                nonlocal episode_count
                episode_count += 1
                done = episode_count % 5 == 0
                if done:
                    episode_count = 0
                return (
                    np.zeros(10),
                    1.0,
                    done,
                    {'success': True, 'constraint_cost': 0.0}
                )
            
            mock_environment.step = mock_step
            
            results = manager.evaluate_model(
                mock_agent,
                mock_environment,
                protocols=['standard']
            )
            
            assert 'protocols' in results
            assert 'standard' in results['protocols']
            assert 'summary' in results


class TestSchedulers:
    """Test parameter schedulers."""
    
    def test_scheduler_config(self):
        """Test scheduler configuration."""
        config = SchedulerConfig(
            scheduler_type="linear",
            initial_value=0.001,
            final_value=0.0001,
            total_steps=1000
        )
        
        assert config.scheduler_type == "linear"
        assert config.initial_value == 0.001
        assert config.final_value == 0.0001
    
    def test_linear_scheduler(self):
        """Test linear parameter scheduler."""
        config = SchedulerConfig(
            scheduler_type="linear",
            initial_value=1.0,
            final_value=0.0,
            total_steps=10
        )
        
        scheduler = LinearScheduler(config)
        
        # First step
        value = scheduler.step()
        assert value == 0.9  # (1.0 - 0.0) / 10 = 0.1 decrease per step
        
        # After all steps
        for _ in range(9):
            value = scheduler.step()
        
        assert abs(value - 0.0) < 1e-6
    
    def test_exponential_scheduler(self):
        """Test exponential parameter scheduler."""
        config = SchedulerConfig(
            scheduler_type="exponential",
            initial_value=1.0,
            decay_rate=0.9
        )
        
        scheduler = ExponentialScheduler(config)
        
        # First step
        value = scheduler.step()
        assert abs(value - 0.9) < 1e-6
        
        # Second step
        value = scheduler.step()
        assert abs(value - 0.81) < 1e-6
    
    def test_scheduler_manager(self):
        """Test scheduler manager."""
        manager = SchedulerManager()
        
        # Add scheduler
        config = SchedulerConfig(
            scheduler_type="linear",
            initial_value=0.001,
            final_value=0.0001,
            total_steps=100
        )
        
        manager.add_parameter_scheduler("learning_rate", config)
        
        # Step all schedulers
        values = manager.step_all()
        
        assert "learning_rate" in values
        assert values["learning_rate"] < 0.001  # Should decrease
    
    def test_common_schedulers_creation(self):
        """Test creation of common schedulers."""
        # Mock optimizers
        policy_optimizer = Mock()
        policy_optimizer.param_groups = [{'lr': 0.001}]
        
        value_optimizer = Mock()
        value_optimizer.param_groups = [{'lr': 0.001}]
        
        manager = create_common_schedulers(policy_optimizer, value_optimizer)
        
        assert "policy" in manager.lr_schedulers
        assert "value" in manager.lr_schedulers
        assert "constraint_threshold" in manager.constraint_schedulers


class TestDistributedTraining:
    """Test distributed training functionality."""
    
    def test_distributed_config(self):
        """Test distributed configuration."""
        config = DistributedConfig(
            use_distributed=True,
            world_size=2,
            rank=0,
            backend="gloo"
        )
        
        assert config.use_distributed
        assert config.world_size == 2
        assert config.rank == 0
        assert config.backend == "gloo"
    
    def test_distributed_manager(self):
        """Test distributed manager (without actual distributed setup)."""
        config = DistributedConfig(use_distributed=False)
        manager = DistributedManager(config)
        
        # Should not initialize distributed training
        manager.setup()
        assert not manager.is_initialized
        
        # Should be main process when not distributed
        assert manager.is_main_process()
        assert manager.get_rank() == 0
        assert manager.get_world_size() == 1
    
    def test_distributed_launcher_port_finding(self):
        """Test finding free port for distributed training."""
        port = DistributedLauncher.get_free_port()
        
        assert isinstance(port, str)
        assert int(port) > 0
        assert int(port) < 65536
    
    @patch('torch.cuda.is_available')
    def test_single_node_multiprocess_config(self, mock_cuda_available):
        """Test single-node multiprocess configuration."""
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.device_count', return_value=2):
            config = DistributedLauncher.setup_single_node_multiprocess()
            
            assert config.use_distributed
            assert config.world_size == 2
            assert config.master_addr == "localhost"
            assert config.backend == "nccl"


class TestIntegration:
    """Integration tests for the complete training system."""
    
    @pytest.fixture
    def complete_setup(self):
        """Set up complete training environment."""
        temp_dir = tempfile.mkdtemp()
        
        # Configuration
        config = TrainingConfig(
            max_iterations=3,
            rollout_length=50,
            batch_size=16,
            save_dir=temp_dir,
            save_frequency=1
        )
        
        # Mock environment
        env = Mock()
        env.observation_space.shape = (10,)
        env.action_space.shape = (4,)
        env.reset.return_value = np.zeros(10)
        
        episode_step = 0
        def mock_step(*args):
            nonlocal episode_step
            episode_step += 1
            done = episode_step >= 20  # End episode after 20 steps
            if done:
                episode_step = 0
            return (
                np.zeros(10),
                1.0,
                done,
                {'success': True, 'constraint_cost': 0.0}
            )
        
        env.step = mock_step
        env.get_safety_constraints.return_value = {'collision': 0.1}
        
        yield config, env, temp_dir
        
        shutil.rmtree(temp_dir)
    
    def test_full_training_pipeline(self, complete_setup):
        """Test complete training pipeline integration."""
        config, env, temp_dir = complete_setup
        
        with patch('safe_rl_human_robot.src.algorithms.cpo.CPOAgent') as MockAgent:
            # Mock agent
            mock_agent = Mock()
            mock_agent.select_action.return_value = np.array([0.5, 0.5, 0.5, 0.5])
            mock_agent.train_policy.return_value = {'policy_loss': 0.1}
            mock_agent.train_value_function.return_value = {'value_loss': 0.2}
            mock_agent.compute_advantages.return_value = (
                np.ones(50), np.ones(50), np.ones(50)
            )
            MockAgent.return_value = mock_agent
            
            # Mock networks for checkpointing
            mock_policy = Mock()
            mock_policy.state_dict.return_value = {'param': torch.tensor([1.0])}
            mock_value = Mock()
            mock_value.state_dict.return_value = {'param': torch.tensor([2.0])}
            
            # Create trainer
            trainer = CPOTrainer(config, env)
            trainer.policy_net = mock_policy
            trainer.value_net = mock_value
            trainer.policy_optimizer = Mock()
            trainer.value_optimizer = Mock()
            
            # Run training
            results = trainer.train()
            
            # Verify results
            assert isinstance(results, TrainingResults)
            assert results.total_iterations == config.max_iterations
            assert len(results.training_history) > 0
            
            # Verify checkpoints were saved
            checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith('checkpoint_')]
            assert len(checkpoint_files) >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])