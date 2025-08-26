"""
Enterprise Model Registry for Safe RL Production System.

This module provides comprehensive model lifecycle management including:
- Model versioning and metadata tracking
- Model validation and testing
- A/B testing and gradual rollout
- Model performance monitoring
- Model governance and compliance
"""

import asyncio
import logging
import json
import hashlib
import pickle
import torch
import mlflow
import boto3
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Float, Boolean, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
import docker
from kubernetes import client, config as k8s_config
import semver
from cryptography.fernet import Fernet
import joblib
from concurrent.futures import ThreadPoolExecutor
import shutil

logger = logging.getLogger(__name__)

Base = declarative_base()


class ModelStatus(Enum):
    """Model lifecycle status."""
    REGISTERED = "registered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TEST = "a_b_test"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    safety_violations: Optional[float] = None
    constraint_satisfaction: Optional[float] = None
    inference_latency_p95: Optional[float] = None
    inference_latency_p99: Optional[float] = None
    throughput: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""
    name: str
    version: str
    algorithm: str
    framework: str  # torch, tensorflow, sklearn, etc.
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Model specifications
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    
    # Compliance and governance
    approval_status: str = "pending"
    approved_by: str = ""
    approval_date: Optional[datetime] = None
    compliance_checks: Dict[str, bool] = field(default_factory=dict)
    
    # Dependencies and requirements
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ""
    system_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage tracking
    parent_models: List[str] = field(default_factory=list)
    training_data_version: str = ""
    training_run_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'algorithm': self.algorithm,
            'framework': self.framework,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'description': self.description,
            'tags': self.tags,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
            'hyperparameters': self.hyperparameters,
            'training_config': self.training_config,
            'metrics': self.metrics.to_dict(),
            'approval_status': self.approval_status,
            'approved_by': self.approved_by,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'compliance_checks': self.compliance_checks,
            'dependencies': self.dependencies,
            'python_version': self.python_version,
            'system_requirements': self.system_requirements,
            'parent_models': self.parent_models,
            'training_data_version': self.training_data_version,
            'training_run_id': self.training_run_id
        }


@dataclass
class DeploymentConfig:
    """Model deployment configuration."""
    strategy: DeploymentStrategy
    target_environment: str  # staging, production
    traffic_percentage: float = 100.0  # For canary/A-B testing
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    health_check_config: Dict[str, Any] = field(default_factory=dict)
    rollback_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)


class ModelStorage(ABC):
    """Abstract base class for model storage backends."""
    
    @abstractmethod
    async def store_model(self, model: Any, metadata: ModelMetadata, artifacts: Dict[str, Any] = None) -> str:
        """Store model with metadata."""
        pass
    
    @abstractmethod
    async def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata."""
        pass
    
    @abstractmethod
    async def delete_model(self, model_id: str) -> bool:
        """Delete model and associated artifacts."""
        pass
    
    @abstractmethod
    async def list_models(self, filters: Dict[str, Any] = None) -> List[ModelMetadata]:
        """List models with optional filtering."""
        pass


class MLflowModelStorage(ModelStorage):
    """MLflow-based model storage implementation."""
    
    def __init__(self, tracking_uri: str, s3_bucket: Optional[str] = None):
        self.tracking_uri = tracking_uri
        self.s3_bucket = s3_bucket
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    async def store_model(self, model: Any, metadata: ModelMetadata, artifacts: Dict[str, Any] = None) -> str:
        """Store model in MLflow."""
        try:
            # Create or get experiment
            experiment_name = f"safe_rl_{metadata.algorithm}"
            try:
                experiment_id = self.client.create_experiment(experiment_name)
            except mlflow.exceptions.MlflowException:
                experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log model based on framework
                if metadata.framework == 'torch':
                    mlflow.pytorch.log_model(model, "model")
                elif metadata.framework == 'sklearn':
                    mlflow.sklearn.log_model(model, "model")
                else:
                    # Generic Python model
                    mlflow.pyfunc.log_model("model", python_model=model)
                
                # Log metadata as parameters and tags
                mlflow.log_params(metadata.hyperparameters)
                mlflow.log_params(metadata.training_config)
                
                # Log metrics
                for key, value in metadata.metrics.to_dict().items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                # Log tags
                for tag in metadata.tags:
                    mlflow.set_tag(f"tag_{tag}", True)
                
                mlflow.set_tag("algorithm", metadata.algorithm)
                mlflow.set_tag("version", metadata.version)
                mlflow.set_tag("created_by", metadata.created_by)
                
                # Log additional artifacts
                if artifacts:
                    for name, artifact in artifacts.items():
                        if isinstance(artifact, (str, Path)):
                            mlflow.log_artifact(str(artifact), name)
                        else:
                            # Save as pickle and log
                            artifact_path = f"/tmp/{name}.pkl"
                            with open(artifact_path, 'wb') as f:
                                pickle.dump(artifact, f)
                            mlflow.log_artifact(artifact_path, name)
                
                # Register model in model registry
                model_uri = f"runs:/{run.info.run_id}/model"
                registered_model = mlflow.register_model(
                    model_uri, 
                    f"{metadata.name}_{metadata.algorithm}"
                )
                
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Failed to store model in MLflow: {e}")
            raise
    
    async def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load model from MLflow."""
        try:
            # Load model
            model_uri = f"runs:/{model_id}/model"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Get run information
            run = self.client.get_run(model_id)
            
            # Reconstruct metadata
            metadata = ModelMetadata(
                name=run.data.tags.get('mlflow.runName', 'unknown'),
                version=run.data.tags.get('version', '1.0.0'),
                algorithm=run.data.tags.get('algorithm', 'unknown'),
                framework='mlflow',  # Will be detected from model
                created_at=datetime.fromtimestamp(run.info.start_time / 1000),
                created_by=run.data.tags.get('created_by', 'unknown'),
                hyperparameters=run.data.params,
                training_run_id=model_id
            )
            
            # Reconstruct metrics
            metrics = ModelMetrics()
            for key, value in run.data.metrics.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
                else:
                    metrics.custom_metrics[key] = value
            metadata.metrics = metrics
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model from MLflow: {e}")
            raise
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete model from MLflow."""
        try:
            # Delete run
            self.client.delete_run(model_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete model from MLflow: {e}")
            return False
    
    async def list_models(self, filters: Dict[str, Any] = None) -> List[ModelMetadata]:
        """List models from MLflow."""
        try:
            # Get all experiments
            experiments = self.client.search_experiments()
            models = []
            
            for experiment in experiments:
                # Search runs in experiment
                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    max_results=1000
                )
                
                for run in runs:
                    # Create metadata from run
                    metadata = ModelMetadata(
                        name=run.data.tags.get('mlflow.runName', 'unknown'),
                        version=run.data.tags.get('version', '1.0.0'),
                        algorithm=run.data.tags.get('algorithm', 'unknown'),
                        framework='mlflow',
                        created_at=datetime.fromtimestamp(run.info.start_time / 1000),
                        created_by=run.data.tags.get('created_by', 'unknown'),
                        hyperparameters=run.data.params,
                        training_run_id=run.info.run_id
                    )
                    
                    # Apply filters if provided
                    if filters:
                        matches = True
                        for key, value in filters.items():
                            if hasattr(metadata, key):
                                if getattr(metadata, key) != value:
                                    matches = False
                                    break
                            elif key in run.data.tags:
                                if run.data.tags[key] != str(value):
                                    matches = False
                                    break
                        
                        if not matches:
                            continue
                    
                    models.append(metadata)
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models from MLflow: {e}")
            return []


class ModelValidator:
    """Validates models before deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_tests = []
    
    def add_validation_test(self, test_func: callable):
        """Add a custom validation test."""
        self.validation_tests.append(test_func)
    
    async def validate_model(self, model: Any, metadata: ModelMetadata, 
                           test_data: Optional[Any] = None) -> Dict[str, Any]:
        """Comprehensive model validation."""
        validation_results = {
            'passed': True,
            'tests': {},
            'metrics': {},
            'issues': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Basic model tests
            await self._test_model_loadable(model, validation_results)
            await self._test_model_inference(model, metadata, validation_results)
            await self._test_input_validation(model, metadata, validation_results)
            await self._test_performance_requirements(model, metadata, validation_results)
            
            # Safety-specific tests
            await self._test_safety_constraints(model, validation_results)
            await self._test_robustness(model, validation_results)
            
            # Business logic tests
            if test_data is not None:
                await self._test_accuracy(model, test_data, validation_results)
                await self._test_bias_fairness(model, test_data, validation_results)
            
            # Custom validation tests
            for test_func in self.validation_tests:
                try:
                    test_name = test_func.__name__
                    result = await test_func(model, metadata, test_data)
                    validation_results['tests'][test_name] = result
                    if not result.get('passed', True):
                        validation_results['passed'] = False
                        validation_results['issues'].append(f"Custom test '{test_name}' failed: {result.get('message', 'Unknown error')}")
                except Exception as e:
                    validation_results['tests'][test_func.__name__] = {'passed': False, 'error': str(e)}
                    validation_results['passed'] = False
                    validation_results['issues'].append(f"Custom test '{test_func.__name__}' error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            validation_results['passed'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results
    
    async def _test_model_loadable(self, model: Any, results: Dict[str, Any]):
        """Test if model can be loaded and initialized."""
        try:
            # Try to access model attributes/methods
            if hasattr(model, 'predict'):
                results['tests']['model_loadable'] = {'passed': True, 'message': 'Model is loadable'}
            else:
                results['tests']['model_loadable'] = {'passed': False, 'message': 'Model missing predict method'}
                results['passed'] = False
                results['issues'].append('Model does not have predict method')
        except Exception as e:
            results['tests']['model_loadable'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
            results['issues'].append(f'Model loading error: {str(e)}')
    
    async def _test_model_inference(self, model: Any, metadata: ModelMetadata, results: Dict[str, Any]):
        """Test model inference with dummy data."""
        try:
            # Generate dummy input based on schema
            dummy_input = self._generate_dummy_input(metadata.input_schema)
            
            # Time inference
            start_time = time.time()
            prediction = model.predict(dummy_input)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            results['tests']['model_inference'] = {
                'passed': True, 
                'inference_time_ms': inference_time,
                'output_shape': getattr(prediction, 'shape', 'unknown')
            }
            results['metrics']['inference_time_ms'] = inference_time
            
        except Exception as e:
            results['tests']['model_inference'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
            results['issues'].append(f'Model inference error: {str(e)}')
    
    async def _test_input_validation(self, model: Any, metadata: ModelMetadata, results: Dict[str, Any]):
        """Test model input validation."""
        try:
            # Test with invalid input shapes/types
            test_cases = [
                np.random.rand(1, 1),  # Wrong shape
                "invalid_input",       # Wrong type
                None,                  # None input
                np.array([]),         # Empty array
            ]
            
            robust_to_invalid_input = True
            for i, invalid_input in enumerate(test_cases):
                try:
                    model.predict(invalid_input)
                    # If no exception, check if output makes sense
                    logger.warning(f"Model accepted invalid input case {i}")
                except (ValueError, TypeError, RuntimeError) as e:
                    # Expected behavior - model should reject invalid input
                    continue
                except Exception as e:
                    # Unexpected error
                    robust_to_invalid_input = False
                    break
            
            results['tests']['input_validation'] = {
                'passed': robust_to_invalid_input,
                'message': 'Model handles invalid inputs appropriately' if robust_to_invalid_input else 'Model does not handle invalid inputs properly'
            }
            
            if not robust_to_invalid_input:
                results['passed'] = False
                results['issues'].append('Model does not handle invalid inputs properly')
                
        except Exception as e:
            results['tests']['input_validation'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
            results['issues'].append(f'Input validation test error: {str(e)}')
    
    async def _test_performance_requirements(self, model: Any, metadata: ModelMetadata, results: Dict[str, Any]):
        """Test model against performance requirements."""
        try:
            requirements = self.config.get('performance_requirements', {})
            max_inference_time = requirements.get('max_inference_time_ms', 1000)
            
            # Run multiple inference tests
            inference_times = []
            dummy_input = self._generate_dummy_input(metadata.input_schema)
            
            for _ in range(10):
                start_time = time.time()
                model.predict(dummy_input)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
            
            avg_time = np.mean(inference_times)
            p95_time = np.percentile(inference_times, 95)
            
            performance_ok = p95_time <= max_inference_time
            
            results['tests']['performance_requirements'] = {
                'passed': performance_ok,
                'avg_inference_time_ms': avg_time,
                'p95_inference_time_ms': p95_time,
                'max_allowed_ms': max_inference_time
            }
            
            results['metrics']['avg_inference_time_ms'] = avg_time
            results['metrics']['p95_inference_time_ms'] = p95_time
            
            if not performance_ok:
                results['passed'] = False
                results['issues'].append(f'Model too slow: {p95_time:.2f}ms > {max_inference_time}ms')
                
        except Exception as e:
            results['tests']['performance_requirements'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
            results['issues'].append(f'Performance test error: {str(e)}')
    
    async def _test_safety_constraints(self, model: Any, results: Dict[str, Any]):
        """Test safety-specific constraints."""
        try:
            # Generate test cases for safety
            dummy_inputs = [self._generate_dummy_input() for _ in range(100)]
            
            safety_violations = 0
            for dummy_input in dummy_inputs:
                try:
                    prediction = model.predict(dummy_input)
                    
                    # Check for safety violations (customize based on domain)
                    if hasattr(prediction, '__iter__'):
                        # Check if prediction is within safe bounds
                        prediction_array = np.array(prediction)
                        if np.any(np.isnan(prediction_array)) or np.any(np.isinf(prediction_array)):
                            safety_violations += 1
                        elif np.any(np.abs(prediction_array) > 10):  # Example safety bound
                            safety_violations += 1
                            
                except Exception:
                    safety_violations += 1
            
            safety_rate = 1.0 - (safety_violations / len(dummy_inputs))
            min_safety_rate = self.config.get('min_safety_rate', 0.95)
            
            safety_ok = safety_rate >= min_safety_rate
            
            results['tests']['safety_constraints'] = {
                'passed': safety_ok,
                'safety_rate': safety_rate,
                'violations': safety_violations,
                'total_tests': len(dummy_inputs)
            }
            
            results['metrics']['safety_rate'] = safety_rate
            
            if not safety_ok:
                results['passed'] = False
                results['issues'].append(f'Safety rate too low: {safety_rate:.3f} < {min_safety_rate}')
                
        except Exception as e:
            results['tests']['safety_constraints'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
            results['issues'].append(f'Safety test error: {str(e)}')
    
    async def _test_robustness(self, model: Any, results: Dict[str, Any]):
        """Test model robustness to input perturbations."""
        try:
            base_input = self._generate_dummy_input()
            base_prediction = model.predict(base_input)
            
            # Add noise and test consistency
            noise_levels = [0.01, 0.05, 0.1]
            consistency_scores = []
            
            for noise_level in noise_levels:
                predictions = []
                for _ in range(10):
                    noisy_input = base_input + np.random.normal(0, noise_level, base_input.shape)
                    try:
                        pred = model.predict(noisy_input)
                        predictions.append(pred)
                    except Exception:
                        predictions.append(None)
                
                # Calculate consistency (how similar predictions are)
                valid_predictions = [p for p in predictions if p is not None]
                if valid_predictions:
                    pred_array = np.array(valid_predictions)
                    consistency = 1.0 - np.std(pred_array) / (np.mean(np.abs(pred_array)) + 1e-8)
                    consistency_scores.append(max(0, consistency))
                else:
                    consistency_scores.append(0)
            
            avg_consistency = np.mean(consistency_scores)
            min_consistency = self.config.get('min_robustness', 0.8)
            
            robust_enough = avg_consistency >= min_consistency
            
            results['tests']['robustness'] = {
                'passed': robust_enough,
                'consistency_score': avg_consistency,
                'noise_levels_tested': noise_levels
            }
            
            results['metrics']['robustness_score'] = avg_consistency
            
            if not robust_enough:
                results['passed'] = False
                results['issues'].append(f'Model not robust enough: {avg_consistency:.3f} < {min_consistency}')
                
        except Exception as e:
            results['tests']['robustness'] = {'passed': False, 'error': str(e)}
            results['passed'] = False
            results['issues'].append(f'Robustness test error: {str(e)}')
    
    async def _test_accuracy(self, model: Any, test_data: Any, results: Dict[str, Any]):
        """Test model accuracy on test data."""
        try:
            # Assume test_data is tuple of (X, y)
            X_test, y_test = test_data
            predictions = model.predict(X_test)
            
            # Calculate accuracy metrics (customize based on task type)
            if len(y_test.shape) == 1 or y_test.shape[1] == 1:
                # Classification or regression
                mse = np.mean((predictions - y_test) ** 2)
                mae = np.mean(np.abs(predictions - y_test))
                
                results['tests']['accuracy'] = {
                    'passed': True,
                    'mse': float(mse),
                    'mae': float(mae)
                }
                
                results['metrics']['test_mse'] = mse
                results['metrics']['test_mae'] = mae
            
        except Exception as e:
            results['tests']['accuracy'] = {'passed': False, 'error': str(e)}
            logger.warning(f'Accuracy test error: {str(e)}')
    
    async def _test_bias_fairness(self, model: Any, test_data: Any, results: Dict[str, Any]):
        """Test model for bias and fairness."""
        try:
            # This is a placeholder for bias testing
            # In practice, this would involve checking performance across different demographic groups
            results['tests']['bias_fairness'] = {
                'passed': True,
                'message': 'Bias testing not implemented - requires domain-specific logic'
            }
            
        except Exception as e:
            results['tests']['bias_fairness'] = {'passed': False, 'error': str(e)}
    
    def _generate_dummy_input(self, input_schema: Dict[str, Any] = None):
        """Generate dummy input for testing."""
        if input_schema and 'shape' in input_schema:
            shape = input_schema['shape']
            return np.random.rand(*shape)
        else:
            # Default dummy input
            return np.random.rand(1, 10)  # Assume 10-dimensional input


class ModelRegistry:
    """Enterprise model registry with full lifecycle management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_backend = None
        self.validator = None
        self.db_engine = None
        self.session_factory = None
        
        # Initialize components
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize registry components."""
        # Initialize storage backend
        storage_config = self.config.get('storage', {})
        storage_type = storage_config.get('type', 'mlflow')
        
        if storage_type == 'mlflow':
            self.storage_backend = MLflowModelStorage(
                tracking_uri=storage_config.get('tracking_uri', 'http://localhost:5000'),
                s3_bucket=storage_config.get('s3_bucket')
            )
        
        # Initialize validator
        validation_config = self.config.get('validation', {})
        self.validator = ModelValidator(validation_config)
        
        # Initialize database for metadata
        db_config = self.config.get('database', {})
        if db_config:
            self.db_engine = create_engine(db_config['url'])
            self.session_factory = sessionmaker(bind=self.db_engine)
        
        logger.info("Model registry initialized")
    
    async def register_model(self, model: Any, metadata: ModelMetadata, 
                           artifacts: Dict[str, Any] = None, validate: bool = True) -> str:
        """Register a new model."""
        try:
            logger.info(f"Registering model: {metadata.name} v{metadata.version}")
            
            # Validate model if requested
            if validate:
                validation_results = await self.validator.validate_model(model, metadata)
                if not validation_results['passed']:
                    raise ValueError(f"Model validation failed: {validation_results['issues']}")
                
                # Update metadata with validation metrics
                for key, value in validation_results['metrics'].items():
                    if hasattr(metadata.metrics, key):
                        setattr(metadata.metrics, key, value)
                    else:
                        metadata.metrics.custom_metrics[key] = value
            
            # Generate version if not provided
            if not metadata.version:
                metadata.version = await self._generate_version(metadata.name)
            
            # Store model
            model_id = await self.storage_backend.store_model(model, metadata, artifacts)
            
            # Update metadata with model ID
            metadata.training_run_id = model_id
            
            # Store metadata in database
            if self.session_factory:
                await self._store_metadata_in_db(metadata, model_id)
            
            logger.info(f"Successfully registered model: {metadata.name} v{metadata.version} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def get_model(self, name: str, version: str = None, stage: str = None) -> Tuple[Any, ModelMetadata]:
        """Get model by name and version/stage."""
        try:
            # Find model ID
            filters = {'name': name}
            if version:
                filters['version'] = version
            
            models = await self.storage_backend.list_models(filters)
            
            if not models:
                raise ValueError(f"Model not found: {name}")
            
            # If multiple models found, select based on criteria
            if len(models) > 1:
                if stage:
                    # Filter by stage if provided
                    stage_models = [m for m in models if m.approval_status == stage]
                    if stage_models:
                        models = stage_models
                
                # Sort by version (semantic versioning)
                models.sort(key=lambda m: semver.VersionInfo.parse(m.version), reverse=True)
            
            selected_model = models[0]
            
            # Load model
            model, metadata = await self.storage_backend.load_model(selected_model.training_run_id)
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to get model {name}: {e}")
            raise
    
    async def list_models(self, filters: Dict[str, Any] = None) -> List[ModelMetadata]:
        """List all models with optional filtering."""
        return await self.storage_backend.list_models(filters)
    
    async def delete_model(self, name: str, version: str) -> bool:
        """Delete a model."""
        try:
            # Find model
            models = await self.storage_backend.list_models({'name': name, 'version': version})
            if not models:
                return False
            
            model_metadata = models[0]
            
            # Delete from storage
            success = await self.storage_backend.delete_model(model_metadata.training_run_id)
            
            # Delete metadata from database
            if success and self.session_factory:
                await self._delete_metadata_from_db(model_metadata.training_run_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete model {name} v{version}: {e}")
            return False
    
    async def promote_model(self, name: str, version: str, stage: str) -> bool:
        """Promote model to a different stage."""
        try:
            # Find model
            models = await self.storage_backend.list_models({'name': name, 'version': version})
            if not models:
                return False
            
            model_metadata = models[0]
            
            # Update stage
            model_metadata.approval_status = stage
            model_metadata.approval_date = datetime.now()
            
            # Update in database
            if self.session_factory:
                await self._update_metadata_in_db(model_metadata)
            
            logger.info(f"Promoted model {name} v{version} to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {name} v{version}: {e}")
            return False
    
    async def compare_models(self, model_specs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Compare multiple models."""
        try:
            comparison_results = {
                'models': {},
                'comparison_metrics': {},
                'recommendations': []
            }
            
            models_data = []
            for name, version in model_specs:
                try:
                    model, metadata = await self.get_model(name, version)
                    models_data.append((model, metadata))
                    comparison_results['models'][f"{name}_v{version}"] = metadata.to_dict()
                except Exception as e:
                    logger.error(f"Failed to load model {name} v{version}: {e}")
                    continue
            
            if len(models_data) < 2:
                raise ValueError("Need at least 2 models to compare")
            
            # Compare metrics
            metric_names = set()
            for _, metadata in models_data:
                metric_names.update(metadata.metrics.to_dict().keys())
            
            for metric_name in metric_names:
                metric_values = {}
                for _, metadata in models_data:
                    model_key = f"{metadata.name}_v{metadata.version}"
                    metric_dict = metadata.metrics.to_dict()
                    metric_values[model_key] = metric_dict.get(metric_name)
                
                comparison_results['comparison_metrics'][metric_name] = metric_values
            
            # Generate recommendations
            comparison_results['recommendations'] = self._generate_model_recommendations(models_data)
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise
    
    async def _generate_version(self, model_name: str) -> str:
        """Generate next version number for model."""
        try:
            existing_models = await self.storage_backend.list_models({'name': model_name})
            
            if not existing_models:
                return "1.0.0"
            
            # Find highest version
            versions = [m.version for m in existing_models if m.version]
            versions.sort(key=lambda v: semver.VersionInfo.parse(v), reverse=True)
            
            if versions:
                latest_version = semver.VersionInfo.parse(versions[0])
                return str(latest_version.bump_minor())  # Increment minor version
            else:
                return "1.0.0"
                
        except Exception as e:
            logger.warning(f"Failed to generate version for {model_name}: {e}")
            return "1.0.0"
    
    async def _store_metadata_in_db(self, metadata: ModelMetadata, model_id: str):
        """Store model metadata in database."""
        if not self.session_factory:
            return
        
        session = self.session_factory()
        try:
            # This would require defining proper SQLAlchemy models
            # Simplified implementation
            logger.debug(f"Storing metadata for model {model_id} in database")
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to store metadata in database: {e}")
        finally:
            session.close()
    
    async def _update_metadata_in_db(self, metadata: ModelMetadata):
        """Update model metadata in database."""
        if not self.session_factory:
            return
        
        session = self.session_factory()
        try:
            logger.debug(f"Updating metadata for model {metadata.name} in database")
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update metadata in database: {e}")
        finally:
            session.close()
    
    async def _delete_metadata_from_db(self, model_id: str):
        """Delete model metadata from database."""
        if not self.session_factory:
            return
        
        session = self.session_factory()
        try:
            logger.debug(f"Deleting metadata for model {model_id} from database")
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete metadata from database: {e}")
        finally:
            session.close()
    
    def _generate_model_recommendations(self, models_data: List[Tuple[Any, ModelMetadata]]) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        # Simple recommendation logic
        if len(models_data) >= 2:
            # Compare safety metrics
            safety_scores = []
            for _, metadata in models_data:
                safety_score = metadata.metrics.constraint_satisfaction or 0
                safety_scores.append((metadata.name, metadata.version, safety_score))
            
            # Recommend model with highest safety score
            safety_scores.sort(key=lambda x: x[2], reverse=True)
            best_safety = safety_scores[0]
            recommendations.append(f"For safety: {best_safety[0]} v{best_safety[1]} (score: {best_safety[2]:.3f})")
            
            # Compare performance metrics
            perf_scores = []
            for _, metadata in models_data:
                # Use accuracy or f1_score if available
                perf_score = metadata.metrics.accuracy or metadata.metrics.f1_score or 0
                perf_scores.append((metadata.name, metadata.version, perf_score))
            
            perf_scores.sort(key=lambda x: x[2], reverse=True)
            best_perf = perf_scores[0]
            recommendations.append(f"For performance: {best_perf[0]} v{best_perf[1]} (score: {best_perf[2]:.3f})")
        
        return recommendations


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'storage': {
            'type': 'mlflow',
            'tracking_uri': 'http://localhost:5000',
            's3_bucket': 'model-artifacts'
        },
        'validation': {
            'performance_requirements': {
                'max_inference_time_ms': 100
            },
            'min_safety_rate': 0.95,
            'min_robustness': 0.8
        },
        'database': {
            'url': 'postgresql://user:pass@localhost:5432/model_registry'
        }
    }
    
    async def main():
        # Create registry
        registry = ModelRegistry(config)
        
        # Example model registration
        dummy_model = lambda x: np.random.rand(len(x), 1)  # Dummy model
        
        metadata = ModelMetadata(
            name="safe_rl_controller",
            version="1.0.0",
            algorithm="SAC_Lagrangian",
            framework="torch",
            description="Safe RL controller for human-robot interaction"
        )
        
        # Register model
        model_id = await registry.register_model(dummy_model, metadata)
        print(f"Registered model with ID: {model_id}")
        
        # List models
        models = await registry.list_models()
        print(f"Found {len(models)} models")
    
    asyncio.run(main())