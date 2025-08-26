"""
Production Configuration Management System for Safe RL Human-Robot Shared Control.

Comprehensive configuration management with environment-specific settings,
validation, deployment detection, and robot-specific parameter management.
"""

import os
import sys
import json
import yaml
import logging
import socket
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Type
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from abc import ABC, abstractmethod
import copy
from datetime import datetime
import hashlib

# Configuration validation imports
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

try:
    import pydantic
    from pydantic import BaseModel, validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object


class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()
    SIMULATION = auto()
    
    @classmethod
    def from_string(cls, env_str: str) -> 'Environment':
        """Convert string to Environment enum."""
        env_map = {
            'dev': cls.DEVELOPMENT,
            'development': cls.DEVELOPMENT,
            'test': cls.TESTING,
            'testing': cls.TESTING,
            'stage': cls.STAGING,
            'staging': cls.STAGING,
            'prod': cls.PRODUCTION,
            'production': cls.PRODUCTION,
            'sim': cls.SIMULATION,
            'simulation': cls.SIMULATION
        }
        return env_map.get(env_str.lower(), cls.DEVELOPMENT)


class ConfigValidationLevel(Enum):
    """Configuration validation strictness levels."""
    MINIMAL = auto()      # Basic type checking
    STANDARD = auto()     # Type + range checking
    STRICT = auto()       # Full validation + consistency
    PRODUCTION = auto()   # Strict + security checks


class RobotType(Enum):
    """Robot hardware types."""
    EXOSKELETON = auto()
    WHEELCHAIR = auto()
    ROBOTIC_ARM = auto()
    MOBILE_BASE = auto()
    HYBRID_SYSTEM = auto()
    SIMULATION = auto()


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    validation_level: ConfigValidationLevel = ConfigValidationLevel.STANDARD
    timestamp: datetime = field(default_factory=datetime.now)
    config_hash: str = ""
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'validation_level': self.validation_level.name,
            'timestamp': self.timestamp.isoformat(),
            'config_hash': self.config_hash
        }


@dataclass
class SafetyConfig:
    """Safety system configuration."""
    emergency_stop_enabled: bool = True
    watchdog_timeout_sec: float = 2.0
    safety_monitoring_rate_hz: float = 100.0
    force_limits_enabled: bool = True
    position_limits_enabled: bool = True
    velocity_limits_enabled: bool = True
    redundant_checking: bool = True
    fail_safe_mode: str = "emergency_stop"  # emergency_stop, safe_position, shutdown
    
    # Force limits (N)
    max_force_x: float = 50.0
    max_force_y: float = 50.0
    max_force_z: float = 100.0
    max_torque_x: float = 20.0
    max_torque_y: float = 20.0
    max_torque_z: float = 10.0


@dataclass
class HardwareConfig:
    """Hardware-specific configuration."""
    robot_type: str = "exoskeleton"
    hardware_id: str = "safe_rl_robot_001"
    communication_protocol: str = "ros"  # ros, serial, can, ethernet
    control_frequency_hz: float = 100.0
    
    # Joint configuration
    num_joints: int = 6
    joint_names: List[str] = field(default_factory=lambda: [
        "shoulder_pitch", "shoulder_roll", "elbow", 
        "wrist_pitch", "wrist_roll", "gripper"
    ])
    
    # Position limits (radians)
    position_limits_min: List[float] = field(default_factory=lambda: [-1.5] * 6)
    position_limits_max: List[float] = field(default_factory=lambda: [1.5] * 6)
    
    # Velocity limits (rad/s)
    velocity_limits: List[float] = field(default_factory=lambda: [2.0] * 6)
    
    # Effort limits (Nm)
    effort_limits: List[float] = field(default_factory=lambda: [100.0] * 6)
    
    # Communication settings
    ros_namespace: str = "/safe_rl"
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 115200
    can_interface: str = "can0"
    ethernet_address: str = "192.168.1.100"
    ethernet_port: int = 8080


@dataclass  
class AlgorithmConfig:
    """Safe RL algorithm configuration."""
    algorithm_type: str = "cpo"  # cpo, trpo, ppo_lagrangian
    policy_network_type: str = "mlp"  # mlp, rnn, transformer
    
    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation_function: str = "tanh"
    learning_rate: float = 3e-4
    
    # Training parameters
    batch_size: int = 2048
    num_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Safety constraints
    cost_limit: float = 25.0
    constraint_violation_penalty: float = 100.0
    safety_margin: float = 0.1
    
    # CPO specific
    cpo_damping_coeff: float = 0.1
    cpo_max_kl: float = 0.01
    
    # Environment interaction
    max_episode_steps: int = 1000
    num_parallel_envs: int = 8
    total_training_steps: int = 1000000


@dataclass
class UserPreferences:
    """User-specific preferences."""
    user_id: str = "default_user"
    preferred_assistance_level: float = 0.5  # 0.0 = minimal, 1.0 = maximum
    preferred_control_mode: str = "shared"  # human, shared, autonomous
    
    # Interface preferences
    feedback_type: str = "haptic"  # haptic, visual, audio
    feedback_intensity: float = 0.7
    
    # Safety preferences
    conservative_mode: bool = False
    custom_safety_limits: Dict[str, float] = field(default_factory=dict)
    
    # Personalization
    adaptation_enabled: bool = True
    learning_from_demonstrations: bool = True
    user_model_path: str = ""


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_path: str = "logs/safe_rl.log"
    max_log_file_size_mb: int = 100
    backup_count: int = 5
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_rate_hz: float = 10.0
    
    # Data logging
    log_sensor_data: bool = True
    log_control_commands: bool = True
    log_safety_events: bool = True
    log_user_interactions: bool = True
    
    # Remote monitoring
    enable_remote_monitoring: bool = False
    monitoring_server_url: str = ""
    monitoring_api_key: str = ""


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    # Metadata
    config_version: str = "1.0.0"
    environment: str = "development"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Core configurations
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProductionConfig':
        """Create configuration from dictionary."""
        # Handle nested dataclass creation
        if 'safety' in config_dict and isinstance(config_dict['safety'], dict):
            config_dict['safety'] = SafetyConfig(**config_dict['safety'])
        
        if 'hardware' in config_dict and isinstance(config_dict['hardware'], dict):
            config_dict['hardware'] = HardwareConfig(**config_dict['hardware'])
        
        if 'algorithm' in config_dict and isinstance(config_dict['algorithm'], dict):
            config_dict['algorithm'] = AlgorithmConfig(**config_dict['algorithm'])
        
        if 'user_preferences' in config_dict and isinstance(config_dict['user_preferences'], dict):
            config_dict['user_preferences'] = UserPreferences(**config_dict['user_preferences'])
        
        if 'logging' in config_dict and isinstance(config_dict['logging'], dict):
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        
        return cls(**config_dict)


class EnvironmentDetector:
    """Detect and determine deployment environment."""
    
    @staticmethod
    def detect_environment() -> Environment:
        """Auto-detect current deployment environment."""
        
        # Check environment variables first
        env_var = os.environ.get('SAFE_RL_ENV', '').lower()
        if env_var:
            try:
                return Environment.from_string(env_var)
            except:
                pass
        
        # Check for CI/CD indicators
        if any(var in os.environ for var in ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS', 'JENKINS_URL']):
            return Environment.TESTING
        
        # Check for production indicators
        if EnvironmentDetector._is_production_environment():
            return Environment.PRODUCTION
        
        # Check for staging indicators
        if EnvironmentDetector._is_staging_environment():
            return Environment.STAGING
        
        # Check for simulation indicators
        if EnvironmentDetector._is_simulation_environment():
            return Environment.SIMULATION
        
        # Default to development
        return Environment.DEVELOPMENT
    
    @staticmethod
    def _is_production_environment() -> bool:
        """Check if running in production."""
        production_indicators = [
            '/opt/safe_rl',  # Production install path
            '/usr/local/safe_rl',
            'production' in socket.gethostname().lower(),
            'prod' in socket.gethostname().lower(),
            os.environ.get('NODE_ENV') == 'production'
        ]
        
        return any(production_indicators)
    
    @staticmethod
    def _is_staging_environment() -> bool:
        """Check if running in staging."""
        staging_indicators = [
            'staging' in socket.gethostname().lower(),
            'stage' in socket.gethostname().lower(),
            os.environ.get('NODE_ENV') == 'staging'
        ]
        
        return any(staging_indicators)
    
    @staticmethod
    def _is_simulation_environment() -> bool:
        """Check if running in simulation."""
        simulation_indicators = [
            os.environ.get('GAZEBO_MASTER_URI'),
            os.environ.get('DISPLAY'),  # X11 for GUI simulation
            'sim' in socket.gethostname().lower(),
            platform.system() == 'Darwin'  # macOS often used for sim
        ]
        
        return any(simulation_indicators)
    
    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """Get comprehensive environment information."""
        return {
            'detected_environment': EnvironmentDetector.detect_environment().name,
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'working_directory': str(Path.cwd()),
            'environment_variables': {
                key: value for key, value in os.environ.items()
                if key.startswith('SAFE_RL') or key in ['NODE_ENV', 'CI', 'HOME', 'USER']
            },
            'timestamp': datetime.now().isoformat()
        }


class ConfigValidator:
    """Comprehensive configuration validation system."""
    
    def __init__(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_configuration(self, config: ProductionConfig) -> ConfigValidationResult:
        """Comprehensive configuration validation."""
        result = ConfigValidationResult(
            is_valid=True,
            validation_level=self.validation_level
        )
        
        # Calculate config hash
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        result.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Basic validation
        self._validate_basic_structure(config, result)
        
        # Type and range validation
        if self.validation_level.value >= ConfigValidationLevel.STANDARD.value:
            self._validate_types_and_ranges(config, result)
        
        # Consistency validation
        if self.validation_level.value >= ConfigValidationLevel.STRICT.value:
            self._validate_consistency(config, result)
        
        # Production security validation
        if self.validation_level.value >= ConfigValidationLevel.PRODUCTION.value:
            self._validate_production_security(config, result)
        
        return result
    
    def _validate_basic_structure(self, config: ProductionConfig, result: ConfigValidationResult):
        """Validate basic configuration structure."""
        required_sections = ['safety', 'hardware', 'algorithm', 'user_preferences', 'logging']
        
        for section in required_sections:
            if not hasattr(config, section) or getattr(config, section) is None:
                result.add_error(f"Missing required configuration section: {section}")
        
        # Validate version format
        if not config.config_version or not isinstance(config.config_version, str):
            result.add_error("Invalid or missing config_version")
        
        # Validate environment
        try:
            Environment.from_string(config.environment)
        except:
            result.add_error(f"Invalid environment: {config.environment}")
    
    def _validate_types_and_ranges(self, config: ProductionConfig, result: ConfigValidationResult):
        """Validate parameter types and ranges."""
        
        # Safety configuration validation
        safety = config.safety
        if safety.watchdog_timeout_sec <= 0 or safety.watchdog_timeout_sec > 60:
            result.add_error("watchdog_timeout_sec must be between 0 and 60 seconds")
        
        if safety.safety_monitoring_rate_hz <= 0 or safety.safety_monitoring_rate_hz > 1000:
            result.add_error("safety_monitoring_rate_hz must be between 0 and 1000 Hz")
        
        if safety.max_force_x <= 0 or safety.max_force_x > 1000:
            result.add_warning("max_force_x seems unusually high or low")
        
        # Hardware configuration validation
        hardware = config.hardware
        if hardware.num_joints <= 0 or hardware.num_joints > 50:
            result.add_error("num_joints must be between 1 and 50")
        
        if len(hardware.joint_names) != hardware.num_joints:
            result.add_error("Number of joint_names must match num_joints")
        
        if hardware.control_frequency_hz <= 0 or hardware.control_frequency_hz > 1000:
            result.add_error("control_frequency_hz must be between 0 and 1000 Hz")
        
        # Validate limit arrays have correct length
        for limit_name in ['position_limits_min', 'position_limits_max', 'velocity_limits', 'effort_limits']:
            limits = getattr(hardware, limit_name)
            if len(limits) != hardware.num_joints:
                result.add_error(f"{limit_name} must have {hardware.num_joints} values")
        
        # Algorithm configuration validation
        algorithm = config.algorithm
        if algorithm.learning_rate <= 0 or algorithm.learning_rate > 1:
            result.add_error("learning_rate must be between 0 and 1")
        
        if algorithm.batch_size <= 0 or algorithm.batch_size > 100000:
            result.add_error("batch_size must be between 1 and 100000")
        
        if algorithm.gamma <= 0 or algorithm.gamma > 1:
            result.add_error("gamma must be between 0 and 1")
        
        if algorithm.cost_limit < 0:
            result.add_error("cost_limit must be non-negative")
        
        # User preferences validation
        user_prefs = config.user_preferences
        if user_prefs.preferred_assistance_level < 0 or user_prefs.preferred_assistance_level > 1:
            result.add_error("preferred_assistance_level must be between 0 and 1")
        
        if user_prefs.feedback_intensity < 0 or user_prefs.feedback_intensity > 1:
            result.add_error("feedback_intensity must be between 0 and 1")
    
    def _validate_consistency(self, config: ProductionConfig, result: ConfigValidationResult):
        """Validate configuration consistency and logical relationships."""
        
        # Check position limits consistency
        hardware = config.hardware
        for i in range(len(hardware.position_limits_min)):
            if hardware.position_limits_min[i] >= hardware.position_limits_max[i]:
                result.add_error(f"Joint {i}: position_limits_min must be less than position_limits_max")
        
        # Safety and hardware frequency consistency
        if config.safety.safety_monitoring_rate_hz < config.hardware.control_frequency_hz:
            result.add_warning("Safety monitoring rate should be at least as high as control frequency")
        
        # Algorithm and hardware consistency
        if config.algorithm.max_episode_steps * config.hardware.control_frequency_hz > 3600:
            result.add_warning("Episode might be very long (>1 hour) - consider reducing max_episode_steps")
        
        # User preferences and safety consistency
        if config.user_preferences.preferred_assistance_level > 0.8 and config.user_preferences.conservative_mode:
            result.add_warning("High assistance with conservative mode may be overly restrictive")
        
        # Logging configuration consistency
        if config.logging.enable_remote_monitoring and not config.logging.monitoring_server_url:
            result.add_error("monitoring_server_url required when enable_remote_monitoring is True")
    
    def _validate_production_security(self, config: ProductionConfig, result: ConfigValidationResult):
        """Validate production security requirements."""
        
        # Check for development settings in production
        if Environment.from_string(config.environment) == Environment.PRODUCTION:
            
            # Logging should not be too verbose in production
            if config.logging.log_level.upper() == "DEBUG":
                result.add_warning("DEBUG logging not recommended for production")
            
            # Safety should be fully enabled
            if not config.safety.emergency_stop_enabled:
                result.add_error("Emergency stop must be enabled in production")
            
            if not config.safety.redundant_checking:
                result.add_error("Redundant checking must be enabled in production")
            
            # Monitoring should be enabled
            if not config.logging.enable_performance_monitoring:
                result.add_warning("Performance monitoring recommended for production")
            
            # Check for insecure defaults
            if config.hardware.ethernet_address == "192.168.1.100":
                result.add_warning("Default ethernet address should be changed for production")
        
        # Check for potential security issues
        if config.logging.monitoring_api_key == "":
            if config.logging.enable_remote_monitoring:
                result.add_error("API key required for remote monitoring")
        
        # Validate file paths are secure
        log_path = Path(config.logging.log_file_path)
        if log_path.is_absolute() and not str(log_path).startswith(('/var/log', '/tmp', str(Path.home()))):
            result.add_warning("Log file path may not be secure")


class ConfigurationManager:
    """Production configuration management system."""
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validator = ConfigValidator()
        
        # Environment-specific config file patterns
        self.config_patterns = {
            Environment.DEVELOPMENT: "development.yaml",
            Environment.TESTING: "testing.yaml", 
            Environment.STAGING: "staging.yaml",
            Environment.PRODUCTION: "production.yaml",
            Environment.SIMULATION: "simulation.yaml"
        }
        
        # Current configuration cache
        self._current_config: Optional[ProductionConfig] = None
        self._config_file_path: Optional[Path] = None
    
    def load_configuration(self, 
                          environment: Optional[Environment] = None,
                          config_file: Optional[Union[str, Path]] = None,
                          validation_level: ConfigValidationLevel = ConfigValidationLevel.STANDARD) -> ProductionConfig:
        """Load configuration for specified environment."""
        
        # Auto-detect environment if not specified
        if environment is None:
            environment = EnvironmentDetector.detect_environment()
        
        # Determine config file path
        if config_file:
            config_path = Path(config_file)
        else:
            config_filename = self.config_patterns[environment]
            config_path = self.config_dir / config_filename
        
        self._config_file_path = config_path
        
        # Load configuration
        if config_path.exists():
            config = self._load_config_file(config_path)
        else:
            self.logger.warning(f"Config file not found: {config_path}, creating default")
            config = self._create_default_config(environment)
            self.save_configuration(config, config_path)
        
        # Apply environment-specific overrides
        config = self._apply_environment_overrides(config, environment)
        
        # Validate configuration
        self.validator.validation_level = validation_level
        validation_result = self.validator.validate_configuration(config)
        
        if not validation_result.is_valid:
            error_msg = f"Configuration validation failed: {validation_result.errors}"
            self.logger.error(error_msg)
            if validation_level == ConfigValidationLevel.PRODUCTION:
                raise ValueError(error_msg)
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning: {warning}")
        
        self._current_config = config
        return config
    
    def save_configuration(self, 
                          config: ProductionConfig, 
                          config_file: Optional[Union[str, Path]] = None) -> bool:
        """Save configuration to file."""
        
        if config_file:
            config_path = Path(config_file)
        elif self._config_file_path:
            config_path = self._config_file_path
        else:
            environment = Environment.from_string(config.environment)
            config_filename = self.config_patterns[environment]
            config_path = self.config_dir / config_filename
        
        try:
            # Update timestamp
            config.updated_at = datetime.now().isoformat()
            
            # Save to YAML format for readability
            config_dict = config.to_dict()
            
            # Create backup if file exists
            if config_path.exists():
                backup_path = config_path.with_suffix(f'.backup.{int(datetime.now().timestamp())}')
                config_path.rename(backup_path)
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _load_config_file(self, config_path: Path) -> ProductionConfig:
        """Load configuration from file."""
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            config = ProductionConfig.from_dict(config_dict)
            self.logger.info(f"Configuration loaded from: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def _create_default_config(self, environment: Environment) -> ProductionConfig:
        """Create default configuration for environment."""
        
        config = ProductionConfig(environment=environment.name.lower())
        
        # Environment-specific defaults
        if environment == Environment.PRODUCTION:
            config.safety.redundant_checking = True
            config.safety.fail_safe_mode = "emergency_stop"
            config.logging.log_level = "INFO"
            config.logging.enable_performance_monitoring = True
            
        elif environment == Environment.DEVELOPMENT:
            config.logging.log_level = "DEBUG"
            config.logging.log_to_console = True
            config.safety.watchdog_timeout_sec = 5.0  # More forgiving
            
        elif environment == Environment.TESTING:
            config.safety.emergency_stop_enabled = False  # For testing
            config.logging.log_level = "DEBUG"
            config.algorithm.total_training_steps = 10000  # Faster testing
            
        elif environment == Environment.SIMULATION:
            config.hardware.robot_type = "simulation"
            config.hardware.communication_protocol = "simulation"
            config.safety.watchdog_timeout_sec = 10.0  # Simulation can be slower
            
        return config
    
    def _apply_environment_overrides(self, config: ProductionConfig, environment: Environment) -> ProductionConfig:
        """Apply environment-specific configuration overrides."""
        
        if environment.name.lower() in config.environment_overrides:
            overrides = config.environment_overrides[environment.name.lower()]
            
            # Deep merge overrides into configuration
            config_dict = config.to_dict()
            self._deep_merge_dict(config_dict, overrides)
            
            return ProductionConfig.from_dict(config_dict)
        
        return config
    
    def _deep_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source dictionary into target dictionary."""
        
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def get_robot_specific_config(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get robot-specific configuration parameters."""
        
        robot_config_file = self.config_dir / f"robots/{robot_id}.yaml"
        
        if robot_config_file.exists():
            try:
                with open(robot_config_file, 'r') as f:
                    robot_config = yaml.safe_load(f)
                return robot_config
            except Exception as e:
                self.logger.error(f"Failed to load robot config for {robot_id}: {e}")
        
        return None
    
    def save_user_preferences(self, user_id: str, preferences: UserPreferences) -> bool:
        """Save user-specific preferences."""
        
        user_config_dir = self.config_dir / "users"
        user_config_dir.mkdir(exist_ok=True)
        
        user_config_file = user_config_dir / f"{user_id}.yaml"
        
        try:
            with open(user_config_file, 'w') as f:
                yaml.dump(asdict(preferences), f, default_flow_style=False)
            
            self.logger.info(f"User preferences saved for: {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save user preferences for {user_id}: {e}")
            return False
    
    def load_user_preferences(self, user_id: str) -> UserPreferences:
        """Load user-specific preferences."""
        
        user_config_file = self.config_dir / f"users/{user_id}.yaml"
        
        if user_config_file.exists():
            try:
                with open(user_config_file, 'r') as f:
                    prefs_dict = yaml.safe_load(f)
                return UserPreferences(**prefs_dict)
            except Exception as e:
                self.logger.error(f"Failed to load user preferences for {user_id}: {e}")
        
        # Return default preferences if file doesn't exist or failed to load
        return UserPreferences(user_id=user_id)
    
    def migrate_configuration(self, from_version: str, to_version: str) -> bool:
        """Migrate configuration from one version to another."""
        
        # This would implement version-specific migration logic
        # For now, just log the migration request
        self.logger.info(f"Configuration migration requested: {from_version} -> {to_version}")
        
        # Placeholder for actual migration logic
        return True
    
    def get_current_config(self) -> Optional[ProductionConfig]:
        """Get currently loaded configuration."""
        return self._current_config
    
    def validate_current_config(self, validation_level: ConfigValidationLevel = ConfigValidationLevel.STANDARD) -> ConfigValidationResult:
        """Validate currently loaded configuration."""
        
        if self._current_config is None:
            result = ConfigValidationResult(is_valid=False)
            result.add_error("No configuration currently loaded")
            return result
        
        self.validator.validation_level = validation_level
        return self.validator.validate_configuration(self._current_config)


def validate_configuration(config: Dict[str, Any]) -> ConfigValidationResult:
    """Standalone configuration validation function."""
    
    try:
        # Convert dict to ProductionConfig object
        prod_config = ProductionConfig.from_dict(config)
        
        # Validate using ConfigValidator
        validator = ConfigValidator(ConfigValidationLevel.STANDARD)
        return validator.validate_configuration(prod_config)
        
    except Exception as e:
        result = ConfigValidationResult(is_valid=False)
        result.add_error(f"Configuration format error: {e}")
        return result


def create_sample_configurations():
    """Create sample configuration files for all environments."""
    
    config_manager = ConfigurationManager()
    
    for environment in Environment:
        config = config_manager._create_default_config(environment)
        config_filename = config_manager.config_patterns[environment]
        config_path = config_manager.config_dir / config_filename
        
        config_manager.save_configuration(config, config_path)
        print(f"Created sample configuration: {config_path}")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Safe RL Configuration Management System")
    print("=" * 50)
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Auto-detect and load configuration
    print(f"Detected environment: {EnvironmentDetector.detect_environment().name}")
    
    try:
        config = config_manager.load_configuration()
        print(f"Loaded configuration for: {config.environment}")
        
        # Validate configuration
        validation_result = config_manager.validate_current_config()
        print(f"Configuration valid: {validation_result.is_valid}")
        
        if validation_result.errors:
            print("Errors:", validation_result.errors)
        if validation_result.warnings:
            print("Warnings:", validation_result.warnings)
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        
        # Create sample configurations
        print("\nCreating sample configurations...")
        create_sample_configurations()