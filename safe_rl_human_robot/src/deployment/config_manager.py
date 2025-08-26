"""
Configuration Management System for Safe RL Deployment.

This module provides comprehensive configuration management with validation,
versioning, hot-reloading, and environment-specific configurations for
production Safe RL systems.

Key Features:
- Hierarchical configuration management
- Schema validation and type checking
- Hot configuration reloading
- Environment-specific configurations
- Configuration versioning and rollback
- Encrypted sensitive configuration
- Configuration auditing and compliance
"""

import os
import json
import yaml
import time
import threading
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from enum import Enum, auto
import jsonschema
from cryptography.fernet import Fernet
import watchdog.observers
import watchdog.events

logger = logging.getLogger(__name__)


class ConfigType(Enum):
    """Configuration types."""
    SYSTEM = auto()
    HARDWARE = auto()
    SAFETY = auto()
    POLICY = auto()
    ROS = auto()
    DEPLOYMENT = auto()
    SECRETS = auto()


class Environment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing" 
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ConfigMetadata:
    """Configuration metadata."""
    version: str
    created_at: float
    modified_at: float
    environment: Environment
    author: str
    checksum: str
    encrypted: bool = False
    validated: bool = False
    applied_at: Optional[float] = None


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    path: str  # JSON path to validate
    rule_type: str  # required, range, enum, format, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    severity: str = "error"  # error, warning, info


class ConfigurationSchema:
    """Configuration schema for validation."""
    
    SYSTEM_SCHEMA = {
        "type": "object",
        "properties": {
            "node_name": {"type": "string", "minLength": 1},
            "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]},
            "max_cpu_percent": {"type": "number", "minimum": 0, "maximum": 100},
            "max_memory_mb": {"type": "number", "minimum": 0},
            "enable_profiling": {"type": "boolean"},
            "heartbeat_interval_s": {"type": "number", "minimum": 0.1, "maximum": 60}
        },
        "required": ["node_name", "log_level"]
    }
    
    HARDWARE_SCHEMA = {
        "type": "object",
        "properties": {
            "interfaces": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "device_id": {"type": "string"},
                        "device_type": {"type": "string", "enum": ["exoskeleton", "wheelchair", "manipulator"]},
                        "communication_protocol": {"type": "string", "enum": ["serial", "can", "ethernet"]},
                        "sampling_rate": {"type": "number", "minimum": 10, "maximum": 10000},
                        "safety_limits": {"type": "object"}
                    },
                    "required": ["device_id", "device_type"]
                }
            }
        }
    }
    
    SAFETY_SCHEMA = {
        "type": "object",
        "properties": {
            "monitoring_frequency": {"type": "number", "minimum": 100, "maximum": 10000},
            "emergency_stop_enabled": {"type": "boolean"},
            "position_limits": {"type": "object"},
            "velocity_limits": {"type": "object"},
            "force_limits": {"type": "object"},
            "workspace_boundaries": {"type": "object"},
            "predictive_monitoring": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "prediction_horizon_s": {"type": "number", "minimum": 0.1, "maximum": 5.0}
                }
            }
        },
        "required": ["monitoring_frequency", "emergency_stop_enabled"]
    }
    
    POLICY_SCHEMA = {
        "type": "object", 
        "properties": {
            "model_path": {"type": "string", "minLength": 1},
            "model_type": {"type": "string", "enum": ["pytorch", "tensorflow", "onnx"]},
            "input_dim": {"type": "integer", "minimum": 1},
            "output_dim": {"type": "integer", "minimum": 1},
            "inference_device": {"type": "string", "enum": ["cpu", "cuda", "auto"]},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 1000},
            "max_inference_time_ms": {"type": "number", "minimum": 1, "maximum": 1000}
        },
        "required": ["model_path", "input_dim", "output_dim"]
    }
    
    ROS_SCHEMA = {
        "type": "object",
        "properties": {
            "namespace": {"type": "string"},
            "topics": {
                "type": "object",
                "properties": {
                    "joint_states": {"type": "string"},
                    "cmd_vel": {"type": "string"},
                    "safety_status": {"type": "string"}
                }
            },
            "publish_rate": {"type": "number", "minimum": 1, "maximum": 1000},
            "queue_size": {"type": "integer", "minimum": 1, "maximum": 1000}
        }
    }
    
    @classmethod
    def get_schema(cls, config_type: ConfigType) -> Dict[str, Any]:
        """Get schema for configuration type."""
        schema_map = {
            ConfigType.SYSTEM: cls.SYSTEM_SCHEMA,
            ConfigType.HARDWARE: cls.HARDWARE_SCHEMA, 
            ConfigType.SAFETY: cls.SAFETY_SCHEMA,
            ConfigType.POLICY: cls.POLICY_SCHEMA,
            ConfigType.ROS: cls.ROS_SCHEMA
        }
        return schema_map.get(config_type, {})


class ConfigFileWatcher(watchdog.events.FileSystemEventHandler):
    """File system watcher for configuration hot-reloading."""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger(f"{__name__}.ConfigFileWatcher")
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            self.logger.info(f"Configuration file modified: {event.src_path}")
            # Debounce multiple rapid changes
            time.sleep(0.1)
            self.config_manager._reload_config_file(event.src_path)


class ConfigurationManager:
    """
    Comprehensive configuration management system.
    
    Provides hierarchical configuration management with validation,
    hot-reloading, versioning, and environment-specific configurations.
    """
    
    def __init__(self, config_dir: Union[str, Path], environment: Environment = Environment.DEVELOPMENT):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration storage
        self.configurations: Dict[ConfigType, Dict[str, Any]] = {}
        self.metadata: Dict[ConfigType, ConfigMetadata] = {}
        self.schemas: Dict[ConfigType, Dict[str, Any]] = {}
        
        # Configuration history for rollback
        self.config_history: Dict[ConfigType, List[Dict[str, Any]]] = {
            config_type: [] for config_type in ConfigType
        }
        
        # Validation rules
        self.validation_rules: Dict[ConfigType, List[ConfigValidationRule]] = {}
        
        # Change callbacks
        self.change_callbacks: Dict[ConfigType, List[Callable]] = {
            config_type: [] for config_type in ConfigType
        }
        
        # File watching
        self.file_observer: Optional[watchdog.observers.Observer] = None
        self.file_watcher: Optional[ConfigFileWatcher] = None
        
        # Encryption for sensitive configs
        self.encryption_key: Optional[bytes] = None
        self.fernet: Optional[Fernet] = None
        
        # Thread safety
        self.config_lock = threading.RLock()
        
        # Initialize
        self._setup_directories()
        self._load_encryption_key()
        self._load_all_configurations()
        self._setup_file_watching()
        
        self.logger.info(f"Configuration manager initialized for environment: {environment.value}")
    
    def _setup_directories(self):
        """Setup configuration directories."""
        try:
            # Create base directories
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different config types
            for config_type in ConfigType:
                type_dir = self.config_dir / config_type.name.lower()
                type_dir.mkdir(exist_ok=True)
            
            # Create environment-specific directories
            for env in Environment:
                env_dir = self.config_dir / "environments" / env.value
                env_dir.mkdir(parents=True, exist_ok=True)
                
            # Create backups directory
            (self.config_dir / "backups").mkdir(exist_ok=True)
            
            # Create schemas directory
            schemas_dir = self.config_dir / "schemas"
            schemas_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"Configuration directories setup complete: {self.config_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup configuration directories: {e}")
    
    def _load_encryption_key(self):
        """Load or generate encryption key for sensitive configurations."""
        try:
            key_file = self.config_dir / ".encryption_key"
            
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Generate new key
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
                
                # Set restrictive permissions
                os.chmod(key_file, 0o600)
            
            self.fernet = Fernet(self.encryption_key)
            self.logger.info("Encryption key loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load encryption key: {e}")
    
    def _load_all_configurations(self):
        """Load all configuration files."""
        try:
            with self.config_lock:
                for config_type in ConfigType:
                    self._load_configuration(config_type)
            
            self.logger.info("All configurations loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {e}")
    
    def _load_configuration(self, config_type: ConfigType):
        """Load configuration for specific type."""
        try:
            # Load base configuration
            base_config = self._load_config_file(config_type, "base")
            
            # Load environment-specific overrides
            env_config = self._load_config_file(config_type, self.environment.value)
            
            # Merge configurations (environment overrides base)
            merged_config = self._merge_configs(base_config, env_config)
            
            # Validate configuration
            if self._validate_configuration(config_type, merged_config):
                self.configurations[config_type] = merged_config
                
                # Create metadata
                self.metadata[config_type] = ConfigMetadata(
                    version=self._calculate_config_version(merged_config),
                    created_at=time.time(),
                    modified_at=time.time(),
                    environment=self.environment,
                    author=os.environ.get('USER', 'unknown'),
                    checksum=self._calculate_checksum(merged_config),
                    validated=True
                )
                
                self.logger.info(f"Configuration loaded: {config_type.name}")
            else:
                self.logger.error(f"Configuration validation failed: {config_type.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration {config_type.name}: {e}")
    
    def _load_config_file(self, config_type: ConfigType, variant: str) -> Dict[str, Any]:
        """Load configuration file."""
        try:
            config_file = None
            
            # Try different file extensions and locations
            possible_paths = [
                self.config_dir / config_type.name.lower() / f"{variant}.yaml",
                self.config_dir / config_type.name.lower() / f"{variant}.yml", 
                self.config_dir / config_type.name.lower() / f"{variant}.json",
                self.config_dir / "environments" / variant / f"{config_type.name.lower()}.yaml",
                self.config_dir / "environments" / variant / f"{config_type.name.lower()}.yml",
                self.config_dir / "environments" / variant / f"{config_type.name.lower()}.json"
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_file = path
                    break
            
            if not config_file:
                return {}  # Return empty config if file doesn't exist
            
            # Load file
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
            
            # Handle encrypted configurations
            if config.get('_encrypted', False):
                config = self._decrypt_configuration(config)
            
            return config or {}
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_type.name}/{variant}: {e}")
            return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        if not base:
            return override.copy() if override else {}
        if not override:
            return base.copy()
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_configuration(self, config_type: ConfigType, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema."""
        try:
            # Get schema for config type
            schema = ConfigurationSchema.get_schema(config_type)
            
            if not schema:
                self.logger.warning(f"No schema available for {config_type.name}")
                return True  # Skip validation if no schema
            
            # Validate using jsonschema
            jsonschema.validate(config, schema)
            
            # Apply custom validation rules
            return self._apply_custom_validation_rules(config_type, config)
            
        except jsonschema.ValidationError as e:
            self.logger.error(f"Configuration validation error for {config_type.name}: {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"Configuration validation failed for {config_type.name}: {e}")
            return False
    
    def _apply_custom_validation_rules(self, config_type: ConfigType, config: Dict[str, Any]) -> bool:
        """Apply custom validation rules."""
        try:
            rules = self.validation_rules.get(config_type, [])
            
            for rule in rules:
                if not self._check_validation_rule(config, rule):
                    if rule.severity == "error":
                        self.logger.error(f"Validation rule failed: {rule.error_message}")
                        return False
                    else:
                        self.logger.warning(f"Validation warning: {rule.error_message}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Custom validation failed: {e}")
            return False
    
    def _check_validation_rule(self, config: Dict[str, Any], rule: ConfigValidationRule) -> bool:
        """Check individual validation rule."""
        try:
            # Get value at path
            value = self._get_config_value_by_path(config, rule.path)
            
            if rule.rule_type == "required":
                return value is not None
            elif rule.rule_type == "range":
                min_val = rule.parameters.get('min')
                max_val = rule.parameters.get('max')
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
                return True
            elif rule.rule_type == "enum":
                return value in rule.parameters.get('values', [])
            # Add more rule types as needed
            
            return True
            
        except Exception:
            return False
    
    def _get_config_value_by_path(self, config: Dict[str, Any], path: str) -> Any:
        """Get configuration value by dot-separated path."""
        try:
            keys = path.split('.')
            value = config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception:
            return None
    
    def _setup_file_watching(self):
        """Setup file system watching for hot-reloading."""
        try:
            if self.file_observer:
                return  # Already setup
            
            self.file_watcher = ConfigFileWatcher(self)
            self.file_observer = watchdog.observers.Observer()
            
            # Watch configuration directories
            self.file_observer.schedule(
                self.file_watcher,
                str(self.config_dir),
                recursive=True
            )
            
            self.file_observer.start()
            self.logger.info("File watching started for hot configuration reloading")
            
        except Exception as e:
            self.logger.error(f"Failed to setup file watching: {e}")
    
    def _reload_config_file(self, file_path: str):
        """Reload configuration file after modification."""
        try:
            file_path = Path(file_path)
            
            # Determine config type from file path
            config_type = None
            for ct in ConfigType:
                if ct.name.lower() in file_path.parts:
                    config_type = ct
                    break
            
            if not config_type:
                return  # Not a recognized config file
            
            self.logger.info(f"Reloading configuration: {config_type.name}")
            
            with self.config_lock:
                # Save current config to history
                if config_type in self.configurations:
                    history = self.config_history[config_type]
                    history.append(self.configurations[config_type].copy())
                    
                    # Keep only last 10 versions
                    if len(history) > 10:
                        history.pop(0)
                
                # Reload configuration
                self._load_configuration(config_type)
                
                # Notify callbacks
                self._notify_config_change(config_type)
            
        except Exception as e:
            self.logger.error(f"Failed to reload config file {file_path}: {e}")
    
    def _notify_config_change(self, config_type: ConfigType):
        """Notify registered callbacks of configuration changes."""
        try:
            callbacks = self.change_callbacks.get(config_type, [])
            
            for callback in callbacks:
                try:
                    callback(config_type, self.configurations[config_type])
                except Exception as e:
                    self.logger.error(f"Configuration change callback failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to notify config change: {e}")
    
    def _calculate_config_version(self, config: Dict[str, Any]) -> str:
        """Calculate configuration version hash."""
        try:
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:12]
        except Exception:
            return f"v{int(time.time())}"
    
    def _calculate_checksum(self, config: Dict[str, Any]) -> str:
        """Calculate configuration checksum."""
        try:
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception:
            return ""
    
    def _encrypt_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration data."""
        try:
            if not self.fernet:
                return config
            
            config_str = json.dumps(config)
            encrypted_data = self.fernet.encrypt(config_str.encode())
            
            return {
                '_encrypted': True,
                '_data': encrypted_data.decode()
            }
            
        except Exception as e:
            self.logger.error(f"Configuration encryption failed: {e}")
            return config
    
    def _decrypt_configuration(self, encrypted_config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt configuration data."""
        try:
            if not self.fernet or not encrypted_config.get('_encrypted'):
                return encrypted_config
            
            encrypted_data = encrypted_config['_data'].encode()
            decrypted_str = self.fernet.decrypt(encrypted_data).decode()
            
            return json.loads(decrypted_str)
            
        except Exception as e:
            self.logger.error(f"Configuration decryption failed: {e}")
            return {}
    
    # Public API methods
    
    def get_config(self, config_type: ConfigType) -> Dict[str, Any]:
        """Get configuration for specified type."""
        with self.config_lock:
            return self.configurations.get(config_type, {}).copy()
    
    def get_config_value(self, config_type: ConfigType, path: str, default: Any = None) -> Any:
        """Get specific configuration value by path."""
        with self.config_lock:
            config = self.configurations.get(config_type, {})
            value = self._get_config_value_by_path(config, path)
            return value if value is not None else default
    
    def set_config_value(self, config_type: ConfigType, path: str, value: Any) -> bool:
        """Set specific configuration value."""
        try:
            with self.config_lock:
                config = self.configurations.get(config_type, {}).copy()
                
                # Set value at path
                keys = path.split('.')
                current = config
                
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                current[keys[-1]] = value
                
                # Validate updated configuration
                if self._validate_configuration(config_type, config):
                    self.configurations[config_type] = config
                    self._update_metadata(config_type)
                    self._notify_config_change(config_type)
                    return True
                else:
                    self.logger.error(f"Configuration validation failed for {path}={value}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to set config value {path}={value}: {e}")
            return False
    
    def update_config(self, config_type: ConfigType, updates: Dict[str, Any]) -> bool:
        """Update configuration with dictionary of changes."""
        try:
            with self.config_lock:
                config = self.configurations.get(config_type, {}).copy()
                
                # Apply updates
                config = self._merge_configs(config, updates)
                
                # Validate updated configuration
                if self._validate_configuration(config_type, config):
                    self.configurations[config_type] = config
                    self._update_metadata(config_type)
                    self._notify_config_change(config_type)
                    return True
                else:
                    self.logger.error("Configuration validation failed for updates")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, config_type: ConfigType, encrypt_secrets: bool = True) -> bool:
        """Save configuration to file."""
        try:
            with self.config_lock:
                config = self.configurations.get(config_type, {})
                
                if not config:
                    return False
                
                # Determine output file path
                output_file = self.config_dir / config_type.name.lower() / f"{self.environment.value}.yaml"
                
                # Encrypt if needed
                if encrypt_secrets and config_type == ConfigType.SECRETS:
                    config = self._encrypt_configuration(config)
                
                # Create backup
                self._create_config_backup(config_type)
                
                # Write configuration
                with open(output_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                
                self.logger.info(f"Configuration saved: {output_file}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save configuration {config_type.name}: {e}")
            return False
    
    def rollback_config(self, config_type: ConfigType, versions_back: int = 1) -> bool:
        """Rollback configuration to previous version."""
        try:
            with self.config_lock:
                history = self.config_history.get(config_type, [])
                
                if len(history) < versions_back:
                    self.logger.error(f"Not enough history for rollback: {len(history)} < {versions_back}")
                    return False
                
                # Get previous version
                rollback_config = history[-versions_back]
                
                # Validate rollback configuration
                if self._validate_configuration(config_type, rollback_config):
                    self.configurations[config_type] = rollback_config.copy()
                    self._update_metadata(config_type)
                    self._notify_config_change(config_type)
                    
                    self.logger.info(f"Configuration rolled back: {config_type.name}")
                    return True
                else:
                    self.logger.error("Rollback configuration validation failed")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Configuration rollback failed: {e}")
            return False
    
    def register_change_callback(self, config_type: ConfigType, callback: Callable):
        """Register callback for configuration changes."""
        with self.config_lock:
            self.change_callbacks[config_type].append(callback)
    
    def add_validation_rule(self, config_type: ConfigType, rule: ConfigValidationRule):
        """Add custom validation rule."""
        with self.config_lock:
            if config_type not in self.validation_rules:
                self.validation_rules[config_type] = []
            self.validation_rules[config_type].append(rule)
    
    def get_config_metadata(self, config_type: ConfigType) -> Optional[ConfigMetadata]:
        """Get configuration metadata."""
        with self.config_lock:
            return self.metadata.get(config_type)
    
    def validate_all_configurations(self) -> Dict[ConfigType, bool]:
        """Validate all loaded configurations."""
        results = {}
        
        with self.config_lock:
            for config_type, config in self.configurations.items():
                results[config_type] = self._validate_configuration(config_type, config)
        
        return results
    
    def _update_metadata(self, config_type: ConfigType):
        """Update configuration metadata."""
        if config_type in self.metadata:
            self.metadata[config_type].modified_at = time.time()
            self.metadata[config_type].version = self._calculate_config_version(
                self.configurations[config_type]
            )
            self.metadata[config_type].checksum = self._calculate_checksum(
                self.configurations[config_type]
            )
    
    def _create_config_backup(self, config_type: ConfigType):
        """Create configuration backup."""
        try:
            if config_type not in self.configurations:
                return
            
            backup_dir = self.config_dir / "backups"
            timestamp = int(time.time())
            backup_file = backup_dir / f"{config_type.name.lower()}_{timestamp}.yaml"
            
            with open(backup_file, 'w') as f:
                yaml.dump(self.configurations[config_type], f, default_flow_style=False)
            
            # Keep only recent backups (last 20)
            backups = sorted(backup_dir.glob(f"{config_type.name.lower()}_*.yaml"))
            if len(backups) > 20:
                for old_backup in backups[:-20]:
                    old_backup.unlink()
                    
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
    
    def shutdown(self):
        """Shutdown configuration manager."""
        try:
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join(timeout=2.0)
            
            self.logger.info("Configuration manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Configuration manager shutdown failed: {e}")


# Example usage and configuration templates
EXAMPLE_CONFIGURATIONS = {
    ConfigType.SYSTEM: {
        "node_name": "safe_rl_system",
        "log_level": "INFO",
        "max_cpu_percent": 80.0,
        "max_memory_mb": 2048,
        "enable_profiling": False,
        "heartbeat_interval_s": 1.0
    },
    
    ConfigType.HARDWARE: {
        "interfaces": [
            {
                "device_id": "exo_arm_left",
                "device_type": "exoskeleton", 
                "communication_protocol": "can",
                "sampling_rate": 1000.0,
                "safety_limits": {
                    "max_torque": 50.0,
                    "max_velocity": 2.0,
                    "position_limits": [-1.57, 1.57]
                }
            }
        ]
    },
    
    ConfigType.SAFETY: {
        "monitoring_frequency": 2000.0,
        "emergency_stop_enabled": True,
        "position_limits": {
            "joint_0": [-1.57, 1.57],
            "joint_1": [-1.0, 1.0]
        },
        "velocity_limits": {
            "joint_0": 2.0,
            "joint_1": 1.5
        },
        "force_limits": {
            "max_force": 50.0
        },
        "predictive_monitoring": {
            "enabled": True,
            "prediction_horizon_s": 0.5
        }
    }
}


def create_example_configurations(config_dir: str):
    """Create example configuration files."""
    config_dir = Path(config_dir)
    
    for config_type, config_data in EXAMPLE_CONFIGURATIONS.items():
        type_dir = config_dir / config_type.name.lower()
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Create base configuration
        base_file = type_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"Created example configuration: {base_file}")


if __name__ == "__main__":
    """Example usage of configuration manager."""
    import tempfile
    import sys
    
    # Create temporary config directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating configuration manager in: {temp_dir}")
        
        # Create example configurations
        create_example_configurations(temp_dir)
        
        # Initialize configuration manager
        config_manager = ConfigurationManager(temp_dir, Environment.DEVELOPMENT)
        
        # Example operations
        print("\n=== Configuration Manager Example ===")
        
        # Get system configuration
        system_config = config_manager.get_config(ConfigType.SYSTEM)
        print(f"System config: {system_config}")
        
        # Get specific value
        log_level = config_manager.get_config_value(ConfigType.SYSTEM, "log_level")
        print(f"Log level: {log_level}")
        
        # Update configuration
        print("\nUpdating log level to DEBUG...")
        config_manager.set_config_value(ConfigType.SYSTEM, "log_level", "DEBUG")
        
        # Verify update
        new_log_level = config_manager.get_config_value(ConfigType.SYSTEM, "log_level")
        print(f"New log level: {new_log_level}")
        
        # Validate all configurations
        validation_results = config_manager.validate_all_configurations()
        print(f"\nValidation results: {validation_results}")
        
        # Shutdown
        config_manager.shutdown()
        
        print("Configuration manager example complete!")