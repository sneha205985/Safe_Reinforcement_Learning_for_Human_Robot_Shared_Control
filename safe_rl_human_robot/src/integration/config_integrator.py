"""
Configuration Integration Module for Safe RL System.

Standardizes configuration loading across all system components and resolves
configuration dependencies and conflicts.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
from enum import Enum

try:
    from ..config.settings import SafeRLConfig
    from ..deployment.config_manager import ConfigurationManager, ConfigType, Environment
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config.settings import SafeRLConfig
    from deployment.config_manager import ConfigurationManager, ConfigType, Environment

logger = logging.getLogger(__name__)


class ConfigurationPriority(Enum):
    """Configuration source priorities."""
    DEFAULT = 1
    CONFIG_FILE = 2
    ENVIRONMENT = 3
    COMMAND_LINE = 4


@dataclass
class ConfigSource:
    """Configuration source metadata."""
    source_type: ConfigurationPriority
    source_path: Optional[str] = None
    timestamp: float = 0.0
    checksum: str = ""


@dataclass
class UnifiedConfig:
    """Unified configuration for the entire Safe RL system."""
    
    # Core system configuration
    safe_rl: SafeRLConfig = field(default_factory=SafeRLConfig)
    
    # Component-specific configurations
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    safety_config: Dict[str, Any] = field(default_factory=dict)
    policy_config: Dict[str, Any] = field(default_factory=dict)
    ros_config: Dict[str, Any] = field(default_factory=dict)
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    
    # System metadata
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    config_sources: List[ConfigSource] = field(default_factory=list)
    
    # Runtime settings
    debug_mode: bool = False
    profile_enabled: bool = False
    log_level: str = "INFO"
    
    def validate(self) -> bool:
        """Validate the unified configuration."""
        try:
            # Validate SafeRLConfig (has built-in validation)
            self.safe_rl._validate_config()
            
            # Validate component configurations
            return self._validate_component_configs()
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_component_configs(self) -> bool:
        """Validate component-specific configurations."""
        
        # Hardware configuration validation
        if self.hardware_config:
            if not self._validate_hardware_config():
                return False
        
        # Safety configuration validation
        if self.safety_config:
            if not self._validate_safety_config():
                return False
        
        # Policy configuration validation
        if self.policy_config:
            if not self._validate_policy_config():
                return False
        
        return True
    
    def _validate_hardware_config(self) -> bool:
        """Validate hardware configuration."""
        required_keys = ['interfaces']
        for key in required_keys:
            if key not in self.hardware_config:
                logger.error(f"Missing required hardware config key: {key}")
                return False
        
        # Validate interfaces
        for interface in self.hardware_config.get('interfaces', []):
            if 'device_id' not in interface or 'device_type' not in interface:
                logger.error("Hardware interface missing required fields")
                return False
        
        return True
    
    def _validate_safety_config(self) -> bool:
        """Validate safety configuration."""
        required_keys = ['monitoring_frequency', 'emergency_stop_enabled']
        for key in required_keys:
            if key not in self.safety_config:
                logger.error(f"Missing required safety config key: {key}")
                return False
        
        # Validate monitoring frequency
        freq = self.safety_config.get('monitoring_frequency', 0)
        if freq < 100 or freq > 10000:
            logger.error(f"Invalid monitoring frequency: {freq}")
            return False
        
        return True
    
    def _validate_policy_config(self) -> bool:
        """Validate policy configuration."""
        if 'model_path' in self.policy_config:
            model_path = Path(self.policy_config['model_path'])
            if not model_path.exists():
                logger.warning(f"Policy model path does not exist: {model_path}")
        
        return True


class ConfigurationIntegrator:
    """
    Integrates configurations from multiple sources with proper prioritization.
    
    Handles configuration loading, merging, validation, and hot-reloading
    for the entire Safe RL system.
    """
    
    def __init__(self, 
                 config_dir: Union[str, Path],
                 environment: Environment = Environment.DEVELOPMENT):
        """
        Initialize configuration integrator.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Deployment environment
        """
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration manager for advanced features
        self.config_manager = ConfigurationManager(config_dir, environment)
        
        # Current unified configuration
        self.unified_config: Optional[UnifiedConfig] = None
        
        # Configuration change callbacks
        self.change_callbacks: List[callable] = []
        
        # Register for configuration changes
        for config_type in ConfigType:
            self.config_manager.register_change_callback(
                config_type, 
                self._on_config_change
            )
    
    def load_unified_config(self, 
                          config_file: Optional[str] = None,
                          override_values: Optional[Dict[str, Any]] = None) -> UnifiedConfig:
        """
        Load and create unified configuration from all sources.
        
        Args:
            config_file: Optional specific config file to load
            override_values: Optional values to override in configuration
            
        Returns:
            UnifiedConfig: Integrated configuration
        """
        try:
            self.logger.info("Loading unified configuration...")
            
            # Start with default configuration
            unified_config = UnifiedConfig(environment=self.environment)
            
            # Load from configuration manager
            self._load_from_config_manager(unified_config)
            
            # Load from specific config file if provided
            if config_file:
                self._load_from_config_file(unified_config, config_file)
            
            # Apply environment variable overrides
            self._apply_environment_overrides(unified_config)
            
            # Apply direct overrides
            if override_values:
                self._apply_overrides(unified_config, override_values)
            
            # Validate configuration
            if not unified_config.validate():
                raise ValueError("Configuration validation failed")
            
            self.unified_config = unified_config
            self.logger.info("Unified configuration loaded successfully")
            
            return unified_config
            
        except Exception as e:
            self.logger.error(f"Failed to load unified configuration: {e}")
            raise
    
    def _load_from_config_manager(self, unified_config: UnifiedConfig):
        """Load configurations from the configuration manager."""
        try:
            # Load component configurations
            unified_config.hardware_config = self.config_manager.get_config(ConfigType.HARDWARE)
            unified_config.safety_config = self.config_manager.get_config(ConfigType.SAFETY)
            unified_config.policy_config = self.config_manager.get_config(ConfigType.POLICY)
            unified_config.ros_config = self.config_manager.get_config(ConfigType.ROS)
            unified_config.deployment_config = self.config_manager.get_config(ConfigType.DEPLOYMENT)
            
            # Add source metadata
            unified_config.config_sources.append(ConfigSource(
                source_type=ConfigurationPriority.CONFIG_FILE,
                source_path=str(self.config_dir),
                timestamp=time.time()
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to load from configuration manager: {e}")
    
    def _load_from_config_file(self, unified_config: UnifiedConfig, config_file: str):
        """Load configuration from specific file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                config_path = self.config_dir / config_file
            
            if not config_path.exists():
                self.logger.warning(f"Configuration file not found: {config_file}")
                return
            
            # Load file
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)
            
            # Apply configuration
            self._merge_config(unified_config, file_config)
            
            # Add source metadata
            unified_config.config_sources.append(ConfigSource(
                source_type=ConfigurationPriority.CONFIG_FILE,
                source_path=str(config_path),
                timestamp=config_path.stat().st_mtime
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
    
    def _apply_environment_overrides(self, unified_config: UnifiedConfig):
        """Apply environment variable overrides."""
        try:
            env_overrides = {}
            
            # Check for environment variables with SAFE_RL_ prefix
            for key, value in os.environ.items():
                if key.startswith('SAFE_RL_'):
                    config_key = key[8:].lower().replace('_', '.')
                    env_overrides[config_key] = self._parse_env_value(value)
            
            if env_overrides:
                self._apply_overrides(unified_config, env_overrides)
                
                unified_config.config_sources.append(ConfigSource(
                    source_type=ConfigurationPriority.ENVIRONMENT,
                    source_path="environment_variables"
                ))
            
        except Exception as e:
            self.logger.error(f"Failed to apply environment overrides: {e}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Try number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _apply_overrides(self, unified_config: UnifiedConfig, overrides: Dict[str, Any]):
        """Apply override values to configuration."""
        def set_nested_value(obj, path: str, value: Any):
            keys = path.split('.')
            current = obj
            
            for key in keys[:-1]:
                if not hasattr(current, key):
                    setattr(current, key, {})
                current = getattr(current, key)
                if isinstance(current, dict) and key not in current:
                    current[key] = {}
                    current = current[key]
            
            if hasattr(current, keys[-1]):
                setattr(current, keys[-1], value)
            elif isinstance(current, dict):
                current[keys[-1]] = value
        
        for key, value in overrides.items():
            try:
                set_nested_value(unified_config, key, value)
            except Exception as e:
                self.logger.warning(f"Failed to apply override {key}={value}: {e}")
    
    def _merge_config(self, unified_config: UnifiedConfig, new_config: Dict[str, Any]):
        """Merge new configuration into unified config."""
        for key, value in new_config.items():
            if hasattr(unified_config, key):
                current_value = getattr(unified_config, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    self._deep_merge_dict(current_value, value)
                else:
                    setattr(unified_config, key, value)
    
    def _deep_merge_dict(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge two dictionaries."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def _on_config_change(self, config_type: ConfigType, new_config: Dict[str, Any]):
        """Handle configuration changes from config manager."""
        if not self.unified_config:
            return
        
        try:
            # Update the appropriate configuration section
            config_mapping = {
                ConfigType.HARDWARE: 'hardware_config',
                ConfigType.SAFETY: 'safety_config',
                ConfigType.POLICY: 'policy_config',
                ConfigType.ROS: 'ros_config',
                ConfigType.DEPLOYMENT: 'deployment_config'
            }
            
            if config_type in config_mapping:
                attr_name = config_mapping[config_type]
                setattr(self.unified_config, attr_name, new_config.copy())
                
                # Validate updated configuration
                if self.unified_config.validate():
                    self.logger.info(f"Configuration updated: {config_type.name}")
                    self._notify_change_callbacks()
                else:
                    self.logger.error(f"Invalid configuration update: {config_type.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle config change: {e}")
    
    def _notify_change_callbacks(self):
        """Notify registered callbacks of configuration changes."""
        for callback in self.change_callbacks:
            try:
                callback(self.unified_config)
            except Exception as e:
                self.logger.error(f"Configuration change callback failed: {e}")
    
    def register_change_callback(self, callback: callable):
        """Register callback for configuration changes."""
        self.change_callbacks.append(callback)
    
    def get_config(self) -> Optional[UnifiedConfig]:
        """Get current unified configuration."""
        return self.unified_config
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        if not self.unified_config:
            return False
        
        try:
            if not config_file:
                config_file = f"unified_config_{self.environment.value}.yaml"
            
            output_path = self.config_dir / config_file
            
            # Convert to dictionary
            config_dict = {
                'safe_rl': self.unified_config.safe_rl.to_dict(),
                'hardware_config': self.unified_config.hardware_config,
                'safety_config': self.unified_config.safety_config,
                'policy_config': self.unified_config.policy_config,
                'ros_config': self.unified_config.ros_config,
                'deployment_config': self.unified_config.deployment_config,
                'environment': self.unified_config.environment.value,
                'version': self.unified_config.version,
                'debug_mode': self.unified_config.debug_mode,
                'profile_enabled': self.unified_config.profile_enabled,
                'log_level': self.unified_config.log_level
            }
            
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_config_summary(self) -> str:
        """Get configuration summary."""
        if not self.unified_config:
            return "No configuration loaded"
        
        summary = [
            "Safe RL System Configuration Summary",
            "=" * 40,
            f"Environment: {self.unified_config.environment.value}",
            f"Version: {self.unified_config.version}",
            f"Debug Mode: {self.unified_config.debug_mode}",
            f"Log Level: {self.unified_config.log_level}",
            "",
            "Configuration Sources:"
        ]
        
        for i, source in enumerate(self.unified_config.config_sources, 1):
            summary.append(f"  {i}. {source.source_type.name}: {source.source_path or 'N/A'}")
        
        summary.extend([
            "",
            "Component Configurations:",
            f"  Hardware Interfaces: {len(self.unified_config.hardware_config.get('interfaces', []))}",
            f"  Safety Monitoring: {'Enabled' if self.unified_config.safety_config.get('emergency_stop_enabled', False) else 'Disabled'}",
            f"  Policy Model: {'Loaded' if self.unified_config.policy_config.get('model_path') else 'Not specified'}",
            f"  ROS Integration: {'Configured' if self.unified_config.ros_config else 'Not configured'}",
            "",
            self.unified_config.safe_rl.summary()
        ])
        
        return "\n".join(summary)
    
    def shutdown(self):
        """Shutdown configuration integrator."""
        try:
            self.config_manager.shutdown()
            self.logger.info("Configuration integrator shutdown complete")
        except Exception as e:
            self.logger.error(f"Configuration integrator shutdown failed: {e}")


import time  # Import at module level


def create_unified_config_template(output_dir: str):
    """Create template unified configuration files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Template configuration
    template_config = {
        'safe_rl': {
            'experiment_name': 'safe_rl_human_robot_shared_control',
            'version': '1.0.0',
            'optimization_method': 'cpo',
            'device': 'cpu'
        },
        'hardware_config': {
            'interfaces': [
                {
                    'device_id': 'example_device',
                    'device_type': 'exoskeleton',
                    'communication_protocol': 'serial',
                    'sampling_rate': 1000
                }
            ]
        },
        'safety_config': {
            'monitoring_frequency': 2000,
            'emergency_stop_enabled': True,
            'position_limits': {},
            'velocity_limits': {},
            'force_limits': {}
        },
        'policy_config': {
            'model_path': '/path/to/trained/model',
            'model_type': 'pytorch',
            'input_dim': 12,
            'output_dim': 6,
            'inference_device': 'cpu'
        },
        'ros_config': {
            'namespace': '/safe_rl',
            'topics': {
                'joint_states': '/joint_states',
                'cmd_vel': '/cmd_vel',
                'safety_status': '/safety_status'
            }
        },
        'deployment_config': {
            'mode': 'development',
            'monitoring_enabled': True,
            'logging_level': 'INFO'
        },
        'environment': 'development',
        'version': '1.0.0',
        'debug_mode': True,
        'profile_enabled': False,
        'log_level': 'INFO'
    }
    
    # Save template files
    for env in ['development', 'testing', 'production']:
        config_file = output_dir / f"unified_config_{env}.yaml"
        template_config['environment'] = env
        template_config['debug_mode'] = (env == 'development')
        
        with open(config_file, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)
        
        print(f"Created configuration template: {config_file}")


if __name__ == "__main__":
    import tempfile
    import sys
    
    # Create example
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Creating configuration integrator example in: {temp_dir}")
        
        # Create template configurations
        create_unified_config_template(temp_dir)
        
        # Initialize configuration integrator
        integrator = ConfigurationIntegrator(temp_dir, Environment.DEVELOPMENT)
        
        # Load unified configuration
        config = integrator.load_unified_config()
        
        # Print configuration summary
        print("\n" + integrator.get_config_summary())
        
        # Save configuration
        integrator.save_config()
        
        # Shutdown
        integrator.shutdown()
        
        print("Configuration integrator example complete!")