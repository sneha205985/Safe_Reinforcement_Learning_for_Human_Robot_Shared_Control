#!/usr/bin/env python3
"""
Configuration Management System Validation Test
Comprehensive testing of configuration loading, validation, and environment detection
"""

import sys
import os
import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.production_config import (
    ConfigurationManager,
    ConfigValidator,
    EnvironmentDetector,
    ProductionConfig,
    Environment,
    ConfigValidationLevel,
    validate_configuration
)

def setup_logging():
    """Setup logging for configuration tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_environment_detection():
    """Test environment detection capabilities"""
    logger = logging.getLogger("test_env_detection")
    logger.info("Testing environment detection...")
    
    results = {}
    
    try:
        # Test basic detection
        detected_env = EnvironmentDetector.detect_environment()
        results['basic_detection'] = f"PASS - Detected: {detected_env.name}"
        logger.info(f"✅ Detected environment: {detected_env.name}")
        
        # Test environment info gathering
        env_info = EnvironmentDetector.get_environment_info()
        if 'detected_environment' in env_info and 'hostname' in env_info:
            results['env_info'] = "PASS - Complete environment information"
            logger.info(f"✅ Environment info: {env_info['hostname']}")
        else:
            results['env_info'] = "FAIL - Incomplete environment information"
        
        # Test environment string conversion
        for env_str in ['dev', 'production', 'staging', 'simulation']:
            try:
                env = Environment.from_string(env_str)
                results[f'string_conversion_{env_str}'] = f"PASS - {env.name}"
            except Exception as e:
                results[f'string_conversion_{env_str}'] = f"FAIL - {e}"
        
    except Exception as e:
        results['environment_detection'] = f"FAIL - {e}"
        logger.error(f"❌ Environment detection failed: {e}")
    
    return results

def test_configuration_loading():
    """Test configuration loading from files"""
    logger = logging.getLogger("test_config_loading")
    logger.info("Testing configuration loading...")
    
    results = {}
    config_manager = ConfigurationManager()
    
    # Test loading each environment configuration
    environments = [Environment.DEVELOPMENT, Environment.PRODUCTION, 
                   Environment.STAGING, Environment.SIMULATION]
    
    for env in environments:
        try:
            config = config_manager.load_configuration(
                environment=env,
                validation_level=ConfigValidationLevel.STANDARD
            )
            
            if config and config.environment == env.name.lower():
                results[f'load_{env.name.lower()}'] = "PASS"
                logger.info(f"✅ Loaded {env.name} configuration successfully")
            else:
                results[f'load_{env.name.lower()}'] = "FAIL - Invalid configuration loaded"
                
        except Exception as e:
            results[f'load_{env.name.lower()}'] = f"FAIL - {e}"
            logger.error(f"❌ Failed to load {env.name} config: {e}")
    
    # Test loading non-existent configuration
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_manager = ConfigurationManager(temp_dir)
            config = temp_config_manager.load_configuration(Environment.DEVELOPMENT)
            
            if config:
                results['create_default'] = "PASS - Default config created"
                logger.info("✅ Default configuration created successfully")
            else:
                results['create_default'] = "FAIL - No default config created"
                
    except Exception as e:
        results['create_default'] = f"FAIL - {e}"
        logger.error(f"❌ Default config creation failed: {e}")
    
    return results

def test_configuration_validation():
    """Test comprehensive configuration validation"""
    logger = logging.getLogger("test_config_validation")
    logger.info("Testing configuration validation...")
    
    results = {}
    validator = ConfigValidator()
    
    # Test valid configuration validation
    try:
        config_manager = ConfigurationManager()
        valid_config = config_manager.load_configuration(
            Environment.DEVELOPMENT,
            validation_level=ConfigValidationLevel.MINIMAL
        )
        
        validation_result = validator.validate_configuration(valid_config)
        
        if validation_result.is_valid:
            results['valid_config'] = "PASS - Valid configuration accepted"
            logger.info("✅ Valid configuration validation passed")
        else:
            results['valid_config'] = f"FAIL - Valid config rejected: {validation_result.errors}"
            
    except Exception as e:
        results['valid_config'] = f"FAIL - {e}"
        logger.error(f"❌ Valid config validation failed: {e}")
    
    # Test invalid configuration scenarios
    invalid_configs = {
        'negative_watchdog_timeout': {
            'config_version': '1.0.0',
            'environment': 'development',
            'safety': {'watchdog_timeout_sec': -1.0},
            'hardware': {'num_joints': 6},
            'algorithm': {'learning_rate': 0.001},
            'user_preferences': {'preferred_assistance_level': 0.5},
            'logging': {'log_level': 'INFO'}
        },
        
        'invalid_learning_rate': {
            'config_version': '1.0.0',
            'environment': 'development', 
            'safety': {'watchdog_timeout_sec': 2.0},
            'hardware': {'num_joints': 6},
            'algorithm': {'learning_rate': 2.0},  # Invalid: > 1.0
            'user_preferences': {'preferred_assistance_level': 0.5},
            'logging': {'log_level': 'INFO'}
        },
        
        'mismatched_joint_arrays': {
            'config_version': '1.0.0',
            'environment': 'development',
            'safety': {'watchdog_timeout_sec': 2.0},
            'hardware': {
                'num_joints': 6,
                'joint_names': ['joint1', 'joint2'],  # Should have 6 names
                'position_limits_min': [-1.0, -1.0, -1.0]  # Should have 6 limits
            },
            'algorithm': {'learning_rate': 0.001},
            'user_preferences': {'preferred_assistance_level': 0.5},
            'logging': {'log_level': 'INFO'}
        }
    }
    
    for test_name, invalid_config_dict in invalid_configs.items():
        try:
            validation_result = validate_configuration(invalid_config_dict)
            
            if not validation_result.is_valid:
                results[f'invalid_{test_name}'] = f"PASS - Correctly rejected"
                logger.info(f"✅ Invalid config correctly rejected: {test_name}")
            else:
                results[f'invalid_{test_name}'] = "FAIL - Invalid config accepted"
                logger.error(f"❌ Invalid config incorrectly accepted: {test_name}")
                
        except Exception as e:
            results[f'invalid_{test_name}'] = f"FAIL - {e}"
            logger.error(f"❌ Validation test failed for {test_name}: {e}")
    
    # Test different validation levels
    try:
        config_manager = ConfigurationManager()
        config = config_manager.load_configuration(Environment.PRODUCTION, ConfigValidationLevel.MINIMAL)
        
        for level in [ConfigValidationLevel.MINIMAL, ConfigValidationLevel.STANDARD, 
                     ConfigValidationLevel.STRICT, ConfigValidationLevel.PRODUCTION]:
            validator.validation_level = level
            result = validator.validate_configuration(config)
            
            results[f'validation_level_{level.name.lower()}'] = f"PASS - {len(result.errors)} errors, {len(result.warnings)} warnings"
            
    except Exception as e:
        results['validation_levels'] = f"FAIL - {e}"
        logger.error(f"❌ Validation level testing failed: {e}")
    
    return results

def test_robot_and_user_configs():
    """Test robot-specific and user-specific configuration loading"""
    logger = logging.getLogger("test_robot_user_configs")
    logger.info("Testing robot and user configuration management...")
    
    results = {}
    config_manager = ConfigurationManager()
    
    # Test robot-specific configuration loading
    try:
        robot_config = config_manager.get_robot_specific_config("exoskeleton_v1")
        
        if robot_config and 'robot_info' in robot_config:
            results['robot_config_loading'] = "PASS"
            logger.info("✅ Robot-specific configuration loaded successfully")
        else:
            results['robot_config_loading'] = "FAIL - Invalid robot config format"
            
    except Exception as e:
        results['robot_config_loading'] = f"FAIL - {e}"
        logger.error(f"❌ Robot config loading failed: {e}")
    
    # Test user preferences loading
    try:
        user_prefs = config_manager.load_user_preferences("patient_001")
        
        if user_prefs and user_prefs.user_id == "patient_001":
            results['user_prefs_loading'] = "PASS"
            logger.info("✅ User preferences loaded successfully")
        else:
            results['user_prefs_loading'] = "FAIL - Invalid user preferences"
            
    except Exception as e:
        results['user_prefs_loading'] = f"FAIL - {e}"
        logger.error(f"❌ User preferences loading failed: {e}")
    
    # Test user preferences saving
    try:
        from config.production_config import UserPreferences
        
        test_prefs = UserPreferences(
            user_id="test_user",
            preferred_assistance_level=0.8,
            preferred_control_mode="autonomous"
        )
        
        save_success = config_manager.save_user_preferences("test_user", test_prefs)
        
        if save_success:
            # Try to load back the saved preferences
            loaded_prefs = config_manager.load_user_preferences("test_user")
            
            if (loaded_prefs.user_id == "test_user" and 
                loaded_prefs.preferred_assistance_level == 0.8):
                results['user_prefs_saving'] = "PASS"
                logger.info("✅ User preferences save/load cycle successful")
            else:
                results['user_prefs_saving'] = "FAIL - Load data doesn't match saved data"
        else:
            results['user_prefs_saving'] = "FAIL - Save operation failed"
            
    except Exception as e:
        results['user_prefs_saving'] = f"FAIL - {e}"
        logger.error(f"❌ User preferences saving failed: {e}")
    
    return results

def test_configuration_persistence():
    """Test configuration saving and loading persistence"""
    logger = logging.getLogger("test_config_persistence")
    logger.info("Testing configuration persistence...")
    
    results = {}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create a temporary configuration manager
            temp_config_manager = ConfigurationManager(temp_dir)
            
            # Load and modify a configuration
            config = temp_config_manager.load_configuration(Environment.DEVELOPMENT)
            
            # Modify some values
            original_timeout = config.safety.watchdog_timeout_sec
            config.safety.watchdog_timeout_sec = 3.14159  # Distinctive value
            config.hardware.hardware_id = "persistence_test_robot"
            
            # Save the modified configuration
            save_success = temp_config_manager.save_configuration(config)
            
            if save_success:
                # Load the configuration again in a new manager
                new_config_manager = ConfigurationManager(temp_dir)
                reloaded_config = new_config_manager.load_configuration(Environment.DEVELOPMENT)
                
                # Check if modifications persisted
                if (reloaded_config.safety.watchdog_timeout_sec == 3.14159 and
                    reloaded_config.hardware.hardware_id == "persistence_test_robot"):
                    results['config_persistence'] = "PASS"
                    logger.info("✅ Configuration persistence working correctly")
                else:
                    results['config_persistence'] = "FAIL - Changes not persisted correctly"
            else:
                results['config_persistence'] = "FAIL - Save operation failed"
                
        except Exception as e:
            results['config_persistence'] = f"FAIL - {e}"
            logger.error(f"❌ Configuration persistence test failed: {e}")
    
    return results

def test_environment_specific_overrides():
    """Test environment-specific configuration overrides"""
    logger = logging.getLogger("test_env_overrides")
    logger.info("Testing environment-specific overrides...")
    
    results = {}
    
    try:
        config_manager = ConfigurationManager()
        
        # Test that different environments have different settings
        dev_config = config_manager.load_configuration(Environment.DEVELOPMENT)
        prod_config = config_manager.load_configuration(Environment.PRODUCTION)
        
        # Development should have more permissive safety settings
        if (dev_config.safety.watchdog_timeout_sec > prod_config.safety.watchdog_timeout_sec and
            dev_config.logging.log_level == "DEBUG" and
            prod_config.logging.log_level == "INFO"):
            
            results['env_specific_settings'] = "PASS"
            logger.info("✅ Environment-specific settings correctly applied")
        else:
            results['env_specific_settings'] = "FAIL - Settings not environment-appropriate"
        
        # Test staging is between dev and prod in strictness  
        staging_config = config_manager.load_configuration(Environment.STAGING)
        
        if (dev_config.safety.watchdog_timeout_sec > staging_config.safety.watchdog_timeout_sec > 
            prod_config.safety.watchdog_timeout_sec):
            results['staging_between_dev_prod'] = "PASS"
            logger.info("✅ Staging configuration appropriately positioned")
        else:
            results['staging_between_dev_prod'] = "FAIL - Staging settings not intermediate"
            
    except Exception as e:
        results['env_overrides'] = f"FAIL - {e}"
        logger.error(f"❌ Environment override testing failed: {e}")
    
    return results

def run_comprehensive_test():
    """Run all configuration system tests"""
    logger = setup_logging()
    logger.info("Starting comprehensive configuration system validation...")
    
    all_results = {}
    
    print("🔧 Safe RL Configuration Management System - Validation Test")
    print("=" * 70)
    
    # Run all test suites
    test_suites = [
        ("Environment Detection", test_environment_detection),
        ("Configuration Loading", test_configuration_loading), 
        ("Configuration Validation", test_configuration_validation),
        ("Robot & User Configs", test_robot_and_user_configs),
        ("Configuration Persistence", test_configuration_persistence),
        ("Environment Overrides", test_environment_specific_overrides)
    ]
    
    for suite_name, test_function in test_suites:
        print(f"\n📋 Testing {suite_name}...")
        try:
            suite_results = test_function()
            all_results[suite_name] = suite_results
            
            passed = sum(1 for result in suite_results.values() if result.startswith("PASS"))
            total = len(suite_results)
            print(f"   Results: {passed}/{total} tests passed")
            
            if passed < total:
                print("   Failed tests:")
                for test_name, result in suite_results.items():
                    if result.startswith("FAIL"):
                        print(f"   ❌ {test_name}: {result}")
            
        except Exception as e:
            all_results[suite_name] = {"error": f"FAIL - Test suite error: {e}"}
            logger.error(f"❌ Test suite {suite_name} failed: {e}")
    
    # Generate summary report
    print(f"\n📊 CONFIGURATION SYSTEM VALIDATION SUMMARY")
    print("=" * 70)
    
    total_tests = 0
    passed_tests = 0
    
    for suite_name, suite_results in all_results.items():
        suite_passed = sum(1 for result in suite_results.values() if result.startswith("PASS"))
        suite_total = len(suite_results)
        total_tests += suite_total
        passed_tests += suite_passed
        
        status_icon = "✅" if suite_passed == suite_total else "❌"
        print(f"{status_icon} {suite_name}: {suite_passed}/{suite_total}")
    
    overall_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🏆 OVERALL RESULTS:")
    print(f"   Tests Passed: {passed_tests}/{total_tests}")
    print(f"   Success Rate: {overall_score:.1f}%")
    
    if overall_score >= 90:
        print(f"   Status: 🟢 EXCELLENT - Configuration system ready for production")
    elif overall_score >= 80:
        print(f"   Status: 🟡 GOOD - Minor issues to address")
    elif overall_score >= 70:
        print(f"   Status: 🟠 ACCEPTABLE - Some fixes needed")
    else:
        print(f"   Status: 🔴 NEEDS WORK - Significant issues found")
    
    print(f"\n📄 Test completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return overall_score >= 80

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)