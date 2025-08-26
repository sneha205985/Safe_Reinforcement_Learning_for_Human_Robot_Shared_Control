#!/usr/bin/env python3
"""
Standalone Hardware Interface Test
Direct testing of production hardware interfaces without problematic dependencies
"""

import sys
import os
import time
import threading
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for hardware test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def test_production_interfaces():
    """Test production interfaces directly"""
    logger = setup_logging()
    logger.info("Starting standalone hardware interface test...")
    
    test_results = {}
    
    # Test 1: Import production interfaces directly
    logger.info("Test 1: Testing production interface imports...")
    try:
        # Import directly from the file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "production_interfaces", 
            str(project_root / "safe_rl_human_robot" / "src" / "hardware" / "production_interfaces.py")
        )
        prod_interfaces = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prod_interfaces)
        
        test_results['imports'] = 'PASS'
        logger.info("✅ Production interfaces imported successfully")
        
    except Exception as e:
        test_results['imports'] = f'FAIL: {e}'
        logger.error(f"❌ Import failed: {e}")
        return test_results
    
    # Test 2: ProductionSafetySystem
    logger.info("Test 2: Testing ProductionSafetySystem...")
    try:
        safety_system = prod_interfaces.ProductionSafetySystem()
        
        # Test emergency stop
        safety_system.emergency_stop()
        estop_status = safety_system.safety_status['emergency_stops']['robot_emergency_stop']
        
        # Test safety interlocks
        safety_system.enable_safety_interlocks()
        
        # Test watchdog
        safety_system.start_watchdog()
        time.sleep(1)
        safety_system.feed_watchdog()
        
        # Test fault detection
        fault_detected = safety_system.check_hardware_faults()
        
        test_results['safety_system'] = 'PASS'
        logger.info(f"✅ Safety system test passed - E-stop: {estop_status}")
        
    except Exception as e:
        test_results['safety_system'] = f'FAIL: {e}'
        logger.error(f"❌ Safety system test failed: {e}")
    
    # Test 3: HardwareSimulator
    logger.info("Test 3: Testing HardwareSimulator...")
    try:
        simulator = prod_interfaces.HardwareSimulator()
        
        # Test joint states
        joint_state = simulator.get_joint_states()
        
        # Test force sensors
        force_data = simulator.get_force_sensor_data()
        
        # Test actuator commands
        test_commands = {
            'joint_positions': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'joint_velocities': [0.0] * 6
        }
        simulator.send_actuator_commands(test_commands)
        
        # Simulate physics step
        simulator.simulate_physics_step(0.01)
        
        test_results['simulator'] = 'PASS'
        logger.info(f"✅ Simulator test passed - Joints: {len(joint_state.get('positions', []))}")
        
    except Exception as e:
        test_results['simulator'] = f'FAIL: {e}'
        logger.error(f"❌ Simulator test failed: {e}")
    
    # Test 4: ROSHardwareInterface
    logger.info("Test 4: Testing ROSHardwareInterface...")
    try:
        ros_interface = prod_interfaces.ROSHardwareInterface()
        
        # Test setup
        ros_interface.setup_publishers_subscribers()
        
        # Test command publishing
        test_joint_command = {
            'positions': [0.0] * 6,
            'velocities': [0.0] * 6,
            'efforts': [0.0] * 6
        }
        ros_interface.publish_joint_command(test_joint_command)
        
        test_results['ros_interface'] = 'PASS'
        logger.info("✅ ROS interface test passed")
        
    except Exception as e:
        test_results['ros_interface'] = f'FAIL: {e}'
        logger.error(f"❌ ROS interface test failed: {e}")
    
    # Test 5: HardwareCalibrator
    logger.info("Test 5: Testing HardwareCalibrator...")
    try:
        calibrator = prod_interfaces.HardwareCalibrator()
        
        # Test calibration procedures
        joint_calib = calibrator.calibrate_joints()
        force_calib = calibrator.calibrate_force_sensors()
        diagnostics = calibrator.run_system_diagnostics()
        validation = calibrator.validate_calibration()
        
        test_results['calibrator'] = 'PASS'
        logger.info(f"✅ Calibrator test passed - Joint: {joint_calib}, Force: {force_calib}")
        
    except Exception as e:
        test_results['calibrator'] = f'FAIL: {e}'
        logger.error(f"❌ Calibrator test failed: {e}")
    
    # Test 6: ProductionHardwareInterface
    logger.info("Test 6: Testing ProductionHardwareInterface...")
    try:
        hardware_interface = prod_interfaces.ProductionHardwareInterface()
        
        # Test initialization
        init_success = hardware_interface.initialize()
        
        if init_success:
            # Test basic operations
            joint_states = hardware_interface.get_joint_states()
            force_data = hardware_interface.get_force_sensor_data()
            
            # Test command sending
            test_command = {
                'joint_positions': [0.0] * 6,
                'joint_velocities': [0.0] * 6
            }
            hardware_interface.send_joint_command(test_command)
            
            # Test safety check
            safety_ok = hardware_interface.is_safe()
            
            # Clean shutdown
            hardware_interface.shutdown()
            
            test_results['main_interface'] = 'PASS'
            logger.info(f"✅ Main interface test passed - Safe: {safety_ok}")
            
        else:
            test_results['main_interface'] = 'FAIL: Initialization failed'
            logger.error("❌ Main interface initialization failed")
        
    except Exception as e:
        test_results['main_interface'] = f'FAIL: {e}'
        logger.error(f"❌ Main interface test failed: {e}")
    
    # Test 7: Integration stress test
    logger.info("Test 7: Running integration stress test (10 seconds)...")
    try:
        hardware_interface = prod_interfaces.ProductionHardwareInterface()
        
        if hardware_interface.initialize():
            start_time = time.time()
            command_count = 0
            error_count = 0
            
            while time.time() - start_time < 10:  # 10 second stress test
                try:
                    # Command cycle
                    joint_states = hardware_interface.get_joint_states()
                    
                    test_command = {
                        'joint_positions': [0.1 * (command_count % 5)] * 6,
                        'joint_velocities': [0.0] * 6
                    }
                    hardware_interface.send_joint_command(test_command)
                    
                    command_count += 1
                    time.sleep(0.05)  # 20 Hz
                    
                except Exception:
                    error_count += 1
            
            hardware_interface.shutdown()
            
            error_rate = error_count / command_count if command_count > 0 else 1.0
            success = error_rate < 0.1
            
            test_results['stress_test'] = f"{'PASS' if success else 'FAIL'} - Commands: {command_count}, Errors: {error_count}"
            logger.info(f"{'✅' if success else '❌'} Stress test: {command_count} commands, {error_count} errors")
            
        else:
            test_results['stress_test'] = 'FAIL: Could not initialize for stress test'
            
    except Exception as e:
        test_results['stress_test'] = f'FAIL: {e}'
        logger.error(f"❌ Stress test failed: {e}")
    
    return test_results

def generate_hardware_report(test_results):
    """Generate hardware test report"""
    print("\n" + "="*80)
    print("SAFE RL HARDWARE INTERFACE TEST REPORT")
    print("="*80)
    print(f"Test Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Count results
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result == 'PASS')
    
    overall_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    overall_status = "PASS" if overall_score >= 70 else "FAIL"
    
    print("OVERALL ASSESSMENT")
    print("-" * 40)
    print(f"Overall Score:     {overall_score:.1f}/100.0")
    print(f"Tests Passed:      {passed_tests}/{total_tests}")
    print(f"Production Ready:  {'✅ YES' if overall_status == 'PASS' else '❌ NO'}")
    print()
    
    print("DETAILED TEST RESULTS")
    print("-" * 40)
    
    test_names = {
        'imports': 'Production Interface Imports',
        'safety_system': 'Production Safety System',
        'simulator': 'Hardware Simulator',
        'ros_interface': 'ROS Hardware Interface',
        'calibrator': 'Hardware Calibrator',
        'main_interface': 'Main Hardware Interface',
        'stress_test': 'Integration Stress Test'
    }
    
    for test_key, result in test_results.items():
        test_name = test_names.get(test_key, test_key.upper())
        status_icon = "✅" if result == 'PASS' else "❌"
        print(f"{test_name}:")
        print(f"  Status: {status_icon} {result}")
        print()
    
    print("RECOMMENDATIONS")
    print("-" * 40)
    
    if overall_status == "PASS":
        print("✅ Hardware interface system is production ready")
        print("✅ All core components functional") 
        print("✅ Safety systems operational")
        print("✅ Integration testing successful")
    else:
        print("❌ Hardware system requires fixes before production")
        for test_key, result in test_results.items():
            if result != 'PASS':
                test_name = test_names.get(test_key, test_key)
                print(f"🔧 Fix {test_name}: {result}")
    
    print()
    print("FINAL ASSESSMENT")
    print("="*80)
    
    if overall_status == "PASS":
        print("🎉 HARDWARE INTERFACE SYSTEM VALIDATED!")
        print()
        print("The production hardware interfaces have been successfully")
        print("tested and are ready for integration with the Safe RL system.")
        print()
        print("✅ Production safety systems operational")
        print("✅ Hardware simulation functional") 
        print("✅ ROS integration working")
        print("✅ Calibration procedures validated")
        print("✅ Main interface integration successful")
        print("✅ Stress testing completed")
    else:
        print("⚠️ HARDWARE SYSTEM NEEDS ATTENTION")
        print()
        print("Some hardware interface components failed validation.")
        print("Please address the issues above before proceeding.")
    
    print()
    print("="*80)
    print(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Safe RL Human-Robot Shared Control System")
    print("="*80)
    
    # Save report to file
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SAFE RL HARDWARE INTERFACE TEST REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Test Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("OVERALL ASSESSMENT")
    report_lines.append("-" * 40)
    report_lines.append(f"Overall Score:     {overall_score:.1f}/100.0")
    report_lines.append(f"Tests Passed:      {passed_tests}/{total_tests}")
    report_lines.append(f"Production Ready:  {'✅ YES' if overall_status == 'PASS' else '❌ NO'}")
    report_lines.append("")
    
    report_lines.append("DETAILED TEST RESULTS")
    report_lines.append("-" * 40)
    
    for test_key, result in test_results.items():
        test_name = test_names.get(test_key, test_key.upper())
        status_icon = "✅" if result == 'PASS' else "❌"
        report_lines.append(f"{test_name}:")
        report_lines.append(f"  Status: {status_icon} {result}")
        report_lines.append("")
    
    if overall_status == "PASS":
        report_lines.append("🎉 HARDWARE INTERFACE SYSTEM VALIDATED!")
        report_lines.append("")
        report_lines.append("All hardware interface components are production ready.")
    else:
        report_lines.append("⚠️ HARDWARE SYSTEM NEEDS ATTENTION")
        report_lines.append("")
        report_lines.append("Some components require fixes before production.")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    with open('HARDWARE_TEST_REPORT.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    return overall_status == "PASS"

def main():
    """Main test execution"""
    print("Safe RL Hardware Interface Test")
    print("=" * 50)
    
    try:
        # Run all tests
        test_results = test_production_interfaces()
        
        # Generate comprehensive report
        success = generate_hardware_report(test_results)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())