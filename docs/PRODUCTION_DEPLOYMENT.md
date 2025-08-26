# 🚀 Production Deployment Guide

## Safe RL Human-Robot Shared Control System

**Version:** 1.0.0  
**Last Updated:** 2025-08-26  
**Deployment Level:** Production Ready

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Pre-Deployment Requirements](#pre-deployment-requirements)
3. [Step-by-Step Deployment Guide](#step-by-step-deployment-guide)
4. [Configuration Management](#configuration-management)
5. [Performance Monitoring Setup](#performance-monitoring-setup)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Maintenance Procedures](#maintenance-procedures)
8. [Emergency Procedures](#emergency-procedures)

---

## 🎯 System Overview

The Safe RL Human-Robot Shared Control System is an advanced rehabilitation robotics platform that combines:
- **Safe Reinforcement Learning** algorithms for adaptive control
- **Multi-modal hardware interfaces** (exoskeletons, wheelchairs)
- **Real-time safety monitoring** with emergency stop capabilities
- **User personalization** and adaptive assistance
- **Production-grade reliability** and monitoring

### Key Safety Features
- ⚡ **Sub-second emergency stop** response times
- 🛡️ **Multi-layer safety validation** with redundant checking
- 📊 **Real-time performance monitoring** and alerting
- 🔒 **Production security** with audit logging
- 🔄 **Automated backup and recovery** systems

---

## 🖥️ Pre-Deployment Requirements

### Hardware Requirements

#### **Minimum System Specifications**
- **CPU**: 8-core Intel/AMD processor (3.0GHz+)
- **Memory**: 16GB RAM (32GB recommended)
- **Storage**: 500GB SSD (NVMe recommended)
- **Network**: Gigabit Ethernet with low latency
- **GPU**: Optional - NVIDIA RTX 3070+ for enhanced ML performance

#### **Robot Hardware Support**
- **Exoskeletons**: EXO-V1, EXO-V2 series
- **Wheelchairs**: CHAIR-V2, CHAIR-V3 series
- **Communication**: CAN bus, Ethernet, ROS-compatible
- **Safety Hardware**: Emergency stop buttons, force sensors, watchdog timers

### Software Dependencies

#### **Core System Requirements**
```bash
# Operating System
Ubuntu 20.04 LTS or Ubuntu 22.04 LTS (recommended)

# Python Environment
Python 3.8+ (Python 3.10 recommended)
pip 21.0+
virtualenv or conda

# ROS (Robot Operating System)
ROS Noetic (Ubuntu 20.04) or ROS2 Humble (Ubuntu 22.04)

# Hardware Communication
can-utils
python-can
pyserial
```

#### **Python Dependencies**
```bash
# Core ML and RL
torch>=1.12.0
gymnasium>=0.26.0
stable-baselines3>=1.7.0
numpy>=1.21.0
scipy>=1.8.0

# Hardware Interface
rospy (ROS1) or rclpy (ROS2)
pyserial>=3.5
python-can>=4.0.0

# Configuration and Monitoring
PyYAML>=6.0
jsonschema>=4.0.0
psutil>=5.8.0
prometheus-client>=0.14.0

# Safety and Validation
cryptography>=3.4.0
pydantic>=1.9.0
```

### Network and Security Requirements

#### **Network Configuration**
- **Production Network**: Isolated VLAN with firewall protection
- **Monitoring Network**: Separate network for telemetry and monitoring
- **Emergency Network**: Backup communication channel
- **Bandwidth**: Minimum 100Mbps for real-time operation

#### **Security Requirements**
- **Firewall**: Hardware firewall with intrusion detection
- **Certificates**: SSL/TLS certificates for encrypted communication
- **Authentication**: Multi-factor authentication for admin access
- **Audit Logging**: Centralized logging with tamper protection

---

## 📦 Step-by-Step Deployment Guide

### Step 1: Environment Preparation

#### **1.1 System Installation**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    can-utils \
    python3-dev \
    python3-pip \
    python3-venv

# Install ROS (Ubuntu 20.04 - ROS Noetic)
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt update
sudo apt install -y ros-noetic-desktop-full

# Setup ROS environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### **1.2 User and Permissions Setup**
```bash
# Create safe-rl user
sudo useradd -m -s /bin/bash safe-rl
sudo usermod -aG dialout,can,sudo safe-rl

# Create application directories
sudo mkdir -p /opt/safe-rl
sudo chown safe-rl:safe-rl /opt/safe-rl

# Setup log directories
sudo mkdir -p /var/log/safe-rl
sudo chown safe-rl:safe-rl /var/log/safe-rl
```

### Step 2: Application Deployment

#### **2.1 Code Deployment**
```bash
# Switch to safe-rl user
sudo su - safe-rl

# Clone the repository
cd /opt/safe-rl
git clone https://github.com/your-org/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control.git
cd Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### **2.2 Configuration Setup**
```bash
# Copy production configuration
cp config/production.yaml config/current_config.yaml

# Set environment variables
echo "export SAFE_RL_ENV=production" >> ~/.bashrc
echo "export SAFE_RL_CONFIG_PATH=/opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control/config" >> ~/.bashrc
source ~/.bashrc

# Validate configuration
python3 scripts/test_configuration_system.py
```

### Step 3: Hardware Setup and Calibration

#### **3.1 Hardware Connection Verification**
```bash
# Test CAN bus connection
sudo modprobe can
sudo ip link set can0 up type can bitrate 1000000

# Verify CAN communication
candump can0 &
cansend can0 123#DEADBEEF

# Test serial connections
ls -la /dev/ttyACM* /dev/ttyUSB*
```

#### **3.2 Robot Calibration**
```bash
# Run hardware calibration
python3 -m safe_rl_human_robot.calibration.calibrate_robot \
    --robot-type exoskeleton \
    --robot-id prod_exo_001 \
    --config-file config/robots/exoskeleton_v1.yaml

# Verify calibration results
python3 scripts/validate_hardware_calibration.py
```

### Step 4: System Services Setup

#### **4.1 Systemd Services**
Create systemd service files:

```bash
# Safe RL Control Service
sudo tee /etc/systemd/system/safe-rl-control.service > /dev/null <<EOF
[Unit]
Description=Safe RL Human-Robot Control System
After=network.target
Requires=network.target

[Service]
Type=simple
User=safe-rl
Group=safe-rl
WorkingDirectory=/opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control
Environment=SAFE_RL_ENV=production
ExecStart=/opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control/venv/bin/python -m safe_rl_human_robot.main
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Safety Monitor Service
sudo tee /etc/systemd/system/safe-rl-safety.service > /dev/null <<EOF
[Unit]
Description=Safe RL Safety Monitor
After=network.target safe-rl-control.service
Requires=network.target

[Service]
Type=simple
User=safe-rl
Group=safe-rl
WorkingDirectory=/opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control
Environment=SAFE_RL_ENV=production
ExecStart=/opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control/venv/bin/python -m safe_rl_human_robot.safety.monitor
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable safe-rl-control.service
sudo systemctl enable safe-rl-safety.service
sudo systemctl start safe-rl-control.service
sudo systemctl start safe-rl-safety.service
```

#### **4.2 Service Verification**
```bash
# Check service status
sudo systemctl status safe-rl-control.service
sudo systemctl status safe-rl-safety.service

# Monitor logs
sudo journalctl -u safe-rl-control.service -f
sudo journalctl -u safe-rl-safety.service -f
```

### Step 5: Security Hardening

#### **5.1 Firewall Configuration**
```bash
# Install and configure UFW
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port as needed)
sudo ufw allow 22/tcp

# Allow application ports
sudo ufw allow 8080/tcp  # Web interface
sudo ufw allow 8443/tcp  # Secure API

# Enable firewall
sudo ufw enable
```

#### **5.2 SSL/TLS Setup**
```bash
# Generate SSL certificates (use proper CA in production)
sudo mkdir -p /opt/safe-rl/certs
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /opt/safe-rl/certs/server.key \
    -out /opt/safe-rl/certs/server.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=safe-rl.local"

sudo chown -R safe-rl:safe-rl /opt/safe-rl/certs
sudo chmod 600 /opt/safe-rl/certs/server.key
```

---

## ⚙️ Configuration Management

### Configuration Files Structure
```
config/
├── production.yaml          # Production environment settings
├── staging.yaml            # Staging environment settings  
├── development.yaml         # Development settings
├── simulation.yaml          # Simulation environment
├── robots/                  # Robot-specific configurations
│   ├── exoskeleton_v1.yaml
│   └── wheelchair_v2.yaml
└── users/                   # User preferences
    ├── patient_001.yaml
    └── researcher_001.yaml
```

### Configuration Management Procedures

#### **Loading Configuration**
```python
from config.production_config import ConfigurationManager

# Initialize configuration manager
config_manager = ConfigurationManager()

# Load environment-specific configuration
config = config_manager.load_configuration()

# Load robot-specific parameters
robot_config = config_manager.get_robot_specific_config("exoskeleton_v1")

# Load user preferences
user_prefs = config_manager.load_user_preferences("patient_001")
```

#### **Configuration Validation**
```bash
# Validate all configurations
python3 scripts/test_configuration_system.py

# Validate specific environment
python3 -c "
from config.production_config import ConfigurationManager
cm = ConfigurationManager()
config = cm.load_configuration()
result = cm.validate_current_config()
print(f'Valid: {result.is_valid}')
if result.errors:
    print(f'Errors: {result.errors}')
"
```

#### **Configuration Updates**
```bash
# Backup current configuration
cp config/production.yaml config/production.yaml.backup.$(date +%Y%m%d_%H%M%S)

# Update configuration (example: increase safety monitoring rate)
sed -i 's/safety_monitoring_rate_hz: .*/safety_monitoring_rate_hz: 250.0/' config/production.yaml

# Validate updated configuration
python3 scripts/test_configuration_system.py

# Restart services to apply changes
sudo systemctl restart safe-rl-control.service
```

---

## 📊 Performance Monitoring Setup

### Monitoring Architecture
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification
- **Node Exporter**: System metrics collection

### Prometheus Setup

#### **Installation**
```bash
# Download and install Prometheus
cd /tmp
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
tar xvfz prometheus-2.45.0.linux-amd64.tar.gz
sudo mv prometheus-2.45.0.linux-amd64 /opt/prometheus

# Create prometheus user
sudo useradd --no-create-home --shell /bin/false prometheus
sudo chown -R prometheus:prometheus /opt/prometheus
```

#### **Configuration**
```yaml
# /opt/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "safe_rl_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - localhost:9093

scrape_configs:
  - job_name: 'safe-rl-system'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 5s
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

#### **Alert Rules**
```yaml
# /opt/prometheus/safe_rl_rules.yml
groups:
  - name: safe_rl_alerts
    rules:
    - alert: SafeRLSystemDown
      expr: up{job="safe-rl-system"} == 0
      for: 30s
      labels:
        severity: critical
      annotations:
        summary: "Safe RL system is down"
        
    - alert: HighCPUUsage
      expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High CPU usage detected"
        
    - alert: EmergencyStopActivated
      expr: safe_rl_emergency_stop_active == 1
      for: 0s
      labels:
        severity: critical
      annotations:
        summary: "Emergency stop has been activated"
        
    - alert: SafetyViolation
      expr: safe_rl_safety_violations_total > 0
      for: 0s
      labels:
        severity: critical
      annotations:
        summary: "Safety violation detected"
```

### Grafana Dashboard Setup

#### **Key Performance Indicators (KPIs)**

**System Health Dashboard:**
- System uptime and availability
- CPU, memory, and disk usage
- Network throughput and latency
- Service status and response times

**Safety Monitoring Dashboard:**
- Emergency stop status
- Safety violation counts
- Watchdog timer status
- Force and position limit violations

**Performance Dashboard:**
- Control loop frequency
- Algorithm convergence metrics
- Hardware response times
- User interaction latency

**User Activity Dashboard:**
- Active sessions count
- User assistance levels
- Task completion rates
- Error rates by user type

### Custom Metrics Implementation

```python
# Add to your application code
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
CONTROL_LOOP_DURATION = Histogram('safe_rl_control_loop_duration_seconds', 
                                 'Time spent in control loop')
SAFETY_VIOLATIONS = Counter('safe_rl_safety_violations_total',
                           'Total number of safety violations',
                           ['violation_type'])
EMERGENCY_STOP_STATUS = Gauge('safe_rl_emergency_stop_active',
                             'Emergency stop status (1=active, 0=inactive)')

# Use metrics in code
@CONTROL_LOOP_DURATION.time()
def control_loop():
    # Control loop implementation
    pass

def handle_safety_violation(violation_type):
    SAFETY_VIOLATIONS.labels(violation_type=violation_type).inc()

def update_emergency_stop_status(active):
    EMERGENCY_STOP_STATUS.set(1 if active else 0)

# Start metrics server
start_http_server(8080)
```

---

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### **System Startup Issues**

**Problem**: Safe RL control service fails to start
```bash
# Check service logs
sudo journalctl -u safe-rl-control.service -n 50

# Common solutions:
# 1. Check configuration validity
python3 scripts/test_configuration_system.py

# 2. Verify hardware connections
sudo dmesg | grep -i usb
candump can0

# 3. Check permissions
ls -la /dev/ttyACM* /dev/ttyUSB*
sudo usermod -aG dialout safe-rl

# 4. Restart services
sudo systemctl restart safe-rl-control.service
```

**Problem**: Hardware not detected
```bash
# Check hardware connections
lsusb  # USB devices
ls /dev/tty*  # Serial ports
ip link show  # Network interfaces

# Test CAN bus
sudo modprobe can
sudo ip link set can0 up type can bitrate 1000000
candump can0

# Check hardware configuration
python3 -c "
from config.production_config import ConfigurationManager
cm = ConfigurationManager()
robot_config = cm.get_robot_specific_config('exoskeleton_v1')
print(robot_config['communication'])
"
```

#### **Performance Issues**

**Problem**: High latency or low control frequency
```bash
# Check system resources
htop
iostat -x 1
sar -n DEV 1

# Check network latency
ping -c 10 192.168.1.100  # Replace with robot IP

# Monitor control loop timing
tail -f /var/log/safe-rl/performance.log | grep "control_loop_duration"

# Adjust CPU governor
sudo cpupower frequency-set -g performance
```

**Problem**: Memory leaks or high memory usage
```bash
# Monitor memory usage
watch -n 1 'free -m'
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full python3 -m safe_rl_human_robot.main

# Restart services if needed
sudo systemctl restart safe-rl-control.service
```

#### **Safety System Issues**

**Problem**: Emergency stop not responding
```bash
# Check emergency stop hardware
# 1. Verify physical button connections
# 2. Test software emergency stop
python3 -c "
from safe_rl_human_robot.safety.emergency_stop import EmergencyStopManager
esm = EmergencyStopManager()
esm.activate_emergency_stop('test')
print(f'Emergency stop active: {esm.is_emergency_stop_active()}')
"

# 3. Check safety monitor service
sudo systemctl status safe-rl-safety.service
sudo journalctl -u safe-rl-safety.service -n 20
```

**Problem**: Safety violations occurring frequently
```bash
# Analyze safety logs
grep "safety_violation" /var/log/safe-rl/*.log | tail -20

# Check safety limits configuration
python3 -c "
from config.production_config import ConfigurationManager
cm = ConfigurationManager()
config = cm.load_configuration()
print('Force limits:', config.safety.max_force_x, config.safety.max_force_y, config.safety.max_force_z)
print('Position limits enabled:', config.safety.position_limits_enabled)
"

# Adjust safety parameters if needed (with extreme caution)
# Only modify after thorough analysis and approval
```

### Diagnostic Commands

#### **System Health Check**
```bash
#!/bin/bash
# /opt/safe-rl/scripts/health_check.sh

echo "=== Safe RL System Health Check ==="
echo "Timestamp: $(date)"
echo ""

echo "=== Service Status ==="
systemctl is-active safe-rl-control.service
systemctl is-active safe-rl-safety.service
echo ""

echo "=== Hardware Status ==="
echo "CAN interfaces:"
ip link show | grep can
echo ""
echo "Serial ports:"
ls -la /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "No serial ports found"
echo ""

echo "=== System Resources ==="
echo "CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo "Memory usage:"
free -m | awk 'NR==2{printf "%.1f%%\n", $3*100/$2}'
echo "Disk usage:"
df -h | grep -E '^/dev/' | awk '{print $5 " " $6}'
echo ""

echo "=== Configuration Status ==="
cd /opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control
python3 -c "
from config.production_config import ConfigurationManager
cm = ConfigurationManager()
try:
    config = cm.load_configuration()
    result = cm.validate_current_config()
    print(f'Configuration valid: {result.is_valid}')
    if result.errors:
        print(f'Errors: {result.errors}')
except Exception as e:
    print(f'Configuration error: {e}')
"

echo "=== Health Check Complete ==="
```

---

## 🔄 Maintenance Procedures

### Regular Maintenance Schedule

#### **Daily Maintenance (Automated)**
- **System Health Checks**: Automated monitoring and alerting
- **Log Rotation**: Automatic log file rotation and archival
- **Backup Verification**: Verify automated backup completion
- **Performance Monitoring**: Check KPIs and alert thresholds

#### **Weekly Maintenance**
```bash
# System updates (non-critical)
sudo apt update
sudo apt list --upgradable

# Log analysis
cd /var/log/safe-rl
grep -i error *.log | tail -50
grep -i warning *.log | tail -50

# Performance review
python3 /opt/safe-rl/scripts/weekly_performance_report.py

# Configuration backup
cp -r /opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control/config \
      /opt/safe-rl/backups/config_$(date +%Y%m%d)
```

#### **Monthly Maintenance**
```bash
# Full system backup
rsync -av /opt/safe-rl/ /backup/safe-rl-$(date +%Y%m)

# Hardware calibration verification
python3 scripts/validate_hardware_calibration.py

# Security audit
sudo lynis audit system

# Performance optimization
python3 /opt/safe-rl/scripts/performance_optimization.py

# Update documentation
git pull origin main  # Update to latest documentation
```

#### **Quarterly Maintenance**
- **Full System Testing**: Complete system validation including hardware tests
- **Security Updates**: Apply critical security patches
- **Performance Tuning**: Optimize system parameters based on usage patterns
- **Hardware Inspection**: Physical inspection of robot hardware
- **Staff Training**: Update staff on new procedures and features

### Maintenance Commands

#### **System Cleanup**
```bash
# Clean log files older than 30 days
find /var/log/safe-rl -name "*.log" -mtime +30 -delete

# Clean temporary files
rm -rf /tmp/safe-rl-*
rm -rf /opt/safe-rl/cache/*

# Clean old backups (keep last 10)
ls -t /opt/safe-rl/backups/config_* | tail -n +11 | xargs rm -rf
```

#### **Performance Optimization**
```bash
# Optimize Python bytecode
python3 -m compileall /opt/safe-rl/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control/

# Clear system caches
sudo sync
sudo echo 3 > /proc/sys/vm/drop_caches

# Optimize database (if applicable)
# sqlite3 /opt/safe-rl/data/safe_rl.db "VACUUM;"
```

---

## 🚨 Emergency Procedures

### Emergency Response Protocols

#### **Immediate Response (First 60 seconds)**

**Step 1: Ensure Safety**
1. **Physical Emergency Stop**: Press red emergency stop button
2. **Software Emergency Stop**: Use keyboard shortcut `Ctrl+Alt+E`
3. **Power Disconnect**: If needed, disconnect main power to robot
4. **Clear Area**: Ensure all personnel are clear of robot workspace

**Step 2: Assess Situation**
1. Check for injuries - call medical assistance if needed
2. Identify the cause of emergency (hardware, software, user error)
3. Document incident with timestamp and description
4. Take photos/videos if safe to do so

**Step 3: System Isolation**
```bash
# Stop all Safe RL services immediately
sudo systemctl stop safe-rl-control.service
sudo systemctl stop safe-rl-safety.service

# Disable automatic restart
sudo systemctl disable safe-rl-control.service

# Check system status
python3 /opt/safe-rl/scripts/emergency_status_check.py
```

#### **Emergency Contact Information**

**Primary Contacts:**
- **System Administrator**: +1-XXX-XXX-XXXX
- **Safety Engineer**: +1-XXX-XXX-XXXX  
- **Medical Emergency**: 911
- **Technical Support**: support@safe-rl.com

**Secondary Contacts:**
- **Hardware Vendor Support**: +1-XXX-XXX-XXXX
- **IT Department**: +1-XXX-XXX-XXXX
- **Facility Management**: +1-XXX-XXX-XXXX

#### **Emergency Recovery Procedures**

**Hardware Emergency Recovery:**
```bash
# 1. Hardware reset sequence
sudo systemctl stop safe-rl-control.service
sleep 5

# 2. Hardware power cycle
# Physical power cycle of robot hardware (manual procedure)

# 3. Communication reset
sudo modprobe -r can
sudo modprobe can
sudo ip link set can0 up type can bitrate 1000000

# 4. Test hardware communication
candump can0 &
cansend can0 123#00000000

# 5. Run hardware diagnostics
python3 scripts/hardware_diagnostics.py --emergency-mode

# 6. If diagnostics pass, restart system
sudo systemctl start safe-rl-control.service
```

**Software Emergency Recovery:**
```bash
# 1. Kill all Safe RL processes
sudo pkill -f safe_rl_human_robot
sudo systemctl stop safe-rl-control.service
sudo systemctl stop safe-rl-safety.service

# 2. Clear any locks or temporary files
rm -f /opt/safe-rl/run/*.lock
rm -rf /tmp/safe-rl-*

# 3. Restart in safe mode
export SAFE_RL_EMERGENCY_MODE=true
sudo systemctl start safe-rl-safety.service
sleep 10
sudo systemctl start safe-rl-control.service

# 4. Monitor for 5 minutes before normal operation
tail -f /var/log/safe-rl/emergency.log
```

### Incident Reporting

#### **Incident Report Template**
```
SAFE RL INCIDENT REPORT
======================
Incident ID: INC-$(date +%Y%m%d-%H%M%S)
Date/Time: $(date)
Reporter: [Name and Role]
Severity: [Critical/High/Medium/Low]

INCIDENT DESCRIPTION:
- What happened?
- When did it occur?
- Who was involved?
- What system components were affected?

IMMEDIATE ACTIONS TAKEN:
- Emergency stops activated?
- System services stopped?
- Personnel evacuated?
- Medical attention required?

SYSTEM STATE AT TIME OF INCIDENT:
- Active users: [Number and IDs]
- Robot configuration: [Hardware type and settings]
- Software version: [Version numbers]
- Environmental conditions: [Temperature, etc.]

ROOT CAUSE ANALYSIS:
- Primary cause:
- Contributing factors:
- System logs analysis:

CORRECTIVE ACTIONS:
- Immediate fixes applied:
- Long-term improvements planned:
- Staff training requirements:

LESSONS LEARNED:
- What could have prevented this?
- Process improvements needed:
- Documentation updates required:

Report completed by: [Name and Signature]
Date: $(date)
```

#### **Post-Incident Procedures**
1. **Immediate Documentation**: Complete incident report within 2 hours
2. **System Quarantine**: Keep system offline until investigation complete
3. **Evidence Preservation**: Preserve all logs, configurations, and hardware state
4. **Stakeholder Notification**: Inform management and relevant parties
5. **Investigation**: Conduct thorough root cause analysis
6. **Corrective Actions**: Implement fixes and preventive measures
7. **System Validation**: Complete testing before returning to service
8. **Follow-up**: Monitor system closely for 48 hours after restart

---

## 📚 Additional Resources

### Training Materials
- **Operator Training Manual**: `docs/OPERATOR_MANUAL.md`
- **System Administration Guide**: `docs/ADMIN_GUIDE.md`
- **Emergency Procedures**: `docs/EMERGENCY_PROCEDURES.md`
- **Video Training Library**: Available on internal portal

### Technical Documentation
- **API Reference**: `docs/API_REFERENCE.md`
- **Configuration Schema**: `docs/CONFIG_SCHEMA.md`
- **Hardware Interface Specifications**: `docs/HARDWARE_SPECS.md`
- **Safety Standards Compliance**: `docs/SAFETY_COMPLIANCE.md`

### Support Resources
- **Technical Support Portal**: https://support.safe-rl.com
- **Knowledge Base**: https://kb.safe-rl.com
- **Community Forum**: https://community.safe-rl.com
- **Bug Tracking**: https://bugs.safe-rl.com

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-08-26  
**Next Review Date**: 2025-11-26  
**Approved By**: [System Administrator Name]

---

*This document is confidential and proprietary. Distribution is restricted to authorized personnel only.*