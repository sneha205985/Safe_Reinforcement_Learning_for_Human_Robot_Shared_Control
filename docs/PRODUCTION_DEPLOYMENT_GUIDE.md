# Production Deployment Guide
## Safe RL Human-Robot Shared Control System

### Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Architecture](#deployment-architecture)
4. [Environment Setup](#environment-setup)
5. [Deployment Configurations](#deployment-configurations)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Security Configuration](#security-configuration)
8. [Disaster Recovery](#disaster-recovery)
9. [Maintenance and Operations](#maintenance-and-operations)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide provides comprehensive instructions for deploying the Safe RL Human-Robot Shared Control System in production environments. The system supports three deployment modes:

- **Edge Deployment**: Optimized for edge computing with offline capabilities
- **Cloud Deployment**: Scalable cloud-native deployment using AWS ECS
- **Hybrid Deployment**: Intelligent orchestration between edge and cloud resources

### Key Features

- **High Availability**: Multi-region deployment with automatic failover
- **Auto-scaling**: Dynamic scaling based on load and performance metrics
- **Security**: Enterprise-grade security with RBAC, MFA, and encryption
- **Monitoring**: Comprehensive observability with real-time metrics and alerting
- **Compliance**: GDPR, CCPA, and model governance compliance built-in
- **Disaster Recovery**: Automated backup and recovery procedures

---

## Prerequisites

### System Requirements

**Minimum Hardware Requirements:**
- CPU: 8 cores (16 recommended)
- Memory: 16GB RAM (32GB recommended)
- Storage: 100GB SSD (500GB recommended)
- GPU: NVIDIA GPU with 8GB VRAM (for inference acceleration)

**Software Requirements:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- Kubernetes 1.25+ (for cloud deployment)
- Python 3.9+
- PostgreSQL 14+
- Redis 7+

### Cloud Requirements (AWS)

**Required AWS Services:**
- Amazon ECS with Fargate
- Amazon RDS (PostgreSQL)
- Amazon ElastiCache (Redis)
- Amazon S3
- Amazon CloudWatch
- AWS Secrets Manager
- Application Load Balancer
- Route 53 (for DNS)

**IAM Permissions:**
- ECS task execution and management
- RDS and ElastiCache access
- S3 read/write permissions
- CloudWatch logs and metrics
- Secrets Manager access

---

## Deployment Architecture

### Edge Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Device   │    │   Local Cache   │    │  Safety Monitor │
│                 │    │                 │    │                 │
│ Safe RL API     │◄───┤ Model Registry  │    │ Real-time       │
│ Policy Inference│    │ Data Storage    │    │ Monitoring      │
│ Optimization    │    │ Metrics         │    │ Alerting        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Data Sync      │
                    │  (Optional)     │
                    │  Cloud Backup   │
                    └─────────────────┘
```

### Cloud Deployment Architecture

```
                    ┌─────────────────┐
                    │ Application     │
                    │ Load Balancer   │
                    └─────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│ Safe RL API   │ │ Policy        │ │ Safety        │
│ (ECS Fargate) │ │ Inference     │ │ Monitor       │
│ Auto-scaling  │ │ (ECS Fargate) │ │ (ECS Fargate) │
└───────────────┘ └───────────────┘ └───────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
┌───▼───┐        ┌────────▼────────┐        ┌───▼───┐
│ RDS   │        │ ElastiCache     │        │  S3   │
│ (PostgreSQL)   │ (Redis)         │        │ Models│
└───────┘        └─────────────────┘        └───────┘
```

### Hybrid Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Hybrid Orchestrator                     │
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Intelligent     │    │ Load Balancer                   │ │
│  │ Decision Engine │◄───┤ (ML-based routing)             │ │
│  └─────────────────┘    └─────────────────────────────────┘ │
└─────────┬───────────────────────────────────┬───────────────┘
          │                                   │
    ┌─────▼─────┐                       ┌─────▼─────┐
    │  Edge     │                       │  Cloud    │
    │  Nodes    │◄─────────────────────►│  Services │
    │           │     Data Sync &       │           │
    │  - Local  │     Model Dist.       │  - ECS    │
    │  - Fast   │                       │  - Auto   │
    │  - Offline│                       │  - Scale  │
    └───────────┘                       └───────────┘
```

---

## Environment Setup

### 1. Clone and Setup Repository

```bash
git clone https://github.com/your-org/Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control.git
cd Safe_Reinforcement_Learning_for_Human_Robot_Shared_Control
```

### 2. Environment Configuration

Create environment-specific configuration files:

```bash
# Development environment
cp config/environments/development.yaml.example config/environments/development.yaml

# Production environment
cp config/environments/production.yaml.example config/environments/production.yaml

# Edge environment
cp config/environments/edge.yaml.example config/environments/edge.yaml
```

### 3. SSL Certificates (Production)

Generate or obtain SSL certificates:

```bash
# Create SSL directory
mkdir -p deployment/ssl

# For development (self-signed)
openssl req -x509 -newkey rsa:4096 -keyout deployment/ssl/key.pem -out deployment/ssl/cert.pem -days 365 -nodes

# For production, obtain certificates from a CA or use Let's Encrypt
# Place certificates in deployment/ssl/
```

### 4. Environment Variables

Create `.env` file in the project root:

```bash
# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=safe_rl
DATABASE_USER=postgres
DATABASE_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password

# JWT Configuration
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24

# AWS Configuration (for cloud deployment)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=safe-rl-models
ECR_REPOSITORY=your-account.dkr.ecr.us-east-1.amazonaws.com/safe-rl

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=your_grafana_password
ALERTMANAGER_WEBHOOK_URL=your_webhook_url

# Security Configuration
ENCRYPTION_KEY=your_encryption_key_here
MFA_ENABLED=true
SECURITY_SCAN_ENABLED=true
```

---

## Deployment Configurations

### Edge Deployment

**Use Case**: Factory floors, autonomous vehicles, robotics labs with limited connectivity

**Deployment Steps**:

1. **Prepare Edge Environment**:
   ```bash
   # Create required directories
   sudo mkdir -p /opt/safe-rl/edge/{models,data,logs,cache,registry,safety,metrics,influx}
   sudo chown -R $USER:$USER /opt/safe-rl/edge/
   ```

2. **Deploy Edge Stack**:
   ```bash
   cd deployment/edge
   docker-compose -f docker-compose.edge.yml up -d
   ```

3. **Verify Edge Deployment**:
   ```bash
   # Check service health
   curl http://localhost:8000/health
   
   # Check monitoring
   open http://localhost:9090  # Prometheus
   open http://localhost:9000  # MinIO (local model registry)
   ```

**Edge Configuration Options**:
- `OFFLINE_MODE=true` - Enable offline operation
- `MODEL_CACHE_SIZE=2GB` - Local model cache size
- `INFERENCE_BATCH_SIZE=8` - Optimized for edge hardware
- `GPU_MEMORY_LIMIT=4GB` - GPU memory allocation

### Cloud Deployment (AWS ECS)

**Use Case**: Large-scale production deployments requiring high availability and auto-scaling

**Deployment Steps**:

1. **Build and Push Images**:
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
   
   # Build and push images
   docker build -t safe-rl-api:latest -f deployment/docker/Dockerfile .
   docker tag safe-rl-api:latest your-account.dkr.ecr.us-east-1.amazonaws.com/safe-rl-api:latest
   docker push your-account.dkr.ecr.us-east-1.amazonaws.com/safe-rl-api:latest
   ```

2. **Deploy CloudFormation Stack**:
   ```bash
   cd deployment/cloud
   aws cloudformation deploy \
     --template-file aws-ecs.yml \
     --stack-name safe-rl-production \
     --parameter-overrides \
       Environment=production \
       VpcId=vpc-12345678 \
       SubnetIds=subnet-12345678,subnet-87654321 \
       ClusterName=safe-rl-cluster \
       ImageRepository=your-account.dkr.ecr.us-east-1.amazonaws.com \
       DatabaseEndpoint=your-rds-endpoint.amazonaws.com \
       RedisEndpoint=your-redis-endpoint.cache.amazonaws.com \
     --capabilities CAPABILITY_IAM
   ```

3. **Configure Auto-scaling**:
   ```bash
   # Auto-scaling is configured in the CloudFormation template
   # Monitor scaling activities
   aws application-autoscaling describe-scaling-activities \
     --service-namespace ecs \
     --resource-id service/safe-rl-cluster/safe-rl-production-safe-rl-api
   ```

### Hybrid Deployment

**Use Case**: Organizations needing both edge capabilities and cloud scalability

**Deployment Steps**:

1. **Deploy Hybrid Orchestrator**:
   ```bash
   cd deployment/hybrid
   
   # Set environment variables
   export CLOUD_ENDPOINT=https://your-cloud-alb-endpoint.com
   export EDGE_ENDPOINTS=http://edge1:8000,http://edge2:8000
   export CLOUD_REGION=us-east-1
   
   # Deploy hybrid stack
   docker-compose -f docker-compose.hybrid.yml up -d
   ```

2. **Configure Edge Nodes**:
   ```bash
   # On each edge node, register with hybrid orchestrator
   curl -X POST http://hybrid-orchestrator:9000/register \
     -H "Content-Type: application/json" \
     -d '{"node_id": "edge-node-1", "endpoint": "http://edge-node-1:8000", "capabilities": ["inference", "safety-monitoring"]}'
   ```

3. **Monitor Hybrid Operations**:
   ```bash
   # Access hybrid monitoring dashboard
   open http://localhost:3000  # Grafana hybrid dashboard
   
   # Check orchestrator status
   curl http://localhost:9000/status
   ```

---

## Monitoring and Observability

### Metrics Collection

The system collects comprehensive metrics across all deployment modes:

**Performance Metrics**:
- Request latency and throughput
- Model inference time
- Resource utilization (CPU, memory, GPU)
- Cache hit rates

**Safety Metrics**:
- Safety constraint violations
- Human intervention rates
- System failure rates
- Recovery times

**Business Metrics**:
- User satisfaction scores
- Task completion rates
- System availability
- Cost metrics (cloud deployments)

### Alerting Configuration

**Critical Alerts**:
- Safety constraint violations
- System failures
- High error rates (>5%)
- High latency (>2 seconds)

**Warning Alerts**:
- High resource utilization (>80%)
- Model drift detection
- Security anomalies
- Backup failures

### Grafana Dashboards

Access monitoring dashboards:
- **Production Dashboard**: http://your-grafana-url:3000/d/production
- **Safety Dashboard**: http://your-grafana-url:3000/d/safety
- **Performance Dashboard**: http://your-grafana-url:3000/d/performance

---

## Security Configuration

### Authentication and Authorization

**Multi-Factor Authentication (MFA)**:
```python
# Enable MFA in configuration
security:
  mfa:
    enabled: true
    providers: ["totp", "sms"]
    backup_codes: true
```

**Role-Based Access Control (RBAC)**:
```yaml
roles:
  admin:
    permissions: ["*"]
  engineer:
    permissions: ["models:read", "models:write", "experiments:*"]
  operator:
    permissions: ["models:read", "monitoring:read"]
  auditor:
    permissions: ["logs:read", "compliance:read"]
```

### Network Security

**TLS Configuration**:
```bash
# Enable TLS in all services
export TLS_ENABLED=true
export MTLS_ENABLED=true  # For service-to-service communication
```

**Firewall Rules**:
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 80/tcp      # HTTP
sudo ufw allow 443/tcp     # HTTPS
sudo ufw allow 8000/tcp    # Safe RL API
sudo ufw enable
```

### Data Encryption

**At-Rest Encryption**:
- Database: PostgreSQL with TDE enabled
- File Storage: AES-256 encryption for all model files
- Backups: Encrypted using AWS KMS or similar

**In-Transit Encryption**:
- All API communications use TLS 1.3
- Inter-service communication uses mTLS
- Message queues use SSL/TLS encryption

---

## Disaster Recovery

### Backup Strategy

**Automated Backups**:
- **Full Backups**: Daily at 2:00 AM
- **Incremental Backups**: Every 4 hours
- **Model Backups**: After each model update
- **Configuration Backups**: After each configuration change

**Backup Locations**:
- Local storage (24-hour retention)
- Cloud storage (30-day retention)
- Off-site storage (1-year retention)

### Recovery Procedures

**Service Recovery**:
```bash
# Check disaster recovery system status
python -m safe_rl_human_robot.src.infrastructure.disaster_recovery status

# Initiate recovery for hardware failure
python -m safe_rl_human_robot.src.infrastructure.disaster_recovery recover \
  --disaster-type hardware_failure \
  --affected-services safe_rl_api,policy_inference
```

**Data Recovery**:
```bash
# List available backups
python -m safe_rl_human_robot.src.infrastructure.disaster_recovery list-backups

# Restore from specific backup
python -m safe_rl_human_robot.src.infrastructure.disaster_recovery restore \
  --backup-id full_backup_20241225_020000
```

### Recovery Time Objectives (RTO)

| Service | Criticality | RTO | RPO |
|---------|------------|-----|-----|
| Safe RL API | Critical | 15 minutes | 5 minutes |
| Policy Inference | High | 30 minutes | 15 minutes |
| Safety Monitor | Critical | 10 minutes | 2 minutes |
| Model Registry | Medium | 60 minutes | 30 minutes |
| Monitoring | Medium | 60 minutes | 30 minutes |

---

## Maintenance and Operations

### Regular Maintenance Tasks

**Daily**:
- Monitor system health dashboards
- Review security alerts
- Check backup completion status
- Verify model performance metrics

**Weekly**:
- Review system performance trends
- Update security patches
- Clean up old logs and metrics
- Test disaster recovery procedures

**Monthly**:
- Update model versions
- Review and rotate secrets
- Performance optimization review
- Security audit and compliance check

### Health Checks

**Automated Health Checks**:
```bash
# API health check
curl -f http://localhost:8000/health || echo "API health check failed"

# Database connectivity
curl -f http://localhost:8000/health/database || echo "Database health check failed"

# Model registry health
curl -f http://localhost:8000/health/models || echo "Model registry health check failed"
```

**Manual Health Checks**:
```bash
# Check service status
docker-compose ps

# Check resource usage
docker stats

# Check logs for errors
docker-compose logs --tail=100 | grep ERROR
```

### Updates and Upgrades

**Rolling Updates (Zero Downtime)**:
```bash
# Update Safe RL API
docker-compose pull safe-rl-api
docker-compose up -d --no-deps safe-rl-api

# Verify update success
curl http://localhost:8000/version
```

**Model Updates**:
```bash
# Deploy new model version
python -m safe_rl_human_robot.src.data_management.model_registry deploy \
  --model-name safe_rl_v2.1 \
  --deployment-strategy canary \
  --canary-percentage 10
```

---

## Troubleshooting

### Common Issues

**Issue: High Memory Usage**
```bash
# Check memory usage by service
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Solution: Adjust memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G  # Increase memory limit
```

**Issue: Model Loading Failures**
```bash
# Check model registry connectivity
curl http://localhost:9000/health

# Check model file integrity
python -c "
from safe_rl_human_robot.src.data_management.model_registry import ModelRegistry
registry = ModelRegistry()
registry.validate_model('model_id')
"
```

**Issue: High Latency**
```bash
# Check inference performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8001/inference

# Enable performance profiling
export PROFILING_ENABLED=true
docker-compose restart policy-inference
```

### Log Analysis

**Centralized Logging**:
```bash
# View all service logs
docker-compose logs -f

# Filter by service
docker-compose logs -f safe-rl-api

# Search for errors
docker-compose logs | grep -i error | tail -20
```

**Log Locations**:
- Application logs: `/opt/safe-rl/*/logs/`
- System logs: `/var/log/safe_rl/`
- Audit logs: `/opt/safe-rl/audit/`

### Performance Debugging

**CPU Profiling**:
```bash
# Enable CPU profiling
export CPU_PROFILING=true
docker-compose restart

# Generate profiling report
python -m safe_rl_human_robot.utils.profiling generate_cpu_report
```

**Memory Profiling**:
```bash
# Monitor memory usage
python -m safe_rl_human_robot.utils.profiling monitor_memory --duration 300
```

### Emergency Procedures

**Emergency Shutdown**:
```bash
# Graceful shutdown
docker-compose down --timeout 30

# Force shutdown (if graceful fails)
docker-compose kill
docker-compose down
```

**Emergency Rollback**:
```bash
# Rollback to previous version
docker-compose down
git checkout previous-stable-tag
docker-compose up -d
```

**Contact Information**:
- **On-call Engineer**: +1-555-SAFE-RL1
- **Security Team**: security@yourorg.com
- **DevOps Team**: devops@yourorg.com

---

## Performance Optimization Tips

1. **Model Optimization**: Use quantized models for edge deployment
2. **Caching Strategy**: Implement aggressive caching for frequently accessed models
3. **Load Balancing**: Use ML-based load balancing in hybrid deployments
4. **Resource Allocation**: Monitor and adjust CPU/memory limits based on usage patterns
5. **Database Optimization**: Use connection pooling and query optimization
6. **Network Optimization**: Enable compression and use CDNs for static assets

---

This deployment guide provides the foundation for running the Safe RL Human-Robot Shared Control System in production. For specific customizations or advanced configurations, consult the detailed API documentation and reach out to the development team.