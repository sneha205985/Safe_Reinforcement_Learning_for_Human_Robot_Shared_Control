# API Documentation
## Safe RL Human-Robot Shared Control System

### Table of Contents
1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core APIs](#core-apis)
4. [Model Management APIs](#model-management-apis)
5. [Safety Monitoring APIs](#safety-monitoring-apis)
6. [Data Management APIs](#data-management-apis)
7. [Monitoring & Metrics APIs](#monitoring--metrics-apis)
8. [Administrative APIs](#administrative-apis)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)
11. [SDK Examples](#sdk-examples)

---

## Overview

The Safe RL API provides RESTful endpoints for managing safe reinforcement learning models, policy inference, safety monitoring, and system administration. All APIs use JSON for request/response bodies and support standard HTTP methods.

**Base URL**: `https://api.safe-rl.yourorg.com/v1`

**Supported Formats**:
- Request: `application/json`
- Response: `application/json`
- File uploads: `multipart/form-data`

---

## Authentication

### JWT Bearer Token Authentication

All API requests require a valid JWT bearer token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

### Obtaining Access Tokens

**POST** `/auth/login`

```json
{
  "username": "user@example.com",
  "password": "secure_password",
  "mfa_token": "123456"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user123",
    "email": "user@example.com",
    "roles": ["engineer", "model-operator"]
  }
}
```

### Token Refresh

**POST** `/auth/refresh`

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

---

## Core APIs

### Policy Inference

**POST** `/inference/policy`

Execute policy inference for safe decision making.

**Request**:
```json
{
  "model_id": "safe_rl_v2.1",
  "state": {
    "robot_position": [1.0, 2.0, 0.5],
    "human_position": [2.0, 2.5, 0.0],
    "environment_state": {
      "obstacles": [[0.5, 1.0], [3.0, 4.0]],
      "goal": [5.0, 5.0, 0.0]
    },
    "sensor_data": {
      "lidar": [0.1, 0.2, 0.3, ...],
      "camera": "base64_encoded_image_data"
    }
  },
  "safety_constraints": {
    "max_velocity": 2.0,
    "min_human_distance": 1.5,
    "collision_avoidance": true
  },
  "inference_options": {
    "return_confidence": true,
    "return_safety_scores": true,
    "explain_decision": true
  }
}
```

**Response**:
```json
{
  "inference_id": "inf_abc123",
  "action": {
    "type": "move",
    "parameters": {
      "velocity": [1.2, 0.8, 0.0],
      "direction": 45.5,
      "duration": 0.1
    }
  },
  "confidence": 0.92,
  "safety_scores": {
    "collision_risk": 0.05,
    "human_safety": 0.98,
    "constraint_satisfaction": 0.95
  },
  "explanation": {
    "primary_factors": ["human_proximity", "goal_alignment"],
    "safety_considerations": ["maintain_safe_distance", "avoid_obstacles"],
    "alternative_actions": [
      {
        "action": {"type": "stop"},
        "confidence": 0.87,
        "reason": "conservative_safety_approach"
      }
    ]
  },
  "metadata": {
    "model_version": "2.1.0",
    "inference_time_ms": 45,
    "gpu_utilization": 0.75
  }
}
```

### Batch Inference

**POST** `/inference/batch`

Execute batch policy inference for multiple states.

**Request**:
```json
{
  "model_id": "safe_rl_v2.1",
  "states": [
    {
      "request_id": "req_001",
      "state": { /* state data */ },
      "safety_constraints": { /* constraints */ }
    },
    {
      "request_id": "req_002",
      "state": { /* state data */ },
      "safety_constraints": { /* constraints */ }
    }
  ],
  "batch_options": {
    "max_batch_size": 32,
    "timeout_seconds": 30
  }
}
```

### Safety Validation

**POST** `/safety/validate`

Validate an action against safety constraints before execution.

**Request**:
```json
{
  "proposed_action": {
    "type": "move",
    "parameters": {
      "velocity": [2.0, 1.0, 0.0],
      "duration": 0.2
    }
  },
  "current_state": { /* current state */ },
  "safety_constraints": { /* constraints */ },
  "validation_level": "strict"
}
```

**Response**:
```json
{
  "validation_id": "val_xyz789",
  "is_safe": true,
  "safety_score": 0.94,
  "violations": [],
  "recommendations": [
    {
      "type": "velocity_reduction",
      "suggested_velocity": [1.5, 0.8, 0.0],
      "reason": "approaching_human_workspace"
    }
  ],
  "alternative_actions": [
    {
      "action": { /* modified action */ },
      "safety_score": 0.98
    }
  ]
}
```

---

## Model Management APIs

### List Models

**GET** `/models`

**Query Parameters**:
- `status`: Filter by status (active, archived, training)
- `version`: Filter by version
- `limit`: Number of results (default: 50, max: 200)
- `offset`: Pagination offset

**Response**:
```json
{
  "models": [
    {
      "id": "safe_rl_v2.1",
      "name": "Safe RL Human-Robot v2.1",
      "version": "2.1.0",
      "status": "active",
      "created_at": "2024-12-20T10:00:00Z",
      "updated_at": "2024-12-25T15:30:00Z",
      "metrics": {
        "accuracy": 0.945,
        "safety_score": 0.987,
        "inference_time_ms": 42
      },
      "deployment_info": {
        "deployed_at": "2024-12-25T16:00:00Z",
        "deployment_environment": "production",
        "resource_usage": {
          "gpu_memory_mb": 2048,
          "inference_qps": 150
        }
      }
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

### Get Model Details

**GET** `/models/{model_id}`

**Response**:
```json
{
  "id": "safe_rl_v2.1",
  "name": "Safe RL Human-Robot v2.1",
  "version": "2.1.0",
  "status": "active",
  "description": "Advanced safe reinforcement learning model for human-robot shared control with improved safety constraints and human behavior prediction",
  "architecture": {
    "type": "actor_critic",
    "layers": 6,
    "parameters": 12500000,
    "input_dimensions": [64, 64, 3],
    "output_dimensions": [8]
  },
  "training_info": {
    "dataset": "human_robot_interactions_v3",
    "training_duration_hours": 72,
    "training_episodes": 1000000,
    "convergence_metrics": {
      "final_reward": 847.2,
      "safety_violations": 0.002
    }
  },
  "validation_results": {
    "safety_tests_passed": 245,
    "safety_tests_failed": 0,
    "performance_benchmarks": {
      "task_completion_rate": 0.952,
      "human_satisfaction": 4.7,
      "collision_rate": 0.001
    }
  },
  "compliance": {
    "bias_assessment": "passed",
    "explainability_score": 0.89,
    "privacy_compliance": "gdpr_compliant"
  }
}
```

### Deploy Model

**POST** `/models/{model_id}/deploy`

**Request**:
```json
{
  "deployment_strategy": "blue_green",
  "target_environment": "production",
  "resource_allocation": {
    "gpu_memory_gb": 4,
    "cpu_cores": 2,
    "max_qps": 200
  },
  "deployment_options": {
    "canary_percentage": 10,
    "health_check_timeout": 300,
    "rollback_on_failure": true
  }
}
```

### Model Performance Metrics

**GET** `/models/{model_id}/metrics`

**Query Parameters**:
- `start_time`: Start time (ISO 8601)
- `end_time`: End time (ISO 8601)
- `metric_types`: Comma-separated list of metric types

**Response**:
```json
{
  "metrics": {
    "inference_latency": {
      "values": [45, 42, 48, 44, 46],
      "timestamps": ["2024-12-25T10:00:00Z", "2024-12-25T10:05:00Z", ...],
      "unit": "milliseconds",
      "aggregation": "average"
    },
    "safety_score": {
      "values": [0.987, 0.989, 0.985, 0.988, 0.990],
      "timestamps": ["2024-12-25T10:00:00Z", "2024-12-25T10:05:00Z", ...],
      "unit": "score",
      "aggregation": "average"
    },
    "error_rate": {
      "values": [0.002, 0.001, 0.003, 0.002, 0.001],
      "timestamps": ["2024-12-25T10:00:00Z", "2024-12-25T10:05:00Z", ...],
      "unit": "percentage",
      "aggregation": "average"
    }
  },
  "summary": {
    "total_inferences": 15420,
    "avg_latency_ms": 45.2,
    "avg_safety_score": 0.9878,
    "error_rate": 0.0018
  }
}
```

---

## Safety Monitoring APIs

### Real-time Safety Status

**GET** `/safety/status`

**Response**:
```json
{
  "overall_status": "healthy",
  "last_updated": "2024-12-25T15:45:30Z",
  "active_sessions": 12,
  "safety_metrics": {
    "violations_last_hour": 0,
    "violations_last_24h": 2,
    "human_interventions_last_hour": 1,
    "system_reliability": 0.9995
  },
  "active_constraints": [
    {
      "id": "min_human_distance",
      "type": "distance_constraint",
      "value": 1.5,
      "unit": "meters",
      "status": "active"
    },
    {
      "id": "max_velocity",
      "type": "velocity_constraint", 
      "value": 2.0,
      "unit": "m/s",
      "status": "active"
    }
  ],
  "alerts": [
    {
      "id": "alert_001",
      "severity": "warning",
      "message": "Human proximity detected - reducing velocity",
      "timestamp": "2024-12-25T15:42:15Z",
      "resolved": false
    }
  ]
}
```

### Safety Incidents

**GET** `/safety/incidents`

**Query Parameters**:
- `severity`: Filter by severity (critical, high, medium, low)
- `status`: Filter by status (open, investigating, resolved)
- `start_time`: Start time for incident search
- `limit`: Number of results

**Response**:
```json
{
  "incidents": [
    {
      "id": "inc_001",
      "severity": "high",
      "status": "resolved",
      "title": "Safety constraint violation - minimum distance",
      "description": "Robot moved within 1.2m of human operator, below 1.5m threshold",
      "occurred_at": "2024-12-25T14:22:10Z",
      "resolved_at": "2024-12-25T14:25:45Z",
      "context": {
        "session_id": "sess_abc123",
        "model_id": "safe_rl_v2.1",
        "robot_position": [2.1, 2.8, 0.0],
        "human_position": [2.0, 2.9, 0.0],
        "action_taken": "emergency_stop"
      },
      "root_cause": "sensor_calibration_drift",
      "corrective_actions": [
        "Recalibrated distance sensors",
        "Updated safety thresholds",
        "Increased monitoring frequency"
      ],
      "prevention_measures": [
        "Implement predictive sensor drift detection",
        "Add redundant distance measurements"
      ]
    }
  ],
  "total": 5,
  "open_incidents": 0,
  "critical_incidents_24h": 0
}
```

### Report Safety Incident

**POST** `/safety/incidents`

**Request**:
```json
{
  "severity": "high",
  "title": "Unexpected robot behavior during human approach",
  "description": "Robot exhibited jerky movements when human approached from blind spot",
  "context": {
    "session_id": "sess_def456",
    "model_id": "safe_rl_v2.1",
    "timestamp": "2024-12-25T15:30:00Z",
    "environment_conditions": "factory_floor_simulation",
    "sensor_data": { /* relevant sensor readings */ }
  },
  "reporter": {
    "user_id": "user123",
    "role": "safety_operator"
  }
}
```

---

## Data Management APIs

### Dataset Management

**GET** `/datasets`

List available training datasets.

**Response**:
```json
{
  "datasets": [
    {
      "id": "human_robot_v3",
      "name": "Human-Robot Interaction Dataset v3",
      "version": "3.2.0",
      "size_gb": 45.7,
      "created_at": "2024-11-15T10:00:00Z",
      "metadata": {
        "num_episodes": 50000,
        "num_environments": 12,
        "human_participants": 150,
        "annotation_quality": 0.96
      },
      "compliance": {
        "consent_obtained": true,
        "anonymized": true,
        "gdpr_compliant": true
      }
    }
  ]
}
```

### Data Pipeline Status

**GET** `/data/pipeline/status`

**Response**:
```json
{
  "pipeline_status": "running",
  "last_run": "2024-12-25T12:00:00Z",
  "next_scheduled_run": "2024-12-25T18:00:00Z",
  "stages": [
    {
      "name": "data_ingestion",
      "status": "completed",
      "duration_seconds": 1847,
      "records_processed": 12500
    },
    {
      "name": "data_validation",
      "status": "completed",
      "duration_seconds": 324,
      "validation_errors": 0
    },
    {
      "name": "data_transformation",
      "status": "running",
      "progress_percent": 65,
      "estimated_completion": "2024-12-25T15:30:00Z"
    }
  ],
  "quality_metrics": {
    "data_completeness": 0.987,
    "data_accuracy": 0.994,
    "data_consistency": 0.991
  }
}
```

---

## Monitoring & Metrics APIs

### System Metrics

**GET** `/metrics/system`

**Query Parameters**:
- `start_time`: Start time (ISO 8601)
- `end_time`: End time (ISO 8601)
- `granularity`: Time granularity (1m, 5m, 1h, 1d)

**Response**:
```json
{
  "metrics": {
    "cpu_usage": {
      "current": 45.2,
      "average": 42.8,
      "peak": 78.5,
      "unit": "percent"
    },
    "memory_usage": {
      "current": 6.8,
      "average": 6.2,
      "peak": 8.9,
      "unit": "GB"
    },
    "gpu_usage": {
      "current": 67.3,
      "average": 62.1,
      "peak": 89.2,
      "unit": "percent"
    },
    "disk_usage": {
      "current": 156.7,
      "available": 843.3,
      "unit": "GB"
    },
    "network_io": {
      "inbound_mbps": 12.5,
      "outbound_mbps": 8.3,
      "unit": "Mbps"
    }
  },
  "alerts": [
    {
      "metric": "gpu_usage",
      "threshold": 85.0,
      "current_value": 67.3,
      "status": "ok"
    }
  ]
}
```

### Performance Analytics

**GET** `/analytics/performance`

**Response**:
```json
{
  "time_period": {
    "start": "2024-12-24T00:00:00Z",
    "end": "2024-12-25T00:00:00Z"
  },
  "summary": {
    "total_requests": 24580,
    "successful_requests": 24563,
    "error_rate": 0.0007,
    "average_response_time_ms": 47.2,
    "p95_response_time_ms": 89.1,
    "p99_response_time_ms": 156.7
  },
  "performance_trends": {
    "response_time_trend": "stable",
    "error_rate_trend": "decreasing",
    "throughput_trend": "increasing"
  },
  "bottlenecks": [
    {
      "component": "model_inference",
      "impact": "medium",
      "description": "GPU memory allocation causing occasional delays",
      "recommendation": "Consider model quantization or increased GPU memory"
    }
  ]
}
```

---

## Administrative APIs

### User Management

**GET** `/admin/users`

**Response**:
```json
{
  "users": [
    {
      "id": "user123",
      "email": "engineer@example.com",
      "name": "Jane Engineer",
      "roles": ["engineer", "model-operator"],
      "status": "active",
      "last_login": "2024-12-25T14:30:00Z",
      "permissions": [
        "models:read",
        "models:write",
        "inference:execute",
        "safety:monitor"
      ]
    }
  ],
  "total": 45
}
```

### System Configuration

**GET** `/admin/config`

**PUT** `/admin/config`

**Request**:
```json
{
  "inference": {
    "default_timeout_seconds": 30,
    "max_batch_size": 64,
    "gpu_memory_limit_gb": 8
  },
  "safety": {
    "default_constraints": {
      "min_human_distance_m": 1.5,
      "max_velocity_ms": 2.0
    },
    "violation_alert_threshold": 1
  },
  "monitoring": {
    "metrics_retention_days": 30,
    "alert_cooldown_minutes": 5
  }
}
```

### System Health

**GET** `/admin/health`

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-12-25T15:45:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connection_pool": "8/20"
    },
    "model_registry": {
      "status": "healthy",
      "models_loaded": 3,
      "cache_hit_rate": 0.94
    },
    "safety_monitor": {
      "status": "healthy",
      "active_constraints": 5,
      "last_check": "2024-12-25T15:44:55Z"
    },
    "message_queue": {
      "status": "healthy",
      "queue_depth": 23,
      "processing_rate": 150
    }
  },
  "version": "2.1.0",
  "uptime_seconds": 2591856
}
```

---

## Error Handling

### Standard Error Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Safety constraints validation failed",
    "details": {
      "constraint_violations": [
        {
          "constraint": "min_human_distance",
          "required": 1.5,
          "actual": 1.2,
          "severity": "high"
        }
      ]
    },
    "request_id": "req_abc123",
    "timestamp": "2024-12-25T15:30:00Z"
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Request validation failed | 400 |
| `AUTHENTICATION_ERROR` | Authentication failed | 401 |
| `AUTHORIZATION_ERROR` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `SAFETY_VIOLATION` | Safety constraint violated | 422 |
| `MODEL_ERROR` | Model inference error | 422 |
| `INTERNAL_ERROR` | Internal server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |

---

## Rate Limiting

Rate limits are applied per API key and user:

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Authentication | 10 requests | 1 minute |
| Inference | 1000 requests | 1 minute |
| Model Management | 100 requests | 1 minute |
| Safety Monitoring | 500 requests | 1 minute |
| Admin APIs | 50 requests | 1 minute |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 982
X-RateLimit-Reset: 1640448000
```

---

## SDK Examples

### Python SDK

```python
from safe_rl_client import SafeRLClient

# Initialize client
client = SafeRLClient(
    base_url="https://api.safe-rl.yourorg.com/v1",
    api_key="your_api_key"
)

# Authenticate
token = client.auth.login(
    username="user@example.com",
    password="password",
    mfa_token="123456"
)

# Execute inference
result = client.inference.predict(
    model_id="safe_rl_v2.1",
    state={
        "robot_position": [1.0, 2.0, 0.5],
        "human_position": [2.0, 2.5, 0.0],
        "environment_state": {
            "obstacles": [[0.5, 1.0], [3.0, 4.0]],
            "goal": [5.0, 5.0, 0.0]
        }
    },
    safety_constraints={
        "max_velocity": 2.0,
        "min_human_distance": 1.5
    }
)

print(f"Recommended action: {result.action}")
print(f"Safety score: {result.safety_scores['human_safety']}")
```

### JavaScript/Node.js SDK

```javascript
const SafeRLClient = require('safe-rl-client');

const client = new SafeRLClient({
    baseURL: 'https://api.safe-rl.yourorg.com/v1',
    apiKey: 'your_api_key'
});

// Execute inference
async function performInference() {
    try {
        const result = await client.inference.predict({
            model_id: 'safe_rl_v2.1',
            state: {
                robot_position: [1.0, 2.0, 0.5],
                human_position: [2.0, 2.5, 0.0],
                environment_state: {
                    obstacles: [[0.5, 1.0], [3.0, 4.0]],
                    goal: [5.0, 5.0, 0.0]
                }
            },
            safety_constraints: {
                max_velocity: 2.0,
                min_human_distance: 1.5
            }
        });

        console.log('Recommended action:', result.action);
        console.log('Safety score:', result.safety_scores.human_safety);
    } catch (error) {
        console.error('Inference failed:', error.message);
    }
}
```

### cURL Examples

```bash
# Authentication
curl -X POST https://api.safe-rl.yourorg.com/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "user@example.com",
    "password": "password",
    "mfa_token": "123456"
  }'

# Inference
curl -X POST https://api.safe-rl.yourorg.com/v1/inference/policy \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "safe_rl_v2.1",
    "state": {
      "robot_position": [1.0, 2.0, 0.5],
      "human_position": [2.0, 2.5, 0.0]
    },
    "safety_constraints": {
      "max_velocity": 2.0,
      "min_human_distance": 1.5
    }
  }'
```

---

For additional support or questions about the API, please contact the development team or refer to the interactive API documentation available at `/docs` endpoint.