# API Reference

This document provides a comprehensive reference for the AI-Playground API, including endpoints, parameters, and example usage.

## Table of Contents

- [REST API](#rest-api)
    - [Authentication](#authentication)
    - [Model Management](#model-management)
    - [Inference](#inference)
    - [Hardware Management](#hardware-management)
    - [System](#system)
- [Python API](#python-api)
    - [Client](#client)
    - [Models](#models)
    - [Hardware](#hardware)
    - [Utilities](#utilities)
- [Error Handling](#error-handling)
- [Common Data Structures](#common-data-structures)

## REST API

Base URL: `https://<server>:<port>/api/v1`

### Authentication

#### Obtain API Token

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "username": "user",
  "password": "password"
}
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Refresh Token

```http
POST /api/v1/auth/refresh
Authorization: Bearer <refresh_token>
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Model Management

#### List Available Models

```http
GET /api/v1/models
Authorization: Bearer <token>
```

Response:

```json
{
  "models": [
    {
      "id": "text-generation-large",
      "name": "Text Generation (Large)",
      "type": "text-generation",
      "size": "large",
      "description": "Large text generation model",
      "created_at": "2023-05-10T12:00:00Z",
      "updated_at": "2023-05-10T12:00:00Z",
      "metadata": {
        "parameters": 7000000000,
        "context_length": 4096
      }
    },
    {
      "id": "image-classification",
      "name": "Image Classification",
      "type": "image-classification",
      "size": "medium",
      "description": "General image classification model",
      "created_at": "2023-05-09T10:30:00Z",
      "updated_at": "2023-05-09T10:30:00Z",
      "metadata": {
        "classes": 1000,
        "architecture": "resnet50"
      }
    }
  ]
}
```

#### Get Model Details

```http
GET /api/v1/models/{model_id}
Authorization: Bearer <token>
```

Response:

```json
{
  "id": "text-generation-large",
  "name": "Text Generation (Large)",
  "type": "text-generation",
  "size": "large",
  "description": "Large text generation model",
  "created_at": "2023-05-10T12:00:00Z",
  "updated_at": "2023-05-10T12:00:00Z",
  "metadata": {
    "parameters": 7000000000,
    "context_length": 4096,
    "supported_hardware": ["arc", "nvidia", "cpu"],
    "recommended_hardware": "arc",
    "file_size_mb": 13500
  },
  "performance": {
    "arc": {
      "tokens_per_second": 80,
      "memory_required_mb": 12000
    },
    "nvidia": {
      "tokens_per_second": 65,
      "memory_required_mb": 12000
    },
    "cpu": {
      "tokens_per_second": 5,
      "memory_required_mb": 16000
    }
  }
}
```

#### Upload a Model

```http
POST /api/v1/models/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data

model_file: <file>
metadata: {
  "name": "Custom Text Generation",
  "type": "text-generation",
  "description": "Custom text generation model",
  "metadata": {
    "parameters": 1000000,
    "context_length": 2048
  }
}
```

Response:

```json
{
  "id": "custom-text-generation",
  "name": "Custom Text Generation",
  "type": "text-generation",
  "size": "custom",
  "description": "Custom text generation model",
  "created_at": "2023-06-15T08:45:00Z",
  "updated_at": "2023-06-15T08:45:00Z",
  "metadata": {
    "parameters": 1000000,
    "context_length": 2048
  },
  "status": "uploaded"
}
```

#### Delete a Model

```http
DELETE /api/v1/models/{model_id}
Authorization: Bearer <token>
```

Response:

```json
{
  "status": "success",
  "message": "Model deleted successfully"
}
```

### Inference

#### Run Inference

```http
POST /api/v1/inference/{model_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "hardware_profile": "arc-optimized"
}
```

Response:

```json
{
  "id": "infer-123456",
  "result": "Once upon a time in a distant kingdom, there lived a wise old king...",
  "usage": {
    "prompt_tokens": 4,
    "generated_tokens": 100,
    "total_tokens": 104
  },
  "hardware_info": {
    "device": "Intel Arc A770",
    "optimizations": ["mixed_precision", "graph_optimization"]
  },
  "performance": {
    "duration_ms": 1250,
    "tokens_per_second": 80
  }
}
```

#### Run Batch Inference

```http
POST /api/v1/inference/batch/{model_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "inputs": [
    {"prompt": "Tell me about cats"},
    {"prompt": "Tell me about dogs"}
  ],
  "max_tokens": 50,
  "hardware_profile": "auto"
}
```

Response:

```json
{
  "id": "batch-789012",
  "results": [
    {
      "input_id": 0,
      "result": "Cats are small carnivorous mammals known for their agility...",
      "usage": {
        "prompt_tokens": 4,
        "generated_tokens": 50,
        "total_tokens": 54
      }
    },
    {
      "input_id": 1,
      "result": "Dogs are domesticated mammals known for their loyalty...",
      "usage": {
        "prompt_tokens": 4,
        "generated_tokens": 50,
        "total_tokens": 54
      }
    }
  ],
  "hardware_info": {
    "device": "Intel Arc A770",
    "optimizations": ["batched_execution", "mixed_precision"]
  },
  "performance": {
    "duration_ms": 1500,
    "average_time_per_input_ms": 750
  }
}
```

#### Start Async Inference

```http
POST /api/v1/inference/async/{model_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "prompt": "Write a comprehensive essay about artificial intelligence",
  "max_tokens": 2000,
  "temperature": 0.8,
  "hardware_profile": "high-quality"
}
```

Response:

```json
{
  "job_id": "async-345678",
  "status": "pending",
  "created_at": "2023-06-15T09:10:00Z",
  "estimated_completion_time": "2023-06-15T09:11:30Z",
  "queue_position": 1
}
```

#### Get Async Job Status/Results

```http
GET /api/v1/inference/jobs/{job_id}
Authorization: Bearer <token>
```

Response (pending):

```json
{
  "job_id": "async-345678",
  "status": "running",
  "created_at": "2023-06-15T09:10:00Z",
  "started_at": "2023-06-15T09:10:05Z",
  "progress": 0.45,
  "estimated_completion_time": "2023-06-15T09:11:30Z"
}
```

Response (completed):

```json
{
  "job_id": "async-345678",
  "status": "completed",
  "created_at": "2023-06-15T09:10:00Z",
  "started_at": "2023-06-15T09:10:05Z",
  "completed_at": "2023-06-15T09:11:25Z",
  "result": "Artificial intelligence (AI) refers to the simulation of human intelligence...",
  "usage": {
    "prompt_tokens": 8,
    "generated_tokens": 2000,
    "total_tokens": 2008
  },
  "hardware_info": {
    "device": "Intel Arc A770",
    "optimizations": ["mixed_precision", "graph_optimization"]
  },
  "performance": {
    "duration_ms": 80000,
    "tokens_per_second": 25
  }
}
```

### Hardware Management

#### Get Hardware Information

```http
GET /api/v1/hardware/info
Authorization: Bearer <token>
```

Response:

```json
{
  "detected_hardware": [
    {
      "type": "gpu",
      "name": "Intel(R) Arc(TM) A770 Graphics",
      "backend": "xpu",
      "memory_mb": 16384,
      "driver_version": "31.0.101.4575",
      "is_available": true,
      "is_optimal": true
    },
    {
      "type": "cpu",
      "name": "Intel(R) Core(TM) i9-13900K",
      "backend": "cpu",
      "cores": 24,
      "threads": 32,
      "memory_mb": 32768,
      "is_available": true,
      "is_optimal": false
    }
  ],
  "active_device": {
    "type": "gpu",
    "name": "Intel(R) Arc(TM) A770 Graphics",
    "backend": "xpu"
  },
  "supported_backends": ["xpu", "npu", "cuda", "cpu"],
  "optimal_device_for_workloads": {
    "text-generation": "gpu",
    "image-generation": "gpu",
    "classification": "gpu",
    "embedding": "gpu"
  }
}
```

#### Set Hardware Optimization

```http
POST /api/v1/hardware/optimize
Authorization: Bearer <token>
Content-Type: application/json

{
  "device_type": "arc",
  "settings": {
    "precision": "mixed",
    "batch_size": 4,
    "memory_optimization": "balanced"
  }
}
```

Response:

```json
{
  "status": "success",
  "active_device": {
    "type": "gpu",
    "name": "Intel(R) Arc(TM) A770 Graphics",
    "backend": "xpu"
  },
  "settings": {
    "precision": "mixed",
    "batch_size": 4,
    "memory_optimization": "balanced"
  },
  "estimated_performance_gain": "30%"
}
```

#### Create Hardware Profile

```http
POST /api/v1/hardware/profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "low-memory-arc",
  "description": "Profile optimized for systems with limited memory",
  "device_type": "arc",
  "settings": {
    "precision": "int8",
    "batch_size": 1,
    "memory_optimization": "conservative",
    "dynamic_shape": true,
    "enable_kernel_caching": true
  }
}
```

Response:

```json
{
  "id": "profile-123456",
  "name": "low-memory-arc",
  "description": "Profile optimized for systems with limited memory",
  "device_type": "arc",
  "created_at": "2023-06-15T10:15:00Z",
  "settings": {
    "precision": "int8",
    "batch_size": 1,
    "memory_optimization": "conservative",
    "dynamic_shape": true,
    "enable_kernel_caching": true
  }
}
```

#### List Hardware Profiles

```http
GET /api/v1/hardware/profiles
Authorization: Bearer <token>
```

Response:

```json
{
  "profiles": [
    {
      "id": "profile-123456",
      "name": "low-memory-arc",
      "description": "Profile optimized for systems with limited memory",
      "device_type": "arc",
      "created_at": "2023-06-15T10:15:00Z"
    },
    {
      "id": "arc-optimized",
      "name": "Arc Optimized",
      "description": "Default optimization profile for Intel Arc GPUs",
      "device_type": "arc",
      "created_at": "2023-05-01T00:00:00Z"
    },
    {
      "id": "nvidia-optimized",
      "name": "NVIDIA Optimized",
      "description": "Default optimization profile for NVIDIA GPUs",
      "device_type": "nvidia",
      "created_at": "2023-05-01T00:00:00Z"
    }
  ]
}
```

### System

#### Get System Status

```http
GET /api/v1/system/status
Authorization: Bearer <token>
```

Response:

```json
{
  "status": "healthy",
  "version": "2.3.0",
  "uptime_seconds": 3600,
  "active_jobs": 2,
  "queued_jobs": 1,
  "hardware_status": {
    "gpu": "healthy",
    "cpu": "healthy",
    "memory": "healthy"
  },
  "resource_usage": {
    "cpu_percent": 35,
    "memory_percent": 45,
    "gpu_percent": 70,
    "gpu_memory_percent": 60
  }
}
```

#### Get Performance Metrics

```http
GET /api/v1/system/metrics
Authorization: Bearer <token>
Query parameters:
  start_time: 2023-06-14T00:00:00Z
  end_time: 2023-06-15T00:00:00Z
  interval: 1h
```

Response:

```json
{
  "metrics": {
    "request_count": [
      {"timestamp": "2023-06-14T00:00:00Z", "value": 120},
      {"timestamp": "2023-06-14T01:00:00Z", "value": 150},
      {"timestamp": "2023-06-14T02:00:00Z", "value": 180}
    ],
    "average_response_time_ms": [
      {"timestamp": "2023-06-14T00:00:00Z", "value": 250},
      {"timestamp": "2023-06-14T01:00:00Z", "value": 280},
      {"timestamp": "2023-06-14T02:00:00Z", "value": 260}
    ],
    "gpu_utilization_percent": [
      {"timestamp": "2023-06-14T00:00:00Z", "value": 65},
      {"timestamp": "2023-06-14T01:00:00Z", "value": 78},
      {"timestamp": "2023-06-14T02:00:00Z", "value": 72}
    ]
  },
  "summary": {
    "total_requests": 450,
    "average_response_time_ms": 263,
    "average_gpu_utilization_percent": 71.7
  }
}
```

#### Get System Logs

```http
GET /api/v1/system/logs
Authorization: Bearer <token>
Query parameters:
  level: info
  service: inference
  limit: 10
```

Response:

```json
{
  "logs": [
    {
      "timestamp": "2023-06-15T10:30:45Z",
      "level": "info",
      "service": "inference",
      "message": "Inference request completed successfully",
      "metadata": {
        "model_id": "text-generation-large",
        "job_id": "infer-123456",
        "duration_ms": 1250
      }
    },
    {
      "timestamp": "2023-06-15T10:29:30Z",
      "level": "info",
      "service": "inference",
      "message": "Inference request received",
      "metadata": {
        "model_id": "text-generation-large",
        "job_id": "infer-123456"
      }
    }
  ]
}
```

## Python API

### Client

#### Initialization

```python
from ai_playground import AIPlayground

# Initialize with default settings
client = AIPlayground()

# Or with custom settings
client = AIPlayground(
    api_key="your-api-key",
    api_url="http://localhost:8000/api/v1",
    hardware_profile="arc-optimized"
)
```

#### Authentication

```python
# Authenticate with username and password
client = AIPlayground()
client.authenticate(username="user", password="password")

# Authenticate with API key
client = AIPlayground(api_key="your-api-key")

# Refresh token
client.refresh_token()
```

### Models

#### Listing and Loading Models

```python
# List available models
models = client.list_models()
for model in models:
    print(f"{model.id}: {model.name} ({model.type})")

# Get model details
model_info = client.get_model_info("text-generation-large")
print(f"Parameters: {model_info.metadata['parameters']}")

# Load a model
model = client.load_model("text-generation-large")
```

#### Running Inference

```python
# Basic text generation
result = model.generate(prompt="Tell me a story about a dragon")
print(result)

# With parameters
result = model.generate(
    prompt="Tell me a story about a dragon",
    max_tokens=200,
    temperature=0.8,
    top_p=0.9
)

# Batch inference
results = model.generate_batch(
    prompts=["Tell me about dragons", "Tell me about unicorns"],
    max_tokens=100
)
for i, result in enumerate(results):
    print(f"Result {i}: {result}")

# Async inference
job = model.generate_async(prompt="Write a long essay about AI")
print(f"Job ID: {job.id}, Status: {job.status}")

# Check job status
job.refresh()
if job.is_complete():
    result = job.get_result()
    print(result)
```

#### Upload and Management

```python
# Upload a model
client.upload_model(
    name="my-custom-model",
    path="/path/to/model.onnx",
    model_type="text-generation",
    metadata={"author": "Example User", "parameters": 1000000}
)

# Delete a model
client.delete_model("my-custom-model")
```

### Hardware

#### Getting Hardware Information

```python
# Get hardware information
hardware_info = client.get_hardware_info()
print(f"Active device: {hardware_info['active_device']['name']}")

# List detected hardware
for device in hardware_info["detected_hardware"]:
    print(f"{device['name']} ({device['type']}) - {device['memory_mb']}MB")
```

#### Hardware Configuration

```python
# Set active hardware
client.set_hardware(device_type="arc")

# Set hardware optimization settings
client.optimize_hardware(
    device_type="arc",
    settings={
        "precision": "mixed",
        "batch_size": 4,
        "memory_optimization": "balanced"
    }
)

# Create hardware profile
client.create_hardware_profile(
    name="custom-arc-profile",
    description="My custom Arc GPU profile",
    device_type="arc",
    settings={
        "precision": "int8",
        "batch_size": 1,
        "memory_optimization": "conservative"
    }
)

# Use hardware profile with a model
model.set_hardware_profile("custom-arc-profile")

# List hardware profiles
profiles = client.list_hardware_profiles()
for profile in profiles:
    print(f"{profile['name']} ({profile['device_type']})")
```

### Utilities

#### Measuring Performance

```python
# Benchmark model performance
benchmark = client.benchmark_model(
    model_id="text-generation-large",
    prompt="Once upon a time",
    max_tokens=100,
    iterations=5
)
print(f"Average time: {benchmark['avg_inference_time_ms']}ms")
print(f"Tokens per second: {benchmark['tokens_per_second']}")

# Compare performance across hardware
comparisons = client.compare_hardware_performance(
    model_id="text-generation-large",
    prompt="Once upon a time",
    max_tokens=100,
    hardware_types=["arc", "cuda", "cpu"]
)
for hw_type, perf in comparisons.items():
    print(f"{hw_type}: {perf['tokens_per_second']} tokens/sec")
```

#### Optimizing Models

```python
# Optimize a model for specific hardware
optimized_model = client.optimize_model(
    model_id="text-generation-large",
    target_hardware="arc",
    precision="mixed"
)

# Quantize a model to reduce memory footprint
quantized_model = client.quantize_model(
    model_id="text-generation-large",
    quantization_type="int8"
)
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information in the response body:

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "Parameter 'max_tokens' must be a positive integer",
    "details": {
      "parameter": "max_tokens",
      "value": -10,
      "constraint": "Must be greater than 0"
    }
  }
}
```

In the Python client, exceptions are raised for error conditions:

```python
from ai_playground.exceptions import (
    APIError,
    AuthenticationError,
    ModelNotFoundError,
    HardwareNotSupportedError,
    InferenceError
)

try:
    model = client.load_model("nonexistent-model")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")

try:
    client.set_hardware(device_type="unknown")
except HardwareNotSupportedError as e:
    print(f"Hardware not supported: {e}")
    # Fall back to CPU
    client.set_hardware(device_type="cpu")
```

## Common Data Structures

### Model

```json
{
  "id": "text-generation-large",
  "name": "Text Generation (Large)",
  "type": "text-generation",
  "size": "large",
  "description": "Large text generation model",
  "created_at": "2023-05-10T12:00:00Z",
  "updated_at": "2023-05-10T12:00:00Z",
  "metadata": {
    "parameters": 7000000000,
    "context_length": 4096,
    "supported_hardware": ["arc", "nvidia", "cpu"],
    "recommended_hardware": "arc"
  }
}
```

### Hardware Device

```json
{
  "type": "gpu",
  "name": "Intel(R) Arc(TM) A770 Graphics",
  "backend": "xpu",
  "memory_mb": 16384,
  "driver_version": "31.0.101.4575",
  "is_available": true,
  "is_optimal": true
}
```

### Hardware Profile

```json
{
  "id": "profile-123456",
  "name": "low-memory-arc",
  "description": "Profile optimized for systems with limited memory",
  "device_type": "arc",
  "created_at": "2023-06-15T10:15:00Z",
  "settings": {
    "precision": "int8",
    "batch_size": 1,
    "memory_optimization": "conservative",
    "dynamic_shape": true,
    "enable_kernel_caching": true
  }
}
```

### Inference Result

```json
{
  "id": "infer-123456",
  "result": "Once upon a time in a distant kingdom, there lived a wise old king...",
  "usage": {
    "prompt_tokens": 4,
    "generated_tokens": 100,
    "total_tokens": 104
  },
  "hardware_info": {
    "device": "Intel Arc A770",
    "optimizations": ["mixed_precision", "graph_optimization"]
  },
  "performance": {
    "duration_ms": 1250,
    "tokens_per_second": 80
  }
}
```

---
**Previous**: [Python API Reference](python-api.md) | **Next**: [Configuration Reference](configuration.md) | **See also**: [API Design](../architecture/api-design.md)
