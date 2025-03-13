
# API Design

This document describes the API design for AI-Playground, covering both the REST API and Python API interfaces.

## Design Principles

The AI-Playground APIs are designed with the following principles:

1. **Consistency**: APIs follow consistent patterns and naming conventions

1. **Simplicity**: Common operations are simple and intuitive

1. **Flexibility**: Advanced options are available when needed

1. **Documentation**: All APIs are well-documented with examples

1. **Versioning**: APIs are versioned to ensure backward compatibility

1. **Error handling**: Clear error messages and appropriate status codes

## REST API

The REST API provides HTTP endpoints for interacting with AI-Playground services.

### Base URL

```text

https://<server>:<port>/api/v1

```text

### Authentication

The API supports token-based authentication:

```http

Authorization: Bearer <api_token>

```text

### Endpoints

#### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models` | GET | List available models |
| `/models/{model_id}` | GET | Get model details |
| `/models/upload` | POST | Upload a model |
| `/models/{model_id}` | DELETE | Delete a model |


#### Inference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/inference/{model_id}` | POST | Run inference with a model |
| `/inference/batch/{model_id}` | POST | Run batch inference |
| `/inference/async/{model_id}` | POST | Start async inference job |
| `/inference/jobs/{job_id}` | GET | Get async job status/results |


#### Hardware Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hardware/info` | GET | Get hardware information |
| `/hardware/optimize` | POST | Set hardware optimization |
| `/hardware/profile` | POST | Create hardware profile |
| `/hardware/profiles` | GET | List hardware profiles |


#### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/system/status` | GET | Get system status |
| `/system/metrics` | GET | Get performance metrics |
| `/system/logs` | GET | Get system logs |


### Request Examples

#### Running an Inference

Request:

```http

POST /api/v1/inference/text-generation
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 100,
  "temperature": 0.7,
  "hardware_profile": "arc-optimized"
}

```text

Response:

```http

HTTP/1.1 200 OK
Content-Type: application/json

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

```text

#### Batch Processing

Request:

```http

POST /api/v1/inference/batch/image-classification
Content-Type: application/json

{
  "inputs": [
    {"image_url": "https://example.com/image1.jpg"},
    {"image_url": "https://example.com/image2.jpg"}
  ],
  "hardware_profile": "auto"
}

```text

Response:

```http

HTTP/1.1 200 OK
Content-Type: application/json

{
  "id": "batch-789012",
  "results": [
    {
      "image_id": 0,
      "classifications": [
        {"label": "dog", "confidence": 0.92},
        {"label": "golden retriever", "confidence": 0.85}
      ]
    },
    {
      "image_id": 1,
      "classifications": [
        {"label": "cat", "confidence": 0.97},
        {"label": "tabby", "confidence": 0.82}
      ]
    }
  ],
  "hardware_info": {
    "device": "Intel Arc A770",
    "optimizations": ["batched_execution", "mixed_precision"]
  },
  "performance": {
    "duration_ms": 350,
    "images_per_second": 5.7
  }
}

```text

### Error Handling

Errors are returned as JSON with appropriate HTTP status codes:

```http

HTTP/1.1 400 Bad Request
Content-Type: application/json

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

```text

Common status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid parameters
- `401 Unauthorized`: Authentication failure
- `403 Forbidden`: Permission denied
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

## Python API

The Python API provides a programmatic interface for integrating AI-Playground into Python applications.

### Installation

```bash

pip install ai-playground

```text

### Client Initialization

```python

from ai_playground import AIPlayground

## Initialize with default settings

client = AIPlayground()

## Or with custom settings

client = AIPlayground(
  api_key="your-api-key",
  api_url="<http://localhost:8000/api/v1",
  hardware_profile="arc-optimized"
)

```text

### Model Management

```python

## List available models

models = client.list_models()

## Load a model

model = client.load_model("text-generation-large")

## Get model details

model_info = model.get_info()

## Upload a model

client.upload_model(
  name="my-custom-model",
  path="/path/to/model.onnx",
  model_type="text-generation",
  metadata={"author": "Example User"}
)

```text

### Inference

```python

## Basic inference

result = model.generate(prompt="Tell me a story about a dragon")

## With parameters

result = model.generate(
  prompt="Tell me a story about a dragon",
  max_tokens=200,
  temperature=0.8,
  top_p=0.9
)

## Batch inference

results = model.generate_batch(
  prompts=["Tell me about dragons", "Tell me about unicorns"],
  max_tokens=100
)

## Async inference

job = model.generate_async(prompt="Write a long essay about AI")

## Check status later

if job.is_complete():
  result = job.get_result()

```text

### Hardware Management

```python

## Get hardware information

hardware_info = client.get_hardware_info()

## Set active hardware

client.set_hardware(device_type="arc")

## Create hardware profile

client.create_hardware_profile(
  name="low-memory",
  settings={
    "precision": "int8",
    "batch_size": 1,
    "dynamic_shape": True
  }
)

## Use hardware profile

model.set_hardware_profile("low-memory")

```text

### Error Handling

```python

from ai_playground.exceptions import ModelNotFoundError, HardwareNotSupportedError

try:
  model = client.load_model("nonexistent-model")
except ModelNotFoundError as e:
  print(f"Model not found: {e}")

try:
  client.set_hardware(device_type="unknown")
except HardwareNotSupportedError as e:
  print(f"Hardware not supported: {e}")

## Fall back to CPU

client.set_hardware(device_type="cpu")

```text

## CLI Interface

The command-line interface provides a way to interact with AI-Playground from the terminal.

### Basic Usage

```bash

## Get help

ai-playground --help

## List models

ai-playground models list

## Run inference

ai-playground infer --model text-generation --prompt "Hello, world" --output output.txt

## Get hardware info

ai-playground hardware info

```text

### Advanced Usage

```bash

## Run with specific hardware

ai-playground infer --model image-gen --prompt "A red apple" --hardware arc

## Batch processing

ai-playground infer-batch --model classifier --input-file images.txt --output results.json

## Create hardware profile

ai-playground hardware create-profile --name arc-optimized --precision fp16 --batch-size 4

```text

## WebSocket API

For applications requiring real-time updates, a WebSocket API is available.

### Connection

```javascript

const socket = new WebSocket('wss://your-server/api/v1/ws');

socket.onopen = () => {
  console.log('Connected to AI-Playground');

  // Authentication
  socket.send(JSON.stringify({
    type: 'auth',
    api_key: 'your-api-key'
  }));
};

```text

### Streaming Inference

```javascript

// Request streaming inference
socket.send(JSON.stringify({
  type: 'inference',
  model_id: 'text-generation',
  params: {
    prompt: 'Write a story about',
    max_tokens: 100,
    stream: true
  }
}));

// Handle streaming responses
socket.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'token') {
    console.log('New token:', data.token);
    // Append token to UI
  } else if (data.type === 'completion') {
    console.log('Inference complete');
    // Update UI to show completion
  } else if (data.type === 'error') {
    console.error('Error:', data.message);
  }
};

```text

## API Versioning

AI-Playground APIs are versioned to ensure backward compatibility:

1. **REST API**: Version in URL path (e.g., `/api/v1/models`)

1. **Python API**: Version in package (e.g., `from ai_playground.v1 import AIPlayground`)

When breaking changes are necessary, a new version is released with:

- Comprehensive documentation of changes
- Migration guides
- Deprecation notices in the older version
- Reasonable overlap period for both versions

## Security Considerations

The API implements several security measures:

1. **Authentication**: Token-based auth for all API calls

1. **Authorization**: Role-based access control for sensitive operations

1. **Rate limiting**: Prevents abuse of the API

1. **Input validation**: Thoroughly validates all input data

1. **TLS encryption**: All API traffic is encrypted

## Additional Resources

- [API Reference](../reference/api.md): Complete API documentation
- [Python Client Examples](../examples/python-client.md): Code examples for Python client
- [REST API Examples](../examples/rest-api.md): Examples for REST API usage
- [WebSocket Examples](../examples/websocket.md): Examples for WebSocket API usage

---
**Previous**: [Architecture Overview](overview.md) | **Next**: [Hardware Integration](hardware-integration.md) | __See also_*: [Python API Reference](../reference/python-api.md)


```text

```text`
