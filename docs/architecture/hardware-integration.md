# Hardware Integration Architecture

This document explains how AI-Playground integrates with different hardware platforms to provide optimized performance across various devices.

## Overview

AI-Playground's hardware integration architecture is designed to:

1. **Abstract hardware differences**: Shield users from hardware-specific implementation details
2. **Maximize performance**: Leverage hardware-specific optimizations when available
3. **Provide graceful fallbacks**: Work even when optimal hardware is unavailable
4. **Support seamless transitions**: Allow easy switching between hardware options
5. **Enable extensibility**: Make it easy to add support for new hardware

## Hardware Support Layers

The hardware integration consists of several layers:

```
┌────────────────────────────────────────────────────────────────┐
│                     Application Layer                          │
└────────────────────────────────────────────────────────────────┘
                │                 │
                ▼                 ▼
┌────────────────────────────────────────────────────────────────┐
│                       Device Selector                          │
└────────────────────────────────────────────────────────────────┘
                │                 │
                ▼                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    Hardware Abstraction Layer                  │
└────────────────────────────────────────────────────────────────┘
        │             │              │               │
        ▼             ▼              ▼               ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐ ┌────────────────┐
│ Intel XPU    │ │ Intel NPU│ │ NVIDIA CUDA│ │ CPU Fallback   │
│ Backend      │ │ Backend  │ │ Backend    │ │ Backend        │
└──────────────┘ └──────────┘ └────────────┘ └────────────────┘
        │             │              │               │
        ▼             ▼              ▼               ▼
┌──────────────┐ ┌──────────┐ ┌────────────┐ ┌────────────────┐
│ Intel        │ │ Intel    │ │ NVIDIA     │ │ CPU Only       │
│ Arc GPUs     │ │ NPU      │ │ GPUs       │ │ Systems        │
└──────────────┘ └──────────┘ └────────────┘ └────────────────┘
```

### 1. Application Layer

The top-level interface for models and services. This layer is hardware-agnostic and interacts with the hardware integration through the device selector.

### 2. Device Selector

Chooses the appropriate backend based on:

- Available hardware
- Model requirements
- User preferences
- Performance profiles

### 3. Hardware Abstraction Layer (HAL)

Provides a unified interface to different hardware backends through:

- Common API for all backends
- Hardware-specific optimizers
- Memory management adapters
- Performance monitoring tools

### 4. Hardware-Specific Backends

Implements hardware-specific optimizations:

- Intel XPU Backend (for Intel Arc GPUs)
- Intel NPU Backend (for integrated Neural Processing Units)
- NVIDIA CUDA Backend (for NVIDIA GPUs)
- CPU Fallback Backend (for systems without specialized hardware)

## Hardware Detection Process

The hardware detection process consists of the following steps:

1. **System Probing**: Query the system for available hardware
2. **Capability Assessment**: Determine the capabilities of detected hardware
3. **Driver Validation**: Check for required drivers and their versions
4. **Feature Verification**: Test for specific hardware features
5. **Priority Assignment**: Assign priorities to available hardware options

### Detection Implementation

```python
def detect_hardware():
    """
    Detect available hardware and return prioritized list of device types.
    """
    available_devices = []
    
    # Check for Intel Arc GPUs
    gpu_info = get_gpu_info()
    for gpu in gpu_info:
        if "Intel(R) Arc(TM)" in gpu:
            available_devices.append({
                "type": "arc",
                "name": gpu,
                "priority": 100,
                "backend": "xpu"
            })
        elif "Intel(R) Battlemage(TM)" in gpu:
            available_devices.append({
                "type": "bmg",
                "name": gpu,
                "priority": 100,
                "backend": "xpu"
            })
    
    # Check for NVIDIA GPUs
    for gpu in gpu_info:
        if "NVIDIA" in gpu:
            available_devices.append({
                "type": "nvidia",
                "name": gpu,
                "priority": 90,
                "backend": "cuda"
            })
    
    # Check for NPU
    if has_dptf_driver() and has_npu_capability():
        available_devices.append({
            "type": "npu",
            "name": "Integrated Neural Processing Unit",
            "priority": 80,
            "backend": "npu"
        })
    
    # Always add CPU as fallback
    available_devices.append({
        "type": "cpu",
        "name": "CPU",
        "priority": 10,
        "backend": "cpu"
    })
    
    # Sort by priority (highest first)
    return sorted(available_devices, key=lambda x: x["priority"], reverse=True)
```

## Environment Setup

Once the hardware is detected, the appropriate environment is set up:

### Intel Arc GPU (XPU) Environment

```python
def setup_arc_environment():
    """Set up environment for Intel Arc GPUs."""
    os.environ["SYCL_CACHE_PERSISTENT"] = "1"
    os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
    os.environ["ENABLE_L0_PROGRAM_CREATION_CACHE"] = "1"
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
    
    try:
        import intel_extension_for_pytorch as ipex
        torch.xpu.set_device(0)
        print("Intel Extension for PyTorch and XPU backend enabled")
    except ImportError:
        print("Intel Extension for PyTorch not found, running with limited optimizations")
```

### Intel NPU Environment

```python
def setup_npu_environment():
    """Set up environment for Intel NPU."""
    os.environ["DNNL_DEFAULT_FPMATH_MODE"] = "BF16"
    os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_AMX"
    
    try:
        import neural_compressor
        print("Neural Compressor found, NPU optimizations enabled")
    except ImportError:
        print("Neural Compressor not found, running with limited NPU optimizations")
```

### NVIDIA GPU Environment

```python
def setup_nvidia_environment():
    """Set up environment for NVIDIA GPUs."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"CUDA enabled: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch with CUDA not found")
```

## Model Optimization

The hardware integration includes model optimization for each backend:

### XPU Optimization

```python
def optimize_model_for_xpu(model):
    """Optimize a PyTorch model for Intel XPU (Arc GPU)."""
    import intel_extension_for_pytorch as ipex
    import torch
    
    # Convert to XPU
    model = model.to("xpu")
    
    # Apply IPEX optimizations
    model = ipex.optimize(model, dtype=torch.float16)
    
    # Trace the model if possible
    try:
        example_input = torch.rand(1, 3, 224, 224).to("xpu")
        model = torch.jit.trace(model, example_input)
        model = torch.jit.freeze(model)
    except Exception as e:
        print(f"Model tracing failed: {e}")
    
    return model
```

### NPU Optimization

```python
def optimize_model_for_npu(model):
    """Optimize a model for Intel NPU."""
    from neural_compressor.experimental import Quantization, common
    
    # Initialize quantization
    quantizer = Quantization("npu_config.yaml")
    quantizer.model = model
    
    # Define calibration dataloader
    calibration_data = get_calibration_data()
    quantizer.calib_dataloader = calibration_data
    
    # Quantize the model
    quantized_model = quantizer.fit()
    
    return quantized_model
```

### CUDA Optimization

```python
def optimize_model_for_cuda(model):
    """Optimize a PyTorch model for NVIDIA CUDA."""
    import torch
    
    # Move to CUDA
    model = model.to("cuda")
    
    # Enable CUDA optimization
    if hasattr(model, "half") and torch.cuda.is_available():
        model = model.half()  # Use FP16 if available
    
    # Trace and compile the model if possible
    try:
        example_input = torch.rand(1, 3, 224, 224).to("cuda")
        model = torch.jit.trace(model, example_input)
        model = torch.jit.freeze(model)
    except Exception as e:
        print(f"Model tracing failed: {e}")
    
    return model
```

## Memory Management

Each hardware backend has its own memory management strategy:

### XPU Memory Management

```python
def manage_xpu_memory(batch_size, model_size):
    """Manage memory for XPU execution."""
    import torch
    import intel_extension_for_pytorch as ipex
    
    # Get available memory
    total_mem = torch.xpu.get_device_properties(0).total_memory
    reserved_mem = torch.xpu.memory_reserved(0)
    allocated_mem = torch.xpu.memory_allocated(0)
    free_mem = total_mem - reserved_mem
    
    # Calculate optimal batch size based on available memory
    estimated_batch_memory = model_size * 4  # Rough estimate
    optimal_batch_size = min(batch_size, max(1, free_mem // estimated_batch_memory))
    
    # Set up memory pool
    torch.xpu.empty_cache()
    
    return optimal_batch_size
```

### CUDA Memory Management

```python
def manage_cuda_memory(batch_size, model_size):
    """Manage memory for CUDA execution."""
    import torch
    
    # Get available memory
    total_mem = torch.cuda.get_device_properties(0).total_memory
    reserved_mem = torch.cuda.memory_reserved(0)
    allocated_mem = torch.cuda.memory_allocated(0)
    free_mem = total_mem - reserved_mem
    
    # Calculate optimal batch size based on available memory
    estimated_batch_memory = model_size * 4  # Rough estimate
    optimal_batch_size = min(batch_size, max(1, free_mem // estimated_batch_memory))
    
    # Set up memory pool
    torch.cuda.empty_cache()
    
    return optimal_batch_size
```

## Hardware-Specific Configurations

Each hardware type has specific configurations to optimize performance:

### Intel Arc Configuration

```json
{
  "hardware": "arc",
  "memory": {
    "max_batch_size": "auto",
    "preallocate": true,
    "offload_to_host": true
  },
  "execution": {
    "precision": "mixed",
    "preferred_format": "bf16",
    "optimize_for_inference": true,
    "max_compile_time_seconds": 60
  },
  "optimizations": {
    "enable_concurrent_execution": true,
    "enable_tensor_parallelism": true,
    "enable_kernel_caching": true
  },
  "environment_variables": {
    "SYCL_CACHE_PERSISTENT": "1",
    "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu"
  }
}
```

### NPU Configuration

```json
{
  "hardware": "npu",
  "memory": {
    "max_batch_size": 1,
    "preallocate": false
  },
  "execution": {
    "precision": "bf16",
    "preferred_format": "bf16",
    "optimize_for_inference": true
  },
  "optimizations": {
    "enable_winograd": true,
    "enable_layer_fusion": true
  },
  "environment_variables": {
    "DNNL_DEFAULT_FPMATH_MODE": "BF16",
    "ONEDNN_MAX_CPU_ISA": "AVX512_CORE_AMX"
  }
}
```

## Hardware Abstraction Interface

The Hardware Abstraction Layer provides a unified interface for all backends:

```python
class HardwareBackend:
    """Interface for hardware backends."""
    
    def __init__(self, config=None):
        """Initialize the backend with optional configuration."""
        self.config = config or {}
        self.device_type = "cpu"  # Default device type
    
    def setup(self):
        """Set up the environment for this backend."""
        raise NotImplementedError
    
    def is_available(self):
        """Check if this backend is available on the current system."""
        raise NotImplementedError
    
    def optimize_model(self, model):
        """Optimize a model for this backend."""
        raise NotImplementedError
    
    def run_inference(self, model, inputs, **kwargs):
        """Run inference with the given model and inputs."""
        raise NotImplementedError
    
    def get_memory_info(self):
        """Get memory information for this backend."""
        raise NotImplementedError
    
    def cleanup(self):
        """Clean up resources used by this backend."""
        raise NotImplementedError
```

## Backend Implementation Example

Here's an example of implementing the XPU backend:

```python
class XPUBackend(HardwareBackend):
    """Backend for Intel XPU (Arc GPUs)."""
    
    def __init__(self, config=None):
        """Initialize the XPU backend."""
        super().__init__(config)
        self.device_type = "xpu"
    
    def setup(self):
        """Set up the XPU environment."""
        # Set environment variables
        os.environ["SYCL_CACHE_PERSISTENT"] = "1"
        os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
        
        # Import required libraries
        try:
            import intel_extension_for_pytorch as ipex
            import torch
            self.torch = torch
            self.ipex = ipex
            return True
        except ImportError:
            print("Intel Extension for PyTorch not found")
            return False
    
    def is_available(self):
        """Check if XPU is available."""
        try:
            import intel_extension_for_pytorch as ipex
            import torch
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except ImportError:
            return False
    
    def optimize_model(self, model):
        """Optimize model for XPU."""
        if not self.is_available():
            return model
        
        # Move model to XPU
        model = model.to("xpu")
        
        # Apply IPEX optimizations
        precision = self.config.get("precision", "mixed")
        if precision == "mixed" or precision == "fp16":
            model = self.ipex.optimize(model, dtype=self.torch.float16)
        else:
            model = self.ipex.optimize(model)
        
        return model
    
    def run_inference(self, model, inputs, **kwargs):
        """Run inference on XPU."""
        if isinstance(inputs, dict):
            # Convert input dict values to XPU
            inputs = {k: v.to("xpu") if hasattr(v, "to") else v 
                     for k, v in inputs.items()}
            with self.torch.no_grad():
                outputs = model(**inputs)
        else:
            # Convert inputs to XPU
            if hasattr(inputs, "to"):
                inputs = inputs.to("xpu")
            with self.torch.no_grad():
                outputs = model(inputs)
        
        # Convert outputs back to CPU if needed
        if kwargs.get("return_cpu", True):
            if isinstance(outputs, dict):
                outputs = {k: v.to("cpu") if hasattr(v, "to") else v 
                          for k, v in outputs.items()}
            elif hasattr(outputs, "to"):
                outputs = outputs.to("cpu")
        
        return outputs
    
    def get_memory_info(self):
        """Get XPU memory information."""
        if not self.is_available():
            return {"error": "XPU not available"}
        
        device = self.torch.xpu.current_device()
        total_mem = self.torch.xpu.get_device_properties(device).total_memory
        reserved_mem = self.torch.xpu.memory_reserved(device)
        allocated_mem = self.torch.xpu.memory_allocated(device)
        free_mem = total_mem - reserved_mem
        
        return {
            "total": total_mem,
            "reserved": reserved_mem,
            "allocated": allocated_mem,
            "free": free_mem
        }
    
    def cleanup(self):
        """Clean up XPU resources."""
        if self.is_available():
            self.torch.xpu.empty_cache()
```

## Adding New Hardware Support

To add support for a new hardware platform:

1. **Create a new backend class** inheriting from `HardwareBackend`
2. **Implement required methods** for the new hardware
3. **Add detection logic** to identify the new hardware
4. **Create optimization profiles** for the new hardware
5. **Register the backend** with the hardware abstraction layer

```python
# Example: Adding support for a new hardware type

# 1. Create backend class
class NewHardwareBackend(HardwareBackend):
    """Backend for new hardware type."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.device_type = "new_hardware"
    
    # Implement required methods
    def setup(self):
        """Set up environment for new hardware."""
        # Setup code
        return True
    
    def is_available(self):
        """Check if new hardware is available."""
        # Detection code
        return has_new_hardware()
    
    # ... implement other methods

# 2. Add detection logic
def detect_new_hardware():
    """Detect if new hardware is available."""
    # Detection code
    return True if new_hardware_found() else False

# 3. Register backend
def register_new_hardware():
    """Register new hardware backend."""
    backend_registry.register("new_hardware", NewHardwareBackend)
```

## Performance Monitoring

The hardware integration includes performance monitoring capabilities:

```python
def monitor_hardware_performance(backend, model, inputs, iterations=10):
    """
    Monitor and benchmark hardware performance.
    
    Args:
        backend: Hardware backend to use
        model: Model to benchmark
        inputs: Inputs for the model
        iterations: Number of iterations to run
        
    Returns:
        Performance metrics
    """
    # Warm-up run
    backend.run_inference(model, inputs)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(iterations):
        backend.run_inference(model, inputs)
    end_time = time.time()
    
    # Get memory usage
    memory_info = backend.get_memory_info()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time = total_time / iterations
    throughput = iterations / total_time
    
    return {
        "backend": backend.device_type,
        "avg_inference_time_ms": avg_time * 1000,
        "throughput_per_second": throughput,
        "iterations": iterations,
        "memory": memory_info
    }
```

## Additional Resources

- [Hardware Compatibility Guide](../hardware/compatibility.md)
- [Hardware Optimization Guide](../hardware/optimization.md)
- [Intel Arc GPU Guide](../hardware/device-specific/intel-arc.md)
- [NPU Integration Guide](../hardware/device-specific/intel-npu.md)
- [NVIDIA GPU Guide](../hardware/device-specific/nvidia.md)

---
**Previous**: [API Design](api-design.md) | **Next**: [Data Flow](data-flow.md) | **See also**: [Architecture Overview](overview.md)
