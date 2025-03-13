
# Intel Arc GPUs Guide {#intel-arc-gpus-guide}

This guide provides detailed information for users running AI-Playground on Intel Arc GPUs.

## Supported Arc GPU Models {#supported-arc-gpu-models}

AI-Playground supports all Intel Arc GPU models:

| Model | Memory | Recommended for | Performance Level |
|-------|--------|-----------------|------------------|
| A770 | 16GB | Large model inference, batch processing | Excellent |
| A750 | 8GB | Medium to large models | Very Good |
| A580 | 8GB | Medium models | Good |
| A380 | 6GB | Small to medium models | Good |
| A310 | 4GB | Small models only | Basic |

## Hardware Requirements {#hardware-requirements}

### Driver Requirements {#driver-requirements}

| OS | Minimum Driver Version | Recommended Driver |
|----|------------------------|-------------------|
| Windows | 31.0.101.4255 | Latest available |
| Linux | Mesa 23.1 | Mesa 23.3+ |

### System Requirements {#system-requirements}

- PCIe 4.0 x8 or x16 slot

- 350W+ power supply (450W+ recommended for A770)

- External power connectors (for models A580 and above)

- 16GB+ system RAM

## Installation and Setup {#installation-and-setup}

### Driver Installation {#driver-installation}

#### Windows {#windows}

1. Download the latest driver from [Intel's download center](https://downloadcenter.intel.com/product/226793/Intel-Arc-A-series-Graphics)

1. Install the driver package

1. Restart your system

1. Verify installation with:

   ```bash

   # Run hardware detection

   python hardware_detection.py

   ```

#### Linux {#linux}

1. Update your system:

   ```bash

   sudo apt update && sudo apt upgrade

   ```

1. Install required packages:

   ```bash

   sudo apt install mesa-utils

   ```

1. Verify installation:

   ```bash

   glxinfo | grep "OpenGL renderer"

   ```

### Environment Setup {#environment-setup}

```bash

#
 Setup environment optimized for Arc GPUs {#setup-environment-optimized-for-arc-gpus}

python setup_hardware_env.py --hardware acm

```
This will

install the required dependencies including:

- Intel Extension for PyTorch (IPEX)

- Intel Neural Compressor

- oneDNN optimizations

- XPU backend libraries

## Optimizing for Arc GPUs {#optimizing-for-arc-gpus}

### XPU-specific Code {#xpu-specific-code}

Use the "xpu" device in your code:

```pytho
n

import torch
import intel_extension_for_pytorch as ipex

# Move model to XPU {#move-model-to-xpu}

model = model.to("xpu")

# Move input tensors to XPU {#move-input-tensors-to-xpu}

input_tensor = input_tensor.to("xpu")

# Run inference {#run-inference}

with torch.xpu.amp.autocast(dtype=torch.bfloat16):
    output = model(input_tensor)

```

### Env

i {#envi}

ronment Variables {#environment-variables}

Set these environment variables for optimal performance:

```bas
h

# Windows (PowerShell) {#windows-powershell}

$env:ZE_AFFINITY_MASK = "0.0"
$env:SYCL_CACHE_PERSISTENT = "1"
$env:IPEX_XPU_MAX_STREAMS = "8"

# Linux (Bash) {#linux-bash}

export ZE_AFFINITY_MASK="0.0"
export SYCL_CACHE_PERSISTENT="1"
export IPEX_XPU_MAX_STREAMS="8"

```

### M

e {#me}

mory Management {#memory-management}

Arc GPUs benefit from careful memory management:

```p
y
thon

# Clear XPU cache when needed {#clear-xpu-cache-when-needed}

torch.xpu.empty_cache()

# Monitor memory usage {#monitor-memory-usage}

print(f"Memory allocated: {torch.xpu.memory_allocated() / 1e9:.2f} GB")
print(f"Memory reserved: {torch.xpu.memory_reserved() / 1e9:.2f} GB")

```

##

T {#t}

roubleshooting Arc-Specific Issues {#troubleshooting-arc-specific-issues}

### Common Issues and Solutions {#common-issues-and-solutions}

| Issue | Solution |
|-------|----------|
| "No XPU devices found" | Update drivers and ensure IPEX is installed correctly |
| Out of memory errors | Reduce batch size or use mixed precision |
| Performance lower than expected | Check power limits and thermal throttling |
| System crash during inference | Update drivers and reduce workload size |

### Debugging Tools {#debugging-tools}

``
`
bash

# Check GPU information {#check-gpu-information}

python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.xpu.get_device_properties(0))"

# Run diagnostic tool {#run-diagnostic-tool}

python service/tools/intel_gpu_diagnostics.py

```

#

#
 Performance Tuning {#performance-tuning}

### Model Optimization {#model-optimization}

1. **Quantization**:

   ```python

   from intel_extension_for_pytorch.quantization import prepare, convert

   # Prepare model for quantization

   qconfig = ipex.quantization.default_static_qconfig
   prepared_model = prepare(model, qconfig, example_inputs=example_inputs)

   # Convert to quantized model

   quantized_model = convert(prepared_model)

   ```

1. **BF16 Mixed Precision**:

   ```python

   with torch.xpu.amp.autocast(dtype=torch.bfloat16):
       output = model(input_tensor)

   ```

### Batch Size Optimization {#batch-size-optimization}

Test different batch sizes to find the optimal value for your specific Arc GPU model:

`
``python

# Example batch size benchmark {#example-batch-size-benchmark}

batch_sizes = [1, 2, 4, 8, 16]
results = {}

for bs in batch_sizes:

    # Test inference speed with batch size bs

    # Record timing information

```
Typical optimal batch sizes:

- A770: 8-16

- A750: 4-8

- A380: 2-4

## Comparing with Other GPUs {#comparing-with-other-gpus}

| Task | Arc A770 | RTX 3070 | Notes |
|------|----------|----------|-------|
| SD XL Inference | ~5.2 it/s | ~6.8 it/s | Arc more power efficient |
| LoRA Training | ~0.9 it/s | ~1.2 it/s | Similar memory usage |
| LLM Inference | ~22 tok/s | ~28 tok/s | Arc benefits from BF16 |

## Additional Resources {#additional-resources}

- [Intel Developer Documentation](https://developer.intel.com/arctgpu)

- [Intel Extension for PyTorch Documentation](https://intel.github.io/intel-extension-for-pytorch/)

- [XPU Migration Guide](https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/docs/tutorials/xpu_migration_guide.md)

---
**Previous**: [Hardware Optimization](../optimization.md) | **See also**: [Meteor Lake Guide](intel-meteor-lake.md)

```markdown

```