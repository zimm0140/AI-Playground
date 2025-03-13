# Hardware Optimization Guide

This guide provides detailed information on optimizing AI-Playground for different hardware configurations to achieve the best performance.

## General Optimization Principles

These optimization principles apply to all hardware configurations:

1. *_Use the right environment__: Let the automatic hardware detection choose the optimal configuration
2. __Update drivers__: Always use the latest drivers for your hardware
3. __Close background applications__: Minimize resource competition
4. __Monitor resource usage__: Use monitoring tools to identify bottlenecks
5. __Batch processing__: Use appropriate batch sizes for your hardware
6. __Mixed precision__: Enable mixed precision where appropriate

## Intel Arc GPUs Optimization

Intel Arc GPUs (Alchemist and newer) offer excellent performance with these optimizations:

### Environment Setup

\`\`\`text\`bash

## Set up environment with Arc optimizations

python setup_hardware_env.py --hardware acm

```text`text

### Performance Tuning

1. __Enable Intel XPU backend__:

   ```python
   ## In your Python code

   import intel_extension_for_pytorch as ipex
   model = model.to("xpu")
   ```text

1. __Use XPU-specific thread count__:

   ```python
   import os
   ## For Arc A770

   os.environ["ZE_AFFINITY_MASK"] = "0.0"
   ```text

1. __Optimize memory usage__:

   ```python
   ## Clear cache between processing

   import torch
   torch.xpu.empty_cache()
   ```text

1. __Enable mixed precision__:

   ```python
   ## Use BF16 for Arc GPUs

   import torch
   with torch.xpu.amp.autocast(dtype=torch.bfloat16):

```text

   ## Your model inference code

```text

```text

   output = model(input)

```text
   ```text

### Arc-Specific Settings

| Setting | Value | Description |
|---------|-------|-------------|
| `ZE_AFFINITY_MASK` | "0.0" | Control which GPU tile is used |
| `SYCL_CACHE_PERSISTENT` | "1" | Enable persistent SYCL cache |
| `IPEX_XPU_MAX_STREAMS` | "8" | Maximum number of streams |
| `DPCT_SYSTEM_MEMORY_GRANULARITY_LEVEL` | "fine" | Memory granularity |


## Intel Meteor Lake Optimization

Intel Meteor Lake CPUs with integrated GPUs benefit from these optimizations:

### Environment Setup

```bash

## Set up environment with Meteor Lake optimizations

python setup_hardware_env.py --hardware mtl

```text

### NPU Acceleration

```python

## Use NPU for compatible operations

os.environ["PYTORCH_MTL_NPU_MODE"] = "1"

```text

### Power Management

```bash

## Set high performance power plan on Windows

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

```text

## NVIDIA GPUs Optimization

For NVIDIA GPUs, consider these optimizations:

### CUDA Optimization

```python

## Set memory allocation strategy

torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available VRAM

## Enable TF32 for better performance (RTX 30/40 series)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

```text

### Multi-GPU Setup

```python

## Use DataParallel for multiple GPUs

model = torch.nn.DataParallel(model)

```text

## CPU-Only Optimization

For systems without GPUs:

```python

## Set thread count to optimize for your CPU

import torch
torch.set_num_threads(8)  # Adjust based on your CPU cores

## Enable MKL optimizations

import os
os.environ["MKL_NUM_THREADS"] = "8"

```text

## Memory Optimization

### Reduce Memory Usage

```python

## Use gradient checkpointing

model.gradient_checkpointing_enable()

## Offload to CPU when appropriate

offload_config = {"offload_buffers": True}

```text

### Optimize for Limited VRAM

For systems with limited GPU memory:

1. Use smaller batch sizes
2. Use 16-bit precision where possible
3. Consider model pruning for inference
4. Use model splitting techniques for large models

## Benchmarking and Performance Measurement

To measure and optimize performance:

```bash

## Run benchmarking tool

python service/tools/benchmark.py --hardware acm --model sd_xl

```text

The tool will report:

- Inference time per image
- Memory usage
- Optimal batch size
- Bottleneck analysis

## Platform-Specific Recommendations

| Platform | Recommended Settings |
|----------|---------------------|
| Intel Arc A770 | XPU backend, BF16 precision, 8 streams |
| Intel Arc A380 | XPU backend, FP32 precision, 4 streams |
| Intel Core Ultra 7 | NPU acceleration, high power mode |
| NVIDIA RTX 3080+ | TF32 precision, CUDA graphs |
| NVIDIA GTX 1660 | FP16 precision, reduced batch size |
| CPU-only | Thread optimization, quantized models |


## Advanced Configuration

For advanced users, edit `/uvfast.json` to fine-tune hardware configurations:

```json
{
  "hardware_optimizations": {

```text

"acm": {
  "thread_count": 8,
  "memory_fraction": 0.8,
  "mixed_precision": true
}

```text
  }
}

```text

---
__Previous__: [Hardware Compatibility](compatibility.md) | __Next__: [Intel Arc Guide](device-specific/intel-arc.md) | __See also_*: [Performance
Troubleshooting](../reference/troubleshooting.md)

```text`

```text`
