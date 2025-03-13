
# Hardware Optimization Guide {#hardware-optimization-guide}

This guide provides detailed information on optimizing AI-Playground for different hardware configurations to achieve the best performance.

## General Optimization Principles {#general-optimization-principles}

These optimization principles apply to all hardware configurations:

1. *_Use the right environment__: Let the automatic hardware detection choose the optimal configuration

1. **Update drivers**: Always use the latest drivers for your hardware

1. **Close background applications**: Minimize resource competition

1. **Monitor resource usage**: Use monitoring tools to identify bottlenecks

1. **Batch processing**: Use appropriate batch sizes for your hardware

1. **Mixed precision**: Enable mixed precision where appropriate

## Intel Arc GPUs Optimization {#intel-arc-gpus-optimization}

Intel Arc GPUs (Alchemist and newer) offer excellent performance with these optimizations:

### Environment Setup {#environment-setup}

\`\`\`text\`bash

## Set up environment with Arc optimizations {#set-up-environment-with-arc-optimizations}

python setup_hardware_env.py --hardware acm

```text`text

### Performance Tuning {#performance-tuning}

1. **Enable Intel XPU backend**:

   ```python

   ## In your Python code

   import intel_extension_for_pytorch as ipex
   model = model.to("xpu")

   ```

1. **Use XPU-specific thread count**:

   ```python

   import os

   ## For Arc A770

   os.environ["ZE_AFFINITY_MASK"] = "0.0"

   ```

1. **Optimize memory usage**:

   ```python

   ## Clear cache between processing

   import torch
   torch.xpu.empty_cache()

   ```

1. **Enable mixed precision**:

   ```python

   ## Use BF16 for Arc GPUs

   import torch
   with torch.xpu.amp.autocast(dtype=torch.bfloat16):

```text
   ## Your model i

nference code

```text

```text
   output = mode

l(input)

```text

```text

### Arc-Speci

fic Settings {#arc-specific-settings}

| Setting | Value | Description |
|---------|-------|-------------|
| `ZE_AFFINITY_MASK` | "0.0" | Control which GPU tile is used |
| `SYCL_CACHE_PERSISTENT` | "1" | Enable persistent SYCL cache |
| `IPEX_XPU_MAX_STREAMS` | "8" | Maximum number of streams |
| `DPCT_SYSTEM_MEMORY_GRANULARITY_LEVEL` | "fine" | Memory granularity |


## Intel Meteor Lake Optimization {#intel-meteor-lake-optimization}

Intel Meteor Lake CPUs with integrated GPUs benefit from these optimizations:

### Environment Setup {#environment-setup}

```bash

## Set up environ

ment with Meteor Lake optimizations {#set-up-environment-with-meteor-lake-optimizations}

python setup_hardware_env.py --hardware mtl

```text

### NPU Acc

eleration {#npu-acceleration}

```python

## Use NPU for

compatible operations {#use-npu-for-compatible-operations}

os.environ["PYTORCH_MTL_NPU_MODE"] = "1"

```text

### Power

 Management {#power-management}

```bash

## Set high p

erformance power plan on Windows {#set-high-performance-power-plan-on-windows}

powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

```text

## NVID

IA GPUs Optimization {#nvidia-gpus-optimization}

For NVIDIA GPUs, consider these optimizations:

### CUDA Optimization {#cuda-optimization}

```python

## Set memo

ry allocation strategy {#set-memory-allocation-strategy}

torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available VRAM

## Enable TF32 for better performance (RTX 30/40 series) {#enable-tf32-for-better-performance-rtx-3040-series}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

```text

### M

ulti-GPU Setup {#multi-gpu-setup}

```python

## Use Da

taParallel for multiple GPUs {#use-dataparallel-for-multiple-gpus}

model = torch.nn.DataParallel(model)

```text
##
CPU-Only Optimization {#cpu-only-optimization}

For systems without GPUs:

```python

## Set

thread count to optimize for your CPU {#set-thread-count-to-optimize-for-your-cpu}

import torch
torch.set_num_threads(8)  # Adjust based on your CPU cores

## Enable MKL optimizations {#enable-mkl-optimizations}

import os
os.environ["MKL_NUM_THREADS"] = "8"

```text
#

# Memory Optimization {#memory-optimization}

### Reduce Memory Usage {#reduce-memory-usage}

```python

## Us

e gradient checkpointing {#use-gradient-checkpointing}

model.gradient_checkpointing_enable()

## Offload to CPU when appropriate {#offload-to-cpu-when-appropriate}

offload_config = {"offload_buffers": True}

```text

### Optimize for Limited VRAM {#optimize-for-limited-vram}

For systems with limited GPU memory:

1. Use smaller batch sizes

1. Use 16-bit precision where possible

1. Consider model pruning for inference

1. Use model splitting techniques for large models

## Benchmarking and Performance Measurement {#benchmarking-and-performance-measurement}

To measure and optimize performance:

```bash

##
Run benchmarking tool {#run-benchmarking-tool}

python service/tools/benchmark.py --hardware acm --model sd_xl

``
`
The tool will report:

- Inference time per image

- Memory usage

- Optimal batch size

- Bottleneck analysis

## Platform-Specific Recommendations {#platform-specific-recommendations}

| Platform | Recommended Settings |
|----------|---------------------|
| Intel Arc A770 | XPU backend, BF16 precision, 8 streams |
| Intel Arc A380 | XPU backend, FP32 precision, 4 streams |
| Intel Core Ultra 7 | NPU acceleration, high power mode |
| NVIDIA RTX 3080+ | TF32 precision, CUDA graphs |
| NVIDIA GTX 1660 | FP16 precision, reduced batch size |
| CPU-only | Thread optimization, quantized models |


## Advanced Configuration {#advanced-configuration}

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
**Previous**: [Hardware Compatibility](compatibility.md) | **Next**: [Intel Arc Guide](device-specific/intel-arc.md) | __See also_*: [Performance
Troubleshooting](../reference/troubleshooting.md)

```text`

```text`
