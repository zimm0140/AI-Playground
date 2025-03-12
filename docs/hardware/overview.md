# Hardware Overview

AI-Playground is designed to work optimally across various hardware configurations, with special optimizations for Intel hardware platforms. This guide provides an overview of
supported hardware and how the system adapts to different configurations.

## Supported Hardware Platforms

AI-Playground supports the following hardware platforms with varying levels of optimization:

| Platform | Code | Support Level | Description |
|----------|------|--------------|-------------|
| Base (CPU) | `base` | Full | Standard CPU support for all platforms |
| Intel Arc GPUs | `acm` | Optimized | Intel Arc GPUs with specialized acceleration |
| Intel Battlemage GPUs | `bmg` | Optimized | Intel Battlemage GPUs (next-gen) |
| Intel Meteor Lake | `mtl` | Optimized | Intel Meteor Lake CPUs with integrated GPU/NPU |
| Intel Lunar Lake | `lnl` | Optimized | Intel Lunar Lake CPUs |
| OpenVINO | `ovino` | Optimized | Systems with OpenVINO runtime |
| Arc Limited | `arl_h` | Optimized | Systems with limited Arc features |
| NVIDIA GPUs | - | Standard | Standard PyTorch CUDA support |
| AMD GPUs | - | Basic | Limited support via ROCm |


## Automatic Hardware Detection

AI-Playground includes a sophisticated hardware detection system that identifies your specific hardware configuration and sets up the environment accordingly:

\`\`\`text\`bash

## Automatic detection and setup

python setup_hardware_env.py

````text

The detection system:

1. Identifies CPU architecture and features
2. Detects available GPUs and their capabilities
3. Checks for specialized hardware like Intel NPUs
4. Verifies the presence of optimization libraries like OpenVINO
5. Selects the most appropriate configuration based on findings

## Hardware-specific Optimizations

### Intel Arc GPUs (`acm`)

Intel Arc GPUs receive special optimizations:

- Intel Extension for PyTorch (IPEX) integration
- XPU-specific kernels and operations
- Optimized memory management
- Hardware-aware scheduling

### Intel Meteor Lake (`mtl`)

Meteor Lake systems benefit from:

- NPU acceleration for certain operations
- Integrated GPU optimizations
- CPU efficiency core utilization
- Power-aware workload distribution

### OpenVINO Integration (`ovino`)

Systems with OpenVINO benefit from:

- Model compilation for faster inference
- Quantization optimizations
- Multi-device execution
- Hardware abstraction for portability

## Manual Hardware Configuration

If you want to override the automatic detection, you can specify the hardware type:

```bash

## Force a specific hardware configuration

python setup_hardware_env.py --hardware acm  # For Intel Arc GPUs

```text

## Hardware-specific Dependencies

Each hardware configuration has specific dependencies:

- `requirements-hardware-base.txt`: Base requirements for all platforms
- `requirements-hardware-acm.txt`: Intel Arc GPU specific packages
- `requirements-hardware-ovino.txt`: OpenVINO specific packages

These are automatically installed based on your detected hardware.

## Performance Considerations

Different hardware platforms have different performance characteristics:

- *_Intel Arc GPUs__: Best for parallel operations and high-throughput processing
- __Intel Meteor Lake__: Good balance of CPU and GPU capabilities, with NPU for AI acceleration
- __NVIDIA GPUs__: Strong general-purpose GPU computing
- __CPU-only__: Works everywhere but with reduced performance for compute-intensive tasks

## Hardware Compatibility

For detailed compatibility information including recommended drivers and configurations, see the [Hardware Compatibility](compatibility.md) guide.

For optimization techniques specific to your hardware, see the [Hardware Optimization](optimization.md) guide.

---
__Previous__: [Migration Guide](../getting-started/migration.md) | __Next__: [Hardware Compatibility](compatibility.md) | __See also_*: [Device-Specific Guides](device-specific/)
```text`
````

