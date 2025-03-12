# Hardware Optimization Guide for AI Applications

This guide explains how to optimize your AI applications for Intel hardware using our hardware-aware environment management system.

## Table of Contents

1. [Introduction](#introduction)
2. [Hardware Detection](#hardware-detection)
3. [Model Optimization](#model-optimization)
4. [Performance Benchmarking](#performance-benchmarking)
5. [Precision Settings](#precision-settings)
6. [Memory Management](#memory-management)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)
9. [FAQ](#faq)

## Overview

Our hardware-aware environment management system automatically detects your Intel hardware and sets up the appropriate environment for optimal performance with AI frameworks. The
system supports:

- *_Intel Arc GPUs__ via XPU backends using Intel® Extension for PyTorch
- **Intel CPUs** with OpenVINO optimizations
- **Standard CPUs** as a fallback option

The system integrates with popular AI frameworks like LangChain and Stable Diffusion to provide seamless acceleration without changing your application code.

## Hardware Types

The system recognizes the following hardware types:

| Type | Description | Optimizations |
|------|-------------|--------------|
| `acm` | Intel Arc GPUs | XPU backends, Intel® Extension for PyTorch |
| `ovino` | Intel CPUs with OpenVINO | OpenVINO runtime optimizations |
| `base` | Standard hardware | Standard PyTorch CPU operations |


## Environment Setup

### Basic Setup

To set up your environment for the detected hardware:

\`\`\`text\`bash

## Set up for automatically detected hardware

python uvfast.py setup

## Set up for specific hardware

python uvfast.py setup --hardware acm
python uvfast.py setup --hardware ovino

## Include development dependencies

python uvfast.py setup --dev

````text

### Using Lockfiles for Reproducible Environments

To ensure reproducible environments, use lockfiles:

```bash

## Generate lockfile for the current hardware

python uvfast.py lock

## Generate lockfile for a specific hardware type

python uvfast.py lock --hardware acm

## Generate lockfiles for all hardware types

python uvfast.py lock --all

## Sync environment from lockfile

python uvfast.py sync

```text

## Using the AI Framework Integration

Our framework integrates with popular AI libraries to provide optimized performance:

```python

## Example of hardware-aware AI framework usage

from examples.ai_frameworks_integration import configure_hardware, setup_langchain_model

## Configure hardware and get device

device, hardware_type = configure_hardware()

## Set up LangChain model with hardware-specific optimizations

llm = setup_langchain_model(device, hardware_type)

## Use the model

response = llm("Explain quantum computing in simple terms.")
print(response)

```text

## Working with LangChain

To optimize LangChain performance on Intel hardware:

### Basic Usage

```python
from examples.ai_frameworks_integration import configure_hardware, setup_langchain_model

## Auto-configure hardware

device, hw_type = configure_hardware()

## Set up LangChain with hardware optimizations

llm = setup_langchain_model(

```text

device=device,
hardware_type=hw_type,
model_id="microsoft/Phi-3-mini-4k-instruct"  # Change to your preferred model

```text

)

## Use the optimized model

response = llm("Explain the theory of relativity in simple terms.")
print(response)

```text

### Advanced Configuration

For complex LangChain applications, you can create hardware-aware chains:

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from examples.ai_frameworks_integration import configure_hardware, setup_langchain_model

## Auto-configure hardware

device, hw_type = configure_hardware()
llm = setup_langchain_model(device, hw_type)

## Create a prompt template

template = """
Answer the following question about {topic}.

Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["topic", "question"])

## Create a chain

chain = LLMChain(llm=llm, prompt=prompt)

## Run the chain

response = chain.run(topic="quantum computing", question="What is quantum entanglement?")

```text

## Working with Stable Diffusion

To optimize Stable Diffusion on Intel hardware:

### Basic Usage

```python
from examples.ai_frameworks_integration import configure_hardware, setup_stable_diffusion

## Auto-configure hardware

device, hw_type = configure_hardware()

## Set up Stable Diffusion with hardware optimizations

pipeline, compel = setup_stable_diffusion(device, hw_type)

## Generate an image

prompt = "a photo of an astronaut riding a horse on mars, highly detailed"
conditioned_prompt = compel(prompt)
image = pipeline(prompt_embeds=conditioned_prompt).images[0]
image.save("astronaut_on_mars.png")

```text

### Optimizing for Speed

For faster inference with reduced quality:

```python

## For Intel Arc GPUs using Intel® Extension for PyTorch

## Lower precision and fewer steps for faster generation

pipeline.set_progress_bar_config(disable=True)
image = pipeline(

```text

prompt="a photo of an astronaut riding a horse on mars",
num_inference_steps=15,  # Reduced from default 50

```text

```text

height=512,  # Smaller size

```text

```text

width=512

```text
).images[0]

```text

### Optimizing for Quality

For higher quality images with longer generation time:

```python

## Higher quality settings

image = pipeline(

```text

prompt="a photo of an astronaut riding a horse on mars, highly detailed",
num_inference_steps=50,
guidance_scale=8.5,
height=768,
width=768

```text
).images[0]

```text

## Performance Benchmarking

To benchmark your hardware and identify optimal settings:

```bash

## Run all benchmarks

python benchmarks/hardware_benchmark.py

## Run specific benchmarks

python benchmarks/hardware_benchmark.py --matrix  # Matrix multiplication only

python benchmarks/hardware_benchmark.py --model   # Model inference only

python benchmarks/hardware_benchmark.py --sd      # Stable Diffusion only

## Specify iterations and output file

python benchmarks/hardware_benchmark.py --iterations 10 --output results.json

```text

### Interpreting Benchmark Results

The benchmark tool measures:

- __Matrix multiplication__: Fundamental operation for linear algebra in ML models
- __Convolution operations__: Key for computer vision models
- __Model inference speed__: LLM inference performance
- __Image generation_*: Stable Diffusion image generation time

Lower times indicate better performance. Compare results across hardware types to determine the best configuration for your workload.

## Troubleshooting

### Common Issues

#### Intel Arc GPU Not Detected

If your Intel Arc GPU is not detected:

1. Ensure you have the latest Intel GPU drivers installed
2. Check that the environment variable `XPU_VISIBLE_DEVICES` is set correctly
3. Verify that Intel® Extension for PyTorch is installed:

   ```bash
   pip install intel-extension-for-pytorch
   ```text

#### OpenVINO Issues

For OpenVINO problems:

1. Ensure OpenVINO is correctly installed
2. Verify your CPU is compatible with OpenVINO
3. Check for environment variable conflicts

#### Performance Issues

If you experience slower than expected performance:

1. Use the benchmarking tool to identify bottlenecks
2. Try different batch sizes and model configurations
3. Update to the latest versions of Intel® Extension for PyTorch and OpenVINO

## Advanced Configuration

### Custom Hardware Detection

You can customize hardware detection by editing `hardware_detection.py`:

```python
def detect_hardware_type():

```text

"""
Custom hardware detection logic
"""

## Your custom logic here

```text

return "acm"  # or "ovino", "base"

```text

```text

### Environment Variables

#### For Intel Arc GPUs

```bash

## Important environment variables for Intel Arc GPUs

export XPU_VISIBLE_DEVICES=0  # Specify which GPU to use

export SYCL_CACHE_PERSISTENT=1  # Improve startup time

export IPEX_XPU_ONEDNN_LAYOUT=1  # Optimize memory layout

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # Improve performance

```text

#### For OpenVINO

```bash

## Important environment variables for OpenVINO

export OPENVINO_THREADING=TBB  # Use TBB threading

export OMP_NUM_THREADS=8  # Control number of OpenMP threads

```text

### Configuring uvfast.json

You can create a `uvfast.json` file in your project root to customize behavior:

```json
{
  "hardware_types": ["base", "acm", "ovino"],
  "default_hardware": "auto",
  "lockfile_settings": {

```text

"auto_sync": true,
"include_dev": true

```text
  },
  "environment_settings": {

```text

"acm": {
  "extra_env_vars": {

```text

"XPU_VISIBLE_DEVICES": "0",
"SYCL_CACHE_PERSISTENT": "1"

```text
  },
  "extra_packages": [

```text

"intel-extension-for-pytorch"

```text
  ]
},
"ovino": {
  "extra_env_vars": {

```text

"OPENVINO_THREADING": "TBB"

```text
  },
  "extra_packages": [

```text

"openvino"

```text
  ]
}

```text
  }
}

```text

This configuration allows for customized settings per hardware type, including environment variables and additional packages.
```text`
```

