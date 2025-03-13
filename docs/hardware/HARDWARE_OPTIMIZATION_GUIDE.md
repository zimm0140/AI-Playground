# Hardware Optimization Guide for AI Applications {#hardware-optimization-guide-for-ai-applications}

This guide explains how to optimize your AI applications for Intel hardware using our hardware-aware environment management system.


## Table of Contents {#table-of-content {#table-of-contents-table-of-content}

s}


1. [Overview](#overview)


1. [Hardware Types](#hardware-types)


1. [Environment Setup](#environment-setup

)


1. [Using the AI Framework Integration](#using-the-ai-framework-integratio

n)


1. [Working with LangChain](#working-with-langcha

in)


1. [Working with Stable Diffusion](#working-with-stable-diffus

ion)


1. [Performance Benchmarking](#performance-benchmar

king)


1. [Troubleshooting](#troublesho

oting)


1. [Advanced Configuration](#advanced-configu

ration)


1. [Overview](#o

verview)


1. [Hardware Types](#hardwa

re-types)


1. [Environment Setup](#environm

ent-setup)


1. [Using the AI Framework Integration](#using-the-ai-framework-i

ntegration)


1. [Working with LangChain](#working-wit

h-langchain)


1. [Working with Stable Diffusion](#working-with-stab

le-diffusion)


1. [Performance Benchmarking](#performance

-benchmarking)


1. [Troubleshooting](#t

roubleshooting)


1. [Advanced Configuration](#advanced-configuratio

n)


## Overview {#overv {#overview-overv}

iew}

Our hardware-aware environment management system automatically detects your Intel hardware and sets up the appropriate environment for optimal performance with AI frameworks. The
system supports:


- **Intel Arc GPUs** via XPU backends using Intel® Extension for PyTorch


- **Intel CPUs** with OpenVINO optimizations


- **Standard CPUs** as a fallback option

The system integrates with popular AI frameworks like LangChain and Stable Diffusion to provide seamless acceleration without changing your application c
ode.


## Hardware Types {#hardware- {#hardware-types-hardware-}

types}

The system recognizes the following hardware types:

| Type | Description | Optimizations |
|------|-------------|---------------|
| `acm` | Intel Arc GPUs | XPU backends, Intel® Extension for PyTorch |
| `ovino` | Intel CPUs with OpenVINO | OpenVINO runtime optimizations |
| `base` | Standard hardware | Standard PyTorch CPU operations |


## Environment Setup {#environmen {#environment-setup-environmen}

t-setup}


### Basic Setup {#ba {#basic-setup-ba}

sic-setup}
To set up your environment for the detected hardware
:

```bash


# Set up for 
a

utomatically detected hardware {#set-up-for-automatically-detect {#set-up-for-automatically-detected-hardware-set-up-for-automatically-detect}

ed-hardware}
python uvf
ast.py setup


# Set up for specific hardware {#set-up-for-spec {#set-up-for-specific-hardware-set-up-for-spec}

ific-hardware}
python uvfast.py setup --hardware acm
python uvfast.py setup --
hardware ovino


# Include development dependencies {#include-developme {#include-development-dependencies-include-developme}

nt-dependencies}
python uvfast.py
setup --dev

```text

### Using Lockf {#usin
g

-lockf}

iles for Reproducible Environments {#using-lockfiles-for-reproduc
ible-environments}
To ensure reproducible environments, use l
ockfiles:

```bash


# Generate 
l

ockfile for the current hardware {#generate-lockfile-for-t {#generate-lockfile-for-the-current-hardware-generate-lockfile-for-t}

he-current-hardware}
p
ython uvfast.py lock


# Generate lockfile for a specific hardware type {#generate-lockfile-for-a-s {#generate-lockfile-for-a-specific-hardware-type-generate-lockfile-for-a-s}

pecific-hardware-type}
python uvfast.
py lock --hardware acm


# Generate lockfiles for all hardware types {#generate-lockfiles {#generate-lockfiles-for-all-hardware-types-generate-lockfiles}

-for-all-hardware-types}
pyt
hon uvfast.py lock --all


# Sync environment from lockfile {#sync- {#sync-environment-from-lockfile-sync-}

environment-from-lockfile}

python uvfast.py syn```text``


## Using the A {#using-the-a}

I Framework Integration {#using-t
he-ai-framework-integration}

Our framework integrates with popular AI libraries to provide optim
ized performance:

```python


# Example of hardwar
e

-aware AI framework usage {#example-of-hard {#example-of-hardware-aware-ai-framework-usage-example-of-hard}

ware-aware-ai-framework-usage}
from examples.ai_frameworks_integration import configure_h
ardware, setup_langchain_model


# Configure hardware and get device {#co {#configure-hardware-and-get-device-co}

nfigure-hardware-and-get-device}
device, hard
ware_type = configure_hardware()


# Set up LangChain model with hardware-specific optimizations {#set-up-langchain-model-wit {#set-up-langchain-model-with-hardware-specific-optimizations-set-up-langchain-model-wit}

h-hardware-specific-optimizations}
llm = setup_lang
chain_model(device, hardware_typ
e)


# Use {#use} the model {#use-the-model}

response = llm("Explain quantum computing in
simple terms.")
print(respo```
```text

## Work {#
work

}ing wi {#working-wi}

th LangChain {#working-with-langchain}

To optimize La
ngChain performance on Intel h
ardware:


### Basic Usage {#basic-usage}

```python

from examples.ai_f
r
ameworks_integration import c
onfigure_hardware, setup_langchain_model


# Auto-con {#auto-con}

figure hardware {#auto-configure-hardwa
re}
device, hw_type = configure_hardware()


# Set up LangChain with hardware optimizations {#s {#set-up-langchain-with-hardware-optimizations-s}

et-up-langchain-with-hardware-optimizations}
llm = setup_langchain_model(
    device=device,
    hardware_type=hw_type,
    model_id="microsoft/Phi-3-mini-4k-in
struct"  # Change to your preferred model

)


# Use {#use}

the optimized model {#use-the-optimized-model}
response = llm("Explain the theory of rela
tivity in simple terms.")
print(re```textse)

```text

### Adva
nc {

#advanc}

ed Configuration {#advanced-configuration}

For complex LangChain applications, you can create hardware-aware chains:

```python

from langchain.p
r
ompts import PromptTemplate
from langchain.chains import LLMChain
from examples.ai_frameworks_integration
import configure_hardware, setup_langchain_model


#
Auto-configure hardware {#auto-configure-hardware}

device, hw_type = configure_hardw
are()
llm = setup_langchain_model(device, hw_type)


#
Create a prompt template {#create-a-prompt-template}

template = """
Answer the following question about {topic}.

Question: {question}
"""
prompt = PromptTemplate(templ
ate=template, input_variables=["to
pic", "question"])


# Create a chain {#cr {#create-a-chain-cr}

eate-a-chain}
chain = LLMChain(l
lm=llm, prompt=prompt)


# Run the chain {#run-the-chain}

response = chain.run(topic="quantum co
mputing", question="What is quantum entan```textent?")

```text

## Wor
k {#

work}

in
g with Stable Diffusion {#working-with-stable-d
iffusion}

To optimize Stable
Diffusion on Intel hardware:


### Basic Usage {#basic-usage}

```python

from examples.
a
i_frameworks_i
ntegration import configure_hardware, setup_stable_d
iffusion


# Auto-con {#auto-con}figure hardware {#a

uto-configure-hardware}
device, hw_type = configure_hardware()


# Set up Stable Diffusion with hardware opti {#set-up-stable-diffusion-with-hardware-opti}

mizations {#set-up-stable-diffusion-with-hardware-optimizat
ions}
pipeline, compel = setup_stable_di
ffusion(device, hw_type)


# Generate an image {#generate-an-image}

prompt = "a photo of an astronaut riding a horse on mars, highly detailed"
conditioned_prompt = compel(prompt)
image = pipeline(prompt_embeds=cond
itioned_prompt).images[0]
image.save("astronaut_
```textars.png")

```text

### 
Opt
{#opt}i {#opti}

mizing for Speed {#optimiz
ing-for-speed}
For faster inference with reduced quality:

```python


# For Intel 
A

rc GPUs using Intel® Extensi {#for-intel-arc-gpus-using-intel-extensi}

on for PyTorch {#for-intel-arc-gpus-using-intel-extension-for-pytorch}


# Lower precision and fewer steps for fa {#lower-precision-and-fewer-steps-for-fa}

ster generation {#lower-precision-and-fewer-steps-for-faster-generation}
pipeline.set_progress_bar_config(disable=True)
image = pipeline(
    prompt="a photo of an astronaut riding a horse on mars",
    num_inference_steps=15,  # Reduced from def
ault 50

    height=512,  # Smaller size

    width=```
).images[0]

```text

##
# Op

t {#opt}

imizing for Quality {#optimizing-for-
quality}
For higher quality images with longer gener
ation time:

```python


# Higher q
u

ality settings {#higher-quality-settings}

image = pipeline(
    prompt="a photo of an astronaut riding a horse on mars, highly detailed",
    num_inference_step
s=50,
    guidance_scale=8.5,
    height=768,
    wid```text768
).images[0]

```text


## P

er {#per}

formance Benchmarking {#performance-ben
chmarking}

To benchmark your hardware and
 identify optimal settings:

```bash


#
Run al
l

 benchmarks {#run-all-benchmarks}

python benc
hmarks/hardware_benchmark.py


# Run specific benchmarks {#run-specific-benchmarks}

python benchmarks/hardware_benchmark.py --matrix  # Matrix multiplication only

python benchmarks/hardware_benchmark.py --model   # Model inferen
ce only

python benchmarks/hardware_benchmark.py --sd      # Stable Diffus
ion only


# Specify iterations and output file {#specify-iterations-and-output-file {#specify-iterations-and-output-file-specify-iterations-and-output-file}

}
python benchmarks/hardware_benchmark.py --iterations 10 ```texttput r
esults.json

```text


##
# I {#i}

nterpreting Benchmark Results {#interpreting-benchmark-results}
The benchmark tool measures:


- **Matrix multiplication**: Fundamental operation for linear algebra in ML models


- **Convolution operations**: Key for computer vision models


- **Model inference speed**: LLM inference performance


- **Image generation**: Stable Diffusion image generation time

Lower times indicate better performance. Compar
e results across hardware types to de

termine the best configuration for

 your workload.


## Troubleshooting {#troubleshooting}


###
Common Issues {#common-issues}


#### I {#i}ntel Arc GPU Not Detected {#intel-arc-gpu-not-detected}

If your Intel Arc GPU
is not detected:


1. Ensure you have the latest Intel GPU

drivers installed


1. Check that the environment variable `XPU_VISIBLE_DEVICES

` is set correctly


1. Verify that Intel® Extensi

on for PyTorch is installed:

   ```bas
h

   pip install intel-extension-for-pytorch

   ```


#### OpenVINO Issues {#openvino-issues}

For
 OpenVINO problems:


1. Ensure OpenVINO is

correctly installe
d


1. Verify your CPU is comp

atible with Open
VINO


1. Check for environment variable conflicts


#### Per {#per}formance Issues {#performance-issues}

If you experience slower than
 expected performance:


1. Use the benchmarking tool

to identify bottlenecks


1. Try different batch sizes a

nd mod
el configurations


1. Update to the latest versions

 of Intel® Extension for PyTorch and OpenVINO


## Advanced {#advanced}

 Configuration {#advanced-configuration}


### Custom Hardware Detection {#custom-hardware-detection}

You can customize hardware detection by editing `hardware_detection.py`:

```python

def de
t
ect_hardware_type():
    """
    Custom hardwar
e detection logic
    """

    # Your custom logic

 here

    return "ac```text# or "ovino", "base
"

```te
xt

###
Environment Variables {#environment-variables}


#### For Intel Arc GPUs {#for-intel-arc-gpus}

```bash


# Im
p

ortant environment variables for Intel Arc GPUs {#important-environment-variables-for-intel-arc-gpus}

export XPU_VISIBLE_DEVICES=0  # Specify which GPU to use

export SYCL_CACHE_PERSISTENT=1  # Improve startup time

export IPEX_XPU_ONEDNN_LAYOUT=1  # O
ptimize memory layout

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1  # Improve performance

```

text

## ## For OpenVINO {#for-openvino}

```bash

# Important environment variables for OpenVINO

export OPENVINO_THREADING=TBB  # Use TBB threading

export OMP_NUM_THREADS=8  # Control number of OpenMP threads

```


### Configuring uvfast.json {#configuring-uvfastjson}

You can create a `uvfast.json` file in your project root to customize behavior:

`
``json

{

  "hardware_types": ["base", "acm", "ovino"],
  "default_hardware": "auto",
  "lockfile_settings": {
    "auto_sync": true,
    "include_dev": true
  },
  "environment_settings": {
    "acm": {
      "extra_env_vars": {
        "XPU_VISIBLE_DEVICES": "0",
        "SYCL_CACHE_PERSISTENT": "1"
      },
      "extra_packages": [
        "intel-extension-for-pytorch"
      ]
    },
    "ovino": {
      "extra_env_vars": {
        "OPENVINO_THREADING": "TBB"
      },
      "extra_packages": [
     ```textpenvino"
      ]
    }
  }
}

```text

This configuration allows for customized settings per hardware type, including environment```te```textbles and additional packages.


```text
```text
