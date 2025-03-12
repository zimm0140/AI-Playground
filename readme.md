# AI Playground

<a href="https://scan.coverity.com/projects/ai-playground">
  <img alt="Coverity Scan Build Status" src="https://scan.coverity.com/projects/30694/badge.svg"/>
</a>
<img alt="Version" src="https://img.shields.io/badge/version-2.2.1--beta-blue"/>
<img alt="Platform" src="https://img.shields.io/badge/platform-Windows-lightgrey"/>

![image](https://github.com/user-attachments/assets/ee1efc30-4dd1-4934-9233-53fba00c71bd)

## Overview

AI Playground is an open-source project and AI PC starter app optimized for Intel® Arc™ GPUs. Create AI-generated images, stylize existing photos, and interact with powerful
chatbots—all locally on your PC.

This project showcases the capabilities of Intel® Arc™ GPUs for AI workloads through XPU implementation, enabling high-performance generative AI experiences on consumer hardware.

## Project Structure

The project is organized into the following directories:

- **docs/**: Documentation files
  - **development/**: Development guides and reports
  - **hardware/**: Hardware-specific documentation
  - **user-guides/**: End-user documentation
  - **workflows/**: Workflow documentation

- **tools/**: Utility scripts and tools
  - **linting/**: Code and documentation linting tools
  - **formatting/**: Code and documentation formatting tools
  - **hardware/**: Hardware detection and setup tools
  - **scripts/**: General utility scripts

- **config/**: Configuration files
  - Environment configuration
  - Linting and formatting configuration
  - Application configuration

- **docker/**: Docker configuration
  - Dockerfile and docker-compose.yml

- **legal/**: Legal documentation
  - License files
  - Notices and disclaimers

- **service/**: Backend services
- **WebUI/**: Frontend web interface
- **langchain_community/**: LangChain integration
- **OpenVINO/**: OpenVINO integration
- **LlamaCPP/**: Llama.cpp integration
- **acceleration/**: Acceleration libraries
- **hardware_aware/**: Hardware-aware features
- **examples/**: Example code and usage
- **benchmarks/**: Performance benchmarks
- **tests/**: Test suite

### ✨ Key Features

- 🖼️ **Generate stunning images** using state-of-the-art AI models
- 💬 **Chat with AI** using local large language models
- 🎬 **Create AI videos** with advanced animation techniques
- ⚡ **Hardware-optimized** for Intel® Arc™ GPUs and Intel Core Ultra processors
- 🛠️ **Extensible architecture** for adding new models and capabilities

AI Playground supports a wide range of generative AI capabilities:

- **Image Generation**: Stable Diffusion 1.5, SDXL, Flux.1-Schnell, LTX-Video
- **Language Models**:

```

- Safetensor PyTorch LLMs: DeepSeek R1, Phi3, Qwen2, Mistral
- GGUF LLMs: Llama 3.1, Llama 3.2
- OpenVINO: TinyLlama, Mistral 7B, Phi3 mini, Phi3.5 mini

```

### 🚀 What's New in v2.2.1

- **Fixed Video Generation**: Resolved issues affecting video creation pipeline
- **Enhanced Image Generation**: Improved performance and stability
- **UI Improvements**: Better user experience and workflow
- **Expanded Hardware Support**: Optimized for latest Intel hardware

For complete release details, see the [v2.2.1 Release Notes](https://github.com/intel/AI-Playground/releases/tag/v2.2.1-beta).

## Documentation

For comprehensive documentation, see the [AI-Playground Documentation](docs/index.md) covering:

- [Getting Started Guide](docs/getting-started/quickstart.md)
- [Hardware Optimization](docs/hardware/optimization.md)
- [Developer Documentation](docs/development/contributing.md)
- [API Reference](docs/reference/api.md)

## README.md

- English (readme.md)

## Min Specs

AI Playground requires the following hardware and software:

- **Operating System**: Windows OS
- **Processor/GPU**: One of the following:

```

- Intel Core Ultra-H Processor
- Intel Core Ultra 200V series processor
- Intel Arc GPU Series A or Series B (discrete) with 8GB of vRAM

```

For detailed hardware compatibility information, see our [Hardware Compatibility Guide](docs/hardware/compatibility.md).

## Installation - Packaged Installer

Starting from v2.0, there is a single packaged installer that works for all supported hardware mentioned above. This installer simplifies the process for end users to install AI
Playground and get it running on their PC. Please note that while this makes the installation process easier, this is open-source beta software, and there may be component and
version conflicts. Refer to the Troubleshooting section for known issues.

### Quick Installation Steps

1. **Download** the latest release below
2. **Run** the installer (completes quickly, installs Electron frontend)
3. **Launch** the application and follow prompts to install backend components
4. **Enjoy** creating with AI!

### Download the installer

:new: **AI Playground 2.2.1 Beta (all SKUs)** - [Release Notes](https://github.com/intel/AI-Playground/releases/tag/v2.2.1-beta) |
[Download](https://github.com/intel/AI-Playground/releases/download/v2.2.1-beta/AI.Playground-2.2.1-beta.exe) :new:
> [!IMPORTANT]
> This release fixes video generation and image generation bugs from 2.2

### Installation Process for v2.0

1. The installer only installs the Electron frontend, so it completes very quickly.
2. On the first run, you need to install additional backend components for AI Playground to function properly. This process requires a strong and open network and may **take
several minutes**.
3. Download the Users Guide for application information: [AI Playground Users Guide](https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf)

For more detailed installation instructions, see our [Installation Guide](docs/getting-started/installation.md).

### Troubleshooting Installation

If your installation is blocked or interrupted, review the following troubleshooting steps. If issues persist, generate a log by pressing `CTRL+SHIFT+I`, selecting the console
tab, and copying the relevant error messages. Share these details through:

- GitHub Issues on this repository
- Intel Insiders Discord
- Graphics forum on Intel's support site

#### Common Installation Issues

1. **Restart**: Time-out issues may appear as failed installations but often resolve after restarting AI Playground.

2. **Verify Intel Arc GPU**: Ensure your system has the required GPU:
   - Open Device Manager (Start Menu → type "Device Manager")
   - Under Display Adapters, verify you have an Intel Arc GPU listed
   - If you only see "Intel(R) Graphics," your system does not meet the minimum specifications

3. **Network Issues**: Backend component installation requires:
   - Open network connection
   - Disabled firewall (temporarily)
   - System settings that prevent sleep during installation

4. **Missing Libraries**: Some Windows systems require additional libraries:
   - Install the [64-bit VC++ redistribution from Microsoft](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
   - We recommend doing this after updating Graphics drivers

5. **Python Conflicts**: Existing Python installations may cause conflicts:
   - Uninstall existing Python environments
   - Restart your system
   - Reinstall AI Playground

6. **Temporary Files**: Interrupted installations may leave behind files that block reinstallation:
   - Remove temporary installation files
   - Perform a clean installation

For additional troubleshooting resources, see our [Common Problems Guide](docs/troubleshooting/common-problems.md).

## Project Development

### Checkout Source Code

To get started, clone the repository and navigate to the project directory:

```

git clone -b dev https://github.com/intel/AI-Playground.git
cd AI-Playground

```

### Install Node.js Dependencies

1. Install the Node.js development environment from [Node.js](https://nodejs.org/en/download).

2. Navigate to the `WebUI` directory and install all Node.js dependencies:

```

cd WebUI
npm install

```

### Prepare Python Environment

1. Install Miniforge to manage your Conda environment: https://github.com/conda-forge/miniforge

2. Create a Conda environment with Python 3.11 and libuv:

   ```

   conda create -n cp311_libuv python=3.11 libuv -y
   ```

3. Locate the path to your newly created Conda environment:

   ```

   conda env list | findstr cp311_libuv
   ```

4. In the `WebUI` directory, execute the `fetch-build-resources` script, replacing `<path_to_cp311_libuv_conda_env>` with the actual path you copied in the previous step:

   ```

   npm run fetch-build-resources -- --conda_env_dir=<path_to_cp311_libuv_conda_env>
   ```

5. Run the `prepare-build` script:

   ```

   npm run prepare-build
   ```

You should now have a basic Python environment located at `build-envs\online\prototype-python-env`.

### Launch the application

To start the application in development mode, run:

```

npm run dev

```

### (Optional) Build the installer

To build the installer, run:

```

npm run build

```

The installer executable will be located in the `release` folder.

For more detailed development information, see our [Contributing Guide](docs/development/contributing.md) and [Architecture Overview](docs/architecture/overview.md).

## Model Support

AI Playground supports PyTorch LLM, SD1.5, and SDXL models. While AI Playground does not ship with any models, it provides convenient access to models either:

- Directly through the application interface
- By downloading models from HuggingFace.co or CivitAI.com and placing them in the appropriate model folder

### Integrated Models

The following models are linked directly from the application:

| Model | License | Background Information |
|-------|---------|------------------------|
| Dreamshaper 8 Model | [License](https://huggingface.co/spaces/CompVis/stable-diffusion-license) | [Model Card](https://huggingface.co/Lykon/dreamshaper-8) |
| Dreamshaper 8 Inpainting Model | [License](https://huggingface.co/spaces/CompVis/stable-diffusion-license) | [Model Card](https://huggingface.co/Lykon/dreamshaper-8-inpainting) |
| JuggernautXL v9 Model | [License](https://huggingface.co/spaces/CompVis/stable-diffusion-license) | [Model Card](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9) |
| Phi3-mini-4k-instruct | [License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE) | [Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) |
| bge-large-en-v1.5 | [License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md) | [Model Card](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Latent Consistency Model (LCM) LoRA: SD1.5 | [License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | [Model Card](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) |
| Latent Consistency Model (LCM) LoRA:SDXL | [License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | [Model Card](https://huggingface.co/latent-consistency/lcm-lora-sdxl) |


⚠️ **Important**: Always check license terms for any model used in AI Playground, particularly noting any restrictions on usage.

### Using Alternative Models

- Refer to the [User Guide](https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf) for detailed instructions
- Watch our [video tutorial](https://www.youtube.com/watch?v=1FXrk9Xcx2g) for adding alternative Stable Diffusion models

For hardware-specific model optimization, see our [Hardware Optimization Guide](docs/hardware/optimization.md).

## Community and Support

### Getting Help

If you need assistance with AI Playground, there are several ways to get help:

- **[GitHub Issues](https://github.com/intel/AI-Playground/issues)**: Report bugs or request features
- **[Intel Insiders Discord](https://discord.gg/intel-insiders)**: Discuss with community members
- **[Intel Developer Forum](https://community.intel.com/t5/Accelerated-Computing/bd-p/accelerated-computing)**: Ask questions about development

### Contributing

We welcome contributions to the AI Playground project! See our [Contributing Guide](docs/development/contributing.md) for details on:

- Reporting bugs
- Suggesting enhancements
- Submitting pull requests
- Development workflow

## Legal Information

### Notices and Disclaimers

For information on AI Playground terms, license, and disclaimers, visit:

- [License](https://github.com/intel/ai-playground/blob/main/LICENSE)
- [Notices & Disclaimers](https://github.com/intel/ai-playground/blob/main/notices-disclaimers.md)

The software may include third-party components with separate legal notices or governed by other agreements, as described in the Third Party Notices file accompanying the software.
