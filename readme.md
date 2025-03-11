# AI Playground

A modern AI development environment with support for multiple models and backends.

## Installation

### Requirements

- Python 3.10 or newer (3.13 recommended for development)
- uv (recommended) or pip for package management

### Primary Method (Recommended)

```bash

# Install uv (once)

# On macOS and Linux

curl -LsSf <https://astral.sh/uv/install.sh> | sh

# On Windows

powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"


# Clone the repository

git clone <https://github.com/zimm0140/AI-Playground.git>
cd AI-Playground

# Create virtual environment and install dependencies

uv venv
uv pip install -e .

# For development dependencies

uv pip install -e ".[dev]"

```text

### Alternative Method (Modern)

```bash

# Install Rye (once)

curl -sSf <https://rye-up.com/get> | bash

# Clone the repository

git clone <https://github.com/zimm0140/AI-Playground.git>
cd AI-Playground

# Setup development environment with Rye

rye sync

```text

### Traditional Method (Compatible with Upstream)

```bash

# Clone the repository

git clone <https://github.com/zimm0140/AI-Playground.git>
cd AI-Playground

# Install dependencies

pip install -e .

# For development dependencies

pip install -e ".[dev]"

```text

## Python 3.10+ Migration

As of version 2.2.0, we have migrated to Python 3.10+ to leverage modern language features and better type checking. For developers, we maintain a hybrid approach with:

- Support for both traditional (pip) and modern (uv/Rye) installation methods
- Comprehensive CI testing across multiple Python versions
- Updated type annotations for compatibility

See [MIGRATION.md](MIGRATION.md) for detailed guidance on the migration process, including:
- Type annotation changes
- Dependency management
- Common issues and solutions

## Development

### Running Tests

```bash

# Using uv (recommended)

uv run pytest

# Using Rye

rye run pytest

# Traditional method

pytest

```text

### Linting and Type Checking

```bash

# Install pre-commit hooks

pre-commit install

# Run manually

pre-commit run --all-files

# Using uv for on-demand checks

uv run ruff check .
uv run mypy .

```text

## Supported Features

- **Text Generation**: Leverage state-of-the-art LLMs
- **Image Generation**: Create and manipulate images with Stable Diffusion
- **Multi-backend Support**: Support for Llama.cpp, OpenVINO, and more
- **GPU Acceleration**: Optimized for Intel GPUs through XPU hijacks

## Project Structure

- `/service` - Core API services
- `/LlamaCPP` - Llama.cpp integration
- `/OpenVINO` - OpenVINO integration
- `/.github` - CI/CD workflows

## CI/CD Pipeline

Our project uses a hybrid approach for CI/CD:
- Python 3.10+ compatibility (3.13 recommended for development)
- Dual testing with both traditional pip and modern Rye
- Comprehensive linting and type checking

## Contributing

1. Fork the repository
1. Create your feature branch (`git checkout -b feature/amazing-feature`)
1. Commit your changes (`git commit -m 'Add some amazing feature'`)
1. Push to the branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

Before submitting, please ensure:
- Tests pass with both installation methods
- Pre-commit hooks run without errors
- Documentation is updated as needed

<a href="<https://scan.coverity.com/projects/ai-playground>">
  <img alt="Coverity Scan Build Status"

```text
   src="<https://scan.coverity.com/projects/30694/badge.svg">/>

```text
</a>

![image](https://github.com/user-attachments/assets/ee1efc30-4dd1-4934-9233-53fba00c71bd)

This example is based on the xpu implementation of Intel® Arc™ GPU.

Welcome to AI Playground open source project and AI PC starter app for doing AI image creation, image stylizing, and chatbot on a PC powered by an Intel® Arc™ GPU. AI Playground
leverages libraries from GitHub and Huggingface which may not be available in all countries world-wide.  AI Playground supports many Gen AI libraries and models including:

- Image Diffusion: Stable Diffusion 1.5, SDXL, Flux.1-Schnell, LTX-Video
- LLM: Safetensor PyTorch LLMs - DeepSeek R1 models, Phi3, Qwen2, Mistral, GGUF LLMs -  Llama 3.1, Llama 3.2: OpenVINO - TinyLlama, Mistral 7B, Phi3 mini, Phi3.5 mini

## README.md

- English (readme.md)

## Min Specs

AI Playground alpha and beta installers are currently available downloadable executables, or available as a source code from our Github repository.  To run AI Playground you must
have a PC that meets the following specifications

- Windows OS
- Intel Core Ultra-H Processor, Intel Core Ultra 200V series processor OR Intel Arc GPU Series A or Series B (discrete) with 8GB of vRAM

## Installation - Packaged Installer

Starting from v2.0, there is a single packaged installer that works for all supported hardware mentioned above. This installer simplifies the process for end users to install AI
Playground and get it running on their PC. Please note that while this makes the installation process easier, this is open-source beta software, and there may be component and
version conflicts. Refer to the Troubleshooting section for known issues.

## # Download the installer

:new: **AI Playground 2.2.1 Beta (all SKUs)** - [Release Notes](https://github.com/intel/AI-Playground/releases/tag/v2.2.1-beta) | [Download](https://github.
com/intel/AI-Playground/releases/download/v2.2.1-beta/AI.Playground-2.2.1-beta.exe) :new:
> [!IMPORTANT]
> This release fixes video generation and image generation bugs from 2.2

## # Installation Process for v2.0

1. The installer only installs the Electron frontend, so it completes very quickly.
1. On the first run, you need to install additional backend components for AI Playground to function properly. This process requires a strong and open network and may **take
several minutes**.
1. Download the Users Guide for application information: [AI Playground Users Guide](https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf)

## # Troubleshooting Installation

The following are known situations where your installation may be blocked or interrupted.  Review the following to remedy installations issues.  If installation issues persist,
generate a copy of the log by typing CTRL+SHIFT+I, select the console tab and copy the last few entries of the log written where the installer failed.  Provide these details to us
via the issues tab here, or via the Intel Insiders Discord, or Graphics forum on Intel's support site.

1. **Restart**: Time-out issues have been sighted, which show as a failed install but resolve when restarting AI Playground
1. **Verify Intel Arc GPU**: Ensure your system has an Intel Arc GPU. Go to your Windows Start Menu, type "Device Manager," and under Display Adapters, check the name of your GPU
device. It should describe an Intel Arc GPU. If so, then you you have a GPU that means our minimum specifications.  If it says "Intel(R) Graphics," your system does not have a
built-in Intel Arc GPU and does not meet the minimum specifications.
1. **Interrupted Installation**: The online installation for backend components can be interrupted or blocked by an IT network, firewall, or sleep settings. Ensure you are on an
open network, with the firewall off, and set sleep settings to stay awake when powered on.
1. **Missing Libraries**: Some Windows systems may be missing needed libraries. This can be fixed by installing the 64-bit VC++ redistribution from Microsoft [here](https://learn.
microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170). It is recommended this be done after updating the Graphics drivers. Then install AI Playground.
1. **Python Conflict**: Some PCs with an existing installation of Python can cause a conflict with AI Playground installation, where the wrong or conflicting packages are
installed due to the incorrect version or location of Python on the system.  This is usually remedied by uninstalling Python environment, restarting and reinstalling AI Playground
1. **Temp Files**: Should the installation be interrupted because of any of the above issues it is possible that temporary installation files have been left behind and trying to
install with these files in place can block the installation. Remove these files or do a clean install of AI Playground to remedy

## Project Development

## # Checkout Source Code

To get started, clone the repository and navigate to the project directory:

```cmd
git clone -b dev <https://github.com/intel/AI-Playground.git>
cd AI-Playground

```text

## # Install Node.js Dependencies

1. Install the Node.js development environment from [Node.js](https://nodejs.org/en/download).

1. Navigate to the `WebUI` directory and install all Node.js dependencies:

```cmd
cd WebUI
npm install

```text

## # Prepare Python Environment

1. Install Miniforge to manage your Conda environment: <https://github.com/conda-forge/miniforge>

1. Create a Conda environment with Python 3.11 and libuv:

```text
conda create -n cp311_libuv python=3.11 libuv -y

```text
1. Locate the path to your newly created Conda environment:

```text
conda env list | findstr cp311_libuv


```text
1. In the `WebUI` directory, execute the `fetch-build-resources` script, replacing `<path_to_cp311_libuv_conda_env>` with the actual path you copied in the previous step:

```text
npm run fetch-build-resources -- --conda_env_dir=<path_to_cp311_libuv_conda_env>

```text
1. Run the `prepare-build` script:

```text
npm run prepare-build

```text
You should now have a basic Python environment located at `build-envs\online\prototype-python-env`.

## # Launch the application

To start the application in development mode, run:

```text
npm run dev

```text

## # (Optional) Build the installer

To build the installer, run:

```text
npm run build

```text
The installer executable will be located in the `release` folder.

## CI Features

## # ComfyUI Workflow Validation

AI Playground includes automated validation for ComfyUI workflows in the CI pipeline. This ensures that all workflow JSON files in the `WebUI/external/workflows` directory are
properly structured and executable.

The validation process includes:

- Structural validation of workflow JSON files
- Analysis of model, custom node, and hardware requirements
- Simulation of workflow execution without requiring actual models

For more details, see the [ComfyUI Workflow Validation documentation](docs/comfyui_workflow_validation.md).

## Developer Environment Setup

## # Virtual Environment Setup

For development work, we recommend using a virtual environment to isolate dependencies. AI Playground supports both Conda and venv-based workflows:

## # # Option 1: Using Conda (Recommended)

```cmd

## Create a conda environment

conda env create -f environment.yml

## Activate the environment

conda activate ai-playground-env

## Install requirements

pip install -r requirements.txt

```text

## # # Option 2: Using venv

```cmd

## Create a virtual environment

python -m venv .venv

## Activate on Windows

.\.venv\Scripts\activate

## Activate on macOS/Linux

source .venv/bin/activate

## Install requirements

pip install -r requirements.txt

```text

## # Quick Setup

We provide a setup script that checks your environment and installs dependencies:

```cmd

## On Windows

scripts\setup_env.bat

## On macOS/Linux

python scripts/setup_env.py

```text
See [CONTRIBUTING.md](CONTRIBUTING.md) for more detailed information about development workflows.

## Model Support

AI Playground supports PyTorch LLM, SD1.5, and SDXL models. AI Playground does not ship with any models but does make  models available for all features either directly from the
interface or indirectly by the users downloading models from HuggingFace.co or CivitAI.com and placing them in the appropriate model folder.

Models currently linked from the application

| Model                                      | License                                                                                                                                                                      | Background Information/Model Card                                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Dreamshaper 8 Model                        | [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)                                             | [site](https://huggingface.co/Lykon/dreamshaper-8)                               |
| Dreamshaper 8 Inpainting Model             | [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)                                             | [site](https://huggingface.co/Lykon/dreamshaper-8-inpainting)         |
| JuggernautXL v9 Model                      | [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)                                             | [site](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)           |
| Phi3-mini-4k-instruct                      | [license](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE)                 | [site](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)     |
| bge-large-en-v1.5                          | [license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)                 | [site](https://huggingface.co/BAAI/bge-large-en-v1.5)                         |
| Latent Consistency Model (LCM) LoRA: SD1.5 | [license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | [site](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) |
| Latent Consistency Model (LCM) LoRA:SDXL   | [license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | [site](https://huggingface.co/latent-consistency/lcm-lora-sdxl)     |


Be sure to check license terms for any model used in AI Playground especially taking note of any restrictions.

## # Use Alternative Models

Check the [User Guide](https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf) for details or [watch this video](https://www.youtube.
com/watch?v=1FXrk9Xcx2g) on how to add alternative Stable Diffusion models to AI Playground

## # Notices and Disclaimers

For information on AI Playground terms, license and disclaimers, visit the project and files on GitHub repo:</br >
[License](https://github.com/intel/ai-playground/blob/main/LICENSE) | [Notices & Disclaimers](https://github.com/intel/ai-playground/blob/main/notices-disclaimers.md)

The software may include third party components with separate legal notices or governed by other agreements, as may be described in the Third Party Notices file accompanying the
software.

# Using Makefile (optional)

We provide a Makefile for common development tasks:

```bash

# Set up development environment

make setup

# Run tests

make test

# Run linters

make lint

# Format code

make format

# Sync dependencies

make sync

# Show all available commands

make help

```text

## Advanced uv Features

### Using Lockfiles

```bash

# Create a lockfile

uv pip compile requirements.txt -o requirements.lock

# Install from lockfile

uv pip sync requirements.lock

```text

### Dependency Auditing

```bash

# Check for vulnerabilities

uv pip audit

```text

### Creating Isolated Environments for Scripts

```bash

# Run a Python script in an isolated environment

uv run script.py

```text

### Installing Command-line Tools

```bash

# Install a tool globally

uv tool install ruff

# Run a tool without installing

uv tool run ruff check .

```
