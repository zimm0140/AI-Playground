# uvfast Command Cheat Sheet

This document provides a quick reference for common `uvfast.py` commands.

## Basic Commands

### Setup Environment

```bash

# Auto-detect hardware and set up environment

python uvfast.py setup

# With development dependencies

python uvfast.py setup --dev

# For specific hardware

python uvfast.py setup --hardware acm
python uvfast.py setup --hardware ovino
python uvfast.py setup --hardware mtl

# Skip lockfile generation/usage

python uvfast.py setup --no-lockfile

```text

### Show Environment Information

```bash

# Display hardware detection and environment info

python uvfast.py info

```text

### Generate Lockfiles

```bash

# Generate lockfiles for all hardware types

python uvfast.py lockfiles --all

# Generate for specific hardware

python uvfast.py lockfiles --hardware acm
python uvfast.py lockfiles --hardware acm --dev

```text

### Run Commands

```bash

# Run pytest

python uvfast.py run pytest

# Run with specific test path

python uvfast.py run pytest tests/test_api.py

# Run linting

python uvfast.py run ruff check .
python uvfast.py run ruff format .

# Run type checking

python uvfast.py run mypy

# Run any command

python uvfast.py run python -m your_module

```text

### Legacy Installation

```bash

# Install using traditional approach but with uv speed

python uvfast.py legacy-install --dev

```text

## Using Wrapper Scripts

### Linux/macOS

```bash

# Make script executable

chmod +x scripts/uvfast.sh

# Run commands through the wrapper

./scripts/uvfast.sh setup --dev
./scripts/uvfast.sh run pytest

```text

### Windows

```powershell

# Run commands through the PowerShell wrapper

.\scripts\uvfast.ps1 setup --dev
.\scripts\uvfast.ps1 run pytest

```text

## Hardware-Specific Tips

### Intel Arc GPUs (acm)

```bash

# Set up for Arc GPUs

python uvfast.py setup --hardware acm --dev

# Run GPU-specific tests

python uvfast.py run pytest tests/hardware/test_gpu.py

```text

### OpenVINO (ovino)

```bash

# Set up for OpenVINO

python uvfast.py setup --hardware ovino --dev

# Run OpenVINO-specific tests

python uvfast.py run pytest tests/openvino/

```text

## Configuration

Edit `uvfast.json` to customize settings:

```json
{

```text

"hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino"],
"lockfiles_dir": ".lockfiles",
"venv_dir": ".venv",
"cache_dir": ".uvcache",
"parallel_jobs": 4

```text
}

```
