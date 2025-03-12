# uvfast: Modern Python Environment Management

This document provides a step-by-step guide for implementing and using the `uvfast` system in your Python projects.

## Implementation Steps

### 1. Core Files

1. **Create the uvfast.py script**:

   - Copy the `uvfast.py` script to your project root
   - Make it executable: `chmod +x uvfast.py` (on Unix/Linux/macOS)

1. **Create convenience wrappers**:

   - For Unix/Linux/macOS: Create `scripts/uvfast.sh`
   - For Windows: Create `scripts/uvfast.ps1`
   - Make them executable: `git update-index --chmod=+x scripts/uvfast.sh scripts/uvfast.ps1`

1. **Create the configuration file**:

   - Create `uvfast.json` in your project root with your project-specific settings

### 2. Requirements Files

1. **Base requirements**:

   - Ensure you have a `requirements.txt` file with core dependencies

1. **Development requirements**:

   - Create `requirements-dev.txt` with development dependencies
   - Include testing, linting, and type checking packages

1. **Hardware-specific requirements** (optional):

   - Create separate files for different hardware configurations:

````text

 - `requirements-hardware-base.txt`
 - `requirements-hardware-acm.txt` (for Intel Arc GPUs)
 - `requirements-hardware-ovino.txt` (for OpenVINO)
 - Add any other hardware-specific configurations

```text

### 3. Documentation

1. **Update your README.md**:
   - Add installation instructions
   - Explain the dual approach (traditional pip vs. uvfast)

1. **Add a QUICKSTART.md**:
   - Include basic usage examples
   - List common commands

1. **Add a cheatsheet**:
   - Create `UVFAST_CHEATSHEET.md` with common commands

### 4. CI/CD Integration

1. **GitHub Actions**:
   - Create or update `.github/workflows/ci.yml`
   - Use the uvfast script to set up environments
   - Run tests and linting

## Usage Guide

### Basic Commands

```bash

## Setup environment with development dependencies

python uvfast.py setup --dev

## Show environment information

python uvfast.py info

## Generate lockfiles for all hardware types

python uvfast.py lock

## Run tests

python uvfast.py run pytest

## Run linting

python uvfast.py run ruff check .

## Run type checking

python uvfast.py run mypy .

```text

### Using Wrapper Scripts

```bash

## Unix/Linux/macOS

./scripts/uvfast.sh setup --dev

## Windows PowerShell

.\scripts\uvfast.ps1 setup --dev

```text

### Hardware-Specific Setup

```bash

## Setup for Intel Arc GPUs

python uvfast.py setup --hardware acm --dev

## Setup for OpenVINO

python uvfast.py setup --hardware ovino --dev

```text

## Implementation Example

### uvfast.json

```json
{
  "project_name": "my-project",
  "python_version": "3.10",
  "hardware_types": ["base", "acm", "ovino"],
  "default_hardware": "base",
  "requirements": {

```text

"base": "requirements.txt",
"dev": "requirements-dev.txt",
"hardware": {
  "base": "requirements-hardware-base.txt",
  "acm": "requirements-hardware-acm.txt",
  "ovino": "requirements-hardware-ovino.txt"
}

```text
  },
  "lockfiles": {

```text

"base": "requirements.lock",
"hardware": {
  "base": "requirements-hardware-base.lock",
  "acm": "requirements-hardware-acm.lock",
  "ovino": "requirements-hardware-ovino.lock"
}

```text
  }
}

```text

## Benefits of Using uvfast

1. **Modern tooling**: Leverages `uv` for faster package installation
1. **Reproducible environments**: Uses lockfiles for consistent dependencies
1. **Hardware-specific setups**: Easily manage different hardware configurations
1. **CI/CD integration**: Streamlined testing across platforms
1. **Developer convenience**: Simple commands for common tasks

## Best Practices

1. **Keep configuration up to date**: Update `uvfast.json` when adding new hardware configurations
1. **Generate lockfiles after requirements changes**: Run `uvfast.py lock` after updating any requirements files
1. **Use wrappers for consistency**: Encourage team members to use the wrapper scripts
1. **Include in CI**: Integrate with your CI/CD pipeline for consistent testing
1. **Document hardware-specific needs**: Make sure to document any hardware-specific considerations

## Troubleshooting

1. **Environment issues**:
   - Try recreating the environment: `uvfast.py setup --clean`
   - Check if all requirements files exist

1. **Lockfile conflicts**:
   - Resolve conflicts in requirements files
   - Regenerate lockfiles

1. **Script permissions**:
   - Ensure scripts are executable

1. **Missing dependencies**:
   - Check hardware-specific requirements
   - Verify lockfiles are up to date

````
