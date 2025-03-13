# uvfast: Modern Python Environment Management

This document provides a step-by-step guide for implementing and using the `uvfast` system in your Python projects.

## Implementation Steps

### 1. Core Files

1. *_Create the uvfast.py script__:

   - Copy the `uvfast.py` script to your project root
   - Make it executable: `chmod +x uvfast.py` (on Unix/Linux/macOS)

1. __Create convenience wrappers__:

   - For Unix/Linux/macOS: Create `scripts/uvfast.sh`
   - For Windows: Create `scripts/uvfast.ps1`
   - Make them executable: `git update-index --chmod=+x scripts/uvfast.sh scripts/uvfast.ps1`

1. __Create the configuration file__:

   - Create `uvfast.json` in your project root with your project-specific settings

### 2. Requirements Files

1. __Base requirements__:

   - Ensure you have a `requirements.txt` file with core dependencies

1. __Development requirements__:

   - Create `requirements-dev.txt` with development dependencies
   - Include testing, linting, and type checking packages

1. __Hardware-specific requirements__ (optional):

   - Create separate files for different hardware configurations:

\`\`\`text\`text

- `requirements-hardware-base.txt`
- `requirements-hardware-acm.txt` (for Intel Arc GPUs)
- `requirements-hardware-ovino.txt` (for OpenVINO)
- Add any other hardware-specific configurations

```text`text

### 3. Documentation

1. __Update your README.md__:
   - Add installation instructions
   - Explain the dual approach (traditional pip vs. uvfast)

1. __Add a QUICKSTART.md__:
   - Include basic usage examples
   - List common commands

1. __Add a cheatsheet__:
   - Create `UVFAST_CHEATSHEET.md` with common commands

### 4. CI/CD Integration

1. __GitHub Actions__:
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

1. __Modern tooling__: Leverages `uv` for faster package installation
2. __Reproducible environments__: Uses lockfiles for consistent dependencies
3. __Hardware-specific setups__: Easily manage different hardware configurations
4. __CI/CD integration__: Streamlined testing across platforms
5. __Developer convenience__: Simple commands for common tasks

## Best Practices

1. __Keep configuration up to date__: Update `uvfast.json` when adding new hardware configurations
2. __Generate lockfiles after requirements changes__: Run `uvfast.py lock` after updating any requirements files
3. __Use wrappers for consistency__: Encourage team members to use the wrapper scripts
4. __Include in CI__: Integrate with your CI/CD pipeline for consistent testing
5. __Document hardware-specific needs__: Make sure to document any hardware-specific considerations

## Troubleshooting

1. __Environment issues__:
   - Try recreating the environment: `uvfast.py setup --clean`
   - Check if all requirements files exist

1. __Lockfile conflicts__:
   - Resolve conflicts in requirements files
   - Regenerate lockfiles

1. __Script permissions__:
   - Ensure scripts are executable

1. __Missing dependencies_*:
   - Check hardware-specific requirements
   - Verify lockfiles are up to date

```text`

```text`
