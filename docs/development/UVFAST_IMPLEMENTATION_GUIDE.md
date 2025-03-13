
# uvfast: Modern Python Environment Management

This document provides a step-by-step guide for implementing and using the `uvfast` system in your Python projects.

## Implementation Steps

### 1. Core Files

1. *_Create the uvfast.py script__:

   - Copy the `uvfast.py` script to your project root
   - Make it executable: `chmod +x uvfast.py` (on Unix/Linux/macOS)

2. **Create convenience wrappers**:

   - For Unix/Linux/macOS: Create `scripts/uvfast.sh`
   - For Windows: Create `scripts/uvfast.ps1`
   - Make them executable: `git update-index --chmod=+x scripts/uvfast.sh scripts/uvfast.ps1`

2. **Create the configuration file**:

   - Create `uvfast.json` in your project root with your project-specific settings

### 2. Requirements Files

2. **Base requirements**:

   - Ensure you have a `requirements.txt` file with core dependencies

2. **Development requirements**:

   - Create `requirements-dev.txt` with development dependencies
   - Include testing, linting, and type checking packages

2. **Hardware-specific requirements** (optional):

   - Create separate files for different hardware configurations:

\`\`\`text\`text

- `requirements-hardware-base.txt`
- `requirements-hardware-acm.txt` (for Intel Arc GPUs)
- `requirements-hardware-ovino.txt` (for OpenVINO)
- Add any other hardware-specific configurations

```text`text

### 3. Documentation

2. **Update your README.md**:

   - Add installation instructions
   - Explain the dual approach (traditional pip vs. uvfast)

2. **Add a QUICKSTART.md**:

   - Include basic usage examples
   - List common commands

2. **Add a cheatsheet**:

   - Create `UVFAST_CHEATSHEET.md` with common commands

### 4. CI/CD Integration

2. **GitHub Actions**:

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

2. **Modern tooling**: Leverages `uv` for faster package installation


2. **Reproducible environments**: Uses lockfiles for consistent dependencies


2. **Hardware-specific setups**: Easily manage different hardware configurations


2. **CI/CD integration**: Streamlined testing across platforms


2. **Developer convenience**: Simple commands for common tasks

## Best Practices

2. **Keep configuration up to date**: Update `uvfast.json` when adding new hardware configurations


2. **Generate lockfiles after requirements changes**: Run `uvfast.py lock` after updating any requirements files


2. **Use wrappers for consistency**: Encourage team members to use the wrapper scripts


2. **Include in CI**: Integrate with your CI/CD pipeline for consistent testing


2. **Document hardware-specific needs**: Make sure to document any hardware-specific considerations

## Troubleshooting

2. **Environment issues**:

   - Try recreating the environment: `uvfast.py setup --clean`
   - Check if all requirements files exist

2. **Lockfile conflicts**:

   - Resolve conflicts in requirements files
   - Regenerate lockfiles

2. **Script permissions**:

   - Ensure scripts are executable

2. __Missing dependencies_*:

   - Check hardware-specific requirements
   - Verify lockfiles are up to date

```text`

```text`
