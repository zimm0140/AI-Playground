
# uvfast: Modern Python Environment Management {#uvfast-modern-python-environment-management}

This document provides a step-by-step guide for implementing and using the `uvfast` system in your Python projects.

## Implementation Steps {#implementation-steps}

### 1. Core Files {#core-files}

1. *_Create the uvfast.py script__:

   - Copy the `uvfast.py` script to your project root
   - Make it executable: `chmod +x uvfast.py` (on Unix/Linux/macOS)

1. **Create convenience wrappers**:

   - For Unix/Linux/macOS: Create `scripts/uvfast.sh`
   - For Windows: Create `scripts/uvfast.ps1`
   - Make them executable: `git update-index --chmod=+x scripts/uvfast.sh scripts/uvfast.ps1`

1. **Create the configuration file**:

   - Create `uvfast.json` in your project root with your project-specific settings

### 2. Requirements Files {#requirements-files}

1. **Base requirements**:

   - Ensure you have a `requirements.txt` file with core dependencies

1. **Development requirements**:

   - Create `requirements-dev.txt` with development dependencies
   - Include testing, linting, and type checking packages

1. **Hardware-specific requirements** (optional):

   - Create separate files for different hardware configurations:

\`\`\`text\`text

- `requirements-hardware-base.txt`

- `requirements-hardware-acm.txt` (for Intel Arc GPUs)

- `requirements-hardware-ovino.txt` (for OpenVINO)

- Add any other hardware-specific configurations

````

### 3. Documentation {#documentation}

1. **Update your README.md**:

   - Add installation instructions
   - Explain the dual approach (traditional pip vs. uvfast)

1. **Add a QUICKSTART.md**:

   - Include basic usage examples
   - List common commands

1. **Add a cheatsheet**:

   - Create `UVFAST_CHEATSHEET.md` with common commands

### 4. CI/CD Integration {#cicd-integration}

1. **GitHub Actions**:

   - Create or update `.github/workflows/ci.yml`
   - Use the uvfast script to set up environments
   - Run tests and linting

## Usage Guide {#usage-guide}

### Basic Commands {#basic-commands}

```bash

## Setup environment with development dependencies {#setup-environment-with-development-dependencies}

python uvfast.py setup --dev

## Show environment information {#show-environment-information}

python uvfast.py info

## Generate lockfiles for all hardware types {#generate-lockfiles-for-all-hardware-types}

python uvfast.py lock

## Run tests {#run-tests}

python uvfast.py run pytest

## Run linting {#run-linting}

python uvfast.py run ruff check .

## Run type checking {#run-type-checking}

python uvfast.py run mypy .

```

### Usi

ng {#using}

 Wrapper Scripts {#using-wrapper-scripts}

```bas
h

## Unix/Linux/macOS {#unixlinuxmacos}

./scripts/uvfast.sh setup --dev

## Windows PowerShell {#windows-powershell}

.\scripts\uvfast.ps1 setup --dev

```

### H

ar {#har}

dware-Specific Setup {#hardware-specific-setup}

```b
as
h

## Setup for Intel Arc GPUs {#setup-for-intel-arc-gpus}

python uvfast.py setup --hardware acm --dev

## Setup for OpenVINO {#setup-for-openvino}

python uvfast.py setup --hardware ovino --dev

```

##

Im {#im}

plementation Example {#implementation-example}

### uvfast.json {#uvfastjson}

``
`j
son

{
  "project_name": "my-project",
  "python_version": "3.10",
  "hardware_types": ["base", "acm", "ovino"],
  "default_hardware": "base",
  "requirements": {

```

"
ba
se": "requirements.txt",
"dev": "requirements-dev.txt",
"hardware": {
  "base": "requirements-hardware-base.txt",
  "acm": "requirements-hardware-acm.txt",
  "ovino": "requirements-hardware-ovino.txt"
}

``
`
  },
  "lockfiles": {

```

"
base": "requirements.lock",
"hardware": {
  "base": "requirements-hardware-base.lock",
  "acm": "requirements-hardware-acm.lock",
  "ovino": "requirements-hardware-ovino.lock"
}

```

  }
}

```

## Benefits of Using uvfast {#benefits-of-using-uvfast}

1. **Modern tooling**: Leverages `uv` for faster package installation

1. **Reproducible environments**: Uses lockfiles for consistent dependencies

1. **Hardware-specific setups**: Easily manage different hardware configurations

1. **CI/CD integration**: Streamlined testing across platforms

1. **Developer convenience**: Simple commands for common tasks

## Best Practices {#best-practices}

1. **Keep configuration up to date**: Update `uvfast.json` when adding new hardware configurations

1. **Generate lockfiles after requirements changes**: Run `uvfast.py lock` after updating any requirements files

1. **Use wrappers for consistency**: Encourage team members to use the wrapper scripts

1. **Include in CI**: Integrate with your CI/CD pipeline for consistent testing

1. **Document hardware-specific needs**: Make sure to document any hardware-specific considerations

## Troubleshooting {#troubleshooting}

1. **Environment issues**:

   - Try recreating the environment: `uvfast.py setup --clean`
   - Check if all requirements files exist

1. **Lockfile conflicts**:

   - Resolve conflicts in requirements files
   - Regenerate lockfiles

1. **Script permissions**:

   - Ensure scripts are executable

1. __Missing dependencies_*:

   - Check hardware-specific requirements
   - Verify lockfiles are up to date

````

````