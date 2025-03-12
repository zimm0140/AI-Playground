# Scripts

This directory contains general utility scripts for the AI Playground project.

## Contents

- `uvfast.py`: Script for managing Python environments with UVFast
- `uvfast_lockfile_enhancements.py`: Enhancements for UVFast lockfile handling
- `ci_maintenance.py`: Script for maintaining CI/CD workflows
- `validate_colorize.py`: Script for validating color schemes
- `test_venv.py`: Script for testing virtual environments

## Usage

### UVFast

To manage Python environments with UVFast:

```

python tools/scripts/uvfast.py [command]

```

For lockfile enhancements:

```

python tools/scripts/uvfast_lockfile_enhancements.py [file]

```

### CI Maintenance

To maintain CI/CD workflows:

```

python tools/scripts/ci_maintenance.py

```

### Testing

To test virtual environments:

```

python tools/scripts/test_venv.py

```

## Related Configuration

These scripts use configuration files from the `config` directory:

- `uvfast.json`: Configuration for UVFast
- Various requirements files: `requirements*.txt` and `requirements*.lock`
