# CI Scripts

This directory contains scripts used by the CI workflow in GitHub Actions. These scripts have been modularized from the main workflow file to improve maintainability and readability.

## Script Overview

| Script | Description |
|--------|-------------|
| `analyze_test_coverage.sh` | Measures test coverage and generates detailed coverage reports with badges |
| `catalog_hardware.sh` | Catalogues hardware-specific dependency files and creates documentation about supported hardware configurations |
| `check_code_quality.sh` | Runs code quality checks including syntax validation, style checking, and detection of common anti-patterns |
| `check_shell_scripts.sh` | Checks shell scripts for syntax and style issues using shellcheck |
| `check_tool_compatibility.sh` | Checks compatibility with different tool versions and generates reports |
| `cpu_mode_patches.sh` | Creates mock implementations of hardware-dependent modules to ensure tests can run in CPU-only mode in CI |
| `custom_test_runner.py` | Custom test runner that handles import errors gracefully and continues despite failures |
| `fix_ci_issues.py` | Applies patches to source code files to make them compatible with the CI environment |
| `generate_api_docs.sh` | Generates API documentation from Python docstrings |
| `generate_compatibility_report.sh` | Creates a report documenting compatibility with different Python versions and environments |
| `generate_summary.sh` | Creates a detailed summary of the CI run for better visibility in GitHub Actions |
| `scan_dependencies.sh` | Scans dependencies for known security vulnerabilities using Safety |
| `verify_environment.sh` | Performs comprehensive checks of the environment to ensure all patches are working |

## Enhanced Features

The CI workflow now includes:

1. **Multi-platform Testing** - Tests on both Linux and Windows to ensure cross-platform compatibility
1. **Security Scanning** - Checks dependencies for known vulnerabilities
1. **Test Coverage Analysis** - Measures and reports code coverage with visual indicators
1. **Shell Script Linting** - Ensures shell scripts follow best practices
1. **Comprehensive Reporting** - Generates detailed reports for all aspects of the CI process

## Usage

These scripts are called from the main CI workflow file (`.github/workflows/main.yml`). They can also be run manually for local testing:

```bash
# Example of running a script locally
cd /path/to/repository
.github/workflows/scripts/check_code_quality.sh
```

## Maintenance

When modifying the CI workflow:

1. Prefer to modify the individual scripts rather than inlining code in the main workflow file
1. Keep scripts focused on a single responsibility
1. Include appropriate documentation and error handling in each script
1. Ensure all scripts have proper execution permissions (`chmod +x`)
1. Update the Windows job in main.yml if adding new functionality that should work cross-platform 
