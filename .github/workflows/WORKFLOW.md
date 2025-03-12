# CI Workflow Overview

This document provides an overview of the CI workflow implemented in this repository.

## Workflow Diagram

```text
┌────────────────────────────┐                  ┌────────────────────────────┐
│                            │                  │                            │
│       Linux Matrix         │                  │      Windows Matrix        │
│    (Python 3.9, 3.10)      │                  │     (Python 3.10)          │
│                            │                  │                            │
└───────────┬────────────────┘                  └──────────────┬─────────────┘
            │                                                  │
            ▼                                                  ▼
┌────────────────────────────┐                  ┌────────────────────────────┐
│                            │                  │                            │
│  1. Environment Setup      │                  │  1. Environment Setup      │
│  - Install Dependencies    │                  │  - Install Dependencies    │
│  - Configure Environment   │                  │  - Configure Environment   │
│                            │                  │                            │
└───────────┬────────────────┘                  └──────────────┬─────────────┘
            │                                                  │
            ▼                                                  ▼
┌────────────────────────────┐                  ┌────────────────────────────┐
│                            │                  │                            │
│  2. Hardware Patching      │                  │  2. Basic Testing          │
│  - Create CPU Mode Patches │                  │  - Syntax Checks           │
│  - Mock Intel Extensions   │                  │  - Run Tests               │
│  - Fix Code Issues         │                  │                            │
│                            │                  │                            │
└───────────┬────────────────┘                  └──────────────┬─────────────┘
            │                                                  │
            ▼                                                  ▼
┌────────────────────────────┐                  ┌────────────────────────────┐
│                            │                  │                            │
│  3. Analysis & Testing     │                  │  3. Reporting              │
│  - Code Quality Checks     │                  │  - Generate Report         │
│  - Security Scanning       │                  │  - Upload Artifacts        │
│  - Shell Script Linting    │                  │                            │
│  - Run Tests               │                  │                            │
│  - Test Coverage Analysis  │                  │                            │
│                            │                  │                            │
└───────────┬────────────────┘                  └──────────────┬─────────────┘
            │                                                  │
            ▼                                                  │
┌────────────────────────────┐                                 │
│                            │                                 │
│  4. Documentation          │                                 │
│  - API Documentation       │                                 │
│  - Compatibility Reports   │                                 │
│  - Tool Compatibility      │                                 │
│                            │                                 │
└───────────┬────────────────┘                                 │
            │                                                  │
            ▼                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              Summary Job                                    │
│                                                                             │
│  - Collects results from all jobs                                           │
│  - Generates cross-platform compatibility report                            │
│  - Creates final summary                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### Linux Workflow

The Linux workflow performs comprehensive testing and analysis:

1. **Environment Setup**
   - Installs dependencies
   - Configures the environment for testing

1. **Hardware Patching**
   - Creates CPU mode patches for hardware acceleration libraries
   - Mocks hardware-dependent modules to enable testing in CI environment
   - Fixes code issues specific to the CI environment

1. **Analysis & Testing**
   - Runs code quality checks with yapf
   - Performs security scanning of dependencies with Safety
   - Lints shell scripts with Shellcheck
   - Runs tests with a custom resilient test runner
   - Analyzes test coverage

1. **Documentation**
   - Generates API documentation
   - Creates compatibility reports
   - Analyzes tool compatibility

### Windows Workflow

The Windows workflow runs a streamlined version of tests:

1. **Environment Setup**
   - Installs dependencies with Windows-specific paths
   - Configures environment variables

1. **Basic Testing**
   - Checks for Python syntax errors
   - Runs tests with the custom test runner
   - Checks code formatting

1. **Reporting**
   - Generates Windows-specific compatibility report
   - Uploads artifacts for review

### Summary Job

The summary job integrates the results from all platforms:

- Collects results from Linux and Windows jobs
- Generates a cross-platform compatibility report
- Creates a final summary with status badges

## Artifacts

The CI workflow produces several artifacts that provide detailed information about the codebase:

- **Code Quality Reports**: Results of syntax and style checks
- **Security Reports**: Findings from dependency vulnerability scanning
- **Coverage Reports**: Code coverage metrics with visualizations
- **Shell Script Analysis**: Results of shell script linting
- **API Documentation**: Generated API documentation from docstrings
- **Hardware Support Matrix**: Documentation of supported hardware configurations
- **Compatibility Reports**: Reports on compatibility with different Python versions and platforms

These artifacts are available for download from the GitHub Actions workflow run page.
