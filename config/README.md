# Configuration Directory

This directory contains configuration files for the AI Playground project.

## Contents

### Linting and Formatting Configuration

- `.markdownlint.yaml`: Configuration for markdown linting
- `.prettierrc` and `.prettierrc.json`: Configuration for Prettier code formatter
- `.pre-commit-config.yaml`: Configuration for pre-commit hooks
- `mypy.ini`: Configuration for mypy type checking
- `pyrightconfig.json`: Configuration for Pyright type checking

### Environment Configuration

- `environment.yml`: Conda environment configuration
- `requirements*.txt`: Python package requirements for different environments
- `requirements*.lock`: Locked versions of Python package requirements

### Workflow Configuration

- `workflow_dashboard.json`: Configuration for workflow dashboard
- `workflow_requirements_results.json`: Results from workflow requirements analysis
- `schema_validation_report.json`: Schema validation report

### Application Configuration

- `uvfast.json`: Configuration for UVFast package

## Usage

These configuration files are used by various tools and scripts in the project. Most are automatically loaded by their respective tools when run from the project root directory.

For example, the markdown linting configuration is used by the markdown linting scripts:

````text

python tools/linting/fix_markdown_lint.py

```text

Some configuration files may need to be symlinked or copied to the project root when used by tools that don't support custom configuration paths.
````
