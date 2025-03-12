# Helper Scripts

This directory contains various helper scripts to assist with development tasks in the AI Playground project.

## Available Scripts

### `run_with_uv.sh` and `run_with_uv.ps1`

Cross-platform wrapper scripts for running commands with `uv`:

- `run_with_uv.sh` - For Unix/Linux/macOS
- `run_with_uv.ps1` - For Windows

#### Usage

\`\`\`text\`bash

## Unix/Linux/macOS

./scripts/run_with_uv.sh [command]

## Windows

.\\scripts\\run_with_uv.ps1 [command]

````text

#### Available Commands

- `run <script.py>` - Run a Python script in an isolated environment
- `test` - Run pytest
- `lint` - Run linters (ruff, mypy)
- `format` - Format code with ruff
- `sync` - Sync dependencies from lockfiles
- `audit` - Run security audit
- `tool <tool_name>` - Install and run a tool
- `clean` - Clean temporary files
- `help` - Show help message

### `example_script.py`

Demonstrates using `uv` with inline dependencies to fetch and display GitHub repository data.

#### Usage

```bash

## Run directly with uv (automatically installs dependencies)

uv run scripts/example_script.py [organization_name] [num_repos]

## Or run through the wrapper script

./scripts/run_with_uv.sh run scripts/example_script.py [organization_name] [num_repos]

```text

### `fix_type_annotations.py`

Tool to scan Python files and detect type annotations that could be updated for Python 3.10+ compatibility.

#### Usage

```bash

## Scan the entire project

python scripts/fix_type_annotations.py .

## Scan a specific file or directory

python scripts/fix_type_annotations.py path/to/file_or_dir

## Run in dry-run mode (don't make changes)

python scripts/fix_type_annotations.py --dry-run .

## Show detailed information about changes

python scripts/fix_type_annotations.py --verbose .

```text

### `setup_dev_environment.py`

One-click setup script for new developers to set up the complete development environment.

#### Usage

```bash

## Run the setup script

python scripts/setup_dev_environment.py

```text

This script:

- Installs uv if not already installed
- Creates a virtual environment
- Installs all dependencies from lockfiles
- Sets up pre-commit hooks
- Configures VS Code settings
- Provides guidance on next steps

## Adding New Scripts

When adding new helper scripts to this directory:

1. Follow the naming conventions: descriptive names in snake_case
1. Add appropriate shebang lines and docstrings
1. Make shell scripts executable: `chmod +x scripts/your_script.sh`
1. Update this README with documentation for the script
1. Include both Unix/Linux/macOS and Windows versions when applicable

## Best Practices

- Use proper error handling in scripts
- Include clear help messages and usage instructions
- Make scripts robust to different environments
- Test scripts on multiple platforms when possible
- Follow consistent coding style within scripts

```text`

````
