# AI Playground Quick Start Guide

This guide will help you get started with AI Playground using uv, an extremely fast Python package and project manager.

## Prerequisites

- Python 3.10 or newer (3.13 recommended)
- Git

## Installation

### 1. Install uv

On macOS and Linux:

```bash
curl -LsSf <https://astral.sh/uv/install.sh> | sh

```text

On Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"


```text

### 2. Clone the Repository

```bash
git clone <https://github.com/zimm0140/AI-Playground.git>
cd AI-Playground

```text

### 3. Set Up Development Environment

#### Option 1: One-click Setup (Recommended)

We provide a one-click setup script that handles everything for you:

```bash

# Run the setup script

python scripts/setup_dev_environment.py

```text

This script:

- Installs uv if not already installed
- Creates a virtual environment
- Installs all dependencies from lockfiles
- Sets up pre-commit hooks
- Configures VS Code settings

#### Option 2: Manual Setup

```bash

# Create a virtual environment with uv

uv venv

# Activate the virtual environment (depends on your shell)

# On Windows

.venv\Scripts\activate

# On macOS/Linux

source .venv/bin/activate

# Install dependencies for development

uv pip install -e ".[dev]"

# Install pre-commit hooks

pre-commit install

```text

## Running Tests

With virtual environment activated:

```bash

# Run all tests

pytest

# Or using uv directly (no need to activate environment)

uv run pytest

```text

## Development Workflow

1. **Edit Code**: Make your changes to the codebase.

1. **Run Tests**: Verify your changes work:
   ```bash
   uv run pytest
   ```text

1. **Check Code Quality**: Run pre-commit hooks:
   ```bash
   pre-commit run --all-files
   ```text

1. **Fix Issues**: Address any linting or type checking errors.

1. **Commit**: Commit your changes:
   ```bash
   git add .
   git commit -m "Your meaningful commit message"
   ```text

## Common Tasks

### Adding New Dependencies

To add a new dependency:

```bash

# Regular dependency

uv pip install some-package

# Add to pyproject.toml and setup.py

python .github/sync_dependencies.py

```text

### Running Specific Tests

```bash

# Run specific test file

uv run pytest tests/test_specific.py

# Run tests with specific name pattern

uv run pytest -k "test_pattern"

```text

### Running Linters Individually

```bash

# Run Ruff linter

uv run ruff check .

# Run mypy type checker

uv run mypy .

```text

## Troubleshooting

If you encounter dependency issues:

```bash

# Re-sync the environment

uv pip sync requirements.txt

# For development packages

uv pip install -e ".[dev]"

```text

For more detailed information, see the full [documentation](README.md) and [migration guide](MIGRATION.md).

## Optional: Shell Completion

Set up shell completion for uv to make your development experience smoother:

### Bash

```bash
uv completion bash > ~/.uv-completion.bash
echo 'source ~/.uv-completion.bash' >> ~/.bashrc

```text

### Zsh

```bash
uv completion zsh > ~/.zsh/_uv
echo 'fpath=(~/.zsh $fpath)' >> ~/.zshrc
echo 'autoload -Uz compinit && compinit' >> ~/.zshrc

```text

### Fish

```bash
uv completion fish > ~/.config/fish/completions/uv.fish

```text

### PowerShell

```powershell
uv completion powershell | Out-File -Encoding utf8 -FilePath (Join-Path $PROFILE.CurrentUserAllHosts "uv.ps1")

```text

## Using Helper Scripts

We've provided convenient wrapper scripts for common uv operations:

### On Unix/Linux/macOS

```bash

# Make the script executable first

chmod +x scripts/run_with_uv.sh

# Run tests

./scripts/run_with_uv.sh test

# Run linters

./scripts/run_with_uv.sh lint

# Format code

./scripts/run_with_uv.sh format

# Run the example script with isolated dependencies

./scripts/run_with_uv.sh run scripts/example_script.py

# See all available commands

./scripts/run_with_uv.sh help

```text

### On Windows

```powershell

# Run tests

.\scripts\run_with_uv.ps1 test

# Run linters

.\scripts\run_with_uv.ps1 lint

# Format code

.\scripts\run_with_uv.ps1 format

# See all available commands

.\scripts\run_with_uv.ps1 help

```text

## Using Lockfiles

For reproducible environments, we maintain lockfiles for exact dependency versions:

```bash

# Sync your environment with locked dependencies

uv pip sync requirements.lock

# For development dependencies

uv pip sync requirements-dev.lock

# Or both at once

uv pip sync requirements.lock requirements-dev.lock

# To update lockfiles after changing requirements.txt

uv pip compile requirements.txt --output-file requirements.lock
uv pip compile requirements-dev.txt --output-file requirements-dev.lock

```text

## Docker Integration

We provide a Dockerfile that uses uv for faster builds:

```bash

# Build the development image

docker build --target development -t ai-playground-dev .

# Build the production image

docker build --target production -t ai-playground .

# Run the container

docker run -p 5000:5000 ai-playground

```
