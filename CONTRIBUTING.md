
# Contributing to AI Playground

Thank you for your interest in contributing to AI Playground! This guide will help you get started with our development workflow using modern Python tools.

## Development Environment

We use `uv`, an extremely fast Python package manager, for dependency management and development workflows.

### Setting Up Your Environment

1. **Install uv**:

   ```text`bash


   ## Unix/Linux/macOS

   curl -LsSf https://astral.sh/uv/install.sh | sh

   ## Windows

   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   ```text

   ```text`

1. **Clone and Setup**:

   ```text`bash

   git clone https://github.com/zimm0140/AI-Playground.git
   cd AI-Playground

   ## Create and activate a virtual environment

   uv venv
   source .venv/bin/activate  # Unix/Linux/macOS

   .venv\Scripts\activate     # Windows

   ## Install dependencies

   uv pip sync requirements.lock requirements-dev.lock

   ## Install pre-commit hooks

   pre-commit install

   ```text

   ```text`

## Development Workflow

### Using Helper Scripts

We provide convenient scripts for common development tasks:

```text`bash

## Unix/Linux/macOS

./scripts/run_with_uv.sh test    # Run tests

./scripts/run_with_uv.sh lint    # Run linters

./scripts/run_with_uv.sh format  # Format code

## Windows

.\scripts\run_with_uv.ps1 test
.\scripts\run_with_uv.ps1 lint
.\scripts\run_with_uv.ps1 format

```text

### Before Submitting a Pull Request

1. **Ensure all tests pass**:

   ```bash

   ./scripts/run_with_uv.sh test

   ```text

1. **Check code quality**:

   ```bash

   ./scripts/run_with_uv.sh lint

   ```text

1. **Format your code**:

   ```bash

   ./scripts/run_with_uv.sh format

   ```text

1. **Update lockfiles if you've changed dependencies**:

   ```bash

   uv pip compile requirements.txt --output-file requirements.lock
   uv pip compile requirements-dev.txt --output-file requirements-dev.lock

   ```text

## Type Annotations

We use Python type annotations and verify them with mypy. For Python 3.10+ compatibility:

1. Use `Union` and `Optional` from the `typing` module


1. For Python 3.10+, you can use the `|` operator for union types, but be consistent


1. Use `TypeVar` for generic type annotations

## Commit Messages

Please use clear, descriptive commit messages that explain what changes you've made and why. Follow this format:

```text

Area: Brief description of what changed

More detailed explanation if needed

```text

For example:

```text

CI: Add uv support to GitHub Actions

- Add workflow file for uv-based testing
- Update lockfiles for dependency tracking
- Add helper scripts for common tasks

```text

## Need Help?

If you have questions about the development process or need help with your contribution, please:

1. Check the documentation in QUICKSTART.md and MIGRATION.md


1. Open an issue with the "question" label


1. Ask for help in pull request comments

Thank you for contributing to AI Playground!

```text`
