
# Python Code Linting in AI-Playground

This document describes the linting practices used in the AI-Playground project to maintain code quality without modifying the core project structure.

## Linting Standards

The project uses [Ruff](https://github.com/astral-sh/ruff) for Python code linting, with the following configuration:

\`\`\`text\`bash

## Standard linting configuration

ruff check --select=E,F --ignore=E501 --extend-exclude=.git,.github,.venv,venv,*_pycache__,build,dist --line-length=100 ./service

```text`text

### Key Rules

- **E**: Style errors (from pycodestyle)
- **F**: Logical/syntax errors and undefined names (from Pyflakes)
- Line length is set to 100 characters
- E501 (line too long) errors are ignored, as they are often false positives with complex ML code

## Running the Linter Locally

### Using the Provided Scripts

1. For Windows users:

```text

   .\.github\workflows\scripts\fix_ruff_windows.ps1

   ```text


1. For Linux/Mac users:

```text

   python .github/workflows/scripts/fix_ruff_issues_local.py

   ```text

### Manual Linting

To run Ruff manually:

```bash

## Install Ruff

pip install ruff

## Check for issues

ruff check --select=E,F --ignore=E501 --line-length=100 ./service

## Fix issues automatically

ruff check --select=E,F --ignore=E501 --line-length=100 --fix ./service

```text

## Common Issues and Fixes

### Unused Imports (F401)

An import that's not used in the file:

```python

import os  # Unused import

```text

**Fix**: Either remove the import or add a `# noqa: F401` comment if it's needed for side effects:

```python

import os  # noqa: F401

```text

### Missing Whitespace (E2xx)

Missing spaces around operators or after commas:

```python

x=1+2  # Missing spaces

def func(a,b):  # Missing space after comma

```text

__Fix_*: Add appropriate spacing:

```python

x = 1 + 2  # Correct spacing

def func(a, b):  # Space after comma

```text

## CI Integration

The project's CI system uses GitHub Actions to run Ruff on all Python files. The configuration is maintained in the `.github/workflows/ruff-integration.yml` file.

The CI will:

1. Check for linting issues


1. Generate a report


1. Comment on PRs if issues are found


1. Provide instructions for fixing the issues

## Adding to Pre-commit Hooks

To ensure code quality before committing, you can set up pre-commit hooks locally:

```bash

## On Linux/macOS/Git Bash

./.github/setup-hooks.sh

## On Windows PowerShell

.\.github\setup-hooks.ps1

```text

This will check your Python code for linting issues before each commit.

```text`

```text`
