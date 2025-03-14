
# Python Code Linting in AI-Playground {#python-code-linting-in-ai-playground}

This document describes the linting practices used in the AI-Playground project to maintain code quality without modifying the core project structure.

## Linting Standards {#linting-standards}

The project uses [Ruff](https://github.com/astral-sh/ruff) for Python code linting, with the following configuration:

\`\`\`text\`bash

## Standard linting configuration {#standard-linting-configuration}

ruff check --select=E,F --ignore=E501 --extend-exclude=.git,.github,.venv,venv,*_pycache__,build,dist --line-length=100 ./service

````

### Key Rules {#key-rules}

- **E**: Style errors (from pycodestyle)

- **F**: Logical/syntax errors and undefined names (from Pyflakes)

- Line length is set to 100 characters

- E501 (line too long) errors are ignored, as they are often false positives with complex ML code

## Running the Linter Locally {#running-the-linter-locally}

### Using the Provided Scripts {#using-the-provided-scripts}

1. For Windows users:

```

   .\.
gi
thub\workflows\scripts\fix_ruff_windows.ps1

   ```

1. For Linux/Mac users:

```

   python
 .
github/workflows/scripts/fix_ruff_issues_local.py

   ```

### Manual Linting {#manual-linting}

To run Ruff manually:

```bash

## Install Ruff {#install-ruff}

pip install ruff

## Check for issues {#check-for-issues}

ruff check --select=E,F --ignore=E501 --line-length=100 ./service

## Fix issues automatically {#fix-issues-automatically}

ruff check --select=E,F --ignore=E501 --line-length=100 --fix ./service

```

## Comm

on {#common}

 Issues and Fixes {#common-issues-and-fixes}

### Unused Imports (F401) {#unused-imports-f401}

An import that's not used in the file:

```pyt
ho
n

import os  # Unused import

```

**Fix
**
: Either remove the import or add a `# noqa: F401` comment if it's needed for side effects:

```p
yt
hon

import os  # noqa: F401

```

###

 M {#m}

issing Whitespace (E2xx) {#missing-whitespace-e2xx}

Missing spaces around operators or after commas:

``
`p
ython

x=1+2  # Missing spaces

def func(a,b):  # Missing space after comma

```

_
_F
ix_*: Add appropriate spacing:

``
`python

x = 1 + 2  # Correct spacing

def func(a, b):  # Space after comma

```

#

# CI Integration {#ci-integration}

The project's CI system uses GitHub Actions to run Ruff on all Python files. The configuration is maintained in the `.github/workflows/ruff-integration.yml` file.

The CI will:

1. Check for linting issues

1. Generate a report

1. Comment on PRs if issues are found

1. Provide instructions for fixing the issues

## Adding to Pre-commit Hooks {#adding-to-pre-commit-hooks}

To ensure code quality before committing, you can set up pre-commit hooks locally:

```bash

## On Linux/macOS/Git Bash {#on-linuxmacosgit-bash}

./.github/setup-hooks.sh

## On Windows PowerShell {#on-windows-powershell}

.\.github\setup-hooks.ps1

```

This will check your Python code for linting issues before each commit.

````

````