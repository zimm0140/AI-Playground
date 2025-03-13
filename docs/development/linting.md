# Linting Guide

This guide outlines the linting practices used in the AI-Playground project to maintain code quality and consistency.

## Linting Tools

AI-Playground primarily uses the following linting tools:

- *_[Ruff](https://github.com/astral-sh/ruff)__: A fast Python linter that combines multiple linting tools
- __[mypy](https://mypy.readthedocs.io/)__: A static type checker for Python
- __[markdownlint](https://github.com/DavidAnson/markdownlint)__: A linter for Markdown files

## Python Linting Configuration

### Ruff Configuration

The project uses Ruff with the following settings:

\`\`\`text\`toml

## in pyproject.toml

[tool.ruff]
target-version = "py310"
line-length = 100
select = ["E", "F", "I", "W", "N", "B", "C4", "UP", "T20"]
ignore = ["E501"]
extend-exclude = \[".git", ".github", ".venv", "venv", "__pycache__", "build", "dist"\]

```text`text

#### Key Rules

- __E__: Style errors (from pycodestyle)
- __F__: Logical/syntax errors and undefined names (from Pyflakes)
- __I__: Import sorting (from isort)
- __W__: Warnings (from pycodestyle)
- __N__: Naming conventions (from pep8-naming)
- __B__: Bug detection (from flake8-bugbear)
- __C4__: Comprehension complexity (from flake8-comprehensions)
- __UP__: Python upgrade suggestions (from pyupgrade)
- __T20__: Print statement detection (from flake8-print)

### Type Checking with mypy

For static type checking, we use mypy with these settings:

```toml

## in pyproject.toml

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
strict_optional = true

```text

## Running Linters Locally

### Using the Provided Scripts

1. For Windows users:

   ```powershell
   .\.github\workflows\scripts\fix_ruff_windows.ps1
   ```text

1. For Linux/Mac users:

   ```bash
   python .github/workflows/scripts/fix_ruff_issues_local.py
   ```text

### Manual Linting

To run Ruff manually:

```bash

## Install Ruff

pip install ruff

## Check for issues

ruff check .

## Fix issues automatically

ruff check --fix .

```text

To run mypy:

```bash

## Install mypy

pip install mypy

## Run type checking

mypy .

```text

To run markdownlint on Markdown files:

```bash

## Install markdownlint (requires Node.js)

npm install -g markdownlint-cli

## Check Markdown files

markdownlint "__/_.md"

```text

## Common Linting Issues and Fixes

### Unused Imports (F401)

An import that's not used in the file:

```python
import os  # Unused import

```text

*_Fix__: Either remove the import or add a `# noqa: F401` comment if it's needed for side effects:

```python
import os  # noqa: F401

```text

### Missing Whitespace (E2xx)

Missing spaces around operators or after commas:

```python
x=1+2  # Missing spaces

def func(a,b):  # Missing space after comma

```text

__Fix__: Add appropriate spacing:

```python
x = 1 + 2  # Correct spacing

def func(a, b):  # Space after comma

```text

### Type Annotation Issues

Missing or incorrect type annotations:

```python
def process_data(data):  # Missing type annotations

```text

return data + 1

```text

```text

__Fix__: Add proper type annotations:

```python
def process_data(data: int) -> int:

```text

return data + 1

```text

```text

### Hardware-Specific Import Issues

Importing hardware-specific modules that might not be available:

```python
import intel_extension_for_pytorch  # May not be available on all systems

```text

__Fix__: Use conditional imports:

```python
try:

```text

import intel_extension_for_pytorch
HAS_INTEL_EXTENSION = True

```text
except ImportError:

```text

HAS_INTEL_EXTENSION = False

```text

```text

## CI Integration

The project's CI system uses GitHub Actions to run linters on all files. The configuration is maintained in the following files:

- `.github/workflows/ruff-integration.yml` (for Ruff)
- `.github/workflows/type-check.yml` (for mypy)
- `.github/workflows/docs-check.yml` (for markdownlint)

The CI will:

1. Check for linting issues
2. Generate a report
3. Comment on PRs if issues are found
4. Provide instructions for fixing the issues

## Pre-commit Hooks

To ensure code quality before committing, you can set up pre-commit hooks locally:

```bash

## On Linux/macOS/Git Bash

./.github/setup-hooks.sh

## On Windows PowerShell

.\.github\setup-hooks.ps1

```text

This will check your code for linting issues before each commit.

## Temporary Disabling of Linter Rules

There are cases where linter rules need to be temporarily disabled:

```python

## In situations where a line is necessarily long

long_url = "<https://very-long-url-that-cannot-be-split.com/path/to/resource">  # noqa: E501

## When using a variable name that doesn't match conventions

def connect_to_API():  # noqa: N802

```text

pass

```text

```text

Use `# noqa:` comments sparingly and only when necessary.

## Hardware-Specific Linting Considerations

When writing hardware-specific code:

1. Use conditional imports for hardware-specific dependencies
2. Consider using feature checking rather than relying on specific hardware
3. Add appropriate comments where hardware specifics affect code structure
4. Use type annotations that reflect hardware-specific considerations

```python
def optimize_for_hardware(model: torch.nn.Module, hardware_type: str) -> torch.nn.Module:

```text

"""
Optimize model for specific hardware.

```text

```text

Args:

```text

model: The PyTorch model
hardware_type: One of "acm", "bmg", or "base"

```text

```text

```text

Returns:

```text

Optimized model

```text
"""
if hardware_type == "acm":

```text

try:

```text

import intel_extension_for_pytorch as ipex  # noqa: F401

```text

```text

```text

```text

```text

```text

model = ipex.optimize(model)

```text
except ImportError:

```text

pass  # Fall back to unoptimized model

```text

```text

```text

```text

return model

```text

```text

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [markdownlint Rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Code Quality Standards](code-quality.md)

---
__Previous__: [Testing Guide](testing.md) | __Next__: [Project Architecture](../architecture/overview.md) | __See also_*: [Code Quality Standards](code-quality.md)


```text`

```text`
