# Migration Guide: Python 3.10+ Upgrade

This document outlines the process of migrating AI Playground to Python 3.10+ while maintaining backward compatibility with upstream.

## Overview

We've implemented a pragmatic approach that:
1. Modernizes the development environment and CI with tools like uv and Rye
1. Maintains backward compatibility using traditional pip installation
1. Updates type annotations for Python 3.10+ compatibility
1. Implements better linting and formatting tools

## For Developers

### Getting Started

Choose one of the following development approaches:

#### Primary Method (Recommended - uv)

```bash

# Install uv

# On macOS and Linux

curl -LsSf <https://astral.sh/uv/install.sh> | sh

# On Windows

powershell -ExecutionPolicy ByPass -c "irm <https://astral.sh/uv/install.ps1> | iex"


# Set up the environment

uv venv
uv pip install -e ".[dev]"

# Install pre-commit hooks

pre-commit install

```text

#### Alternative Modern Method (Rye)

```bash

# Install Rye

curl -sSf <https://rye-up.com/get> | bash

# Set up the environment

rye sync

# Install pre-commit hooks

pre-commit install

```text

#### Traditional Method

```bash

# Install dependencies using pip

pip install -e .

# Install development dependencies

pip install -e ".[dev]"

# Install pre-commit hooks

pre-commit install

```text

### Type Annotation Changes

We've updated our type annotations to be compatible with Python 3.10+:

1. Replaced pipe syntax (`|`) with `Union` from typing:
   ```python
   # Before (Python 3.10+)

   def some_function(param: str | int) -> list[str] | None:


```text

   ...

```text

   # After (Compatible with Python 3.10+)

   from typing import Union, List, Optional
   def some_function(param: Union[str, int]) -> Optional[List[str]]:

```text

   ...

```text
   ```text

1. Fixed Optional handling:
   ```python
   # Before (problematic)

   os.path.join(maybe_none, "subdir")  # Type error if maybe_none is None

   # After (safe)

   path = os.path.join(maybe_none or "", "subdir")
   ```text

### Dependencies Management

When adding or updating dependencies:

1. Edit either `setup.py` or `pyproject.toml`
1. Run the sync script to keep them in sync:
   ```bash
   python .github/sync_dependencies.py
   ```text

### CI Pipeline

Our CI now uses a hybrid approach with:
- Tests on multiple Python versions (3.10, 3.11, 3.13)
- Testing with multiple installation methods (uv, Rye, and pip)
- Comprehensive linting and type checking using uv for improved performance

## Common Issues and Solutions

### Type Checking Failures

If you encounter type checking errors:

1. Import necessary types from `typing` module
1. Replace pipe syntax (`|`) with `Union[Type1, Type2]`
1. Fix `Optional` type handling with safe defaults

Example:

```python

# Error-prone

def process_file(file_path: Optional[str]) -> None:

```text

with open(os.path.join(file_path, "subfile"), "r") as f:

```text

...

```text

```text

# Fixed

def process_file(file_path: Optional[str]) -> None:

```text

path = file_path or ""
with open(os.path.join(path, "subfile"), "r") as f:

```text

...

```text

```text

```text

### Package Compatibility

Some packages may require updates for Python 3.10+ compatibility. Check for:

1. Deprecated `collections` imports (use `collections.abc` instead)
1. Updated typing syntax
1. Changes in function signatures

## Future Improvements

- Gradually adopt more Python 3.10+ features
- Migrate to native type annotations as Python 3.9 support is phased out
- Consider adopting Rust extensions for performance-critical code
- Explore uv's workspace features for better monorepo support

## Help and Support

If you encounter issues during migration, please:
1. Check this guide for solutions
1. Review the existing issues on GitHub
1. Open a new issue with detailed reproduction steps
