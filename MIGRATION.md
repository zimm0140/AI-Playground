# Migration Guide: Python 3.10+ Upgrade

This document outlines the process of migrating AI Playground to Python 3.10+ while maintaining backward compatibility with upstream.

## Overview

We've implemented a pragmatic approach that:
1. Modernizes the development environment and CI with tools like Rye
2. Maintains backward compatibility using traditional pip installation
3. Updates type annotations for Python 3.10+ compatibility
4. Implements better linting and formatting tools

## For Developers

### Getting Started

Choose either the traditional or modern development approach:

#### Traditional Method
```bash
# Install dependencies using pip
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Modern Method (Recommended)
```bash
# Install Rye
curl -sSf https://rye-up.com/get | bash

# Set up the environment
rye sync

# Install pre-commit hooks
pre-commit install
```

### Type Annotation Changes

We've updated our type annotations to be compatible with Python 3.10+:

1. Replaced pipe syntax (`|`) with `Union` from typing:
   ```python
   # Before (Python 3.10+)
   def some_function(param: str | int) -> list[str] | None:
       ...

   # After (Compatible with Python 3.10+)
   from typing import Union, List, Optional
   def some_function(param: Union[str, int]) -> Optional[List[str]]:
       ...
   ```

2. Fixed Optional handling:
   ```python
   # Before (problematic)
   os.path.join(maybe_none, "subdir")  # Type error if maybe_none is None

   # After (safe)
   path = os.path.join(maybe_none or "", "subdir")
   ```

### Dependencies Management

When adding or updating dependencies:

1. Edit either `setup.py` or `pyproject.toml`
2. Run the sync script to keep them in sync:
   ```bash
   python .github/sync_dependencies.py
   ```

### CI Pipeline

Our CI now uses a hybrid approach with:
- Tests on multiple Python versions (3.10, 3.11, 3.13)
- Both traditional (pip) and modern (Rye) installation methods
- Comprehensive linting and type checking

## Common Issues and Solutions

### Type Checking Failures

If you encounter type checking errors:

1. Import necessary types from `typing` module
2. Replace pipe syntax (`|`) with `Union[Type1, Type2]`
3. Fix `Optional` type handling with safe defaults

Example:
```python
# Error-prone:
def process_file(file_path: Optional[str]) -> None:
    with open(os.path.join(file_path, "subfile"), "r") as f:
        ...

# Fixed:
def process_file(file_path: Optional[str]) -> None:
    path = file_path or ""
    with open(os.path.join(path, "subfile"), "r") as f:
        ...
```

### Package Compatibility

Some packages may require updates for Python 3.10+ compatibility. Check for:

1. Deprecated `collections` imports (use `collections.abc` instead)
2. Updated typing syntax
3. Changes in function signatures

## Future Improvements

- Gradually adopt more Python 3.10+ features
- Migrate to native type annotations as Python 3.9 support is phased out
- Consider adopting Rust extensions for performance-critical code

## Help and Support

If you encounter issues during migration, please:
1. Check this guide for solutions
2. Review the existing issues on GitHub
3. Open a new issue with detailed reproduction steps 