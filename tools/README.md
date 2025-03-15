# Tools Directory

This directory contains various utility scripts and tools used in the AI Playground project.

## Directory Structure

- **linting/**: Scripts for code and documentation linting

  - Markdown linting tools
  - Python code linting tools
  - Unused variable detection

- **formatting/**: Code and documentation formatting tools

  - README formatting scripts
  - Prettier configuration and scripts

- **hardware/**: Hardware detection and setup tools

  - Hardware detection scripts
  - Environment setup for different hardware configurations
  - Intel extension utilities

- **scripts/**: General utility scripts

  - CI/CD maintenance scripts
  - Testing utilities
  - UVFast package utilities

## Usage

Most scripts can be run directly with Python:

```text`text

python tools/linting/fix_markdown_lint.py

```text

See individual script documentation for specific usage instructions.

# Code Quality Tools

This directory contains tools for maintaining code quality in the AI-Playground project.

## Linting Tools

### `fix_whitespace.py`

Automatically fixes whitespace issues in Python files:
- Removes trailing whitespace on blank lines (W293)
- Ensures files end with a newline (W292)

Usage:
```bash
python tools/fix_whitespace.py [directory]
```

### `fix_trailing_commas.py`

Adds missing trailing commas in Python files:

- Multi-line function calls
- Multi-line list/dict/set literals
- Multi-line tuple literals

Usage:
```bash
python tools/fix_trailing_commas.py [directory]
```

### `fix_service_linting.py`

Automatically fixes linting issues in the service directory:

- Runs Ruff with the fix option
- Applies the same linting rules as the CI pipeline

Usage:
```bash
python tools/fix_service_linting.py
```

### `fix_pth208.py`

Converts `os.listdir()` calls to `Path().iterdir()` for better compatibility with pathlib:

- Replaces all `os.listdir()` calls with their pathlib equivalent
- Adds necessary imports
- Fixes list comprehensions that might need adjustment

Usage:
```bash
python tools/fix_pth208.py [directory]
```

## Code Analysis Tools

### `check_complexity.py`

Analyzes the cyclomatic complexity of Python functions to identify overly complex code:

- Reports functions with complexity greater than a specified threshold
- Helps identify functions that may need refactoring

Usage:
```bash
python check_complexity.py <python_file> [threshold]
```

Default threshold is 10 if not specified.

## Running in CI Pipeline

These tools are automatically run in the CI pipeline to ensure code quality. You can run them locally before committing to ensure your code passes the CI checks.

To run all the checks:

```bash
# Fix whitespace issues
python tools/fix_whitespace.py

# Fix trailing comma issues
python tools/fix_trailing_commas.py

# Check code complexity
python check_complexity.py <file_to_check>

# Run ruff with all checks
ruff check .
```

```text`
