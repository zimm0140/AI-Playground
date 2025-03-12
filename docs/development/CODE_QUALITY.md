# Code Quality Guidelines

This document outlines the code quality standards and tools used in this project.

## Linting and Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for Python linting and formatting. Ruff is a fast Python linter
written in Rust that combines the functionality of multiple Python linting tools.

### Common Linting Issues

1. **Unused Imports (F401)**
   - Remove unused imports or add `# noqa: F401` with an explanation if the import is needed for side effects.

   - Example: `import module  # noqa: F401 - Import needed for registration`

1. **Unused Variables (F841)**
   - Use `_` for variables you don't need: `_, result = function_with_multiple_returns()`
   - Or add `# noqa: F841` if you need to keep the variable for clarity.

1. **Module Level Import Not at Top of File (E402)**
   - Keep all imports at the top of the file.
   - If you need to modify environment variables before importing, add `# noqa: E402` to the imports.

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run checks before each commit. This helps catch issues early.

### Setup

1. Install pre-commit:

```bash
pip install pre-commit

```text
1. Install the hooks:

```bash
pre-commit install

```text
1. Run against all files:

```bash
pre-commit run --all-files

```text

## CI/CD Checks

Our GitHub Actions workflow runs the following checks:

1. **Linting**: Ruff checks for code style and common errors.
1. **Type Checking**: Mypy verifies type annotations.
1. **Unit Tests**: Pytest runs our test suite.
1. **Pre-commit**: Ensures all pre-commit hooks pass.

## Fixing Linting Issues

We've created helper scripts to fix common linting issues:

1. **check_linting.py**: Checks for linting issues in key directories.
1. **fix_unused_variables.py**: Fixes unused variable warnings.

Run these scripts to quickly identify and fix issues:

```bash
python check_linting.py
python fix_unused_variables.py

```text

## Best Practices

1. **Run pre-commit before pushing**: This catches issues before they reach CI.
1. **Add meaningful comments**: Especially when using `# noqa` directives.

1. **Keep imports organized**: Standard library first, then third-party, then local.
1. **Use type annotations**: They improve code readability and catch errors.
1. **Write tests**: Aim for high test coverage, especially for critical code paths.

## Troubleshooting

If you encounter CI failures:

1. Check the CI logs to identify the specific issues.
1. Run `python check_linting.py` locally to see if you can reproduce the issues.
1. Fix the issues manually or use the helper scripts.
1. Run pre-commit to verify your fixes before pushing again.
