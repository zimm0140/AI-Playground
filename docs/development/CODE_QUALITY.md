# Code Quality Guidelines

This document outlines the code quality standards and tools used in this project.

## Linting and Formatting

We use [Ruff](https://github.com/astral-sh/ruff) for Python linting and formatting. Ruff is a fast Python linter
written in Rust that combines the functionality of multiple Python linting tools.

### Common Linting Issues

1. *_Unused Imports (F401)__

   - Remove unused imports or add `# noqa: F401` with an explanation if the import is needed for side effects.

   - Example: `import module  # noqa: F401 - Import needed for registration`

1. __Unused Variables (F841)__

   - Use `_` for variables you don't need: `_, result = function_with_multiple_returns()`
   - Or add `# noqa: F841` if you need to keep the variable for clarity.

1. __Module Level Import Not at Top of File (E402)__

   - Keep all imports at the top of the file.
   - If you need to modify environment variables before importing, add `# noqa: E402` to the imports.

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to run checks before each commit. This helps catch issues early.

### Setup

1. Install pre-commit:

\`\`\`text\`bash
pip install pre-commit

````text
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

1. __Linting__: Ruff checks for code style and common errors.
2. __Type Checking__: Mypy verifies type annotations.
3. __Unit Tests__: Pytest runs our test suite.
4. __Pre-commit__: Ensures all pre-commit hooks pass.

## Fixing Linting Issues

We've created helper scripts to fix common linting issues:

1. __check_linting.py__: Checks for linting issues in key directories.
2. __fix_unused_variables.py__: Fixes unused variable warnings.

Run these scripts to quickly identify and fix issues:

```bash
python check_linting.py
python fix_unused_variables.py

```text

## Best Practices

1. __Run pre-commit before pushing__: This catches issues before they reach CI.
2. __Add meaningful comments__: Especially when using `# noqa` directives.

1. __Keep imports organized__: Standard library first, then third-party, then local.
2. __Use type annotations__: They improve code readability and catch errors.
3. __Write tests_*: Aim for high test coverage, especially for critical code paths.

## Troubleshooting

If you encounter CI failures:

1. Check the CI logs to identify the specific issues.
2. Run `python check_linting.py` locally to see if you can reproduce the issues.
3. Fix the issues manually or use the helper scripts.
4. Run pre-commit to verify your fixes before pushing again.
```text`

````

