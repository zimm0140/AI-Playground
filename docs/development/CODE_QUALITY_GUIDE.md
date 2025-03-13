# Code Quality Guide

This document outlines the code quality standards, tools, and practices implemented in the AI Playground project.

## Standards and Tools

### Python Code Quality

- *_Linting__: We use `flake8`, `pylint`, and `black` for Python code linting and formatting.
- __Type Checking__: `mypy` is used for static type checking.
- __Import Sorting__: `isort` ensures imports are organized consistently.
- __Code Complexity__: We monitor cyclomatic complexity with `flake8-complexity`.

### Markdown Documentation

- __Linting__: `markdownlint` ensures consistent and readable documentation.
- __Formatting Rules__: Line length, heading spacing, list formatting, and more are enforced.
- __Automation__: Custom scripts (`fix_markdown_lint.py` and `fix_readme.py`) automate fixing common issues.

### CI/CD Integration

- __GitHub Actions__: Automated quality checks run on every push and pull request.
- __Pre-commit Hooks__: Local checks run before committing to prevent introducing issues.
- __Auto-fixing__: Some issues are automatically fixed during CI runs.

## Best Practices

### Python

1. Use type hints for all function parameters and return values.
2. Follow PEP 8 style guidelines.
3. Keep functions small and focused (preferably under 50 lines).
4. Write docstrings for all modules, classes, and functions.
5. Use meaningful variable and function names.

### Documentation

1. Keep documentation up-to-date with code changes.
2. Use consistent formatting in markdown files.
3. Document complex functionality with examples.
4. Include installation and setup instructions for developers.

## Pre-commit Hooks

Pre-commit hooks are configured to run the following checks:

- `black`: Format Python code
- `isort`: Sort Python imports
- `flake8`: Lint Python code
- `mypy`: Check Python types
- `markdownlint`: Lint markdown documents

To set up pre-commit hooks:

\`\`\`text\`bash
pip install pre-commit
pre-commit install

```text`text

## Troubleshooting Common Issues

### Markdown Linting

Common issues and solutions:

1. __Line Length (MD013)__: Break long lines or use the `<!-- markdownlint-disable MD013 -->` comment to disable for specific sections.
2. __Multiple Top-level Headings (MD025)__: Use only one H1 (#) heading per document.
3. __List Formatting__: Ensure lists have blank lines before and after, and use consistent formatting (- for unordered, 1. for ordered).

### Python Linting

1. __Import Issues__: Update `pyrightconfig.json` or `mypy.ini` to handle special imports.
2. __Line Length__: Use line breaks or, in rare cases, `# noqa: E501` to ignore specific lines.

1. __Type Checking_*: Use `# type: ignore` for legitimate cases where types cannot be properly resolved.

## Contact

For questions about code quality standards or help with resolving issues, please open an issue on GitHub or contact the project maintainers.

```text`

```text`
