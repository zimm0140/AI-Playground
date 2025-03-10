# Code Quality Guide

This document outlines the code quality standards, tools, and practices implemented in the AI Playground project.

## Standards and Tools

### Python Code Quality

- **Linting**: We use `flake8`, `pylint`, and `black` for Python code linting and formatting.
- **Type Checking**: `mypy` is used for static type checking.
- **Import Sorting**: `isort` ensures imports are organized consistently.
- **Code Complexity**: We monitor cyclomatic complexity with `flake8-complexity`.

### Markdown Documentation

- **Linting**: `markdownlint` ensures consistent and readable documentation.
- **Formatting Rules**: Line length, heading spacing, list formatting, and more are enforced.
- **Automation**: Custom scripts (`fix_markdown_lint.py` and `fix_readme.py`) automate fixing common issues.

### CI/CD Integration

- **GitHub Actions**: Automated quality checks run on every push and pull request.
- **Pre-commit Hooks**: Local checks run before committing to prevent introducing issues.
- **Auto-fixing**: Some issues are automatically fixed during CI runs.

## Best Practices

### Python

1. Use type hints for all function parameters and return values.
1. Follow PEP 8 style guidelines.
1. Keep functions small and focused (preferably under 50 lines).
1. Write docstrings for all modules, classes, and functions.
1. Use meaningful variable and function names.

### Documentation

1. Keep documentation up-to-date with code changes.
1. Use consistent formatting in markdown files.
1. Document complex functionality with examples.
1. Include installation and setup instructions for developers.

## Pre-commit Hooks

Pre-commit hooks are configured to run the following checks:

- `black`: Format Python code
- `isort`: Sort Python imports
- `flake8`: Lint Python code
- `mypy`: Check Python types
- `markdownlint`: Lint markdown documents

To set up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install

```text
## Troubleshooting Common Issues

### Markdown Linting

Common issues and solutions:

1. **Line Length (MD013)**: Break long lines or use the `<!-- markdownlint-disable MD013 -->` comment to disable for specific sections.
1. **Multiple Top-level Headings (MD025)**: Use only one H1 (#) heading per document.
1. **List Formatting**: Ensure lists have blank lines before and after, and use consistent formatting (- for unordered, 1. for ordered).

### Python Linting

1. **Import Issues**: Update `pyrightconfig.json` or `mypy.ini` to handle special imports.
1. **Line Length**: Use line breaks or, in rare cases, `# noqa: E501` to ignore specific lines.

1. **Type Checking**: Use `# type: ignore` for legitimate cases where types cannot be properly resolved.

## Contact

For questions about code quality standards or help with resolving issues, please open an issue on GitHub or contact the project maintainers.
