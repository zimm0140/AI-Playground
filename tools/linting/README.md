# Linting Tools

This directory contains scripts for linting and fixing common code quality issues in the AI Playground project.

## Contents

- `fix_markdown_lint.py`: Script to fix common markdown linting issues
- `fix_markdown_advanced.py`: Advanced script for fixing markdown linting issues
- `check_linting.py`: Script to check if all linting issues have been fixed
- `fix_lint_issues.py`: Script to fix common linting issues in Python code
- `fix_unused_variables.py`: Script to fix unused variable warnings

## Usage

### Markdown Linting

To fix markdown linting issues:

````text

python tools/linting/fix_markdown_lint.py [directory_or_file]

```text

For advanced markdown fixes:

```text

python tools/fix_markdown_advanced.py [directory_or_file]

```text

### Python Linting

To check for linting issues:

```text

python tools/linting/check_linting.py

```text

To fix common linting issues:

```text

python tools/linting/fix_lint_issues.py [file]

```text

To fix unused variable warnings:

```text

python tools/linting/fix_unused_variables.py [file]

```text

## Configuration

These tools use configuration files from the `config` directory:

- `.markdownlint.yaml`: Configuration for markdown linting
- `.prettierrc`, `.prettierrc.json`: Configuration for Prettier code formatter
- `mypy.ini`: Configuration for mypy type checking
- `pyrightconfig.json`: Configuration for Pyright type checking
- `.pre-commit-config.yaml`: Configuration for pre-commit hooks

Note: Copies of these configuration files are also available in the project root directory for compatibility with tools that expect them there.

````
