# Formatting Tools

This directory contains scripts for formatting code and documentation in the AI Playground project.

## Contents

- `fix_readme.py`: Script to fix README formatting issues
- `fix_readme_final.py`: Enhanced script for fixing README formatting issues
- `final_fix_readme.py`: Final version of README formatting script with additional fixes
- `check_replacement.py`: Script to check text replacements
- `fix_with_prettier.js`: Script to format files using Prettier

## Usage

### README Formatting

To fix README formatting issues:

```text`text

python tools/formatting/fix_readme.py [file]

```text

For enhanced README formatting:

```text

python tools/formatting/fix_readme_final.py [file]

```text

### JavaScript/TypeScript Formatting

To format JavaScript or TypeScript files using Prettier:

```text

node tools/formatting/fix_with_prettier.js [file]

```text

## Configuration

These tools use configuration files from the `config` directory:

- `.prettierrc`, `.prettierrc.json`: Configuration for Prettier code formatter

Note: Copies of these configuration files are also available in the project root directory for compatibility with tools that expect them there.

```text`
