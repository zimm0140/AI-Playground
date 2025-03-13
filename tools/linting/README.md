# Linting Tools

This directory contains scripts for linting and fixing common code quality issues in the AI Playground project.

## Contents

- `fix_lint_issues.py`: Enhanced script to fix common linting issues with progress tracking
- `track_technical_debt.py`: Script to track and analyze technical debt over time
- `check_linting.py`: Script to check if all linting issues have been fixed
- `fix_markdown_lint.py`: Script to fix common markdown linting issues
- `fix_markdown_advanced.py`: Advanced script for fixing markdown linting issues
- `fix_unused_variables.py`: Script to fix unused variable warnings

## Technical Debt Management

We use a systematic approach to manage technical debt:

### Priority Levels

1. **High Priority**
   - Unused imports (F401)
   - Trailing whitespace (W291)
   - Undefined names (F821)
   - Basic naming conventions (N801-N803)

2. **Medium Priority**
   - Complex structures (C901)
   - Line length issues (E501)
   - Import placement (E402)
   - Return statement consistency (RET503-RET505)

3. **Low Priority**
   - Security issues (S*)
   - Path handling (PTH*)
   - Code structure (SIM*)

### Usage

#### Fix Linting Issues

```bash
# Fix all issues in a directory
python tools/linting/fix_lint_issues.py [directory]

# Fix specific rules
python tools/linting/fix_lint_issues.py [directory] --rules F401 W291
```

#### Track Technical Debt

```bash
# Generate technical debt report
python tools/linting/track_technical_debt.py
```

The script will:

- Analyze current technical debt
- Track progress over time
- Generate recommendations
- Save historical data

#### Check Linting Status

```bash
python tools/linting/check_linting.py
```

### Reports

The tools generate several reports:

- `technical_debt_report.md`: Current state and trends
- `lint_report.md`: Details of recent fixes
- `technical_debt_history.json`: Historical data

### Configuration

Linting rules are configured in:

- `pyproject.toml`: Python linting configuration
- `.markdownlint.yaml`: Markdown linting rules

## Best Practices

1. **Regular Monitoring**
   - Run `track_technical_debt.py` weekly
   - Review trends and adjust priorities
   - Focus on high-priority issues first

2. **Incremental Fixes**
   - Fix issues in small batches
   - Test thoroughly after each fix
   - Document any manual fixes needed

3. **Code Review**
   - Check linting before review
   - Prevent new technical debt
   - Use automated fixes when possible

## Contributing

When adding new code:

1. Run linting checks locally
2. Fix high-priority issues immediately
3. Document any technical debt added
4. Update debt tracking if needed

## Usage

### Markdown Linting

To fix markdown linting issues:

```text`text

python tools/linting/fix_markdown_lint.py [directory_or_file]

```text

For advanced markdown fixes:

```text

python tools/fix_markdown_advanced.py [directory_or_file]

```text

### Python Linting

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

```text`
