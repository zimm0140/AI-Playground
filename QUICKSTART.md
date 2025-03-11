# AI Playground Quick Start Guide

This guide will help you get started with AI Playground using uv, an extremely fast Python package and project manager.

## Prerequisites

- Python 3.10 or newer (3.13 recommended)
- Git

## Installation

### 1. Install uv

On macOS and Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the Repository

```bash
git clone https://github.com/zimm0140/AI-Playground.git
cd AI-Playground
```

### 3. Set Up Development Environment

```bash
# Create a virtual environment with uv
uv venv

# Activate the virtual environment (depends on your shell)
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies for development
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Running Tests

With virtual environment activated:
```bash
# Run all tests
pytest

# Or using uv directly (no need to activate environment)
uv run pytest
```

## Development Workflow

1. **Edit Code**: Make your changes to the codebase.

2. **Run Tests**: Verify your changes work:
   ```bash
   uv run pytest
   ```

3. **Check Code Quality**: Run pre-commit hooks:
   ```bash
   pre-commit run --all-files
   ```

4. **Fix Issues**: Address any linting or type checking errors.

5. **Commit**: Commit your changes:
   ```bash
   git add .
   git commit -m "Your meaningful commit message"
   ```

## Common Tasks

### Adding New Dependencies

To add a new dependency:
```bash
# Regular dependency
uv pip install some-package
# Add to pyproject.toml and setup.py
python .github/sync_dependencies.py
```

### Running Specific Tests

```bash
# Run specific test file
uv run pytest tests/test_specific.py

# Run tests with specific name pattern
uv run pytest -k "test_pattern"
```

### Running Linters Individually

```bash
# Run Ruff linter
uv run ruff check .

# Run mypy type checker
uv run mypy .
```

## Troubleshooting

If you encounter dependency issues:
```bash
# Re-sync the environment
uv pip sync requirements.txt

# For development packages
uv pip install -e ".[dev]"
```

For more detailed information, see the full [documentation](README.md) and [migration guide](MIGRATION.md).

## Optional: Shell Completion

Set up shell completion for uv to make your development experience smoother:

### Bash

```bash
uv completion bash > ~/.uv-completion.bash
echo 'source ~/.uv-completion.bash' >> ~/.bashrc
```

### Zsh

```bash
uv completion zsh > ~/.zsh/_uv
echo 'fpath=(~/.zsh $fpath)' >> ~/.zshrc
echo 'autoload -Uz compinit && compinit' >> ~/.zshrc
```

### Fish

```bash
uv completion fish > ~/.config/fish/completions/uv.fish
```

### PowerShell

```powershell
uv completion powershell | Out-File -Encoding utf8 -FilePath (Join-Path $PROFILE.CurrentUserAllHosts "uv.ps1")
``` 