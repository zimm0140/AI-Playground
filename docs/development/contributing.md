# Contributing to AI-Playground

Thank you for your interest in contributing to AI-Playground! This guide will help you get started with the development process.

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue in the [issue tracker](https://github.com/intel/AI-Playground/issues) with the following information:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, hardware configuration)

### Suggesting Enhancements

We welcome suggestions for enhancements. Please create an issue with:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or mockups
- How the enhancement would benefit users

### Pull Requests

1. Fork the repository
1. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
1. Make your changes
1. Run tests to ensure your changes don't break existing functionality
1. Commit your changes (`git commit -m 'Add some amazing feature'`)
1. Push to your branch (`git push origin feature/amazing-feature`)
1. Open a Pull Request

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Setting Up Your Development Environment

1. Clone the repository:

   ```bash
   git clone <https://github.com/intel/AI-Playground.git>
   cd AI-Playground
   ```text

1. Set up a virtual environment:

   ```bash
   # Using venv

   python -m venv .venv

   # Activate on Windows

   .venv\Scripts\activate

   # Activate on macOS/Linux

   source .venv/bin/activate
   ```text

1. Install dependencies:

   ```bash
   # For automatic hardware detection and environment setup

   python setup_hardware_env.py --dev
   ```text

### Using uv for Dependency Management

We recommend using `uv` for faster dependency management:

```bash

# Install uv

curl -LsSf <https://astral.sh/uv/install.sh> | sh  # Unix/Linux/macOS

# or

powershell -c "irm <https://astral.sh/uv/install.ps1> | iex"  # Windows

# Install dependencies with uv

uv pip install -e ".[dev]"

```text

## Development Workflow

1. **Before Making Changes**:
   - Ensure you're working with the latest code: `git pull origin main`
   - Create a new branch for your feature or fix
   - Install development dependencies

1. **Making Changes**:
   - Write clean, well-documented code
   - Follow the existing code style
   - Add tests for new functionality

1. **Testing Your Changes**:
   - Run the existing test suite: `pytest`
   - Add tests for your new functionality
   - Ensure all tests pass

1. **Submitting Changes**:
   - Commit your changes with a clear message
   - Push your branch to your fork
   - Create a Pull Request against the main branch

## Coding Standards

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Write meaningful docstrings for functions and classes
- Use descriptive variable and function names
- Keep functions focused on a single responsibility

## Testing

- Add tests for new functionality
- Run the full test suite before submitting a PR
- Use test fixtures when appropriate
- Mock external dependencies in tests

## Documentation

- Update documentation to reflect your changes
- Document new features or changed behavior
- Use clear, concise language
- Include examples where appropriate

## Review Process

After submitting a PR, maintainers will review your changes. They may suggest improvements or changes. Once approved, your changes will be merged into the main branch.

## Hardware-Specific Contributions

When contributing code that deals with specific hardware:

1. Clearly document hardware requirements
1. Add defensive checks for hardware availability
1. Provide fallback implementations when possible
1. Test on multiple hardware configurations if possible

---
**Next**: [Code Quality Standards](code-quality.md) | **See also**: [Testing Guide](testing.md)
