.PHONY: lint format test ci-check install-hooks clean build help

# Default target
all: lint format test

# Install dev dependencies
setup:
	pip install -r requirements-dev.txt
	pre-commit install

# Run linting checks
lint:
	ruff check .

# Fix linting issues automatically
lint-fix:
	ruff check --fix .

# Format code
format:
	ruff format .

# Check formatting without changing files
format-check:
	ruff format --check .

# Run tests
test:
	pytest

# Run all CI checks locally
ci-check: lint format-check test

# Install pre-commit hooks
install-hooks:
	pre-commit install

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

# Build package
build:
	python -m build

# Show help
help:
	@echo "Available commands:"
	@echo "  make setup         - Install development dependencies"
	@echo "  make lint          - Run linting checks"
	@echo "  make lint-fix      - Fix linting issues automatically"
	@echo "  make format        - Format code"
	@echo "  make format-check  - Check formatting without changing files"
	@echo "  make test          - Run tests"
	@echo "  make ci-check      - Run all CI checks locally"
	@echo "  make install-hooks - Install pre-commit hooks"
	@echo "  make clean         - Clean up cache files"
	@echo "  make build         - Build package"
	@echo "  make help          - Show this help message" 