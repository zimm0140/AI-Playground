.PHONY: setup test lint format clean check audit sync help

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
UV := uv

# Development setup
setup: ## Set up development environment
	$(UV) venv
	$(UV) pip install -e ".[dev]"
	pre-commit install

# Testing
test: ## Run tests
	$(UV) run pytest

# Linting
lint: ## Run linters
	$(UV) run ruff check .
	$(UV) run mypy .

# Formatting
format: ## Format code
	$(UV) run ruff format .

# Cleaning
clean: ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ 
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Dependency checking
check: ## Check dependencies for updates
	$(UV) pip check

# Security audit
audit: ## Run security audit
	$(UV) pip audit

# Sync dependencies
sync: ## Sync dependencies from requirements.txt
	$(UV) pip sync requirements.txt

# Help message
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) 