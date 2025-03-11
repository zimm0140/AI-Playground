#!/bin/bash
# Helper script to run commands with uv

# Set error handling
set -e

# Ensure uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Function to display help
function show_help {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  run <script.py>        Run a Python script in an isolated environment"
    echo "  test                   Run pytest"
    echo "  lint                   Run linters (ruff, mypy)"
    echo "  format                 Format code with ruff"
    echo "  sync                   Sync dependencies from lockfiles"
    echo "  audit                  Run security audit"
    echo "  tool <tool_name>       Install and run a tool"
    echo "  clean                  Clean temporary files"
    echo "  help                   Show this help message"
    echo ""
    exit 0
}

# Check if no arguments were passed
if [ $# -eq 0 ]; then
    show_help
fi

# Process commands
case "$1" in
    run)
        if [ -z "$2" ]; then
            echo "Error: No script specified."
            echo "Usage: $0 run <script.py>"
            exit 1
        fi
        echo "Running $2 with uv..."
        uv run "$2"
        ;;
    test)
        shift
        echo "Running tests with uv..."
        uv run pytest "$@"
        ;;
    lint)
        echo "Running linters with uv..."
        uv run ruff check .
        uv run mypy .
        ;;
    format)
        echo "Formatting code with uv..."
        uv run ruff format .
        ;;
    sync)
        echo "Syncing dependencies with uv..."
        uv pip sync requirements.lock requirements-dev.lock
        ;;
    audit)
        echo "Running security audit with uv..."
        uv pip audit
        ;;
    tool)
        if [ -z "$2" ]; then
            echo "Error: No tool specified."
            echo "Usage: $0 tool <tool_name>"
            exit 1
        fi
        shift
        echo "Running tool with uv: $*"
        uv tool run "$@"
        ;;
    clean)
        echo "Cleaning temporary files..."
        rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ 
        find . -type d -name __pycache__ -exec rm -rf {} +
        find . -type d -name "*.egg-info" -exec rm -rf {} +
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        ;;
esac 