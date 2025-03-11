#!/bin/bash
# Convenience wrapper script for uvfast.py on Linux/macOS

# Script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Check if uvfast.py exists
if [ ! -f "uvfast.py" ]; then
    echo "Error: uvfast.py not found in $PROJECT_ROOT"
    exit 1
fi

# Pass all arguments to uvfast.py
python uvfast.py "$@" 