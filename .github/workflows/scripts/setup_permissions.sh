#!/bin/bash
# Set executable permissions for workflow scripts and set up development environment

set -e  # Exit on error

SCRIPTS_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"

echo "Setting up development environment..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv (recommended)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ Installed uv"
    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✓ uv already installed"
fi

# Ensure tomli and tomli_w are installed for dependency sync
echo "Installing required packages for dependency sync..."
uv pip install tomli tomli_w
echo "✓ Installed tomli and tomli_w"

# Install pre-commit if not installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    uv pip install pre-commit
    echo "✓ Installed pre-commit"
else
    echo "✓ pre-commit already installed"
fi

# Make fix_markdown_lint.py executable
chmod +x "$SCRIPTS_DIR/fix_markdown_lint.py"
echo "✓ Set executable permissions for fix_markdown_lint.py"

# Make sync_dependencies.py executable
chmod +x "$REPO_ROOT/.github/sync_dependencies.py"
echo "✓ Set executable permissions for sync_dependencies.py"

# Make any other scripts in .github/workflows/scripts executable
find "$SCRIPTS_DIR" -name "*.py" -type f -exec chmod +x {} \;
find "$SCRIPTS_DIR" -name "*.sh" -type f -exec chmod +x {} \;
echo "✓ Set executable permissions for all scripts in $SCRIPTS_DIR"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
cd "$REPO_ROOT"
pre-commit install
echo "✓ Installed pre-commit hooks"

# Sync dependencies
echo "Syncing dependencies..."
python "$REPO_ROOT/.github/sync_dependencies.py"
echo "✓ Synced dependencies"

# Run markdown fixer
echo "Fixing markdown issues..."
python "$SCRIPTS_DIR/fix_markdown_lint.py" "$REPO_ROOT"
echo "✓ Fixed markdown issues"

echo "Done setting up development environment."
echo -e "\nYou can now run tests with:"
echo "  uv run pytest"
echo "Or using the traditional method:"
echo "  pytest"
echo "To run pre-commit checks: pre-commit run --all-files" 