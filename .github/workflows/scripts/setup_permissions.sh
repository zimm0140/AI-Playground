#!/bin/bash
# Set executable permissions for workflow scripts

set -e  # Exit on error

SCRIPTS_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(cd "$SCRIPTS_DIR/../.." && pwd)"

echo "Setting executable permissions for workflow scripts..."

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

echo "Done setting executable permissions." 