#!/bin/bash
#
# Setup Git hooks for the project
# Run this script from the project root to set up pre-commit hooks
#

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up Git hooks for AI-Playground project...${NC}"

# Detect if we're running on Windows
case "$(uname -s)" in
    CYGWIN*|MINGW*|MSYS*)
        # Windows detected - run the PowerShell script if available
        echo -e "${YELLOW}Windows detected, using PowerShell setup if available...${NC}"
        if command -v pwsh >/dev/null 2>&1; then
            pwsh -File ".github/setup-hooks.ps1"
            exit $?
        elif command -v powershell >/dev/null 2>&1; then
            powershell -File ".github/setup-hooks.ps1"
            exit $?
        else
            echo -e "${YELLOW}PowerShell not found, falling back to bash setup...${NC}"
        fi
        ;;
esac

# Ensure the hooks directory exists
mkdir -p .git/hooks

# Copy the pre-commit hook and make it executable
cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Copy the PowerShell hook for Windows users who might use Git Bash
cp .github/hooks/pre-commit.ps1 .git/hooks/pre-commit.ps1

# Install any other hooks here as needed

echo -e "${GREEN}Git hooks successfully installed!${NC}"
echo -e "Pre-commit hooks will now run automatically when you commit."
echo -e "To bypass hooks temporarily, use ${YELLOW}git commit --no-verify${NC}" 