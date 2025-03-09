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

# Ensure the hooks directory exists
mkdir -p .git/hooks

# Copy the pre-commit hook
cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install any other hooks here as needed

echo -e "${GREEN}Git hooks successfully installed!${NC}"
echo -e "Pre-commit hooks will now run automatically when you commit."
echo -e "To bypass hooks temporarily, use ${YELLOW}git commit --no-verify${NC}" 