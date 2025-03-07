#!/bin/bash
# Script to update all GitHub Actions to their latest versions
# This helps keep GitHub Actions up to date and avoid deprecation warnings

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}     GitHub Actions Update Script      ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Make sure all scripts are executable
chmod +x .github/workflows/scripts/update_artifact_version.sh
chmod +x .github/workflows/scripts/update_checkout_version.sh
chmod +x .github/workflows/scripts/update_cache_version.sh
chmod +x .github/workflows/scripts/update_github_script_version.sh

# Run all update scripts
echo -e "\n${YELLOW}Updating upload-artifact action from v3 to v4...${NC}"
bash .github/workflows/scripts/update_artifact_version.sh

echo -e "\n${YELLOW}Updating checkout action from v3 to v4...${NC}"
bash .github/workflows/scripts/update_checkout_version.sh

echo -e "\n${YELLOW}Updating cache action from v3 to v4...${NC}"
bash .github/workflows/scripts/update_cache_version.sh

echo -e "\n${YELLOW}Updating github-script action from v6 to v7...${NC}"
bash .github/workflows/scripts/update_github_script_version.sh

echo -e "\n${GREEN}All GitHub Actions have been updated to their latest versions!${NC}"
echo -e "${BLUE}===============================================${NC}" 