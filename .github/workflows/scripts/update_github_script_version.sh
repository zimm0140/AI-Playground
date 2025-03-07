#!/bin/bash
# Script to update all instances of actions/github-script from v6 to v7
# This helps keep GitHub Actions up to date

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Updating GitHub Action references from github-script@v6 to v7${NC}"

# Find all workflow files
WORKFLOW_FILES=$(find .github/workflows -name "*.yml" -o -name "*.yaml")

# Counter for files updated
COUNT=0

for file in $WORKFLOW_FILES; do
    # Look for the outdated action in the file
    if grep -q "actions/github-script@v6" "$file"; then
        echo -e "${YELLOW}Updating file:${NC} $file"
        
        # Replace v6 with v7
        sed -i "s/actions\/github-script@v6/actions\/github-script@v7/g" "$file"
        
        # Confirm the replacement was made
        if grep -q "actions/github-script@v7" "$file"; then
            echo -e "  ${GREEN}✓${NC} Successfully updated"
            ((COUNT++))
        else
            echo -e "  ${RED}✗${NC} Failed to update"
        fi
    fi
done

echo -e "\n${GREEN}Update complete!${NC}"
echo -e "Updated $COUNT workflow files." 