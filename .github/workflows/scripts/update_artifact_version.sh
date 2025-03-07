#!/bin/bash
# Script to update all instances of actions/upload-artifact from v3 to v4
# This helps fix the GitHub Actions deprecation warning

set -e  # Exit on error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Updating GitHub Action references from upload-artifact@v3 to v4${NC}"

# Find all workflow files
WORKFLOW_FILES=$(find .github/workflows -name "*.yml" -o -name "*.yaml")

# Counter for files updated
COUNT=0

for file in $WORKFLOW_FILES; do
    # Look for the deprecated action in the file
    if grep -q "actions/upload-artifact@v3" "$file"; then
        echo -e "${YELLOW}Updating file:${NC} $file"
        
        # Replace v3 with v4
        sed -i "s/actions\/upload-artifact@v3/actions\/upload-artifact@v4/g" "$file"
        
        # Confirm the replacement was made
        if grep -q "actions/upload-artifact@v4" "$file"; then
            echo -e "  ${GREEN}✓${NC} Successfully updated"
            ((COUNT++))
        else
            echo -e "  ${RED}✗${NC} Failed to update"
        fi
    fi
done

echo -e "\n${GREEN}Update complete!${NC}"
echo -e "Updated $COUNT workflow files." 