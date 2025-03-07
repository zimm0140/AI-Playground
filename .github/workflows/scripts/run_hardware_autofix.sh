#!/bin/bash
# Hardware Compatibility Auto-fix Runner
# This script runs the hardware compatibility auto-fix tool

set -e  # Exit on any error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
DRY_RUN=true
HIGH_PRIORITY_ONLY=true

# Process command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --apply)
      DRY_RUN=false
      shift
      ;;
    --all-priorities)
      HIGH_PRIORITY_ONLY=false
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --apply           Apply changes (default: dry run)"
      echo "  --all-priorities  Apply both high and medium priority recommendations (default: high only)"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create output directories
OUTPUT_DIR="ci_artifacts/hardware_compatibility"
RECOMMENDATIONS_DIR="$OUTPUT_DIR/recommendations"
AUTOFIX_DIR="$OUTPUT_DIR/autofix"
mkdir -p "$OUTPUT_DIR" "$RECOMMENDATIONS_DIR" "$AUTOFIX_DIR"

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}     Hardware Compatibility Auto-fix Tool      ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Step 1: Ensure we have the compatibility data
echo -e "\n${YELLOW}STEP 1: Checking for hardware compatibility data...${NC}"
if [ ! -f "$OUTPUT_DIR/hardware_compatibility_data.json" ]; then
    echo -e "${YELLOW}Compatibility data not found, running compatibility tester...${NC}"
    
    if [ -f .github/workflows/scripts/run_hardware_advisor.sh ]; then
        .github/workflows/scripts/run_hardware_advisor.sh
    else
        echo -e "${RED}Error: run_hardware_advisor.sh not found. Please run hardware compatibility tests first.${NC}"
        exit 1
    fi
fi

# Step 2: Ensure we have recommendations
echo -e "\n${YELLOW}STEP 2: Checking for advisor recommendations...${NC}"
if [ ! -f "$RECOMMENDATIONS_DIR/resolution_plan.md" ]; then
    echo -e "${RED}Error: Resolution plan not found. Please run hardware compatibility advisor first.${NC}"
    exit 1
fi

# Step 3: Run the auto-fix tool
echo -e "\n${YELLOW}STEP 3: Running Auto-fix Tool...${NC}"

# Build the command with appropriate options
CMD="python .github/workflows/scripts/hardware_compatibility_autofix.py --github-summary"

# Add options based on user preference
if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
    echo -e "${YELLOW}Running in DRY RUN mode. No actual changes will be made.${NC}"
else
    echo -e "${RED}WARNING: Running in APPLY mode. Changes will be applied!${NC}"
    echo -e "${YELLOW}Backups will be created with .bak extension.${NC}"
fi

if [ "$HIGH_PRIORITY_ONLY" = true ]; then
    CMD="$CMD --high-priority-only"
    echo -e "${YELLOW}Applying HIGH PRIORITY recommendations only.${NC}"
else
    CMD="$CMD --all-priorities"
    echo -e "${YELLOW}Applying ALL recommendations (high and medium priority).${NC}"
fi

# Run the command
echo -e "${BLUE}Executing: $CMD${NC}"
eval $CMD

# Check exit code
if [ $? -eq 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        echo -e "\n${GREEN}Auto-fix simulation completed successfully!${NC}"
        echo -e "${YELLOW}To apply the changes, run this script with --apply${NC}"
    else
        echo -e "\n${GREEN}Auto-fix applied successfully!${NC}"
        echo -e "${YELLOW}Backups of modified files were created with .bak extension.${NC}"
    fi
else
    echo -e "\n${RED}Auto-fix ${DRY_RUN:+simulation }failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}Hardware Compatibility Auto-fix complete!${NC}"
echo -e "Reports generated:"
echo -e "  - ${YELLOW}Auto-fix Report:${NC} $AUTOFIX_DIR/autofix_report.md"
echo -e "${BLUE}===============================================${NC}"

# Provide next steps
if [ "$DRY_RUN" = true ]; then
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "1. Review the auto-fix report"
    echo -e "2. Run with --apply to apply the changes"
    echo -e "3. Run the hardware compatibility tester again to verify conflicts are resolved"
else
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "1. Verify that conflicts have been resolved by running the hardware compatibility tester"
    echo -e "2. Commit the changes to the requirements files"
    echo -e "3. If needed, restore from backups (*.bak files)"
fi 