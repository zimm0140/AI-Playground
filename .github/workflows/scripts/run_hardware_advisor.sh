#!/bin/bash
# Hardware Compatibility Advisor Runner
# This script runs both the hardware compatibility tester and advisor

set -e  # Exit on any error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create output directories
OUTPUT_DIR="ci_artifacts/hardware_compatibility"
RECOMMENDATIONS_DIR="$OUTPUT_DIR/recommendations"
mkdir -p "$OUTPUT_DIR" "$RECOMMENDATIONS_DIR"

echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}     Hardware Compatibility Advisor Suite      ${NC}"
echo -e "${BLUE}===============================================${NC}"

# Step 1: Run the compatibility tester
echo -e "\n${YELLOW}STEP 1: Running Hardware Compatibility Tester...${NC}"
if [ -f .github/workflows/scripts/test_hardware_compatibility.sh ]; then
    .github/workflows/scripts/test_hardware_compatibility.sh
else
    echo -e "${YELLOW}Running tester directly...${NC}"
    python .github/workflows/scripts/hardware_compatibility_tester.py --output-dir "$OUTPUT_DIR" --github-summary
fi

# Check if the compatibility data file exists
if [ ! -f "$OUTPUT_DIR/hardware_compatibility_data.json" ]; then
    echo -e "${RED}Error: Compatibility data file not found. Tester may have failed.${NC}"
    exit 1
fi

# Step 2: Run the compatibility advisor
echo -e "\n${YELLOW}STEP 2: Running Hardware Compatibility Advisor...${NC}"
python .github/workflows/scripts/hardware_compatibility_advisor.py \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$RECOMMENDATIONS_DIR" \
    --github-summary

# Step 3: Generate combined report
echo -e "\n${YELLOW}STEP 3: Generating Combined Report...${NC}"

# Create combined summary
cat > "$OUTPUT_DIR/combined_summary.md" << EOF
# Hardware Compatibility Suite Summary

This report combines results from both the Hardware Compatibility Tester and Advisor.

## Overview

The Hardware Compatibility Suite performs three main functions:
1. **Testing**: Analyzes package dependencies across hardware platforms
2. **Advising**: Generates recommendations for resolving compatibility issues
3. **Auto-fix**: Automatically applies recommended changes to standardize package versions

## Testing Results

$([ -f "$OUTPUT_DIR/hardware_compatibility_report.md" ] && cat "$OUTPUT_DIR/hardware_compatibility_report.md" | grep -A 10 "## Overview" | grep -v "## Overview" || echo "Testing results not available")

## Compatibility Matrix

$([ -f "$OUTPUT_DIR/hardware_compatibility_data.json" ] && python -c "
import json
with open('$OUTPUT_DIR/hardware_compatibility_data.json', 'r') as f:
    data = json.load(f)
matrix = data.get('compatibility_matrix', {})
if matrix:
    print('| Platform |' + ' | '.join(sorted(matrix.keys())) + ' |')
    print('|----------|' + '|'.join(['------' for _ in matrix.keys()]) + '|')
    for hw1 in sorted(matrix.keys()):
        row = f'| {hw1} |'
        for hw2 in sorted(matrix.keys()):
            if hw1 == hw2:
                row += ' — |'
            elif hw2 in matrix.get(hw1, {}):
                row += f' {matrix[hw1][hw2][\"score\"]} |'
            else:
                row += ' N/A |'
        print(row)
else:
    print('Compatibility matrix not available')
" || echo "Compatibility matrix not available")

## Advisor Recommendations

$([ -f "$RECOMMENDATIONS_DIR/resolution_plan.md" ] && cat "$RECOMMENDATIONS_DIR/resolution_plan.md" | grep -A 10 "## Recommendations Summary" | grep -v "## Recommendations Summary" || echo "Advisor recommendations not available")

## Next Steps

1. Review the detailed [compatibility report]($OUTPUT_DIR/hardware_compatibility_report.md)
2. Examine the complete [resolution plan]($RECOMMENDATIONS_DIR/resolution_plan.md)
3. Run the auto-fix tool to automatically apply recommended changes:
   \`\`\`
   .github/workflows/scripts/run_hardware_autofix.sh [--apply] [--all-priorities]
   \`\`\`
4. Re-run the hardware compatibility suite to verify improvements
EOF

echo -e "\n${GREEN}Hardware Compatibility Suite Complete!${NC}"
echo -e "Reports generated:"
echo -e "  - ${YELLOW}Compatibility Report:${NC} $OUTPUT_DIR/hardware_compatibility_report.md"
echo -e "  - ${YELLOW}Resolution Plan:${NC} $RECOMMENDATIONS_DIR/resolution_plan.md"
echo -e "  - ${YELLOW}Combined Summary:${NC} $OUTPUT_DIR/combined_summary.md"
echo -e "${BLUE}===============================================${NC}"

# Step 4: Offer to run the auto-fix tool
echo -e "\n${YELLOW}STEP 4: Auto-fix Option${NC}"
echo -e "Would you like to run the auto-fix tool to automatically apply recommended changes?"
echo -e "  1) Run auto-fix in simulation mode (dry run, no changes applied)"
echo -e "  2) Run auto-fix and apply changes (backups will be created)"
echo -e "  3) Skip auto-fix"
echo -e ""
read -p "Enter your choice (1-3): " choice

case $choice in
  1)
    echo -e "\n${YELLOW}Running auto-fix in simulation mode...${NC}"
    .github/workflows/scripts/run_hardware_autofix.sh
    ;;
  2)
    echo -e "\n${YELLOW}Running auto-fix and applying changes...${NC}"
    .github/workflows/scripts/run_hardware_autofix.sh --apply
    ;;
  3)
    echo -e "\n${YELLOW}Skipping auto-fix.${NC}"
    echo -e "You can run it later with: .github/workflows/scripts/run_hardware_autofix.sh"
    ;;
  *)
    echo -e "\n${RED}Invalid choice. Skipping auto-fix.${NC}"
    echo -e "You can run it later with: .github/workflows/scripts/run_hardware_autofix.sh"
    ;;
esac 