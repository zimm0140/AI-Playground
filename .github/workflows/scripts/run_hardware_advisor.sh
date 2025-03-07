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

The Hardware Compatibility Suite performs two main functions:
1. **Testing**: Analyzes package dependencies across hardware platforms
2. **Advising**: Generates recommendations for resolving compatibility issues

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
3. Apply suggested changes to standardize package versions across platforms
4. Re-run the hardware compatibility suite to verify improvements
EOF

echo -e "\n${GREEN}Hardware Compatibility Suite Complete!${NC}"
echo -e "Reports generated:"
echo -e "  - ${YELLOW}Compatibility Report:${NC} $OUTPUT_DIR/hardware_compatibility_report.md"
echo -e "  - ${YELLOW}Resolution Plan:${NC} $RECOMMENDATIONS_DIR/resolution_plan.md"
echo -e "  - ${YELLOW}Combined Summary:${NC} $OUTPUT_DIR/combined_summary.md"
echo -e "${BLUE}===============================================${NC}" 