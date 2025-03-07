#!/bin/bash
# Hardware Compatibility Testing Script
# This script runs the hardware compatibility tests and analyzes the results

set -e  # Exit on any error

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directory
OUTPUT_DIR="ci_artifacts/hardware_compatibility"
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Starting Hardware Compatibility Tests${NC}"
echo "======================================"

# Generate timestamp for reports
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Timestamp: $TIMESTAMP"

# Run the Python hardware compatibility tester
echo -e "\n${YELLOW}Running hardware compatibility analysis...${NC}"
python .github/workflows/scripts/hardware_compatibility_tester.py --output-dir "$OUTPUT_DIR" --github-summary

# Check the exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}Hardware compatibility analysis completed successfully!${NC}"
else
    echo -e "\n${YELLOW}Hardware compatibility analysis completed with warnings.${NC}"
fi

# Create a visual compatibility matrix using ASCII art
if [ -f "$OUTPUT_DIR/hardware_compatibility_data.json" ]; then
    echo -e "\n${GREEN}Generating visual compatibility matrix...${NC}"
    
    # Use Python to generate a simple ASCII compatibility matrix
    python - <<EOF
import json
import os

# Load the compatibility data
data_file = "$OUTPUT_DIR/hardware_compatibility_data.json"
with open(data_file, 'r') as f:
    data = json.load(f)

matrix = data.get('compatibility_matrix', {})
if not matrix:
    print("No compatibility data found.")
    exit(0)

# Get all hardware platforms
platforms = sorted(matrix.keys())

# Define color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# Generate ASCII art matrix
print("\nHardware Compatibility Matrix:")
print("=" * 80)

# Header row
header = "Platform".ljust(15)
for hw in platforms:
    header += hw.ljust(15)
print(header)
print("-" * 80)

# Data rows
for hw1 in platforms:
    row = hw1.ljust(15)
    for hw2 in platforms:
        if hw1 == hw2:
            row += "---".ljust(15)
        elif hw2 in matrix.get(hw1, {}):
            score = matrix[hw1][hw2]['score']
            if score == "100% Compatible":
                row += f"{GREEN}Compatible{NC}".ljust(25)
            elif score == "High":
                row += f"{YELLOW}High{NC}".ljust(19)
            elif score == "Medium":
                row += f"{YELLOW}Medium{NC}".ljust(21)
            elif score == "Low":
                row += f"{RED}Low{NC}".ljust(18)
            else:
                row += score.ljust(15)
        else:
            row += "N/A".ljust(15)
    print(row)

print("=" * 80)

# Summary of conflicts
conflicts = data.get('conflicts', [])
high_priority = [c for c in conflicts if c['severity'] == 'high']

if high_priority:
    print(f"\n{RED}High Priority Conflicts:{NC}")
    for conflict in high_priority:
        package = conflict['package']
        platforms = ", ".join(sorted(conflict['versions'].keys()))
        print(f"- {package}: Different versions on {platforms}")

medium_priority = [c for c in conflicts if c['severity'] == 'medium']
if medium_priority:
    print(f"\n{YELLOW}Medium Priority Conflicts:{NC}")
    for conflict in medium_priority:
        package = conflict['package']
        platforms = ", ".join(sorted(conflict['versions'].keys()))
        print(f"- {package}: Different versions on {platforms}")

if not conflicts:
    print(f"\n{GREEN}No compatibility issues detected!{NC}")

print("\nSee detailed report at: $OUTPUT_DIR/hardware_compatibility_report.md")
EOF
fi

# Run additional hardware-specific package tests if needed
if [ -f "service/requirements-acm.txt" ]; then
    echo -e "\n${YELLOW}Running ACM-specific package tests...${NC}"
    # Only run the pip check, don't actually install the packages in CI
    pip check service/requirements-acm.txt || echo "ACM requirements need attention"
fi

# Create a summary file for CI
cat > "$OUTPUT_DIR/hardware_compatibility_summary.txt" <<EOF
Hardware Compatibility Test Summary
==================================
Timestamp: $TIMESTAMP
Output Directory: $OUTPUT_DIR

See the detailed report for more information about any compatibility issues.

Next Steps:
1. Review the compatibility matrix in the detailed report
2. Address any high priority conflicts
3. Consider standardizing package versions across hardware platforms
4. Re-run the compatibility tests after making changes
EOF

echo -e "\n${GREEN}Hardware compatibility testing complete!${NC}"
echo -e "Report generated at: ${YELLOW}$OUTPUT_DIR/hardware_compatibility_report.md${NC}"
echo "====================================" 