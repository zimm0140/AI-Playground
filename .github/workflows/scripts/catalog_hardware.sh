#!/bin/bash
# Hardware Cataloging Script
# This script documents all hardware-specific requirements files for better visibility

echo "Cataloging hardware-specific requirements files"
mkdir -p ci_artifacts

# Find all hardware-specific requirements files
find . -name "requirements-*.txt" -o -name "OpenVINO/requirements.txt" -o -name "LlamaCPP/requirements.txt" > ci_artifacts/hw_req_files.txt

# Analyze requirements for documentation
echo "# Hardware Acceleration Support Matrix" > ci_artifacts/hw_support.md
echo "| Hardware | Requirements File | Key Packages |" >> ci_artifacts/hw_support.md
echo "|----------|------------------|-------------|" >> ci_artifacts/hw_support.md

# Process each file
while IFS= read -r file; do
  # Extract hardware identifier from filename
  hw_id=$(basename "$file" | sed -E 's/requirements-?(.*)\.txt/\1/')
  [ -z "$hw_id" ] && hw_id=$(dirname "$file" | xargs basename)
  
  # Get key packages (focus on torch, intel extensions, etc.)
  key_pkgs=$(grep -E "torch|intel|ipex|bigdl|onednn" "$file" | grep -v "^#" | sed -e 's/^[[:space:]]*//' | tr '\n' ', ' | sed 's/,$//')
  
  # Add to documentation
  echo "| $hw_id | $file | $key_pkgs |" >> ci_artifacts/hw_support.md
done < ci_artifacts/hw_req_files.txt

echo "Hardware support documentation created at ci_artifacts/hw_support.md" 