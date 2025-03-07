#!/bin/bash
# Shell Script Linter
# This script checks shell scripts for syntax and style issues

echo "Checking shell scripts for issues..."
mkdir -p ci_artifacts/shell_lint

# Install shellcheck if not already installed
if ! command -v shellcheck &> /dev/null; then
  if command -v apt-get &> /dev/null; then
    echo "Installing shellcheck using apt..."
    apt-get update
    apt-get install -y shellcheck
  elif command -v brew &> /dev/null; then
    echo "Installing shellcheck using brew..."
    brew install shellcheck
  else
    echo "WARNING: Could not install shellcheck. Skipping shell script checks."
    exit 0
  fi
fi

# Find all shell scripts
find_shell_scripts() {
  find . -type f -name "*.sh" | grep -v "node_modules" | grep -v "venv"
}

# Count shell scripts
SHELL_SCRIPTS=$(find_shell_scripts)
SCRIPT_COUNT=$(echo "$SHELL_SCRIPTS" | wc -l)
SCRIPT_COUNT=${SCRIPT_COUNT// /}

if [ "$SCRIPT_COUNT" -eq "0" ]; then
  echo "No shell scripts found to check."
  echo "## Shell Script Analysis" >> $GITHUB_STEP_SUMMARY
  echo "" >> $GITHUB_STEP_SUMMARY
  echo "No shell scripts found for analysis." >> $GITHUB_STEP_SUMMARY
  exit 0
fi

echo "Found $SCRIPT_COUNT shell scripts to check."

# Run shellcheck on all shell scripts
echo "$SHELL_SCRIPTS" | while read -r script; do
  if [ -f "$script" ]; then
    echo "Checking $script..."
    
    # Run shellcheck and save results
    shellcheck -f json "$script" > "ci_artifacts/shell_lint/$(basename "$script").json" || true
    shellcheck -f checkstyle "$script" > "ci_artifacts/shell_lint/$(basename "$script").checkstyle.xml" || true
    
    # Get error count
    ERROR_COUNT=$(shellcheck "$script" 2>&1 | grep -c "^In" || echo "0")
    
    if [ "$ERROR_COUNT" -eq "0" ]; then
      echo "✓ $script is clean"
    else
      echo "✗ $script has $ERROR_COUNT issues"
    fi
  fi
done

# Combine all results
cat ci_artifacts/shell_lint/*.json > ci_artifacts/shell_lint/combined_results.json 2>/dev/null || echo "[]" > ci_artifacts/shell_lint/combined_results.json

# Count total issues
TOTAL_ISSUES=$(grep -c "code\":" ci_artifacts/shell_lint/combined_results.json || echo "0")

# Generate report for GitHub summary
echo "## Shell Script Analysis" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "| Metric | Value |" >> $GITHUB_STEP_SUMMARY
echo "|--------|-------|" >> $GITHUB_STEP_SUMMARY
echo "| Scripts Analyzed | $SCRIPT_COUNT |" >> $GITHUB_STEP_SUMMARY

if [ "$TOTAL_ISSUES" -eq "0" ]; then
  echo "| Shellcheck Issues | :white_check_mark: 0 |" >> $GITHUB_STEP_SUMMARY
  echo "" >> $GITHUB_STEP_SUMMARY
  echo ":tada: All shell scripts are clean!" >> $GITHUB_STEP_SUMMARY
else
  echo "| Shellcheck Issues | :warning: $TOTAL_ISSUES |" >> $GITHUB_STEP_SUMMARY
  echo "" >> $GITHUB_STEP_SUMMARY
  echo ":warning: Found $TOTAL_ISSUES potential issues in shell scripts. See shell-lint artifact for details." >> $GITHUB_STEP_SUMMARY
fi

echo "Shell script check completed. Reports saved to ci_artifacts/shell_lint/" 