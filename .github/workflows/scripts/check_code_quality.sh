#!/bin/bash
# Code Quality Check Script
# This script runs various code quality checks including syntax validation,
# style checking, and detection of common anti-patterns

echo "Running code quality checks..."
mkdir -p ci_artifacts

# Check for Python syntax errors
echo "Checking for Python syntax errors..."
find . -name "*.py" -not -path "*/\.*" -not -path "*/venv/*" -print0 | xargs -0 -n1 python -m py_compile || echo "Some files have syntax errors"

# Run basic style check with yapf
echo "Checking code formatting with yapf..."
yapf --diff --recursive --exclude="venv/*" . > yapf_report.txt
if [ -s yapf_report.txt ]; then
  echo "Some files need formatting (full report in yapf_report.txt artifact):"
  head -n 20 yapf_report.txt
else
  echo "All files are properly formatted according to yapf!"
fi

# Check for common anti-patterns
echo "Checking for common anti-patterns..."
grep_results=$(mktemp)
(
  # Check for print statements (which should be replaced with proper logging)
  echo "== Prints that might need to be converted to logging: =="
  grep -r --include="*.py" "print(" . --include="*.py" | grep -v "tests\|examples\|__pycache__" | wc -l
  
  # Check for TODOs
  echo "== TODOs remaining in code: =="
  grep -r --include="*.py" "TODO" . --include="*.py" | grep -v "tests\|examples\|__pycache__" | wc -l
) > $grep_results

cat $grep_results
cp $grep_results ci_artifacts/code_quality_summary.txt

echo "Code quality check completed successfully!" 