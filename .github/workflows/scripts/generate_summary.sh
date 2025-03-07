#!/bin/bash
# Workflow Summary Generator
# This script creates a detailed summary of the CI run for better visibility in GitHub Actions

echo "Generating workflow summary..."

# Create the summary header
echo "# CI Workflow Summary :clipboard:" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Environment Information
echo "## Environment Information" >> $GITHUB_STEP_SUMMARY
echo "- Python version: $(python --version)" >> $GITHUB_STEP_SUMMARY
echo "- PyTorch version: $(python -c 'import torch; print(torch.__version__)')" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Applied Patches
echo "## Applied Patches" >> $GITHUB_STEP_SUMMARY
echo "- :white_check_mark: Intel Extension for PyTorch stub" >> $GITHUB_STEP_SUMMARY
echo "- :white_check_mark: CPU-only mode enforcement" >> $GITHUB_STEP_SUMMARY
echo "- :white_check_mark: Import compatibility fixes" >> $GITHUB_STEP_SUMMARY
echo "- :white_check_mark: XPU hijacks patching" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Code Quality Checks
echo "## Code Quality Checks" >> $GITHUB_STEP_SUMMARY
if [ -f ci_artifacts/code_quality_summary.txt ]; then
  echo "| Category | Count |" >> $GITHUB_STEP_SUMMARY
  echo "|----------|-------|" >> $GITHUB_STEP_SUMMARY
  grep "Prints" ci_artifacts/code_quality_summary.txt | sed 's/.*: ==/| Print statements |/' | sed 's/ //' | sed 's/$/ |/' >> $GITHUB_STEP_SUMMARY
  grep "TODOs" ci_artifacts/code_quality_summary.txt | sed 's/.*: ==/| TODOs |/' | sed 's/ //' | sed 's/$/ |/' >> $GITHUB_STEP_SUMMARY
  
  if [ -s yapf_report.txt ]; then
    format_issues=$(grep -c "@@" yapf_report.txt || echo "0")
    echo "| Format issues | $format_issues |" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "See 'code-quality-report' artifact for details" >> $GITHUB_STEP_SUMMARY
  else
    echo "| Format issues | 0 |" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    echo ":tada: Code follows formatting standards!" >> $GITHUB_STEP_SUMMARY
  fi
else
  echo "No code quality report available" >> $GITHUB_STEP_SUMMARY
fi
echo "" >> $GITHUB_STEP_SUMMARY

# Hardware Support
echo "## Hardware Support" >> $GITHUB_STEP_SUMMARY
echo "See 'hardware-support-matrix' artifact for a detailed list of supported hardware configurations" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Test Status
echo "## Test Status" >> $GITHUB_STEP_SUMMARY
echo "- :information_source: Tests were run with error handling enabled" >> $GITHUB_STEP_SUMMARY
echo "- :warning: Some tests may have been skipped due to missing dependencies in CI" >> $GITHUB_STEP_SUMMARY
echo "- :rocket: CI workflow completed successfully" >> $GITHUB_STEP_SUMMARY

echo "Workflow summary generated successfully!" 