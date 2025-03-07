#!/bin/bash
# Dependency Security Scanner
# This script scans Python dependencies for known security vulnerabilities

echo "Scanning dependencies for security vulnerabilities..."
mkdir -p ci_artifacts/security

# Install safety scanner
python -m pip install safety

# Generate requirements file if not present
if [ ! -f requirements.txt ]; then
  echo "No requirements.txt found, generating from installed packages..."
  python -m pip freeze > requirements.txt
fi

# Run security scan on dependencies
echo "Running security scan on main requirements..."
python -m safety check -r requirements.txt --output json > ci_artifacts/security/main_requirements_scan.json 2>/dev/null || true
python -m safety check -r requirements.txt --output text > ci_artifacts/security/main_requirements_scan.txt || true

# Check service requirements if they exist
if [ -f service/requirements.txt ]; then
  echo "Running security scan on service requirements..."
  python -m safety check -r service/requirements.txt --output json > ci_artifacts/security/service_requirements_scan.json 2>/dev/null || true
  python -m safety check -r service/requirements.txt --output text > ci_artifacts/security/service_requirements_scan.txt || true
fi

# Generate security report summary for GitHub step summary
echo "## Security Scan Results" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Count vulnerabilities
MAIN_VULN_COUNT=$(grep -c "vulnerability found" ci_artifacts/security/main_requirements_scan.txt || echo "0")
SERVICE_VULN_COUNT=0
if [ -f ci_artifacts/security/service_requirements_scan.txt ]; then
  SERVICE_VULN_COUNT=$(grep -c "vulnerability found" ci_artifacts/security/service_requirements_scan.txt || echo "0")
fi
TOTAL_VULN_COUNT=$((MAIN_VULN_COUNT + SERVICE_VULN_COUNT))

# Add summary table
echo "| Package Source | Status |" >> $GITHUB_STEP_SUMMARY
echo "|---------------|--------|" >> $GITHUB_STEP_SUMMARY
if [ "$MAIN_VULN_COUNT" -eq "0" ]; then
  echo "| Main requirements | :white_check_mark: No vulnerabilities |" >> $GITHUB_STEP_SUMMARY
else
  echo "| Main requirements | :warning: $MAIN_VULN_COUNT vulnerabilities found |" >> $GITHUB_STEP_SUMMARY
fi

if [ -f service/requirements.txt ]; then
  if [ "$SERVICE_VULN_COUNT" -eq "0" ]; then
    echo "| Service requirements | :white_check_mark: No vulnerabilities |" >> $GITHUB_STEP_SUMMARY
  else
    echo "| Service requirements | :warning: $SERVICE_VULN_COUNT vulnerabilities found |" >> $GITHUB_STEP_SUMMARY
  fi
fi

echo "" >> $GITHUB_STEP_SUMMARY
if [ "$TOTAL_VULN_COUNT" -eq "0" ]; then
  echo ":tada: No security vulnerabilities found in dependencies!" >> $GITHUB_STEP_SUMMARY
else
  echo ":warning: Found $TOTAL_VULN_COUNT potential security issues. See security-report artifact for details." >> $GITHUB_STEP_SUMMARY
fi

echo "Security scan completed. Reports saved to ci_artifacts/security/" 