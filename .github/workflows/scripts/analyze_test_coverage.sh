#!/bin/bash
# Test Coverage Analysis
# This script measures test coverage and generates coverage reports

echo "Analyzing test coverage..."
mkdir -p ci_artifacts/coverage

# Install coverage tools
python -m pip install coverage pytest pytest-cov

# Create .coveragerc file if it doesn't exist
if [ ! -f .coveragerc ]; then
  echo "Creating default .coveragerc configuration..."
  cat > .coveragerc << EOF
[run]
source = service
omit =
    */site-packages/*
    */tests/*
    */__pycache__/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
EOF
fi

# Run coverage analysis
echo "Running test coverage analysis..."
if [ -d "service" ]; then
  python -m pytest service --cov=service --cov-report=xml:ci_artifacts/coverage/coverage.xml --cov-report=html:ci_artifacts/coverage/html || true
else
  # If no service directory, run on the whole project
  python -m pytest --cov --cov-report=xml:ci_artifacts/coverage/coverage.xml --cov-report=html:ci_artifacts/coverage/html || true
fi

# Extract coverage percentage
if [ -f "ci_artifacts/coverage/coverage.xml" ]; then
  COVERAGE_PCT=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('ci_artifacts/coverage/coverage.xml'); root = tree.getroot(); print(root.attrib.get('line-rate', '0.0'))")
  COVERAGE_PCT=$(python -c "print(round(float('$COVERAGE_PCT') * 100, 2))")
else
  COVERAGE_PCT="0.0"
fi

# Generate simple text report
python -m coverage report > ci_artifacts/coverage/coverage_report.txt || true

# Add coverage info to GitHub step summary
echo "## Test Coverage Results" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

# Define a function to get an emoji for the coverage value
get_coverage_emoji() {
  local coverage=$1
  if (( $(echo "$coverage >= 80" | bc -l) )); then
    echo ":green_circle:"
  elif (( $(echo "$coverage >= 60" | bc -l) )); then
    echo ":yellow_circle:"
  else
    echo ":red_circle:"
  fi
}

COVERAGE_EMOJI=$(get_coverage_emoji $COVERAGE_PCT)

echo "| Metric | Value |" >> $GITHUB_STEP_SUMMARY
echo "|--------|-------|" >> $GITHUB_STEP_SUMMARY
echo "| Overall Coverage | $COVERAGE_EMOJI $COVERAGE_PCT% |" >> $GITHUB_STEP_SUMMARY

# Add coverage badge
echo "" >> $GITHUB_STEP_SUMMARY
echo "![Coverage](https://img.shields.io/badge/coverage-$COVERAGE_PCT%25-${COVERAGE_PCT//.})" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "See coverage-report artifact for complete details" >> $GITHUB_STEP_SUMMARY

echo "Coverage analysis completed. Reports saved to ci_artifacts/coverage/" 