#!/bin/bash
# Docker-based Testing Script
# This script runs tests inside Docker containers to ensure consistent environments

echo "Setting up Docker-based testing environment..."
mkdir -p ci_artifacts/docker

# Check if Docker is available
if ! command -v docker &> /dev/null; then
  echo "ERROR: Docker is not installed. Skipping Docker-based testing."
  echo "## Docker Testing" >> $GITHUB_STEP_SUMMARY
  echo "" >> $GITHUB_STEP_SUMMARY
  echo "⚠️ Docker is not available. Tests were not run in containerized environment." >> $GITHUB_STEP_SUMMARY
  exit 0
fi

# Create a Dockerfile for testing
cat > Dockerfile.ci << EOF
FROM python:${1:-3.10}-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY service/requirements.txt ./service/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r service/requirements.txt \
    && pip install pytest pytest-cov yapf

# Copy the code
COPY . .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=-1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV XPU_VISIBLE_DEVICES=-1
ENV FORCE_CPU_ONLY=1

# Create necessary directories
RUN mkdir -p ci_artifacts/coverage
EOF

echo "Building Docker image for testing..."
docker build -t ci-test-image -f Dockerfile.ci .

# Run tests in Docker
echo "Running tests in Docker container..."
docker run --name ci-test-container \
  -v "$(pwd)/ci_artifacts:/app/ci_artifacts" \
  ci-test-image \
  bash -c "python -m pytest service --cov=service --cov-report=xml:ci_artifacts/coverage/docker_coverage.xml --cov-report=html:ci_artifacts/coverage/docker_html"

# Check if tests ran successfully
DOCKER_EXIT_CODE=$?

# Run code quality checks in Docker
echo "Running code quality checks in Docker container..."
docker run --name ci-lint-container \
  -v "$(pwd)/ci_artifacts:/app/ci_artifacts" \
  ci-test-image \
  bash -c "yapf --diff --recursive --exclude='venv/*' . > ci_artifacts/docker_yapf_report.txt"

# Generate Docker test summary
echo "## Docker Testing Results" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY

if [ $DOCKER_EXIT_CODE -eq 0 ]; then
  echo "✅ Tests passed successfully in Docker container" >> $GITHUB_STEP_SUMMARY
else
  echo "❌ Tests failed in Docker container (Exit code: $DOCKER_EXIT_CODE)" >> $GITHUB_STEP_SUMMARY
fi

# Extract coverage from Docker run
if [ -f "ci_artifacts/coverage/docker_coverage.xml" ]; then
  COVERAGE_PCT=$(python -c "import xml.etree.ElementTree as ET; tree = ET.parse('ci_artifacts/coverage/docker_coverage.xml'); root = tree.getroot(); print(round(float(root.attrib.get('line-rate', '0.0')) * 100, 2))")
  
  # Get emoji for coverage value
  if (( $(echo "$COVERAGE_PCT >= 80" | bc -l) )); then
    COVERAGE_EMOJI=":green_circle:"
  elif (( $(echo "$COVERAGE_PCT >= 60" | bc -l) )); then
    COVERAGE_EMOJI=":yellow_circle:"
  else
    COVERAGE_EMOJI=":red_circle:"
  fi
  
  echo "| Docker Test Coverage | $COVERAGE_EMOJI $COVERAGE_PCT% |" >> $GITHUB_STEP_SUMMARY
else
  echo "⚠️ Could not retrieve coverage information from Docker tests" >> $GITHUB_STEP_SUMMARY
fi

# Clean up Docker resources
echo "Cleaning up Docker resources..."
docker rm ci-test-container ci-lint-container 2>/dev/null || true
docker rmi ci-test-image 2>/dev/null || true

echo "Docker testing completed" 