#!/bin/bash
# Dependency Update Checker
# This script checks for outdated dependencies and generates an update report

echo "Checking for dependency updates..."
mkdir -p ci_artifacts/dependencies

# Install pip-tools and other utilities
python -m pip install pip-tools pipdeptree

# Function to check for updates in a requirements file
check_requirements_file() {
  local req_file="$1"
  local output_prefix="$2"
  
  if [ ! -f "$req_file" ]; then
    echo "Requirements file not found: $req_file"
    return 1
  fi
  
  echo "Checking updates for $req_file..."
  
  # Generate constraints from current requirements
  pip-compile --no-header --no-emit-index-url --output-file="$output_prefix-current.txt" "$req_file" >/dev/null 2>&1
  
  # Generate updated requirements
  pip-compile --upgrade --no-header --no-emit-index-url --output-file="$output_prefix-updated.txt" "$req_file" >/dev/null 2>&1
  
  # Compare the files
  echo "Differences between current and updated dependencies in $req_file:"
  diff -u "$output_prefix-current.txt" "$output_prefix-updated.txt" > "$output_prefix-diff.txt" || true
  
  # Count the number of outdated packages
  local outdated_count=$(grep -c "^+" "$output_prefix-diff.txt" || echo "0")
  echo "$outdated_count packages need updates in $req_file"
  
  return $outdated_count
}

# Generate dependency tree for documentation
echo "Generating dependency tree..."
python -m pipdeptree --exclude pip,pipdeptree,pip-tools,setuptools,wheel > ci_artifacts/dependencies/dependency_tree.txt

# Check main requirements file
if [ -f requirements.txt ]; then
  check_requirements_file "requirements.txt" "ci_artifacts/dependencies/main"
  MAIN_OUTDATED=$?
else
  MAIN_OUTDATED=0
fi

# Check service requirements file
if [ -f service/requirements.txt ]; then
  check_requirements_file "service/requirements.txt" "ci_artifacts/dependencies/service"
  SERVICE_OUTDATED=$?
else
  SERVICE_OUTDATED=0
fi

# Generate a summary report
TOTAL_OUTDATED=$((MAIN_OUTDATED + SERVICE_OUTDATED))

echo "## Dependency Update Check" >> $GITHUB_STEP_SUMMARY
echo "" >> $GITHUB_STEP_SUMMARY
echo "| Requirements File | Status |" >> $GITHUB_STEP_SUMMARY
echo "|------------------|--------|" >> $GITHUB_STEP_SUMMARY

if [ -f requirements.txt ]; then
  if [ $MAIN_OUTDATED -eq 0 ]; then
    echo "| requirements.txt | :white_check_mark: Up to date |" >> $GITHUB_STEP_SUMMARY
  else
    echo "| requirements.txt | :warning: $MAIN_OUTDATED updates available |" >> $GITHUB_STEP_SUMMARY
  fi
fi

if [ -f service/requirements.txt ]; then
  if [ $SERVICE_OUTDATED -eq 0 ]; then
    echo "| service/requirements.txt | :white_check_mark: Up to date |" >> $GITHUB_STEP_SUMMARY
  else
    echo "| service/requirements.txt | :warning: $SERVICE_OUTDATED updates available |" >> $GITHUB_STEP_SUMMARY
  fi
fi

echo "" >> $GITHUB_STEP_SUMMARY
if [ $TOTAL_OUTDATED -eq 0 ]; then
  echo ":tada: All dependencies are up to date!" >> $GITHUB_STEP_SUMMARY
else
  echo ":warning: $TOTAL_OUTDATED package updates available. See dependency-report artifact for details." >> $GITHUB_STEP_SUMMARY
fi

# Create Dependabot configuration file if it doesn't exist
if [ ! -f .github/dependabot.yml ]; then
  echo "Creating Dependabot configuration file..."
  mkdir -p .github
  cat > .github/dependabot.yml << EOF
# Dependabot configuration file
# See: https://docs.github.com/github/administering-a-repository/configuration-options-for-dependency-updates

version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "python"

  # Service Python dependencies
  - package-ecosystem: "pip"
    directory: "/service"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
      - "python"
      - "service"

  # GitHub Actions dependencies
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
      - "github_actions"
EOF

  echo "Dependabot configuration created at .github/dependabot.yml"
  echo "" >> $GITHUB_STEP_SUMMARY
  echo "📦 Created Dependabot configuration file at .github/dependabot.yml" >> $GITHUB_STEP_SUMMARY
fi

echo "Dependency update check completed. Reports saved to ci_artifacts/dependencies/" 