#!/bin/bash
# CI Performance Profiler
# This script measures and analyzes CI step durations to identify bottlenecks

echo "Profiling CI performance..."
mkdir -p ci_artifacts/performance

# Function to measure execution time of a command
measure_performance() {
  local step_name="$1"
  local command="$2"
  
  echo "Measuring performance of: $step_name"
  
  # Record start time
  local start_time=$(date +%s.%N)
  
  # Execute the command
  eval "$command"
  local exit_code=$?
  
  # Record end time
  local end_time=$(date +%s.%N)
  
  # Calculate duration
  local duration=$(echo "$end_time - $start_time" | bc)
  
  # Log performance data
  echo "$step_name,$duration,$exit_code" >> ci_artifacts/performance/performance_data.csv
  
  echo "Step '$step_name' completed in $duration seconds (exit code: $exit_code)"
  
  return $exit_code
}

# Initialize performance data file
echo "Step,Duration,ExitCode" > ci_artifacts/performance/performance_data.csv

# Define thresholds for performance warning
SLOW_THRESHOLD=10.0  # seconds
VERY_SLOW_THRESHOLD=30.0  # seconds

# Profile environment setup
measure_performance "Environment check" "python -c 'import sys; print(sys.version)'"

# Profile dependency resolution (using pip download without installing as a test)
measure_performance "Dependency resolution" "python -m pip download -d /tmp/pip_download_test --no-deps --no-binary :all: pytest >/dev/null 2>&1"

# Profile import speed test (helps identify slow imports in the codebase)
cat > import_speed_test.py << EOF
import time
import sys
from importlib import import_module

def measure_import_time(module_name):
    start_time = time.time()
    try:
        import_module(module_name)
        import_time = time.time() - start_time
        print(f"{module_name},{import_time:.4f},0")
        return True
    except ImportError as e:
        import_time = time.time() - start_time
        print(f"{module_name},{import_time:.4f},1")
        return False

# Common modules to test
modules = [
    'numpy', 
    'torch', 
    'pandas', 
    'pytest', 
    'coverage',
    'service'
]

print("Module,ImportTime,Error")
for module in modules:
    measure_import_time(module)
EOF

measure_performance "Module import speed" "python import_speed_test.py > ci_artifacts/performance/import_times.csv"

# Profile test discovery time
measure_performance "Test discovery" "python -m pytest --collect-only -q || true"

# Basic test directory walk performance
measure_performance "Directory traversal" "find . -name '*.py' -type f | wc -l"

# Analyze the performance data
echo "Analyzing performance data..."

# Calculate totals and identify bottlenecks
python -c "
import csv
import os

# Read the performance data
data = []
with open('ci_artifacts/performance/performance_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Sort steps by duration
sorted_data = sorted(data, key=lambda x: float(x['Duration']), reverse=True)

# Calculate total and average
total_duration = sum(float(step['Duration']) for step in data)
avg_duration = total_duration / len(data) if data else 0

# Identify bottlenecks (steps taking > 20% of total time)
bottlenecks = [step for step in data if float(step['Duration']) / total_duration > 0.2]

# Create performance summary
with open('ci_artifacts/performance/performance_summary.md', 'w') as f:
    f.write('# CI Performance Summary\n\n')
    f.write(f'Total profiled time: {total_duration:.2f} seconds\n')
    f.write(f'Average step time: {avg_duration:.2f} seconds\n\n')
    
    f.write('## Step Durations\n\n')
    f.write('| Step | Duration (s) | % of Total |\n')
    f.write('|------|--------------|------------|\n')
    for step in sorted_data:
        pct = (float(step['Duration']) / total_duration) * 100
        f.write(f'| {step[\"Step\"]} | {float(step[\"Duration\"]):.2f} | {pct:.1f}% |\n')
    
    if bottlenecks:
        f.write('\n## Identified Bottlenecks\n\n')
        for step in bottlenecks:
            pct = (float(step['Duration']) / total_duration) * 100
            f.write(f'- **{step[\"Step\"]}** takes {float(step[\"Duration\"]):.2f}s ({pct:.1f}% of total time)\n')
        
        f.write('\n### Optimization Suggestions\n\n')
        f.write('- Consider parallelizing or optimizing the identified bottleneck steps\n')
        f.write('- Use caching strategies for dependencies and test results\n')
        f.write('- Review module import times for slow loading dependencies\n')
"

# Read the performance summary for GitHub step summary
if [ -f ci_artifacts/performance/performance_summary.md ]; then
  echo "## CI Performance Profile" >> $GITHUB_STEP_SUMMARY
  echo "" >> $GITHUB_STEP_SUMMARY
  
  # Extract and add key metrics
  TOTAL_TIME=$(grep "Total profiled time:" ci_artifacts/performance/performance_summary.md | cut -d' ' -f4)
  echo "| Metric | Value |" >> $GITHUB_STEP_SUMMARY
  echo "|--------|-------|" >> $GITHUB_STEP_SUMMARY
  echo "| Total profiled time | ${TOTAL_TIME}s |" >> $GITHUB_STEP_SUMMARY
  
  # Extract and add bottlenecks if any
  if grep -q "Identified Bottlenecks" ci_artifacts/performance/performance_summary.md; then
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "### Performance Bottlenecks" >> $GITHUB_STEP_SUMMARY
    echo "" >> $GITHUB_STEP_SUMMARY
    
    grep -A 10 "^- \*\*" ci_artifacts/performance/performance_summary.md >> $GITHUB_STEP_SUMMARY || true
    
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "See performance-report artifact for complete details" >> $GITHUB_STEP_SUMMARY
  else
    echo "" >> $GITHUB_STEP_SUMMARY
    echo "No significant performance bottlenecks identified." >> $GITHUB_STEP_SUMMARY
  fi
fi

# Clean up temporary files
rm -f import_speed_test.py

echo "Performance profiling completed. Reports saved to ci_artifacts/performance/" 