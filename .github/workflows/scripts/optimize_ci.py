#!/usr/bin/env python3
"""
CI Optimization Script

This script analyzes and optimizes GitHub Actions workflow files to improve performance,
reduce run time, and fix common issues. It implements best practices like:

1. Proper caching configuration
2. Conditional job steps to skip unnecessary work
3. Efficient dependency installation
4. Parallel job execution where appropriate
5. Removal of redundant actions
"""

import glob
import re

def optimize_ci_workflows():
    """Find and optimize CI workflow files"""
    print("Optimizing CI workflow files...")
    
    # Find all workflow files
    workflow_files = glob.glob(".github/workflows/*.yml") + glob.glob(".github/workflows/*.yaml")
    
    if not workflow_files:
        print("No workflow files found.")
        return
    
    print(f"Found {len(workflow_files)} workflow files to optimize.")
    
    for file_path in workflow_files:
        print(f"Optimizing {file_path}...")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Apply general optimizations
        content = optimize_actions_versions(content)
        content = optimize_caching(content)
        content = optimize_pip_install(content)
        content = add_conditional_execution(content)
        content = optimize_checkout_depth(content)
        content = cleanup_redundant_steps(content)
        
        # Write back the optimized content
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Optimized {file_path}")
    
    print("CI workflow optimization complete!")

def optimize_actions_versions(content):
    """Update GitHub Actions to use the latest stable versions"""
    # Regular expressions to identify and update action versions
    version_patterns = [
        # Update checkout action from v2/v3 to v4
        (r'uses: actions/checkout@v[23]', 'uses: actions/checkout@v4'),
        
        # Update setup-python action from v1/v2/v3 to v4
        (r'uses: actions/setup-python@v[123]', 'uses: actions/setup-python@v4'),
        
        # Update cache action from v1/v2 to v3
        (r'uses: actions/cache@v[12]', 'uses: actions/cache@v3'),
        
        # Update upload-artifact action from v1/v2/v3 to v4
        (r'uses: actions/upload-artifact@v[123]', 'uses: actions/upload-artifact@v4'),
        
        # Update download-artifact action from v1/v2/v3 to v4
        (r'uses: actions/download-artifact@v[123]', 'uses: actions/download-artifact@v4')
    ]
    
    updated = False
    for pattern, replacement in version_patterns:
        if re.search(pattern, content):
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                updated = True
    
    if updated:
        print("  ↑ Updated action versions to latest stable releases")
    
    return content

def optimize_caching(content):
    """Optimize caching configurations for better performance"""
    # Check if we have proper pip caching
    pip_cache_pattern = r'actions/cache@v\d+.*\s+path:.*pip.*cache'
    has_pip_cache = bool(re.search(pip_cache_pattern, content, re.DOTALL))
    
    # Proper pip cache template for different OS
    pip_cache_template = """      - name: Configure pip caching
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/pip
            ${{ runner.os == 'Windows' && '~\\\\AppData\\\\Local\\\\pip\\\\Cache' || runner.os == 'macOS' && '~/Library/Caches/pip' || '~/.cache/pip' }}
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-pip-
"""
    
    # Add pip cache if it doesn't exist
    if not has_pip_cache and "setup-python" in content:
        # Find setup-python step and insert cache after it
        setup_python_pattern = r'(- name:.*setup.*python.*\n(?:\s+.*\n)*?\s+python-version:.*\n)'
        content = re.sub(setup_python_pattern, r'\1\n' + pip_cache_template, content)
        print("  + Added optimized pip caching")
    
    # Add dependency hash to cache key if not present
    if "cache" in content and "hashFiles" not in content:
        # Update cache keys with proper file hashing
        content = re.sub(
            r'(key:.*?)(\n)',
            r'\1-${{ hashFiles(\'**/requirements*.txt\', \'**/package.json\', \'**/yarn.lock\', \'**/pnpm-lock.yaml\') }}\2',
            content
        )
        print("  ↑ Added dependency hash to cache keys")
    
    return content

def optimize_pip_install(content):
    """Optimize pip install commands for better performance"""
    # Make sure pip is upgraded
    if "pip install" in content and "pip install --upgrade pip" not in content:
        # Add pip upgrade
        content = re.sub(
            r'(python -m pip install)',
            r'python -m pip install --upgrade pip\n          \1',
            content
        )
        print("  + Added pip upgrade step")
    
    # Use pip compile if multiple requirements files
    if re.search(r'pip install.*requirements\.txt.*\n.*pip install.*requirements', content, re.MULTILINE):
        # Suggest combining requirements files
        print("  ℹ Suggestion: Consider using pip-compile to combine requirements files")
    
    return content

def add_conditional_execution(content):
    """Add conditional execution to skip unnecessary steps"""
    # Add conditional execution to upload-artifact steps if not present
    if "uses: actions/upload-artifact" in content:
        upload_pattern = r'(- name:.*upload.*\n\s+uses: actions/upload-artifact.*\n)(?!\s+if:)'
        upload_with_condition = r'\1      if: always()\n'
        
        # Add if: always() condition to ensure artifacts are uploaded even if tests fail
        if re.search(upload_pattern, content):
            content = re.sub(upload_pattern, upload_with_condition, content)
            print("  + Added 'if: always()' to artifact upload steps")
    
    # Add conditional execution for tests if not present
    test_patterns = [
        r'(- name:.*run tests.*\n\s+run:.*\n)(?!\s+if:)',
        r'(- name:.*pytest.*\n\s+run:.*\n)(?!\s+if:)'
    ]
    for pattern in test_patterns:
        if re.search(pattern, content):
            # We aren't adding a condition, just noting it would be good to have
            print("  ℹ Suggestion: Consider adding conditions to test steps for skip capability")
            break
    
    return content

def optimize_checkout_depth(content):
    """Optimize checkout depth for faster clones"""
    # Add fetch-depth to checkout action if not present
    checkout_pattern = r'uses: actions/checkout@v\d+\s*\n(?!\s+with:[\s\S]*?fetch-depth:)'
    if re.search(checkout_pattern, content):
        # Add fetch-depth: 1 to checkout action
        checkout_with_depth = r'uses: actions/checkout@v4\n        with:\n          fetch-depth: 1'
        content = re.sub(r'uses: actions/checkout@v\d+', checkout_with_depth, content)
        print("  + Added fetch-depth: 1 to checkout action for faster clones")
    
    return content

def cleanup_redundant_steps(content):
    """Remove or combine redundant steps to reduce workflow time"""
    # Check for multiple similar pip install steps that could be combined
    pip_install_count = len(re.findall(r'pip install', content))
    if pip_install_count > 3:
        print(f"  ℹ Suggestion: Consider combining {pip_install_count} pip install steps")
    
    # Count steps in the workflow
    step_count = len(re.findall(r'- name:', content))
    if step_count > 20:
        print(f"  ℹ Suggestion: Workflow has {step_count} steps, consider grouping related steps")
    
    return content

if __name__ == "__main__":
    optimize_ci_workflows() 