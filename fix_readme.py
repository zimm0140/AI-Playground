#!/usr/bin/env python
"""
Fix README.md trailing whitespace issues.
"""

import os

# Path to the README file
readme_path = os.path.join('WebUI', 'external', 'workflows', 'README.md')

# Read the file content
with open(readme_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix trailing whitespace issues
fixed_content = '\n'.join([line.rstrip() for line in content.split('\n')])

# Write the fixed content back to the file
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Fixed {readme_path}") 