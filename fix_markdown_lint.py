#!/usr/bin/env python3
"""
Script to fix common markdown linting issues automatically.

This script addresses:
- MD009: Trailing spaces
- MD031: Blank lines around fenced code blocks
- MD047: Single trailing newline at end of file
- MD040: Adding language to code blocks
"""

import os
import re
import glob
from pathlib import Path

def fix_trailing_spaces(content):
    """Fix trailing whitespace (MD009)."""
    # Replace trailing spaces but preserve intentional line breaks (two spaces)
    lines = []
    for line in content.splitlines():
        if line.rstrip() == "":
            lines.append("")  # Empty lines should have no trailing space
        elif line.endswith("  "):
            lines.append(line)  # Keep lines with exactly two trailing spaces (markdown line break)
        else:
            lines.append(line.rstrip())  # Remove trailing spaces
    return "\n".join(lines)

def fix_code_blocks(content):
    """Ensure code blocks have blank lines around them (MD031) and a language (MD040)."""
    # Pattern to find code blocks
    pattern = re.compile(r'(```[^\n]*\n[\s\S]*?```)', re.MULTILINE)
    
    # Fix code blocks
    def code_block_fix(match):
        block = match.group(1)
        
        # Check if there's a language specified
        if block.startswith('```\n'):
            # Add 'text' as default language
            block = '```text\n' + block[4:]
        
        # Ensure there's a blank line before and after (if not at the start/end of the file)
        if not block.startswith('\n\n') and not block.startswith('\n'):
            block = '\n' + block
        if not block.endswith('\n\n') and not block.endswith('\n'):
            block = block + '\n'
            
        return block
    
    return pattern.sub(code_block_fix, content)

def ensure_trailing_newline(content):
    """Ensure file ends with exactly one newline (MD047)."""
    return content.rstrip('\n') + '\n'

def fix_markdown_file(file_path):
    """Apply all fixes to a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        content = fix_trailing_spaces(content)
        content = fix_code_blocks(content)
        content = ensure_trailing_newline(content)
        
        # Write changes if needed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed linting issues in {file_path}")
            return True
        else:
            print(f"✓ No fixable issues found in {file_path}")
            return False
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return False

def main():
    """Find and fix markdown files."""
    # Find all markdown files
    markdown_files = []
    for extension in ['*.md', '*.markdown']:
        markdown_files.extend(glob.glob(f"**/{extension}", recursive=True))
    
    # Add specific files that need attention based on the linting output
    priority_files = [
        'CODE_QUALITY.md',
        '.github/workflows/README.md',
        '.github/workflows/WORKFLOW.md',
        'CONTRIBUTING.md',
        'docs/comfyui_workflow_validation.md',
        'WebUI/external/components/README.md',
        'PR-CHANGES.md',
        'readme.md',
        'workflows-document.md'
    ]
    
    # Process priority files first
    processed = set()
    fixed_count = 0
    
    print(f"🔍 Found {len(markdown_files)} markdown files")
    
    # Process priority files first
    for file in priority_files:
        if os.path.exists(file):
            if fix_markdown_file(file):
                fixed_count += 1
            processed.add(file)
    
    # Process remaining files
    for file in markdown_files:
        if file not in processed:
            if fix_markdown_file(file):
                fixed_count += 1
    
    print(f"\n✅ Fixed issues in {fixed_count} files")
    print("\nNote: Some markdown issues may require manual fixing:")
    print("1. MD033: Replace <br> tags with proper Markdown line breaks (two spaces at end of line)")
    print("2. MD025/MD024: Fix duplicate or multiple headings manually")
    print("3. MD032: Ensure lists have blank lines before and after them")
    print("4. Check markdown files with a markdown linter after running this script")

if __name__ == "__main__":
    main() 