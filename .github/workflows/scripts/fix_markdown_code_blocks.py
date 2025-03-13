#!/usr/bin/env python3
"""
Script to automatically fix markdown code blocks without language specifiers.
This script addresses the MD040 linting issue.
"""

import re
import sys
from pathlib import Path


def fix_code_blocks_without_language(file_path):
    """Fix code blocks without language specifiers in a markdown file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # This regex matches a fence (```) that is not immediately followed by a word character
    pattern = r"```(?!\w+)"

    # Replace them with ```text
    new_content = re.sub(pattern, "```text", content)

    # Only write if changes were made
    if new_content != content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Fixed code blocks in {file_path}")
        return True

    return False


def find_markdown_files():
    """Find all markdown files in the repository."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    md_files = list(repo_root.glob("**/*.md"))

    # Filter out virtual environments or other non-project paths
    excluded_patterns = [".venv", "venv", ".git", "__pycache__", "node_modules", "build", "dist"]

    return [str(path) for path in md_files if not any(pattern in str(path) for pattern in excluded_patterns)]


def main():
    """Find and fix code blocks without language specifiers in markdown files."""
    md_files = find_markdown_files()
    print(f"Found {len(md_files)} markdown files to check")

    fixed_count = 0
    for file_path in md_files:
        if fix_code_blocks_without_language(file_path):
            fixed_count += 1

    print(f"Fixed code blocks in {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
