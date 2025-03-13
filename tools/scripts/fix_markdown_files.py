#!/usr/bin/env python3
"""
Script to fix common markdown issues in documentation files.
This is primarily used in CI to automatically fix markdown issues.
"""

import argparse
import os
import re
from pathlib import Path


def fix_markdown_file(file_path):
    """Fix common markdown issues in a file."""
    print(f"Processing {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Fix duplicate TOC sections (remove duplicate Table of Contents blocks)
    toc_pattern = re.compile(r"(## Table of Contents.*?)(?=##|\Z)", re.DOTALL)
    toc_matches = list(toc_pattern.finditer(content))

    if len(toc_matches) > 1:
        # Keep only the first TOC section
        toc_text = toc_matches[0].group(1)
        # Replace content with first match and remove others
        for match in toc_matches[1:]:
            content = content.replace(match.group(1), "")

    # Fix ordered list numbering (make all numbers sequential)
    list_pattern = re.compile(r"^(\s*)\d+\.\s", re.MULTILINE)
    list_matches = list(list_pattern.finditer(content))

    if list_matches:
        lines = content.splitlines()
        current_list_indent = None
        current_number = 0

        for i, line in enumerate(lines):
            match = list_pattern.match(line)
            if match:
                indent = match.group(1)
                # If this is a new list or different indentation level
                if current_list_indent != indent:
                    current_list_indent = indent
                    current_number = 1
                else:
                    current_number += 1

                # Replace the number in the line
                line_prefix = f"{indent}{current_number}. "
                line_content = list_pattern.sub("", line)
                lines[i] = line_prefix + line_content
            elif line.strip() == "" or not line.startswith(current_list_indent if current_list_indent else ""):
                # Reset list tracking on blank lines or when indentation changes
                current_list_indent = None
                current_number = 0

        content = "\n".join(lines)

    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Fix code block syntax (ensure proper code block markers)
    content = re.sub(r"```text", "```", content)

    # Ensure there's a blank line before and after headings
    content = re.sub(r"([^\n])\n(#{1,6}\s)", r"\1\n\n\2", content)
    content = re.sub(r"(#{1,6}[^\n]+)\n([^\n])", r"\1\n\n\2", content)

    # Check if any changes were made
    if content != original_content:
        print(f"✅ Fixed issues in {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    print(f"No issues found in {file_path}")
    return False


def find_markdown_files(directory, recursive=False, exclude=None):
    """Find all markdown files in a directory."""
    if exclude is None:
        exclude = []

    # Convert exclude to absolute paths
    exclude = [os.path.abspath(Path(ex)) for ex in exclude]

    md_files = []
    search_dir = Path(directory)

    if recursive:
        pattern = "**/*.md"
    else:
        pattern = "*.md"

    for file_path in search_dir.glob(pattern):
        abs_path = os.path.abspath(file_path)
        if not any(abs_path.startswith(ex) for ex in exclude):
            md_files.append(file_path)

    return md_files


def main():
    parser = argparse.ArgumentParser(description="Fix common markdown issues")
    parser.add_argument("--dir", default="docs", help="Directory to search for markdown files")
    parser.add_argument("--recursive", action="store_true", help="Search recursively")
    parser.add_argument("--exclude", nargs="+", default=["node_modules"], help="Directories to exclude")

    args = parser.parse_args()

    md_files = find_markdown_files(args.dir, args.recursive, args.exclude)

    if not md_files:
        print(f"No markdown files found in {args.dir}")
        return

    print(f"Found {len(md_files)} markdown files to process")

    fixed_files = 0
    for file_path in md_files:
        if fix_markdown_file(file_path):
            fixed_files += 1

    print(f"Fixed issues in {fixed_files} files")


if __name__ == "__main__":
    main()
