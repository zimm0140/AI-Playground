#!/usr/bin/env python3
"""
Script to fix common markdown linting issues automatically.

This script addresses:
- MD009: Trailing spaces
- MD012: Multiple consecutive blank lines
- MD029: Ordered list item prefix (preserves sequential numbering)
- MD031: Blank lines around fenced code blocks
- MD040: Code blocks without requiring language specifiers
- MD047: Single trailing newline at end of file
- MD004: Unordered list style (using dashes)
- MD022: Headings surrounded by blank lines
- MD026: Remove trailing punctuation in headings
- MD010: Convert hard tabs to spaces
"""

import glob
import os
import re


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


def fix_consecutive_blank_lines(content):
    """Fix multiple consecutive blank lines (MD012)."""
    # Replace 2+ consecutive blank lines with a single blank line
    return re.sub(r"\n{3,}", "\n\n", content)


def fix_heading_spacing(content):
    """Fix spacing around headings (MD022)."""
    # Ensure headings have blank lines before and after them
    lines = content.splitlines()
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Check if this is a heading
        if re.match(r"^#{1,6}\s+", line):
            # Add blank line before heading if not at start of document
            # and previous line is not blank
            if i > 0 and result and result[-1].strip():
                result.append("")

            # Add the heading
            result.append(line)

            # Add blank line after heading if next line is not blank
            # and not at end of document
            if i < len(lines) - 1 and lines[i + 1].strip():
                result.append("")
                # Skip adding the blank line again if we just added one
                if i < len(lines) - 1 and not lines[i + 1].strip():
                    i += 1
        else:
            result.append(line)
        i += 1

    return "\n".join(result)


def fix_heading_punctuation(content):
    """Remove trailing punctuation from headings (MD026)."""
    # Replace trailing punctuation in headings
    return re.sub(r"^(#{1,6}\s+.*?)[.,;:!。，；：！](\s*)$", r"\1\2", content, flags=re.MULTILINE)


def fix_list_style(content):
    """Fix unordered list style (MD004) and preserve ordered list numbering (MD029)."""
    lines = content.splitlines()
    result = []

    for line in lines:
        # Fix unordered lists (asterisks/plus to dashes)
        if re.match(r"^\s*[*+]\s", line):
            line = re.sub(r"^(\s*)[*+](\s)", r"\1-\2", line)

        # Fix hard tabs (MD010)
        line = line.replace("\t", "    ")

        # We preserve ordered list numbering by not modifying the numbers

        result.append(line)

    return "\n".join(result)


def ensure_blank_lines_around_lists(content):
    """Ensure lists have blank lines before and after them (MD032)."""
    lines = content.splitlines()
    result = []
    in_list = False
    i = 0

    while i < len(lines):
        line = lines[i]
        # Check if this line starts a list item
        is_list_item = re.match(r"^\s*([*+-]|\d+\.)\s", line)

        if is_list_item and not in_list:
            # Starting a new list - add blank line before if needed
            if i > 0 and result and result[-1].strip():
                result.append("")
            in_list = True

        # Check if we're exiting a list
        if in_list and not is_list_item and line.strip():
            # Exiting list to non-empty line - add blank line if needed
            if result and result[-1].strip():
                result.append("")
            in_list = False

        # Add the current line
        result.append(line)
        i += 1

    return "\n".join(result)


def fix_code_blocks(content):
    """Ensure code blocks have blank lines around them (MD031) without requiring language specifiers."""
    lines = content.splitlines()
    result = []
    in_code_block = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect start/end of code blocks
        if line.startswith("```"):
            # If starting a code block and previous line is not blank
            if not in_code_block and i > 0 and result and result[-1].strip():
                result.append("")  # Add blank line before code block

            # Add the code block marker
            result.append(line)
            in_code_block = not in_code_block

            # If ending a code block and next line exists and is not blank
            if not in_code_block and i < len(lines) - 1 and lines[i + 1].strip():
                result.append("")  # Add blank line after code block
        else:
            result.append(line)

        i += 1

    return "\n".join(result)


def ensure_trailing_newline(content):
    """Ensure file ends with exactly one newline (MD047)."""
    return content.rstrip("\n") + "\n"


def fix_markdown_file(file_path):
    """Apply all fixes to a markdown file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes in a specific order
        content = fix_trailing_spaces(content)
        content = fix_consecutive_blank_lines(content)
        content = fix_heading_spacing(content)
        content = fix_heading_punctuation(content)
        content = fix_list_style(content)
        content = ensure_blank_lines_around_lists(content)
        content = fix_code_blocks(content)
        content = ensure_trailing_newline(content)

        # Write changes if needed
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed linting issues in {file_path}")
            return True
        print(f"✓ No fixable issues found in {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return False


def main():
    """Find and fix markdown files."""
    # Find all markdown files
    markdown_files = []
    for extension in ["*.md", "*.markdown"]:
        markdown_files.extend(glob.glob(f"**/{extension}", recursive=True))

    # Add specific files that need attention based on the linting output
    priority_files = [
        "readme.md",
        "CONTRIBUTING.md",
        "CODE_QUALITY.md",
        ".github/workflows/README.md",
        ".github/workflows/WORKFLOW.md",
        "docs/comfyui_workflow_validation.md",
        "WebUI/external/components/README.md",
        "PR-CHANGES.md",
        "workflows-document.md",
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
        if file not in processed and fix_markdown_file(file):
            fixed_count += 1

    print(f"\n✅ Fixed issues in {fixed_count} files")
    print("\nNote: Some markdown issues may require manual fixing:")
    print("1. MD013/line-length: Lines exceeding 180 characters (consider breaking these manually)")
    print("2. MD025/single-title: Multiple top-level headings in the same document")
    print("3. MD033/no-inline-html: Replace HTML with Markdown syntax where possible")
    print("4. Check markdown files with a markdown linter after running this script")


if __name__ == "__main__":
    main()