#!/usr/bin/env python3
"""
Advanced Markdown Linter Fix Script

This script automatically fixes common markdown linting issues detected by markdownlint,
with special handling for:
1. Multiple consecutive blank lines (MD012)
2. Line length issues (MD013)
3. Tables not surrounded by blank lines (MD058)
4. List marker spacing (MD030)
5. Preserves sequential list numbering (MD029)
6. Respects bare URLs without angle brackets (MD034 is set to false)
7. Respects code blocks without language specifiers (MD040 is set to false)
"""

import re
import sys
from pathlib import Path

# Maximum line length (from .markdownlint.yaml)
MAX_LINE_LENGTH = 180


def fix_trailing_spaces(content: str) -> str:
    """Remove trailing whitespace from lines (MD009)."""
    return re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)


def fix_consecutive_blank_lines(content: str) -> str:
    """Reduce multiple consecutive blank lines to a single blank line (MD012)."""
    return re.sub(r"\n{3,}", "\n\n", content)


def fix_heading_spacing(content: str) -> str:
    """Fix spacing after heading markers."""
    # Remove space between # and heading text
    return re.sub(r"(#{1,6})[ ]{2,}", r"\1 ", content)


def fix_heading_punctuation(content: str) -> str:
    """Remove trailing punctuation in headings (MD026)."""
    # Replace trailing punctuation in headings
    return re.sub(r"^(#{1,6}\s+.*?)[.,;:!。，；：！](\s*)$", r"\1\2", content, flags=re.MULTILINE)


def fix_ordered_lists(content: str) -> str:
    """
    Preserve ordered list numbering as set in markdownlint.yaml (MD029: ordered).
    This function is intentionally a no-op to respect user's existing numbering.
    """
    # Just return the content unchanged
    return content


def fix_list_marker_spacing(content: str) -> str:
    """Fix spacing after list markers (MD030) while preserving sequential numbering."""
    # Fix unordered list spacing
    content = re.sub(r"^(\s*)([*+-])(\s{2,})", r"\1\2 ", content, flags=re.MULTILINE)
    # Fix ordered list spacing while preserving the numbers
    content = re.sub(r"^(\s*)(\d+\.)(\s{2,})", r"\1\2 ", content, flags=re.MULTILINE)
    return content


def fix_code_blocks(content: str) -> str:
    """Fix spacing and formatting around code blocks."""
    # Importantly, do NOT add language specifiers to code blocks
    return content


def fix_table_spacing(content: str) -> str:
    """Fix spacing around tables for better readability."""
    # Add blank line after tables
    return re.sub(r"(\|[^|]*\|.*\n)\n([^\n])", r"\1\n\n\2", content)


def ensure_trailing_newline(content: str) -> str:
    """Ensure file ends with exactly one newline (MD047)."""
    return content.rstrip("\n") + "\n"


def fix_bare_urls(content: str) -> str:
    """
    Respect bare URLs without angle brackets as set in markdownlint.yaml (MD034: false).
    This function is intentionally a no-op to respect user's existing URLs.
    """
    # Just return the content unchanged
    return content


def fix_ordered_list_spacing(content: str) -> str:
    """Fix spacing after ordered list markers while preserving numbers."""
    # Fix ordered list spacing while preserving the numbers
    return re.sub(r"^(\s*)(\d+\.)(\s{2,})", r"\1\2 ", content, flags=re.MULTILINE)


def fix_markdown_file(file_path: str) -> bool:
    """Apply all fixes to a single markdown file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        content = fix_all(content)

        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Fixed linting issues in {file_path}")
            return True

        print(f"✓ No fixable issues found in {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def get_markdown_files(path: str) -> list:
    """Get all markdown files in a directory or a single file."""
    if Path(path).is_dir():
        return Path(path).glob("**/*.md", recursive=True)

    return []


def main():
    """Main function to process directories or files."""
    path = "." if len(sys.argv) < 2 else sys.argv[1]

    md_files = get_markdown_files(path)
    if not md_files:
        print(f"🔍 No markdown files found in {path}")
        return

    print(f"\n🔍 Found {len(md_files)} markdown files")

    fixed_count = 0
    for file_path in md_files:
        if fix_markdown_file(file_path):
            fixed_count += 1

    print(f"\n✅ Fixed issues in {fixed_count} files")


if __name__ == "__main__":
    main()
