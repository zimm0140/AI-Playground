#!/usr/bin/env python3
"""
Script to automatically fix common markdown linting issues.
"""
import os
import re
import sys
from re import Pattern


def fix_trailing_spaces(content: str) -> str:
    """Remove trailing spaces at the end of lines."""
    lines = content.splitlines()
    fixed_lines: list[str] = [line.rstrip() for line in lines]
    return "\n".join(fixed_lines) + "\n"


def fix_consecutive_blank_lines(content: str) -> str:
    """Ensure no more than one consecutive blank line."""
    pattern: Pattern[str] = re.compile(r"\n{3,}")
    return pattern.sub("\n\n", content)


def fix_emphasis_as_heading(content: str) -> str:
    """Convert emphasis used as heading to proper headings."""
    lines = content.splitlines()
    fixed_lines: list[str] = []

    for line in lines:
        # Match emphasis patterns used at the start of a line (**text** or __text__)
        if re.match(r"^\s*(\*\*|__).+(\*\*|__)\s*$", line):
            # Extract text from emphasis
            emphasis_text = re.sub(r"^\s*(\*\*|__)(.*?)(\*\*|__)\s*$", r"\2", line)
            # Replace with heading (level 3)
            fixed_lines.append(f"### {emphasis_text}")
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines) + "\n"


def fix_line_length(content: str, max_length: int = 180) -> str:
    """Break long lines at appropriate boundaries."""
    lines = content.splitlines()
    fixed_lines: list[str] = []

    for line in lines:
        # Skip headings, code blocks, and tables
        if (
            line.startswith("#")
            or line.startswith("```")
            or line.startswith("|")
            or len(line) <= max_length
        ):
            fixed_lines.append(line)
        else:
            # Try to break at punctuation or spaces
            current_pos = 0
            while current_pos < len(line):
                if current_pos + max_length >= len(line):
                    fixed_lines.append(line[current_pos:])
                    break

                # Find a good breaking point
                break_pos = line.rfind(" ", current_pos, current_pos + max_length)
                if break_pos == -1 or break_pos <= current_pos:
                    # No space found, just break at max_length
                    break_pos = current_pos + max_length

                fixed_lines.append(line[current_pos:break_pos])
                current_pos = break_pos + 1  # Skip the space

    return "\n".join(fixed_lines) + "\n"


def fix_first_line_heading(content: str) -> str:
    """Ensure first line is a top-level heading if it's not."""
    lines = content.splitlines()
    if not lines:
        return content

    first_line = lines[0]
    if not first_line.startswith("# "):
        # Check if there's any heading in the first 3 lines
        has_heading = any(line.startswith("#") for line in lines[:3])

        if not has_heading:
            # Extract a title from the filename or first line
            title = (
                os.path.basename(sys.argv[1])
                .replace(".md", "")
                .replace("-", " ")
                .replace("_", " ")
                .title()
            )
            lines.insert(0, f"# {title}")

    return "\n".join(lines) + "\n"


def fix_ordered_list_prefixes(content: str) -> str:
    """Preserve sequential numbering for ordered lists."""
    lines = content.splitlines()
    fixed_lines: list[str] = []

    # We're not modifying the ordered list prefixes anymore
    # Just pass through the content
    return content


def fix_code_blocks(content: str) -> str:
    """Add language specifiers to fenced code blocks."""
    lines = content.splitlines()
    fixed_lines: list[str] = []
    in_code_block = False

    for line in lines:
        # Just pass through code blocks without modification
        fixed_lines.append(line)

        # Keep track of code block state to handle nested blocks properly
        if line.strip() == "```" and not in_code_block:
            in_code_block = True
        elif line.startswith("```") and not in_code_block:
            in_code_block = True
        elif line.strip() == "```" and in_code_block:
            in_code_block = False

    return "\n".join(fixed_lines) + "\n"


def fix_markdown_file(file_path: str, dry_run: bool = False) -> None:
    """Apply all fixes to a markdown file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply all fixes
        content = fix_trailing_spaces(content)
        content = fix_consecutive_blank_lines(content)
        content = fix_emphasis_as_heading(content)
        content = fix_line_length(content)
        # Skip ordered list prefix fixing
        # content = fix_ordered_list_prefixes(content)
        # Skip code block language specifier addition
        # content = fix_code_blocks(content)

        # Only write if changes were made and not in dry-run mode
        if content != original_content:
            if dry_run:
                print(f"Would fix issues in {file_path}")
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Fixed issues in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)


def find_markdown_files(directory: str) -> list[str]:
    """Recursively find all markdown files in the given directory."""
    result: list[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".md"):
                result.append(os.path.join(root, file))
    return result


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    if len(sys.argv) > 1 and sys.argv[1] != "--dry-run":
        directory = sys.argv[1]
    else:
        directory = "."

    markdown_files = find_markdown_files(directory)
    for file_path in markdown_files:
        fix_markdown_file(file_path, dry_run)

    print(f"Processed {len(markdown_files)} markdown files")
