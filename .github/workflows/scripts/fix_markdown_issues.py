#!/usr/bin/env python3
"""
Script to fix common markdown issues in files.
This is used by the CI workflow to automatically fix simple markdown problems.
"""

import os
import re
from pathlib import Path
from typing import List


def github_heading_id(text: str) -> str:
    """Generate GitHub-style heading ID from heading text."""
    # Remove leading numbers and periods
    text = re.sub(r"^\d+\.?\s+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove any character that is not alphanumeric, space, or hyphen
    text = re.sub(r"[^\w\s-]", "", text)

    # Replace spaces with hyphens
    text = re.sub(r"\s+", "-", text)

    return text


def fix_markdown_file(file_path: str) -> bool:
    """Fix common issues in a markdown file."""
    print(f"Processing {file_path}...")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Store original content to check if changes were made
    original_content = content

    # Fix 1: Table of Contents deduplication and correct numbering
    content = fix_table_of_contents(content)

    # Fix 2: Code block syntax
    content = fix_code_blocks(content)

    # Fix 3: Ordered list numbering
    content = fix_ordered_lists(content)

    # Fix 4: Heading IDs and link fragments
    content = fix_link_fragments(content)

    # Fix 5: Table formatting
    content = fix_table_formatting(content)

    # Fix 6: Ensure proper spacing around headings
    content = fix_blanks_around_headings(content)

    # Fix 7: Ensure proper spacing around lists
    content = fix_blanks_around_lists(content)

    # Fix 8: Ensure proper spacing around fenced code blocks
    content = fix_blanks_around_fences(content)

    # Fix 9: Add language to fenced code blocks
    content = fix_fenced_code_language(content)

    # Fix 10: Remove trailing whitespace
    content = fix_trailing_whitespace(content)

    # Fix 11: Remove multiple consecutive blank lines
    content = fix_consecutive_blank_lines(content)

    # Fix 12: Ensure single trailing newline
    content = fix_file_ending(content)

    # Write changes if content was modified
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✓ Fixed issues in {file_path}")
        return True
    print(f"✓ No issues to fix in {file_path}")
    return False


def fix_table_of_contents(content: str) -> str:
    """Fix duplicate and incorrectly numbered Table of Contents sections."""
    # Check if there are multiple TOC sections
    toc_sections = re.findall(r"## Table of Contents.*?(?=\n##|\Z)", content, re.DOTALL)

    if len(toc_sections) <= 1:
        return content

    # Extract all heading links from the TOC sections
    all_links = []
    for section in toc_sections:
        links = re.findall(r"\d+\.\s+\[(.*?)\]\((.*?)\)", section)
        all_links.extend(links)

    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for text, url in all_links:
        if url not in seen:
            unique_links.append((text, url))
            seen.add(url)

    # Create a new TOC with sequential numbering
    new_toc = "## Table of Contents {#table-of-contents}\n\n"
    for i, (text, url) in enumerate(unique_links, 1):
        new_toc += f"{i}. [{text}]({url})\n"
    new_toc += "\n"

    # Replace all TOC sections with the new one
    for section in toc_sections:
        content = content.replace(section, new_toc)
        # Only replace the first occurrence to avoid further duplication
        break

    # Remove any remaining TOC sections
    for section in toc_sections[1:]:
        content = content.replace(section, "")

    return content


def fix_code_blocks(content: str) -> str:
    """Fix code block syntax issues."""
    # Replace ```text with just ``` (when it's used as a closing tag)
    content = re.sub(r"```text\s*$", "```", content, flags=re.MULTILINE)

    # Fix improper or malformed code blocks
    content = re.sub(r"```\s*\n\s*```", "```text\n\n```", content)

    return content


def fix_ordered_lists(content: str) -> str:
    """Fix ordered list numbering."""
    # Identify sections with ordered lists
    sections = re.split(r"(?=^##? .*$)", content, flags=re.MULTILINE)
    fixed_sections = []

    for section in sections:
        # Skip if no ordered list in section
        if not re.search(r"^\d+\. ", section, re.MULTILINE):
            fixed_sections.append(section)
            continue

        # Split section into lines
        lines = section.split("\n")
        fixed_lines = []
        counter = 1
        in_ordered_list = False

        for line in lines:
            # Check if line starts with a number and period
            list_match = re.match(r"^(\d+)\. (.*)", line)
            if list_match:
                if not in_ordered_list:
                    in_ordered_list = True
                    counter = 1
                line = f"{counter}. {list_match.group(2)}"
                counter += 1
            elif line.strip() == "" and in_ordered_list:
                # A blank line might end the list
                in_ordered_list = False
            elif in_ordered_list and line.strip() and not line.strip().startswith("   "):
                # If we hit a non-blank, non-indented line, the list is over
                in_ordered_list = False

            fixed_lines.append(line)

        fixed_sections.append("\n".join(fixed_lines))

    return "".join(fixed_sections)


def fix_link_fragments(content: str) -> str:
    """Fix heading IDs and link fragments."""
    # Find all headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+\{#(.*?)\})?\s*$", re.MULTILINE)
    headings = list(heading_pattern.finditer(content))

    # If no headings, return original content
    if not headings:
        return content

    # Generate IDs for headings that don't have them
    heading_replacements = {}
    for match in headings:
        level, text, existing_id = match.groups()
        if not existing_id:
            new_id = github_heading_id(text)
            new_heading = f"{level} {text} {{#{new_id}}}"
            heading_replacements[match.group(0)] = new_heading

    # Apply heading replacements
    for old, new in heading_replacements.items():
        content = content.replace(old, new)

    # Fix malformed heading IDs with duplicate or nested IDs
    pattern = re.compile(r"(#{1,6}\s+.*?)\s+\{#(.*?)\s+\{#(.*?)\}\s*\}", re.MULTILINE)
    content = pattern.sub(r"\1 {#\3}", content)

    # Fix link fragments in the content
    link_pattern = re.compile(r"\[(.*?)\]\(#(.*?)\)")
    for match in link_pattern.finditer(content):
        link_text, fragment = match.groups()

        # Check if fragment exists in any heading ID
        found = False
        heading_ids = []

        for heading_match in headings:
            _, heading_text, heading_id = heading_match.groups()
            heading_ids.append(heading_id or github_heading_id(heading_text))

            # If heading has explicit ID, use that
            if heading_id and heading_id == fragment:
                found = True
                break

            # If no explicit ID, check the auto-generated ID
            if not heading_id and github_heading_id(heading_text) == fragment:
                found = True
                break

        if not found:
            # Try to find the heading text and generate proper fragment
            for heading_match in headings:
                _, heading_text, _ = heading_match.groups()
                if heading_text.lower() == link_text.lower():
                    new_fragment = github_heading_id(heading_text)
                    if new_fragment in heading_ids:
                        old_link = f"[{link_text}](#{fragment})"
                        new_link = f"[{link_text}](#{new_fragment})"
                        content = content.replace(old_link, new_link)
                    break

    return content


def fix_table_formatting(content: str) -> str:
    """Fix table formatting issues."""
    # Match markdown tables (standard format)
    table_pattern = re.compile(r"^([|]?.*[|].*[|]?)\s*\n([|]?[ :]*[-]+[ :]*[|][ :]*[-]+[ :]*.*[|]?)\s*\n", re.MULTILINE)

    # Find all tables
    for match in table_pattern.finditer(content):
        header_row = match.group(1)
        separator_row = match.group(2)

        # Check if header has leading and trailing pipes
        has_leading_pipe = header_row.startswith("|")
        has_trailing_pipe = header_row.endswith("|")

        # Check if separator has leading and trailing pipes
        sep_has_leading_pipe = separator_row.startswith("|")
        sep_has_trailing_pipe = separator_row.endswith("|")

        # If any row is missing leading or trailing pipes, fix all rows
        if not (has_leading_pipe and has_trailing_pipe and sep_has_leading_pipe and sep_has_trailing_pipe):
            # Extract the table content
            # Find the end of the table by looking for lines with pipes
            lines = content[match.start() :].split("\n")
            table_end_line = 2  # We already matched header and separator rows

            # Process remaining rows
            for i, line in enumerate(lines[2:], 2):
                if "|" in line:
                    table_end_line = i + 1
                else:
                    break

            # Extract the table
            table_text = "\n".join(lines[:table_end_line])

            # Process table rows
            fixed_rows = []
            rows = table_text.split("\n")

            # Determine column count by inspecting separator row
            sep_parts = separator_row.strip("|").split("|")
            column_count = len(sep_parts)

            for i, row in enumerate(rows):
                # Clean the row by removing leading/trailing pipes and spaces
                clean_row = row.strip()
                if not clean_row.startswith("|"):
                    clean_row = "|" + clean_row
                if not clean_row.endswith("|"):
                    clean_row = clean_row + "|"

                # Ensure correct number of columns
                row_parts = clean_row.strip("|").split("|")
                if len(row_parts) < column_count:
                    # Add missing cells
                    for _ in range(column_count - len(row_parts)):
                        clean_row = clean_row[:-1] + " |"

                fixed_rows.append(clean_row)

            # Replace the original table with the fixed one
            fixed_table = "\n".join(fixed_rows)
            content = content[: match.start()] + fixed_table + content[match.start() + len(table_text) :]

    return content


def fix_blanks_around_headings(content: str) -> str:
    """Fix blank lines around headings."""
    # Match headings (with or without IDs)
    heading_pattern = re.compile(r"^(#{1,6}\s+.*?(?:\s+\{#.*?\})?)$", re.MULTILINE)

    # Find all headings
    matches = list(heading_pattern.finditer(content))

    # Process in reverse to avoid position shifts
    for match in reversed(matches):
        start = match.start()
        end = match.end()

        # Check if there's a blank line before the heading
        # (unless it's at the start of the file)
        if start > 0:
            # Look back to find the previous non-blank line
            prev_text = content[:start].rstrip()
            if prev_text and not prev_text.endswith("\n\n"):
                # Fix: add blank line before heading
                content = content[:start] + "\n" + content[start:]
                # Adjust end position
                end += 1

        # Check if there's a blank line after the heading
        # Look ahead to find the next line
        if end < len(content) and not content[end : end + 2].startswith("\n\n"):
            # Fix: add blank line after heading
            content = content[:end] + "\n" + content[end:]

    return content


def fix_blanks_around_lists(content: str) -> str:
    """Fix blank lines around lists."""
    # Match list markers (ordered and unordered)
    list_markers = [
        r"^\d+\.\s",  # Ordered list
        r"^[-*+]\s",  # Unordered list
    ]

    # First pass: add blank lines before list starts
    for marker_pattern in list_markers:
        pattern = re.compile(marker_pattern, re.MULTILINE)

        # Find all list starts
        matches = list(pattern.finditer(content))

        # Process in reverse to avoid position shifts
        for match in reversed(matches):
            start = match.start()

            # Skip if at start of file
            if start == 0:
                continue

            # Check if there's a blank line before the list start
            prev_text = content[:start].rstrip()
            if prev_text and not prev_text.endswith("\n\n") and not prev_text.endswith(":\n"):
                # Don't add blank line if preceded by a colon (likely part of a description)
                if not re.search(r":\s*$", content[:start].split("\n")[-1]):
                    # Fix: add blank line before list
                    content = content[:start] + "\n" + content[start:]

    # Second pass: add blank lines after lists - analyze line by line
    lines = content.split("\n")
    result_lines = []
    in_list = False
    list_indent = 0

    for i, line in enumerate(lines):
        is_list_item = any(re.match(pattern, line.lstrip()) for pattern in list_markers)
        current_indent = len(line) - len(line.lstrip())

        # Detect list start
        if is_list_item and not in_list:
            in_list = True
            list_indent = current_indent
            result_lines.append(line)
        # Detect list continuation (indented content or list items at same indent level)
        elif in_list and (
            not line.strip() or current_indent > list_indent or (current_indent == list_indent and is_list_item)
        ):
            result_lines.append(line)
        # Detect list end - content that's not part of the list
        elif in_list:
            in_list = False
            # Check if we need a blank line
            if result_lines and result_lines[-1].strip():
                result_lines.append("")
            result_lines.append(line)
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def fix_blanks_around_fences(content: str) -> str:
    """Fix blank lines around fenced code blocks."""
    # Find positions of all code fence markers
    fence_pattern = re.compile(r"^```\w*\s*$", re.MULTILINE)
    positions = [(m.start(), m.group(0)) for m in fence_pattern.finditer(content)]

    # Skip if fewer than 2 fence markers found
    if len(positions) < 2:
        return content

    # Group fences into opening/closing pairs
    fence_pairs = []
    open_fence = None

    for pos, fence in positions:
        if open_fence is None:
            open_fence = (pos, fence)
        else:
            fence_pairs.append((open_fence, (pos, fence)))
            open_fence = None

    # Process fence pairs from the end to avoid position shifts
    adjustments = 0
    for (start_pos, start_fence), (end_pos, end_fence) in reversed(fence_pairs):
        # Adjust positions for previous insertions
        start_pos += adjustments
        end_pos += adjustments

        # Check for blank line before opening fence
        if start_pos > 0:
            # Look back to find the last non-blank line
            prev_text = content[:start_pos].rstrip()
            if prev_text and not prev_text.endswith("\n\n"):
                # Add blank line before fence
                content = content[:start_pos] + "\n" + content[start_pos:]
                adjustments += 1
                end_pos += 1  # Adjust end position

        # Check for blank line after closing fence
        fence_end = end_pos + len(end_fence)
        if fence_end < len(content):
            # Look ahead to see if there's a blank line
            if not content[fence_end : fence_end + 2].startswith("\n\n"):
                # Add blank line after fence
                content = content[:fence_end] + "\n" + content[fence_end:]
                adjustments += 1

    return content


def fix_fenced_code_language(content: str) -> str:
    """Add language specifier to fenced code blocks."""
    # Match opening code fences without language
    fence_pattern = re.compile(r"^```\s*$", re.MULTILINE)

    # Find all fences without language specifier
    matches = list(fence_pattern.finditer(content))

    # Process in reverse to avoid position shifts
    for match in reversed(matches):
        # Replace with default language 'text'
        content = content[: match.start()] + "```text" + content[match.end() :]

    return content


def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from all lines."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        fixed_lines.append(line.rstrip())

    return "\n".join(fixed_lines)


def fix_consecutive_blank_lines(content: str) -> str:
    """Remove multiple consecutive blank lines."""
    # Replace 3 or more consecutive blank lines with 2 blank lines
    return re.sub(r"\n{3,}", "\n\n", content)


def fix_file_ending(content: str) -> str:
    """Ensure single trailing newline at end of file."""
    # Remove all trailing newlines
    content = content.rstrip("\n")

    # Add a single newline
    return content + "\n"


def find_markdown_files(directory: str = ".", recursive: bool = True, exclude_patterns=None) -> List[str]:
    """Find all markdown files in the directory."""
    if exclude_patterns is None:
        exclude_patterns = []

    path = Path(directory)
    if recursive:
        files = [str(file) for file in path.glob("**/*.md")]
    else:
        files = [str(file) for file in path.glob("*.md")]

    # Apply exclusion patterns
    for pattern in exclude_patterns:
        files = [f for f in files if pattern not in f]

    return files


def main() -> None:
    """Main function to run the script."""
    # Default to fixing docs directory if no arguments
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "docs")

    # Find and fix markdown files
    exclude_patterns = ["node_modules", ".git", ".venv", "venv"]
    md_files = find_markdown_files(docs_dir, recursive=True, exclude_patterns=exclude_patterns)

    print(f"Found {len(md_files)} markdown files to process")

    fixed_count = 0
    for file_path in md_files:
        if fix_markdown_file(file_path):
            fixed_count += 1

    print(f"\nFixed {fixed_count} out of {len(md_files)} markdown files")

    if fixed_count > 0:
        print("✅ Successfully fixed markdown issues")
    else:
        print("✓ No issues found to fix")


if __name__ == "__main__":
    main()
