#!/usr/bin/env python3

import re
import sys
from pathlib import Path


def github_heading_id(text):
    """Generate a GitHub-style heading ID from heading text."""
    # Remove leading numbers and dots (e.g., "1. " or "1.1 ")
    text = re.sub(r"^\d+(\.\d+)*\s+", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation except hyphens
    text = re.sub(r"[^\w\s-]", "", text)

    # Replace spaces with hyphens
    text = re.sub(r"\s+", "-", text)

    # Remove consecutive hyphens
    text = re.sub(r"-+", "-", text)

    # Remove leading and trailing hyphens
    text = text.strip("-")

    return text


def fix_markdown_issues(file_path):
    """Fix common markdown issues including:
    - MD051: Link fragments should be valid
    - MD029: Ordered list item prefixes
    - MD022: Headings surrounded by blank lines
    - MD032: Lists surrounded by blank lines
    - MD047: Files end with a single newline
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Step 1: First ensure proper blank lines
    # Fix blank lines around headings (MD022)
    modified_content = re.sub(r"([^\n])\n(#{1,6}\s+)", r"\1\n\n\2", content)
    modified_content = re.sub(r"(#{1,6}\s+.*?)\n([^\n#])", r"\1\n\n\2", modified_content)

    # Fix blank lines around lists (MD032)
    modified_content = re.sub(r"([^\n])\n(\d+\.\s+)", r"\1\n\n\2", modified_content)
    modified_content = re.sub(r"([^\n])\n(-\s+)", r"\1\n\n\2", modified_content)
    modified_content = re.sub(r"(\d+\.\s+.*?)\n([^\n\d-])", r"\1\n\n\2", modified_content)
    modified_content = re.sub(r"(-\s+.*?)\n([^\n\d-])", r"\1\n\n\2", modified_content)

    # Step 2: Create a fresh table of contents with sequential numbering (MD029)
    # Find all headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*?)$", re.MULTILINE)
    headings = []

    # Get all headings except 'Table of Contents'
    for match in heading_pattern.finditer(modified_content):
        level, heading_text = match.groups()
        level_num = len(level)
        if "Table of Contents" not in heading_text and level_num > 1:  # Skip the main title and TOC
            headings.append((level_num, heading_text.strip()))

    # Find Table of Contents section
    toc_pattern = re.compile(r"## Table of Contents.*?\n(.*?)(?=\n##|\Z)", re.DOTALL)
    toc_match = toc_pattern.search(modified_content)

    if toc_match and headings:
        # Create new TOC with proper numbering and correct links
        new_toc_lines = []
        for i, (level, heading) in enumerate(headings, 1):
            if level == 2:  # Main sections (##)
                # Create a slug for the heading
                slug = re.sub(r"[^\w\s-]", "", heading).lower()
                slug = re.sub(r"\s+", "-", slug)
                new_toc_lines.append(f"{i}. [{heading}](#{slug})")
                print(f"Added TOC entry {i}: {heading}")

        # Replace old TOC with new one
        new_toc = "\n".join(new_toc_lines) + "\n\n"
        modified_content = re.sub(toc_pattern, f"## Table of Contents\n\n{new_toc}", modified_content)

        # Now add IDs to the main headings (level 2)
        for level, heading in headings:
            if level == 2:  # Main sections (##)
                slug = re.sub(r"[^\w\s-]", "", heading).lower()
                slug = re.sub(r"\s+", "-", slug)

                # Replace heading with one that has an ID
                original_heading = f"## {heading}"
                modified_heading = f"## {heading} {{#{slug}}}"
                modified_content = modified_content.replace(original_heading, modified_heading)
                print(f"Added ID to heading: {heading} -> {slug}")

    # Step 3: Fix nested ordered lists (MD029) in content - use the same number (1.) for all nested lists
    lines = modified_content.split("\n")
    for i in range(len(lines)):
        line = lines[i]
        if re.match(r"^\s*\d+\.", line):
            # Check if this is a nested list
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                # Replace number with 1.
                lines[i] = re.sub(r"^\s*\d+\.", f"{' ' * indent}1.", line)

    modified_content = "\n".join(lines)

    # Step 4: Ensure single trailing newline (MD047)
    modified_content = modified_content.rstrip("\n") + "\n"

    # Write the modified content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Fixed markdown issues in {file_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_markdown_issues.py <markdown_file1> [<markdown_file2> ...]")
        sys.exit(1)

    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File {file_path} does not exist.")
            continue

        fix_markdown_issues(file_path)


if __name__ == "__main__":
    main()
