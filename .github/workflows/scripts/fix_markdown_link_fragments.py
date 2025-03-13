#!/usr/bin/env python3
"""
Script to automatically fix invalid link fragments in markdown files.
This script addresses the MD051 linting issue.
"""

import re
import sys
from pathlib import Path


def extract_headings(content):
    """Extract all headings from markdown content and return their IDs."""
    # Find all headings (lines starting with #)
    heading_pattern = r"^(#+)\s+(.+)$"
    headings = re.findall(heading_pattern, content, re.MULTILINE)

    heading_ids = {}
    for level, text in headings:
        # Convert heading text to ID format (lowercase, replace spaces with hyphens)
        heading_id = text.strip().lower().replace(" ", "-")
        # Remove special characters
        heading_id = re.sub(r"[^\w\-]", "", heading_id)
        heading_ids[text.strip()] = heading_id

    return heading_ids


def fix_link_fragments(file_path):
    """Fix invalid link fragments in a markdown file."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Extract headings and their IDs
    heading_ids = extract_headings(content)

    # Find all link fragments
    link_pattern = r"\[(.*?)\]\(#(.*?)\)"
    links = re.findall(link_pattern, content)

    modified = False
    for link_text, fragment in links:
        # Check if the fragment doesn't match any heading ID
        if fragment not in heading_ids.values():
            # Try to find a matching heading
            for heading_text, heading_id in heading_ids.items():
                # If the link text is similar to a heading text
                if link_text.lower() in heading_text.lower() or heading_text.lower() in link_text.lower():
                    # Replace the invalid fragment with the correct heading ID
                    content = content.replace(f"[{link_text}](#{fragment})", f"[{link_text}](#{heading_id})")
                    print(
                        f"Fixed link fragment in {file_path}: [{link_text}](#{fragment}) -> [{link_text}](#{heading_id})"
                    )
                    modified = True
                    break

    # Only write if changes were made
    if modified:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
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
    """Find and fix invalid link fragments in markdown files."""
    md_files = find_markdown_files()
    print(f"Found {len(md_files)} markdown files to check")

    fixed_count = 0
    for file_path in md_files:
        if fix_link_fragments(file_path):
            fixed_count += 1

    print(f"Fixed link fragments in {fixed_count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
