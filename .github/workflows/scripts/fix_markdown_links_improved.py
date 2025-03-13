#!/usr/bin/env python3
"""
Advanced script to fix invalid link fragments in markdown files.
This script addresses persistent MD051 linting issues by directly targeting
problematic files with custom fixes.
"""

import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_headings(file_content):
    """Extract headings from markdown content and map them to their link fragment form."""
    heading_pattern = r"^(#{1,6})\s+(.+)$"
    headings = {}

    for line in file_content.splitlines():
        match = re.match(heading_pattern, line.strip())
        if match:
            heading_level, heading_text = match.groups()
            # Create link fragment format (lowercase, spaces to dashes, remove special chars)
            fragment = heading_text.lower()
            fragment = re.sub(r"[^\w\s-]", "", fragment)  # Remove special characters
            fragment = re.sub(r"\s+", "-", fragment)  # Replace spaces with dashes
            headings[heading_text] = fragment

    return headings


def fix_blank_lines_around_headings(content):
    """Ensure headings are surrounded by blank lines."""
    # Find all headings
    heading_pattern = r"(^|\n)([^\n]+\n)(#{1,6}\s+.+)(\n[^\n]+)"

    # Replace with proper spacing
    def add_blank_lines(match):
        before = match.group(1) + match.group(2)
        heading = match.group(3)
        after = match.group(4)

        # If there's not already a blank line before
        if not before.endswith("\n\n"):
            before = before.rstrip("\n") + "\n\n"

        # If there's not already a blank line after
        if not after.startswith("\n"):
            after = "\n" + after

        return before + heading + after

    return re.sub(heading_pattern, add_blank_lines, content, flags=re.MULTILINE)


def fix_blank_lines_around_lists(content):
    """Ensure lists are surrounded by blank lines."""
    # Find start of lists
    list_start_pattern = r"(^|\n)([^\n-*\d]+\n)([-*]\s+.+|\d+\.\s+.+)"

    # Replace with proper spacing before lists
    def add_blank_line_before_list(match):
        before = match.group(1) + match.group(2)
        list_item = match.group(3)

        # If there's not already a blank line before
        if not before.endswith("\n\n"):
            before = before.rstrip("\n") + "\n\n"

        return before + list_item

    content = re.sub(list_start_pattern, add_blank_line_before_list, content, flags=re.MULTILINE)

    # Find end of lists
    list_end_pattern = r"(^|\n)([-*]\s+.+|\d+\.\s+.+)(\n)([^-*\d\n])"

    # Replace with proper spacing after lists
    def add_blank_line_after_list(match):
        before = match.group(1)
        list_item = match.group(2)
        newline = match.group(3)
        after = match.group(4)

        return before + list_item + "\n\n" + after

    return re.sub(list_end_pattern, add_blank_line_after_list, content, flags=re.MULTILINE)


def fix_blank_lines_around_code_blocks(content):
    """Ensure code blocks are surrounded by blank lines."""
    # Find code blocks without proper spacing
    code_block_pattern = r"(^|\n)([^\n`]+\n)(```[\w]*\n.*?\n```)(\n[^\n`]+)"

    # Replace with proper spacing
    def add_blank_lines_around_code(match):
        before = match.group(1) + match.group(2)
        code_block = match.group(3)
        after = match.group(4)

        # If there's not already a blank line before
        if not before.endswith("\n\n"):
            before = before.rstrip("\n") + "\n\n"

        # If there's not already a blank line after
        if not after.startswith("\n"):
            after = "\n" + after

        return before + code_block + after

    return re.sub(code_block_pattern, add_blank_lines_around_code, content, flags=re.DOTALL | re.MULTILINE)


def fix_hardware_optimization_guide(file_path):
    """Fix link fragments in the hardware optimization guide."""
    print(f"Fixing link fragments in {file_path}")

    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # Extract headings and their corresponding fragments
    headings = extract_headings(content)

    # Find and update TOC
    toc_pattern = r"(## Table of Contents\s+)(?:\d+\.\s+\[(.*?)\]\(#.*?\)\s*)+"
    toc_match = re.search(toc_pattern, content, re.DOTALL)

    if toc_match:
        toc_start = toc_match.group(1)
        new_toc = toc_start + "\n"  # Add extra newline for spacing

        # Extract TOC entries
        entry_pattern = r"\d+\.\s+\[(.*?)\]\(#.*?\)"
        entries = re.findall(entry_pattern, toc_match.group(0))

        # Rebuild TOC with correct fragments
        for i, entry in enumerate(entries, 1):
            if entry in headings:
                new_toc += f"{i}. [{entry}](#{headings[entry]})\n"
            else:
                print(f"Warning: Could not find heading for TOC entry: {entry}")
                # Create a reasonable fallback
                fallback_fragment = entry.lower().replace(" ", "-")
                fallback_fragment = re.sub(r"[^\w\-]", "", fallback_fragment)
                new_toc += f"{i}. [{entry}](#{fallback_fragment})\n"

        # Add blank line after TOC
        new_toc += "\n"

        # Replace old TOC with new one
        content = content.replace(toc_match.group(0), new_toc)

    # Fix spacing around headings and lists
    content = fix_blank_lines_around_headings(content)
    content = fix_blank_lines_around_lists(content)
    content = fix_blank_lines_around_code_blocks(content)

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def fix_api_reference(file_path):
    """Fix link fragments in the API reference file."""
    print(f"Fixing link fragments in {file_path}")

    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # Extract headings and their corresponding fragments
    headings = extract_headings(content)

    # Find and update TOC
    toc_section = re.search(r"## Table of Contents\s+(.*?)##", content, re.DOTALL)
    if toc_section:
        toc_content = toc_section.group(1)

        # Extract TOC entries
        toc_entry_pattern = r"[-*]\s+\[(.*?)\]\(#.*?\)"
        toc_entries = re.findall(toc_entry_pattern, toc_content)

        new_toc_content = "## Table of Contents\n\n"

        # Rebuild TOC with correct fragments
        for entry in toc_entries:
            # Find the corresponding heading section
            for heading_text, fragment in headings.items():
                if heading_text.lower() == entry.lower() or heading_text.lower().startswith(entry.lower()):
                    new_toc_content += f"- [{entry}](#{fragment})\n"
                    break
            else:
                print(f"Warning: Could not find heading for TOC entry: {entry}")
                # Create a reasonable fallback
                fallback_fragment = entry.lower().replace(" ", "-")
                fallback_fragment = re.sub(r"[^\w\-]", "", fallback_fragment)
                new_toc_content += f"- [{entry}](#{fallback_fragment})\n"

        # Add blank line after TOC
        new_toc_content += "\n"

        # Replace old TOC with new one
        content = content.replace(toc_section.group(1), new_toc_content)

    # Fix spacing around headings and lists
    content = fix_blank_lines_around_headings(content)
    content = fix_blank_lines_around_lists(content)
    content = fix_blank_lines_around_code_blocks(content)

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def fix_migration_guide(file_path):
    """Fix link fragments in the migration guide."""
    print(f"Fixing link fragments in {file_path}")

    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # Extract headings and their corresponding fragments
    headings = extract_headings(content)

    # Find and update TOC entries
    toc_pattern = r"\[(.*?)\]\(#.*?\)"

    def replace_link(match):
        link_text = match.group(1)

        # Try to find a matching heading
        for heading_text, fragment in headings.items():
            if heading_text.lower() == link_text.lower() or heading_text.lower().startswith(link_text.lower()):
                return f"[{link_text}](#{fragment})"

            # Handle special cases like "CI/CD Pipeline Updates" -> "CI/CD Pipeline"
            if "CI/CD Pipeline" in heading_text and "CI/CD Pipeline" in link_text:
                return f"[{link_text}](#{fragment})"

        # If no matching heading, create a reasonable fallback
        print(f"Warning: Could not find heading for link: {link_text}")
        fallback_fragment = link_text.lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "")
        fallback_fragment = re.sub(r"[^\w\s-]", "", fallback_fragment)
        fallback_fragment = re.sub(r"\s+", "-", fallback_fragment)
        return f"[{link_text}](#{fallback_fragment})"

    # Replace all links in the content
    updated_content = re.sub(toc_pattern, replace_link, content)

    # Fix spacing around headings and lists
    updated_content = fix_blank_lines_around_headings(updated_content)
    updated_content = fix_blank_lines_around_lists(updated_content)
    updated_content = fix_blank_lines_around_code_blocks(updated_content)

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)


def main():
    """Main function to fix markdown link fragments."""
    base_dir = os.getcwd()

    # Fix Hardware Optimization Guide
    hardware_guide_path = os.path.join(base_dir, "docs", "hardware", "HARDWARE_OPTIMIZATION_GUIDE.md")
    if os.path.exists(hardware_guide_path):
        fix_hardware_optimization_guide(hardware_guide_path)

    # Fix API Reference
    api_reference_path = os.path.join(base_dir, "docs", "reference", "api.md")
    if os.path.exists(api_reference_path):
        fix_api_reference(api_reference_path)

    # Fix Migration Guide
    migration_guide_path = os.path.join(base_dir, "docs", "development", "MIGRATION.md")
    if os.path.exists(migration_guide_path):
        fix_migration_guide(migration_guide_path)

    print(
        f"Fixed link fragments and formatting in a total of {sum(1 for p in [hardware_guide_path, api_reference_path, migration_guide_path] if os.path.exists(p))} files.",
    )


if __name__ == "__main__":
    main()
