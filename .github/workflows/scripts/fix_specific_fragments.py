#!/usr/bin/env python3
"""
fix_specific_fragments.py

A targeted script to fix known problematic link fragments in specific files.
This complements the more general fix_markdown_all.py script.
"""

import re
import sys
from pathlib import Path


def fix_hardware_guide(file_path):
    """Fix specific link fragments in HARDWARE_OPTIMIZATION_GUIDE.md"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Replace problematic fragments with correct ones
    replacements = {
        r"\[Using the AI Framework Integration\]\(#using-the-ai-framework-integration\)": r"[Using the AI Framework Integration](#using-the-ai-framework-integration)",
        r"\[Working with LangChain\]\(#working-with-langchain\)": r"[Working with LangChain](#working-with-langchain)",
        r"\[Working with Stable Diffusion\]\(#working-with-stable-diffusion\)": r"[Working with Stable Diffusion](#working-with-stable-diffusion)",
        r"\[Performance Benchmarking\]\(#performance-benchmarking\)": r"[Performance Benchmarking](#performance-benchmarking)",
        r"\[Troubleshooting\]\(#troubleshooting\)": r"[Troubleshooting](#troubleshooting)",
        r"\[Advanced Configuration\]\(#advanced-configuration\)": r"[Advanced Configuration](#advanced-configuration)",
    }

    # Apply each replacement
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    # Fix ordered list prefixes in the table of contents
    lines = content.splitlines()
    fixed_lines = []
    in_toc = False
    counter = 0

    for line in lines:
        # Check if we're in the Table of Contents section
        if re.match(r"^##\s+Table\s+of\s+Contents", line, re.IGNORECASE):
            in_toc = True
            counter = 0
            fixed_lines.append(line)
            continue

        # If in TOC, fix ordered list prefixes
        if in_toc:
            # Detect if we've left the TOC section
            if line.startswith("##"):
                in_toc = False
                fixed_lines.append(line)
                continue

            # Fix ordered list items
            match = re.match(r"^(\s*)(\d+)\.\s+(.*?)$", line)
            if match:
                counter += 1
                indent, _, rest = match.groups()
                fixed_lines.append(f"{indent}{counter}. {rest}")
                continue

        fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    # Save the fixed content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed: {file_path}")
    return True


def fix_migration_guide(file_path):
    """Fix specific link fragments in MIGRATION.md"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Replace problematic fragments with correct ones
    replacements = {
        r"\[Using Lockfiles for Reproducible Environments\]\(#using-lockfiles-for-reproducible-environments\)": r"[Using Lockfiles for Reproducible Environments](#using-lockfiles-for-reproducible-environments)",
        r"\[Working with Docker\]\(#working-with-docker\)": r"[Working with Docker](#working-with-docker)",
        r"\[CI/CD Pipeline Updates\]\(#cicd-pipeline-updates\)": r"[CI/CD Pipeline Updates](#cicd-pipeline-updates)",
        r"\[Migration FAQs\]\(#migration-faqs\)": r"[Migration FAQs](#migration-faqs)",
    }

    # Apply each replacement
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    # Fix ordered list numbering for problematic sections
    # The MD029 error indicates there are lists that have incorrect numbering
    # We'll look for these specific sections and fix them

    # Define sections and their corresponding patterns
    sections_to_fix = [
        (r"### Why Migrate to uv\?", r"### Step-by-Step Migration"),
        (r"### Step-by-Step Migration", r"## Using Lockfiles for Reproducible Environments"),
    ]

    # Process content section by section
    for start_pattern, end_pattern in sections_to_fix:
        # Extract the section
        start_match = re.search(start_pattern, content)
        end_match = re.search(end_pattern, content)

        if start_match and end_match and start_match.start() < end_match.start():
            section_start = start_match.end()
            section_end = end_match.start()

            section = content[section_start:section_end]

            # Fix ordered list numbering in this section
            fixed_section = fix_ordered_lists_in_section(section)

            # Replace the section in the content
            content = content[:section_start] + fixed_section + content[section_end:]

    # Save the fixed content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed: {file_path}")
    return True


def fix_ordered_lists_in_section(section):
    """Fix ordered list numbering in a specific section"""
    lines = section.splitlines()
    fixed_lines = []
    current_list = []
    current_indent = ""
    in_list = False

    for line in lines:
        list_match = re.match(r"^(\s*)(\d+)\.\s+(.*?)$", line)

        if list_match:
            # This is a list item
            indent = list_match.group(1)
            content = list_match.group(3)

            # Start or continue a list
            if not in_list or indent != current_indent:
                # If we were in a list, process and add it to fixed lines
                if in_list:
                    fixed_lines.extend(number_list_items(current_list, current_indent))
                    current_list = []

                # Start a new list
                in_list = True
                current_indent = indent
                current_list.append(content)
            else:
                # Continue the current list
                current_list.append(content)
        else:
            # Not a list item
            if in_list:
                # End of a list, process and add it
                fixed_lines.extend(number_list_items(current_list, current_indent))
                current_list = []
                in_list = False

            # Add the non-list line
            fixed_lines.append(line)

    # Handle any remaining list
    if in_list:
        fixed_lines.extend(number_list_items(current_list, current_indent))

    return "\n".join(fixed_lines)


def number_list_items(items, indent):
    """Number list items incrementally"""
    numbered_lines = []
    for i, item in enumerate(items, 1):
        numbered_lines.append(f"{indent}{i}. {item}")
    return numbered_lines


def main():
    """Main function to fix specific files"""
    # Check if specific files were provided as arguments
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default files to fix
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        files = [
            repo_root / "docs" / "hardware" / "HARDWARE_OPTIMIZATION_GUIDE.md",
            repo_root / "docs" / "development" / "MIGRATION.md",
        ]

    # Fix each file
    for file_path in files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            continue

        if file_path.name == "HARDWARE_OPTIMIZATION_GUIDE.md":
            fix_hardware_guide(file_path)
        elif file_path.name == "MIGRATION.md":
            fix_migration_guide(file_path)
        else:
            print(f"Skipping: {file_path} (not configured for specific fixes)")

    return 0


if __name__ == "__main__":
    sys.exit(main())