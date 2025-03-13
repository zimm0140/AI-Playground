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


def fix_link_fragments(file_path):
    """Fix link fragments in a markdown file by normalizing heading IDs and TOC links."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # First pass: update headings to use auto-generated GitHub-style IDs (except Table of Contents)
    heading_ids = {}
    heading_pattern = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+\{#([\w-]+)\})?$", re.MULTILINE)
    modified_content = content
    for match in heading_pattern.finditer(content):
        level, heading_text, existing_id = match.groups()
        if heading_text.strip() == "Table of Contents":
            continue
        computed_id = github_heading_id(heading_text)
        if existing_id != computed_id:
            new_heading = f"{level} {heading_text} {{#{computed_id}}}"
            modified_content = modified_content.replace(match.group(0), new_heading)
            print(f"Updated heading ID for '{heading_text}': {existing_id} -> {computed_id}")
        heading_ids[heading_text.strip()] = computed_id

    content = modified_content

    # Second pass: Update the Table of Contents links
    toc_pattern = re.compile(r"## Table of Contents.*?\n(.*?)(?=\n##|\Z)", re.DOTALL)
    toc_match = toc_pattern.search(content)
    if toc_match:
        toc_content = toc_match.group(1)
        link_pattern = re.compile(r"\[(.*?)\]\((#.*?)\)")
        new_toc = toc_content
        for match in link_pattern.finditer(toc_content):
            link_text, link_fragment = match.groups()
            clean_link_text = link_text.strip()
            if clean_link_text in heading_ids:
                new_fragment = f"#{heading_ids[clean_link_text]}"
                old_link = f"[{link_text}]({link_fragment})"
                new_link = f"[{link_text}]({new_fragment})"
                new_toc = new_toc.replace(old_link, new_link)
                print(f"Updated link: {old_link} -> {new_link}")
            else:
                for h_text, h_id in heading_ids.items():
                    if clean_link_text in h_text or h_text in clean_link_text:
                        new_fragment = f"#{h_id}"
                        old_link = f"[{link_text}]({link_fragment})"
                        new_link = f"[{link_text}]({new_fragment})"
                        new_toc = new_toc.replace(old_link, new_link)
                        print(f"Updated link (partial match): {old_link} -> {new_link}")
                        break
        content = content.replace(toc_match.group(1), new_toc)

    # Write the modified content back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Fixed link fragments in {file_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_link_fragments.py <markdown_file1> [<markdown_file2> ...]")
        sys.exit(1)

    for file_path in sys.argv[1:]:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File {file_path} does not exist.")
            continue

        fix_link_fragments(file_path)


if __name__ == "__main__":
    main()
