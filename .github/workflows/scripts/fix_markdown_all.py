#!/usr/bin/env python3
"""
fix_markdown_all.py

A consolidated script that addresses:
- MD051: Invalid link fragments
- MD050: Strong style (use asterisks, not underscores)
- MD029: Ordered list prefix (incremental numbers)
- MD032/MD022: Spacing around elements (lists, headings, code blocks)
- MD040: Language specifiers for code blocks
- MD047: Single trailing newline
- And more as needed

Run it before or during CI to fix issues automatically.
"""

import re
import sys
from pathlib import Path


def extract_headings(content: str) -> dict:
    """
    Extract headings from content and create a mapping of heading text to valid fragment.
    Also creates variations of the heading text for better matching.
    """
    headings = {}
    heading_variations = {}

    for match in re.finditer(r"^(#+)\s+(.*?)$", content, re.MULTILINE):
        heading_text = match.group(2).strip()
        fragment = heading_text.lower()
        # Convert to lower case, remove punctuation, replace spaces with '-'
        fragment = re.sub(r"[^\w\s-]", "", fragment)  # remove punctuation
        fragment = re.sub(r"\s+", "-", fragment)  # spaces to dashes
        fragment = fragment.strip("-")  # remove leading/trailing dashes

        # Store the heading
        headings[heading_text] = fragment

        # Create variations of the heading for more robust matching
        # Simple lowercase version
        heading_variations[heading_text.lower()] = fragment

        # Version without special chars
        simple_heading = re.sub(r"[^\w\s-]", "", heading_text)
        heading_variations[simple_heading] = fragment
        heading_variations[simple_heading.lower()] = fragment

    return headings, heading_variations


def normalize_fragment_text(text: str) -> str:
    """
    Normalize fragment text to improve matching.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^\w\s-]", "", text)

    # Replace spaces with dashes
    text = re.sub(r"\s+", "-", text)

    # Remove leading/trailing dashes
    text = text.strip("-")

    return text


def fix_link_fragments(content: str) -> str:
    """
    Convert heading-based links to valid link fragments.
    Example:
      [Section](#my heading) -> [Section](#my-heading)
    """
    # Extract headings to create mappings
    headings, heading_variations = extract_headings(content)

    # Create a list of all section titles for case-insensitive matching
    sections_lowercase = {title.lower(): fragment for title, fragment in headings.items()}

    # Function to replace link fragments
    def _fix_fragment(match):
        link_text = match.group(1)
        fragment_text = match.group(2)

        # 1. Direct match with a heading
        if fragment_text in headings:
            return f"[{link_text}](#{headings[fragment_text]})"

        # 2. Match with a variation
        if fragment_text in heading_variations:
            return f"[{link_text}](#{heading_variations[fragment_text]})"

        # 3. Try case-insensitive match
        if fragment_text.lower() in sections_lowercase:
            return f"[{link_text}](#{sections_lowercase[fragment_text.lower()]})"

        # 4. Try matching with normalized fragment
        normalized = normalize_fragment_text(fragment_text)
        for section, fragment in sections_lowercase.items():
            if normalized == normalize_fragment_text(section):
                return f"[{link_text}](#{fragment})"

        # 5. Some common substitutions for known patterns
        if "using the ai framework integration" in fragment_text.lower():
            return f"[{link_text}](#using-the-ai-framework-integration)"
        if "working with langchain" in fragment_text.lower():
            return f"[{link_text}](#working-with-langchain)"
        if "working with stable diffusion" in fragment_text.lower():
            return f"[{link_text}](#working-with-stable-diffusion)"
        if "performance benchmarking" in fragment_text.lower():
            return f"[{link_text}](#performance-benchmarking)"
        if "troubleshooting" in fragment_text.lower():
            return f"[{link_text}](#troubleshooting)"
        if "advanced configuration" in fragment_text.lower():
            return f"[{link_text}](#advanced-configuration)"
        if "using lockfiles for reproducible environments" in fragment_text.lower():
            return f"[{link_text}](#using-lockfiles-for-reproducible-environments)"
        if "working with docker" in fragment_text.lower():
            return f"[{link_text}](#working-with-docker)"
        if "ci/cd pipeline updates" in fragment_text.lower() or "cicd pipeline updates" in fragment_text.lower():
            return f"[{link_text}](#cicd-pipeline-updates)"
        if "migration faqs" in fragment_text.lower():
            return f"[{link_text}](#migration-faqs)"

        # 6. For completely unmatched fragments, create a valid one
        valid_fragment = normalize_fragment_text(fragment_text)
        return f"[{link_text}](#{valid_fragment})"

    # Regex: captures link text and fragment
    pattern = r"\[(.*?)\]\(#(.*?)\)"
    content = re.sub(pattern, _fix_fragment, content)
    return content


def fix_strong_style(content: str) -> str:
    """
    MD050: Convert __text__ to **text**
    """
    return re.sub(r"__(.+?)__", r"**\1**", content)


def fix_ordered_lists(content: str) -> str:
    """
    MD029: Ensure ordered lists use incremental numbers (1/2/3...).
    """
    lines = content.splitlines()
    fixed_lines = []

    # Special handling for table of contents
    in_toc = False
    toc_counter = 0

    for i, line in enumerate(lines):
        # Check if we're in a table of contents section
        if re.match(r"^#+\s+Table\s+of\s+Contents", line, re.IGNORECASE):
            in_toc = True
            toc_counter = 0
            fixed_lines.append(line)
            continue

        # If in TOC, handle ordered list items specially
        if in_toc:
            # Detect a new heading which would end the TOC
            if line.startswith("#"):
                in_toc = False
                fixed_lines.append(line)
                continue

            # Detect list items in TOC
            match = re.match(r"^(\s*)(\d+)\.\s+(.*?)$", line)
            if match:
                toc_counter += 1
                indent, _, rest = match.groups()
                fixed_lines.append(f"{indent}{toc_counter}. {rest}")
                continue

        # Standard approach for non-TOC ordered lists
        match = re.match(r"^(\s*)(\d+)\.\s+(.*?)$", line)
        if match:
            # This is a list item, but we're not in TOC
            # Handle it according to normal list rules
            indent, number, rest = match.groups()

            # Try to find the previous list item to determine the number
            prev_number = 0
            for j in range(i - 1, -1, -1):
                prev_match = re.match(r"^(\s*)(\d+)\.\s+(.*?)$", lines[j])
                if prev_match and prev_match.group(1) == indent:
                    try:
                        prev_number = int(prev_match.group(2))
                        break
                    except ValueError:
                        pass

            # Increment the number
            current_number = prev_number + 1
            fixed_lines.append(f"{indent}{current_number}. {rest}")
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_spacing(content: str) -> str:
    """
    MD032/MD022: Ensure blank lines around headings, lists, code blocks, etc.
    """
    lines = content.splitlines()
    fixed_lines = []

    # Add blank line at the beginning if needed
    if (
        lines
        and lines[0].strip()
        and (lines[0].startswith("#") or re.match(r"^\d+\.", lines[0]) or lines[0].startswith("```"))
    ):
        fixed_lines.append("")

    for i, line in enumerate(lines):
        # Current line is a heading, list item, or code block start
        is_special_line = (
            line.lstrip().startswith("#") or re.match(r"^\s*\d+\.", line) or line.lstrip().startswith("```")
        )

        # If we need a blank line before this line
        if is_special_line and i > 0 and lines[i - 1].strip():
            # Skip adding blank line after heading if next line is a list or code block
            prev_is_heading = lines[i - 1].lstrip().startswith("#")
            if not (prev_is_heading and is_special_line):
                fixed_lines.append("")

        fixed_lines.append(line)

        # If we need a blank line after this line
        if is_special_line and i < len(lines) - 1 and lines[i + 1].strip():
            # Don't add blank line after heading if the next line is a list or code block
            next_is_special = (
                lines[i + 1].lstrip().startswith("#")
                or re.match(r"^\s*\d+\.", lines[i + 1])
                or lines[i + 1].lstrip().startswith("```")
            )
            if not (line.lstrip().startswith("#") and next_is_special):
                fixed_lines.append("")

    return "\n".join(fixed_lines)


def fix_code_blocks(content: str) -> str:
    """
    MD040: Add language specifiers if missing (e.g., ``` -> ```text).
    """
    # This pattern matches a triple backtick not followed by a language specifier
    return re.sub(r"```(\s*?)\n", "```text\n", content)


def ensure_trailing_newline(content: str) -> str:
    """
    MD047: Ensure file ends with a single newline.
    """
    content = content.rstrip("\n")  # Remove all trailing newlines
    return content + "\n"  # Add a single newline


def fix_markdown_file(file_path: Path) -> bool:
    """
    Apply all fix functions in sequence to a single file.
    Returns True if changes were made.
    """
    try:
        original_content = file_path.read_text(encoding="utf-8")
        content = original_content

        # Apply all fixes
        content = fix_link_fragments(content)
        content = fix_strong_style(content)
        content = fix_ordered_lists(content)
        content = fix_spacing(content)
        content = fix_code_blocks(content)
        content = ensure_trailing_newline(content)

        # Check if any changes were made
        changed = content != original_content
        if changed:
            file_path.write_text(content, encoding="utf-8")
            print(f"Fixed: {file_path}")

        return changed
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def main() -> int:
    """
    Run fix functions on all .md files in the repository.
    Exclude node_modules, .git, .venv, venv, etc.

    Can also specify specific files by passing them as command-line arguments.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    exclude_dirs = ["node_modules", ".git", ".venv", "venv", ".github/workflows-archive"]

    # If specific files were provided as arguments, only process those
    if len(sys.argv) > 1:
        md_files = [Path(file) for file in sys.argv[1:]]
    else:
        md_files = list(repo_root.glob("**/*.md"))

    fixed_count = 0
    total_files = 0

    for md_file in md_files:
        if any(excl in str(md_file) for excl in exclude_dirs):
            continue
        total_files += 1
        if fix_markdown_file(md_file):
            fixed_count += 1

    print(f"Processed {total_files} markdown files, fixed {fixed_count} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
