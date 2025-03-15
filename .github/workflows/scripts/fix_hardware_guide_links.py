#!/usr/bin/env python3
"""
fix_hardware_guide_links.py

A simple script to directly fix the link fragments in HARDWARE_OPTIMIZATION_GUIDE.md
"""

import sys
from pathlib import Path


def fix_hardware_guide_links(file_path):
    """Fix the link fragments in HARDWARE_OPTIMIZATION_GUIDE.md"""
    try:
        # Read the file
        content = Path(file_path).read_text(encoding="utf-8")

        # Define the replacements
        link_fixes = [
            # Raw string with escaping
            (
                r"\[Using the AI Framework Integration\]\(#using-the-ai-framework-integration\)",
                "[Using the AI Framework Integration](#using-the-ai-framework-integration)",
            ),
            (
                r"\[Working with LangChain\]\(#working-with-langchain\)",
                "[Working with LangChain](#working-with-langchain)",
            ),
            (
                r"\[Working with Stable Diffusion\]\(#working-with-stable-diffusion\)",
                "[Working with Stable Diffusion](#working-with-stable-diffusion)",
            ),
            (
                r"\[Performance Benchmarking\]\(#performance-benchmarking\)",
                "[Performance Benchmarking](#performance-benchmarking)",
            ),
            (r"\[Troubleshooting\]\(#troubleshooting\)", "[Troubleshooting](#troubleshooting)"),
            (
                r"\[Advanced Configuration\]\(#advanced-configuration\)",
                "[Advanced Configuration](#advanced-configuration)",
            ),
        ]

        # Apply the replacements
        for old, new in link_fixes:
            content = content.replace(old, new)

        # Ensure file ends with a single newline
        if not content.suffix == '.\n'):
            content += "\n"

        # Write the file back
        Path(file_path).write_text(content, encoding="utf-8")
        print(f"Fixed link fragments in {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {e}", file=sys.stderr)
        return False


def main():
    """Main function"""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    file_path = repo_root / "docs" / "hardware" / "HARDWARE_OPTIMIZATION_GUIDE.md"

    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    if fix_hardware_guide_links(file_path):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())