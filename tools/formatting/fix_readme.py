#!/usr/bin/env python3
"""
Script to fix specific formatting issues in readme.md
"""

import re


def fix_readme():
    """Apply specific fixes to readme.md."""
    try:
        with open("readme.md", "r", encoding="utf-8") as f:
            content = f.read()

        # Fix HTML tags for badges
        content = re.sub(r'<a href="<(https?://[^>]+)>">', r'<a href="\1">', content)
        content = re.sub(r'src="<([^>]+)>/>"', r'src="\1" />', content)

        # Fix URL formatting
        content = re.sub(r"(?<!\]\()(?<!\<)(https?://[^\s<>]+)(?!\>)", r"<\1>", content)

        # Fix hard tabs and ensure proper indentation
        lines = []
        for line in content.split("\n"):
            # Replace tabs with spaces
            line = line.replace("\t", "    ")
            # Remove strange characters that appear at line endings
            line = re.sub(r"\s+$", "", line)
            lines.append(line)

        content = "\n".join(lines)

        # Fix list formatting
        content = re.sub(r"^\*\t", "- ", content, flags=re.MULTILINE)
        content = re.sub(r"^\*\s", "- ", content, flags=re.MULTILINE)

        # Ensure headings have spaces after #
        content = re.sub(r"^(#+)([^\s])", r"\1 \2", content, flags=re.MULTILINE)

        # Remove trailing punctuation in headers
        content = re.sub(r"^(#+\s+.*?)[.:;,!?](\s*)$", r"\1\2", content, flags=re.MULTILINE)

        # Fix multiple consecutive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Fix heading spacing
        lines = content.split("\n")
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # Check if this is a heading
            if re.match(r"^#+\s+", line):
                # Add blank line before heading if not at start and previous line not blank
                if i > 0 and result and result[-1].strip():
                    result.append("")

                # Add the heading
                result.append(line)

                # Add blank line after heading if next line not blank
                if i < len(lines) - 1 and lines[i + 1].strip():
                    result.append("")
            else:
                result.append(line)
            i += 1

        content = "\n".join(result)

        # Fix list numbering
        lines = content.split("\n")
        result = []
        list_counter = 0
        in_list = False

        for line in lines:
            # Check if this is a numbered list item
            ordered_list_match = re.match(r"^(\s*)(\d+)\.(\s+)(.*)", line)

            if ordered_list_match:
                indent, num, spaces, text = ordered_list_match.groups()
                if not in_list:
                    list_counter = 1
                    in_list = True
                else:
                    list_counter += 1

                # Replace with correct numbering
                line = f"{indent}{list_counter}.{spaces}{text}"
            elif line.strip() == "":
                # Reset list counter on blank line
                in_list = False
            elif not re.match(r"^\s*-\s+", line):
                # Not a list item and not a blank line - end of list
                in_list = False

            result.append(line)

        content = "\n".join(result)

        # Ensure single trailing newline
        content = content.rstrip("\n") + "\n"

        # Fix double hash in headers (# # to #)
        content = re.sub(r"^#\s+#\s+", "## ", content, flags=re.MULTILINE)

        with open("readme.md", "w", encoding="utf-8") as f:
            f.write(content)

        print("✅ Successfully fixed readme.md")

    except Exception as e:
        print(f"❌ Error fixing readme.md: {str(e)}")


if __name__ == "__main__":
    fix_readme()
