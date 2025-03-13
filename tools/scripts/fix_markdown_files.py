#!/usr/bin/env python3
"""
Markdown file fixer that handles common markdown issues including:
- Table of Contents deduplication and correct numbering
- Code block syntax fixing
- Ordered list numbering
- Proper spacing around headings and lists
- Link fragments
- Table formatting
- Unique headings
"""

import sys
import os
import re
from pathlib import Path
import argparse
from typing import List, Dict, Tuple, Set

try:
    import markdown_it
    from markdown_it import MarkdownIt
    has_markdown_it = True
except ImportError:
    has_markdown_it = False


def fix_markdown_file(file_path: str, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """
    Fix common issues in markdown files.
    
    Args:
        file_path: Path to markdown file
        dry_run: Don't write changes, just report them
        
    Returns:
        Tuple of (success, list of changes made)
    """
    print(f"Processing {file_path}...")
    changes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Store original content to check if changes were made
    original_content = content
    
    # Fix 1: Table of Contents deduplication and correct numbering
    content, toc_changes = fix_table_of_contents(content)
    changes.extend(toc_changes)
    
    # Fix 2: Code block syntax
    content, code_changes = fix_code_blocks(content)
    changes.extend(code_changes)
    
    # Fix 3: Ordered list numbering
    content, list_changes = fix_ordered_lists(content)
    changes.extend(list_changes)
    
    # Fix 4: Heading IDs and link fragments
    content, link_changes = fix_link_fragments(content)
    changes.extend(link_changes)
    
    # Fix 5: Table formatting (MD055/MD056)
    content, table_changes = fix_table_formatting(content)
    changes.extend(table_changes)
    
    # Fix 6: Unique headings (MD024)
    content, heading_uniq_changes = fix_duplicate_headings(content)
    changes.extend(heading_uniq_changes)
    
    # Fix 7: Ensure proper spacing around headings (MD022)
    content, heading_changes = fix_blanks_around_headings(content)
    changes.extend(heading_changes)
    
    # Fix 8: Ensure proper spacing around lists (MD032)
    content, list_spacing_changes = fix_blanks_around_lists(content)
    changes.extend(list_spacing_changes)
    
    # Fix 9: Ensure proper spacing around fenced code blocks (MD031)
    content, code_spacing_changes = fix_blanks_around_fences(content)
    changes.extend(code_spacing_changes)
    
    # Fix 10: Add language to fenced code blocks (MD040)
    content, code_lang_changes = fix_fenced_code_language(content)
    changes.extend(code_lang_changes)
    
    # Fix 11: Remove trailing whitespace on all lines
    content, whitespace_changes = fix_trailing_whitespace(content)
    changes.extend(whitespace_changes)
    
    # Fix 12: Remove multiple consecutive blank lines
    content, blank_line_changes = fix_consecutive_blank_lines(content)
    changes.extend(blank_line_changes)
    
    # Fix 13: Ensure single trailing newline at end of file
    content, eol_changes = fix_file_ending(content)
    changes.extend(eol_changes)

    # Fix 14: Run spacing fixes a second time to address any missed spots
    # after all other changes
    content, heading_changes2 = fix_blanks_around_headings(content)
    content, list_spacing_changes2 = fix_blanks_around_lists(content)
    content, code_spacing_changes2 = fix_blanks_around_fences(content)
    changes.extend(heading_changes2)
    changes.extend(list_spacing_changes2)
    changes.extend(code_spacing_changes2)
    
    # Write changes if content was modified and not in dry run mode
    if content != original_content and not dry_run:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed {len(changes)} issues in {file_path}")
    elif content != original_content and dry_run:
        print(f"🔍 Would fix {len(changes)} issues in {file_path} (dry run)")
    else:
        print(f"✓ No issues to fix in {file_path}")
    
    return content != original_content, changes


def fix_table_formatting(content: str) -> Tuple[str, List[str]]:
    """Fix table formatting issues (MD055/MD056)."""
    changes = []
    
    # Match markdown tables (standard format)
    # Look for table header row and separator row
    table_pattern = re.compile(r'^([|]?.*[|].*[|]?)\s*\n([|]?[ :]*[-]+[ :]*[|][ :]*[-]+[ :]*.*[|]?)\s*\n', re.MULTILINE)
    
    # Find all tables
    for match in table_pattern.finditer(content):
        header_row = match.group(1)
        separator_row = match.group(2)
        
        # Check if header has leading and trailing pipes
        has_leading_pipe = header_row.startswith('|')
        has_trailing_pipe = header_row.endswith('|')
        
        # Check if separator has leading and trailing pipes
        sep_has_leading_pipe = separator_row.startswith('|')
        sep_has_trailing_pipe = separator_row.endswith('|')
        
        # If any row is missing leading or trailing pipes, fix all rows
        if not (has_leading_pipe and has_trailing_pipe and sep_has_leading_pipe and sep_has_trailing_pipe):
            # Extract the table content
            table_start = match.start()
            
            # Find the end of the table by looking for lines with pipes
            lines = content[match.start():].split('\n')
            table_end_line = 2  # We already matched header and separator rows
            
            # Process remaining rows
            for i, line in enumerate(lines[2:], 2):
                if '|' in line:
                    table_end_line = i + 1
                else:
                    break
            
            # Extract the table
            table_text = '\n'.join(lines[:table_end_line])
            
            # Process table rows
            fixed_rows = []
            total_pipes = 0
            rows = table_text.split('\n')
            
            # Determine column count by inspecting separator row
            sep_parts = separator_row.strip('|').split('|')
            column_count = len(sep_parts)
            
            for i, row in enumerate(rows):
                # Clean the row by removing leading/trailing pipes and spaces
                clean_row = row.strip()
                if not clean_row.startswith('|'):
                    clean_row = '|' + clean_row
                if not clean_row.endswith('|'):
                    clean_row = clean_row + '|'
                
                # Ensure correct number of columns
                row_parts = clean_row.strip('|').split('|')
                if len(row_parts) < column_count:
                    # Add missing cells
                    for _ in range(column_count - len(row_parts)):
                        clean_row = clean_row[:-1] + ' |'
                
                fixed_rows.append(clean_row)
            
            # Replace the original table with the fixed one
            fixed_table = '\n'.join(fixed_rows)
            content = content[:match.start()] + fixed_table + content[match.start() + len(table_text):]
            
            changes.append("Fixed table formatting (added missing pipes and ensured consistent column count)")
    
    return content, changes


def fix_duplicate_headings(content: str) -> Tuple[str, List[str]]:
    """Fix duplicate headings (MD024)."""
    changes = []
    
    # Match headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+\{#(.*?)\})?\s*$', re.MULTILINE)
    
    # Find all headings
    headings = list(heading_pattern.finditer(content))
    
    # Track used heading texts and IDs
    used_heading_texts = {}
    used_ids = set()
    heading_replacements = {}
    
    for match in headings:
        level, text, heading_id = match.groups()
        
        # Check for duplicate heading text at the same level
        if (level, text) in used_heading_texts:
            # Create a unique suffix for the heading
            counter = used_heading_texts.get((level, text), 0) + 1
            used_heading_texts[(level, text)] = counter
            
            new_text = f"{text} ({counter})"
            
            # If it has an explicit ID, keep it; otherwise generate unique one
            if not heading_id:
                new_id = github_heading_id(new_text)
                while new_id in used_ids:
                    new_id = f"{new_id}-{counter}"
                
                heading_replacements[match.group(0)] = f"{level} {new_text} {{#{new_id}}}"
                used_ids.add(new_id)
                changes.append(f"Fixed duplicate heading: '{text}' -> '{new_text}'")
        else:
            used_heading_texts[(level, text)] = 0
            
            # Add ID to our tracking set if it has one
            if heading_id:
                used_ids.add(heading_id)
    
    # Apply replacements
    for old, new in heading_replacements.items():
        content = content.replace(old, new)
    
    return content, changes


def fix_blanks_around_headings(content: str) -> Tuple[str, List[str]]:
    """Fix blank lines around headings (MD022)."""
    changes = []
    
    # Match headings (with or without IDs)
    heading_pattern = re.compile(r'^(#{1,6}\s+.*?(?:\s+\{#.*?\})?)$', re.MULTILINE)
    
    # Find all headings
    matches = list(heading_pattern.finditer(content))
    
    # Process in reverse to avoid position shifts
    for match in reversed(matches):
        heading = match.group(1)
        start = match.start()
        end = match.end()
        
        # Check if there's a blank line before the heading
        # (unless it's at the start of the file)
        if start > 0:
            # Look back to find the previous non-blank line
            prev_text = content[:start].rstrip()
            if prev_text and not prev_text.endswith('\n\n'):
                # Fix: add blank line before heading
                content = content[:start] + '\n' + content[start:]
                # Adjust end position
                end += 1
                changes.append(f"Added blank line before heading: '{heading.strip()}'")
        
        # Check if there's a blank line after the heading
        # Look ahead to find the next line
        if end < len(content) and not content[end:end+2].startswith('\n\n'):
            # Fix: add blank line after heading
            content = content[:end] + '\n' + content[end:]
            changes.append(f"Added blank line after heading: '{heading.strip()}'")
    
    return content, changes


def fix_blanks_around_lists(content: str) -> Tuple[str, List[str]]:
    """Fix blank lines around lists (MD032)."""
    changes = []
    
    # Match list markers (ordered and unordered)
    list_markers = [
        r'^\d+\.\s', # Ordered list
        r'^[-*+]\s'  # Unordered list
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
            if prev_text and not prev_text.endswith('\n\n') and not prev_text.endswith(':\n'):
                # Don't add blank line if preceded by a colon (likely part of a description)
                if not re.search(r':\s*$', content[:start].split('\n')[-1]):
                    # Fix: add blank line before list
                    content = content[:start] + '\n' + content[start:]
                    changes.append(f"Added blank line before list")
    
    # Second pass: add blank lines after lists - analyze line by line
    lines = content.split('\n')
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
        elif in_list and (not line.strip() or current_indent > list_indent or 
                          (current_indent == list_indent and is_list_item)):
            result_lines.append(line)
        # Detect list end - content that's not part of the list
        elif in_list:
            in_list = False
            # Check if we need a blank line
            if result_lines and result_lines[-1].strip():
                result_lines.append('')
                changes.append("Added blank line after list")
            result_lines.append(line)
        else:
            result_lines.append(line)
    
    return '\n'.join(result_lines), changes


def fix_blanks_around_fences(content: str) -> Tuple[str, List[str]]:
    """Fix blank lines around fenced code blocks (MD031)."""
    changes = []
    
    # Find positions of all code fence markers
    fence_pattern = re.compile(r'^```\w*\s*$', re.MULTILINE)
    positions = [(m.start(), m.group(0)) for m in fence_pattern.finditer(content)]
    
    # Skip if fewer than 2 fence markers found
    if len(positions) < 2:
        return content, changes
    
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
            if prev_text and not prev_text.endswith('\n\n'):
                # Add blank line before fence
                content = content[:start_pos] + '\n' + content[start_pos:]
                adjustments += 1
                end_pos += 1  # Adjust end position
                changes.append("Added blank line before code block")
        
        # Check for blank line after closing fence
        fence_end = end_pos + len(end_fence)
        if fence_end < len(content):
            # Look ahead to see if there's a blank line
            if not content[fence_end:fence_end+2].startswith('\n\n'):
                # Add blank line after fence
                content = content[:fence_end] + '\n' + content[fence_end:]
                adjustments += 1
                changes.append("Added blank line after code block")
    
    return content, changes


def fix_fenced_code_language(content: str) -> Tuple[str, List[str]]:
    """Add language specifier to fenced code blocks (MD040)."""
    changes = []
    
    # Match opening code fences without language
    fence_pattern = re.compile(r'^```\s*$', re.MULTILINE)
    
    # Find all fences without language specifier
    matches = list(fence_pattern.finditer(content))
    
    # Process in reverse to avoid position shifts
    for match in reversed(matches):
        # Replace with default language 'text'
        content = content[:match.start()] + '```text' + content[match.end():]
        changes.append("Added default language 'text' to code block")
    
    return content, changes


def fix_file_ending(content: str) -> Tuple[str, List[str]]:
    """Ensure single trailing newline at end of file (MD047)."""
    changes = []
    
    # Remove all trailing newlines
    content = content.rstrip('\n')
    
    # Add a single newline
    content = content + '\n'
    changes.append("Ensured single trailing newline")
    
    return content, changes


def fix_table_of_contents(content: str) -> Tuple[str, List[str]]:
    """Fix duplicate and incorrectly numbered Table of Contents sections."""
    changes = []
    
    # Check if there are multiple TOC sections
    toc_sections = re.findall(r'## Table of Contents\s+(?:(?:\d+\.\s+\[.*?\]\(.*?\)\s*)+)', content, re.DOTALL)
    
    if len(toc_sections) <= 1:
        return content, changes
    
    # Extract all heading links from the TOC sections
    all_links = []
    for section in toc_sections:
        links = re.findall(r'\d+\.\s+\[(.*?)\]\((.*?)\)', section)
        all_links.extend(links)
    
    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for text, url in all_links:
        if url not in seen:
            unique_links.append((text, url))
            seen.add(url)
    
    # Create a new TOC with sequential numbering
    new_toc = "## Table of Contents\n\n"
    for i, (text, url) in enumerate(unique_links, 1):
        new_toc += f"{i}. [{text}]({url})\n"
    new_toc += "\n"
    
    # Replace all TOC sections with the new one
    for section in toc_sections:
        content = content.replace(section, new_toc)
        changes.append(f"Deduplicated and fixed numbering in Table of Contents")
        # Only replace the first occurrence to avoid further duplication
        break
    
    # Remove any remaining TOC sections
    for section in toc_sections[1:]:
        content = content.replace(section, "")
        changes.append(f"Removed duplicate Table of Contents section")
    
    return content, changes


def fix_code_blocks(content: str) -> Tuple[str, List[str]]:
    """Fix code block syntax issues."""
    changes = []
    
    # Replace ```text with just ``` (when it's used as a closing tag)
    pattern = r'```text\s*$'
    replacement = '```'
    matches = re.findall(pattern, content, re.MULTILINE)
    if matches:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        changes.append(f"Fixed {len(matches)} improper code block closings (```text -> ```)")
    
    return content, changes


def fix_ordered_lists(content: str) -> Tuple[str, List[str]]:
    """Fix ordered list numbering."""
    changes = []
    
    # Identify sections with ordered lists
    sections = re.split(r'(?=^##? .*$)', content, flags=re.MULTILINE)
    fixed_sections = []
    
    for i, section in enumerate(sections):
        # Skip if no ordered list in section
        if not re.search(r'^\d+\. ', section, re.MULTILINE):
            fixed_sections.append(section)
            continue
        
        # Split section into lines
        lines = section.split('\n')
        fixed_lines = []
        counter = 1
        in_ordered_list = False
        
        for line in lines:
            # Check if line starts with a number and period
            list_match = re.match(r'^(\d+)\. (.*)', line)
            if list_match:
                if not in_ordered_list:
                    in_ordered_list = True
                    counter = 1
                line = f"{counter}. {list_match.group(2)}"
                counter += 1
            elif line.strip() == '' and in_ordered_list:
                # A blank line might end the list
                in_ordered_list = False
            elif in_ordered_list and line.strip() and not line.strip().startswith('   '):
                # If we hit a non-blank, non-indented line, the list is over
                in_ordered_list = False
            
            fixed_lines.append(line)
        
        fixed_section = '\n'.join(fixed_lines)
        if fixed_section != section:
            changes.append(f"Fixed ordered list numbering in section {i+1}")
        
        fixed_sections.append(fixed_section)
    
    return ''.join(fixed_sections), changes


def github_heading_id(text: str) -> str:
    """
    Generate GitHub-style heading ID from heading text.
    
    Args:
        text: Heading text
        
    Returns:
        GitHub-style heading ID
    """
    # Remove leading numbers and periods (e.g., "1. Introduction" -> "Introduction")
    text = re.sub(r'^\d+\.?\s+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any character that is not alphanumeric, space, or hyphen
    text = re.sub(r'[^\w\s-]', '', text)
    
    # Replace spaces with hyphens
    text = re.sub(r'\s+', '-', text)
    
    return text


def fix_link_fragments(content: str) -> Tuple[str, List[str]]:
    """Fix heading IDs and link fragments."""
    changes = []
    
    # Find all headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+\{#(.*?)\})?\s*$', re.MULTILINE)
    headings = list(heading_pattern.finditer(content))
    
    # If no headings, return original content
    if not headings:
        return content, changes
    
    # Generate IDs for headings that don't have them
    heading_replacements = {}
    for match in headings:
        level, text, existing_id = match.groups()
        if not existing_id:
            new_id = github_heading_id(text)
            new_heading = f"{level} {text} {{#{new_id}}}"
            heading_replacements[match.group(0)] = new_heading
            changes.append(f"Added ID '{new_id}' to heading '{text}'")
    
    # Apply heading replacements
    for old, new in heading_replacements.items():
        content = content.replace(old, new)
    
    # Fix link fragments in the content
    link_pattern = re.compile(r'\[(.*?)\]\(#(.*?)\)')
    for match in link_pattern.finditer(content):
        link_text, fragment = match.groups()
        
        # Check if fragment exists in any heading ID
        found = False
        for heading_match in headings:
            _, heading_text, heading_id = heading_match.groups()
            
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
                    old_link = f"[{link_text}](#{fragment})"
                    new_link = f"[{link_text}](#{new_fragment})"
                    content = content.replace(old_link, new_link)
                    changes.append(f"Fixed link fragment: {fragment} -> {new_fragment}")
                    break
    
    return content, changes


def fix_trailing_whitespace(content: str) -> Tuple[str, List[str]]:
    """Remove trailing whitespace from all lines."""
    changes = []
    
    lines = content.split('\n')
    fixed_lines = []
    trailing_whitespace_count = 0
    
    for line in lines:
        stripped = line.rstrip()
        if stripped != line:
            trailing_whitespace_count += 1
        fixed_lines.append(stripped)
    
    if trailing_whitespace_count > 0:
        changes.append(f"Removed trailing whitespace from {trailing_whitespace_count} lines")
    
    return '\n'.join(fixed_lines), changes


def fix_consecutive_blank_lines(content: str) -> Tuple[str, List[str]]:
    """Remove multiple consecutive blank lines."""
    changes = []
    
    # Replace 3 or more consecutive blank lines with 2 blank lines
    pattern = r'\n{3,}'
    replacement = '\n\n'
    matches = re.findall(pattern, content)
    
    if matches:
        content = re.sub(pattern, replacement, content)
        changes.append(f"Normalized {len(matches)} instances of excessive blank lines")
    
    return content, changes


def find_markdown_files(directory: str = ".", recursive: bool = True) -> List[str]:
    """Find all markdown files in the directory."""
    path = Path(directory)
    if recursive:
        return [str(file) for file in path.glob("**/*.md")]
    else:
        return [str(file) for file in path.glob("*.md")]


def main():
    parser = argparse.ArgumentParser(description="Fix common issues in markdown files")
    parser.add_argument("files", nargs="*", help="Markdown files to fix")
    parser.add_argument("--dir", "-d", help="Directory to scan for markdown files")
    parser.add_argument("--recursive", "-r", action="store_true", 
                        help="Recursively scan directories for markdown files")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Don't write changes, just report them")
    parser.add_argument("--exclude", "-e", 
                        help="Exclude pattern (e.g., 'node_modules')")
    parser.add_argument("--fix-all", "-a", action="store_true",
                        help="Apply all fixes including more intensive ones like table fixing")
    args = parser.parse_args()
    
    if not has_markdown_it:
        print("Warning: markdown-it-py is not installed. Some advanced features may not work.")
        print("Install with: pip install markdown-it-py")
    
    files_to_fix = []
    
    if args.files:
        files_to_fix.extend(args.files)
    
    if args.dir:
        files_to_fix.extend(find_markdown_files(args.dir, args.recursive))
    
    if not files_to_fix and not args.dir:
        # Default to current directory
        files_to_fix.extend(find_markdown_files(".", True))
    
    # Remove duplicates while preserving order
    files_to_fix = list(dict.fromkeys(files_to_fix))
    
    # Apply exclusion pattern if provided
    if args.exclude:
        original_count = len(files_to_fix)
        files_to_fix = [f for f in files_to_fix if args.exclude not in f]
        excluded_count = original_count - len(files_to_fix)
        if excluded_count > 0:
            print(f"Excluded {excluded_count} files matching pattern '{args.exclude}'")
    
    fixed_count = 0
    total_changes = 0
    
    print(f"Processing {len(files_to_fix)} markdown files...")
    
    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            continue
            
        if not file_path.endswith('.md'):
            print(f"Skipping non-markdown file: {file_path}")
            continue
            
        fixed, changes = fix_markdown_file(file_path, args.dry_run)
        if fixed:
            fixed_count += 1
            total_changes += len(changes)
    
    if args.dry_run:
        print(f"\n🔍 Dry run: Would fix {total_changes} issues in {fixed_count} files")
    else:
        print(f"\n✅ Fixed {total_changes} issues in {fixed_count} files")


if __name__ == "__main__":
    main()
