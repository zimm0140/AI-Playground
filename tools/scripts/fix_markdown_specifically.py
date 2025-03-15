#!/usr/bin/env python3

import re


def fix_hardware_guide():
    """Fix specific issues in HARDWARE_OPTIMIZATION_GUIDE.md"""
    file_path = "docs/hardware/HARDWARE_OPTIMIZATION_GUIDE.md"
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Fix Table of Contents numbering and links
    new_toc = """## Table of Contents

1. [Overview](#overview)
2. [Hardware Types](#hardware-types)
3. [Environment Setup](#environment-setup)
4. [Using the AI Framework Integration](#using-the-ai-framework-integration)
5. [Working with LangChain](#working-with-langchain)
6. [Working with Stable Diffusion](#working-with-stable-diffusion)
7. [Performance Benchmarking](#performance-benchmarking)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

"""

    # Replace TOC section
    toc_pattern = re.compile(r"## Table of Contents.*?\n\n", re.DOTALL)
    modified_content = re.sub(toc_pattern, new_toc, content)

    # Ensure proper trailing newline
    modified_content = modified_content.rstrip("\n") + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Fixed markdown issues in {file_path}")


def fix_migration_guide():
    """Fix specific issues in MIGRATION.md"""
    file_path = "docs/development/MIGRATION.md"
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Fix Table of Contents numbering and links
    new_toc = """## Table of Contents

1. [Migrating from pip to uv](#migrating-from-pip-to-uv)
2. [Updating Type Annotations for Python 3.10+](#updating-type-annotations-for-python-310)
3. [Using Lockfiles for Reproducible Environments](#using-lockfiles-for-reproducible-environments)
4. [Working with Docker](#working-with-docker)
5. [CI/CD Pipeline Updates](#cicd-pipeline-updates)
6. [Migration FAQs](#migration-faqs)

"""

    # Replace TOC section
    toc_pattern = re.compile(r"## Table of Contents.*?\n\n", re.DOTALL)
    modified_content = re.sub(toc_pattern, new_toc, content)

    # Ensure proper trailing newline
    modified_content = modified_content.rstrip("\n") + "\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"Fixed markdown issues in {file_path}")


def main():
    # Fix both files with specific targeted fixes
    fix_hardware_guide()
    fix_migration_guide()


if __name__ == "__main__":
    main()