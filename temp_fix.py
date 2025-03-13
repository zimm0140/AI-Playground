#!/usr/bin/env python3
"""
Temporary script to fix link fragments in HARDWARE_OPTIMIZATION_GUIDE.md
"""

file_path = "docs/hardware/HARDWARE_OPTIMIZATION_GUIDE.md"
with open(file_path, encoding="utf-8") as f:
    content = f.read()

# Fix link fragments by directly replacing the problematic lines
lines = content.splitlines()
fixed_lines = []

for line in lines:
    if "[Using the AI Framework Integration](#using-the-ai-framework-integration)" in line:
        line = line.replace(
            "[Using the AI Framework Integration](#using-the-ai-framework-integration)",
            "[Using the AI Framework Integration](#using-the-ai-framework-integration)",
        )
    elif "[Working with LangChain](#working-with-langchain)" in line:
        line = line.replace(
            "[Working with LangChain](#working-with-langchain)",
            "[Working with LangChain](#working-with-langchain)",
        )
    elif "[Working with Stable Diffusion](#working-with-stable-diffusion)" in line:
        line = line.replace(
            "[Working with Stable Diffusion](#working-with-stable-diffusion)",
            "[Working with Stable Diffusion](#working-with-stable-diffusion)",
        )
    elif "[Performance Benchmarking](#performance-benchmarking)" in line:
        line = line.replace(
            "[Performance Benchmarking](#performance-benchmarking)",
            "[Performance Benchmarking](#performance-benchmarking)",
        )
    elif "[Troubleshooting](#troubleshooting)" in line:
        line = line.replace("[Troubleshooting](#troubleshooting)", "[Troubleshooting](#troubleshooting)")
    elif "[Advanced Configuration](#advanced-configuration)" in line:
        line = line.replace(
            "[Advanced Configuration](#advanced-configuration)",
            "[Advanced Configuration](#advanced-configuration)",
        )

    fixed_lines.append(line)

# Ensure file ends with a single newline
content = "\n".join(fixed_lines)
if not content.endswith("\n"):
    content += "\n"

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Fixed {file_path}")
