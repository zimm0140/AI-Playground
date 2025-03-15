#!/usr/bin/env python3
"""Check the complexity of Python functions in a file."""

import ast
import sys
from pathlib import Path

try:
    import mccabe
except ImportError:
    print("mccabe module not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mccabe"])
    import mccabe


def check_file_complexity(filename, threshold=10):
    """Check the complexity of all functions in a file."""
    print(f"Checking complexity of {filename}...")

    with open(filename, encoding="utf-8") as f:
        code = f.read()

    try:
        tree = ast.parse(code, filename)
    except SyntaxError as e:
        print(f"Syntax error in {filename}: {e}")
        return

    visitor = mccabe.PathGraphingAstVisitor()
    visitor.preorder(tree, visitor)

    found_complex = False
    for graph in visitor.graphs.values():
        if graph.complexity() > threshold:
            found_complex = True
            print(f"  Function {graph.entity}: complexity={graph.complexity()}")

    if not found_complex:
        print(f"  No functions with complexity > {threshold} found.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <python_file> [threshold]")
        sys.exit(1)

    filename = sys.argv[1]
    threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if not Path(filename).exists():
        print(f"File {filename} does not exist.")
        sys.exit(1)

    check_file_complexity(filename, threshold)


if __name__ == "__main__":
    main() 
