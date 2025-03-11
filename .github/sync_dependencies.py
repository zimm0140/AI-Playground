#!/usr/bin/env python3
"""
Script to synchronize dependencies between setup.py and pyproject.toml.
This ensures that both traditional and modern builds use the same dependencies.
"""

import os
import re
import sys
import tomli
import tomli_w
from typing import Dict, List, Set, Tuple, Optional, Any

# Regular expression patterns to extract dependencies from setup.py
INSTALL_REQUIRES_PATTERN = r"install_requires\s*=\s*\[([\s\S]*?)\]"
EXTRAS_REQUIRE_PATTERN = r"extras_require\s*=\s*\{([\s\S]*?)\}"
EXTRA_PATTERN = r"['\"](.*?)['\"]\s*:\s*\[([\s\S]*?)\]"
DEPENDENCY_PATTERN = r"['\"]([^'\"]+?)['\"]"

def parse_setup_py_dependencies(setup_py_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Parse dependencies from setup.py file.
    
    Returns:
        Tuple containing install_requires list and extras_require dictionary
    """
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    install_requires: List[str] = []
    extras_require: Dict[str, List[str]] = {}
    
    # Extract install_requires
    install_requires_match = re.search(INSTALL_REQUIRES_PATTERN, content)
    if install_requires_match:
        install_requires_block = install_requires_match.group(1)
        install_requires = [
            match.group(1) for match in re.finditer(DEPENDENCY_PATTERN, install_requires_block)
        ]
    
    # Extract extras_require
    extras_require_match = re.search(EXTRAS_REQUIRE_PATTERN, content)
    if extras_require_match:
        extras_block = extras_require_match.group(1)
        for extra_match in re.finditer(EXTRA_PATTERN, extras_block):
            extra_name = extra_match.group(1)
            extra_deps_block = extra_match.group(2)
            extras_require[extra_name] = [
                match.group(1) for match in re.finditer(DEPENDENCY_PATTERN, extra_deps_block)
            ]
    
    return install_requires, extras_require

def load_pyproject_toml(pyproject_path: str) -> Dict[str, Any]:
    """Load pyproject.toml file."""
    with open(pyproject_path, 'rb') as f:
        return tomli.load(f)

def save_pyproject_toml(pyproject_path: str, data: Dict[str, Any]) -> None:
    """Save pyproject.toml file."""
    with open(pyproject_path, 'wb') as f:
        tomli_w.dump(data, f)

def update_pyproject_from_setup(setup_py_path: str, pyproject_path: str) -> None:
    """
    Update pyproject.toml dependencies based on setup.py.
    """
    install_requires, extras_require = parse_setup_py_dependencies(setup_py_path)
    
    try:
        pyproject_data = load_pyproject_toml(pyproject_path)
    except FileNotFoundError:
        print(f"Error: {pyproject_path} not found.")
        sys.exit(1)
    
    # Update project dependencies
    if "project" not in pyproject_data:
        pyproject_data["project"] = {}
    
    pyproject_data["project"]["dependencies"] = install_requires
    
    # Update optional dependencies
    if extras_require:
        if "optional-dependencies" not in pyproject_data["project"]:
            pyproject_data["project"]["optional-dependencies"] = {}
        
        for extra_name, deps in extras_require.items():
            pyproject_data["project"]["optional-dependencies"][extra_name] = deps
    
    # Update Rye dev-dependencies if present
    if "tool" in pyproject_data and "rye" in pyproject_data["tool"]:
        if "dev" in extras_require:
            pyproject_data["tool"]["rye"]["dev-dependencies"] = extras_require["dev"]
    
    save_pyproject_toml(pyproject_path, pyproject_data)
    print(f"Successfully updated {pyproject_path} based on {setup_py_path}")

def update_setup_from_pyproject(setup_py_path: str, pyproject_path: str) -> None:
    """
    Update setup.py dependencies based on pyproject.toml.
    This is more complex and requires string manipulation since setup.py is not a data format.
    """
    try:
        pyproject_data = load_pyproject_toml(pyproject_path)
    except FileNotFoundError:
        print(f"Error: {pyproject_path} not found.")
        sys.exit(1)
    
    # Get dependencies from pyproject.toml
    dependencies = pyproject_data.get("project", {}).get("dependencies", [])
    optional_dependencies = pyproject_data.get("project", {}).get("optional-dependencies", {})
    
    # Read setup.py
    with open(setup_py_path, 'r', encoding='utf-8') as f:
        setup_content = f.read()
    
    # Replace install_requires
    if dependencies:
        deps_str = ",\n        ".join([f'"{dep}"' for dep in dependencies])
        install_requires_replacement = f"install_requires=[\n        {deps_str}\n    ]"
        setup_content = re.sub(
            r"install_requires\s*=\s*\[[\s\S]*?\]",
            install_requires_replacement,
            setup_content
        )
    
    # Replace extras_require
    if optional_dependencies:
        extras_str_parts = []
        for extra_name, deps in optional_dependencies.items():
            deps_str = ",\n            ".join([f'"{dep}"' for dep in deps])
            extras_str_parts.append(f'"{extra_name}": [\n            {deps_str}\n        ]')
        
        extras_str = ",\n        ".join(extras_str_parts)
        extras_require_replacement = f"extras_require={{\n        {extras_str}\n    }}"
        
        setup_content = re.sub(
            r"extras_require\s*=\s*\{[\s\S]*?\}",
            extras_require_replacement,
            setup_content
        )
    
    # Write back to setup.py
    with open(setup_py_path, 'w', encoding='utf-8') as f:
        f.write(setup_content)
    
    print(f"Successfully updated {setup_py_path} based on {pyproject_path}")

def sync_dependencies(direction: str = "both") -> None:
    """
    Synchronize dependencies between setup.py and pyproject.toml.
    
    Args:
        direction: 'to_pyproject', 'to_setup', or 'both'
    """
    setup_py_path = "setup.py"
    pyproject_path = "pyproject.toml"
    
    if not os.path.exists(setup_py_path):
        print(f"Error: {setup_py_path} not found.")
        sys.exit(1)
    
    if not os.path.exists(pyproject_path):
        print(f"Error: {pyproject_path} not found.")
        sys.exit(1)
    
    if direction in ("to_pyproject", "both"):
        update_pyproject_from_setup(setup_py_path, pyproject_path)
    
    if direction in ("to_setup", "both"):
        update_setup_from_pyproject(pyproject_path, setup_py_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        direction = sys.argv[1]
        if direction not in ("to_pyproject", "to_setup", "both"):
            print("Usage: python sync_dependencies.py [to_pyproject|to_setup|both]")
            sys.exit(1)
    else:
        direction = "both"
    
    sync_dependencies(direction) 