#!/usr/bin/env python3
"""Setup script for hardware_detection package."""

from setuptools import setup

# Metadata is specified in pyproject.toml
setup(
    # Package-specific data
    package_data={
        "hardware_detection": ["py.typed"],
    },
    # Entry points for CLI tools
    entry_points={
        "console_scripts": [
            "hardware-detection=hardware_detection.cli:main",
        ],
    },
)
