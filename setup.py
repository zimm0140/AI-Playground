#!/usr/bin/env python3
"""Setup script for hardware_detection package."""

from setuptools import find_packages, setup

setup(
    name="hardware_detection",
    version="1.0.0",
    description="Hardware detection for ML applications",
    long_description=open("hardware_detection/README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Playground Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    # Keep dependencies minimal for upstream acceptance
    install_requires=[],
    package_data={
        "hardware_detection": ["py.typed"],
    },
    entry_points={
        "console_scripts": [
            "hardware-detect=hardware_detection.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
