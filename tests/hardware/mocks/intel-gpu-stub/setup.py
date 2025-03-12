#!/usr/bin/env python3
"""Setup script for intel-gpu-stub package."""

from setuptools import find_packages, setup

setup(
    name="intel-gpu-stub",
    version="1.0.0",
    description="Dummy Intel GPU package for testing",
    author="Test Author",
    author_email="test@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
)
