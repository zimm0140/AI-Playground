#!/usr/bin/env python3
"""Setup script for openvino-dummy package."""

from setuptools import find_packages, setup

setup(
    name="openvino-dummy",
    version="2023.1.0",
    description="Dummy OpenVINO package for testing",
    author="Test Author",
    author_email="test@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
)
