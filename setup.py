#!/usr/bin/env python3
"""Setup script for hardware detection package."""

from setuptools import find_packages, setup

setup(
    name="hardware_detection",
    version="0.1.0",
    description="Hardware detection module for ML workloads",
    author="AI Playground Team",
    author_email="example@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "typing-extensions>=4.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
)
