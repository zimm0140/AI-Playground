#!/usr/bin/env python3
"""Setup script for hardware detection package."""

from setuptools import find_packages, setup

setup(
    name="hardware_detection",
    version="1.0.0",
    description="Hardware detection module for identifying specialized hardware",
    author="Intel Corporation",
    author_email="support@intel.com",
    packages=find_packages(),
    package_data={
        "hardware_detection": ["py.typed"],
    },
    python_requires=">=3.8",
    install_requires=[
        "typing-extensions>=4.0.0",
        "types-requests>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
