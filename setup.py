#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Maintain compatibility with pip install -e
setup(
    name="ai-playground",
    version="2.2.0",
    description="AI Playground with compatibility for upstream merging",
    author="AI Playground Team",
    author_email="contributor@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "requests>=2.28.0",
        "flask>=2.2.0",
        "pillow>=9.3.0",
        "marshmallow-dataclass>=8.5.3",
        "langchain>=0.3.0",
        "compel>=2.0.0",
        "pytest>=7.3.1",
        "mypy>=1.3.0",
        "types-requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "mypy>=1.3.0",
            "ruff>=0.0.272",
            "pre-commit>=3.3.2",
            "markdownlint-cli>=0.35.0",
            "types-requests>=2.28.0",
            "tomli>=2.0.1",
            "tomli-w>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.13",
    ],
) 