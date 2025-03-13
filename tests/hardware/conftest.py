#!/usr/bin/env python3
"""Pytest configuration for hardware tests."""

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def mock_dir():
    """Create and return a mock directory for hardware tests."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)
    return mock_dir


@pytest.fixture
def base_environment(mock_dir, monkeypatch):
    """Set up a base environment for testing."""
    # Set environment variables
    monkeypatch.setenv("UVFAST_MOCK_DIR", str(mock_dir))
    if "SIMULATED_HARDWARE" in os.environ:
        monkeypatch.delenv("SIMULATED_HARDWARE")

    # Create mock files
    with mock_dir.joinpath("gpu_info.txt").open("w", encoding="utf-8") as f:
        f.write("Generic GPU\n")

    with mock_dir.joinpath("cpu_info.txt").open("w", encoding="utf-8") as f:
        f.write("vendor: Generic\n")
        f.write("name: Generic CPU\n")
        f.write("cores: 4\n")

    yield

    # Clean up is handled by the mock_dir fixture


@pytest.fixture
def intel_arc_environment(mock_dir, monkeypatch):
    """Set up an Intel Arc GPU environment for testing."""
    # Set environment variables
    monkeypatch.setenv("UVFAST_MOCK_DIR", str(mock_dir))
    monkeypatch.setenv("SIMULATED_HARDWARE", "acm")

    # Create mock files
    with mock_dir.joinpath("gpu_info.txt").open("w", encoding="utf-8") as f:
        f.write("Intel(R) Arc(TM) A770 Graphics\n")

    with mock_dir.joinpath("cpu_info.txt").open("w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i9-13900K\n")
        f.write("cores: 24\n")

    # Create a mock intel-gpu-stub package
    sys.path.insert(0, str(mock_dir))
    intel_gpu_dir = mock_dir / "intel_gpu_stub"
    intel_gpu_dir.mkdir(exist_ok=True)

    with intel_gpu_dir.joinpath("__init__.py").open("w", encoding="utf-8") as f:
        f.write("# Mock Intel GPU package\n")
        f.write("__version__ = '1.0.0'\n")

    yield

    # Clean up
    sys.path.remove(str(mock_dir))


@pytest.fixture
def openvino_environment(mock_dir, monkeypatch):
    """Set up an OpenVINO environment for testing."""
    # Set environment variables
    monkeypatch.setenv("UVFAST_MOCK_DIR", str(mock_dir))
    monkeypatch.setenv("SIMULATED_HARDWARE", "ovino")

    # Create mock files
    with mock_dir.joinpath("gpu_info.txt").open("w", encoding="utf-8") as f:
        f.write("Intel(R) UHD Graphics 770\n")

    with mock_dir.joinpath("cpu_info.txt").open("w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i7-1370P\n")
        f.write("cores: 16\n")

    # Create a mock openvino package
    sys.path.insert(0, str(mock_dir))
    openvino_dir = mock_dir / "openvino"
    openvino_dir.mkdir(exist_ok=True)

    with openvino_dir.joinpath("__init__.py").open("w", encoding="utf-8") as f:
        f.write("# Mock OpenVINO package\n")
        f.write("__version__ = '2023.1.0'\n")

    yield

    # Clean up
    sys.path.remove(str(mock_dir))


@pytest.fixture(params=["base", "acm", "ovino"])
def all_environments(request, mock_dir, monkeypatch):
    """Parametrized fixture for all hardware environments."""
    hardware_type = request.param

    # Set environment variables
    monkeypatch.setenv("UVFAST_MOCK_DIR", str(mock_dir))

    if hardware_type == "base":
        if "SIMULATED_HARDWARE" in os.environ:
            monkeypatch.delenv("SIMULATED_HARDWARE")

        with mock_dir.joinpath("gpu_info.txt").open("w", encoding="utf-8") as f:
            f.write("Generic GPU\n")

        with mock_dir.joinpath("cpu_info.txt").open("w", encoding="utf-8") as f:
            f.write("vendor: Generic\n")
            f.write("name: Generic CPU\n")
            f.write("cores: 4\n")

    elif hardware_type == "acm":
        monkeypatch.setenv("SIMULATED_HARDWARE", "acm")

        with mock_dir.joinpath("gpu_info.txt").open("w", encoding="utf-8") as f:
            f.write("Intel(R) Arc(TM) A770 Graphics\n")

        with mock_dir.joinpath("cpu_info.txt").open("w", encoding="utf-8") as f:
            f.write("vendor: Intel\n")
            f.write("name: Intel(R) Core(TM) i9-13900K\n")
            f.write("cores: 24\n")

    elif hardware_type == "ovino":
        monkeypatch.setenv("SIMULATED_HARDWARE", "ovino")

        with mock_dir.joinpath("gpu_info.txt").open("w", encoding="utf-8") as f:
            f.write("Intel(R) UHD Graphics 770\n")

        with mock_dir.joinpath("cpu_info.txt").open("w", encoding="utf-8") as f:
            f.write("vendor: Intel\n")
            f.write("name: Intel(R) Core(TM) i7-1370P\n")
            f.write("cores: 16\n")

    return hardware_type
