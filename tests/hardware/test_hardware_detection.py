#!/usr/bin/env python3
"""Tests for hardware detection module."""

import os
import platform

import pytest

from tools.hardware.hardware_detection import (
    __version__,
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    is_openvino_available,
)


def test_version():
    """Test that the module has a version."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_detect_hardware_type():
    """Test hardware type detection."""
    # When running in CI with SIMULATED_HARDWARE set
    if "SIMULATED_HARDWARE" in os.environ:
        expected = os.environ["SIMULATED_HARDWARE"]
        actual = detect_hardware_type()
        assert actual == expected, f"Expected {expected}, got {actual}"
    else:
        # When running without simulation, should return a valid type
        hw_type = detect_hardware_type()
        assert hw_type in ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"]


def test_get_gpu_info():
    """Test GPU info detection."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)

    # When running in CI with SIMULATED_HARDWARE set
    if os.environ.get("SIMULATED_HARDWARE") == "acm":
        assert any("Arc" in gpu for gpu in gpus)
    elif os.environ.get("SIMULATED_HARDWARE") == "ovino":
        assert any("Intel" in gpu for gpu in gpus)


def test_get_cpu_info():
    """Test CPU info detection."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info

    # When running in CI with SIMULATED_HARDWARE set
    if os.environ.get("SIMULATED_HARDWARE") == "acm":
        assert "i9" in cpu_info["name"]
    elif os.environ.get("SIMULATED_HARDWARE") == "ovino":
        assert "i7" in cpu_info["name"]


def test_get_hardware_info():
    """Test hardware info retrieval."""
    info = get_hardware_info()
    assert isinstance(info, dict)
    assert "system" in info
    assert "gpus" in info
    assert "cpu" in info
    assert "detected_hardware" in info

    # Hardware type should match what we expect
    if "SIMULATED_HARDWARE" in os.environ:
        expected = os.environ["SIMULATED_HARDWARE"]
        assert info["detected_hardware"] == expected


def test_openvino_available():
    """Test OpenVINO availability check."""
    result = is_openvino_available()
    assert isinstance(result, bool)

    # In simulated environment, we should get specific results
    if os.environ.get("SIMULATED_HARDWARE") == "ovino":
        assert result is True


def test_hardware_info():
    """Test hardware info collection."""
    info = get_hardware_info()
    assert isinstance(info, dict)
    assert "system" in info
    assert "python_version" in info
    assert "gpus" in info
    assert "cpu" in info
    assert "detected_hardware" in info
    assert "openvino_available" in info

    # System should match platform.system()
    assert info["system"] == platform.system()

    # Python version should match platform.python_version()
    assert info["python_version"] == platform.python_version()


@pytest.mark.parametrize("hardware_type", ["base", "acm", "ovino"])
def test_simulated_hardware_detection(hardware_type, monkeypatch):
    """Test hardware detection with simulated environment."""
    # Set up simulation
    monkeypatch.setenv("SIMULATED_HARDWARE", hardware_type)

    # Test detection
    detected = detect_hardware_type()
    assert detected == hardware_type

    # Test that info is consistent
    info = get_hardware_info()
    assert info["detected_hardware"] == hardware_type

    if hardware_type == "acm":
        assert any("Arc" in gpu for gpu in info["gpus"])
    elif hardware_type == "ovino":
        assert info["openvino_available"] is True
