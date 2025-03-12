#!/usr/bin/env python3
"""
Tests for specific hardware configurations to verify detection logic.
These tests focus on simulated hardware environments with specific configurations.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest


def test_hardware_detection_module_exists():
    """Verify the hardware detection module exists."""
    module_path = Path("tools/hardware/hardware_detection.py")
    assert module_path.exists(), "Hardware detection module does not exist"

    # Try to import it
    spec = importlib.util.spec_from_file_location("hardware_detection", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["hardware_detection"] = module
    spec.loader.exec_module(module)

    # Check for required functions
    assert hasattr(module, "detect_hardware_type"), "Missing detect_hardware_type function"
    assert hasattr(module, "get_gpu_info"), "Missing get_gpu_info function"
    assert hasattr(module, "get_cpu_info"), "Missing get_cpu_info function"
    assert hasattr(module, "get_hardware_info"), "Missing get_hardware_info function"


@pytest.mark.parametrize("hardware_type", ["base", "acm", "ovino"])
def test_specific_hardware_detection(hardware_type):
    """Test hardware detection with specific hardware types."""
    # Set up environment for this test
    os.environ["SIMULATED_HARDWARE"] = hardware_type

    # Import the module (which should get our environment variable)
    from tools.hardware.hardware_detection import (
        detect_hardware_type,
        get_hardware_info,
    )

    # Test detection
    detected = detect_hardware_type()
    assert detected == hardware_type, f"Expected {hardware_type}, got {detected}"

    # Test hardware info
    info = get_hardware_info()
    assert info["detected_hardware"] == hardware_type

    if hardware_type == "acm":
        # Check for Arc GPU in the info
        assert info["gpus"], "No GPUs found in acm hardware type"
        assert any("Arc" in gpu for gpu in info["gpus"]), "Arc GPU not found in acm hardware type"

        # Check CPU info
        assert "i9" in info["cpu"]["name"], "Expected i9 CPU in acm hardware type"

    elif hardware_type == "ovino":
        # Check for Intel GPU in the info
        assert info["gpus"], "No GPUs found in ovino hardware type"
        assert any("Intel" in gpu for gpu in info["gpus"]), "Intel GPU not found in ovino hardware type"

        # Check CPU info
        assert "i7" in info["cpu"]["name"], "Expected i7 CPU in ovino hardware type"

        # Check OpenVINO availability
        assert info["openvino_available"], "OpenVINO should be available in ovino hardware type"


def test_environment_variables():
    """Test that environment variables are correctly set and used."""
    # Test with a custom mock directory
    custom_mock_dir = ".uvfast/custom_mock"
    os.makedirs(custom_mock_dir, exist_ok=True)

    # Create a custom GPU info file
    with open(f"{custom_mock_dir}/gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Custom Test GPU\n")

    # Set the environment variable
    original_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    os.environ["UVFAST_MOCK_DIR"] = custom_mock_dir

    try:
        # Import the module with our custom environment
        from tools.hardware.hardware_detection import get_gpu_info

        # Get GPU info
        gpus = get_gpu_info()

        # Verify our custom GPU is detected
        assert any("Custom Test GPU" in gpu for gpu in gpus), "Custom GPU not detected from mock dir"

    finally:
        # Restore original environment
        if original_mock_dir:
            os.environ["UVFAST_MOCK_DIR"] = original_mock_dir
        else:
            del os.environ["UVFAST_MOCK_DIR"]


def test_config_loading():
    """Test loading of uvfast.json configuration."""
    # Create a test config file
    config_path = Path(".uvfast/uvfast.json")
    os.makedirs(config_path.parent, exist_ok=True)

    config_content = {
        "hardware_types": ["test_hw1", "test_hw2", "base"],
        "default_hardware": "test_hw1",
        "detection": {
            "test_hw1": {"gpu_name_pattern": "Test GPU 1"},
            "test_hw2": {"cpu_name_pattern": "Test CPU 2"},
        },
    }

    import json

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_content, f, indent=2)

    try:
        # Import the module
        from tools.hardware.hardware_detection import load_config

        # Load the config
        config = load_config()

        # Verify the config is loaded correctly
        assert config["default_hardware"] == "test_hw1", "Config not loaded correctly"
        assert "test_hw1" in config["hardware_types"], "Hardware types not loaded correctly"
        assert "gpu_name_pattern" in config["detection"]["test_hw1"], "Detection config not loaded correctly"

    finally:
        # Clean up
        if config_path.exists():
            os.unlink(config_path)
