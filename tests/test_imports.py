#!/usr/bin/env python3
"""
Tests for hardware detection import compatibility.

This module verifies that both the new package structure
and the legacy import paths work as expected.
"""

import warnings

import pytest


def test_new_import():
    """Test importing from the new package structure."""
    # Suppress the deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Import from the new package
        import hardware_detection

        # Check version is available
        assert hasattr(hardware_detection, "__version__")
        assert hardware_detection.__version__ == "1.0.0"

        # Check core functions are available
        assert hasattr(hardware_detection, "detect_hardware_type")
        assert hasattr(hardware_detection, "get_gpu_info")
        assert hasattr(hardware_detection, "get_cpu_info")
        assert hasattr(hardware_detection, "get_hardware_info")
        assert hasattr(hardware_detection, "print_hardware_info")


def test_legacy_import():
    """Test importing from the legacy location."""
    # We expect a deprecation warning
    with pytest.warns(DeprecationWarning):
        import tools.hardware.hardware_detection

    # Check version is available
    assert hasattr(tools.hardware.hardware_detection, "__version__")

    # Check core functions are available
    assert hasattr(tools.hardware.hardware_detection, "detect_hardware_type")
    assert hasattr(tools.hardware.hardware_detection, "get_gpu_info")
    assert hasattr(tools.hardware.hardware_detection, "get_cpu_info")
    assert hasattr(tools.hardware.hardware_detection, "get_hardware_info")
    assert hasattr(tools.hardware.hardware_detection, "print_hardware_info")


def test_api_consistency():
    """Test that both APIs provide the same functionality."""
    # Suppress deprecation warnings for this test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Import from both locations
        import hardware_detection
        import tools.hardware.hardware_detection

        # Verify version consistency
        assert hardware_detection.__version__ == tools.hardware.hardware_detection.__version__

        # Test a few key functions produce the same results
        hw_type1 = hardware_detection.detect_hardware_type()
        hw_type2 = tools.hardware.hardware_detection.detect_hardware_type()
        assert hw_type1 == hw_type2

        # Check hardware info functions return similar structures
        info1 = hardware_detection.get_hardware_info()
        info2 = tools.hardware.hardware_detection.get_hardware_info()

        # Check that both dictionaries have the same keys
        assert set(info1.keys()) == set(info2.keys())

        # Check that the detected hardware is the same
        assert info1["detected_hardware"] == info2["detected_hardware"]


if __name__ == "__main__":
    # Run the tests if file is executed directly
    pytest.main(["-xvs", __file__])
