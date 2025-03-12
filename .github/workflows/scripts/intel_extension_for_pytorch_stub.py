#!/usr/bin/env python3
"""
Intel Extension for PyTorch Stub Module for CI Testing

This module provides a simple stub implementation of the intel_extension_for_pytorch module
for use in CI environments where the actual Intel modules are not installed.
"""


class XPUBackend:
    """Mock XPU backend class."""

    @staticmethod
    def is_available():
        """Check if XPU is available."""
        return True

    @staticmethod
    def device_count():
        """Get mock device count."""
        return 1


class DeviceProperties:
    """Mock device properties."""

    def __init__(self, name="Intel Arc A770 Graphics (Mock)"):
        self.name = name
        self.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB
        self.device_type = "XPU"


# Mock versions and attributes
__version__ = "2.0.110+mock"
_C = None  # Mock _C module


# Mock functions
def xpu_device_name():
    """Get the XPU device name."""
    return "Intel Arc A770 Graphics (Mock)"


def optimize():
    """Mock optimize function."""
    # Does nothing
    return


# Create module-level variables for easier importing
def __getattr__(name):
    """Handle attribute access for mock imports."""
    return lambda *args, **kwargs: None
