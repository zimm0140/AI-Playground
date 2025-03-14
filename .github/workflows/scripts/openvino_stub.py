#!/usr/bin/env python3
"""
OpenVINO Stub Module for CI Testing

This module provides a simple stub implementation of the openvino module
for use in CI environments where the actual OpenVINO package is not installed.
"""


class Runtime:
    """Mock OpenVINO Runtime class."""

    def __init__(self, *args, **kwargs):
        self.version = "STUB.2023.0.0"

    def get_property(self, prop):
        """Get mock property value."""
        return f"STUB_{prop}"

    def set_property(self, prop, value):
        """Set mock property value."""
        pass

    def compile_model(self, model, device="CPU"):
        """Compile a mock model."""
        return CompiledModel()


class CompiledModel:
    """Mock OpenVINO CompiledModel class."""

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def infer(self, inputs):
        """Mock inference method."""
        return {"output": [1.0, 2.0, 3.0]}


# Module-level properties and functions
AVAILABLE_DEVICES = ["CPU"]
version = "STUB.2023.0.0"


def core():
    """Create a mock OpenVINO Core instance."""
    return Runtime()


# Create module-level variables for easier importing
def __getattr__(name):
    """Handle attribute access for mock imports."""
    return lambda *args, **kwargs: None
