#!/usr/bin/env python3
"""
Setup Mock Modules for CI Testing

This script sets up mock modules for hardware-dependent packages that might
not be available in CI environments.
"""

import sys
from pathlib import Path


def setup_intel_xpu_module():
    """Set up mock intel_extension_for_pytorch.xpu module."""
    xpu_dir = Path("intel_extension_for_pytorch/xpu")
    xpu_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py with mock implementations
    with open(xpu_dir / "__init__.py", "w") as f:
        f.write(
            """
\"\"\"Mock XPU module.\"\"\"

import torch

def is_available():
    \"\"\"Mock is_available function.\"\"\"
    return True

def device_count():
    \"\"\"Mock device_count function.\"\"\"
    return 1

def current_device():
    \"\"\"Mock current_device function.\"\"\"
    return 0

def synchronize():
    \"\"\"Mock synchronize function.\"\"\"
    pass

def empty_cache():
    \"\"\"Mock empty_cache function.\"\"\"
    pass

class DeviceProperties:
    \"\"\"Mock DeviceProperties class.\"\"\"
    def __init__(self, index=0):
        self.name = "Intel Arc A770 Graphics (Mock)"
        self.total_memory = 16 * 1024 * 1024 * 1024  # 16 GB

def get_device_properties(device):
    \"\"\"Mock get_device_properties function.\"\"\"
    return DeviceProperties(device)

def pin_memory(tensor):
    \"\"\"Mock pin_memory function.\"\"\"
    return tensor
"""
        )

    # Create __pycache__ to avoid warnings
    (xpu_dir / "__pycache__").mkdir(exist_ok=True)

    print(f"Created mock module: {xpu_dir}")
    return True


def setup_openvino_runtime_module():
    """Set up mock openvino.runtime module."""
    runtime_dir = Path("openvino/runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py with mock implementations
    with open(runtime_dir / "__init__.py", "w") as f:
        f.write(
            """
\"\"\"Mock OpenVINO Runtime module.\"\"\"

class Core:
    \"\"\"Mock Core class.\"\"\"
    def __init__(self):
        self.devices = ["CPU"]

    def compile_model(self, model, device="CPU"):
        \"\"\"Mock compile_model function.\"\"\"
        return CompiledModel()

    def get_versions(self, device="CPU"):
        \"\"\"Mock get_versions function.\"\"\"
        return {device: {"major": "2023", "minor": "0", "build": "0"}}

    def get_property(self, property_name, device="CPU"):
        \"\"\"Mock get_property function.\"\"\"
        return f"MOCK_{property_name}_{device}"

    def get_available_devices(self):
        \"\"\"Mock get_available_devices function.\"\"\"
        return self.devices

class CompiledModel:
    \"\"\"Mock CompiledModel class.\"\"\"
    def __init__(self):
        pass

    def infer(self, inputs):
        \"\"\"Mock infer function.\"\"\"
        return {"output": [1.0, 2.0, 3.0]}

    def inputs(self):
        \"\"\"Mock inputs function.\"\"\"
        return []

    def outputs(self):
        \"\"\"Mock outputs function.\"\"\"
        return []

class Layout:
    \"\"\"Mock Layout class.\"\"\"
    NCHW = "NCHW"
    NHWC = "NHWC"

class Type:
    \"\"\"Mock Type class.\"\"\"
    f32 = "f32"
    f16 = "f16"
    i32 = "i32"
    i64 = "i64"
"""
        )

    # Create __pycache__ to avoid warnings
    (runtime_dir / "__pycache__").mkdir(exist_ok=True)

    print(f"Created mock module: {runtime_dir}")
    return True


def setup_all_mock_modules():
    """Set up all mock modules."""
    success = True
    success &= setup_intel_xpu_module()
    success &= setup_openvino_runtime_module()
    return success


if __name__ == "__main__":
    print("Setting up mock modules for CI testing...")
    if setup_all_mock_modules():
        print("Successfully set up all mock modules")
        sys.exit(0)
    else:
        print("Failed to set up some mock modules")
        sys.exit(1)
