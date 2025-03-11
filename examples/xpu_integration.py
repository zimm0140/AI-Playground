#!/usr/bin/env python3
"""
XPU Integration Example

This script demonstrates how to integrate hardware detection with XPU hijacks
to automatically configure PyTorch for the appropriate backend based on the 
available hardware.
"""

import importlib.util
import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from the root
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

try:
    # Import hardware detection module
    from hardware_detection import detect_hardware_type, get_hardware_info
except ImportError:
    print("Error: hardware_detection.py not found.")
    print(f"Make sure it exists in {root_dir}")
    sys.exit(1)


def is_package_available(package_name: str) -> bool:
    """Check if a package is available."""
    return importlib.util.find_spec(package_name) is not None


def configure_torch_backend():
    """Configure the appropriate torch backend based on hardware."""
    # Detect hardware type
    hardware_type = detect_hardware_type()
    print(f"Detected hardware type: {hardware_type}")

    # Get hardware information
    hardware_info = get_hardware_info()
    print(f"GPUs: {hardware_info['gpus']}")

    # Configure based on hardware type
    if hardware_type == "acm" and is_package_available("intel_extension_for_pytorch"):
        print("Configuring for Intel Arc GPU with IPEX...")

        # Import PyTorch first
        import torch

        try:
            # Import Intel Extension for PyTorch
            import intel_extension_for_pytorch as ipex

            # Apply XPU hijacks
            try:
                from service.xpu_hijacks import ipex_hijacks

                ipex_hijacks()
                print("Successfully applied XPU hijacks")
            except ImportError:
                print("Warning: Could not import xpu_hijacks. Some functionality may be limited.")

            # Set device to XPU
            print("Setting default device to XPU")
            os.environ["XPU_VISIBLE_DEVICES"] = "0"
            device = torch.device("xpu:0")

            return "xpu", device
        except ImportError:
            print("Warning: Intel Extension for PyTorch not found, falling back to CPU")
            device = torch.device("cpu")
            return "cpu", device

    elif hardware_type == "ovino" and is_package_available("openvino"):
        print("Configuring for OpenVINO...")

        # Import PyTorch first
        import torch

        try:
            # Import OpenVINO
            import openvino

            print("Setting up OpenVINO integration")
            # For demonstration - in a real application you would
            # configure OpenVINO specific optimizations here

            # Use CPU device for OpenVINO
            device = torch.device("cpu")

            return "openvino", device
        except ImportError:
            print("Warning: OpenVINO not found, falling back to CPU")
            device = torch.device("cpu")
            return "cpu", device

    else:
        print("Using default CPU backend")
        import torch

        device = torch.device("cpu")
        return "cpu", device


def test_torch_tensor_creation(device):
    """Test creating a PyTorch tensor on the configured device."""
    import torch

    print(f"\nTesting tensor creation on {device}...")
    try:
        # Create a tensor on the device
        x = torch.randn(2, 3, device=device)
        print(f"Successfully created tensor with shape {x.shape} on {device}")
        print(f"Tensor device: {x.device}")
        print(f"Tensor dtype: {x.dtype}")

        # Do a simple operation
        y = x + 1
        print("Successfully performed tensor operation: x + 1")

        return True
    except Exception as e:
        print(f"Error creating tensor on {device}: {e}")
        return False


def test_neural_network(device):
    """Test a simple neural network on the configured device."""
    import torch
    import torch.nn as nn

    print(f"\nTesting neural network on {device}...")
    try:
        # Define a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 2)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        # Create and move model to device
        model = SimpleModel().to(device)
        print(f"Successfully created and moved model to {device}")

        # Create input tensor
        x = torch.randn(1, 10, device=device)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        print("Successfully ran model inference")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"Error testing neural network on {device}: {e}")
        return False


def main():
    """Main function demonstrating hardware-aware PyTorch configuration."""
    print("XPU Integration Example")
    print("======================\n")

    # Configure the appropriate backend based on hardware
    backend, device = configure_torch_backend()

    # Test tensor creation
    test_torch_tensor_creation(device)

    # Test neural network
    test_neural_network(device)

    print("\nExecution complete!")


if __name__ == "__main__":
    main()
