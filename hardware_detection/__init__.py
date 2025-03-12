"""Hardware detection package for identifying and utilizing specialized hardware."""

__version__ = "1.0.0"

from hardware_detection.core import (
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    is_openvino_available,
    print_hardware_info,
)

__all__ = [
    "detect_hardware_type",
    "get_cpu_info",
    "get_gpu_info",
    "get_hardware_info",
    "is_openvino_available",
    "print_hardware_info",
]
