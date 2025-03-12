"""Root-level tests for hardware detection package."""

import pytest

from hardware_detection import (
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    is_openvino_available,
)


def test_package_import():
    """Test that the package can be imported."""
    import hardware_detection

    assert hasattr(hardware_detection, "__version__")
    assert isinstance(hardware_detection.__version__, str)


def test_detect_hardware_type():
    """Test hardware type detection."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "base")
        assert detect_hardware_type() == "base"

        mp.setenv("SIMULATED_HARDWARE", "acm")
        assert detect_hardware_type() == "acm"

        mp.setenv("SIMULATED_HARDWARE", "ovino")
        assert detect_hardware_type() == "ovino"


def test_get_gpu_info():
    """Test GPU info detection."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "base")
        gpus = get_gpu_info()
        assert isinstance(gpus, list)
        assert len(gpus) > 0
        assert "Generic GPU" in gpus[0]

        mp.setenv("SIMULATED_HARDWARE", "acm")
        gpus = get_gpu_info()
        assert isinstance(gpus, list)
        assert len(gpus) > 0
        assert "Arc" in gpus[0]

        mp.setenv("SIMULATED_HARDWARE", "ovino")
        gpus = get_gpu_info()
        assert isinstance(gpus, list)
        assert len(gpus) > 0
        assert "UHD" in gpus[0]


def test_get_cpu_info():
    """Test CPU info detection."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "base")
        cpu_info = get_cpu_info()
        assert isinstance(cpu_info, dict)
        assert "name" in cpu_info
        assert "Generic" in cpu_info["name"]
        assert cpu_info["cores"] == 4

        mp.setenv("SIMULATED_HARDWARE", "acm")
        cpu_info = get_cpu_info()
        assert isinstance(cpu_info, dict)
        assert "name" in cpu_info
        assert "i9" in cpu_info["name"]
        assert cpu_info["cores"] == 24

        mp.setenv("SIMULATED_HARDWARE", "ovino")
        cpu_info = get_cpu_info()
        assert isinstance(cpu_info, dict)
        assert "name" in cpu_info
        assert "i7" in cpu_info["name"]
        assert cpu_info["cores"] == 16


def test_get_hardware_info():
    """Test hardware info retrieval."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "base")
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert "system" in info
        assert "gpus" in info
        assert "cpu" in info
        assert "detected_hardware" in info
        assert info["detected_hardware"] == "base"


def test_is_openvino_available():
    """Test OpenVINO availability detection."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "ovino")
        assert is_openvino_available()

        mp.setenv("SIMULATED_HARDWARE", "base")
        assert not is_openvino_available()
