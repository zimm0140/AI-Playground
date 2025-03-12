"""Tests for hardware detection module."""

import pytest

from hardware_detection import (
    detect_hardware_type,
    get_cpu_info,
    get_gpu_info,
    get_hardware_info,
    is_openvino_available,
)


def test_version():
    """Test that the module has a version."""
    from hardware_detection import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)


def test_detect_hardware_type_base(mock_base_env):
    """Test hardware type detection with base environment."""
    hw_type = detect_hardware_type()
    assert hw_type == "base", f"Expected base, got {hw_type}"


def test_detect_hardware_type_acm(mock_acm_env):
    """Test hardware type detection with Intel Arc environment."""
    hw_type = detect_hardware_type()
    assert hw_type == "acm", f"Expected acm, got {hw_type}"


def test_detect_hardware_type_ovino(mock_ovino_env):
    """Test hardware type detection with OpenVINO environment."""
    hw_type = detect_hardware_type()
    assert hw_type == "ovino", f"Expected ovino, got {hw_type}"


def test_get_gpu_info_base(mock_base_env):
    """Test GPU info detection with base environment."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    assert "Generic GPU" in gpus[0]


def test_get_gpu_info_acm(mock_acm_env):
    """Test GPU info detection with Intel Arc environment."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    assert "Arc" in gpus[0]


def test_get_gpu_info_ovino(mock_ovino_env):
    """Test GPU info detection with OpenVINO environment."""
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    assert "UHD" in gpus[0]


def test_get_cpu_info_base(mock_base_env):
    """Test CPU info detection with base environment."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info
    assert "Generic" in cpu_info["name"]
    assert cpu_info["cores"] == 4


def test_get_cpu_info_acm(mock_acm_env):
    """Test CPU info detection with Intel Arc environment."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info
    assert "i9" in cpu_info["name"]
    assert cpu_info["cores"] == 24


def test_get_cpu_info_ovino(mock_ovino_env):
    """Test CPU info detection with OpenVINO environment."""
    cpu_info = get_cpu_info()
    assert isinstance(cpu_info, dict)
    assert "name" in cpu_info
    assert "i7" in cpu_info["name"]
    assert cpu_info["cores"] == 16


def test_get_hardware_info(mock_acm_env):
    """Test hardware info retrieval."""
    info = get_hardware_info()
    assert isinstance(info, dict)
    assert "system" in info
    assert "gpus" in info
    assert "cpu" in info
    assert "detected_hardware" in info
    assert info["detected_hardware"] == "acm"


def test_is_openvino_available():
    """Test OpenVINO availability detection."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "ovino")
        assert is_openvino_available()

        mp.setenv("SIMULATED_HARDWARE", "base")
        assert not is_openvino_available()


def test_platform_specific_detection():
    """Test platform-specific detection paths."""
    # Test Windows path
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "base")
        info = get_hardware_info()
        assert isinstance(info, dict)
        assert "system" in info
        assert "gpus" in info
        assert "cpu" in info
        assert "detected_hardware" in info
        assert info["detected_hardware"] == "base"


def test_detect_hardware_type_env():
    """Test hardware type detection with environment variables."""
    # Test with environment variable
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SIMULATED_HARDWARE", "base")
        assert detect_hardware_type() == "base"

        mp.setenv("SIMULATED_HARDWARE", "acm")
        assert detect_hardware_type() == "acm"

        mp.setenv("SIMULATED_HARDWARE", "ovino")
        assert detect_hardware_type() == "ovino"


def test_get_gpu_info_env():
    """Test GPU info detection with environment variables."""
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


def test_get_cpu_info_env():
    """Test CPU info detection with environment variables."""
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
