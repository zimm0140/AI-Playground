"""
Pytest configuration and fixtures for testing.
"""

import json
import os
import platform
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add the GitHub workflows scripts directory to the Python path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(script_dir, ".github", "workflows", "scripts"))

# Add the project root to the Python path
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Add the root directory to the Python path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Set environment variable for CI testing
if "CI" in os.environ and "SIMULATED_HARDWARE" not in os.environ:
    os.environ["SIMULATED_HARDWARE"] = "base"

# Make sure pytest can find the package in CI
try:
    import hardware_detection
except ImportError:
    print("WARNING: Failed to import hardware_detection package")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")

    # Try installing the package in CI
    if "CI" in os.environ:
        import subprocess

        print("Installing package in development mode...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])

        # Second import attempt
        try:
            import hardware_detection

            print("Successfully imported hardware_detection after installation")
        except ImportError:
            print("Failed to import hardware_detection even after installation")


@pytest.fixture
def sample_workflow_traditional():
    """Fixture providing a sample workflow in traditional format."""
    return {
        "name": "Test Workflow",
        "version": "1.0.0",
        "nodes": {
            "1": {
                "class_type": "CheckpointLoader",
                "inputs": {"ckpt_name": "model.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a photo of a cat", "clip": ["1", 0]},
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": 42,
                    "steps": 20,
                },
            },
        },
        "links": [
            [1, 0, 2, 1],  # Model to CLIP
            [1, 0, 3, 0],  # Model to KSampler
            [2, 0, 3, 1],  # Positive prompt to KSampler
            [4, 0, 3, 2],  # Negative prompt to KSampler
            [5, 0, 3, 3],  # Latent image to KSampler
        ],
    }


@pytest.fixture
def sample_workflow_api_format():
    """Fixture providing a sample workflow in API format with nested nodes."""
    return {
        "name": "Test API Workflow",
        "version": "1.0.0",
        "comfyUiApiWorkflow": {
            "nodes": {
                "1": {
                    "class_type": "CheckpointLoader",
                    "inputs": {"ckpt_name": "model.safetensors"},
                },
                "2": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "vae.safetensors"},
                },
            },
            "links": [[1, 0, 3, 0], [2, 0, 3, 1]],
        },
    }


@pytest.fixture
def sample_workflow_direct_nodes():
    """Fixture providing a sample workflow in API format with direct nodes."""
    return {
        "name": "Test Direct Nodes Workflow",
        "version": "1.0.0",
        "comfyUiApiWorkflow": {
            "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
            "2": {
                "class_type": "SaveImage",
                "inputs": {"images": ["1", 0], "filename_prefix": "output"},
            },
            "links": [[1, 0, 2, 0]],
        },
    }


@pytest.fixture
def sample_hardware_detection_config():
    """Fixture providing a sample hardware detection configuration."""
    return {
        "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
        "default_hardware": "base",
        "detection": {
            "acm": {"gpu_name_pattern": "Intel.*Arc|Intel.*A[1-9][0-9][0-9]"},
            "bmg": {"gpu_name_pattern": "Intel.*Battlemage|Intel.*B[1-9][0-9][0-9]"},
            "mtl": {"cpu_name_pattern": "Intel.*Core.*Ultra"},
            "lnl": {"cpu_name_pattern": "Intel.*Lunar Lake"},
            "ovino": {"platform_flags": ["has_openvino"]},
            "arl_h": {"platform_flags": ["arc_specific_flag"]},
        },
    }


@pytest.fixture
def sample_uvfast_config():
    """Fixture providing a sample uvfast configuration."""
    return {
        "project_name": "test-project",
        "python_version": "3.10",
        "hardware_types": ["base", "acm", "bmg", "mtl", "lnl", "ovino", "arl_h"],
        "default_hardware": "base",
        "requirements": {
            "base": "requirements.txt",
            "dev": "requirements-dev.txt",
            "hardware": {
                "base": "requirements-hardware-base.txt",
                "acm": "requirements-hardware-acm.txt",
                "bmg": "requirements-hardware-bmg.txt",
                "mtl": "requirements-hardware-mtl.txt",
                "lnl": "requirements-hardware-lnl.txt",
                "ovino": "requirements-hardware-ovino.txt",
                "arl_h": "requirements-hardware-arl_h.txt",
            },
        },
        "lockfiles": {
            "base": "requirements.lock",
            "dev": "requirements-dev.lock",
            "hardware": {
                "base": "requirements-hardware-base.lock",
                "acm": "requirements-hardware-acm.lock",
                "bmg": "requirements-hardware-bmg.lock",
                "mtl": "requirements-hardware-mtl.lock",
                "lnl": "requirements-hardware-lnl.lock",
                "ovino": "requirements-hardware-ovino.lock",
                "arl_h": "requirements-hardware-arl_h.lock",
            },
        },
    }


@pytest.fixture
def mock_gpu_info():
    """Fixture providing common mock GPU information for different hardware."""
    return {
        "acm": ["Intel(R) Arc(TM) A770 Graphics"],
        "bmg": ["Intel(R) Battlemage(TM) B770 Graphics"],
        "mtl": ["Intel(R) Graphics"],
        "none": ["NVIDIA GeForce RTX 3080"],
        "multiple": ["NVIDIA GeForce RTX 3080", "Intel(R) Arc(TM) A770 Graphics"],
    }


@pytest.fixture
def mock_cpu_info():
    """Fixture providing common mock CPU information for different hardware."""
    return {
        "mtl": {
            "name": "Intel(R) Core(TM) Ultra 7 155H",
            "manufacturer": "Intel Corporation",
        },
        "lnl": {
            "name": "Intel(R) Core(TM) Ultra Lunar Lake",
            "manufacturer": "Intel Corporation",
        },
        "standard": {
            "name": "Intel(R) Core(TM) i9-9900K",
            "manufacturer": "Intel Corporation",
        },
        "amd": {
            "name": "AMD Ryzen 9 5950X",
            "manufacturer": "Advanced Micro Devices, Inc.",
        },
        "apple": {"name": "Apple M1 Pro"},
    }


@pytest.fixture
def mock_subprocess_run():
    """Fixture providing a mock for subprocess.run that returns success."""
    mock = MagicMock()
    mock.return_value.returncode = 0
    return mock


@pytest.fixture
def temp_venv_path(tmpdir):
    """Fixture providing a temporary virtual environment path."""
    venv_dir = tmpdir.mkdir(".venv")
    if platform.system() == "Windows":
        scripts_dir = venv_dir.mkdir("Scripts")
        python_exe = scripts_dir.join("python.exe")
        python_exe.write("#!/bin/env python\n")
    else:
        bin_dir = venv_dir.mkdir("bin")
        python_exe = bin_dir.join("python")
        python_exe.write("#!/bin/env python\n")

    return Path(str(venv_dir))


@pytest.fixture
def temp_config_file(tmpdir):
    """Fixture providing a temporary configuration file."""
    config_file = tmpdir.join("uvfast.json")
    config_data = {
        "project_name": "test-project",
        "python_version": "3.10",
        "hardware_types": ["base", "acm"],
        "default_hardware": "base",
    }
    config_file.write(json.dumps(config_data))
    return Path(str(config_file))


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment for all tests."""
    # Store original environment
    old_env = {
        "SIMULATED_HARDWARE": os.environ.get("SIMULATED_HARDWARE"),
        "CI_TESTING": os.environ.get("CI_TESTING"),
        "UVFAST_MOCK_DIR": os.environ.get("UVFAST_MOCK_DIR"),
    }

    # Set test environment
    os.environ["CI_TESTING"] = "true"
    os.environ["SIMULATED_HARDWARE"] = "base"

    # Create mock directory if it doesn't exist
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)
    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)

    yield

    # Restore original environment
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def mock_dir():
    """Create a temporary directory for mock files."""
    mock_dir = Path(".uvfast/mock")
    mock_dir.mkdir(parents=True, exist_ok=True)
    return mock_dir


@pytest.fixture
def mock_base_env(mock_dir):
    """Set up a mock base environment."""
    # Create mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Generic GPU\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Generic\n")
        f.write("name: Generic CPU\n")
        f.write("cores: 4\n")

    # Set environment variables
    old_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    old_sim_hw = os.environ.get("SIMULATED_HARDWARE")

    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)
    os.environ["SIMULATED_HARDWARE"] = "base"

    yield

    # Restore environment variables
    if old_mock_dir:
        os.environ["UVFAST_MOCK_DIR"] = old_mock_dir
    else:
        os.environ.pop("UVFAST_MOCK_DIR", None)

    if old_sim_hw:
        os.environ["SIMULATED_HARDWARE"] = old_sim_hw
    else:
        os.environ.pop("SIMULATED_HARDWARE", None)


@pytest.fixture
def mock_acm_env(mock_dir):
    """Set up a mock Intel Arc environment."""
    # Create mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Intel(R) Arc(TM) A770 Graphics\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i9-13900K\n")
        f.write("cores: 24\n")

    # Set environment variables
    old_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    old_sim_hw = os.environ.get("SIMULATED_HARDWARE")

    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)
    os.environ["SIMULATED_HARDWARE"] = "acm"

    yield

    # Restore environment variables
    if old_mock_dir:
        os.environ["UVFAST_MOCK_DIR"] = old_mock_dir
    else:
        os.environ.pop("UVFAST_MOCK_DIR", None)

    if old_sim_hw:
        os.environ["SIMULATED_HARDWARE"] = old_sim_hw
    else:
        os.environ.pop("SIMULATED_HARDWARE", None)


@pytest.fixture
def mock_ovino_env(mock_dir):
    """Set up a mock OpenVINO environment."""
    # Create mock files
    with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
        f.write("Intel(R) UHD Graphics 770\n")

    with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
        f.write("vendor: Intel\n")
        f.write("name: Intel(R) Core(TM) i7-1370P\n")
        f.write("cores: 16\n")

    # Set environment variables
    old_mock_dir = os.environ.get("UVFAST_MOCK_DIR")
    old_sim_hw = os.environ.get("SIMULATED_HARDWARE")

    os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)
    os.environ["SIMULATED_HARDWARE"] = "ovino"

    yield

    # Restore environment variables
    if old_mock_dir:
        os.environ["UVFAST_MOCK_DIR"] = old_mock_dir
    else:
        os.environ.pop("UVFAST_MOCK_DIR", None)

    if old_sim_hw:
        os.environ["SIMULATED_HARDWARE"] = old_sim_hw
    else:
        os.environ.pop("SIMULATED_HARDWARE", None)
