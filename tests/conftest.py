"""
Pytest configuration and fixtures for testing.
"""

import os
import sys
import pytest

# Add the GitHub workflows scripts directory to the Python path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(script_dir, ".github", "workflows", "scripts"))


@pytest.fixture
def sample_workflow_traditional():
    """Fixture providing a sample workflow in traditional format."""
    return {
        "name": "Test Workflow",
        "version": "1.0.0",
        "nodes": {
            "1": {
                "class_type": "CheckpointLoader",
                "inputs": {
                    "ckpt_name": "model.safetensors"
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "a photo of a cat",
                    "clip": ["1", 0]
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": 42,
                    "steps": 20
                }
            }
        },
        "links": [
            [1, 0, 2, 1],  # Model to CLIP
            [1, 0, 3, 0],  # Model to KSampler
            [2, 0, 3, 1],  # Positive prompt to KSampler
            [4, 0, 3, 2],  # Negative prompt to KSampler
            [5, 0, 3, 3]   # Latent image to KSampler
        ]
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
                    "inputs": {
                        "ckpt_name": "model.safetensors"
                    }
                },
                "2": {
                    "class_type": "VAELoader",
                    "inputs": {
                        "vae_name": "vae.safetensors"
                    }
                }
            },
            "links": [
                [1, 0, 3, 0],
                [2, 0, 3, 1]
            ]
        }
    }


@pytest.fixture
def sample_workflow_direct_nodes():
    """Fixture providing a sample workflow in API format with direct nodes."""
    return {
        "name": "Test Direct Nodes Workflow",
        "version": "1.0.0",
        "comfyUiApiWorkflow": {
            "1": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": "input.png"
                }
            },
            "2": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["1", 0],
                    "filename_prefix": "output"
                }
            },
            "links": [
                [1, 0, 2, 0]
            ]
        }
    } 