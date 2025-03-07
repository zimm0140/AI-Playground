"""
OpenVINO Model Configuration Module
----------------------------------
This module defines configuration settings for the OpenVINO backend implementation,
including model paths for the Large Language Models.

The configuration is used by the OpenVINO implementation to locate model files
for inference. It provides a centralized location for path configurations,
making it easier to update paths without modifying the core implementation code.
"""

# Dictionary of model paths for OpenVINO backend
# - openvinoLLM: Path to the directory containing OpenVINO IR format language models
openVINOConfig = {
    "openvinoLLM": "../service/models/llm/openvino",
}

