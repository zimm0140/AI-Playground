"""
LlamaCPP Model Configuration Module
----------------------------------
This module defines configuration settings for the Llama.cpp backend implementation,
including model paths and hardware acceleration options.

The configuration is used by the LlamaCPP implementation to locate model files
and determine which hardware acceleration to use for inference.
"""

# Dictionary of model paths for LlamaCPP backend
# - ggufLLM: Path to the directory containing GGUF format language models
llamaCppConfig = {
    "ggufLLM": "../service/models/llm/ggufLLM",
}

# Target device for model inference
# "xpu" refers to Intel's XPU (unified GPU/accelerator) architecture
device = "xpu"
