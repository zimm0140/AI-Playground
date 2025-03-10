"""
Llama.cpp LLM Parameters Module
------------------------------
This module defines parameter classes used to configure and control
Large Language Model (LLM) inference through the Llama.cpp runtime.

The LLMParams class encapsulates all configuration options needed for text generation,
including conversation prompts, device selection, RAG configuration, and model settings.
"""

from typing import Any, Dict, List


class LLMParams:
    """
    Parameter class for Llama.cpp LLM inference.

    This class encapsulates all configuration parameters needed for text generation
    with Large Language Models optimized through Llama.cpp. It handles both
    basic configuration (model selection, device) and advanced generation parameters.

    Attributes:
        prompt: List of conversation turns as dictionaries with 'question' from user
               and optional 'answer' from the model
        device: Target device ID for inference (e.g., CPU=0, GPU=1)
        enable_rag: Whether to use Retrieval Augmented Generation for enhanced responses
        model_repo_id: Identifier for the model to use (filename or repository ID)
        max_tokens: Maximum number of tokens to generate in the response
        generation_parameters: Additional parameters for controlling text generation
                              (temperature, top_p, etc.)
    """

    prompt: List[Dict[str, str]]
    device: int
    enable_rag: bool
    model_repo_id: str
    max_tokens: int
    generation_parameters: Dict[str, Any]

    def __init__(
        self,
        prompt: list,
        device: int,
        enable_rag: bool,
        model_repo_id: str,
        max_tokens: int,
        **kwargs,
    ) -> None:
        """
        Initialize LLM parameters with required and optional configuration.

        Args:
            prompt: Conversation history as a list of dictionaries, each containing
                   'question' from user and optional 'answer' from model
            device: Device ID for model inference (integer, e.g., CPU=0, GPU=1)
            enable_rag: Whether to enable Retrieval Augmented Generation
            model_repo_id: Model identifier (filename or repository ID)
            max_tokens: Maximum number of tokens to generate in the response
            **kwargs: Additional generation parameters like temperature, top_p, etc.
                     These are passed to the generation_parameters dictionary
        """
        self.prompt = prompt
        self.device = device
        self.enable_rag = enable_rag
        self.model_repo_id = model_repo_id
        self.max_tokens = max_tokens
        self.generation_parameters = kwargs
