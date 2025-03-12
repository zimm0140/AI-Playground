"""
Llama.cpp LLM Interface Module
----------------------------
This module defines an abstract interface for Large Language Model implementations
using the Llama.cpp optimization framework.

The LLMInterface abstract base class enforces a consistent API for different LLM
backend implementations, allowing for interchangeable model backends while
maintaining a unified interface for client code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from llama_params import LLMParams


class LLMInterface(ABC):
    """
    Abstract interface for Large Language Model implementations using Llama.cpp.

    This abstract base class defines the required methods that any LLM
    implementation must provide to be compatible with the Llama.cpp framework.
    It provides a standard interface for model loading/unloading, text generation,
    and backend identification.

    Attributes:
        stop_generate: Boolean flag to signal text generation should stop
        _model: Optional reference to the underlying model object
    """

    stop_generate: bool
    _model: Optional[object]

    @abstractmethod
    def load_model(self, params: LLMParams, **kwargs):
        """
        Load the specified language model into memory.

        This method initializes the model based on the provided parameters
        and prepares it for inference.

        Args:
            params: Configuration parameters for the model
            **kwargs: Additional backend-specific parameters

        Returns:
            Implementation-defined, typically a success indicator or the model itself
        """
        pass

    @abstractmethod
    def unload_model(self):
        """
        Unload the current model from memory.

        This method frees resources associated with the loaded model
        and resets the model state.

        Returns:
            Implementation-defined, typically a success indicator
        """
        pass

    @abstractmethod
    def create_chat_completion(self, messages: List[Dict[str, str]]):
        """
        Generate a text completion based on the provided conversation messages.

        This method handles the core text generation functionality, taking a conversation
        history and producing a model response.

        Args:
            messages: List of conversation turns, typically with 'role' and 'content' keys

        Returns:
            Implementation-defined, typically the generated text or a response object
        """
        pass

    @abstractmethod
    def get_backend_type(self):
        """
        Get the type identifier for this LLM backend implementation.

        This method allows client code to identify which specific backend
        implementation is being used.

        Returns:
            A string or identifier representing the backend type
        """
        pass
