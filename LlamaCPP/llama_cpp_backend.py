"""
Llama.cpp Backend Implementation Module
------------------------------------
This module provides a concrete implementation of the LLMInterface for the Llama.cpp
runtime. It handles model loading, text generation, and resource management
for Large Language Models using the Llama.cpp C++ library with Python bindings.

The LlamaCpp class implements all the abstract methods defined in LLMInterface,
providing the functionality necessary for integrating Llama.cpp-optimized models
with the rest of the system.
"""

import gc
from os import path
from typing import Callable, Dict, List

import model_config
from llama_cpp import CreateChatCompletionStreamResponse, Iterator, Llama
from llama_interface import LLMInterface
from llama_params import LLMParams


class LlamaCpp(LLMInterface):
    """
    Llama.cpp backend implementation for Large Language Models.

    This class provides concrete implementations of the abstract methods
    defined in LLMInterface, using the Llama.cpp runtime to handle model
    loading, inference, and resource management.

    The class manages model loading, caching, and generation of text
    responses based on conversation history through the Llama.cpp library.
    """

    def __init__(self):
        """
        Initialize the Llama.cpp backend.

        Sets up initial state with no loaded model, generation flag set to false,
        and no previous model identifier.
        """
        self._model = None
        self.stop_generate = False
        self._last_repo_id = None

    def load_model(
        self,
        params: LLMParams,
        n_gpu_layers: int = -1,
        context_length: int = 16000,
        callback: Callable[[str], None] = None,
    ):
        """
        Load a model based on the provided parameters.

        This method checks if the requested model is already loaded. If not,
        it unloads any existing model and loads the requested one. Progress
        can be reported via an optional callback function.

        Args:
            params: Configuration parameters including model repository ID
            n_gpu_layers: Number of layers to offload to GPU (-1 for auto-determination)
            context_length: Maximum context window size for the model
            callback: Optional function called with status updates during loading
                     - "start" when loading begins
                     - "finish" when loading completes
        """
        model_repo_id = params.model_repo_id
        if self._model is None or self._last_repo_id != model_repo_id:
            if callback is not None:
                callback("start")
            self.unload_model()

            model_base_path = model_config.llamaCppConfig.get("ggufLLM")
            namespace, repo, *model = model_repo_id.split("/")
            model_path = path.abspath(path.join(model_base_path, "---".join([namespace, repo]), "---".join(model)))

            self._model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=context_length,
                verbose=False,
            )

            self._last_repo_id = model_repo_id
            if callback is not None:
                callback("finish")

    def create_chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1024):
        """
        Generate text completion based on conversation messages.

        This method uses the loaded Llama.cpp model to generate a response
        to the provided conversation history, streaming the results token by token.

        Args:
            messages: List of conversation messages with roles and content
            max_tokens: Maximum number of tokens to generate (default: 1024)

        Returns:
            Iterator of streaming response chunks from the Llama.cpp library
        """
        completion: Iterator[CreateChatCompletionStreamResponse] = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
        )
        return completion

    def unload_model(self):
        """
        Unload the current model and free resources.

        This method properly closes the Llama.cpp model if one exists,
        deletes the reference, runs garbage collection to free memory,
        and resets the model reference to None.
        """
        if self._model is not None:
            self._model.close()
            del self._model
        gc.collect()
        self._model = None

    def get_backend_type(self):
        """
        Get the identifier for this backend implementation.

        This method returns a string identifying this backend as the Llama.cpp
        implementation, allowing client code to know which specific backend
        is being used.

        Returns:
            String "llama_cpp" identifying this backend type
        """
        return "llama_cpp"
