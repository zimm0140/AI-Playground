"""
OpenVINO Backend Implementation Module
------------------------------------
This module provides a concrete implementation of the LLMInterface for the OpenVINO
runtime. It handles model loading, text generation, and resource management
for Large Language Models optimized with Intel's OpenVINO toolkit.

The OpenVino class implements all the abstract methods defined in LLMInterface,
providing the functionality necessary for integrating OpenVINO-optimized models
with the rest of the system.
"""

import gc
from collections.abc import Callable
from os import path

import openvino_genai
import openvino_model_config as model_config
from openvino_interface import LLMInterface
from openvino_params import LLMParams


class OpenVino(LLMInterface):
    """
    OpenVINO backend implementation for Large Language Models.

    This class provides concrete implementations of the abstract methods
    defined in LLMInterface, using the OpenVINO runtime to handle model
    loading, inference, and resource management.

    The class manages model loading, caching, and generation of text
    responses based on conversation history.
    """

    def __init__(self):
        """
        Initialize the OpenVINO backend.

        Sets up initial state with no loaded model, generation flag set to false,
        and no previous model identifier.
        """
        self._model = None
        self.stop_generate = False
        self._last_repo_id = None

    def load_model(self, params: LLMParams, callback: Callable[[str], None] = None):
        """
        Load a model based on the provided parameters.

        This method checks if the requested model is already loaded. If not,
        it unloads any existing model and loads the requested one. Progress
        can be reported via an optional callback function.

        Args:
            params: Configuration parameters including model repository ID
            callback: Optional function called with status updates during loading
                      - "start" when loading begins
                      - model_repo_id during loading
                      - "finish" when loading completes
        """
        model_repo_id = params.model_repo_id
        if self._model is None or self._last_repo_id != model_repo_id:
            if callback is not None:
                callback("start")
            self.unload_model()
            callback(params.model_repo_id)

            model_base_path = model_config.openVINOConfig.get("openvinoLLM")
            model_name = model_repo_id.replace("/", "---")
            model_path = path.abspath(path.join(model_base_path, model_name))

            # Enable compilation cache for better performance
            enable_compile_cache = {}
            enable_compile_cache["CACHE_DIR"] = "llm_cache"
            self._model = openvino_genai.LLMPipeline(model_path, "AUTO", **enable_compile_cache)
            self._tokenizer = self._model.get_tokenizer()

            self._last_repo_id = model_repo_id
            if callback is not None:
                callback("finish")

    def create_chat_completion(
        self,
        messages: list[dict[str, str]],
        streamer: Callable[[str], None],
        max_tokens: int = 1024,
    ):
        """
        Generate text completion based on conversation messages.

        This method applies a chat template to the messages, configures
        the generation parameters, and streams the generated tokens via
        the provided callback function.

        Args:
            messages: List of conversation messages with roles and content
            streamer: Callback function called with each generated token
            max_tokens: Maximum number of tokens to generate (default: 1024)

        Returns:
            The result from the model's generate method
        """
        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = max_tokens

        full_prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self._model.generate(full_prompt, config, streamer)

    def unload_model(self):
        """
        Unload the current model and free resources.

        This method deletes the current model object if one exists,
        runs garbage collection to free memory, and resets the model
        reference to None.
        """
        if self._model is not None:
            del self._model
        gc.collect()
        self._model = None

    def get_backend_type(self):
        """
        Get the identifier for this backend implementation.

        Returns:
            String "openvino" identifying this backend type
        """
        return "openvino"
