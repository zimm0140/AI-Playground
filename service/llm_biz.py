"""
LLM Business Logic Module
------------------------
This module provides a comprehensive interface for working with Large Language Models (LLMs).

Features:
- Model loading and efficient management with Intel XPU acceleration
- Streaming text generation with real-time output
- Chat history management with templating support
- Retrieval-Augmented Generation (RAG) integration
- Performance measurement and optimization
- Graceful stopping mechanism and resource cleanup

This module serves as the core LLM engine, handling the complexities of prompt processing,
model inference, memory management, and performance metrics collection.
"""

# Load model directly
import gc
import threading
import time
import traceback
import torch
import logging
import sys

from typing import Any, List, Dict
from os import path
from transformers import (
    TextIteratorStreamer,
    StoppingCriteriaList,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

try:
    from ipex_llm.transformers import AutoModelForCausalLM
except ModuleNotFoundError:
    from transformers import AutoModelForCausalLM
from typing import Callable
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)
import service_config


class LLMParams:
    """
    Configuration parameters for LLM generation.
    
    This class encapsulates all the necessary parameters for controlling
    the behavior of the LLM generation process, including model selection,
    device placement, and generation settings.
    
    Attributes:
        prompt: List of message dictionaries in the chat format
        device: Device ID for model placement (XPU device index)
        enable_rag: Whether to enable Retrieval-Augmented Generation
        model_repo_id: Model identifier for loading from disk/repository
        max_tokens: Maximum number of tokens to generate
        print_metrics: Whether to print performance metrics after generation
        generation_parameters: Additional parameters for the generation process
    """
    prompt: List[Dict[str, str]]
    device: int
    enable_rag: bool
    model_repo_id: str
    max_tokens: int
    print_metrics: bool
    generation_parameters: Dict[str, Any]


    def __init__(
            self,
            prompt: list,
            device: int,
            enable_rag: bool,
            model_repo_id: str,
            max_tokens: int,
            print_metrics: bool = True,
            **kwargs
    ) -> None:
        """
        Initialize LLMParams with the provided configuration.
        
        Args:
            prompt: List of message dictionaries for the conversation
            device: XPU device index to use for inference
            enable_rag: Flag to enable Retrieval-Augmented Generation
            model_repo_id: Identifier for the model to load
            max_tokens: Maximum number of tokens to generate
            print_metrics: Whether to print performance metrics
            **kwargs: Additional generation parameters to pass to the model
        """
        self.prompt = prompt
        self.device = device
        self.enable_rag = enable_rag
        self.model_repo_id = model_repo_id
        self.max_tokens = max_tokens
        self.print_metrics = print_metrics
        self.generation_parameters = kwargs


# Format string for RAG prompts, combining context with the user question
RAG_PROMPT_FORMAT = "Answer the questions based on the information below. \n{context}\n\nQuestion: {prompt}"

# Global state variables for model management
_model: PreTrainedModel = None  # Currently loaded model
_generating = False  # Flag indicating if generation is in progress
_stop_generate = False  # Flag to request generation stopping
_stop_event = threading.Event()  # Event for synchronizing stop requests
_last_repo_id: str = None  # Last loaded model repo ID for caching
_default_prompt = {
    "role": "system",
    "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user. Please keep the output text language the same as the user input.",
}


def user_stop(input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
    """
    Callback function to check if generation should stop.
    
    Used as a stopping criterion during generation to check if the
    user has requested to stop the generation process.
    
    Args:
        input_ids: Token IDs generated so far
        scores: Token scores
        **kwargs: Additional arguments (unused)
        
    Returns:
        bool: True if generation should stop, False otherwise
    """
    global _stop_generate
    return _stop_generate


def stream_chat_generate(
        model: PreTrainedModel,
        args: dict,
        error_callback: Callable[[Exception], None] = None,
):
    """
    Generate text using a model with streaming output.
    
    This function is designed to be run in a separate thread, allowing
    generated text to be streamed via the TextIteratorStreamer in args.
    
    Args:
        model: The pre-trained language model to use for generation
        args: Dictionary of arguments to pass to the model's generate method
        error_callback: Optional callback for handling exceptions
    """
    try:
        model.generate(**args)
        sys.stdout.flush()
    except Exception as ex:
        traceback.print_exc()
        if error_callback is not None:
            error_callback(ex)


def generate(
        prompt: List[Dict[str, str]],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int,
        error_callback: Callable[[Exception], None] = None,
):
    """
    Prepare and start text generation from the given prompt.
    
    This function handles:
    1. Formatting the chat history with templates
    2. Truncating long prompts to fit model context
    3. Setting up stopping criteria and streaming
    4. Starting the generation in a background thread
    
    Args:
        prompt: List of message dictionaries to generate from
        model: The model to use for generation
        tokenizer: Tokenizer matching the model
        max_new_tokens: Maximum number of new tokens to generate
        error_callback: Optional callback for error handling
        
    Returns:
        TextIteratorStreamer: Streamer object to iterate over generated tokens
    """
    logging.debug(f"got prompt: {prompt}")
    global _stop_generate, _default_prompt
    _stop_generate = False

    # Prepare chat history with default system prompt
    chat_history = [_default_prompt]
    prompt_len = prompt.__len__()
    i = 0
    while i < prompt_len:
        chat_history.append({"role": "user", "content": prompt[i].get("question")})
        if i < prompt_len - 1:
            chat_history.append(
                {"role": "assistant", "content": prompt[i].get("answer")}
            )
        i = i + 1

    # Apply chat template to format the conversation
    new_prompt = tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )

    # Truncate prompt if it's too long for the model's context window
    while len(tokenizer.tokenize(new_prompt)) > 2000:
        chat_history.remove(chat_history[1])
        new_prompt = tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    # Tokenize the prompt and prepare model inputs
    model_inputs = tokenizer(new_prompt, return_tensors="pt").to(service_config.device)
    ##tensor: torch.Tensor = encoding.get("input_ids")

    # Set up stopping criteria for generation
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(CustomStopCriteria(user_stop))

    # Configure streamer for token-by-token output
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Prepare generation arguments
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        num_beams=1,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
    )

    # Start generation in a separate thread
    chat_thread = threading.Thread(
        target=stream_chat_generate,
        kwargs=dict(model=model, args=generate_kwargs, error_callback=error_callback),
    )

    chat_thread.start()

    return streamer


def process_rag(
        prompt: str,
        text_out_callback: Callable[[str, int], None] = None,
):
    """
    Process a prompt with Retrieval-Augmented Generation.
    
    Queries a retrieval system to find relevant context for the prompt,
    then formats the prompt with retrieved context for improved generation.
    
    Args:
        prompt: The user's input prompt
        text_out_callback: Optional callback for sending retrieved information
        
    Returns:
        str: RAG-enhanced prompt with relevant context
    """
    import rag

    # Initialize RAG and move to correct device
    rag.to(service_config.device)
    
    # Query RAG system for relevant context
    query_success, context, rag_source = rag.query(prompt)
    if query_success:
        print("rag query input\r\n{}output:\r\n{}".format(prompt, context))
        # Format the prompt with the retrieved context
        prompt = RAG_PROMPT_FORMAT.format(prompt=prompt, context=context)
        if text_out_callback is not None:
            text_out_callback(rag_source, 2)
    return prompt


def chat(
        params: LLMParams,
        load_model_callback: Callable[[str], None] = None,
        text_out_callback: Callable[[str, int], None] = None,
        metrics_callback: Callable[[dict], None] = None,
        error_callback: Callable[[Exception], None] = None,
):
    """
    Main entry point for LLM chat functionality.
    
    This function:
    1. Handles model loading if needed
    2. Processes RAG queries if enabled
    3. Performs generation with streaming output
    4. Collects and reports performance metrics
    
    Args:
        params: Configuration parameters for the chat session
        load_model_callback: Optional callback for model loading events
        text_out_callback: Optional callback for streaming text output
        metrics_callback: Optional callback for performance metrics
        error_callback: Optional callback for error handling
    """
    global _model, _last_repo_id, _generating, _tokenizer, _stop_generate

    try:
        # if prev genera not finish, stop it
        stop_generate()

        # Set device and extract parameters
        torch.xpu.set_device(params.device)
        service_config.device = f"xpu:{params.device}"
        prompt = params.prompt
        enable_rag = params.enable_rag
        model_repo_id = params.model_repo_id
        max_tokens = params.max_tokens

        _generating = True
        _stop_generate = False

        # Load model if not already loaded or if model changed
        if _model is None or _last_repo_id != model_repo_id:
            # if change model, free used resources
            if _model is not None:
                del _model
                gc.collect()
                torch.xpu.empty_cache()

            # Construct model path from repository ID
            model_base_path = service_config.service_model_paths.get("llm")
            model_name = model_repo_id.replace("/", "---")
            model_path = path.abspath(path.join(model_base_path, model_name))

            # load model
            if load_model_callback is not None:
                load_model_callback("start")
            start = time.time()

            # Set quantization parameters
            load_in_low_bit = "sym_int4"

            # Load model with Intel optimizations
            _model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                load_in_low_bit=load_in_low_bit,
                # load_in_4bit=True,
            )

            _tokenizer = AutoTokenizer.from_pretrained(model_path)

            _last_repo_id = model_repo_id

            print(
                "load llm model {} finish. cost {}s".format(
                    model_repo_id, round(time.time() - start, 3)
                )
            )
            if load_model_callback is not None:
                load_model_callback("finish")

        # Check if generation should stop
        assert_stop_generate()

        # Process prompt with RAG if enabled
        if enable_rag:
            last_prompt = prompt[prompt.__len__() - 1]
            last_prompt.__setitem__(
                "question", process_rag(last_prompt.get("question"), text_out_callback)
            )

        # Move model to specified device
        _model = _model.to(service_config.device)

        # Initialize metrics tracking
        num_tokens = 0
        start_time = time.time()
        is_first = True
        first_token_time = 0
        last_token_time = 0
        
        # Start generation with metrics collection
        with torch.inference_mode():
            all_stream_output = ""
            for stream_output in generate(
                    prompt, _model, _tokenizer, max_tokens, error_callback
            ):
                assert_stop_generate()

                # Track token timing for metrics
                num_tokens += 1
                if is_first:
                    first_token_time = time.time()
                    is_first = False

                # Process and output generated text
                if stream_output != "":
                    all_stream_output += stream_output
                    print(stream_output, end="")
                    text_out_callback(stream_output, 1)

        # Finalize metrics
        last_token_time = time.time()
        torch.xpu.empty_cache()

        # Calculate and report performance metrics
        metrics_data = {
            "type": "metrics",
            "num_tokens": num_tokens,
            "total_time": last_token_time - start_time,
            "overall_tokens_per_second": num_tokens / (last_token_time - start_time),
            "second_plus_tokens_per_second": (num_tokens - 1) / (last_token_time - first_token_time),
            "first_token_latency": first_token_time - start_time,
            "after_token_latency": (last_token_time - first_token_time) / (num_tokens - 1) if num_tokens > 1 else None
        }

        metrics_callback(metrics_data)

        # Log metrics if enabled
        if params.print_metrics:
            logging.info(f"""
                    ----------inference finish----------
                    num_tokens : {metrics_data['num_tokens']}
                    total_time : {metrics_data['total_time']:.4f} s
                    overall tokens/s : {metrics_data['overall_tokens_per_second']:.4f}
                    2nd+ token/s : {metrics_data['second_plus_tokens_per_second']:.4f}
                    first_token_latency : {metrics_data['first_token_latency']:.4f} s
                    after_token_latency : {metrics_data['after_token_latency']:.4f} s
                    """)

    finally:
        _generating = False


def stop_generate():
    """
    Stop any ongoing generation process.
    
    Sets flags to stop generation and waits for the process to acknowledge
    the stop request via the stop event.
    """
    global _stop_generate, _generating, _stop_event
    if _generating:
        _stop_generate = True
        _stop_event.clear()
        _stop_event.wait()
        _generating = False
        _stop_generate = False


def assert_stop_generate():
    """
    Check if generation should stop and raise an exception if so.
    
    This function is called during generation to check if the process
    should stop, and raises a StopGenerateException if requested.
    This allows for graceful interruption of the generation process.
    
    Raises:
        StopGenerateException: If generation should stop
    """
    global _stop_generate, _stop_event
    if _stop_generate:
        _stop_event.set()
        raise StopGenerateException()


def dispose():
    """
    Clean up resources used by the LLM module.
    
    Stops any ongoing generation, deletes the model,
    and clears GPU memory caches to prevent memory leaks.
    """
    global _stop_generate, _model
    stop_generate()

    del _model
    _model = None
    gc.collect()
    torch.xpu.empty_cache()


class StopGenerateException(Exception):
    """
    Exception raised when generation is stopped by user request.
    
    This exception is used for flow control to gracefully exit
    from the generation process when requested by the user.
    """
    def __str__(self):
        return "user stop llm generate"


class CustomStopCriteria(StoppingCriteria):
    """
    Custom stopping criteria for text generation.
    
    This class implements the HuggingFace StoppingCriteria interface
    to allow for custom stopping logic during generation.
    """
    def __init__(self, stop_callback):
        """
        Initialize with a callback function that determines when to stop.
        
        Args:
            stop_callback: Function that returns True when generation should stop
        """
        self.stop_callback = stop_callback

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Determine if generation should stop.
        
        Delegates to the stop_callback function to decide if
        generation should stop based on current state.
        
        Args:
            input_ids: Token IDs generated so far
            scores: Token scores from the model
            **kwargs: Additional arguments passed to the callback
            
        Returns:
            bool: True if generation should stop, False otherwise
        """
        return self.stop_callback(input_ids, scores, **kwargs)
