"""
Llama.cpp LLM Adapter Module
--------------------------
This module provides an adapter between the LLMInterface and Server-Sent Events (SSE)
streaming protocol for web API integration. It enables real-time streaming of LLM
outputs with metrics collection and error handling.

The module implements:
1. The LlmSseAdapter class for streaming LLM responses via SSE
2. Helper functions for prompt conversion and RAG (Retrieval Augmented Generation)
3. Metrics collection for token generation speed and latency measurements
"""

import json
import logging
import threading
import traceback
from collections.abc import Callable
from queue import Empty, Queue

from llama_interface import LLMInterface
from llama_params import LLMParams

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RAG_PROMPT_FORMAT = "Answer the questions based on the information below. \n{context}\n\nQuestion: {prompt}"


class LlmSseAdapter:
    """
    Adapter for streaming LLM responses using Server-Sent Events (SSE) protocol.

    This class bridges the LLMInterface implementation with a streaming response
    mechanism suitable for web APIs. It manages a background thread for LLM inference,
    collects performance metrics, and formats the output as SSE events.

    Attributes:
        msg_queue: Queue for messages to be sent as SSE events
        finish: Flag indicating whether the generation is complete
        singal: Threading event for synchronization
        llm_interface: The LLM implementation to use for text generation
        should_stop: Flag indicating whether generation should be stopped
    """

    msg_queue: Queue
    finish: bool
    singal: threading.Event
    llm_interface: LLMInterface
    should_stop: bool

    def __init__(self, llm_interface: LLMInterface):
        """
        Initialize the SSE adapter with an LLM implementation.

        Args:
            llm_interface: An implementation of LLMInterface to use for text generation
        """
        self.msg_queue = Queue(-1)
        self.finish = False
        self.singal = threading.Event()
        self.llm_interface = llm_interface
        self.should_stop = False

    def put_msg(self, data):
        """
        Add a message to the queue and signal the waiting thread.

        Args:
            data: The message data to be queued
        """
        self.msg_queue.put_nowait(data)
        self.singal.set()

    def load_model_callback(self, event: str):
        """
        Callback for model loading status updates.

        Args:
            event: The loading event ("start", model ID, or "finish")
        """
        data = {"type": "load_model", "event": event}
        self.put_msg(data)

    def text_in_callback(self, msg: str):
        """
        Callback for input text events.

        Args:
            msg: The input text message
        """
        data = {"type": "text_in", "value": msg}
        self.put_msg(data)

    def text_out_callback(self, msg: str, type=1):
        """
        Callback for output text events.

        Args:
            msg: The output text message
            type: Message type (1 for normal output, 2 for RAG sources)
        """
        data = {"type": "text_out", "value": msg, "dtype": type}
        self.put_msg(data)

    def first_latency_callback(self, first_latency: str):
        """
        Callback for first token latency metric.

        Args:
            first_latency: The measured latency for the first token
        """
        data = {"type": "first_token_latency", "value": first_latency}
        self.put_msg(data)

    def after_latency_callback(self, after_latency: str):
        """
        Callback for subsequent tokens latency metric.

        Args:
            after_latency: The measured average latency for tokens after the first
        """
        data = {"type": "after_token_latency", "value": after_latency}
        self.put_msg(data)

    def sr_latency_callback(self, sr_latency: str):
        """
        Callback for sampling rate latency metric.

        Args:
            sr_latency: The measured sampling rate latency
        """
        data = {"type": "sr_latency", "value": sr_latency}
        self.put_msg(data)

    def error_callback(self, ex: Exception):
        """
        Callback for error handling.

        Processes different types of exceptions and sends appropriate error messages.
        The method handles specific exception types differently, including:
        - NotImplementedError for repository access
        - RuntimeError for general runtime issues
        - Other exceptions as unknown

        Some exception handlers are commented out but preserved for future use.

        Args:
            ex: The exception that occurred
        """
        if isinstance(ex, NotImplementedError) and ex.__str__() == "Access to repositories lists is not implemented.":
            self.put_msg(
                {
                    "type": "error",
                    "err_type": "repositories_not_found",
                }
            )
        # elif isinstance(ex, NotEnoughDiskSpaceException):
        #     self.put_msg(
        #         {
        #             "type": "error",
        #             "err_type": "not_enough_disk_space",
        #             "need": bytes2human(ex.requires_space),
        #             "free": bytes2human(ex.free_space),
        #         }
        #     )
        # elif isinstance(ex, DownloadException):
        #     self.put_msg({"type": "error", "err_type": "download_exception"})
        # # elif isinstance(ex, llm_biz.StopGenerateException):
        # #     pass
        elif isinstance(ex, RuntimeError):
            self.put_msg({"type": "error", "err_type": "runtime_error"})
        else:
            self.put_msg({"type": "error", "err_type": "unknown_exception"})
        logging.error(f"exception:{str(ex)}")

    def text_conversation(self, params: LLMParams):
        """
        Start a text conversation in a background thread and return a generator
        for streaming results.

        This method initializes the generation process in a separate thread and returns
        a generator that can be used to stream the results. It sets up the necessary
        callbacks for handling metrics and errors.

        Args:
            params: Parameters for the text generation

        Returns:
            Generator yielding SSE-formatted message strings
        """
        thread = threading.Thread(
            target=self.text_conversation_run,
            args=[params],
        )
        thread.start()
        return self.generator()

    def stream_function(self, stream):
        """
        Process the stream data and send it through the message queue.

        Args:
            stream: Generator of stream data
        """
        for s in stream:
            if self.llm_interface.stop_generate:
                self.llm_interface.stop_generate = False
                break

            if self.llm_interface.get_backend_type() == "ipex_llm":
                # transformer style
                self.text_out_callback(s)
            else:
                # openai style
                self.text_out_callback(s["choices"][0]["delta"].get("content", ""))

        self.put_msg({"type": "finish"})

    def text_conversation_run(
        self,
        params: LLMParams,
    ):
        """
        Run the text conversation in the current thread.

        This method is the core implementation of the text conversation functionality,
        which is run in a background thread by text_conversation. It handles the
        request parameters, calls the LLM, and processes the results.

        Args:
            params: Parameters for the text generation

        Returns:
            None, results are sent through callbacks
        """
        try:
            self.finish = False
            self.should_stop = False

            # Ensure context is properly handled
            context = params.context
            if context:
                params.prompt = RAG_PROMPT_FORMAT.format(prompt=params.prompt, context=context)

            # Handle RAG if requested
            if params.use_rag:
                params.prompt = process_rag(params.prompt, params.device, self.text_out_callback)

            # Note: The abstract interface defines only the 'messages' parameter,
            # but implementations like LlamaCpp also have a 'max_tokens' parameter.
            # This inconsistency causes type errors.
            # Convert the prompt format if using messages
            if params.messages:
                params.messages = convert_prompt(params.messages)

            self.llm_interface.load_model(params, callback=self.load_model_callback)

            prompt = params.prompt
            full_prompt = convert_prompt(prompt)
            # Note: The abstract interface defines only the 'messages' parameter, but implementations
            # like LlamaCpp also have a 'max_tokens' parameter. This inconsistency causes type errors.
            # Using type: ignore disables type checking for this line
            stream = self.llm_interface.create_chat_completion(  # type: ignore
                full_prompt, params.max_tokens
            )
            self.stream_function(stream)

        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            self.finish = True
            self.singal.set()

    def generator(self):
        """
        Generate SSE events for streaming LLM responses.

        This method is a generator that yields SSE events containing the LLM's
        generated text and any metadata. It handles the synchronization between
        the inference thread and the web server, ensuring smooth streaming.

        Yields:
            SSE event strings in the format "data:{json_data}\0"
        """
        while True:
            while not self.msg_queue.empty():
                try:
                    data = self.msg_queue.get_nowait()
                    msg = f"data:{json.dumps(data)}\0"
                    logging.debug(msg)
                    yield msg
                except Empty:
                    break
            if not self.finish:
                self.singal.clear()
                self.singal.wait()
            else:
                break


# Default system prompt for the assistant
_default_prompt = {
    "role": "system",
    "content": (
        "You are a helpful digital assistant. Please provide safe, ethical and "
        "accurate information to the user. Please keep the output text language "
        "the same as the user input."
    ),
}


def convert_prompt(prompt: list[dict[str, str]]):
    """
    Convert the API prompt format to the chat history format expected by the LLM.

    This function transforms the API's question-answer pair format into the
    role-content format used by the LLM, adding a default system message.

    Args:
        prompt: List of question-answer pairs from the API

    Returns:
        List of role-content dictionaries for the LLM
    """
    chat_history = [_default_prompt]
    prompt_len = prompt.__len__()
    i = 0
    while i < prompt_len:
        question = prompt[i].get("question")
        if question is not None:
            chat_history.append({"role": "user", "content": question})

        if i < prompt_len - 1:
            answer = prompt[i].get("answer")
            if answer is not None:
                chat_history.append({"role": "assistant", "content": answer})
        i = i + 1
    return chat_history


def process_rag(
    prompt: str,
    device: str,
    text_out_callback: Callable[[str, int], None] | None = None,
):
    """
    Process a prompt using Retrieval Augmented Generation (RAG).

    This function queries a RAG system to retrieve relevant context
    for the prompt and formats it into a new prompt that includes
    the retrieved information.

    Args:
        prompt: The original query prompt
        device: The device to use for RAG processing
        text_out_callback: Optional callback for sending source information

    Returns:
        Augmented prompt with retrieved context
    """
    # Using try/except block to handle potential import error
    try:
        import rag  # type: ignore

        rag.to(device)
        query_success, context, rag_source = rag.query(prompt)
        if query_success:
            logging.info(f"rag query input\r\n{prompt}\noutput:\r\n{context}")
            prompt = RAG_PROMPT_FORMAT.format(prompt=prompt, context=context)
            if text_out_callback is not None:
                text_out_callback(rag_source, 2)
        return prompt
    except ImportError:
        logging.warning("RAG module couldn't be imported, returning original prompt")
        return prompt
