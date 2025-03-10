"""
OpenVINO LLM Adapter Module
--------------------------
This module provides an adapter between the LLMInterface and Server-Sent Events (SSE)
streaming protocol for web API integration. It enables real-time streaming of LLM
outputs with metrics collection and error handling.

The module implements:
1. The LLM_SSE_Adapter class for streaming LLM responses via SSE
2. Helper functions for prompt conversion and RAG (Retrieval Augmented Generation)
3. Metrics collection for token generation speed and latency measurements
"""

import threading
from queue import Empty, Queue
import json
import time
import traceback
from typing import Dict, List, Callable
from openvino_interface import LLMInterface
from openvino_params import LLMParams

# Format template for augmenting prompts with retrieved context in RAG mode
RAG_PROMPT_FORMAT = "Answer the questions based on the information below. \n{context}\n\nQuestion: {prompt}"


class LLM_SSE_Adapter:
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
        num_tokens: Counter for generated tokens
        start_time: Timestamp when generation started
        first_token_time: Timestamp when first token was generated
        last_token_time: Timestamp when last token was generated
        is_first_token: Flag indicating if the next token is the first one
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
        self.num_tokens = 0
        self.start_time = 0
        self.first_token_time = 0
        self.last_token_time = 0
        self.is_first_token = True

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

        Args:
            ex: The exception that occurred
        """
        if (
            isinstance(ex, NotImplementedError)
            and ex.__str__() == "Access to repositories lists is not implemented."
        ):
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
        self.put_msg(f"exception:{str(ex)}")

    def text_conversation(self, params: LLMParams):
        """
        Start a text conversation in a background thread and return a generator for streaming results.

        This is the main entry point for client code. It starts the text generation
        in a background thread and returns a generator that yields SSE-formatted events.

        Args:
            params: LLM parameters including prompt and configuration options

        Returns:
            A generator yielding SSE-formatted messages
        """
        thread = threading.Thread(
            target=self.text_conversation_run,
            args=[params],
        )
        thread.start()
        return self.generator()

    def stream_function(self, output):
        """
        Callback function for processing each token generated by the LLM.

        This function is passed to the LLM to handle each generated token.
        It tracks metrics and checks for stop signals.

        Args:
            output: The generated token text

        Returns:
            Boolean indicating whether generation should stop (True) or continue (False)
        """
        if self.is_first_token:
            self.first_token_time = time.time()
            self.is_first_token = False

        self.text_out_callback(output)
        self.num_tokens += 1

        if self.llm_interface.stop_generate:
            self.put_msg("Stopping generation.")
            return True  # Stop generation

        return False

    def text_conversation_run(
        self,
        params: LLMParams,
    ):
        """
        Main method that runs the text generation process.

        This method loads the model, processes the prompt, runs inference,
        collects metrics, and handles errors. It runs in a background thread.

        Args:
            params: LLM parameters including prompt and configuration options
        """
        try:
            self.llm_interface.load_model(params, callback=self.load_model_callback)

            # Reset metrics tracking
            self.num_tokens = 0
            self.start_time = time.time()
            self.first_token_time = 0
            self.last_token_time = 0
            self.is_first_token = True

            prompt = params.prompt
            full_prompt = convert_prompt(prompt)
            self.llm_interface.create_chat_completion(
                full_prompt, self.stream_function, params.max_tokens
            )

            # Calculate and send metrics
            self.last_token_time = time.time()
            metrics_data = {
                "type": "metrics",
                "num_tokens": self.num_tokens,
                "total_time": self.last_token_time - self.start_time,
                "overall_tokens_per_second": self.num_tokens
                / (self.last_token_time - self.start_time)
                if self.num_tokens > 0
                else 0,
                "second_plus_tokens_per_second": (self.num_tokens - 1)
                / (self.last_token_time - self.first_token_time)
                if self.num_tokens > 1
                else None,
                "first_token_latency": self.first_token_time - self.start_time
                if self.num_tokens > 0
                else None,
                "after_token_latency": (self.last_token_time - self.first_token_time)
                / (self.num_tokens - 1)
                if self.num_tokens > 1
                else None,
            }
            self.put_msg(metrics_data)
            self.put_msg({"type": "finish"})

        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            self.llm_interface.stop_generate = False
            self.finish = True
            self.singal.set()

    def generator(self):
        """
        Generator function that yields SSE-formatted messages from the queue.

        This function is returned by text_conversation and yields messages
        in SSE format (data:JSON\0) until generation is complete.

        Yields:
            SSE-formatted message strings
        """
        while True:
            while not self.msg_queue.empty():
                try:
                    data = self.msg_queue.get_nowait()
                    msg = f"data:{json.dumps(data)}\0"
                    print(msg)
                    yield msg
                except Empty(Exception):
                    break
            if not self.finish:
                self.singal.clear()
                self.singal.wait()
            else:
                break


# Default system prompt for the assistant
_default_prompt = {
    "role": "system",
    "content": "You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user. Please keep the output text language the same as the user input.",
}


def convert_prompt(prompt: List[Dict[str, str]]):
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
        chat_history.append({"role": "user", "content": prompt[i].get("question")})
        if i < prompt_len - 1:
            chat_history.append(
                {"role": "assistant", "content": prompt[i].get("answer")}
            )
        i = i + 1
    return chat_history


def process_rag(
    prompt: str,
    device: str,
    text_out_callback: Callable[[str, int], None] = None,
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
    import rag

    rag.to(device)
    query_success, context, rag_source = rag.query(prompt)
    if query_success:
        print("rag query input\r\n{}output:\r\n{}".format(prompt, context))
        prompt = RAG_PROMPT_FORMAT.format(prompt=prompt, context=context)
        if text_out_callback is not None:
            text_out_callback(rag_source, 2)
    return prompt
