"""
Llama.cpp LLM Adapter Module
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
#from model_downloader import NotEnoughDiskSpaceException, DownloadException
#from psutil._common import bytes2human
from llama_interface import LLMInterface
from llama_params import LLMParams


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
        print(f"exception:{str(ex)}")

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
    

    def stream_function(self, stream):  
        """
        Process the streaming output from the LLM and collect performance metrics.
        
        This function iterates through each token of the generated output,
        sends it to clients via the text_out_callback, and collects performance
        metrics such as token generation speed and latency.
        
        It handles different output formats based on the backend type.
        
        Args:
            stream: Iterator of generated tokens from the LLM
        """
        num_tokens = 0
        start_time = time.time()
        is_first = True
        first_token_time = 0.0
        last_token_time = 0.0

        for output in stream:
            if self.llm_interface.stop_generate:
                self.llm_interface.stop_generate = False
                break
            
            if self.llm_interface.get_backend_type() == "ipex_llm":
                # transformer style
                self.text_out_callback(output)
            else:
                # openai style
                self.text_out_callback(output["choices"][0]["delta"].get("content",""))
                num_tokens += 1

                if is_first:
                    first_token_time = time.time()
                    is_first = False

        last_token_time = time.time()

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

        self.put_msg(metrics_data)

        self.put_msg({"type": "finish"})

    def text_conversation_run(
        self,
        params: LLMParams,
    ):
        """
        Main method that runs the text generation process.
        
        This method loads the model, processes the prompt (including RAG if enabled),
        runs inference, and handles errors. It runs in a background thread.
        
        Args:
            params: LLM parameters including prompt and configuration options
        """
        try:
            self.llm_interface.load_model(params, callback=self.load_model_callback)
            
            prompt = params.prompt
            if params.enable_rag:
                last_prompt = prompt[prompt.__len__() - 1]
                last_prompt.__setitem__(
                    "question", process_rag(last_prompt.get("question"), params.device)
                )

            full_prompt = convert_prompt(prompt)
            stream = self.llm_interface.create_chat_completion(full_prompt, params.max_tokens)
            self.stream_function(stream)	
            
        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
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