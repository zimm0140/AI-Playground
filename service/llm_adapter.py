"""
LLM SSE Adapter Module
----------------------
This module provides an adapter for integrating Large Language Model (LLM) operations with Server-Sent Events (SSE).

It facilitates streaming of messages related to model loading, text input, text output, latency measurements,
metrics reporting, and error handling. The adapter wraps the functionalities provided by the llm_biz module
and also handles errors from model downloads using model_downloader exceptions.
"""

import json
import threading
import traceback
from queue import Empty, Queue

import llm_biz
from model_downloader import DownloadException, NotEnoughDiskSpaceException
from psutil._common import bytes2human


class LlmSseAdapter:
    """
    Adapter class for managing SSE messages for LLM interactions.

    This class maintains an internal message queue and provides callbacks for various events such as model
    loading, text input and output, latency measurements, and error handling. It also manages the generation
    of SSE-formatted messages from the internal queue.

    Attributes:
        msg_queue (Queue): Queue to store outgoing messages.
        finish (bool): Flag indicating whether the processing is finished.
        singal (threading.Event): Event used to signal waiting threads when new messages are available.
        metrics_data: Container for storing metrics data.
    """

    msg_queue: Queue
    finish: bool
    singal: threading.Event

    def __init__(self):
        """
        Initialize the LLM_SSE_Adapter with an empty message queue and unsignaled event.
        """
        self.msg_queue = Queue(-1)
        self.finish = False
        self.singal = threading.Event()
        self.metrics_data = None

    def put_msg(self, data):
        """
        Add a message to the internal queue and signal waiting threads.

        Args:
            data: The message data to be enqueued.
        """
        self.msg_queue.put_nowait(data)
        self.singal.set()

    def load_model_callback(self, event: str):
        """
        Callback for model loading events.

        Args:
            event (str): A string representing the event status (e.g., 'start', 'finish').
        """
        data = {"type": "load_model", "event": event}
        self.put_msg(data)

    def text_in_callback(self, msg: str):
        """
        Callback for incoming text messages.

        Args:
            msg (str): The incoming text message.
        """
        data = {"type": "text_in", "value": msg}
        self.put_msg(data)

    def text_out_callback(self, msg: str, type=1):
        """
        Callback for outgoing text messages.

        Args:
            msg (str): The text message to be sent.
            type: An identifier for the message type (default is 1).
        """
        data = {"type": "text_out", "value": msg, "dtype": type}
        self.put_msg(data)

    def first_latency_callback(self, first_latency: str):
        """
        Callback for reporting the first token latency.

        Args:
            first_latency (str): The latency value for the first token.
        """
        data = {"type": "first_token_latency", "value": first_latency}
        self.put_msg(data)

    def after_latency_callback(self, after_latency: str):
        """
        Callback for reporting the latency after the first token.

        Args:
            after_latency (str): The latency value after the first token.
        """
        data = {"type": "after_token_latency", "value": after_latency}
        self.put_msg(data)

    def sr_latency_callback(self, sr_latency: str):
        """
        Callback for reporting super-resolution latency.

        Args:
            sr_latency (str): The super-resolution latency value.
        """
        data = {"type": "sr_latency", "value": sr_latency}
        self.put_msg(data)

    def error_callback(self, ex: Exception):
        """
        Callback for handling errors during LLM operations.

        Depending on the exception type, an appropriate error message is enqueued.

        Args:
            ex (Exception): The exception that was raised.
        """
        if isinstance(ex, NotImplementedError) and ex.__str__() == "Access to repositories lists is not implemented.":
            self.put_msg(
                {
                    "type": "error",
                    "err_type": "repositories_not_found",
                },
            )
        elif isinstance(ex, NotEnoughDiskSpaceException):
            self.put_msg(
                {
                    "type": "error",
                    "err_type": "not_enough_disk_space",
                    "need": bytes2human(ex.requires_space),
                    "free": bytes2human(ex.free_space),
                },
            )
        elif isinstance(ex, DownloadException):
            self.put_msg({"type": "error", "err_type": "download_exception"})
        elif isinstance(ex, llm_biz.StopGenerateException):
            pass
        elif isinstance(ex, RuntimeError):
            self.put_msg({"type": "error", "err_type": "runtime_error"})
        else:
            self.put_msg({"type": "error", "err_type": "unknown_exception"})
        print(f"exception:{str(ex)}")

    def metrics_callback(self, msg: dict):
        """
        Callback for capturing performance metrics.

        Args:
            msg (dict): Metrics data as a dictionary.
        """
        self.metrics_data = msg

    def text_conversation(self, params: llm_biz.LLMParams):
        """
        Initiate a text conversation session using LLM parameters.

        This function starts a separate thread that runs the conversation
        and returns a generator for streaming SSE messages.

        Args:
            params (llm_biz.LLMParams): Configuration parameters for the chat session.

        Returns:
            Generator yielding SSE-formatted messages.
        """
        thread = threading.Thread(
            target=self.text_conversation_run,
            args=[params],
        )
        thread.start()
        return self.generator()

    def text_conversation_run(
        self,
        params: llm_biz.LLMParams,
    ):
        """
        Execute the text conversation by invoking the llm_biz.chat function.

        This method calls the llm_biz.chat function with the appropriate callbacks and
        enqueues the resulting metrics and finish messages upon completion.

        Args:
            params (llm_biz.LLMParams): Configuration parameters for the chat session.
        """
        try:
            llm_biz.chat(
                params=params,
                load_model_callback=self.load_model_callback,
                text_out_callback=self.text_out_callback,
                error_callback=self.error_callback,
                metrics_callback=self.metrics_callback,
            )
            self.put_msg(self.metrics_data)
            self.put_msg({"type": "finish"})

        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            self.finish = True
            self.singal.set()

    def generator(self):
        """
        Generator yielding SSE-formatted messages from the internal queue.

        Continuously checks the message queue for new messages and yields them until
        the finish flag is set.

        Yields:
            str: SSE-formatted string messages containing JSON data.
        """
        while True:
            while not self.msg_queue.empty():
                try:
                    data = self.msg_queue.get_nowait()
                    msg = f"data:{json.dumps(data)}\0"
                    yield msg
                except Empty(Exception):
                    break
            if not self.finish:
                self.singal.clear()
                self.singal.wait()
            else:
                break
