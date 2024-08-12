# --- Standard Library Imports ---
import threading
from queue import Empty, Queue
import json
import traceback

# --- Third-Party Imports ---
from psutil._common import bytes2human

# --- Local Imports ---
from service import llm_biz
from service.model_downloader import NotEnoughDiskSpaceException, DownloadException


class LLM_SSE_Adapter:
    """
    Adapter class to handle SSE (Server-Sent Events) for an LLM (Large Language Model).
    """

    def __init__(self):
        """
        Initializes the LLM_SSE_Adapter with a message queue, finish flag, and signal event.
        """
        self.msg_queue = Queue()
        self.finish = False
        self.signal = threading.Event()

    def put_msg(self, data: dict):
        """
        Puts a message in the queue and sets the signal event.

        :param data: The data to put in the message queue.
        """
        self.msg_queue.put_nowait(data)
        self.signal.set()

    def load_model_callback(self, event: str):
        """
        Callback for when a model is loaded.

        :param event: The event message.
        """
        self.put_msg({"type": "load_model", "event": event})

    def text_in_callback(self, msg: str):
        """
        Callback for text input.

        :param msg: The input message.
        """
        self.put_msg({"type": "text_in", "value": msg})

    def text_out_callback(self, msg: str, type: int = 1):
        """
        Callback for text output.

        :param msg: The output message.
        :param type: The type of the message.
        """
        self.put_msg({"type": "text_out", "value": msg, "dtype": type})

    def first_latency_callback(self, first_latency: str):
        """
        Callback for the first token latency.

        :param first_latency: The first token latency.
        """
        self.put_msg({"type": "first_token_latency", "value": first_latency})

    def after_latency_callback(self, after_latency: str):
        """
        Callback for the after token latency.

        :param after_latency: The after token latency.
        """
        self.put_msg({"type": "after_token_latency", "value": after_latency})

    def sr_latency_callback(self, sr_latency: str):
        """
        Callback for the speech recognition latency.

        :param sr_latency: The speech recognition latency.
        """
        self.put_msg({"type": "sr_latency", "value": sr_latency})

    def error_callback(self, ex: Exception):
        """
        Callback for handling errors.

        :param ex: The exception that occurred.
        """
        if isinstance(ex, NotImplementedError) and str(ex) == "Access to repositories lists is not implemented.":
            self.put_msg({"type": "error", "err_type": "repositories_not_found"})
        elif isinstance(ex, NotEnoughDiskSpaceException):
            self.put_msg({
                "type": "error",
                "err_type": "not_enough_disk_space",
                "need": bytes2human(ex.requires_space),
                "free": bytes2human(ex.free_space),
            })
        elif isinstance(ex, DownloadException):
            self.put_msg({"type": "error", "err_type": "download_exception"})
        elif isinstance(ex, llm_biz.StopGenerateException):
            pass  # No action needed
        elif isinstance(ex, RuntimeError):
            self.put_msg({"type": "error", "err_type": "runtime_error"})
        else:
            self.put_msg({"type": "error", "err_type": "unknown_exception"})
        print(f"exception: {str(ex)}")

    def text_conversation(self, params: llm_biz.LLMParams):
        """
        Starts a text conversation in a new thread.

        :param params: Parameters for the LLM conversation.
        :return: Generator that yields messages from the queue.
        """
        thread = threading.Thread(target=self.text_conversation_run, args=[params])
        thread.start()
        return self.generator()

    def text_conversation_run(self, params: llm_biz.LLMParams):
        """
        Runs the text conversation.

        :param params: Parameters for the LLM conversation.
        """
        try:
            llm_biz.chat(
                params=params,
                load_model_callback=self.load_model_callback,
                text_out_callback=self.text_out_callback,
                error_callback=self.error_callback,
            )
            self.put_msg({"type": "finish"})
        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            self.finish = True
            self.signal.set()

    def generator(self):
        """
        Generator that yields messages from the queue.

        :yield: Message from the queue.
        """
        while True:
            while not self.msg_queue.empty():
                try:
                    data = self.msg_queue.get_nowait()
                    yield f"data:{json.dumps(data)}\0"
                except Empty:
                    break
            if not self.finish:
                self.signal.clear()
                self.signal.wait()
            else:
                break
