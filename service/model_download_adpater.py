"""
Model Downloader Adapter Module
------------------------------
This module provides a high-level adapter for downloading AI models from various sources.

It abstracts away the specifics of different download sources by providing:
- A unified interface for initiating downloads
- Server-Sent Events (SSE) for real-time progress reporting
- Consistent error handling across download sources
- Support for both local file downloads and Hugging Face Hub models

The primary class is Model_Downloader_Adapter, which coordinates downloads
and streams progress events back to the client.
"""

import json
import os
import threading
from queue import Empty, Queue

import aipg_utils as utils
import realesrgan
from file_downloader import FileDownloader
from model_downloader import (
    DownloadException,
    HFPlaygroundDownloader,
    NotEnoughDiskSpaceException,
)
from psutil._common import bytes2human
from web_request_bodies import DownloadModelData


class Model_Downloader_Adapter:
    """
    Adapter class that unifies the interface for downloading models from different sources.

    This class wraps both general file downloads and Hugging Face model downloads,
    providing a consistent interface and event stream for progress reporting.

    Attributes:
        msg_queue: Queue for messages to be sent to the client
        finish: Flag indicating if download is complete
        singal: Event for synchronizing the generator with download threads
        file_downloader: Downloader for general file downloads
        hf_downloader: Downloader for Hugging Face models
        has_error: Flag indicating if an error occurred during download
        user_stop: Flag indicating if the user requested to stop the download
    """

    msg_queue: Queue
    finish: bool
    singal: threading.Event
    file_downloader: FileDownloader
    hf_downloader: HFPlaygroundDownloader
    has_error: bool
    user_stop: bool

    def __init__(self, hf_token=None):
        """
        Initialize the Model_Downloader_Adapter.

        Sets up the message queue, events, and downloaders with appropriate callbacks.

        Args:
            hf_token: Optional Hugging Face authentication token for accessing gated models
        """
        self.msg_queue = Queue(-1)
        self.finish = False
        self.user_stop = False
        self.singal = threading.Event()
        self.file_downloader = FileDownloader()
        self.file_downloader.on_download_progress = self.download_model_progress_callback
        self.file_downloader.on_download_completed = self.download_model_completed_callback
        self.hf_downloader = HFPlaygroundDownloader(hf_token)
        self.hf_downloader.on_download_progress = self.download_model_progress_callback
        self.hf_downloader.on_download_completed = self.download_model_completed_callback

    def put_msg(self, data):
        """
        Add a message to the queue and signal waiting threads.

        Args:
            data: Message data to be sent to the client
        """
        self.msg_queue.put_nowait(data)
        self.singal.set()

    def download_model_progress_callback(self, repo_id: str, download_size: int, total_size: int, speed: int):
        """
        Callback for reporting download progress.

        Formats download progress information and adds it to the message queue.

        Args:
            repo_id: Identifier for the model being downloaded
            download_size: Number of bytes downloaded so far
            total_size: Total size of the download in bytes
            speed: Current download speed in bytes per second
        """
        print(
            f"download {repo_id} {bytes2human(download_size)}/{bytes2human(total_size)} speed {bytes2human(speed)}",
        )
        data = {
            "type": "download_model_progress",
            "repo_id": repo_id,
            "download_size": bytes2human(download_size),
            "total_size": bytes2human(total_size),
            "percent": round(download_size / total_size * 100, 2),
            "speed": f"{bytes2human(speed)}/s",
        }
        self.put_msg(data)

    def download_model_completed_callback(self, repo_id: str, ex: Exception):
        """
        Callback triggered when a download completes or fails.

        Reports completion status and cleans up the adapter instance.

        Args:
            repo_id: Identifier for the model that was downloaded
            ex: Exception if download failed, None if successful
        """
        global _adapter
        if ex is not None:
            self.put_msg({"type": "error", "err_type": "download_exception"})
            self.has_error = True
            self.finish = True
        else:
            self.put_msg({"type": "download_model_completed", "repo_id": repo_id})
        _adapter = None

    def error_callback(self, ex: Exception):
        """
        Handle various types of download errors.

        Maps different exception types to appropriate error messages
        and adds them to the message queue.

        Args:
            ex: The exception that occurred
        """
        self.has_error = True
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
        elif isinstance(ex, RuntimeError):
            self.put_msg({"type": "error", "err_type": "runtime_error"})
        else:
            self.put_msg({"type": "error", "err_type": "unknown_exception"})
        print(f"exception:{str(ex)}")

    def download(self, model_download_list: list[DownloadModelData]):
        """
        Start downloading a list of models.

        Launches a download thread and returns a generator for streaming events.

        Args:
            model_download_list: List of models to download

        Returns:
            Generator yielding SSE messages for progress updates
        """
        self.has_error = False
        threading.Thread(target=self.__start_download, kwargs={"model_download_list": model_download_list}).start()
        return self.generator()

    def __start_download(self, model_download_list: list[DownloadModelData]):
        """
        Download thread that processes each model in the list.

        Handles different model types by using the appropriate downloader,
        and reports completion or errors.

        Args:
            model_download_list: List of models to download
        """
        self.finish = False
        self.user_stop = False
        try:
            for item in model_download_list:
                if self.user_stop:
                    break
                if self.has_error:
                    break
                if item.type == 4:
                    # Type 4 is ESRGAN, needs special handling with file_downloader
                    self.file_downloader.download_file(
                        realesrgan.ESRGAN_MODEL_URL,
                        os.path.join(
                            utils.get_model_path(item.type, item.backend), os.path.basename(realesrgan.ESRGAN_MODEL_URL),
                        ),
                    )
                else:
                    # All other model types use hf_downloader
                    self.hf_downloader.download(item.repo_id, item.type, item.backend)
            self.put_msg({"type": "allComplete"})
            self.finish = True
        except Exception as ex:
            self.error_callback(ex)

    def stop_download(self):
        """
        Stop any ongoing downloads.

        Sets a flag to stop future downloads and signals the active downloaders
        to cancel their operations.
        """
        self.user_stop = True
        if not self.file_downloader.completed:
            self.file_downloader.stop_download()
        if not self.hf_downloader.completed:
            self.hf_downloader.stop_download()

    def generator(self):
        """
        Generator that yields download progress events as SSE messages.

        Continuously checks the message queue for new messages and yields
        them until the download is complete or stopped.

        Yields:
            SSE-formatted string messages containing JSON data
        """
        while True:
            while not self.msg_queue.empty():
                try:
                    data = self.msg_queue.get_nowait()
                    msg = f"data:{json.dumps(data)}\0"
                    yield msg
                except Empty:
                    break
            if not self.finish:
                self.singal.clear()
                self.singal.wait()
            else:
                break


# Global singleton instance of the adapter
_adapter: Model_Downloader_Adapter = None
