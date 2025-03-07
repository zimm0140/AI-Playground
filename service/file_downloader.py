"""
File Downloader Module
---------------------
This module provides functionality for downloading files with progress tracking and resume capability.

The FileDownloader class handles downloading files from URLs, with support for:
- Resuming interrupted downloads
- Progress reporting via callbacks
- Download cancellation
- Automatic retries on failure
"""

from io import BufferedWriter
import os
import time
import traceback
from typing import Callable
import requests
from threading import Thread
from exceptions import DownloadException


class FileDownloader:
    """
    A utility class for downloading files with progress tracking and resume capability.
    
    This class supports downloading files from URLs with features such as progress reporting,
    download resumption, retry on failure, and cancellation.
    
    Attributes:
        on_download_progress: Callback function for reporting download progress.
            Called with (filename, bytes_downloaded, total_bytes, speed).
        on_download_completed: Callback function for reporting download completion.
            Called with (filename, error) where error is None if successful.
        url: The URL being downloaded.
        filename: The local path where the file is being saved.
        basename: The base name of the file being downloaded.
        total_size: Total size of the file in bytes.
        download_size: Number of bytes downloaded so far.
        download_stop: Flag to indicate if download should be stopped.
        prev_sec_download_size: Size at the previous progress report (for speed calculation).
    """
    on_download_progress: Callable[[str, int, int, int], None] = None
    on_download_completed: Callable[[str, Exception], None] = None
    url: str
    filename: str
    basename: str
    total_size: int
    download_size: int
    download_stop: bool
    prev_sec_download_size: int

    def __init__(self):
        """
        Initialize a new FileDownloader with default state.
        
        Sets up initial values for tracking download progress and status.
        """
        self.download_stop = False
        self.download_size = 0
        self.completed = False
        self.total_size = 0
        self.prev_sec_download_size = 0
        self.report_thread = None

    def download_file(self, url: str, file_path: str):
        """
        Download a file from the specified URL to the given local path.
        
        This method coordinates the download process, handling initialization,
        progress reporting, and cleanup. It notifies completion via the callback.
        
        Args:
            url: The URL to download from.
            file_path: The local path where the file will be saved.
        """
        self.url = url
        self.basename = os.path.basename(file_path)
        self.download_stop = False
        self.filename = file_path
        self.prev_sec_download_size = 0
        self.download_size = 0
        self.completed = False
        self.report_thread = None
        error = None
        report_thread = None
        try:
            response, fw = self.__init_download(self.url, self.filename)
            self.total_size = int(response.headers.get("Content-Length"))
            if self.on_download_progress is not None:
                report_thread = self.__start_report_download_progress()
            self.__start_download(response, fw)
        except Exception as e:
            error = e
        finally:
            self.completed = True
            if report_thread is not None:
                report_thread.join()

        if self.on_download_completed is not None:
            self.on_download_completed(self.basename, error)

    def __init_download(
        self, url: str, file_path: str
    ) -> tuple[requests.Response, BufferedWriter]:
        """
        Initialize the download by checking if the file exists and setting up appropriate requests.
        
        If the file already exists partially, this method will set up a Range request to resume
        the download from where it left off.
        
        Args:
            url: The URL to download from.
            file_path: The local path where the file will be saved.
            
        Returns:
            A tuple containing:
                - The HTTP response object with the download stream.
                - The file writer object for writing the downloaded content.
        """
        if os.path.exists(file_path):
            start_pos = os.path.getsize(file_path)
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            start_pos = 0

        if start_pos > 0:
            # download skip exists part
            response = requests.get(
                url,
                stream=True,
                verify=False,
                headers={"Range": f"bytes={start_pos}-"},
            )
            fw = open(file_path, "ab")
        else:
            response = requests.get(url, stream=True, verify=False)
            fw = open(file_path, "wb")

        return response, fw

    def __start_download(self, response: requests.Response, fw: BufferedWriter):
        """
        Perform the actual file download from the HTTP response stream.
        
        This method reads the response in chunks and writes them to the file.
        It includes retry logic to handle temporary network failures.
        
        Args:
            response: The HTTP response object with the download stream.
            fw: The file writer object for writing the downloaded content.
            
        Raises:
            DownloadException: If the download fails after multiple retries.
        """
        retry = 0
        while True:
            try:
                with response:
                    with fw:
                        for bytes in response.iter_content(chunk_size=4096):
                            self.download_size += bytes.__len__()
                            fw.write(bytes)

                            if self.download_stop:
                                print(
                                    f"FileDownloader thread {Thread.native_id} exit by stop"
                                )
                                break
                break
            except Exception:
                traceback.print_exc()
                retry += 1
                if retry > 3:
                    raise DownloadException(self.url)
                else:
                    print(
                        f"FileDownloader thread {Thread.native_id} retry {retry} times"
                    )
                    time.sleep(1)
                    response, fw = self.__init_download(self.url, self.filename)

    def __start_report_download_progress(self):
        """
        Start a background thread for reporting download progress.
        
        Returns:
            The created thread object that reports progress.
        """
        report_thread = Thread(target=self.__report_download_progress)
        report_thread.start()
        return report_thread

    def __report_download_progress(self):
        """
        Periodically report download progress via the callback.
        
        This method is meant to be run in a separate thread. It calls the progress callback
        once per second, providing the current download status and speed.
        """
        while not self.download_stop and not self.completed:
            self.on_download_progress(
                self.basename,
                self.download_size,
                self.total_size,
                self.download_size - self.prev_sec_download_size,
            )

            self.prev_sec_download_size = self.download_size
            time.sleep(1)

    def stop_download(self):
        """
        Signal the download to stop.
        
        This sets a flag that will cause the download loop to exit 
        at the next convenient opportunity.
        """
        self.download_stop = True
