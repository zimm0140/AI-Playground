import logging
import time
from io import BufferedWriter
from pathlib import Path
from threading import Thread, Lock
from typing import Callable, Optional, Tuple

import requests
import traceback

from exceptions import DownloadException

class FileDownloader:
    """
    A class for downloading files from a given URL to a local file system, 
    with support for progress reporting, download completion notification, 
    and stopping the download process.

    Attributes:
        on_download_progress (Optional[Callable[[str, int, int, int], None]]): 
            A callback function that reports the download progress. It receives 
            the filename, the number of bytes downloaded so far, the total file size, 
            and the download speed in bytes per second.
        on_download_completed (Optional[Callable[[str, Optional[Exception]], None]]): 
            A callback function that is called when the download is completed. It receives 
            the filename and an optional exception if an error occurred during the download.
        url (str): The URL of the file being downloaded.
        filename (str): The full path of the local file where the downloaded content is saved.
        basename (str): The name of the file being downloaded (without the path).
        total_size (int): The total size of the file being downloaded in bytes.
        download_size (int): The number of bytes downloaded so far.
        download_stop (bool): A flag indicating whether the download process should be stopped.
        prev_sec_download_size (int): The number of bytes downloaded in the previous second, 
            used for calculating the download speed.
        lock (Lock): A threading lock to ensure thread-safe access to shared resources.
    """
    on_download_progress: Optional[Callable[[str, int, int, int], None]] = None
    on_download_completed: Optional[Callable[[str, Optional[Exception]], None]] = None
    url: str
    filename: str
    basename: str
    total_size: int
    download_size: int
    download_stop: bool
    prev_sec_download_size: int
    lock: Lock

    def __init__(self):
        """
        Initializes the FileDownloader with default values for all attributes.
        Sets up a lock for thread safety and initializes the logger.
        """
        self.download_stop = False
        self.download_size = 0
        self.completed = False
        self.total_size = 0
        self.prev_sec_download_size = 0
        self.report_thread: Optional[Thread] = None
        self.lock = Lock()
        logging.basicConfig(level=logging.INFO)

    def download_file(self, url: str, file_path: str) -> None:
        """
        Initiates the file download process.

        This method begins the download of the file from the provided URL to the specified 
        local file path. It manages the download process, handles errors, and invokes 
        callbacks for progress reporting and completion.

        Args:
            url (str): The URL of the file to download.
            file_path (str): The local file path where the downloaded file will be saved.

        Raises:
            DownloadException: Raised if the download fails after multiple retries.
        """
        self.url = url
        self.basename = Path(file_path).name
        self.download_stop = False
        self.filename = file_path
        self.prev_sec_download_size = 0
        self.download_size = 0
        self.completed = False
        self.report_thread = None
        error: Optional[Exception] = None
        report_thread: Optional[Thread] = None
        try:
            response, fw = self.__init_download(self.url, self.filename)
            self.total_size = int(response.headers.get("Content-Length", 0))
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

    def __init_download(self, url: str, file_path: str) -> Tuple[requests.Response, BufferedWriter]:
        """
        Prepares the download by initiating the HTTP request and setting up the local file.

        This method supports resuming an interrupted download if the file already exists 
        locally by requesting the remaining portion of the file from the server.

        Args:
            url (str): The URL of the file to download.
            file_path (str): The local file path where the downloaded file will be saved.

        Returns:
            Tuple[requests.Response, BufferedWriter]: 
                A tuple containing the HTTP response object and a buffered writer object 
                for writing the downloaded content to the file.

        Raises:
            requests.RequestException: If the HTTP request fails.
            OSError: If the file operations (opening, writing) fail.
        """
        file_path = Path(file_path)
        if file_path.exists():
            start_pos = file_path.stat().st_size
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            start_pos = 0

        headers = {"Range": f"bytes={start_pos}-"} if start_pos > 0 else {}
        response = requests.get(url, stream=True, verify=False, headers=headers)
        mode = "ab" if start_pos > 0 else "wb"
        fw = open(file_path, mode)

        return response, fw

    def __start_download(self, response: requests.Response, fw: BufferedWriter) -> None:
        """
        Handles the main file download process, including retries on failure.

        This method reads the file content from the HTTP response in chunks and writes 
        it to the local file. If the download is interrupted by an exception, it will retry 
        up to three times before failing.

        Args:
            response (requests.Response): The HTTP response object containing the file data.
            fw (BufferedWriter): The file writer object used to save the downloaded content.

        Raises:
            DownloadException: If the download fails after three retries.
        """
        retry = 0
        while True:
            try:
                with response, fw:
                    for bytes in response.iter_content(chunk_size=4096):
                        with self.lock:
                            self.download_size += len(bytes)
                        fw.write(bytes)

                        if self.download_stop:
                            logging.info(f"FileDownloader thread {threading.current_thread().ident} exit by stop")
                            break
                break
            except Exception:
                traceback.print_exc()
                retry += 1
                if retry > 3:
                    raise DownloadException(self.url)
                else:
                    logging.info(f"FileDownloader thread {threading.current_thread().ident} retry {retry} times")
                    time.sleep(1)
                    response, fw = self.__init_download(self.url, self.filename)

    def __start_report_download_progress(self) -> Thread:
        """
        Starts a separate thread to periodically report the download progress.

        The thread repeatedly invokes the `on_download_progress` callback (if provided) 
        with the current download status.

        Returns:
            Thread: The thread object responsible for reporting download progress.
        """
        report_thread = Thread(target=self.__report_download_progress)
        report_thread.start()
        return report_thread

    def __report_download_progress(self) -> None:
        """
        Reports download progress at regular intervals (1 second).

        This method is intended to run in a separate thread. It reports the download 
        progress by calling the `on_download_progress` callback with the current 
        download state, including the amount downloaded and the download speed.
        """
        while not self.download_stop and not self.completed:
            with self.lock:
                downloaded_size = self.download_size
            self.on_download_progress(
                self.basename,
                downloaded_size,
                self.total_size,
                downloaded_size - self.prev_sec_download_size,
            )

            self.prev_sec_download_size = downloaded_size
            time.sleep(1)

    def stop_download(self) -> None:
        """
        Stops the download process.

        Sets a flag that signals the download process to stop after the current chunk 
        is processed. This method is typically called from another thread to gracefully 
        stop the download.
        """
        self.download_stop = True