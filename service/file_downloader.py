import logging
import time
import traceback
from threading import Thread, Lock
from pathlib import Path
from typing import Callable, Optional, Tuple
from io import BufferedWriter

import requests

from exceptions import DownloadException


class FileDownloader:
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
        Download a file from the given URL to the specified file path.

        Args:
            url (str): The URL of the file to download.
            file_path (str): The local file path to save the downloaded file.
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
        Initialize the download process, supporting resume if the file already exists.

        Args:
            url (str): The URL of the file to download.
            file_path (str): The local file path to save the downloaded file.

        Returns:
            Tuple[requests.Response, BufferedWriter]: The HTTP response and file writer object.
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
        Start the download process and handle retries in case of failures.

        Args:
            response (requests.Response): The HTTP response object.
            fw (BufferedWriter): The file writer object.
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
                            logging.info(f"FileDownloader thread {Thread.current_thread().ident} exit by stop")
                            break
                break
            except Exception:
                traceback.print_exc()
                retry += 1
                if retry > 3:
                    raise DownloadException(self.url)
                else:
                    logging.info(f"FileDownloader thread {Thread.current_thread().ident} retry {retry} times")
                    time.sleep(1)
                    response, fw = self.__init_download(self.url, self.filename)

    def __start_report_download_progress(self) -> Thread:
        """
        Start a separate thread to report download progress.

        Returns:
            Thread: The thread object for reporting download progress.
        """
        report_thread = Thread(target=self.__report_download_progress)
        report_thread.start()
        return report_thread

    def __report_download_progress(self) -> None:
        """
        Report download progress periodically.
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
        Stop the download process.
        """
        self.download_stop = True