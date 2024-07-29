from io import BufferedWriter
import time
import traceback
from typing import Callable, Optional, Tuple
import requests
from threading import Thread
from exceptions import DownloadException
from pathlib import Path


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

    def __init__(self):
        self.download_stop = False
        self.download_size = 0
        self.completed = False
        self.total_size = 0
        self.prev_sec_download_size = 0
        self.report_thread: Optional[Thread] = None

    def download_file(self, url: str, file_path: str) -> None:
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
        file_path = Path(file_path)
        if file_path.exists():
            start_pos = file_path.stat().st_size
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            start_pos = 0

        if start_pos > 0:
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

    def __start_download(self, response: requests.Response, fw: BufferedWriter) -> None:
        retry = 0
        while True:
            try:
                with response, fw:
                    for bytes in response.iter_content(chunk_size=4096):
                        self.download_size += len(bytes)
                        fw.write(bytes)

                        if self.download_stop:
                            print(f"FileDownloader thread {Thread.native_id} exit by stop")
                            break
                break
            except Exception:
                traceback.print_exc()
                retry += 1
                if retry > 3:
                    raise DownloadException(self.url)
                else:
                    print(f"FileDownloader thread {Thread.native_id} retry {retry} times")
                    time.sleep(1)
                    response, fw = self.__init_download(self.url, self.filename)

    def __start_report_download_progress(self) -> Thread:
        report_thread = Thread(target=self.__report_download_progress)
        report_thread.start()
        return report_thread

    def __report_download_progress(self) -> None:
        while not self.download_stop and not self.completed:
            self.on_download_progress(
                self.basename,
                self.download_size,
                self.total_size,
                self.download_size - self.prev_sec_download_size,
            )

            self.prev_sec_download_size = self.download_size
            time.sleep(1)

    def stop_download(self) -> None:
        self.download_stop = True