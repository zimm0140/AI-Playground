"""
Stable Diffusion Server-Sent Events Adapter Module
-------------------------------------------------
This module implements an adapter for Stable Diffusion image generation that uses
Server-Sent Events (SSE) to stream generation progress and results back to the client.

The SD_SSE_Adapter class handles:
- Model downloading and loading progress communication
- Step-by-step generation progress updates with optional image previews
- Final image output and saving
- Error handling for various generation issues
- Generation history logging
"""

import json
import os
import threading
import traceback
from datetime import datetime
from queue import Empty, Queue
from typing import Any

import aipg_utils as utils
import paint_biz
from model_downloader import DownloadException, NotEnoughDiskSpaceException
from PIL import Image
from psutil._common import bytes2human


class SD_SSE_Adapter:
    """
    Adapter class that uses Server-Sent Events to stream Stable Diffusion generation progress.

    Acts as a bridge between the Stable Diffusion generation process and the web API,
    collecting events and streaming them to the client.

    Attributes:
        msg_queue: Queue to store messages to be sent to the client
        finish: Boolean flag indicating if generation is complete
        singal: Threading event for controlling message flow
        url_root: Base URL of the server
        save_image_path: Directory path where generated images are saved
    """

    msg_queue: Queue
    finish: bool
    singal: threading.Event
    url_root: str
    save_image_path: str

    def __init__(self, url_root: str):
        """
        Initialize the SD_SSE_Adapter with the server's URL root.

        Sets up message queue, control flags, and determines the appropriate
        output path for saving generated images based on the user's environment.

        Args:
            url_root: Base URL of the server
        """
        self.msg_queue = Queue(-1)
        self.finish = False
        self.singal = threading.Event()
        self.url_root = url_root
        if os.getenv("USERPROFILE"):
            self.save_image_path = os.path.join(os.getenv("USERPROFILE"), "Documents", "AI-Playground", "media")
        elif os.getenv("HOME"):
            self.save_image_path = os.path.join(os.getenv("HOME"), "AI-Playground", "media")
        else:
            self.save_image_path = os.path.join("static", "sd_out")

    def put_msg(self, data):
        """
        Add a message to the queue and signal that new data is available.

        Args:
            data: Message data to be queued for sending to the client
        """
        self.msg_queue.put_nowait(data)
        self.singal.set()

    def download_model_progress_callback(self, repo_id: str, download_size: int, total_size: int, speed: int):
        """
        Callback for tracking model download progress.

        Creates a progress message with download statistics and adds it to the message queue.

        Args:
            repo_id: Repository ID of the model being downloaded
            download_size: Current bytes downloaded
            total_size: Total bytes to download
            speed: Download speed in bytes per second
        """
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
        Callback triggered when model download completes or fails.

        Args:
            repo_id: Repository ID of the downloaded model
            ex: Exception if download failed, None if successful
        """
        if ex is not None:
            self.put_msg({"type": "error", "value": "DownloadModelFailed"})
        else:
            self.put_msg({"type": "download_model_completed", "repo_id": repo_id})

    def load_model_callback(self, event: str):
        """
        Callback for model loading events.

        Args:
            event: Description of the current loading stage
        """
        data = {"type": "load_model", "event": event}
        self.put_msg(data)

    def load_model_components_callback(self, event: str):
        """
        Callback for loading specific model components.

        Args:
            event: Description of the component being loaded
        """
        data = {"type": "load_model_components", "event": event}
        self.put_msg(data)

    def step_end_callback(
        self,
        index: int,
        step: int,
        total_step: int,
        preview_enabled: bool,
        image: Image.Image | None,
    ):
        """
        Callback triggered at the end of each generation step.

        Sends progress updates to the client and includes preview images
        if enabled and available.

        Args:
            index: Index of the current image being generated
            step: Current generation step number
            total_step: Total number of steps for generation
            preview_enabled: Whether preview images are enabled
            image: Preview image if available, None otherwise
        """
        if preview_enabled and image is not None:
            image = utils.image_to_base64(image)
        elif not preview_enabled:
            image = f"{self.url_root}/static/assets/aipg.png"

        data = {
            "type": "step_end",
            "index": index,
            "step": step,
            "total_step": total_step,
            "image": image,
        }
        self.put_msg(data)

    def image_out_callback(
        self,
        index: int,
        image: Image.Image | None,
        params: paint_biz.TextImageParams = None,
        safe_check_pass: bool = True,
    ):
        """
        Callback triggered when a final image is generated.

        Saves the image to disk, logs generation parameters, and sends
        the image location and metadata to the client.

        Args:
            index: Index of the generated image
            image: The final generated image
            params: Parameters used for generation
            safe_check_pass: Whether the image passed safety checks
        """
        now = datetime.now()
        folder = now.strftime("%d_%m_%Y")
        base_name = now.strftime("%H%M%S")
        image_name = f"{base_name}.png"
        filename = os.path.join(self.save_image_path, folder, image_name)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir)
        image.save(filename)
        utils.cache_file(filename, os.path.getsize(filename))

        response_params = self.get_response_params(image, os.path.getsize(filename), params)
        try:
            self.log_to_file(params, folder, base_name)
        except Exception:
            traceback.print_exc()
            pass

        image_location = f"{folder}/{image_name}"

        data = {
            "type": "image_out",
            "index": index,
            "image": image_location,
            "params": response_params,
            "safe_check_pass": safe_check_pass,
        }
        self.put_msg(data)

    def error_callback(self, ex: Exception):
        """
        Callback for handling errors during generation.

        Maps different exception types to appropriate error messages.

        Args:
            ex: The exception that occurred
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
        elif isinstance(ex, paint_biz.StopGenerateException):
            pass
        elif isinstance(ex, RuntimeError):
            self.put_msg({"type": "error", "err_type": "runtime_error"})
        else:
            self.put_msg({"type": "error", "err_type": "unknown_exception"})
        print(f"exception:{str(ex)}")

    def generate(self, params: paint_biz.TextImageParams):
        """
        Start image generation in a separate thread and return a generator for streaming events.

        Args:
            params: Parameters for image generation

        Returns:
            A generator yielding SSE messages
        """
        thread = threading.Thread(
            target=self.generate_run,
            args=[params],
        )
        thread.start()
        return self.generator()

    def generate_run(
        self,
        params: paint_biz.TextImageParams
        | paint_biz.ImageToImageParams
        | paint_biz.UpscaleImageParams
        | paint_biz.InpaintParams
        | paint_biz.OutpaintParams,
    ):
        """
        Run the image generation process with the given parameters.

        Sets up the necessary callbacks and handles exceptions.

        Args:
            params: Parameters for image generation, can be of various types
                   depending on the generation mode
        """
        try:
            paint_biz.load_model_callback = self.load_model_callback
            paint_biz.load_model_components_callback = self.load_model_components_callback
            paint_biz.step_end_callback = self.step_end_callback
            paint_biz.image_out_callback = self.image_out_callback
            paint_biz.download_progress_callback = self.download_model_progress_callback
            paint_biz.download_completed_callback = self.download_model_completed_callback
            paint_biz.generate(params=params)
        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex)
        finally:
            self.finish = True
            self.singal.set()

    def generator(self):
        """
        Generator function that yields SSE messages from the queue.

        Continues yielding messages until generation is finished.

        Yields:
            Formatted SSE messages containing JSON data
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

    def get_response_params(self, image: Image.Image, size: int, params: paint_biz.TextImageParams):
        """
        Extract and format parameters to include in the response.

        Creates a dictionary of parameters for the client, filtering out
        unnecessary or large parameters.

        Args:
            image: The generated image
            size: Size of the image file in bytes
            params: Original generation parameters

        Returns:
            Dictionary of formatted parameters for the response
        """
        response_params = {
            "width": image.width,
            "height": image.height,
            "size": bytes2human(size),
        }

        for key, value in params.__dict__.items():
            if key in [
                "generate_number",
                "image_preview",
                "width",
                "height",
            ] or isinstance(value, Image.Image):
                continue
            response_params.__setitem__(key, value)

        return response_params

    def log_to_file(self, params: Any, folder: str, base_name: str):
        """
        Log generation parameters and history to files.

        Creates and updates a history.json file with generation parameters,
        and sets up an HTML viewer if it doesn't exist.

        Args:
            params: Generation parameters
            folder: Directory where the image is saved
            base_name: Base filename for the image
        """
        from shutil import copyfile

        image_folder_path = os.path.join(self.save_image_path, f"{folder}")
        json_path = os.path.join(image_folder_path, "history.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r+") as f:
                    f.seek(12)
                    history_json = json.load(f)
            except Exception:
                os.remove(json_path)
                history_json = []
        else:
            history_json = []

        param_list = []
        for k, v in params.__dict__.items():
            if k in ("generate_number", "image_preview"):
                continue
            if k in ("image", "mask_image"):
                # currently, this option does not occur and would be, moreover,
                # explicitly filtered out in get_response_params(). It is therefore
                # uncertain, from where the reference images would be drawn from.
                # This clause should, thus, be changed once the need occurs
                base_output = os.path.abspath("./static/")
                save_path = os.path.abspath(str(v))
                save_path = save_path.replace(base_output, "../../").replace("\\", "/")
                param_list.append(
                    {
                        "name": k,
                        "type": "image",
                        "value": save_path,
                    },
                )
            else:
                param_list.append(
                    {
                        "name": k,
                        "value": v,
                        "type": "normal",
                    },
                )

        history_item = {
            "out_image": f"./{base_name}.png",
            "params": param_list,
        }

        history_json.insert(0, history_item)

        with open(json_path, "w") as f:
            f.write("let history=")
            json.dump(history_json, f)

        html_path = os.path.join(image_folder_path, "history.html")
        if not os.path.exists(html_path):
            copyfile("./static/assets/history_template.html", html_path)
            template_path = os.path.abspath("./static/assets/")
            css_path = os.path.join(template_path, "history.css")
            js_path = os.path.join(template_path, "history.js")
            with open(html_path, "r+") as file:
                content = file.read()
                content = content.replace("{css_path}", css_path)
                content = content.replace("{js_path}", js_path)
                file.seek(0)
                file.write(content)
                file.truncate()
