# --- Standard Library Imports ---
from datetime import datetime
import json
import os
import threading
from queue import Empty, Queue
import traceback
from typing import Any

# --- Third-Party Imports ---
from PIL import Image
from psutil._common import bytes2human

# --- Local Application Imports ---
import paint_biz
from model_downloader import NotEnoughDiskSpaceException, DownloadException
import utils

class SD_SSE_Adapter:
    """
    Adapter class for handling Server-Sent Events (SSE) for Stable Diffusion image generation.
    This class manages communication between the backend image generation process and 
    the frontend client, providing real-time updates on progress, errors, and the final image output.
    """
    msg_queue: Queue  # Queue to store messages for the client
    finish: bool  # Flag indicating if the generation process is finished
    singal: threading.Event  # Threading event for signaling message availability
    url_root: str  # Root URL for constructing image paths

    def __init__(self, url_root: str):
        """
        Initializes the SD_SSE_Adapter with the root URL for serving images.

        Args:
            url_root (str): The base URL for accessing generated images.
        """
        self.msg_queue = Queue(-1) # Initialize a queue with unlimited size
        self.finish = False  # Set the finish flag to False initially
        self.singal = threading.Event() # Initialize the threading event
        self.url_root = url_root # Store the root URL

    def put_msg(self, data):
        """
        Puts a message into the message queue and signals the waiting thread.

        Args:
            data (Any): The data to be sent as a message.
        """
        self.msg_queue.put_nowait(data) # Add data to the queue without blocking
        self.singal.set()  # Signal the event to indicate a new message

    def download_model_progress_callback(
        self, repo_id: str, download_size: int, total_size: int, speed: int
    ):
        """
        Callback function to handle model download progress updates.

        Args:
            repo_id (str): The ID of the repository from which the model is downloaded.
            download_size (int): The size of the downloaded data in bytes.
            total_size (int): The total size of the model in bytes.
            speed (int): The download speed in bytes per second.
        """
        data = {
            "type": "download_model_progress",
            "repo_id": repo_id,
            "download_size": bytes2human(download_size),
            "total_size": bytes2human(total_size),
            "percent": round(download_size / total_size * 100, 2),
            "speed": "{}/s".format(bytes2human(speed)),
        }
        self.put_msg(data)  # Put the progress data into the message queue

    def download_model_completed_callback(self, repo_id: str, ex: Exception):
        """
        Callback function to handle model download completion or failure.

        Args:
            repo_id (str): The ID of the repository from which the model was downloaded.
            ex (Exception): The exception object if a download error occurred, otherwise None.
        """
        if ex is not None:
            # Send an error message if the download failed
            self.put_msg({"type": "error", "value": "DownloadModelFailed"}) 
        else:
            # Send a completion message if the download was successful
            self.put_msg({"type": "download_model_completed", "repo_id": repo_id})

    def load_model_callback(self, event: str):
        """
        Callback function to handle model loading events.

        Args:
            event (str): A string describing the model loading event.
        """
        data = {"type": "load_model", "event": event}
        self.put_msg(data) # Put the model loading event data into the message queue

    def load_model_components_callback(self, event: str):
        """
        Callback function to handle model components loading events.

        Args:
            event (str): A string describing the model components loading event.
        """
        data = {"type": "load_model_components","event": event}
        self.put_msg(data) # Put the model components loading event data into the message queue

    def step_end_callback(
        self,
        index: int,
        step: int,
        total_step: int,
        preview_enabled: bool,
        image: Image.Image | None,
    ):
        """
        Callback function to handle the end of a generation step.

        Args:
            index (int): Index of the image being generated.
            step (int): Current step number in the generation process.
            total_step (int): Total number of steps in the generation process.
            preview_enabled (bool): Boolean flag indicating if image previews are enabled.
            image (Image.Image | None): The generated preview image, if available.
        """
        if preview_enabled and image is not None:
            # Convert image to base64 if previews are enabled
            image = utils.image_to_base64(image) 
        elif not preview_enabled:
            # Otherwise, use a placeholder image URL
            image = f"{self.url_root}/static/assets/aipg.png" 

        data = {
            "type": "step_end",
            "index": index,
            "step": step,
            "total_step": total_step,
            "image": image, # Include image data or placeholder URL
        }
        self.put_msg(data) # Put the step-end data into the message queue

    def image_out_callback(
        self,
        index: int,
        image: Image.Image | None,
        params: paint_biz.TextImageParams = None,
        safe_check_pass: bool = True
    ):
        """
        Callback function to handle the output of a generated image.

        Args:
            index (int): Index of the generated image.
            image (Image.Image | None): The generated image.
            params (paint_biz.TextImageParams, optional): Parameters used for generation.
            safe_check_pass (bool): Indicates if the generated image passed safety checks.
        """
        now = datetime.now()
        folder = now.strftime("%d_%m_%Y")
        base_name = now.strftime("%H%M%S")
        filename = "static/sd_out/{}/{}.png".format(folder, base_name)
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.makedirs(dir) # Create the output directory if it doesn't exist
        image.save(filename) # Save the generated image to the specified path
        utils.cache_file(filename, os.path.getsize(filename)) # Cache the generated image

        response_params = self.get_response_params(image, os.path.getsize(filename), params)
        try:
            # Attempt to log the generation parameters to a file
            self.log_to_file(params, folder, base_name)
        except Exception:
            traceback.print_exc() # Print any exceptions that occur during logging

        image_url = f"{self.url_root}/{filename}"
        data = {
            "type": "image_out",
            "index": index,
            "image": image_url, # Include the URL of the generated image
            "params": response_params,
            "safe_check_pass":safe_check_pass,
        }
        self.put_msg(data) # Put the image output data into the message queue

    def error_callback(self, ex: Exception):
        """
        Callback function to handle errors during image generation or model downloading.

        Args:
            ex (Exception): The exception object representing the error.
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
        elif isinstance(ex, NotEnoughDiskSpaceException):
            self.put_msg(
                {
                    "type": "error",
                    "err_type": "not_enough_disk_space",
                    "need": bytes2human(ex.requires_space),
                    "free": bytes2human(ex.free_space),
                }
            )
        elif isinstance(ex, DownloadException):
            self.put_msg({"type": "error", "err_type": "download_exception"})
        elif isinstance(ex, paint_biz.StopGenerateException):
            pass  # No specific action for StopGenerateException (handled elsewhere)
        elif isinstance(ex, RuntimeError):
            self.put_msg({"type": "error", "err_type": "runtime_error"})
        else:
            self.put_msg({"type": "error", "err_type": "unknow_exception"})
        print(f"exception:{str(ex)}")

    def generate(self, params: paint_biz.TextImageParams):
        """
        Initiates the image generation process in a separate thread.

        Args:
            params (paint_biz.TextImageParams): Parameters for image generation.
        
        Returns:
            Generator: A generator that yields messages to be sent to the client.
        """
        thread = threading.Thread(target=self.generate_run, args=[params])
        thread.start() # Start the image generation in a new thread
        return self.generator()  # Return a generator to yield messages to the client

    def generate_run(
        self,
        params: paint_biz.TextImageParams
        | paint_biz.ImageToImageParams
        | paint_biz.UpscaleImageParams
        | paint_biz.InpaintParams
        | paint_biz.OutpaintParams,
    ):
        """
        The main function for running the image generation process in a separate thread.

        Args:
            params: Parameters for image generation (can be one of the defined parameter types).
        """
        try:
            paint_biz.load_model_callback = self.load_model_callback
            paint_biz.load_model_components_callback = self.load_model_components_callback
            paint_biz.step_end_callback = self.step_end_callback
            paint_biz.image_out_callback = self.image_out_callback
            paint_biz.download_progress_callback = self.download_model_progress_callback
            paint_biz.download_completed_callback = (
                self.download_model_completed_callback
            )
            paint_biz.generate(params=params)  # Start the image generation process
        except Exception as ex:
            traceback.print_exc()
            self.error_callback(ex) # Handle any exceptions using the error callback
        finally:
            self.finish = True  # Set the finish flag to True when generation is complete
            self.singal.set() # Signal the threading event

    def generator(self):
        """
        A generator function that yields messages from the queue to the client. 

        Yields:
            str: A Server-Sent Event (SSE) formatted message.
        """
        while True:
            while not self.msg_queue.empty():
                try:
                    # Get a message from the queue
                    data = self.msg_queue.get_nowait() 
                    # Format the message for SSE
                    msg = f"data:{json.dumps(data)}\0" 
                    yield msg # Yield the message to be sent to the client
                except Empty:
                    break # Break the inner loop if the queue is empty
            if not self.finish:
                self.singal.clear() # Clear the signal if generation is not finished
                self.singal.wait()  # Wait for the signal indicating a new message
            else:
                break # Break the outer loop if the generation process is finished

    def get_response_params(
        self, image: Image.Image, size: int, params: paint_biz.TextImageParams
    ):
        """
        Generates a dictionary of parameters for the image generation response.

        Args:
            image (Image.Image): The generated image.
            size (int): The size of the generated image in bytes.
            params (paint_biz.TextImageParams): Parameters used for image generation.

        Returns:
            dict: A dictionary of response parameters.
        """
        response_params = {
            "width": image.width,  # Image width
            "height": image.height, # Image height
            "size": bytes2human(size), # Human-readable image size
        }

        for key, value in params.__dict__.items():
            # Exclude specific parameters from the response
            if key in [
                "generate_number",
                "image_preview",
                "width",
                "height"
            ] or isinstance(value, Image.Image):
                continue
            response_params.__setitem__(key, value) # Add other parameters to the response

        return response_params

    def log_to_file(
        self, params: Any, folder: str, base_name: str
    ):
        """
        Logs the image generation parameters and the path to the output image to a history file.

        Args:
            params (Any): The parameters used for image generation.
            folder (str): The folder where the history file is stored.
            base_name (str): The base filename for the output image.
        """
        from shutil import copyfile

        json_path = f"./static/sd_out/{folder}/history.json"
        base_output = os.path.abspath("./static/")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r+") as f:
                    f.seek(12) # Seek to the beginning of the JSON data
                    history_json = json.load(f) # Load the existing history
            except Exception:
                os.remove(json_path) # Remove the history file if it's corrupted
                history_json = [] # Start a new history list
        else:
            history_json = []  # Start a new history list

        param_list = []
        for k, v in params.__dict__.items():
            if k == "generate_number" or k == "image_preview":
                continue # Skip these parameters
            elif k == "image" or k == "mask_image":
                # For image parameters, store the relative path
                save_path = os.path.abspath(str(v))
                save_path = save_path.replace(base_output, "../../").replace('\\', '/')
                param_list.append(
                    {
                        "name": k,
                        "type": "image",
                        "value": save_path, 
                    }
                )
            else:
                # For other parameters, store the value directly
                param_list.append(
                    {
                        "name": k,
                        "value": v,
                        "type": "normal",
                    }
                )

        history_item = {
            "out_image": f"./{base_name}.png", # Path to the output image
            "params": param_list, # Parameters used for generation
        }

        history_json.insert(0, history_item)  # Insert the new item at the beginning

        with open(json_path, "w") as f:
            # Write the history data to the file
            f.write("let hisotry=") 
            json.dump(history_json, f)
        
        html_path = f"./static/sd_out/{folder}/history.html"
        if not os.path.exists(html_path):
            # Copy the history template if it doesn't exist
            copyfile("./static/assets/hisotory_template.html", html_path)