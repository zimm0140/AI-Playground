from datetime import datetime
import os
import threading
from flask import Flask, jsonify, request, Response, stream_with_context
from llm_adapter import LLM_SSE_Adapter
from sd_adapter import SD_SSE_Adapter
import model_download_adpater
from paint_biz import (
    TextImageParams,
    ImageToImageParams,
    InpaintParams,
    OutpaintParams,
    UpscaleImageParams,
)
import paint_biz
import llm_biz
import utils
import rag
import model_config
from model_downloader import HFPlaygroundDownloader
from psutil._common import bytes2human
import traceback

# Initialize the Flask application
app = Flask(__name__)

# Route for LLM (Language Model) chat API
@app.route("/api/llm/chat", methods=["POST"])
def llm_chat():
    """
    Handles LLM chat requests and streams responses.
    
    Expects a JSON payload with parameters for the LLM chat.
    Returns a streaming response with the chat output.
    """
    paint_biz.dispose_basic_model()  # Dispose of the basic model if in use
    params = request.get_json()  # Parse JSON parameters from the request
    llm_params = llm_biz.LLMParams(**params)  # Initialize LLM parameters
    sse_invoker = LLM_SSE_Adapter()  # Create an SSE adapter for LLM
    it = sse_invoker.text_conversation(llm_params)  # Start the conversation
    return Response(stream_with_context(it), content_type="text/event-stream")  # Stream response as SSE

# Route to stop LLM generation
@app.route("/api/llm/stopGenerate", methods=["GET"])
def stop_llm_generate():
    """Stops the ongoing LLM generation process."""
    import llm_biz  # This import could be moved to the top of the file
    llm_biz.stop_generate()  # Stop the LLM generation
    return jsonify({"code": 0, "message": "success"})  # Return success message

# Route for Stable Diffusion image generation API
@app.route("/api/sd/generate", methods=["POST"])
def sd_generate():
    """
    Handles image generation requests using Stable Diffusion.
    Supports various modes:
    - mode 0: Text-to-Image
    - mode 1: Upscale Image
    - mode 2: Image-to-Image
    - mode 3: Inpainting
    - mode 4: Outpainting

    Expects form data with parameters for image generation.
    Returns a streaming response with image generation progress.
    """
    llm_biz.dispose()  # Dispose LLM to free resources
    mode = request.form.get("mode", default=0, type=int)  # Get the mode from request

    # Initialize parameters based on the selected mode
    if mode != 0:  # Handle modes that require an input image
        if mode == 1:  # Upscale Image mode
            params = UpscaleImageParams()
            params.scale = request.form.get("scale", 1.5, type=float)
        elif mode == 2:  # Image-to-Image mode
            params = ImageToImageParams()
        elif mode == 3:  # Inpainting mode
            params = InpaintParams()
            params.mask_image = cache_mask_image()
        elif mode == 4:  # Outpainting mode
            params = OutpaintParams()
            params.direction = request.form.get("direction", "right", type=str)

        # Cache the input image and get denoise parameter
        params.image = cache_input_image()
        params.denoise = request.form.get("denoise", 0.5, type=float)
    else:
        params = TextImageParams()  # Text-to-Image mode

    # Set common parameters for all modes
    base_params = params  # Consider refactoring this to avoid assigning to a new variable
    base_params.device = request.form.get("device", default=0, type=int)
    base_params.prompt = request.form.get("prompt", default="", type=str)
    base_params.model_name = request.form["model_repo_id"]
    base_params.mode = mode
    base_params.width = request.form.get("width", default=512, type=int)
    base_params.negative_prompt = request.form.get("negative_prompt", default="", type=str)
    base_params.height = request.form.get("height", default=512, type=int)
    base_params.generate_number = request.form.get("generate_number", default=1, type=int)
    base_params.inference_steps = request.form.get("inference_steps", default=12, type=int)
    base_params.guidance_scale = request.form.get("guidance_scale", default=7.5, type=float)
    base_params.seed = request.form.get("seed", default=-1, type=int)
    base_params.scheduler = request.form.get("scheduler", default="None", type=str)
    base_params.lora = request.form.get("lora", default="None", type=str)
    base_params.image_preview = request.form.get("image_preview", default=0, type=int)
    base_params.safe_check = request.form.get("safe_check", default=1, type=int)

    # Initialize the SSE adapter for Stable Diffusion
    sse_invoker = SD_SSE_Adapter(request.url_root)
    # Start the image generation process
    it = sse_invoker.generate(params)
    # Return a streaming response with the generated image(s)
    return Response(stream_with_context(it), content_type="text/event-stream")

# Route to stop Stable Diffusion image generation
@app.route("/api/sd/stopGenerate", methods=["GET"])
def stop_sd_generate():
    """Stops the ongoing SD generation process."""
    import paint_biz  # This import could be moved to the top of the file
    paint_biz.stop_generate()  # Stop the SD generation
    return jsonify({"code": 0, "message": "success"})  # Return success message

# Route to initialize application settings
@app.route("/api/init", methods=["POST"])
def get_init_settings():
    """
    Initializes settings based on POSTed configuration.

    Expects a JSON payload with configurations.
    Returns a JSON response with available schedulers.
    """
    import schedulers_util  # This import could be moved to the top of the file

    # Get the configuration settings from the request body
    post_config: dict = request.get_json()

    # Update the global configuration with the received settings
    for k, v in post_config.items():
        if k in model_config.config:  # More Pythonic way to check for key existence
            model_config.config[k] = v  # Update configuration using dictionary syntax

    # Return the available schedulers as a JSON response
    return jsonify(schedulers_util.schedulers)

# Route to get supported graphics configurations
@app.route("/api/getGraphics", methods=["POST"])
def get_graphics():
    """Retrieves supported graphics configurations based on the environment."""
    env = request.form.get("env", default="ultra", type=str)  # Get environment type
    return jsonify(utils.get_support_graphics(env))  # Return supported graphics

# Route to gracefully exit the application (Caution: This may need additional security)
@app.route("/api/applicationExit", methods=["GET"])
def applicationExit():
    """
    Terminates the application by sending a SIGINT signal.
    
    Warning: This endpoint should be protected in a production environment to prevent unauthorized termination.
    """
    from signal import SIGINT
    pid = os.getpid()  # Get the current process ID
    os.kill(pid, SIGINT)  # Send SIGINT to terminate the process

# Route to check if specified models exist on the server
@app.route("/api/checkModelExist", methods=["POST"])
def check_model_exist():
    """
    Checks if models exist locally.

    Expects a JSON payload with a list of models to check.
    Returns a JSON response with the existence status of each model.
    """
    model_list = request.get_json()  # Get the list of models from the request body
    result_list = []

    # Iterate over each model in the list and check if it exists
    for item in model_list:
        repo_id = item["repo_id"]
        model_type = item["type"]
        exist = utils.check_model_exist(model_type, repo_id)  # Corrected the typo here
        result_list.append({"repo_id": repo_id, "type": model_type, "exist": exist})

    # Return the existence check results as a JSON response
    return jsonify({"code": 0, "message": "success", "exists": result_list})

# Cache to store model sizes
size_cache = dict()
# Thread lock for thread-safe operations on the cache
lock = threading.Lock()

# Route to get the sizes of specified models
@app.route("/api/getModelSize", methods=["POST"])
def get_model_size():
    """
    Retrieves the sizes of specified models.

    Expects a JSON payload with a list of models.
    Returns a JSON response with the size of each requested model.
    """
    import concurrent.futures  # This import could be moved to the top of the file

    model_list = request.get_json()  # Get the list of models from the request body
    result_dict = dict()
    request_list = []

    # Check the size cache for each model
    for item in model_list:
        repo_id = item["repo_id"]
        model_type = item["type"]
        key = f"{repo_id}_{model_type}"
        total_size = size_cache.get(key)
        if total_size is None:
            # If the model size is not cached, add it to the request list
            request_list.append((repo_id, model_type))
        else:
            # If the model size is cached, add it to the result dictionary
            result_dict[key] = bytes2human(total_size, "%(value).2f%(symbol)s")  # More Pythonic dictionary syntax

    # If there are models to check, use a thread pool to calculate their sizes concurrently
    if len(request_list) > 0:  # More Pythonic way to check list length
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fill_size_execute, repo_id, model_type, result_dict)
                for repo_id, model_type in request_list
            ]
            concurrent.futures.wait(futures)
            executor.shutdown()

    # Return the model sizes as a JSON response
    return jsonify({"code": 0, "message": "success", "sizeList": result_dict})

# Function to retrieve and cache the size of a specific model
def fill_size_execute(repo_id: str, model_type: int, result_dict: dict):
    """
    Retrieves and caches the size of a model.

    Args:
        repo_id (str): The repository ID of the model.
        model_type (int): The type of the model.
        result_dict (dict): The dictionary to store the result.
    """
    key = f"{repo_id}_{model_type}"

    # Get the model size based on the type
    if model_type == 4:
        total_size = utils.get_ESRGAN_size()
    else:
        total_size = HFPlaygroundDownloader().get_model_total_size(repo_id, model_type)

    # Update the size cache with the model size in a thread-safe manner
    with lock:
        size_cache[key] = total_size  # More Pythonic dictionary syntax
        result_dict[key] = bytes2human(total_size, "%(value).2f%(symbol)s")  # More Pythonic dictionary syntax

# Route to enable Retrieval-Augmented Generation (RAG) feature
@app.route("/api/llm/enableRag", methods=["POST"])
def enable_rag():
    """
    Enables RAG (Retrieval-Augmented Generation) for LLMs.

    Expects form data with 'repo_id' and 'device'.
    Returns a JSON response indicating success.
    """
    if not rag.Is_Inited:  # If RAG is not initialized
        repo_id = request.form.get("repo_id", default="", type=str)
        device = request.form.get("device", default=0, type=int)
        rag.init(repo_id, device)  # Initialize RAG with the given repo_id and device
    # Return a success message
    return jsonify({"code": 0, "message": "success"})

# Route to disable RAG feature
@app.route("/api/llm/disableRag", methods=["GET"])
def disable_rag():
    """Disables RAG for LLMs."""
    if rag.Is_Inited:
        rag.dispose()  # Dispose of the RAG resources
    return jsonify({"code": 0, "message": "success"})  # Return success message

# Route to download a model
@app.route("/api/downloadModel", methods=["POST"])
def download_model():
    """
    Handles model download requests.

    Expects a JSON payload with a list of models to download.
    Returns a streaming response with download progress.
    """
    model_list = request.get_json()  # Get the list of models to download from the request
    
    # If a download is already in progress, stop it
    if model_download_adpater._adapter is not None:
        model_download_adpater._adapter.stop_download()
    try:
        # Create a new Model_Downloader_Adapter to handle the download
        model_download_adpater._adapter = model_download_adpater.Model_Downloader_Adapter()
        iterator = model_download_adpater._adapter.download(model_list)  # Start downloading models
        # Return a streaming response to show download progress
        return Response(stream_with_context(iterator), content_type="text/event-stream")
    except Exception as e:
        # Handle exceptions during the download process
        traceback.print_exc()  # Print the traceback for debugging
        model_download_adpater._adapter.stop_download()  # Stop the download in case of error
        error_message = '{{"type": "error", "err_type": "{}"}}'.format(e)
        # Return an error message as a streaming response
        return Response(stream_with_context([error_message]), content_type="text/event-stream")

# Route to stop an ongoing model download
@app.route("/api/stopDownloadModel", methods=["GET"])
def stop_download_model():
    """Stops an ongoing model download."""
    # Stop the download if the adapter is active
    if model_download_adpater._adapter is not None:
        model_download_adpater._adapter.stop_download()
    return jsonify({"code": 0, "message": "success"})  # Return success message

# Route to get the list of RAG files
@app.route("/api/llm/getRagFiles", methods=["GET"])
def get_rag_files():
    """Retrieves a list of RAG files with their filenames and MD5 checksums."""
    try:
        result_list = list()  # Initialize an empty list to store the results
        index_list = rag.get_index_list()  # Get the list of RAG indices
        # Check if index_list is not None (avoid potential errors)
        if index_list is not None:  
            # Iterate through each index in the list
            for index in index_list:
                # Append a dictionary containing the filename and its MD5 checksum to the result list
                result_list.append({"filename": index.get("name"), "md5": index.get("md5")}) 

        return jsonify({"code": 0, "message": "success", "data": result_list})  # Return success with the file list
    except Exception as e:
        traceback.print_exc()  # Print the traceback in case of an exception
        return jsonify({"code": -1, "message": "failed"})  # Return a failure message

# Route to upload a RAG file
@app.route("/api/llm/uploadRagFile", methods=["POST"])
def upload_rag_file():
    """Handles RAG file uploads and returns the MD5 checksum of the uploaded file."""
    try:
        path = request.form.get("path")  # Retrieve the file path from the request form
        code, md5 = rag.add_index_file(path)  # Add the file to the RAG index and get its MD5 checksum
        return jsonify({"code": code, "message": "success", "md5": md5})  # Return a success message with the MD5 checksum
    except Exception as e:
        traceback.print_exc()  # Print the traceback if an exception occurs
        return jsonify({"code": -1, "message": "failed", "path": path})  # Return a failure message with the file path

# Route to delete a RAG file from the index
@app.route("/api/llm/deleteRagIndex", methods=["POST"])
def delete_rag_file():
    """Deletes a RAG file from the index based on its MD5 checksum."""
    try:
        md5_checksum = request.form.get("md5")  # Get the MD5 checksum of the file to delete
        rag.delete_index(md5_checksum)  # Delete the file from the RAG index
        return jsonify({"code": 0, "message": "success"})  # Return a success message
    except Exception as e:
        traceback.print_exc()  # Print the traceback for debugging purposes
        return jsonify({"code": -1, "message": "failed"})  # Return a failure message

def cache_input_image():
    """
    Caches the uploaded input image for Stable Diffusion and returns the file path.
    
    Handles various image file types (JPEG, GIF, BMP, PNG) and generates a 
    timestamped file name for storage.
    """
    file = request.files.get("image")  # Get the uploaded image file from the request
    ext = ".png"  # Set the default file extension to PNG
    # Determine the correct file extension based on the content type
    if file.content_type == "image/jpeg":
        ext = ".jpg"
    elif file.content_type == "image/gif":
        ext = ".gif"
    elif file.content_type == "image/bmp":
        ext = ".bmp"

    # Generate a timestamped filename
    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    file_path = os.path.abspath(os.path.join("./static/sd_input/", folder, base_name + ext))

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Save the file to the generated path
    file.save(file_path)
    # Cache the file with its size
    utils.cache_file(file_path, file.__sizeof__())  
    file.stream.close()  # Close the file stream
    return file_path  # Return the file path

def cache_mask_image():
    """Caches the uploaded mask image for inpainting and returns the file path."""
    # Get mask dimensions from request
    mask_width = request.form.get("mask_width", default=512, type=int)
    mask_height = request.form.get("mask_height", default=512, type=int)
    # Generate a PIL Image from the uploaded mask image data
    mask_image = utils.generate_mask_image(
        request.files.get("mask_image").stream.read(), mask_width, mask_height
    )

    # Generate a timestamped filename for the mask image
    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    file_path = os.path.abspath(os.path.join("static/sd_mask/", folder, base_name + ".png"))

    # Create the directory for the mask image if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Save the mask image to the generated file path
    mask_image.save(file_path)
    # Cache the file path and size
    utils.cache_file(file_path, os.path.getsize(file_path)) 
    return file_path  # Return the file path

# Main entry point for running the Flask application
if __name__ == "__main__":
    import argparse
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="AI Playground Web service")
    parser.add_argument('--port', type=int, default=59999, help='Service listen port')
    args = parser.parse_args()
    # Run the Flask app on the specified host and port
    app.run(host="127.0.0.1", port=args.port)