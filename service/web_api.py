from datetime import datetime
import threading
from typing import Dict, Any, List, Tuple

from flask import Flask, jsonify, request, Response, stream_with_context, Blueprint
from werkzeug.exceptions import BadRequest, InternalServerError
from pathlib import Path
import logging

from llm_adapter import LLM_SSE_Adapter
from sd_adapter import SD_SSE_Adapter
import model_download_adapter
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

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprints for organizing routes
llm_bp = Blueprint("llm", __name__, url_prefix="/api/llm")
sd_bp = Blueprint("sd", __name__, url_prefix="/api/sd")
model_bp = Blueprint("model", __name__, url_prefix="/api")

# --- Constants ---
SUCCESS_CODE = 0
FAILURE_CODE = -1
SUCCESS_MESSAGE = "success"
FAILED_MESSAGE = "failed"
DEFAULT_MODE = 0
DEFAULT_IMAGE_SIZE = 512
DEFAULT_SCALE = 1.5
DEFAULT_DENOISE = 0.5
DEFAULT_GENERATE_NUMBER = 1
DEFAULT_INFERENCE_STEPS = 12
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_SEED = -1
DEFAULT_SCHEDULER = "None"
DEFAULT_LORA = "None"
DEFAULT_IMAGE_PREVIEW = 0
DEFAULT_SAFE_CHECK = 1

# --- Helper Functions ---

def validate_json_input(required_fields: List[str]):
    """
    Decorator to validate JSON input for API endpoints.
    Raises a BadRequest exception if the request is not JSON or if any required field is missing.

    Args:
        required_fields (List[str]): A list of required fields in the JSON payload.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                raise BadRequest("Invalid content type. JSON expected.")
            data = request.get_json()
            for field in required_fields:
                if field not in data:
                    raise BadRequest(f"Missing required field: {field}")
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Global error handler for the application. 
    Catches all exceptions, logs them, and returns a JSON error response. 
    For BadRequest exceptions, it returns a 400 status code. 
    For all other exceptions, it returns a 500 status code.
    """
    if isinstance(e, BadRequest):
        return jsonify({"error": str(e)}), 400
    logger.exception("An error occurred:")
    return jsonify({"error": "An internal error occurred."}), 500

# --- LLM Routes ---

@llm_bp.route("/chat", methods=["POST"])
@validate_json_input(["prompt"])
def llm_chat():
    """
    Handles LLM chat requests and streams responses.

    Expects a JSON payload with:
    - prompt: The chat prompt.

    Returns a streaming response with the chat output.
    """
    try:
        paint_biz.dispose_basic_model()  # Dispose of the basic model if in use
        params = request.get_json()
        llm_params = llm_biz.LLMParams(**params)
        sse_invoker = LLM_SSE_Adapter()
        it = sse_invoker.text_conversation(llm_params)
        return Response(stream_with_context(it), content_type="text/event-stream")
    except Exception as e:
        logger.exception("Error in LLM chat route.")
        return jsonify({"error": "An error occurred during LLM chat."}), 500

@llm_bp.route("/stopGenerate", methods=["GET"])
def stop_llm_generate():
    """Stops the ongoing LLM generation process. Returns a JSON success response."""
    llm_biz.stop_generate()
    return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})

# --- Stable Diffusion Routes ---

@sd_bp.route("/generate", methods=["POST"])
def sd_generate():
    """
    Handles image generation requests using Stable Diffusion.

    Expects form data with the following parameters:
    - mode: The generation mode (0-4).
        - 0: Text-to-Image
        - 1: Upscale Image
        - 2: Image-to-Image
        - 3: Inpainting
        - 4: Outpainting
    - device: The index of the device to use (defaults to 0).
    - prompt: The text prompt for image generation.
    - model_repo_id: The repository ID of the model to use.
    - width: The width of the generated image (defaults to 512).
    - negative_prompt: The negative prompt for image generation.
    - height: The height of the generated image (defaults to 512).
    - generate_number: The number of images to generate (defaults to 1).
    - inference_steps: The number of inference steps (defaults to 12).
    - guidance_scale: The guidance scale (defaults to 7.5).
    - seed: The random seed (defaults to -1).
    - scheduler: The scheduler to use (defaults to "None").
    - lora: The LORA to use (defaults to "None").
    - image_preview: Whether to enable image preview (defaults to 0).
    - safe_check: Whether to enable safety checks (defaults to 1).
    - image: The input image for modes 1, 2, 3, and 4.
    - denoise: The denoising strength for modes 1, 2, 3, and 4 (defaults to 0.5).
    - scale: The scaling factor for Upscale Image mode (mode 1, defaults to 1.5).
    - mask_image: The mask image for Inpainting mode (mode 3).
    - direction: The direction for Outpainting mode (mode 4, defaults to "right").

    Returns a streaming response with image generation progress.
    """
    try:
        llm_biz.dispose()  # Dispose of any existing LLM resources
        mode = int(request.form.get("mode", DEFAULT_MODE))
        if mode not in range(5):
            raise BadRequest("Invalid mode specified.")

        # Initialize parameters based on the selected mode
        if mode == 1:  # Upscale Image mode
            params = UpscaleImageParams(
                scale=float(request.form.get("scale", DEFAULT_SCALE))
            )
        elif mode == 2:  # Image-to-Image mode
            params = ImageToImageParams()
        elif mode == 3:  # Inpainting mode
            params = InpaintParams(
                mask_image=cache_mask_image()
            )
        elif mode == 4:  # Outpainting mode
            params = OutpaintParams(
                direction=request.form.get("direction", "right")
            )
        else:
            params = TextImageParams()

        # Set common parameters for all modes
        params.device = int(request.form.get("device", 0))
        params.prompt = request.form.get("prompt", "")
        params.model_name = request.form["model_repo_id"]
        params.mode = mode
        params.width = int(request.form.get("width", DEFAULT_IMAGE_SIZE))
        params.negative_prompt = request.form.get("negative_prompt", "")
        params.height = int(request.form.get("height", DEFAULT_IMAGE_SIZE))
        params.generate_number = int(request.form.get("generate_number", DEFAULT_GENERATE_NUMBER))
        params.inference_steps = int(request.form.get("inference_steps", DEFAULT_INFERENCE_STEPS))
        params.guidance_scale = float(request.form.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        params.seed = int(request.form.get("seed", DEFAULT_SEED))
        params.scheduler = request.form.get("scheduler", DEFAULT_SCHEDULER)
        params.lora = request.form.get("lora", DEFAULT_LORA)
        params.image_preview = int(request.form.get("image_preview", DEFAULT_IMAGE_PREVIEW))
        params.safe_check = int(request.form.get("safe_check", DEFAULT_SAFE_CHECK))

        if mode != 0:  # Modes that require input image
            params.image = cache_input_image()
            params.denoise = float(request.form.get("denoise", DEFAULT_DENOISE))

        sse_invoker = SD_SSE_Adapter(request.url_root)
        it = sse_invoker.generate(params)
        return Response(stream_with_context(it), content_type="text/event-stream")
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Error in SD generate route.")
        return jsonify({"error": "An error occurred during image generation."}), 500

@sd_bp.route("/stopGenerate", methods=["GET"])
def stop_sd_generate():
    """Stops the ongoing SD generation process. Returns a JSON success response."""
    paint_biz.stop_generate()
    return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})

# --- Model Management Routes ---

@model_bp.route("/init", methods=["POST"])
def get_init_settings():
    """
    Initializes settings based on POSTed configuration.

    Expects a JSON payload with configurations.
    Returns a JSON response with available schedulers.
    """
    try:
        post_config: dict = request.get_json()
        for k, v in post_config.items():
            if k in model_config.config:
                model_config.config[k] = v
        return jsonify(schedulers_util.schedulers)
    except Exception as e:
        logger.exception("Error in init route.")
        return jsonify({"error": "An error occurred during initialization."}), 500

@model_bp.route("/getGraphics", methods=["POST"])
def get_graphics():
    """
    Retrieves supported graphics configurations based on the environment. 
    Takes a POST request with a form parameter 'env' (defaults to "ultra").
    Returns a JSON response with the supported graphics configurations.
    """
    env = request.form.get("env", default="ultra", type=str)
    return jsonify(utils.get_support_graphics(env))

@model_bp.route("/applicationExit", methods=["GET"])
def application_exit(): # Renamed for consistency
    """
    Terminates the application by sending a SIGINT signal.

    Warning: This endpoint should be protected in a production environment 
    to prevent unauthorized termination.
    """
    from signal import SIGINT
    pid = os.getpid() 
    os.kill(pid, SIGINT)

@model_bp.route("/checkModelExist", methods=["POST"])
@validate_json_input(["models"])
def check_model_exist():
    """
    Checks if models exist locally.

    Expects a JSON payload with a list of models to check:
    - models: A list of dictionaries, each with 'repo_id' and 'type'.

    Returns a JSON response with the existence status of each model.
    """
    try:
        model_list: List[Dict[str, Any]] = request.get_json()["models"]
        result_list = []

        for item in model_list:
            repo_id = item["repo_id"]
            model_type = item["type"]
            exist = utils.check_model_exist(model_type, repo_id)
            result_list.append({"repo_id": repo_id, "type": model_type, "exist": exist})

        return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE, "exists": result_list})
    except Exception as e:
        logger.exception("Error in checkModelExist route.")
        return jsonify({"error": "An error occurred during model existence check."}), 500

# --- Model Size Cache ---

# This could be improved with a more robust caching mechanism (e.g., using cachetools)
size_cache = dict()
lock = threading.Lock()

@model_bp.route("/getModelSize", methods=["POST"])
@validate_json_input(["models"])
def get_model_size():
    """
    Retrieves the sizes of specified models. 

    Expects a JSON payload with:
    - models: A list of dictionaries, each with 'repo_id' and 'type'.

    Returns a JSON response with the size of each requested model.
    """
    try:
        model_list: List[Dict[str, Any]] = request.get_json()["models"]
        result_dict = {}
        request_list = []

        for item in model_list:
            repo_id = item["repo_id"]
            model_type = item["type"]
            key = f"{repo_id}_{model_type}"
            total_size = size_cache.get(key)
            if total_size is None:
                request_list.append((repo_id, model_type))
            else:
                result_dict[key] = bytes2human(total_size, "%(value).2f%(symbol)s")

        if request_list:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(fill_size_execute, repo_id, model_type, result_dict)
                    for repo_id, model_type in request_list
                ]
                concurrent.futures.wait(futures)
                executor.shutdown()

        return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE, "sizeList": result_dict})
    except Exception as e:
        logger.exception("Error in getModelSize route.")
        return jsonify({"error": "An error occurred during model size retrieval."}), 500

def fill_size_execute(repo_id: str, model_type: int, result_dict: dict):
    """
    Retrieves the size of a model and updates the cache. 
    This function is designed to be run in a separate thread.
    """
    key = f"{repo_id}_{model_type}"
    if model_type == 4:
        total_size = utils.get_ESRGAN_size()
    else:
        total_size = HFPlaygroundDownloader().get_model_total_size(repo_id, model_type)

    with lock:
        size_cache[key] = total_size
        result_dict[key] = bytes2human(total_size, "%(value).2f%(symbol)s")

# --- RAG Routes ---

@llm_bp.route("/enableRag", methods=["POST"])
def enable_rag():
    """
    Enables RAG (Retrieval-Augmented Generation) for LLMs.

    Expects form data with:
    - repo_id: The repository ID for the RAG model.
    - device: The device index to use.

    Returns a JSON success response.
    """
    if not rag.Is_Inited:
        repo_id = request.form.get("repo_id", default="", type=str)
        device = request.form.get("device", default=0, type=int)
        rag.init(repo_id, device)
    return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})

@llm_bp.route("/disableRag", methods=["GET"])
def disable_rag():
    """Disables RAG for LLMs. Returns a JSON success response."""
    if rag.Is_Inited:
        rag.dispose()
    return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})

@llm_bp.route("/getRagFiles", methods=["GET"])
def get_rag_files():
    """
    Retrieves a list of RAG files with their filenames and MD5 checksums. 
    Returns a JSON response with the file data.
    """
    try:
        result_list = list()
        index_list = rag.get_index_list()
        if index_list is not None:
            for index in index_list:
                result_list.append({"filename": index.get("name"), "md5": index.get("md5")})
        return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE, "data": result_list})
    except Exception as e:
        logger.exception("Error in getRagFiles route.")
        return jsonify({"error": "An error occurred while retrieving RAG files."}), 500

@llm_bp.route("/uploadRagFile", methods=["POST"])
def upload_rag_file():
    """
    Handles RAG file uploads.

    Expects form data with:
    - path: The path to the RAG file.

    Returns a JSON response indicating success and the MD5 checksum of the uploaded file.
    """
    try:
        path = request.form.get("path")
        code, md5 = rag.add_index_file(path)
        return jsonify({"code": code, "message": SUCCESS_MESSAGE, "md5": md5})
    except Exception as e:
        logger.exception("Error in uploadRagFile route.")
        return jsonify({"error": "An error occurred during RAG file upload."}), 500

@llm_bp.route("/deleteRagIndex", methods=["POST"])
def delete_rag_file():
    """
    Deletes a RAG file from the index.

    Expects form data with:
    - md5: The MD5 checksum of the file to delete.

    Returns a JSON success response.
    """
    try:
        md5_checksum = request.form.get("md5")
        rag.delete_index(md5_checksum)
        return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})
    except Exception as e:
        logger.exception("Error in deleteRagIndex route.")
        return jsonify({"error": "An error occurred during RAG file deletion."}), 500

# --- Model Downloading ---

@model_bp.route("/downloadModel", methods=["POST"])
@validate_json_input(["models"])
def download_model():
    """
    Handles model download requests and streams progress updates.

    Expects a JSON payload with:
    - models: A list of dictionaries, each containing the model information.

    Returns a streaming response with download progress.
    """
    try:
        model_list: List[Dict[str, Any]] = request.get_json()["models"]
        # Ensure thread safety when stopping downloads
        if model_download_adapter._adapter is not None:
            with lock:
                model_download_adapter._adapter.stop_download()
        model_download_adapter._adapter = model_download_adapter.Model_Downloader_Adapter()
        iterator = model_download_adapter._adapter.download(model_list)
        return Response(stream_with_context(iterator), content_type="text/event-stream")
    except Exception as e:
        logger.exception("Error in downloadModel route.")
        return jsonify({"error": "An error occurred during model download."}), 500

@model_bp.route("/stopDownloadModel", methods=["GET"])
def stop_download_model():
    """Stops an ongoing model download. Returns a JSON success response."""
    # Ensure thread safety when stopping downloads
    if model_download_adapter._adapter is not None:
        with lock:
            model_download_adapter._adapter.stop_download()
    return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})

# --- Image Caching ---

def cache_input_image():
    """
    Caches the uploaded input image and returns the file path.
    Handles file extensions based on content type and creates a timestamped filename.
    Raises a BadRequest exception if no image file is provided.
    """
    file = request.files.get("image")
    if not file:
        raise BadRequest("No image file provided.")

    file_extensions = {
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/png": ".png"
    }
    ext = file_extensions.get(file.content_type, ".png")

    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    # Use pathlib for file path construction
    save_dir = Path("./static/sd_input") / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{base_name}{ext}" 

    file.save(file_path)
    utils.cache_file(file_path, file.__sizeof__())
    file.stream.close()
    return str(file_path) # Return as string for compatibility

def cache_mask_image():
    """
    Caches the uploaded mask image and returns the file path.
    Creates a timestamped filename and saves the mask as a PNG image.
    """
    mask_width = int(request.form.get("mask_width", DEFAULT_IMAGE_SIZE))
    mask_height = int(request.form.get("mask_height", DEFAULT_IMAGE_SIZE))
    mask_image = utils.generate_mask_image(
        request.files.get("mask_image").stream.read(), mask_width, mask_height
    )

    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    # Use pathlib for file path construction
    save_dir = Path("./static/sd_mask") / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{base_name}.png"

    mask_image.save(file_path)
    utils.cache_file(file_path, file_path.stat().st_size) # Get file size using pathlib
    return str(file_path) # Return as string for compatibility

# Register blueprints with the Flask application
app.register_blueprint(llm_bp)
app.register_blueprint(sd_bp)
app.register_blueprint(model_bp)

# Main entry point for running the Flask application
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Playground Web service")
    parser.add_argument("--port", type=int, default=59999, help="Service listen port")
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port)