from datetime import datetime
import threading
from typing import Dict, Any, List, Tuple

from flask import Flask, jsonify, request, Response, stream_with_context, Blueprint
from werkzeug.exceptions import BadRequest, NotFound, UnsupportedMediaType
from pathlib import Path
import logging
from PIL import Image
import imghdr
import requests

from llm_adapter import LLM_SSE_Adapter
from sd_adapter import SD_SSE_Adapter
import model_download_adapter
from paint_biz import (
    TextImageParams,
    ImageToImageParams,
    InpaintParams,
    OutpaintParams,
    UpscaleImageParams,
    PaintBiz
)
import llm_biz
import utils
import rag
import model_config
from model_downloader import HFPlaygroundDownloader
from psutil._common import bytes2human
import schedulers_util
from cachetools import TTLCache
from functools import wraps

# --- Initialize Flask App ---

app = Flask(__name__)

# --- Configure Logging ---

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Blueprint Factory Functions --- 

def create_llm_blueprint(llm_biz_instance: llm_biz.LLMBiz, paint_biz_instance: paint_biz.PaintBiz) -> Blueprint: 
    """Creates and returns the LLM blueprint."""
    llm_bp = Blueprint("llm", __name__, url_prefix="/api/llm")

    @llm_bp.route("/chat", methods=["POST"])
    @validate_json_input(["prompt"])
    def llm_chat():
        """Handles LLM chat requests and streams responses."""
        try:
            paint_biz_instance.dispose_basic_model()
            params = request.get_json()
            llm_params = llm_biz_instance.LLMParams(**params)
            sse_invoker = LLM_SSE_Adapter()
            it = sse_invoker.text_conversation(llm_params)
            return Response(stream_with_context(it), content_type="text/event-stream")
        except Exception as e:
            logger.exception("Error in LLM chat route.")
            return jsonify({"error": "An error occurred during LLM chat."}), 500

    @llm_bp.route("/stopGenerate", methods=["GET"])
    def stop_llm_generate():
        """Stops the ongoing LLM generation process."""
        llm_biz_instance.stop_generate()
        return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE}) 

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

            # Basic validation for repo_id
            if not repo_id:
                raise BadRequest("Missing 'repo_id' in form data.")

            try:
                rag.init(repo_id, device)
            except Exception as e:  
                logger.exception("Error initializing RAG.")
                return jsonify({"code": FAILURE_CODE, "message": str(e)}), 500

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
            result_list: List[Dict[str, str]] = []
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
            file_path = request.form.get("path")

            # Validate the file path
            if not file_path:
                raise BadRequest("Missing 'path' in form data.")
            if not Path(file_path).exists():
                raise NotFound(f"File not found: {file_path}")

            code, md5 = rag.add_index_file(file_path)
            return jsonify({"code": code, "message": SUCCESS_MESSAGE, "md5": md5})
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except NotFound as e:
            return jsonify({"error": str(e)}), 404
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

            # Validate the MD5 checksum (You might want to add a more robust check here)
            if not md5_checksum:
                raise BadRequest("Missing 'md5' in form data.")

            rag.delete_index(md5_checksum)
            return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Error in deleteRagIndex route.")
            return jsonify({"error": "An error occurred during RAG file deletion."}), 500

    return llm_bp 

def create_sd_blueprint(paint_biz_instance: paint_biz.PaintBiz, llm_biz_instance: llm_biz.LLMBiz) -> Blueprint:
    """Creates and returns the Stable Diffusion blueprint."""
    sd_bp = Blueprint("sd", __name__, url_prefix="/api/sd")

    @sd_bp.route("/generate", methods=["POST"])
    def sd_generate():
        """Handles image generation requests using Stable Diffusion."""
        try:
            llm_biz_instance.dispose()  
            mode = int(request.form.get("mode", DEFAULT_MODE))
            if mode not in range(5):
                raise BadRequest("Invalid mode specified.")

            # Dictionary mapping modes to parameter classes
            mode_params = {
                1: UpscaleImageParams,
                2: ImageToImageParams,
                3: InpaintParams,
                4: OutpaintParams,
                0: TextImageParams  # Default to TextImageParams
            }

            # Get the appropriate parameter class based on the mode
            params_class = mode_params.get(mode)
            if params_class is None:
                raise BadRequest(f"Invalid mode: {mode}")

            # Initialize parameters using the selected class
            params = params_class()

            # Populate parameters from form data
            if mode != 0:  # For modes requiring an input image
                params.image = cache_input_image()
                params.denoise = float(request.form.get("denoise", DEFAULT_DENOISE))

            if mode == 1:  # Upscale mode requires scale
                params.scale = float(request.form.get("scale", DEFAULT_SCALE))
            elif mode == 3:  # Inpainting mode requires mask_image
                params.mask_image = cache_mask_image()
            elif mode == 4:  # Outpainting mode requires direction
                params.direction = request.form.get("direction", "right")

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
        """Stops the ongoing SD generation process."""
        paint_biz_instance.stop_generate() 
        return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE})

    return sd_bp 

def create_model_blueprint(model_size_cache: TTLCache = None) -> Blueprint:
    """Creates and returns the model management blueprint."""
    if model_size_cache is None:
        model_size_cache = TTLCache(maxsize=100, ttl=3600)  # Initialize if not provided
    lock = threading.Lock()
    model_bp = Blueprint("model", __name__, url_prefix="/api")

    # --- Model Management Routes ---

    @model_bp.route("/init", methods=["POST"])
    @validate_json_input(["config"])
    def get_init_settings():
        """
        Initializes settings based on POSTed configuration.

        Expects a JSON payload with:
        - config: A dictionary containing configuration settings.

        Returns a JSON response with available schedulers.
        """
        try:
            post_config: Dict[str, Any] = request.get_json()["config"]
            
            # Validate configuration settings here before updating (implementation not shown)
            # ...

            for key, value in post_config.items():
                if key in model_config.config:
                    model_config.config[key] = value
            return jsonify(schedulers_util.schedulers)
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
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
    def application_exit(): 
        """
        DISABLED: This route was intended to terminate the application but is disabled
        due to security concerns. In a production environment, this functionality should
        be implemented with proper authentication and authorization mechanisms.
        """
        return jsonify({"error": "This endpoint is disabled."}), 403

    @model_bp.route("/checkModelExist", methods=["POST"])
    @validate_json_input(["models"])
    def check_model_exist():
        """
        Checks if models exist locally.

        Expects a JSON payload with:
        - models: A list of dictionaries, each with 'repo_id' and 'type'.

        Returns a JSON response with the existence status of each model.
        """
        try:
            model_list: List[Dict[str, Any]] = request.get_json()["models"]
            result_list = []

            for item in model_list:
                repo_id = item["repo_id"]
                model_type = item["type"]
                exists = utils.check_model_exist(model_type, repo_id)  # More descriptive variable name
                result_list.append({"repo_id": repo_id, "type": model_type, "exist": exists})

            return jsonify({"code": SUCCESS_CODE, "message": SUCCESS_MESSAGE, "exists": result_list})
        except Exception as e:
            logger.exception("Error in checkModelExist route.")
            return jsonify({"error": "An error occurred during model existence check."}), 500

    # --- Model Size Cache ---

    @model_bp.route("/getModelSize", methods=["POST"])
    @validate_json_input(["models"])
    def get_model_size():
        """
        Retrieves the sizes of specified models, using a cached approach. 

        Expects a JSON payload with:
        - models: A list of dictionaries, each with 'repo_id' and 'type'.

        Returns a JSON response with the size of each requested model.
        """
        try:
            model_list: List[Dict[str, Any]] = request.get_json()["models"]
            result_dict = {}
            request_list = []

            # Iterate through the list of models
            for item in model_list:
                repo_id = item["repo_id"]
                model_type = item["type"]
                key = f"{repo_id}_{model_type}"

                # Check if the model size is in the cache
                with lock: # Acquire the lock before accessing the cache
                    total_size = model_size_cache.get(key)
                
                if total_size is None:
                    # If not in the cache, add it to the list of models to fetch
                    request_list.append((repo_id, model_type))
                else:
                    # If found in the cache, add it to the result dictionary
                    result_dict[key] = bytes2human(total_size, "%(value).2f%(symbol)s")

            # If there are models that need to be fetched
            if request_list:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit tasks to the thread pool to fetch the sizes
                    futures = [
                        executor.submit(fill_size_execute, repo_id, model_type, result_dict)
                        for repo_id, model_type in request_list
                    ]
                    concurrent.futures.wait(futures)  # Wait for all tasks to finish
                    executor.shutdown()

            # Return the results
            return jsonify({"code": success_code, "message": success_message, "sizeList": result_dict})
        except Exception as e:
            logger.exception("Error in getModelSize route.")
            return jsonify({"error": "An error occurred during model size retrieval."}), 500

    def fill_size_execute(repo_id: str, model_type: int, result_dict: dict):
        """
        Retrieves the size of a model, updates the cache, and updates the result dictionary. 
        This function is designed to be run in a separate thread.
        """
        key = f"{repo_id}_{model_type}"

        # Fetch the size based on the model type
        if model_type == 4:
            total_size = utils.get_ESRGAN_size()
        else:
            total_size = HFPlaygroundDownloader().get_model_total_size(repo_id, model_type)

        # Update the cache and result dictionary with the lock acquired
        with lock:
            model_size_cache[key] = total_size 
            result_dict[key] = bytes2human(total_size, "%(value).2f%(symbol)s") 

    @model_bp.route("/downloadModel", methods=["POST"])
    @validate_json_input(["models"])
    def download_model():
        """
        Handles model download requests and streams progress updates.
        """
        try:
            model_list: List[Dict[str, Any]] = request.get_json()["models"]

            if not isinstance(model_list, list):
                raise BadRequest("Invalid model list. Expected a list of model dictionaries.")

            # Create a new adapter instance for each request
            downloader_adapter = model_download_adapter.Model_Downloader_Adapter()
            iterator = downloader_adapter.download(model_list)
            return Response(stream_with_context(iterator), content_type="text/event-stream")
        except requests.exceptions.RequestException as e: 
            logger.exception("Network error during model download.")
            return jsonify({"error": f"Network error: {str(e)}"}), 500
        except BadRequest as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("Error in downloadModel route.")
            return jsonify({"error": "An error occurred during model download."}), 500


    return model_bp

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

# Create instances of the dependencies
paint_biz_instance = paint_biz.PaintBiz() 
llm_biz_instance = llm_biz.LLMBiz()

# Register the blueprints, passing in the dependencies
app.register_blueprint(create_llm_blueprint(llm_biz_instance, paint_biz_instance))
app.register_blueprint(create_sd_blueprint(paint_biz_instance, llm_biz_instance))
app.register_blueprint(create_model_blueprint(size_cache))

# --- Image Caching ---

def cache_input_image():
    """
    Caches the uploaded input image and returns the file path.
    Handles file extensions based on content type and creates a timestamped filename.
    Raises a BadRequest exception if no image file is provided or if the file type is invalid.
    """
    file = request.files.get("image")
    if not file:
        raise BadRequest("No image file provided.")

    # Validate the file type using imghdr
    file_data = file.stream.read()
    file.stream.seek(0) # Reset the stream position
    image_type = imghdr.what(None, h=file_data)
    if image_type is None:
        raise UnsupportedMediaType("Invalid image file type.")

    allowed_types = ['jpeg', 'png', 'gif', 'bmp']
    if image_type not in allowed_types:
        raise UnsupportedMediaType(f"Unsupported image type: {image_type}. Allowed types: {', '.join(allowed_types)}")

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
    utils.cache_file(file_path, file_path.stat().st_size)
    return str(file_path) # Return as string for compatibility

def cache_mask_image():
    """
    Caches the uploaded mask image and returns the file path.
    Creates a timestamped filename and saves the mask as a PNG image.
    Raises a BadRequest exception if no mask image file is provided.
    """
    file = request.files.get("mask_image")
    if not file:
        raise BadRequest("No mask image file provided.")

    mask_width = int(request.form.get("mask_width", default_image_size))
    mask_height = int(request.form.get("mask_height", default_image_size))

    # Validate and process the mask image
    try:
        mask_image = Image.open(file.stream).convert("L") # Convert to grayscale ("L" mode)
    except (IOError, OSError) as e:
        raise BadRequest("Invalid mask image file.") from e

    mask_image = mask_image.resize((mask_width, mask_height))

    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    # Use pathlib for file path construction
    save_dir = Path("./static/sd_mask") / folder
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{base_name}.png"

    mask_image.save(file_path)
    utils.cache_file(file_path, file_path.stat().st_size)
    return str(file_path) 

# Main entry point for running the Flask application
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Playground Web service")
    parser.add_argument("--port", type=int, default=59999, help="Service listen port")
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port)