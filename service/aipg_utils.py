"""
AI Playground Utilities Module
-----------------------------
This module provides utility functions for the AI Playground service, including:

- Image processing and conversion utilities
- Model existence checking across different backends (default, OpenVINO, ComfyUI, llama.cpp)
- Repository ID handling and path construction
- File caching with content-based addressing
- Hardware detection for Intel XPU devices
- Subprocess execution helpers

These utilities support the AI service components by providing common functionality
for working with models, files, and hardware resources.
"""

import base64
import hashlib
import io
import logging
import math
import os
import shlex
import shutil
import subprocess
from typing import IO

import torch
from PIL import Image

from service import service_config


def image_to_base64(image: Image.Image):
    """
    Convert a PIL Image to a base64-encoded data URL string.

    Args:
        image: PIL Image object to convert

    Returns:
        str: Base64-encoded data URL string with PNG format
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64,{}".format(base64.b64encode(buffered.getvalue()).decode("utf-8"))


def generate_mask_image(mask_flag_bytes: bytes, width: int, height: int):
    """
    Generate a mask image from binary mask data.

    Args:
        mask_flag_bytes: Binary data representing the mask
        width: Width of the mask image
        height: Height of the mask image

    Returns:
        Image.Image: PIL RGB image created from the mask data
    """
    import numpy as np
    from PIL import Image

    np_data = np.frombuffer(mask_flag_bytes, dtype=np.uint8)
    image = Image.fromarray(np_data.reshape((height, width)), mode="L").convert("RGB")

    return image


def get_shape_ceil(h: float, w: float):
    """
    Calculate a shape size rounded up to the nearest multiple of 64.

    Args:
        h: Height value
        w: Width value

    Returns:
        float: Square root of area (h*w) rounded up to nearest multiple of 64
    """
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def get_image_shape_ceil(image: Image.Image):
    """
    Get the ceiling of an image shape rounded to the nearest multiple of 64.

    Args:
        image: Input image

    Returns:
        float: Square root of image area rounded up to nearest multiple of 64
    """
    H, W = image.shape[:2]
    return get_shape_ceil(H, W)


def check_mmodel_exist(type: int, repo_id: str, backend: str) -> bool:
    """
    Check if a model exists in the specified backend.

    Delegates to backend-specific functions based on the backend parameter.

    Args:
        type: Integer type identifier for the model
        repo_id: Repository ID of the model
        backend: Backend type ("default", "openvino", "comfyui", or "llama_cpp")

    Returns:
        bool: True if the model exists, False otherwise

    Raises:
        NameError: If an unknown backend is specified
    """
    if backend == "default":
        return check_defaultbackend_mmodel_exist(type, repo_id)
    if backend == "openvino":
        return check_openvino_model_exists(type, repo_id)
    if backend == "comfyui":
        return check_comfyui_model_exists(type, repo_id)
    if backend == "llama_cpp":
        return check_llama_cpp_model_exists(type, repo_id)
    raise NameError("Unknown Backend")


def check_openvino_model_exists(type, repo_id) -> bool:
    """
    Check if an OpenVINO model exists.

    Args:
        type: Model type (unused in this function)
        repo_id: Repository ID of the model

    Returns:
        bool: True if the model exists, False otherwise
    """
    folder_name = repo_local_root_dir_name(repo_id)
    dir = service_config.openvino_model_paths.get("openvinoLLM")
    return os.path.exists(os.path.join(dir, folder_name))


def check_llama_cpp_model_exists(type, repo_id) -> bool:
    """
    Check if a llama.cpp model exists.

    Args:
        type: Model type to convert to string representation
        repo_id: Repository ID of the model

    Returns:
        bool: True if the model exists, False otherwise
    """
    model_dir = service_config.llama_cpp_model_paths.get(convert_model_type(type))
    dir_to_look_for = os.path.join(model_dir, repo_local_root_dir_name(repo_id), extract_model_id_pathsegments(repo_id))
    return os.path.exists(dir_to_look_for)


def check_comfyui_model_exists(type, repo_id) -> bool:
    """
    Check if a ComfyUI model exists.

    Different model types have different directory structures:
    - faceswap/facerestore models use a flat directory structure
    - nsfwdetector has a special directory structure
    - other model types use a nested directory structure

    Args:
        type: Model type to convert to string representation
        repo_id: Repository ID of the model

    Returns:
        bool: True if the model exists, False otherwise
    """
    model_type = convert_model_type(type)
    model_dir = service_config.comfy_ui_model_paths.get(model_type)
    if model_type in ("faceswap", "facerestore"):
        dir_to_look_for = os.path.join(model_dir, flat_repo_local_dir_name(repo_id))
    elif model_type == "nsfwdetector":
        dir_to_look_for = os.path.join(model_dir, "vit-base-nsfw-detector", extract_model_id_pathsegments(repo_id))
    else:
        dir_to_look_for = os.path.join(
            model_dir, repo_local_root_dir_name(repo_id), extract_model_id_pathsegments(repo_id),
        )
    return os.path.exists(dir_to_look_for)


def trim_repo(repo_id):
    """
    Get the first two segments of a repository ID.

    Args:
        repo_id: Repository ID to trim

    Returns:
        str: First two path segments of the repository ID
    """
    return "/".join(repo_id.split("/")[:2])


def extract_model_id_pathsegments(repo_id) -> str:
    """
    Extract all segments after the first two from a repository ID.

    Args:
        repo_id: Repository ID to extract from

    Returns:
        str: Path segments after the first two, joined with "/"
    """
    return "/".join(repo_id.split("/")[2:])


def repo_local_root_dir_name(repo_id):
    """
    Convert the first two segments of a repository ID to a local directory name.

    Replaces "/" with "---" for use in the filesystem.

    Args:
        repo_id: Repository ID to convert

    Returns:
        str: Local directory name based on repository organization/name
    """
    return "---".join(repo_id.split("/")[:2])


def flat_repo_local_dir_name(repo_id):
    """
    Convert an entire repository ID to a flat local directory name.

    Replaces all "/" with "---" for use in the filesystem.

    Args:
        repo_id: Repository ID to convert

    Returns:
        str: Flattened local directory name
    """
    return "---".join(repo_id.split("/"))


def check_defaultbackend_mmodel_exist(type: int, repo_id: str) -> bool:
    """
    Check if a model exists in the default backend.

    Different model types require different checks:
    - LLM (0): Checks for existence of the repository directory
    - Stable Diffusion (1): Checks for model_index.json or single file
    - LoRA (2): Checks for pytorch_lora_weights.safetensors or .bin
    - Other types: Have specific file/directory checks

    Args:
        type: Integer type identifier for the model
        repo_id: Repository ID of the model

    Returns:
        bool: True if the model exists, False otherwise
    """
    folder_name = repo_local_root_dir_name(repo_id)
    if type == 0:
        dir = service_config.service_model_paths.get("llm")
        return os.path.exists(os.path.join(dir, folder_name))
    if type == 1:
        dir = service_config.service_model_paths.get("stableDiffusion")
        if is_single_file(repo_id):
            return os.path.exists(os.path.join(dir, repo_id))
        return os.path.exists(os.path.join(dir, folder_name, "model_index.json"))
    if type == 2:
        dir = service_config.service_model_paths.get("lora")
        if is_single_file(repo_id):
            return os.path.exists(os.path.join(dir, repo_id))
        return os.path.exists(os.path.join(dir, folder_name, "pytorch_lora_weights.safetensors")) or os.path.exists(
            os.path.join(dir, folder_name, "pytorch_lora_weights.bin"),
        )
    if type == 3:
        dir = service_config.service_model_paths.get("vae")
        return os.path.exists(os.path.join(dir, folder_name))
    if type == 4:
        import realesrgan

        dir = service_config.service_model_paths.get("ESRGAN")
        return os.path.exists(os.path.join(dir, realesrgan.ESRGAN_MODEL_URL.split("/")[-1]))
    if type == 5:
        dir = service_config.service_model_paths.get("embedding")
        return os.path.exists(os.path.join(dir, folder_name))
    if type == 6:
        dir = service_config.service_model_paths.get("inpaint")
        if is_single_file(repo_id):
            return os.path.exists(os.path.join(dir, repo_id))
        return os.path.exists(os.path.join(dir, repo_id.replace("/", "---"), "model_index.json"))
    if type == 7:
        dir = service_config.service_model_paths.get("preview")
        return (
            os.path.exists(os.path.join(dir, folder_name, "config.json"))
            or os.path.exists(os.path.join(dir, f"{repo_id}.safetensors"))
            or os.path.exists(os.path.join(dir, f"{repo_id}.bin"))
        )


def convert_model_type(type: int):
    """
    Convert an integer model type to its string representation.

    Handles all known model types for different backends.

    Args:
        type: Integer type identifier for the model

    Returns:
        str: String representation of the model type

    Raises:
        Exception: If an unknown model type is provided
    """
    if type == 0:
        return "llm"
    if type == 1:
        return "stableDiffusion"
    if type == 2:
        return "lora"
    if type == 3:
        return "vae"
    if type == 4:
        return "ESRGAN"
    if type == 5:
        return "embedding"
    if type == 6:
        return "inpaint"
    if type == 7:
        return "preview"
    if type == 8:
        return "ggufLLM"
    if type == 9:
        return "openvinoLLM"
    if type == 100:
        return "unet"
    if type == 101:
        return "clip"
    if type == 102:
        return "vae"
    if type == 103:
        return "defaultCheckpoint"
    if type == 104:
        return "defaultLora"
    if type == 105:
        return "controlNet"
    if type == 106:
        return "faceswap"
    if type == 107:
        return "facerestore"
    if type == 108:
        return "nsfwdetector"
    if type == 109:
        return "checkpoints"
    raise Exception(f"unknown model type value {type}")


def get_model_path(type: int, backend: str) -> str | None:
    """
    Get the base directory path for a model type on a specific backend.

    Args:
        type: Integer type identifier for the model
        backend: Backend type ("default", "llama_cpp", "openvino", or "comfyui")

    Returns:
        str: Directory path for the specified model type and backend
    """
    if backend == "default":
        return service_config.service_model_paths.get(convert_model_type(type))
    if backend == "llama_cpp":
        return service_config.llama_cpp_model_paths.get(convert_model_type(type))
    if backend == "openvino":
        return service_config.openvino_model_paths.get(convert_model_type(type))
    if backend == "comfyui":
        return service_config.comfy_ui_model_paths.get(convert_model_type(type))
    raise NameError("Unknown Backend")


def calculate_md5(file_path: str):
    """
    Calculate the MD5 hash of a file.

    Reads the file in chunks to efficiently handle large files.

    Args:
        file_path: Path to the file to hash

    Returns:
        str: Hexadecimal MD5 hash of the file
    """
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def create_cache_path(md5: str, file_size: int):
    """
    Create a cache path based on an MD5 hash and file size.

    The path is structured with nested directories based on the MD5 hash
    to avoid too many files in a single directory.

    Args:
        md5: MD5 hash of the file
        file_size: Size of the file in bytes

    Returns:
        str: Absolute path to the cache location
    """
    cache_dir = "./cache"
    sub_dirs = [md5[i : i + 4] for i in range(0, len(md5), 4)]
    cache_path = os.path.abspath(os.path.join(cache_dir, *sub_dirs, f"{md5}_{file_size}"))
    return cache_path


def calculate_md5_from_stream(file_stream: IO[bytes]):
    """
    Calculate the MD5 hash of a file from a stream.

    Args:
        file_stream: Stream of bytes to hash

    Returns:
        str: Hexadecimal MD5 hash of the stream contents
    """
    file_hash = hashlib.md5()
    for chunk in iter(lambda: file_stream.read(8192), b""):
        file_hash.update(chunk)
    return file_hash.hexdigest()


def cache_file(file_path: IO[bytes] | str, file_size: int) -> str:
    """
    Cache a file using content-based addressing.

    Creates a hard link between the original file and the cached file,
    which saves disk space while preserving the file in both locations.

    Args:
        file_path: Path to the file to cache
        file_size: Size of the file in bytes
    """
    md5 = calculate_md5(file_path)

    cache_path = create_cache_path(md5, file_size)

    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        os.rename(file_path, cache_path)

    if os.path.exists(file_path):
        os.remove(file_path)
    os.link(cache_path, file_path)

    return cache_path


def is_single_file(filename: str):
    """
    Check if a filename represents a single model file.

    Single files are identified by specific extensions (.safetensors, .bin, .gguf).

    Args:
        filename: Filename to check

    Returns:
        bool: True if it's a single file, False otherwise
    """
    return filename.endswith(".safetensors") or filename.endswith(".bin") or filename.endswith(".gguf")


def get_ESRGAN_size():
    """
    Get the file size of the ESRGAN model.

    Makes a request to the ESRGAN URL and reads the Content-Length header.

    Returns:
        int: Size of the ESRGAN model in bytes
    """
    import realesrgan
    import requests

    response = requests.get(realesrgan.ESRGAN_MODEL_URL, stream=True)
    with response:
        return int(response.headers.get("Content-Length"))


def get_support_graphics():
    """
    Get a list of supported Intel XPU graphics devices.

    Returns:
        list: List of dictionaries with device index and name information
    """
    try:
        device_count = torch.xpu.device_count()
    except Exception:
        device_count = 0
    graphics = []
    for i in range(device_count):
        try:
            device_name = torch.xpu.get_device_name(i)
        except Exception:
            device_name = "Dummy XPU Device"
        graphics.append({"index": i, "name": device_name})
    if len(graphics) == 0:
        # Fallback to a dummy device if none are available
        graphics = [{"index": 0, "name": "Dummy XPU Device"}]
    return graphics


def call_subprocess(process_command: str, cwd: str | None = None) -> str:
    """
    Execute a subprocess command and return the output.

    Uses shlex to handle command parsing and logs the command execution.

    Args:
        process_command: Command string to execute
        cwd: Optional working directory for the command

    Returns:
        str: Output from the command, stripped of trailing whitespace

    Raises:
        subprocess.CalledProcessError: If the command execution fails
    """
    args = shlex.split(process_command)
    try:
        logging.info(f"calling cmd process: {args}")
        output = subprocess.check_output(args, cwd=cwd)
        return output.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to call subprocess {process_command} with error {e}")
        raise e


def remove_existing_filesystem_resource(path: str):
    """
    Remove a file or directory if it exists.

    Handles both files and directories with appropriate removal methods.

    Args:
        path: Path to the file or directory to remove
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
