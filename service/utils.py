import base64
import math
from typing import IO, Union, List, Dict
from pathlib import Path
from PIL import Image
import io
import hashlib
import torch
import numpy as np
import tempfile
import shutil
import re
import requests

import model_config
import realesrgan

# Constants for model types
MODEL_TYPES = {
    0: "llm",
    1: "stableDiffusion",
    2: "lora",
    3: "vae",
    4: "ESRGAN",
    5: "embedding",
    6: "inpaint",
    7: "preview"
}

def image_to_base64(image: Image.Image) -> str:
    """
    Converts a PIL Image to a base64 encoded string.

    Args:
        image (Image.Image): The PIL Image to convert.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    with io.BytesIO() as buffered:
        image.save(buffered, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def generate_mask_image(mask_flag_bytes: bytes, width: int, height: int) -> Image.Image:
    """
    Generates a mask image from a byte string.

    Args:
        mask_flag_bytes (bytes): The byte string representing the mask data.
        width (int): The width of the mask image.
        height (int): The height of the mask image.

    Returns:
        Image.Image: The generated mask image as a PIL Image.
    """
    np_data = np.frombuffer(mask_flag_bytes, dtype=np.uint8)
    image = Image.fromarray(np_data.reshape((height, width)), mode="L").convert("RGB")
    return image


def get_shape_ceil(h: float, w: float) -> float:
    """
    Calculates the ceiling of the square root of the product of height and width, 
    divided by 64.0, and then multiplies by 64.0.

    This function is likely used to determine the appropriate output size for models that 
    require input dimensions to be multiples of 64.

    Args:
        h (float): The height value.
        w (float): The width value.

    Returns:
        float: The calculated ceiling value.
    """
    return math.ceil(((h * w) ** 0.5) / 64.0) * 64.0


def get_image_shape_ceil(image: Image.Image) -> float:
    """
    Gets the ceiling of the shape of an image, ensuring it is a multiple of 64.

    Args:
        image (Image.Image): A PIL Image.

    Returns:
        float: The calculated ceiling value for the image shape.
    """
    width, height = image.size  # Ensure correct order of width and height
    return get_shape_ceil(height, width)


def check_model_exist(type: int, repo_id: str) -> bool:
    """
    Check if a model of a specified type exists in the configured model directory.

    The function checks for the existence of model files based on the provided `type`
    and `repo_id`. It retrieves the base directory for the specified model type from 
    the `model_config` and checks if the necessary files or directories exist within
    that directory.

    Args:
        type (int): An integer representing the type of model (e.g., 0 for LLM, 1 for Stable Diffusion).
        repo_id (str): The repository ID of the model on Hugging Face.

    Returns:
        bool: True if the model exists, False otherwise.

    Raises:
        ValueError: If an unknown model type value is provided.
    """
    folder_name = repo_id.replace("/", "---")
    model_type_paths = {
        0: Path(model_config.config.get(MODEL_TYPES[0])) / folder_name / "config.json",
        1: Path(model_config.config.get(MODEL_TYPES[1])) / folder_name / "model_index.json",
        2: Path(model_config.config.get(MODEL_TYPES[2])) / folder_name,
        3: Path(model_config.config.get(MODEL_TYPES[3])) / folder_name,
        4: Path(model_config.config.get(MODEL_TYPES[4])) / Path(realesrgan.ESRGAN_MODEL_URL).name,
        5: Path(model_config.config.get(MODEL_TYPES[5])) / folder_name,
        6: Path(model_config.config.get(MODEL_TYPES[6])) / folder_name / "model_index.json",
        7: Path(model_config.config.get(MODEL_TYPES[7])) / folder_name / "config.json"
    }

    if type in model_type_paths:
        path = model_type_paths[type]
        if type == 2:  # LORA models
            return any([
                (path / "pytorch_lora_weights.safetensors").exists(),
                (path / "pytorch_lora_weights.bin").exists(),
                is_single_file(repo_id) and path.exists()
            ])
        if type == 7:  # Preview models
            return any([
                path.exists(),
                (Path(model_config.config.get(MODEL_TYPES[7])) / f"{repo_id}.safetensors").exists(),
                (Path(model_config.config.get(MODEL_TYPES[7])) / f"{repo_id}.bin").exists()
            ])
        if is_single_file(repo_id) and type in {1, 2, 6}:
            return path.exists()
        return path.exists()
    else:
        raise ValueError(f"Unknown model type value: {type}")


def convert_model_type(type: int) -> str:
    """
    Converts a model type code (int) to its corresponding string representation.

    Args:
        type (int): An integer representing the model type.

    Returns:
        str: The string representation of the model type.

    Raises:
        ValueError: If an unknown model type value is provided. 
    """
    try:
        return MODEL_TYPES[type]
    except KeyError:
        raise ValueError(f"Unknown model type value: {type}")


def get_model_path(type: int) -> Path:
    """
    Gets the file path associated with a given model type.

    Args:
        type (int): The integer code representing the model type.

    Returns:
        Path: The file path associated with the specified model type.
    """
    return Path(model_config.config.get(convert_model_type(type)))


def calculate_md5(file_path: Path) -> str:
    """
    Calculates the MD5 hash of a file.

    Args:
        file_path (Path): The path to the file.

    Returns:
        str: The MD5 hash of the file.
    """
    with file_path.open("rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def create_cache_path(md5: str, file_size: int) -> Path:
    """
    Creates a path for caching a file based on its MD5 hash and size.

    Args:
        md5 (str): The MD5 hash of the file.
        file_size (int): The size of the file in bytes.

    Returns:
        Path: The constructed cache path for the file.
    """
    cache_dir = Path("./cache")
    sub_dirs = [md5[i:i + 4] for i in range(0, len(md5), 4)]
    return cache_dir.joinpath(*sub_dirs, f"{md5}_{file_size}").resolve()


def calculate_md5_from_stream(file_stream: IO[bytes]) -> str:
    """
    Calculates the MD5 hash from a file stream.

    Args:
        file_stream (IO[bytes]): The file stream to read from.

    Returns:
        str: The MD5 hash calculated from the file stream.
    """
    file_hash = hashlib.md5()
    for chunk in iter(lambda: file_stream.read(8192), b""):
        file_hash.update(chunk)
    return file_hash.hexdigest()


def cache_file(file_path: Union[IO[bytes], Path], file_size: int):
    """
    Caches a file based on its MD5 hash and size.

    The function calculates the MD5 hash of the provided file, creates a directory
    structure for caching based on the hash, and moves the file to the cache location.
    It then creates a hard link from the original file path to the cached file.

    Args:
        file_path (Union[IO[bytes], Path]): The path to the file, either a file-like object or a Path.
        file_size (int): The size of the file in bytes.
    """
    if isinstance(file_path, io.IOBase):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            shutil.copyfileobj(file_path, temp_file)
            temp_file_path = Path(temp_file.name)
        md5 = calculate_md5(temp_file_path)
        cache_path = create_cache_path(md5, file_size)

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file_path.rename(cache_path)
    else:
        md5 = calculate_md5(file_path)
        cache_path = create_cache_path(md5, file_size)

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.rename(cache_path)

        if file_path.exists():
            file_path.unlink()

        try:
            cache_path.link_to(file_path)
        except OSError:
            shutil.copyfile(cache_path, file_path)

def is_single_file(filename: str) -> bool:
    """
    Checks if a filename corresponds to a single model file.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if the filename corresponds to a single file (.safetensors or .bin), False otherwise.
    """
    return filename.endswith(('.safetensors', '.bin'))

def get_ESRGAN_size() -> int:
    """
    Retrieves the size of the ESRGAN model file.

    Returns:
        int: The size of the ESRGAN model file in bytes.
    """
    with requests.Session() as session:
        response = session.head(realesrgan.ESRGAN_MODEL_URL)  # Use a HEAD request to get only the headers
        return int(response.headers.get("Content-Length"))  # Return the content length from the headers.

def get_support_graphics(env_type: str) -> List[Dict[str, Union[int, str]]]:
    """
    Retrieves a list of supported Intel Arc Graphics devices.

    Args:
        env_type (str): The environment type (e.g., "production", "development").

    Returns:
        list: A list of dictionaries, each containing the index and name of a supported device.
    """
    arc_regex = re.compile(r"Intel\(R\) Arc\(TM\) [^ ]+ Graphics")
    device_count = torch.xpu.device_count()
    model_config.env_type = env_type
    graphics = []
    for i in range(device_count):
        device_name = torch.xpu.get_device_name(i)
        if device_name == "Intel(R) Arc(TM) Graphics" or arc_regex.search(device_name):
            graphics.append({"index": i, "name": device_name})
    return graphics