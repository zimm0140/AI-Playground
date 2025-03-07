"""
AI Playground Web API Module
---------------------------
This module implements a Flask-based web API for AI Playground, providing endpoints for:
- Large Language Model (LLM) interaction
- Stable Diffusion (SD) image generation
- Model management (downloading, checking, etc.)
- ComfyUI custom nodes and Python package management
- RAG (Retrieval-Augmented Generation) functionality

The API uses Server-Sent Events (SSE) for streaming responses where appropriate.
"""

import sys

import comfyui_downloader
from web_request_bodies import ComfyUICheckWorkflowRequirementRequest, DownloadModelRequestBody, ComfyUICustomNodesDownloadRequest, ComfyUIPackageInstallRequest


# Credit to https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14186
# Related issues:
# + https://github.com/XPixelGroup/BasicSR/issues/649
# + https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as functional

        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass

from datetime import datetime
import os
import threading
from flask import jsonify, request, Response, stream_with_context
from apiflask import APIFlask

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
import aipg_utils as utils
import rag
import service_config
from model_downloader import HFPlaygroundDownloader
from psutil._common import bytes2human
import traceback

import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


app = APIFlask(__name__)

# =================== CORE SERVICE ENDPOINTS ===================
# These endpoints provide basic service functionality like health checks and service control


@app.get("/healthy")
def healthEndpoint():
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        JSON response with health status.
    """
    return jsonify({"health": "OK"})


@app.get("/api/applicationExit")
def applicationExit():
    """
    Endpoint to terminate the application by sending a SIGINT signal to the process.
    """
    from signal import SIGINT

    pid = os.getpid()
    os.kill(pid, SIGINT)


@app.post("/api/init")
def get_init_settings():
    """
    Initialization endpoint that configures service settings and returns available schedulers.
    
    Processes configuration provided in the request and updates service model paths.
    
    Returns:
        JSON response with available schedulers.
    """
    import schedulers_util

    post_config: dict = request.get_json()
    for k, v in post_config.items():
        if service_config.service_model_paths.__contains__(k):
            service_config.service_model_paths.__setitem__(k, v)

    return jsonify(schedulers_util.schedulers)


@app.post("/api/getGraphics")
def get_graphics():
    """
    Endpoint to retrieve information about supported graphics hardware.
    
    Returns:
        JSON response with graphics hardware information.
    """
    return jsonify(utils.get_support_graphics())


# =================== LLM ENDPOINTS ===================
# Endpoints for Large Language Model interactions


@app.post("/api/llm/chat")
def llm_chat():
    """
    Endpoint for LLM chat interactions.
    
    Disposes of any basic model and streams the LLM response as Server-Sent Events.
    
    Returns:
        Streaming response with the LLM output.
    """
    paint_biz.dispose_basic_model()
    params = request.get_json()
    llm_params = llm_biz.LLMParams(**params)
    sse_invoker = LLM_SSE_Adapter()
    it = sse_invoker.text_conversation(llm_params)
    return Response(stream_with_context(it), content_type="text/event-stream")


@app.get("/api/llm/stopGenerate")
def stop_llm_generate():
    """
    Endpoint to stop ongoing LLM text generation.
    
    Returns:
        JSON response indicating success.
    """
    import llm_biz

    llm_biz.stop_generate()
    return jsonify({"code": 0, "message": "success"})


# =================== STABLE DIFFUSION ENDPOINTS ===================
# Endpoints for image generation and manipulation


@app.post("/api/sd/generate")
def sd_generate():
    """
    Endpoint for Stable Diffusion image generation.
    
    Supports multiple modes:
    - Text-to-image (mode 0)
    - Upscale (mode 1)
    - Image-to-image (mode 2)
    - Inpaint (mode 3)
    - Outpaint (mode 4)
    
    Request format:
    {
        "device": int,
        "prompt": str,
        "model_repo_id": str,
        "mode": int,
        "image?": file,
        "mask?": file
    }
    
    Returns:
        Streaming response with generation progress and results.
    """
    llm_biz.dispose()
    mode = request.form.get("mode", default=0, type=int)
    if mode != 0:
        if mode == 1:
            params = UpscaleImageParams()
            params.scale = request.form.get("scale", 1.5, type=float)
        elif mode == 2:
            params = ImageToImageParams()
        elif mode == 3:
            params = InpaintParams()
            params.mask_image = cache_mask_image()
        elif mode == 4:
            params = OutpaintParams()
            params.direction = request.form.get("direction", "right", type=str)

        params.image = cache_input_image()
        params.denoise = request.form.get("denoise", 0.5, type=float)
    else:
        params = TextImageParams()
    base_params = params
    base_params.device = request.form.get("device", default=0, type=int)
    base_params.prompt = request.form.get("prompt", default="", type=str)
    base_params.model_name = request.form["model_repo_id"]
    base_params.mode = mode
    base_params.width = request.form.get("width", default=512, type=int)
    base_params.negative_prompt = request.form.get(
        "negative_prompt", default="", type=str
    )
    base_params.height = request.form.get("height", default=512, type=int)
    base_params.generate_number = request.form.get(
        "generate_number", default=1, type=int
    )
    base_params.inference_steps = request.form.get(
        "inference_steps", default=12, type=int
    )
    base_params.guidance_scale = request.form.get(
        "guidance_scale", default=7.5, type=float
    )
    base_params.seed = request.form.get("seed", default=-1, type=int)
    base_params.scheduler = request.form.get("scheduler", default="None", type=str)
    base_params.lora = request.form.get("lora", default="None", type=str)
    base_params.image_preview = request.form.get("image_preview", default=0, type=int)
    base_params.safe_check = request.form.get("safe_check", default=1, type=int)
    sse_invoker = SD_SSE_Adapter(request.url_root)
    it = sse_invoker.generate(params)
    return Response(stream_with_context(it), content_type="text/event-stream")


@app.get("/api/sd/stopGenerate")
def stop_sd_generate():
    """
    Endpoint to stop ongoing Stable Diffusion image generation.
    
    Returns:
        JSON response indicating success.
    """
    import paint_biz

    paint_biz.stop_generate()
    return jsonify({"code": 0, "message": "success"})


# =================== MODEL MANAGEMENT ENDPOINTS ===================
# Endpoints for checking and downloading models from HuggingFace


@app.post("/api/checkModelAlreadyLoaded")
@app.input(DownloadModelRequestBody.Schema, location='json', arg_name='download_request_data')
def check_model_already_loaded(download_request_data: DownloadModelRequestBody):
    """
    Endpoint to check if requested models are already loaded.
    
    Args:
        download_request_data: Request body containing model information to check.
        
    Returns:
        JSON response with load status for each requested model.
    """
    result_list = []
    for item in download_request_data.data:
        base_response = {
            "repo_id": item.repo_id,
            "type": item.type,
            "backend": item.backend,
            "already_loaded": utils.check_mmodel_exist(item.type, item.repo_id, item.backend),
        }

        if (item.additionalLicenseLink is not None):
            base_response['additionalLicenseLink'] = item.additionalLicenseLink

        result_list.append(base_response)
    return jsonify({"code": 0, "message": "success", "data": result_list})


@app.get("/api/checkHFRepoExists")
def check_if_huggingface_repo_exists():
    """
    Endpoint to check if a specified HuggingFace repository exists.
    
    Query parameters:
        repo_id: Repository ID to check.
        
    Returns:
        JSON response indicating whether the repository exists.
    """
    repo_id = request.args.get('repo_id')
    downloader = HFPlaygroundDownloader()
    exists = downloader.hf_url_exists(repo_id)
    return jsonify(
        {
            "exists": exists
        }
    )


@app.get("/api/isLLM")
def is_llm():
    """
    Endpoint to check if a specified model is a Language Model.
    
    Query parameters:
        repo_id: Repository ID to check.
        
    Returns:
        JSON response indicating whether the model is an LLM.
    """
    repo_id = request.args.get('repo_id')
    downloader = HFPlaygroundDownloader()
    try:
        model_type_hf = downloader.probe_type(repo_id)
    except Exception:
        model_type_hf = "undefined"
    return jsonify(
        {
            "isllm": model_type_hf == "text-generation"
        }
    )


# Cache for storing model sizes to avoid redundant computations
size_cache = dict()
lock = threading.Lock()


@app.post("/api/isModelGated")
def is_model_gated():
    """
    Endpoint to check if specified models are gated (require authentication).
    
    Returns:
        JSON response with gated status for each requested model.
    """
    list = request.get_json()
    downloader = HFPlaygroundDownloader()
    gated = {item["repo_id"]: downloader.is_gated(item["repo_id"]) for item in list}

    return jsonify(
        {
            "code": 0,
            "message": "success",
            "gatedList": gated,
        }
    )


@app.route("/api/isAccessGranted", methods=["POST"])
def is_access_granted():
    """
    Endpoint to check if the provided token grants access to specified models.
    
    Request format:
        [List of model info objects, HuggingFace token]
        
    Returns:
        JSON response with access status for each requested model.
    """
    list, hf_token = request.get_json()
    downloader = HFPlaygroundDownloader(hf_token)
    accessGranted = { item["repo_id"] : downloader.is_access_granted(item["repo_id"], item["type"], item["backend"]) for item in list }
    return jsonify(
        {
            "accessList": accessGranted
        }
    )


@app.post("/api/getModelSize")
def get_model_size():
    """
    Endpoint to get the size of specified models.
    
    Uses a thread pool executor to fetch sizes in parallel and a cache to avoid
    redundant computations.
    
    Returns:
        JSON response with size information for each requested model.
    """
    import concurrent.futures

    list = request.get_json()
    result_dict = dict()
    request_list = []
    for item in list:
        repo_id = item["repo_id"]
        type = item["type"]
        key = f"{repo_id}_{type}"
        total_size = size_cache.get(key)
        if total_size is None:
            request_list.append((repo_id, type))
        else:
            result_dict.__setitem__(
                key, bytes2human(total_size, "%(value).2f%(symbol)s")
            )

    if request_list.__len__() > 0:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(fill_size_execute, repo_id, type, result_dict)
                for repo_id, type in request_list
            ]
            concurrent.futures.wait(futures)
            executor.shutdown()

    return jsonify(
        {
            "code": 0,
            "message": "success",
            "sizeList": result_dict,
        }
    )


def fill_size_execute(repo_id: str, type: int, result_dict: dict):
    """
    Helper function to fetch model size and update the result dictionary.
    
    Used by the thread pool executor in get_model_size().
    
    Args:
        repo_id: Repository ID of the model.
        type: Type of the model.
        result_dict: Dictionary to store the result.
    """
    key = f"{repo_id}_{type}"
    if type == 4:
        total_size = utils.get_ESRGAN_size()
    else:
        total_size = HFPlaygroundDownloader().get_model_total_size(repo_id, type)
    with lock:
        size_cache.__setitem__(key, total_size)
        result_dict.__setitem__(key, bytes2human(total_size, "%(value).2f%(symbol)s"))


@app.post("/api/downloadModel")
@app.input(DownloadModelRequestBody.Schema, location='json', arg_name='download_request_data')
def download_model(download_request_data: DownloadModelRequestBody):
    """
    Endpoint to download requested models.
    
    Stops any ongoing downloads before starting a new one.
    Uses the Bearer token for authentication if provided.
    
    Args:
        download_request_data: Request body containing models to download.
        
    Returns:
        Streaming response with download progress and status.
    """
    if model_download_adpater._adapter is not None:
        model_download_adpater._adapter.stop_download()
    try:
        model_download_adpater._adapter = (
            model_download_adpater.Model_Downloader_Adapter(
                hf_token=get_bearer_token(request)
            )
        )
        iterator = model_download_adpater._adapter.download(download_request_data.data)
        return Response(stream_with_context(iterator), content_type="text/event-stream")
    except Exception as e:
        traceback.print_exc()

        model_download_adpater._adapter.stop_download()
        ex_str = '{{"type": "error", "err_type": "{}"}}'.format(e)
        return Response(stream_with_context([ex_str]), content_type="text/event-stream")


@app.get("/api/stopDownloadModel")
def stop_download_model():
    """
    Endpoint to stop ongoing model downloads.
    
    Returns:
        JSON response indicating success.
    """
    if model_download_adpater._adapter is not None:
        model_download_adpater._adapter.stop_download()
    return jsonify({"code": 0, "message": "success"})


# =================== RAG FUNCTIONALITY ENDPOINTS ===================
# Endpoints for Retrieval-Augmented Generation operations


@app.post("/api/llm/enableRag")
def enable_rag():
    """
    Endpoint to enable Retrieval-Augmented Generation for LLMs.
    
    Initializes the RAG system with the specified model if not already initialized.
    
    Returns:
        JSON response indicating success.
    """
    if not rag.Is_Inited:
        repo_id = request.form.get("repo_id", default="", type=str)
        device = request.form.get("device", default=0, type=int)
        rag.init(repo_id, device)
    return jsonify({"code": 0, "message": "success"})


@app.get("/api/llm/disableRag")
def disable_rag():
    """
    Endpoint to disable Retrieval-Augmented Generation for LLMs.
    
    Disposes of the RAG system if it is initialized.
    
    Returns:
        JSON response indicating success.
    """
    if rag.Is_Inited:
        rag.dispose()
    return jsonify({"code": 0, "message": "success"})


@app.get("/api/llm/getRagFiles")
def get_rag_files():
    """
    Endpoint to get the list of files indexed for RAG.
    
    Returns:
        JSON response with the list of indexed files, including filenames and MD5 hashes.
    """
    try:
        result_list = list()
        index_list = rag.get_index_list()
        if list is not None:
            for index in index_list:
                result_list.append(
                    {"filename": index.get("name"), "md5": index.get("md5")}
                )

        return jsonify({"code": 0, "message": "success", "data": result_list})
    except Exception:
        traceback.print_exc()
        return jsonify({"code": -1, "message": "failed"})


@app.post("/api/llm/uploadRagFile")
def upload_rag_file():
    """
    Endpoint to upload and index a file for RAG.
    
    Form parameters:
        path: Path to the file to index.
        
    Returns:
        JSON response with success status and the MD5 hash of the indexed file.
    """
    try:
        path = request.form.get("path")
        code, md5 = rag.add_index_file(path)
        return jsonify({"code": code, "message": "success", "md5": md5})
    except Exception:
        traceback.print_exc()
        return jsonify({"code": -1, "message": "failed", path: path})


@app.post("/api/llm/deleteRagIndex")
def delete_rag_file():
    """
    Endpoint to delete a RAG index file.
    
    Form parameters:
        md5: MD5 hash of the index to delete.
        
    Returns:
        JSON response indicating success or failure.
    """
    try:
        path = request.form.get("md5")
        rag.delete_index(path)
        return jsonify({"code": 0, "message": "success"})
    except Exception:
        traceback.print_exc()
        return jsonify({"code": -1, "message": "failed"})


def get_bearer_token(request):
    """
    Helper function to extract the Bearer token from the Authorization header.
    
    Args:
        request: The Flask request object.
        
    Returns:
        The bearer token if present, None otherwise.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    return None


# =================== COMFYUI INTEGRATION ENDPOINTS ===================
# Endpoints for managing ComfyUI custom nodes and packages


@app.post("/api/comfyUi/areCustomNodesLoaded")
@app.input(ComfyUICustomNodesDownloadRequest.Schema, location='json', arg_name='comfyNodeRequest')
def are_custom_nodes_installed(comfyNodeRequest: ComfyUICustomNodesDownloadRequest):
    """
    Endpoint to check if specified ComfyUI custom nodes are installed.
    
    Args:
        comfyNodeRequest: Request body containing custom nodes to check.
        
    Returns:
        JSON response indicating installation status for each custom node.
    """
    response = { f"{x.username}/{x.repoName}" : comfyui_downloader.is_custom_node_installed_with_git_ref(x) for x in comfyNodeRequest.data}
    return jsonify(response)


@app.post("/api/comfyUi/loadCustomNodes")
@app.input(ComfyUICustomNodesDownloadRequest.Schema, location='json', arg_name='comfyNodeRequest')
def install_custom_nodes(comfyNodeRequest: ComfyUICustomNodesDownloadRequest):
    """
    Endpoint to install ComfyUI custom nodes.
    
    Only installs nodes that are not already installed.
    
    Args:
        comfyNodeRequest: Request body containing custom nodes to install.
        
    Returns:
        JSON response with installation results for each node.
    """
    try:
        nodes_to_be_installed = [x for x in comfyNodeRequest.data if not comfyui_downloader.is_custom_node_installed_with_git_ref(x)]
        installation_result = [ {"node": f"{x.username}/{x.repoName}", "success": comfyui_downloader.download_custom_node(x)} for x in nodes_to_be_installed ]
        logging.info(f"custom node installation request result: {installation_result}")
        return jsonify(installation_result)
    except Exception as e:
        return jsonify({'error_message': f'failed to install at least one custom node due to {e}'}), 501


@app.post("/api/comfyUi/installPythonPackage")
@app.input(ComfyUIPackageInstallRequest.Schema, location='json', arg_name='comfyPackageInstallRequest')
def install_python_packages_for_comfy(comfyPackageInstallRequest: ComfyUIPackageInstallRequest):
    """
    Endpoint to install Python packages required by ComfyUI.
    
    Args:
        comfyPackageInstallRequest: Request body containing packages to install.
        
    Returns:
        JSON response with installation results for each package.
    """
    try:
        for package in comfyPackageInstallRequest.data:
            comfyui_downloader.install_pypi_package(package)
        return jsonify({ f"{package}" : {"success": True, "errorMessage": ""} for x in comfyPackageInstallRequest.data})
    except Exception as e:
        return jsonify({'error_message': f'failed to at least one package due to {e}'}), 501


@app.post("/api/comfyUi/checkWorkflowRequirements")
@app.input(ComfyUICheckWorkflowRequirementRequest.Schema , location='json', arg_name='comfyRequirementRequest')
def check_workflow_requirements(comfyRequirementRequest: ComfyUICheckWorkflowRequirementRequest):
    """
    Endpoint to check if ComfyUI workflow requirements are met.
    
    Checks if required custom nodes and Python packages are installed.
    
    Args:
        comfyRequirementRequest: Request body containing requirements to check.
        
    Returns:
        JSON response indicating whether installation is needed.
    """
    try:
        nodes_to_be_installed = [not comfyui_downloader.is_custom_node_installed_with_git_ref(x) for x in comfyRequirementRequest.customNodes]
        packages_to_be_installed = [not comfyui_downloader.is_package_installed(x) for x in comfyRequirementRequest.pythonPackages]
        needs_installation = any(nodes_to_be_installed) or any(packages_to_be_installed)
        return jsonify({'needsInstallation' : needs_installation})
    except Exception as e:
        return jsonify({'errorMessage': f'failed to check for installation {e}'}), 500


# =================== HELPER FUNCTIONS ===================
# Utility functions used by the API endpoints


def cache_input_image():
    """
    Helper function to cache an input image for Stable Diffusion.
    
    Saves the uploaded file to disk with a timestamp-based filename.
    
    Returns:
        The path to the cached image file.
    """
    file = request.files.get("image")
    ext = ".png"
    if file.content_type == "image/jpeg":
        ext = ".jpg"
    elif file.content_type == "image/gif":
        ext = ".gif"
    elif file.content_type == "image/bmp":
        ext = ".bmp"
    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    file_path = os.path.abspath(
        os.path.join("./static/sd_input/", folder, base_name + ext)
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    utils.cache_file(file_path, file.__sizeof__())
    file.stream.close()
    return file_path


def cache_mask_image():
    """
    Helper function to cache a mask image for inpainting.
    
    Processes the uploaded mask image and saves it to disk with a timestamp-based filename.
    
    Returns:
        The path to the cached mask image file.
    """
    mask_width = request.form.get("mask_width", default=512, type=int)
    mask_height = request.form.get("mask_height", default=512, type=int)
    mask_image = utils.generate_mask_image(
        request.files.get("mask_image").stream.read(), mask_width, mask_height
    )
    now = datetime.now()
    folder = now.strftime("%d_%m_%Y")
    base_name = now.strftime("%H%M%S")
    file_path = os.path.abspath(
        os.path.join("static/sd_mask/", folder, base_name + ".png")
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    mask_image.save(file_path)
    utils.cache_file(file_path, os.path.getsize(file_path))
    return file_path


# =================== APPLICATION ENTRY POINT ===================
# Main execution block for running the Flask application


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Playground Web service")
    parser.add_argument("--port", type=int, default=59999, help="Service listen port")
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port)
