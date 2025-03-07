"""
Stable Diffusion Image Generation Module
--------------------------------------
This module provides functionality for generating and manipulating images using Stable Diffusion models.

It supports multiple generation modes:
- Text-to-Image: Generate images from text prompts
- Image-to-Image: Transform existing images based on text prompts
- Inpainting: Fill in masked regions of images
- Outpainting: Extend images beyond their original boundaries
- Upscaling: Increase image resolution with RealESRGAN

The module handles model loading, parameter management, callback processing, and various image manipulation tasks
with optimization for Intel XPU hardware.
"""

import gc
import os
import queue
import random
import time
from typing import Any, Callable, Dict, List
import aipg_utils as utils
import service_config
import inpaint_utils
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    AutoPipelineForInpainting,
    AutoPipelineForImage2Image,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    LCMScheduler,
    AutoencoderTiny,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
import torch
from PIL import Image
from realesrgan import RealESRGANer
import re
import schedulers_util
from compel import Compel
from threading import Event
from xpu_hijacks import ipex_hijacks

# Apply Intel XPU (GPU) hijacks to make PyTorch operations work on Intel GPUs
ipex_hijacks()
print("workarounds applied")


# region class define


class TextImageParams:
    """
    Base parameters class for text-to-image generation.
    
    Attributes:
        device: GPU device ID to use for generation
        prompt: Text prompt describing the desired image
        model_name: Name of the model to use for generation
        mode: Generation mode (0=text2img, 1=upscale, 2=img2img, 3=inpaint, 4=outpaint)
        width: Output image width in pixels
        height: Output image height in pixels
        generate_number: Number of images to generate
        seed: Random seed for reproducibility (-1 for random)
        guidance_scale: How closely to follow the prompt (higher = more faithful)
        inference_steps: Number of denoising steps (more = higher quality but slower)
        negative_prompt: Text describing what to avoid in the image
        lora: LoRA adapter name to use with the model
        scheduler: Name of the diffusion scheduler to use
        image_preview: Whether to enable preview during generation (0=off, 1=on)
        safe_check: Whether to enable safety checking (0=off, 1=on)
    """
    device: int
    prompt: str
    model_name: str
    mode: int
    """
    0-text to image 1-upscale 2-image to image 3-inpaint 4-outpaint
    """
    width: int
    height: int
    generate_number: int
    seed: int
    guidance_scale: int
    inference_steps: int
    negative_prompt: str
    lora: str
    scheduler: str
    image_preview: int
    safe_check: int


class ImageToImageParams(TextImageParams):
    """
    Parameters for image-to-image generation.
    
    Extends TextImageParams with image source and denoising strength.
    
    Attributes:
        image: Path to the source image
        denoise: Strength of transformation (0.0-1.0, higher = more transformation)
    """
    image: str
    denoise: float


class UpscaleImageParams(ImageToImageParams):
    """
    Parameters for image upscaling.
    
    Extends ImageToImageParams with scaling factor.
    
    Attributes:
        scale: Factor by which to upscale the image
    """
    scale: float


class InpaintParams(ImageToImageParams):
    """
    Parameters for image inpainting.
    
    Extends ImageToImageParams with a mask image that defines the area to inpaint.
    
    Attributes:
        mask_image: Path to the mask image (white areas will be inpainted)
    """
    mask_image: str


class OutpaintParams(ImageToImageParams):
    """
    Parameters for image outpainting.
    
    Extends ImageToImageParams with a direction to extend the image.
    
    Attributes:
        direction: Direction to extend the image ("left", "right", "up", "down")
    """
    direction: str


class StopGenerateException(Exception):
    """
    Exception raised when image generation is stopped by user request.
    """
    def __str__(self):
        return "user stop generate image"


class NoWatermark:
    """
    Dummy watermark class to replace the default watermarking in Stable Diffusion.
    
    Used to disable watermarking on generated images.
    """
    def apply_watermark(self, img):
        return img


# endregion


# region global variable

# Main model pipeline for basic image generation tasks
_basic_model_pipe: StableDiffusionPipeline | StableDiffusionXLPipeline = None
# Extended model pipeline for specialized tasks (img2img, inpainting, etc.)
_ext_model_pipe: (
        StableDiffusionPipeline
        | StableDiffusionXLPipeline
        | StableDiffusionImg2ImgPipeline
        | StableDiffusionXLImg2ImgPipeline
        | StableDiffusionInpaintPipeline
        | StableDiffusionXLInpaintPipeline
) = None
# RealESRGAN super-resolution model instance for upscaling
_realESRGANer: RealESRGANer = None
# Track the last used generation mode to avoid unnecessary reloading
_last_mode: int = None
# Track the last model name to avoid unnecessary reloading
_last_model_name: str = None
# Current generation index within a batch
_generate_idx: int
# Default scheduler instance (used as fallback)
_default_scheduler: LCMScheduler = None
# Track the last used LoRA to avoid unnecessary reloading
_last_lora: str = "None"
# Track the last used scheduler to avoid unnecessary reloading
_last_scheduler: str = "None"
# Callback for model loading progress events
load_model_callback: Callable[[str], None] = None
# Callback for model component loading events
load_model_components_callback: Callable[[str], None] = None
# Callback for download progress events
download_progress_callback: Callable[[str, int, int, int], None] = (None,)
# Callback for download completion events
download_completed_callback: Callable[[str, Exception], None] = (None,)
# Callback for step completion during generation
step_end_callback: Callable[[int, int, int, Image.Image | None], None] = None
# Callback for generated image output
image_out_callback: Callable[[int, Image.Image, Any], None] = None
# Tiny autoencoder for efficient preview generation
_taesd_vae: AutoencoderTiny = None
# Type of tiny autoencoder currently loaded (sd1.5 or sdxl)
_taesd_vae_type: str = None
# Whether preview generation is enabled (0=off, 1=on)
_preview_enabled = 0
# Flag to signal generation stopping
_stop_generate = False
# Flag to indicate active generation
_generating = False
# Event for synchronizing stop requests
_stop_event = Event()
# Queue for preview images
_preview_queue = queue.Queue()
# Safety checker instance for content filtering
_safety_checker: StableDiffusionSafetyChecker = None


# endregion


# region load model


def get_basic_model(input_model_name: str) -> DiffusionPipeline | Any:
    """
    Load or retrieve the basic diffusion model pipeline.
    
    This function manages loading and caching of the primary diffusion model.
    If the requested model is already loaded, it returns the cached instance.
    Otherwise, it loads the model from disk, configures it, and caches it for future use.
    
    Args:
        input_model_name: The model name in format "config_key:model_name"
        
    Returns:
        A configured diffusion pipeline ready for use
        
    Raises:
        Exception: If the model cannot be found or loaded
        StopGenerateException: If loading is interrupted by user
    """
    global \
        _last_model_name, \
        _basic_model_pipe, \
        load_model_callback, \
        _last_lora, \
        _last_scheduler, \
        _taesd_vae, \
        _safety_checker

    assert_stop_generate()

    if _last_model_name == input_model_name and _basic_model_pipe is not None:
        return _basic_model_pipe

    mode_name_array = input_model_name.split(":")
    config_key = mode_name_array[0]
    model_name = mode_name_array[1]
    model_base_path = service_config.service_model_paths.get(config_key)

    start = time.time()
    if load_model_callback is not None:
        load_model_callback("start")

    if utils.is_single_file(model_name):
        # single_file_mode
        model_path = os.path.abspath(os.path.join(model_base_path, model_name))

        if not os.path.exists(model_path):
            raise Exception(f'can not find model "{model_name}"', model_path)

        _basic_model_pipe = load_model_from_single_file(model_path)
    else:
        model_floder = model_name.replace("/", "---")
        model_path = os.path.abspath(
            os.path.join(model_base_path, model_floder, "model_index.json")
        )
        if not os.path.exists(model_path):
            raise Exception(f'can not find model "{model_name}"', model_path)

        _basic_model_pipe = load_model_from_pretrained(
            os.path.abspath(os.path.join(model_base_path, model_floder))
        )

    _last_lora = "None"
    _last_scheduler = "None"

    assert_stop_generate()

    _basic_model_pipe.watermark = NoWatermark()
    if isinstance(_basic_model_pipe, StableDiffusionPipeline):
        _safety_checker = _basic_model_pipe.safety_checker
        _basic_model_pipe.safety_checker = None
    else:
        _safety_checker = None
    # perf optimization
    _basic_model_pipe.enable_model_cpu_offload()
    _basic_model_pipe.enable_vae_tiling()
    _basic_model_pipe.to(service_config.device)

    print(
        "load model {} finish. cost {}s".format(
            model_name, round(time.time() - start, 3)
        )
    )

    if load_model_callback is not None:
        load_model_callback("finish")
    _last_model_name = input_model_name

    print(_basic_model_pipe)
    return _basic_model_pipe


def process_preview_taesd():
    """
    Initialize or reload the tiny autoencoder for generation previews.
    
    Loads the appropriate tiny autoencoder model (TAESD) based on the current 
    diffusion model type (SD1.5 or SDXL) to enable efficient generation previews.
    The TAESD provides fast approximate decoding of latent space for previews.
    """
    global _taesd_vae_type, _taesd_vae

    if isinstance(
            _basic_model_pipe,
            StableDiffusionXLPipeline
            | StableDiffusionXLImg2ImgPipeline
            | StableDiffusionXLInpaintPipeline,
    ) and (_taesd_vae_type != "sdxl" or _taesd_vae is None):
        _taesd_vae = AutoencoderTiny.from_pretrained(
            os.path.join(service_config.service_model_paths.get("preview"), "madebyollin---taesdxl"),
            torch_dtype=torch.bfloat16,
        )
        _taesd_vae_type = "sdxl"
    elif isinstance(
            _basic_model_pipe,
            StableDiffusionPipeline
            | StableDiffusionImg2ImgPipeline
            | StableDiffusionInpaintPipeline,
    ) and (_taesd_vae_type != "sd1.5" or _taesd_vae is None):
        _taesd_vae = AutoencoderTiny.from_pretrained(
            os.path.join(service_config.service_model_paths.get("preview"), "madebyollin---taesd"),
            torch_dtype=torch.bfloat16,
        )
        _taesd_vae_type = "sd1.5"

    _taesd_vae.to(service_config.device)


def get_ext_pipe(params: TextImageParams, pipe_classes: List, init_class: any):
    """
    Get or initialize an extended pipeline for specialized tasks.
    
    Creates or reuses an extended pipeline for specific generation tasks like 
    img2img, inpainting, or outpainting. If a suitable pipeline is already 
    loaded, it will be reused; otherwise, a new one will be created.
    
    Args:
        params: Generation parameters
        pipe_classes: List of valid pipeline classes for the task
        init_class: Class to use for initializing a new pipeline
        
    Returns:
        Configured pipeline for the specified task
        
    Raises:
        StopGenerateException: If initialization is interrupted by user
    """
    global _basic_model_pipe, _ext_model_pipe

    if _ext_model_pipe is not None:
        for cls in pipe_classes:
            if isinstance(_ext_model_pipe, cls):
                return _ext_model_pipe
        del _ext_model_pipe
        gc.collect()
        torch.xpu.empty_cache()

    basic_model_pipe = get_basic_model(params.model_name)
    _ext_model_pipe = init_class.from_pipe(basic_model_pipe)
    _ext_model_pipe.to(service_config.device)

    assert_stop_generate()

    return _ext_model_pipe


def load_model_from_single_file(model_signle_file: str):
    """
    Load a diffusion model from a single file (safetensors or ckpt).
    
    Attempts to load either an SDXL or SD1.5 model based on the filename,
    falling back to the other format if the initial attempt fails.
    
    Args:
        model_signle_file: Path to the model file
        
    Returns:
        Loaded diffusion pipeline
    """
    base_name = os.path.basename(model_signle_file)
    is_xl = re.search("[-_]xl[-_\.]", base_name, flags=re.I) is not None
    if is_xl:
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_signle_file, torch_dtype=torch.bfloat16
            )

        except Exception:
            pipe = StableDiffusionPipeline.from_single_file(
                model_signle_file, torch_dtype=torch.bfloat16
            )
    else:
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                model_signle_file, torch_dtype=torch.bfloat16
            )
        except Exception:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_signle_file, torch_dtype=torch.bfloat16
            )
    return pipe


def load_model_from_pretrained(model_dir: str):
    """
    Load a diffusion model from a directory of model components.
    
    Detects and loads the appropriate model precision (fp16 or fp32)
    based on available model files.
    
    Args:
        model_dir: Directory containing the model files
        
    Returns:
        Loaded diffusion pipeline
    """
    if os.path.exists(
            os.path.join(model_dir, "unet/diffusion_pytorch_model.fp32.safetensors")
    ) or os.path.exists(
        os.path.join(model_dir, "unet/diffusion_pytorch_model.fp32.bin")
    ):
        pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            variant="fp32",
            device=service_config.device,
        )
    elif os.path.exists(
            os.path.join(model_dir, "unet/diffusion_pytorch_model.fp16.safetensors")
    ) or os.path.exists(
        os.path.join(model_dir, "unet/diffusion_pytorch_model.fp16.bin")
    ):
        pipe = DiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, variant="fp16"
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    return pipe


def set_lora(pipe: StableDiffusionPipeline | StableDiffusionXLPipeline, lora: str):
    """
    Apply LoRA (Low-Rank Adaptation) weights to a diffusion model.
    
    Loads and applies LoRA weights to customize model behavior.
    Caches the last used LoRA to avoid unnecessary reloading.
    
    Args:
        pipe: The diffusion pipeline to modify
        lora: Name of the LoRA adapter to apply, or "None" to remove
    
    Raises:
        Exception: If the specified LoRA cannot be found
    """
    global \
        _default_scheduler, \
        _last_lora, \
        download_progress_callback, \
        download_completed_callback
    if lora == _last_lora:
        return
    if lora != "None":
        base_path = service_config.service_model_paths.get("lora")

        if utils.is_single_file(lora):
            lora_path = os.path.join(base_path, lora)
            if not os.path.exists(lora_path):
                raise Exception(f"not found lora {lora} in dir {base_path}")
            pipe.load_lora_weights(base_path, weight_name=lora, low_cpu_mem_usage=True)
        else:
            lora_path = os.path.join(base_path, lora.replace("/", "---"))
            if not os.path.exists(lora_path):
                raise Exception(f"not found lora {lora} in dir {base_path}")
            pipe.load_lora_weights(lora_path)

        print(f"Loaded LORA weights from {lora}.")
    elif _last_lora is not None:
        pipe.unload_lora_weights()
        print("Restored the default LORA.")
    _last_lora = lora


def set_scheduler(
        pipe: StableDiffusionPipeline | StableDiffusionXLPipeline, scheduler_name: str
):
    """
    Set the scheduler for a diffusion pipeline.
    
    Changes the noise scheduler used during the diffusion process,
    which affects the image generation quality and characteristics.
    
    Args:
        pipe: The diffusion pipeline to modify
        scheduler_name: Name of the scheduler to use
    """
    global _last_scheduler
    schedulers_util.set_scheduler(pipe, scheduler_name)
    _last_scheduler = scheduler_name


def set_components(
        pipe: (
                StableDiffusionPipeline
                | StableDiffusionXLPipeline
                | StableDiffusionInpaintPipeline
                | StableDiffusionXLInpaintPipeline
        ),
        params: TextImageParams,
):
    """
    Set up the components for a diffusion pipeline based on generation parameters.
    
    Configures the pipeline with the appropriate scheduler, LoRA, safety checker,
    and preview functionality based on the provided parameters.
    
    Args:
        pipe: The diffusion pipeline to configure
        params: Generation parameters to apply
        
    Raises:
        StopGenerateException: If configuration is interrupted by user
    """
    global \
        _last_scheduler, \
        _last_lora, \
        load_model_components_callback, \
        _taesd_vae, \
        _safety_checker

    if load_model_components_callback is not None:
        load_model_components_callback("start")

    if params.image_preview == 1:
        process_preview_taesd()

    if params.safe_check and isinstance(
            pipe, StableDiffusionPipeline | StableDiffusionInpaintPipeline
    ):
        pipe.safety_checker = _safety_checker
    else:
        pipe.safety_checker = None

    if params.scheduler != _last_scheduler:
        set_scheduler(pipe, params.scheduler)
        _last_scheduler = params.scheduler
    assert_stop_generate()
    if params.lora != _last_lora:
        set_lora(pipe, params.lora)
    assert_stop_generate()

    if load_model_components_callback is not None:
        load_model_components_callback("finish")


def get_ESRGANer():
    """
    Get or initialize the RealESRGAN super-resolution model.
    
    Lazy-loads the RealESRGAN model for image upscaling.
    
    Returns:
        Configured RealESRGANer instance ready for upscaling
    """
    global _realESRGANer
    if _realESRGANer is None:
        _realESRGANer = RealESRGANer()
    _realESRGANer.to(service_config.device)
    return _realESRGANer


def convert_prompt_to_compel_format(prompt):
    """
    Convert prompt text to the format expected by Compel.
    
    Transforms common attention weight formats into the syntax used by Compel:
    - (word:1.2) becomes (word)1.2
    - [word] becomes (word)0.909090909
    - [word:1.2] becomes (word)0.9
    
    Args:
        prompt: Original prompt text
        
    Returns:
        Converted prompt compatible with Compel
    """
    # convert prompt to compel supported prompt weighting format
    converted = re.sub(r"\(([^:]+):([\d.]+)\)", r"(\1)\2", prompt)
    converted = re.sub(r"\[([^:\]]+)\]", r"(\1)0.909090909", converted)
    converted = re.sub(r"\[([^:]+):[\d.]+\]", r"(\1)0.9", converted)
    return converted


# endregion


# region preview


def __callback_on_step_end__(
        model: (
                StableDiffusionPipeline
                | StableDiffusionXLPipeline
                | StableDiffusionInpaintPipeline
                | StableDiffusionXLInpaintPipeline
        ),
        step: int,
        timesteps: int,
        callback_kwargs: Dict,
):
    """
    Callback function called at the end of each diffusion step.
    
    Handles progress reporting and preview image generation during the diffusion process.
    If preview generation is enabled, it uses the tiny autoencoder to create
    approximate previews of the current image state every few steps.
    
    Args:
        model: The diffusion model being used
        step: Current step number
        timesteps: Total number of timesteps
        callback_kwargs: Dictionary containing step data, including latents
        
    Returns:
        The unchanged callback_kwargs dictionary
        
    Raises:
        StopGenerateException: If generation is interrupted by user
    """
    global \
        step_end_callback, \
        _generate_idx, \
        _preview_enabled, \
        _taesd_vae, \
        _preview_queue

    assert_stop_generate()

    if step_end_callback is not None:
        if _preview_enabled == 1:
            latents: torch.FloatTensor = callback_kwargs["latents"]
            # put preiview task to preview thread task queue
            if step % 4 == 0:
                with torch.no_grad():
                    image = _taesd_vae.decode(latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
                    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                    # convert to PIL Images
                    image = model.numpy_to_pil(image)
                    step_end_callback(
                        _generate_idx,
                        step,
                        model.num_timesteps,
                        _preview_enabled,
                        image[0],
                    )
            else:
                step_end_callback(
                    _generate_idx, step, model.num_timesteps, _preview_enabled, None
                )

        else:
            step_end_callback(
                _generate_idx, step, model.num_timesteps, _preview_enabled, None
            )

    return callback_kwargs


# endregion


# region generate_image_function


def convet_compel_prompt(
        prompt: str, pipe: StableDiffusionPipeline | StableDiffusionXLPipeline
):
    """
    Process text prompt using Compel for improved text conditioning.
    
    Creates text embeddings for model input using Compel, which provides 
    improved control over text prompt weighting and emphasis.
    
    Args:
        prompt: The text prompt to process
        pipe: The diffusion pipeline that will use the embeddings
        
    Returns:
        Dictionary of inputs for the pipeline containing processed prompt embeddings
    """
    custom_inputs = {}

    if hasattr(pipe, "text_encoder_2") and hasattr(pipe, "tokenizer_2"):
        custom_inputs.update({"prompt": prompt})
        # compel_proc2 = Compel(
        #     tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        #     text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        #     returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        #     requires_pooled=[False, True],
        # )
        # compel_prompt2 = convert_prompt_to_compel_format(prompt)
        # pooled_prompt = [compel_prompt2, ""]
        # prompt_embeds, pooled_prompt_embeds = compel_proc2(pooled_prompt)
        # custom_inputs.update(
        #     {
        #         "prompt_embeds": prompt_embeds,
        #         "pooled_prompt_embeds": pooled_prompt_embeds,
        #     }
        # )
    else:
        compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
        compel_prompt = convert_prompt_to_compel_format(prompt)
        prompt_embeds = compel_proc(compel_prompt)
        custom_inputs.update(
            {
                "prompt_embeds": prompt_embeds,
            }
        )

    return custom_inputs


def text_to_image(
        params: TextImageParams,
):
    """
    Generate images from text prompts.
    
    The core text-to-image generation function that:
    1. Loads the appropriate model
    2. Processes the text prompt with Compel
    3. Runs diffusion with the specified parameters
    4. Outputs the generated images
    
    Supports batch generation with different seeds for each image.
    
    Args:
        params: Text-to-image generation parameters
    """
    global _generate_idx, image_out_callback
    pipe = get_basic_model(params.model_name)
    set_components(pipe, params)
    pipe.to(service_config.device)

    custom_inputs = convet_compel_prompt(params.prompt, pipe)
    seed = params.seed

    _generate_idx = 0

    with torch.inference_mode():
        while _generate_idx < params.generate_number:
            params.seed = (
                random.randint(0, 0xFFFFFFFE)
                if seed == -1
                else seed + _generate_idx & 0xFFFFFFFF
            )
            params.seed = 0 if params.seed == 0xFFFFFFFF else params.seed
            generator = torch.Generator("cpu").manual_seed(params.seed)

            image = pipe(
                width=params.width,
                height=params.height,
                generator=generator,
                num_inference_steps=params.inference_steps,
                guidance_scale=params.guidance_scale,
                negative_prompt=params.negative_prompt,
                callback_on_step_end=__callback_on_step_end__,
                **custom_inputs,
            ).images[0]

            output_image(pipe, image, params)
            _generate_idx += 1


def image_to_image(params: ImageToImageParams):
    """
    Transform an existing image using text prompts.
    
    Performs image-to-image generation by:
    1. Loading the appropriate specialized pipeline
    2. Processing the input image
    3. Running diffusion with the text prompt to transform the image
    4. Outputting the generated images
    
    The denoise parameter controls how much of the original image is preserved.
    
    Args:
        params: Image-to-image generation parameters
    """
    global _generate_idx, image_out_callback
    pipe = get_ext_pipe(
        params,
        [StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
        AutoPipelineForImage2Image,
    )

    set_components(pipe, params)
    pipe.to(service_config.device)
    input_image = Image.open(params.image)
    input_image = (
        input_image.convert("RGB") if input_image.mode != "RGB" else input_image
    )
    if input_image.width != params.width or input_image.height != params.height:
        input_image = input_image.resize((params.width, params.height))

    seed = params.seed

    _generate_idx = 0
    custom_inputs = convet_compel_prompt(params.prompt, pipe)
    seed = params.seed
    with torch.inference_mode():
        while _generate_idx < params.generate_number:
            params.seed = (
                random.randint(0, 0xFFFFFFFE)
                if seed == -1
                else seed + _generate_idx & 0xFFFFFFFF
            )
            params.seed = 0 if params.seed == 0xFFFFFFFF else params.seed
            generator = torch.Generator("cpu").manual_seed(params.seed)

            image = pipe(
                image=input_image,
                width=params.width,
                height=params.height,
                generator=generator,
                strength=params.denoise,
                guidance_scale=params.guidance_scale,
                num_inference_steps=params.inference_steps,
                negative_prompt=params.negative_prompt,
                callback_on_step_end=__callback_on_step_end__,
                **custom_inputs,
            ).images[0]

            output_image(pipe, image, params)

            _generate_idx += 1


def upscale(params: UpscaleImageParams):
    """
    Upscale an image to a higher resolution.
    
    Offers two upscaling modes:
    1. Pure RealESRGAN upscaling (when denoise ≤ 0.1)
    2. Diffusion-enhanced upscaling (when denoise > 0.1)
    
    The second mode combines Stable Diffusion refinement with RealESRGAN
    to produce high-quality upscaled images with enhanced details.
    
    Args:
        params: Upscaling parameters including scale factor and denoise strength
    """
    global image_out_callback, _generate_idx, _ext_model_pipe

    input_image = Image.open(params.image)

    input_image = (
        input_image.convert("RGB") if input_image.mode != "RGB" else input_image
    )

    if params.denoise <= 0.1:
        out_image = Image.fromarray(
            get_ESRGANer().enhance(input_image, params.scale)[0]
        )
        if image_out_callback is not None:
            image_out_callback(0, out_image, params)
    else:
        pipe = get_ext_pipe(
            params,
            [StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline],
            AutoPipelineForImage2Image,
        )
        set_components(pipe, params)
        pipe.to(service_config.device)

        custom_inputs = convet_compel_prompt(params.prompt, pipe)
        seed = params.seed
        _generate_idx = 0
        with torch.inference_mode():
            # while _generate_idx < params.generate_number:
            #     params.seed = (
            #         random.randint(0, 0xFFFFFFFE)
            #         if seed == -1
            #         else seed + _generate_idx & 0xFFFFFFFF
            #     )
            params.seed = 0 if params.seed == 0xFFFFFFFF else params.seed
            generator = torch.Generator("cpu").manual_seed(seed)
            out_image = pipe(
                image=input_image,
                width=input_image.width,
                height=input_image.height,
                generator=generator,
                strength=params.denoise,
                guidance_scale=params.guidance_scale,
                num_inference_steps=params.inference_steps,
                negative_prompt=params.negative_prompt,
                callback_on_step_end=__callback_on_step_end__,
                **custom_inputs,
            ).images[0]
            out_image = Image.fromarray(
                get_ESRGANer().enhance(out_image, params.scale)[0]
            )
            params.width = out_image.width
            params.height = out_image.height
            output_image(pipe, out_image, params)
            # _generate_idx += 1


def inpaint(params: InpaintParams):
    """
    Fill in masked regions of an image using text prompts.
    
    Performs inpainting by:
    1. Loading a specialized inpainting pipeline
    2. Processing the input image and mask
    3. Slicing the image to focus on the masked area
    4. Running diffusion to generate content in the masked region
    5. Blending the new content with the original image
    
    The mask defines which areas will be regenerated (white areas in the mask).
    
    Args:
        params: Inpainting parameters including image path, mask path, and prompt
    """
    global _generate_idx, image_out_callback

    pipe = get_ext_pipe(
        params,
        [StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline],
        AutoPipelineForInpainting,
    )

    set_components(pipe, params)
    pipe.to(service_config.device)

    input_image = Image.open(params.image)
    mask_image = Image.open(params.mask_image)
    input_image = (
        input_image.convert("RGB") if input_image.mode != "RGB" else input_image
    )
    mask_image = mask_image.convert("RGB") if mask_image.mode != "RGB" else mask_image

    slice_image, mask_image, slice_box = inpaint_utils.pre_input_and_mask(
        input_image, mask_image
    )

    slice_w, slice_h = slice_image.size
    out_width, out_height, out_radio = inpaint_utils.calc_out_size(
        slice_w, slice_h, isinstance(pipe, StableDiffusionXLInpaintPipeline)
    )
    if out_radio != 1:
        slice_image = slice_image.resize((out_width, out_height))
        mask_image = mask_image.resize((out_width, out_height))

    mask_image = pipe.mask_processor.blur(mask_image, blur_factor=33)
    seed = params.seed
    _generate_idx = 0

    custom_inputs = convet_compel_prompt(params.prompt, pipe)
    with torch.inference_mode():
        while _generate_idx < params.generate_number:
            params.seed = (
                random.randint(0, 0xFFFFFFFE)
                if seed == -1
                else seed + _generate_idx & 0xFFFFFFFF
            )
            params.seed = 0 if params.seed == 0xFFFFFFFF else params.seed
            generator = torch.Generator("cpu").manual_seed(params.seed)

            repainted_image: Image.Image = pipe(
                **custom_inputs,
                image=slice_image,
                mask_image=mask_image,
                strength=params.denoise,
                width=out_width,
                height=out_height,
                generator=generator,
                guidance_scale=params.guidance_scale,
                num_inference_steps=params.inference_steps,
                negative_prompt=params.negative_prompt,
                callback_on_step_end=__callback_on_step_end__,
                force_unmasked_unchanged=True,
            ).images[0]

            gen_image = pipe.image_processor.apply_overlay(
                mask_image, slice_image, repainted_image
            )

            if out_radio != 1:
                realESRGANer = get_ESRGANer()
                gen_image = Image.fromarray(
                    realESRGANer.enhance(gen_image, out_radio)[0]
                )

            slice_width = slice_box[2] - slice_box[0]
            slice_height = slice_box[3] - slice_box[1]
            if gen_image.height != slice_width or gen_image.width != slice_height:
                gen_image = gen_image.resize((slice_width, slice_height))

            input_image.paste(gen_image, slice_box)

            output_image(pipe, input_image, params)
            _generate_idx += 1


def outpaint(params: OutpaintParams):
    """
    Extend an image beyond its original boundaries using text prompts.
    
    Performs outpainting by:
    1. Loading a specialized inpainting pipeline
    2. Expanding the original image with transparent/blank areas
    3. Creating a mask for the expanded areas
    4. Running diffusion to generate content in the expanded areas
    5. Blending the new content with the original image
    
    The direction parameter determines which side to expand the image.
    
    Args:
        params: Outpainting parameters including image path, direction, and prompt
    """
    from outpaint_utils import preprocess_outpaint

    global _generate_idx, image_out_callback

    pipe = get_ext_pipe(
        params,
        [StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline],
        AutoPipelineForInpainting,
    )
    set_components(pipe, params)

    pipe.to(service_config.device)
    if isinstance(pipe, StableDiffusionXLInpaintPipeline):
        max_size = 1536
    else:
        max_size = 768

    ori_image = Image.open(params.image)

    if ori_image.mode != "RGB":
        ori_image = ori_image.convert("RGB")

    new_width = inpaint_utils.make_multiple_of_8(ori_image.width)
    new_height = inpaint_utils.make_multiple_of_8(ori_image.height)
    if new_width != ori_image.width or new_height != ori_image.width:
        ori_image = ori_image.resize((new_width, new_height))

    expand_image, inpaint_mask = preprocess_outpaint(params.direction, ori_image)

    inpaint_image, scale_ratio = inpaint_utils.resize_by_max(expand_image, max_size)
    out_width = inpaint_utils.make_multiple_of_8(inpaint_image.width)
    out_height = inpaint_utils.make_multiple_of_8(inpaint_image.height)

    if out_width != inpaint_image.width or out_height != inpaint_image.height:
        inpaint_image = inpaint_image.resize((out_width, out_height))

    inpaint_mask = inpaint_mask.resize((out_width, out_height))

    seed = params.seed
    _generate_idx = 0
    custom_inputs = convet_compel_prompt(params.prompt, pipe)
    with torch.inference_mode():
        while _generate_idx < params.generate_number:
            params.seed = (
                random.randint(0, 0xFFFFFFFE)
                if seed == -1
                else seed + _generate_idx & 0xFFFFFFFF
            )
            params.seed = 0 if params.seed == 0xFFFFFFFF else params.seed
            generator = torch.Generator("cpu").manual_seed(params.seed)
            repainted_image: Image.Image = pipe(
                **custom_inputs,
                image=inpaint_image,
                mask_image=inpaint_mask,
                strength=params.denoise,
                width=out_width,
                height=out_height,
                generator=generator,
                guidance_scale=params.guidance_scale,
                num_inference_steps=params.inference_steps,
                negative_prompt=params.negative_prompt,
                callback_on_step_end=__callback_on_step_end__,
                force_unmasked_unchanged=True,
            ).images[0]

            unmasked_unchanged_image = pipe.image_processor.apply_overlay(
                inpaint_mask, inpaint_image, repainted_image
            )

            if scale_ratio != 1:
                unmasked_unchanged_image = Image.fromarray(
                    get_ESRGANer().enhance(unmasked_unchanged_image, scale_ratio)[0]
                )

            output_image(pipe, unmasked_unchanged_image, params)
            _generate_idx += 1


def is_image_completely_black(image: Image):
    """
    Check if an image is entirely black.
    
    Used for safety checking to detect if an image was filtered out.
    
    Args:
        image: The PIL image to check
        
    Returns:
        True if the image is completely black, False otherwise
    """
    pixels = image.getdata()
    return all(pixel == (0, 0, 0) for pixel in pixels)


def output_image(
        pipe: StableDiffusionPipeline | StableDiffusionXLPipeline,
        image: Image.Image,
        params: TextImageParams,
):
    """
    Process and output a generated image.
    
    Handles safety checking and sends the image to the output callback.
    
    Args:
        pipe: The diffusion pipeline that generated the image
        image: The generated image
        params: The parameters used for generation
    """
    global image_out_callback, _safety_checker, _generate_idx
    passed_safety_check = not is_image_completely_black(image)
    if image_out_callback is not None:
        image_out_callback(_generate_idx, image, params, passed_safety_check)


def generate(params: TextImageParams):
    """
    Main entry point for image generation.
    
    Dispatches to the appropriate generation function based on the mode parameter:
    - Mode 0: Text-to-image
    - Mode 1: Upscale
    - Mode 2: Image-to-image
    - Mode 3: Inpaint
    - Mode 4: Outpaint
    
    Args:
        params: Generation parameters
    """
    global \
        _last_model_name, \
        _last_mode, \
        _basic_model_pipe, \
        _ext_model_pipe, \
        _realESRGANer, \
        _stop_generate, \
        _generating, \
        _preview_enabled

    try:
        stop_generate()
        torch.xpu.set_device(params.device)
        # service_config.device = f"xpu:{params.device}"
        if _last_model_name != params.model_name:
            # hange model dispose basic model
            if _basic_model_pipe is not None:
                dispose_basic_model()

        _preview_enabled = params.image_preview

        _stop_generate = False

        _generating = True

        print("receive params", vars(params))
        if params.mode == 1:
            upscale(params)
        elif params.mode == 2:
            image_to_image(params)
        elif params.mode == 3:
            inpaint(params)
        elif params.mode == 4:
            outpaint(params)
        else:
            text_to_image(params)
        _last_mode = params.mode

        torch.xpu.empty_cache()
    finally:
        _generating = False


# endregion


def dispose_basic_model():
    """
    Clean up the basic model pipeline resources.
    
    Releases memory used by the basic model pipeline, extended pipeline,
    and tiny autoencoder. Resets state variables and clears GPU cache.
    """
    global \
        _basic_model_pipe, \
        _ext_model_pipe, \
        _taesd_vae, \
        _last_lora, \
        _last_scheduler, \
        _last_mode, \
        _last_model_name

    stop_generate()

    if _ext_model_pipe is not None:
        del _ext_model_pipe
        _ext_model_pipe = None
    if _basic_model_pipe is not None:
        for key in _basic_model_pipe.components:
            del _basic_model_pipe.components[key]
        del _basic_model_pipe
        _basic_model_pipe = None
    if _taesd_vae is not None:
        del _taesd_vae
        _taesd_vae = None

    _last_lora = "None"
    _last_scheduler = "None"
    _last_model_name = None
    _last_mode = None

    gc.collect()
    torch.xpu.empty_cache()


def dispose_ext_model():
    """
    Clean up the extended model pipeline resources.
    
    Releases memory used by the extended model pipeline and clears GPU cache.
    """
    global _ext_model_pipe
    del _ext_model_pipe
    _ext_model_pipe = None
    gc.collect()
    torch.xpu.empty_cache()


def dispose():
    """
    Clean up all model resources.
    
    Releases memory used by the RealESRGAN model and all diffusion pipelines.
    Called when shutting down or needing to free all resources.
    """
    global _realESRGANer, _preview_thread
    if _realESRGANer is not None:
        del _realESRGANer
        _realESRGANer = None
    dispose_basic_model()


def stop_generate():
    """
    Stop any ongoing image generation process.
    
    Sets a flag to request generation stopping and waits for the process
    to acknowledge the stop request via an event.
    """
    global _stop_generate, _generating, _stop_event
    if _generating:
        _stop_generate = True
        _stop_event.clear()
        _stop_event.wait()
        _generating = False
        _stop_generate = False


def assert_stop_generate():
    """
    Check if generation should stop and raise an exception if so.
    
    Called at various points during generation to allow early termination.
    Signals that the stop was acknowledged by setting an event.
    
    Raises:
        StopGenerateException: If generation stop has been requested
    """
    global _stop_generate, _stop_event
    if _stop_generate:
        _stop_event.set()
        raise StopGenerateException()


def clear_xpu_cache():
    """
    Clear the GPU (XPU) memory cache.
    
    Utility function for manual memory management to free GPU memory.
    """
    torch.xpu.empty_cache()
