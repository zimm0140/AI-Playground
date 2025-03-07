"""
AI Playground Service Configuration Module
-----------------------------------------
This module contains configuration settings for the AI Playground service, defining:
- File paths for various model types
- Integration paths for ComfyUI
- External tool configurations
- Hardware settings

The paths are structured to maintain organization of different model types and to support
multiple AI backends including standard PyTorch models, ComfyUI, LlamaCPP, and OpenVINO.
"""

# Path to optional JSON config file (currently commented out)
# CONFIG_PATH = "./service_config.json"

# Main service model paths for the core AI Playground functionality
service_model_paths = {
    "llm": "./models/llm/checkpoints",              # Large Language Models
    "embedding": "./models/llm/embedding",          # Text embeddings for LLMs
    "stableDiffusion": "./models/stable_diffusion/checkpoints",  # SD model checkpoints
    "lora": "./models/stable_diffusion/lora",       # LoRA adapters for SD
    "vae": "./models/stable_diffusion/vae",         # Variational autoencoders for SD
    "inpaint": "./models/stable_diffusion/inpaint", # Inpainting models
    "ESRGAN": "./models/stable_diffusion/ESRGAN",   # Enhanced Super-Resolution GAN models
    "preview": "./models/stable_diffusion/preview", # Preview images
}


# ComfyUI integration configuration
comfy_ui_root_path = "../ComfyUI"                   # Root path to ComfyUI installation

# Git configuration for custom nodes installation
git = {
    "rootDirPath": "../portable-git",               # Root directory for portable Git
    "exePath": "../portable-git/cmd/git.exe",       # Executable path for Git
}

# Python environment for ComfyUI
comfyui_python_exe = "../comfyui-backend-env/python.exe"  # Python executable for ComfyUI
comfyui_python_env = "../comfyui-backend-env"             # Python environment directory

# Model paths specific to ComfyUI
comfy_ui_model_paths = {
    "checkpoints": f"{comfy_ui_root_path}/models/checkpoints",   # SD checkpoints
    "unet": f"{comfy_ui_root_path}/models/unet",                 # UNet models
    "clip": f"{comfy_ui_root_path}/models/clip",                 # CLIP models
    "vae": f"{comfy_ui_root_path}/models/vae",                   # VAE models
    "faceswap": f"{comfy_ui_root_path}/models/insightface",      # Face swap models
    "facerestore": f"{comfy_ui_root_path}/models/facerestore_models",  # Face restoration
    "nsfwdetector": f"{comfy_ui_root_path}/models/nsfw_detector",  # NSFW content detection
    "controlNet": f"{comfy_ui_root_path}/models/controlnet",     # ControlNet models
    "defaultCheckpoint": "./models/stable_diffusion/checkpoints",  # Default SD checkpoint
    "defaultLora": "./models/stable_diffusion/lora",             # Default LoRA location
}

# Llama.cpp model paths for GGUF format models
llama_cpp_model_paths = {
    "ggufLLM": "./models/llm/ggufLLM",             # GGUF formatted LLM models
}

# OpenVINO model paths for optimized models
openvino_model_paths = {
    "openvinoLLM": "./models/llm/openvino",         # OpenVINO optimized LLM models
}

# Default compute device - Intel XPU (Arc GPUs)
device = "xpu"
