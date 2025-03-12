# ComfyUI Workflow Requirements Report

Generated on: 2025-03-10 05:02:58

## Summary

- Total workflows analyzed: 8
- Successfully analyzed: 8

## Aggregate Requirements

### Models

| Model | Workflows |
|-------|----------|
| RunDiffusion---Juggernaut-XL-v9\vae\diffusion_pytorch_model.fp16.safetensors (vae) | FaceSwapHD.json, Line2ImageHD-Fast.json, Line2ImageHD-Quality.json |
| black-forest-labs---FLUX.1-schnell\ae.safetensors (vae) | fluxQ4.json, fluxQ8.json |
| latent-consistency---lcm-lora-sdxl\pytorch_lora_weights.safetensors (lora) | FaceSwapHD.json, Line2ImageHD-Fast.json |
| stabilityai---control-lora\control-LoRAs-rank128\control-lora-canny-rank128.safetensors (controlnet) | Line2ImageHD-Fast.json, Line2ImageHD-Quality.json |


### Custom Nodes

| Custom Node Extension | Workflows |
|----------------------|----------|
| comfyui-face-swap | CopyFace.json, FaceSwapHD.json |


### Python Packages

| Package | Workflows |
|---------|----------|
| opencv-python | CopyFace.json, FaceSwapHD.json |
| insightface | CopyFace.json, FaceSwapHD.json |
| onnxruntime-gpu | CopyFace.json, FaceSwapHD.json |


### Memory Requirements

- Minimum memory required: 0.0GB
- Maximum memory required: 0.0GB

| Memory Requirement | Workflows |
|-------------------|----------|
| 7GB | Colorize.json, CopyFace.json, FaceSwapHD.json, Video.json, fluxQ4.json, fluxQ8.json |
| 8GB | Line2ImageHD-Fast.json, Line2ImageHD-Quality.json |


## Individual Workflow Requirements

### Colorize.json

✅ **Successfully analyzed**

### Memory Requirements

- Minimum: 7GB
- Recommended: 10GB

### CopyFace.json

✅ **Successfully analyzed**

### Required Custom Nodes

- comfyui-face-swap

### Required Python Packages

- opencv-python
- insightface
- onnxruntime-gpu

### Memory Requirements

- Minimum: 7GB
- Recommended: 10GB

### FaceSwapHD.json

✅ **Successfully analyzed**

### Required Models

- Vae: RunDiffusion---Juggernaut-XL-v9\vae\diffusion_pytorch_model.fp16.safetensors
- Lora: latent-consistency---lcm-lora-sdxl\pytorch_lora_weights.safetensors

### Required Custom Nodes

- comfyui-face-swap

### Required Python Packages

- opencv-python
- insightface
- onnxruntime-gpu

### Memory Requirements

- Minimum: 7GB
- Recommended: 10GB

### Line2ImageHD-Fast.json

✅ **Successfully analyzed**

### Required Models

- Vae: RunDiffusion---Juggernaut-XL-v9\vae\diffusion_pytorch_model.fp16.safetensors
- Lora: latent-consistency---lcm-lora-sdxl\pytorch_lora_weights.safetensors
- Controlnet: stabilityai---control-lora\control-LoRAs-rank128\control-lora-canny-rank128.safetensors

### Memory Requirements

- Minimum: 8GB
- Recommended: 12GB

### Line2ImageHD-Quality.json

✅ **Successfully analyzed**

### Required Models

- Vae: RunDiffusion---Juggernaut-XL-v9\vae\diffusion_pytorch_model.fp16.safetensors
- Controlnet: stabilityai---control-lora\control-LoRAs-rank128\control-lora-canny-rank128.safetensors

### Memory Requirements

- Minimum: 8GB
- Recommended: 12GB

### Video.json

✅ **Successfully analyzed**

### Memory Requirements

- Minimum: 7GB
- Recommended: 10GB

### fluxQ4.json

✅ **Successfully analyzed**

### Required Models

- Vae: black-forest-labs---FLUX.1-schnell\ae.safetensors

### Memory Requirements

- Minimum: 7GB
- Recommended: 10GB

### fluxQ8.json

✅ **Successfully analyzed**

### Required Models

- Vae: black-forest-labs---FLUX.1-schnell\ae.safetensors

### Memory Requirements

- Minimum: 7GB
- Recommended: 10GB
