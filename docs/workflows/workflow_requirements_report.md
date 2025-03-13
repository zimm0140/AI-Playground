
# ComfyUI Workflow Requirements Analysis {#comfyui-workflow-requirements-analysis}

Analysis run: 2025-03-10 21:11:51

## Summary {#summary}

Total workflows: 8
Successfully analyzed: 8

## Memory Requirements {#memory-requirements}

Minimum memory needed: 7GB
Maximum memory needed: 8GB

### Workflows by Memory Requirement {#workflows-by-memory-requirement}

| Memory (GB) | Workflows |
| ----------- | --------- |
| 7 | 6 |
| 8 | 2 |


## Model Usage {#model-usage}

| Model | Workflows |
| ----- | --------- |
| RunDiffusion---Juggernaut-XL-v9\\vae\\diffusion_pytorch_model.fp16.safetensors (vae) | FaceSwapHD.json, Line2ImageHD-Fast.json, Line2ImageHD-Quality.json |
| black-forest-labs---FLUX.1-schnell\\ae.safetensors (vae) | fluxQ4.json, fluxQ8.json |
| latent-consistency---lcm-lora-sdxl\\pytorch_lora_weights.safetensors (lora) | FaceSwapHD.json, Line2ImageHD-Fast.json |
| stabilityai---control-lora\\control-LoRAs-rank128\\control-lora-canny-rank128.safetensors (controlnet) | Line2ImageHD-Fast.json, Line2ImageHD-Quality.json |


## Custom Node Extensions {#custom-node-extensions}

| Extension | Workflows |
| --------- | --------- |
| comfyui-face-swap | CopyFace.json, FaceSwapHD.json |


## Python Package Requirements {#python-package-requirements}

| Package | Required By |
| ------- | ----------- |
| opencv-python | CopyFace.json, FaceSwapHD.json |
| insightface | CopyFace.json, FaceSwapHD.json |
| onnxruntime-gpu | CopyFace.json, FaceSwapHD.json |


## Individual Workflow Details {#individual-workflow-details}

### Colorize.json {#colorizejson}

### Memory Requirements {#memory-requirements}

- Minimum: 7GB

- Recommended: 12GB

### CopyFace.json {#copyfacejson}

### Required Custom Nodes {#required-custom-nodes}

- comfyui-face-swap

### Python Package Dependencies {#python-package-dependencies}

- opencv-python

- insightface

- onnxruntime-gpu

### Memory Requirements {#memory-requirements}

- Minimum: 7GB

- Recommended: 12GB

### FaceSwapHD.json {#faceswaphdjson}

### Required Models {#required-models}

- Vae: RunDiffusion---Juggernaut-XL-v9\\vae\\diffusion_pytorch_model.fp16.safetensors

- Lora: latent-consistency---lcm-lora-sdxl\\pytorch_lora_weights.safetensors

### Required Custom Nodes {#required-custom-nodes}

- comfyui-face-swap

### Python Package Dependencies {#python-package-dependencies}

- opencv-python

- insightface

- onnxruntime-gpu

### Memory Requirements {#memory-requirements}

- Minimum: 7GB

- Recommended: 12GB

### Line2ImageHD-Fast.json {#line2imagehd-fastjson}

### Required Models {#required-models}

- Vae: RunDiffusion---Juggernaut-XL-v9\\vae\\diffusion_pytorch_model.fp16.safetensors

- Lora: latent-consistency---lcm-lora-sdxl\\pytorch_lora_weights.safetensors

- Controlnet: stabilityai---control-lora\\control-LoRAs-rank128\\control-lora-canny-rank128.safetensors

### Memory Requirements {#memory-requirements}

- Minimum: 8GB

- Recommended: 13GB

### Line2ImageHD-Quality.json {#line2imagehd-qualityjson}

### Required Models {#required-models}

- Vae: RunDiffusion---Juggernaut-XL-v9\\vae\\diffusion_pytorch_model.fp16.safetensors

- Controlnet: stabilityai---control-lora\\control-LoRAs-rank128\\control-lora-canny-rank128.safetensors

### Memory Requirements {#memory-requirements}

- Minimum: 8GB

- Recommended: 13GB

### Video.json {#videojson}

### Memory Requirements {#memory-requirements}

- Minimum: 7GB

- Recommended: 12GB

### fluxQ4.json {#fluxq4json}

### Required Models {#required-models}

- Vae: black-forest-labs---FLUX.1-schnell\\ae.safetensors

### Memory Requirements {#memory-requirements}

- Minimum: 7GB

- Recommended: 12GB

### fluxQ8.json {#fluxq8json}

### Required Models {#required-models}

- Vae: black-forest-labs---FLUX.1-schnell\\ae.safetensors

### Memory Requirements {#memory-requirements}

- Minimum: 7GB

- Recommended: 12GB
