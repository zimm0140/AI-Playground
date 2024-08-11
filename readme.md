# AI Playground

![image](https://github.com/user-attachments/assets/66086f2c-216e-4a79-8ff9-01e04db7e71d)

This example is based on the xpu implementation of Intel Arc A-Series dGPU and Ultra iGPU

Welcome to AI Playground beta open source project and AI PC starter app for doing AI image creation, image stylizing, and chatbot on a PC powered by an Intel® Arc™ GPU.  AI Playground leverages libraries from GitHub and Huggingface which may not be available in all countries world-wide.

## README.md
- English (readme.md)

## Min Specs
AI Playground beta is currently available as a packaged installer, or available as a source code from our Github repository.  To run AI Playground you must have a PC that meets the following specifications

*	Windows OS
*	Intel Core Ultra-H Processor (coming soon) OR Intel Arc GPU (discrete) with 8GB of vRAM

## Installation - Packaged Installer: 
AI Playground has multiple packaged installers, each specific to the hardware. 
1. Choose the correct installer (for Desktop systems with Intel Arc GPUs,or for Intel Core Ultra-H systems), download to your PC then run the installer.
2. The installer will have two phases.  It will first install components and environment from the installer. The second phase will pull in components from their source. </b >
This second phase of installation **will take several minutes** and require a steady internet connection.
3. On first run, the load screen will take up to a minute
4. Download the Users Guide for application information

*	AI Playground for Desktop-dGPU - [Release Notes](https://github.com/intel/AI-Playground/releases/tag/v1.0beta) | [Download](https://github.com/intel/AI-Playground/releases/download/v1.0beta/AI.Playground-v1.0b-Desktop_dGPU.exe)

*	AI Playground for Intel Core Ultra-H  - coming soon.

*	[AI Playground Users Guide](https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf)


## Project Development
### Dev Environment Setup (Backend, Python)

1. **Install Intel oneAPI Base Toolkit:**

    - Download and install the latest Intel oneAPI Base Toolkit from [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html).
    - Ensure you select all the necessary components, including the **Intel® oneAPI DPC++/C++ Compiler** and **Intel® oneAPI Math Kernel Library (oneMKL)**.

2. **Create and Activate the Conda Environment:**

   - Open a terminal or command prompt and navigate to the project root directory (AI-Playground).
   - Run the following command to create and activate the environment based on your hardware:

     **For Core Ultra-H:**
     ```bash
     conda env create -f environment-ultra.yml
     conda activate aipg_xpu_ultra
     ```

     **For Arc A-Series dGPUs:**
     ```bash
     conda env create -f environment-arc.yml
     conda activate aipg_xpu_arc
     ```


3. **Download and Install the Intel Extension for PyTorch AOT Packages:**

    - **Important:** Ensure you select the correct wheel file corresponding to your hardware and Python version from the [Intel Extension for PyTorch releases page](https://github.com/intel/intel-extension-for-pytorch/releases).

    - **For Core Ultra-H:**
      ```bash
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/intel_extension_for_pytorch-2.1.20+mtl-cp310-cp310-win_amd64.whl & \
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/torch-2.1.0a0+git7bcf7da-cp310-cp310-win_amd64.whl & \
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/torchaudio-2.1.0+6ea1133-cp310-cp310-win_amd64.whl & \
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.20%2Bmtl%2Boneapi/torchvision-0.16.0+fbb4cc5-cp310-cp310-win_amd64.whl
      ```

    - **For Arc A-Series dGPU:**
      ```bash
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torch-2.1.0a0+cxx11.abi-cp310-cp310-win_amd64.whl & \
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/intel_extension_for_pytorch-2.1.10+xpu-cp310-cp310-win_amd64.whl & \
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torchvision-0.16.0a0+cxx11.abi-cp310-cp310-win_amd64.whl & \
      pip install https://github.com/Nuullll/intel-extension-for-pytorch/releases/download/v2.1.10%2Bxpu/torchaudio-2.1.0a0+cxx11.abi-cp310-cp310-win_amd64.whl
      ```

4. **Verify the XPU Environment Setup:**

    ```bash
    python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())]"
    ```

### Linking Dev Environment to Project Environment

1.  **Switch to the project root directory:** (AI-Playground)
    ```bash
    cd AI-Playground 
    ```

2.  **View the Conda environment path (on Windows):** 
    ```powershell
    conda env list | findstr aipg_xpu
    ```
    This command will show the path to your `aipg_xpu` environment.

3.  **Create a symbolic link:**

    - **Using PowerShell:**
        ```powershell
        New-Item -ItemType Junction -Path ".\env" -Target "C:\Users\YourUserName\.conda\envs\aipg_xpu"
        ```
        (Replace `"C:\Users\YourUserName\.conda\envs\aipg_xpu"` with the actual path obtained from the previous step.)

    - **Using Command Prompt (cmd):**
        ```cmd
        mklink /J ".\env" "C:\Users\YourUserName\.conda\envs\aipg_xpu"
        ```
        (Replace `"C:\Users\YourUserName\.conda\envs\aipg_xpu"` with the actual path obtained from the previous step.)

### WebUI (Node.js + electron)

1.  **Install Node.js development environment:** Download and install from [Node.js download page](https://nodejs.org/).
2.  **Switch to the WebUI directory and install all Node.js dependencies.**
    ```bash
    cd WebUI
    npm install
    ```
3.  **In the WebUI directory, run the below command to get started with development**
    ```bash
    npm run dev
    ```

## Model Support
AI Playground supports PyTorch LLM, SD1.5, and SDXL models. AI Playground does not ship with any models but does make  models available for all features either directly from the interface or indirectly by the users downloading models from HuggingFace.com or CivitAI.com and placing them in the appropriate model folder. 

Models currently linked from the application 
| Model                                      | License                                                                                                                                                                      | Background Information/Model Card                                                                                      |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Dreamshaper 8 Model                        | [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)                                             | [site](https://huggingface.co/Lykon/dreamshaper-8)                               |
| Dreamshaper 8 Inpainting Model             | [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)                                             | [site](https://huggingface.co/Lykon/dreamshaper-8-inpainting)         |
| JuggernautXL v9 Model                      | [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)                                             | [site](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9)           |
| Phi3-mini-4k-instruct                      | [license](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE)                 | [site](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)     |
| bge-large-en-v1.5                          | [license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)                 | [site](https://huggingface.co/BAAI/bge-large-en-v1.5)                         |
| Latent Consistency Model (LCM) LoRA: SD1.5 | [license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | [site](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) |
| Latent Consistency Model (LCM) LoRA:SDXL   | [license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | [site](https://huggingface.co/latent-consistency/lcm-lora-sdxl)     |

Be sure to check license terms for any model used in AI Playground especially taking note of any restrictions.

### Use Alternative Models
Check the [User Guide](https://github.com/intel/ai-playground/blob/main/AI%20Playground%20Users%20Guide.pdf) for details or [watch this video](https://www.youtube.com/watch?v=1FXrk9Xcx2g) on how to add alternative Stable Diffusion models to AI Playground

### Notices and Disclaimers: 
For information on AI Playground terms, license and disclaimers, visit the project and files on GitHub repo:</br >
[License](https://github.com/intel/ai-playground/blob/main/LICENSE) | [Notices & Disclaimers](https://github.com/intel/ai-playground/blob/main/notices-disclaimers.md)

The software may include third party components with separate legal notices or governed by other agreements, as may be described in the Third Party Notices file accompanying the software.