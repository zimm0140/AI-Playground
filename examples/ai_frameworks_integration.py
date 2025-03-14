#!/usr/bin/env python3
"""
Hardware-Aware AI Framework Integration Example

This example demonstrates how to use the hardware detection system with LangChain
and Stable Diffusion to optimize performance on different Intel hardware.
"""

import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

import torch
from compel import Compel
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Import the hardware detection module
from hardware_detection import detect_hardware_type


def configure_hardware():
    """Configure the environment based on detected hardware"""
    hardware_type = detect_hardware_type()
    print(f"Detected hardware: {hardware_type}")

    if hardware_type == "acm":  # Intel Arc GPUs
        try:
            # Check for Intel IPEX availability
            has_ipex = importlib.util.find_spec("intel_extension_for_pytorch") is not None
            if has_ipex:
                # Import but use it immediately to avoid unused import warning
                import intel_extension_for_pytorch as ipex
                _ = ipex.__name__
                
                from service.xpu_hijacks import ipex_hijacks
                _ = ipex_hijacks.__name__

            # Apply XPU hijacks to redirect CUDA calls to XPU
            ipex_hijacks()

            # Set environment variables for Intel GPU
            os.environ["SYCL_CACHE_PERSISTENT"] = "1"
            os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"

            device = torch.device("xpu")
            print("Intel Arc GPU configured successfully")
            return device, hardware_type
        except ImportError:
            print("Intel Extension for PyTorch not found, falling back to CPU.")
            device = torch.device("cpu")

    elif hardware_type == "ovino":  # OpenVINO
        try:
            # Check for OpenVINO availability
            has_openvino = importlib.util.find_spec("openvino") is not None
            if has_openvino:
                # Import but use it immediately to avoid unused import warning
                import openvino
                _ = openvino.__name__

            device = torch.device("cpu")  # OpenVINO optimizes on CPU
            print("OpenVINO environment configured")
            return device, hardware_type
        except ImportError:
            print("OpenVINO not found, falling back to CPU.")
            device = torch.device("cpu")

    # Default fallback to CPU
    device = torch.device("cpu")
    return device, hardware_type


def setup_langchain_model(device, hardware_type, model_id="microsoft/Phi-3-mini-4k-instruct"):
    """Set up a LangChain model with hardware-specific optimizations"""
    print(f"Setting up LangChain with {model_id} on {device}")

    # Hardware-specific model loading
    if hardware_type == "acm":
        # Intel Arc GPU optimization
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True).to(
            device,
        )

    elif hardware_type == "ovino":
        # OpenVINO optimization
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        # Here you would convert to OpenVINO IR format
        # This is a simplified example
        print("Note: Full OpenVINO conversion would be done here")

    else:
        # Standard CPU loading
        model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, trust_remote_code=True)

    # Set up tokenizer and pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Create text generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        device=device if hardware_type != "ovino" else "cpu",
    )

    # Wrap with LangChain
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm


def setup_stable_diffusion(device, hardware_type):
    """Set up Stable Diffusion with hardware-specific optimizations"""
    print(f"Setting up Stable Diffusion on {device}")

    # This is a simplified example - in practice you would:
    # 1. Load the appropriate SD model based on hardware
    # 2. Configure the model with optimal settings
    # 3. Set up the pipeline

    if hardware_type == "acm":
        # Intel Arc GPU specific settings for Stable Diffusion
        try:
            import intel_extension_for_pytorch as ipex
            from diffusers import StableDiffusionPipeline

            # Example of loading model for Intel Arc
            pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

            # Optional: Optimize the pipeline
            pipeline = ipex.optimize(pipeline)

            # Set up Compel for prompt conditioning
            compel = Compel(
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                truncate_long_prompts=False,
            )

            return pipeline, compel
        except ImportError:
            print("Required libraries for Arc GPU not available")

    elif hardware_type == "ovino":
        # OpenVINO optimization for Stable Diffusion
        try:
            from diffusers import StableDiffusionPipeline

            # Example for OpenVINO - this would typically involve conversion to IR
            pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

            # Note: In a real implementation, you would convert components to OpenVINO
            print("Note: Full OpenVINO conversion would be done here")

            # Set up Compel for prompt conditioning
            compel = Compel(
                tokenizer=pipeline.tokenizer,
                text_encoder=pipeline.text_encoder,
                truncate_long_prompts=False,
            )

            return pipeline, compel
        except ImportError:
            print("Required libraries for OpenVINO not available")

    # Default CPU implementation
    try:
        from diffusers import StableDiffusionPipeline

        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

        # Set up Compel for prompt conditioning
        compel = Compel(
            tokenizer=pipeline.tokenizer,
            text_encoder=pipeline.text_encoder,
            truncate_long_prompts=False,
        )

        return pipeline, compel
    except ImportError:
        print("Diffusers library not available")
        return None, None


def main():
    """Main function to demonstrate hardware-aware AI framework integration"""
    print("Configuring hardware for AI frameworks...")
    device, hardware_type = configure_hardware()

    # Example 1: Set up LangChain with hardware optimization
    llm = setup_langchain_model(device, hardware_type)
    if llm:
        # Test the model
        print("\nTesting LangChain model:")
        response = llm("Explain quantum computing in simple terms.")
        print(f"Response: {response}")

    # Example 2: Set up Stable Diffusion with hardware optimization
    sd_pipeline, compel = setup_stable_diffusion(device, hardware_type)
    if sd_pipeline and compel:
        print("\nStable Diffusion pipeline set up successfully")
        print("You can now generate images with hardware-optimized settings")

        # Note: Actual image generation would be done here
        # This is commented out as it requires significant resources
        """
        prompt = "a photo of an astronaut riding a horse on mars"
        conditioned_prompt = compel(prompt)
        image = sd_pipeline(prompt_embeds=conditioned_prompt).images[0]
        image.save("astronaut_on_mars.png")
        """


if __name__ == "__main__":
    main()
