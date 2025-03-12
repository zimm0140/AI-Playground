"""
Text Generation Service Module
-----------------------------
This module provides a service for generating text using pre-trained language models.

Features:
- Streamed text generation with real-time output
- Intel XPU (GPU) acceleration support
- Performance timing and metrics
- Chat-style interaction using message templates

This module demonstrates how to efficiently stream outputs from large language models
using PyTorch and the Transformers library with Intel hardware acceleration.
"""

import time
import traceback

# Load model directly
from threading import Thread

import intel_extension_for_pytorch as ipex
import torch
from transformers import PreTrainedModel, TextIteratorStreamer, pipeline


def stream_chat_generate(model: PreTrainedModel, args: dict):
    """
    Generate text using a pre-trained language model with streaming output.

    This function is designed to be run in a separate thread, allowing the
    generated text to be streamed back to the main thread via a TextIteratorStreamer.
    It also measures and reports the time taken for generation.

    Args:
        model: The pre-trained language model to use for generation
        args: Dictionary of arguments to pass to the model's generate method,
              including input tensors and the streamer object

    Note:
        This function catches and reports any exceptions that occur during generation
        to prevent thread crashes.
    """
    try:
        print("generate start")
        start = time.time()
        model.generate(**args)
        end = time.time()
        print(f"generate finish. cost {end - start}s")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    # Initialize the text generation pipeline with the Phi-3-mini model
    # using bfloat16 precision for improved performance
    pipe = pipeline(
        "text-generation",
        model="microsoft/Phi-3-mini-4k-instruct",
        torch_dtype=torch.bfloat16,
    )

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?",
        },
    ]

    # Prepare the model for inference by setting evaluation mode
    # and moving it to the Intel XPU (GPU) device
    pipe.model.eval()
    pipe.model.to("xpu")

    # Optimize the model for Intel hardware using IPEX
    model = ipex.optimize(pipe.model, dtype=torch.bfloat16)

    # Format the conversation using the model's chat template
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
    )

    # Encode the prompt into token IDs and move to the XPU device
    encoding = pipe.tokenizer.encode_plus(prompt, return_tensors="pt").to("xpu")
    tensor: torch.Tensor = encoding.get("input_ids")

    # Create a streaming interface to get generated tokens incrementally
    streamer = TextIteratorStreamer(
        pipe.tokenizer,
        skip_prompt=False,  # skip prompt in the generated tokens
        skip_special_tokens=True,
    )

    # Configure generation parameters for quality and performance
    generate_kwargs = dict(
        inputs=tensor,
        streamer=streamer,
        num_beams=1,
        do_sample=True,
        max_new_tokens=256,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

    # Ensure any pending XPU operations are completed before generation
    torch.xpu.synchronize()

    # Start generation in a separate thread to enable streaming
    Thread(target=stream_chat_generate, args=(pipe.model, generate_kwargs)).start()

    # Consume and print the streamed output tokens as they're generated
    for stream_output in streamer:
        print(stream_output, end="")
    print()
