"""
Llama.cpp Web API Service
------------------------
A Flask-based web service that provides API endpoints for interacting with Llama.cpp
language models. This service enables text generation via a RESTful API with
streaming response capabilities.

The API supports:
- Health status checks
- LLM chat completions with streaming responses
- Model unloading to free resources
- Stopping ongoing text generation

The service uses Llama.cpp's optimized C++ implementation for efficient inference
on various hardware including CPUs and GPUs.
"""

import os

# Ensure Llama.cpp libraries are in the PATH
os.environ["PATH"] = os.path.abspath("../llama-cpp-env/Library/bin") + os.pathsep + os.environ["PATH"]

# Import after setting PATH environment variable
from apiflask import APIFlask  # noqa: E402
from flask import Response, jsonify, request, stream_with_context  # noqa: E402
from llama_adapter import LLM_SSE_Adapter  # noqa: E402
from llama_cpp_backend import LlamaCpp  # noqa: E402
from llama_params import LLMParams  # noqa: E402

# Initialize Flask application and Llama.cpp backend
app = APIFlask(__name__)
llm_backend = LlamaCpp()


@app.get("/health")
def health():
    """Health check endpoint.

    Provides a simple way to verify the service is running and responsive.

    Returns:
        JSON response with status code and success message.
    """
    return jsonify({"code": 0, "message": "success"})


@app.post("/api/llm/chat")
def llm_chat():
    """LLM chat endpoint that handles text generation requests.

    Processes the incoming JSON parameters, initializes the LLM with those parameters,
    and returns a streaming response with generated text using Server-Sent Events (SSE).

    The function:
    1. Extracts parameters from the request JSON
    2. Converts them to an LLMParams object
    3. Creates an adapter for streaming
    4. Generates text and streams it back to the client

    Returns:
        Streaming response with generated text chunks.
    """
    params = request.get_json()
    params.pop("print_metrics", None)  # Remove print_metrics if present
    llm_params = LLMParams(**params)  # Convert JSON to LLMParams object
    sse_invoker = LLM_SSE_Adapter(llm_backend)  # Create adapter for streaming
    it = sse_invoker.text_conversation(llm_params)  # Generate text iterator
    return Response(stream_with_context(it), content_type="text/event-stream")


@app.post("/api/free")
def free():
    """Frees resources by unloading the model from memory.

    This endpoint allows clients to explicitly release memory used by LLM models
    when they're no longer needed, which is useful for resource management.

    Returns:
        JSON response with status code and success message.
    """
    llm_backend.unload_model()
    return jsonify({"code": 0, "message": "success"})


@app.get("/api/llm/stopGenerate")
def stop_llm_generate():
    """Stops any ongoing text generation process.

    Sets a flag in the backend to stop the generation process, which is useful
    for canceling long generations or when the user no longer needs the output.

    Returns:
        JSON response with status code and success message.
    """
    llm_backend.stop_generate = True
    return jsonify({"code": 0, "message": "success"})


if __name__ == "__main__":
    """Main entry point when script is run directly.

    This block:
    1. Parses command-line arguments for configuration
    2. Starts the Flask web server on the specified port
    3. Configures the server to run on localhost (127.0.0.1)
    4. Disables the reloader to prevent duplicate process issues
    """
    import argparse

    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="AI Playground Web service")
    parser.add_argument("--port", type=int, default=59997, help="Service listen port")
    args = parser.parse_args()

    # Start the Flask application with the specified configuration
    app.run(host="127.0.0.1", port=args.port, use_reloader=False)
