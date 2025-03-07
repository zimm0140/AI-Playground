"""
OpenVINO Web API Service
------------------------
A Flask-based web service that provides API endpoints for interacting with OpenVINO-optimized
language models (LLMs). This service enables text generation via a RESTful API with
streaming response capabilities.

The API supports:
- Health status checks
- LLM chat completions with streaming responses
- Model unloading to free resources
- Stopping ongoing text generation

The service uses OpenVINO's optimized runtime for efficient inference on Intel hardware.
"""

import os
# Ensure OpenVINO libraries are in the PATH
os.environ['PATH'] = os.path.abspath('../openvino-env/Library/bin') + os.pathsep + os.environ['PATH']
from apiflask import APIFlask
from flask import jsonify, request, Response, stream_with_context
from openvino_backend import OpenVino
from openvino_adapter import LLM_SSE_Adapter
from openvino_params import LLMParams

# Initialize Flask application and OpenVINO backend
app = APIFlask(__name__)
llm_backend = OpenVino()


@app.get("/health")
def health():
    """Health check endpoint.
    
    Returns:
        JSON response with status code and success message.
    """
    return jsonify({"code": 0, "message": "success"})


@app.post("/api/llm/chat")
def llm_chat():
    """LLM chat endpoint that handles text generation requests.
    
    Processes the incoming JSON parameters, initializes the LLM with those parameters,
    and returns a streaming response with generated text using Server-Sent Events (SSE).
    
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
    
    Returns:
        JSON response with status code and success message.
    """
    llm_backend.unload_model()
    return jsonify({"code": 0, "message": "success"})


@app.get("/api/llm/stopGenerate")
def stop_llm_generate():
    """Stops any ongoing text generation process.
    
    Sets a flag in the backend to stop the generation process.
    
    Returns:
        JSON response with status code and success message.
    """
    llm_backend.stop_generate = True
    return jsonify({"code": 0, "message": "success"})


if __name__ == "__main__":
    """Run the web service when the script is executed directly.
    
    Parses command line arguments for port configuration and starts the Flask server.
    """
    import argparse

    parser = argparse.ArgumentParser(description="AI Playground Web service")
    parser.add_argument("--port", type=int, default=59997, help="Service listen port")
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, use_reloader=False)
