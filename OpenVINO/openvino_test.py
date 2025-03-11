"""
OpenVINO LLM Chat API Test Client
---------------------------------
This script demonstrates how to interact with the OpenVINO Web API service 
for language model chat functionality. It shows how to:

1. Create a properly formatted request to the LLM chat endpoint
2. Process streaming Server-Sent Events (SSE) responses
3. Parse and extract text data from the response

The example initializes a conversation with the LLM, setting a persona
and then asking a question to test the response.
"""

import requests
import pytest, socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
if sock.connect_ex(("127.0.0.1", 29000)) != 0:
    pytest.skip("Skipping OpenVINO tests because the server is not available", allow_module_level=True)

# Define the LLM chat API endpoint URL
url = "http://127.0.0.1:29000/api/llm/chat"

# Configure the parameters for the LLM request
# - prompt: Array of conversation turns with 'question' from user and optional 'answer' from LLM
# - device: Target inference device (empty string uses default)
# - enable_rag: Whether to use Retrieval Augmented Generation
# - model_repo_id: The specific LLM model to use
params = {
    "prompt": [
        {"question": "Your name is Luca", "answer": "My name is Luca."},
        {"question": "What is your name?"},
    ],
    "device": "",
    "enable_rag": False,
    "model_repo_id": "meta-llama-3.1-8b-instruct-q5_k_m.gguf",
}

# Send the POST request with streaming enabled
response = requests.post(url, json=params, stream=True)

# Check if the response status code is 200 (OK)
response.raise_for_status()

# Counter variable for tracking events
e = 1

# Iterate over the streaming response line by line
for line in response.iter_lines():
    e += 1
    if line:
        # Decode the line (assuming UTF-8 encoding)
        decoded_line = line.decode("utf-8")

        # SSE events typically start with "data: "
        if decoded_line.startswith("data:"):
            # Extract the data part
            data = decoded_line[len("data:") :]
            print(data)  # Process the data as needed
