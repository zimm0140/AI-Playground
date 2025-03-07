"""
API Integration Test Module
--------------------------
This module contains integration tests for the AI Playground web API service.

It tests core functionality of the service including:
- Graphics device detection and enumeration
- Service initialization with model paths
- LLM (Language Model) chat capabilities

The tests use unittest framework and Flask's test client to directly
interact with the application without running a full server.
"""

import sys
import os
import unittest
import logging
import json


class TestAPI(unittest.TestCase):
    """
    Test suite for the AI Playground web API.
    
    This class tests several API endpoints to verify the core functionality
    of the service. It sets up the test environment, defines test cases, and
    provides helper methods for common operations like payload creation and
    response parsing.
    """
    
    def setUp(self):
        """
        Set up the test environment before each test.
        
        This method:
        - Adds the service directory to the Python path
        - Configures model paths for different model types
        - Initializes the Flask test client
        - Sets up test data like device info and model IDs
        """
        self.service_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        sys.path.insert(0, self.service_dir)

        self.model_dir = os.path.abspath(os.path.join(self.service_dir, "models"))
        self.model_paths = {
            "llm": os.path.join(self.model_dir, "llm", "checkpoints"),
            "embedding": os.path.join(self.model_dir, "llm", "embedding"),
            "inpaint": os.path.join(self.model_dir, "stable_diffusion", "inpaint"),
            "lora": os.path.join(self.model_dir, "stable_diffusion", "lora"),
            "stableDiffusion": os.path.join(
                self.model_dir, "stable_diffusion", "checkpoints"
            ),
            "vae": os.path.join(self.model_dir, "stable_diffusion", "vae"),
        }

        from web_api import app

        self.app = app.test_client()

        self.devices = {}
        self.llm_model_id = "microsoft/Phi-3-mini-4k-instruct"

    def test_get_graphics(self):
        """
        Test the /api/getGraphics endpoint.
        
        This test:
        - Makes a POST request to the graphics endpoint
        - Verifies the response contains supported graphics devices
        - Logs device information for debugging
        - Stores device info for potential use in other tests
        - Asserts that at least one device is available
        - Verifies the HTTP status code is 200 (OK)
        """
        response = self.app.post("/api/getGraphics")
        supported_graphics = response.get_json()
        for graphics in supported_graphics:
            logging.info(f"Device #{graphics['index']}: {graphics['name']}")
            self.devices[graphics["index"]] = graphics["name"]
        self.assertGreater(len(supported_graphics), 0)
        self.assertEqual(response.status_code, 200)

    def test_init(self):
        """
        Test the /api/init endpoint for service initialization.
        
        This test:
        - Makes a POST request with model paths to initialize the service
        - Verifies the response contains the expected set of schedulers
        - Confirms all expected scheduler names are present
        - Checks that the HTTP status code is 200 (OK)
        """
        response = self.app.post("/api/init", json=self.model_paths)
        schedulers = response.get_json()
        self.assertEqual(
            set(schedulers),
            {
                "DPM++ 2M",
                "DPM++ 2M Karras",
                "DPM++ SDE",
                "DPM++ SDE Karras",
                "DPM2",
                "DPM2 Karras",
                "DPM2 a",
                "DPM2 a Karras",
                "Euler",
                "Euler a",
                "Heun",
                "LMS",
                "LMS Karras",
                "DEIS",
                "UniPC",
                "DDIM",
                "DDPM",
                "EDM Euler",
                "PNDM",
                "LCM",
            },
        )
        self.assertEqual(response.status_code, 200)

    def get_llm_chat_payload(self, prompt):
        """
        Create a payload for an LLM chat request.
        
        This helper method generates a standard payload with the given prompt
        and consistent settings for device, RAG, and model ID.
        
        Args:
            prompt (str): The prompt text to send to the language model
            
        Returns:
            dict: A formatted payload dictionary ready for the chat API
        """
        return {
            "device": 0,
            "enable_rag": False,
            "model_repo_id": self.llm_model_id,
            "prompt": [{"question": prompt, "answer": ""}],
        }

    def decode_stream(self, stream_data):
        """
        Decode a stream of Server-Sent Events (SSE) data.
        
        This helper method:
        - Splits the binary stream by null bytes
        - Extracts data sections from the SSE format
        - Parses JSON from each data section
        - Collects parsed data into a list
        
        Args:
            stream_data (bytes): Raw binary SSE data from API response
            
        Returns:
            list: List of parsed JSON objects from the stream
            
        Raises:
            AssertionError: If JSON parsing fails for any event
        """
        event_data = []
        for line in stream_data.split(b"\x00"):
            if line.startswith(b"data:"):
                data_json = line.split(b"data:")[1].strip()
                try:
                    data = json.loads(data_json)
                    event_data.append(data)
                except json.JSONDecodeError:
                    self.fail(f"Failed to decode JSON: {data_json}")
        return event_data

    def llm_warmup(self):
        """
        Perform a warm-up request to the LLM chat API.
        
        This helper method:
        - Sends a simple greeting to prepare the model
        - Verifies the response status code
        - Checks that the response contains valid data
        
        This warm-up helps ensure the model is loaded and ready
        before running more complex test cases.
        """
        logging.info("Warming up LLM...")
        response = self.app.post("/api/llm/chat", json=self.get_llm_chat_payload("hi"))
        self.assertEqual(response.status_code, 200)

        event_data = self.decode_stream(response.data)
        self.assertGreater(len(event_data), 0)

    def test_llm_chat(self):
        """
        Test the /api/llm/chat endpoint for language model interaction.
        
        This test:
        - Warms up the model with a simple request
        - Sends a more complex prompt asking about why the sky is blue
        - Verifies the response status code is 200 (OK)
        - Checks that the response contains valid streaming data
        """
        self.llm_warmup()

        logging.info("Testing LLM chat...")
        response = self.app.post(
            "/api/llm/chat",
            json=self.get_llm_chat_payload(
                "Please explain in detail: why is sky blue?"
            ),
        )
        self.assertEqual(response.status_code, 200)

        event_data = self.decode_stream(response.data)
        self.assertGreater(len(event_data), 0)


if __name__ == "__main__":
    unittest.main()
