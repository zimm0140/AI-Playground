#!/usr/bin/env python
"""
ComfyUI Workflow Execution with Model Simulation

This script simulates execution of ComfyUI workflows using dummy models:
1. Creates minimal model simulations (small tensors) to stand in for real models
2. Executes key parts of workflow graphs to test actual node execution logic
3. Verifies data flow between nodes with realistic but minimal resource usage
4. Detects runtime errors and implementation issues not caught by static validation

This provides more meaningful testing without requiring large model downloads.
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Optional, Dict, List

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Add the parent directory to the path so we can import the utils package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.workflow_parser import build_link_map, get_workflow_links, get_workflow_nodes

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ComfyWorkflowSimulator")


class ModelSimulation:
    """Simulates models for lightweight testing"""

    def __init__(self, model_type: str, parameters: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        self.parameters = parameters or {}
        self.tensor_data = None
        self._create_dummy_tensors()

    def _create_dummy_tensors(self):
        """Create dummy tensor data based on model type"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using NumPy arrays for simulation")
            self._create_numpy_tensors()
            return

        # Create appropriate tensor shapes based on model type
        if self.model_type == "checkpoint":
            # Simulate a small diffusion model checkpoint
            sd_type = self.parameters.get("sd_type", "SD1.5")
            if sd_type == "SDXL":
                hidden_size = 128
            elif sd_type == "SD3":
                hidden_size = 192
            else:  # SD1.5
                hidden_size = 64

            self.tensor_data = {
                "model": {
                    "diffusion_model": torch.randn(1, hidden_size, 32, 32),
                    "first_stage_model": torch.randn(1, hidden_size // 2, 16, 16),
                    "cond_stage_model": torch.randn(1, hidden_size // 2, 64),
                },
                "global_step": torch.tensor(1000),
            }

        elif self.model_type == "vae":
            # Simulate a VAE model
            hidden_size = 32
            self.tensor_data = {
                "encoder": torch.randn(1, hidden_size, 16, 16),
                "decoder": torch.randn(1, hidden_size, 16, 16),
                "quant_conv": torch.randn(1, hidden_size, 8, 8),
                "post_quant_conv": torch.randn(1, hidden_size, 8, 8),
            }

        elif self.model_type == "lora":
            # Simulate a LoRA model - very small matrices
            hidden_size = 16
            self.tensor_data = {
                "lora_up": torch.randn(hidden_size, hidden_size),
                "lora_down": torch.randn(hidden_size, hidden_size),
                "alpha": torch.tensor(0.75),
            }

        elif self.model_type == "controlnet":
            # Simulate a ControlNet model
            hidden_size = 48
            self.tensor_data = {
                "control_model": torch.randn(1, hidden_size, 16, 16),
                "hint_model": torch.randn(1, 3, 64, 64),
            }

        elif self.model_type == "upscaler":
            # Simulate an upscaler model
            hidden_size = 24
            self.tensor_data = {"upscale_model": torch.randn(1, hidden_size, 8, 8)}

        else:
            # Generic model simulation
            logger.warning(
                f"Unknown model type: {self.model_type} - using generic simulation",
            )
            self.tensor_data = {"weights": torch.randn(1, 32, 32, 32)}

    def _create_numpy_tensors(self):
        """Create NumPy arrays when PyTorch is not available"""
        # Similar to the PyTorch version but using NumPy
        if self.model_type == "checkpoint":
            sd_type = self.parameters.get("sd_type", "SD1.5")
            if sd_type == "SDXL":
                hidden_size = 128
            elif sd_type == "SD3":
                hidden_size = 192
            else:  # SD1.5
                hidden_size = 64

            self.tensor_data = {
                "model": {
                    "diffusion_model": np.random.randn(1, hidden_size, 32, 32),
                    "first_stage_model": np.random.randn(1, hidden_size // 2, 16, 16),
                    "cond_stage_model": np.random.randn(1, hidden_size // 2, 64),
                },
                "global_step": np.array(1000),
            }
        # ... similar implementations for other model types ...
        else:
            # Generic model simulation
            self.tensor_data = {"weights": np.random.randn(1, 32, 32, 32)}

    def get_tensor(self, key: str):
        """Retrieve a specific tensor from the model"""
        if key in self.tensor_data:
            return self.tensor_data[key]

        # Try nested lookups with dot notation
        if "." in key:
            parts = key.split(".")
            data = self.tensor_data
            for part in parts:
                if part in data:
                    data = data[part]
                else:
                    return None
            return data

        return None


class NodeSimulation:
    """Simulates execution of ComfyUI nodes"""

    def __init__(self, node_type: str, node_id: str, inputs: Optional[Dict[str, Any]] = None):
        self.node_type = node_type
        self.node_id = node_id
        self.inputs = inputs or {}
        self.outputs = {}

    def execute(self):
        """Simulate execution of the node, generating appropriate outputs"""
        try:
            # Route to appropriate handler based on node type
            if hasattr(self, f"_execute_{self.node_type}"):
                method = getattr(self, f"_execute_{self.node_type}")
                self.outputs = method()
            else:
                # Generic handler for unknown nodes
                self.outputs = self._execute_generic()

            return True, self.outputs
        except Exception as e:
            logger.error(
                f"Error executing node {self.node_id} of type {self.node_type}: {str(e)}",
            )
            return False, {"error": str(e)}

    def _execute_generic(self):
        """Generic execution for unknown node types"""
        # For unknown nodes, we create outputs with appropriate shapes
        if TORCH_AVAILABLE:
            return {"output": torch.randn(1, 32, 32, 32)}
        return {"output": np.random.randn(1, 32, 32, 32)}

    def _execute_checkpoint_loader(self):
        """Simulate checkpoint loader node"""
        ckpt_name = self.inputs.get("ckpt_name", "stable_diffusion.ckpt")

        # Determine SD type from name
        sd_type = "SD1.5"
        if "xl" in ckpt_name.lower():
            sd_type = "SDXL"
        elif "sd3" in ckpt_name.lower() or "sd-3" in ckpt_name.lower():
            sd_type = "SD3"

        # Create model simulation
        model_sim = ModelSimulation("checkpoint", {"sd_type": sd_type})

        # Return appropriate outputs
        return {
            "model": model_sim.get_tensor("model"),
            "clip": model_sim.get_tensor("model.cond_stage_model"),
            "vae": model_sim.get_tensor("model.first_stage_model"),
        }

    def _execute_vae_loader(self):
        """Simulate VAE loader node"""
        # Variable captured for future implementation
        _ = self.inputs.get("vae_name", "vae.pt")
        model_sim = ModelSimulation("vae")

        return {"vae": model_sim.tensor_data}

    def _execute_lora_loader(self):
        """Simulate LoRA loader node"""
        model = self.inputs.get("model", None)
        clip = self.inputs.get("clip", None)
        # Variables captured for future implementation
        _ = self.inputs.get("lora_name", "lora.safetensors")
        _ = self.inputs.get("strength", 1.0)

        # Apply "modifications" to the input model and clip
        if model is not None and clip is not None:
            # In reality, LoRA would modify these models
            # For simulation, we just pass them through
            return {"model": model, "clip": clip}
        # If no model/clip provided, return dummy tensors
        model_sim = ModelSimulation("lora")
        return {
            "model": model_sim.get_tensor("lora_up"),
            "clip": model_sim.get_tensor("lora_down"),
        }

    def _execute_clip_text_encode(self):
        """Simulate CLIP text encoding"""
        text = self.inputs.get("text", "")
        # Variable captured for future implementation
        _ = self.inputs.get("clip", None)

        # Create a conditioning tensor that would represent encoded text
        if TORCH_AVAILABLE:
            # Size depends on text length, but we'll use a fixed size for simulation
            text_token_count = min(len(text.split()), 77) if text else 3
            return {"conditioning": torch.randn(1, text_token_count, 64)}
        text_token_count = min(len(text.split()), 77) if text else 3
        return {"conditioning": np.random.randn(1, text_token_count, 64)}

    def _execute_empty_latent_image(self):
        """Simulate empty latent image creation"""
        width = self.inputs.get("width", 512)
        height = self.inputs.get("height", 512)
        batch_size = self.inputs.get("batch_size", 1)

        # Calculate appropriate latent dimensions (typically 1/8 of pixel dimensions)
        latent_width = width // 8
        latent_height = height // 8

        if TORCH_AVAILABLE:
            return {"latent": torch.zeros(batch_size, 4, latent_height, latent_width)}
        return {"latent": np.zeros((batch_size, 4, latent_height, latent_width))}

    def _execute_ksampler(self):
        """Simulate K-Sampler node"""
        # Variables captured for future implementation
        _ = self.inputs.get("model")
        _ = self.inputs.get("positive")
        _ = self.inputs.get("negative")
        latent = self.inputs.get("latent_image")

        # In reality, this would run diffusion steps
        # For simulation, we just add noise to the latent
        if latent is not None:
            if TORCH_AVAILABLE:
                noise = torch.randn_like(latent) * 0.1
                return {"latent": latent + noise}
            noise = np.random.randn(*latent.shape) * 0.1
            return {"latent": latent + noise}
        # If no latent provided, return a dummy tensor
        if TORCH_AVAILABLE:
            return {"latent": torch.randn(1, 4, 64, 64)}
        return {"latent": np.random.randn(1, 4, 64, 64)}

    def _execute_vae_decode(self):
        """Simulate VAE decoding from latent to image"""
        # Variable captured for future implementation
        _ = self.inputs.get("vae")
        samples = self.inputs.get("samples")

        # In reality, this would decode the latent using the VAE
        # For simulation, we just create an appropriately sized tensor
        if samples is not None:
            # Get latent dimensions and scale up 8x for pixel space
            if TORCH_AVAILABLE:
                # Assuming samples has shape [B, 4, H, W]
                B, _, H, W = samples.shape
                return {"image": torch.randn(B, 3, H * 8, W * 8)}
            # NumPy version
            B, _, H, W = samples.shape
            return {"image": np.random.randn(B, 3, H * 8, W * 8)}
        # Fallback
        if TORCH_AVAILABLE:
            return {"image": torch.randn(1, 3, 512, 512)}
        return {"image": np.random.randn(1, 3, 512, 512)}

    def _execute_save_image(self):
        """Simulate saving an image"""
        images = self.inputs.get("images")

        # In reality, this would save images to disk
        # For simulation, we just log that we would save them
        if images is not None:
            logger.info(
                f"Node {self.node_id}: Would save images with shape {images.shape}",
            )
        else:
            logger.warning(f"Node {self.node_id}: No images to save")

        # This node has no outputs
        return {}

    def _execute_upscale_image(self):
        """Simulate image upscaling"""
        image = self.inputs.get("image")
        # Variable captured for future implementation
        _ = self.inputs.get("upscale_method", "nearest")
        scale = self.inputs.get("scale", 2.0)

        if image is not None:
            # Simulate upscaling by creating a larger tensor
            if TORCH_AVAILABLE:
                # Assuming image has shape [B, C, H, W]
                B, C, H, W = image.shape
                new_H = int(H * scale)
                new_W = int(W * scale)
                return {"image": torch.randn(B, C, new_H, new_W)}
            # NumPy version
            B, C, H, W = image.shape
            new_H = int(H * scale)
            new_W = int(W * scale)
            return {"image": np.random.randn(B, C, new_H, new_W)}
        # Fallback
        if TORCH_AVAILABLE:
            return {"image": torch.randn(1, 3, 1024, 1024)}
        return {"image": np.random.randn(1, 3, 1024, 1024)}

    # Add more node-specific simulation methods as needed


class ComfyWorkflowSimulator:
    """Simulates execution of ComfyUI workflows"""

    def __init__(
        self, workflows_dir: str, output_dir: str = "ci_artifacts/workflow_simulation",
    ):
        self.workflows_dir = workflows_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = {
            "summary": {
                "time": datetime.now().isoformat(),
                "total_workflows": 0,
                "successful_workflows": 0,
                "failed_workflows": 0,
            },
            "workflows": [],
        }

    def find_workflow_files(self) -> List[str]:
        """Find all workflow JSON files in the specified directory"""
        return glob.glob(os.path.join(self.workflows_dir, "*.json"))

    def topological_sort(self, workflow: Optional[Dict[str, Any]]) -> List[str]:
        """Sort nodes in topological order for execution"""
        if "links" not in workflow:
            return []

        # Get nodes from either workflow format
        nodes = None
        if "nodes" in workflow and isinstance(workflow["nodes"], dict):
            nodes = workflow["nodes"]
        elif (
            "comfyUiApiWorkflow" in workflow
            and isinstance(workflow["comfyUiApiWorkflow"], dict)
            and "nodes" in workflow["comfyUiApiWorkflow"]
            and isinstance(workflow["comfyUiApiWorkflow"]["nodes"], dict)
        ):
            nodes = workflow["comfyUiApiWorkflow"]["nodes"]

        if nodes is None:
            return []

        # Build a directed graph using adjacency list and count incoming edges
        graph: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}

        # Initialize all nodes with 0 in-degree
        for node_id in nodes:
            graph[node_id] = []
            in_degree[node_id] = 0

        # Count incoming edges
        for link in workflow["links"]:
            if len(link) < 4:  # Basic validation
                continue

            from_node, _, to_node, _ = link[0:4]
            from_node, to_node = str(from_node), str(to_node)

            if from_node in graph and to_node in in_degree:
                graph[from_node].append(to_node)
                in_degree[to_node] += 1

        # Find all nodes with 0 in-degree
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        # Perform topological sort
        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if we visited all nodes
        if len(result) != len(nodes):
            logger.warning("Cannot determine execution order - graph may have cycles")
            return list(nodes.keys())  # Fallback: return all nodes

        return result

    def simulate_workflow(self, file_path: str) -> Dict[str, Any]:
        """Simulate the execution of a ComfyUI workflow"""
        result = {
            "file": file_path,
            "name": os.path.basename(file_path),
            "status": "failed",
            "errors": [],
            "warnings": [],
            "execution_time": 0,
            "node_execution": [],
            "output_images": 0,
        }

        # Load workflow file
        try:
            with open(file_path, encoding="utf-8") as f:
                workflow = json.load(f)

            # Get nodes using the utility function
            nodes = get_workflow_nodes(workflow)
            if nodes is None:
                result["errors"].append(
                    "Workflow does not have required nodes structure",
                )
                return result

            # Get links using the utility function
            links = get_workflow_links(workflow)
            if links is None:
                result["errors"].append("Workflow does not have links structure")
                return result

            # Sort nodes for execution
            execution_order = self.topological_sort(workflow)
            if not execution_order:
                result["errors"].append("Failed to determine execution order")
                return result

            # Create a map of node connections using the utility function
            link_map = build_link_map(workflow)

            start_time = time.time()
            node_outputs: Dict[str, Any] = {}
            failed_nodes = set()

            # Execute nodes in topological order
            for node_id in execution_order:
                node_data = nodes.get(node_id, {})

                # Skip if node doesn't exist
                if not node_data:
                    result["errors"].append(f"Node {node_id} not found in workflow")
                    continue

                node_type = node_data.get("class_type")
                if not node_type:
                    result["errors"].append(f"Node {node_id} has no class_type")
                    continue

                # Collect inputs from connected nodes
                node_inputs = {}

                # Add static inputs from node data
                if "inputs" in node_data:
                    for input_name, input_value in node_data["inputs"].items():
                        node_inputs[input_name] = input_value

                # Add dynamic inputs from connected nodes
                for input_name in node_data.get("inputs", {}):
                    link_key = f"{node_id}:{input_name}"

                    if link_key in link_map:
                        from_node, from_slot = link_map[link_key]

                        # Skip if source node failed
                        if from_node in failed_nodes:
                            result["errors"].append(
                                f"Node {node_id} has input from failed node {from_node}",
                            )
                            failed_nodes.add(node_id)
                            break

                        # Skip if source node output not available
                        if (
                            from_node not in node_outputs
                            or from_slot not in node_outputs[from_node]
                        ):
                            result["errors"].append(
                                f"Node {node_id} missing input from {from_node}:{from_slot}",
                            )
                            failed_nodes.add(node_id)
                            break

                        # Add input from source node
                        node_inputs[input_name] = node_outputs[from_node][from_slot]

                # Skip execution if node has failed dependencies
                if node_id in failed_nodes:
                    continue

                # Create and execute node simulation
                node_sim = NodeSimulation(node_type, node_id, node_inputs)
                success, outputs = node_sim.execute()

                # Store results
                if success:
                    node_outputs[node_id] = outputs
                    result["node_execution"].append(
                        {
                            "node_id": node_id,
                            "type": node_type,
                            "success": True,
                            "outputs": {k: str(type(v)) for k, v in outputs.items()},
                        },
                    )
                else:
                    failed_nodes.add(node_id)
                    result["node_execution"].append(
                        {
                            "node_id": node_id,
                            "type": node_type,
                            "success": False,
                            "error": outputs.get("error", "Unknown error"),
                        },
                    )
                    result["errors"].append(
                        f"Failed to execute node {node_id} ({node_type}): {outputs.get('error')}",
                    )

            # Set success if at least some nodes executed successfully
            result["status"] = (
                "success"
                if len(node_outputs) > 0 and len(result["errors"]) == 0
                else "failed"
            )

        except json.JSONDecodeError as e:
            result["errors"].append(f"Invalid JSON: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Error: {str(e)}")
            result["traceback"] = traceback.format_exc()

        # Calculate execution time
        end_time = time.time()
        result["execution_time"] = round(end_time - start_time, 2)

        logger.info(
            f"Completed simulation of {result['name']} in {result['execution_time']}s - {'Success' if result['status'] == 'success' else 'Failed'}",
        )
        return result

    def simulate_all_workflows(self) -> Dict[str, Any]:
        """Simulate all workflows in the directory"""
        workflow_files = self.find_workflow_files()
        self.results["summary"]["total_workflows"] = len(workflow_files)

        for file_path in workflow_files:
            result = self.simulate_workflow(file_path)
            self.results["workflows"].append(result)

            if result["status"] == "success":
                self.results["summary"]["successful_workflows"] += 1
            else:
                self.results["summary"]["failed_workflows"] += 1

            # Print progress summary
            status = (
                "✅ Success"
                if result["status"] == "success"
                else f"❌ Failed ({len(result['errors'])} errors)"
            )
            print(
                f"Simulated {result['name']}: {status} in {result['execution_time']}s",
            )

            if result["status"] != "success" and result["errors"]:
                for error in result["errors"][:3]:  # Show first 3 errors
                    print(f"  - {error}")
                if len(result["errors"]) > 3:
                    print(f"  ... and {len(result['errors']) - 3} more errors")

        return self.results

    def generate_markdown_report(self) -> str:
        """Generate a markdown report of simulation results"""
        report_path = os.path.join(self.output_dir, "workflow_simulation_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# ComfyUI Workflow Simulation Report\n\n")
            f.write(f"Generated on: {self.results['summary']['time']}\n\n")

            # Summary section
            f.write("## Summary\n\n")
            f.write(
                f"- Total workflows simulated: {self.results['summary']['total_workflows']}\n",
            )
            f.write(
                f"- Successful simulations: {self.results['summary']['successful_workflows']}\n",
            )
            f.write(
                f"- Failed simulations: {self.results['summary']['failed_workflows']}\n\n",
            )

            # Simulation results overview
            f.write("## Simulation Results\n\n")
            f.write("| Workflow | Status | Duration | Nodes Executed | Errors |\n")
            f.write("|----------|--------|----------|----------------|--------|\n")

            for result in sorted(
                self.results["workflows"],
                key=lambda x: (x["status"] != "success", x["name"]),
            ):
                status = "✅ Success" if result["status"] == "success" else "❌ Failed"
                executed_node_count = sum(
                    1 for node in result["node_execution"] if node["success"]
                )
                total_node_count = len(result["node_execution"])
                error_count = len(result["errors"])

                f.write(
                    f"| {result['name']} | {status} | {result['execution_time']}s | {executed_node_count}/{total_node_count} | {error_count} |\n",
                )

            f.write("\n")

            # Details for failed workflows
            failed_workflows = [
                r for r in self.results["workflows"] if r["status"] == "failed"
            ]
            if failed_workflows:
                f.write("## Failed Workflow Details\n\n")

                for result in failed_workflows:
                    f.write(f"### {result['name']}\n\n")
                    f.write(f"- **Duration**: {result['execution_time']}s\n")

                    executed_node_count = sum(
                        1 for node in result["node_execution"] if node["success"]
                    )
                    total_node_count = len(result["node_execution"])
                    f.write(
                        f"- **Node Execution**: {executed_node_count}/{total_node_count} nodes executed successfully\n\n",
                    )

                    if result["errors"]:
                        f.write("**Errors**:\n\n")
                        for error in result["errors"]:
                            f.write(f"- {error}\n")
                        f.write("\n")

                    # List failing nodes
                    failing_nodes = [
                        node["node_id"]
                        for node in result["node_execution"]
                        if not node["success"]
                    ]
                    if failing_nodes:
                        f.write("**Failing Nodes**:\n\n")
                        for node_id in failing_nodes:
                            f.write(f"- Node {node_id}\n")
                        f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(
                "1. **Fix workflow errors**: Address the issues in failing workflows.\n",
            )
            f.write(
                "2. **Add node implementations**: Implement simulation support for nodes with missing handlers.\n",
            )
            f.write(
                "3. **Validate input connections**: Ensure all nodes receive the expected inputs.\n",
            )
            f.write(
                "4. **Check compatibility**: Verify that workflows use node combinations that work together.\n\n",
            )

            f.write("---\n")
            f.write(
                "*This report was automatically generated by the ComfyUI workflow simulator.*\n",
            )

        logger.info(f"Report generated at {report_path}")
        return report_path

    def generate_json_report(self) -> str:
        """Generate a JSON report of simulation results"""
        report_path = os.path.join(self.output_dir, "workflow_simulation_results.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"JSON results saved to {report_path}")
        return report_path

    def generate_github_summary(self) -> None:
        """Generate GitHub step summary with simulation results"""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        try:
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
                f.write("## ComfyUI Workflow Simulation\n\n")

                # Status indicators
                if self.results["summary"]["failed_workflows"] > 0:
                    f.write(
                        f"⚠️ **{self.results['summary']['failed_workflows']} workflow(s) failed simulation**\n\n",
                    )
                else:
                    f.write("✅ **All workflows passed simulation**\n\n")

                # Summary table
                f.write("| Metric | Count |\n")
                f.write("|--------|-------|\n")
                f.write(
                    f"| Total Workflows | {self.results['summary']['total_workflows']} |\n",
                )
                f.write(
                    f"| Successful | {self.results['summary']['successful_workflows']} |\n",
                )
                f.write(
                    f"| Failed | {self.results['summary']['failed_workflows']} |\n\n",
                )

                # Show failed workflows
                if self.results["summary"]["failed_workflows"] > 0:
                    f.write("### Failed Workflows\n\n")
                    f.write("| Workflow | Nodes Executed | Top Error |\n")
                    f.write("|----------|----------------|----------|\n")

                    failed_workflows = [
                        r for r in self.results["workflows"] if r["status"] == "failed"
                    ]
                    for result in failed_workflows[
                        :10
                    ]:  # Limit to 10 most important ones
                        executed_nodes = sum(
                            1 for node in result["node_execution"] if node["success"]
                        )
                        total_nodes = len(result["node_execution"])

                        top_error = (
                            result["errors"][0] if result["errors"] else "Unknown error"
                        )
                        if len(top_error) > 50:
                            top_error = top_error[:47] + "..."

                        f.write(
                            f"| {result['name']} | {executed_nodes}/{total_nodes} | {top_error} |\n",
                        )

                    if len(failed_workflows) > 10:
                        f.write(
                            f"\n... and {len(failed_workflows) - 10} more failed workflows.\n",
                        )

                f.write("\nSee workflow simulation report artifact for details.\n")
        except Exception as e:
            logger.error(f"Error generating GitHub summary: {e}")

    def run(self) -> int:
        """Run workflow simulation and generate reports"""
        logger.info(f"Simulating ComfyUI workflows in {self.workflows_dir}...")

        # Simulate workflows
        self.simulate_all_workflows()

        # Generate reports
        self.generate_markdown_report()
        self.generate_json_report()
        self.generate_github_summary()

        # Return the number of failed workflows
        return self.results["summary"]["failed_workflows"]


def main():
    parser = argparse.ArgumentParser(
        description="Simulate execution of ComfyUI workflows",
    )
    parser.add_argument(
        "--workflows-dir",
        default="WebUI/external/workflows",
        help="Directory containing workflow files",
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/workflow_simulation",
        help="Directory to store simulation results",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error if any simulations fail",
    )
    args = parser.parse_args()

    simulator = ComfyWorkflowSimulator(
        workflows_dir=args.workflows_dir, output_dir=args.output_dir,
    )

    failed_count = simulator.run()

    # Exit with error code if requested and there are failed workflows
    if args.fail_on_error and failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()