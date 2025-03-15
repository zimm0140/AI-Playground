#!/usr/bin/env python
"""
ComfyUI Workflow Requirements Analyzer

This script analyzes ComfyUI workflow JSON files to determine:
1. Required models (checkpoints, LoRAs, VAEs, ControlNets)
2. Required custom nodes (extensions)
3. Required Python packages
4. Hardware requirements (GPU memory, etc.)

The analyzer supports two workflow formats:
- Traditional format with top-level "nodes" property
- Workflow API format with nodes defined in "comfyUiApiWorkflow.nodes"

It produces a report detailing the requirements for executing each workflow.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

from utils.workflow_parser import get_workflow_nodes


class WorkflowRequirementsAnalyzer:
    """Analyzer for ComfyUI workflow dependencies"""

    def __init__(self, workflows_dir, output_dir="ci_artifacts/workflow_requirements"):
        self.workflows_dir = workflows_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Known model loading nodes and their associated model types
        self.model_nodes = {
            "CheckpointLoader": "checkpoint",
            "CheckpointLoaderSimple": "checkpoint",
            "VAELoader": "vae",
            "LoraLoader": "lora",
            "ControlNetLoader": "controlnet",
            "DiffControlNetLoader": "controlnet",
            "CLIPVisionLoader": "clip_vision",
            "UnetLoader": "unet",
            "UpscaleModelLoader": "upscaler",
            "StyleModelLoader": "style_model",
            "IPAdapterModelLoader": "ip_adapter",
            "LoadLora": "lora",
            "FluxLoaderQ4": "flux_checkpoint",
            "FluxLoaderQ8": "flux_checkpoint",
        }

        # Node types that indicate the need for custom node extensions
        self.custom_node_types = {
            # Flux nodes
            "FluxNodes": "comfyui-flux-nodes",
            "FluxImplicitKSampler": "comfyui-flux-nodes",
            "FluxQ4": "comfyui-flux-nodes",
            "FluxQ8": "comfyui-flux-nodes",
            # Face swap nodes
            "FaceSwapNode": "comfyui-face-swap",
            "CopyFace": "comfyui-face-swap",
            "ReActorFaceSwap": "comfyui-face-swap",
            # IP-Adapter nodes
            "IPAdapterApply": "comfyui-ip-adapter",
            "IPAdapterModelLoader": "comfyui-ip-adapter",
            # Animation nodes
            "AnimateDiff": "comfyui-animatediff",
            "VideoLinearCFGGuidance": "comfyui-animatediff",
            "LoadVideo": "comfyui-animatediff",
            "SaveVideo": "comfyui-animatediff",
            # Line art and colorization nodes
            "LineArtProcessor": "comfyui-lineart-preprocessor",
            "ColorizeImage": "comfyui-colorization",
        }

        # Custom node package dependencies
        self.node_package_dependencies = {
            "comfyui-flux-nodes": ["torch>=2.0", "safetensors"],
            "comfyui-face-swap": ["opencv-python", "insightface", "onnxruntime-gpu"],
            "comfyui-ip-adapter": ["transformers", "diffusers", "accelerate"],
            "comfyui-animatediff": ["torch>=2.0", "einops", "opencv-python"],
            "comfyui-lineart-preprocessor": ["opencv-python", "numpy"],
            "comfyui-colorization": ["pytorch-lightning", "diffusers"],
        }

        # Approximate GPU memory requirements by node type
        self.gpu_memory_estimates = {
            "CheckpointLoader": {
                "SD1.5": 2.0,  # GB
                "SDXL": 6.0,  # GB
                "SD3": 8.0,  # GB
            },
            "VAELoader": 0.5,  # GB
            "ControlNetLoader": 1.0,  # GB per ControlNet
            "LoraLoader": 0.2,  # GB per LoRA
            "KSampler": {
                "base": 1.0,  # GB
                "per_batch": 0.5,  # GB per batch size
            },
            "UpscaleModelLoader": 1.0,  # GB
            "FaceRestoration": 1.0,  # GB
            "AnimateDiff": 2.0,  # GB
            "IPAdapterModelLoader": 1.0,  # GB
        }

        # Results storage
        self.results = {
            "summary": {
                "total_workflows": 0,
                "analyzed_workflows": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "workflows": [],
            "aggregate": {
                "models": defaultdict(int),
                "custom_nodes": defaultdict(int),
                "python_packages": defaultdict(int),
                "memory_requirements": {
                    "min": 0,
                    "max": 0,
                    "workflows_by_memory": defaultdict(list),
                },
            },
        }

    def find_workflow_files(self):
        """Find all workflow JSON files in the specified directory"""
        return glob.glob(os.path.join(self.workflows_dir, "*.json"))

    def analyze_workflow(self, workflow_file):
        """
        Analyze a single workflow file to determine its requirements.

        Args:
            workflow_file (str): Path to the workflow JSON file

        Returns:
            dict: Analysis results including models, custom nodes, and memory requirements
        """
        filename = os.path.basename(workflow_file)

        # Initialize result structure
        workflow_result = self._initialize_workflow_result(filename, workflow_file)

        # Load workflow JSON
        workflow, is_valid = self._load_workflow_json(workflow_file, workflow_result)
        if not is_valid:
            return workflow_result

        # Get nodes using the utility function
        nodes = get_workflow_nodes(workflow)
        if not nodes:
            workflow_result["errors"].append("Workflow does not have required structure")
            return workflow_result

        # Process workflow nodes to extract requirements
        workflow_result = self._process_workflow_nodes(nodes, workflow_result)

        # Mark as successfully analyzed
        workflow_result["is_analyzed"] = True
        return workflow_result

    def _initialize_workflow_result(self, filename, workflow_file):
        """Initialize the result structure for a workflow analysis."""
        return {
            "filename": filename,
            "path": workflow_file,
            "models": {
                "checkpoint": [],
                "vae": [],
                "lora": [],
                "controlnet": [],
                "clip_vision": [],
                "unet": [],
                "upscaler": [],
                "style_model": [],
                "ip_adapter": [],
                "flux_checkpoint": [],
            },
            "custom_nodes": [],
            "python_packages": [],
            "memory_required": {
                "min": 4,  # Minimum 4GB as base requirement
                "recommended": 8,  # Recommended 8GB as base
            },
            "is_analyzed": False,
            "errors": [],
        }

    def _load_workflow_json(self, workflow_file, workflow_result):
        """Load and parse workflow JSON file."""
        try:
            with open(workflow_file, encoding="utf-8") as f:
                workflow = json.load(f)

            # Check if the workflow has the expected structure
            if not isinstance(workflow, dict):
                workflow_result["errors"].append("Workflow is not a valid JSON object")
                return workflow, False

            return workflow, True

        except json.JSONDecodeError as e:
            workflow_result["errors"].append(f"Invalid JSON: {str(e)}")
            return None, False
        except Exception as e:
            workflow_result["errors"].append(f"Error reading file: {str(e)}")
            return None, False

    def _process_workflow_nodes(self, nodes, workflow_result):
        """Process workflow nodes to extract requirements and estimate memory needs."""
        # Initialize tracking variables
        checkpoint_sd_type = "SD1.5"  # Default to SD1.5 for memory estimation
        batch_size = 1  # Default batch size
        has_upscaling = False
        control_net_count = 0
        lora_count = 0

        # Process each node
        for _node_id, node_data in nodes.items():
            if "class_type" not in node_data:
                continue

            class_type = node_data["class_type"]

            # Process model loading nodes
            if class_type in self.model_nodes:
                model_type = self.model_nodes[class_type]
                model_path = self._extract_model_path(node_data)

                # Update workflow result with model information
                if model_path and model_path not in workflow_result["models"][model_type]:
                    workflow_result["models"][model_type].append(model_path)

                # Update tracking variables for memory estimation
                if model_type == "controlnet":
                    control_net_count += 1
                elif model_type == "lora":
                    lora_count += 1

                # Detect model type for memory estimation
                if model_type == "checkpoint" and model_path:
                    checkpoint_sd_type = self._determine_checkpoint_type(model_path)

            # Process custom node extensions
            if class_type in self.custom_node_types:
                self._add_custom_node_requirements(class_type, workflow_result)

            # Check for batch size in samplers
            batch_size = max(batch_size, self._extract_batch_size(node_data, batch_size))

            # Check for upscalers
            if "Upscale" in class_type or class_type in ["UpscaleModelLoader"]:
                has_upscaling = True

        # Calculate memory requirements
        self._calculate_memory_requirements(
            workflow_result,
            checkpoint_sd_type,
            control_net_count,
            lora_count,
            batch_size,
            has_upscaling,
        )

        return workflow_result

    def _extract_model_path(self, node_data):
        """Extract model path from node inputs."""
        if "inputs" not in node_data:
            return None

        inputs = node_data["inputs"]
        model_path_keys = [
            "ckpt_name",
            "model_name",
            "vae_name",
            "lora_name",
            "control_net_name",
            "model",
            "swap_model",
            "face_restore_model",
        ]

        for key in model_path_keys:
            if key in inputs:
                return inputs[key]

        return None

    def _determine_checkpoint_type(self, model_path):
        """Determine the checkpoint type based on model path."""
        if "xl" in model_path.lower():
            return "SDXL"
        elif "sd3" in model_path.lower() or "sd_3" in model_path.lower():
            return "SD3"
        return "SD1.5"

    def _add_custom_node_requirements(self, class_type, workflow_result):
        """Add custom node and its dependencies to requirements."""
        custom_node = self.custom_node_types[class_type]

        if custom_node not in workflow_result["custom_nodes"]:
            workflow_result["custom_nodes"].append(custom_node)

            # Add any required Python packages for this custom node
            if custom_node in self.node_package_dependencies:
                for package in self.node_package_dependencies[custom_node]:
                    if package not in workflow_result["python_packages"]:
                        workflow_result["python_packages"].append(package)

    def _extract_batch_size(self, node_data, default_batch_size):
        """Extract batch size from node data."""
        if "inputs" in node_data and "batch_size" in node_data["inputs"]:
            try:
                batch_input = node_data["inputs"]["batch_size"]
                if isinstance(batch_input, (int, float)):
                    return max(default_batch_size, int(batch_input))
            except (ValueError, TypeError):
                pass
        return default_batch_size

    def _calculate_memory_requirements(
        self,
        workflow_result,
        checkpoint_sd_type,
        control_net_count,
        lora_count,
        batch_size,
        has_upscaling,
    ):
        """Calculate GPU memory requirements based on detected models and settings."""
        min_memory = 4.0  # Base minimum (comfyUI overhead + basic model)
        recommended_memory = 8.0  # Base recommended

        # Add checkpoint memory based on type
        if checkpoint_sd_type in self.gpu_memory_estimates["CheckpointLoader"]:
            checkpoint_memory = self.gpu_memory_estimates["CheckpointLoader"][checkpoint_sd_type]
            min_memory += checkpoint_memory
            recommended_memory += checkpoint_memory

        # Add memory for ControlNets
        if control_net_count > 0:
            control_net_memory = control_net_count * self.gpu_memory_estimates["ControlNetLoader"]
            min_memory += control_net_memory
            recommended_memory += control_net_memory * 1.5  # Extra buffer for recommended

        # Add memory for LoRAs
        if lora_count > 0:
            lora_memory = lora_count * self.gpu_memory_estimates["LoraLoader"]
            min_memory += lora_memory
            recommended_memory += lora_memory

        # Add memory for sampling with batch size
        if "KSampler" in self.gpu_memory_estimates:
            sampler_memory = (
                self.gpu_memory_estimates["KSampler"]["base"]
                + (batch_size - 1) * self.gpu_memory_estimates["KSampler"]["per_batch"]
            )
            min_memory += sampler_memory
            recommended_memory += sampler_memory * 1.5  # Extra buffer for larger images

        # Add upscaling memory if needed
        if has_upscaling and "UpscaleModelLoader" in self.gpu_memory_estimates:
            upscale_memory = self.gpu_memory_estimates["UpscaleModelLoader"]
            min_memory += upscale_memory
            recommended_memory += upscale_memory * 2  # Upscaling large images needs more memory

        # Round up to whole GB and update result
        workflow_result["memory_required"]["min"] = round(min_memory)
        workflow_result["memory_required"]["recommended"] = round(recommended_memory)

    def analyze_all_workflows(self):
        """
        Analyze all workflow files in the specified directory.

        Returns:
            dict: Analysis results for all workflows
        """
        all_workflow_files = self.find_workflow_files()
        self.results["summary"]["total_workflows"] = len(all_workflow_files)

        # Sort workflow files for consistent execution
        workflow_files = sorted(all_workflow_files)

        # Skip any readme.md files that may have a .json extension
        analyzed_files = [f for f in workflow_files if os.path.basename(f).lower() != "readme.md"]

        for file_path in analyzed_files:
            workflow_result = self.analyze_workflow(file_path)
            self.results["workflows"].append(workflow_result)

            if workflow_result["is_analyzed"]:
                self.results["summary"]["analyzed_workflows"] += 1

        return self.results

    def aggregate_results(self, results):
        """
        Aggregate results from multiple workflows for summary.

        Args:
            results (dict): Raw analysis results from all workflows

        Returns:
            dict: Aggregated results
        """
        # Deep copy to avoid modifying original
        results = json.loads(json.dumps(results))

        # Process each analyzed workflow
        for workflow in results["workflows"]:
            if not workflow["is_analyzed"]:
                continue

            # Aggregate memory requirements
            self._aggregate_memory_requirements(results, workflow)

            # Aggregate model information
            self._aggregate_models(results, workflow)

            # Aggregate custom nodes
            self._aggregate_custom_nodes(results, workflow)

            # Aggregate Python packages
            self._aggregate_python_packages(results, workflow)

        return results

    def _aggregate_memory_requirements(self, results, workflow):
        """Aggregate memory requirements across workflows."""
        min_memory = workflow["memory_required"]["min"]

        # Initialize workflows_by_memory if needed
        if "workflows_by_memory" not in results["aggregate"]["memory_requirements"]:
            results["aggregate"]["memory_requirements"]["workflows_by_memory"] = {}

        # Group workflows by memory requirement
        if min_memory not in results["aggregate"]["memory_requirements"]["workflows_by_memory"]:
            results["aggregate"]["memory_requirements"]["workflows_by_memory"][min_memory] = []

        results["aggregate"]["memory_requirements"]["workflows_by_memory"][min_memory].append(
            workflow["filename"],
        )

        # Track min/max memory across all workflows
        results["aggregate"]["memory_requirements"]["max"] = max(
            results["aggregate"]["memory_requirements"]["max"],
            min_memory,
        )

        # Update minimum memory requirement if needed
        if (
            results["aggregate"]["memory_requirements"]["min"] == 0
            or min_memory < results["aggregate"]["memory_requirements"]["min"]
        ):
            results["aggregate"]["memory_requirements"]["min"] = min_memory

    def _aggregate_models(self, results, workflow):
        """Aggregate model information across workflows."""
        for model_type, models in workflow["models"].items():
            for model in models:
                # Initialize model type dictionary if needed
                if model_type not in results["aggregate"]["models"]:
                    results["aggregate"]["models"][model_type] = {}

                # Initialize model list if needed
                if model not in results["aggregate"]["models"][model_type]:
                    results["aggregate"]["models"][model_type][model] = []

                # Add workflow to the list for this model
                results["aggregate"]["models"][model_type][model].append(
                    workflow["filename"],
                )

    def _aggregate_custom_nodes(self, results, workflow):
        """Aggregate custom node information across workflows."""
        for node in workflow["custom_nodes"]:
            if node not in results["aggregate"]["custom_nodes"]:
                results["aggregate"]["custom_nodes"][node] = []

            results["aggregate"]["custom_nodes"][node].append(workflow["filename"])

    def _aggregate_python_packages(self, results, workflow):
        """Aggregate Python package information across workflows."""
        for package in workflow["python_packages"]:
            if package not in results["aggregate"]["python_packages"]:
                results["aggregate"]["python_packages"][package] = []

            results["aggregate"]["python_packages"][package].append(
                workflow["filename"],
            )

    def save_results_to_json(self, results, output_file="workflow_requirements_results.json"):
        """Save the results to a JSON file"""
        output_path = os.path.join(self.output_dir, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return output_path

    def generate_markdown_report(self, results):
        """
        Generate a Markdown report from the analysis results.

        Args:
            results (dict): Analysis results from all workflows

        Returns:
            str: Markdown report content
        """
        report = []

        # Add title and summary
        self._add_report_header(report, results)

        # Add memory requirements section
        self._add_memory_requirements_section(report, results)

        # Add model usage section
        self._add_model_usage_section(report, results)

        # Add custom node usage section
        self._add_custom_node_section(report, results)

        # Add Python package requirements section
        self._add_python_package_section(report, results)

        # Add individual workflow details section
        self._add_workflow_details_section(report, results)

        # Join all sections with newlines
        return "\n".join(report)

    def _add_report_header(self, report, results):
        """Add title and summary section to the report."""
        report.append("# ComfyUI Workflow Requirements Analysis")
        report.append("")
        report.append(f"Analysis run: {results['summary']['time']}")
        report.append("")
        report.append("## Summary")
        report.append("")
        report.append(f"Total workflows: {results['summary']['total_workflows']}")
        report.append(f"Successfully analyzed: {results['summary']['analyzed_workflows']}")
        report.append("")

    def _add_memory_requirements_section(self, report, results):
        """Add memory requirements section to the report."""
        if results["aggregate"]["memory_requirements"]["max"] > 0:
            report.append("## Memory Requirements")
            report.append("")
            report.append(
                f"Minimum memory needed: {results['aggregate']['memory_requirements']['min']}GB",
            )
            report.append(
                f"Maximum memory needed: {results['aggregate']['memory_requirements']['max']}GB",
            )
            report.append("")
            report.append("### Workflows by Memory Requirement")
            report.append("")
            report.append("| Memory (GB) | Workflows |")
            report.append("| ----------- | --------- |")

            for memory, workflows in sorted(
                results["aggregate"]["memory_requirements"]["workflows_by_memory"].items(),
            ):
                report.append(f"| {memory} | {len(workflows)} |")

            report.append("")

    def _add_model_usage_section(self, report, results):
        """Add model usage section to the report."""
        if any(results["aggregate"]["models"].values()):
            report.append("## Model Usage")
            report.append("")
            report.append("| Model | Workflows |")
            report.append("| ----- | --------- |")

            for model_type, models in results["aggregate"]["models"].items():
                for model, workflows in models.items():
                    report.append(f"| {model} ({model_type}) | {', '.join(workflows)} |")

            report.append("")

    def _add_custom_node_section(self, report, results):
        """Add custom node usage section to the report."""
        if results["aggregate"]["custom_nodes"]:
            report.append("## Custom Node Extensions")
            report.append("")
            report.append("| Extension | Workflows |")
            report.append("| --------- | --------- |")

            for node, workflows in results["aggregate"]["custom_nodes"].items():
                report.append(f"| {node} | {', '.join(workflows)} |")

            report.append("")

    def _add_python_package_section(self, report, results):
        """Add Python package requirements section to the report."""
        if results["aggregate"]["python_packages"]:
            report.append("## Python Package Requirements")
            report.append("")
            report.append("| Package | Required By |")
            report.append("| ------- | ----------- |")

            for package, workflows in results["aggregate"]["python_packages"].items():
                report.append(f"| {package} | {', '.join(workflows)} |")

            report.append("")

    def _add_workflow_details_section(self, report, results):
        """Add individual workflow details section to the report."""
        report.append("## Individual Workflow Details")
        report.append("")

        for workflow in results["workflows"]:
            report.append(f"### {workflow['filename']}")
            report.append("")

            if workflow["is_analyzed"]:
                self._add_workflow_model_details(report, workflow)
                self._add_workflow_custom_node_details(report, workflow)
                self._add_workflow_python_package_details(report, workflow)
                self._add_workflow_memory_details(report, workflow)
            else:
                self._add_workflow_error_details(report, workflow)

    def _add_workflow_model_details(self, report, workflow):
        """Add model details for a workflow to the report."""
        # Add models
        if any(workflow["models"].values()):
            report.append("**Required Models:**\n")
            for model_type, models in workflow["models"].items():
                if models:
                    report.append(
                        f"- {model_type.capitalize()}: {', '.join(models)}",
                    )
            report.append("")

    def _add_workflow_custom_node_details(self, report, workflow):
        """Add custom node details for a workflow to the report."""
        # Add custom nodes
        if workflow["custom_nodes"]:
            report.append("**Required Custom Nodes:**\n")
            for node in workflow["custom_nodes"]:
                report.append(f"- {node}")
            report.append("")

    def _add_workflow_python_package_details(self, report, workflow):
        """Add Python package details for a workflow to the report."""
        # Add Python packages
        if workflow["python_packages"]:
            report.append("**Python Package Dependencies:**\n")
            for package in workflow["python_packages"]:
                report.append(f"- {package}")
            report.append("")

    def _add_workflow_memory_details(self, report, workflow):
        """Add memory requirement details for a workflow to the report."""
        # Add memory requirements
        report.append("**Memory Requirements:**\n")
        report.append(f"- Minimum: {workflow['memory_required']['min']}GB")
        report.append(
            f"- Recommended: {workflow['memory_required']['recommended']}GB",
        )
        report.append("")

    def _add_workflow_error_details(self, report, workflow):
        """Add error details for a failed workflow analysis to the report."""
        report.append("❌ **Analysis failed**\n")
        for error in workflow["errors"]:
            report.append(f"- Error: {error}")
        report.append("")

    def save_markdown_report(self, report, output_file="workflow_requirements_report.md"):
        """Save the Markdown report to a file"""
        output_path = os.path.join(self.output_dir, output_file)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        return output_path


def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(
        description="Analyze ComfyUI workflow requirements",
    )
    parser.add_argument(
        "--workflows-dir",
        required=True,
        help="Directory containing workflow JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/workflow_requirements",
        help="Directory to write output files",
    )

    args = parser.parse_args()

    analyzer = WorkflowRequirementsAnalyzer(args.workflows_dir, args.output_dir)
    results = analyzer.analyze_all_workflows()
    aggregated_results = analyzer.aggregate_results(results)

    json_file = analyzer.save_results_to_json(aggregated_results)
    print(f"Saved JSON results to {json_file}")

    markdown_report = analyzer.generate_markdown_report(aggregated_results)
    md_file = analyzer.save_markdown_report(markdown_report)
    print(f"Saved Markdown report to {md_file}")


if __name__ == "__main__":
    main()