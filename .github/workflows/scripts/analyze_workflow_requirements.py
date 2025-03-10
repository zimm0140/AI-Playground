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

import os
import json
import argparse
import glob
from datetime import datetime
from collections import defaultdict

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
            "FluxLoaderQ8": "flux_checkpoint"
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
            "ColorizeImage": "comfyui-colorization"
        }
        
        # Custom node package dependencies
        self.node_package_dependencies = {
            "comfyui-flux-nodes": ["torch>=2.0", "safetensors"],
            "comfyui-face-swap": ["opencv-python", "insightface", "onnxruntime-gpu"],
            "comfyui-ip-adapter": ["transformers", "diffusers", "accelerate"],
            "comfyui-animatediff": ["torch>=2.0", "einops", "opencv-python"],
            "comfyui-lineart-preprocessor": ["opencv-python", "numpy"],
            "comfyui-colorization": ["pytorch-lightning", "diffusers"]
        }
        
        # Approximate GPU memory requirements by node type
        self.gpu_memory_estimates = {
            "CheckpointLoader": {
                "SD1.5": 2.0,  # GB
                "SDXL": 6.0,   # GB
                "SD3": 8.0     # GB
            },
            "VAELoader": 0.5,  # GB
            "ControlNetLoader": 1.0,  # GB per ControlNet
            "LoraLoader": 0.2,  # GB per LoRA
            "KSampler": {
                "base": 1.0,  # GB
                "per_batch": 0.5  # GB per batch size
            },
            "UpscaleModelLoader": 1.0,  # GB
            "FaceRestoration": 1.0,  # GB
            "AnimateDiff": 2.0,  # GB
            "IPAdapterModelLoader": 1.0  # GB
        }
        
        # Results storage
        self.results = {
            "summary": {
                "total_workflows": 0,
                "analyzed_workflows": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "workflows": [],
            "aggregate": {
                "models": defaultdict(int),
                "custom_nodes": defaultdict(int),
                "python_packages": defaultdict(int),
                "memory_requirements": {
                    "min": 0,
                    "max": 0,
                    "workflows_by_memory": defaultdict(list)
                }
            }
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
        workflow_result = {
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
                "flux_checkpoint": []
            },
            "custom_nodes": [],
            "python_packages": [],
            "memory_required": {
                "min": 4,  # Minimum 4GB as base requirement
                "recommended": 8  # Recommended 8GB as base
            },
            "is_analyzed": False,
            "errors": []
        }
        
        # Load and parse the workflow file
        try:
            with open(workflow_file, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            workflow_result["errors"].append(f"Invalid JSON: {str(e)}")
            return workflow_result
        except Exception as e:
            workflow_result["errors"].append(f"Error reading file: {str(e)}")
            return workflow_result
        
        # Check if the workflow has the expected structure
        if not isinstance(workflow, dict):
            workflow_result["errors"].append("Workflow is not a valid JSON object")
            return workflow_result
        
        # Get nodes using the utility function
        nodes = get_workflow_nodes(workflow)
        
        if not nodes:
            workflow_result["errors"].append("Workflow does not have required structure")
            return workflow_result
        
        # Extract model requirements
        checkpoint_sd_type = "SD1.5"  # Default to SD1.5 for memory estimation
        batch_size = 1  # Default batch size
        has_upscaling = False
        control_net_count = 0
        lora_count = 0
        
        for node_id, node_data in nodes.items():
            if "class_type" not in node_data:
                continue
                
            class_type = node_data["class_type"]
            
            # Check for model loading nodes
            if class_type in self.model_nodes:
                model_type = self.model_nodes[class_type]
                
                # Extract model path/name if available
                model_path = None
                if "inputs" in node_data:
                    inputs = node_data["inputs"]
                    
                    # Different model nodes have different input parameter names
                    if "ckpt_name" in inputs:
                        model_path = inputs["ckpt_name"]
                    elif "model_name" in inputs:
                        model_path = inputs["model_name"]
                    elif "vae_name" in inputs:
                        model_path = inputs["vae_name"]
                    elif "lora_name" in inputs:
                        model_path = inputs["lora_name"]
                    elif "control_net_name" in inputs:
                        model_path = inputs["control_net_name"]
                    elif "model" in inputs:
                        model_path = inputs["model"]
                    elif "swap_model" in inputs:
                        model_path = inputs["swap_model"]
                    elif "face_restore_model" in inputs:
                        model_path = inputs["face_restore_model"]
                
                # Add to the appropriate model type list if not already present
                if model_path and model_path not in workflow_result["models"][model_type]:
                    workflow_result["models"][model_type].append(model_path)
                
                # Count specific model types for memory estimation
                if model_type == "controlnet":
                    control_net_count += 1
                elif model_type == "lora":
                    lora_count += 1
                
                # Detect model type for memory estimation
                if model_type == "checkpoint" and model_path:
                    if "xl" in model_path.lower():
                        checkpoint_sd_type = "SDXL"
                    elif "sd3" in model_path.lower() or "sd_3" in model_path.lower():
                        checkpoint_sd_type = "SD3"
            
            # Check for custom nodes
            if class_type in self.custom_node_types:
                custom_node = self.custom_node_types[class_type]
                if custom_node not in workflow_result["custom_nodes"]:
                    workflow_result["custom_nodes"].append(custom_node)
                    
                    # Add associated Python packages
                    if custom_node in self.node_package_dependencies:
                        for package in self.node_package_dependencies[custom_node]:
                            if package not in workflow_result["python_packages"]:
                                workflow_result["python_packages"].append(package)
            
            # Check for batch size
            if "inputs" in node_data and "batch_size" in node_data["inputs"]:
                try:
                    batch_size = max(batch_size, int(node_data["inputs"]["batch_size"]))
                except (ValueError, TypeError):
                    pass
            
            # Check for upscaling operations
            if "Upscale" in class_type or class_type in ["UpscaleModelLoader"]:
                has_upscaling = True
        
        # Calculate memory requirements
        memory_required = 4.0  # Base memory requirement in GB
        
        # Add checkpoint memory
        if checkpoint_sd_type in self.gpu_memory_estimates["CheckpointLoader"]:
            memory_required += self.gpu_memory_estimates["CheckpointLoader"][checkpoint_sd_type]
        
        # Add ControlNet memory
        memory_required += control_net_count * self.gpu_memory_estimates["ControlNetLoader"]
        
        # Add LoRA memory
        memory_required += lora_count * self.gpu_memory_estimates["LoraLoader"]
        
        # Add KSampler memory based on batch size
        if "KSampler" in self.gpu_memory_estimates:
            memory_required += self.gpu_memory_estimates["KSampler"]["base"]
            memory_required += (batch_size - 1) * self.gpu_memory_estimates["KSampler"]["per_batch"]
        
        # Add upscaling memory if needed
        if has_upscaling and "UpscaleModelLoader" in self.gpu_memory_estimates:
            memory_required += self.gpu_memory_estimates["UpscaleModelLoader"]
        
        # Update memory requirements
        workflow_result["memory_required"]["min"] = max(workflow_result["memory_required"]["min"], int(memory_required))
        workflow_result["memory_required"]["recommended"] = max(workflow_result["memory_required"]["recommended"], int(memory_required * 1.5))
        
        # Mark as successfully analyzed
        workflow_result["is_analyzed"] = True
        
        return workflow_result

    def analyze_all_workflows(self):
        """
        Analyze all workflow files in the specified directory.
        
        Returns:
            dict: Analysis results for all workflows
        """
        workflow_files = []
        
        # Find all JSON files in the workflows directory
        for file_path in glob.glob(os.path.join(self.workflows_dir, "*.json")):
            if os.path.isfile(file_path) and file_path.endswith('.json'):
                workflow_files.append(file_path)
        
        # Skip README.md and other non-workflow files
        workflow_files = [f for f in workflow_files if os.path.basename(f).lower() != "readme.md"]
        
        # Initialize results
        results = {
            "summary": {
                "total_workflows": len(workflow_files),
                "analyzed_workflows": 0,
                "failed_workflows": 0
            },
            "workflows": [],
            "aggregate": {
                "models": {},
                "custom_nodes": {},
                "python_packages": {},
                "memory_requirements": {
                    "min": 0,
                    "max": 0,
                    "workflows_by_memory": {}
                }
            }
        }
        
        # Analyze each workflow
        print(f"Analyzing ComfyUI workflows in {self.workflows_dir}...")
        for workflow_file in sorted(workflow_files):
            workflow_result = self.analyze_workflow(workflow_file)
            results["workflows"].append(workflow_result)
            
            # Update summary
            if workflow_result["is_analyzed"]:
                results["summary"]["analyzed_workflows"] += 1
                status = "✅ Analyzed"
            else:
                results["summary"]["failed_workflows"] += 1
                status = f"❌ Failed ({len(workflow_result['errors'])} errors)"
            
            print(f"Analyzed {os.path.basename(workflow_file)}: {status}")
        
        # Aggregate results
        for workflow in results["workflows"]:
            if not workflow["is_analyzed"]:
                continue
                
            # Aggregate models
            for model_type, models in workflow["models"].items():
                for model in models:
                    if model_type not in results["aggregate"]["models"]:
                        results["aggregate"]["models"][model_type] = {}
                    
                    if model not in results["aggregate"]["models"][model_type]:
                        results["aggregate"]["models"][model_type][model] = []
                    
                    results["aggregate"]["models"][model_type][model].append(workflow["filename"])
            
            # Aggregate custom nodes
            for node in workflow["custom_nodes"]:
                if node not in results["aggregate"]["custom_nodes"]:
                    results["aggregate"]["custom_nodes"][node] = []
                
                results["aggregate"]["custom_nodes"][node].append(workflow["filename"])
            
            # Aggregate Python packages
            for package in workflow["python_packages"]:
                if package not in results["aggregate"]["python_packages"]:
                    results["aggregate"]["python_packages"][package] = []
                
                results["aggregate"]["python_packages"][package].append(workflow["filename"])
            
            # Aggregate memory requirements
            min_memory = workflow["memory_required"]["min"]
            results["aggregate"]["memory_requirements"]["min"] = min(
                results["aggregate"]["memory_requirements"]["min"] or min_memory,
                min_memory
            )
            results["aggregate"]["memory_requirements"]["max"] = max(
                results["aggregate"]["memory_requirements"]["max"],
                workflow["memory_required"]["recommended"]
            )
            
            memory_key = f"{min_memory}GB"
            if memory_key not in results["aggregate"]["memory_requirements"]["workflows_by_memory"]:
                results["aggregate"]["memory_requirements"]["workflows_by_memory"][memory_key] = []
            
            results["aggregate"]["memory_requirements"]["workflows_by_memory"][memory_key].append(workflow["filename"])
        
        return results

    def generate_markdown_report(self, results):
        """
        Generate a markdown report from the analysis results.
        
        Args:
            results (dict): Analysis results
            
        Returns:
            str: Markdown report
        """
        report = []
        
        # Header
        report.append("# ComfyUI Workflow Requirements Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        report.append("## Summary\n")
        report.append(f"- Total workflows analyzed: {results['summary']['total_workflows']}")
        report.append(f"- Successfully analyzed: {results['summary']['analyzed_workflows']}\n")
        
        # Aggregate Requirements
        report.append("## Aggregate Requirements\n")
        
        # Models
        report.append("### Models\n")
        report.append("| Model | Workflows |")
        report.append("|-------|----------|")
        
        for model_type, models in results["aggregate"]["models"].items():
            for model, workflows in models.items():
                report.append(f"| {model} ({model_type}) | {', '.join(workflows)} |")
        
        report.append("")
        
        # Custom Nodes
        report.append("### Custom Nodes\n")
        report.append("| Custom Node Extension | Workflows |")
        report.append("|----------------------|----------|")
        
        for node, workflows in results["aggregate"]["custom_nodes"].items():
            report.append(f"| {node} | {', '.join(workflows)} |")
        
        report.append("")
        
        # Python Packages
        report.append("### Python Packages\n")
        report.append("| Package | Workflows |")
        report.append("|---------|----------|")
        
        for package, workflows in results["aggregate"]["python_packages"].items():
            report.append(f"| {package} | {', '.join(workflows)} |")
        
        report.append("")
        
        # Memory Requirements
        report.append("### Memory Requirements\n")
        
        min_memory = results["aggregate"]["memory_requirements"]["min"]
        max_memory = results["aggregate"]["memory_requirements"]["max"]
        
        report.append(f"- Minimum memory required: {min_memory/1024:.1f}GB")
        report.append(f"- Maximum memory required: {max_memory/1024:.1f}GB\n")
        
        report.append("| Memory Requirement | Workflows |")
        report.append("|-------------------|----------|")
        
        for memory, workflows in results["aggregate"]["memory_requirements"]["workflows_by_memory"].items():
            report.append(f"| {memory} | {', '.join(workflows)} |")
        
        report.append("")
        
        # Individual Workflow Requirements
        report.append("## Individual Workflow Requirements\n")
        
        for workflow in results["workflows"]:
            report.append(f"### {workflow['filename']}\n")
            
            if workflow["is_analyzed"]:
                report.append("✅ **Successfully analyzed**\n")
                
                # Models
                if any(workflow["models"].values()):
                    report.append("**Required Models:**\n")
                    for model_type, models in workflow["models"].items():
                        if models:
                            report.append(f"- {model_type.capitalize()}: {', '.join(models)}")
                    report.append("")
                
                # Custom Nodes
                if workflow["custom_nodes"]:
                    report.append("**Required Custom Nodes:**\n")
                    for node in workflow["custom_nodes"]:
                        report.append(f"- {node}")
                    report.append("")
                
                # Python Packages
                if workflow["python_packages"]:
                    report.append("**Required Python Packages:**\n")
                    for package in workflow["python_packages"]:
                        report.append(f"- {package}")
                    report.append("")
                
                # Memory Requirements
                report.append("**Memory Requirements:**\n")
                report.append(f"- Minimum: {workflow['memory_required']['min']}GB")
                report.append(f"- Recommended: {workflow['memory_required']['recommended']}GB")
                report.append("")
            else:
                report.append("❌ **Analysis failed**\n")
                for error in workflow["errors"]:
                    report.append(f"- Error: {error}")
                report.append("")
        
        return "\n".join(report)

    def save_results(self, results):
        """
        Save analysis results to output files.
        
        Args:
            results (dict): Analysis results
        """
        # Save JSON results
        json_output_path = os.path.join(self.output_dir, "workflow_requirements_results.json")
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Generate and save markdown report
        markdown_report = self.generate_markdown_report(results)
        markdown_output_path = os.path.join(self.output_dir, "workflow_requirements_report.md")
        with open(markdown_output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        print(f"Report generated at {markdown_output_path}")
        print(f"JSON results saved to {json_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ComfyUI workflow requirements")
    parser.add_argument("--workflows-dir", help="Directory containing workflow JSON files")
    parser.add_argument("--output-dir", default="workflow_requirements_output", help="Output directory for reports")
    
    args = parser.parse_args()
    
    analyzer = WorkflowRequirementsAnalyzer(args.workflows_dir, args.output_dir)
    results = analyzer.analyze_all_workflows()
    analyzer.save_results(results)


if __name__ == "__main__":
    main() 