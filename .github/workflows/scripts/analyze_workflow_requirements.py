#!/usr/bin/env python
"""
ComfyUI Workflow Requirements Analyzer

This script analyzes ComfyUI workflow JSON files to determine:
1. Required models (checkpoints, LoRAs, VAEs, ControlNets)
2. Required custom nodes (extensions)
3. Required Python packages
4. Hardware requirements (GPU memory, etc.)

It produces a report detailing the requirements for executing each workflow.
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from collections import defaultdict


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
                "SD1.5": 4, # 4GB base model
                "SDXL": 8,  # 8GB base model
                "SD3": 12   # 12GB base model (estimated)
            },
            "VAELoader": 0.5,  # 0.5GB for VAE
            "KSampler": {
                "base": 2,    # 2GB overhead for sampling
                "per_batch": 0.5 # Additional 0.5GB per batch size
            },
            "AnimateDiff": 2,  # 2GB overhead for animation
            "ControlNetApply": 1, # 1GB per ControlNet
            "LoraLoader": 0.2,    # 0.2GB per LoRA
            "UpscaleImage": 2      # 2GB for upscaling
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

    def analyze_workflow(self, file_path):
        """Analyze a single workflow file for its requirements"""
        filename = os.path.basename(file_path)
        
        workflow_result = {
            "filename": filename,
            "path": file_path,
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
                "min": 4,  # Minimum 4GB estimated baseline
                "recommended": 8  # Default recommendation
            },
            "is_analyzed": False,
            "errors": []
        }
        
        # Try to parse the JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            workflow_result["errors"].append(f"Invalid JSON: {str(e)}")
            return workflow_result
        except Exception as e:
            workflow_result["errors"].append(f"Error reading file: {str(e)}")
            return workflow_result
        
        # Check if the workflow has the expected structure - accept either top level nodes or comfyUiApiWorkflow.nodes
        if not isinstance(workflow, dict):
            workflow_result["errors"].append("Workflow is not a valid JSON object")
            return workflow_result
        
        # Look for nodes in either location
        nodes = None
        if "nodes" in workflow:
            nodes = workflow["nodes"]
        elif "comfyUiApiWorkflow" in workflow and isinstance(workflow["comfyUiApiWorkflow"], dict) and "nodes" in workflow["comfyUiApiWorkflow"]:
            nodes = workflow["comfyUiApiWorkflow"]["nodes"]
        
        if not nodes:
            workflow_result["errors"].append("Workflow does not have required nodes structure")
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
                
            node_type = node_data["class_type"]
            
            # Check if this is a model loading node
            if node_type in self.model_nodes:
                model_type = self.model_nodes[node_type]
                
                # Try to extract the model name if available
                model_name = None
                if "inputs" in node_data and isinstance(node_data["inputs"], dict):
                    if "ckpt_name" in node_data["inputs"]:
                        model_name = node_data["inputs"]["ckpt_name"]
                    elif "model_name" in node_data["inputs"]:
                        model_name = node_data["inputs"]["model_name"]
                    elif "lora_name" in node_data["inputs"]:
                        model_name = node_data["inputs"]["lora_name"]
                    elif "vae_name" in node_data["inputs"]:
                        model_name = node_data["inputs"]["vae_name"]
                
                # Add model to requirements
                if model_name and model_name not in workflow_result["models"][model_type]:
                    workflow_result["models"][model_type].append(model_name)
                    
                    # Check for SDXL or SD3 models for memory estimation
                    if model_type == "checkpoint" and model_name:
                        model_name_lower = model_name.lower()
                        if "sdxl" in model_name_lower or "xl" in model_name_lower:
                            checkpoint_sd_type = "SDXL"
                        elif "sd3" in model_name_lower or "sd-3" in model_name_lower:
                            checkpoint_sd_type = "SD3"
            
            # Count control nets
            if node_type == "ControlNetApply":
                control_net_count += 1
                
            # Count LoRAs    
            if node_type == "LoraLoader":
                lora_count += 1
                
            # Check for upscaling
            if node_type in ["UpscaleImage", "ImageUpscaleWithModel"]:
                has_upscaling = True
                
            # Check batch size
            if node_type in ["KSampler", "KSamplerAdvanced"] and "inputs" in node_data:
                if "batch_size" in node_data["inputs"]:
                    try:
                        batch_size = max(batch_size, int(node_data["inputs"]["batch_size"]))
                    except (ValueError, TypeError):
                        pass
            
            # Check for custom nodes
            if node_type in self.custom_node_types:
                custom_node_package = self.custom_node_types[node_type]
                if custom_node_package not in workflow_result["custom_nodes"]:
                    workflow_result["custom_nodes"].append(custom_node_package)
                    
                    # Add associated Python package dependencies
                    if custom_node_package in self.node_package_dependencies:
                        for package in self.node_package_dependencies[custom_node_package]:
                            if package not in workflow_result["python_packages"]:
                                workflow_result["python_packages"].append(package)
        
        # Estimate memory requirements
        # Start with base model size
        memory_required = self.gpu_memory_estimates["CheckpointLoader"][checkpoint_sd_type]
        
        # Add memory for VAE
        if workflow_result["models"]["vae"]:
            memory_required += self.gpu_memory_estimates["VAELoader"]
        
        # Add sampling overhead
        memory_required += self.gpu_memory_estimates["KSampler"]["base"]
        memory_required += self.gpu_memory_estimates["KSampler"]["per_batch"] * (batch_size - 1)
        
        # Add ControlNet memory
        memory_required += control_net_count * self.gpu_memory_estimates["ControlNetApply"]
        
        # Add LoRA memory
        memory_required += lora_count * self.gpu_memory_estimates["LoraLoader"]
        
        # Add upscaling memory if needed
        if has_upscaling:
            memory_required += self.gpu_memory_estimates["UpscaleImage"]
        
        # If using AnimateDiff, add its overhead
        if "comfyui-animatediff" in workflow_result["custom_nodes"]:
            memory_required += self.gpu_memory_estimates["AnimateDiff"]
        
        # Update memory requirements
        workflow_result["memory_required"]["min"] = round(memory_required, 1)
        workflow_result["memory_required"]["recommended"] = round(memory_required * 1.5, 1)  # Add 50% buffer for recommended
        
        workflow_result["is_analyzed"] = True
        return workflow_result

    def analyze_all_workflows(self):
        """Analyze all workflow files"""
        workflow_files = self.find_workflow_files()
        self.results["summary"]["total_workflows"] = len(workflow_files)
        
        for file_path in workflow_files:
            workflow_result = self.analyze_workflow(file_path)
            self.results["workflows"].append(workflow_result)
            
            if workflow_result["is_analyzed"]:
                self.results["summary"]["analyzed_workflows"] += 1
                
                # Update aggregate statistics
                for model_type, models in workflow_result["models"].items():
                    for model in models:
                        self.results["aggregate"]["models"][f"{model_type}: {model}"] += 1
                
                for node in workflow_result["custom_nodes"]:
                    self.results["aggregate"]["custom_nodes"][node] += 1
                
                for package in workflow_result["python_packages"]:
                    self.results["aggregate"]["python_packages"][package] += 1
                
                # Track memory requirements
                memory_required = workflow_result["memory_required"]["min"]
                memory_category = f"{memory_required:.1f}GB"
                self.results["aggregate"]["memory_requirements"]["workflows_by_memory"][memory_category].append(workflow_result["filename"])
                
                # Update min/max memory stats
                if self.results["aggregate"]["memory_requirements"]["min"] == 0 or memory_required < self.results["aggregate"]["memory_requirements"]["min"]:
                    self.results["aggregate"]["memory_requirements"]["min"] = memory_required
                
                if memory_required > self.results["aggregate"]["memory_requirements"]["max"]:
                    self.results["aggregate"]["memory_requirements"]["max"] = memory_required
                
            # Print progress
            status = "✅ Analyzed" if workflow_result["is_analyzed"] else f"❌ Failed ({len(workflow_result['errors'])} errors)"
            print(f"Analyzed {workflow_result['filename']}: {status}")

        return self.results

    def generate_markdown_report(self):
        """Generate a markdown report of requirements analysis"""
        report_path = os.path.join(self.output_dir, "workflow_requirements_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ComfyUI Workflow Requirements Report\n\n")
            f.write(f"Generated on: {self.results['summary']['time']}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- Total workflows analyzed: {self.results['summary']['total_workflows']}\n")
            f.write(f"- Successfully analyzed: {self.results['summary']['analyzed_workflows']}\n\n")
            
            # Aggregate statistics
            f.write("## Aggregate Requirements\n\n")
            
            # Models statistics
            f.write("### Models\n\n")
            f.write("| Model | Workflows |\n")
            f.write("|-------|----------|\n")
            
            # Sort by frequency (descending)
            for model, count in sorted(self.results["aggregate"]["models"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {model} | {count} |\n")
            
            f.write("\n")
            
            # Custom nodes statistics
            f.write("### Custom Nodes\n\n")
            f.write("| Custom Node Extension | Workflows |\n")
            f.write("|----------------------|----------|\n")
            
            for node, count in sorted(self.results["aggregate"]["custom_nodes"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {node} | {count} |\n")
            
            f.write("\n")
            
            # Python packages statistics
            f.write("### Python Packages\n\n")
            f.write("| Package | Workflows |\n")
            f.write("|---------|----------|\n")
            
            for package, count in sorted(self.results["aggregate"]["python_packages"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {package} | {count} |\n")
            
            f.write("\n")
            
            # Memory requirements statistics
            f.write("### Memory Requirements\n\n")
            f.write(f"- Minimum memory required: {self.results['aggregate']['memory_requirements']['min']:.1f}GB\n")
            f.write(f"- Maximum memory required: {self.results['aggregate']['memory_requirements']['max']:.1f}GB\n\n")
            
            f.write("| Memory Requirement | Workflows |\n")
            f.write("|-------------------|----------|\n")
            
            # Sort memory categories
            memory_categories = sorted(self.results["aggregate"]["memory_requirements"]["workflows_by_memory"].keys(), 
                                     key=lambda x: float(x.replace("GB", "")))
            
            for category in memory_categories:
                workflows = self.results["aggregate"]["memory_requirements"]["workflows_by_memory"][category]
                f.write(f"| {category} | {len(workflows)} |\n")
            
            f.write("\n")
            
            # Individual workflow details
            f.write("## Individual Workflow Requirements\n\n")
            
            for workflow in self.results["workflows"]:
                f.write(f"### {workflow['filename']}\n\n")
                
                if not workflow["is_analyzed"]:
                    f.write("❌ **Analysis failed**\n\n")
                    for error in workflow["errors"]:
                        f.write(f"- Error: {error}\n")
                    f.write("\n")
                    continue
                
                # Memory requirements
                f.write("**Memory Requirements**:\n")
                f.write(f"- Minimum: {workflow['memory_required']['min']}GB\n")
                f.write(f"- Recommended: {workflow['memory_required']['recommended']}GB\n\n")
                
                # Models
                f.write("**Required Models**:\n")
                has_models = False
                
                for model_type, models in workflow["models"].items():
                    if models:
                        has_models = True
                        type_name = model_type.replace("_", " ").title()
                        f.write(f"- {type_name}: {', '.join(models)}\n")
                
                if not has_models:
                    f.write("- No explicit model requirements found\n")
                
                f.write("\n")
                
                # Custom nodes
                f.write("**Required Custom Nodes**:\n")
                
                if workflow["custom_nodes"]:
                    for node in workflow["custom_nodes"]:
                        f.write(f"- {node}\n")
                else:
                    f.write("- No custom nodes required\n")
                
                f.write("\n")
                
                # Python packages
                f.write("**Required Python Packages**:\n")
                
                if workflow["python_packages"]:
                    for package in workflow["python_packages"]:
                        f.write(f"- {package}\n")
                else:
                    f.write("- No additional Python packages required\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Ensure Required Models Availability**: Pre-download the most commonly used models to avoid runtime downloads.\n")
            f.write("2. **Install Custom Node Extensions**: The custom node extensions listed should be installed before running the workflows.\n")
            f.write("3. **Install Python Dependencies**: Add required Python packages to your CI environment.\n")
            f.write("4. **GPU Memory Considerations**: Ensure sufficient GPU memory is available for the workflows you plan to test.\n")
            f.write("5. **Base Configuration**: Set up a baseline test environment that can run at least the simplest workflows.\n\n")
            
            f.write("---\n")
            f.write("*This report was automatically generated by the CI workflow requirements analyzer.*\n")
        
        print(f"Report generated at {report_path}")
        return report_path

    def generate_json_report(self):
        """Generate a JSON report of requirements analysis"""
        report_path = os.path.join(self.output_dir, "workflow_requirements_results.json")
        
        # Convert defaultdicts to regular dicts for JSON serialization
        json_results = self.results.copy()
        json_results["aggregate"]["models"] = dict(json_results["aggregate"]["models"])
        json_results["aggregate"]["custom_nodes"] = dict(json_results["aggregate"]["custom_nodes"])
        json_results["aggregate"]["python_packages"] = dict(json_results["aggregate"]["python_packages"])
        json_results["aggregate"]["memory_requirements"]["workflows_by_memory"] = {
            k: v for k, v in json_results["aggregate"]["memory_requirements"]["workflows_by_memory"].items()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"JSON results saved to {report_path}")
        return report_path
        
    def generate_github_summary(self):
        """Generate GitHub step summary with requirements analysis"""
        if not os.environ.get('GITHUB_STEP_SUMMARY'):
            return
        
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write("## ComfyUI Workflow Requirements Analysis\n\n")
            
            # Summary stats
            f.write(f"✅ **Analyzed {self.results['summary']['analyzed_workflows']} of {self.results['summary']['total_workflows']} workflows**\n\n")
            
            # Top requirements
            f.write("### Top Requirements\n\n")
            
            # Top models
            f.write("**Most Common Models**:\n")
            for model, count in sorted(self.results["aggregate"]["models"].items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"- {model} ({count} workflows)\n")
            f.write("\n")
            
            # Top custom nodes
            if self.results["aggregate"]["custom_nodes"]:
                f.write("**Most Common Custom Nodes**:\n")
                for node, count in sorted(self.results["aggregate"]["custom_nodes"].items(), key=lambda x: x[1], reverse=True)[:5]:
                    f.write(f"- {node} ({count} workflows)\n")
                f.write("\n")
            
            # Memory summary
            f.write("**Memory Requirements**:\n")
            f.write(f"- Range: {self.results['aggregate']['memory_requirements']['min']:.1f}GB - {self.results['aggregate']['memory_requirements']['max']:.1f}GB\n")
            
            # Show distribution of memory requirements
            if self.results["aggregate"]["memory_requirements"]["workflows_by_memory"]:
                memory_categories = sorted(self.results["aggregate"]["memory_requirements"]["workflows_by_memory"].keys(), 
                                         key=lambda x: float(x.replace("GB", "")))
                memory_counts = [len(self.results["aggregate"]["memory_requirements"]["workflows_by_memory"][cat]) for cat in memory_categories]
                
                f.write("- Distribution:\n")
                for i, category in enumerate(memory_categories):
                    count = memory_counts[i]
                    percent = 100 * count / self.results["summary"]["analyzed_workflows"]
                    f.write(f"  - {category}: {count} workflows ({percent:.1f}%)\n")
            
            f.write("\nSee workflow requirements report artifact for details.\n")

    def run(self):
        """Run the workflow requirements analysis"""
        print(f"Analyzing ComfyUI workflows in {self.workflows_dir}...")
        self.analyze_all_workflows()
        self.generate_markdown_report()
        self.generate_json_report()
        self.generate_github_summary()
        
        # Return the number of workflows successfully analyzed
        return self.results["summary"]["analyzed_workflows"]


def main():
    parser = argparse.ArgumentParser(description="Analyze ComfyUI workflow requirements")
    parser.add_argument("--workflows-dir", default="WebUI/external/workflows", help="Directory containing workflow files")
    parser.add_argument("--output-dir", default="ci_artifacts/workflow_requirements", help="Directory to store analysis results")
    args = parser.parse_args()
    
    analyzer = WorkflowRequirementsAnalyzer(
        workflows_dir=args.workflows_dir,
        output_dir=args.output_dir
    )
    
    analyzed_count = analyzer.run()
    
    # Return success if at least one workflow was analyzed
    sys.exit(0 if analyzed_count > 0 else 1)


if __name__ == "__main__":
    main() 