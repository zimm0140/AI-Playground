#!/usr/bin/env python
"""
ComfyUI Workflow Validator

This script validates ComfyUI workflow JSON files to ensure they are:
1. Well-formed JSON
2. Contain the expected structure
3. Have valid connections between nodes
4. Use node types that are likely to be available

The script produces a report of any issues found in the workflows.
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path


class ComfyWorkflowValidator:
    """Validator for ComfyUI workflow files"""

    def __init__(self, workflows_dir, output_dir="ci_artifacts/workflow_validation"):
        self.workflows_dir = workflows_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Common node types in ComfyUI
        self.known_node_types = {
            # Input nodes
            "CLIPTextEncode", "CLIPSetLastLayer", "LoraLoader", "CheckpointLoader", 
            "ControlNetLoader", "LoadImage", "LoadImageMask", "VAELoader",
            
            # Processing nodes
            "KSampler", "KSamplerAdvanced", "SamplerCustom", "VAEDecode", "VAEEncode",
            "ControlNetApply", "SetLatentNoiseMask", "LatentUpscale", "UpscaleImage",
            
            # Output/utility nodes
            "SaveImage", "PreviewImage", "EmptyLatentImage", "ConditioningCombine",
            "ConditioningAverage", "ConditioningSetArea", "SetModelWeights",
            
            # Advanced nodes
            "LatentComposite", "ImageComposite", "ImageScaleBy", "CropImage",
            "FaceDetailer", "DynamicThresholding", "ColorCorrect", "Reroute",
            
            # Animation nodes
            "AnimateDiff", "VideoLinearCFGGuidance", "LoadVideo", "SaveVideo",
            
            # Flux specific nodes
            "FluxNodes", "FluxImplicitKSampler", "FluxQ4", "FluxQ8", 
            
            # LoRA nodes
            "LoraLoader", "DiffControlNetLoader",
            
            # FaceSwap nodes
            "FaceSwapNode", "IPAdapterApply", "CopyFace",
            
            # Colorization nodes
            "ColorizeImage", "LineArtProcessor"
        }

        # Results storage
        self.results = {
            "summary": {
                "total_workflows": 0,
                "valid_workflows": 0,
                "invalid_workflows": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "workflows": []
        }

    def find_workflow_files(self):
        """Find all workflow JSON files in the specified directory"""
        return glob.glob(os.path.join(self.workflows_dir, "*.json"))

    def validate_workflow(self, file_path):
        """Validate a single workflow file"""
        filename = os.path.basename(file_path)
        file_result = {
            "filename": filename,
            "path": file_path,
            "is_valid": False,
            "issues": []
        }

        # Try to parse the JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            file_result["issues"].append({
                "type": "json_error",
                "message": f"Invalid JSON: {str(e)}"
            })
            return file_result
        except Exception as e:
            file_result["issues"].append({
                "type": "file_error",
                "message": f"Error reading file: {str(e)}"
            })
            return file_result

        # Check if the workflow has the expected structure
        if not isinstance(workflow, dict):
            file_result["issues"].append({
                "type": "structure_error",
                "message": "Workflow is not a dictionary object"
            })
            return file_result

        # Check for the presence of nodes
        if "nodes" not in workflow:
            file_result["issues"].append({
                "type": "structure_error",
                "message": "Workflow does not contain 'nodes' key"
            })
            return file_result

        # Check if nodes is a dictionary
        if not isinstance(workflow["nodes"], dict):
            file_result["issues"].append({
                "type": "structure_error",
                "message": "'nodes' is not a dictionary object"
            })
            return file_result

        # Check for empty nodes
        if not workflow["nodes"]:
            file_result["issues"].append({
                "type": "content_warning",
                "message": "Workflow contains no nodes"
            })

        # Collect node IDs and types
        node_ids = set()
        node_types = {}
        unknown_node_types = set()

        for node_id, node_data in workflow["nodes"].items():
            node_ids.add(node_id)
            
            # Check if node has a type
            if "class_type" not in node_data:
                file_result["issues"].append({
                    "type": "node_error",
                    "message": f"Node {node_id} does not have a 'class_type'"
                })
                continue
                
            node_type = node_data["class_type"]
            node_types[node_id] = node_type
            
            # Check if node type is known
            if node_type not in self.known_node_types:
                unknown_node_types.add(node_type)

        # Report unknown node types
        if unknown_node_types:
            file_result["issues"].append({
                "type": "node_warning",
                "message": f"Workflow uses {len(unknown_node_types)} unknown node types: {', '.join(unknown_node_types)}"
            })

        # Validate connections
        if "links" in workflow:
            # Check if links is a list
            if not isinstance(workflow["links"], list):
                file_result["issues"].append({
                    "type": "structure_error",
                    "message": "'links' is not a list"
                })
            else:
                for i, link in enumerate(workflow["links"]):
                    # Check if link has the right structure
                    if not isinstance(link, list) or len(link) < 4:
                        file_result["issues"].append({
                            "type": "link_error",
                            "message": f"Link at index {i} has invalid format"
                        })
                        continue
                        
                    from_node, from_slot, to_node, to_slot = link[0:4]
                    
                    # Check if source node exists
                    if str(from_node) not in node_ids:
                        file_result["issues"].append({
                            "type": "link_error",
                            "message": f"Link references non-existent source node {from_node}"
                        })
                    
                    # Check if target node exists
                    if str(to_node) not in node_ids:
                        file_result["issues"].append({
                            "type": "link_error",
                            "message": f"Link references non-existent target node {to_node}"
                        })

        # Mark as valid if no issues were found
        if not file_result["issues"]:
            file_result["is_valid"] = True

        return file_result

    def validate_all_workflows(self):
        """Validate all workflow files"""
        workflow_files = self.find_workflow_files()
        self.results["summary"]["total_workflows"] = len(workflow_files)
        
        for file_path in workflow_files:
            file_result = self.validate_workflow(file_path)
            self.results["workflows"].append(file_result)
            
            if file_result["is_valid"]:
                self.results["summary"]["valid_workflows"] += 1
            else:
                self.results["summary"]["invalid_workflows"] += 1
                
            # Print progress
            status = "✅ Valid" if file_result["is_valid"] else f"❌ Invalid ({len(file_result['issues'])} issues)"
            print(f"Validated {file_result['filename']}: {status}")

        return self.results

    def generate_markdown_report(self):
        """Generate a markdown report of validation results"""
        report_path = os.path.join(self.output_dir, "workflow_validation_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ComfyUI Workflow Validation Report\n\n")
            f.write(f"Generated on: {self.results['summary']['time']}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- Total workflows analyzed: {self.results['summary']['total_workflows']}\n")
            f.write(f"- Valid workflows: {self.results['summary']['valid_workflows']}\n")
            f.write(f"- Invalid workflows: {self.results['summary']['invalid_workflows']}\n\n")
            
            # Status overview
            f.write("## Workflow Status Overview\n\n")
            f.write("| Workflow | Status | Issues |\n")
            f.write("|----------|--------|--------|\n")
            
            for workflow in self.results["workflows"]:
                status = "✅ Valid" if workflow["is_valid"] else "❌ Invalid"
                issues_count = len(workflow["issues"])
                f.write(f"| {workflow['filename']} | {status} | {issues_count} |\n")
            
            f.write("\n")
            
            # Detailed issues section
            if self.results["summary"]["invalid_workflows"] > 0:
                f.write("## Detailed Issues\n\n")
                
                for workflow in self.results["workflows"]:
                    if not workflow["is_valid"]:
                        f.write(f"### {workflow['filename']}\n\n")
                        
                        for issue in workflow["issues"]:
                            issue_type = issue["type"].replace("_", " ").title()
                            f.write(f"- **{issue_type}**: {issue['message']}\n")
                        
                        f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Fix JSON formatting errors**: Ensure all workflow files contain valid JSON syntax.\n")
            f.write("2. **Add missing node types**: If unknown node types are legitimate, consider adding them to the validator's known node types list.\n")
            f.write("3. **Fix broken links**: Ensure all node connections reference valid nodes in the workflow.\n")
            f.write("4. **Standardize workflows**: Consider standardizing workflows or adding documentation for custom node types.\n\n")
            
            f.write("---\n")
            f.write("*This report was automatically generated by the CI workflow validation script.*\n")
        
        print(f"Report generated at {report_path}")
        return report_path

    def generate_json_report(self):
        """Generate a JSON report of validation results"""
        report_path = os.path.join(self.output_dir, "workflow_validation_results.json")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"JSON results saved to {report_path}")
        return report_path
        
    def generate_github_summary(self):
        """Generate GitHub step summary with validation results"""
        if not os.environ.get('GITHUB_STEP_SUMMARY'):
            return
        
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write("## ComfyUI Workflow Validation\n\n")
            
            # Status indicators
            if self.results["summary"]["invalid_workflows"] > 0:
                f.write(f"⚠️ **Found {self.results['summary']['invalid_workflows']} invalid workflow(s)**\n\n")
            else:
                f.write("✅ **All workflows are valid**\n\n")
            
            # Summary table
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Workflows | {self.results['summary']['total_workflows']} |\n")
            f.write(f"| Valid | {self.results['summary']['valid_workflows']} |\n")
            f.write(f"| Invalid | {self.results['summary']['invalid_workflows']} |\n\n")
            
            # List invalid workflows if any
            if self.results["summary"]["invalid_workflows"] > 0:
                f.write("### Invalid Workflows\n\n")
                
                for workflow in self.results["workflows"]:
                    if not workflow["is_valid"]:
                        issue_types = set(issue["type"] for issue in workflow["issues"])
                        issue_summary = ", ".join(t.replace("_", " ").title() for t in issue_types)
                        f.write(f"- **{workflow['filename']}**: {issue_summary} ({len(workflow['issues'])} issues)\n")
                
                f.write("\nSee workflow validation report artifact for details.\n")

    def run(self):
        """Run the workflow validation process"""
        print(f"Validating ComfyUI workflows in {self.workflows_dir}...")
        self.validate_all_workflows()
        self.generate_markdown_report()
        self.generate_json_report()
        self.generate_github_summary()
        
        # Return the number of invalid workflows (for exit code)
        return self.results["summary"]["invalid_workflows"]


def main():
    parser = argparse.ArgumentParser(description="Validate ComfyUI workflow files")
    parser.add_argument("--workflows-dir", default="WebUI/external/workflows", help="Directory containing workflow files")
    parser.add_argument("--output-dir", default="ci_artifacts/workflow_validation", help="Directory to store validation results")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with error if any workflows are invalid")
    args = parser.parse_args()
    
    validator = ComfyWorkflowValidator(
        workflows_dir=args.workflows_dir,
        output_dir=args.output_dir
    )
    
    invalid_count = validator.run()
    
    # Exit with error code if requested and there are invalid workflows
    if args.fail_on_error and invalid_count > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main() 