#!/usr/bin/env python3
"""
Workflow Documentation Generator

This script automatically generates markdown documentation from workflow JSON files.
It creates comprehensive documentation including descriptions, requirements, and usage.

Usage:
    python generate_workflow_docs.py --workflows-dir DIR --output-dir DIR [--create-index]
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime

def sanitize_filename(name):
    """Convert a string to a valid filename."""
    return re.sub(r'[^\w\-\.]', '_', name)

def load_workflow(workflow_file):
    """Load a workflow from a JSON file."""
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading workflow file: {str(e)}")
        return None

def format_model_link(model):
    """Format a model name as a link if it looks like a Hugging Face path."""
    model_path = model.get("model", "")
    if '/' in model_path and not model_path.startswith(('http://', 'https://')):
        # Looks like a HF path (username/model_name)
        parts = model_path.split('/')
        if len(parts) >= 2:
            return f"[{model_path}](https://huggingface.co/{parts[0]}/{parts[1]})"
    return model_path

def generate_workflow_doc(workflow, workflow_file):
    """Generate markdown documentation for a workflow."""
    doc = []
    
    # Header and basic info
    doc.append(f"# {workflow.get('name', 'Unnamed Workflow')}")
    doc.append("")
    
    # Add version if available
    if "version" in workflow:
        doc.append(f"**Version:** {workflow['version']}")
        doc.append("")
    
    # Tags
    if "tags" in workflow and workflow["tags"]:
        doc.append("**Tags:** " + ", ".join([f"`{tag}`" for tag in workflow["tags"]]))
        doc.append("")
    
    # Description
    if "description" in workflow and workflow["description"]:
        doc.append("## Description")
        doc.append("")
        doc.append(workflow["description"])
        doc.append("")
    
    # System Requirements
    if "requirements" in workflow and workflow["requirements"]:
        doc.append("## System Requirements")
        doc.append("")
        for req in workflow["requirements"]:
            doc.append(f"- {req}")
        doc.append("")
    
    # Models and Technical Requirements
    if "comfyUIRequirements" in workflow:
        reqs = workflow["comfyUIRequirements"]
        doc.append("## Technical Requirements")
        doc.append("")
        
        # Models
        if "requiredModels" in reqs and reqs["requiredModels"]:
            doc.append("### Required Models")
            doc.append("")
            doc.append("| Type | Model | Additional Info |")
            doc.append("|------|-------|----------------|")
            
            for model in reqs["requiredModels"]:
                model_type = model.get("type", "")
                model_link = format_model_link(model)
                additional_info = model.get("additionalLicenceLink", "")
                if additional_info:
                    additional_info = f"[License]({additional_info})"
                
                doc.append(f"| {model_type} | {model_link} | {additional_info} |")
            
            doc.append("")
        
        # Custom Nodes
        if "customNodes" in reqs and reqs["customNodes"]:
            doc.append("### Required Custom Nodes")
            doc.append("")
            for node in reqs["customNodes"]:
                if '/' in node and '@' in node:
                    # Format: username/repo@commit
                    parts = node.split('@')
                    repo = parts[0]
                    commit = parts[1] if len(parts) > 1 else ""
                    doc.append(f"- [{repo}](https://github.com/{repo}) (commit: `{commit}`)")
                else:
                    doc.append(f"- {node}")
            doc.append("")
        
        # Python Packages
        if "pythonPackages" in reqs and reqs["pythonPackages"]:
            doc.append("### Required Python Packages")
            doc.append("")
            for package in reqs["pythonPackages"]:
                doc.append(f"- `{package}`")
            doc.append("")
    
    # Default Settings
    if "defaultSettings" in workflow and workflow["defaultSettings"]:
        doc.append("## Default Settings")
        doc.append("")
        doc.append("| Setting | Value |")
        doc.append("|---------|-------|")
        
        for key, value in workflow["defaultSettings"].items():
            doc.append(f"| {key} | `{value}` |")
        doc.append("")
    
    # Inputs
    if "inputs" in workflow and workflow["inputs"]:
        doc.append("## User Inputs")
        doc.append("")
        doc.append("| Label | Type | Default Value |")
        doc.append("|-------|------|--------------|")
        
        for input_item in workflow["inputs"]:
            label = input_item.get("label", input_item.get("nodeInput", "Unnamed Input"))
            input_type = input_item.get("type", "")
            default_value = input_item.get("defaultValue", "")
            
            # Format default value based on type
            if input_type == "image" and isinstance(default_value, str) and default_value.startswith("data:image"):
                default_value = "(embedded image)"
            elif isinstance(default_value, (dict, list)):
                default_value = json.dumps(default_value)[:20] + "..."
            
            doc.append(f"| {label} | {input_type} | `{default_value}` |")
        doc.append("")
    
    # Outputs
    if "outputs" in workflow and workflow["outputs"]:
        doc.append("## Outputs")
        doc.append("")
        doc.append("| Name | Type |")
        doc.append("|------|------|")
        
        for output in workflow["outputs"]:
            name = output.get("name", "Unnamed Output")
            output_type = output.get("type", "")
            doc.append(f"| {name} | {output_type} |")
        doc.append("")
    
    # Change Log
    if "changeLog" in workflow and workflow["changeLog"]:
        doc.append("## Change Log")
        doc.append("")
        
        for entry in workflow["changeLog"]:
            version = entry.get("version", "")
            date_str = entry.get("date", "")
            doc.append(f"### Version {version} ({date_str})")
            doc.append("")
            
            for change in entry.get("changes", []):
                doc.append(f"- {change}")
            doc.append("")
    
    # Add footer with generation info
    doc.append("---")
    doc.append(f"*Documentation generated on {datetime.now().strftime('%Y-%m-%d')}*")
    
    return "\n".join(doc)

def generate_index(workflows_dir, output_dir):
    """Generate an index page with links to all workflow documentation."""
    index = ["# ComfyUI Workflows", "", "This directory contains documentation for all available ComfyUI workflows.", ""]
    
    # Collect workflow information
    workflows = []
    for filename in os.listdir(workflows_dir):
        if filename.endswith('.json'):
            workflow_file = os.path.join(workflows_dir, filename)
            workflow = load_workflow(workflow_file)
            if workflow:
                doc_filename = sanitize_filename(workflow.get("name", filename)) + ".md"
                workflows.append({
                    "name": workflow.get("name", filename),
                    "tags": workflow.get("tags", []),
                    "description": workflow.get("description", ""),
                    "doc_file": doc_filename,
                    "display_priority": workflow.get("displayPriority", 0)
                })
    
    # Sort by display priority (higher first) and then by name
    workflows.sort(key=lambda w: (-w.get("display_priority", 0), w["name"]))
    
    # Create table
    index.append("## Available Workflows")
    index.append("")
    index.append("| Workflow | Tags | Description |")
    index.append("|----------|------|-------------|")
    
    for workflow in workflows:
        name_with_link = f"[{workflow['name']}]({workflow['doc_file']})"
        tags = ", ".join([f"`{tag}`" for tag in workflow.get("tags", [])])
        description = workflow.get("description", "")[:100] + ("..." if len(workflow.get("description", "")) > 100 else "")
        index.append(f"| {name_with_link} | {tags} | {description} |")
    
    index.append("")
    index.append("---")
    index.append(f"*Index generated on {datetime.now().strftime('%Y-%m-%d')}*")
    
    # Write index file
    index_path = os.path.join(output_dir, "README.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(index))
    
    print(f"Generated index: {index_path}")
    return True

def generate_all_docs(workflows_dir, output_dir, create_index=False):
    """Generate documentation for all workflow files in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    success_count = 0
    error_count = 0
    
    for filename in os.listdir(workflows_dir):
        if filename.endswith('.json'):
            workflow_file = os.path.join(workflows_dir, filename)
            workflow = load_workflow(workflow_file)
            
            if workflow:
                try:
                    doc_content = generate_workflow_doc(workflow, workflow_file)
                    doc_filename = sanitize_filename(workflow.get("name", filename)) + ".md"
                    doc_path = os.path.join(output_dir, doc_filename)
                    
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(doc_content)
                    
                    print(f"Generated documentation for {filename} -> {doc_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error generating documentation for {filename}: {str(e)}")
                    error_count += 1
            else:
                error_count += 1
    
    if create_index:
        generate_index(workflows_dir, output_dir)
    
    print(f"\nDocumentation generation complete:")
    print(f"Successfully generated: {success_count}")
    print(f"Errors: {error_count}")
    
    return error_count == 0

def main():
    parser = argparse.ArgumentParser(description="Generate documentation for ComfyUI workflow files")
    parser.add_argument("--workflows-dir", required=True, help="Directory containing workflow JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to write documentation files")
    parser.add_argument("--create-index", action="store_true", help="Create an index page with links to all workflows")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.workflows_dir):
        print(f"Error: Workflows directory {args.workflows_dir} not found")
        sys.exit(1)
    
    success = generate_all_docs(args.workflows_dir, args.output_dir, args.create_index)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 