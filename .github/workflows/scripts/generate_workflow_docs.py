#!/usr/bin/env python3
"""
Workflow Documentation Generator

This script automatically generates markdown documentation from workflow JSON files.
It creates comprehensive documentation including descriptions, requirements, and usage.

Usage:
    python generate_workflow_docs.py --workflows-dir DIR --output-dir DIR [--create-index] [--create-gallery]
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime


def sanitize_filename(name):
    """Convert a string to a valid filename."""
    return re.sub(r"[^\w\-\.]", "_", name)


def load_workflow(workflow_file):
    """Load a workflow from a JSON file."""
    try:
        with open(workflow_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading workflow file: {str(e)}")
        return None


def format_model_link(model):
    """Format a model name as a link if it looks like a Hugging Face path."""
    model_path = model.get("model", "")
    if "/" in model_path and not model_path.startswith(("http://", "https://")):
        # Looks like a HF path (username/model_name)
        parts = model_path.split("/")
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

    # Examples (new section)
    if "examples" in workflow and workflow["examples"]:
        doc.append("## Examples")
        doc.append("")

        for i, example in enumerate(workflow["examples"]):
            title = example.get("title", f"Example {i+1}")
            doc.append(f"### {title}")
            doc.append("")

            if "description" in example:
                doc.append(example["description"])
                doc.append("")

            # Display input and output images side by side if available
            if "inputImage" in example or "outputImage" in example:
                doc.append('<div class="example-images">')

                if "inputImage" in example:
                    doc.append('<div class="input-image">')
                    doc.append("<p><strong>Input:</strong></p>")
                    doc.append(
                        f'<img src="{example["inputImage"]}" alt="Input for {title}" />'
                    )
                    doc.append("</div>")

                if "outputImage" in example:
                    doc.append('<div class="output-image">')
                    doc.append("<p><strong>Output:</strong></p>")
                    doc.append(
                        f'<img src="{example["outputImage"]}" alt="Output for {title}" />'
                    )
                    doc.append("</div>")

                doc.append("</div>")
                doc.append("")

            # Display input settings if available
            if "inputSettings" in example and example["inputSettings"]:
                doc.append("**Settings used:**")
                doc.append("")
                doc.append("```json")
                doc.append(json.dumps(example["inputSettings"], indent=2))
                doc.append("```")
                doc.append("")

    # Resource Estimation (new section)
    if "resourceEstimation" in workflow:
        doc.append("## Resource Requirements")
        doc.append("")

        resources = workflow["resourceEstimation"]
        doc.append("| Resource | Requirement |")
        doc.append("|----------|-------------|")

        if "vramMinimum" in resources:
            doc.append(f"| Minimum VRAM | {resources['vramMinimum']} MB |")

        if "vramRecommended" in resources:
            doc.append(f"| Recommended VRAM | {resources['vramRecommended']} MB |")

        if "diskSpace" in resources:
            doc.append(f"| Disk Space | {resources['diskSpace']} MB |")

        if "cpuUsage" in resources:
            doc.append(f"| CPU Usage | {resources['cpuUsage'].capitalize()} |")

        doc.append("")

    # Components (new section)
    if "components" in workflow and workflow["components"]:
        doc.append("## Components")
        doc.append("")
        doc.append("This workflow uses the following reusable components:")
        doc.append("")

        for component in workflow["components"]:
            component_id = component.get("componentId", "")
            component_type = component.get("componentType", "")
            description = component.get("description", "")

            doc.append(f"- **{component_id}** ({component_type}): {description}")

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
                if "/" in node and "@" in node:
                    # Format: username/repo@commit
                    parts = node.split("@")
                    repo = parts[0]
                    commit = parts[1] if len(parts) > 1 else ""
                    doc.append(
                        f"- [{repo}](https://github.com/{repo}) (commit: `{commit}`)"
                    )
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
            label = input_item.get(
                "label", input_item.get("nodeInput", "Unnamed Input")
            )
            input_type = input_item.get("type", "")
            default_value = input_item.get("defaultValue", "")

            # Format default value based on type
            if (
                input_type == "image"
                and isinstance(default_value, str)
                and default_value.startswith("data:image")
            ):
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


def generate_gallery(workflows_dir, output_dir):
    """Generate a visual gallery of workflow examples."""
    gallery = [
        "# ComfyUI Workflow Gallery",
        "",
        "Browse visual examples of available workflows:",
        "",
    ]

    # Collect workflow information with examples
    workflows_with_examples = []

    for filename in os.listdir(workflows_dir):
        if filename.endswith(".json"):
            workflow_file = os.path.join(workflows_dir, filename)
            workflow = load_workflow(workflow_file)

            if workflow and "examples" in workflow and workflow["examples"]:
                # Get the first example with an output image
                example = next(
                    (ex for ex in workflow["examples"] if "outputImage" in ex), None
                )

                if example:
                    workflows_with_examples.append(
                        {
                            "name": workflow.get("name", filename),
                            "description": workflow.get("description", ""),
                            "tags": workflow.get("tags", []),
                            "outputImage": example.get("outputImage", ""),
                            "doc_file": sanitize_filename(
                                workflow.get("name", filename)
                            )
                            + ".md",
                            "display_priority": workflow.get("displayPriority", 0),
                        }
                    )

    # Sort by display priority (higher first) and then by name
    workflows_with_examples.sort(
        key=lambda w: (-w.get("display_priority", 0), w["name"])
    )

    # Create gallery grid
    gallery.append('<div class="workflow-gallery">')

    for workflow in workflows_with_examples:
        gallery.append('<div class="workflow-card">')
        gallery.append(f'<a href="{workflow["doc_file"]}">')
        gallery.append(
            f'<img src="{workflow["outputImage"]}" alt="{workflow["name"]}" />'
        )
        gallery.append(f'<h3>{workflow["name"]}</h3>')
        gallery.append("</a>")

        # Add tags if available
        if workflow["tags"]:
            gallery.append('<div class="tags">')
            for tag in workflow["tags"]:
                gallery.append(f'<span class="tag">{tag}</span>')
            gallery.append("</div>")

        # Add short description
        if workflow["description"]:
            short_desc = workflow["description"][:100] + (
                "..." if len(workflow["description"]) > 100 else ""
            )
            gallery.append(f"<p>{short_desc}</p>")

        gallery.append("</div>")

    gallery.append("</div>")

    # Add CSS for the gallery
    gallery.append(
        """
<style>
.workflow-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.workflow-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  transition: transform 0.3s ease;
}

.workflow-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.workflow-card img {
  width: 100%;
  height: 200px;
  object-fit: cover;
}

.workflow-card h3 {
  padding: 10px;
  margin: 0;
  font-size: 18px;
}

.workflow-card p {
  padding: 0 10px 10px;
  margin: 0;
  color: #666;
  font-size: 14px;
}

.tags {
  padding: 0 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}

.tag {
  background: #f0f0f0;
  padding: 3px 8px;
  border-radius: 4px;
  font-size: 12px;
  color: #555;
}

.example-images {
  display: flex;
  gap: 20px;
  margin: 20px 0;
}

.input-image, .output-image {
  flex: 1;
}

.input-image img, .output-image img {
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 4px;
}
</style>
"""
    )

    # Write gallery file
    gallery_path = os.path.join(output_dir, "gallery.md")
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gallery))

    print(f"Generated gallery: {gallery_path}")
    return True


def generate_index(workflows_dir, output_dir):
    """Generate an index page with links to all workflow documentation."""
    index = [
        "# ComfyUI Workflows",
        "",
        "This directory contains documentation for all available ComfyUI workflows.",
        "",
    ]

    # Collect workflow information
    workflows = []
    for filename in os.listdir(workflows_dir):
        if filename.endswith(".json"):
            workflow_file = os.path.join(workflows_dir, filename)
            workflow = load_workflow(workflow_file)
            if workflow:
                doc_filename = sanitize_filename(workflow.get("name", filename)) + ".md"
                workflows.append(
                    {
                        "name": workflow.get("name", filename),
                        "tags": workflow.get("tags", []),
                        "description": workflow.get("description", ""),
                        "doc_file": doc_filename,
                        "display_priority": workflow.get("displayPriority", 0),
                    }
                )

    # Sort by display priority (higher first) and then by name
    workflows.sort(key=lambda w: (-w.get("display_priority", 0), w["name"]))

    # Add link to gallery if it exists
    if os.path.exists(os.path.join(output_dir, "gallery.md")):
        index.append("## Visual Gallery")
        index.append("")
        index.append("Browse workflows visually in our [Gallery](gallery.md).")
        index.append("")

    # Create table
    index.append("## Available Workflows")
    index.append("")
    index.append("| Workflow | Tags | Description |")
    index.append("|----------|------|-------------|")

    for workflow in workflows:
        name_with_link = f"[{workflow['name']}]({workflow['doc_file']})"
        tags = ", ".join([f"`{tag}`" for tag in workflow.get("tags", [])])
        description = workflow.get("description", "")[:100] + (
            "..." if len(workflow.get("description", "")) > 100 else ""
        )
        index.append(f"| {name_with_link} | {tags} | {description} |")

    index.append("")
    index.append("---")
    index.append(f"*Index generated on {datetime.now().strftime('%Y-%m-%d')}*")

    # Write index file
    index_path = os.path.join(output_dir, "README.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index))

    print(f"Generated index: {index_path}")
    return True


def generate_all_docs(
    workflows_dir, output_dir, create_index=False, create_gallery=False
):
    """Generate documentation for all workflow files in a directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    success_count = 0
    error_count = 0

    for filename in os.listdir(workflows_dir):
        if filename.endswith(".json"):
            workflow_file = os.path.join(workflows_dir, filename)
            workflow = load_workflow(workflow_file)

            if workflow:
                try:
                    doc_content = generate_workflow_doc(workflow, workflow_file)
                    doc_filename = (
                        sanitize_filename(workflow.get("name", filename)) + ".md"
                    )
                    doc_path = os.path.join(output_dir, doc_filename)

                    with open(doc_path, "w", encoding="utf-8") as f:
                        f.write(doc_content)

                    print(f"Generated documentation for {filename} -> {doc_path}")
                    success_count += 1
                except Exception as e:
                    print(f"Error generating documentation for {filename}: {str(e)}")
                    error_count += 1
            else:
                error_count += 1

    if create_gallery:
        generate_gallery(workflows_dir, output_dir)

    if create_index:
        generate_index(workflows_dir, output_dir)

    print("\nDocumentation generation complete:")
    print(f"Successfully generated: {success_count}")
    print(f"Errors: {error_count}")

    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation for ComfyUI workflow files"
    )
    parser.add_argument(
        "--workflows-dir",
        required=True,
        help="Directory containing workflow JSON files",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write documentation files"
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create an index page with links to all workflows",
    )
    parser.add_argument(
        "--create-gallery",
        action="store_true",
        help="Create a visual gallery of workflow examples",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.workflows_dir):
        print(f"Error: Workflows directory {args.workflows_dir} not found")
        sys.exit(1)

    success = generate_all_docs(
        args.workflows_dir, args.output_dir, args.create_index, args.create_gallery
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
