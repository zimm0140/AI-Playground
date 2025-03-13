#!/usr/bin/env python
"""
ComfyUI Workflow Execution Test

This script performs lightweight validation of ComfyUI workflow execution:
1. Verifies that the workflow JSON can be parsed into a valid graph structure
2. Checks for circular dependencies in the graph
3. Verifies that node connections are type-compatible (when type information is available)
4. Simulates workflow execution without requiring actual model loading or GPU resources

This allows testing the basic correctness of workflows without running the full ComfyUI environment.
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict, deque
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ComfyWorkflowTest")


class ComfyWorkflowTester:
    """Tester for ComfyUI workflow files"""

    def __init__(self, workflows_dir, output_dir="ci_artifacts/workflow_tests"):
        self.workflows_dir = workflows_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Known node types and their input/output slot types (simplified)
        self.node_types = {
            # Input nodes
            "CLIPTextEncode": {
                "inputs": {"text": "string", "clip": "clip_model"},
                "outputs": {"conditioning": "conditioning"},
            },
            "CheckpointLoader": {
                "inputs": {"ckpt_name": "string"},
                "outputs": {"model": "model", "clip": "clip_model", "vae": "vae"},
            },
            "VAELoader": {"inputs": {"vae_name": "string"}, "outputs": {"vae": "vae"}},
            "LoraLoader": {
                "inputs": {
                    "model": "model",
                    "clip": "clip_model",
                    "lora_name": "string",
                    "strength": "float",
                },
                "outputs": {"model": "model", "clip": "clip_model"},
            },
            "LoadImage": {
                "inputs": {"image": "string"},
                "outputs": {"image": "image", "mask": "mask"},
            },
            "EmptyLatentImage": {
                "inputs": {"width": "int", "height": "int", "batch_size": "int"},
                "outputs": {"latent": "latent"},
            },
            # Processing nodes
            "KSampler": {
                "inputs": {
                    "model": "model",
                    "seed": "int",
                    "steps": "int",
                    "cfg": "float",
                    "sampler_name": "string",
                    "scheduler": "string",
                    "positive": "conditioning",
                    "negative": "conditioning",
                    "latent_image": "latent",
                    "denoise": "float",
                },
                "outputs": {"latent": "latent"},
            },
            "VAEDecode": {
                "inputs": {"samples": "latent", "vae": "vae"},
                "outputs": {"image": "image"},
            },
            "VAEEncode": {
                "inputs": {"pixels": "image", "vae": "vae"},
                "outputs": {"latent": "latent"},
            },
            "UpscaleImage": {
                "inputs": {
                    "image": "image",
                    "upscale_method": "string",
                    "scale": "float",
                },
                "outputs": {"image": "image"},
            },
            # Output nodes
            "SaveImage": {
                "inputs": {"images": "image", "filename_prefix": "string"},
                "outputs": {},
            },
            "PreviewImage": {"inputs": {"images": "image"}, "outputs": {}},
        }

        # Simplified type compatibility matrix
        self.type_compatibility = {
            "string": ["string"],
            "int": ["int", "float"],
            "float": ["float", "int"],
            "model": ["model"],
            "clip_model": ["clip_model"],
            "vae": ["vae"],
            "conditioning": ["conditioning"],
            "latent": ["latent"],
            "image": ["image"],
            "mask": ["mask"],
        }

        # Results storage
        self.results = {
            "summary": {
                "total_workflows": 0,
                "passed_workflows": 0,
                "failed_workflows": 0,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "workflow_results": [],
        }

    def find_workflow_files(self):
        """Find all workflow JSON files in the specified directory"""
        return glob.glob(os.path.join(self.workflows_dir, "*.json"))

    def check_circular_dependencies(self, workflow):
        """Check if there are circular dependencies in the workflow graph"""
        if "nodes" not in workflow or "links" not in workflow:
            return False, "Workflow missing nodes or links"

        # Build a directed graph using adjacency list
        graph = defaultdict(list)

        for link in workflow["links"]:
            if len(link) < 4:  # Basic validation
                continue

            from_node, _, to_node, _ = link[0:4]
            graph[str(from_node)].append(str(to_node))

        # Check for cycles using BFS
        def has_cycle(node, visited, path):
            visited.add(node)
            path.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, path):
                        return True
                elif neighbor in path:
                    return True

            path.remove(node)
            return False

        visited = set()
        path = set()

        for node in graph:
            if node not in visited and has_cycle(node, visited, path):
                return (
                    True,
                    f"Circular dependency detected starting from node {node}",
                )

        return False, "No circular dependencies"

    def topological_sort(self, workflow):
        """Sort nodes in topological order for execution"""
        if "links" not in workflow:
            return [], "Workflow missing links"

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
            return [], "Workflow missing nodes or has invalid structure"

        # Build a directed graph using adjacency list and count incoming edges
        graph = defaultdict(list)
        in_degree = defaultdict(int)

        # Initialize all nodes with 0 in-degree
        for node_id in nodes:
            in_degree[str(node_id)] = 0

        # Count incoming edges
        for link in workflow["links"]:
            if len(link) < 4:  # Basic validation
                continue

            from_node, _, to_node, _ = link[0:4]
            from_node, to_node = str(from_node), str(to_node)

            graph[from_node].append(to_node)
            in_degree[to_node] += 1

        # Find all nodes with 0 in-degree
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []

        # Perform topological sort
        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check if we visited all nodes
        if len(result) != len(nodes):
            return [], "Cannot determine execution order due to cycles"

        return result, "Execution order determined successfully"

    def check_type_compatibility(self, from_type, to_type):
        """Check if the output type is compatible with the input type"""
        if from_type not in self.type_compatibility:
            return True  # Unknown type, assume compatible

        return to_type in self.type_compatibility[from_type]

    def simulate_execution(self, workflow):
        """Simulate the execution of a workflow without actually running it"""
        issues = []

        # Check for circular dependencies
        has_cycle, cycle_message = self.check_circular_dependencies(workflow)
        if has_cycle:
            issues.append(f"Error: {cycle_message}")
            return False, issues

        # Get execution order
        execution_order, order_message = self.topological_sort(workflow)
        if not execution_order:
            issues.append(f"Error: {order_message}")
            return False, issues

        # Build links map
        links_map = {}

        for link in workflow["links"]:
            if len(link) < 4:  # Basic validation
                issues.append(f"Warning: Invalid link format {link}")
                continue

            from_node, from_slot, to_node, to_slot = link[0:4]
            from_node, to_node = str(from_node), str(to_node)

            # Create a unique key for the link
            link_key = f"{to_node}:{to_slot}"
            links_map[link_key] = (from_node, from_slot)

        # Simulate execution of each node
        node_outputs = {}

        for node_id in execution_order:
            node_data = workflow["nodes"].get(node_id, {})

            if "class_type" not in node_data:
                issues.append(f"Warning: Node {node_id} has no class_type")
                continue

            node_type = node_data["class_type"]

            # Skip unknown node types
            if node_type not in self.node_types:
                logger.debug(f"Skipping unknown node type: {node_type}")
                continue

            # Get node type definition
            node_def = self.node_types[node_type]

            # Check input connections
            node_inputs = {}
            if "inputs" in node_data:
                for input_name, input_value in node_data["inputs"].items():
                    # Skip inputs that don't match our definition
                    if input_name not in node_def["inputs"]:
                        continue

                    # Add static input
                    node_inputs[input_name] = input_value

            # Process dynamic inputs from links
            for input_name in node_def["inputs"]:
                link_key = f"{node_id}:{input_name}"

                if link_key in links_map:
                    from_node, from_slot = links_map[link_key]

                    # Check if source node has been processed
                    if from_node not in node_outputs:
                        issues.append(
                            f"Error: Source node {from_node} not processed before {node_id}"
                        )
                        continue

                    # Check if source node has the output
                    if from_slot not in node_outputs[from_node]:
                        issues.append(
                            f"Error: Source node {from_node} has no output {from_slot}"
                        )
                        continue

                    # Get output value and type
                    output_value, output_type = node_outputs[from_node][from_slot]

                    # Check type compatibility if we have type information
                    input_type = node_def["inputs"].get(input_name)
                    if (
                        input_type
                        and output_type
                        and not self.check_type_compatibility(output_type, input_type)
                    ):
                        issues.append(
                            f"Warning: Type mismatch: {from_node}:{from_slot} ({output_type}) -> {node_id}:{input_name} ({input_type})"
                        )

                    # Add to node inputs
                    node_inputs[input_name] = output_value

            # Simulate node execution
            node_output = {}

            # For each output defined by node type, generate a simulated output
            for output_name, output_type in node_def["outputs"].items():
                # In a real execution, this would be the actual output value
                # For simulation, we just use a placeholder
                output_value = f"value_{node_id}_{output_name}"
                node_output[output_name] = (output_value, output_type)

            # Store node outputs
            node_outputs[node_id] = node_output

        # Check if all output nodes have been connected
        output_node_types = ["SaveImage", "PreviewImage"]
        has_output = False

        for node_id, node_data in workflow["nodes"].items():
            if (
                "class_type" in node_data
                and node_data["class_type"] in output_node_types
            ):
                # Check if this output node has required inputs
                if node_id in node_outputs:
                    has_output = True
                    break

        if not has_output:
            issues.append("Warning: No output nodes connected in workflow")

        success = len(issues) == 0 or all(
            issue.startswith("Warning") for issue in issues
        )
        return success, issues

    def test_workflow(self, file_path):
        """Test a single workflow file"""
        filename = os.path.basename(file_path)
        logger.info(f"Testing workflow: {filename}")

        result = {
            "filename": filename,
            "path": file_path,
            "passed": False,
            "issues": [],
            "execution_time": 0,
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        start_time = datetime.now()

        try:
            # Load the workflow
            with open(file_path, encoding="utf-8") as f:
                workflow = json.load(f)

            # Simulate execution
            success, issues = self.simulate_execution(workflow)

            # Store results
            result["passed"] = success
            result["issues"] = issues

        except json.JSONDecodeError as e:
            result["issues"].append(f"Error: Invalid JSON: {str(e)}")
        except Exception as e:
            result["issues"].append(f"Error: {str(e)}")

        # Calculate execution time
        end_time = datetime.now()
        result["execution_time"] = (end_time - start_time).total_seconds()

        return result

    def test_all_workflows(self):
        """Test all workflow files"""
        workflow_files = self.find_workflow_files()
        self.results["summary"]["total_workflows"] = len(workflow_files)

        for file_path in workflow_files:
            result = self.test_workflow(file_path)
            self.results["workflow_results"].append(result)

            if result["passed"]:
                self.results["summary"]["passed_workflows"] += 1
                logger.info(
                    f"✅ Passed: {result['filename']} ({result['execution_time']:.2f}s)"
                )
            else:
                self.results["summary"]["failed_workflows"] += 1
                logger.info(
                    f"❌ Failed: {result['filename']} ({result['execution_time']:.2f}s)"
                )
                for issue in result["issues"]:
                    logger.info(f"  - {issue}")

        logger.info(f"Tested {self.results['summary']['total_workflows']} workflows")
        logger.info(f"Passed: {self.results['summary']['passed_workflows']}")
        logger.info(f"Failed: {self.results['summary']['failed_workflows']}")

        return self.results

    def generate_markdown_report(self):
        """Generate a markdown report of test results"""
        report_path = os.path.join(self.output_dir, "workflow_test_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# ComfyUI Workflow Test Report\n\n")
            f.write(f"Generated on: {self.results['summary']['time']}\n\n")

            # Summary section
            f.write("## Summary\n\n")
            f.write(
                f"- Total workflows tested: {self.results['summary']['total_workflows']}\n"
            )
            f.write(f"- Passed: {self.results['summary']['passed_workflows']}\n")
            f.write(f"- Failed: {self.results['summary']['failed_workflows']}\n\n")

            # Test results overview
            f.write("## Test Results\n\n")
            f.write("| Workflow | Status | Duration | Issues |\n")
            f.write("|----------|--------|----------|--------|\n")

            # Sort by status (failed first) then by name
            sorted_results = sorted(
                self.results["workflow_results"],
                key=lambda x: (x["passed"], x["filename"]),
            )

            for result in sorted_results:
                status = "✅ Pass" if result["passed"] else "❌ Fail"
                issues_count = len(result["issues"])
                issues_text = (
                    f"{issues_count} issue{'s' if issues_count != 1 else ''}"
                    if issues_count > 0
                    else "None"
                )

                f.write(
                    f"| {result['filename']} | {status} | {result['execution_time']:.2f}s | {issues_text} |\n"
                )

            # Failed workflow details
            failed_results = [
                r for r in self.results["workflow_results"] if not r["passed"]
            ]

            if failed_results:
                f.write("\n## Failed Workflow Details\n\n")

                for result in failed_results:
                    f.write(f"### {result['filename']}\n\n")

                    for issue in result["issues"]:
                        issue_type = "Error" if issue.startswith("Error") else "Warning"
                        f.write(f"- **{issue_type}**: {issue}\n")

                    f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(
                "1. **Fix Circular Dependencies**: Workflows should not contain circular references between nodes.\n"
            )
            f.write(
                "2. **Connect Output Nodes**: Ensure workflows have properly connected output nodes (SaveImage or PreviewImage).\n"
            )
            f.write(
                "3. **Fix Type Mismatches**: Ensure connections between nodes use compatible types.\n"
            )
            f.write(
                "4. **Update Node Definitions**: If the test reports unknown node types, they may need to be added to the test definitions.\n\n"
            )

            f.write("---\n")
            f.write(
                "*This report was automatically generated by the ComfyUI workflow test script.*\n"
            )

        logger.info(f"Report generated at {report_path}")
        return report_path

    def generate_json_report(self):
        """Generate a JSON report of test results"""
        report_path = os.path.join(self.output_dir, "workflow_test_results.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"JSON results saved to {report_path}")
        return report_path

    def generate_github_summary(self):
        """Generate GitHub step summary with test results"""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## ComfyUI Workflow Tests\n\n")

            # Status indicators
            if self.results["summary"]["failed_workflows"] > 0:
                f.write(
                    f"⚠️ **{self.results['summary']['failed_workflows']} workflow(s) failed simulation**\n\n"
                )
            else:
                f.write("✅ **All workflows passed simulation**\n\n")

            # Summary table
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(
                f"| Total Workflows | {self.results['summary']['total_workflows']} |\n"
            )
            f.write(f"| Passed | {self.results['summary']['passed_workflows']} |\n")
            f.write(f"| Failed | {self.results['summary']['failed_workflows']} |\n\n")

            # Show failed workflows if any
            if self.results["summary"]["failed_workflows"] > 0:
                f.write("### Failed Workflows\n\n")

                failed_results = [
                    r for r in self.results["workflow_results"] if not r["passed"]
                ]
                for result in failed_results[:10]:  # Show at most 10 failures
                    error_count = sum(
                        1 for issue in result["issues"] if issue.startswith("Error")
                    )
                    warning_count = len(result["issues"]) - error_count

                    f.write(
                        f"- **{result['filename']}**: {error_count} errors, {warning_count} warnings\n"
                    )

                if len(failed_results) > 10:
                    f.write(f"\n... and {len(failed_results) - 10} more.\n")

                f.write("\nSee workflow test report artifact for details.\n")

    def run(self):
        """Run the workflow testing process"""
        logger.info(f"Testing ComfyUI workflows in {self.workflows_dir}...")
        self.test_all_workflows()
        self.generate_markdown_report()
        self.generate_json_report()
        self.generate_github_summary()

        # Return the number of failed workflows
        return self.results["summary"]["failed_workflows"]


def main():
    parser = argparse.ArgumentParser(description="Test ComfyUI workflow files")
    parser.add_argument(
        "--workflows-dir",
        default="WebUI/external/workflows",
        help="Directory containing workflow files",
    )
    parser.add_argument(
        "--output-dir",
        default="ci_artifacts/workflow_tests",
        help="Directory to store test results",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with error if any workflows fail tests",
    )
    args = parser.parse_args()

    tester = ComfyWorkflowTester(
        workflows_dir=args.workflows_dir, output_dir=args.output_dir
    )

    failed_count = tester.run()

    # Exit with error code if requested and there are failed workflows
    if args.fail_on_error and failed_count > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
