#!/usr/bin/env python
"""
ComfyUI Workflow Version Tracker

This script tracks changes to ComfyUI workflow files over time:
1. Calculates a unique fingerprint/hash for each workflow's structure
2. Maintains a version history for workflows with timestamps and changes
3. Detects breaking changes that may affect compatibility
4. Generates reports on workflow evolution and stability

This helps track workflow changes over time and identify potential compatibility issues.
"""

import os
import sys
import json
import hashlib
import argparse
import glob
import difflib
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set


class WorkflowVersion:
    """Represents a specific version of a workflow"""
    def __init__(self, 
                 hash_value: str,
                 timestamp: datetime.datetime,
                 node_count: int, 
                 link_count: int,
                 structure_hash: str,
                 changes: List[str] = None):
        self.hash_value = hash_value
        self.timestamp = timestamp
        self.node_count = node_count
        self.link_count = link_count
        self.structure_hash = structure_hash
        self.changes = changes or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "hash": self.hash_value,
            "timestamp": self.timestamp.isoformat(),
            "node_count": self.node_count,
            "link_count": self.link_count,
            "structure_hash": self.structure_hash,
            "changes": self.changes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowVersion':
        """Create from dictionary"""
        return cls(
            hash_value=data["hash"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            node_count=data["node_count"],
            link_count=data["link_count"],
            structure_hash=data["structure_hash"],
            changes=data.get("changes", [])
        )


class WorkflowHistory:
    """Represents the version history of a workflow"""
    def __init__(self, 
                 workflow_id: str, 
                 filename: str, 
                 versions: List[WorkflowVersion] = None):
        self.workflow_id = workflow_id
        self.filename = filename
        self.versions = versions or []
    
    def add_version(self, version: WorkflowVersion) -> None:
        """Add a new version to the history"""
        self.versions.append(version)
    
    def get_latest_version(self) -> Optional[WorkflowVersion]:
        """Get the most recent version"""
        if not self.versions:
            return None
        return sorted(self.versions, key=lambda v: v.timestamp, reverse=True)[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "workflow_id": self.workflow_id,
            "filename": self.filename,
            "versions": [v.to_dict() for v in self.versions]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowHistory':
        """Create from dictionary"""
        history = cls(
            workflow_id=data["workflow_id"],
            filename=data["filename"]
        )
        history.versions = [WorkflowVersion.from_dict(v) for v in data.get("versions", [])]
        return history


class WorkflowVersionTracker:
    """Tracks versions of ComfyUI workflows"""
    
    def __init__(self, 
                 workflows_dir: str, 
                 history_file: str = None,
                 output_dir: str = "ci_artifacts/workflow_versions"):
        self.workflows_dir = workflows_dir
        self.history_file = history_file or os.path.join(output_dir, "workflow_history.json")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing history if available
        self.workflow_history: Dict[str, WorkflowHistory] = {}
        if os.path.exists(self.history_file):
            self.load_history()
    
    def load_history(self) -> None:
        """Load workflow history from file"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for history_data in data.get("workflows", []):
                    history = WorkflowHistory.from_dict(history_data)
                    self.workflow_history[history.workflow_id] = history
            print(f"Loaded history for {len(self.workflow_history)} workflows")
        except Exception as e:
            print(f"Error loading history: {e}")
            # Start with empty history if loading fails
            self.workflow_history = {}
    
    def save_history(self) -> None:
        """Save workflow history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "last_updated": datetime.datetime.now().isoformat(),
                    "workflows": [h.to_dict() for h in self.workflow_history.values()]
                }, f, indent=2)
            print(f"Saved history to {self.history_file}")
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def find_workflow_files(self) -> List[str]:
        """Find all workflow JSON files in the specified directory"""
        return glob.glob(os.path.join(self.workflows_dir, "*.json"))
    
    def calculate_hash(self, content: Dict) -> str:
        """Calculate a hash of the workflow content"""
        serialized = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def calculate_structure_hash(self, workflow: Dict) -> str:
        """Calculate a hash representing the workflow structure (nodes and connections)
        This ignores node positions, workflow title, and specific parameter values.
        It focuses on the types of nodes, their input parameters (keys only), and how they're connected.
        """
        structure = {}
        
        # Extract node types and their connections
        nodes = None
        if "nodes" in workflow and isinstance(workflow["nodes"], dict):
            nodes = workflow["nodes"]
        elif ("comfyUiApiWorkflow" in workflow and 
              isinstance(workflow["comfyUiApiWorkflow"], dict) and 
              "nodes" in workflow["comfyUiApiWorkflow"] and
              isinstance(workflow["comfyUiApiWorkflow"]["nodes"], dict)):
            nodes = workflow["comfyUiApiWorkflow"]["nodes"]
            
        if nodes:
            nodes_structure = {}
            for node_id, node_data in nodes.items():
                node_structure = {
                    "class_type": node_data.get("class_type"),
                    # Include only input keys but not their values
                    "input_keys": sorted(list(node_data.get("inputs", {}).keys()))
                }
                nodes_structure[node_id] = node_structure
            structure["nodes"] = nodes_structure
        
        # Extract link structure (connections between nodes)
        if "links" in workflow:
            # Only store the node and slot information, not any additional metadata
            links_structure = []
            for link in workflow["links"]:
                if len(link) >= 4:  # Basic validation
                    link_structure = link[:4]  # from_node, from_slot, to_node, to_slot
                    links_structure.append(link_structure)
            structure["links"] = sorted(links_structure)
        
        serialized = json.dumps(structure, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def detect_changes(self, old_workflow: Dict, new_workflow: Dict) -> List[str]:
        """Detect changes between two versions of a workflow"""
        changes = []
        
        # Get nodes from either workflow format
        def get_nodes(workflow):
            if "nodes" in workflow and isinstance(workflow["nodes"], dict):
                return workflow["nodes"]
            elif ("comfyUiApiWorkflow" in workflow and 
                  isinstance(workflow["comfyUiApiWorkflow"], dict) and 
                  "nodes" in workflow["comfyUiApiWorkflow"] and
                  isinstance(workflow["comfyUiApiWorkflow"]["nodes"], dict)):
                return workflow["comfyUiApiWorkflow"]["nodes"]
            return {}
            
        old_nodes_dict = get_nodes(old_workflow)
        new_nodes_dict = get_nodes(new_workflow)
        
        # Check for added/removed/changed nodes
        old_nodes = set(old_nodes_dict.keys())
        new_nodes = set(new_nodes_dict.keys())
        
        added_nodes = new_nodes - old_nodes
        removed_nodes = old_nodes - new_nodes
        common_nodes = old_nodes.intersection(new_nodes)
        
        if added_nodes:
            node_types = [new_nodes_dict[node_id].get("class_type", "unknown") 
                         for node_id in added_nodes]
            changes.append(f"Added {len(added_nodes)} node(s): {', '.join(node_types)}")
        
        if removed_nodes:
            node_types = [old_nodes_dict[node_id].get("class_type", "unknown") 
                         for node_id in removed_nodes]
            changes.append(f"Removed {len(removed_nodes)} node(s): {', '.join(node_types)}")
        
        # Check for changed node configurations
        changed_nodes = []
        for node_id in common_nodes:
            old_node = old_nodes_dict[node_id]
            new_node = new_nodes_dict[node_id]
            
            # Check if node type changed
            if old_node.get("class_type") != new_node.get("class_type"):
                changed_nodes.append(node_id)
                changes.append(f"Node {node_id} type changed from {old_node.get('class_type')} to {new_node.get('class_type')}")
            
            # Check if inputs changed significantly (keys added/removed)
            old_inputs = set(old_node.get("inputs", {}).keys())
            new_inputs = set(new_node.get("inputs", {}).keys())
            
            if old_inputs != new_inputs:
                changed_nodes.append(node_id)
                changes.append(f"Node {node_id} inputs changed")
        
        # Check for changed connections
        old_links = old_workflow.get("links", [])
        new_links = new_workflow.get("links", [])
        
        # Simplify links to tuples for comparison
        old_link_tuples = [tuple(link[:4]) if len(link) >= 4 else tuple(link) for link in old_links]
        new_link_tuples = [tuple(link[:4]) if len(link) >= 4 else tuple(link) for link in new_links]
        
        old_link_set = set(old_link_tuples)
        new_link_set = set(new_link_tuples)
        
        added_links = new_link_set - old_link_set
        removed_links = old_link_set - new_link_set
        
        if added_links:
            changes.append(f"Added {len(added_links)} connection(s)")
        
        if removed_links:
            changes.append(f"Removed {len(removed_links)} connection(s)")
        
        return changes
    
    def is_breaking_change(self, changes: List[str]) -> bool:
        """Determine if changes might be breaking"""
        # Consider removal of nodes or connections as potentially breaking
        for change in changes:
            if change.startswith("Removed") or "type changed" in change:
                return True
        return False
    
    def track_workflow(self, file_path: str) -> Tuple[bool, WorkflowHistory]:
        """Track a single workflow file, updating its version history"""
        filename = os.path.basename(file_path)
        print(f"Tracking workflow: {filename}")
        
        try:
            # Load the workflow
            with open(file_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
            
            # Calculate workflow ID and hashes
            workflow_id = f"{filename}"
            content_hash = self.calculate_hash(workflow)
            structure_hash = self.calculate_structure_hash(workflow)
            
            # Count nodes and links
            node_count = len(workflow.get("nodes", {}))
            link_count = len(workflow.get("links", []))
            
            # Check if we already have history for this workflow
            is_new = False
            if workflow_id not in self.workflow_history:
                is_new = True
                self.workflow_history[workflow_id] = WorkflowHistory(workflow_id, filename)
            
            history = self.workflow_history[workflow_id]
            latest = history.get_latest_version()
            
            # Determine if this is a new version
            changes = []
            if latest is None:
                # First version
                changes = ["Initial version"]
            elif latest.hash_value != content_hash:
                # Content has changed, analyze the differences
                # For this, we need to load the previous version's content
                # This is a simplification - in a real system, we might store the actual content
                if os.path.exists(file_path + ".prev"):
                    with open(file_path + ".prev", 'r', encoding='utf-8') as f:
                        prev_workflow = json.load(f)
                    changes = self.detect_changes(prev_workflow, workflow)
                else:
                    changes = ["Content changed, but unable to compare with previous version"]
            else:
                # No changes, just return the history
                return is_new, history
            
            # Create a new version
            new_version = WorkflowVersion(
                hash_value=content_hash,
                timestamp=datetime.datetime.now(),
                node_count=node_count,
                link_count=link_count,
                structure_hash=structure_hash,
                changes=changes
            )
            
            # Add to history
            history.add_version(new_version)
            
            # Save current content for future comparison
            with open(file_path + ".prev", 'w', encoding='utf-8') as f:
                json.dump(workflow, f, indent=2)
            
            return is_new, history
            
        except json.JSONDecodeError as e:
            print(f"Error parsing workflow {filename}: {e}")
        except Exception as e:
            print(f"Error tracking workflow {filename}: {e}")
        
        return False, None
    
    def track_all_workflows(self) -> Dict[str, WorkflowHistory]:
        """Track all workflows in the directory"""
        workflow_files = self.find_workflow_files()
        print(f"Found {len(workflow_files)} workflow files")
        
        new_workflows = []
        updated_workflows = []
        
        for file_path in workflow_files:
            is_new, history = self.track_workflow(file_path)
            if history:
                if is_new:
                    new_workflows.append(history.filename)
                else:
                    latest = history.get_latest_version()
                    if latest and latest.changes:
                        updated_workflows.append(history.filename)
        
        print(f"New workflows: {len(new_workflows)}")
        for name in new_workflows:
            print(f"  - {name}")
        
        print(f"Updated workflows: {len(updated_workflows)}")
        for name in updated_workflows:
            print(f"  - {name}")
        
        # Save the updated history
        self.save_history()
        
        return self.workflow_history
    
    def generate_version_report(self) -> str:
        """Generate a report of workflow versions"""
        report_path = os.path.join(self.output_dir, "workflow_versions_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ComfyUI Workflow Version Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- Total workflows tracked: {len(self.workflow_history)}\n")
            
            # Count total versions
            total_versions = sum(len(h.versions) for h in self.workflow_history.values())
            f.write(f"- Total versions recorded: {total_versions}\n")
            
            # Recent changes
            recent_changes = []
            breaking_changes = []
            
            for workflow_id, history in self.workflow_history.items():
                if not history.versions:
                    continue
                    
                latest = history.get_latest_version()
                if not latest:
                    continue
                
                # Consider changes in the last week as recent
                one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
                if latest.timestamp >= one_week_ago and latest.changes:
                    recent_changes.append((history.filename, latest))
                
                # Check for breaking changes
                if self.is_breaking_change(latest.changes):
                    breaking_changes.append((history.filename, latest))
            
            f.write(f"- Workflows with recent changes: {len(recent_changes)}\n")
            f.write(f"- Workflows with potentially breaking changes: {len(breaking_changes)}\n\n")
            
            # Recent changes section
            if recent_changes:
                f.write("## Recent Changes\n\n")
                for filename, version in sorted(recent_changes, key=lambda x: x[1].timestamp, reverse=True):
                    f.write(f"### {filename}\n\n")
                    f.write(f"Updated: {version.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("Changes:\n")
                    for change in version.changes:
                        f.write(f"- {change}\n")
                    f.write("\n")
            
            # Breaking changes section
            if breaking_changes:
                f.write("## Potentially Breaking Changes\n\n")
                f.write("The following workflows have changes that might affect compatibility:\n\n")
                for filename, version in breaking_changes:
                    f.write(f"### {filename}\n\n")
                    f.write(f"Updated: {version.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("Changes:\n")
                    for change in version.changes:
                        f.write(f"- {change}\n")
                    f.write("\n")
            
            # Individual workflow histories
            f.write("## Workflow Histories\n\n")
            for workflow_id, history in sorted(self.workflow_history.items(), key=lambda x: x[1].filename):
                f.write(f"### {history.filename}\n\n")
                f.write(f"Total versions: {len(history.versions)}\n\n")
                
                if history.versions:
                    f.write("| Version | Date | Nodes | Links | Changes |\n")
                    f.write("|---------|------|-------|-------|--------|\n")
                    
                    # Sort versions by timestamp (newest first)
                    sorted_versions = sorted(history.versions, key=lambda v: v.timestamp, reverse=True)
                    
                    for version in sorted_versions:
                        changes_summary = ", ".join(version.changes) if version.changes else "No changes"
                        # Truncate long change summaries
                        if len(changes_summary) > 50:
                            changes_summary = changes_summary[:47] + "..."
                        
                        f.write(f"| {version.hash_value[:8]} | {version.timestamp.strftime('%Y-%m-%d')} | {version.node_count} | {version.link_count} | {changes_summary} |\n")
                
                f.write("\n")
                
            f.write("\n---\n")
            f.write("*This report was automatically generated by the ComfyUI workflow version tracker.*\n")
        
        print(f"Report generated at {report_path}")
        return report_path
    
    def generate_compatibility_matrix(self) -> str:
        """Generate a compatibility matrix showing which workflows are compatible with different configurations"""
        report_path = os.path.join(self.output_dir, "workflow_compatibility_matrix.md")
        
        # Define hardware configurations to check against
        hw_configs = {
            "minimum": {
                "description": "Minimum Requirements (8GB GPU)",
                "memory": 8
            },
            "recommended": {
                "description": "Recommended Requirements (12GB GPU)",
                "memory": 12
            },
            "high_end": {
                "description": "High-End System (24GB+ GPU)",
                "memory": 24
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ComfyUI Workflow Compatibility Matrix\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Compatibility matrix
            f.write("## Hardware Compatibility Matrix\n\n")
            f.write("This matrix shows which workflows are expected to be compatible with different hardware configurations:\n\n")
            
            f.write("| Workflow | Min Memory | ")
            for config_id, config in hw_configs.items():
                f.write(f"{config['description']} | ")
            f.write("\n")
            
            f.write("|----------|------------|")
            for _ in hw_configs:
                f.write("------------|")
            f.write("\n")
            
            # Estimate memory requirements for each workflow
            # This is a simplification - in a real system, we would analyze each workflow in detail
            for workflow_id, history in sorted(self.workflow_history.items(), key=lambda x: x[1].filename):
                latest = history.get_latest_version()
                if not latest:
                    continue
                
                # Estimate memory based on node and link count (very simplistic)
                # In a real system, we would use the actual analysis from analyze_workflow_requirements.py
                # For now, this is just a placeholder estimate
                estimated_memory = max(4, latest.node_count * 0.5 + latest.link_count * 0.2)
                
                f.write(f"| {history.filename} | {estimated_memory:.1f}GB | ")
                
                # Check compatibility with each hardware configuration
                for config_id, config in hw_configs.items():
                    if estimated_memory <= config["memory"]:
                        f.write("✅ Compatible | ")
                    elif estimated_memory <= config["memory"] * 1.2:
                        f.write("⚠️ Marginal | ")
                    else:
                        f.write("❌ Not Compatible | ")
                
                f.write("\n")
            
            f.write("\n")
            f.write("### Compatibility Levels\n\n")
            f.write("- ✅ **Compatible**: The workflow should run smoothly on this hardware.\n")
            f.write("- ⚠️ **Marginal**: The workflow may run, but could experience performance issues or out-of-memory errors.\n")
            f.write("- ❌ **Not Compatible**: The workflow is unlikely to run successfully on this hardware.\n\n")
            
            f.write("## Notes\n\n")
            f.write("1. Memory estimates are based on workflow complexity and may vary based on specific models used.\n")
            f.write("2. Actual compatibility depends on specific models loaded, resolution settings, and batch sizes.\n")
            f.write("3. Consider using smaller models, lower resolutions, or disabling specific nodes to improve compatibility.\n\n")
            
            f.write("---\n")
            f.write("*This matrix was automatically generated by the ComfyUI workflow version tracker.*\n")
        
        print(f"Compatibility matrix generated at {report_path}")
        return report_path
    
    def generate_github_summary(self) -> None:
        """Generate GitHub step summary with workflow version information"""
        if not os.environ.get('GITHUB_STEP_SUMMARY'):
            return
        
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as f:
            f.write("## ComfyUI Workflow Versions\n\n")
            
            # Count workflows with recent changes
            recent_changes = []
            breaking_changes = []
            
            for workflow_id, history in self.workflow_history.items():
                if not history.versions:
                    continue
                    
                latest = history.get_latest_version()
                if not latest:
                    continue
                
                # Consider changes in the last week as recent
                one_week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
                if latest.timestamp >= one_week_ago and latest.changes:
                    recent_changes.append((history.filename, latest))
                
                # Check for breaking changes
                if self.is_breaking_change(latest.changes):
                    breaking_changes.append((history.filename, latest))
            
            # Status indicators
            if breaking_changes:
                f.write(f"⚠️ **{len(breaking_changes)} workflow(s) have potentially breaking changes**\n\n")
            elif recent_changes:
                f.write(f"ℹ️ **{len(recent_changes)} workflow(s) have recent changes**\n\n")
            else:
                f.write("✅ **All workflows are stable**\n\n")
            
            # Summary stats
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Workflows | {len(self.workflow_history)} |\n")
            total_versions = sum(len(h.versions) for h in self.workflow_history.values())
            f.write(f"| Total Versions | {total_versions} |\n")
            f.write(f"| Recent Changes | {len(recent_changes)} |\n")
            f.write(f"| Breaking Changes | {len(breaking_changes)} |\n\n")
            
            # Show recent changes
            if recent_changes:
                f.write("### Recent Changes\n\n")
                
                for filename, version in sorted(recent_changes, key=lambda x: x[1].timestamp, reverse=True)[:5]:  # Show at most 5
                    change_summary = ", ".join(version.changes) if version.changes else "No changes"
                    # Truncate long summaries
                    if len(change_summary) > 80:
                        change_summary = change_summary[:77] + "..."
                    
                    f.write(f"- **{filename}** ({version.timestamp.strftime('%Y-%m-%d')}): {change_summary}\n")
                
                if len(recent_changes) > 5:
                    f.write(f"\n... and {len(recent_changes) - 5} more.\n")
                
                f.write("\nSee workflow version report artifact for details.\n")
    
    def run(self) -> None:
        """Run the workflow version tracking process"""
        print(f"Tracking ComfyUI workflow versions in {self.workflows_dir}...")
        self.track_all_workflows()
        self.generate_version_report()
        self.generate_compatibility_matrix()
        self.generate_github_summary()


def main():
    parser = argparse.ArgumentParser(description="Track ComfyUI workflow versions")
    parser.add_argument("--workflows-dir", default="WebUI/external/workflows", help="Directory containing workflow files")
    parser.add_argument("--history-file", default=None, help="File to store version history (default: ci_artifacts/workflow_versions/workflow_history.json)")
    parser.add_argument("--output-dir", default="ci_artifacts/workflow_versions", help="Directory to store output reports")
    args = parser.parse_args()
    
    tracker = WorkflowVersionTracker(
        workflows_dir=args.workflows_dir,
        history_file=args.history_file,
        output_dir=args.output_dir
    )
    
    tracker.run()


if __name__ == "__main__":
    main() 