"""
Utilities for parsing ComfyUI workflow files.
Centralizes the logic for handling different workflow formats.
"""

from typing import Dict, Any, Optional, List, Tuple, Union


def get_workflow_nodes(workflow: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract nodes from a workflow, handling different formats:
    - Traditional format with top-level nodes
    - API format with nodes inside comfyUiApiWorkflow.nodes
    - API format with nodes directly inside comfyUiApiWorkflow
    
    Args:
        workflow: The loaded workflow JSON
        
    Returns:
        Dict containing the nodes, or None if nodes can't be found
    """
    if "nodes" in workflow and isinstance(workflow["nodes"], dict):
        # Traditional ComfyUI format with top-level nodes
        return workflow["nodes"]
    elif "comfyUiApiWorkflow" in workflow and isinstance(workflow["comfyUiApiWorkflow"], dict):
        if "nodes" in workflow["comfyUiApiWorkflow"] and isinstance(workflow["comfyUiApiWorkflow"]["nodes"], dict):
            # API format with nodes inside comfyUiApiWorkflow.nodes
            return workflow["comfyUiApiWorkflow"]["nodes"]
        else:
            # API format with nodes directly inside comfyUiApiWorkflow (numeric keys)
            # Check if it has node-like structure (typically with keys that are numeric strings)
            has_nodes = any(
                isinstance(v, dict) and "class_type" in v 
                for k, v in workflow["comfyUiApiWorkflow"].items()
            )
            if has_nodes:
                return workflow["comfyUiApiWorkflow"]
    
    return None


def get_workflow_links(workflow: Dict[str, Any]) -> Optional[List[List[Any]]]:
    """
    Extract links from a workflow, handling different formats.
    
    Args:
        workflow: The loaded workflow JSON
        
    Returns:
        List of links, or None if links can't be found
    """
    if "links" in workflow and isinstance(workflow["links"], list):
        # Traditional format with top-level links
        return workflow["links"]
    elif "comfyUiApiWorkflow" in workflow and isinstance(workflow["comfyUiApiWorkflow"], dict):
        if "links" in workflow["comfyUiApiWorkflow"] and isinstance(workflow["comfyUiApiWorkflow"]["links"], list):
            # API format with links inside comfyUiApiWorkflow
            return workflow["comfyUiApiWorkflow"]["links"]
    
    return None


def build_link_map(workflow: Dict[str, Any]) -> Dict[str, Tuple[str, str, str]]:
    """
    Build a map of node connections from a workflow.
    
    Args:
        workflow: The loaded workflow JSON
        
    Returns:
        Dict mapping target slots to source nodes and slots
    """
    link_map = {}
    links = get_workflow_links(workflow)
    
    if not links:
        return link_map
    
    for link in links:
        # Skip malformed links
        if len(link) < 4:
            continue
        
        # Format: [from_node, from_slot, to_node, to_slot, ...]
        from_node, from_slot, to_node, to_slot = link[:4]
        
        # Ensure node IDs are strings
        from_node = str(from_node)
        to_node = str(to_node)
        
        # Create a unique key for the target slot
        target_key = f"{to_node}:{to_slot}"
        
        # Map the target slot to the source node and slot
        link_map[target_key] = (from_node, from_slot, target_key)
    
    return link_map


def get_workflow_attribute(workflow: Dict[str, Any], attribute: str) -> Any:
    """
    Get an attribute from a workflow, checking both top-level and inside comfyUiApiWorkflow.
    
    Args:
        workflow: The loaded workflow JSON
        attribute: The attribute name to retrieve
        
    Returns:
        The attribute value if found, None otherwise
    """
    if attribute in workflow:
        return workflow[attribute]
    elif "comfyUiApiWorkflow" in workflow and isinstance(workflow["comfyUiApiWorkflow"], dict):
        if attribute in workflow["comfyUiApiWorkflow"]:
            return workflow["comfyUiApiWorkflow"][attribute]
    
    return None


def get_node_class_type(node: Dict[str, Any]) -> Optional[str]:
    """
    Get the class_type of a node, which indicates its functionality.
    
    Args:
        node: The node dictionary
        
    Returns:
        The class_type string if found, None otherwise
    """
    return node.get("class_type")


def get_node_inputs(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the inputs of a node, handling different formats.
    
    Args:
        node: The node dictionary
        
    Returns:
        Dict of input parameters, or empty dict if none found
    """
    if "inputs" in node and isinstance(node["inputs"], dict):
        return node["inputs"]
    return {} 