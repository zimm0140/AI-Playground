#!/usr/bin/env python
"""
Unit tests for workflow_parser.py

These tests verify that the workflow parsing functions correctly handle
different workflow formats and edge cases.
"""

import os
import sys
import pytest
from typing import Dict, Any

# Add the GitHub workflows scripts directory to the Python path
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(script_dir, ".github", "workflows", "scripts"))

from utils.workflow_parser import (
    get_workflow_nodes,
    get_workflow_links,
    build_link_map,
    get_workflow_attribute,
    get_node_class_type,
    get_node_inputs
)


def test_get_workflow_nodes_traditional_format():
    """Test extracting nodes from traditional format workflow"""
    workflow = {
        "nodes": {
            "1": {"class_type": "LoadImage"},
            "2": {"class_type": "SaveImage"}
        }
    }
    nodes = get_workflow_nodes(workflow)
    assert nodes is not None
    assert "1" in nodes
    assert "2" in nodes
    assert nodes["1"]["class_type"] == "LoadImage"
    assert nodes["2"]["class_type"] == "SaveImage"


def test_get_workflow_nodes_api_format_nested():
    """Test extracting nodes from API format with nested nodes"""
    workflow = {
        "comfyUiApiWorkflow": {
            "nodes": {
                "1": {"class_type": "LoadImage"},
                "2": {"class_type": "SaveImage"}
            }
        }
    }
    nodes = get_workflow_nodes(workflow)
    assert nodes is not None
    assert "1" in nodes
    assert "2" in nodes
    assert nodes["1"]["class_type"] == "LoadImage"
    assert nodes["2"]["class_type"] == "SaveImage"


def test_get_workflow_nodes_api_format_direct():
    """Test extracting nodes from API format with direct nodes"""
    workflow = {
        "comfyUiApiWorkflow": {
            "1": {"class_type": "LoadImage"},
            "2": {"class_type": "SaveImage"}
        }
    }
    nodes = get_workflow_nodes(workflow)
    assert nodes is not None
    assert "1" in nodes
    assert "2" in nodes
    assert nodes["1"]["class_type"] == "LoadImage"
    assert nodes["2"]["class_type"] == "SaveImage"


def test_get_workflow_nodes_invalid():
    """Test handling of invalid workflow formats"""
    # Empty workflow
    workflow: Dict[str, Any] = {}
    nodes = get_workflow_nodes(workflow)
    assert nodes is None
    
    # Workflow with missing nodes structure
    workflow = {"not_a_valid_key": {}}
    nodes = get_workflow_nodes(workflow)
    assert nodes is None
    
    # Workflow with non-dict nodes
    workflow = {"nodes": "not a dict"}
    nodes = get_workflow_nodes(workflow)
    assert nodes is None
    
    # Workflow with empty comfyUiApiWorkflow
    workflow = {"comfyUiApiWorkflow": {}}
    nodes = get_workflow_nodes(workflow)
    assert nodes is None


def test_get_workflow_links_traditional_format():
    """Test extracting links from traditional format workflow"""
    workflow = {
        "links": [
            [1, 0, 2, 0],
            [3, 0, 4, 0]
        ]
    }
    links = get_workflow_links(workflow)
    assert links is not None
    assert len(links) == 2
    assert links[0] == [1, 0, 2, 0]
    assert links[1] == [3, 0, 4, 0]


def test_get_workflow_links_api_format():
    """Test extracting links from API format workflow"""
    workflow = {
        "comfyUiApiWorkflow": {
            "links": [
                [1, 0, 2, 0],
                [3, 0, 4, 0]
            ]
        }
    }
    links = get_workflow_links(workflow)
    assert links is not None
    assert len(links) == 2
    assert links[0] == [1, 0, 2, 0]
    assert links[1] == [3, 0, 4, 0]


def test_get_workflow_links_invalid():
    """Test handling of invalid links formats"""
    # Empty workflow
    workflow: Dict[str, Any] = {}
    links = get_workflow_links(workflow)
    assert links is None
    
    # Workflow with missing links
    workflow = {"nodes": {}}
    links = get_workflow_links(workflow)
    assert links is None
    
    # Workflow with non-list links
    workflow = {"links": "not a list"}
    links = get_workflow_links(workflow)
    assert links is None


def test_build_link_map():
    """Test building a map of node connections"""
    workflow = {
        "links": [
            [1, 0, 2, 0],
            [3, 0, 4, 0]
        ]
    }
    link_map = build_link_map(workflow)
    assert "2:0" in link_map
    assert "4:0" in link_map
    
    # Note: from_slot is kept as a number, not converted to string
    assert link_map["2:0"][0] == "1"  # from_node is string
    assert link_map["2:0"][1] == 0    # from_slot is kept as number
    assert link_map["2:0"][2] == "2:0"  # target_key is string


def test_build_link_map_with_malformed_links():
    """Test building a link map with malformed links"""
    workflow = {
        "links": [
            [1, 0, 2],  # Missing to_slot
            [3, 0, 4, 0]
        ]
    }
    link_map = build_link_map(workflow)
    assert len(link_map) == 1
    assert "4:0" in link_map
    
    # Note: from_slot is kept as a number, not converted to string
    assert link_map["4:0"][0] == "3"  # from_node is string
    assert link_map["4:0"][1] == 0    # from_slot is kept as number
    assert link_map["4:0"][2] == "4:0"  # target_key is string


def test_get_workflow_attribute():
    """Test getting attributes from workflows"""
    workflow = {
        "name": "Test Workflow",
        "comfyUiApiWorkflow": {
            "version": "1.0.0"
        }
    }
    
    # Get attribute from top level
    name = get_workflow_attribute(workflow, "name")
    assert name == "Test Workflow"
    
    # Get attribute from comfyUiApiWorkflow
    version = get_workflow_attribute(workflow, "version")
    assert version == "1.0.0"
    
    # Get non-existent attribute
    description = get_workflow_attribute(workflow, "description")
    assert description is None


def test_get_node_class_type():
    """Test getting class_type from nodes"""
    node = {"class_type": "LoadImage", "inputs": {}}
    class_type = get_node_class_type(node)
    assert class_type == "LoadImage"
    
    # Node without class_type
    node = {"inputs": {}}
    class_type = get_node_class_type(node)
    assert class_type is None


def test_get_node_inputs():
    """Test getting inputs from nodes"""
    node = {
        "class_type": "LoadImage",
        "inputs": {
            "image": "test.png",
            "scale": 1.0
        }
    }
    inputs = get_node_inputs(node)
    assert inputs == {"image": "test.png", "scale": 1.0}
    
    # Node without inputs
    node = {"class_type": "LoadImage"}
    inputs = get_node_inputs(node)
    assert inputs == {} 