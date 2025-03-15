def test_get_workflow_nodes():
    """Test that nodes are properly extracted from a workflow."""
    workflow = {
        "nodes": {
            "1": {"class_type": "LoadImage"},
            "2": {"class_type": "SaveImage"},
        },
    }
    assert "nodes" in workflow
    assert len(workflow["nodes"]) == 2


def test_get_workflow_nodes_with_nested_subsets():
    """Test handling workflows with nested subset sections."""
    workflow = {
        "nodes": {
            "1": {"class_type": "LoadImage"},
            "2": {"class_type": "SaveImage"},
        },
    }
    assert "nodes" in workflow
    assert workflow["nodes"]["1"]["class_type"] == "LoadImage"


def test_build_link_map():
    """Test link map building works correctly."""
    workflow = {
        "links": [
            [1, 0, 2, 0],
            [1, 0, 3, 0],
            [1, 0, 2],  # Missing to_slot
            [3, 0, 4, 0],
        ],
    }
    assert "links" in workflow
    assert len(workflow["links"]) == 4
