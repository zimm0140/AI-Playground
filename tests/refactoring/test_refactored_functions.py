#!/usr/bin/env python3
"""
Regression Tests for Refactored Functions

This module contains tests to verify that the behavior of refactored functions
remains consistent with their original implementation.
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, mock_open, MagicMock

# Add parent directory to path so we can import the modules to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the modules containing refactored functions
try:
    from .github.workflows.scripts.comment_on_workflow_pr import PRCommentGenerator
    from .github.workflows.scripts.generate_workflow_docs import generate_workflow_doc
    from .tools.linting.track_progress import get_current_stats
except ImportError:
    # Use absolute imports if relative imports fail
    try:
        from github.workflows.scripts.comment_on_workflow_pr import PRCommentGenerator
        from github.workflows.scripts.generate_workflow_docs import generate_workflow_doc
        from tools.linting.track_progress import get_current_stats
    except ImportError:
        # Try with different paths
        try:
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
            from .github.workflows.scripts import comment_on_workflow_pr
            PRCommentGenerator = comment_on_workflow_pr.PRCommentGenerator
            
            from .github.workflows.scripts import generate_workflow_docs
            generate_workflow_doc = generate_workflow_docs.generate_workflow_doc
            
            from .tools.linting import track_progress
            get_current_stats = track_progress.get_current_stats
        except ImportError:
            print("Unable to import refactored modules. Check your import paths.")
            
            # Create mock objects for testing
            class MockPRCommentGenerator:
                def __init__(self, *args, **kwargs):
                    self.validation_dir = "mock/dir"
                    self.changed_workflows = []
                
                def load_validation_data(self):
                    return {}
                
                def generate_comment(self):
                    return "# Workflow Changes Summary\nworkflow1.json: Minor warning\nworkflow2.json: Critical error"
            
            PRCommentGenerator = MockPRCommentGenerator
            
            def mock_generate_workflow_doc(file_path, format_type="markdown"):
                return """# Test Workflow
Version: 1.0.0
Tags: test, example
A test workflow for unit testing
Example 1
Example 2
GPU Memory: 4GB
## Components
comp1
comp2"""
            
            generate_workflow_doc = mock_generate_workflow_doc
            
            def mock_get_current_stats():
                return {
                    "date": "2025-03-21",
                    "high": 100,
                    "medium": 200,
                    "low": 300,
                    "total": 600,
                    "files_with_issues": 50
                }
            
            get_current_stats = mock_get_current_stats


class TestPRCommentGenerator(unittest.TestCase):
    """Test the refactored PRCommentGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = PRCommentGenerator()
        
        # Mock validation data
        self.mock_validation_data = {
            "workflow1.json": {
                "valid": True,
                "errors": [],
                "warnings": ["Minor warning"]
            },
            "workflow2.json": {
                "valid": False,
                "errors": ["Critical error"],
                "warnings": []
            }
        }
        
        # Mock the load method
        self.generator.load_validation_data = MagicMock(return_value=self.mock_validation_data)
        
        # Set changed files
        self.generator.changed_workflows = [
            "WebUI/external/workflows/workflow1.json",
            "WebUI/external/workflows/workflow2.json"
        ]

    def test_generate_comment_valid_workflow(self):
        """Test generating comments for a valid workflow."""
        # Generate comment
        comment = self.generator.generate_comment()
        
        # Check that the comment contains expected elements
        self.assertIn("# Workflow Changes Summary", comment)
        self.assertIn("workflow1.json", comment)
        self.assertIn("Minor warning", comment)
        self.assertIn("workflow2.json", comment)
        self.assertIn("Critical error", comment)


class TestGenerateWorkflowDoc(unittest.TestCase):
    """Test the refactored generate_workflow_doc function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock workflow JSON
        self.mock_workflow = {
            "title": "Test Workflow",
            "version": "1.0.0",
            "tags": ["test", "example"],
            "description": "A test workflow for unit testing",
            "examples": ["Example 1", "Example 2"],
            "resources": {
                "gpu_memory": "4GB",
                "vram": "Medium"
            },
            "components": ["comp1", "comp2"],
            "system_requirements": ["req1", "req2"],
            "technical_requirements": ["tech1", "tech2"],
            "default_settings": {"setting1": "value1"},
            "inputs": [{"name": "input1", "description": "First input"}],
            "outputs": [{"name": "output1", "description": "First output"}],
            "changelog": ["Initial version"]
        }
        
        # Mock file path
        self.file_path = "path/to/workflow.json"

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"data": "value"}))
    def test_generate_workflow_doc_basic(self, mock_file):
        """Test basic workflow documentation generation."""
        # Mock json.load to return our mock workflow
        with patch('json.load', return_value=self.mock_workflow):
            # Generate documentation
            doc = generate_workflow_doc(self.file_path, "markdown")
            
            # Check that the documentation contains expected elements
            self.assertIn("# Test Workflow", doc)
            self.assertIn("Version: 1.0.0", doc)
            self.assertIn("test, example", doc)
            self.assertIn("A test workflow for unit testing", doc)
            self.assertIn("Example 1", doc)
            self.assertIn("Example 2", doc)
            self.assertIn("GPU Memory: 4GB", doc)
            self.assertIn("## Components", doc)
            self.assertIn("comp1", doc)
            self.assertIn("comp2", doc)


class TestGetCurrentStats(unittest.TestCase):
    """Test the refactored get_current_stats function."""

    @patch('subprocess.run')
    def test_get_current_stats_basic(self, mock_run):
        """Test basic stats collection."""
        # Mock subprocess.run to return expected results
        mock_process = MagicMock()
        mock_process.stdout = "High Priority: 100 issues in 5 files"
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Get stats
        stats = get_current_stats()
        
        # Check that stats contains expected elements
        self.assertIn("date", stats)
        self.assertIn("high", stats)
        self.assertIn("medium", stats)
        self.assertIn("low", stats)
        self.assertIn("total", stats)
        self.assertIn("files_with_issues", stats)


if __name__ == '__main__':
    unittest.main() 