"""
Web Request Body Definitions Module
----------------------------------
This module defines data structures used for parsing and validating web request bodies
in the service. It utilizes marshmallow_dataclass for data validation and serialization/deserialization.

Each class represents a specific request body structure used in the API endpoints.
"""

from typing import List, Optional

import marshmallow_dataclass
from marshmallow import EXCLUDE


@marshmallow_dataclass.dataclass
class DownloadModelData:
    """
    Data structure for model download requests.
    
    Attributes:
        type: Integer identifier for the model type.
        repo_id: String identifier for the model repository.
        backend: String identifier for the backend service.
        additionalLicenseLink: Optional URL to additional license information.
    """
    class Meta:
        unknown = EXCLUDE  # Ignores unknown fields in the request
    type : int
    repo_id : str
    backend : str
    additionalLicenseLink: Optional[str]

@marshmallow_dataclass.dataclass
class DownloadModelRequestBody:
    """
    Request body for downloading multiple models.
    
    Attributes:
        data: List of DownloadModelData objects to be processed.
    """
    data : List[DownloadModelData]

@marshmallow_dataclass.dataclass
class ComfyUICustomNodesGithubRepoId:
    """
    Identifies a GitHub repository containing ComfyUI custom nodes.
    
    Attributes:
        username: GitHub username of the repository owner.
        repoName: Name of the GitHub repository.
        gitRef: Optional git reference (branch, tag, or commit hash).
    """
    username: str
    repoName: str
    gitRef: Optional[str]

@marshmallow_dataclass.dataclass
class ComfyUICustomNodesDownloadRequest:
    """
    Request body for downloading ComfyUI custom nodes from GitHub repositories.
    
    Attributes:
        data: List of GitHub repository identifiers to download custom nodes from.
    """
    data : List[ComfyUICustomNodesGithubRepoId]

@marshmallow_dataclass.dataclass
class ComfyUICheckWorkflowRequirementRequest:
    """
    Request body for checking ComfyUI workflow requirements.
    
    Attributes:
        pythonPackages: List of Python package names required by the workflow.
        customNodes: List of GitHub repositories containing custom nodes required by the workflow.
    """
    pythonPackages : List[str]
    customNodes : List[ComfyUICustomNodesGithubRepoId]

@marshmallow_dataclass.dataclass
class ComfyUIPackageInstallRequest:
    """
    Request body for installing Python packages for ComfyUI.
    
    Attributes:
        data: List of Python package names to install.
    """
    data : List[str]
