"""
Web Request Body Definitions Module
----------------------------------
This module defines data structures used for parsing and validating web request bodies
in the service. It utilizes marshmallow_dataclass for data validation and serialization/deserialization.

Each class represents a specific request body structure used in the API endpoints.

The dataclasses define:
- Structure validation for API request bodies
- Type conversion and validation of input parameters
- Serialization/deserialization between JSON and Python objects

This module serves as a contract between the client and server, ensuring that
requests conform to the expected structure before being processed by the service.
"""

from typing import List, Optional

import marshmallow_dataclass
from marshmallow import EXCLUDE  # Used to ignore unknown fields in requests


@marshmallow_dataclass.dataclass
class DownloadModelData:
    """
    Data structure for model download requests.
    
    Represents a single model to be downloaded from a repository.
    Used in API endpoints related to model management and downloading.
    
    Attributes:
        type: Integer identifier for the model type (e.g., 0=SD, 1=LLM, etc.).
        repo_id: String identifier for the model repository (e.g., "runwayml/stable-diffusion-v1-5").
        backend: String identifier for the backend service (e.g., "pytorch", "onnx").
        additionalLicenseLink: Optional URL to additional license information that may need to be
                              accepted before downloading the model.
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
    
    Used in the "/api/downloadModel" endpoint to specify a batch of models to download.
    The service processes each model in the list sequentially.
    
    Attributes:
        data: List of DownloadModelData objects to be processed, each representing 
              a separate model to download.
    """
    data : List[DownloadModelData]

@marshmallow_dataclass.dataclass
class ComfyUICustomNodesGithubRepoId:
    """
    Identifies a GitHub repository containing ComfyUI custom nodes.
    
    Custom nodes extend the functionality of ComfyUI with additional processing nodes,
    allowing for more complex workflows and integrations.
    
    Attributes:
        username: GitHub username of the repository owner (e.g., "comfyanonymous").
        repoName: Name of the GitHub repository containing custom nodes (e.g., "ComfyUI_experiments").
        gitRef: Optional git reference (branch, tag, or commit hash) to checkout. If None,
                the default branch (usually 'main' or 'master') will be used.
    """
    username: str
    repoName: str
    gitRef: Optional[str]

@marshmallow_dataclass.dataclass
class ComfyUICustomNodesDownloadRequest:
    """
    Request body for downloading ComfyUI custom nodes from GitHub repositories.
    
    Used in the "/api/comfyUi/loadCustomNodes" endpoint to install custom nodes
    that extend the functionality of ComfyUI.
    
    Attributes:
        data: List of GitHub repository identifiers to download custom nodes from.
              Each entry specifies a separate repository containing custom nodes.
    """
    data : List[ComfyUICustomNodesGithubRepoId]

@marshmallow_dataclass.dataclass
class ComfyUICheckWorkflowRequirementRequest:
    """
    Request body for checking ComfyUI workflow requirements.
    
    Used in the "/api/comfyUi/checkWorkflowRequirements" endpoint to verify whether
    all requirements for a workflow are installed before attempting to run it.
    
    Attributes:
        pythonPackages: List of Python package names required by the workflow.
                        These are packages that need to be installed via pip.
        customNodes: List of GitHub repositories containing custom nodes required by the workflow.
                    These need to be installed from GitHub before the workflow can run.
    """
    pythonPackages : List[str]
    customNodes : List[ComfyUICustomNodesGithubRepoId]

@marshmallow_dataclass.dataclass
class ComfyUIPackageInstallRequest:
    """
    Request body for installing Python packages for ComfyUI.
    
    Used in the "/api/comfyUi/installPythonPackage" endpoint to install
    Python dependencies required by ComfyUI custom nodes.
    
    Attributes:
        data: List of Python package names to install via pip. May include
              version specifiers (e.g., "numpy>=1.20.0").
    """
    data : List[str]
