"""
Hugging Face Model Downloader API Module
---------------------------------------
This module provides an interface for querying information about models from the Hugging Face Hub.
It allows for enumerating files in a repository and collecting metadata about available models.

This is used for exploring models before downloading them with the more comprehensive downloader modules.
"""

import sys
from json import dumps
from os import path

from huggingface_hub import HfFileSystem, hf_hub_url


class ModelDownloaderApi:
    """
    API for listing and exploring models and their files on the Hugging Face Hub.

    This class provides methods to query information about models in the Hugging Face Hub,
    including listing files, computing total size, and generating download URLs.

    Attributes:
        repo_id: Identifier of the Hugging Face repository (e.g., "runwayml/stable-diffusion-v1-5").
        file_queue: List of files found in the repository.
        total_size: Total size of all files in the repository in bytes.
        fs: HuggingFace FileSystem interface for accessing repository contents.
        repo_folder: Local folder name derived from the repository ID.
    """

    repo_id: str
    file_queue: list
    total_size: int
    fs: HfFileSystem
    repo_folder: str

    def __init__(self):
        """
        Initialize the ModelDownloaderApi with an empty file queue and HF filesystem.
        """
        self.file_queue = []
        self.fs = HfFileSystem()

    def get_info(self, repo_id: str, is_sd=False):
        """
        Get information about a Hugging Face repository and enumerate its files.

        Sets the repo_id and repo_folder attributes, clears any existing file queue,
        and calls enum_file_list to populate the file queue with files from the repository.

        Args:
            repo_id: The Hugging Face repository ID to query.
            is_sd: Boolean flag indicating if this is a Stable Diffusion model, which
                  affects how files are enumerated.

        Returns:
            Dictionary containing repository information including file list and total size.
        """
        self.repo_id = repo_id
        self.repo_folder = repo_id.replace("/", "---")
        self.file_queue.clear()
        self.total_size = 0
        self.enum_file_list(repo_id, is_sd, True)
        print(dumps({"total_size": self.total_size, "file_list": self.file_queue}))

        # Also return a dictionary for API usage
        return {
            "repo_id": repo_id,
            "repo_folder": self.repo_folder,
            "file_list": self.file_queue,
            "total_size": self.total_size,
        }

    def enum_file_list(self, enum_path: str, is_sd=False, is_root=True):
        """
        Enumerate files in a repository path and add them to the file queue.

        Recursively explores the repository structure, filtering out unwanted files,
        adding file information to the file_queue and accumulating the total size.

        Args:
            enum_path: The repository path or ID to enumerate.
            is_sd: Boolean flag indicating if this is a Stable Diffusion model,
                  which affects file filtering (ignores root .safetensors, .pt, and .ckpt files).
            is_root: Boolean flag indicating if this is the root call in the recursion.

        Returns:
            The updated file queue if this is the root call, otherwise None.
        """
        list = self.fs.ls(enum_path, detail=True)
        for item in list:
            name: str = item.get("name")
            size: int = item.get("size")
            type: str = item.get("type")
            if type == "directory":
                self.enum_file_list(name, is_sd, False)
            else:
                # sd model ignore root .safetensors .pt .ckpt files
                if (
                    is_sd
                    and is_root
                    and (name.endswith(".safetensors") or name.endswith(".pt") or name.endswith(".ckpt"))
                ) or (
                    name.endswith(".png")
                    or name.endswith(".gitattributes")
                    or name.endswith(".md")
                    or name.endswith(".jpg")
                ):
                    continue

                self.total_size += size
                relative_path = path.relpath(name, self.repo_id)
                subfolder = path.dirname(relative_path).replace("\\", "/")
                filename = path.basename(relative_path)
                url = hf_hub_url(repo_id=self.repo_id, filename=filename, subfolder=subfolder)
                self.file_queue.append(
                    {
                        "name": name.replace(self.repo_id, self.repo_folder),
                        "size": size,
                        "url": url,
                    }
                )

        if is_root:
            return self.file_queue


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.exit(1)
    else:
        ModelDownloaderApi().get_info(sys.argv[1], int(sys.argv[2]) != 0 if sys.argv.__len__() > 2 else False)
