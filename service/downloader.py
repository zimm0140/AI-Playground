"""
Hugging Face Model Downloader API Module
---------------------------------------
This module provides an interface for querying information about models from the Hugging Face Hub.
It allows for enumerating files in a repository and collecting metadata about available models.

This is used for exploring models before downloading them with the more comprehensive downloader modules.
"""

from json import dumps
from os import path
import sys
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
        self.file_queue = list()
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
        self.enum_file_list("/", is_sd)
        return {
            "repo_id": repo_id,
            "repo_folder": self.repo_folder,
            "file_list": self.file_queue,
            "total_size": self.total_size,
        }

    def enum_file_list(self, enum_path: str, is_sd=False, is_root=True):
        """
        Enumerate files in a repository path and add them to the file queue.
        
        Recursively explores the repository structure, adding file information
        to the file_queue and accumulating the total size.
        
        Args:
            enum_path: The path within the repository to enumerate.
            is_sd: Boolean flag indicating if this is a Stable Diffusion model,
                  which may require special handling for certain file types.
            is_root: Boolean flag indicating if this is the root call in the recursion.
                    
        Returns:
            The updated file queue if this is the root call, otherwise None.
        """
        print(f"enum_path={enum_path}")
        file_list = self.fs.ls(path.join(self.repo_id, enum_path), detail=True)
        
        for file in file_list:
            filetype = file["type"]
            filename = path.basename(file["name"])
            full_name = file["name"]
            file_rel_path = full_name.replace(f"{self.repo_id}/", "")
            if filetype == "directory":
                self.enum_file_list(file_rel_path, is_sd, False)
            elif filetype == "file":
                if is_sd:
                    # If is a safetensor file, is the model. Else is an extra file
                    is_model = filename.endswith(".safetensors")
                    is_extra = not is_model
                else:
                    # If it's not a sd model, all files are model files
                    is_model = True
                    is_extra = False
                fsize = file["size"]
                self.total_size += fsize
                url = hf_hub_url(self.repo_id, file_rel_path)
                # file_path = path.join(self.repo_folder, file_rel_path)
                file_info = {
                    "name": filename,
                    "path": file_rel_path,
                    "url": url,
                    "size": fsize,
                    "is_model": is_model,
                    "is_extra": is_extra,
                }
                self.file_queue.append(file_info)
                # print(f'file_path={file_path}')
                # print(f'url={url}')
        
        if is_root:
            return self.file_queue


if __name__ == "__main__":
    if len(sys.argv) == 1:
        exit(1)
    else:
        ModelDownloaderApi().get_info(
            sys.argv[1], int(sys.argv[2]) != 0 if sys.argv.__len__() > 2 else False
        )
