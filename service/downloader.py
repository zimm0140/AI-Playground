import logging
from json import dumps
import sys
from huggingface_hub import HfFileSystem, hf_hub_url
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloaderApi:
    """
    API for retrieving information about models on Hugging Face Hub and constructing a download queue.

    This class interacts with the Hugging Face file system to retrieve details about a specific model repository,
    including the total size of the files and a list of individual files with their metadata. It also provides
    functionality for constructing download URLs and filtering out unnecessary files based on specified criteria.
    """
    repo_id: str  # Repository ID on Hugging Face Hub
    file_queue: List[Dict[str, Any]]  # Queue to store file metadata
    total_size: int  # Total size of all files in the repository
    fs: HfFileSystem  # File system interface for interacting with the Hugging Face Hub
    repo_folder: str  # Local folder name corresponding to the repository

    def __init__(self, fs: HfFileSystem = None):
        """
        Initializes the ModelDownloaderApi.

        Args:
            fs (HfFileSystem, optional): A Hugging Face file system object. If not provided, a new instance
                of `HfFileSystem` is created.
        """
        self.file_queue = []  # Initialize the file queue
        self.fs = fs if fs else HfFileSystem()  # Use provided file system or default to HfFileSystem
        self.total_size = 0  # Initialize the total size counter

    def get_info(self, repo_id: str, is_sd: bool = False) -> Dict[str, Any]:
        """
        Retrieves information about a model repository from the Hugging Face Hub.

        This method fetches the total size of the repository and a list of files with their metadata
        (name, size, and download URL). It also applies filtering to exclude specific files
        based on the `is_sd` flag.

        Args:
            repo_id (str): The ID of the repository on the Hugging Face Hub. 
            is_sd (bool, optional): A flag indicating if the model is a Stable Diffusion model. 
                If True, specific files associated with Stable Diffusion models are excluded. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing the 'total_size' (in bytes) and a 'file_list' of the 
                model repository. The 'file_list' is a list of dictionaries, each representing a file 
                and containing its 'name', 'size', and 'url'.

        Raises:
            Exception: If an error occurs during the information retrieval process, it is logged and re-raised.
        """
        try:
            self.repo_id = repo_id
            self.repo_folder = repo_id.replace('/', '---')
            self.file_queue.clear()
            self.total_size = 0
            self.process_repository(repo_id, is_sd)
            return {"total_size": self.total_size, "file_list": self.file_queue}
        except Exception as e:
            logger.error(f"An error occurred while fetching info for repo '{repo_id}': {e}")
            raise

    def process_repository(self, repo_id: str, is_sd: bool) -> None:
        """
        Initiates the processing of the model repository.

        This method starts the recursive enumeration of files in the repository, applying filtering
        based on the `is_sd` flag. 

        Args:
            repo_id (str): The ID of the model repository on the Hugging Face Hub.
            is_sd (bool): A flag indicating if the model is a Stable Diffusion model.
        """
        self.enum_file_list(repo_id, is_sd, True)

    def enum_file_list(self, enum_path: str, is_sd: bool = False, is_root: bool = True) -> None:
        """
        Recursively enumerates files and directories within a given path.

        This method traverses the directory structure of the repository, processing files and 
        recursively calling itself for subdirectories. It applies filtering to exclude unnecessary
        files. 

        Args:
            enum_path (str): The current path being enumerated.
            is_sd (bool, optional): A flag indicating if the model is a Stable Diffusion model. Defaults to False.
            is_root (bool, optional): A flag indicating if the current path is the root of the repository. 
                Defaults to True.

        Raises:
            Exception: If any error occurs during file listing, it is logged and re-raised.
        """
        try:
            items = self.fs.ls(enum_path, detail=True)
            for item in items:
                self.process_item(item, is_sd, is_root)
        except Exception as e:
            logger.error(f"An error occurred while enumerating files in '{enum_path}': {e}")
            raise

    def process_item(self, item: Dict[str, Any], is_sd: bool, is_root: bool) -> None:
        """
        Processes an individual item (file or directory) in the repository.

        For directories, it recursively calls `enum_file_list`. For files, it checks if the file 
        should be ignored, and if not, adds its information to the download queue. 

        Args:
            item (Dict[str, Any]): A dictionary containing the metadata of the file or directory.
            is_sd (bool): A flag indicating if special handling for Stable Diffusion models is needed.
            is_root (bool): A flag indicating if the current item is in the root directory of the repository. 
        """
        name = self.normalize_path(item.get("name"))
        size = item.get("size")
        item_type = item.get("type")

        if item_type == "directory":
            self.enum_file_list(name, is_sd, False)
        else:
            if not self.should_ignore_file(name, is_sd, is_root):
                self.add_file_to_queue(name, size)

    def should_ignore_file(self, name: str, is_sd: bool, is_root: bool) -> bool:
        """
        Determines if a file should be ignored based on its name and the type of model.

        This method implements filtering rules to exclude specific files from the download queue. 
        For Stable Diffusion models, it ignores model files (`.safetensors`, `.pt`, `.ckpt`) in 
        the root directory. Common unnecessary files like images, git attributes, and 
        documentation files are also ignored.

        Args:
            name (str): The name of the file.
            is_sd (bool): A flag indicating if the model is a Stable Diffusion model.
            is_root (bool):  A flag indicating if the file is in the root directory of the repository.

        Returns:
            bool: True if the file should be ignored, False otherwise.
        """
        sd_ignored_extensions = [".safetensors", ".pt", ".ckpt"]
        common_ignored_extensions = [".png", ".gitattributes", ".md", ".jpg"]

        if is_sd and is_root and any(name.endswith(ext) for ext in sd_ignored_extensions):
            return True
        if any(name.endswith(ext) for ext in common_ignored_extensions):
            return True
        return False

    def add_file_to_queue(self, name: str, size: int) -> None:
        """
        Adds a file to the download queue and updates the total size.

        Args:
            name (str): The name of the file.
            size (int): The size of the file in bytes.
        """
        self.total_size += size
        url = self.construct_url(name)
        self.file_queue.append({"name": name.replace(self.repo_id, self.repo_folder), "size": size, "url": url})

    def construct_url(self, name: str) -> str:
        """
        Constructs the download URL for a file on the Hugging Face Hub.

        Args:
            name (str): The name of the file.

        Returns:
            str: The constructed download URL.

        Raises:
            Exception: If an error occurs while constructing the URL, it is logged and re-raised.
        """
        try:
            # Calculate the relative path.
            relative_path = Path(name).relative_to(self.repo_id).as_posix()
            subfolder = Path(relative_path).parent.as_posix()
            filename = Path(relative_path).name
            subfolder = '' if subfolder == '.' else subfolder
            return hf_hub_url(repo_id=self.repo_id, filename=filename, subfolder=subfolder)
        except Exception as e:
            logger.error(f"An error occurred while constructing URL for '{name}': {e}")
            raise

    def normalize_path(self, name: str) -> str:
        """
        Normalizes a file path to POSIX format.

        Args:
            name (str): The file path to normalize.

        Returns:
            str: The normalized file path.

        Raises:
            ValueError: If the input path is empty or None.
            Exception: If any other error occurs during path normalization.
        """
        if not name:
            logger.error("Invalid path: Path is empty or None.")
            raise ValueError("Invalid path")
        try:
            return Path(name).as_posix()
        except Exception as e:
            logger.error(f"An error occurred while normalizing path '{name}': {e}")
            raise

# --- Main Function ---
def main() -> None:
    """
    Main function to execute the script from the command line.

    Parses command-line arguments, initializes the `ModelDownloaderApi`, and retrieves 
    repository information. Prints the results as a JSON string.

    Usage: 
        python downloader.py <repo_id> [is_sd]

    Args:
        repo_id (str): The ID of the repository to process.
        is_sd (int, optional):  A flag (0 or 1) indicating whether special handling is 
            required for Stable Diffusion models. Defaults to 0 (False).
    """
    if len(sys.argv) < 2:
        logger.error("Usage: downloader.py <repo_id> [is_sd]")
        exit(1)
    else:
        try:
            repo_id = sys.argv[1]
            is_sd = int(sys.argv[2]) != 0 if len(sys.argv) > 2 else False
            downloader = ModelDownloaderApi()
            info = downloader.get_info(repo_id, is_sd)
            print(dumps(info))
        except ValueError as e:
            logger.error(f"Invalid argument: {e}")
            exit(1)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            exit(1)

if __name__ == "__main__":
    main()