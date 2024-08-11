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

    def __init__(self, fs: HfFileSystem = None):
        """
        Initializes the ModelDownloaderApi instance.

        Args:
            fs (HfFileSystem, optional): A Hugging Face file system object used to interact with the repository.
                If not provided, a new instance of `HfFileSystem` is created.
        """
        self.repo_id: str = ""  # Repository ID on Hugging Face Hub
        self.file_queue: List[Dict[str, Any]] = []  # Queue to store file metadata
        self.total_size: int = 0  # Total size of all files in the repository
        self.fs: HfFileSystem = fs if fs else HfFileSystem()  # File system interface for interacting with the Hugging Face Hub
        self.repo_folder: str = ""  # Local folder name corresponding to the repository

    def get_info(self, repo_id: str, is_sd: bool = False) -> Dict[str, Any]:
        """
        Retrieves information about a model repository from the Hugging Face Hub.

        This method calculates the total size of the repository and generates a list of files with their
        corresponding metadata (name, size, and download URL). It also applies filtering to exclude specific files
        based on the `is_sd` flag.

        Args:
            repo_id (str): The ID of the repository on the Hugging Face Hub. It should be in the format "username/repo_name".
            is_sd (bool, optional): A flag indicating whether the repository contains a Stable Diffusion model.
                If True, specific files associated with Stable Diffusion models (e.g., `.safetensors`, `.pt`, `.ckpt`)
                are excluded. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'total_size' (int): The cumulative size of all files in the repository (in bytes).
                - 'file_list' (List[Dict[str, Any]]): A list of dictionaries, each representing a file
                  with its 'name', 'size', and 'url'.

        Raises:
            FileNotFoundError: If the specified repository is not found on the Hugging Face Hub.
            ValueError: If the provided repository ID is invalid (e.g., missing a '/' separator).
            Exception: If any other error occurs during the information retrieval process.
        """
        try:
            if not repo_id or '/' not in repo_id:
                raise ValueError("Invalid repository ID format. Expected format: 'username/repo_name'.")
            self.repo_id = repo_id
            self.repo_folder = repo_id.replace('/', '---')  # Replace '/' with '---' to create a valid folder name
            self.file_queue.clear()  # Clear any existing files in the queue
            self.total_size = 0  # Reset the total size
            self.process_repository(repo_id, is_sd)  # Process the repository to gather file information
            return {"total_size": self.total_size, "file_list": self.file_queue}
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Invalid repository '{repo_id}': {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while fetching info for repo '{repo_id}': {e}")
            raise

    def process_repository(self, repo_id: str, is_sd: bool) -> None:
        """
        Initiates the processing of the model repository by enumerating its files and directories.

        This method starts the recursive enumeration of files in the repository, applying filtering based on the `is_sd` flag.

        Args:
            repo_id (str): The ID of the model repository on the Hugging Face Hub.
            is_sd (bool): A flag indicating if the model is a Stable Diffusion model. If True, the method will exclude
                certain files specific to Stable Diffusion models.
        """
        self.enum_file_list(repo_id, is_sd, True)

    def enum_file_list(self, enum_path: str, is_sd: bool = False, is_root: bool = True, visited_paths: set = None) -> None:
        """
        Recursively enumerates files and directories within a given path.

        This method traverses the directory structure of the repository, processing files and recursively calling itself
        for subdirectories. It applies filtering to exclude unnecessary files.

        Args:
            enum_path (str): The current path being enumerated.
            is_sd (bool, optional): A flag indicating if the model is a Stable Diffusion model. Defaults to False.
            is_root (bool, optional): A flag indicating if the current path is the root of the repository. Defaults to True.
            visited_paths (set, optional): A set to keep track of visited paths to prevent infinite recursion
                in case of circular symbolic links.

        Raises:
            FileNotFoundError: If the provided path does not exist on the Hugging Face Hub.
            PermissionError: If there are permission issues accessing the path.
            Exception: If any other error occurs during file enumeration.
        """
        if visited_paths is None:
            visited_paths = set()

        if enum_path in visited_paths:
            logger.warning(f"Skipping circular reference at '{enum_path}'")
            return
        visited_paths.add(enum_path)

        try:
            items = self.fs.ls(enum_path, detail=True)  # List all files and directories in the current path
            for item in items:
                self.process_item(item, is_sd, is_root, visited_paths=visited_paths)
        except FileNotFoundError as e:
            logger.error(f"Path '{enum_path}' not found: {e}")
        except PermissionError as e:
            logger.error(f"Permission denied for '{enum_path}': {e}")
        except Exception as e:
            logger.error(f"An error occurred while enumerating files in '{enum_path}': {e}")
            raise

    def process_item(self, item: Dict[str, Any], is_sd: bool, is_root: bool, visited_paths: set) -> None:
        """
        Processes an individual item (file or directory) in the repository.

        For directories, it recursively calls `enum_file_list`. For files, it checks if the file should be ignored,
        and if not, adds its information to the download queue.

        Args:
            item (Dict[str, Any]): A dictionary containing the metadata of the file or directory.
            is_sd (bool): A flag indicating if special handling for Stable Diffusion models is needed.
            is_root (bool): A flag indicating if the current item is in the root directory of the repository.
            visited_paths (set): A set to keep track of visited paths to prevent infinite recursion
                in case of circular symbolic links.
        """
        name = self.normalize_path(item.get("name"))
        size = item.get("size")
        item_type = item.get("type")

        if item_type == "directory":
            self.enum_file_list(name, is_sd, False, visited_paths=visited_paths)
        else:
            if not self.should_ignore_file(name, is_sd, is_root):
                self.add_file_to_queue(name, size)

    def should_ignore_file(self, name: str, is_sd: bool, is_root: bool) -> bool:
        """
        Determines if a file should be ignored based on its name and the type of model.

        This method implements filtering rules to exclude specific files from the download queue.
        For Stable Diffusion models, it ignores model files (`.safetensors`, `.pt`, `.ckpt`) in the root directory.
        Common unnecessary files like images, git attributes, and documentation files are also ignored.

        Args:
            name (str): The name of the file.
            is_sd (bool): A flag indicating if the model is a Stable Diffusion model.
            is_root (bool): A flag indicating if the file is in the root directory of the repository.

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
        self.total_size += size  # Update the total size with the file's size
        url = self.construct_url(name)  # Construct the download URL for the file
        # Add the file's information to the queue, replacing the repository ID with the local folder name
        self.file_queue.append({"name": name.replace(self.repo_id, self.repo_folder), "size": size, "url": url})

    def construct_url(self, name: str) -> str:
        """
        Constructs the download URL for a file on the Hugging Face Hub.

        Args:
            name (str): The name of the file.

        Returns:
            str: The constructed download URL.

        Raises:
            ValueError: If the file path cannot be made relative to the repository ID.
            Exception: If any other error occurs during URL construction, it is logged.
        """
        try:
            relative_path = Path(name).relative_to(self.repo_id).as_posix()  # Calculate the relative path
            subfolder = Path(relative_path).parent.as_posix()  # Extract the subfolder from the relative path
            filename = Path(relative_path).name  # Get the filename
            subfolder = '' if subfolder == '.' else subfolder  # Handle files in the root directory
            return hf_hub_url(repo_id=self.repo_id, filename=filename, subfolder=subfolder)  # Construct the URL
        except ValueError as e:
            logger.error(f"Cannot make '{name}' relative to '{self.repo_id}': {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while constructing URL for '{name}': {e}")
            raise

    def normalize_path(self, name: str) -> str:
        """
        Normalizes a file path to POSIX format, which is platform-independent.

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
            return Path(name).as_posix()  # Convert the path to POSIX format
        except Exception as e:
            logger.error(f"An error occurred while normalizing path '{name}': {e}")
            raise

# --- Main Function ---
def main() -> None:
    """
    The main function to execute the script. It processes command-line arguments, 
    initializes the ModelDownloaderApi class, and retrieves repository information.

    If insufficient arguments are provided, it logs an error and exits.
    """
    if len(sys.argv) < 2:
        logger.error("Usage: downloader.py <repo_id> [is_sd]")  # Print usage if insufficient arguments are provided
        exit(1)  # Exit with error status
    elif len(sys.argv) == 2: # Handle case where is_sd is not provided
        repo_id = sys.argv[1]
        is_sd = False  # Default is_sd to False
    else:
        try:
            repo_id = sys.argv[1]  # Get the repository ID from the command line
            # Validate repo_id format
            if not repo_id or '/' not in repo_id:
                raise ValueError("Invalid repository ID format.")
            is_sd = int(sys.argv[2]) != 0 if len(sys.argv) > 2 else False  # Determine the is_sd flag from the command line
            downloader = ModelDownloaderApi()  # Initialize the ModelDownloaderApi class
            info = downloader.get_info(repo_id, is_sd)  # Retrieve repository information
            print(dumps(info))  # Print the repository information as a JSON string
        except ValueError as e:
            logger.error(f"Invalid argument: {e}")  # Log error if there's a problem with the arguments
            exit(1)  # Exit with error status
        except Exception as e:
            logger.error(f"An error occurred: {e}")  # Log any other errors
            exit(1)  # Exit with error status

if __name__ == "__main__":
    main()  # Run the main function if the script is executed directly