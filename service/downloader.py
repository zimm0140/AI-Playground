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
    repo_id: str
    file_queue: List[Dict[str, Any]]
    total_size: int
    fs: HfFileSystem
    repo_folder: str

    def __init__(self, fs: HfFileSystem = None):
        self.file_queue = []
        self.fs = fs if fs else HfFileSystem()
        self.total_size = 0

    def get_info(self, repo_id: str, is_sd: bool = False) -> Dict[str, Any]:
        try:
            self.repo_id = repo_id
            self.repo_folder = repo_id.replace('/', '---')
            self.file_queue.clear()
            self.total_size = 0
            self.enum_file_list(repo_id, is_sd, True)
            return {"total_size": self.total_size, "file_list": self.file_queue}
        except Exception as e:
            logger.error(f"An error occurred while fetching info for repo '{repo_id}': {e}")
            raise

    def enum_file_list(self, enum_path: str, is_sd: bool = False, is_root: bool = True) -> None:
        try:
            items = self.fs.ls(enum_path, detail=True)
            for item in items:
                name = self.normalize_path(item.get("name"))
                size = item.get("size")
                item_type = item.get("type")
                if item_type == "directory":
                    self.enum_file_list(name, is_sd, False)
                else:
                    if self.should_ignore_file(name, is_sd, is_root):
                        continue
                    self.total_size += size
                    url = self.construct_url(name)
                    self.file_queue.append({"name": name.replace(self.repo_id, self.repo_folder), "size": size, "url": url})
        except Exception as e:
            logger.error(f"An error occurred while enumerating files in '{enum_path}': {e}")
            raise

    def should_ignore_file(self, name: str, is_sd: bool, is_root: bool) -> bool:
        sd_ignored_extensions = [".safetensors", ".pt", ".ckpt"]
        common_ignored_extensions = [".png", ".gitattributes", ".md", ".jpg"]

        if is_sd and is_root and any(name.endswith(ext) for ext in sd_ignored_extensions):
            return True
        if any(name.endswith(ext) for ext in common_ignored_extensions):
            return True
        return False

    def construct_url(self, name: str) -> str:
        try:
            relative_path = Path(name).relative_to(self.repo_id).as_posix()
            subfolder = Path(relative_path).parent.as_posix()
            filename = Path(relative_path).name
            subfolder = '' if subfolder == '.' else subfolder
            return hf_hub_url(repo_id=self.repo_id, filename=filename, subfolder=subfolder)
        except Exception as e:
            logger.error(f"An error occurred while constructing URL for '{name}': {e}")
            raise

    def normalize_path(self, name: str) -> str:
        try:
            return Path(name).as_posix()
        except Exception as e:
            logger.error(f"An error occurred while normalizing path '{name}': {e}")
            raise

# --- Main Function ---
def main() -> None:
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