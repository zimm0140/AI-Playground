from json import dumps
import sys
from huggingface_hub import HfFileSystem, hf_hub_url
from pathlib import Path
from typing import List, Dict, Any

class ModelDownloaderApi:
    repo_id: str
    file_queue: List[Dict[str, Any]]
    total_size: int
    fs: HfFileSystem
    repo_folder: str

    def __init__(self):
        self.file_queue = []
        self.fs = HfFileSystem()
        self.total_size = 0

    def get_info(self, repo_id: str, is_sd: bool = False) -> Dict[str, Any]:
        self.repo_id = repo_id
        self.repo_folder = repo_id.replace('/', '---')
        self.file_queue.clear()
        self.total_size = 0
        self.enum_file_list(repo_id, is_sd, True)
        return {"total_size": self.total_size, "file_list": self.file_queue}

    def enum_file_list(self, enum_path: str, is_sd: bool = False, is_root: bool = True) -> None:
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

    def should_ignore_file(self, name: str, is_sd: bool, is_root: bool) -> bool:
        if is_sd and is_root and any(name.endswith(ext) for ext in [".safetensors", ".pt", ".ckpt"]):
            return True
        if any(name.endswith(ext) for ext in [".png", ".gitattributes", ".md", ".jpg"]):
            return True
        return False

    def construct_url(self, name: str) -> str:
        relative_path = Path(name).relative_to(self.repo_id).as_posix()
        subfolder = Path(relative_path).parent.as_posix()
        filename = Path(relative_path).name
        subfolder = '' if subfolder == '.' else subfolder
        return hf_hub_url(repo_id=self.repo_id, filename=filename, subfolder=subfolder)

    def normalize_path(self, name: str) -> str:
        return Path(name).as_posix()

# --- Main Function ---
def main() -> None:
    if len(sys.argv) == 1:
        exit(1)
    else:
        downloader = ModelDownloaderApi()
        info = downloader.get_info(
            sys.argv[1], int(sys.argv[2]) != 0 if len(sys.argv) > 2 else False
        )
        print(dumps(info))

if __name__ == "__main__":
    main()