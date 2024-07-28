import pytest
from unittest.mock import patch, MagicMock
import json
from service.downloader import ModelDownloaderApi, hf_hub_url

@pytest.fixture
def mock_hf_filesystem():
    with patch('service.downloader.HfFileSystem') as mock_fs_class:
        mock_fs_instance = mock_fs_class.return_value
        mock_fs_instance.ls = MagicMock()
        yield mock_fs_instance

@pytest.fixture
def mock_hf_hub_url():
    with patch('service.downloader.hf_hub_url') as mock_url:
        yield mock_url

@pytest.fixture
def downloader_api(mock_hf_filesystem):
    return ModelDownloaderApi()

# --- Initialization ---
def test_init(downloader_api):
    """
    Verifies the correct initialization of ModelDownloaderApi.
    Ensures that the fs attribute is a MagicMock instance,
    and the file_queue and total_size are correctly initialized.
    """
    # Check that the fs attribute is an instance of MagicMock
    assert isinstance(downloader_api.fs, MagicMock), "The fs attribute should be a MagicMock instance."

    # Check that the file_queue is initialized as an empty list
    assert downloader_api.file_queue == [], "The file_queue should be initialized as an empty list."

    # Check that the total_size is initialized to 0
    assert downloader_api.total_size == 0, "The total_size should be initialized to 0."

# --- get_info Method Tests ---
def test_get_info_empty_repo(downloader_api, mock_hf_filesystem):
    """
    Tests get_info with an empty repository.
    Ensures that the total_size is 0 and the file_queue is empty.
    """
    # Mock the ls method to return an empty list
    mock_hf_filesystem.ls.return_value = []

    # Call the method under test
    downloader_api.get_info("empty-repo")

    # Assert the expected results
    assert downloader_api.total_size == 0, "Expected total_size to be 0 for an empty repository"
    assert downloader_api.file_queue == [], "Expected file_queue to be empty for an empty repository"

def test_get_info_single_file(downloader_api, mock_hf_filesystem, mock_hf_hub_url):
    """
    Tests get_info with a single file repository.
    Ensures that the total_size and file_queue reflect the single file.
    """
    # Mock the ls method to return a single file
    mock_hf_filesystem.ls.return_value = [
        {"name": "single-file-repo/myfile.txt", "size": 42, "type": "file"}
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Call the method under test
    downloader_api.get_info("single-file-repo")

    # Assert the expected results
    assert downloader_api.total_size == 42, "Expected total_size to be the size of the single file"
    assert downloader_api.file_queue == [
        {"name": "single-file-repo/myfile.txt", "size": 42, "url": "mock_url"}
    ], "Expected file_queue to contain the single file"

def test_get_info_with_directory(downloader_api, mock_hf_filesystem, mock_hf_hub_url):
    """
    Tests get_info with nested directories and files.
    Ensures that the total_size and file_queue reflect the nested structure.
    """
    # Mock the ls method to return nested directories and files
    mock_hf_filesystem.ls.side_effect = [
        [
            {"name": "repo-with-dir/subdir", "size": 0, "type": "directory"},
            {"name": "repo-with-dir/file1.txt", "size": 100, "type": "file"},
        ],
        [{"name": "repo-with-dir/subdir/file2.txt", "size": 250, "type": "file"}]
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Call the method under test
    downloader_api.get_info("repo-with-dir")

    # Assert the expected results
    assert downloader_api.total_size == 350
    assert set(tuple(item.items()) for item in downloader_api.file_queue) == set(tuple(item.items()) for item in [
        {"name": "repo-with-dir/file1.txt", "size": 100, "url": "mock_url"},
        {"name": "repo-with-dir/subdir/file2.txt", "size": 250, "url": "mock_url"}
    ])

    # Print the actual file_queue for debugging
    print("Actual file_queue:", downloader_api.file_queue)

def test_get_info_with_invalid_repo_id(downloader_api, mock_hf_filesystem):
    """
    Tests get_info with an invalid repository ID.
    Ensures that an exception is raised for invalid repository IDs.
    """
    # Mock the ls method to raise an Exception
    mock_hf_filesystem.ls.side_effect = Exception("Invalid repository ID")

    # Assert that the method raises an Exception
    with pytest.raises(Exception, match="Invalid repository ID"):
        downloader_api.get_info("invalid_repo_id")

# --- enum_file_list Method Tests ---
@pytest.mark.parametrize("is_sd, expected_total_size, expected_file_list", [
    (False, 300, [
        {"name": "repo/file1.txt", "size": 100, "url": "mock_url"},
        {"name": "repo/file2.safetensors", "size": 200, "url": "mock_url"},
    ]),
    (True, 100, [
        {"name": "repo/file1.txt", "size": 100, "url": "mock_url"}
    ])
])
def test_enum_file_list(downloader_api, mock_hf_filesystem, mock_hf_hub_url, is_sd, expected_total_size, expected_file_list):
    """
    Tests enum_file_list for standard and Stable Diffusion models.
    Ensures that files are correctly enumerated and their sizes and URLs are accurate.
    """
    # Mock the ls method to return files
    mock_hf_filesystem.ls.return_value = [
        {"name": "repo/file1.txt", "size": 100, "type": "file"},
        {"name": "repo/file2.safetensors", "size": 200, "type": "file"},
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Set repo_id and repo_folder
    downloader_api.repo_id = "repo"
    downloader_api.repo_folder = "repo"

    # Call the method under test
    downloader_api.enum_file_list("repo", is_sd)

    # Assert the expected results
    assert downloader_api.total_size == expected_total_size, f"Expected total_size to be {expected_total_size}"
    assert downloader_api.file_queue == expected_file_list, "Expected file_queue to match the expected file list"

def test_enum_file_list_nested(downloader_api, mock_hf_filesystem, mock_hf_hub_url):
    """
    Tests enum_file_list with nested directories.
    Ensures that nested directories and their files are correctly enumerated.
    """
    # Mock the ls method to return nested directories and files
    mock_hf_filesystem.ls.side_effect = [
        [
            {"name": "nested-repo/file1.txt", "size": 100, "type": "file"},
            {"name": "nested-repo/subdir", "type": "directory"}
        ],
        [
            {"name": "nested-repo/subdir/file2.txt", "size": 200, "type": "file"}
        ]
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Set repo_id and repo_folder
    downloader_api.repo_id = "nested-repo"
    downloader_api.repo_folder = "nested-repo"

    # Call the method under test
    downloader_api.enum_file_list("nested-repo")

    # Assert the expected results
    assert downloader_api.total_size == 300
    assert downloader_api.file_queue == [
        {"name": "nested-repo/file1.txt", "size": 100, "url": "mock_url"},
        {"name": "nested-repo/subdir/file2.txt", "size": 200, "url": "mock_url"}
    ]

@pytest.mark.parametrize("test_filename, should_ignore", [
    ("file.png", True),
    ("file.md", True),
    ("file.txt", False)
])
def test_enum_file_list_ignored_files(downloader_api, mock_hf_filesystem, mock_hf_hub_url, test_filename, should_ignore):
    """
    Tests ignoring specific file types in enum_file_list.
    Ensures that certain file types are ignored during enumeration.
    """
    # Mock the ls method to return different types of files
    mock_hf_filesystem.ls.return_value = [
        {"name": f"repo/{test_filename}", "size": 100, "type": "file"}
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Set repo_id and repo_folder
    downloader_api.repo_id = "repo"
    downloader_api.repo_folder = "repo"

    # Call the method under test
    downloader_api.enum_file_list("repo")

    # Assert the expected results
    if should_ignore:
        assert downloader_api.total_size == 0, f"Expected total_size to be 0 for ignored file type {test_filename}"
        assert downloader_api.file_queue == [], f"Expected file_queue to be empty for ignored file type {test_filename}"
    else:
        assert downloader_api.total_size == 100, f"Expected total_size to be 100 for non-ignored file type {test_filename}"
        assert downloader_api.file_queue == [
            {"name": f"repo/{test_filename}", "size": 100, "url": "mock_url"}
        ], f"Expected file_queue to contain the file {test_filename}"

# --- Stable Diffusion Model Handling ---
def test_enum_file_list_sd_model_root(downloader_api, mock_hf_filesystem, mock_hf_hub_url):
    """
    Tests ignoring specific files at the root for Stable Diffusion models.
    Ensures that files like .safetensors at the root level are ignored.
    """
    # Mock the ls method to return root files for SD model
    mock_hf_filesystem.ls.return_value = [
        {"name": "sd-model/model.safetensors", "size": 5000, "type": "file"},
        {"name": "sd-model/config.json", "size": 100, "type": "file"}
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Set repo_id and repo_folder
    downloader_api.repo_id = "sd-model"
    downloader_api.repo_folder = "sd-model"

    # Call the method under test with is_sd=True
    downloader_api.enum_file_list("sd-model", is_sd=True)

    # Assert the expected results
    assert downloader_api.total_size == 100, "Expected total_size to be 100, ignoring root .safetensors files"
    assert downloader_api.file_queue == [
        {"name": "sd-model/config.json", "size": 100, "url": "mock_url"}
    ], "Expected file_queue to contain only non-ignored files at the root"

def test_enum_file_list_sd_model_subdir(downloader_api, mock_hf_filesystem, mock_hf_hub_url):
    """
    Tests including specific files in subdirectories for Stable Diffusion models.
    Ensures that files like .safetensors in subdirectories are included.
    """
    # Mock the ls method to return subdirectory and files for SD model
    mock_hf_filesystem.ls.side_effect = [
        [{"name": "sd-model/subdir", "size": 0, "type": "directory"}],
        [{"name": "sd-model/subdir/model.safetensors", "size": 5000, "type": "file"}]
    ]
    # Mock the hf_hub_url function to return a fixed URL
    mock_hf_hub_url.return_value = "mock_url"

    # Set repo_id and repo_folder
    downloader_api.repo_id = "sd-model"
    downloader_api.repo_folder = "sd-model"

    # Call the method under test with is_sd=True
    downloader_api.enum_file_list("sd-model", is_sd=True)

    assert downloader_api.total_size == 5000
    assert downloader_api.file_queue == [
        {"name": "sd-model/subdir/model.safetensors", "size": 5000, "url": "mock_url"}
    ]


# --- URL Construction ---
@pytest.mark.parametrize("url_test_filename, url_test_subfolder, expected_url", [
    ("model.bin", "weights", "https://huggingface.co/test-repo/resolve/main/weights/model.bin"),
    ("config.json", "", "https://huggingface.co/test-repo/resolve/main/config.json"),
])
def test_url_construction(downloader_api, mock_hf_filesystem, mock_hf_hub_url, url_test_filename, url_test_subfolder, expected_url):
    """
    Verifies correct URL construction in enum_file_list.
    Ensures that the constructed URL matches the expected URL for given filename and subfolder.
    """
    # Mock the ls method to return a single file
    mock_hf_filesystem.ls.return_value = [
        {"name": f"test-repo/{url_test_subfolder if url_test_subfolder else ''}/{url_test_filename}", "size": 100, "type": "file"}
    ]
    # Mock the hf_hub_url function to return the expected URL
    mock_hf_hub_url.return_value = expected_url

    # Set repo_id and repo_folder
    downloader_api.repo_id = "test-repo"
    downloader_api.repo_folder = "test-repo"

    # Call the method under test
    downloader_api.enum_file_list("test-repo")

    # Assert the expected results
    assert downloader_api.file_queue[0]["url"] == expected_url, f"Expected URL to be {expected_url}"

# --- Error Handling ---
def test_enum_file_list_error_handling(downloader_api, mock_hf_filesystem):
    """
    Tests general error handling in enum_file_list.
    Ensures that an exception is raised when there is a general error listing files.
    """
    # Mock the ls method to raise a generic Exception
    mock_hf_filesystem.ls.side_effect = Exception("Error listing files")

    # Assert that the method raises an Exception with the expected message
    with pytest.raises(Exception, match="Error listing files"):
        downloader_api.enum_file_list("error-repo")


def test_enum_file_list_network_error(downloader_api, mock_hf_filesystem):
    """
    Tests network error handling in enum_file_list.
    Ensures that an OSError is raised when there is a network error.
    """
    # Mock the ls method to raise an OSError
    mock_hf_filesystem.ls.side_effect = OSError("Network error")

    # Assert that the method raises an OSError with the expected message
    with pytest.raises(OSError, match="Network error"):
        downloader_api.enum_file_list("error-repo")


def test_enum_file_list_permission_error(downloader_api, mock_hf_filesystem):
    """
    Tests permission error handling in enum_file_list.
    Ensures that a PermissionError is raised when there is a permission denied error.
    """
    # Mock the ls method to raise a PermissionError
    mock_hf_filesystem.ls.side_effect = PermissionError("Permission denied")

    # Assert that the method raises a PermissionError with the expected message
    with pytest.raises(PermissionError, match="Permission denied"):
        downloader_api.enum_file_list("restricted-repo")

# --- Main Function ---
def test_main_function():
    """
    Tests the main function's behavior.
    Ensures that the get_info method is called with the correct arguments when the script is executed.
    """
    with patch('service.downloader.ModelDownloaderApi') as MockModelDownloaderApi, \
         patch('sys.argv', ['downloader.py', 'test-repo', '1']):  
        
        # No need to import __main__
        from service.downloader import main 

        # Mock the return value of get_info to be JSON serializable
        mock_instance = MockModelDownloaderApi.return_value
        mock_instance.get_info.return_value = {"total_size": 100, "file_list": [{"name": "test-repo/file.txt", "size": 100, "url": "mock_url"}]}

        # Call the main function
        main() 

        mock_instance.get_info.assert_called_once_with("test-repo", True)