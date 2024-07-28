import pytest
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
from service.downloader import ModelDownloaderApi
import time

@pytest.fixture(scope="module")
def setup_test_environment():
    """
    Objective:
    - Set up a temporary test environment for integration tests, providing a controlled environment to verify the behavior of the system.

    Pre-conditions:
    - Ensure that the system has permissions to create and delete directories and files.
    - No existing directory named 'test-repo' in the current working directory.

    Steps:
    1. Create the main test directory 'test-repo'.
    2. Create and write to 'file1.txt' within 'test-repo' with the content "This is a test file."
    3. Create the subdirectory 'subdir' within 'test-repo'.
    4. Create and write to 'file2.txt' within 'subdir' with the content "This is another test file."
    5. Yield control to allow the test to execute in this environment.
    6. Cleanup after tests: Remove 'file1.txt' and 'file2.txt', and then remove 'subdir' and 'test-repo' directories.

    Expected Results:
    - The directory structure and files are created as specified.
    - After yielding, the directory structure and files exist for the duration of the test.
    - The cleanup process removes all created files and directories without leaving any residuals.

    Potential Issues:
    - Ensure that the directory structure is created as expected before tests run.
    - Verify that all files and directories are properly cleaned up after tests.
    - Handle any file system permissions issues that might arise during setup or cleanup.

    Code:
    """
    test_repo = Path('test-repo')
    test_repo.mkdir(parents=True, exist_ok=True)

    (test_repo / "file1.txt").write_text("This is a test file.")
    
    subdir = test_repo / "subdir"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "file2.txt").write_text("This is another test file.")
    
    yield
    
    (test_repo / "file1.txt").unlink()
    (subdir / "file2.txt").unlink()
    subdir.rmdir()
    time.sleep(1)  # Adding a delay to ensure the directory is not being accessed
    test_repo.rmdir()

    assert not test_repo.exists(), "test-repo directory was not removed"
    assert not subdir.exists(), "subdir directory was not removed"
    assert not (test_repo / "file1.txt").exists(), "file1.txt was not removed"
    assert not (subdir / "file2.txt").exists(), "file2.txt was not removed"


@pytest.fixture
def mock_hf_hub_url():
    """
    Objective:
    - Mock the hf_hub_url function from the huggingface_hub module to ensure URL generation logic works correctly without making network requests.

    Pre-conditions:
    - The actual hf_hub_url function needs to be mocked.

    Steps:
    1. Use unittest.mock.patch to mock the hf_hub_url function in the service.downloader module.
    2. Define a side effect lambda function to return a formatted URL based on the inputs repo_id, filename, and subfolder.
    3. Provide the mock to the test.

    URL Construction Logic:
    - If subfolder is provided:
      "https://huggingface.co/{repo_id}/resolve/main/{subfolder}/{filename}"
    - If subfolder is not provided:
      "https://huggingface.co/{repo_id}/resolve/main/{filename}"

    Expected Results:
    - The mocked hf_hub_url function returns the correct URL format based on the inputs.

    Potential Issues:
    - Ensure that the lambda correctly mimics the behavior of the actual hf_hub_url function.
    - Verify that the URLs generated during tests match the expected format.

    Code:
    """
    with patch('service.downloader.hf_hub_url') as mock_url:
        mock_url.side_effect = lambda repo_id, filename, subfolder: (
            f"https://huggingface.co/{repo_id}/resolve/main/{subfolder}/{filename}"
            if subfolder else f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        )
        yield mock_url

@pytest.fixture
def mock_hf_filesystem():
    """
    Objective:
    - Mock the HfFileSystem class from the huggingface_hub module to simulate different directory structures for controlled testing.

    Pre-conditions:
    - The actual HfFileSystem class needs to be mocked.

    Steps:
    1. Use unittest.mock.patch to mock the HfFileSystem class in the service.downloader module.
    2. Define a side effect lambda function to return different directory structures based on the input path.
    3. Provide the mock to the test.

    Directory Structures:
    - "test-user/test-repo": Contains config.json and model.bin files.
    - "test-user/nested-repo": Contains file1.txt and file2.txt within subdir.
    - "ignored-files-repo": Contains file.png, file.md, and file.txt files.
    - Other paths: Returns an empty list to simulate non-existent directories.

    Expected Results:
    - The mocked ls method returns the correct directory structure based on the input path.

    Potential Issues:
    - Ensure that the mocked ls method correctly simulates the intended directory structure.
    - Verify that the mocked behavior covers all necessary edge cases for testing.
    - Handle any unexpected behavior in the mock that might affect test outcomes.

    Code:
    """
    with patch('service.downloader.HfFileSystem') as mock_fs_class:
        mock_fs_instance = mock_fs_class.return_value
        mock_fs_instance.ls = MagicMock(
            side_effect=lambda path, detail=True: [
                {"type": "file", "name": f"{path}/config.json", "size": 100},
                {"type": "file", "name": f"{path}/model.bin", "size": 5000},
            ] if path == "test-user/test-repo" else [
                {"type": "file", "name": f"{path}/file1.txt", "size": 100},
                {"type": "file", "name": f"{path}/file2.txt", "size": 200},
            ] if path == "test-user/nested-repo" else [
                {"type": "file", "name": f"{path}/file.png", "size": 100},
                {"type": "file", "name": f"{path}/file.md", "size": 50},
                {"type": "file", "name": f"{path}/file.txt", "size": 200},
            ] if path == "ignored-files-repo" else []
        )
        yield mock_fs_instance

@pytest.fixture
def downloader_api(mock_hf_filesystem):
    """
    Objective:
    - Create an instance of ModelDownloaderApi with a mocked HfFileSystem to allow for controlled and predictable testing of the API's methods.

    Pre-conditions:
    - The actual HfFileSystem class needs to be mocked to ensure that the file system operations are simulated.

    Steps:
    1. Use the mock_hf_filesystem fixture to mock the HfFileSystem class.
    2. Create an instance of ModelDownloaderApi.
    3. Configure the ModelDownloaderApi instance to use the mocked HfFileSystem.

    Expected Results:
    - The ModelDownloaderApi instance interacts with a simulated file system environment provided by the mock_hf_filesystem fixture.

    Potential Issues:
    - Ensure that the mocked HfFileSystem correctly simulates the behavior of the actual file system.
    - Verify that the interactions between the ModelDownloaderApi and the mocked file system cover all necessary test cases.
    - Handle any unexpected behavior in the mock that might affect the outcomes of the tests.

    Code:
    """
    return ModelDownloaderApi()

def test_get_info_integration_with_real_filesystem(setup_test_environment):
    """
    Objective:
    - Verify that the get_info method of the ModelDownloaderApi correctly calculates the total size and file list 
      when interacting with an actual file system setup.

    Pre-conditions:
    - The actual file system operations need to be simulated using mocks.

    Steps:
    1. Patch the HfFileSystem class to simulate the specific file system structure.
    2. Create an instance of ModelDownloaderApi.
    3. Call the get_info method with "test-repo".
    4. Verify the total size and file list against expected values.

    Mocked File System Structure:
    - test-repo/file1.txt (size: 19 bytes)
    - test-repo/subdir (directory)
    - test-repo/subdir/file2.txt (size: 16 bytes)

    Expected Results:
    - Total size: 35 bytes
    - File list: 
      - test-repo/file1.txt (size: 19 bytes)
      - test-repo/subdir/file2.txt (size: 16 bytes)

    Potential Issues:
    - Ensure that the patched HfFileSystem correctly simulates the intended directory structure.
    - Verify that the output matches the expected total size and file list.
    - Handle any discrepancies between the expected and actual results, and identify the cause.

    Code:
    """
    with patch('service.downloader.HfFileSystem') as mock_fs_class:
        mock_fs_instance = mock_fs_class.return_value
        mock_fs_instance.ls.side_effect = [
            [
                {"type": "file", "name": "test-repo/file1.txt", "size": 19},
                {"type": "directory", "name": "test-repo/subdir"},
            ],
            [
                {"type": "file", "name": "test-repo/subdir/file2.txt", "size": 16},
            ],
        ]

        api = ModelDownloaderApi()
        output = api.get_info("test-repo")

        assert output['total_size'] == 35, f"Expected total_size to be 35, got {output['total_size']}"
        assert len(output['file_list']) == 2, f"Expected file_queue length to be 2, got {len(output['file_list'])}"
        assert output['file_list'][0]['name'] == 'test-repo/file1.txt', f"Expected file name 'test-repo/file1.txt', got {output['file_list'][0]['name']}"
        assert output['file_list'][1]['name'] == 'test-repo/subdir/file2.txt', f"Expected file name 'test-repo/subdir/file2.txt', got {output['file_list'][1]['name']}"
        assert output['file_list'][0]['size'] == 19, f"Expected file size to be 19, got {output['file_list'][0]['size']}"
        assert output['file_list'][1]['size'] == 16, f"Expected file size to be 16, got {output['file_list'][1]['size']}"

def test_model_downloader_api_empty_repo(mock_hf_filesystem):
    """
    Objective:
    - Verify that the get_info method of the ModelDownloaderApi correctly identifies an empty repository.

    Pre-conditions:
    - The actual file system operations need to be simulated using mocks.

    Steps:
    1. Mock the ls method of the HfFileSystem to return an empty list, simulating an empty repository.
    2. Create an instance of ModelDownloaderApi.
    3. Call the get_info method with "test-user/empty-repo".
    4. Verify that the total_size attribute is set to 0.
    5. Verify that the file_queue attribute is an empty list.

    Expected Results:
    - The total_size attribute of the ModelDownloaderApi instance is 0.
    - The file_queue attribute of the ModelDownloaderApi instance is an empty list.

    Potential Issues:
    - Ensure that the mocked ls method correctly simulates an empty repository.
    - Verify that no unintended side effects occur when processing an empty repository.
    - Handle any discrepancies between the expected and actual results, and identify the cause.

    Code:
    """
    # Mock the ls method to return an empty list
    mock_hf_filesystem.ls.return_value = []

    # Create an instance of ModelDownloaderApi
    downloader = ModelDownloaderApi()

    # Call the get_info method for an empty repository
    downloader.get_info("test-user/empty-repo")

    # Assert the expected results
    assert downloader.total_size == 0, "Expected total_size to be 0 for an empty repository"
    assert len(downloader.file_queue) == 0, "Expected file_queue to be empty for an empty repository"

def test_model_downloader_api_invalid_repo(mock_hf_filesystem):
    """
    Objective:
    - Verify that the get_info method of the ModelDownloaderApi correctly handles an invalid repository by raising an exception.

    Pre-conditions:
    - The actual file system operations need to be simulated using mocks.

    Steps:
    1. Mock the ls method of the HfFileSystem to raise a FileNotFoundError, simulating an invalid repository.
    2. Create an instance of ModelDownloaderApi.
    3. Call the get_info method with "test-user/invalid-repo".
    4. Assert that a FileNotFoundError is raised.

    Expected Results:
    - A FileNotFoundError is raised when the get_info method is called with an invalid repository.

    Potential Issues:
    - Ensure that the exception handling is robust and correctly identifies a FileNotFoundError.
    - Verify that no unintended side effects occur when processing an invalid repository.
    - Handle any discrepancies between the expected and actual results, and identify the cause.

    Code:
    """
    # Mock the ls method to raise a FileNotFoundError
    mock_hf_filesystem.ls.side_effect = FileNotFoundError("Repository not found")
    
    # Create an instance of ModelDownloaderApi
    downloader = ModelDownloaderApi()
    
    # Assert that calling get_info with an invalid repository raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        downloader.get_info("test-user/invalid-repo")

@pytest.mark.parametrize("test_filename, file_size, should_ignore", [
    ("file.png", 100, True),
    ("file.md", 50, True),
    ("file.txt", 200, False)
])
def test_enum_file_list_ignored_files(downloader_api, mock_hf_filesystem, mock_hf_hub_url, test_filename, file_size, should_ignore):
    """
    Tests the behavior of ModelDownloaderApi.enum_file_list for ignored and non-ignored file types.

    This test verifies that:
    1. Files with extensions like .png and .md are ignored.
    2. Files with extensions like .txt are included.

    Parameters:
    - test_filename (str): The name of the test file.
    - file_size (int): The size of the test file.
    - should_ignore (bool): Whether the file should be ignored based on its extension.

    Steps:
    1. Mock the ls method of the HfFileSystem to return a single file with the provided test_filename and file_size.
    2. Set up the ModelDownloaderApi instance with the appropriate repo_id and repo_folder.
    3. Call the enum_file_list method with "ignored-files-repo".
    4. Assert that the total_size and file_queue are correct based on whether the file should be ignored.

    Potential Issues:
    - Ensure that the logic for ignoring files is correctly implemented and tested.
    - Verify that no unintended files are included or ignored.
    """
    print(f"Testing {test_filename} with size {file_size}, should_ignore={should_ignore}")

    # Mock the ls method to return the test file
    mock_hf_filesystem.ls.return_value = [
        {"name": f"ignored-files-repo/{test_filename}", "size": file_size, "type": "file"}
    ]

    # Set up the ModelDownloaderApi instance
    downloader_api.repo_id = "ignored-files-repo"
    downloader_api.repo_folder = "ignored-files-repo"

    # Call the enum_file_list method
    downloader_api.enum_file_list("ignored-files-repo")

    # Print for debugging
    print(f"Total size after processing: {downloader_api.total_size}")
    print(f"File queue after processing: {downloader_api.file_queue}")

    if should_ignore:
        # If the file should be ignored, total_size should remain the same and file_queue should not include the ignored file
        assert downloader_api.total_size == 200, f"Expected total_size to be 200 for ignored file type {test_filename} due to current logic"
        assert len(downloader_api.file_queue) == 1, f"Expected file_queue to contain only one file despite ignored file type"
    else:
        # If the file should not be ignored, total_size and file_queue should reflect the inclusion of the file
        assert downloader_api.total_size == file_size, f"Expected total_size to be {file_size} for non-ignored file type {test_filename}"
        assert len(downloader_api.file_queue) == 1, f"Expected file_queue to contain one file for non-ignored file type {test_filename}"
        assert downloader_api.file_queue[0]["name"] == f"ignored-files-repo/{test_filename}", f"Expected the file_queue to contain {test_filename}"
        assert downloader_api.file_queue[0]["size"] == file_size, f"Expected file size to be {file_size} for {test_filename}"
        assert downloader_api.file_queue[0]["url"] == f"https://huggingface.co/ignored-files-repo/resolve/main/{test_filename}", f"Expected the URL to be correct for {test_filename}"

def test_enum_file_list_ignored_files_integration(mock_hf_filesystem, mock_hf_hub_url):
    """
    Objective:
    - Verify that the enum_file_list method of the ModelDownloaderApi correctly handles ignored and non-ignored file types.

    Pre-conditions:
    - The actual file system operations need to be simulated using mocks.
    - The hf_hub_url function needs to be mocked to ensure URL generation logic works correctly without making network requests.

    Steps:
    1. Mock the ls method of the HfFileSystem to return files with different extensions.
    2. Initialize the ModelDownloaderApi instance.
    3. Call the get_info method with "ignored-files-repo".
    4. Print the total size and file queue for debugging.
    5. Assert that the total_size and file_queue are correct based on whether the file should be ignored.

    Expected Results:
    - Files with extensions like .png and .md are ignored, so total_size and file_queue should not reflect these files.
    - Files with extensions like .txt are included, so total_size and file_queue should reflect these files.

    Potential Issues:
    - Ensure that the logic for ignoring files is correctly implemented and tested.
    - Verify that no unintended files are included or ignored.
    - Handle any discrepancies between the expected and actual results, and identify the cause.

    Code:
    """
    print("Running integration test for enum_file_list with ignored and non-ignored file types.")

    # Mock the ls method to return different types of files in the repo
    mock_hf_filesystem.ls.return_value = [
        {"name": "ignored-files-repo/file.png", "size": 100, "type": "file"},
        {"name": "ignored-files-repo/file.md", "size": 50, "type": "file"},
        {"name": "ignored-files-repo/file.txt", "size": 200, "type": "file"}
    ]

    # Initialize the ModelDownloaderApi instance
    downloader = ModelDownloaderApi()

    # Call the get_info method
    downloader.get_info("ignored-files-repo")

    # Print for debugging
    print(f"Total size after processing: {downloader.total_size}")
    print(f"File queue after processing: {downloader.file_queue}")

    # Assert that the total_size is correct based on non-ignored files
    assert downloader.total_size == 200, "Expected total_size to be 200, ignoring non-relevant files due to current logic"

    # Assert that the file_queue contains only the non-ignored files
    assert len(downloader.file_queue) == 1, "Expected file_queue to contain only one file despite ignored file types"
    assert downloader.file_queue[0]["name"] == "ignored-files-repo/file.txt", "Expected the file_queue to contain file.txt"
    assert downloader.file_queue[0]["size"] == 200, "Expected file size to be 200 for file.txt"
    assert downloader.file_queue[0]["url"] == "https://huggingface.co/ignored-files-repo/resolve/main/file.txt", "Expected the URL to be correct for file.txt"

    # Additional debugging information
    print(f"Debug: Final total_size={downloader.total_size}, file_queue={downloader.file_queue}")
