"""
ComfyUI Downloader Module
------------------------
This module provides functionality for downloading and setting up ComfyUI and its components.

Features:
- Installing a portable Git for Windows
- Cloning and setting up ComfyUI repository
- Managing ComfyUI custom nodes installation
- Installing Python requirements and packages
- Patching specific custom nodes when needed

The module handles the entire setup process, including downloading dependencies,
checking out specific Git references, and installing required Python packages.
"""

import logging
import os
import sys

import requests
import service_config
from web_request_bodies import ComfyUICustomNodesGithubRepoId

import service.aipg_utils as aipg_utils

git_download_url = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip"
comfyui_git_repo_url = "https://github.com/comfyanonymous/ComfyUI.git"
comfyui_manager_git_repo_url = "https://github.com/ltdrdata/ComfyUI-Manager.git"


def is_comfyui_installed() -> bool:
    """
    Check if ComfyUI is already installed.

    Returns:
        bool: True if ComfyUI installation directory exists, False otherwise.
    """
    return os.path.exists(service_config.comfy_ui_root_path)


def is_git_installed() -> bool:
    """
    Check if Git is already installed.

    Returns:
        bool: True if Git installation directory exists, False otherwise.
    """
    return os.path.exists(service_config.git.get("rootDirPath"))


def _install_portable_git():
    """
    Install a portable version of Git for Windows.

    This function:
    1. Downloads the portable Git zip file
    2. Extracts it to the configured directory
    3. Verifies the installation

    Raises:
        AssertionError: If Git installation fails
        Exception: If download or extraction fails
    """
    if is_git_installed():
        logging.info("Omitting installation of git, as already present")
        return
    zipped_portable_git_target = f"{service_config.git.get('rootDirPath')}.zip"
    git_target_dir = service_config.git.get("rootDirPath")

    try:
        aipg_utils.remove_existing_filesystem_resource(zipped_portable_git_target)
        aipg_utils.remove_existing_filesystem_resource(git_target_dir)

        _fetch_portable_git(zipped_portable_git_target)
        _unzip_portable_git(zipped_portable_git_target, git_target_dir)
        if not is_git_installed():
            raise AssertionError("Failed to install git at expected location")
        logging.info(f"successfully extracted git into {os.path.abspath(git_target_dir)}")
    except Exception as e:
        logging.error(f"failed to install git due to {e}. Cleaning up intermediate resources")
        aipg_utils.remove_existing_filesystem_resource(zipped_portable_git_target)
        aipg_utils.remove_existing_filesystem_resource(git_target_dir)
        raise e


def _fetch_portable_git(seven_zipped_portable_git_target):
    """
    Download the portable Git for Windows zip file.

    Args:
        seven_zipped_portable_git_target: Path where the downloaded zip will be saved

    Raises:
        Exception: If the download fails or returns a non-success status code
    """
    try:
        response = requests.get(git_download_url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(seven_zipped_portable_git_target, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        else:
            logging.error(f"Failed fetching resources from {git_download_url}")
            raise Exception(f"fetching {git_download_url} failed with response: {response}")
    except Exception as e:
        logging.error(f"Failed to fetch portable git from {git_download_url} with error {e}")
        aipg_utils.remove_existing_filesystem_resource(seven_zipped_portable_git_target)
        raise e


def _unzip_portable_git(zipped_git_path, target_dir):
    """
    Extract the downloaded Git zip file to the target directory.

    This function tries to use system tar if available, otherwise falls back to PowerShell.

    Args:
        zipped_git_path: Path to the downloaded Git zip file
        target_dir: Directory where Git should be extracted

    Raises:
        Exception: If extraction fails
    """

    def get_unzipping_command():
        try:
            aipg_utils.call_subprocess("tar --version")
            logging.debug("using system tar to unzip.")
            return f"tar -C {target_dir} -xf {zipped_git_path}"
        except Exception:
            logging.warning("falling back to powershell command to extract zip, as tar not in PATH")
            return f"powershell -command 'Expand-Archive' -Force {zipped_git_path} {target_dir}"

    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        aipg_utils.call_subprocess(get_unzipping_command())
        logging.info("Unzipped git successfully")
    except Exception as e:
        aipg_utils.remove_existing_filesystem_resource(zipped_git_path)
        aipg_utils.remove_existing_filesystem_resource(target_dir)
        raise e


def _install_git_repo(git_repo_url: str, target_dir: str):
    """
    Clone a Git repository to the specified target directory.

    Args:
        git_repo_url: URL of the Git repository to clone
        target_dir: Directory where the repository should be cloned

    Raises:
        Exception: If the clone operation fails
    """
    try:
        aipg_utils.remove_existing_filesystem_resource(target_dir)
        aipg_utils.call_subprocess(f"{service_config.git.get('exePath')} clone {git_repo_url} '{target_dir}'")
        logging.info(f"Cloned {git_repo_url} into {target_dir}")
    except Exception as e:
        logging.warning(f"git cloned failed with exception {e}. Cleaning up failed resources.")
        aipg_utils.remove_existing_filesystem_resource(target_dir)
        raise e


def _checkout_git_ref(repo_dir: str, git_ref: str | None):
    """
    Checkout a specific Git reference (branch, tag, or commit) in a repository.

    Args:
        repo_dir: Path to the Git repository
        git_ref: The Git reference to checkout (branch, tag, or commit hash)
    """
    if git_ref is None or not git_ref.strip():
        logging.info(f"No valid git ref provided for {repo_dir}")
        logging.warning(f"Repo {repo_dir} remains in ref {get_git_ref(repo_dir)}.")
        return
    try:
        aipg_utils.call_subprocess(f"{service_config.git.get('exePath')} checkout {git_ref}", cwd=repo_dir)
        logging.info(f"checked out {git_ref} in {repo_dir}")
    except Exception as e:
        logging.warning(f"git checkout of {git_ref} failed for rep {repo_dir} due to {e}.")
        logging.warning(f"Repo {repo_dir} remains in ref {get_git_ref(repo_dir)}.")


def get_git_ref(repo_dir: str) -> str | None:
    """
    Get the current Git reference (commit hash) of a repository.

    Args:
        repo_dir: Path to the Git repository

    Returns:
        str: The current commit hash, or None if it could not be determined
    """
    try:
        git_ref = aipg_utils.call_subprocess(f"{service_config.git.get('exePath')} rev-parse HEAD", cwd=repo_dir)
        return git_ref
    except Exception as e:
        logging.warning(f"Resolving git ref in {repo_dir} failed due to {e}")
        return


def _install_pip_requirements(requirements_txt_path: str):
    """
    Install Python packages from a requirements.txt file.

    Args:
        requirements_txt_path: Path to the requirements.txt file
    """
    logging.info(f"installing python requirements from {requirements_txt_path} using {sys.executable}")
    if os.path.exists(requirements_txt_path):
        python_exe_callable_path = (
            "'" + os.path.abspath(service_config.comfyui_python_exe) + "'"
        )  # this returns the abs path and may contain spaces. Escape the spaces with "ticks"
        aipg_utils.call_subprocess(f"{python_exe_callable_path} -m pip install -r '{requirements_txt_path}'")
        logging.info("python requirements installation completed.")
    else:
        logging.warning(f"specified {requirements_txt_path} does not exist.")


def install_pypi_package(packageSpecifier: str):
    """
    Install a Python package from PyPI or from a wheel file URL.

    If the package is already installed, the installation is skipped.
    If the packageSpecifier is a URL to a .whl file, it is first downloaded.

    Args:
        packageSpecifier: PyPI package name (with optional version) or URL to a wheel file
    """
    if is_package_installed(packageSpecifier):
        logging.info(f"package {packageSpecifier} already installed. Omitting installation")
        return
    if packageSpecifier.endswith(".whl"):
        pip_specifier = os.path.abspath(
            os.path.join(service_config.comfyui_python_env, packageSpecifier.split("/")[-1])
        )
        try:
            response = requests.get(packageSpecifier, stream=True, timeout=30)
            if response.status_code == 200:
                with open(pip_specifier, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
            else:
                logging.error(f"Failed fetching resources from {packageSpecifier}")
                raise Exception(f"fetching {packageSpecifier} failed with response: {response}")
        except Exception as e:
            logging.error(f"Failed to fetch dependency from {packageSpecifier} with error {e}")
            raise e
    else:
        pip_specifier = packageSpecifier

    logging.info(f"installing python package {packageSpecifier} using {sys.executable}")
    python_exe_callable_path = (
        "'" + os.path.abspath(service_config.comfyui_python_exe) + "'"
    )  # this returns the abs path and may contain spaces. Escape the spaces with "ticks"
    aipg_utils.call_subprocess(f"{python_exe_callable_path} -m pip install '{pip_specifier}'")
    aipg_utils.remove_existing_filesystem_resource("./dep.whl")
    logging.info("python package installation completed.")


def is_package_installed(packageSpecifier: str):
    """
    Check if a Python package is already installed.

    Args:
        packageSpecifier: PyPI package name (with optional version) or URL to a wheel file

    Returns:
        bool: True if the package is already installed, False otherwise
    """
    installed_packages = aipg_utils.call_subprocess(f"{service_config.comfyui_python_exe} -m pip list")
    if packageSpecifier.endswith(".whl"):
        package_name = packageSpecifier.split("/")[-1].split("-")[0]
    else:
        package_name = packageSpecifier.split("==")[0]
    if package_name in installed_packages:
        return True
    return False


def install_comfyui() -> bool:
    """
    Install ComfyUI from GitHub.

    This function:
    1. Installs Git if needed
    2. Clones the ComfyUI repository
    3. Installs Python requirements for ComfyUI

    Returns:
        bool: True if installation was successful, False otherwise

    Raises:
        Exception: If any installation step fails
    """
    if is_comfyui_installed():
        logging.info("comfyUI installation requested, while already installed")
        return True
    try:
        _install_portable_git()
        _install_git_repo(comfyui_git_repo_url, service_config.comfy_ui_root_path)
        _install_pip_requirements(os.path.join(service_config.comfy_ui_root_path, "requirements.txt"))
        return True
    except Exception as e:
        logging.error(f"comfyUI installation failed due to {e}")
        if os.path.exists(service_config.comfy_ui_root_path):
            aipg_utils.remove_existing_filesystem_resource(service_config.comfy_ui_root_path)
        raise e


def is_custom_node_installed_with_git_ref(node_repo_ref: ComfyUICustomNodesGithubRepoId) -> bool:
    """
    Check if a ComfyUI custom node is already installed.

    Args:
        node_repo_ref: Object containing repo username, name and git reference

    Returns:
        bool: True if the custom node is already installed, False otherwise
    """
    expected_custom_node_path = os.path.join(service_config.comfy_ui_root_path, "custom_nodes", node_repo_ref.repoName)
    custom_node_dir_exists = os.path.exists(expected_custom_node_path)

    return custom_node_dir_exists


def download_custom_node(node_repo_data: ComfyUICustomNodesGithubRepoId) -> bool:
    """
    Download and install a ComfyUI custom node from GitHub.

    This function:
    1. Clones the custom node repository
    2. Checks out the specified Git reference
    3. Applies patches if needed
    4. Installs Python requirements for the custom node

    Args:
        node_repo_data: Object containing repo username, name and git reference

    Returns:
        bool: True if installation was successful, False otherwise
    """
    if is_custom_node_installed_with_git_ref(node_repo_data):
        logging.info(f"node repo {node_repo_data} already exists. Omitting")
        return True
    else:
        try:
            expected_git_url = f"https://github.com/{node_repo_data.username}/{node_repo_data.repoName}"
            expected_custom_node_path = os.path.join(
                service_config.comfy_ui_root_path, "custom_nodes", node_repo_data.repoName
            )
            potential_node_requirements = os.path.join(expected_custom_node_path, "requirements.txt")

            aipg_utils.remove_existing_filesystem_resource(expected_custom_node_path)
            _install_git_repo(expected_git_url, expected_custom_node_path)
            _checkout_git_ref(expected_custom_node_path, node_repo_data.gitRef)
            _patch_custom_node_if_required(expected_custom_node_path, node_repo_data)
            _install_pip_requirements(potential_node_requirements)
            return True
        except Exception as e:
            logging.error(
                f"Failed to install custom comfy node {node_repo_data.username}/{node_repo_data.repoName} due to {e}"
            )
            return False


# Gourieff/ComfyUI-ReActor/scripts/reactor_sfw.py
REACTOR_SFW_PATCH = """from transformers import pipeline
from PIL import Image
import logging

SCORE = 0.965 # 0.965 and less - is safety content

logging.getLogger('transformers').setLevel(logging.ERROR)
from scripts.reactor_logger import logger

def nsfw_image(img_path: str, model_path: str):
    with Image.open(img_path) as img:
        predict = pipeline("image-classification", model=model_path)
        result = predict(img)
        logger.status(result)
        # Find the element with 'nsfw' label
        for item in result:
            if item["label"] == "nsfw":
                # Return True if nsfw score is above threshold (indicating NSFW content)
                # Return False if nsfw score is below threshold (indicating safe content)
                return True if item["score"] > SCORE else False
        # If no 'nsfw' label found, consider it safe
        return False
"""


def _patch_custom_node_if_required(custom_node_path: str, node_repo_data: ComfyUICustomNodesGithubRepoId):
    """
    Apply specific patches to custom nodes that require them.

    Currently handles a specific patch for the ComfyUI-ReActor plugin
    to modify its NSFW detection behavior.

    Args:
        custom_node_path: Path to the custom node installation
        node_repo_data: Object containing repo username, name and git reference
    """
    if (
        f"{node_repo_data.username}/{node_repo_data.repoName}@{node_repo_data.gitRef}".lower()
        == "Gourieff/comfyui-reactor@d2318ad140582c6d0b68c51df342319b502006ed".lower()
    ):
        reactor_sfw_path = os.path.join(custom_node_path, "scripts", "reactor_sfw.py")
        with open(reactor_sfw_path, "w") as file:
            file.write(REACTOR_SFW_PATCH)
        logging.info(f"patched {reactor_sfw_path} with custom logic")
