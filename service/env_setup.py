# Standard library imports
import subprocess  # Allows for executing shell commands from within Python.
import sys         # Provides access to system-specific parameters and functions.
import argparse    # Handles command-line argument parsing.
import logging     # Provides a flexible framework for emitting log messages from Python programs.
from typing import List  # Allows for type hinting, ensuring better code readability and error checking.

# External library imports
from rich.progress import Progress  # Used to create and manage progress bars in the console.
from rich.console import Console    # Provides advanced console output capabilities, including styled text.
from rich.traceback import install  # Enhances Python's error tracebacks for better readability.

# Install enhanced traceback handling using `rich`
install()

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level (int): The logging level. Default is `logging.INFO`.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(level=level)
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def install_package(package_name: str) -> bool:
    """Install a package using pip.

    Args:
        package_name (str): The name of the package to install.

    Returns:
        bool: True if installation succeeds, False otherwise.
    """
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, "--no-cache-dir", "--no-warn-script-location"])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def install_packages_from_file(requirements_file: str) -> None:
    """Install packages listed in a requirements file.

    Args:
        requirements_file (str): The path to the requirements file.
    """
    try:
        with open(requirements_file, 'r') as f:
            packages: List[str] = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Requirements file {requirements_file} not found.")
        return

    with Console() as console:
        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]Installing packages...", total=len(packages))

            for package in packages:
                result: bool = install_package(package)
                if result:
                    progress.update(task, advance=1, description=f"[green]Installed {package}")
                else:
                    progress.update(task, advance=1, description=f"[red]Failed to install {package}")

                # Optional: Print update to console
                console.print(f"[cyan]Installing {package}... [green]Done" if result else f"[cyan]Installing {package}... [red]Failed")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Setup script for environment.')
    parser.add_argument('-f', '--file', required=True, help='Requirement file location')
    return parser.parse_args()

def setup_env() -> None:
    """Set up the environment by installing packages from a requirements file."""
    args = parse_arguments()
    install_packages_from_file(args.file)

if __name__ == "__main__":
    setup_env()
