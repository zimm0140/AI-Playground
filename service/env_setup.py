import subprocess
import sys
import argparse
import logging
from rich.progress import Progress
from rich.console import Console
from rich.traceback import install
from typing import List, Optional

install()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_package(package_name: str) -> bool:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, "--no-cache-dir", "--no-warn-script-location"])
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        return False

def install_packages_from_file(requirements_file: str) -> None:
    try:
        with open(requirements_file, 'r') as f:
            packages: List[str] = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logger.error(f"Requirements file {requirements_file} not found.")
        return

    with Progress(transient=True) as progress:
        task = progress.add_task("[cyan]Installing packages...", total=len(packages))
        console = Console()

        for package in packages:
            result: bool = install_package(package)
            if result:
                progress.update(task, advance=1, description=f"[green]Installed {package}")
            else:
                progress.update(task, advance=1, description=f"[red]Failed to install {package}")

            # Optional: Print update to console
            console.print(f"[cyan]Installing {package}... [green]Done" if result else f"[cyan]Installing {package}... [red]Failed")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Setup script for environment.')
    parser.add_argument('-f', '--file', required=True, help='requirement file location')
    return parser.parse_args()

def setup_env() -> None:
    args = parse_arguments()
    install_packages_from_file(args.file)

if __name__ == "__main__":
    setup_env()
