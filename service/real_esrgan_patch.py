"""
Real-ESRGAN Import Patcher
-------------------------
This module provides a utility for patching import statements in Real-ESRGAN code.

It specifically addresses an import compatibility issue with torchvision by replacing
references to the deprecated 'functional_tensor' module with the standard 'functional' module.
This patch enables Real-ESRGAN to work with newer versions of torchvision.
"""

import sys


def patch_import(file_path):
    """
    Patch the torchvision import statements in the specified file.

    This function replaces imports from 'torchvision.transforms.functional_tensor'
    with equivalent imports from 'torchvision.transforms.functional' to maintain
    compatibility with newer versions of torchvision.

    Args:
        file_path: Path to the Python file that needs to be patched
    """
    # Read the file
    with open(file_path) as file:
        lines = file.readlines()

    # Patch the import line
    with open(file_path, "w") as file:
        for line in lines:
            # Replace the old import line with the new one
            if "from torchvision.transforms.functional_tensor import rgb_to_grayscale" in line:
                line = line.replace(
                    "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                    "from torchvision.transforms.functional import rgb_to_grayscale",
                )
            file.write(line)

    print(f"Patched {file_path} successfully.")


# Main execution block for command-line usage
if __name__ == "__main__":
    # Check for correct command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python patch_import.py <path_to_file>")
        sys.exit(1)

    # Get file path from command-line argument
    file_path = sys.argv[1]
    # Execute the patching operation
    patch_import(file_path)
