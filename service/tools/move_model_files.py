"""
Model Files Migration Utility
----------------------------
This script migrates model files from a source directory to a target directory.

It's designed to be invoked by the installer to restore model files from a backup
location to their intended destination. The script:
- Checks if source and target directories exist
- Logs all operations to a log file in the target directory
- For each file in the source directory:
  - If the corresponding target file exists, removes it
  - Moves the file from source to target

Usage:
    python move_model_files.py <src_dir> <target_dir>

Where:
    <src_dir>: Source directory containing the model files to be moved
    <target_dir>: Target directory where model files should be placed
"""

# To be invoked by installer
# Usage: python move_model_files.py <src_dir> <target_dir>

import os
import sys


# Validate command line arguments
if len(sys.argv) != 3:
    print("Usage: python move_model_files.py <src_dir> <target_dir>")
    sys.exit(1)


# Extract and validate source and target directories
src_dir = sys.argv[1]
target_dir = sys.argv[2]
if not os.path.exists(src_dir):
    print("Backup model directory does not exist: " + src_dir)
    sys.exit(1)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


# Set up logging
log_file = os.path.join(target_dir, "copy.log")
if os.path.exists(log_file):
    os.remove(log_file)


def log(msg):
    """
    Log a message both to stdout and the log file.
    
    Args:
        msg: The message to log
    """
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def move_model_files(src_dir, target_dir):
    """
    Move all model files from source directory to target directory.
    
    For each file in the source directory tree, this function:
    1. Calculates the corresponding target path
    2. Removes any existing file at the target path
    3. Creates target subdirectories if they don't exist
    4. Moves the file from source to target, preserving the directory structure
    
    Args:
        src_dir: Source directory containing model files
        target_dir: Target directory where model files should be placed
        
    Raises:
        SystemExit: If any error occurs during the move operation
    """
    try:
        # for each file in src_dir, move it to target_dir if target path does not exist
        # otherwise, remove the target file and move the backup file to target path
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                src_file = os.path.join(root, file)
                target_file = src_file.replace(src_dir, target_dir)
                if os.path.exists(target_file):
                    os.remove(target_file)
                    log(f"Removed existing {target_file}")
                tdir = os.path.dirname(target_file)
                if not os.path.exists(tdir):
                    os.makedirs(tdir)
                os.rename(src_file, target_file)
                log(f"Moved {src_file} to {target_file}")
    except Exception as e:
        log("Failed to recover model files: " + str(e))
        sys.exit(1)


# Execute the file migration
move_model_files(src_dir, target_dir)
