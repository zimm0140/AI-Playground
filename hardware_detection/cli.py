#!/usr/bin/env python3
"""CLI for hardware detection."""

import argparse
import logging
import sys
from pathlib import Path

from hardware_detection import __version__
from hardware_detection.core import (
    detect_hardware_type,
    get_hardware_info,
    print_hardware_info,
)


def setup_logging(debug: bool = False) -> None:
    """Set up logging for the CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[HARDWARE] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main() -> int:
    """Run the CLI."""
    parser = argparse.ArgumentParser(
        description="Hardware detection CLI for identifying specialized hardware."
    )
    parser.add_argument(
        "--version", action="version", version=f"hardware_detection {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show hardware information")
    info_parser.add_argument(
        "--json", action="store_true", help="Output in JSON format"
    )

    # Detect command
    detect_parser = subparsers.add_parser("detect", help="Detect hardware type")

    # Mock command
    mock_parser = subparsers.add_parser("mock", help="Create mock environment")
    mock_parser.add_argument(
        "--hardware-type",
        choices=["base", "acm", "ovino"],
        default="base",
        help="Type of hardware to mock",
    )
    mock_parser.add_argument(
        "--mock-dir",
        type=Path,
        default=Path(".uvfast/mock"),
        help="Directory to store mock files",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.debug)

    # Default to info command if no command specified
    if not args.command:
        args.command = "info"

    # Run the appropriate command
    if args.command == "info":
        if args.json:
            import json

            print(json.dumps(get_hardware_info(), indent=2))
        else:
            print_hardware_info(verbose=args.verbose)
    elif args.command == "detect":
        print(detect_hardware_type())
    elif args.command == "mock":
        import json
        import os

        # Create mock directory
        mock_dir = args.mock_dir
        mock_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables
        os.environ["SIMULATED_HARDWARE"] = args.hardware_type
        os.environ["UVFAST_MOCK_DIR"] = str(mock_dir)

        # Create mock files based on hardware type
        if args.hardware_type == "base":
            with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
                f.write("Generic GPU\n")

            with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
                f.write("vendor: Generic\n")
                f.write("name: Generic CPU\n")
                f.write("cores: 4\n")

        elif args.hardware_type == "acm":
            with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
                f.write("Intel(R) Arc(TM) A770 Graphics\n")

            with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
                f.write("vendor: Intel\n")
                f.write("name: Intel(R) Core(TM) i9-13900K\n")
                f.write("cores: 24\n")

        elif args.hardware_type == "ovino":
            with open(mock_dir / "gpu_info.txt", "w", encoding="utf-8") as f:
                f.write("Intel(R) UHD Graphics 770\n")

            with open(mock_dir / "cpu_info.txt", "w", encoding="utf-8") as f:
                f.write("vendor: Intel\n")
                f.write("name: Intel(R) Core(TM) i7-1370P\n")
                f.write("cores: 16\n")

        print(f"Mock environment for {args.hardware_type} created at {mock_dir}")

        # Show the resulting hardware detection
        print("\nDetected hardware:")
        print_hardware_info(verbose=args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
