#!/usr/bin/env python3
"""
Automated Linting Fixer

This script automatically fixes high-priority linting issues across the codebase
using Ruff. It focuses on issues that can be automatically fixed without risk.

Usage:
    python tools/linting/auto_fix_linting.py [--dry-run] [--verbose]

Options:
    --dry-run   Show what would be fixed without making changes
    --verbose   Show detailed output
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# High-priority issues that can be fixed automatically
HIGH_PRIORITY_RULES = [
    "F401",  # Unused imports
    "F841",  # Unused variables
    "W291",  # Trailing whitespace
    "E711",  # Comparison to None should be 'if cond is None:'
    "E712",  # Comparison to True/False should be 'if cond is True:'
    "E713",  # Test for membership should be 'not in'
    "E714",  # Test for identity should be 'is not'
]

# Medium-priority issues that can be fixed automatically
MEDIUM_PRIORITY_RULES = [
    "E501",  # Line too long
    "E225",  # Missing whitespace around operator
    "E231",  # Missing whitespace after ','
    "E261",  # At least two spaces before inline comment
    "E271",  # Multiple spaces after keyword
    "E272",  # Multiple spaces before keyword
    "E303",  # Too many blank lines
    "E304",  # Blank lines found after function decorator
]


def setup_logging(verbose: bool) -> logging.Logger:
    """Set up logging with appropriate verbosity level."""
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Automated linting fixer")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument(
        "--priority", choices=["high", "medium", "all"], default="high", help="Priority level of issues to fix"
    )
    return parser.parse_args()


def run_ruff_fix(rules: list[str], dry_run: bool, logger: logging.Logger) -> bool:
    """Run Ruff to automatically fix issues."""
    cmd = ["python", "-m", "ruff", "check", ".", "--select", ",".join(rules)]

    if not dry_run:
        cmd.append("--fix")

    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            logger.info("No issues found to fix")
            return True

        if dry_run:
            logger.info("Issues that would be fixed in a real run:")
            logger.info(result.stdout)
        else:
            logger.info(f"Fixed {rules} issues")
            if result.stderr:
                logger.warning(f"Warnings during fix: {result.stderr}")

        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error running Ruff: {e}")
        return False


def get_rules_for_priority(priority: str) -> list[str]:
    """Get the rules to fix based on priority level."""
    if priority == "high":
        return HIGH_PRIORITY_RULES
    elif priority == "medium":
        return MEDIUM_PRIORITY_RULES
    elif priority == "all":
        return HIGH_PRIORITY_RULES + MEDIUM_PRIORITY_RULES
    else:
        raise ValueError(f"Unknown priority level: {priority}")


def main() -> int:
    """Main function to run the automated fixer."""
    args = parse_args()
    logger = setup_logging(args.verbose)

    logger.info("Starting automated linting fixer")
    if args.dry_run:
        logger.info("DRY RUN MODE: No changes will be made")

    rules = get_rules_for_priority(args.priority)
    logger.info(f"Fixing {args.priority} priority issues: {', '.join(rules)}")

    success = run_ruff_fix(rules, args.dry_run, logger)

    if success:
        logger.info("Linting fix completed successfully")
        return 0
    else:
        logger.warning("Linting fix completed with issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
