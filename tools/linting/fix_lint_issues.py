#!/usr/bin/env python3
"""
Script to automatically fix common linting issues in the codebase.

Enhanced version with:
- Progress tracking
- Selective rule fixing
- Detailed reporting
- Automatic backup
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LintingFixer:
    def __init__(self, directories: List[str], rules: Optional[List[str]] = None):
        self.directories = directories
        self.rules = rules or ["F401", "W291", "F821", "N801", "N802", "N803"]
        self.stats: Dict[str, int] = {rule: 0 for rule in self.rules}
        self.backup_dir = Path("lint_backups") / datetime.now().strftime("%Y%m%d_%H%M%S")

    def backup_file(self, file_path: Path) -> None:
        """Create a backup of the file before modifying it."""
        backup_path = self.backup_dir / file_path.relative_to(Path.cwd())
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)

    def fix_file(self, file_path: Path) -> Tuple[bool, Dict[str, int]]:
        """Fix linting issues in a single file."""
        try:
            # Backup the file
            self.backup_file(file_path)

            # Run ruff with --fix for each rule
            fixes_applied = {rule: 0 for rule in self.rules}
            for rule in self.rules:
                result = subprocess.run(
                    ["ruff", "check", "--fix", f"--select={rule}", str(file_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    fixes_applied[rule] = len(result.stdout.splitlines())

            # Run formatter after fixes
            subprocess.run(
                ["ruff", "format", str(file_path)],
                check=False,
                capture_output=True,
            )

            return True, fixes_applied
        except Exception as e:
            print(f"❌ Error processing {file_path}: {str(e)}")
            return False, {}

    def generate_report(self, start_time: datetime) -> str:
        """Generate a detailed report of fixes applied."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        report = [
            "# Linting Fix Report",
            f"\nRun completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {duration:.2f} seconds\n",
            "## Rules Fixed",
        ]

        for rule, count in self.stats.items():
            report.append(f"- {rule}: {count} fixes")

        return "\n".join(report)

    def run(self) -> bool:
        """Run the linting fixes on all directories."""
        start_time = datetime.now()
        success = True

        for directory in self.directories:
            if not Path(directory).exists():
                print(f"⚠️ Directory {directory} does not exist, skipping.")
                continue

            print(f"🛠️ Fixing issues in {directory}...")
            py_files = list(Path(directory).rglob("*.py"))

            for i, file_path in enumerate(py_files, 1):
                print(f"Processing {file_path} ({i}/{len(py_files)})")
                file_success, fixes = self.fix_file(file_path)
                success &= file_success

                # Update statistics
                for rule, count in fixes.items():
                    self.stats[rule] += count

        # Generate and save report
        report = self.generate_report(start_time)
        report_path = Path("lint_report.md")
        report_path.write_text(report)
        print(f"\nReport saved to {report_path}")

        return success


def main():
    """Run the enhanced linting fixer."""
    parser = argparse.ArgumentParser(description="Fix linting issues with enhanced tracking")
    parser.add_argument("directories", nargs="*", default=["."], help="Directories to process")
    parser.add_argument("--rules", nargs="*", help="Specific rules to fix")
    args = parser.parse_args()

    fixer = LintingFixer(args.directories, args.rules)
    success = fixer.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
