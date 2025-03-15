#!/usr/bin/env python
"""
Platform Compatibility Checker

This script analyzes the codebase for potential platform-specific issues
that might cause compatibility problems across different operating systems.
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict


class PlatformCompatibilityChecker:
    def __init__(self, report_dir="ci_artifacts/platform_checks"):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

        # Define patterns to check for platform-specific issues
        self.patterns = [
            {
                "name": "windows_path_separator",
                "description": "Windows-specific path separator (backslash)",
                "pattern": r'[\'"][a-zA-Z]:\\[^\'"]*[\'"]|[\'"][^\'"]*\\\\[^\'"]*[\'"]',
                "platforms": ["windows"],
                "severity": "warning",
            },
            {
                "name": "unix_path_separator",
                "description": "Unix-specific path separator (forward slash)",
                "pattern": r'[\'"]/[^\'"]*[\'"]',
                "platforms": ["unix", "macos"],
                "severity": "info",
            },
            {
                "name": "windows_line_endings",
                "description": "Windows-specific line endings (CRLF)",
                "pattern": r"\r\n",
                "platforms": ["windows"],
                "severity": "warning",
            },
            {
                "name": "environment_variable_case",
                "description": "Environment variable case sensitivity issues",
                "pattern": r'os\.environ\[[\'"](?!PATH|TEMP|TMP|HOME|USER)[A-Z_]+[\'"]\]',
                "platforms": ["unix", "macos", "windows"],
                "severity": "warning",
            },
            {
                "name": "platform_specific_commands",
                "description": "Platform-specific command execution",
                "pattern": r'(?:os\.system|subprocess\.(?:call|run|Popen))\([\'"](?:cmd|powershell|bash|sh|systemctl)',
                "platforms": ["unix", "macos", "windows"],
                "severity": "warning",
            },
            {
                "name": "windows_registry",
                "description": "Windows Registry access",
                "pattern": r"winreg|_winreg|HKEY_|OpenKey|REG_",
                "platforms": ["windows"],
                "severity": "error",
            },
            {
                "name": "unix_specific_paths",
                "description": "Unix-specific paths",
                "pattern": r'[\'"](?:/etc/|/var/|/usr/|/proc/|/sys/)',
                "platforms": ["unix", "macos"],
                "severity": "error",
            },
            {
                "name": "macos_specific_apis",
                "description": "macOS-specific APIs",
                "pattern": r"NSApplication|CFBundle|NSURL|Foundation|AppKit|Cocoa",
                "platforms": ["macos"],
                "severity": "error",
            },
            {
                "name": "file_permission_unix",
                "description": "Unix-specific file permission operations",
                "pattern": r"os\.chmod\([^)]+, 0o[0-7]{3}\)|stat\.S_I[RWX]",
                "platforms": ["unix", "macos"],
                "severity": "warning",
            },
            {
                "name": "drive_letters",
                "description": "Windows drive letters",
                "pattern": r'[\'"][A-Za-z]:/',
                "platforms": ["windows"],
                "severity": "error",
            },
            {
                "name": "os_conditional_blocks",
                "description": "Explicit OS conditional checks",
                "pattern": r"if\s+(?:sys\.platform|platform\.system\(\))\s*(?:==|!=|in|not in)",
                "platforms": ["unix", "macos", "windows"],
                "severity": "info",
            },
        ]

    def find_files(self, extensions=None):
        """Find all files to check for platform-specific issues."""
        if extensions is None:
            extensions = [".py", ".sh", ".bat", ".ps1", ".cmd"]

        files = []
        for ext in extensions:
            for file in glob.glob(f"**/*{ext}", recursive=True):
                # Skip files in version control, virtual environments, and CI directories
                if any(
                    path in file for path in [".git/", "venv/", ".github/workflows/"]
                ):
                    continue
                files.append(file)

        return sorted(files)

    def check_file(self, file_path):
        """Check a file for platform-specific issues."""
        issues = []

        if not os.path.exists(file_path):
            return issues

        # Check file extension for inherently platform-specific files
        _, ext = os.path.splitext(file_path.lower())
        if ext in [".bat", ".ps1", ".cmd"]:
            issues.append(
                {
                    "file": file_path,
                    "line": 0,
                    "pattern_name": "windows_specific_script",
                    "description": f"File type '{ext}' is Windows-specific",
                    "platforms": ["windows"],
                    "severity": "error",
                },
            )
            return issues  # No need to check the content further

        if ext in [".sh"]:
            issues.append(
                {
                    "file": file_path,
                    "line": 0,
                    "pattern_name": "unix_specific_script",
                    "description": f"File type '{ext}' is Unix/macOS-specific",
                    "platforms": ["unix", "macos"],
                    "severity": "error",
                },
            )
            return issues  # No need to check the content further

        # Read the file and check its content
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Check for line endings
                if "\r\n" in content:
                    issues.append(
                        {
                            "file": file_path,
                            "line": 0,
                            "pattern_name": "windows_line_endings",
                            "description": "File uses Windows-specific line endings (CRLF)",
                            "platforms": ["windows"],
                            "severity": "info",
                        },
                    )

                # Check against other patterns
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    for pattern in self.patterns:
                        matches = re.search(pattern["pattern"], line)
                        if matches:
                            issues.append(
                                {
                                    "file": file_path,
                                    "line": i + 1,
                                    "pattern_name": pattern["name"],
                                    "description": f"{pattern['description']}: {matches.group(0)}",
                                    "platforms": pattern["platforms"],
                                    "severity": pattern["severity"],
                                },
                            )

                # Check for shebang in script files
                if lines and lines[0].startswith("#!") and "/bin/" in lines[0]:
                    issues.append(
                        {
                            "file": file_path,
                            "line": 1,
                            "pattern_name": "unix_shebang",
                            "description": f"Unix-specific shebang: {lines[0]}",
                            "platforms": ["unix", "macos"],
                            "severity": "warning",
                        },
                    )
        except Exception as e:
            issues.append(
                {
                    "file": file_path,
                    "line": 0,
                    "pattern_name": "file_error",
                    "description": f"Error reading file: {str(e)}",
                    "platforms": [],
                    "severity": "error",
                },
            )

        return issues

    def check_platform_specific_imports(self, file_path):
        """Check for platform-specific imports in Python files."""
        issues = []

        if not file_path.lower().endswith(".py"):
            return issues

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Check for platform-specific imports
                platform_imports = {
                    "windows": [
                        "winreg",
                        "_winreg",
                        "msvcrt",
                        "win32api",
                        "win32con",
                        "win32com",
                        "win32gui",
                    ],
                    "unix": ["fcntl", "grp", "pwd", "termios"],
                    "macos": ["AppKit", "Cocoa", "CoreFoundation", "objc"],
                }

                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if not line.strip().startswith("#") and re.search(
                        r"^\s*(?:import|from)\s+", line,
                    ):
                        for platform, imports in platform_imports.items():
                            for import_name in imports:
                                if re.search(
                                    rf"(?:import|from)\s+{import_name}(?:\s+|\.|$)",
                                    line,
                                ):
                                    issues.append(
                                        {
                                            "file": file_path,
                                            "line": i + 1,
                                            "pattern_name": f"{platform}_specific_import",
                                            "description": f"{platform.capitalize()}-specific import: {import_name}",
                                            "platforms": [platform],
                                            "severity": "error",
                                        },
                                    )
        except Exception as e:
            issues.append(
                {
                    "file": file_path,
                    "line": 0,
                    "pattern_name": "file_error",
                    "description": f"Error reading file: {str(e)}",
                    "platforms": [],
                    "severity": "error",
                },
            )

        return issues

    def check_platform_specific_code_blocks(self, file_path):
        """Check for platform-specific code blocks in Python files."""
        issues = []

        if not file_path.lower().endswith(".py"):
            return issues

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Check for platform-specific conditional blocks
                platform_conditions = [
                    r'if\s+sys\.platform\s*==\s*[\'"]win32[\'"]',
                    r'if\s+sys\.platform\s*==\s*[\'"]darwin[\'"]',
                    r'if\s+sys\.platform\s*==\s*[\'"]linux[\'"]',
                    r'if\s+platform\.system\(\)\s*==\s*[\'"]Windows[\'"]',
                    r'if\s+platform\.system\(\)\s*==\s*[\'"]Darwin[\'"]',
                    r'if\s+platform\.system\(\)\s*==\s*[\'"]Linux[\'"]',
                    r'if\s+os\.name\s*==\s*[\'"]nt[\'"]',
                    r'if\s+os\.name\s*==\s*[\'"]posix[\'"]',
                ]

                lines = content.split("\n")
                for i, line in enumerate(lines):
                    for condition in platform_conditions:
                        matches = re.search(condition, line)
                        if matches:
                            issues.append(
                                {
                                    "file": file_path,
                                    "line": i + 1,
                                    "pattern_name": "platform_specific_condition",
                                    "description": f"Platform-specific condition: {matches.group(0)}",
                                    "platforms": ["cross-platform"],
                                    "severity": "info",  # This is actually a good practice for cross-platform code
                                },
                            )
        except Exception as e:
            issues.append(
                {
                    "file": file_path,
                    "line": 0,
                    "pattern_name": "file_error",
                    "description": f"Error reading file: {str(e)}",
                    "platforms": [],
                    "severity": "error",
                },
            )

        return issues

    def generate_report(self, all_issues):
        """Generate a comprehensive report of all platform-specific issues."""
        # Organize issues by file
        issues_by_file = defaultdict(list)
        for issue in all_issues:
            issues_by_file[issue["file"]].append(issue)

        # Organize issues by severity
        issues_by_severity = defaultdict(list)
        for issue in all_issues:
            issues_by_severity[issue["severity"]].append(issue)

        # Organize issues by platform
        issues_by_platform = defaultdict(list)
        for issue in all_issues:
            for platform in issue.get("platforms", []):
                issues_by_platform[platform].append(issue)

        # Save detailed issues to JSON
        json_path = os.path.join(self.report_dir, "platform_issues.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_issues, f, indent=2)

        # Generate markdown report
        markdown_path = os.path.join(
            self.report_dir, "platform_compatibility_report.md",
        )
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("# Platform Compatibility Report\n\n")
            f.write(
                "This report identifies potential platform-specific code that may cause compatibility issues.\n\n",
            )

            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- **Total files analyzed**: {len(issues_by_file)}\n")
            f.write(f"- **Total issues found**: {len(all_issues)}\n")
            f.write(
                f"- **Error level issues**: {len(issues_by_severity.get('error', []))}\n",
            )
            f.write(
                f"- **Warning level issues**: {len(issues_by_severity.get('warning', []))}\n",
            )
            f.write(
                f"- **Info level issues**: {len(issues_by_severity.get('info', []))}\n\n",
            )

            # Platform breakdown
            f.write("## Platform-Specific Issues\n\n")
            f.write("| Platform | Issues |\n")
            f.write("|----------|--------|\n")
            for platform, issues in sorted(issues_by_platform.items()):
                if platform == "cross-platform":
                    continue  # Skip cross-platform issues in this table
                f.write(f"| {platform.capitalize()} | {len(issues)} |\n")
            f.write("\n")

            # Files with most issues
            f.write("## Files with Platform-Specific Issues\n\n")
            f.write("| File | Total Issues | Errors | Warnings | Info |\n")
            f.write("|------|--------------|--------|----------|------|\n")
            for file, issues in sorted(
                issues_by_file.items(), key=lambda x: len(x[1]), reverse=True,
            ):
                errors = sum(1 for i in issues if i["severity"] == "error")
                warnings = sum(1 for i in issues if i["severity"] == "warning")
                infos = sum(1 for i in issues if i["severity"] == "info")
                f.write(
                    f"| {file} | {len(issues)} | {errors} | {warnings} | {infos} |\n",
                )
            f.write("\n")

            # Detailed issues by file
            f.write("## Detailed Issues\n\n")
            for file, issues in sorted(issues_by_file.items()):
                f.write(f"### {file}\n\n")

                for issue in sorted(
                    issues,
                    key=lambda x: (
                        x["severity"] != "error",
                        x["severity"] != "warning",
                        x["line"],
                    ),
                ):
                    severity_emoji = (
                        "🔴"
                        if issue["severity"] == "error"
                        else "⚠️"
                        if issue["severity"] == "warning"
                        else "ℹ️"
                    )
                    platforms = ", ".join(
                        p.capitalize() for p in issue.get("platforms", [])
                    )
                    line_info = f"Line {issue['line']}: " if issue["line"] else ""

                    f.write(
                        f"- {severity_emoji} **{issue['pattern_name']}** ({platforms}) - {line_info}{issue['description']}\n",
                    )

                f.write("\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(
                "1. **Use os.path for file operations**: Replace hardcoded path separators with `os.path.join()`\n",
            )
            f.write(
                "2. **Use pathlib for modern path handling**: Consider using the cross-platform `pathlib` module\n",
            )
            f.write(
                "3. **Check platform conditionally**: Use `if sys.platform == 'win32'` for platform-specific code\n",
            )
            f.write(
                "4. **Use consistent line endings**: Configure your editor to use LF (Unix-style) line endings\n",
            )
            f.write(
                "5. **Abstract platform-specific operations**: Create utility functions to abstract OS-specific code\n\n",
            )

            f.write("## Next Steps\n\n")
            f.write(
                "- Review error-level issues first as they're most likely to cause compatibility problems\n",
            )
            f.write(
                "- Consider adding cross-platform tests to verify behavior across different operating systems\n",
            )
            f.write(
                "- Use techniques like dependency injection to make platform-specific code more testable\n\n",
            )

            # Footer
            f.write("---\n\n")
            f.write(
                f"*This report was automatically generated by the CI process on {os.popen('date').read().strip()}*\n",
            )

        return markdown_path, json_path

    def generate_github_summary(self, all_issues):
        """Generate GitHub step summary from the analysis results."""
        if not os.environ.get("GITHUB_STEP_SUMMARY"):
            return

        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write("## Platform Compatibility Check\n\n")

            # Count issues by severity
            error_count = sum(1 for i in all_issues if i["severity"] == "error")
            warning_count = sum(1 for i in all_issues if i["severity"] == "warning")
            info_count = sum(1 for i in all_issues if i["severity"] == "info")

            # Count issues by platform
            platform_counts = defaultdict(int)
            for issue in all_issues:
                for platform in issue.get("platforms", []):
                    platform_counts[platform] += 1

            f.write("### Overview\n\n")
            f.write(
                f"Found **{len(all_issues)}** potential platform compatibility issues\n\n",
            )

            # Status indicators
            if error_count > 0:
                f.write(
                    "🔴 **Platform-specific issues found that may break cross-platform compatibility**\n\n",
                )
            elif warning_count > 0:
                f.write(
                    "⚠️ **Minor platform-specific issues found that should be reviewed**\n\n",
                )
            else:
                f.write("✅ **No critical platform-specific issues found**\n\n")

            # Issues breakdown
            f.write("| Severity | Count |\n")
            f.write("|----------|-------|\n")
            f.write(f"| Errors | {error_count} |\n")
            f.write(f"| Warnings | {warning_count} |\n")
            f.write(f"| Info | {info_count} |\n\n")

            # Platform breakdown
            f.write("| Platform | Issues |\n")
            f.write("|----------|--------|\n")
            for platform, count in sorted(platform_counts.items()):
                if platform == "cross-platform":
                    continue
                f.write(f"| {platform.capitalize()} | {count} |\n")
            f.write("\n")

            # Provide recommendation based on findings
            if error_count > 0:
                f.write(
                    "⚠️ **Recommendation**: Review error-level issues immediately to ensure cross-platform compatibility\n",
                )
            elif warning_count > 0:
                f.write(
                    "ℹ️ **Recommendation**: Consider refactoring code with platform-specific patterns for better compatibility\n",
                )
            else:
                f.write(
                    "✅ **Recommendation**: Continue maintaining good cross-platform practices\n",
                )

            f.write(
                "\nSee platform compatibility report artifact for detailed information.\n",
            )

    def run(self):
        """Run the platform compatibility check on all relevant files."""
        files = self.find_files()
        all_issues = []

        print(f"Analyzing {len(files)} files for platform-specific issues...")

        for file_path in files:
            print(f"Checking {file_path}...")

            # Basic pattern checks
            issues = self.check_file(file_path)
            all_issues.extend(issues)

            # Python-specific checks
            if file_path.lower().endswith(".py"):
                # Check for platform-specific imports
                import_issues = self.check_platform_specific_imports(file_path)
                all_issues.extend(import_issues)

                # Check for platform-specific code blocks
                code_block_issues = self.check_platform_specific_code_blocks(file_path)
                all_issues.extend(code_block_issues)

            # Print summary of issues
            errors = sum(1 for i in issues if i["severity"] == "error")
            warnings = sum(1 for i in issues if i["severity"] == "warning")
            infos = sum(1 for i in issues if i["severity"] == "info")

            if errors or warnings or infos:
                issue_summary = []
                if errors:
                    issue_summary.append(f"{errors} errors")
                if warnings:
                    issue_summary.append(f"{warnings} warnings")
                if infos:
                    issue_summary.append(f"{infos} info")
                print(f"  Found {', '.join(issue_summary)}")
            else:
                print("  ✓ No platform-specific issues")

        # Generate reports
        markdown_path, json_path = self.generate_report(all_issues)
        self.generate_github_summary(all_issues)

        print(
            f"Platform compatibility analysis complete. Reports saved to {markdown_path} and {json_path}",
        )

        # Return exit code based on errors
        return 1 if any(issue["severity"] == "error" for issue in all_issues) else 0


def main():
    parser = argparse.ArgumentParser(
        description="Check for platform-specific issues in the codebase",
    )
    parser.add_argument(
        "--report-dir",
        default="ci_artifacts/platform_checks",
        help="Directory to store the report",
    )
    args = parser.parse_args()

    checker = PlatformCompatibilityChecker(report_dir=args.report_dir)
    # Exit with code 0 to allow CI to continue regardless of findings
    # The report will still show issues that need to be addressed
    checker.run()
    sys.exit(0)


if __name__ == "__main__":
    main()