#!/usr/bin/env python
"""
Example Script with Inline Dependencies

This script demonstrates uv's ability to handle inline dependencies.
It will fetch information about GitHub repositories for a given organization.

Usage:
    python -m scripts.example_script [organization_name] [num_repos]

    # Or with uv (which will automatically install dependencies):
    uv run scripts/example_script.py [organization_name] [num_repos]

Dependencies:
    # uv-x-package: requests>=2.28.0
    # uv-x-package: rich>=13.0.0
    # uv-x-package: tqdm>=4.66.0
"""

import argparse
import sys
from typing import Any

# These imports are from packages specified in the uv-x-package comments
import requests
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


def get_github_repos(org_name: str, n: int = 5) -> list[dict[str, Any]]:
    """Fetch GitHub repositories for a given organization.

    Args:
        org_name: The GitHub organization name
        n: Number of repositories to fetch

    Returns:
        List of repository data dictionaries
    """
    url = f"https://api.github.com/orgs/{org_name}/repos"
    headers = {"Accept": "application/vnd.github.v3+json"}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        repos = response.json()
        return repos[:n]
    except requests.RequestException as e:
        print(f"Error fetching repositories: {e}")
        return []


def display_repos(repos: list[dict[str, Any]]) -> None:
    """Display repository information in a formatted table.

    Args:
        repos: List of repository data dictionaries
    """
    console = Console()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="dim")
    table.add_column("Stars", justify="right")
    table.add_column("Forks", justify="right")
    table.add_column("Language", style="dim")
    table.add_column("Description")

    for repo in repos:
        table.add_row(
            repo["name"],
            str(repo["stargazers_count"]),
            str(repo["forks_count"]),
            repo.get("language", "N/A"),
            repo.get("description", "No description") or "No description",
        )

    console.print(table)


def main() -> None:
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description="Fetch and display GitHub repositories for an organization")
    parser.add_argument(
        "org_name", nargs="?", default="astral-sh", help="GitHub organization name (default: astral-sh)"
    )
    parser.add_argument(
        "num_repos", nargs="?", type=int, default=5, help="Number of repositories to display (default: 5)"
    )
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold green]Fetching repositories for {args.org_name}...[/bold green]")

    with console.status("[bold green]Fetching data from GitHub API...[/bold green]"):
        repos = get_github_repos(args.org_name, args.num_repos)

    if repos:
        console.print(f"\n[bold]Found {len(repos)} repositories for {args.org_name}:[/bold]\n")
        display_repos(repos)
    else:
        console.print(f"[bold red]No repositories found for {args.org_name}[/bold red]")

    # Demo of tqdm progress bar
    console.print("\n[italic]Demonstrating tqdm progress bar:[/italic]")
    for _ in tqdm(range(10), desc="Processing"):
        import time

        time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
