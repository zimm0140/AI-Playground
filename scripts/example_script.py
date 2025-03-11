#!/usr/bin/env python
"""
Example script demonstrating uv inline dependencies.

To run this script with uv:
    uv run scripts/example_script.py

Dependencies:
# uv: requests>=2.28.0
# uv: tqdm>=4.65.0
# uv: rich>=12.0.0
"""
import json
import requests
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

def get_github_repos(org_name, n=5):
    """Fetch top repositories for a GitHub organization."""
    url = f"https://api.github.com/orgs/{org_name}/repos?sort=stars&per_page={n}"
    response = requests.get(url, headers={"Accept": "application/vnd.github.v3+json"})
    return response.json()

def display_repos(repos):
    """Display the repositories in a pretty table."""
    table = Table(title=f"Top GitHub Repositories for {repos[0]['owner']['login']}")
    
    table.add_column("Name", style="cyan")
    table.add_column("Stars", justify="right", style="green")
    table.add_column("Language", style="magenta")
    table.add_column("Description")
    
    for repo in repos:
        table.add_row(
            repo["name"],
            str(repo["stargazers_count"]),
            repo["language"] or "None",
            (repo["description"] or "")[:50] + ("..." if repo["description"] and len(repo["description"]) > 50 else "")
        )
    
    console.print(table)

def main():
    orgs = ["astral-sh", "microsoft", "google"]
    
    for org in tqdm(orgs, desc="Fetching organizations"):
        console.print(f"\n[bold blue]Organization:[/] {org}")
        repos = get_github_repos(org)
        display_repos(repos)
        console.print("\n" + "-" * 80 + "\n")

if __name__ == "__main__":
    console.print("[bold yellow]Starting GitHub Repository Explorer[/]")
    main()
    console.print("[bold green]Done![/]") 