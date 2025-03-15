import re


def generate_heading_id(heading: str) -> str:
    """Generate a GitHub-style heading ID from a markdown heading."""
    # Remove any HTML tags
    heading = re.sub(r"<[^>]+>", "", heading)

    # Convert to lowercase
    heading = heading.lower()

    # Replace spaces with hyphens
    heading = heading.replace(" ", "-")

    # Remove all characters except alphanumeric and hyphens
    heading = re.sub(r"[^a-z0-9-]", "", heading)

    # Replace multiple hyphens with a single hyphen
    heading = re.sub(r"-+", "-", heading)

    # Remove leading and trailing hyphens
    heading = heading.strip("-")

    return heading


def main():
    """Test the heading ID generator with some example headings."""
    test_headings = [
        "Using Lockfiles for Reproducible Environments",
        "Working with Docker",
        "CI/CD Pipeline Updates",
        "Migration FAQs",
        "Using the AI Framework Integration",
        "Working with LangChain",
        "Working with Stable Diffusion",
        "Performance Benchmarking",
        "Troubleshooting",
    ]

    print("GitHub-style heading IDs:")
    for heading in test_headings:
        heading_id = generate_heading_id(heading)
        print(f"{heading} -> #{heading_id}")


if __name__ == "__main__":
    main()