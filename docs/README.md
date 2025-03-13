# Documentation Directory

This directory contains all documentation for the AI Playground project.

## Directory Structure

- *_development/__: Development-related documentation

  - Code quality guides
  - Contributing guidelines
  - Security policies
  - Modernization reports
  - Implementation guides

- __hardware/__: Hardware-specific documentation

  - Hardware optimization guides
  - Hardware compatibility information
  - Hardware-aware features

- __user-guides/__: End-user documentation

  - User manuals and guides
  - PDF documentation

- __workflows/_*: Workflow documentation

  - CI/CD workflow documentation
  - Requirements and reports

## Documentation Format

Most documentation is written in Markdown format and can be viewed directly on GitHub or through the project's documentation site generated with MkDocs.

## Building Documentation

The documentation site can be built using MkDocs:

```text`text

mkdocs build

```text

To serve the documentation locally:

```text

mkdocs serve

```text

See the `mkdocs.yml` file in the root directory for configuration details.

## Documentation Structure

The documentation is organized into the following sections:

- `getting-started/`: Guides for new users to get started with AI-Playground
- `hardware/`: Hardware compatibility and optimization guides
- `development/`: Guides for contributors and developers
- `architecture/`: Documentation on system architecture and design
- `reference/`: API references and configuration documentation
- `examples/`: Example code and usage patterns
- `troubleshooting/`: Guides for resolving common issues
- `community/`: Community guidelines and support information
- `releases/`: Release notes and version history
- `maintenance/`: Documentation about maintaining the documentation itself

## Key Files

- `index.md`: Main entry point for documentation
- `documentation-overview.md`: Comprehensive overview of all documentation
- `mkdocs.yml`: Configuration for the MkDocs documentation generator (in project root)

## Documentation Standards

All documentation follows these standards:

- Markdown for all documentation files

- Consistent headers using ATX style (# for headers)

- Code examples in fenced code blocks with appropriate language tags

- Relative links between documents

- Images stored in the `assets/` directory

## Contributing to Documentation

Contributions to documentation are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a branch for your changes
3. Make your changes following our documentation standards
4. Submit a pull request

For more details, see [Contributing to Documentation](development/contributing.md).

## Related Tools

Documentation tools and scripts are located in:

- `tools/linting/fix_markdown_lint.py`: Script to fix common markdown linting issues
- `tools/fix_markdown_advanced.py`: Advanced script for fixing markdown linting issues
- `tools/formatting/fix_readme.py`: Script to fix README formatting issues

## Configuration

Documentation linting is configured in:

- `config/.markdownlint.yaml`: Configuration for markdown linting

## Contact

If you have questions about the documentation, please open an issue or contact the maintainers at <example@example.com>.

```text`
