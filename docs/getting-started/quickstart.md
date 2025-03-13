
# Quick Start Guide

This guide will help you quickly set up and start using AI-Playground.

## Prerequisites

- Python 3.10 or higher
- Git
- 16GB+ of RAM recommended
- NVIDIA GPU or Intel GPU (optional, but recommended for better performance)

## Installation

1. Clone the repository:

   \`\`\`text\`bash
   git clone <https://github.com/intel/AI-Playground.git>
   cd AI-Playground

   ```text`text

   ```text`

   ```text`

1. Set up the environment:

   \`\`\`text\`bash

   ## For automatic hardware detection and environment setup

   python setup_hardware_env.py

   ```text`text

   This will:
   - Detect your hardware configuration
   - Create a Python virtual environment
   - Install the appropriate dependencies for your hardware

   ```text`

   ```text`

## Basic Usage

1. Activate the virtual environment:

   \`\`\`text\`bash

   ## On Windows

   .venv\\Scripts\\activate

   ## On Linux/macOS

   source .venv/bin/activate

   ```text`text

   ```text`

   ```text`

1. Run the service:

   \`\`\`text\`bash
   python service/main.py

   ```text`text

   ```text`

   ```text`

1. Access the web interface by opening `<http://localhost:8000`> in your browser.

## Hardware-specific Optimizations

AI-Playground automatically detects and optimizes for your hardware:

- Intel Arc GPUs: Uses Intel Extension for PyTorch and Intel optimized packages
- Intel Meteor Lake CPUs: Utilizes optimized NPU and GPU capabilities
- NVIDIA GPUs: Standard PyTorch with CUDA acceleration
- CPU-only: Optimized CPU inferencing

## Next Steps

- [Installation Guide](installation.md) - For detailed installation instructions
- [Hardware Optimization](../hardware/optimization.md) - Learn how to optimize for your specific hardware
- [Example Workflows](../reference/examples.md) - Explore example workflows and use cases

***_****_****_****_****_****_****_****_****_****_****_****_****_****__

**Next**: [Installation Guide](installation.md) | __See also_*: [Hardware Overview](../hardware/overview.md)