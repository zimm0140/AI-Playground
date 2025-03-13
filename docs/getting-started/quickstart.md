
# Quick Start Guide {#quick-start-guide}

This guide will help you quickly set up and start using AI-Playground.

## Prerequisites {#prerequisites}

- Python 3.10 or higher

- Git

- 16GB+ of RAM recommended

- NVIDIA GPU or Intel GPU (optional, but recommended for better performance)

## Installation {#installation}

1. Clone the repository:

   \`\`\`text\`bash
   git clone <https://github.com/intel/AI-Playground.git>
   cd AI-Playground

   ````

   ````

   ````

1. Set up the environment:

   \`\`\`text\`bash

   ## For automatic hardware detection and environment setup

   python setup_hardware_env.py

   ````

   This will:
   - Detect your hardware configuration
   - Create a Python virtual environment
   - Install the appropriate dependencies for your hardware

   ````

   ````

## Basic Usage {#basic-usage}

1. Activate the virtual environment:

   \`\`\`text\`bash

   ## On Windows

   .venv\\Scripts\\activate

   ## On Linux/macOS

   source .venv/bin/activate

   ````

   ````

   ````

1. Run the service:

   \`\`\`text\`bash
   python service/main.py

   ````

   ````

   ````

1. Access the web interface by opening `<http://localhost:8000`> in your browser.

## Hardware-specific Optimizations {#hardware-specific-optimizations}

AI-Playground automatically detects and optimizes for your hardware:

- Intel Arc GPUs: Uses Intel Extension for PyTorch and Intel optimized packages

- Intel Meteor Lake CPUs: Utilizes optimized NPU and GPU capabilities

- NVIDIA GPUs: Standard PyTorch with CUDA acceleration

- CPU-only: Optimized CPU inferencing

## Next Steps {#next-steps}

- [Installation Guide](installation.md) - For detailed installation instructions

- [Hardware Optimization](../hardware/optimization.md) - Learn how to optimize for your specific hardware

- [Example Workflows](../reference/examples.md) - Explore example workflows and use cases

***_****_****_****_****_****_****_****_****_****_****_****_****_****__

**Next**: [Installation Guide](installation.md) | __See also_*: [Hardware Overview](../hardware/overview.md)
