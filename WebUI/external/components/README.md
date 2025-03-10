# ComfyUI Component Library

This directory contains reusable components for ComfyUI workflows. Components are modular, self-contained parts of workflows that can be reused across multiple different workflows.

## What are Components?

Components are collections of connected nodes that perform a specific function, such as:

- Image loading and preprocessing
- Text embedding with specific models
- Common sampling configurations
- Output processing and saving

Each component is defined in its own JSON file with standard inputs and outputs, making them easy to integrate into larger workflows.

## Using Components

To use a component in your workflow:

1. Reference it in your workflow file's `components` array with the appropriate `componentId`
1. Import the component nodes into your workflow by using the component loader
1. Connect the component's inputs and outputs to other parts of your workflow

## Component Structure

Each component follows this standard format:

```json
{
  "name": "Component Name",
  "description": "What this component does",
  "version": "1.0.0",
  "type": "image_loader|text_encoder|sampler|output_processor|etc",
  "inputs": [

```text
{
  "name": "input_1",
  "type": "string|number|image|etc",
  "description": "Description of this input"
}

```text
  ],
  "outputs": [

```text
{
  "name": "output_1",
  "type": "latent|image|conditioning|etc",
  "description": "Description of this output"
}

```text
  ],
  "nodes": {

```text
"1": {
  "class_type": "NodeType",
  "inputs": {

```text
"param1": "value1"

```text
  }
},
"2": {
  "class_type": "AnotherNodeType",
  "inputs": {

```text
"param1": ["1", 0]

```text
  }
}

```text
  },
  "inputMappings": {

```text
"input_1": {
  "nodeId": "1",
  "inputName": "param1"
}

```text
  },
  "outputMappings": {

```text
"output_1": {
  "nodeId": "2",
  "outputIndex": 0
}

```text
  }
}

```text

## Available Components

- **Image Loaders**: Components for loading and processing input images
- **Text Encoders**: Components for encoding text prompts with different models
- **Samplers**: Common sampling configurations for different quality/speed tradeoffs
- **Model Loaders**: Standard configurations for loading models
- **Output Processors**: Components for post-processing and saving outputs

## Creating Components

To create a new component:

1. Identify a reusable pattern in your workflows
1. Extract the relevant nodes and connections
1. Define clear inputs and outputs
1. Add appropriate documentation
1. Save it in this directory with a descriptive filename

## Best Practices

- Keep components focused on a single responsibility
- Document all inputs and outputs clearly
- Version your components using semantic versioning
- Test components in isolation before including them in workflows
- Consider resource requirements when designing components
