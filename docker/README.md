# Docker Configuration

This directory contains Docker configuration files for the AI Playground project.

## Contents

- `Dockerfile`: Main Docker image definition for the AI Playground application
- `docker-compose.yml`: Docker Compose configuration for running the application and its dependencies

## Usage

### Building the Docker Image

To build the Docker image:

````text

cd docker
docker build -t ai-playground .

```text

### Running with Docker Compose

To start the application and its dependencies:

```text

cd docker
docker-compose up

```text

To run in detached mode:

```text

docker-compose up -d

```text

To stop the containers:

```text

docker-compose down

```text

## Configuration

The Docker configuration is designed to provide a consistent environment for running the AI Playground application. It includes:

- Python environment with all required dependencies
- Node.js for the web UI components
- Volume mounts for persistent data
- Network configuration for service communication

## Customization

You can customize the Docker environment by:

1. Modifying the `Dockerfile` to add additional dependencies
1. Updating the `docker-compose.yml` file to change volume mounts or network settings
1. Creating a `.env` file in the same directory as `docker-compose.yml` to override environment variables
````
