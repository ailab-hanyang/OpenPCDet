# OpenPCDet Docker Environment Setup

This repository provides a Docker environment setup for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) project, optimized for GPU-based deep learning on LiDAR datasets. The Docker image includes all necessary dependencies, such as PyTorch with CUDA support, OpenCV, and ROS (Robot Operating System).

## Prerequisites

Before you begin, ensure that you have the following installed on your system:

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Docker Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Docker Image Content

The Docker image contains:

- **NVIDIA TensorRT**: For optimized deep learning inference.
- **OpenCV**: Version 4.2.0 for computer vision tasks.
- **PyTorch**: Version 2.1.1 with CUDA 12.1 support.
- **ROS Noetic**: For integrating with robotic systems.
- **OpenPCDet dependencies**: Includes libraries such as spconv, onnx, and others required by OpenPCDet.

## Building the Docker Image

1. Clone the repository:

   ```bash
   git clone https://github.com/ailab-konkuk/OpenPCDet.git
   cd OpenPCDet
   ```

2. Build the Docker image using the provided Dockerfile:

   ```bash
   docker build -f docker/env.Dockerfile -t openpcdet-env docker/
   ```

## Running the Container with Docker Compose

1. Start the container with Docker Compose:

   ```bash
   docker compose up --build -d
   ```

   This command builds and starts the container in detached mode.

## Using the Container

Once the container is running, you can access it with the following command:

```bash
docker exec -it openpcdet bash
```

### Inside the Container

- **Source ROS Noetic**: The ROS environment is already set up for you. It sources automatically when you start a new shell session.
  
- **OpenPCDet**: The `PYTHONPATH` is set to include the OpenPCDet directory. You can run your OpenPCDet scripts directly.

## Environment Variables

The following environment variables are set within the container:

- `PYTHONPATH`: Includes the OpenPCDet directory.
- `DISPLAY`: Set to support GUI applications via X11 forwarding.
- `NVIDIA_VISIBLE_DEVICES` and `NVIDIA_DRIVER_CAPABILITIES`: Configured for GPU support.

## Customization

You can customize the Dockerfile and `docker-compose.yml` according to your specific requirements, such as adding additional dependencies or modifying user permissions.

## Notes

- **ROS 2 Support**: If you need ROS 2 instead of ROS Noetic, you can uncomment the relevant lines in the Dockerfile for ROS 2 Foxy installation.
- **Volume Mounts**: The Docker Compose file mounts your current directory to the container's working directory, so any changes you make to your files on the host are immediately reflected in the container.

## Troubleshooting

If you encounter issues with missing modules or other errors, ensure that the paths and environment variables are set correctly within the container. You can manually source the environment files or re-run the setup commands as needed.
