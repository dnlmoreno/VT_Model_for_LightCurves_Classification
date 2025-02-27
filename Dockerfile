# Use a base image with CUDA 12.1 and PyTorch support
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv python3-dev \
    git curl unzip ffmpeg \
    texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set the working directory
WORKDIR /app

# Copy the dependency file
COPY requirements.txt .

# Install dependencies, excluding nvidia-nccl-cu12 and triton if they cause issues
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Set the default command when the container starts
CMD ["/bin/bash"]

