# 1. Base Image: NVIDIA CUDA 11.8 with cuDNN 8 on Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 2. Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 3. Install system dependencies including Python 3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    && \
    rm -rf /var/lib/apt/lists/*

# 4. Set python3.8 as the default python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 5. Upgrade pip and install gdown
RUN pip install --upgrade pip && pip install gdown

# 6. Install a specific version of PyTorch compatible with CUDA 11.8
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 7. Set working directory
WORKDIR /app

# 8. Copy the requirements file
COPY ./LineVul/requirements.txt /app/requirements.txt

# 9. Create a requirements file for the Docker environment.
# Exclude torch (already installed) and nvidia packages (provided by base image).
# Downgrade packages for compatibility.
RUN grep -v 'torch' requirements.txt | grep -v 'nvidia' | \
    sed 's/matplotlib==3.9.0/matplotlib==3.5.3/' | \
    sed 's/contourpy==1.2.1/contourpy==1.0.7/' | \
    sed 's/scipy==1.14.0/scipy==1.10.1/' > requirements.docker.txt

# 10. Install the Python dependencies
RUN pip install -r requirements.docker.txt

# 11. Copy the rest of the application code
COPY ./LineVul /app/

# 12. Download the big-vul dataset into the image
RUN gdown "https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V" -O /app/data/big-vul_dataset/test.csv && \
    gdown "https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw" -O /app/data/big-vul_dataset/train.csv && \
    gdown "https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ" -O /app/data/big-vul_dataset/val.csv

# 13. Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]