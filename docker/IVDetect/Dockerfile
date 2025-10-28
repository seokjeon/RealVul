# Docker image for reproducing the VulnerabilityDetectionResearch/IVDetect experiments
#
# This image provides:
#   * CUDA-enabled PyTorch 1.9 stack with torch-geometric 2.0.1
#   * System build tooling for the GloVe embeddings and Joern graph extraction
#   * A ready-to-use checkout of the IVDetect code under /app/IVDetect

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# 1. System packages and Python 3.8 toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3-pip \
        git \
        wget \
        curl \
        ca-certificates \
        unzip \
        build-essential \
        libffi-dev \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev \
        openjdk-11-jre-headless \
        graphviz \
        libgraphviz-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --no-cache-dir --upgrade pip

# 2. CUDA 11.1 compatible PyTorch 1.9 stack and torch-geometric dependencies
RUN pip install --no-cache-dir \
        # IVDetect 레포에서 제공된 데이터셋은 CUDA 11.0 기반 PyTorch 1.7.1 버전에서 전처리되어있습니다.
        torch==1.7.1+cu110 \
        torchvision==0.8.2+cu110 \
        torchaudio==0.7.2 \
        -f https://download.pytorch.org/whl/torch_stable.html

        # IVDetect 레포에서 requirements.txt에 명시된 버전은 CUDA 11.1 기반 PyTorch 1.9.0 버전입니다.
        # torch==1.9.0+cu111 \
        # torchvision==0.10.0+cu111 \
        # torchaudio==0.9.0 \
        # --extra-index-url https://download.pytorch.org/whl/cu111

RUN pip install --no-cache-dir \
        # IVDetect 레포에서 제공된 데이터셋에 맞춘 torch 의존성 버전
        torch-scatter==2.0.7 \
        torch-sparse==0.6.10 \
        torch-cluster==1.5.9 \
        torch-spline-conv==1.2.1 \
        torch-geometric==1.7.2 \
        -f https://data.pyg.org/whl/torch-1.7.0+cu110.html && \
        
        # IVDetect 레포에서 PyTorch 1.9.0 기반 torch 의존성 버전
        # torch-scatter==2.0.8 \
        # torch-sparse==0.6.12 \
        # torch-cluster==1.5.9 \
        # torch-spline-conv==1.2.1 \
        # -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install --no-cache-dir torch-geometric==2.0.1

# 3. Remaining Python user-land dependencies needed by IVDetect
RUN pip install --no-cache-dir \
        numpy==1.21.6 \
        scipy==1.7.3 \
        pandas==0.25.2 \
        scikit-learn==0.24.2 \
        gensim==4.3.2 \
        tqdm==4.66.5 \
        sentencepiece==0.2.0 \
        networkx==2.5.1 \
        tensorboard==2.11.2 \
        matplotlib==3.5.3 \
        PyYAML==6.0.1 \
        nni==2.10 \
        pygraphviz==1.11

# 4. Install Joern (legacy version aligned with the paper instructions)
ARG JOERN_VERSION=1.1.249
RUN mkdir -p /opt && \
    curl -L "https://github.com/joernio/joern/releases/download/v${JOERN_VERSION}/joern-cli.zip" -o /tmp/joern.zip && \
    unzip /tmp/joern.zip -d /opt && \
    mv /opt/joern-cli /opt/joern-cli-${JOERN_VERSION} && \
    rm /tmp/joern.zip && \
    ln -s /opt/joern-cli-${JOERN_VERSION}/joern /usr/local/bin/joern && \
    ln -s /opt/joern-cli-${JOERN_VERSION}/joern-parse /usr/local/bin/joern-parse && \
    ln -s /opt/joern-cli-${JOERN_VERSION}/joern-export /usr/local/bin/joern-export && \
    ln -s /opt/joern-cli-${JOERN_VERSION}/joern-flow /usr/local/bin/joern-flow

# 5. Copy IVDetect sources into the image
WORKDIR /app
COPY VulnerabilityDetectionResearch/IVDetect /app/IVDetect

# Pre-create experiment output directories commonly referenced by the scripts
RUN mkdir -p /app/IVDetect/data/pyg_graph \
    /app/IVDetect/data/test_graph \
    /app/IVDetect/data/train_graph \
    /app/IVDetect/data/valid_graph \
    /app/IVDetect/result \
    /app/IVDetect/model

ENV PYTHONPATH=/app/IVDetect \
    PATH=/app/IVDetect/glove:$PATH

WORKDIR /app/IVDetect

# Default command keeps the container alive for interactive experimentation
CMD ["bash"]
