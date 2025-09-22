# PyTorch 1.9.0과 관련 패키지들을 위한 Dockerfile (GPU 지원)
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-l", "-c"]

# 시스템 패키지 업데이트 및 필요한 의존성 설치
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    p7zip-full unzip zip graphviz graphviz-dev pkg-config pigz\
    && rm -rf /var/lib/apt/lists/*

# Python 3.8을 기본으로 설정
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Python 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# pip 업그레이드 및 패키지 설치
RUN pip install --no-cache-dir --upgrade pip

# PyTorch 관련 패키지들을 먼저 설치 (GPU 버전, 의존성 순서 고려)
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==0.9.0 --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch-geometric==2.3.1
# torch-scatter와 torch-sparse 설치 (torch-geometric 의존성)
RUN pip install --no-cache-dir torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
# 나머지 패키지들 설치
RUN pip install --no-cache-dir \
    black==21.6b0 \
    mypy==0.910 \
    pytest==6.2.4 \
    dpu-utils \
    datasets==3.1.0 \
    wordninja==2.0.0 \
    gensim==3.8.3 \
    networkx==2.5.1 \
    pytorch-lightning==1.9.5 \
    torchmetrics==0.11.4 \
    commode-utils==0.3.1 \
    tqdm==4.67.1 \
    omegaconf==2.2.0 \
    jupyter \
    setuptools==59.5.0 ray clang gdown libclang mpmath nltk openpyxl safetensors sympy transformers pygraphviz

WORKDIR /tmp

RUN curl -s "https://get.sdkman.io" | bash && echo 'source ~/.sdkman/bin/sdkman-init.sh' >> ~/.profile
RUN sdk install gradle 4.10.3 && sdk install java 11.0.25-ms

## 각종 도구(Joern) 설치
WORKDIR /tools

RUN git clone https://github.com/VulDetProject/ReVeal.git \
	&& cd ./ReVeal/code-slicer/joern \
	&& ./build.sh

# 기본 명령어 설정 - 컨테이너가 계속 실행되도록
CMD ["tail", "-f", "/dev/null"]
