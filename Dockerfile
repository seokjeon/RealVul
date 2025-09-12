# 참고: 이 Dockerfile은 LineVul 학습과 훈련을 위해 최적화 되어있습니다. 다른 모델 학습을 위해 사용하시려면 수정하십시오.

# 1. Base Image: NVIDIA CUDA 11.8 with cuDNN 8 on Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 2. Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 3. Install system dependencies and set up Python environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip

# 6. Install a specific version of PyTorch compatible with CUDA 11.8
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 7. Set working directory
WORKDIR /app

# 8. Create a requirements file for the linevul Docker environment.
# 아래의 세 패키지는 충돌을 피하기 위해 다운그레이드 됨
# matplotlib==3.5.3
# contourpy==1.0.7
# scipy==1.10.1
RUN <<EOF pip install -r /dev/stdin
matplotlib==3.5.3
contourpy==1.0.7
scipy==1.10.1
beautifulsoup4==4.12.3
captum==0.7.0
certifi==2024.7.4
charset-normalizer==3.3.2
cycler==0.12.1
filelock==3.15.4
fonttools==4.53.1
fsspec==2024.6.1
gdown==5.2.0
huggingface-hub==0.24.5
idna==3.7
joblib==1.4.2
kiwisolver==1.4.5
numpy==1.24.2
packaging==24.1
pandas==1.5.2
pillow==10.4.0
pyparsing==3.1.2
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
regex==2024.7.24
requests==2.32.3
scikit-learn==1.2.1
six==1.16.0
soupsieve==2.5
threadpoolctl==3.5.0
tokenizers==0.13.2
tqdm==4.66.5
transformers==4.26.0
typing_extensions==4.12.2
urllib3==2.2.2
EOF


# 9. Copy the rest of the application code
COPY ./LineVul /app/

# 10. Download the big-vul dataset, model into the image
RUN mkdir -p /app/LineVul/data/big-vul_dataset &&     mkdir -p /app/LineVul/linevul/saved_models/checkpoint-best-f1 &&     gdown "https://drive.google.com/uc?id=1h0iFJbc5DGXCXXvvR6dru_Dms_b2zW4V" -O /app/LineVul/data/big-vul_dataset/test.csv &&     gdown "https://drive.google.com/uc?id=1ldXyFvHG41VMrm260cK_JEPYqeb6e6Yw" -O /app/LineVul/data/big-vul_dataset/train.csv &&     gdown "https://drive.google.com/uc?id=1yggncqivMcP0tzbh8-8Eu02Edwcs44WZ" -O /app/LineVul/data/big-vul_dataset/val.csv &&     gdown "https://drive.google.com/uc?id=1oodyQqRb9jEcvLMVVKILmu8qHyNwd-zH" -O /app/LineVul/linevul/saved_models/checkpoint-best-f1/12heads_linevul_model.bin

# 11. Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]
