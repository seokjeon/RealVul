# 참고: 이 Dockerfile은 LineVul 학습과 훈련을 위해 최적화 되어있습니다. 다른 모델 학습을 위해 사용하시려면 수정하십시오.

# 1. Base Image: NVIDIA CUDA 11.8 with cuDNN 8 on Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 2. Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# ENV SLURM_TMPDIR=/tmp/slurm_job

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

# 4. pip install for the linevul Docker environment.
# 아래의 세 패키지는 충돌을 피하기 위해 다운그레이드 됨
# matplotlib==3.5.3
# contourpy==1.0.7
# scipy==1.10.1
RUN <<EOF pip install -r /dev/stdin
#  pip에게 CUDA 11.8용으로 컴파일된 PyTorch 관련 패키지들은 기본 저장소가 아닌 이 주소에서 찾으라고 지시, 그외의 것은 PyPI 참조
--index-url https://download.pytorch.org/whl/cu118
--extra-index-url https://pypi.org/simple
# CUDA 11.8용 PyTorch 스택
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2

# 데이터/수치 스택(파이썬 3.8~3.11 권장)
numpy==1.24.2
scipy==1.10.1
pandas==1.5.2
scikit-learn==1.2.1

# 시각화
matplotlib==3.5.3
contourpy==1.0.7
kiwisolver==1.4.5
cycler==0.12.1
fonttools==4.53.1
pyparsing==3.1.2
pillow==10.4.0

# HF & NLP (구버전 트랜스포머 유지 시 허브 다운그레이드 권고)
transformers==4.26.0
tokenizers==0.13.2
huggingface-hub==0.13.4

# 기타
beautifulsoup4==4.12.3
soupsieve==2.5
regex==2024.7.24
tqdm==4.66.5
requests==2.32.3
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2024.1
filelock==3.15.4
fsspec==2024.6.1
gdown==5.2.0
idna==3.7
joblib==1.4.2
packaging==24.1
certifi==2024.7.4
six==1.16.0
PyYAML==6.0.1
typing_extensions==4.12.2
threadpoolctl==3.5.0

# 분산
ray==2.9.3
pydriller==2.5
omegaconf==2.3.0
EOF


# 5. Set directory, file
WORKDIR /app
RUN mkdir -p /app/Experiments/LineVul/best_model
# mkdir -p /data/project_files && \
# mkdir -p $SLURM_TMPDIR # Real_Vul 데이터셋이 주어지므로 사용하지 않음
COPY . /app
RUN cp /app/LineVul/linevul/saved_models/checkpoint-best-f1/12heads_linevul_model.bin /app/Experiments/LineVul/best_model/ && \
    cp /app/Experiments/LineVul/best_model/12heads_linevul_model.bin /app/Experiments/LineVul/best_model/pytorch_model.bin


# HuggingFace 에서 기본 설정 config.json 로드
RUN python - <<'PY'
from transformers import RobertaConfig
config = RobertaConfig.from_pretrained("microsoft/codebert-base")
config.num_labels = 2              # Line-level classification
config.save_pretrained("/app/Experiments/LineVul/best_model")
PY

# 6. Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]
