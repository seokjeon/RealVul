# RealVul

## 0. RealVul 준비
0. git clone --recursive https://github.com/seokjeon/RealVul
1. [RealVul 공식 repository](https://zenodo.org/records/12707476)에서 `Replication Package.zip`를 다운로드 합니다.
2. 압축 파일 내 Dataset 폴더를 RealVul 폴더로 옮깁니다.

## 1. LineVul 재현 방법

**요구사항:**
- NVIDIA GPU
- NVIDIA 드라이버
- Docker

**단계별 안내:**

0.  **서브모듈 초기화**
    서브모듈 LineVul 하위 모듈을 초기화하고 데이터를 가져옵니다.
    ```sh
    git submodule update --init --recursive
    ``` 

1.  **Docker 이미지 빌드**

    프로젝트 루트 디렉터리에서 다음 명령어를 실행하여 Docker 이미지를 빌드합니다. 
    이 과정은 의존성 패키지와 `big-vul` 데이터셋을 모두 이미지에 포함시키며, 최초 빌드 시 시간이 다소 소요될 수 있습니다.

    ```sh
    docker build --no-cache -t linevul-env .
    ```

2.  **Docker 컨테이너 실행**

    빌드된 이미지로 컨테이너를 생성하고 백그라운드에서 실행합니다.

    - `-d`: 백그라운드 실행
    - `--gpus all`: 호스트의 모든 NVIDIA GPU를 컨테이너에 할당
    - `--name linevul-container`: 컨테이너의 이름을 `linevul-container`로 지정
    - `-v .:/app`: 호스트의 현재 프로젝트 폴더를 컨테이너의 `/app` 폴더와 동기화 (학습된 모델이나 로그를 호스트에서 바로 확인하기 위함)

    ```sh
    docker run -d --gpus all --name linevul-container -v .:/app linevul-env
    ```

3.  **모델 학습 & 테스트 실행**

    `docker exec`를 사용하여 실행 중인 컨테이너 내부에서 `linevul_main.py` 스크립트를 실행합니다.

    - `-w /app/LineVul/linevul`: 컨테이너 내부의 작업 디렉터리를 지정합니다.
    - `2>&1 | tee train.log`: 학습 과정의 모든 출력(표준 출력 및 오류)을 터미널과 `LineVul/linevul/train.log` 파일 양쪽에 기록합니다.

    > **참고:** 아래 명령어의 `--epochs` 값은 논문과 동일한 10으로 설정되어 있습니다. 필요에 맞게 조절하십시오.
    > **참고:** 도커 빌드 도중 gdown 차단으로 컨테이너에 데이터셋과 모델이 없을수 있습니다. 확인후 학습을 수행하십시오. 

    **host os Linux**
    ```sh
    # 1. 학습 + 테스트
    docker exec -w /app/LineVul/linevul linevul-container python linevul_main.py \
      --output_dir ./saved_models \
      --model_type roberta \
      --tokenizer_name microsoft/codebert-base \
      --model_name_or_path microsoft/codebert-base \
      --do_train \
      --do_test \
      --train_data_file ../data/big-vul_dataset/train.csv \
      --eval_data_file ../data/big-vul_dataset/val.csv \
      --test_data_file ../data/big-vul_dataset/test.csv \
      --epochs 10 \
      --block_size 512 \
      --train_batch_size 16 \
      --eval_batch_size 16 \
      --learning_rate 2e-5 \
      --max_grad_norm 1.0 \
      --evaluate_during_training \
      --seed 123456 2>&1 | tee train.log

    # 2. 테스트
    docker exec -w /app/LineVul/linevul linevul-container python linevul_main.py \
    --model_name=12heads_linevul_model.bin \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=../data/big-vul_dataset/train.csv \
    --eval_data_file=../data/big-vul_dataset/val.csv \
    --test_data_file=../data/big-vul_dataset/test.csv \
    --block_size 512 \
    --eval_batch_size 512 2>&1 | tee test.log
    ```

    **host os Windows**
    ```sh
    # 1. 학습 + 테스트
    docker exec -w /app/LineVul/linevul linevul-container python linevul_main.py `
    --output_dir ./saved_models `
    --model_type roberta `
    --tokenizer_name microsoft/codebert-base `
    --model_name_or_path microsoft/codebert-base `
    --do_train `
    --do_test `
    --train_data_file ../data/big-vul_dataset/train.csv `
    --eval_data_file ../data/big-vul_dataset/val.csv `
    --test_data_file ../data/big-vul_dataset/test.csv `
    --epochs 10 `
    --block_size 512 `
    --train_batch_size 16 `
    --eval_batch_size 16 `
    --learning_rate 2e-5 `
    --max_grad_norm 1.0 `
    --evaluate_during_training `
    --seed 123456 *>&1 | Tee-Object -FilePath train.log

    # 2. 테스트
    docker exec -w /app/LineVul/linevul linevul-container python linevul_main.py `
    --model_name=12heads_linevul_model.bin `
    --output_dir=./saved_models `
    --model_type=roberta `
    --tokenizer_name=microsoft/codebert-base `
    --model_name_or_path=microsoft/codebert-base `
    --do_test `
    --train_data_file=../data/big-vul_dataset/train.csv `
    --eval_data_file=../data/big-vul_dataset/val.csv `
    --test_data_file=../data/big-vul_dataset/test.csv `
    --block_size 512 `
    --eval_batch_size 512 *>&1 | Tee-Object -FilePath test.log
    ```

4.  **학습/테스트 과정 모니터링**

    호스트 머신에서 다음 명령어를 실행하여 `train.log`, `test.log` 파일들을 실시간으로 확인하며 진행 상황을 모니터링할 수 있습니다.

    ```sh
    tail -f ./LineVul/linevul/train.log
    tail -f ./LineVul/linevul/test.log
    ```


## DeepWukong 재현
1. mkdir Experiments/DeepWukong/data
2. 도커 이미지를 생성한다.
    - GPU 사용 시: `docker-compose build deepwukong`
    - CPU 사용 시: `docker-compose build deepwukong_without_gpu`

### pretrained_model로 deepwukong 데이터셋(SARD-CWE119) 평가
0. deepwukong의 실험을 재현하기 위해 [data](https://github.com/jumormt/DeepWukong?tab=readme-ov-file#setup)와 [pretrained_model](https://github.com/jumormt/DeepWukong?tab=readme-ov-file#one-step-evaluation)를 다운로드 받아 압축 해제 후 Experiments/DeepWukong/data로 옮긴다. `7z x Data.7z -o/code/models/DeepWukong/data/`
1. 도커 컨테이너를 실행한다.
    - GPU 사용 시: `docker-compose up deepwukong -d`
    - CPU 사용 시: `docker-compose up deepwukong_without_gpu -d`
2. Experiments/DeepWukong/config/config.yaml에서 `split_folder_name`을 `CWE119`로 변경
3. `PYTORCH_JIT=0 SLURM_TMPDIR=. python evaluate.py ./data/DeepWukong --root_folder_path ./data --split_folder_name CWE119`

### pretrained_model로 realvul 평가
0. 도커 컨테이너를 실행한다.
    - GPU 사용 시: `docker-compose up deepwukong -d`
    - CPU 사용 시: `docker-compose up deepwukong_without_gpu -d`
0. 
1. Experiments/DeepWukong/config/config.yaml에서 `csv_data_path`을 `/data/dataset/all_csv.tar.gz`로 변경
2. Experiments/DeepWukong/deepwukong_pipeline.sh의 `project_name`을 `all`로 변경, `SLURM_TMPDIR`을 `/code/models/DeepWukong/data/realvul`으로 변경
3. mkdir -p /code/models/DeepWukong/data/

### 그 외 다른 소프트웨어 테스트 시
1. Experiments/DeepWukong/deepwukong_pipeline.sh의 `tar -xf "/data/dataset/${project_name}_source_code.tar.xz" -C $SLURM_TMPDIR`를 `tar -xf "/data/dataset/${project_name}_source_code.tar.gz" -C $SLURM_TMPDIR`로 변경해야 함. all_source_code 만 tar.xz고 나머지는 gz임.