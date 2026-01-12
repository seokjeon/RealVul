#!/bin/bash
unset RAY_USE_MULTIPROCESSING_CPU_COUNT

enable_archive=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-archive)
            enable_archive=true
            shift
            ;;
        *)
            ARGUMENT=${1:-"all"}
            shift
            ;;
    esac
done

DS_NAME=$(dirname "$ARGUMENT")
PROJECT_NAME=$(basename "$ARGUMENT")

SLURM_TMPDIR="data/${ARGUMENT}"
if [ -d "$SLURM_TMPDIR" ]; then
    rm -rf "$SLURM_TMPDIR"/*
fi
mkdir -p "$SLURM_TMPDIR" || { echo "Failed to create $SLURM_TMPDIR" >&2; exit 1; }
cd $SLURM_TMPDIR

# Source code preparation (prepare.sh에서 다운로드한 파일을 압축 해제)
# docker-compose.yml에서 /data/dataset으로 마운트됨
if [ "$ARGUMENT" = "all" ]; then
    # 전체 데이터 (이미 압축 해제되어 있음)
    if [ ! -d "/data/dataset/all_source_code" ]; then
        echo "Extracting all_source_code..."
        tar -xf "/data/dataset/RealVul_Dataset-all_source_code.tar.xz" -C "/data/dataset/"
        mv "/data/dataset/source_code" "/data/dataset/all_source_code"
    fi
    ln -s "/data/dataset/all_source_code" "source_code"
else
    # 특정 프로젝트 데이터
    echo "Extracting ${PROJECT_NAME} source code..."
    project_src_tar_gz="/data/dataset/${DS_NAME}/${PROJECT_NAME}_source_code.tar.gz"
    
    # 압축 파일 찾기
    if [ -f "${project_src_tar_gz}" ]; then
        # 압축 해제
        tar -xf "$project_src_tar_gz" -C .
    else
        echo "Error: Cannot find source code archive for ${PROJECT_NAME}" >&2
        exit 1
    fi
    
    
    # 파일 확장자 변환 (.c로 통일)
    find source_code/ -type f -exec sh -c 'mv "$1" "${1%.*}.c"' _ {} \;
    
    # 데이터셋 CSV 파일 찾기 및 복사
    if [ -f "/data/dataset/${DS_NAME}/${PROJECT_NAME}_dataset.csv" ]; then
        cp "/data/dataset/${DS_NAME}/${PROJECT_NAME}_dataset.csv" "/data/dataset/${DS_NAME}-${PROJECT_NAME}_dataset.csv"
    fi
fi

#sGeneration of PDG
/tools/ReVeal/code-slicer/joern/joern-parse "./source_code" # TODO: joern-parse 경로 수정 필요

mkdir csv && find parsed/source_code/ -mindepth 1 -maxdepth 1 -type d | xargs -I{} mv {} csv/ # mv source_code/ ../csv && cd .. # root@22995bd65f6d:/code/models/DeepWukong/data/all# mv parsed csv
if [ "$enable_archive" = true ]; then
    tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/${DS_NAME}-${PROJECT_NAME}}_csv.tar.gz" csv # tar -zcvf "Dataset/${project_name}_csv.tar.gz" csv
# 압축 해제 시, tar --use-compress-program=pigz -xvf archive.tar.gz -C /path/to/dest
fi

#Generation of XFG
cd -
PROJECT_NAME="all" SLURM_TMPDIR="." python3 "data_generator.py" -c "./config/config.yaml"
if [ "$enable_archive" = true ]; then
    tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/XFG_${DS_NAME}-${PROJECT_NAME}.tar.gz" XFG # 원래 이건데 왼쪽으로 해봄 tar -zcf "/data/dataset/XFG_${project_name}.tar.gz" XFG
fi
#Symbolize and Split Dataset
python3 "preprocess/dataset_generator.py" -c "./config/config.yaml"
#cd $SLURM_TMPDIR
if [ "$enable_archive" = true ]; then
    tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/XFG_${DS_NAME}-${PROJECT_NAME}__processed.tar.gz" XFG # tar -zcf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG
fi
#cd -

#Word Embedding Pretraining
python3 "preprocess/word_embedding.py" -c "./config/config.yaml"

# #Training and Testing
# SLURM_TMPDIR="." python3 "run.py" -c "./config/config.yaml"