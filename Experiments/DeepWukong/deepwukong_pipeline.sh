#!/bin/bash
unset RAY_USE_MULTIPROCESSING_CPU_COUNT
project_name="all"
SLURM_TMPDIR="data/${project_name}"
if [ ! -d "$SLURM_TMPDIR" ]; then
    mkdir -p "$SLURM_TMPDIR" || { echo "Failed to create $SLURM_TMPDIR" >&2; exit 1; }
fi
cd $SLURM_TMPDIR

#source_code extraction
ln -s "/data/dataset/all_source_code" "source_code" # TODO: 다른 SW 테스트 할때는 tar -xf "Dataset/${project_name}_source_code.tar.gz" -C $SLURM_TMPDIR

#sGeneration of PDG
/tools/ReVeal/code-slicer/joern/joern-parse "./source_code" # TODO: joern-parse 경로 수정 필요

mkdir csv && find parsed/source_code/ -mindepth 1 -maxdepth 1 -type d | xargs -I{} mv {} csv/ # mv source_code/ ../csv && cd .. # root@22995bd65f6d:/code/models/DeepWukong/data/all# mv parsed csv
# tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/my${project_name}_csv.tar.gz" csv # tar -zcvf "Dataset/${project_name}_csv.tar.gz" csv
# 압축 해제 시, tar --use-compress-program=pigz -xvf archive.tar.gz -C /path/to/dest

#Generation of XFG
cd ../../
project_name="all" SLURM_TMPDIR="." python "data_generator.py" -c "./config/config.yaml"
#tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/myXFG_${project_name}.tar.gz" XFG # 원래 이건데 왼쪽으로 해봄 tar -zcf "/data/dataset/XFG_${project_name}.tar.gz" XFG

#Symbolize and Split Dataset
python "preprocess/dataset_generator.py" -c "./config/config.yaml"
#cd $SLURM_TMPDIR
#tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/myXFG_${project_name}__processed.tar.gz" XFG # tar -zcf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG
#cd -

#Word Embedding Pretraining
python "preprocess/word_embedding.py" -c "./config/config.yaml"

#Training and Testing
SLURM_TMPDIR="." python "run.py" -c "./config/config.yaml"

