#!/bin/bash

project_name="all"
SLURM_TMPDIR="/code/models/DeepWukong/data/realvul"
cd $SLURM_TMPDIR

#source_code extraction
tar -xf "/data/dataset/${project_name}_source_code.tar.xz" -C $SLURM_TMPDIR
mv "SLURM_TMPDIR/${project_name}_source_code" "SLURM_TMPDIR/source_code"

#sGeneration of PDG
/tools/ReVeal/code-slicer/joern/joern-parse source_code
mv parsed csv
tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/${project_name}_csv.tar.gz" csv # tar -zcvf "/data/dataset/${project_name}_csv.tar.gz" csv

#Generation of XFG
python "/code/models/DeepWukong/data_generator.py" -c "/config/config.yaml"
cd $SLURM_TMPDIR
tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/XFG_${project_name}.tar.gz" XFG # 원래 이건데 왼쪽으로 해봄 tar -zcf "/data/dataset/XFG_${project_name}.tar.gz" XFG

#Symbolize and Split Dataset
python "/code/models/DeepWukong/preprocess/dataset_generator.py" -c "/config/config.yaml"
cd $SLURM_TMPDIR
tar -I 'pigz -p $(nproc) -6' -cf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG # tar -zcf "/data/dataset/XFG_${project_name}__processed.tar.gz" XFG

#Word Embedding Pretraining
python "/code/models/DeepWukong/preprocess/word_embedding.py" -c "/config/config.yaml"

#Training and Testing
python "/code/models/DeepWukong/run.py" -c "/config/config.yaml"

