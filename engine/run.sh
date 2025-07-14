#!/bin/bash

#SBATCH --job-name=inflibstrain
#SBATCH --partition=medium
#SBATCH --nodelist=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=44G
#SBATCH --time=48:00:00
#SBATCH --out=/home/bwalidain/birrulwldain/logs/train-%j.out
#SBATCH --err=/home/bwalidain/birrulwldain/logs/train-%j.err

source "/home/bwalidain/miniconda3/etc/profile.d/conda.sh"
conda activate rapids-25.06


python "/home/bwalidain/birrulwldain/train.py" \
    --dataset_path "/home/bwalidain/birrulwldain/data/dataset-50.h5" \
    --element_map_path "/home/bwalidain/birrulwldain/data/element-map-18a.json" \
    --model_dir "/home/bwalidain/birrulwldain/models" \
    --results_dir "/home/bwalidain/birrulwldain/logs" \
    --epochs 100 \
    --batch_size 26 \
    --lr 1e-4 #\
############    --resume-from "/home/bwalidain/birrulwldain/models/informer_multilabel_model.pth"
