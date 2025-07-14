#!/bin/bash

#SBATCH --job-name=inflibseval
#SBATCH --partition=short
#SBATCH --nodelist=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=44G
#SBATCH --time=01:00:00          # Batas waktu 1 jam untuk evaluasi

#SBATCH --out=/home/bwalidain/birrulwldain/logs/eval-%j.out


source "/home/bwalidain/miniconda3/etc/profile.d/conda.sh"
conda activate rapids-25.06

# --- KONFIGURASI PATH (Sesuaikan jika perlu) ---
BASE_DIR="/home/bwalidain/birrulwldain"
MODEL_PATH="${BASE_DIR}/models/informer_multilabel_model-3.pth"
DATASET_PATH="${BASE_DIR}/data/dataset-50.h5"
ELEMENT_MAP_PATH="${BASE_DIR}/data/element-map-18a.json"
RESULTS_DIR="${BASE_DIR}/results"

# --- PILIH MODE EKSEKUSI ---
# Hapus komentar pada salah satu baris di bawah ini untuk menjalankan mode yang diinginkan.

--- Contoh 1: Menjalankan evaluasi kuantitatif pada split 'test' ---
python "${BASE_DIR}/eval.py" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --element_map_path "${ELEMENT_MAP_PATH}" \
    --results_dir "${RESULTS_DIR}" \
    --mode eval \
    --split test \
    --threshold 0.5

# --- Contoh 2: Menganalisis dan membuat plot untuk satu sampel dari file H5 ---
# python "${BASE_DIR}/eval.py" \
#     --model_path "${MODEL_PATH}" \
#     --dataset_path "${DATASET_PATH}" \
#     --element_map_path "${ELEMENT_MAP_PATH}" \
#     --results_dir "${RESULTS_DIR}" \
#     --mode analyze_h5 \
#     --split validation \
#     --sample_idx 150

# --- Contoh 3: Menganalisis data lapangan dari file .asc ---
# ASC_FILE_PATH="${BASE_DIR}/data/asc/nama_file.asc" # Ganti dengan path file .asc Anda
# python "${BASE_DIR}/eval.py" \
#     --model_path "${MODEL_PATH}" \
#     --dataset_path "${DATASET_PATH}" \
#     --element_map_path "${ELEMENT_MAP_PATH}" \
#     --results_dir "${RESULTS_DIR}" \
#     --mode analyze_asc \
#     --asc_path "${ASC_FILE_PATH}"
