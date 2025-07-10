#!/bin/bash
#SBATCH --job-name=spec_worker
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=24:00:00

# Nama file log sekarang akan berisi Job ID, lebih sederhana dan pasti berfungsi.
# Kita akan tahu ini chunk berapa dari isi lognya.
#SBATCH --output=/home/bwalidain/birrulwldain/logs/worker_manual_%j.out
#SBATCH --error=/home/bwalidain/birrulwldain/logs/worker_manual_%j.err


### ======================================================================
### Menangkap Argumen dari Command Line
### ======================================================================
#
# Cek apakah argumen nomor chunk DAN argumen suffix file (misal: '2' dari combinations-2.json) diberikan
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Anda harus memberikan nomor chunk DAN suffix file JSON sebagai argumen." >&2
    echo "Contoh: sbatch job_manual.sh 1 2  (untuk combinations-2-1.json)" >&2
    exit 1
fi

# Ambil argumen pertama ($1) sebagai nomor chunk yang akan dijalankan
CHUNK_ID=$1
# Ambil argumen kedua ($2) sebagai suffix file dari nama file JSON asli (misal, '2' dari combinations-2.json)
FILE_SUFFIX=$2 

### ======================================================================
### Persiapan Lingkungan
### ======================================================================
echo "--- Memulai Pekerjaan Manual untuk Chunk ${CHUNK_ID} (File Suffix: ${FILE_SUFFIX}) ---"
echo "Job ID: $SLURM_JOB_ID"
date

# Mengaktifkan lingkungan Conda
source /home/bwalidain/miniconda3/etc/profile.d/conda.sh
conda activate bw

# Mengatur variabel lingkungan untuk optimasi
export MKL_VERBOSE=0
export OMP_NUM_THREADS=4

### ======================================================================
### Eksekusi Pekerjaan
### ======================================================================

# Pindah ke direktori kerja utama
WORK_DIR="/home/bwalidain/birrulwldain"
cd "${WORK_DIR}"
echo "Direktori kerja saat ini: $(pwd)"

# Tentukan file input dan output berdasarkan CHUNK_ID dan FILE_SUFFIX
# Nama file sekarang: combinations-{FILE_SUFFIX}-{CHUNK_ID}.json
INPUT_JSON="combinations-${FILE_SUFFIX}-${CHUNK_ID}.json"
OUTPUT_H5="dataset-${FILE_SUFFIX}-${CHUNK_ID}.h5" # Sesuaikan nama output H5 jika perlu

echo "File Input untuk task ini: ${INPUT_JSON}"
echo "File Output untuk task ini: ${OUTPUT_H5}"

# Menjalankan skrip worker Python
python /home/bwalidain/birrulwldain/job.py \
    --input-json "${INPUT_JSON}" \
    --output-h5 "${OUTPUT_H5}"

echo "--- Pekerjaan Manual Selesai ---"
date

conda activate bw
python job.py --input-json combinations-10-1.json --output-h5 dataset-10-1.h5