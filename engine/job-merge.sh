#!/bin/bash
#SBATCH --job-name=dataset_merger
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # 8 CPU sudah lebih dari cukup untuk I/O dan numpy
#SBATCH --mem=128G             # Alokasi RAM besar untuk menampung data gabungan
#SBATCH --time=04:00:00          # Batas waktu 4 jam

# Nama file log untuk pekerjaan merger
#SBATCH --output=/home/bwalidain/birrulwldain/logs/merger_%j.out
#SBATCH --error=/home/bwalidain/birrulwldain/logs/merger_%j.err

### ======================================================================
### Menangkap Argumen dari Command Line
### ======================================================================
# Cek apakah argumen suffix file diberikan saat menjalankan sbatch
if [ -z "$1" ]; then
    echo "Error: Anda harus memberikan suffix file JSON sebagai argumen (misal: '2' untuk dataset-2-*.h5)." >&2
    echo "Contoh: sbatch dataset_merger.sh 2" >&2
    exit 1
fi

# Ambil argumen pertama ($1) sebagai suffix file (misal, '2' dari dataset-2-*.h5)
FILE_SUFFIX=$1 

### ======================================================================
### Persiapan Lingkungan
### ======================================================================
echo "--- Memulai Pekerjaan Penggabungan (Merger) ---"
echo "Job ID: $SLURM_JOB_ID"
date

# Mengaktifkan lingkungan Conda
source /home/bwalidain/miniconda3/etc/profile.d/conda.sh
conda activate bw

### ======================================================================
### Eksekusi Pekerjaan
### ======================================================================

# Pindah ke direktori kerja tempat file-file .h5 parsial berada
WORK_DIR="/home/bwalidain/birrulwldain" # Sesuaikan dengan path Anda
cd "${WORK_DIR}"
echo "Direktori kerja saat ini: $(pwd)"

# Tentukan pola nama file input. Ini adalah HANYA NAMA FILE, bukan path lengkap.
# Contoh: "dataset-10-*.h5"
FILE_NAME_PATTERN="dataset-${FILE_SUFFIX}-*.h5"

# Tentukan path output file H5 gabungan
# Pastikan ini adalah path lengkap ke mana Anda ingin file akhir disimpan.
OUTPUT_H5="${WORK_DIR}/dataset-${FILE_SUFFIX}.h5" # Menggunakan WORK_DIR untuk path lengkap

echo "Pola Nama File Input yang dicari: ${FILE_NAME_PATTERN}"
echo "File Output gabungan: ${OUTPUT_H5}"

# Temukan semua file yang cocok di direktori kerja saat ini.
# Menggunakan find dengan jalur lengkap dari WORK_DIR untuk keandalan.
# `find "${WORK_DIR}" -maxdepth 1 -name "${FILE_NAME_PATTERN}" -print0`
#     - find di WORK_DIR
#     - maxdepth 1: hanya direktori ini, bukan subdirektori
#     - name "${FILE_NAME_PATTERN}": cari file dengan pola nama ini
#     - print0: pisahkan hasil dengan null character, aman untuk nama file dengan spasi/karakter khusus
# `xargs -0 echo`: Mengambil daftar file yang dipisahkan null dan mencetaknya, dipisahkan spasi.
# Hasilnya adalah string "path/to/file1.h5 path/to/file2.h5 ..."
FOUND_FILES=$(find "${WORK_DIR}" -maxdepth 1 -name "${FILE_NAME_PATTERN}" -print0 | xargs -0 echo)

# Cek apakah ada file yang ditemukan
if [ -z "$FOUND_FILES" ]; then
    echo "ERROR: Tidak ada file yang cocok dengan pola ${FILE_NAME_PATTERN} di ${WORK_DIR}. Mohon periksa." >&2
    exit 1
fi
echo "File-file yang ditemukan: ${FOUND_FILES}"

# Menjalankan skrip merger Python
# Argumen --inputs harus menerima daftar path file yang dipisahkan spasi.
# Bash akan memperluas $FOUND_FILES menjadi argumen terpisah untuk --inputs.
python /home/bwalidain/birrulwldain/merge.py \
    --inputs ${FOUND_FILES} \
    --output "${OUTPUT_H5}"

echo "--- Pekerjaan Penggabungan Selesai ---"
date

python merge.py --inputs dataset-200-*.h5 --output dataset-200.h5