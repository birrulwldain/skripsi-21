#!/bin/bash

# Skrip pembersih file temporary LaTeX dengan konfirmasi
# Simpan dengan nama clean_tex.sh

# Mencari file temporary LaTeX secara rekursif
files=$(find . -type f \( \
    -name "*.aux" -o \
    -name "*.log" -o \
    -name "*.lof" -o \
    -name "*.lot" -o \
    -name "*.out" -o \
    -name "*.blg" -o \
    -name "*.bbl" -o \
    -name "*.bcf" -o \
    -name "*.apc" -o \
    -name "*.toc" -o \
    -name "*.synctex.gz" -o \
    -name "*.run.xml" \
\))


# Minta konfirmasi
# Hapus file
echo "$files" | xargs rm -f
echo "Semua file temporary LaTeX di direktori saat ini dan subfolder telah dihapus!"
