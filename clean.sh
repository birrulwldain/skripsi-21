#!/bin/bash

# Skrip pembersih file temporary LaTeX
# Simpan dengan nama clean_tex.sh

rm -f *.aux \
    *.log \
    *.lof \
    *.lot \
    *.out \
    *.blg \
    *.bbl \
    *.bcf \
    *.apc \
    *.toc \
    *.synctex.gz \
    *.run.xml

echo "Semua file temporary LaTeX telah dihapus!"