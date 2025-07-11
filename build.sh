pdflatex -interaction=nonstopmode --shell-escape 1proposal.tex;
biber 1proposal.aux ;
pdflatex -interaction=nonstopmode --shell-escape 1proposal.tex;
pdflatex -interaction=nonstopmode --shell-escape 1proposal.tex;

# Uncommnet bagian ini jika inigin mengkompile file seminar-hasil
pdflatex -interaction=nonstopmode --shell-escape 2hasil.tex;
biber 2hasil.aux ;
pdflatex -interaction=nonstopmode --shell-escape 2hasil.tex;
pdflatex -interaction=nonstopmode --shell-escape 2hasil.tex;

# Uncomment bagian ini jika ingin mencompile file skripsi
pdflatex -interaction=nonstopmode --shell-escape 3skripsi.tex;
biber 3skripsi.aux ;
pdflatex -interaction=nonstopmode --shell-escape 3skripsi.tex;
pdflatex -interaction=nonstopmode --shell-escape 3skripsi.tex;



#!/bin/bash

# Daftar DOI yang valid dari referensi
dois=(
    "https://doi.org/10.1063/1.2832689"
    "https://doi.org/10.1109/72.279181"
    "https://doi.org/10.1016/j.neunet.2012.09.005"
    "https://doi.org/10.1039/D3DD00198A"
    "https://doi.org/10.1038/s41467-025-56481-x"
    "https://doi.org/10.48550/arXiv.1406.1078"
    "https://doi.org/10.48550/arXiv.1511.07289"
    "https://doi.org/10.1016/j.sab.2021.106113"
    "https://doi.org/10.1016/j.sab.2014.06.006"
    "https://doi.org/10.18653/v1/N19-1423"
    "https://doi.org/10.1016/j.geodrs.2019.e00308"
    "https://doi.org/10.1007/s11467-020-1007-z"
    "https://doi.org/10.1016/j.infrared.2024.105384"
    "https://doi.org/10.1016/j.sab.2022.106576"
    "https://doi.org/10.1063/5.0206788"
    "https://doi.org/10.1016/j.sab.2023.106695"
    "https://doi.org/10.1109/ICASSP.2013.6639067"
    "https://doi.org/10.1016/j.apgeochem.2021.104929"
    "https://doi.org/10.1016/j.sab.2021.106242"
    "https://doi.org/10.1039/D1JA00124J"
    "https://doi.org/10.1162/neco.1997.9.8.1735"
    "https://doi.org/10.1177/0003702820920635"
    "https://doi.org/10.1016/S0584-8547(99)00136-6"
    "https://dx.doi.org/10.18434/T4W30F"
    "https://doi.org/10.1021/acs.analchem.0c04545"
    "https://doi.org/10.1145/3664647.3681059"
    "https://doi.org/10.48550/arXiv.2103.14030"
    "https://doi.org/10.1119/1.4913497"
    "https://doi.org/10.1002/9780470027318.a9021.pub3"
    "https://doi.org/10.1080/05704928.2022.2085297"
    "https://doi.org/10.1177/0003702818783307"
    "https://doi.org/10.1016/j.trac.2021.116476"
    "https://doi.org/10.1016/j.sab.2020.105897"
    "https://doi.org/10.1016/0040-9383(95)00022-4"
    "https://doi.org/10.1016/0584-8547(88)80002-0"
    "https://doi.org/10.1162/jocn.1991.3.1.71"
    "https://doi.org/10.48550/arXiv.1706.03762"
    "https://doi.org/10.1002/andp.19293940503"
    "https://doi.org/10.1021/acs.jcim.4c00378"
    "https://doi.org/10.1016/j.talanta.2025.127937"
    "https://doi.org/10.1109/ACCESS.2019.2936508"
    "https://doi.org/10.1016/j.fmre.2021.05.005"
)

# Perintah untuk membuka link (sesuaikan dengan sistem operasi)
# macOS: gunakan 'open'
# Linux: gunakan 'xdg-open'
# Windows: gunakan 'start' (uncomment baris di bawah untuk Windows)
OPEN_CMD="open"  # macOS
# OPEN_CMD="xdg-open"  # Linux
# OPEN_CMD="start"  # Windows

# Loop untuk membuka setiap DOI di browser
for doi in "${dois[@]}"; do
    $OPEN_CMD "$doi" &
done

echo "Semua link DOI dibuka di browser."