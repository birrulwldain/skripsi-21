# -*- coding: utf-8 -*-
from pylatex import Document, Section, Command, NoEscape, Package
from pylatex.utils import italic
import os

# --- Konfigurasi Dokumen ---
output_path = "generated_proposal"

# Informasi Dokumen (disesuaikan dengan nama variabel di .cls)
info = {
    "judulcover": "Analisis Prediktif Spektrum Emisi Laser-Induced Breakdown Spectroscopy (LIBS) Multi-Elemen Berbasis Simulasi Sintetis dengan Informer",
    "judul": "Analisis Prediktif Spektrum Emisi Laser-Induced Breakdown Spectroscopy (LIBS) Multi-Elemen Berbasis Simulasi Sintetis dengan Informer",
    "judulinggris": "Predictive Analysis of Multi-Element Laser-Induced Breakdown Spectroscopy (LIBS) Emission Spectra Based on Synthetic Simulation with Informer",
    "fullname": "Birrul Walidain",
    "idnum": "2008102010010",
    "degree": "Sarjana Sains",
    "yearsubmit": "Juli, 2025",
    "dept": "Fisika",
    "firstsupervisor": r"Prof. Dr. Eng. Nasrullah, S.Si., M.T.",
    "firstnip": "197607031995121001",
    "secondsupervisor": r"Dr. Khairun Saddami, S.T.",
    "secondnip": "199103182022031008",
    "kajur": r"Dr. Saumi Syahreza, S.Si., M.Si.",
    "kajurnip": "197609172005011002",
    "dekan": r"Prof. Dr. Taufik Fuadi Abidin, S.Si., M.Tech.",
    "dekannip": "197010081994031002",
    "kaprodi": r"Dr. Saumi Syahreza, S.Si., M.Si.",
    "kaprodinip": "197609172005011002",
    "approval_date": "5 Juli 2025"
}

# --- Inisialisasi Dokumen PyLaTeX ---
# Menggunakan class 1proposal dan opsi proposal
doc = Document(documentclass=NoEscape('./lib/1proposal'),
               document_options=['proposal', 'dvipsnames'], 
               fontenc='T1',
               inputenc='utf8',
               lmodern=False)

# --- PREAMBLE SETUP ---

# Menambahkan path ke folder lib/, language/, dan include/
for path in ['lib/', 'language/', 'include/']:
    doc.preamble.append(NoEscape(r'\makeatletter'))
    doc.preamble.append(NoEscape(r'\def\input@path{{./' + path + r'}}'))
    doc.preamble.append(NoEscape(r'\makeatother'))

# Menambahkan package yang dibutuhkan
doc.packages.append(Package('biblatex', options=['style=apa', 'backend=biber', 'natbib=true']))
doc.packages.append(Package('comment'))
# (Paket-paket lain dari .cls sudah otomatis dimuat, tidak perlu didaftarkan ulang)
# Namun, untuk kejelasan, kita tetap bisa menambahkan yang penting
doc.packages.append(Package('setspace')) 

# Menambahkan resource bibliography
doc.preamble.append(NoEscape(r'\addbibresource{daftar-pustaka.bib}'))

# [FIX] Mendefinisikan variabel LaTeX dengan nama yang BENAR sesuai .cls
for key, value in info.items():
    # Khusus judulinggris, menggunakan italic
    if key == 'judulinggris':
        doc.preamble.append(Command(key, NoEscape(italic(value))))
    else:
        doc.preamble.append(Command(key, value))

# --- BAGIAN ISI DOKUMEN ---

# [FIX] Menggunakan perintah \cover dan \approvalpage yang benar dari .cls
doc.append(NoEscape(r'\cover'))
doc.append(NoEscape(r'\approvalpage'))

# Daftar Isi, dll. akan dibuat secara otomatis oleh kelas 'report'
# yang menjadi dasar .cls Anda
doc.append(NoEscape(r'\tableofcontents'))
doc.append(NoEscape(r'\listoffigures'))
doc.append(NoEscape(r'\listoftables'))

# Kembali ke penomoran Arab untuk Bab
doc.append(NoEscape(r'\clearpage'))
doc.append(NoEscape(r'\pagenumbering{arabic}'))

# Meng-include file-file bab
doc.append(NoEscape(r'\input{bab1.tex}'))
doc.append(NoEscape(r'\input{bab2.tex}'))
doc.append(NoEscape(r'\input{bab3.tex}'))
doc.append(NoEscape(r'\input{bab4.tex}'))
doc.append(NoEscape(r'\input{bab5.tex}'))

# Mencetak daftar pustaka
doc.append(NoEscape(r'\printbibliography'))

# --- Generate PDF ---
# --- Generate .tex file ---
try:
    doc.generate_tex(output_path)
    print(f"✅ File '{output_path}.tex' berhasil dibuat.")
    print("   Sekarang, lanjutkan dengan kompilasi manual di terminal.")
except Exception as e:
    print(f"❌ Terjadi error saat membuat file .tex: {e}")