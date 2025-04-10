Berikut adalah struktur manajemen proyek GitHub yang terintegrasi dengan LaTeX dan pengembangan prototipe ilmiah, dirancang khusus untuk kebutuhan penelitian Anda:

---

### **Struktur Direktori Rekomendasi**
```bash
LIBS-Spectral-Simulation/
├── .github/                  # GitHub Actions, issue templates
│   └── workflows/
│       └── compile-latex.yml
├── literature/               # Tinjauan pustaka & referensi
│   ├── papers/               # PDF paper yang relevan
│   ├── bib/                  # File .bib
│   └── literature_review.tex # Draft tinjauan pustaka
├── src/                      Kode simulasi & AI
│   ├── physics_model/        # Simulasi fisika dasar (Python/Matlab)
│   │   ├── boltzmann_saha.py
│   │   └── plasma_simulation.ipynb
│   ├── GAN/                  # Implementasi PA-DCGAN
│   │   ├── model.py
│   │   ├── train_gan.ipynb
│   │   └── requirements.txt
│   └── utils/                # Tools preprocessing data
│       ├── nist_parser.py
│       └── spectral_analysis.py
├── data/                     # Dataset & output simulasi
│   ├── raw/                  # Data mentah (NIST ASD, dll)
│   ├── synthetic/            # Spektrum sintetis (HDF5/CSV)
│   └── experimental/         # Data LIBS eksperimen (jika ada)
├── output/                   # Hasil ilmiah (LaTeX-ready)
│   ├── figures/              # Plot spektrum, arsitektur GAN
│   │   ├── spectra/
│   │   └── model_diagrams/
│   ├── tables/               # Tabel parameter, hasil eksperimen
│   └── equations/            # Persamaan fisika dalam format .tex
├── thesis/                   # Penulisan laporan/thesis
│   ├── chapters/
│   │   ├── 1_introduction.tex
│   │   ├── 2_literature.tex
│   │   └── 3_methodology.tex
│   ├── thesis.tex            # File master LaTeX
│   └── styles/
│       └── unsrt-custom.bst  # Gaya sitasi kustom
└── .gitignore                # Exclude: .aux, .log, dataset besar
```

---

### **Best Practices untuk Version Control**
1. **`.gitignore` untuk LaTeX & Python**:
   ```gitignore
   # LaTeX
   *.aux
   *.log
   *.bbl
   *.blg
   *.out

   # Python
   __pycache__/
   *.pyc
   .venv/

   # Data besar (gunakan Git LFS)
   *.h5
   *.csv
   ```

2. **Branching Strategy**:
   - `main`: Versi stabil (hanya hasil final).
   - `dev`: Pengembangan aktif.
   - `feature/physic-model`: Fitur spesifik (misal: implementasi model fisika).
   - `experiment/gan-augmentation`: Eksperimen riset.

3. **Commit Messages Bermakna**:
   - `feat: Add Boltzmann equation solver`
   - `fix: Correct plasma density calculation`
   - `docs: Update literature review section 2.1`

---

### **Integrasi Workflow Penelitian**
1. **Dari Simulasi ke LaTeX**:
   - Hasil simulasi (dari `src/`) → diekspor ke `data/synthetic/` → diolah menjadi plot (Python/Matlab) → simpan ke `output/figures/` → referensi di LaTeX via `\includegraphics`.

2. **Automation Script**:
   ```bash
   # compile_and_push.sh
   python src/utils/generate_figures.py  # Generate figures
   cd thesis && latexmk -pdf thesis.tex  # Compile LaTeX
   git add output/figures/* thesis/thesis.pdf
   git commit -m "docs: Update results section with new spectra"
   git push origin dev
   ```

3. **GitHub Actions untuk LaTeX**:
   - Otomatisasi kompilasi LaTeX di GitHub:
     ```yaml
     # .github/workflows/compile-latex.yml
     name: Compile LaTeX
     on: [push]
     jobs:
       build:
         runs-on: ubuntu-latest
         steps:
           - name: Compile LaTeX
             uses: xu-cheng/latex-action@v2
             with:
               root_file: thesis/thesis.tex
               args: -pdf -interaction=nonstopmode
     ```

---

### **Fitur Kolaboratif untuk Riset**
1. **GitHub Issues**:
   - Template issue untuk:  
     - **Bug**: "Synthetic spectra mismatch NIST data at 500-600nm"  
     - **Enhancement**: "Implement uncertainty quantification in GAN"  
     - **Question**: "Optimal plasma temp range for volcanic soils?"

2. **Project Board**:
   Buat board dengan kolom:  
   - `Backlog` → `Literature Review` → `Simulation` → `Validation` → `Writing`

3. **Wiki Repositori**:
   - Dokumentasi spesifik proyek:  
     - "Cara menjalankan simulasi fisika dasar"  
     - "Struktur dataset sintetik"  
     - "Panduan kontribusi untuk kolaborator"

---

### **Tips untuk Output Ilmiah dalam LaTeX**
1. **Modularisasi Dokumen**:
   ```latex
   % File: thesis/thesis.tex
   \documentclass{report}
   \input{styles/packages}
   \begin{document}
   \input{chapters/1_introduction}
   \input{chapters/2_literature}
   \input{chapters/3_methodology}
   \end{document}
   ```

2. **Versioning Gambar**:
   - Nama file gambar: `fig_spectra_v3.pdf` (v3 = versi ke-3).
   - Gunakan `\includegraphics` dengan label deskriptif:
     ```latex
     \begin{figure}
       \centering
       \includegraphics[width=0.8\textwidth]{figures/spectra/fe_8000K_v2.pdf}
       \caption{Synthetic Fe spectrum at 8000 K (v2).}
       \label{fig:fe-spectrum}
     \end{figure}
     ```

3. **BibTeX Terorganisir**:
   - Pisahkan referensi per bab:
     ```bib
     @phdthesis{smith2023libs,
       author = "Smith, John",
       title  = "Advanced LIBS for Geochemical Analysis",
       school = "MIT",
       year   = "2023",
       file   = "literature/papers/smith2023.pdf"
     }
     ```

---

### **Contoh Workflow Harian**
1. **Pagi**:
   - `git pull` → Update repositori.
   - Kerjakan simulasi fisika di branch `feature/physic-model`.

2. **Siang**:
   - Hasilkan plot baru → commit ke `output/figures/`.
   - Tulis bagian metodologi di `thesis/chapters/3_methodology.tex`.

3. **Sore**:
   - `git push` → Trigger GitHub Actions untuk kompilasi LaTeX.
   - Review PDF yang di-generate otomatis di GitHub.

---

Dengan struktur ini, prototipe kode dan penulisan ilmiah dapat berkembang secara paralel, terintegrasi rapi, dan siap untuk kolaborasi atau publikasi terbuka.