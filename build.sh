pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex;
biber seminar-proposal.aux ;
pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex;
pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex;

# Uncommnet bagian ini jika inigin mengkompile file seminar-hasil
pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex;
biber seminar-hasil.aux ;
pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex;
pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex;

# Uncomment bagian ini jika ingin mencompile file skripsi
pdflatex -interaction=nonstopmode --shell-escape skripsi.tex;
biber skripsi.aux ;
pdflatex -interaction=nonstopmode --shell-escape skripsi.tex;
pdflatex -interaction=nonstopmode --shell-escape skripsi.tex;
