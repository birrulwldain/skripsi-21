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
