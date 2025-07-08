pdflatex -interaction=nonstopmode --shell-escape 1proposal.tex
biber 1proposal.aux
pdflatex -interaction=nonstopmode --shell-escape 1proposal.tex
pdflatex -interaction=nonstopmode --shell-escape 1proposal.tex

@REM Uncommnet bagian ini jika inigin mengkompile file seminar-hasil
@REM pdflatex -interaction=nonstopmode --shell-escape 2hasil.tex
@REM biber 2hasil.aux 
@REM pdflatex -interaction=nonstopmode --shell-escape 2hasil.tex
@REM pdflatex -interaction=nonstopmode --shell-escape 2hasil.tex

@REM Uncomment bagian ini jika ingin mencompile file skripsi
@REM pdflatex -interaction=nonstopmode --shell-escape 3skripsi.tex
@REM biber 3skripsi.aux 
@REM pdflatex -interaction=nonstopmode --shell-escape 3skripsi.tex
@REM pdflatex -interaction=nonstopmode --shell-escape 3skripsi.tex