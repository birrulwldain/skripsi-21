pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex
biber seminar-proposal.aux
pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex
pdflatex -interaction=nonstopmode --shell-escape seminar-proposal.tex

@REM Uncommnet bagian ini jika inigin mengkompile file seminar-hasil
@REM pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex
@REM biber seminar-hasil.aux 
@REM pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex
@REM pdflatex -interaction=nonstopmode --shell-escape seminar-hasil.tex

@REM Uncomment bagian ini jika ingin mencompile file skripsi
@REM pdflatex -interaction=nonstopmode --shell-escape skripsi.tex
@REM biber skripsi.aux 
@REM pdflatex -interaction=nonstopmode --shell-escape skripsi.tex
@REM pdflatex -interaction=nonstopmode --shell-escape skripsi.tex