# Configuration
MAIN = report
LATEX = pdflatex
BIB = biber

# Règle par défaut : compile le PDF
compile: $(MAIN).pdf

# Compilation PDF avec biber
$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN).tex

# Nettoyage des fichiers temporaires générés par la compilation
clean:
	find . -type f \( \
		-name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.toc" \
		-o -name "*.lof" -o -name "*.lot" -o -name "*.bbl" -o -name "*.blg" \
		-o -name "*.maf" -o -name "*.mtc*" -o -name "*.thm" -o -name "*.synctex.gz" \
		-o -name "*.ptc*" -o -name "*.fls" -o -name "*.fdb_latexmk" \
		-o -name "*.bcf" -o -name "*.run.xml" \
	\) -delete

# Nettoyage complet : enlève le PDF et les fichiers temporaires
cleanall: clean
	rm -f $(MAIN).pdf

# Visualiser le PDF
view: $(MAIN).pdf
	open $(MAIN).pdf

build:
	make clean
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex
	make clean