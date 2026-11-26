# Regenerate the measured numbers / figures / table, then build main.pdf.
# Needs a LaTeX engine: latexmk + pdflatex (apt: texlive-latex-extra texlive-science latexmk), or tectonic.
PY ?= python3

all: assets pdf

assets:
	cd ../speed_vlm && $(PY) baselines.py && $(PY) paper_figs.py

pdf:
	@if command -v latexmk >/dev/null; then latexmk -pdf -interaction=nonstopmode main.tex; \
	elif command -v tectonic >/dev/null; then tectonic main.tex; \
	else echo "no LaTeX engine (install texlive or tectonic)"; exit 1; fi

clean:
	latexmk -C 2>/dev/null || true

.PHONY: all assets pdf clean
