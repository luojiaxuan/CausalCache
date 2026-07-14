.PHONY: paper clean-paper

paper:
	mkdir -p output/pdf
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../output/pdf main.tex
	cp output/pdf/main.pdf output/pdf/causalcache_aaai27.pdf

clean-paper:
	cd paper && latexmk -C -outdir=../output/pdf main.tex
	rm -f output/pdf/causalcache_aaai27.pdf
