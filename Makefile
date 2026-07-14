.PHONY: paper clean-paper test validate-contract synthetic-phase0

paper:
	mkdir -p output/pdf
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../output/pdf main.tex
	cp output/pdf/main.pdf output/pdf/causalcache_aaai27.pdf

clean-paper:
	cd paper && latexmk -C -outdir=../output/pdf main.tex
	rm -f output/pdf/causalcache_aaai27.pdf

test:
	cd code && python3 -m unittest discover -s tests -v

validate-contract:
	cd code && python3 -m scripts.validate_contract \
		--config configs/phase0_contract.json \
		--decision ../data/fixtures/validated_decision.json

synthetic-phase0:
	cd code && python3 -m scripts.run_synthetic_attribution \
		--contract configs/phase0_contract.json \
		--scenario configs/synthetic_phase0.json \
		--output-json ../data/results/synthetic_phase0/summary.json \
		--output-csv ../data/results/synthetic_phase0/per_seed.csv \
		--output-report ../data/results/synthetic_phase0/README.md
