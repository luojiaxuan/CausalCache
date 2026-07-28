.PHONY: figures paper supplement paper-all clean-paper test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces validate-restoration-v2-executor-dispatch validate-restoration-v2-selection validate-restoration-v2-ocr-config validate-restoration-v2-ocr-artifact validate-restoration-v2-baselines synthetic-phase0

FIGURE_SOURCE_DATE_EPOCH ?= 1785168000
FIGURE_PDF_TMP_DIR ?= tmp/paper-figures
GS_OUTLINE_FLAGS = -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.7 -dNoOutputFonts -dOmitInfoDate=true -dOmitID=true -dDeterministicIDs=true

figures:
	python3 paper/figures/build_causalcache_figures.py
	mkdir -p $(FIGURE_PDF_TMP_DIR)
	SOURCE_DATE_EPOCH=$(FIGURE_SOURCE_DATE_EPOCH) rsvg-convert -f pdf -o $(FIGURE_PDF_TMP_DIR)/causalcache_overview.pdf paper/figures/causalcache_overview.svg
	gs $(GS_OUTLINE_FLAGS) -sOutputFile=$(FIGURE_PDF_TMP_DIR)/causalcache_overview_warm.pdf $(FIGURE_PDF_TMP_DIR)/causalcache_overview.pdf
	gs $(GS_OUTLINE_FLAGS) -sOutputFile=paper/figures/causalcache_overview.pdf $(FIGURE_PDF_TMP_DIR)/causalcache_overview.pdf
	SOURCE_DATE_EPOCH=$(FIGURE_SOURCE_DATE_EPOCH) rsvg-convert -f png -w 2016 -o paper/figures/causalcache_overview.png paper/figures/causalcache_overview.svg
	SOURCE_DATE_EPOCH=$(FIGURE_SOURCE_DATE_EPOCH) rsvg-convert -f pdf -o $(FIGURE_PDF_TMP_DIR)/causalcache_qualitative_cart.pdf paper/figures/causalcache_qualitative_cart.svg
	gs $(GS_OUTLINE_FLAGS) -sOutputFile=$(FIGURE_PDF_TMP_DIR)/causalcache_qualitative_cart_warm.pdf $(FIGURE_PDF_TMP_DIR)/causalcache_qualitative_cart.pdf
	gs $(GS_OUTLINE_FLAGS) -sOutputFile=paper/figures/causalcache_qualitative_cart.pdf $(FIGURE_PDF_TMP_DIR)/causalcache_qualitative_cart.pdf
	SOURCE_DATE_EPOCH=$(FIGURE_SOURCE_DATE_EPOCH) rsvg-convert -f png -w 2016 -o paper/figures/causalcache_qualitative_cart.png paper/figures/causalcache_qualitative_cart.svg

paper: figures
	mkdir -p output/pdf
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../output/pdf main.tex
	cp output/pdf/main.pdf output/pdf/causalcache_aaai27.pdf

supplement:
	mkdir -p output/pdf
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../output/pdf supplement.tex
	cp output/pdf/supplement.pdf output/pdf/causalcache_aaai27_supplement.pdf

paper-all: paper supplement

clean-paper:
	cd paper && latexmk -C -outdir=../output/pdf main.tex
	cd paper && latexmk -C -outdir=../output/pdf supplement.tex
	rm -f output/pdf/causalcache_aaai27.pdf output/pdf/causalcache_aaai27_supplement.pdf

test:
	cd code && python3 -m unittest discover -s tests -v

validate-contract:
	cd code && python3 -m scripts.validate_contract \
		--config configs/phase0_contract.json \
		--decision ../data/fixtures/validated_decision.json

validate-restoration-v2:
	cd code && python3 -m scripts.validate_restoration_v2_contract \
		--config configs/causalcache_restoration_v2.json

validate-restoration-v2-interfaces:
	cd code && python3 -m scripts.validate_restoration_v2_interfaces \
		--contract configs/causalcache_restoration_v2.json \
		--action-fixture ../data/fixtures/gui_owl_v2_action_roundtrip.json \
		--prompt-fixture ../data/fixtures/restoration_v2_prompt_low_fidelity.json \
		--interface-manifest ../data/manifests/restoration_v2_interfaces.json

validate-restoration-v2-executor-dispatch:
	cd code && python3 -m scripts.validate_restoration_v2_executor_evidence validate \
		--summary ../data/results/restoration_v2_executor_dispatch/summary-rv2-20260715T101814Z-53016a40.json

validate-restoration-v2-selection:
	cd code && python3 -m scripts.validate_restoration_v2_selection \
		--v2-contract configs/causalcache_restoration_v2.json \
		--selection ../data/manifests/restoration_v2_selection.json \
		--exposure ../data/manifests/restoration_v2_exposure.json

validate-restoration-v2-ocr-config:
	cd code && python3 -m scripts.validate_restoration_v2_ocr_backend config-only \
		--backend-config configs/restoration_v2_ocr_backend.json \
		--fixture ../data/fixtures/restoration_v2_ocr_golden.json

validate-restoration-v2-ocr-artifact:
	cd code && python3 -m scripts.validate_restoration_v2_ocr_backend artifact-source \
		--backend-config configs/restoration_v2_ocr_backend.json \
		--artifact-manifest ../data/manifests/restoration_v2_ocr_backend.json \
		--repository-root ..

validate-restoration-v2-baselines:
	cd code && python3 -m scripts.validate_restoration_v2_baselines \
		--manifest ../data/manifests/restoration_v2_baselines.json \
		--repository-root ..

synthetic-phase0:
	cd code && python3 -m scripts.run_synthetic_attribution \
		--contract configs/phase0_contract.json \
		--scenario configs/synthetic_phase0.json \
		--output-json ../data/results/synthetic_phase0/summary.json \
		--output-csv ../data/results/synthetic_phase0/per_seed.csv \
		--output-report ../data/results/synthetic_phase0/README.md
