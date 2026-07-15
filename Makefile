.PHONY: paper clean-paper test validate-contract validate-restoration-v2 validate-restoration-v2-interfaces validate-restoration-v2-executor-dispatch validate-restoration-v2-selection synthetic-phase0

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

synthetic-phase0:
	cd code && python3 -m scripts.run_synthetic_attribution \
		--contract configs/phase0_contract.json \
		--scenario configs/synthetic_phase0.json \
		--output-json ../data/results/synthetic_phase0/summary.json \
		--output-csv ../data/results/synthetic_phase0/per_seed.csv \
		--output-report ../data/results/synthetic_phase0/README.md
