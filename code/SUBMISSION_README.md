# CausalCache — AAAI-27 code supplement

This archive contains the implementation, experiment entry points, tests,
dependency locks, and paper-facing configurations for CausalCache. It is a
code-only artifact: datasets, task records, screenshots, trajectories, model
weights, checkpoints, paper sources, PDFs, and experiment outputs are not
included.

## Paper configuration

The paper uses a fixed active history-image budget of `B=4`. CausalCache is the
`HGKV + selector` arm; `Frozen + selector` is a separate ablation. The
paper-facing evaluation contract is
`code/configs/causalcache_paper_evaluation.json`.

For OSWorld-Verified, `max_steps=30` is the main matched allocation comparison.
The earlier `max_steps=15` run is a short-horizon sensitivity inherited from an
example baseline invocation; it is not an official task-success setting. For
MobileWorld, all paper headline values are three-run aggregates.

## Layout

```text
code/
├── causalcache/     # core library, HGKV, rendering, selectors, evaluation
├── integrations/    # benchmark integration adapters
├── scripts/         # training, scoring, reduction, and runner entry points
├── tests/           # unit and contract tests
├── configs/         # paper-facing, model, and matched-baseline configs
└── requirements/    # optional dependency locks
```

The Python package named `causalcache.data` contains only Python source for
dataset interfaces; it does not contain dataset records or generated data.

## Installation and tests

Python 3.10 or newer is required. From the extracted archive root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q code/tests/test_osworld_gui_owl.py
```

Model-backed execution additionally requires the dependencies and model
snapshots named by the included configs. Benchmark datasets and environments
must be obtained from their upstream projects. Paths are supplied through CLI
arguments or local configs; no private storage layout is required by the core
library. The remaining tests are retained for auditability but require their
corresponding optional dependencies and benchmark fixtures, which this
code-only archive intentionally excludes.

## Rebuilding this archive

From the full source repository:

```bash
make submission-code
```

The packager creates a deterministic ZIP, records the source Git revision and
per-file SHA-256 manifest, rejects data/model/output file types, excludes
machine-specific shell runscripts, and anonymizes author-specific identifiers
in the generated artifact.
