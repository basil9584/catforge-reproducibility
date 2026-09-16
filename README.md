# CatFORGE — Reproducibility Package

Seeded synthetic-data generation, quantitative evaluation, and generalized
prediction for heterogeneous-catalysis tabular data — packaged for independent
verification of the CatFORGE manuscript.

This repository contains the **scientific core** of CatFORGE without the
production web service: the synthesis, evaluation, and prediction modules, a
configuration-driven study runner, a 51-row smoke-test dataset, and the
protocol for promoting a demo run into a paper-grade reproduction.

> **Release status: functional demo, not yet a paper reproduction.**
> The bundled demo proves the wiring works end-to-end. It does **not**
> reproduce any manuscript table, figure, or numerical claim. Do not cite this
> repository as the paper's code release until the permitted study datasets,
> exact per-study configurations, split manifests, expected outputs, completed
> citation record, and archived release tag described in
> [PUBLISHING_CHECKLIST.md](PUBLISHING_CHECKLIST.md) are in place.

Related documents:

- [docs/REPRODUCIBILITY_PROTOCOL.md](docs/REPRODUCIBILITY_PROTOCOL.md) — required
  arrangement and run record for every paper study.
- [docs/NATURE_CATALYSIS_CODE_RELEASE_NOTES.md](docs/NATURE_CATALYSIS_CODE_RELEASE_NOTES.md) —
  what editors and reviewers expect in a *Nature Catalysis* code release.
- [RELEASE_SCOPE.md](RELEASE_SCOPE.md) — what is included and what is
  deliberately excluded (web UI, auth, databases, credentials, pickles).
- [SECURITY.md](SECURITY.md) — data-handling rules and preflight checks.
- [PUBLISHING_CHECKLIST.md](PUBLISHING_CHECKLIST.md) — blockers before a public,
  paper-linked release.

---

## 1. What this package does

CatFORGE addresses a practical catalysis problem: experimental datasets are
small, expensive, and heterogeneous, which makes it hard to train reliable
predictive models or to decide which experiment to run next. The scientific
core in this repository covers three steps:

1. **Synthesis** (`src/models/synthesis/data_synthesizer.py`) — learn a tabular
   generative model from a real catalyst dataset and sample physically
   plausible synthetic rows with an explicit random seed. Supported generators
   (via [SDV](https://sdv.dev/)): Gaussian Copula, CopulaGAN, CTGAN.
2. **Evaluation** (`src/evaluators/main_evaluator.py`) — score the synthetic
   table against the real table: statistical fidelity (KS/TV per column,
   correlation similarity, missing-value similarity), detectability (can a
   classifier distinguish real from synthetic?), dimensionality-reduction
   diagnostics (PCA/UMAP), and target-conditional ML efficacy
   (train-on-synthetic / test-on-real for the declared target column).
3. **Prediction** (`src/models/prediction/generalized_predictor.py`) —
   target-agnostic regression workflow (PLS-augmented voting regressor over
   XGBoost / Random Forest with feature selection) plus SHAP/LIME explanations
   and diagnostic plots.

`scripts/reproduce_study.py` chains these steps without the Flask/React
platform so a reviewer can rerun an entire study from a single tracked JSON
config.

The full deployed platform (upload → preview → configure → synthesize →
evaluate → predict → optimize → complete, with Firebase auth, tiered limits,
and active-learning optimization) lives outside this repository. The browser
UI is orchestration and presentation only; it performs no scientific
computation needed to check the paper's numbers, which is why it is excluded
here. See [RELEASE_SCOPE.md](RELEASE_SCOPE.md).

## 2. Repository layout

```
catforge-reproducibility/
├── src/
│   ├── models/synthesis/data_synthesizer.py      # seeded SDV synthesis core
│   ├── models/prediction/generalized_predictor.py # PLS + voting regressor, SHAP/LIME
│   └── evaluators/main_evaluator.py              # fidelity, detectability, efficacy
├── scripts/
│   ├── reproduce_study.py                        # config-driven study runner
│   └── preflight_release.py                      # hygiene / release-blocker check
├── configs/
│   ├── demo_study.json                           # runnable 51-row demo config
│   ├── paper_study.template.json                 # template for each paper study
│   └── README.md                                 # config authoring rules
├── data/
│   ├── demo/catalyst_demo_dataset.csv            # 51-row smoke-test input ONLY
│   ├── raw/ / processed/ / splits/               # paper data goes here (curated)
│   └── metadata/data_manifest.csv                # provenance + SHA-256 per file
├── tests/test_reproducibility_core.py            # config/data alignment, efficacy, determinism
├── results/expected/                             # committed expected metric/figure-source files
├── docs/                                         # protocol + journal notes
├── requirements.txt                              # pinned scientific environment
└── LICENSE / CITATION.cff.template               # MIT licence / citation template
```

Run artifacts are written to `artifacts/<study-id>/` and are git-ignored by
design — the repository tracks *inputs and scripts*, not outputs.

## 3. Requirements

- **Python:** 3.12 or 3.13 (CI reference: 3.13-slim Docker image in the full
  platform; local verification on 3.12 works).
- **OS:** Linux, macOS, or Windows. Linux is recommended for exact
  reproduction; UMAP and torch builds can differ subtly across platforms.
- **Hardware:** CPU is sufficient for the demo. CTGAN/CopulaGAN training on
  paper-scale data benefits from more RAM; no GPU is required.
- **Disk:** ~2 GB for a fresh venv (torch + SDV + SHAP are the heavy
  dependencies).

Dependencies are pinned in [`requirements.txt`](requirements.txt)
(pandas, numpy, scikit-learn, xgboost, sdv, sdmetrics, matplotlib, plotly,
kaleido, umap-learn, torch, shap, lime, pytest). Treat these pins as the
baseline, not yet as an archival lockfile — a paper-linked release must
additionally record the validated platform, Python build, and package hashes
or a container digest (see publishing checklist).

> **CPU-only torch:** the PyPI torch wheel is very large and defaults to CUDA
> builds. If you want a smaller CPU-only install, install torch from the CPU
> index **before** installing the rest:
>
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

## 4. Installation

```bash
git clone <this-repository-url>
cd catforge-reproducibility

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the install:

```bash
python scripts/preflight_release.py
python -m pytest -q
```

`preflight_release.py` checks that the scientific-core files are present and
that no credentials, databases, or unreviewed model artifacts are tracked. The
`--strict-public-release` flag additionally fails while the paper-data
manifest, completed `CITATION.cff`, or other publication blockers are missing
— this is expected in the current staging state.

## 5. Quickstart — run the bundled demo

The demo uses the 51-row file in `data/demo/` and the fully specified config
in `configs/demo_study.json` (seed 42, CTGAN requested, 51 synthetic rows,
explicit Input/Output column roles). It exercises synthesis + evaluation; add
`--with-prediction` for the costlier prediction/explanation step.

```bash
python scripts/reproduce_study.py --config configs/demo_study.json
python scripts/reproduce_study.py --config configs/demo_study.json --with-prediction

# Independent run in a fresh folder (recommended for verification):
python scripts/reproduce_study.py --config configs/demo_study.json --output-dir artifacts/demo-rerun-01

# Full gate:
python -m pytest -q
```

Expected layout after a run (all under the chosen `--output-dir`, default
`artifacts/demo-catalyst/`):

```
artifacts/demo-catalyst/
├── job_details.json        # legacy synthesizer input derived from the tracked config
├── synthesis.log           # run log (includes seed + effective generator)
├── <synthetic csv>         # sampled synthetic table
├── evaluation.json         # fidelity, detectability, efficacy scores
├── run_manifest.json       # config SHA-256, data SHA-256, seed, requested generator
└── prediction/             # only with --with-prediction
    ├── prediction_self_test.csv
    └── prediction_manifest.json  # feature stats, SHAP/LIME/scatter file list
```

`run_manifest.json` is the audit anchor: it binds the study ID to the exact
config bytes, input-data bytes, seed, and generator metadata. Quote its
SHA-256 values when reporting a verification.

## 6. Study configuration reference

Each paper study gets **one reviewed JSON file** copied from
`configs/paper_study.template.json` — never reconstructed from memory or from
opaque historical job folders. Required fields:

| Field | Meaning |
|---|---|
| `study_id` | Stable identifier; also the default artifact folder name. |
| `seed` | Master seed forwarded to NumPy/torch and the synthesizer. Default `42`. |
| `real_data_path` | Repo-relative path to the tracked input (e.g. `data/processed/<study>.csv`). |
| `target_column` | Primary target for target-conditional efficacy. Must exist in the data. |
| `synthesis.model_selected` | Requested generator (`GaussianCopula`, `CopulaGAN`, or `CTGAN`). |
| `synthesis.num_samples` | Number of synthetic rows to sample. |
| `synthesis.columns[]` | Every included column with `name`, `column_role` (`Input`/`Output`), `column_data_type` (`continuous`/`categorical`). |
| `prediction.columns[]` | Same column set in predictor notation (`name`, `role`, `dataType`). Required for `--with-prediction`. |

Important: the synthesis core may select a **different effective generator**
at runtime for small tables and reports this override in its warnings/result
metadata (covered by `test_seeded_small_table_synthesis_is_deterministic`).
Always record the *effective* generator alongside the *requested* one; the
runner preserves both (`requested_generator` in the manifest plus the full
`synthesis_result` payload).

CLI options for `reproduce_study.py`:

```
--config PATH        Tracked study JSON (required).
--output-dir DIR     Artifact directory (default: artifacts/<study-id>).
--with-prediction    Also run the prediction self-test + SHAP/LIME.
--seed INT           Override the config seed for a sensitivity check.
--overwrite          Allow writing into a non-empty output directory.
```

## 7. Evaluation and prediction methods (summary)

- **Statistical fidelity:** per-column KS (continuous) / TV (categorical)
  similarity, inter-column correlation similarity, missing-value-pattern
  similarity (via sdmetrics-backed routines in `main_evaluator.py`).
- **Detectability:** logistic-detection score — how easily a classifier
  separates real from synthetic rows. Lower detectability means higher
  fidelity, but check it jointly with efficacy.
- **Target-conditional efficacy:** train-on-synthetic / test-on-real R² (or
  appropriate metric) for the declared target — the closest proxy to
  "is the synthetic table useful for the paper's prediction task?".
- **Prediction workflow:** PLS dimensionality handling + voting regressor
  (XGBoost + Random Forest) with recursive feature elimination, cross-validated
  metrics, SHAP global attributions, LIME local explanations, and
  predicted-vs-actual diagnostics.

Full metric definitions and protocols belong in the manuscript Methods and in
the evaluator/predictor docstrings; this README states the pipeline shape so a
reviewer knows which output file answers which question.

## 8. Reproducibility design — and its current limits

What is deterministic today:

- Master seed (`seed` in config, default `42`) is fixed before any stochastic
  step (NumPy + torch RNGs in `run_synthesis`).
- Tracked configs pin the input file, column roles/types, sample counts, and
  target; the runner checksums both config and data into `run_manifest.json`.
- Tests pin config↔data column alignment, efficacy direction on a known
  relationship, and seeded small-table synthesis determinism.

What is **not** yet paper-grade:

- Only the demo dataset is bundled. The manuscript's study datasets, row-ID
  split manifests, preprocessing scripts, per-study configs, expected outputs,
  and figure-generation scripts are absent by design until data owners approve
  redistribution (see `data/README.md`, `data/metadata/README.md`,
  `data/splits/README.md`).
- `requirements.txt` pins are not an archival lockfile. Validate on a clean
  machine and archive exact hashes or a container digest before release.
- Deep-learning generators (CTGAN/CopulaGAN) can vary across torch/CUDA
  builds despite seeding; the Gaussian Copula path is fully deterministic and
  preferable for exact-match checks.

Independent verification procedure: clone into a clean directory, install from
`requirements.txt`, run the demo plus `pytest`, then each paper-study command
from the protocol, and diff `evaluation.json` / `results/expected/` hashes. A
second author should perform this comparison before submission
(see [PUBLISHING_CHECKLIST.md](PUBLISHING_CHECKLIST.md)).

## 9. From demo to paper reproduction

1. Curate each approved input into `data/processed/` (keeping the permitted
   original in `data/raw/` where possible) and add a SHA-256 provenance row to
   `data/metadata/data_manifest.csv`.
2. Add a row-ID-based split manifest in `data/splits/` — never infer splits
   from row order.
3. Freeze one config per study from `configs/paper_study.template.json`,
   recording requested *and* effective generators.
4. Run without the web app:
   `python scripts/reproduce_study.py --config configs/<study>.json --output-dir artifacts/<study> --with-prediction`.
5. Add a tracked script that builds every figure-source CSV, table, and graphic
   from run outputs — no hand-edited numbers.
6. Commit lightweight expected outputs under `results/expected/` with hashes
   and have a co-author rerun on a clean clone.

The full checklist is in [PUBLISHING_CHECKLIST.md](PUBLISHING_CHECKLIST.md).

## 10. Tests

```bash
python -m pytest -q
```

Covered: demo config↔data column alignment, target-efficacy direction on a
known linear relationship, seeded small-table synthesis determinism (including
the small-data generator-override warning), and predictor importability
without the web app. `pytest.ini` sets `pythonpath = src` so tests import the
research package directly.

## 11. Security and data policy

Never commit `.env` files, API keys, Firebase/service-account JSON, user
uploads, databases, session/job folders, logs, unreviewed `.pkl`/`.joblib`
files, or any data without a documented owner and redistribution permission.
These patterns are git-ignored and caught by `preflight_release.py`. If a
secret was ever pushed or shared outside the authorized team, rotate it with
the provider and purge it from history before publishing. Details in
[SECURITY.md](SECURITY.md).

## 12. Citation

A citable release needs a completed `CITATION.cff` (see
[`CITATION.cff.template`](CITATION.cff.template)), a release tag, and — when
the manuscript's code-availability statement cites an archive — a
DOI-granting deposit (e.g. Zenodo) of the exact tagged source. Until then:

```bibtex
@software{catforge_reproducibility,
  title  = {CatFORGE Reproducibility Package},
  author = {{CatFORGE authors}},
  year   = {2026},
  note   = {Seeded synthesis, evaluation, and prediction core; staging release}
}
```

## 13. License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 CatFORGE authors.

Code and data are licensed separately: the MIT licence covers the software in
`src/`, `scripts/`, `configs/`, and `tests/`. Every dataset needs its own
documented licence or written redistribution permission recorded in
`data/metadata/data_manifest.csv` — never assume the code licence covers the
data.

## 14. Code-availability statement (template for the manuscript)

> Code for synthesis, evaluation, and generalized prediction is available at
> [REPOSITORY URL] (release [TAG], archived at [DOI URL]). The repository
> includes the exact source version, pinned dependencies, tracked study
> configurations, data-manifest checksums, and run instructions to reproduce
> all computational results. The web deployment layer is not required for
> reproduction and is excluded from this release.

Replace all bracketed fields with verified values before submission, and check
the current *Nature Catalysis* editorial policies, since requirements can
change: <https://www.nature.com/natcatal/editorial-policies>.
