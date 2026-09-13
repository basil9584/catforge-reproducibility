# Reproducibility protocol for paper studies

The demo proves that the repository wiring is functional. A scientific paper run requires a separate, auditable record for every study.

1. **Curate the input.** Place an approved, immutable analysis input in `data/processed/`, retain its permitted source in `data/raw/` when possible, and add a provenance row with a SHA-256 checksum to `data/metadata/data_manifest.csv`.
2. **Preserve identifiers and splits.** Add a row-ID-based split manifest in `data/splits/`; do not reconstruct train/test membership from a row order or memory.
3. **Freeze the run configuration.** Copy `configs/paper_study.template.json` to an explicit study filename. Specify every input/output column, type, requested generator, sample count, target and seed. Record any core-code rule that changes the effective generator at runtime.
4. **Run without the web app.** Execute `python scripts/reproduce_study.py --config configs/<study>.json --output-dir artifacts/<study> --with-prediction`. The runner emits a config/data checksum and core metadata in `run_manifest.json`.
5. **Generate manuscript artifacts.** Add a tracked script that converts the run outputs into every figure-source CSV, table and final graphic. Do not hand-edit numbers or figures after generation.
6. **Verify independently.** Commit lightweight expected metric JSON/CSV outputs under `results/expected/`, record their hashes, and have a co-author rerun the workflow on a clean clone.

For data that cannot be openly redistributed, the manifest and README must state the access procedure, eligibility, persistent source, preprocessing and expected checksums. A repository link alone is not sufficient if a reader cannot identify the exact data and configuration that generated the reported result.
