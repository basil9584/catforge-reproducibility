"""Run one explicit CatFORGE study without the web application.

The configuration is deliberately tracked separately from code so a paper run
can preserve its input file, selected columns, random seed and requested
generator. This script is a study runner, not a substitute for a manuscript
figure-generation workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from evaluators.main_evaluator import CustomJsonEncoder, run_evaluation
from models.prediction.generalized_predictor import train_and_predict_on_dataframe
from models.synthesis.data_synthesizer import DEFAULT_SEED, run_synthesis


def sha256_file(path: Path) -> str:
    """Return the checksum of an exact input or configuration file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_repository_path(path: Path) -> str:
    """Use a portable relative path where possible in a run manifest."""
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_input_path(value: str) -> Path:
    """Resolve a configuration path relative to the repository root."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=CustomJsonEncoder)
        handle.write("\n")


def validate_config(config: dict[str, Any]) -> None:
    required = {"study_id", "real_data_path", "synthesis"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Configuration is missing required keys: {', '.join(missing)}")

    synthesis = config["synthesis"]
    if not isinstance(synthesis, dict):
        raise ValueError("'synthesis' must be an object.")
    for key in ("model_selected", "num_samples", "columns"):
        if key not in synthesis:
            raise ValueError(f"'synthesis.{key}' is required.")
    if not isinstance(synthesis["columns"], list) or not synthesis["columns"]:
        raise ValueError("'synthesis.columns' must be a non-empty list.")


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {path}. "
            "Choose a new --output-dir or pass --overwrite intentionally."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_study(
    config_path: Path,
    output_dir: Path,
    *,
    with_prediction: bool,
    seed_override: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)

    source_data_path = resolve_input_path(config["real_data_path"])
    if not source_data_path.is_file():
        raise FileNotFoundError(f"Input data not found: {source_data_path}")

    output_dir = prepare_output_directory(output_dir, overwrite)
    synthesis_config = config["synthesis"]
    seed = int(seed_override if seed_override is not None else config.get("seed", DEFAULT_SEED))

    # The core synthesizer accepts this legacy job-details shape. Keep the
    # generated details file in the ignored run directory, while the canonical
    # study configuration remains version-controlled in configs/.
    job_details = {
        "original_file_path": str(source_data_path),
        "model_selected": synthesis_config["model_selected"],
        "synthesis_parameters": {"num_samples": int(synthesis_config["num_samples"])},
        "column_configs": synthesis_config["columns"],
    }
    job_details_path = output_dir / "job_details.json"
    write_json(job_details_path, job_details)

    synthesis_result = run_synthesis(
        str(config["study_id"]),
        str(output_dir),
        str(job_details_path),
        str(output_dir / "synthesis.log"),
        seed=seed,
    )
    if synthesis_result.get("error"):
        raise RuntimeError(f"Synthesis failed: {synthesis_result['error']}")

    synthetic_data_path = Path(synthesis_result["output_path"])
    target_column = config.get("target_column")
    evaluation_result = run_evaluation(
        str(config["study_id"]),
        str(output_dir),
        str(source_data_path),
        str(synthetic_data_path),
        target_column=target_column,
    )
    write_json(output_dir / "evaluation.json", evaluation_result)

    manifest: dict[str, Any] = {
        "study_id": config["study_id"],
        "configuration": {
            "path": as_repository_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "input_data": {
            "path": as_repository_path(source_data_path),
            "sha256": sha256_file(source_data_path),
        },
        "seed": seed,
        "requested_generator": synthesis_config["model_selected"],
        "target_column": target_column,
        "synthetic_data_path": as_repository_path(synthetic_data_path),
        "synthesis_result": synthesis_result,
        "evaluation_output": "evaluation.json",
    }

    if with_prediction:
        prediction_config = config.get("prediction", {})
        column_configs = prediction_config.get("columns")
        if not column_configs:
            raise ValueError(
                "--with-prediction requires prediction.columns in the study configuration."
            )
        prediction_dir = output_dir / "prediction"
        prediction_dir.mkdir(exist_ok=True)
        predictions, feature_stats, shap_files, lime_files, scatter_files, error = (
            train_and_predict_on_dataframe(
                str(source_data_path), None, column_configs, str(prediction_dir)
            )
        )
        if error:
            raise RuntimeError(f"Prediction failed: {error}")
        if predictions is not None:
            predictions.to_csv(prediction_dir / "prediction_self_test.csv", index=False)
        write_json(
            prediction_dir / "prediction_manifest.json",
            {
                "input_mode": "self-test on the configured study dataset",
                "feature_stats": feature_stats,
                "shap_files": shap_files,
                "lime_files": lime_files,
                "scatter_files": scatter_files,
            },
        )
        manifest["prediction_output"] = "prediction/prediction_manifest.json"

    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a config-defined CatFORGE synthesis/evaluation study."
    )
    parser.add_argument("--config", required=True, type=Path, help="Tracked study JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run-artifact directory (default: artifacts/<study-id>).",
    )
    parser.add_argument(
        "--with-prediction",
        action="store_true",
        help="Also run the generalized prediction self-test and explanations.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configuration seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit overwriting files in a non-empty output directory.",
    )
    args = parser.parse_args()

    try:
        config_path = resolve_input_path(str(args.config))
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        default_output = REPOSITORY_ROOT / "artifacts" / str(config.get("study_id", "study"))
        output_dir = args.output_dir or default_output
        if not output_dir.is_absolute():
            output_dir = REPOSITORY_ROOT / output_dir
        manifest = run_study(
            config_path,
            output_dir,
            with_prediction=args.with_prediction,
            seed_override=args.seed,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Completed study '{manifest['study_id']}'.")
    print(f"Run manifest: {Path(output_dir).resolve() / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
