"""Check that the staged repository has no obvious private-release hazards."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md",
    "requirements.txt",
    "RELEASE_SCOPE.md",
    "SECURITY.md",
    "PUBLISHING_CHECKLIST.md",
    "src/models/synthesis/data_synthesizer.py",
    "src/evaluators/main_evaluator.py",
    "src/models/prediction/generalized_predictor.py",
    "scripts/reproduce_study.py",
    "configs/demo_study.json",
    "data/metadata/data_manifest.csv",
)
SKIP_DIRECTORIES = {".git", ".venv", "venv", "__pycache__", "artifacts"}


def contains_forbidden_file(path: Path) -> str | None:
    name = path.name.lower()
    if name == ".env" or name.startswith(".env."):
        return "environment file"
    if "service-account" in name or "adminsdk" in name or name.startswith("firebase"):
        return "credential-like filename"
    if path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".pkl", ".joblib"}:
        return f"unreviewed runtime artifact ({path.suffix})"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repository hygiene checks.")
    parser.add_argument(
        "--strict-public-release",
        action="store_true",
        help="Fail if the outstanding public-release blockers are not complete.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).is_file():
            failures.append(f"Missing required repository file: {relative_path}")

    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRECTORIES for part in path.relative_to(ROOT).parts):
            continue
        if not path.is_file():
            continue
        reason = contains_forbidden_file(path)
        if reason:
            failures.append(f"Forbidden {reason}: {path.relative_to(ROOT)}")

    manifest_path = ROOT / "data" / "metadata" / "data_manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        if not manifest_rows:
            warnings.append("Data manifest has no paper-study records; staging state is expected.")

    if not (ROOT / "LICENSE").is_file():
        warnings.append("No approved LICENSE is present.")
    if not (ROOT / "CITATION.cff").is_file():
        warnings.append("No completed CITATION.cff is present.")

    if args.strict_public_release:
        if not (ROOT / "LICENSE").is_file():
            failures.append("Public release blocked: choose and add LICENSE.")
        if not (ROOT / "CITATION.cff").is_file():
            failures.append("Public release blocked: complete CITATION.cff.")
        if manifest_path.is_file() and not manifest_rows:
            failures.append("Public release blocked: add verified data-manifest records.")

    for message in failures:
        print(f"FAIL: {message}")
    for message in warnings:
        print(f"WARN: {message}")
    if not failures:
        print("PASS: Repository contains the required scientific-core files and no obvious forbidden files.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
