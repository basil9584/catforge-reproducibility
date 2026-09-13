"""Focused tests for the public scientific-core package.

These tests avoid the production API, databases and historical runtime data.
They confirm that the demo configuration remains aligned with its input data,
that the core modules import independently, and that seeded small-table
synthesis remains reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluators.main_evaluator import compute_target_efficacy
from models.prediction.generalized_predictor import train_and_predict_on_dataframe
from models.synthesis.data_synthesizer import DEFAULT_SEED, run_synthesis


ROOT = Path(__file__).resolve().parents[1]


def test_demo_configuration_matches_demo_columns():
    config = json.loads((ROOT / "configs" / "demo_study.json").read_text(encoding="utf-8"))
    data = pd.read_csv(ROOT / config["real_data_path"])
    synthesis_columns = [item["name"] for item in config["synthesis"]["columns"]]
    prediction_columns = [item["name"] for item in config["prediction"]["columns"]]

    assert synthesis_columns == list(data.columns)
    assert prediction_columns == list(data.columns)
    assert config["target_column"] in data.columns


def test_target_efficacy_is_high_for_matching_relationships():
    rng = np.random.default_rng(12)
    real_x = rng.uniform(0, 10, 120)
    synthetic_x = rng.uniform(0, 10, 120)
    real = pd.DataFrame({"x": real_x, "y": 3 * real_x + rng.normal(0, 0.25, 120)})
    synthetic = pd.DataFrame(
        {"x": synthetic_x, "y": 3 * synthetic_x + rng.normal(0, 0.25, 120)}
    )

    efficacy = compute_target_efficacy(real, synthetic, "y")
    assert efficacy is not None
    assert efficacy["metric"] == "r2"
    assert efficacy["score"] > 0.8


def test_seeded_small_table_synthesis_is_deterministic(tmp_path):
    rng = np.random.default_rng(8)
    real = pd.DataFrame(
        {
            "Temperature": rng.uniform(500, 900, 80).round(2),
            "Pressure": rng.uniform(1, 30, 80).round(2),
            "Catalyst": rng.choice(["Ni", "Pt", "Fe"], 80),
            "Conversion": rng.uniform(10, 95, 80).round(2),
        }
    )
    real_path = tmp_path / "real.csv"
    real.to_csv(real_path, index=False)

    config = {
        "original_file_path": str(real_path),
        "model_selected": "CTGAN",
        "synthesis_parameters": {"num_samples": 16},
        "column_configs": [
            {"name": "Temperature", "column_role": "Input", "column_data_type": "continuous"},
            {"name": "Pressure", "column_role": "Input", "column_data_type": "continuous"},
            {"name": "Catalyst", "column_role": "Input", "column_data_type": "categorical"},
            {"name": "Conversion", "column_role": "Output", "column_data_type": "continuous"},
        ],
    }

    outputs = []
    for label in ("first", "second"):
        run_dir = tmp_path / label
        run_dir.mkdir()
        config_path = run_dir / "job_details.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        result = run_synthesis(
            label,
            str(run_dir),
            str(config_path),
            str(run_dir / "synthesis.log"),
            seed=DEFAULT_SEED,
        )
        assert "error" not in result, result.get("error")
        assert result["warnings"], "Small data should report its generator override."
        outputs.append(pd.read_csv(result["output_path"]))

    pd.testing.assert_frame_equal(outputs[0], outputs[1], check_exact=False, rtol=1e-9)


def test_generalized_predictor_is_available_without_the_web_application():
    assert callable(train_and_predict_on_dataframe)
