# Release scope

## Included scientific core

| Release path | Source reviewed for this staging package | Role |
| --- | --- | --- |
| `src/models/synthesis/data_synthesizer.py` | `backend/models/synthesis/data_synthesizer.py` | Seeded tabular data synthesis |
| `src/evaluators/main_evaluator.py` | `backend/evaluators/main_evaluator.py` | Fidelity, detectability and target-efficacy evaluation |
| `src/models/prediction/generalized_predictor.py` | `backend/models/prediction/generalized_predictor.py` | Generalized prediction, metrics and explainability outputs |
| `data/demo/catalyst_demo_dataset.csv` | `backend/demo_data/catalyst_demo_dataset.csv` | Smoke-test input only |

The `src/*/__init__.py` files are intentionally clean research-package markers. They do not carry over the original application's database models or authentication dependencies.

## Intentionally excluded

- The browser client, compiled frontend bundle and Node dependencies. Editable frontend source is not available in this workspace; the UI is not required for numerical reproduction.
- Flask/API routes, authentication, database schema, deployment configuration and live platform administration.
- Service-account JSON files, `.env` files, credentials, user data, uploads, runtime jobs, logs and databases.
- Pre-trained pickle/joblib artifacts and historical job outputs, whose data provenance and redistribution rights have not been confirmed.
- Candidate-search/optimization and active-learning modules. They are not needed for the reported paper studies and currently require additional dependency/provenance review.
- Manuscript PDFs and existing output packages. A code repository should contain tracked inputs and scripts, not duplicate submission artifacts.

## What this package can and cannot establish

It can run a deterministic demo pipeline and expose the core code needed to build a real study-reproduction workflow. It cannot yet recreate a paper table, figure or numerical claim because the paper-specific source data, preprocessing, exact configurations, splits, effective generator records and expected outputs have not been curated into this package.
