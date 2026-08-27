[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Customer Churn & Financial Retention Platform

![Streamlit ROI simulator](docs/streamlit_preview.png)

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-2E7D32)
![MLflow](https://img.shields.io/badge/MLflow-tracking_%26_registry-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-inference_API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-ROI_simulator-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?logo=docker&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end customer churn prediction system for banking/retail, built around one central idea: **a churn model isn't worth its AUC, it's worth how much money the business stops losing when it's used to decide who to contact.**

## The result, first

Evaluated on a test holdout never seen during training or threshold optimization:

| Strategy | Net financial value | Customers contacted |
|---|---:|---:|
| Contact no one | −$809,055 | 0% |
| Contact everyone (no model) | +$42,717 | 100% |
| **Model's optimal threshold** | **+$75,847** | 78.6% |

The model nearly doubles the return of an indiscriminate campaign, while contacting fewer people. That's the product: not "who is going to churn," but "who is it worth offering something to, and how much is it worth doing."

**The threshold isn't an artifact of a favorable split**: `notebooks/02_LTV_Cost_Sensitive_Thresholding.ipynb` verifies this directly — recalculated from scratch against the holdout's own real LTV distribution (a purely diagnostic exercise, never used as the operational decision), the optimum lands at exactly the same point (threshold = 0.03) as the one chosen on validation. See [Phase 4](#phase-4--ltv-weighted-threshold-analysis-on-holdout) for the full detail.

## Architecture

```
┌────────────────────────┐
│  make_dataset.py        │  10,000 synthetic customers + projected LTV
└────────────┬─────────────┘
             │
             ▼
┌────────────────────────┐
│       train.py           │  LightGBM + 50/30/20 split + MLflow (tracking +
│  (MLflow: local SQLite)  │  registry) + financial-return threshold search
└────────────┬─────────────┘
             │
    ┌────────┴─────────┐
    │  data/processed/   │  churn_model.joblib + model_metadata.json
    └────────┬─────────┘
             │
   ┌─────────┴──────────┐
   ▼                     ▼
┌─────────────┐   ┌──────────────────┐
│  api/main.py │◄──┤ app/streamlit_app │  ROI simulator (recalculates the
│  (FastAPI)   │   │  .py              │  threshold live) + single prediction (via API)
└─────────────┘   └──────────────────┘
       │                    │
       └─────────┬──────────┘
                  ▼
        docker-compose.yml
   (multi-stage Dockerfile, one target per service)
```

`notebooks/02_LTV_Cost_Sensitive_Thresholding.ipynb` runs alongside this pipeline, consuming the trained model and `src/models/train.py` directly to dig into the holdout's LTV distribution and the gains curve (§Phase 4).

## The process (4 phases)

This project was built in phases, delivered and verified one at a time — not written all at once and assumed to work.

### Phase 1 — Data and modeling

`src/data/make_dataset.py` generates 10,000 synthetic bank customers with a logistic churn-risk model (tenure, satisfaction, complaints, activity, number of products) and a **projected LTV** (current monthly revenue × expected future months of useful life, by activity segment).

`src/models/train.py` trains LightGBM on a 50/30/20 split (train / val / test), tracks everything in MLflow (local SQLite, with a working Model Registry), and — the heart of the project — searches for the **decision threshold that maximizes net financial return** on validation, not the one that maximizes F1 or accuracy.

**This didn't work on the first try.** The first version of the optimal threshold was degenerate: 0.01, basically "contact everyone." The reason, once diagnosed: given this problem's cost asymmetry (not contacting a real churner costs their entire LTV; contacting someone who wasn't going to churn only costs the campaign price), the mathematical indifference point is `cost / (LTV × (1 + success_rate))` — and with a trivial campaign cost relative to the LTV at stake, that point falls below almost any predicted probability, so "contact everyone" wins mathematically without the model contributing anything. Adjusting the cost alone wasn't enough either: pushing it too high made the value negative at *every* threshold. The fix was two combined changes: a realistic campaign cost ($100, a real offer — fee discount, bonus — not a trivial phone call) and a less noisy risk model (ROC-AUC 0.75 → 0.82), until the optimum landed in a genuinely interior, profitable region. The full detail is commented in `src/models/train.py`.

### Phase 2 — API and UI

`src/api/main.py`: FastAPI + Pydantic v2, with `lifespan` (not the deprecated `on_event`) to load the model at startup. Before writing it I empirically verified a non-obvious LightGBM-in-production detail: categorical columns need to be cast to `dtype="category"` on every request (otherwise it fails with `"train and valid dataset categorical_feature do not match"`), but **not** replicating the exact category levels seen during training is fine — the booster keeps its own internal mapping (`pandas_categorical`) and applies it on its own. Endpoints: `/health`, `/model/info`, `/predict`, `/predict/batch`.

`src/app/streamlit_app.py`: two tabs. **ROI simulator**, with sliders for campaign cost and success rate that recalculate the optimal threshold and return live over the 10,000 customers. **Single prediction**, which calls the real API (doesn't load the model on its own) — genuine microservices architecture.

### Phase 3 — Containerization and tests

Multi-stage `Dockerfile`: `builder` (compiles with `build-essential`/`cmake`) → `runtime` (minimal shared base, with `libgomp1` installed — without that library, LightGBM fails to import on Debian slim with `libgomp.so.1: cannot open shared object file`, a real gotcha worth documenting) → `api` and `app` as final targets of the same Dockerfile, so the dependency definition isn't duplicated. `docker-compose.yml` brings up both services, mounts `data/` as a read-only volume (model artifacts aren't baked into the image), and uses a `curl`-free healthcheck (via the stdlib's `urllib`) so `app` waits for `api` to actually be ready.

`tests/`: `pytest` integration suite. `conftest.py` auto-generates the dataset and trains the model if they don't exist — verified for real by deleting every artifact and running `pytest tests/` from scratch (16/16 green), not just assumed.

**Verifying Docker without an available daemon.** This execution environment doesn't have Docker installed (neither Docker Desktop nor the CLI), so `docker compose up --build` itself couldn't literally be run. Rather than leaving it at "review it before a real deployment" and moving on, everything that *can* be validated without the daemon was validated, by replicating every Dockerfile stage outside the container:

1. **Clean `requirements.txt` install into an isolated venv** — replicates exactly what the `builder` stage does; no dependency conflicts.
2. **The exact `CMD` commands from the Dockerfile, run as-is** — `uvicorn src.api.main:app --host 0.0.0.0 --port 8000` and `streamlit run src/app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true` both start and serve correctly in that isolated venv.
3. **The exact `healthcheck` command** from `docker-compose.yml` (`python -c "import urllib.request; urllib.request.urlopen(...)"`) run against the live API — responds `{"status":"ok","model_loaded":true}`.
4. **The full end-to-end pipeline** (`make_dataset.py` → `train.py` → serving with the real `CMD`) run in the isolated venv, with no artifacts inherited from another install.
5. **All 16 integration tests**, run from scratch against the freshly trained model.

This doesn't replace an actual `docker compose up --build` — the `COPY` paths, the multi-stage ordering, and the Docker engine itself still haven't literally run — but it covers exactly the code and commands that end up running inside the container, with far more confidence than "it's written carefully." **Recommendation**: before a real deployment, run `docker compose up --build` once on a machine with Docker installed as a final verification step.

### Phase 4 — LTV-weighted threshold analysis on holdout

`notebooks/02_LTV_Cost_Sensitive_Thresholding.ipynb` (executed end to end, zero errors) digs into three questions `train.py` doesn't show in detail:

1. **How is LTV distributed on the holdout?** The average real churner is worth $2,053, with a heavy tail (standard deviation ~$1,730, the same order of magnitude as the mean) — consistent with why the threshold is chosen on a sizeable (30%) validation set before ever touching the holdout.
2. **Is the optimal threshold an artifact of the split, or does it generalize?** Recalculated directly against the holdout's own LTV distribution (purely diagnostic — the real operational decision remains the validation-derived one), the optimum lands at **exactly the same threshold (0.03)** chosen on validation, with a financial-value difference of $0.00. Direct evidence the threshold isn't overfit to one particular split.
3. **Gains curve: retention cost vs. LTV saved.** Ranking the holdout from highest to lowest predicted risk, the model recovers substantially more LTV for the same campaign cost than random ordering does — the real lift, not just an abstract AUC. The exact optimum of the full financial objective (swept customer by customer, including the cost of failing to act on a real churner) lands within 12 customers of the operational threshold chosen by `train.py`'s 0.01-granularity grid search — that search's granularity isn't leaving meaningful value on the table.

## Project structure

```
customer-churn-mlops-platform/
├── data/
│   ├── raw/                        # customers.csv (generated, not versioned)
│   └── processed/                  # churn_model.joblib, model_metadata.json,
│                                    #   figures/threshold_vs_value.png (generated)
├── src/
│   ├── data/
│   │   └── make_dataset.py         # Generates 10,000 synthetic customers + LTV
│   ├── models/
│   │   └── train.py                # LightGBM + MLflow + financial-return threshold
│   ├── api/
│   │   └── main.py                 # FastAPI: /predict, /predict/batch, /model/info
│   └── app/
│       └── streamlit_app.py        # ROI simulator + single prediction
├── notebooks/
│   └── 02_LTV_Cost_Sensitive_Thresholding.ipynb  # LTV on holdout + gains curve
├── tests/
│   ├── conftest.py                 # Auto-bootstraps dataset + model for CI
│   ├── test_data_and_training.py
│   └── test_api.py
├── docs/
│   └── streamlit_preview.png
├── Dockerfile                      # Multi-stage: builder -> runtime -> api / app
├── docker-compose.yml
├── requirements.txt
├── LICENSE
├── README.md
└── README.es.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Generate the synthetic dataset
python -m src.data.make_dataset

# 2. Train (LightGBM + MLflow + financial threshold optimization)
python -m src.models.train

# 3a. Bring up the API
uvicorn src.api.main:app --reload

# 3b. Bring up the UI (in another terminal; waits for the API to be running)
streamlit run src/app/streamlit_app.py

# View the experiments tracked in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Integration tests (auto-generates dataset/model if missing)
pytest tests/ -v
```

### Analysis notebook (optional, not part of the Docker image)

```bash
pip install jupyter ipykernel nbconvert
jupyter nbconvert --to notebook --execute --inplace notebooks/02_LTV_Cost_Sensitive_Thresholding.ipynb
```

Jupyter is deliberately not added to `requirements.txt`: that file is what the `Dockerfile` installs inside the `api`/`app` images, and an analysis notebook has no business being inside a production service container.

## Docker deployment

```bash
# Builds both images (api, app) and brings up the two services
docker compose up --build

# API:       http://localhost:8000/docs  (interactive Swagger UI)
# Streamlit: http://localhost:8501
```

What `docker-compose.yml` does exactly:

- **`api`** builds from the Dockerfile's `api` target (image `churn-api:latest`), exposes port `8000`, mounts `./data` as a read-only volume (`:ro`) — the trained model is read from the host, not baked into the image — and exposes a `healthcheck` that hits `/health` every 10s (with a 15s `start_period`) using the stdlib's `urllib`, without depending on `curl` being installed in the minimal image.
- **`app`** builds from the `app` target (image `churn-streamlit:latest`), exposes port `8501`, also mounts `./data` read-only, receives `API_URL=http://api:8000` as an environment variable to talk to the API by its Docker Compose service name (not `localhost`), and has `depends_on: api: condition: service_healthy` — it won't start until `api`'s healthcheck is green.

Useful commands once it's up:

```bash
docker compose ps                 # status of both services
docker compose logs -f api        # live API logs
curl http://localhost:8000/health # {"status":"ok","model_loaded":true}
docker compose down               # stops and removes the containers (not mounted volumes)
```

**Prerequisite**: `data/processed/churn_model.joblib` and `model_metadata.json` must exist on the host before bringing up the containers (run `python -m src.data.make_dataset` and `python -m src.models.train` first) — the containers *serve* the trained model, they don't train it; the read-only volume is deliberate, it separates the model artifact's lifecycle from the application code's lifecycle.

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status and whether the model is loaded |
| `GET` | `/model/info` | Active model metadata: optimal threshold, AUC, business assumptions |
| `POST` | `/predict` | Prediction for one customer; if `ltv` is sent, adds the expected financial value |
| `POST` | `/predict/batch` | Prediction for a list of customers |

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "credit_score": 550, "geography": "West", "gender": "Male", "age": 35,
  "tenure_years": 0, "balance": 500, "num_products": 1, "has_credit_card": false,
  "is_active_member": false, "estimated_salary": 40000, "num_complaints": 5,
  "satisfaction_score": 1, "monthly_fee_revenue": 25.0, "ltv": 450
}'
# {"churn_probability":0.9741,"optimal_threshold":0.03,"recommended_action":"contact","expected_financial_value":31.51}
```

## Tech stack

| Tool | Role |
|---|---|
| **LightGBM** | Churn classification model (native categoricals) |
| **MLflow** | Experiment tracking, business metrics, and Model Registry (local SQLite) |
| **FastAPI + Pydantic v2** | Inference API with strict schema validation |
| **Streamlit** | ROI simulator UI and single prediction |
| **Docker** | Multi-stage image, one `target` per service |
| **pytest** | Integration suite (data, financial logic, real API) |
| **Jupyter / nbconvert** | `notebooks/02_LTV_Cost_Sensitive_Thresholding.ipynb`, executed end to end and committed with real outputs (§Phase 4) |
| **pandas / numpy / scikit-learn** | Data preparation and modeling utilities |

## Known limitations

- The data is synthetic (a custom generator, no connection to a real banking dataset) — the risk pattern and LTV are design assumptions, documented and calibrated, not observations.
- The optimal threshold depends on `campaign_cost` and `retention_success_rate`, which are adjustable business assumptions (that's why the ROI simulator exists: to explore what happens if they change).
- Threshold selection on a moderately sized validation set is sensitive to LTV's heavy tail — a handful of high-value customers can move the total by tens of thousands of dollars. §Phase 4 measures this sensitivity directly (comparing the validation threshold against the one recalculated on holdout) instead of just warning about it, and finds it stable on this dataset; a production system should keep monitoring this stability against real data.
- Docker verification in this environment was by equivalence (isolated venv + the Dockerfile's exact commands), not a literal `docker compose up --build` run — see §Phase 3 for the exact detail of what was validated and what still needs confirming on a machine with Docker.

## Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)

Code: MIT — see [LICENSE](LICENSE).
