# MLOps Upgrade — Step-by-Step Guide

Turns your `xgb_flask_api` project into a production ML story with:
**MLflow** (experiment tracking) · **pytest** (testing) · **GitHub Actions** (CI/CD) · **AWS App Runner** (cloud deploy) · **Evidently** (drift monitoring).

Estimated time: 15–20 hours. Do the phases in order — each builds on the last.

---

## File map (what goes where in your repo)

```
xgb_flask_api/
├── train.py                     # NEW — training + MLflow logging
├── app.py                       # UPDATED — adds /health + latency logging
├── benchmark.py                 # NEW — p50/p95/p99 latency measurement
├── requirements.txt             # UPDATED — mlflow, evidently, pytest, gunicorn
├── Dockerfile                   # UPDATED — gunicorn, port 8080
├── .dockerignore                # NEW
├── tests/
│   ├── __init__.py
│   ├── test_features.py         # NEW — leakage + feature-logic tests
│   └── test_api.py              # NEW — endpoint tests
├── monitoring/
│   ├── drift_report.py          # NEW — Evidently drift + retrain trigger
│   └── reference_data.csv       # auto-created by train.py
└── .github/
    └── workflows/
        └── ci.yml               # NEW — CI/CD pipeline
```

Copy these files into your repo, keeping the folder structure. Put your
`data/cleaned_sales.csv` under a `data/` folder.

---

## Phase 1 — MLflow experiment tracking (~3 hrs)

**Goal:** every training run logs its params, metrics, and model, so you can
compare experiments in a UI.

1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training (this logs to a local `./mlruns` folder):
   ```bash
   python train.py
   ```
   You'll see it print the test RMSE and improvement, and write
   `final_xgb_model.json`, the encoders, and `monitoring/reference_data.csv`.

3. Open the MLflow UI to see your run:
   ```bash
   mlflow ui
   ```
   Go to `http://127.0.0.1:5000`. You'll see the **demand-forecasting**
   experiment with logged params (depth, lr, ...) and metrics (test_rmse,
   improvement_over_baseline_pct, ...).

4. **Create an experiment comparison** (the resume-worthy artifact): change a
   hyperparameter or `SUBSET_SIZE`, run `python train.py` again, then in the UI
   select both runs and click **Compare**. Screenshot it. You now have
   "tracked and compared experiments with MLflow."

**Interview line:** *"I used MLflow to track every run's params, metrics, and
model artifact, so I could compare configurations objectively instead of eyeballing notebook outputs."*

---

## Phase 2 — Testing with pytest (~2 hrs)

**Goal:** automated tests, especially for the leakage-prone feature logic.

1. Run the suite:
   ```bash
   pytest -v
   ```

2. What's covered:
   - `test_rolling_mean_excludes_current_day` — **the important one.** It proves
     your rolling feature doesn't leak today's sales (asserts the value is 6.0,
     not 7.0). If someone ever removes the `.shift(1)`, this test fails loudly.
   - `test_lag_7_is_seven_days_back` — lag correctness.
   - `test_no_cross_series_leakage` — lags don't bleed between products.
   - `test_api.py` — `/health` works, `/predict` returns a non-negative number,
     bad input returns 400. (These auto-skip if the model isn't built yet.)

**Interview line:** *"I wrote unit tests around the feature engineering,
including a regression test that specifically guards against reintroducing
target leakage in the rolling features."*

---

## Phase 3 — CI/CD with GitHub Actions (~2 hrs)

**Goal:** every push runs tests and builds the Docker image automatically.

1. Commit everything and push to GitHub:
   ```bash
   git add .
   git commit -m "Add MLflow, tests, CI/CD, monitoring"
   git push
   ```

2. On GitHub, open the **Actions** tab. You'll see the **CI/CD** workflow run:
   - **test** job — installs deps, runs `pytest`.
   - **build** job — builds the Docker image (only if tests pass).

3. The green check on your commits is the visible proof of CI/CD. Add a badge to
   your README:
   ```markdown
   ![CI](https://github.com/<you>/xgb_flask_api/actions/workflows/ci.yml/badge.svg)
   ```

**Interview line:** *"I set up GitHub Actions so every push runs the test suite
and builds the container — tests gate the build, so broken code never ships."*

---

## Phase 4 — Deploy to AWS App Runner (~4–5 hrs)

**Goal:** a live public `/predict` URL. App Runner is the simplest path — it
runs a container with no servers to manage.

### 4a. Build & push the image to ECR
```bash
# set your region + account
export AWS_REGION=us-east-1
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REPO=demand-forecast-api

# create the ECR repo (once)
aws ecr create-repository --repository-name $REPO --region $AWS_REGION

# log in, build, tag, push
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $REPO .
docker tag $REPO:latest $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO:latest
```

### 4b. Create the App Runner service (console is easiest)
1. AWS Console → **App Runner** → **Create service**.
2. Source: **Container registry** → **Amazon ECR** → pick your image.
3. Deployment: **Automatic** (redeploys when the image updates).
4. Port: **8080** (matches the Dockerfile).
5. Health check path: **/health**.
6. Create. After a few minutes you get a public URL like
   `https://xxxx.us-east-1.awsapprunner.com`.

### 4c. Test the live endpoint
```bash
curl https://<your-app-runner-url>/health
python benchmark.py --url https://<your-app-runner-url>/predict --n 500
```
`benchmark.py` prints **p50 / p95 / p99 latency**. Record the p95 — that's the
number to put on your resume ("served predictions at ~Xms p95").

**Interview line:** *"I containerized the model and deployed it on AWS App
Runner behind a health check, with a live /predict endpoint at ~X ms p95 latency."*

> **Cost note:** App Runner isn't free. Pause or delete the service when you're
> not demoing it to avoid charges.

---

## Phase 5 — Drift monitoring with Evidently (~3 hrs)

**Goal:** detect when incoming data drifts from training data, and trigger retraining.

1. `train.py` already saved `monitoring/reference_data.csv` (a sample of training data).

2. Run the drift report (with no live data, it simulates drift so you can see it work):
   ```bash
   python monitoring/drift_report.py
   ```
   It prints the number of drifted features and writes an HTML report to
   `monitoring/drift_report.html` — open it in a browser to see the
   feature-by-feature drift visualization.

3. On real production data, point it at a CSV of recent requests:
   ```bash
   python monitoring/drift_report.py --current data/recent_production.csv
   ```

4. **The retraining trigger:** the script exits with code `1` if more than 30%
   of features have drifted. That non-zero exit is what a scheduler (cron, a
   GitHub Actions scheduled job, or an Airflow DAG) uses to kick off retraining
   automatically.

**Interview line:** *"I added Evidently to monitor data drift between training
and production distributions, with a rule that triggers retraining when drift
crosses a threshold — closing the MLOps loop."*

---

## The full production story (say this in interviews)

> "I took the model from a notebook to a monitored production service. Training
> is tracked in MLflow so I can compare experiments. Every push runs a pytest
> suite — including a regression test that guards against feature leakage — and
> builds the container through GitHub Actions. The image deploys to AWS App
> Runner with a health check and a live /predict endpoint at ~X ms p95. And
> Evidently monitors data drift in production and flags when the model should be
> retrained. So it's the full loop: track, test, ship, serve, monitor, retrain."

**Keywords this unlocks on your resume:** MLflow, experiment tracking, model
registry, CI/CD, GitHub Actions, Docker, AWS, ECR, App Runner, model serving,
p95 latency, data drift, model monitoring, Evidently, automated retraining, pytest.

---

## Suggested order if you're short on time

If you can't do all five, this order gives the most resume value per hour:
1. **MLflow** (Phase 1) — cheap, high-signal.
2. **pytest + CI/CD** (Phases 2–3) — one push, big keyword payoff.
3. **Evidently** (Phase 5) — runs locally, no cloud cost.
4. **AWS deploy** (Phase 4) — highest effort/cost; do it last, and take a
   screenshot of the live endpoint so you can demo it even after you tear it down.
