# Model Lifecycle and Retraining Plan: Customer Churn Model

## 1. Purpose

The churn model flags customers who are unlikely to buy again soon, so the business can
act before they are lost. It is trained on the sales data warehouse built in Part 1, served
through a prediction API, and monitored for drift and performance.

**Churn definition.** A customer who has bought before a scoring date is *churned* if they
make **no purchase in the 60 days after that date**. The dataset ends on 9 December 2011, so
a 60-day window leaves several scoring dates whose outcome is fully known. That is what makes
a chronological train, validation and test split possible.

## 2. Lifecycle stages

| # | Stage | What happens | Where |
|---|-------|--------------|-------|
| 1 | Data ingestion and ETL | Monthly files are cleaned, validated and loaded into the PostgreSQL star schema by an Airflow DAG | `ingest.py`, `etl.py`, `dags/ecommerce_etl_dag.py`, Neon |
| 2 | Feature engineering | Six customer features are computed using only transactions on or before the scoring date: `recency_days`, `frequency`, `monetary`, `avg_order_value`, `tenure_days`, `orders_last_90d` | `mlops/train.py` (`build_snapshot`) |
| 3 | Training and evaluation | Logistic Regression, Random Forest and XGBoost are trained on the 9 Jun 2011 snapshot, compared on the 9 Aug 2011 snapshot, and the winner is checked once on the 9 Oct 2011 snapshot | `mlops/train.py` |
| 4 | Experiment tracking | Parameters, metrics and the model artifact of every run are logged to MLflow (experiment `churn-prediction`) | MLflow (`mlflow.db`) |
| 5 | Registration and versioning | The best candidate by validation ROC-AUC is registered as `churn-model`; every training run creates a new version | MLflow Model Registry |
| 6 | Packaging | The latest registered version is exported to a plain folder together with a pinned `requirements.txt`, then baked into a Docker image | `mlops/export_model.py`, `Dockerfile.api` |
| 7 | Serving | FastAPI service with `/predict` and `/health`, input validation, and a log line for every request | `mlops/api.py`, `docker-compose.yml` |
| 8 | Monitoring | A batch job checks service health, input data quality, input drift and performance on labeled periods, then applies the retraining rules | `mlops/monitor.py` |
| 9 | Retraining or rollback | If a trigger fires, the model is retrained, compared against the current one, and promoted or rejected | Section 5 |

## 3. Training and evaluation results

Splits are chronological and use no future information: each snapshot only sees data up to its
own scoring date.

| Split | Scoring date | Customers | Churn rate |
|-------|--------------|-----------|------------|
| Train | 2011-06-09 | 2,800 | 60.5% |
| Validation | 2011-08-09 | 3,204 | 58.3% |
| Test | 2011-10-09 | 3,721 | 51.9% |

Validation results, used to choose the model:

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|-----|---------|
| Logistic Regression | 0.732 | 0.752 | 0.742 | 0.739 |
| Random Forest | 0.708 | 0.819 | 0.759 | 0.739 |
| XGBoost | 0.683 | 0.874 | 0.767 | 0.702 |

**Selection rule:** highest validation ROC-AUC. Logistic Regression and Random Forest are almost
tied, and Logistic Regression was registered. It is also the simplest and fastest to serve.
XGBoost catches the most churners (highest recall) but is wrong more often when it flags a
customer, so the right choice depends on the cost of a missed churner versus a wasted
retention offer.

**Held-out test result (registered model):** precision 0.670, recall 0.730, F1 0.699,
ROC-AUC 0.734. The model ranks customers better than chance but is not highly confident, so it
is best used to prioritise outreach, not to make automatic decisions.

**Versions.** `churn-model` has versions 1 and 2. Both come from running the same training
code on the same data, so they are equivalent. The service always loads the newest version.

## 4. Monitoring plan

`mlops/monitor.py` is run against the prediction log (`logs/predictions.jsonl`) and the
warehouse. It saves `monitoring/report.json`.

| Check | How it is measured | Alert threshold |
|-------|--------------------|-----------------|
| Failures | Share of logged requests with status `failure` | above 2% |
| Latency | p95 of `latency_ms` in the log | above 500 ms |
| Missing inputs | Count of missing feature values in requests | any |
| Out-of-range inputs | Share of requests outside the min to max seen in training, per feature | above 5% |
| Input drift | Population Stability Index (PSI) of each feature against the training distribution, plus a Kolmogorov-Smirnov test | PSI above 0.25 is significant, 0.10 to 0.25 is a warning |
| Model performance | Precision and recall of the deployed model on recent scoring dates whose outcome is now known (2011-08-09, 09-09, 10-09) | precision or recall below 0.60 |

**Observed behaviour.** On normal traffic every feature had PSI below 0.07 and no alerts were
raised. On deliberately shifted traffic, `recency_days` (PSI 1.35), `orders_last_90d` (1.14),
`monetary` (0.43) and `avg_order_value` (0.41) all crossed the alert threshold, and the report
recommended retraining. The predicted churn rate also rose from 67.5% to 79.5%, which is the
expected direction for customers who bought long ago and have no recent orders.

Performance on labeled windows, for the deployed model:

| Scoring date | Precision | Recall | F1 | ROC-AUC |
|--------------|-----------|--------|-----|---------|
| 2011-08-09 | 0.732 | 0.752 | 0.742 | 0.739 |
| 2011-09-09 | 0.693 | 0.740 | 0.716 | 0.724 |
| 2011-10-09 | 0.670 | 0.730 | 0.699 | 0.734 |

Precision is drifting down slowly (0.732 to 0.670) while recall holds. Both stay above the 0.60
floor, so no alert is raised yet, but the trend should be watched. The August window is also the
validation period used to pick the model, so it is slightly optimistic.

## 5. Retraining criteria and procedure

**Retrain when any of these is true:**

1. **Drift:** two or more features have PSI above 0.25.
2. **Performance:** precision or recall falls below 0.60 on any labeled window.
3. **Schedule:** new months of data have been loaded and the training cutoffs are more than
   three months old. Customer tenure and recency shift as customers age, so a model trained on
   old cutoffs sees increasingly unfamiliar inputs.

Service alerts (failure rate, latency, missing or out-of-range inputs) call for an engineering
investigation first. They trigger retraining only if they point to a change in the data itself.

**Procedure:**

1. Load the new data through the Airflow DAG and confirm the row counts reconcile.
2. Move the train, validation and test cutoffs in `mlops/train.py` forward so they cover the
   newest data, keeping the 60-day outcome window fully inside the data.
3. Run `train.py`. This logs the new runs and registers a new version of `churn-model`.
4. **Promotion rule:** evaluate the new version and the current version on the same newest
   labeled window. Promote only if the new version meets the 0.60 precision and recall floors
   and its ROC-AUC is at least as good as the current one. Otherwise keep the current version.
5. Run `export_model.py`, rebuild the containers with `docker compose up --build`, and check
   `/health` (version number) and a sample `/predict` call.
6. Delete `monitoring/reference_features.csv` so the drift reference is rebuilt from the new
   training period, then run `monitor.py` once to confirm there are no alerts.

**Rollback.** Earlier versions stay in the MLflow registry. To roll back, load the earlier
version with `mlflow.sklearn.load_model("models:/churn-model/<version>")`, save it into
`models/churn-model`, and rebuild the container.

## 6. Limitations

- One year of data from a single retailer, so seasonality cannot be separated from drift.
- Churn outcomes are only known 60 days after the scoring date, so live performance always lags.
- The prediction log does not record customer IDs, so live precision and recall cannot be
  computed from it. Performance is instead measured by re-scoring recent periods with known
  outcomes.
- The decision threshold is the default 0.5 and has not been tuned to business costs.
- The monitoring job is run manually. It can be scheduled weekly through Airflow in the same
  way as the ETL DAG; this is a proposed extension and is not implemented.
- The model is saved in cloudpickle format, which can run code when loaded, so model files
  should only ever be loaded from sources you trust.
