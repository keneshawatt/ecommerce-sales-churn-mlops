# Viva Preparation: likely questions and short answers

Answer in your own words. These are the points to cover, written to match what you built.

## The 30-second summary

I built a pipeline for an online retailer's transactions. Data is fetched from the UCI API,
split into monthly files, and loaded by an Airflow DAG into a PostgreSQL star schema on Neon,
with bad rows logged rather than dropped. A Streamlit dashboard shows sales and retention. In
Part 2 I trained a churn model, tracked it in MLflow, served it through FastAPI in Docker, and
added monitoring and a retraining plan.

## Data and ingestion

**Why the UCI Online Retail dataset?** It is the primary dataset named in the assignment, it is
public (CC BY 4.0), and it has invoices, products, customers, countries, quantities and prices:
541,909 transaction lines from December 2010 to December 2011.

**How is the data fetched?** Through the `ucimlrepo` Python package (`fetch_ucirepo(id=352)`), so
anyone can reproduce it without a manual download. An untouched raw copy is saved.

**How did you simulate incremental ingestion?** `ingest.py` splits the data by invoice month into
landing files such as `Apr_2011.csv`, and Airflow processes one file per task, in order.

**What do you log?** `ingestion_log.jsonl` records the run, source, timestamp, status and row
count. The `etl_run_log` table records rows extracted, loaded and rejected for each ETL run.

## Airflow

**What is a DAG?** A directed acyclic graph of tasks with dependencies. Mine is a chain: one task
per monthly file, each running after the previous one.

**Why Airflow?** Scheduling, ordering, retries and visibility of every task's logs. The assignment
asks for an Airflow DAG or an approved equivalent.

**Why BashOperator?** It runs the same `etl.py` script I can run by hand, and passes the database
credentials in as an environment variable.

**What happens when a task fails?** Airflow retries it (the DAG sets one retry), the log shows the
error, and later tasks do not run.

**Did you have problems?** Yes. One task once ran for about 49 minutes with no output and was
killed. The full load completed afterwards. My suspected cause was a stuck session on the
serverless database. Earlier, tasks failed because the DAG could not find the `.env` file when
Airflow parsed it; I fixed that by loading it from an explicit path.

## ETL and data quality

**What cleaning rules do you apply?** Cancelled invoices (starting with `C`), quantity or price
at or below zero or missing, and missing customer ID are rejected. Country names are trimmed and
title-cased, and dates are parsed.

**Why remove cancelled orders?** They are returns, with negative quantities, and would distort
revenue. They are logged, so they are still visible.

**Why log rejected rows instead of dropping them?** So every data quality decision can be audited,
and so I can show how many rows were rejected and why.

**How do you prevent duplicates?** `fact_sales` has a unique constraint on `(invoice_no,
product_key)` and loads with `ON CONFLICT DO NOTHING`, so reprocessing a file cannot double-count.
The rejected-row log replaces a file's earlier rejects when that file is processed again.

**What was your staging layer?** The raw and landing files, plus temporary staging tables that the
cleaned rows are written to before being merged into the warehouse.

**What bug did you find?** After several reruns the error log held more rows than the source data,
because it had no duplicate protection. I truncated the tables, reran cleanly, and then made the
log rerun-safe.

## Warehouse

**What is a star schema?** One fact table surrounded by dimension tables. `fact_sales` holds the
measures (quantity, price, revenue) and the dimensions (customer, product, geography, date) hold
descriptions.

**What is the grain?** One row per invoice line item.

**Why surrogate keys?** They are small integers, stable if a natural ID changes, and faster to
join.

**Why a star schema and not one flat table?** It avoids repeating customer and product details on
every row and makes aggregate queries simple and fast.

**What are the marts?** Views on top of the star schema. `sales_mart` serves the revenue views and
`customer_mart` serves segmentation and retention.

**What is RFM?** Recency (days since last purchase), frequency (number of orders) and monetary
value (total spend). They are standard customer-behaviour features and the base of the churn model.

**What is the retention fact table?** `fact_customer_retention` holds one row per customer as of
the last transaction date: recency, frequency, monetary value and a repeat-customer flag.

**Why Neon?** It is real PostgreSQL in the cloud with no local setup, always reachable from
Airflow and from Docker containers. The free tier sleeps when idle, so my connections use
`pool_pre_ping` and `pool_recycle`.

## Dashboard

**Why Streamlit and not Tableau?** The assignment allows either. Streamlit supports interactive
filtering and lets the dashboard call the model API, which Tableau would not do directly.

**How are credentials handled?** In a `.env` file that is never committed or copied into an image.
Nothing is hardcoded.

## The model

**How is churn defined?** A customer who bought before a scoring date is churned if they make no
purchase in the following 60 days. The data ends on 9 December 2011, so a 60-day window leaves
several scoring dates with fully known outcomes.

**Why a chronological split?** Random splitting would let the model learn from the future. I
score on 9 June (train), 9 August (validation) and 9 October (test), and every feature uses only
data before its scoring date.

**What are the features?** Recency, frequency, monetary value, average order value, tenure, and
orders in the last 90 days.

**Why three models, and why did Logistic Regression win?** The assignment asks for a comparison.
Logistic Regression and Random Forest tied on validation ROC-AUC (0.739), and Logistic Regression
is simpler and faster to serve. XGBoost had the highest recall (0.87) but the lowest precision.

**What do precision and recall mean here?** Precision: of the customers flagged as churners, how
many really churned. Recall: of the customers who really churned, how many were flagged. Which
matters more depends on whether a missed churner or a wasted retention offer costs more.

**Is the model good?** It is modest. Test ROC-AUC is 0.734, precision 0.670, recall 0.730. It ranks
customers better than chance and is useful for prioritising outreach, not for automatic decisions.

**Why is the churn rate so high (52 to 60%)?** Many customers in this dataset buy only once, and
a 60-day window is short for a gift retailer.

**What is class weighting?** `class_weight="balanced"` makes the model treat both classes as
equally important during training.

## MLflow

**Tracking versus registry?** Tracking records each training run's parameters, metrics and
artifacts. The registry stores the chosen model under a name with numbered versions.

**Why are there two versions?** I ran the training script twice on the same data, so versions 1
and 2 are equivalent. The service loads the newest.

**Why export the model to a folder?** A container cannot see my laptop's MLflow database, so the
registered model is exported with its pinned library versions and baked into the image.

## API and Docker

**What does the API do?** `/predict` validates the six features (all must be numbers at or above
zero, otherwise a 422 error), returns the churn probability and label, and logs every request.
`/health` reports the model version.

**Image versus container?** An image is the packaged recipe, and a container is a running copy.
`docker compose` starts the API and the dashboard together.

**Why Python 3.13 in the image?** The model is saved with cloudpickle, which needs the same Python
version it was trained with.

**How do secrets stay out of the images?** `.dockerignore` excludes `.env`, and the dashboard
container receives the database URL at run time.

## Monitoring and retraining

**Drift versus performance monitoring?** Drift asks whether the inputs still look like the
training data. Performance asks whether the predictions are still accurate. Drift can appear
first, because outcomes arrive 60 days late.

**What is PSI?** The Population Stability Index compares how a feature's values are spread across
bins in training versus now. Below 0.10 is stable, 0.10 to 0.25 a warning, above 0.25 significant.
I also run a Kolmogorov-Smirnov test.

**Why did you simulate drifted traffic?** To show the monitoring works. Shifted customers raised
alerts on four features and the report recommended retraining.

**Why can't you compute live precision and recall?** The prediction log has no customer IDs, and
true outcomes only arrive 60 days later. Instead I re-score recent periods with known outcomes.

**When do you retrain?** When two or more features have PSI above 0.25, when precision or recall
falls below 0.60, or on a schedule. A new version is promoted only if it is at least as good on the
newest labeled window.

**What did the performance numbers show?** Precision slid from 0.732 to 0.670 across three
windows while recall held. It is still above the floor, but it is a trend to watch.

## Mistakes and lessons

- **Leaked credentials:** a database password ended up in code and in chats. I rotated it and moved
  all credentials into `.env`. Lesson: never hardcode secrets, and check what you paste.
- **Environment loading:** `.env` was not found under Airflow's working directory. Fixed with an
  explicit path.
- **Stale connections:** the serverless database sleeps, which broke a cached connection. Fixed
  with `pool_pre_ping`.
- **Pooled endpoint:** Neon's pooled connection rejects the `statement_timeout` startup option.
- **Rerun duplication:** the rejected-row log had no duplicate protection. Fixed.

## Limitations and next steps

One year of data from one retailer, so seasonality and drift cannot be separated. Modest model
performance. The 0.5 decision threshold is not tuned to business costs. Monitoring is run by hand
and could be scheduled weekly with Airflow. Next steps would be richer features (product
categories, return history), threshold tuning, and a customer ID in the prediction log.
