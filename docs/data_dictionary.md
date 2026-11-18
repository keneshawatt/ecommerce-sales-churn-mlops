# Data Dictionary: E-Commerce Sales Analytics and Churn Prediction

## 1. Data source

**UCI Online Retail Dataset.** Transaction-level data for a UK-based online gift retailer,
1 December 2010 to 9 December 2011, 541,909 line items.

- Access: fetched programmatically with the `ucimlrepo` Python package
  (`fetch_ucirepo(id=352)`, script `scripts/fetch_data.py`). The page is
  https://archive.ics.uci.edu/dataset/352/online+retail
- Licence: Creative Commons Attribution 4.0 (free to use with credit).
- Citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5BW33

| Raw column | Description |
|------------|-------------|
| InvoiceNo | Invoice number. A leading `C` marks a cancellation |
| StockCode | Product code |
| Description | Product name |
| Quantity | Units on the line |
| InvoiceDate | Date and time of the transaction |
| UnitPrice | Price per unit in pounds sterling |
| CustomerID | Customer identifier (missing for some rows) |
| Country | Customer's country |

## 2. Data layers

| Layer | Contents | Location |
|-------|----------|----------|
| Raw | Untouched copy of the fetched dataset, stamped with a timestamp | `data/raw/` |
| Landing | Raw data split into monthly files (for example `Apr_2011.csv`), simulating incremental monthly extracts | `data/landing/` |
| Staging | Cleaned rows are written to temporary staging tables (`temp_cust`, `temp_prod`, `temp_geo`, `temp_date`, `temp_fact`) and merged into the warehouse with `INSERT ... ON CONFLICT`, then dropped | Neon PostgreSQL |
| Cleaned and analytical | Star schema tables plus the sales and customer marts | Neon PostgreSQL |
| Logs | Ingestion log, ETL run log, rejected-record log, prediction log | `data/logs/`, Neon, `logs/` |

## 3. Star schema (as defined in `sql/schema.sql`)

### dim_customer
| Column | Type | Notes |
|--------|------|-------|
| customer_key | SERIAL, primary key | Surrogate key |
| customer_id | VARCHAR(20), unique | Natural key from the source |
| country | VARCHAR(100) | Country at first load |
| first_purchase_date | DATE | |
| is_active | BOOLEAN | |
| created_at, updated_at | TIMESTAMP | Load audit columns |

### dim_product
| Column | Type | Notes |
|--------|------|-------|
| product_key | SERIAL, primary key | Surrogate key |
| stock_code | VARCHAR(20), unique | Natural key |
| description | TEXT | Truncated to 500 characters on load |
| category | VARCHAR(100) | Reserved for a future product-category enrichment |

### dim_geography
| Column | Type | Notes |
|--------|------|-------|
| geography_key | SERIAL, primary key | Surrogate key |
| country | VARCHAR(100), unique | Title-cased country name |
| region | VARCHAR(100) | Reserved for a future regional grouping |

### dim_date
| Column | Type | Notes |
|--------|------|-------|
| date_key | INTEGER, primary key | `YYYYMMDD` |
| full_date | DATE | |
| day, month, quarter, year | INTEGER | |
| month_name, day_of_week | VARCHAR | |
| is_weekend | BOOLEAN | |

### fact_sales
Grain: one row per invoice line item.

| Column | Type | Notes |
|--------|------|-------|
| sales_key | BIGSERIAL, primary key | |
| invoice_no | VARCHAR(20) | |
| date_key, customer_key, product_key, geography_key | INTEGER | Foreign keys to the dimensions |
| quantity | INTEGER | Always greater than 0 after validation |
| unit_price | NUMERIC(10,2) | Always greater than 0 after validation |
| revenue | NUMERIC(12,2) | `quantity * unit_price`, derived at load time |
| load_ts | TIMESTAMP | |

Unique constraint on `(invoice_no, product_key)` prevents duplicate lines when a file is
reprocessed.

### fact_customer_retention
Grain: one row per customer per snapshot date. Filled by
`scripts/build_marts_and_retention.py`.

| Column | Type | Notes |
|--------|------|-------|
| retention_key | BIGSERIAL, primary key | |
| customer_key | INTEGER | Foreign key to `dim_customer` |
| snapshot_date | DATE | Last transaction date in the warehouse |
| recency_days | INTEGER | Days from the customer's last purchase to the snapshot date |
| frequency | INTEGER | Number of distinct invoices |
| monetary | NUMERIC(12,2) | Total revenue |
| is_repeat_customer | BOOLEAN | True when the customer has more than one invoice |

Unique constraint on `(customer_key, snapshot_date)`; the load is an upsert.

## 4. Operational tables

### etl_run_log (one row per ETL run)
| Column | Notes |
|--------|-------|
| run_id | Primary key |
| source_name | Landing file processed |
| started_at, finished_at | Run timestamps |
| status | `SUCCESS`, `FAILED` or `PARTIAL` |
| rows_extracted, rows_loaded, rows_rejected | Row counts for the run |

### etl_error_log (one row per rejected record)
| Column | Notes |
|--------|-------|
| error_id | Primary key |
| run_id, source_file | Which run and file the row came from |
| row_data | The rejected row, as text |
| error_reason | `cancelled_order`, `invalid_quantity`, `invalid_price` or `missing_customer_id` |
| logged_at | Timestamp |

## 5. Analytical marts (views, `sql/marts.sql`)

### sales_mart
Revenue by date, country and product: `full_date`, `year`, `month`, `month_name`, `country`,
`stock_code`, `description`, `total_quantity`, `total_revenue`, `num_orders`.

### customer_mart
One row per customer: `customer_id`, `country`, `first_purchase_date`, `last_purchase_date`,
`recency_days` (measured from the last transaction date in the data), `frequency`, `monetary`,
`is_repeat_customer`.

## 6. Model features (Part 2)

Computed per customer as of a scoring date, using only transactions on or before that date.

| Feature | Definition |
|---------|------------|
| recency_days | Days since the customer's last purchase |
| frequency | Distinct invoices so far |
| monetary | Total revenue so far |
| avg_order_value | `monetary / frequency` |
| tenure_days | Days since the customer's first purchase |
| orders_last_90d | Distinct invoices in the 90 days before the scoring date |
| **churn (label)** | 1 if the customer makes no purchase in the 60 days after the scoring date, otherwise 0 |

## 7. Prediction log (`logs/predictions.jsonl`)

One JSON object per API request: `timestamp`, `inputs` (the six features), `churn_probability`,
`prediction`, `model_version`, `latency_ms`, `status` (`success` or `failure`), and `error` for
failures.

## 8. Validation rules (applied in `etl.py`)

1. Rows whose `InvoiceNo` starts with `C` (cancellations) are excluded from `fact_sales` and
   logged with reason `cancelled_order`.
2. Rows with `Quantity` less than or equal to 0, or missing, are rejected (`invalid_quantity`).
3. Rows with `UnitPrice` less than or equal to 0, or missing, are rejected (`invalid_price`).
4. Rows with a missing `CustomerID` are rejected (`missing_customer_id`), since they cannot be
   attached to the customer dimension or used for churn labels.
5. Country names are trimmed and title-cased, and dates are parsed to timestamps.
6. Duplicate prevention: the `(invoice_no, product_key)` unique constraint together with
   `ON CONFLICT DO NOTHING` makes reruns safe.
7. Every run records its extracted, loaded and rejected row counts in `etl_run_log`.
8. API inputs are validated by the service: all six features must be numbers greater than or
   equal to 0, otherwise the request is rejected with a 422.
