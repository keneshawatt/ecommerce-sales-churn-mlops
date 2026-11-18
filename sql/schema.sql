-- ============================================================
-- E-Commerce Sales Analytics — Star Schema
-- Run this once against your PostgreSQL database before ETL.
-- ============================================================

-- Raw / staging layer (loose typing, mirrors source file)
CREATE TABLE IF NOT EXISTS staging_sales (
    invoice_no      VARCHAR(20),
    stock_code      VARCHAR(20),
    description     TEXT,
    quantity        INTEGER,
    invoice_date    TIMESTAMP,
    unit_price      NUMERIC(10,2),
    customer_id     VARCHAR(20),
    country         VARCHAR(100),
    source_file     VARCHAR(200),
    load_ts         TIMESTAMP DEFAULT NOW()
);

-- Dimension: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_key        INTEGER PRIMARY KEY,      -- YYYYMMDD
    full_date       DATE NOT NULL,
    day             INTEGER,
    month           INTEGER,
    month_name      VARCHAR(15),
    quarter         INTEGER,
    year            INTEGER,
    day_of_week     VARCHAR(15),
    is_weekend      BOOLEAN
);

-- Dimension: Customer
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key    SERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) UNIQUE NOT NULL,
    country         VARCHAR(100),
    first_purchase_date DATE,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- Dimension: Product
CREATE TABLE IF NOT EXISTS dim_product (
    product_key     SERIAL PRIMARY KEY,
    stock_code      VARCHAR(20) UNIQUE NOT NULL,
    description     TEXT,
    category        VARCHAR(100)              -- derived/bucketed if you choose to add one
);

-- Dimension: Geography
CREATE TABLE IF NOT EXISTS dim_geography (
    geography_key   SERIAL PRIMARY KEY,
    country         VARCHAR(100) UNIQUE NOT NULL,
    region          VARCHAR(100)               -- optional manual/enrichment mapping
);

-- Fact: Sales (grain = one line item per invoice line)
CREATE TABLE IF NOT EXISTS fact_sales (
    sales_key       BIGSERIAL PRIMARY KEY,
    invoice_no      VARCHAR(20) NOT NULL,
    date_key        INTEGER REFERENCES dim_date(date_key),
    customer_key    INTEGER REFERENCES dim_customer(customer_key),
    product_key     INTEGER REFERENCES dim_product(product_key),
    geography_key   INTEGER REFERENCES dim_geography(geography_key),
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(10,2) NOT NULL,
    revenue         NUMERIC(12,2) NOT NULL,
    load_ts         TIMESTAMP DEFAULT NOW(),
    UNIQUE (invoice_no, product_key)           -- crude duplicate guard at line-item grain
);

-- Fact: Customer Retention (recomputed per ETL run / per period)
CREATE TABLE IF NOT EXISTS fact_customer_retention (
    retention_key   BIGSERIAL PRIMARY KEY,
    customer_key    INTEGER REFERENCES dim_customer(customer_key),
    snapshot_date   DATE NOT NULL,
    recency_days    INTEGER,
    frequency       INTEGER,
    monetary        NUMERIC(12,2),
    is_repeat_customer BOOLEAN,
    UNIQUE (customer_key, snapshot_date)
);

-- Data-quality / rejected records log
CREATE TABLE IF NOT EXISTS etl_error_log (
    error_id        BIGSERIAL PRIMARY KEY,
    run_id          VARCHAR(50),
    source_file     VARCHAR(200),
    row_data        TEXT,
    error_reason    VARCHAR(500),
    logged_at       TIMESTAMP DEFAULT NOW()
);

-- ETL run metadata (extraction date, source, status, row count — required by spec)
CREATE TABLE IF NOT EXISTS etl_run_log (
    run_id          VARCHAR(50) PRIMARY KEY,
    source_name     VARCHAR(200),
    started_at      TIMESTAMP,
    finished_at     TIMESTAMP,
    status          VARCHAR(20),               -- SUCCESS / FAILED / PARTIAL
    rows_extracted  INTEGER,
    rows_loaded     INTEGER,
    rows_rejected   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_fact_sales_date ON fact_sales(date_key);
CREATE INDEX IF NOT EXISTS idx_fact_sales_customer ON fact_sales(customer_key);
CREATE INDEX IF NOT EXISTS idx_fact_sales_product ON fact_sales(product_key);
