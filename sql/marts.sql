-- ============================================================
-- Analytical marts, built on top of the star schema.
-- Safe to re-run: the views are dropped and recreated, so an older view with a
-- different column layout cannot block the rebuild.
-- ============================================================

DROP VIEW IF EXISTS sales_mart;
DROP VIEW IF EXISTS customer_mart;

-- Sales mart: revenue trends, top products, country breakdown
CREATE VIEW sales_mart AS
SELECT
    d.full_date,
    d.year,
    d.month,
    d.month_name,
    g.country,
    p.stock_code,
    p.description,
    SUM(f.quantity)  AS total_quantity,
    SUM(f.revenue)   AS total_revenue,
    COUNT(DISTINCT f.invoice_no) AS num_orders
FROM fact_sales f
JOIN dim_date d       ON f.date_key = d.date_key
JOIN dim_geography g  ON f.geography_key = g.geography_key
JOIN dim_product p    ON f.product_key = p.product_key
GROUP BY d.full_date, d.year, d.month, d.month_name, g.country, p.stock_code, p.description;

-- Customer mart: RFM features for segmentation and retention.
-- Recency is measured from the last transaction date in the data (the dataset's
-- "today"), not from CURRENT_DATE, so it stays meaningful for a historical dataset.
CREATE VIEW customer_mart AS
SELECT
    c.customer_id,
    c.country,
    MIN(d.full_date) AS first_purchase_date,
    MAX(d.full_date) AS last_purchase_date,
    ((SELECT MAX(full_date) FROM dim_date) - MAX(d.full_date)) AS recency_days,
    COUNT(DISTINCT f.invoice_no) AS frequency,
    SUM(f.revenue) AS monetary,
    CASE WHEN COUNT(DISTINCT f.invoice_no) > 1 THEN TRUE ELSE FALSE END AS is_repeat_customer
FROM fact_sales f
JOIN dim_customer c ON f.customer_key = c.customer_key
JOIN dim_date d      ON f.date_key = d.date_key
GROUP BY c.customer_id, c.country;
