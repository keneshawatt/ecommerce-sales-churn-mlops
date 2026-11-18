"""
Streamlit dashboard: reads from sales_mart / customer_mart (Neon PostgreSQL) and
calls the churn prediction API for the churn-risk view.

Run locally:  streamlit run dashboard/app.py
In Docker:    docker compose up   (API_URL is set to the api container)
"""
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="E-Commerce Sales Analytics", layout="wide")


@st.cache_resource
def get_engine():
    db_url = os.environ.get("DATABASE_URL")
    return create_engine(db_url, pool_pre_ping=True, pool_recycle=300)


@st.cache_data(ttl=600)
def load_sales_mart():
    return pd.read_sql("SELECT * FROM sales_mart", get_engine())


@st.cache_data(ttl=600)
def load_customer_mart():
    return pd.read_sql("SELECT * FROM customer_mart", get_engine())


@st.cache_data(ttl=600)
def load_orders_by_country():
    """Distinct invoices per country. sales_mart cannot be summed for this, because an invoice
    with several products would be counted once per product."""
    return pd.read_sql(
        "SELECT g.country, COUNT(DISTINCT f.invoice_no) AS orders "
        "FROM fact_sales f JOIN dim_geography g ON f.geography_key = g.geography_key "
        "GROUP BY g.country",
        get_engine(),
    )


@st.cache_data(ttl=600)
def load_as_of_date():
    """Last transaction date in the warehouse: 'today' for the dataset."""
    value = pd.read_sql("SELECT MAX(full_date) AS d FROM dim_date", get_engine())["d"].iloc[0]
    return pd.Timestamp(value)


@st.cache_data(ttl=600)
def load_customer_invoices(customer_id: str) -> pd.DataFrame:
    query = text("""
        SELECT d.full_date, f.invoice_no, SUM(f.revenue) AS revenue
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_key = c.customer_key
        JOIN dim_date d     ON f.date_key = d.date_key
        WHERE c.customer_id = :cid
        GROUP BY d.full_date, f.invoice_no
    """)
    invoices = pd.read_sql(query, get_engine(), params={"cid": customer_id})
    invoices["full_date"] = pd.to_datetime(invoices["full_date"])
    invoices["revenue"] = invoices["revenue"].astype(float)
    return invoices


def customer_features(invoices: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """The same six features the model was trained on, computed as of `as_of`."""
    frequency = invoices["invoice_no"].nunique()
    monetary = invoices["revenue"].sum()
    recent = invoices[invoices["full_date"] > as_of - pd.Timedelta(days=90)]
    return {
        "recency_days": float((as_of - invoices["full_date"].max()).days),
        "frequency": float(frequency),
        "monetary": float(monetary),
        "avg_order_value": float(monetary / frequency),
        "tenure_days": float((as_of - invoices["full_date"].min()).days),
        "orders_last_90d": float(recent["invoice_no"].nunique()),
    }


def predict_churn(features: dict):
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not get a prediction from the API at {API_URL}. "
                 f"Is the API container running? ({exc})")
        return None


def show_prediction(result: dict):
    probability = result["churn_probability"]
    left, right = st.columns([1, 2])
    left.metric("Churn probability", f"{probability:.1%}")
    if result["churn_prediction"] == 1:
        right.error("At risk of churning (no purchase expected in the next 60 days)")
    else:
        right.success("Likely to keep buying")
    right.progress(min(max(probability, 0.0), 1.0))
    st.caption(f"Registered model churn-model, version {result['model_version']}")


st.title("E-Commerce Sales & Customer Analytics")

sales = load_sales_mart()
customers = load_customer_mart()

country_filter = st.sidebar.multiselect(
    "Country", options=sorted(sales["country"].dropna().unique()), default=None
)
if country_filter:
    sales = sales[sales["country"].isin(country_filter)]

orders = load_orders_by_country()
if country_filter:
    orders = orders[orders["country"].isin(country_filter)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"£{sales['total_revenue'].sum():,.0f}")   # source prices are in pounds sterling
col2.metric("Total Orders", f"{int(orders['orders'].sum()):,}")
col3.metric("Repeat Customers", f"{customers['is_repeat_customer'].sum():,}")

st.markdown("---")

# View 1: Revenue trend over time
st.subheader("1. Revenue & Order Trends")
trend = sales.groupby("full_date")[["total_revenue", "num_orders"]].sum().reset_index()
st.plotly_chart(
    px.line(trend, x="full_date", y="total_revenue", title="Revenue Over Time"),
    use_container_width=True,
)

# View 2: Top products
st.subheader("2. Top Products")
top_products = (
    sales.groupby("description")["total_revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
st.plotly_chart(
    px.bar(top_products, x="total_revenue", y="description", orientation="h",
           title="Top 10 Products by Revenue"),
    use_container_width=True,
)

# View 3: Country-wise sales
st.subheader("3. Country-wise Sales")
by_country = (
    sales.groupby("country")["total_revenue"].sum().sort_values(ascending=False).reset_index()
)
st.plotly_chart(
    px.bar(by_country.head(15), x="country", y="total_revenue", title="Revenue by Country"),
    use_container_width=True,
)

# View 4: Customer segmentation (RFM scatter)
st.subheader("4. Customer Segmentation (RFM)")
st.plotly_chart(
    px.scatter(customers, x="recency_days", y="monetary", size="frequency",
               color="is_repeat_customer", title="Recency vs Monetary, sized by Frequency"),
    use_container_width=True,
)

# View 5: Retention / repeat-purchase rate
st.subheader("5. Retention & Repeat-Purchase Analysis")
repeat_rate = customers["is_repeat_customer"].mean() * 100
st.metric("Repeat-Purchase Rate", f"{repeat_rate:.1f}%")
st.plotly_chart(
    px.histogram(customers, x="frequency", nbins=20, title="Distribution of Purchase Frequency"),
    use_container_width=True,
)

# View 6: Churn risk, powered by the registered model through the prediction API
st.markdown("---")
st.subheader("6. Churn Risk Prediction")
st.caption("Churn means no purchase in the 60 days after the scoring date. "
           f"Predictions come from the registered churn model through the API ({API_URL}).")

tab_lookup, tab_whatif = st.tabs(["Look up a customer", "What-if scenario"])

with tab_lookup:
    customer_id = st.selectbox(
        "Customer ID",
        options=sorted(customers["customer_id"].astype(str)),
        index=None,
        placeholder="Choose a customer",
    )
    if customer_id:
        as_of = load_as_of_date()
        features = customer_features(load_customer_invoices(customer_id), as_of)
        st.caption(f"Features computed as of {as_of.date()}, the last transaction date in the data.")
        st.dataframe(pd.DataFrame([features]), hide_index=True, use_container_width=True)
        result = predict_churn(features)
        if result:
            show_prediction(result)

with tab_whatif:
    c1, c2, c3 = st.columns(3)
    recency = c1.number_input("Days since last purchase", min_value=0.0, value=45.0, step=1.0)
    frequency = c2.number_input("Number of orders", min_value=0.0, value=3.0, step=1.0)
    monetary = c3.number_input("Total spend", min_value=0.0, value=520.0, step=10.0)
    c4, c5, c6 = st.columns(3)
    aov = c4.number_input("Average order value", min_value=0.0, value=173.3, step=5.0)
    tenure = c5.number_input("Days since first purchase", min_value=0.0, value=200.0, step=1.0)
    recent_orders = c6.number_input("Orders in the last 90 days", min_value=0.0, value=1.0, step=1.0)
    if st.button("Predict churn risk"):
        result = predict_churn({
            "recency_days": recency, "frequency": frequency, "monetary": monetary,
            "avg_order_value": aov, "tenure_days": tenure, "orders_last_90d": recent_orders,
        })
        if result:
            show_prediction(result)

st.caption("Data source: UCI Online Retail Dataset, processed via the ETL pipeline.")
