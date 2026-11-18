"""
draw_architecture.py
--------------------
Draws the project architecture diagram (all layers, including the MLOps layer).

Run from the project root:
    /opt/anaconda3/bin/python scripts/draw_architecture.py
Output: docs/architecture_diagram.png and docs/architecture_diagram.svg
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY, RED, ORANGE, GREEN, GRAY = "#1F3864", "#C0392B", "#E67E22", "#1E8449", "#555555"
LIGHT = "#F4F6F8"

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def box(x, y, w, h, text, face="white", edge=NAVY, color="black", size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=face, edgecolor=edge, linewidth=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=color, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(x1, y1, x2, y2, color=GRAY, style="-|>", rad=0.0, lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


ax.text(8, 8.72, "E-Commerce Sales Analytics and Customer Churn Prediction: Architecture",
        ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY)

# ------------------------------------------------------------------ data pipeline (Part 1)
ax.add_patch(FancyBboxPatch((0.15, 4.75), 15.7, 3.65, boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor=LIGHT, edgecolor="#C8CDD3", linewidth=1))
ax.text(0.35, 8.2, "DATA PIPELINE (Part 1)", fontsize=10.5, fontweight="bold", color=GRAY, va="center")

lanes = [
    ("SOURCE", NAVY, ["UCI Online Retail\n(ucimlrepo API)", "Monthly extracts\n(simulated CSV files)"]),
    ("INGESTION", RED, ["fetch_data.py + ingest.py\n(Python / Pandas)", "Airflow DAG\necommerce_monthly_etl", "ingestion_log.jsonl"]),
    ("RAW / STAGING", ORANGE, ["data/raw\n(untouched copy)", "data/landing\nMon_YYYY.csv", "temp_* staging tables"]),
    ("TRANSFORMATION", NAVY, ["etl.py clean()\nvalidation rules", "etl_error_log\n(rejected rows)", "etl_run_log\n(row counts)"]),
    ("STORAGE", RED, ["Neon PostgreSQL\nstar schema", "dim_customer, dim_product\ndim_geography, dim_date", "fact_sales\nfact_customer_retention"]),
    ("ANALYTICS", ORANGE, ["sales_mart\ncustomer_mart", "Streamlit dashboard\n(6 views incl. churn risk)"]),
]
lane_w, gap, x0 = 2.3, 0.3, 0.35
for i, (title, color, items) in enumerate(lanes):
    x = x0 + i * (lane_w + gap)
    box(x, 7.55, lane_w, 0.45, title, face=color, edge=color, color="white", size=9.5, bold=True)
    for j, item in enumerate(items):
        box(x, 6.65 - j * 0.85, lane_w, 0.72, item, size=8.6)
    if i < len(lanes) - 1:
        arrow(x + lane_w + 0.02, 7.01, x + lane_w + gap - 0.02, 7.01)

# ------------------------------------------------------------------ MLOps layer (Part 2)
ax.add_patch(FancyBboxPatch((0.15, 0.15), 15.7, 4.35, boxstyle="round,pad=0.02,rounding_size=0.1",
                            facecolor="#F1F8F3", edgecolor="#BFD8C7", linewidth=1))
ax.text(15.65, 4.28, "MLOPS LAYER (Part 2)", fontsize=10.5, fontweight="bold", color=GREEN, va="center", ha="right")

row1 = [
    "Feature engineering\nchurn label: no purchase\nin next 60 days",
    "train.py\nLogistic Regression,\nRandom Forest, XGBoost\n(chronological split)",
    "MLflow\nexperiment tracking +\nmodel registry (churn-model)",
    "export_model.py\nmodels/churn-model\n+ pinned requirements",
    "FastAPI service (Docker)\n/predict  /health\nvalidates every request",
    "Streamlit churn-risk view\n(Docker)\ncalls the API",
]
bw, bg, bx0, by1, bh = 2.3, 0.3, 0.35, 2.75, 1.2
for i, text in enumerate(row1):
    x = bx0 + i * (bw + bg)
    box(x, by1, bw, bh, text, edge=GREEN, size=8.6)
    if i < len(row1) - 1:
        arrow(x + bw + 0.02, by1 + bh / 2, x + bw + bg - 0.02, by1 + bh / 2, color=GREEN)

row2 = [
    (4, "Prediction log\nlogs/predictions.jsonl\n(inputs, probability, latency)"),
    (5, "monitor.py\nservice health, data quality,\ndrift (PSI / KS), performance"),
]
by2, bh2 = 0.55, 1.2
box(bx0 + 4 * (bw + bg), by2, bw, bh2, row2[0][1], edge=GREEN, size=8.6)
box(bx0 + 2 * (bw + bg), by2, bw + 0.0, bh2, row2[1][1], edge=GREEN, size=8.6)
box(bx0 + 0 * (bw + bg), by2, bw + 0.0, bh2,
    "Retraining decision\ndrift, performance, schedule\n(docs/model_lifecycle.md)", face="#FDEDEC", edge=RED, size=8.6)

# arrows: API -> log -> monitor -> retrain -> train
api_x = bx0 + 4 * (bw + bg)
arrow(api_x + bw / 2, by1, api_x + bw / 2, by2 + bh2, color=GREEN)                       # API down to log
arrow(api_x - 0.02, by2 + bh2 / 2, bx0 + 2 * (bw + bg) + bw + 0.02, by2 + bh2 / 2, color=GREEN)   # log to monitor
arrow(bx0 + 2 * (bw + bg) - 0.02, by2 + bh2 / 2, bx0 + bw + 0.02, by2 + bh2 / 2, color=GREEN)     # monitor to retrain
arrow(bx0 + bw / 2 + 0.55, by2 + bh2, bx0 + (bw + bg) + bw / 2 - 0.55, by1, color=RED, rad=-0.15)  # retrain up to train.py
ax.text(bx0 + 0.95, by1 - 0.45, "retrain", fontsize=8.5, color=RED, style="italic", ha="center")

# warehouse feeds feature engineering (elbow route through the gap between the bands)
store_x = x0 + 4 * (lane_w + gap) + lane_w / 2
feat_x = bx0 + bw / 2
ax.plot([store_x, store_x], [4.95, 4.62], color=NAVY, lw=1.6, solid_capstyle="butt")
ax.plot([store_x, feat_x], [4.62, 4.62], color=NAVY, lw=1.6, solid_capstyle="butt")
arrow(feat_x, 4.62, feat_x, by1 + bh + 0.02, color=NAVY)
ax.text(7.0, 4.62, "features read from fact_sales, dim_date and dim_customer", fontsize=8.3,
        color=NAVY, style="italic", ha="center", va="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

# dashboard shares the container stack
out = Path(__file__).resolve().parent.parent / "docs"
out.mkdir(exist_ok=True)
fig.savefig(out / "architecture_diagram.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out / "architecture_diagram.svg", bbox_inches="tight", facecolor="white")
print("Saved", out / "architecture_diagram.png")
