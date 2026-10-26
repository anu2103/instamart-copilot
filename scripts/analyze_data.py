
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Analyse completed orders only for revenue and order-value statistics.
completed = orders.loc[orders["status"].eq("completed")].copy()
completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

completed_city = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one",
)

print("\n" + "=" * 65)
print("1. DATASET SUMMARY")
print("=" * 65)
print(f"Customers: {len(customers):,}")
print(f"Products: {len(products):,}")
print(f"Clean orders: {len(orders):,}")
print(f"Completed orders: {len(completed):,}")

# 2. Revenue by city and product category.
completed_detail = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left",
    validate="many_to_one",
).merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

pivot = pd.pivot_table(
    completed_detail,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total",
)

print("\n" + "=" * 65)
print("2. PANDAS REVENUE PIVOT: CITY x CATEGORY (INR)")
print("=" * 65)
print(pivot.round(2).to_string())

pandas_total = completed["order_amount"].sum()

with sqlite3.connect(DB_PATH) as conn:
    sql_total = pd.read_sql_query(
        """
        SELECT SUM(order_amount) AS total_revenue
        FROM orders
        WHERE status = 'completed'
        """,
        conn,
    ).iloc[0]["total_revenue"]

print(f"\nPandas completed revenue: INR {pandas_total:,.2f}")
print(f"SQL completed revenue:    INR {sql_total:,.2f}")
assert round(pandas_total, 2) == round(sql_total, 2)
print("Pandas and SQL revenue totals MATCH.")

# 3. Named aggregation by city.
city_summary = completed_city.groupby("city").agg(
    total_revenue_inr=("order_amount", "sum"),
    mean_order_amount_inr=("order_amount", "mean"),
    completed_orders=("order_id", "count"),
).sort_values("total_revenue_inr", ascending=False)

print("\n" + "=" * 65)
print("3. COMPLETED ORDER METRICS BY CITY")
print("=" * 65)
print(city_summary.round(2).to_string())

# 4. Distribution and IQR outlier analysis.
amounts = completed["order_amount"].dropna()

q1 = amounts.quantile(0.25)
median = amounts.median()
q3 = amounts.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below = amounts < lower_fence
above = amounts > upper_fence

print("\n" + "=" * 65)
print("4. COMPLETED ORDER AMOUNT DISTRIBUTION (INR)")
print("=" * 65)
print(f"Mean:               {amounts.mean():,.2f}")
print(f"Median:             {median:,.2f}")
print(f"Skewness:           {amounts.skew():.2f}")
print(f"Q1 (25th percentile): {q1:,.2f}")
print(f"Q3 (75th percentile): {q3:,.2f}")
print(f"IQR:                {iqr:,.2f}")
print(f"Lower outlier fence: {lower_fence:,.2f}")
print(f"Upper outlier fence: {upper_fence:,.2f}")
print(f"Outliers below fence: {int(below.sum()):,}")
print(f"Outliers above fence: {int(above.sum()):,}")

# 5. Correlation analysis.
correlation_columns = ["order_amount", "discount_pct", "quantity"]
correlations = completed[correlation_columns].corr(method="pearson")

print("\n" + "=" * 65)
print("5. PEARSON CORRELATIONS: COMPLETED ORDERS")
print("=" * 65)
print(correlations.round(3).to_string())

pairs = [
    ("order_amount", "discount_pct"),
    ("order_amount", "quantity"),
    ("discount_pct", "quantity"),
]
strongest_pair = max(
    pairs,
    key=lambda pair: abs(correlations.loc[pair[0], pair[1]]),
)
print(
    "\nLargest absolute pairwise correlation:",
    f"{strongest_pair[0]} vs {strongest_pair[1]}",
    f"= {correlations.loc[strongest_pair[0], strongest_pair[1]]:.3f}",
)

print(
    "\nInterpretation: Correlation measures association, not causation. "
    "These results alone do not establish that one variable causes changes "
    "in another."
)

print("\nPANDAS ANALYSIS COMPLETED.")

