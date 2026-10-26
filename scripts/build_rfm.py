
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

SNAPSHOT_DATE = pd.Timestamp("2025-10-01 00:00:00")

# Load Part A's cleaned data
customers = pd.read_csv(CLEAN_DIR / "customers.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Parse order timestamps
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="raise")

# Use completed orders only
completed = orders.loc[
    orders["status"].eq("completed")
].copy()

# Calculate RFM for each customer with at least one completed order
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

# Recency is the whole number of days between snapshot and last order
rfm["recency"] = (
    SNAPSHOT_DATE - rfm["last_order_ts"]
).dt.days

# Keep the requested RFM columns and consistent column order
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].sort_values("customer_id").reset_index(drop=True)

# Save intermediate RFM output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rfm.to_csv(OUTPUT_DIR / "rfm.csv", index=False)

# Report reference checks
print("=" * 60)
print("RFM TABLE VALIDATION")
print("=" * 60)

print(f"Customers in clean customer table: {len(customers):,}")
print(f"Customers with completed orders:   {len(rfm):,}")
print(f"Snapshot date: {SNAPSHOT_DATE}")

print("\nRecency (days):")
print(rfm["recency"].describe().to_string())

print("\nFrequency:")
print(rfm["frequency"].describe().to_string())

print("\nMonetary (INR):")
print(rfm["monetary"].describe().to_string())

print("\nReference checks:")
print("Customer count:", len(rfm), "(reference: 1,175)")
print("Recency median:", rfm["recency"].median(), "(reference: 19)")
print("Recency minimum:", rfm["recency"].min(), "(reference: 0)")
print("Recency maximum:", rfm["recency"].max(), "(reference: 180)")
print("Frequency mean:", round(rfm["frequency"].mean(), 2),
      "(reference: 9.46)")
print("Frequency maximum:", rfm["frequency"].max(),
      "(reference: 33)")
print("Monetary mean:", round(rfm["monetary"].mean(), 2),
      "(reference: 2,552.06)")
print("Monetary maximum:", round(rfm["monetary"].max(), 2),
      "(reference: 14,766.55)")

assert len(rfm) == 1175, "Unexpected RFM customer count"
assert rfm["customer_id"].is_unique, "Duplicate customers in RFM"
assert rfm[["recency", "frequency", "monetary"]].notna().all().all()
assert (rfm["frequency"] > 0).all()

print("\nRFM table saved to:", OUTPUT_DIR / "rfm.csv")
print("RFM VALIDATION COMPLETED.")
