
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

checks = []

def check(description, condition):
    assert bool(condition), f"CHECK FAILED: {description}"
    checks.append(description)


# Customer checkpoints
check("1,200 customers", len(customers) == 1200)
check("40 products", len(products) == 40)
check("Six cities", customers["city"].nunique() == 6)
check(
    "Expected city counts",
    customers["city"].value_counts().to_dict() == {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184,
    },
)
check(
    "Expected gender counts",
    customers["gender"].value_counts().to_dict() == {
        "female": 610,
        "male": 560,
        "unknown": 30,
    },
)
check("No missing ages", customers["age"].notna().all())
check(
    "Ages between 18 and 60",
    customers["age"].between(18, 60).all(),
)

# Order checkpoints
check("12,309 clean orders", len(orders) == 12309)
check(
    "One row per order ID",
    orders["order_id"].is_unique,
)
check(
    "No pending orders",
    not orders["status"].eq("pending").any(),
)
check(
    "11,110 completed orders",
    orders["status"].eq("completed").sum() == 11110,
)
check(
    "1,199 cancelled orders",
    orders["status"].eq("cancelled").sum() == 1199,
)
check(
    "No missing order amounts",
    orders["order_amount"].notna().all(),
)
check(
    "No missing completed delivery times",
    orders.loc[
        orders["status"].eq("completed"), "delivery_minutes"
    ].notna().all(),
)
check(
    "All cancelled delivery times missing",
    orders.loc[
        orders["status"].eq("cancelled"), "delivery_minutes"
    ].isna().all(),
)

completed_revenue = orders.loc[
    orders["status"].eq("completed"), "order_amount"
].sum()

check(
    "Completed revenue INR 2,998,668.80",
    round(completed_revenue, 2) == 2998668.80,
)

print("ALL CHECKPOINTS PASSED")

