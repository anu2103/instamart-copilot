
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"

CLEAN.mkdir(parents=True, exist_ok=True)


def is_blank(series):
    """Identify missing values and strings containing only whitespace."""
    return (
        series.isna()
        | series.astype("string").str.strip().eq("").fillna(False)
    )


# ---------------------------------------------------------
# 1. Load the original data
# ---------------------------------------------------------
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

print("RAW DATA")
print(f"Customers: {len(customers)}")
print(f"Products: {len(products)}")
print(f"Orders:   {len(orders)}")


# ---------------------------------------------------------
# 2. Clean customers
# ---------------------------------------------------------
print("\nCLEANING CUSTOMERS")

# Standardise city spelling.
customers["city"] = customers["city"].astype("string").str.strip().str.title()

# Standardise gender spelling.
raw_gender = customers["gender"].copy()
gender = raw_gender.astype("string").str.strip().str.casefold()

male_aliases = {
    "male", "m", "man", "men", "masculine", "boy"
}
female_aliases = {
    "female", "f", "woman", "women", "feminine", "girl"
}

customers["gender"] = gender.map(
    lambda value: (
        "male" if value in male_aliases
        else "female" if value in female_aliases
        else "unknown"
    )
)

print("Original gender spellings:")
print(raw_gender.value_counts(dropna=False).to_string())
print("Clean gender counts:")
print(customers["gender"].value_counts().to_string())

# Convert invalid and missing ages to NaN.
age = pd.to_numeric(customers["age"], errors="coerce")
invalid_age = age.notna() & ~age.between(10, 100)
invalid_age_count = int(invalid_age.sum())
blank_age_count = int(is_blank(customers["age"]).sum())

age = age.mask(invalid_age)

# Fill missing ages using the median within each city.
customers["age"] = age
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

if customers["age"].isna().any():
    raise ValueError(
        "Some customer ages remain missing; check city-level medians."
    )

customers["age"] = customers["age"].astype(int)

# Convert signup date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

print("Distinct cleaned cities:", customers["city"].nunique())
print("Invalid ages changed to missing:", invalid_age_count)
print("Originally blank ages:", blank_age_count)
print("Missing ages after cleaning:", customers["age"].isna().sum())
print("Age range:", customers["age"].min(), "to", customers["age"].max())


# ---------------------------------------------------------
# 3. Clean orders
# ---------------------------------------------------------
print("\nCLEANING ORDERS")

# Convert timestamps before ordering records.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

if orders["order_ts"].isna().any():
    raise ValueError("Some order timestamps could not be parsed.")

# Remove exact duplicate rows.
before_exact_dedup = len(orders)
orders = orders.drop_duplicates().copy()
exact_duplicates_removed = before_exact_dedup - len(orders)

# Retain the latest record for each order_id.
before_latest_dedup = len(orders)
orders = (
    orders.sort_values(
        ["order_id", "order_ts"], kind="mergesort"
    )
    .drop_duplicates(subset=["order_id"], keep="last")
    .copy()
)
superseded_records_removed = before_latest_dedup - len(orders)

# Match product details to each order.
if products["product_id"].duplicated().any():
    raise ValueError("products.csv contains duplicate product_id values.")

orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
    indicator=True,
)

unmatched_products = int(orders["_merge"].eq("left_only").sum())
if unmatched_products:
    raise ValueError(
        f"{unmatched_products} orders have no matching product."
    )

orders = orders.drop(columns="_merge")

# Convert amounts and numeric inputs.
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["price_inr"] = pd.to_numeric(
    orders["price_inr"], errors="coerce"
)

# Recover missing or unparseable amounts from product details.
missing_amount_before = int(orders["order_amount"].isna().sum())

calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(
    calculated_amount
)

if orders["order_amount"].isna().any():
    raise ValueError(
        "Some order amounts remain missing; inspect product prices, "
        "quantities, discounts and raw amounts."
    )

# Set negative or blank delivery times to missing.
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

negative_delivery_count = int(
    orders["delivery_minutes"].lt(0).sum()
)
orders.loc[
    orders["delivery_minutes"].lt(0), "delivery_minutes"
] = float("nan")

# Impute only completed orders using their product category median.
completed_mask = orders["status"].eq("completed")

completed_delivery_medians = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .median()
)

missing_completed_before = int(
    orders.loc[completed_mask, "delivery_minutes"].isna().sum()
)

completed_categories = orders.loc[completed_mask, "category"]
completed_medians_for_rows = completed_categories.map(
    completed_delivery_medians
)

orders.loc[completed_mask, "delivery_minutes"] = (
    orders.loc[completed_mask, "delivery_minutes"].fillna(
        completed_medians_for_rows
    )
)

if orders.loc[completed_mask, "delivery_minutes"].isna().any():
    raise ValueError(
        "Some completed orders could not receive a category median."
    )

# Cancelled orders must retain missing delivery times.
cancelled_mask = orders["status"].eq("cancelled")
orders.loc[cancelled_mask, "delivery_minutes"] = float("nan")

# Round recovered and retained monetary values to two decimal places.
orders["order_amount"] = orders["order_amount"].round(2)

# Drop temporary product details: the final orders table keeps its
# original order columns.
orders = orders.drop(columns=["price_inr", "category"])

# Standard database-friendly timestamp format.
orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# ---------------------------------------------------------
# 4. Save clean outputs
# ---------------------------------------------------------
customers.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(CLEAN / "orders.csv", index=False)

print("\nCLEANING SUMMARY")
print("Exact duplicate rows removed:", exact_duplicates_removed)
print("Older order records removed:", superseded_records_removed)
print("Amounts missing before recovery:", missing_amount_before)
print("Negative delivery values found:", negative_delivery_count)
print("Completed delivery values imputed:", missing_completed_before)

print("\nFINAL ORDER COUNTS")
print(orders["status"].value_counts().to_string())
print("Total clean orders:", len(orders))
print(
    "Completed orders with missing delivery:",
    orders.loc[
        orders["status"].eq("completed"), "delivery_minutes"
    ].isna().sum(),
)
print(
    "Cancelled orders with missing delivery:",
    orders.loc[
        orders["status"].eq("cancelled"), "delivery_minutes"
    ].isna().sum(),
)
print(
    "Completed revenue (INR):",
    f'{orders.loc[orders["status"].eq("completed"), "order_amount"].sum():,.2f}',
)

print("\nClean CSV files saved in:", CLEAN)

