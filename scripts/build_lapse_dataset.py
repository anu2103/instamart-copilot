
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = ROOT / "data" / "clean"
OUTPUT_DIR = ROOT / "outputs"

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="raise"
)
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Features can use only orders strictly before the cutoff.
past_orders = orders.loc[orders["order_ts"] < CUTOFF].copy()

past_completed = past_orders.loc[
    past_orders["status"].eq("completed")
].copy()

# Cohort: customers with at least one completed order before cutoff.
cohort_ids = past_completed["customer_id"].unique()

# Label window: September 1 through September 30, 2025.
september_completed = orders.loc[
    orders["status"].eq("completed")
    & orders["order_ts"].ge(CUTOFF)
    & orders["order_ts"].lt(LABEL_END)
]

september_active_ids = set(
    september_completed["customer_id"].unique()
)

cohort = customers.loc[
    customers["customer_id"].isin(cohort_ids)
].copy()

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_active_ids)
).astype(int)

# Completed-order behaviour, using past data only.
completed_features = (
    past_completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean"),
    )
    .reset_index()
)

completed_features["recency"] = (
    CUTOFF - completed_features["last_order_ts"]
).dt.days

# Cancellation rate among all final-status orders before cutoff.
final_orders = past_orders.loc[
    past_orders["status"].isin(["completed", "cancelled"])
].copy()

final_counts = (
    final_orders.groupby("customer_id")
    .agg(
        total_final_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status",
            lambda values: values.eq("cancelled").sum()
        ),
    )
    .reset_index()
)

final_counts["cancellation_rate"] = (
    final_counts["cancelled_orders"]
    / final_counts["total_final_orders"]
)

# Favourite category from the customer's past completed orders.
completed_with_category = past_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

category_counts = (
    completed_with_category.groupby(
        ["customer_id", "category"]
    )
    .size()
    .reset_index(name="category_orders")
)

# Sorting makes ties deterministic.
category_counts = category_counts.sort_values(
    ["customer_id", "category_orders", "category"],
    ascending=[True, False, True],
)

favourite_category = (
    category_counts.drop_duplicates("customer_id")
    [["customer_id", "category"]]
    .rename(columns={"category": "favourite_category"})
)

# Assemble customer features.
dataset = (
    cohort.merge(
        completed_features.drop(columns="last_order_ts"),
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        final_counts[["customer_id", "cancellation_rate"]],
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    .merge(
        favourite_category,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
)

dataset["tenure_days"] = (
    CUTOFF - dataset["signup_date"]
).dt.days

feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
]

# Do not silently fill missing values here.
# Investigate any missing values before modelling.
missing_counts = dataset[feature_columns].isna().sum()
print("=" * 65)
print("MISSING FEATURE VALUES")
print(missing_counts.to_string())

if missing_counts.any():
    raise ValueError(
        "Missing feature values detected. Review the counts above "
        "before proceeding to preprocessing."
    )

# Explicitly exclude segment_id, segment_name and is_premium.
dataset = dataset[["customer_id"] + feature_columns + ["lapsed"]]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset.to_csv(
    OUTPUT_DIR / "lapse_dataset.csv",
    index=False
)

print("\n" + "=" * 65)
print("LAPSE DATASET VALIDATION")
print("=" * 65)
print("Cohort customers:", len(dataset))
print("Lapsed customers:", int(dataset["lapsed"].sum()))
print("Lapsed rate (%):", round(dataset["lapsed"].mean() * 100, 2))
print("Not lapsed:", int(dataset["lapsed"].eq(0).sum()))
print("Feature columns:", feature_columns)
print("Customer ID included as identifier, not as a model feature.")
print("Segment labels included as features: No")
print("Orders on/after cutoff used in features: No")

assert len(dataset) == 1156, "Unexpected cohort size"
assert int(dataset["lapsed"].sum()) == 443, "Unexpected lapsed count"
assert dataset["customer_id"].is_unique
assert dataset[feature_columns].notna().all().all()

print("\nSaved:", OUTPUT_DIR / "lapse_dataset.csv")
print("LAPSE DATASET BUILD COMPLETED.")
