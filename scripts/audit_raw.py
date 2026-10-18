from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def report_table(name, df):
    print(f"\n{'=' * 70}")
    print(f"TABLE: {name}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (
            df.isna().sum() / len(df) * 100
        ).round(2),
    })
    print(missing.to_string())


# Preserve raw values for auditing.
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

# Empty ticket text must remain an empty string.
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False,
)

report_table("customers.csv", customers)
report_table("products.csv", products)
report_table("orders.csv", orders)

print("\n" + "=" * 70)
print("REQUIRED RAW-DATA CHECKPOINTS")

amount_numeric = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
delivery_numeric = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
age_numeric = pd.to_numeric(
    customers["age"], errors="coerce"
)

print("Raw order rows:", len(orders))
print("Exact duplicate order rows:", orders.duplicated().sum())
print(
    "Pending rows:",
    orders["status"].astype("string").str.strip().str.lower()
    .eq("pending").sum(),
)
print("Distinct order_id:", orders["order_id"].nunique())
print(
    "Non-numeric order_amount:",
    amount_numeric.isna().sum(),
)
print(
    "Negative delivery_minutes:",
    delivery_numeric.lt(0).sum(),
)
print(
    "Blank delivery_minutes:",
    (
        orders["delivery_minutes"].isna()
        | orders["delivery_minutes"].astype(str).str.strip().eq("")
    ).sum(),
)
print(
    "Blank customer age:",
    (
        customers["age"].isna()
        | customers["age"].astype(str).str.strip().eq("")
    ).sum(),
)
print(
    "Ages outside 10-100:",
    (
        age_numeric.notna()
        & ~age_numeric.between(10, 100)
    ).sum(),
)
print(
    "Distinct raw city spellings:",
    customers.loc[
        customers["city"].notna()
        & customers["city"].astype(str).str.strip().ne(""),
        "city",
    ].nunique(),
)
print(
    "Distinct non-blank raw gender spellings:",
    customers.loc[
        customers["gender"].notna()
        & customers["gender"].astype(str).str.strip().ne(""),
        "gender",
    ].nunique(),
)
print(
    "Blank gender:",
    (
        customers["gender"].isna()
        | customers["gender"].astype(str).str.strip().eq("")
    ).sum(),
)

print("\nTicket text preservation check:")
print("Ticket rows:", len(tickets))
print("Empty ticket text values are preserved as strings.")

