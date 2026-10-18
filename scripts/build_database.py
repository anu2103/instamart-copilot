
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

tables = {
    "customers": pd.read_csv(CLEAN / "customers.csv"),
    "products": pd.read_csv(CLEAN / "products.csv"),
    "orders": pd.read_csv(CLEAN / "orders.csv"),
}

with sqlite3.connect(DB_PATH) as conn:
    for table_name, df in tables.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    # Verify the required tables exist and have the expected row counts.
    for table_name, df in tables.items():
        count = pd.read_sql_query(
            f"SELECT COUNT(*) AS n FROM {table_name}", conn
        )["n"].iloc[0]
        assert count == len(df), f"Row count mismatch: {table_name}"
        print(f"{table_name}: {count} rows")

    # Ensure order_ts is stored as text in the requested format.
    order_ts_type = pd.read_sql_query(
        "PRAGMA table_info(orders)", conn
    ).set_index("name").loc["order_ts", "type"]

    print("orders.order_ts SQLite type:", order_ts_type)

print(f"Database created: {DB_PATH}")

