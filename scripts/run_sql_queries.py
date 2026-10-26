
from pathlib import Path
import sqlite3
import re

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

sql_text = SQL_PATH.read_text(encoding="utf-8")

# Remove full-line SQL comments before splitting statements.
sql_text = re.sub(r"(?m)^\s*--.*$", "", sql_text)

queries = [
    statement.strip()
    for statement in sql_text.split(";")
    if statement.strip()
]

with sqlite3.connect(DB_PATH) as conn:
    for i, query in enumerate(queries, start=1):
        cursor = conn.execute(query)

        print(f"\n{'=' * 60}")
        print(f"Q{i}")

        if cursor.description is None:
            print("Statement executed; no result set.")
            continue

        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]

        print(f"Rows: {len(rows)}")
        print(columns)

        for row in rows[:5]:
            print(row)

        if len(rows) > 5:
            print("...")

        if i == 2 and rows:
            print(
                "Q2 total revenue (INR):",
                round(sum(row[2] for row in rows), 2),
            )

        if i == 5:
            print("Q5 customers above average:", len(rows))

        if i == 7 and rows:
            print(
                "Q7 final cumulative revenue (INR):",
                rows[-1][2],
            )

