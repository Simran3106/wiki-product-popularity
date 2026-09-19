import sqlite3
import pandas as pd
import os
import re


PAGEVIEWS_DIR = "pageviews_data"
EDITS_DIR = "edits_data"
DB_PATH = "wiki_data.db"


def safe_filename(name):
    # Make the filename match the way the original CSV files were saved
    return re.sub(r'[<>:"/\\|?*]', "_", name)


# Start with a fresh database
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

# Create empty tables for pageviews and edits
conn.execute("""
    CREATE TABLE pageviews (
        product TEXT,
        date TEXT,
        views INTEGER
    )
""")

conn.execute("""
    CREATE TABLE edits (
        product TEXT,
        date TEXT,
        edit_count INTEGER
    )
""")

# Read the original product names so that the database keeps
# the exact Wikipedia article names
with open("product_list.txt", "r", encoding="utf-8") as f:
    products = f.read().splitlines()

pageview_rows = 0
edit_rows = 0
pageview_products = 0
edit_products = 0

# Load pageview CSVs into the database
for product in products:
    filename = safe_filename(product)
    path = os.path.join(PAGEVIEWS_DIR, f"{filename}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)

        df["product"] = product

        df[["product", "date", "views"]].to_sql(
            "pageviews",
            conn,
            if_exists="append",
            index=False
        )

        pageview_rows += len(df)
        pageview_products += 1


# Load edit CSVs into the database
for product in products:
    filename = safe_filename(product)
    path = os.path.join(EDITS_DIR, f"{filename}.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)

        df["product"] = product

        df[["product", "date", "edit_count"]].to_sql(
            "edits",
            conn,
            if_exists="append",
            index=False
        )

        edit_rows += len(df)
        edit_products += 1


# Add indexes so looking up a product and date is faster
conn.execute("""
    CREATE INDEX idx_pageviews_product_date
    ON pageviews(product, date)
""")

conn.execute("""
    CREATE INDEX idx_edits_product_date
    ON edits(product, date)
""")

conn.commit()
conn.close()

print("Database created successfully.")
print(f"Pageview products loaded: {pageview_products}")
print(f"Pageview rows loaded: {pageview_rows}")
print(f"Edit products loaded: {edit_products}")
print(f"Edit rows loaded: {edit_rows}")
print(f"Saved database to: {DB_PATH}")