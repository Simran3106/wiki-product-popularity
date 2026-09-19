import requests
import pandas as pd
import time
import os
import re

HEADERS = {'User-Agent': 'Student(simran.january01@gmail.com)'}

def get_edit_history(article, start_date, end_date, lang='en'):
    """
    Pulls revision (edit) timestamps for a Wikipedia article between two dates.
    start_date / end_date format: 'YYYY-MM-DD'
    Returns a dataframe with one row per day and an edit_count column.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    all_revisions = []
    rvcontinue = None

    while True:
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": article,
            "rvlimit": "500",
            "rvprop": "timestamp",
            "rvstart": f"{end_date}T23:59:59Z",   # newest first
            "rvend": f"{start_date}T00:00:00Z",
            "format": "json"
        }
        if rvcontinue:
            params["rvcontinue"] = rvcontinue

        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=10)
            data = r.json()
        except Exception as e:
            print(f"  ERROR on {article}: {e}", flush=True)
            return None

        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "revisions" in page_data:
                all_revisions.extend(page_data["revisions"])

        if "continue" in data:
            rvcontinue = data["continue"]["rvcontinue"]
        else:
            break

    if not all_revisions:
        return None

    df = pd.DataFrame(all_revisions)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby("date").size().reset_index(name="edit_count")
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', "_", name)


if __name__ == "__main__":
    # Load your existing product list from the Wikipedia pageviews script
    with open(r"C:\Users\hppav\OneDrive\Documents\VSCODE\project\product_list.txt", "r", encoding="utf-8") as f:
        products = f.read().splitlines()

    print(f"Loaded {len(products)} products", flush=True)

    os.makedirs("edits_data", exist_ok=True)
    success, failed = 0, []

    for i, product in enumerate(products):
        print(f"  [{i+1}/{len(products)}] {product}", flush=True)
        df = get_edit_history(product, "2015-07-01", "2026-07-01")
        if df is not None:
            df.to_csv(f"edits_data/{safe_filename(product)}.csv", index=False)
            success += 1
        else:
            failed.append(product)
        time.sleep(0.3)

    print(f"\nSuccessfully pulled: {success}", flush=True)
    print(f"Failed/no data: {len(failed)}", flush=True)
    print("SCRIPT FINISHED", flush=True)