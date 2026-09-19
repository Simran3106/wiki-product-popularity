import requests
import pandas as pd
import time
import sys

print("SCRIPT STARTED", flush=True)

# ---------- Category pulling ----------

HEADERS = {'User-Agent': 'Student(simran.january01@gmail.com)'}

def get_category_members(category, limit=500):
    print(f"  Requesting category: {category}", flush=True)
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": limit,
        "format": "json"
    }
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
        print(f"    status: {r.status_code}", flush=True)
        if r.status_code != 200:
            print(f"    body preview: {r.text[:200]}", flush=True)
        data = r.json()
    except requests.exceptions.Timeout:
        print(f"  TIMEOUT on {category}", flush=True)
        return []
    except Exception as e:
        print(f"  ERROR on {category}: {e}", flush=True)
        return []
    return [item["title"] for item in data.get("query", {}).get("categorymembers", [])]


def is_probably_junk(title):
    junk_markers = ["List of", "Category:", "(disambiguation)"]
    return any(marker in title for marker in junk_markers)


# Pull categories 2015-2025
print("Starting category pull...", flush=True)
years = range(2015, 2026)
all_products = []

for year in years:
    category = f"Products introduced in {year}"
    members = get_category_members(category)
    print(f"{year}: {len(members)} entries", flush=True)
    all_products.extend(members)
    time.sleep(0.5)

all_products = list(set(all_products))
print(f"Total unique products (before filtering): {len(all_products)}", flush=True)

clean_products = [p for p in all_products if not is_probably_junk(p)]
print(f"Total unique products (after filtering): {len(clean_products)}", flush=True)

# Add your manually seeded known examples
seed_products = [
    "Labubu",
    "Sonny_Angel",
    "Dubai chocolate",
    "Ray-Ban Meta",
    "Fidget spinner",
]

final_products = list(set(clean_products + seed_products))
print(f"Final product list size: {len(final_products)}", flush=True)

print("\n--- FINAL PRODUCT LIST ---", flush=True)
for p in final_products:
    print(p, flush=True)

# Save the list to a file so you have it even if the next step fails
with open("product_list.txt", "w", encoding="utf-8") as f:
    for p in final_products:
        f.write(p + "\n")
print(f"\nSaved product list to product_list.txt ({len(final_products)} products)", flush=True)

# ---------- Pageviews pulling ----------

def wiki(article, start_date, end_date, lang='en'):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{lang}.wikipedia/all-access/all-agents/{article}/daily/{start_date}/{end_date}"
    headers = {'User-Agent': 'Student(simran.january01@gmail.com)'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None

    if 'items' not in data:
        return None

    df = pd.DataFrame(data['items'])
    df['date'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H')
    return df[['date', 'views']]


print("\nStarting pageviews pull...", flush=True)
all_data = {}
failed = []

for i, product in enumerate(final_products):
    print(f"  [{i+1}/{len(final_products)}] {product}", flush=True)
    df = wiki(product, '20150701', '20260701')
    if df is not None:
        all_data[product] = df
    else:
        failed.append(product)
    time.sleep(0.2)

print(f"\nSuccessfully pulled: {len(all_data)}", flush=True)
print(f"Failed/no data: {len(failed)}", flush=True)
import os
import re

os.makedirs("pageviews_data", exist_ok=True)

def safe_filename(name):
    # Replace any character illegal in Windows filenames with underscore
    return re.sub(r'[<>:"/\\|?*]', "_", name)

for product, df in all_data.items():
    safe_name = safe_filename(product)
    df.to_csv(f"pageviews_data/{safe_name}.csv", index=False)

print(f"Saved {len(all_data)} CSV files to pageviews_data/", flush=True)
print("SCRIPT FINISHED", flush=True)