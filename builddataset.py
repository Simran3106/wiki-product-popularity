import pandas as pd
import numpy as np
import sqlite3
from scipy.stats import linregress, skew

DB_PATH = "wiki_data.db"

VIEW_THRESHOLD = 5      # first day considered active
PRE_WINDOW_DAYS = 30    # days used to build features
POST_WINDOW_DAYS = 180  # future period used to define the target
MIN_HISTORY_DAYS = 40    # need at least this much data before + reference point to bother


def compute_features_and_target(product_name, conn):

    # Get all pageviews for this product from the database
    pv = pd.read_sql_query(
        """
        SELECT date, views
        FROM pageviews
        WHERE product = ?
        ORDER BY date
        """,
        conn,
        params=[product_name],
        parse_dates=["date"]
    )

    if pv.empty or len(pv) < MIN_HISTORY_DAYS:
        return None

    # Reference point = first day views go above a small floor (article "took off" enough to track)
    active=pv[pv["views"]>VIEW_THRESHOLD]
    if active.empty:
        return None
    ref_date = active["date"].iloc[0] + pd.Timedelta(days=PRE_WINDOW_DAYS)

    pre = pv[(pv["date"] >= ref_date - pd.Timedelta(days=PRE_WINDOW_DAYS)) & (pv["date"] < ref_date)]
    post = pv[(pv["date"] >= ref_date) & (pv["date"] < ref_date + pd.Timedelta(days=POST_WINDOW_DAYS))]

    if len(pre) < PRE_WINDOW_DAYS * 0.7 or post.empty:
        return None  # not enough data on either side to trust this

    # Build all features using only the pre-window data

    pre_views = pre["views"].values
    days = np.arange(len(pre_views))

    # Basic statistics
    avg_views = np.mean(pre_views)
    median_views = np.median(pre_views)
    max_views = np.max(pre_views)
    min_views = np.min(pre_views)
    range_views = max_views - min_views
    volatility = np.std(pre_views)
    cv = volatility / (avg_views + 1)
    total_views = np.sum(pre_views)

    # Simple growth from start to end of the window
    growth_rate = (pre_views[-1] - pre_views[0]) / max(pre_views[0], 1)

    # Largest increase and decrease between consecutive days
    if len(pre_views) > 1:
        daily_change = np.diff(pre_views)
        largest_jump = np.max(daily_change)
        largest_drop = np.min(daily_change)
    else:
        largest_jump = 0
        largest_drop = 0

    # Count how many days were above the average
    days_above_mean = np.sum(pre_views > avg_views)

    # Check whether the distribution has occasional spikes
    view_skew = skew(pre_views)

    # Fit one straight line across the whole window
    slope, intercept, r_value, p_value, std_err = linregress(days, pre_views)
    trend_slope = slope
    trend_r2 = r_value ** 2

    # Compare the trend in the first half and second half
    mid = len(pre_views) // 2

    if mid >= 2 and len(pre_views[mid:]) >= 2:
        first_days = np.arange(mid)
        second_days = np.arange(len(pre_views) - mid)

        first_slope = linregress(first_days, pre_views[:mid]).slope
        second_slope = linregress(second_days, pre_views[mid:]).slope

        acceleration = second_slope - first_slope
    else:
        acceleration = 0

    # Edit count feature, if available
    
    # Get edit data for this product from the database

    edit_avg = 0
    edit_total = 0

    ed = pd.read_sql_query(
        """
        SELECT date, edit_count
        FROM edits
        WHERE product = ?
        ORDER BY date
        """,
        conn,
        params=[product_name],
        parse_dates=["date"]
    )

    if not ed.empty:
        pre_ed = ed[
            (ed["date"] >= ref_date - pd.Timedelta(days=PRE_WINDOW_DAYS))
            & (ed["date"] < ref_date)
        ]

        if not pre_ed.empty:
            edit_avg = pre_ed["edit_count"].mean()
            edit_total = pre_ed["edit_count"].sum()

    edit_view_ratio = edit_avg / (avg_views + 1)
    peak_post = post["views"].max()

    return{
        "product":product_name,
        "avg_views_pre":avg_views,
        "median_views_pre":median_views,
        "max_views_pre":max_views,
        "min_views_pre":min_views,
        "range_views_pre":range_views,
        "volatility_pre":volatility,
        "cv_pre":cv,
        "total_views_pre":total_views,

        "growth_rate_pre":growth_rate,

        "trend_slope_pre":trend_slope,
        "trend_r2_pre":trend_r2,

        "largest_jump_pre":largest_jump,
        "largest_drop_pre":largest_drop,

        "days_above_mean_pre":days_above_mean,

        "view_skew_pre":view_skew,

        "acceleration_pre":acceleration,

        "avg_edits_pre":edit_avg,
        "total_edits_pre":edit_total,
        "edit_view_ratio_pre":edit_view_ratio,

        "peak_views_post":peak_post,
        "log_peak_views_post":np.log1p(peak_post)

        }

if __name__ == "__main__":

    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)

    with open("product_list.txt", "r", encoding="utf-8") as f:
        products = f.read().splitlines()

    rows = []
    skipped = 0

    for i, product in enumerate(products):

        result = compute_features_and_target(product, conn)

        if result is not None:
            rows.append(result)
        else:
            skipped += 1

    # Close the database connection after all products are processed
    conn.close()

    df = pd.DataFrame(rows)

    print(
        f"Built dataset: {len(df)} products with valid features "
        f"(skipped {skipped})",
        flush=True
    )

    df.to_csv("model_dataset.csv", index=False)

    print("Saved to model_dataset.csv", flush=True)
    print(df.describe())