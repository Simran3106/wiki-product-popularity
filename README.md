# Early Product Popularity Signals

Can a product's early Wikipedia footprint predict how much attention it will get later? This project builds an end-to-end pipeline — data collection, a relational database, feature engineering, and modeling- to test that using pageview and edit-history data for consumer products.

# Problem Setup

For each product, I use the first 30 days after it starts getting meaningful traffic (its "pre-window") to compute features, and then look at the peak pageviews over the following 180 days (the "post-window") as the target. In other words: given only a product's early signal, can you predict how big it eventually gets?

- Target: `log_peak_views_post` — log-transformed peak daily pageviews in the 180 days after the reference point
- Products: 356 (after filtering down to those with sufficient history on both sides)

## Pipeline

1. Data Collection (`wikidata.py`, `wikiedit.py`)- Pulled candidate products from Wikipedia categories, then collected daily pageviews via the Wikimedia REST API and full edit-history via the MediaWiki API.
2. Database (`createdatabase.py`)- Loaded all per-product CSVs into SQLite with indexed tables for fast lookups.
3. Feature Engineering (`builddataset.py`)- For each product, computed 18 features from the 30-day pre-window: view-level stats (avg, median, volatility, coefficient of variation), trend features (slope, acceleration), shape features (skew, largest jump/drop), and edit-activity features.
4. Modeling (`train.py`)- Trained Linear Regression and Random Forest, evaluated with 10-fold cross-validation, and ran feature ablation.

## Results

| Model | Mean CV R² |
| Linear Regression (1 feature: avg views) | 0.587 |
| Linear Regression (all 18 features) | 0.595 |
| Random Forest (all 18 features) | 0.582 |

Key finding: A single feature- average pre-window views, explains nearly all the variance. Engineering 17 additional features added only ~1.4% improvement. The strongest signals by importance were `max_views_pre`, `range_views_pre`, and `volatility_pre` i.e., how big the spike was mattered more than growth-shape features.

## What I'd Improve

- Engineered features were largely redundant with raw view magnitude — worth testing explicitly decorrelated features (e.g., rank-normalized).
- No time-series-aware validation; random splits may leak temporal structure.
- Wikipedia category sourcing is noisy; survivorship bias toward products with good coverage.

## Tech Stack
Python, SQL, SQLite, Pandas, NumPy, Scikit-learn, SciPy, Wikipedia APIs

## How to Run

python wikidata.py        # builds product_list.txt, pulls pageviews
python wikiedit.py         # pulls edit history
python createdatabase.py   # builds wiki_data.db
python builddataset.py     # builds model_dataset.csv
python train.py            # trains and evaluates models
