import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# Load dataset
df = pd.read_csv("model_dataset.csv")
print(f"Loaded {len(df)} products")

FEATURES = [
    "avg_views_pre",
    "median_views_pre",
    "max_views_pre",
    "min_views_pre",
    "range_views_pre",
    "volatility_pre",
    "cv_pre",
    "total_views_pre",
    "growth_rate_pre",
    "trend_slope_pre",
    "trend_r2_pre",
    "largest_jump_pre",
    "largest_drop_pre",
    "days_above_mean_pre",
    "acceleration_pre",
    "avg_edits_pre",
    "total_edits_pre",
    "edit_view_ratio_pre",
]

TARGET = "log_peak_views_post"

df = df.dropna(subset=FEATURES + [TARGET])
df = df[np.isfinite(df[FEATURES]).all(axis=1)]
print(f"After cleaning: {len(df)} products")

log_features = [
    "avg_views_pre","median_views_pre","max_views_pre","min_views_pre",
    "range_views_pre","volatility_pre","cv_pre","total_views_pre",
    "avg_edits_pre","total_edits_pre","edit_view_ratio_pre",
    "growth_rate_pre","trend_slope_pre",
    "largest_jump_pre","largest_drop_pre",
    "acceleration_pre"
]

for col in log_features:
    df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

# Correlation heatmap
corr = df[FEATURES].corr()
plt.figure(figsize=(10,8))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(FEATURES)), FEATURES, rotation=90, fontsize=7)
plt.yticks(range(len(FEATURES)), FEATURES, fontsize=7)
plt.title("Feature Correlation")
plt.tight_layout()
plt.show()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\n--- 80/20 Train/Test Split ---")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

print("\nStandardized Coefficients")
for feat, coef in sorted(zip(FEATURES, model.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"{feat:25s} {coef:.3f}")

perm = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=20,
    random_state=42
)

print("\nPermutation Importance")
for feat, imp in sorted(zip(FEATURES, perm.importances_mean), key=lambda x: x[1], reverse=True):
    print(f"{feat:25s} {imp:.4f}")

# Plots
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Prediction vs Actual")
plt.tight_layout()
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(6,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.hist(residuals, bins=20, edgecolor="black")
plt.title("Residual Distribution")
plt.tight_layout()
plt.show()

cv = KFold(n_splits=10, shuffle=True, random_state=42)

pipeline = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print("\n--- 10 Fold Cross Validation ---")
print(np.round(cv_scores,3))
print(f"Mean R²: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

corr_value, _ = pearsonr(df["avg_views_pre"], y)
print(f"\nCorrelation(avg_views,target): {corr_value:.3f}")

baseline_scores = cross_val_score(
    LinearRegression(),
    df[["avg_views_pre"]],
    y,
    cv=cv,
    scoring="r2"
)

print(f"Baseline-only R²: {baseline_scores.mean():.3f}")
print(f"Full model R²: {cv_scores.mean():.3f}")
print(f"Improvement: {cv_scores.mean()-baseline_scores.mean():.3f}")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,
    random_state=42
)

rf_scores = cross_val_score(
    rf,
    X,
    y,
    cv=cv,
    scoring="r2"
)

print(f"\nRandom Forest Mean R²: {rf_scores.mean():.3f}")

feature_sets = [
    ["avg_views_pre"],
    ["avg_views_pre","trend_slope_pre"],
    ["avg_views_pre","trend_slope_pre","volatility_pre"],
    ["avg_views_pre","trend_slope_pre","volatility_pre","avg_edits_pre"],
    FEATURES
]

print("\nFeature Ablation Study")
for feats in feature_sets:
    if len(feats) == len(FEATURES):
        scores = cv_scores
    else:
        scores = cross_val_score(
            make_pipeline(StandardScaler(), LinearRegression()),
            df[feats],
            y,
            cv=cv,
            scoring="r2"
        )
    print(f"{len(feats):2d} feature(s): {scores.mean():.3f}")


print("\nTop 10 Features by Permutation Importance")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": perm.importances_mean
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print(importance_df.head(10))


# ----------------------------------------------------
# Biggest prediction errors
# ----------------------------------------------------

results = pd.DataFrame({
    "product": df.loc[y_test.index, "product"],
    "actual": y_test,
    "predicted": y_pred
})

results["absolute_error"] = np.abs(
    results["actual"] - results["predicted"]
)

print("\nTop 5 Largest Prediction Errors")

print(
    results.sort_values(
        by="absolute_error",
        ascending=False
    ).head(5)
)
