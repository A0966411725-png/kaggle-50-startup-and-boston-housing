"""
Problem 1 — 50 Startups, solved by hand with scikit-learn following CRISP-DM
(course L6 "write sklearn to solve 50-startup multiple linear regression problem").

Part 1: CRISP-DM pipeline (load from web → one-hot → split → LinearRegression →
        R2 / MAE / MSE / RMSE → actual-vs-predicted plot)
Part 2: forward selection for k = 1..5 (lecture version, scored on the test set)
Step 5: deployment with pickle + gzip

This is the hand-written version; analysis/mlr_workflow.py generalises the same
steps and is what the Claude Skill `mlr-crisp-dm` runs for later problems.
"""

import gzip
import pickle
from io import StringIO
from urllib.error import URLError
from urllib.request import urlopen
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
URL = ("https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/"
       "faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv")

# ---- 1. Business Understanding -------------------------------------------
# Predict a startup's Profit from R&D / Administration / Marketing spend and State,
# and find which factors matter (high R2, low MAE/MSE/RMSE, few features).

# ---- 2. Data Understanding -----------------------------------------------
try:
    with urlopen(URL, timeout=20) as resp:
        data = pd.read_csv(StringIO(resp.read().decode("utf-8")))
except URLError:
    data = pd.read_csv(ROOT / "data" / "50_Startups.csv")  # offline fallback, same file
print(data.head())
print(data.describe())
print("missing values:", int(data.isna().sum().sum()))
print(data.drop(columns="State").corr()["Profit"].round(3))

# ---- 3. Data Preparation -------------------------------------------------
X = pd.get_dummies(data.drop(columns="Profit"), columns=["State"], drop_first=True, dtype=float)
y = data["Profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ---- 4. Modeling ---------------------------------------------------------
model = LinearRegression().fit(X_train, y_train)

# ---- 5. Evaluation -------------------------------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nAll 5 features  R2={r2_score(y_test, y_pred):.4f}  MAE={mean_absolute_error(y_test, y_pred):,.2f}  "
      f"MSE={mse:,.2f}  RMSE={np.sqrt(mse):,.2f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, y_pred)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "--", color="gray")
ax.set(xlabel="Actual Profit", ylabel="Predicted Profit", title="Actual vs. Predicted Profit")
(ROOT / "reports").mkdir(exist_ok=True)
fig.savefig(ROOT / "reports" / "startups_actual_vs_pred.png", dpi=120, bbox_inches="tight")


# Part 2 — forward selection, 1..5 features (lecture: chosen by test RMSE)
def forward_selection(max_features):
    selected, results = [], []
    for k in range(1, max_features + 1):
        best = None
        for f in X_train.columns.difference(selected):
            cols = selected + [f]
            m = LinearRegression().fit(X_train[cols], y_train)
            p = m.predict(X_test[cols])
            rmse = np.sqrt(mean_squared_error(y_test, p))
            if best is None or rmse < best[1]:
                best = (f, rmse, r2_score(y_test, p))
        selected.append(best[0])
        results.append({"Number of Features": k, "Selected Features": list(selected),
                        "RMSE": best[1], "R-squared": best[2]})
    return pd.DataFrame(results)


table = forward_selection(5)
print("\nResults Table:")
print(table.to_string(index=False))

# ---- 6. Deployment -------------------------------------------------------
best_cols = table.loc[table["RMSE"].idxmin(), "Selected Features"]
final = LinearRegression().fit(X_train[best_cols], y_train)
(ROOT / "models").mkdir(exist_ok=True)
with gzip.open(ROOT / "models" / "startups_lecture_l6.pkl.gz", "wb") as fh:
    pickle.dump({"model": final, "features": best_cols}, fh)
with gzip.open(ROOT / "models" / "startups_lecture_l6.pkl.gz", "rb") as fh:
    loaded = pickle.load(fh)
demo = pd.DataFrame([[100000, 150000]], columns=best_cols)
print(f"\nDeployed model {best_cols}: Profit({demo.values.tolist()[0]}) = "
      f"{loaded['model'].predict(demo)[0]:,.2f}")
