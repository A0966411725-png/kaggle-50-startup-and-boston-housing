---
name: mlr-crisp-dm
description: Add a new multiple-linear-regression problem to this repo's MLR Explorer by running the full CRISP-DM workflow (data profiling, one-hot encoding, train/test split, seven feature-selection methods for k=1..p, Optuna-tuned Lasso, statsmodels inference and assumption tests, business-criteria check, pickle+gzip deployment) and registering it in the website's problem dropdown. Use when the user gives a CSV (local path or raw URL) and a target column and wants it analysed or added as "Problem N".
---

# MLR CRISP-DM workflow

This skill turns a tabular regression dataset into a new problem in the
explorer (`index.html`). Problem 1 (50 Startups) was built by hand; this skill
packages exactly what was done there so any later problem is produced the same
way. Every number the site shows is recomputed in the browser by
`js/engine.js` and checked against the bundle this workflow writes.

## Inputs to collect

1. **Dataset** – a CSV path inside `data/`, or a raw URL to download into `data/`.
2. **Target column** – the continuous value to predict.
3. **Business context** – one paragraph: who uses the prediction and why, plus
   measurable success criteria. Ask the user if they are not obvious.

## Steps

1. **Fetch & inspect (Data Understanding).**
   Download the CSV into `data/` if needed, then print `head()`, `dtypes`,
   missing values, `nunique()` and the correlations with the target. Note:
   - columns that are identifiers (row ids, free text) → put them in `drop`;
   - non-numeric columns with ≤ 20 levels are one-hot encoded automatically
     (drop_first); more levels are dropped as ID-like;
   - perfectly collinear columns (e.g. a column that is the sum of others) → `drop`;
   - censored targets, skew or ethically sensitive columns → mention them in `notes.data`.
2. **Write the config** `problems/config/<id>.config.json`, copying
   `problems/config/startups.config.json` as the template:
   `id`, `order` (position in the dropdown), `title`, `short`, `source`
   {name,url}, `csv`, `csv_url` (raw URL; the custom-dataset box recognises
   the same data and reuses the split settings), `target`, `target_label`,
   `unit` {prefix,suffix,decimals}, `labels` (readable names per encoded
   feature), `drop`, `test_size` (0.2), `random_state`, `notes` {data, lecture},
   and `business`:
   - `objective` – the paragraph from the inputs;
   - `success` – a list of `{text, metric, op, value}`; `metric` is one of
     `test_r2`, `test_rmse`, `test_mae`, `cv_rmse`, `k`, `max_p`, `max_vif`
     and `op` one of `>=`, `>`, `<=`, `<`. Decide the criteria **before**
     looking at the results.

   Write the texts in 繁體中文, matching the existing configs.
3. **Run the workflow** (Data Preparation → Modeling → Evaluation → Deployment)
   in a Python environment with `requirements.txt` installed:
   ```bash
   python analysis/mlr_workflow.py problems/config/<id>.config.json
   ```
   (On Windows with Smart App Control, the newest pandas / scikit-learn wheels
   may be blocked; use a virtual environment with the pinned versions.)
   Outputs: `problems/<id>.json` (bundle for the site, registered in
   `problems/index.json`), `reports/<id>_crisp_dm.md`, `models/<id>_ols.pkl.gz`.
4. **Read the printout and the report.** Check the forward-selection table,
   the leakage-free recommendation (min 5-fold CV RMSE) versus the
   lecture-style pick (min test RMSE), the Backward p-value stopping point,
   the assumption tests (RESET, Jarque–Bera, Breusch–Pagan, Durbin–Watson in
   the original row order, VIF) and the success-criteria table. Report unmet
   criteria honestly; do not change the criteria to make them pass.
5. **Verify in the browser.** Serve the repo (`python -m http.server 8000`),
   open `http://localhost:8000/?problem=<id>`, and confirm the green
   "matches sklearn" badge for every method, then try the live predictor.
6. **Document.** Add a section for the problem to `README.md` (dataset,
   CRISP-DM summary, success criteria, key numbers taken from the report) and
   record in the "Skill applied" log which steps were done by this skill.

## Rules

- Never pick features or k using the test set for the *recommended* model;
  the lecture-style test-RMSE pick is shown only as a comparison.
- Keep `analysis/mlr_workflow.py` and `js/engine.js` in sync – they implement
  the same algorithms (lasso/enet path entry order, RFE on standardised X,
  SelectKBest F-test, Backward by smallest |t|, forward & exhaustive search,
  KFold(5) without shuffle, RESET with powers 2–3 of the standardised fitted
  values, Durbin–Watson in the original row order).
- Exhaustive search is skipped when p > 16.
