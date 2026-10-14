"""
Generic CRISP-DM workflow for a multiple-linear-regression problem.

Given a problem config (CSV + target + a short business brief), this script:
  1. loads the data and profiles it                     (Data Understanding)
  2. drops NA rows, one-hot encodes nominal columns      (Data Preparation)
     with drop_first, splits train/test with sklearn
  3. runs six feature-selection methods for k = 1..p     (Modeling)
     Lasso, Forward SFS, RFE, ElasticNet, SelectKBest, Exhaustive
     plus Optuna-tuned Lasso and a mutual-information ranking
  4. evaluates every (method, k) with RMSE / R2 / MAE    (Evaluation)
     on train, 5-fold CV and the held-out test set, and fits
     the recommended model with statsmodels for inference
  5. saves the model (pickle + gzip) and a JSON bundle   (Deployment)
     that the web explorer (index.html) loads; the browser
     re-computes everything and checks it against these numbers

Usage:
    python analysis/mlr_workflow.py problems/config/startups.config.json

The algorithms are mirrored one-to-one in js/engine.js; change both together.
"""

import gzip
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, enet_path, lasso_path
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent.parent
N_ALPHAS = 200
EPS = 1e-4
ENET_L1_RATIO = 0.5
CV_FOLDS = 5
MAX_EXHAUSTIVE_P = 16
MAX_CATEGORIES = 20


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def prepare(df, target, drop):
    """NA rows dropped, nominal columns one-hot encoded (drop_first, sorted levels)."""
    df = df.drop(columns=[c for c in drop if c in df.columns])
    n_raw = len(df)
    df = df.dropna().reset_index(drop=True)
    X = df.drop(columns=[target])
    nominal, dropped_id_like = [], []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            if X[col].nunique() <= MAX_CATEGORIES:
                nominal.append(col)
            else:
                dropped_id_like.append(col)
    X = X.drop(columns=dropped_id_like)
    X = pd.get_dummies(X, columns=nominal, drop_first=True, dtype=float).astype(float)
    return df, X, df[target].astype(float), nominal, dropped_id_like, n_raw


# --------------------------------------------------------------------------- #
# OLS helpers (index lists refer to columns of the full encoded matrix)
# --------------------------------------------------------------------------- #
def ols_fit(X, y):
    m = LinearRegression().fit(X, y)
    return m


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y = np.asarray(y)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def cv_rmse(Xtr, ytr, cols, folds):
    scores = []
    for tr, va in folds:
        m = ols_fit(Xtr[tr][:, cols], ytr[tr])
        scores.append(rmse(ytr[va], m.predict(Xtr[va][:, cols])))
    return float(np.mean(scores))


def evaluate(Xtr, ytr, Xte, yte, cols, folds):
    m = ols_fit(Xtr[:, cols], ytr)
    ptr, pte = m.predict(Xtr[:, cols]), m.predict(Xte[:, cols])
    return {
        "train_rmse": rmse(ytr, ptr),
        "train_r2": r2(ytr, ptr),
        "cv_rmse": cv_rmse(Xtr, ytr, cols, folds),
        "test_rmse": rmse(yte, pte),
        "test_r2": r2(yte, pte),
        "test_mae": float(np.mean(np.abs(yte - pte))),
    }


# --------------------------------------------------------------------------- #
# Feature-selection methods -> feature order or per-k subsets
# --------------------------------------------------------------------------- #
def path_order(coefs):
    """coefs: (p, n_alphas) along a decreasing alpha grid -> entry order."""
    p, n = coefs.shape
    entry = []
    for j in range(p):
        nz = np.flatnonzero(coefs[j] != 0)
        idx = int(nz[0]) if len(nz) else n
        mag = abs(coefs[j, idx]) if idx < n else 0.0
        entry.append((idx, -mag, j))
    return [j for _, _, j in sorted(entry)]


def lasso_order(Xs, yc):
    _, coefs, _ = lasso_path(Xs, yc, eps=EPS, n_alphas=N_ALPHAS, tol=1e-12, max_iter=200000)
    return path_order(coefs)


def enet_order(Xs, yc):
    _, coefs, _ = enet_path(Xs, yc, l1_ratio=ENET_L1_RATIO, eps=EPS, n_alphas=N_ALPHAS,
                            tol=1e-12, max_iter=200000)
    return path_order(coefs)


def rfe_order(Xs, ytr):
    rfe = RFE(LinearRegression(), n_features_to_select=1, step=1).fit(Xs, ytr)
    return [int(j) for j in np.argsort(rfe.ranking_, kind="stable")]


def kbest_order(Xtr, ytr):
    F, _ = f_regression(Xtr, ytr)
    return [int(j) for j in np.argsort(-F, kind="stable")]


def forward_subsets(p, score):
    chosen, out = [], []
    for _ in range(p):
        best, best_s = None, np.inf
        for j in range(p):
            if j in chosen:
                continue
            s = score(chosen + [j])
            if s < best_s:
                best, best_s = j, s
        chosen = chosen + [best]
        out.append(list(chosen))
    return out


def exhaustive_subsets(p, score):
    out = []
    for k in range(1, p + 1):
        best, best_s = None, np.inf
        for combo in itertools.combinations(range(p), k):
            s = score(list(combo))
            if s < best_s:
                best, best_s = list(combo), s
        out.append(best)
    return out


# --------------------------------------------------------------------------- #
# Inference for the recommended model
# --------------------------------------------------------------------------- #
def inference(Xtr_df, ytr, names):
    D = sm.add_constant(Xtr_df[names], has_constant="add")
    res = sm.OLS(ytr.values, D).fit()
    ci = res.conf_int(0.05)
    rows = []
    for i, nm in enumerate(["const"] + names):
        vif = None if i == 0 or len(names) == 1 else float(variance_inflation_factor(D.values, i))
        if i > 0 and len(names) == 1:
            vif = 1.0
        rows.append({
            "name": nm, "coef": float(res.params.iloc[i]), "se": float(res.bse.iloc[i]),
            "t": float(res.tvalues.iloc[i]), "p": float(res.pvalues.iloc[i]),
            "ci_low": float(ci.iloc[i, 0]), "ci_high": float(ci.iloc[i, 1]), "vif": vif,
        })
    return {
        "features": names, "coefficients": rows, "r2_train": float(res.rsquared),
        "adj_r2_train": float(res.rsquared_adj), "f_stat": float(res.fvalue),
        "f_pvalue": float(res.f_pvalue), "df_model": int(res.df_model), "df_resid": int(res.df_resid),
        "durbin_watson": float(sm.stats.durbin_watson(res.resid)),
        "jarque_bera_p": float(sm.stats.jarque_bera(res.resid)[1]),
        "breusch_pagan_p": float(sm.stats.het_breuschpagan(res.resid, D)[1]),
    }


def optuna_lasso(Xs, ytr, folds):
    try:
        import optuna
    except ImportError:
        return None
    from sklearn.linear_model import Lasso
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scale = float(np.std(ytr))

    def objective(trial):
        a = trial.suggest_float("alpha", 1e-4 * scale, 1.0 * scale, log=True)
        errs = []
        for tr, va in folds:
            m = Lasso(alpha=a, max_iter=50000).fit(Xs[tr], ytr[tr])
            errs.append(rmse(ytr[va], m.predict(Xs[va])))
        return float(np.mean(errs))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=60)
    return {"alpha": study.best_params["alpha"], "cv_rmse": study.best_value, "n_trials": 60}


# --------------------------------------------------------------------------- #
def main(config_path):
    t0 = time.time()
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / cfg["csv"])
    target = cfg["target"]
    df, X, y, nominal, id_like, n_raw = prepare(raw, target, cfg.get("drop", []))
    names = list(X.columns)
    p = len(names)

    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=cfg.get("test_size", 0.2),
                                      random_state=cfg.get("random_state", 42))
    Xtr, Xte = X.values[tr_idx], X.values[te_idx]
    ytr, yte = y.values[tr_idx], y.values[te_idx]
    folds = list(KFold(n_splits=CV_FOLDS, shuffle=False).split(Xtr))

    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr)
    yc = ytr - ytr.mean()

    def test_score(cols):
        m = ols_fit(Xtr[:, cols], ytr)
        return rmse(yte, m.predict(Xte[:, cols]))

    def cv_score(cols):
        return cv_rmse(Xtr, ytr, cols, folds)

    orders = {
        "lasso": lasso_order(Xs, yc),
        "elasticnet": enet_order(Xs, yc),
        "rfe": rfe_order(Xs, ytr),
        "kbest": kbest_order(Xtr, ytr),
    }
    subsets = {m: [o[:k] for k in range(1, p + 1)] for m, o in orders.items()}
    for crit, score in (("test", test_score), ("cv", cv_score)):
        subsets[f"forward_{crit}"] = forward_subsets(p, score)
        if p <= MAX_EXHAUSTIVE_P:
            subsets[f"exhaustive_{crit}"] = exhaustive_subsets(p, score)

    table = {}
    for key, subs in subsets.items():
        table[key] = []
        for k, cols in enumerate(subs, start=1):
            row = {"k": k, "features": [names[c] for c in cols]}
            row.update(evaluate(Xtr, ytr, Xte, yte, cols, folds))
            table[key].append(row)

    # Recommended model: leakage-free choice = min CV RMSE over every method and k
    best = min(((key, r) for key, rows in table.items() if not key.endswith("_test") for r in rows),
               key=lambda kr: (kr[1]["cv_rmse"], kr[1]["k"]))
    best_test = min(((key, r) for key, rows in table.items() if not key.endswith("_cv") for r in rows),
                    key=lambda kr: (kr[1]["test_rmse"], kr[1]["k"]))
    rec_names = best[1]["features"]
    Xtr_df = pd.DataFrame(Xtr, columns=names)
    infer = inference(Xtr_df, pd.Series(ytr), rec_names)

    mi = mutual_info_regression(Xtr, ytr, random_state=0)
    mi_rank = [{"feature": names[j], "mi": float(mi[j])} for j in np.argsort(-mi)]
    opt = optuna_lasso(Xs, ytr, folds)
    if opt:
        from sklearn.linear_model import Lasso
        lm = Lasso(alpha=opt["alpha"], max_iter=50000).fit(Xs, ytr)
        pte = lm.predict(scaler.transform(Xte))
        opt.update({"test_rmse": rmse(yte, pte), "test_r2": r2(yte, pte),
                    "n_nonzero": int(np.sum(lm.coef_ != 0)),
                    "kept": [names[j] for j in np.flatnonzero(lm.coef_ != 0)]})

    # Deployment artifact: the recommended OLS model, pickle + gzip
    rec_cols = [names.index(n) for n in rec_names]
    model = ols_fit(Xtr[:, rec_cols], ytr)
    (ROOT / "models").mkdir(exist_ok=True)
    with gzip.open(ROOT / "models" / f"{cfg['id']}_ols.pkl.gz", "wb") as fh:
        pickle.dump({"model": model, "features": rec_names, "target": target,
                     "encoded_columns": names, "nominal": nominal}, fh)

    desc = raw.describe(include="all").T
    profile = []
    for col in raw.columns:
        s = raw[col]
        item = {"name": col, "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            item.update({"mean": float(s.mean()), "std": float(s.std()),
                         "min": float(s.min()), "max": float(s.max())})
            if col != target:
                item["corr_with_target"] = float(raw[[col, target]].corr().iloc[0, 1])
        profile.append(item)

    bundle = {
        "id": cfg["id"], "title": cfg["title"], "short": cfg.get("short", cfg["title"]),
        "source": cfg["source"], "csv": cfg["csv"], "csv_url": cfg.get("csv_url"),
        "target": target, "target_label": cfg.get("target_label", target),
        "unit": cfg.get("unit", {}), "labels": cfg.get("labels", {}), "drop": cfg.get("drop", []),
        "test_size": cfg.get("test_size", 0.2), "random_state": cfg.get("random_state", 42),
        "split": {"train_idx": [int(i) for i in tr_idx], "test_idx": [int(i) for i in te_idx]},
        "n_raw": n_raw, "n": len(df), "nominal": nominal, "dropped_id_like": id_like,
        "encoded_features": names, "profile": profile,
        "orders": {m: [names[j] for j in o] for m, o in orders.items()},
        "reference": table,
        "recommended": {"method": best[0], "k": best[1]["k"], "features": rec_names,
                        "cv_rmse": best[1]["cv_rmse"], "test_rmse": best[1]["test_rmse"],
                        "test_r2": best[1]["test_r2"]},
        "best_by_test": {"method": best_test[0], "k": best_test[1]["k"],
                         "features": best_test[1]["features"],
                         "test_rmse": best_test[1]["test_rmse"], "test_r2": best_test[1]["test_r2"]},
        "inference": infer, "mutual_info": mi_rank, "optuna_lasso": opt,
        "business": cfg.get("business", {}), "notes": cfg.get("notes", {}),
        "versions": {"pandas": pd.__version__, "numpy": np.__version__,
                     "sklearn": __import__("sklearn").__version__, "statsmodels": sm.__version__},
    }
    out = ROOT / "problems" / f"{cfg['id']}.json"
    out.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")

    reg_path = ROOT / "problems" / "index.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else []
    reg = [r for r in reg if r["id"] != cfg["id"]] + [{"id": cfg["id"], "short": bundle["short"],
                                                       "file": f"problems/{cfg['id']}.json",
                                                       "order": cfg.get("order", 99)}]
    reg.sort(key=lambda r: r["order"])
    reg_path.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")

    write_report(bundle, ROOT / "reports" / f"{cfg['id']}_crisp_dm.md")
    print(f"[{cfg['id']}] n={len(df)} p={p} train={len(tr_idx)} test={len(te_idx)}  "
          f"({time.time() - t0:.1f}s)")
    for key in ("forward_test", "lasso", "exhaustive_cv"):
        if key in table:
            print(f"  {key}:")
            for r in table[key]:
                print(f"    k={r['k']:>2}  test RMSE={r['test_rmse']:.4f}  R2={r['test_r2']:.6f}  "
                      f"CV RMSE={r['cv_rmse']:.4f}  {r['features']}")
    print(f"  recommended (min CV RMSE): {bundle['recommended']}")
    print(f"  best by test RMSE (lecture style): {bundle['best_by_test']}")


def fmt(v, unit):
    return f"{unit.get('prefix', '')}{v:,.{unit.get('decimals', 2)}f}{unit.get('suffix', '')}"


def write_report(b, path):
    """Markdown CRISP-DM report generated from the bundle."""
    path.parent.mkdir(exist_ok=True)
    u = b["unit"]
    lines = [f"# {b['title']} — CRISP-DM report", "",
             f"Source: [{b['source']['name']}]({b['source']['url']}) · {b['n']} rows · "
             f"target `{b['target']}` · generated by `analysis/mlr_workflow.py`", ""]
    bu = b["business"]
    lines += ["## 1. Business Understanding", "", bu.get("objective", ""), ""]
    lines += [f"- {s}" for s in bu.get("success", [])] + [""]
    lines += ["## 2. Data Understanding", "",
              "| column | dtype | missing | unique | mean | corr with target |", "|---|---|---|---|---|---|"]
    for c in b["profile"]:
        mean = f"{c['mean']:,.3f}" if "mean" in c else "–"
        corr = f"{c['corr_with_target']:+.3f}" if "corr_with_target" in c else "–"
        lines.append(f"| {c['name']} | {c['dtype']} | {c['missing']} | {c['unique']} | {mean} | {corr} |")
    lines += ["", "## 3. Data Preparation", "",
              f"- Rows: {b['n_raw']} → {b['n']} after dropping NA rows.",
              f"- Nominal columns one-hot encoded with `drop_first=True` (dummy-variable trap): "
              f"{', '.join(b['nominal']) or 'none'}.",
              f"- Encoded predictors (p = {len(b['encoded_features'])}): {', '.join(b['encoded_features'])}.",
              f"- `train_test_split(test_size={b['test_size']}, random_state={b['random_state']})` → "
              f"{len(b['split']['train_idx'])} train / {len(b['split']['test_idx'])} test.", "",
              "## 4. Modeling — feature selection", "",
              "| method | k | features | train RMSE | 5-fold CV RMSE | test RMSE | test R² |",
              "|---|---|---|---|---|---|---|"]
    for key, rows in b["reference"].items():
        for r in rows:
            lines.append(f"| {key} | {r['k']} | {', '.join(r['features'])} | {fmt(r['train_rmse'], u)} | "
                         f"{fmt(r['cv_rmse'], u)} | {fmt(r['test_rmse'], u)} | {r['test_r2']:.4f} |")
    if b["optuna_lasso"]:
        o = b["optuna_lasso"]
        lines += ["", f"Optuna (TPE, {o['n_trials']} trials) tuned Lasso alpha = {o['alpha']:.4g} → "
                      f"CV RMSE {fmt(o['cv_rmse'], u)}, test RMSE {fmt(o['test_rmse'], u)}, "
                      f"R² {o['test_r2']:.4f}, keeps {o['n_nonzero']} features."]
    lines += ["", "Mutual information ranking: " +
              ", ".join(f"{m['feature']} ({m['mi']:.3f})" for m in b["mutual_info"]), "",
              "## 5. Evaluation", ""]
    rec, bt = b["recommended"], b["best_by_test"]
    lines += [f"- Lecture-style choice (min **test** RMSE): `{bt['method']}` k={bt['k']} "
              f"{bt['features']} → test RMSE {fmt(bt['test_rmse'], u)}, R² {bt['test_r2']:.4f}.",
              f"- Leakage-free choice (min **5-fold CV** RMSE on training only): `{rec['method']}` "
              f"k={rec['k']} {rec['features']} → test RMSE {fmt(rec['test_rmse'], u)}, "
              f"R² {rec['test_r2']:.4f}.", "",
              "| term | coef | std err | t | p | 95% CI | VIF |", "|---|---|---|---|---|---|---|"]
    inf = b["inference"]
    for c in inf["coefficients"]:
        vif = "–" if c["vif"] is None else f"{c['vif']:.2f}"
        lines.append(f"| {c['name']} | {c['coef']:,.4f} | {c['se']:,.4f} | {c['t']:.3f} | {c['p']:.4g} | "
                     f"[{c['ci_low']:,.3f}, {c['ci_high']:,.3f}] | {vif} |")
    lines += ["", f"R² (train) {inf['r2_train']:.4f}, adj. R² {inf['adj_r2_train']:.4f}, "
                  f"F({inf['df_model']}, {inf['df_resid']}) = {inf['f_stat']:.2f} (p = {inf['f_pvalue']:.3g}); "
                  f"Durbin–Watson {inf['durbin_watson']:.3f}; Jarque–Bera p {inf['jarque_bera_p']:.3g}; "
                  f"Breusch–Pagan p {inf['breusch_pagan_p']:.3g}.", "",
              "## 6. Deployment", "",
              f"- `models/{b['id']}_ols.pkl.gz` (pickle + gzip) holds the recommended OLS model.",
              "- The web explorer loads this bundle, recomputes every number in the browser and "
              "shows a live predictor.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
