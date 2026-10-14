/*
 * MLR engine — runs entirely in the browser.
 * Mirrors analysis/mlr_workflow.py one-to-one so the numbers can be checked
 * against scikit-learn / statsmodels:
 *   - CSV parsing, NA-row dropping, one-hot (drop_first, sorted levels)
 *   - sklearn train_test_split (numpy RandomState MT19937 permutation)
 *   - KFold(5, shuffle=False) cross-validation
 *   - Lasso / ElasticNet path entry order, RFE, SelectKBest(F-test),
 *     Forward SFS, Exhaustive best-subset
 *   - OLS inference: SE, t, p, 95% CI, VIF, F-test, Durbin–Watson,
 *     Jarque–Bera, Breusch–Pagan (Koenker)
 */
(function (global) {
  "use strict";

  const N_ALPHAS = 200, EPS = 1e-4, ENET_L1 = 0.5, CV_FOLDS = 5;
  const MAX_EXHAUSTIVE_P = 16, MAX_CATEGORIES = 20;
  const NA = new Set(["", "na", "n/a", "nan", "null", "none", "#n/a", "-nan", "<na>"]);

  // ------------------------------------------------------------------ CSV
  function parseCSV(text) {
    const rows = [];
    let row = [], field = "", inQ = false, i = 0;
    text = text.replace(/^﻿/, "");
    const delim = sniffDelimiter(text);
    while (i < text.length) {
      const c = text[i];
      if (inQ) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
          inQ = false; i++; continue;
        }
        field += c; i++; continue;
      }
      if (c === '"') { inQ = true; i++; continue; }
      if (c === delim) { row.push(field); field = ""; i++; continue; }
      if (c === "\r") { i++; continue; }
      if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; i++; continue; }
      field += c; i++;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    const nonEmpty = rows.filter(r => r.length > 1 || (r[0] || "").trim() !== "");
    if (nonEmpty.length < 2) throw new Error("CSV 至少需要標題列和一列資料");
    const header = nonEmpty[0].map(h => h.trim());
    const body = nonEmpty.slice(1).filter(r => r.length === header.length);
    return { header, rows: body, skipped: nonEmpty.length - 1 - body.length };
  }

  function sniffDelimiter(text) {
    const first = text.split("\n", 1)[0];
    const counts = [",", ";", "\t"].map(d => [d, first.split(d).length]);
    counts.sort((a, b) => b[1] - a[1]);
    return counts[0][1] > 1 ? counts[0][0] : ",";
  }

  const isNA = v => NA.has(String(v).trim().toLowerCase());
  const toNum = v => { const s = String(v).trim(); return s === "" ? NaN : Number(s); };

  function columnTypes(table) {
    return table.header.map((h, j) => {
      let numeric = true, nonNA = 0;
      const levels = new Set();
      for (const r of table.rows) {
        const v = r[j];
        if (isNA(v)) continue;
        nonNA++;
        if (numeric && !Number.isFinite(toNum(v))) numeric = false;
        if (levels.size <= MAX_CATEGORIES + 1) levels.add(String(v).trim());
      }
      return { name: h, numeric, nonNA, unique: levels.size };
    });
  }

  // --------------------------------------------------------- preparation
  function prepare(table, opts) {
    const target = opts.target, drop = new Set(opts.drop || []);
    const types = columnTypes(table);
    const tj = table.header.indexOf(target);
    if (tj < 0) throw new Error(`找不到目標欄位 ${target}`);
    if (!types[tj].numeric) throw new Error(`目標欄位 ${target} 不是數值`);
    const used = types.map((t, j) => j).filter(j => !drop.has(table.header[j]));
    const rowIds = [];
    const kept = [];
    table.rows.forEach((r, i) => {
      if (used.every(j => !isNA(r[j]))) { kept.push(r); rowIds.push(i); }
    });
    const n = kept.length;
    const numericCols = [], nominal = [], idLike = [];
    for (const j of used) {
      if (j === tj) continue;
      const t = types[j];
      if (t.numeric) numericCols.push(j);
      else {
        const levels = [...new Set(kept.map(r => String(r[j]).trim()))].sort();
        if (levels.length <= MAX_CATEGORIES) nominal.push({ j, col: t.name, levels });
        else idLike.push(t.name);
      }
    }
    const names = [], X = [], origin = [];
    for (const j of numericCols) {
      names.push(table.header[j]);
      X.push(Float64Array.from(kept, r => toNum(r[j])));
      origin.push({ col: table.header[j], kind: "numeric" });
    }
    for (const nm of nominal) {
      for (const lv of nm.levels.slice(1)) {
        names.push(`${nm.col}_${lv}`);
        X.push(Float64Array.from(kept, r => (String(r[nm.j]).trim() === lv ? 1 : 0)));
        origin.push({ col: nm.col, kind: "dummy", level: lv, reference: nm.levels[0] });
      }
    }
    const y = Float64Array.from(kept, r => toNum(r[tj]));
    return {
      target, names, X, y, n, nRaw: table.rows.length, rowIds, origin,
      nominal: nominal.map(m => ({ col: m.col, levels: m.levels })), idLike,
      dropped: [...drop].filter(d => table.header.includes(d)), types,
    };
  }

  // Remove constant and perfectly collinear columns (Gram–Schmidt on centred
  // training columns); returns the indices to keep plus the dropped names.
  function collinearScreen(prep, trainIdx) {
    const keep = [], dropped = [], basis = [];
    for (let j = 0; j < prep.names.length; j++) {
      const v = Float64Array.from(trainIdx, i => prep.X[j][i]);
      const m = mean(v);
      for (let i = 0; i < v.length; i++) v[i] -= m;
      const norm0 = Math.sqrt(dot(v, v));
      if (norm0 < 1e-12) { dropped.push({ name: prep.names[j], reason: "常數欄位" }); continue; }
      for (const b of basis) { const d = dot(v, b); for (let i = 0; i < v.length; i++) v[i] -= d * b[i]; }
      const norm = Math.sqrt(dot(v, v));
      if (norm / norm0 < 1e-8) { dropped.push({ name: prep.names[j], reason: "完全共線" }); continue; }
      for (let i = 0; i < v.length; i++) v[i] /= norm;
      basis.push(v); keep.push(j);
    }
    return { keep, dropped };
  }

  function subsetPrep(prep, keep) {
    return Object.assign({}, prep, {
      names: keep.map(j => prep.names[j]), X: keep.map(j => prep.X[j]),
      origin: keep.map(j => prep.origin[j]),
    });
  }

  // -------------------------------------- sklearn train_test_split (numpy)
  function MT19937(seed) {
    const mt = new Uint32Array(624);
    let idx = 624;
    mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const s = mt[i - 1] ^ (mt[i - 1] >>> 30);
      mt[i] = (Math.imul(1812433253, s) + i) >>> 0;
    }
    function twist() {
      for (let i = 0; i < 624; i++) {
        const yv = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
        let v = mt[(i + 397) % 624] ^ (yv >>> 1);
        if (yv & 1) v ^= 0x9908b0df;
        mt[i] = v >>> 0;
      }
      idx = 0;
    }
    return function next() {
      if (idx >= 624) twist();
      let yv = mt[idx++];
      yv ^= yv >>> 11;
      yv ^= (yv << 7) & 0x9d2c5680;
      yv ^= (yv << 15) & 0xefc60000;
      yv ^= yv >>> 18;
      return yv >>> 0;
    };
  }

  function randomInterval(next, max) {
    if (max === 0) return 0;
    let mask = max;
    mask |= mask >>> 1; mask |= mask >>> 2; mask |= mask >>> 4;
    mask |= mask >>> 8; mask |= mask >>> 16;
    let v;
    while ((v = (next() & mask) >>> 0) > max);
    return v;
  }

  function sklearnSplit(n, testSize, seed) {
    const next = MT19937(seed);
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i >= 1; i--) {
      const j = randomInterval(next, i);
      const t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    const nTest = Math.ceil(testSize * n);
    return { test: perm.slice(0, nTest), train: perm.slice(nTest) };
  }

  // ---------------------------------------------------------- linear algebra
  function dot(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function mean(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }

  function cholesky(A) {
    const n = A.length, L = A.map(() => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 1e-14 * Math.max(1, Math.abs(A[i][i]))) return null;
          L[i][i] = Math.sqrt(s);
        } else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }

  function cholSolve(L, b) {
    const n = L.length, z = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * z[k]; z[i] = s / L[i][i]; }
    for (let i = n - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
    return x;
  }

  function cholInverse(L) {
    const n = L.length, inv = [];
    for (let j = 0; j < n; j++) { const e = new Float64Array(n); e[j] = 1; inv.push(cholSolve(L, e)); }
    return inv; // symmetric, inv[j] is column j
  }

  // Centred Gram statistics over a set of rows (two-pass for accuracy)
  function gramStats(X, y, rows) {
    const p = X.length, n = rows.length;
    const mx = new Float64Array(p);
    for (let j = 0; j < p; j++) { let s = 0; for (const i of rows) s += X[j][i]; mx[j] = s / n; }
    let my = 0; for (const i of rows) my += y[i]; my /= n;
    const C = X.map((col, j) => Float64Array.from(rows, i => col[i] - mx[j]));
    const yc = Float64Array.from(rows, i => y[i] - my);
    const G = [];
    for (let j = 0; j < p; j++) {
      G.push(new Float64Array(p));
      for (let k = 0; k <= j; k++) { const v = dot(C[j], C[k]); G[j][k] = v; if (k < j) G[k][j] = v; }
    }
    const g = C.map(c => dot(c, yc));
    return { n, mx, my, G, g, syy: dot(yc, yc), rows };
  }

  function fitSubset(st, cols) {
    const A = cols.map(a => Float64Array.from(cols, b => st.G[a][b]));
    const L = cholesky(A);
    if (!L) return null;
    const beta = cholSolve(L, Float64Array.from(cols, a => st.g[a]));
    let b0 = st.my;
    cols.forEach((c, t) => { b0 -= beta[t] * st.mx[c]; });
    return { cols, beta, b0, L };
  }

  function predictRows(X, fit, rows) {
    const out = new Float64Array(rows.length);
    rows.forEach((i, r) => {
      let v = fit.b0;
      for (let t = 0; t < fit.cols.length; t++) v += fit.beta[t] * X[fit.cols[t]][i];
      out[r] = v;
    });
    return out;
  }

  function errStats(y, rows, pred) {
    let sse = 0, sae = 0, my = 0;
    rows.forEach(i => { my += y[i]; });
    my /= rows.length;
    let sst = 0;
    rows.forEach((i, r) => { const e = y[i] - pred[r]; sse += e * e; sae += Math.abs(e); sst += (y[i] - my) ** 2; });
    return { rmse: Math.sqrt(sse / rows.length), r2: 1 - sse / sst, mae: sae / rows.length, sse };
  }

  // -------------------------------------------------------------- Problem
  function Problem(prep, split) {
    const X = prep.X, y = prep.y, p = prep.names.length;
    const train = split.train, test = split.test;
    const full = gramStats(X, y, train);
    // KFold(5, shuffle=False) over training positions, in split order
    const ntr = train.length, folds = [];
    let start = 0;
    for (let f = 0; f < CV_FOLDS; f++) {
      const size = Math.floor(ntr / CV_FOLDS) + (f < ntr % CV_FOLDS ? 1 : 0);
      const val = train.slice(start, start + size);
      const fit = train.slice(0, start).concat(train.slice(start + size));
      folds.push({ st: gramStats(X, y, fit), val });
      start += size;
    }
    const cache = new Map();

    function testRMSE(cols) {
      const f = fitSubset(full, cols);
      if (!f) return Infinity;
      return errStats(y, test, predictRows(X, f, test)).rmse;
    }
    function cvRMSE(cols) {
      let s = 0;
      for (const fd of folds) {
        const f = fitSubset(fd.st, cols);
        if (!f) return Infinity;
        s += errStats(y, fd.val, predictRows(X, f, fd.val)).rmse;
      }
      return s / folds.length;
    }
    function evaluate(cols) {
      const f = fitSubset(full, cols);
      if (!f) return null;
      const tr = errStats(y, train, predictRows(X, f, train));
      const te = errStats(y, test, predictRows(X, f, test));
      return {
        train_rmse: tr.rmse, train_r2: tr.r2, cv_rmse: cvRMSE(cols),
        test_rmse: te.rmse, test_r2: te.r2, test_mae: te.mae,
      };
    }

    // standardised Gram for path / RFE methods
    const sd = Float64Array.from(full.G, (row, j) => Math.sqrt(row[j] / ntr));
    const live = [...Array(p).keys()].filter(j => sd[j] > 0);
    const Gs = full.G.map((row, j) => Float64Array.from(row, (v, k) => v / (sd[j] * sd[k] * ntr)));
    const cs = Float64Array.from(full.g, (v, j) => v / (sd[j] * ntr));

    function pathOrder(l1) {
      const amax = Math.max(...live.map(j => Math.abs(cs[j]))) / l1;
      const w = new Float64Array(p), q = new Float64Array(p);
      const entry = new Array(p).fill(N_ALPHAS), mag = new Float64Array(p);
      for (let a = 0; a < N_ALPHAS; a++) {
        const alpha = amax * Math.pow(EPS, a / (N_ALPHAS - 1));
        for (let sweep = 0; sweep < 100000; sweep++) {
          let maxDelta = 0, maxW = 0;
          for (const j of live) {
            const rho = cs[j] - (q[j] - Gs[j][j] * w[j]);
            const thr = alpha * l1;
            const wn = (Math.sign(rho) * Math.max(Math.abs(rho) - thr, 0)) / (Gs[j][j] + alpha * (1 - l1));
            const d = wn - w[j];
            if (d !== 0) {
              for (const k of live) q[k] += Gs[k][j] * d;
              w[j] = wn;
              maxDelta = Math.max(maxDelta, Math.abs(d));
            }
            maxW = Math.max(maxW, Math.abs(w[j]));
          }
          if (maxDelta <= 1e-13 * Math.max(1, maxW)) break;
        }
        for (const j of live) if (entry[j] === N_ALPHAS && w[j] !== 0) { entry[j] = a; mag[j] = Math.abs(w[j]); }
      }
      return [...Array(p).keys()].sort((i, j) => entry[i] - entry[j] || mag[j] - mag[i] || i - j);
    }

    function rfeOrder() {
      let remaining = [...live];
      const eliminated = [];
      while (remaining.length > 1) {
        const A = remaining.map(a => Float64Array.from(remaining, b => Gs[a][b]));
        const L = cholesky(A);
        const beta = L ? cholSolve(L, Float64Array.from(remaining, a => cs[a])) : new Float64Array(remaining.length);
        let worst = 0;
        for (let t = 1; t < remaining.length; t++) if (Math.abs(beta[t]) < Math.abs(beta[worst])) worst = t;
        eliminated.push(remaining[worst]);
        remaining = remaining.filter((_, t) => t !== worst);
      }
      const dead = [...Array(p).keys()].filter(j => !live.includes(j));
      return remaining.concat(eliminated.reverse(), dead);
    }

    function kbestOrder() {
      const F = [...Array(p).keys()].map(j => {
        if (sd[j] === 0) return -Infinity;
        const r = full.g[j] / Math.sqrt(full.G[j][j] * full.syy);
        return (r * r) / (1 - r * r) * (ntr - 2);
      });
      return [...Array(p).keys()].sort((i, j) => F[j] - F[i] || i - j);
    }

    function forward(score) {
      let chosen = [];
      const out = [];
      for (let step = 0; step < p; step++) {
        let best = -1, bestS = Infinity;
        for (let j = 0; j < p; j++) {
          if (chosen.includes(j)) continue;
          const s = score(chosen.concat([j]));
          if (s < bestS) { best = j; bestS = s; }
        }
        if (best < 0) break;
        chosen = chosen.concat([best]);
        out.push(chosen.slice());
      }
      return out;
    }

    function exhaustive(score) {
      const out = [];
      for (let k = 1; k <= p; k++) {
        const c = [...Array(k).keys()];
        let best = null, bestS = Infinity;
        while (true) {
          const s = score(c);
          if (s < bestS) { best = c.slice(); bestS = s; }
          let i = k - 1;
          while (i >= 0 && c[i] === p - k + i) i--;
          if (i < 0) break;
          c[i]++;
          for (let t = i + 1; t < k; t++) c[t] = c[t - 1] + 1;
        }
        out.push(best);
      }
      return out;
    }

    // method key -> list of subsets (k = 1..p)
    function subsets(key) {
      if (cache.has("s:" + key)) return cache.get("s:" + key);
      let subs;
      const fromOrder = o => o.map((_, k) => o.slice(0, k + 1));
      switch (key) {
        case "lasso": subs = fromOrder(pathOrder(1)); break;
        case "elasticnet": subs = fromOrder(pathOrder(ENET_L1)); break;
        case "rfe": subs = fromOrder(rfeOrder()); break;
        case "kbest": subs = fromOrder(kbestOrder()); break;
        case "forward_test": subs = forward(testRMSE); break;
        case "forward_cv": subs = forward(cvRMSE); break;
        case "exhaustive_test": subs = p <= MAX_EXHAUSTIVE_P ? exhaustive(testRMSE) : null; break;
        case "exhaustive_cv": subs = p <= MAX_EXHAUSTIVE_P ? exhaustive(cvRMSE) : null; break;
        default: throw new Error("unknown method " + key);
      }
      cache.set("s:" + key, subs);
      return subs;
    }

    function table(key) {
      if (cache.has("t:" + key)) return cache.get("t:" + key);
      const subs = subsets(key);
      const rows = subs ? subs.map((cols, i) => Object.assign(
        { k: i + 1, cols, features: cols.map(c => prep.names[c]) }, evaluate(cols))) : null;
      cache.set("t:" + key, rows);
      return rows;
    }

    function inference(cols) {
      const f = fitSubset(full, cols);
      if (!f) return null;
      const n = ntr, k = cols.length, df = n - k - 1;
      const pred = predictRows(X, f, train);
      const resid = Float64Array.from(train, (i, r) => y[i] - pred[r]);
      const sse = dot(resid, resid), sst = full.syy;
      const s2 = sse / df;
      const inv = cholInverse(f.L);
      const tq = tQuantile(0.975, df);
      const coefs = [];
      let v0 = 1 / n;
      for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) v0 += full.mx[cols[a]] * inv[b][a] * full.mx[cols[b]];
      const mk = (name, coef, se, vif) => {
        const t = coef / se;
        return { name, coef, se, t, p: tTwoSided(t, df), ci_low: coef - tq * se, ci_high: coef + tq * se, vif };
      };
      coefs.push(mk("const", f.b0, Math.sqrt(s2 * v0), null));
      cols.forEach((c, t) => coefs.push(mk(prep.names[c], f.beta[t], Math.sqrt(s2 * inv[t][t]),
        k === 1 ? 1 : inv[t][t] * full.G[c][c])));
      const r2 = 1 - sse / sst;
      const fstat = ((sst - sse) / k) / s2;
      // assumption diagnostics
      let num = 0;
      for (let i = 1; i < resid.length; i++) num += (resid[i] - resid[i - 1]) ** 2;
      const dw = num / sse;
      const me = mean(resid);
      let m2 = 0, m3 = 0, m4 = 0;
      for (const e of resid) { const d = e - me; m2 += d * d; m3 += d ** 3; m4 += d ** 4; }
      m2 /= n; m3 /= n; m4 /= n;
      const skew = m3 / Math.pow(m2, 1.5), kurt = m4 / (m2 * m2);
      const jb = n / 6 * (skew * skew + (kurt - 3) ** 2 / 4);
      // Breusch–Pagan (Koenker): n * R² of e² on the same regressors
      const e2 = Float64Array.from(resid, e => e * e);
      const me2 = mean(e2);
      const ge = cols.map(c => { let s = 0; train.forEach((i, r) => { s += (X[c][i] - full.mx[c]) * (e2[r] - me2); }); return s; });
      const ga = cholSolve(f.L, Float64Array.from(ge));
      let sstE = 0; for (const v of e2) sstE += (v - me2) ** 2;
      const ssrE = dot(ga, Float64Array.from(ge));
      const lm = n * (ssrE / sstE);
      return {
        features: cols.map(c => prep.names[c]), cols, coefficients: coefs,
        r2_train: r2, adj_r2_train: 1 - (1 - r2) * (n - 1) / df, f_stat: fstat,
        f_pvalue: fSurvival(fstat, k, df), df_model: k, df_resid: df,
        durbin_watson: dw, skew, kurtosis: kurt, jarque_bera: jb, jarque_bera_p: chi2Survival(jb, 2),
        breusch_pagan: lm, breusch_pagan_p: chi2Survival(lm, k),
        max_vif: k === 1 ? 1 : Math.max(...coefs.slice(1).map(c => c.vif)),
        sigma: Math.sqrt(s2), n_train: n, fit: f,
        // point prediction + 95% prediction interval for x (values in `cols` order)
        predict(xs) {
          let yhat = f.b0;
          xs.forEach((v, t) => { yhat += f.beta[t] * v; });
          const d = Float64Array.from(cols, (c, t) => xs[t] - full.mx[c]);
          const se = Math.sqrt(s2 * (1 + 1 / n + dot(d, cholSolve(f.L, d))));
          return { yhat, lo: yhat - tq * se, hi: yhat + tq * se };
        },
      };
    }

    function testPredictions(cols) {
      const f = fitSubset(full, cols);
      if (!f) return null;
      const pred = predictRows(X, f, test);
      return test.map((i, r) => ({ row: i, id: prep.rowIds[i], actual: y[i], pred: pred[r], resid: y[i] - pred[r] }));
    }

    return {
      prep, split, p, ntr, nte: test.length, table, subsets, inference, testPredictions,
      exhaustiveAllowed: p <= MAX_EXHAUSTIVE_P,
      range(j) {
        let lo = Infinity, hi = -Infinity, s = 0;
        for (const i of train) { const v = X[j][i]; lo = Math.min(lo, v); hi = Math.max(hi, v); s += v; }
        return { min: lo, max: hi, mean: s / train.length };
      },
    };
  }

  // ------------------------------------------------------- distributions
  function logGamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let yv = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (const ci of c) ser += ci / ++yv;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }

  function betacf(a, b, x) {
    const MAXIT = 300, EPSV = 3e-16, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPSV) break;
    }
    return h;
  }

  function betai(a, b, x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }

  function tTwoSided(t, df) { return betai(df / 2, 0.5, df / (df + t * t)); }

  function tQuantile(prob, df) {
    const target = 2 * (1 - prob);
    let lo = 0, hi = 1e3;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tTwoSided(mid, df) > target) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  function fSurvival(F, d1, d2) { return F <= 0 ? 1 : betai(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)); }

  function gammaQ(a, x) {
    if (x <= 0) return 1;
    const gln = logGamma(a);
    if (x < a + 1) {
      let ap = a, sum = 1 / a, del = sum;
      for (let n = 0; n < 1000; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 3e-16) break; }
      return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
    }
    let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-16) break;
    }
    return Math.exp(-x + a * Math.log(x) - gln) * h;
  }

  function chi2Survival(x, k) { return gammaQ(k / 2, x / 2); }

  global.MLR = {
    parseCSV, columnTypes, prepare, collinearScreen, subsetPrep, sklearnSplit, Problem,
    stats: { tTwoSided, tQuantile, fSurvival, chi2Survival },
    MAX_EXHAUSTIVE_P,
  };
})(window);
