/* UI for the MLR Feature Selection Explorer. All numbers come from js/engine.js. */
(function () {
  "use strict";

  const METHODS = [
    { id: "lasso", name: "Lasso", tag: "L1 路徑", color: "--s1" },
    { id: "forward", name: "Forward SFS", tag: "貪婪前進", color: "--s2", crit: true },
    { id: "rfe", name: "RFE", tag: "遞迴消去", color: "--s3" },
    { id: "elasticnet", name: "ElasticNet", tag: "L1 + L2", color: "--s4" },
    { id: "kbest", name: "SelectKBest", tag: "F 檢定", color: "--s5" },
    { id: "exhaustive", name: "Exhaustive", tag: "最佳子集", color: "--s6", crit: true },
    { id: "backward", name: "Backward (p 值)", tag: "講義最常用", color: "--s7" },
  ];
  const N_METHODS = METHODS.length;
  const METHOD_DESC = {
    lasso: "在標準化訓練資料上計算 Lasso 路徑（200 個 α），依特徵「進入模型」的先後排序，取前 k 個再用 OLS 重新估計。",
    forward: "Forward Sequential Selection：每一步加入讓準則（CV 或測試集 RMSE）最低的特徵。",
    rfe: "Recursive Feature Elimination：在標準化資料上反覆用 OLS 擬合，每次刪掉 |係數| 最小的特徵。",
    elasticnet: "ElasticNet 路徑（l1_ratio = 0.5），同時有 L1 篩選和 L2 收縮，依進入順序排序。",
    kbest: "SelectKBest(f_regression)：依單變量 F 統計量（等同 |相關係數|）排序，取前 k 個。",
    exhaustive: "暴力搜尋全部 2^p − 1 個子集，對每個 k 找準則最低的組合（p ≤ 16 時可用）。",
    backward: "Backward Elimination（講義 L6 Part 5「最常用」）：從全部特徵開始，每次刪掉 p 值最大（最不顯著）的特徵；所有係數 p < 0.05 時就是講義的停止點。",
  };
  const OPS = { ">=": (a, b) => a >= b, ">": (a, b) => a > b, "<=": (a, b) => a <= b, "<": (a, b) => a < b };
  const DEFAULT_SUCCESS = [
    { text: "測試集 R² ≥ 0.70", metric: "test_r2", op: ">=", value: 0.7 },
    { text: "所有係數顯著（p < 0.05）", metric: "max_p", op: "<", value: 0.05 },
    { text: "共線性可接受（最大 VIF < 10）", metric: "max_vif", op: "<", value: 10 },
  ];
  // Set to the GitHub repository URL after the repo is created; shows a link in the header.
  const REPO_URL = "";
  const SAMPLES = [
    ["50 Startups", "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"],
    ["Boston Housing", "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"],
    ["Auto MPG", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"],
    ["Diamonds", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv"],
  ];

  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const state = {
    registry: [], cur: null, method: "lasso", criterion: "cv", k: null,
    residAll: false, tab: 0, pred: {}, charts: {}, customTable: null, customDrop: new Set(),
    loadToken: 0, bundles: {}, tables: {},
  };

  // ------------------------------------------------------------ formatting
  function fmtU(v, d) {
    const u = state.cur.unit || {};
    const dec = d ?? (u.decimals ?? 2);
    const s = Math.abs(v).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
    return `${v < 0 ? "−" : ""}${u.prefix || ""}${s}${u.suffix || ""}`;
  }
  const fmtN = (v, d = 4) => Number.isFinite(v) ? v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "–";
  function fmtCoef(v) {
    const a = Math.abs(v);
    if (a >= 1000) return fmtN(v, 2);
    if (a !== 0 && a < 0.001) return v.toExponential(3);
    return fmtN(v, 4);
  }
  const fmtP = p => (p < 0.0001 ? "< 0.0001" : p.toFixed(4));
  const label = name => (state.cur.labels && state.cur.labels[name]) || name;
  const shortLabel = name => label(name).replace(/（.*）/, "");

  // ------------------------------------------------------------ helpers
  const methodById = id => METHODS.find(m => m.id === id);
  const keyOf = (m, crit = state.criterion) => (methodById(m).crit ? `${m}_${crit}` : m);
  const metricOf = (crit = state.criterion) => (crit === "cv" ? "cv_rmse" : "test_rmse");

  function tableFor(m, crit) {
    const P = state.cur.problem;
    if (m === "exhaustive" && !P.exhaustiveAllowed) return null;
    return P.table(keyOf(m, crit));
  }
  function sweetSpot(rows, crit = state.criterion) {
    const metric = metricOf(crit);
    let best = rows[0];
    for (const r of rows) if (r[metric] < best[metric] - 1e-12) best = r;
    return best;
  }
  function globalBest(crit = state.criterion) {
    let best = null;
    for (const m of METHODS) {
      const rows = tableFor(m.id, crit);
      if (!rows) continue;
      const s = sweetSpot(rows, crit);
      if (!best || s[metricOf(crit)] < best.row[metricOf(crit)] - 1e-12) best = { method: m.id, row: s };
    }
    return best;
  }
  function activeModel() {
    if (state.method === "compare") return globalBest();
    const rows = tableFor(state.method);
    return { method: state.method, row: rows[state.k - 1] };
  }

  // ------------------------------------------------------------ loading
  async function loadBundle(file) {
    if (!state.bundles[file]) state.bundles[file] = await (await fetch(file)).json();
    return state.bundles[file];
  }
  async function loadTable(csv) {
    if (!state.tables[csv]) state.tables[csv] = MLR.parseCSV(await (await fetch(csv)).text());
    return state.tables[csv];
  }

  async function init() {
    try {
      state.registry = await (await fetch("problems/index.json")).json();
    } catch (e) {
      state.registry = [];
    }
    const sel = $("problem");
    sel.innerHTML = state.registry.map(r => `<option value="${esc(r.id)}">${esc(r.short)}</option>`).join("") +
      `<option value="custom">Custom · 自訂資料集（輸入任意 CSV 網址或上傳）</option>`;
    sel.addEventListener("change", () => selectProblem(sel.value));

    $("methods").innerHTML = METHODS.map(m =>
      `<button class="chip" data-m="${m.id}"><span class="dot" style="background:var(${m.color})"></span>${m.name}<small>${m.tag}</small></button>`
    ).join("") + `<button class="chip" data-m="compare">比較全部 ${N_METHODS} 種方法</button>`;
    $("methods").addEventListener("click", e => {
      const b = e.target.closest("[data-m]");
      if (!b || b.disabled) return;
      state.method = b.dataset.m;
      if (state.method !== "compare") state.k = sweetSpot(tableFor(state.method)).k;
      render();
    });
    $("criterion").addEventListener("click", e => {
      const b = e.target.closest("[data-c]");
      if (!b) return;
      state.criterion = b.dataset.c;
      if (state.method !== "compare") state.k = sweetSpot(tableFor(state.method)).k;
      render();
    });
    $("ks").addEventListener("click", e => {
      const b = e.target.closest("[data-k]");
      if (!b) return;
      if (state.method === "compare") state.method = globalBest().method;
      state.k = +b.dataset.k;
      render();
    });
    $("resid-toggle").addEventListener("click", () => { state.residAll = !state.residAll; renderResiduals(); });
    $("download-model").addEventListener("click", downloadModel);
    $("crisp-tabs").addEventListener("click", e => {
      const b = e.target.closest("[data-t]");
      if (b) { state.tab = +b.dataset.t; renderCrisp(); }
    });
    $("sel-table").addEventListener("click", e => {
      const tr = e.target.closest("tr[data-k]");
      if (!tr) return;
      if (tr.dataset.m) state.method = tr.dataset.m;
      state.k = +tr.dataset.k;
      render();
    });
    setupCustom();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => state.cur && render());

    if (REPO_URL) { $("repo-link").href = REPO_URL; $("repo-link").classList.remove("hidden"); }
    const params = new URLSearchParams(location.search);
    let want = params.get("problem") || (params.get("csv") ? "custom" : state.registry[0]?.id || "custom");
    if (want !== "custom" && !state.registry.find(r => r.id === want)) want = state.registry[0]?.id || "custom";
    sel.value = want;
    await selectProblem(want);
    if (want === "custom" && params.get("csv")) {
      $("csv-url").value = params.get("csv");
      await loadCustomURL(params.get("target"));
    }
  }

  async function selectProblem(id) {
    const token = ++state.loadToken;
    const params = new URLSearchParams(location.search);
    params.set("problem", id);
    if (id !== "custom") { params.delete("csv"); params.delete("target"); }
    history.replaceState(null, "", "?" + params.toString());
    $("custom").classList.toggle("hidden", id !== "custom");
    if (id === "custom") {
      $("results").classList.toggle("hidden", !(state.cur && state.cur.custom));
      $("method-controls").classList.toggle("hidden", !(state.cur && state.cur.custom));
      if (state.cur && state.cur.custom) render();
      else setHero("自訂資料集 Custom Dataset", "貼上任何 CSV 的 raw 網址（例如 GitHub raw）或上傳檔案，選擇目標欄位後，網站會在瀏覽器內跑完整的 CRISP-DM 與七種特徵選擇。", []);
      return;
    }
    $("results").classList.remove("hidden");
    $("method-controls").classList.remove("hidden");
    const entry = state.registry.find(r => r.id === id);
    document.body.classList.add("loading");
    try {
      const bundle = await loadBundle(entry.file);
      const table = await loadTable(bundle.csv);
      if (token !== state.loadToken) return; // the user picked another problem meanwhile
      const prep = MLR.prepare(table, { target: bundle.target, drop: bundle.drop });
      const jsSplit = MLR.sklearnSplit(prep.n, bundle.test_size, bundle.random_state);
      const split = { train: bundle.split.train_idx, test: bundle.split.test_idx };
      const splitMatches = sameArray(jsSplit.train, split.train) && sameArray(jsSplit.test, split.test);
      const namesMatch = sameArray(prep.names, bundle.encoded_features);
      state.cur = {
        id, bundle, custom: false, title: bundle.title, table, prep,
        problem: MLR.Problem(prep, split), labels: bundle.labels, unit: bundle.unit,
        target: bundle.target, targetLabel: bundle.target_label, source: bundle.source,
        csvUrl: bundle.csv_url, splitMatches, namesMatch, seed: bundle.random_state,
        testSize: bundle.test_size, collinear: [],
      };
      startProblem();
    } catch (err) {
      setHero("載入失敗", String(err), []);
      console.error(err);
    } finally {
      document.body.classList.remove("loading");
    }
  }

  function startProblem() {
    state.pred = {};
    state.residAll = false;
    if (state.method === "exhaustive" && !state.cur.problem.exhaustiveAllowed) state.method = "lasso";
    if (state.method !== "compare") state.k = sweetSpot(tableFor(state.method)).k;
    render();
  }

  const sameArray = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

  // ------------------------------------------------------------ custom data
  function setupCustom() {
    $("samples").insertAdjacentHTML("beforeend", SAMPLES.map(([n, u]) =>
      `<button class="chip" data-url="${esc(u)}">${esc(n)}</button>`).join(""));
    $("samples").addEventListener("click", e => {
      const b = e.target.closest("[data-url]");
      if (!b) return;
      $("csv-url").value = b.dataset.url;
      loadCustomURL();
    });
    $("load-url").addEventListener("click", () => loadCustomURL());
    $("csv-url").addEventListener("keydown", e => { if (e.key === "Enter") loadCustomURL(); });
    $("csv-file").addEventListener("change", async e => {
      const f = e.target.files[0];
      if (!f) return;
      e.target.value = ""; // so choosing the same file again still fires "change"
      customLoaded(await f.text(), f.name, null);
    });
    $("drop-chips").addEventListener("click", e => {
      const b = e.target.closest("[data-col]");
      if (!b) return;
      const c = b.dataset.col;
      state.customDrop.has(c) ? state.customDrop.delete(c) : state.customDrop.add(c);
      b.classList.toggle("on");
    });
    $("analyze").addEventListener("click", analyzeCustom);
  }

  function status(msg, kind) {
    const el = $("custom-status");
    el.className = "status " + (kind || "");
    el.innerHTML = msg;
  }

  async function loadCustomURL(target) {
    const url = $("csv-url").value.trim();
    if (!/^https?:\/\//i.test(url)) { status("請輸入 http(s) 開頭的 CSV 網址。", "err"); return; }
    status("下載中…");
    $("load-url").disabled = true;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await customLoaded(await res.text(), url.split("/").pop(), url, target);
    } catch (err) {
      status(`無法下載：${esc(err.message)}。請確認是 raw CSV 網址，且伺服器允許跨網域讀取（GitHub raw、gist 都可以）；也可以改用「上傳 CSV」。`, "err");
    } finally {
      $("load-url").disabled = false;
    }
  }

  // A custom CSV that is the same data as a built-in problem gets that problem's
  // split settings, so its numbers match the built-in page exactly.
  function fingerprint(table) {
    const norm = v => { const t = String(v).trim(); const n = Number(t); return t !== "" && Number.isFinite(n) ? String(n) : t; };
    return table.header.map(norm).join("\u0001") + "\u0002" + table.rows.map(r => r.map(norm).join("\u0001")).join("\u0002");
  }
  async function matchBuiltin(table) {
    let fp = null;
    for (const r of state.registry) {
      try {
        const b = await loadBundle(r.file);
        const t = await loadTable(b.csv);
        if (t.rows.length !== table.rows.length || t.header.length !== table.header.length) continue;
        fp = fp || fingerprint(table);
        if (fingerprint(t) === fp) return Object.assign({ short: r.short }, b);
      } catch (e) { /* a missing built-in file only disables the match */ }
    }
    return null;
  }

  async function customLoaded(text, name, url, wantTarget) {
    let table;
    try { table = MLR.parseCSV(text); } catch (err) { status(esc(err.message), "err"); return; }
    const types = MLR.columnTypes(table);
    const numeric = types.filter(t => t.numeric).map(t => t.name);
    if (!numeric.length) { status("找不到數值欄位，無法做迴歸。", "err"); return; }
    state.customTable = { table, name, url, types };
    const named = numeric.find(n => /^(target|y|price|profit|medv|mpg|saleprice|charges|quality)$/i.test(n));
    const guess = wantTarget && numeric.includes(wantTarget) ? wantTarget : named || numeric[numeric.length - 1];
    $("target").innerHTML = numeric.map(n => `<option ${n === guess ? "selected" : ""}>${esc(n)}</option>`).join("");
    state.customDrop = new Set(table.header.filter(h => /^(id|unnamed.*|index|row\.?names?)$/i.test(h) || h === ""));
    renderDropChips();
    $("target").onchange = renderDropChips;
    $("custom-config").classList.remove("hidden");
    let msg = `已讀取 <b>${esc(name)}</b>：${table.rows.length} 列 × ${table.header.length} 欄` +
      (table.skipped ? `（${table.skipped} 列欄位數不符已略過）` : "") + "。請確認目標欄位後按「開始分析」。";
    const match = await matchBuiltin(table);
    $("seed").value = 42;
    $("test-size").value = "0.2";
    if (match) {
      if (!wantTarget && numeric.includes(match.target)) $("target").value = match.target;
      $("seed").value = match.random_state;
      $("test-size").value = String(match.test_size);
      state.customDrop = new Set(match.drop);
      renderDropChips();
      msg += `<br>這份資料和「${esc(match.short)}」相同，已自動套用相同的 random_state = ${match.random_state}、test_size = ${match.test_size}，分析結果會和內建問題一致（可自行修改）。`;
    }
    status(msg, "ok");
    if (wantTarget) analyzeCustom();
  }

  function renderDropChips() {
    const { table, types } = state.customTable;
    const target = $("target").value;
    $("drop-chips").innerHTML = table.header.filter(h => h !== target).map((h, i) => {
      const t = types[table.header.indexOf(h)];
      const kind = t.numeric ? "數值" : t.unique <= 20 ? "名目→one-hot" : "文字→自動排除";
      return `<button class="chip ${state.customDrop.has(h) ? "on" : ""}" data-col="${esc(h)}">${esc(h || "(空白)")}<small>${kind}</small></button>`;
    }).join("");
  }

  function analyzeCustom() {
    const { table, name, url } = state.customTable;
    const target = $("target").value;
    const testSize = +$("test-size").value, seed = Math.max(0, Math.floor(+$("seed").value || 0));
    status("分析中…");
    setTimeout(() => {
      try {
        if (seed > 4294967295) throw new Error("random_state 必須介於 0 和 4294967295 之間");
        const drop = [...state.customDrop].filter(c => c !== target);
        let prep = MLR.prepare(table, { target, drop });
        if (prep.n < 10) throw new Error(`去除缺失值後只剩 ${prep.n} 列，資料太少`);
        const split = MLR.sklearnSplit(prep.n, testSize, seed);
        const screen = MLR.collinearScreen(prep, split.train);
        prep = MLR.subsetPrep(prep, screen.keep);
        if (!prep.names.length) throw new Error("沒有可用的特徵欄位");
        if (split.train.length < prep.names.length + 3) throw new Error(`訓練集 ${split.train.length} 列不足以估計 ${prep.names.length} 個特徵`);
        state.cur = {
          id: "custom", bundle: null, custom: true, title: name, table, prep,
          problem: MLR.Problem(prep, split), labels: {}, unit: { prefix: "", suffix: "", decimals: 4 },
          target, targetLabel: target, source: { name: url ? "使用者輸入的網址" : "使用者上傳的檔案", url },
          csvUrl: url, splitMatches: null, namesMatch: true, seed, testSize, collinear: screen.dropped,
        };
        if (url) {
          const params = new URLSearchParams({ problem: "custom", csv: url, target });
          history.replaceState(null, "", "?" + params.toString());
        }
        $("results").classList.remove("hidden");
        $("method-controls").classList.remove("hidden");
        status(`完成：${prep.n} 列、${prep.names.length} 個特徵（${split.train.length} 訓練 / ${split.test.length} 測試）。` +
          (screen.dropped.length ? ` 已自動排除：${screen.dropped.map(d => `${esc(d.name)}（${d.reason}）`).join("、")}。` : ""), "ok");
        startProblem();
      } catch (err) {
        status(esc(err.message), "err");
        console.error(err);
      }
    }, 30);
  }

  // ------------------------------------------------------------ render
  function render() {
    const C = state.cur;
    if (!C) return;
    const P = C.problem;
    $("problem").value = C.custom ? "custom" : C.id;
    const src = $("source-link");
    src.textContent = C.source.url ? `資料來源：${C.source.name}` : "";
    if (C.source.url) src.href = C.source.url; else src.removeAttribute("href");

    // method chips
    document.querySelectorAll("#methods [data-m]").forEach(b => {
      b.classList.toggle("on", b.dataset.m === state.method);
      b.setAttribute("aria-pressed", String(b.dataset.m === state.method));
      b.disabled = b.dataset.m === "exhaustive" && !P.exhaustiveAllowed;
      b.title = b.disabled ? `p = ${P.p} > ${MLR.MAX_EXHAUSTIVE_P}，暴力搜尋太慢` : "";
    });
    document.querySelectorAll("#criterion [data-c]").forEach(b => {
      b.classList.toggle("on", b.dataset.c === state.criterion);
      b.setAttribute("aria-pressed", String(b.dataset.c === state.criterion));
    });
    $("criterion-hint").innerHTML = state.criterion === "cv"
      ? "只用訓練集的 5-fold CV RMSE 來決定 Forward / Exhaustive 的特徵與最佳 k，測試集只拿來做最後評估。"
      : "⚠ 這是講義 L6 的做法：直接用測試集 RMSE 挑特徵。測試分數會偏樂觀（資料洩漏），這裡只用來重現講義結果。";

    const model = activeModel();
    const rows = state.method === "compare" ? null : tableFor(state.method);
    const ss = rows ? sweetSpot(rows) : null;
    if (state.method === "compare") state.k = model.row.k;

    // nested methods add one feature per k, so the chip can name it; exhaustive subsets are not nested
    const nested = rows && state.method !== "exhaustive";
    const stop = state.method === "backward" ? P.backwardStop() : null;
    $("ks").innerHTML = Array.from({ length: P.p }, (_, i) => {
      const k = i + 1;
      const star = ss && ss.k === k, full = k === P.p;
      const extra = star ? "★ 最佳" : full ? "⚠ 全部特徵" : k === stop ? "p 值停止點"
        : nested ? "+ " + esc(shortLabel(rows[i].features[rows[i].features.length - 1])) : "";
      const on = state.method === "compare" ? false : state.k === k;
      return `<button class="chip ${on ? "on" : ""} ${star ? "star" : ""} ${full ? "full" : ""}" data-k="${k}" aria-pressed="${on}">k = ${k}${extra ? `<small>${extra}</small>` : ""}</button>`;
    }).join("");

    renderHero();
    const inf = P.inference(model.row.cols);
    state.inf = inf;
    state.model = model;
    renderKPIs(model, inf, ss);
    renderActive(model);
    renderCharts(model);
    renderEquation(inf);
    renderCoefs(inf);
    renderPredictor(inf);
    renderScatter(model);
    renderAssumptions(inf);
    renderResiduals();
    renderSelTable(model);
    renderCrisp();
  }

  function setHero(title, desc, badges) {
    $("hero-title").textContent = title;
    $("hero-desc").textContent = desc;
    $("hero-badges").innerHTML = badges.map(b => `<span class="badge">${b}</span>`).join("");
  }

  function renderHero() {
    const C = state.cur, P = C.problem;
    const gb = globalBest();
    const crit = state.criterion === "cv" ? "CV" : "Test";
    const fullRow = tableFor("lasso")[P.p - 1];
    const desc = C.bundle ? C.bundle.business.objective
      : `以「${C.target}」為目標，在瀏覽器內比較七種特徵選擇方法與 OLS，找出最精簡、泛化最好的特徵組合。`;
    setHero(C.title, desc, [
      `📊 ${C.prep.n} 筆 · ${P.p} 個特徵（${P.ntr} 訓練 / ${P.nte} 測試）`,
      `🧪 ${N_METHODS} 種特徵選擇方法`,
      `★ 全域最佳（${crit} 準則）：${esc(methodById(gb.method).name)} k=${gb.row.k}，Test RMSE ${fmtU(gb.row.test_rmse)}`,
      `⚠ 全部 ${P.p} 個特徵：Test RMSE ${fmtU(fullRow.test_rmse)}`,
    ]);
  }

  function renderKPIs(model, inf, ss) {
    const r = model.row, P = state.cur.problem;
    const tagK = state.method === "compare" ? `<span class="tag good">全域最佳</span>`
      : ss && ss.k === r.k ? `<span class="tag good">★ 最佳 k</span>`
      : r.k === P.p ? `<span class="tag warn">⚠ 全部特徵</span>` : `<span class="tag info">k = ${r.k}</span>`;
    const fTag = inf.f_pvalue < 0.05 ? `<span class="tag good">p ${inf.f_pvalue < 0.0001 ? "< 0.0001" : "= " + inf.f_pvalue.toFixed(4)}</span>`
      : `<span class="tag warn">p = ${inf.f_pvalue.toFixed(3)}</span>`;
    const tiles = [
      ["Test RMSE", tagK, fmtU(r.test_rmse), `5-fold CV RMSE ${fmtU(r.cv_rmse)} · 越低越好`],
      ["R² Score（測試集）", `<span class="tag info">${(r.test_r2 * 100).toFixed(1)}% Fit</span>`, fmtN(r.test_r2, 4), `解釋測試集 ${(r.test_r2 * 100).toFixed(2)}% 的變異`],
      ["Adjusted R²（訓練集）", "", fmtN(inf.adj_r2_train, 4), `訓練集 R² ${fmtN(inf.r2_train, 4)}，懲罰多餘特徵`],
      ["MAE（測試集）", "", fmtU(r.test_mae), "平均絕對誤差"],
      ["F-Statistic", fTag, fmtN(inf.f_stat, 1), `F(${inf.df_model}, ${inf.df_resid})，整體模型顯著性`],
    ];
    $("kpis").innerHTML = tiles.map(([l, t, v, n]) =>
      `<div class="kpi"><div class="lbl"><span>${l}</span>${t}</div><div class="val num">${v}</div><div class="note">${n}</div></div>`).join("");
  }

  function renderActive(model) {
    const feats = model.row.features;
    const mname = methodById(model.method).name;
    $("active-feats").innerHTML = `<b>目前模型</b>：${esc(mname)} · k = ${model.row.k}　` +
      feats.map(f => `<span class="feat">${esc(shortLabel(f))}</span>`).join("");
    $("verify").innerHTML = verifyBadge(model);
  }

  function verifyBadge(model) {
    const C = state.cur;
    if (C.custom) {
      return `<span class="verify info">ℹ 自訂資料：瀏覽器端計算，切分方式等同 sklearn train_test_split(random_state=${C.seed})</span>`;
    }
    const keys = state.method === "compare"
      ? METHODS.filter(m => m.id !== "exhaustive" || C.problem.exhaustiveAllowed).map(m => keyOf(m.id))
      : [keyOf(state.method)];
    let maxRel = 0, bad = [];
    for (const key of keys) {
      const ref = C.bundle.reference[key];
      const js = C.problem.table(key);
      if (!ref || !js) continue;
      js.forEach((r, i) => {
        const a = new Set(r.features), b = ref[i].features;
        if (b.length !== a.size || !b.every(f => a.has(f))) bad.push(`${key} k=${r.k}`);
        for (const m of ["test_rmse", "cv_rmse", "train_rmse"]) {
          maxRel = Math.max(maxRel, Math.abs(r[m] - ref[i][m]) / Math.abs(ref[i][m]));
        }
      });
    }
    if (!C.splitMatches || !C.namesMatch) bad.push("資料切分");
    if (bad.length) return `<span class="verify warn">⚠ 與 Python 結果不同：${esc(bad.slice(0, 3).join("、"))}</span>`;
    return `<span class="verify good">✓ 與 Python scikit-learn 結果一致（最大相對誤差 ${maxRel.toExponential(1)}）</span>`;
  }

  // ------------------------------------------------------------ charts
  function baseOptions(yTitle, onPick) {
    const ink2 = css("--ink-2"), muted = css("--muted"), grid = css("--grid");
    return {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: css("--surface"), titleColor: css("--ink"), bodyColor: ink2, borderColor: css("--axis"), borderWidth: 1, padding: 10, boxPadding: 4 },
      },
      scales: {
        x: { title: { display: true, text: "特徵數 k", color: muted }, grid: { display: false }, ticks: { color: muted }, border: { color: css("--axis") } },
        y: { title: { display: true, text: yTitle, color: muted }, grid: { color: grid }, ticks: { color: muted }, border: { display: false } },
      },
      onClick: onPick,
      onHover: (e, els) => { e.native.target.style.cursor = els.length ? "pointer" : "default"; },
    };
  }

  function lineDS(label, data, color, dash, highlightIdx) {
    return {
      label, data, borderColor: color, backgroundColor: color, borderWidth: 2, borderDash: dash || [],
      tension: 0, pointRadius: data.map((_, i) => (i === highlightIdx ? 6 : 3)),
      pointHoverRadius: 7, pointBorderColor: css("--surface"), pointBorderWidth: 2,
    };
  }

  function axisTicks(opts) {
    if (opts.scales.y.title.text.startsWith("R²")) opts.scales.y.ticks.callback = v => Number(v).toFixed(3);
    return opts;
  }

  function drawChart(id, config) {
    if (state.charts[id]) state.charts[id].destroy();
    state.charts[id] = new Chart($(id), config);
  }

  function legendHTML(items) {
    return items.map(([name, color, style]) => `<span><i class="${style || ""}" style="border-color:${color}"></i>${esc(name)}</span>`).join("");
  }

  function renderCharts(model) {
    const P = state.cur.problem;
    const labels = Array.from({ length: P.p }, (_, i) => i + 1);
    const pick = (e, els) => {
      if (!els.length) return;
      if (state.method === "compare") state.method = METHODS.filter(m => tableFor(m.id))[els[0].datasetIndex]?.id || model.method;
      state.k = els[0].index + 1;
      render();
    };
    const yU = `RMSE（${state.cur.unit.prefix || ""}${state.cur.unit.suffix || ""}${state.cur.unit.prefix || state.cur.unit.suffix ? "，" : ""}越低越好）`;
    const tipU = ctx => `${ctx.dataset.label}: ${fmtU(ctx.parsed.y)}`;
    const tipR = ctx => `${ctx.dataset.label}: ${fmtN(ctx.parsed.y, 4)}`;
    if (state.method === "compare") {
      const ms = METHODS.filter(m => tableFor(m.id));
      const metric = metricOf();
      const critName = state.criterion === "cv" ? "5-fold CV RMSE" : "Test RMSE";
      $("rmse-title").textContent = `${critName} vs k（${ms.length} 種方法）`;
      $("rmse-sub").textContent = "每條線是一種特徵選擇方法；點任一點可切換到該方法與 k。";
      $("rmse-legend").innerHTML = legendHTML(ms.map(m => [m.name, css(m.color)]));
      drawChart("chart-rmse", {
        type: "line",
        data: { labels, datasets: ms.map(m => lineDS(m.name, tableFor(m.id).map(r => r[metric]), css(m.color), null, m.id === model.method ? model.row.k - 1 : -1)) },
        options: axisTicks(withTip(baseOptions(yU, pick), tipU)),
      });
      $("r2-title").textContent = `Test R² vs k（${ms.length} 種方法）`;
      $("r2-sub").textContent = "越高越好；觀察各方法在哪個 k 達到高峰。";
      $("r2-legend").innerHTML = legendHTML(ms.map(m => [m.name, css(m.color)]));
      drawChart("chart-r2", {
        type: "line",
        data: { labels, datasets: ms.map(m => lineDS(m.name, tableFor(m.id).map(r => r.test_r2), css(m.color), null, m.id === model.method ? model.row.k - 1 : -1)) },
        options: axisTicks(withTip(baseOptions("R²（越高越好）", pick), tipR)),
      });
      return;
    }
    const rows = tableFor(state.method);
    const m = methodById(state.method), col = css(m.color);
    const hi = state.k - 1;
    const ss = sweetSpot(rows);
    $("rmse-title").textContent = `RMSE vs 特徵數 k · ${m.name}`;
    $("rmse-sub").textContent = `Training / Test curve（L6 Step 4）：訓練誤差隨 k 下降；測試 RMSE 在 k=${sweetSpot(rows, "test").k} 最低，CV RMSE 在 k=${sweetSpot(rows, "cv").k} 最低。`;
    $("rmse-legend").innerHTML = legendHTML([["Test RMSE", col], ["5-fold CV RMSE", col, "dash"], ["Train RMSE", col, "dot"]]);
    drawChart("chart-rmse", {
      type: "line",
      data: {
        labels, datasets: [
          lineDS("Test RMSE", rows.map(r => r.test_rmse), col, null, hi),
          lineDS("5-fold CV RMSE", rows.map(r => r.cv_rmse), col, [6, 4], hi),
          lineDS("Train RMSE", rows.map(r => r.train_rmse), col, [2, 3], hi),
        ],
      },
      options: axisTicks(withTip(baseOptions(yU, pick), tipU)),
    });
    $("r2-title").textContent = `R² vs 特徵數 k · ${m.name}`;
    const peak = rows.reduce((a, b) => (b.test_r2 > a.test_r2 ? b : a));
    const trainR2 = state.method === "exhaustive" ? "訓練集 R² 通常隨 k 增加" : "訓練集 R² 只會隨 k 增加（每多一個特徵都不會變差）";
    $("r2-sub").textContent = `Test R² 在 k=${peak.k} 達高峰（${fmtN(peak.test_r2, 4)}）；${trainR2}，不能用來選 k。目前最佳 k（依準則）= ${ss.k}。`;
    $("r2-legend").innerHTML = legendHTML([["Test R²", col], ["Train R²", col, "dot"]]);
    drawChart("chart-r2", {
      type: "line",
      data: { labels, datasets: [lineDS("Test R²", rows.map(r => r.test_r2), col, null, hi), lineDS("Train R²", rows.map(r => r.train_r2), col, [2, 3], hi)] },
      options: axisTicks(withTip(baseOptions("R²（越高越好）", pick), tipR)),
    });
  }

  function withTip(opts, fn) {
    opts.plugins.tooltip.callbacks = { title: items => `k = ${items[0].label}`, label: fn };
    return opts;
  }

  function renderScatter(model) {
    const preds = state.cur.problem.testPredictions(model.row.cols);
    let lo = Infinity, hi = -Infinity;
    for (const p of preds) { lo = Math.min(lo, p.actual, p.pred); hi = Math.max(hi, p.actual, p.pred); }
    const c1 = css("--s1"), muted = css("--muted");
    $("scatter-legend").innerHTML = legendHTML([["測試樣本", c1], ["完美預測 y = x", muted, "dash"]]);
    const opts = baseOptions(`預測值（${state.cur.target}）`);
    opts.interaction = { mode: "nearest", intersect: true };
    opts.scales.x.title.text = `實際值（${state.cur.target}）`;
    opts.scales.x.type = "linear";
    opts.scales.x.grid = { color: css("--grid") };
    opts.plugins.tooltip.callbacks = {
      title: items => (items[0].datasetIndex === 0 ? `樣本 #${preds[items[0].dataIndex].id + 1}` : "y = x"),
      label: ctx => ctx.datasetIndex === 0 ? [`實際 ${fmtU(ctx.parsed.x)}`, `預測 ${fmtU(ctx.parsed.y)}`, `殘差 ${fmtU(ctx.parsed.x - ctx.parsed.y)}`] : "",
    };
    opts.onClick = null;
    drawChart("chart-scatter", {
      type: "scatter",
      data: {
        datasets: [
          { label: "測試樣本", data: preds.map(p => ({ x: p.actual, y: p.pred })), backgroundColor: c1, borderColor: css("--surface"), borderWidth: 1.5, pointRadius: preds.length > 200 ? 3 : 5, pointHoverRadius: 7 },
          { type: "line", label: "y = x", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: muted, borderDash: [6, 4], borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 0 },
        ],
      },
      options: opts,
    });
  }

  // ------------------------------------------------------------ tables & cards
  function renderEquation(inf) {
    const c = inf.coefficients;
    let s = `<b>${esc(state.cur.target)}</b> = ${fmtCoef(c[0].coef)}`;
    for (const t of c.slice(1)) {
      s += ` <span class="term">${t.coef < 0 ? "−" : "+"} ${fmtCoef(Math.abs(t.coef))} × (${esc(shortLabel(t.name))})</span>`;
    }
    $("equation").innerHTML = s;
  }

  function renderCoefs(inf) {
    const head = `<tr><th>Predictor</th><th class="r">Estimate (β)</th><th class="r">Std. Error</th><th class="r">t</th><th>p-value</th><th>95% CI</th><th>VIF</th></tr>`;
    const body = inf.coefficients.map(c => {
      const sig = c.p < 0.05 ? `<span class="tag good">${fmtP(c.p)} ✓</span>` : `<span class="tag warn">${fmtP(c.p)}（n.s.）</span>`;
      let vif = "–";
      if (c.vif != null) {
        const cls = c.vif < 5 ? "good" : c.vif < 10 ? "warn" : "crit";
        vif = `<span class="tag ${cls}">${fmtN(c.vif, 2)}${cls === "good" ? "（OK）" : cls === "warn" ? "（中度）" : "（嚴重）"}</span>`;
      }
      return `<tr><td>${c.name === "const" ? "Intercept（截距）" : esc(label(c.name))}</td><td class="r num mono">${fmtCoef(c.coef)}</td>` +
        `<td class="r num mono">${fmtCoef(c.se)}</td><td class="r num">${fmtN(c.t, 3)}</td><td>${sig}</td>` +
        `<td class="num mono" style="white-space:nowrap">[${fmtCoef(c.ci_low)}, ${fmtCoef(c.ci_high)}]</td><td>${vif}</td></tr>`;
    }).join("");
    $("coef-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
    let foot = `訓練集 n = ${inf.n_train}；殘差標準誤 σ̂ = ${fmtU(inf.sigma)}；p < 0.05 視為顯著。`;
    const B = state.cur.bundle;
    const sameSet = B && B.inference.features.length === inf.features.length &&
      B.inference.features.every(f => inf.features.includes(f));
    if (sameSet) {
      const byName = Object.fromEntries(B.inference.coefficients.map(c => [c.name, c]));
      const maxRel = Math.max(...inf.coefficients.map(c => {
        const r = byName[c.name];
        return Math.max(Math.abs(c.coef - r.coef) / Math.abs(r.coef), Math.abs(c.se - r.se) / r.se, r.p > 1e-12 ? Math.abs(c.p - r.p) / r.p : 0);
      }));
      foot += ` ✓ 與 Python statsmodels OLS 對照一致（最大相對誤差 ${maxRel.toExponential(1)}）。`;
    }
    $("coef-foot").textContent = foot;
  }

  function renderPredictor(inf) {
    const C = state.cur, prep = C.prep, P = C.problem;
    const cols = inf.cols;
    const groups = [];
    cols.forEach(c => {
      const o = prep.origin[c];
      if (!groups.find(g => g.col === o.col)) groups.push({ col: o.col, kind: o.kind, idx: c });
    });
    const html = groups.map(g => {
      if (g.kind === "numeric") {
        const r = P.range(g.idx);
        const integer = Array.from(prep.X[g.idx].slice(0, 200)).every(v => Number.isInteger(v));
        const step = integer ? 1 : niceStep((r.max - r.min) / 200);
        if (state.pred[g.col] == null) state.pred[g.col] = roundTo(r.mean, step);
        const v = state.pred[g.col];
        return `<div class="pred-field"><label>${esc(label(g.col))}<span id="pv-${g.idx}">${fmtVal(v)}</span></label>` +
          `<div class="pair"><input type="range" data-col="${esc(g.col)}" data-idx="${g.idx}" min="${r.min}" max="${r.max}" step="${step}" value="${v}" aria-label="${esc(g.col)}">` +
          `<input type="number" data-col="${esc(g.col)}" data-idx="${g.idx}" step="${step}" value="${v}" aria-label="${esc(g.col)} 數值"></div></div>`;
      }
      const nom = prep.nominal.find(n => n.col === g.col);
      if (state.pred[g.col] == null) state.pred[g.col] = nom.levels[0];
      return `<div class="pred-field"><label>${esc(g.col)}</label><select data-col="${esc(g.col)}">` +
        nom.levels.map((lv, i) => `<option value="${esc(lv)}" ${lv === state.pred[g.col] ? "selected" : ""}>${esc(lv)}${i === 0 ? "（基準 Reference）" : ""}</option>`).join("") +
        `</select></div>`;
    }).join("");
    const unused = [...new Set(prep.origin.map(o => o.col))].filter(c => !groups.find(g => g.col === c));
    $("predictor").innerHTML = html + (unused.length ? `<p class="hint">未納入模型（不影響預測）：${unused.map(esc).join("、")}</p>` : "");
    $("predictor").oninput = e => {
      const el = e.target;
      if (!el.dataset.col) return;
      if (el.tagName !== "SELECT" && (el.value === "" || !Number.isFinite(+el.value))) return;
      state.pred[el.dataset.col] = el.tagName === "SELECT" ? el.value : +el.value;
      if (el.dataset.idx) {
        $("predictor").querySelectorAll(`[data-idx="${el.dataset.idx}"]`).forEach(x => { if (x !== el) x.value = el.value; });
        $(`pv-${el.dataset.idx}`).textContent = fmtVal(+el.value);
      }
      updatePrediction();
    };
    $("pred-label").textContent = `預測 ${C.targetLabel}`;
    const pk = $("download-pkl");
    pk.classList.toggle("hidden", C.custom);
    if (!C.custom) pk.href = `models/${C.id}_ols.pkl.gz`;
    updatePrediction();
  }

  function updatePrediction() {
    const inf = state.inf, prep = state.cur.prep;
    const xs = inf.cols.map(c => {
      const o = prep.origin[c];
      return o.kind === "numeric" ? +state.pred[o.col] : (state.pred[o.col] === o.level ? 1 : 0);
    });
    const r = inf.predict(xs);
    $("pred-value").textContent = fmtU(r.yhat);
    $("pred-pi").textContent = `95% 預測區間 [${fmtU(r.lo)}, ${fmtU(r.hi)}] · ${state.cur.target} = β₀ + Σ βᵢxᵢ`;
  }

  const niceStep = s => { if (s <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(s))); return [1, 2, 5, 10].map(m => m * p).find(v => v >= s); };
  const roundTo = (v, step) => +(Math.round(v / step) * step).toFixed(10);
  const fmtVal = v => Math.abs(v) >= 1000 ? v.toLocaleString("en-US", { maximumFractionDigits: 2 }) : String(+v.toFixed(4));

  function renderAssumptions(inf) {
    const items = [
      ["線性 Linearity", "Ramsey RESET 檢定：加入 ŷ²、ŷ³ 後沒有顯著改善（p > 0.05）才視為線性", `RESET F ${fmtN(inf.reset_f, 2)} · p ${fmtP(inf.reset_p)}`, inf.reset_p > 0.05 ? "good" : "warn"],
      ["殘差常態 Normality", "Jarque–Bera 檢定（p > 0.05 視為常態）", `JB ${fmtN(inf.jarque_bera, 2)} · p ${fmtP(inf.jarque_bera_p)}`, inf.jarque_bera_p > 0.05 ? "good" : "warn"],
      ["同質變異 Homoscedasticity", "Breusch–Pagan（Koenker）檢定（p > 0.05 視為同質）", `LM ${fmtN(inf.breusch_pagan, 2)} · p ${fmtP(inf.breusch_pagan_p)}`, inf.breusch_pagan_p > 0.05 ? "good" : "warn"],
      ["誤差獨立 Independence", "Durbin–Watson 依資料原本的列順序計算（隨機切分後的順序沒有意義），1.5–2.5 視為無自我相關", `DW ${fmtN(inf.durbin_watson, 3)}`, inf.durbin_watson > 1.5 && inf.durbin_watson < 2.5 ? "good" : "warn"],
      ["無共線性 No Multicollinearity", "最大 VIF < 5 良好、5–10 中度、> 10 嚴重", `max VIF ${fmtN(inf.max_vif, 2)}`, inf.max_vif < 5 ? "good" : inf.max_vif < 10 ? "warn" : "crit"],
    ];
    const icon = { good: "✓", warn: "!", crit: "✕" };
    const word = { good: "符合", warn: "需留意", crit: "違反" };
    $("assumptions").innerHTML = items.map(([t, d, v, s]) =>
      `<div class="item"><div class="ic ${s}" aria-hidden="true">${icon[s]}</div><div>` +
      `<div class="t">${t} <span class="tag ${s}">${word[s]}</span></div><div class="v">${v}</div><div class="d">${d}</div></div></div>`).join("");
  }

  function renderResiduals() {
    const C = state.cur, inf = state.inf;
    if (!inf) return;
    const preds = C.problem.testPredictions(inf.cols).sort((a, b) => Math.abs(b.resid) - Math.abs(a.resid));
    const shown = state.residAll ? preds.slice(0, 500) : preds.slice(0, 10);
    const sig = inf.sigma;
    $("resid-sub").textContent = `測試集 ${preds.length} 筆，依 |殘差| 排序。|殘差| > 2σ̂ 標為離群值、> 1σ̂ 標為殘差偏高（σ̂ = ${fmtU(sig)}，訓練集殘差標準誤）。`;
    $("resid-toggle").textContent = state.residAll ? "只顯示前 10 筆" : preds.length > 500 ? "顯示前 500 筆" : `顯示全部 ${preds.length} 筆`;
    $("resid-toggle").classList.toggle("hidden", preds.length <= 10);
    const head = `<tr><th>Rank</th><th>Sample</th><th class="r">Actual</th><th class="r">Predicted</th><th class="r">Residual</th><th class="r">|Error|</th><th class="r">% Error</th><th>Status</th></tr>`;
    const body = shown.map((p, i) => {
      const a = Math.abs(p.resid);
      const st = a > 2 * sig ? ["crit", "✕ 離群值"] : a > sig ? ["warn", "! 殘差偏高"] : ["good", "✓ 正常範圍"];
      const pct = p.actual !== 0 ? (a / Math.abs(p.actual) * 100).toFixed(2) + "%" : "–";
      return `<tr><td>#${i + 1}</td><td>樣本 #${p.id + 1}</td><td class="r num">${fmtU(p.actual)}</td><td class="r num">${fmtU(p.pred)}</td>` +
        `<td class="r num ${p.resid >= 0 ? "pos" : "neg"}">${p.resid >= 0 ? "+" : ""}${fmtU(p.resid)}</td><td class="r num">${fmtU(a)}</td>` +
        `<td class="r num">${pct}</td><td><span class="tag ${st[0]}">${st[1]}</span></td></tr>`;
    }).join("");
    $("resid-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  }

  function renderSelTable(model) {
    const C = state.cur;
    const featCell = fs => fs.map(f => `<span class="feat">${esc(shortLabel(f))}</span>`).join("");
    if (state.method === "compare") {
      $("sel-title").textContent = `${N_METHODS} 種方法比較 Comparison`;
      $("sel-sub").textContent = `每種方法在目前準則（${state.criterion === "cv" ? "5-fold CV" : "測試集"} RMSE）下的最佳 k。點選列可切換到該方法。`;
      const head = `<tr><th>Method</th><th class="r">最佳 k</th><th>Selected Features</th><th class="r">Train RMSE</th><th class="r">CV RMSE</th><th class="r">Test RMSE</th><th class="r">Test R²</th></tr>`;
      const body = METHODS.map(m => {
        const rows = tableFor(m.id);
        if (!rows) return `<tr><td>${m.name}</td><td colspan="6" class="muted">p > ${MLR.MAX_EXHAUSTIVE_P}，未執行</td></tr>`;
        const s = sweetSpot(rows);
        const hl = m.id === model.method ? "hl" : "";
        return `<tr class="${hl}" data-m="${m.id}" data-k="${s.k}" style="cursor:pointer"><td><span class="dot" style="display:inline-block;width:9px;height:9px;border-radius:50%;background:var(${m.color});margin-right:6px"></span>${m.name}${hl ? " ★" : ""}</td>` +
          `<td class="r">${s.k}</td><td>${featCell(s.features)}</td><td class="r num">${fmtU(s.train_rmse)}</td><td class="r num">${fmtU(s.cv_rmse)}</td><td class="r num">${fmtU(s.test_rmse)}</td><td class="r num">${fmtN(s.test_r2, 4)}</td></tr>`;
      }).join("");
      $("sel-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
      return;
    }
    const rows = tableFor(state.method);
    const ss = sweetSpot(rows);
    const m = methodById(state.method);
    $("sel-title").textContent = `特徵選擇結果表 Results Table · ${m.name}`;
    let sub = METHOD_DESC[m.id];
    const stop = state.method === "backward" ? C.problem.backwardStop() : null;
    if (state.method === "backward") {
      sub += stop ? ` 本資料在 k = ${stop} 時所有係數 p < 0.05。` : " 本資料即使只剩 1 個特徵，係數也不顯著。";
    }
    if (C.id === "startups" && state.method === "forward" && state.criterion === "test") {
      sub += " 這張表和講義 L6 的 Results Table 完全相同（RMSE 8274.87 / 8198.80 / 8309.06 / 8409.92 / 9137.99）。";
    }
    $("sel-sub").textContent = sub + " 點選列可切換 k。";
    const head = `<tr><th class="r">k</th><th>Selected Features</th><th class="r">Train RMSE</th><th class="r">5-fold CV RMSE</th><th class="r">Test RMSE</th><th class="r">Test R²</th></tr>`;
    const body = rows.map(r => `<tr class="${r.k === state.k ? "hl" : ""}" data-k="${r.k}" style="cursor:pointer"><td class="r">${r.k}${r.k === ss.k ? " ★" : ""}${r.k === stop ? ` <span class="tag info">p 值停止點</span>` : ""}</td>` +
      `<td>${featCell(r.features)}</td><td class="r num">${fmtU(r.train_rmse)}</td><td class="r num">${fmtU(r.cv_rmse)}</td>` +
      `<td class="r num">${fmtU(r.test_rmse)}</td><td class="r num">${fmtN(r.test_r2, 6)}</td></tr>`).join("");
    $("sel-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
  }

  // ------------------------------------------------------------ CRISP-DM
  const PHASES = ["1 商業理解", "2 資料理解", "3 資料準備", "4 建立模型", "5 模型評估", "6 部署"];

  function renderCrisp() {
    $("crisp-tabs").innerHTML = PHASES.map((p, i) =>
      `<button class="${i === state.tab ? "on" : ""}" data-t="${i}" role="tab" aria-selected="${i === state.tab}">${p}</button>`).join("");
    $("crisp-body").innerHTML = [phaseBusiness, phaseData, phasePrep, phaseModel, phaseEval, phaseDeploy][state.tab]();
    if (state.tab === 1) drawDataCharts();
  }

  // Business success criteria, checked on the leakage-free (CV) recommended model
  function successChecks() {
    const C = state.cur, best = globalBest("cv");
    const inf = C.problem.inference(best.row.cols);
    const values = {
      test_r2: best.row.test_r2, test_rmse: best.row.test_rmse, test_mae: best.row.test_mae,
      cv_rmse: best.row.cv_rmse, k: best.row.k, max_vif: inf.max_vif,
      max_p: Math.max(...inf.coefficients.slice(1).map(c => c.p)),
    };
    const crit = C.bundle ? C.bundle.business.success : DEFAULT_SUCCESS;
    return { best, inf, checks: crit.map(c => Object.assign({}, c, { actual: values[c.metric], passed: OPS[c.op](values[c.metric], c.value) })) };
  }

  function fmtMetric(metric, v) {
    if (["test_rmse", "cv_rmse", "test_mae"].includes(metric)) return fmtU(v);
    if (metric === "k") return String(v);
    if (metric === "max_p") return fmtP(v);
    return fmtN(v, metric === "max_vif" ? 2 : 4);
  }

  function successHTML() {
    const C = state.cur, P = C.problem;
    const { best, inf, checks } = successChecks();
    const rows = checks.map(c => `<tr><td>${esc(c.text)}</td><td class="r num">${fmtMetric(c.metric, c.actual)}</td>` +
      `<td>${c.passed ? `<span class="tag good">✓ 達成</span>` : `<span class="tag crit">✕ 未達成</span>`}</td></tr>`).join("");
    const notes = [];
    if (checks.some(c => c.metric === "max_p" && !c.passed)) {
      const ns = inf.coefficients.slice(1).filter(c => c.p >= 0.05).map(c => `${esc(shortLabel(c.name))}（p = ${c.p.toFixed(3)}）`);
      const stop = P.backwardStop();
      let t = `不顯著的係數：${ns.join("、")}。`;
      if (stop) {
        const r = tableFor("backward")[stop - 1];
        t += ` 若要求所有係數都顯著，Backward（p 值）會停在 k = ${stop}（${r.features.map(f => esc(shortLabel(f))).join("、")}），` +
          `Test RMSE ${fmtU(r.test_rmse)}，推薦模型是 ${fmtU(best.row.test_rmse)}：這是「精簡、可解釋」和「預測誤差」之間的取捨。`;
      }
      notes.push(t);
    }
    if (checks.some(c => ["test_r2", "test_rmse", "test_mae", "cv_rmse"].includes(c.metric) && !c.passed)) {
      notes.push(inf.reset_p < 0.05
        ? `預測誤差未達標，而且 RESET 檢定（p ${fmtP(inf.reset_p)}）顯示存在非線性關係：線性模型已接近極限，可考慮加入轉換或多項式項（例如 log、平方項）。`
        : "預測誤差未達標：可能需要更多特徵或更多資料。");
    }
    return `<p>以只用訓練集 CV 選出的推薦模型（${esc(methodById(best.method).name)} k = ${best.row.k}）逐項檢核${C.bundle ? "" : "（自訂資料使用預設標準）"}：</p>` +
      `<div class="tbl-wrap"><table><thead><tr><th>成功標準</th><th class="r">實際值</th><th>結果</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      notes.map(n => `<div class="callout">${n}</div>`).join("");
  }

  function phaseBusiness() {
    const C = state.cur, B = C.bundle;
    const gbCV = globalBest("cv");
    const obj = B ? B.business.objective : `預測連續數值「${C.target}」，並找出真正影響它的特徵。`;
    const succ = (B ? B.business.success : DEFAULT_SUCCESS).map(c => c.text);
    const met = successChecks().checks;
    return `<h4>Business Understanding</h4><p>${esc(obj)}</p><h4>成功標準 Success Criteria</h4><ul>${succ.map(s => `<li>${esc(s)}</li>`).join("")}</ul>` +
      `<p class="hint">達成 ${met.filter(c => c.passed).length} / ${met.length} 項，逐項檢核見「5 模型評估」。</p>` +
      `<div class="callout">本頁的結論：只用訓練集 CV 挑選時，最佳模型是 <b>${esc(methodById(gbCV.method).name)}</b> k = ${gbCV.row.k}（${gbCV.row.features.map(f => esc(shortLabel(f))).join("、")}），` +
      `測試集 RMSE ${fmtU(gbCV.row.test_rmse)}、R² ${fmtN(gbCV.row.test_r2, 4)}。</div>`;
  }

  function phaseData() {
    const C = state.cur, prep = C.prep, table = C.table;
    const tj = table.header.indexOf(C.target);
    const kept = prep.rowIds.map(i => table.rows[i]);
    const ty = kept.map(r => Number(r[tj]));
    const rows = table.header.map((h, j) => {
      const t = prep.types[j];
      const missing = table.rows.length - t.nonNA;
      let mean = "–", corr = "–";
      if (t.numeric) {
        // pairs of finite values only: a dropped column may still hold NA
        const pairs = kept.map((r, i) => [Number(String(r[j]).trim() || NaN), ty[i]]).filter(([x]) => Number.isFinite(x));
        if (pairs.length) {
          const mx = pairs.reduce((a, [x]) => a + x, 0) / pairs.length;
          mean = fmtVal(mx);
          if (j !== tj) {
            const my = pairs.reduce((a, [, v]) => a + v, 0) / pairs.length;
            let sxy = 0, sxx = 0, syy = 0;
            for (const [x, v] of pairs) { sxy += (x - mx) * (v - my); sxx += (x - mx) ** 2; syy += (v - my) ** 2; }
            const r = sxy / Math.sqrt(sxx * syy);
            corr = Number.isFinite(r) ? (r >= 0 ? "+" : "−") + Math.abs(r).toFixed(3) : "–";
          }
        }
      }
      const role = h === C.target ? "<b>目標 Target</b>" : prep.dropped.includes(h) ? "排除" : prep.idLike.includes(h) ? "ID / 文字（排除）"
        : t.numeric ? "數值 Numeric" : "名目 Nominal";
      return `<tr><td>${esc(h)}</td><td>${role}</td><td class="r">${missing}</td><td class="r">${t.unique > 20 ? "> 20" : t.unique}</td><td class="r num">${mean}</td><td class="r num">${corr}</td></tr>`;
    }).join("");
    const note = C.bundle ? `<p>${esc(C.bundle.notes.data)}</p>` : "";
    return `<h4>Data Understanding</h4><p>資料來源：${C.source.url ? `<a href="${esc(C.source.url)}" target="_blank" rel="noopener">${esc(C.source.name)}</a>` : esc(C.source.name)}` +
      `${C.csvUrl ? `（<a href="${esc(C.csvUrl)}" target="_blank" rel="noopener">raw CSV</a>）` : ""}，共 ${table.rows.length} 列 × ${table.header.length} 欄。</p>${note}` +
      `<div class="tbl-wrap"><table><thead><tr><th>欄位</th><th>角色 / 型態</th><th class="r">缺失</th><th class="r">唯一值</th><th class="r">平均</th><th class="r">與目標相關 r</th></tr></thead><tbody>${rows}</tbody></table></div>` +
      `<h4>各特徵與目標的相關係數（編碼後特徵，全部資料）</h4><div class="legend" id="corr-legend"></div>` +
      `<div class="chart-box" style="height:${Math.max(180, 26 * prep.names.length + 60)}px"><canvas id="chart-corr" aria-label="各特徵與目標的相關係數長條圖"></canvas></div>` +
      `<h4 id="eda-title">最相關的特徵 vs 目標</h4><div class="chart-box"><canvas id="chart-eda" aria-label="最相關特徵與目標的散布圖"></canvas></div>`;
  }

  function pearson(a, b) {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { const da = a[i] - ma, db = b[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
    return sab / Math.sqrt(saa * sbb);
  }

  function drawDataCharts() {
    const C = state.cur, prep = C.prep;
    const corr = prep.names.map((nm, j) => ({ nm, j, r: pearson(prep.X[j], prep.y) }))
      .filter(d => Number.isFinite(d.r)).sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    if (!corr.length) return;
    const pos = css("--s1"), neg = css("--s8");
    $("corr-legend").innerHTML = `<span><i style="border-color:${pos};border-top-width:8px"></i>正相關</span><span><i style="border-color:${neg};border-top-width:8px"></i>負相關</span>`;
    const opts = baseOptions("");
    opts.indexAxis = "y";
    opts.interaction = { mode: "nearest", axis: "y", intersect: false };
    opts.scales.x = { min: -1, max: 1, title: { display: true, text: `與 ${C.target} 的相關係數 r`, color: css("--muted") }, grid: { color: css("--grid") }, ticks: { color: css("--muted") }, border: { display: false } };
    opts.scales.y = { grid: { display: false }, ticks: { color: css("--ink-2"), autoSkip: false }, border: { color: css("--axis") } };
    opts.plugins.tooltip.callbacks = { label: ctx => `r = ${ctx.parsed.x >= 0 ? "+" : "−"}${Math.abs(ctx.parsed.x).toFixed(3)}` };
    opts.onClick = null;
    drawChart("chart-corr", {
      type: "bar",
      data: { labels: corr.map(d => shortLabel(d.nm)), datasets: [{ data: corr.map(d => d.r), backgroundColor: corr.map(d => (d.r >= 0 ? pos : neg)), borderRadius: 4, maxBarThickness: 16 }] },
      options: opts,
    });
    const top = corr[0];
    $("eda-title").textContent = `最相關的特徵：${shortLabel(top.nm)} vs ${C.target}（r = ${top.r.toFixed(3)}）`;
    const sc = baseOptions(C.target);
    sc.interaction = { mode: "nearest", intersect: true };
    sc.scales.x = { type: "linear", title: { display: true, text: shortLabel(top.nm), color: css("--muted") }, grid: { color: css("--grid") }, ticks: { color: css("--muted") }, border: { color: css("--axis") } };
    sc.plugins.tooltip.callbacks = { title: () => "", label: ctx => [`${shortLabel(top.nm)} = ${fmtVal(ctx.parsed.x)}`, `${C.target} = ${fmtVal(ctx.parsed.y)}`] };
    sc.onClick = null;
    drawChart("chart-eda", {
      type: "scatter",
      data: { datasets: [{ data: Array.from(prep.X[top.j], (x, i) => ({ x, y: prep.y[i] })), backgroundColor: pos, borderColor: css("--surface"), borderWidth: 1, pointRadius: prep.n > 2000 ? 2 : 4, pointHoverRadius: 6 }] },
      options: sc,
    });
  }

  function phasePrep() {
    const C = state.cur, prep = C.prep, P = C.problem;
    const nom = prep.nominal.length ? prep.nominal.map(n => `<code>${esc(n.col)}</code> → ${n.levels.length - 1} 個 dummy（基準 = ${esc(n.levels[0])}）`).join("；") : "無";
    let split = `<code>train_test_split(test_size=${C.testSize}, random_state=${C.seed})</code> → ${P.ntr} 筆訓練 / ${P.nte} 筆測試。`;
    if (C.splitMatches === true) split += " ✓ 瀏覽器用 MT19937 重現 numpy 的亂數排列，切分結果和 Python 的索引完全相同。";
    if (C.custom) split += " 瀏覽器用 MT19937 重現 numpy 的 RandomState，切分等同 sklearn。";
    return `<h4>Data Preparation</h4><ul>` +
      `<li>缺失值：含缺失的列直接移除（${prep.nRaw} → ${prep.n} 列）。</li>` +
      `<li>名目變數 One-hot encoding（<code>drop_first=True</code>，避免 dummy variable trap：n 個類別只放 n−1 個）：${nom}。</li>` +
      (prep.idLike.length ? `<li>超過 20 個類別的文字欄位視為 ID，自動排除：${prep.idLike.map(esc).join("、")}。</li>` : "") +
      (prep.dropped.length ? `<li>手動排除：${prep.dropped.map(esc).join("、")}。</li>` : "") +
      (C.collinear.length ? `<li>常數或完全共線欄位已排除：${C.collinear.map(d => `${esc(d.name)}（${d.reason}）`).join("、")}。</li>` : "") +
      `<li>最終特徵 p = ${P.p}：${prep.names.map(n => `<code>${esc(n)}</code>`).join(" ")}</li>` +
      `<li>${split}</li>` +
      `<li>Lasso / ElasticNet / RFE 使用 StandardScaler（只用訓練集的平均與標準差）；交叉驗證為 <code>KFold(5, shuffle=False)</code>。</li></ul>`;
  }

  function phaseModel() {
    const C = state.cur, B = C.bundle;
    let extra = "";
    if (B) {
      const o = B.optuna_lasso;
      if (o) extra += `<h4>Optuna 調參（Python，L7）</h4><p>以 TPE 搜尋 Lasso 的 α（${o.n_trials} 次、5-fold CV）：α = ${o.alpha.toPrecision(4)}，CV RMSE ${fmtU(o.cv_rmse)}，測試集 RMSE ${fmtU(o.test_rmse)}、R² ${fmtN(o.test_r2, 4)}，保留 ${o.n_nonzero} 個特徵（${o.kept.map(f => esc(shortLabel(f))).join("、")}）。</p>`;
      extra += `<h4>Mutual Information 排名（Python，L7 MIF）</h4><p>${B.mutual_info.slice(0, 8).map(m => `${esc(shortLabel(m.feature))} ${m.mi.toFixed(3)}`).join(" › ")}</p>`;
    }
    return `<h4>Modeling</h4><p>基準模型是 OLS 多元線性迴歸（<code>LinearRegression</code>）。每種特徵選擇方法都產生 k = 1…p 的特徵子集，再用 OLS 重新估計，比較偏差與變異的取捨。</p>` +
      `<ul>${METHODS.map(m => `<li><b>${m.name}</b>：${METHOD_DESC[m.id]}</li>`).join("")}</ul>` +
      `<div class="callout">選擇準則：<b>5-fold CV RMSE</b> 只用訓練集，是正確的做法；<b>測試集 RMSE</b> 是講義 L6 的做法，可以重現講義數字，但測試集被拿來挑模型後就不再是獨立評估。</div>` + extra;
  }

  function phaseEval() {
    const C = state.cur, P = C.problem;
    const lect = globalBest("test"), cv = globalBest("cv");
    const rows = tableFor(state.method === "compare" ? cv.method : state.method);
    const first = rows[0], full = rows[P.p - 1];
    const bestTest = rows.reduce((a, b) => (b.test_rmse < a.test_rmse ? b : a));
    const inf = state.inf;
    const mName = methodById(state.method === "compare" ? cv.method : state.method);
    const trainTxt = mName.id === "exhaustive" ? "訓練誤差大致隨 k 下降" : "訓練誤差隨 k 一路下降（多加特徵不會讓訓練誤差變大）";
    const gap = first.test_rmse / bestTest.test_rmse - 1;
    const underfit = bestTest.k === 1 ? "已經是最低的測試誤差"
      : gap > 0.05 ? `比最佳 k 高 ${(gap * 100).toFixed(1)}%，特徵太少（underfit）`
      : `只比最佳 k 高 ${(gap * 100).toFixed(1)}%，單一特徵就解釋了大部分變異`;
    let curve;
    if (bestTest.k === P.p) curve = `測試誤差在用上全部 ${P.p} 個特徵時最低，這個方法下沒有明顯的過擬合；挑選特徵的好處主要是模型更精簡、更好解釋。`;
    else if (bestTest.k === 1) curve = "只用 1 個特徵時測試誤差就最低，之後加入的特徵都沒有改善測試誤差。";
    else curve = `測試誤差在 k = ${bestTest.k} 最低（${fmtU(bestTest.test_rmse)}），用上全部特徵時回升到 ${fmtU(full.test_rmse)}，多出來的特徵開始擬合雜訊（過擬合）。`;
    let py = "";
    if (C.bundle) {
      const bi = C.bundle.inference;
      py = `<h4>Python statsmodels 對照（推薦模型：${bi.features.map(f => esc(shortLabel(f))).join("、")}）</h4>` +
        `<p>R² ${fmtN(bi.r2_train, 4)}、Adj. R² ${fmtN(bi.adj_r2_train, 4)}、F = ${fmtN(bi.f_stat, 2)}（p ${bi.f_pvalue < 1e-4 ? "< 0.0001" : bi.f_pvalue.toFixed(4)}）、Ramsey RESET p ${fmtP(bi.reset_p)}、Durbin–Watson ${fmtN(bi.durbin_watson, 3)}、Jarque–Bera p ${fmtP(bi.jarque_bera_p)}、Breusch–Pagan p ${fmtP(bi.breusch_pagan_p)}。` +
        ` 完整報告：<a href="reports/${esc(C.id)}_crisp_dm.md" target="_blank" rel="noopener">reports/${esc(C.id)}_crisp_dm.md</a></p>`;
    }
    return `<h4>Evaluation</h4><ul>` +
      `<li><b>MSE</b> = 平均平方誤差；<b>RMSE</b> = √MSE，和目標同單位；<b>MAE</b> = 平均絕對誤差，對離群值較不敏感；<b>R²</b> = 模型解釋的變異比例；<b>Adjusted R²</b> 會懲罰多餘特徵。</li>` +
      `<li><b>Underfit / Overfit</b>（${esc(mName.name)}）：k = 1 時 train RMSE ${fmtU(first.train_rmse)}、test ${fmtU(first.test_rmse)}，${underfit}；` +
      `全部 ${P.p} 個特徵時 train ${fmtU(full.train_rmse)}、test ${fmtU(full.test_rmse)}。${trainTxt}；${curve}</li>` +
      `<li><b>講義做法</b>（用測試集挑）：${esc(methodById(lect.method).name)} k = ${lect.row.k}，Test RMSE ${fmtU(lect.row.test_rmse)}，但這個分數偏樂觀。</li>` +
      `<li><b>無資料洩漏做法</b>（用 CV 挑）：${esc(methodById(cv.method).name)} k = ${cv.row.k}，Test RMSE ${fmtU(cv.row.test_rmse)}、R² ${fmtN(cv.row.test_r2, 4)}，這才是誠實的泛化估計。</li>` +
      `<li>目前模型的假設檢驗：RESET p ${fmtP(inf.reset_p)}、DW ${fmtN(inf.durbin_watson, 3)}、JB p ${fmtP(inf.jarque_bera_p)}、BP p ${fmtP(inf.breusch_pagan_p)}、max VIF ${fmtN(inf.max_vif, 2)}（詳見上方五大假設卡片）。</li></ul>` +
      `<h4>對照商業成功標準</h4>` + successHTML() + py;
  }

  function phaseDeploy() {
    const C = state.cur;
    return `<h4>Deployment</h4><ul>` +
      `<li><b>網頁即時預測</b>：上方 Live Predictor 用目前選定的模型計算預測值與 95% 預測區間。</li>` +
      `<li><b>下載模型 JSON</b>：截距、係數、特徵編碼方式與評估指標，可直接在其他程式中使用。</li>` +
      (C.custom ? "" : `<li><b>Python 模型</b>：<code>models/${esc(C.id)}_ols.pkl.gz</code>（pickle + gzip，L6 Step 5），由 <code>analysis/mlr_workflow.py</code> 產生。</li>`) +
      `<li><b>GitHub Pages</b>：<code>.github/workflows/pages.yml</code> 在每次 push 後自動部署這個靜態網站。</li>` +
      `<li><b>新問題</b>：用 Claude Skill <code>mlr-crisp-dm</code>（<code>.claude/skills/mlr-crisp-dm/SKILL.md</code>）跑同一套 workflow，就能把新資料集加進下拉選單；Problem 2 Boston Housing 就是這樣產生的。</li></ul>`;
  }

  function downloadModel() {
    const C = state.cur, inf = state.inf, m = state.model;
    const out = {
      problem: C.title, target: C.target, method: methodById(m.method).name, k: m.row.k,
      criterion: state.criterion === "cv" ? "5-fold CV RMSE" : "test RMSE",
      intercept: inf.coefficients[0].coef,
      coefficients: Object.fromEntries(inf.coefficients.slice(1).map(c => [c.name, c.coef])),
      encoding: inf.cols.map(c => C.prep.origin[c]),
      metrics: { test_rmse: m.row.test_rmse, test_r2: m.row.test_r2, test_mae: m.row.test_mae, cv_rmse: m.row.cv_rmse, adj_r2_train: inf.adj_r2_train },
      sigma: inf.sigma, split: { test_size: C.testSize, random_state: C.seed },
    };
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${C.custom ? "custom" : C.id}_${m.method}_k${m.row.k}_model.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  init();
})();
