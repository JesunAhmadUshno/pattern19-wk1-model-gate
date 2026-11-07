// Pure logic shared by the page and the build-time known-answer check (model.check.mjs).
// No DOM, no fetch: every input is the text or bytes of a repo file.

// ---- Gate: a line-for-line port of src/gate.py (check_eval, check_data_card, check_approval, main) ----
const isNumber = (x) => typeof x === 'number' && Number.isFinite(x);
// Python prints 0.65 as "0.65" and 1.0 as "1.0"; JS would print "1".
const py = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));

function checkEval(evalText, floor, frozenHashText, actualHash) {
  if (evalText == null) return [false, 'MISSING (records/eval_record.json not found)', 'no eval record'];
  const rec = JSON.parse(evalText);
  const frozen = frozenHashText.trim();
  if (rec.test_sha256 !== frozen || actualHash !== frozen)
    return [false, 'FAIL (test set hash mismatch: the scored set is not the frozen set)', 'eval earned on a swapped test set'];
  if (!isNumber(floor)) return [false, 'FAIL (gate_config.json has no numeric floor_auc)', 'no floor in gate_config.json'];
  if (!isNumber(rec.auc)) return [false, 'FAIL (eval record has no numeric auc)', 'eval record has no numeric AUC'];
  if (rec.auc < floor) return [false, `FAIL (AUC ${py(rec.auc)} below floor ${py(floor)})`, 'score below the floor set before training'];
  return [true, `OK (AUC ${py(rec.auc)} >= floor ${py(floor)}, ${rec.test_rows} frozen rows, hash verified)`, ''];
}

function checkDataCard(text) {
  if (text == null || text.length === 0) return [false, 'MISSING (records/data_card.md not found)', 'no data card'];
  return [true, 'OK (records/data_card.md)', ''];
}

function checkApproval(text) {
  if (text != null) {
    for (const line of text.split('\n')) {
      const m = line.trim().match(/^APPROVED\s+(.+?)\s+(\d{4}-\d{2}-\d{2})\s*$/);
      if (m) return [true, `OK (${m[1]}, ${m[2]})`, ''];
    }
  }
  return [false, 'MISSING (no APPROVED line in records/approvals.log)', 'no approval line'];
}

// Returns { out, code } where out is exactly what gate.py prints and code its exit status.
export function runGate({ configText, evalText, dataCard, approvals, frozenHashText, actualHash }) {
  const cfg = JSON.parse(configText);
  const version = cfg.model_version;
  const checks = [
    ['eval record', checkEval(evalText, cfg.floor_auc, frozenHashText, actualHash)],
    ['data card', checkDataCard(dataCard)],
    ['approval line', checkApproval(approvals)],
  ];
  const lines = ['ModelGate promotion check', `model: ${version}`];
  const reasons = [];
  checks.forEach(([name, [ok, detail, reason]], i) => {
    lines.push(`[${i + 1}/3] ${name} ... ${detail}`);
    if (!ok) reasons.push(reason);
  });
  if (reasons.length) {
    lines.push(`PROMOTE ${version}: BLOCKED`, 'reasons: ' + reasons.join('; '));
    return { out: lines.join('\n') + '\n', code: 1, checks };
  }
  lines.push(`PROMOTE ${version}: PASS`);
  return { out: lines.join('\n') + '\n', code: 0, checks };
}

export const toHex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');

// ---- Data ----
export function parseCsv(text) {
  const rows = text.trim().split('\n');
  const header = rows[0].trim().split(',');
  return { header, rows: rows.slice(1).map((r) => r.split(',').map(Number)) };
}

export const TARGET = 'default_next_month';

// Train = credit.csv rows whose ID is not in the frozen test file (reproduces src/train.py's split).
export function split(credit, frozen) {
  const idCol = credit.header.indexOf('ID');
  const testIds = new Set(frozen.rows.map((r) => r[frozen.header.indexOf('ID')]));
  return { train: credit.rows.filter((r) => !testIds.has(r[idCol])), test: frozen.rows };
}

// ---- Model: StandardScaler + L2 logistic regression with sklearn's objective (C = 1, intercept unpenalized) ----
// Minimizes 0.5 * |w|^2 + C * sum(logloss) by Newton's method (IRLS); sklearn's lbfgs reaches the same optimum.
export function fit(header, train, features, C = 1) {
  const cols = features.map((f) => header.indexOf(f));
  const y = header.indexOf(TARGET);
  const n = train.length, p = cols.length + 1;
  const mean = cols.map((c) => train.reduce((s, r) => s + r[c], 0) / n);
  // StandardScaler uses the population standard deviation (ddof 0).
  const sd = cols.map((c, j) => Math.sqrt(train.reduce((s, r) => s + (r[c] - mean[j]) ** 2, 0) / n) || 1);
  const X = new Float64Array(n * p);
  const Y = new Float64Array(n);
  train.forEach((r, i) => {
    X[i * p] = 1;
    cols.forEach((c, j) => { X[i * p + j + 1] = (r[c] - mean[j]) / sd[j]; });
    Y[i] = r[y];
  });
  let w = new Float64Array(p);
  for (let iter = 0; iter < 50; iter++) {
    const g = new Float64Array(p), H = new Float64Array(p * p);
    for (let i = 0; i < n; i++) {
      let z = 0;
      for (let j = 0; j < p; j++) z += X[i * p + j] * w[j];
      const mu = 1 / (1 + Math.exp(-z)), s = mu * (1 - mu), e = mu - Y[i];
      for (let j = 0; j < p; j++) {
        const xj = X[i * p + j];
        g[j] += C * e * xj;
        const sx = C * s * xj;
        for (let k = 0; k <= j; k++) H[j * p + k] += sx * X[i * p + k];
      }
    }
    for (let j = 1; j < p; j++) { g[j] += w[j]; H[j * p + j] += 1; }
    for (let j = 0; j < p; j++) for (let k = 0; k < j; k++) H[k * p + j] = H[j * p + k];
    const step = solve(H, g, p);
    let max = 0;
    for (let j = 0; j < p; j++) { w[j] -= step[j]; max = Math.max(max, Math.abs(step[j])); }
    if (max < 1e-10) break;
  }
  return { cols, mean, sd, w, features };
}

function solve(A, b, p) { // Gaussian elimination with partial pivoting
  const M = Array.from({ length: p }, (_, i) => [...A.slice(i * p, i * p + p), b[i]]);
  for (let c = 0; c < p; c++) {
    let piv = c;
    for (let r = c + 1; r < p; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < p; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= p; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(p).fill(0);
  for (let r = p - 1; r >= 0; r--) {
    let s = M[r][p];
    for (let k = r + 1; k < p; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

export function predict(m, rows) {
  return rows.map((r) => {
    let z = m.w[0];
    m.cols.forEach((c, j) => { z += m.w[j + 1] * (r[c] - m.mean[j]) / m.sd[j]; });
    return 1 / (1 + Math.exp(-z));
  });
}

// ---- Metrics ----
// AUC by the rank-sum (Mann-Whitney) formula with tied scores averaged, as sklearn's roc_auc_score.
// Optional weights w (bootstrap counts) let one pre-sorted order serve every resample.
export function aucSorted(order, score, y, w) {
  let pos = 0, neg = 0, sum = 0, rank = 0;
  for (let i = 0; i < order.length;) {
    let j = i, wp = 0, wn = 0;
    while (j < order.length && score[order[j]] === score[order[i]]) {
      const k = order[j], c = w ? w[k] : 1;
      if (y[k]) wp += c; else wn += c;
      j++;
    }
    const cnt = wp + wn;
    sum += wp * (rank + (cnt + 1) / 2);
    rank += cnt; pos += wp; neg += wn; i = j;
  }
  return (sum - (pos * (pos + 1)) / 2) / (pos * neg);
}

export const auc = (score, y) => aucSorted(score.map((_, i) => i).sort((a, b) => score[a] - score[b]), score, y);

export function roc(score, y) {
  const order = score.map((_, i) => i).sort((a, b) => score[b] - score[a]);
  const P = y.reduce((s, v) => s + v, 0), N = y.length - P;
  const pts = [[0, 0]];
  let tp = 0, fp = 0;
  order.forEach((k, i) => {
    if (y[k]) tp++; else fp++;
    if (i === order.length - 1 || score[order[i + 1]] !== score[k]) pts.push([fp / N, tp / P]);
  });
  return pts;
}

export function confusion(score, y, t) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  score.forEach((s, i) => { if (s >= t) { if (y[i]) tp++; else fp++; } else if (y[i]) fn++; else tn++; });
  const n = y.length;
  return { tp, fp, tn, fn, accuracy: (tp + tn) / n, precision: tp + fp ? tp / (tp + fp) : 0, recall: tp + fn ? tp / (tp + fn) : 0, flagRate: (tp + fp) / n };
}

export function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Percentile bootstrap of the test AUC: resample the scored rows with replacement.
export function bootstrapAuc(score, y, runs = 500, seed = 19) {
  const n = score.length, rand = mulberry32(seed);
  const order = score.map((_, i) => i).sort((a, b) => score[a] - score[b]);
  const w = new Uint32Array(n), out = [];
  for (let b = 0; b < runs; b++) {
    w.fill(0);
    for (let i = 0; i < n; i++) w[Math.floor(rand() * n)]++;
    out.push(aucSorted(order, score, y, w));
  }
  out.sort((a, b) => a - b);
  return { lo: out[Math.floor(0.025 * runs)], hi: out[Math.ceil(0.975 * runs) - 1], runs, seed };
}

// 1-based line number of the first line containing needle, so cited lines track the source.
export const lineOf = (src, needle) => src.split('\n').findIndex((l) => l.includes(needle)) + 1;
