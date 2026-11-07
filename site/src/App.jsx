import { useMemo, useState, useEffect, useId, useRef } from 'react';
import config from '../../gate_config.json';
import configText from '../../gate_config.json?raw';
import evalText from '../../records/eval_record.json?raw';
import dataCardText from '../../records/data_card.md?raw';
import approvalsText from '../../records/approvals.log?raw';
import frozenHashText from '../../records/test_frozen.sha256?raw';
import modelCardText from '../../models/model_v1/model_card.md?raw';
import run1 from '../../demo/run1_blocked.txt?raw';
import run2 from '../../demo/run2_pass.txt?raw';
import gateSrc from '../../src/gate.py?raw';
import trainSrc from '../../src/train.py?raw';
import evaluateSrc from '../../src/evaluate.py?raw';
import testSrc from '../../tests/test_gate.py?raw';
import './app.css';
import { runGate, toHex, parseCsv, split, fit, predict, auc, roc, confusion, bootstrapAuc, lineOf, TARGET } from './model.js';

const REPO = 'https://github.com/JesunAhmadUshno/pattern19-wk1-model-gate';
const blob = (p, line) => `${REPO}/blob/main/${p}${line ? `#L${line}` : ''}`;
const rec = JSON.parse(evalText);
const FROZEN_HASH = frozenHashText.trim();
const APPROVAL_HEADER = approvalsText.split('\n').filter((l) => l.startsWith('#')).join('\n') + '\n';
const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
const fmt = (n) => n.toLocaleString('en-US');
const TRAIN_ROWS = +modelCardText.match(/Train rows: (\d+)/)[1];
const N_TESTS = (testSrc.match(/^    def test_/gm) || []).length;
const PROTECTED = ['ID', 'SEX', 'MARRIAGE', 'AGE'];

const SECTIONS = [['top', 'Overview', '◎'], ['gate', 'Gate', '⛨'], ['runs', 'Recorded runs', '▶'], ['records', 'Evidence', '▤'], ['scorecard', 'Scorecard', '◢'], ['audit', 'Audit', '⚖'], ['review', 'Findings', '✓'], ['reproduce', 'Reproduce', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">ModelGate</a>
      <a className="m" href={blob('src/gate.py')}>gate.py</a>
      <a className="m" href={blob('records/eval_record.json')}>Records</a>
      <a className="m" href={blob('docs/research-review.md')}>Research review</a>
      <span className="spacer" />
      <span className="status"><i /> Gate rerun in browser</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

const sha256 = async (bytes) => (crypto?.subtle ? toHex(await crypto.subtle.digest('SHA-256', bytes)) : null);

// Loads the two CSVs served next to the page, hashes the frozen file and refits the model once.
function useLab() {
  const [lab, setLab] = useState({ status: 'loading' });
  useEffect(() => {
    let live = true;
    const base = import.meta.env.BASE_URL;
    (async () => {
      try {
        const [fb, ct] = await Promise.all([
          fetch(`${base}data/test_frozen.csv`).then((r) => r.arrayBuffer()),
          fetch(`${base}data/credit.csv`).then((r) => r.text()),
        ]);
        const frozenBytes = new Uint8Array(fb);
        const hash = await sha256(frozenBytes);
        await new Promise((r) => setTimeout(r, 0)); // let the page paint before the refit
        const credit = parseCsv(ct), frozen = parseCsv(new TextDecoder().decode(frozenBytes));
        const { train, test } = split(credit, frozen);
        const features = credit.header.filter((h) => h !== TARGET);
        const yi = frozen.header.indexOf(TARGET);
        const y = test.map((r) => r[yi]);
        const t0 = performance.now();
        const p = predict(fit(credit.header, train, features), test);
        const ms = performance.now() - t0;
        if (live) setLab({ status: 'ready', frozenBytes, hash, header: credit.header, train, test, features, y, p, ms, auc: auc(p, y), boot: bootstrapAuc(p, y, 500, 19) });
      } catch (e) {
        if (live) setLab({ status: 'error', error: String(e) });
      }
    })();
    return () => { live = false; };
  }, []);
  return lab;
}

export default function App() {
  useReveal();
  const lab = useLab();
  const [threshold, setThreshold] = useState(0.5);
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <GateConsole lab={lab} />
      <RecordedRuns />
      <Evidence lab={lab} />
      <Scorecard lab={lab} threshold={threshold} setThreshold={setThreshold} />
      <Audit lab={lab} threshold={threshold} />
      <Review />
      <Reproduce />
      <Dock />
      <footer>
        Data: Default of Credit Card Clients, UCI Machine Learning Repository (Yeh and Lien, 2009). Every number on this page is read from the repo’s <code>records/</code>, <code>demo/</code> and <code>gate_config.json</code> at build time, or computed in your browser from <code>data/*.csv</code>, and labelled which.
      </footer>
    </div>
  );
}

function Hero() {
  return (
    <header className="card hero window" id="top" data-title="ModelGate.app">
      <div className="head">
        <p className="eyebrow">Promotion gate · credit-risk model</p>
        <span className="pill good">● PROMOTE {rec.model}: PASS</span>
      </div>
      <h1>A gate that said <em>no</em> first, then passed {rec.model} at AUC {rec.auc}.</h1>
      <p className="sub">
        A logistic-regression credit scorer is promoted only when three evidence records check out: an eval record that beats a floor fixed before training, earned on a hashed frozen test set; a data card; and a signed approval line.
        The gate reads the records, decides, and only then reports. Run 1 had no records and was blocked. Run 2 passed. The console below reruns the same gate logic in your browser on the same files.
      </p>
      <div className="actions">
        <a className="btn primary" href="#gate">Try to break the gate</a>
        <a className="btn ghost" href="#scorecard">Refit the model live</a>
        <span className="hint">Recorded numbers from <code>records/eval_record.json</code></span>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{rec.auc}</div><div className="cap">Test AUC (recorded, sklearn) on the frozen split</div></div>
        <div className="tile"><div className="num">{config.floor_auc}</div><div className="cap">Floor AUC, set {config.floor_set}</div></div>
        <div className="tile"><div className="num">{pct(rec.accuracy)}</div><div className="cap">Accuracy at threshold 0.5 (recorded)</div></div>
        <div className="tile"><div className="num">{fmt(rec.test_rows)}</div><div className="cap">Frozen test rows, sha256 {FROZEN_HASH.slice(0, 8)}…</div></div>
      </div>
    </header>
  );
}

function Check({ checked, onChange, children }) {
  return <label className="check"><input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> {children}</label>;
}

function GateConsole({ lab }) {
  const [hasEval, setHasEval] = useState(true);
  const [hasCard, setHasCard] = useState(true);
  const [approvals, setApprovals] = useState(approvalsText);
  const [floor, setFloor] = useState(config.floor_auc);
  const [tamper, setTamper] = useState(false);
  const [tamperHash, setTamperHash] = useState(null);
  const approvalsId = useId();

  // Tampering edits one digit in the first data row of the real frozen file, then rehashes it here.
  useEffect(() => {
    if (!tamper || lab.status !== 'ready') return;
    const b = lab.frozenBytes.slice();
    const i = b.indexOf(10) + 1; // first byte after the header line
    b[i] = b[i] === 57 ? 56 : b[i] + 1;
    sha256(b).then(setTamperHash);
  }, [tamper, lab]);

  const actualHash = lab.status !== 'ready' ? null : tamper ? tamperHash : lab.hash;
  const result = actualHash == null ? null : runGate({
    configText: JSON.stringify({ ...config, floor_auc: floor }),
    evalText: hasEval ? evalText : null,
    dataCard: hasCard ? dataCardText : null,
    approvals, frozenHashText, actualHash,
  });
  const preset = (shipped) => { setHasEval(shipped); setHasCard(shipped); setApprovals(shipped ? approvalsText : APPROVAL_HEADER); setFloor(config.floor_auc); setTamper(false); };

  return (
    <section id="gate" className="card window reveal" data-title="gate.py — live in your browser">
      <div className="head">
        <div>
          <p className="eyebrow">Live gate console</p>
          <h2>Remove a record, fake an approval, or tamper with the test set.</h2>
          <p className="sub">
            A line-for-line JavaScript port of <a href={blob('src/gate.py')}>src/gate.py</a>, fed the repo’s real records. The build fails unless this port prints exactly the two recorded transcripts in <code>demo/</code>.
            The SHA-256 of <code>data/test_frozen.csv</code> is computed here from the served file.
          </p>
        </div>
        {result && <span className={`pill ${result.code ? 'bad' : 'good'}`}>exit {result.code}: {result.code ? 'BLOCKED' : 'PASS'}</span>}
      </div>
      <div className="grid2">
        <div>
          <div className="checks">
            <Check checked={hasEval} onChange={setHasEval}><code>records/eval_record.json</code> present</Check>
            <Check checked={hasCard} onChange={setHasCard}><code>records/data_card.md</code> present</Check>
            <Check checked={tamper} onChange={setTamper}>Tamper one byte of <code>data/test_frozen.csv</code></Check>
          </div>
          <Range label="Floor AUC in gate_config.json" value={floor} min={0.5} max={0.8} step={0.01} onChange={(v) => setFloor(+v.toFixed(2))} show={floor.toFixed(2)} />
          <div className="field" style={{ marginTop: 14 }}>
            <label htmlFor={approvalsId}>records/approvals.log (editable)</label>
            <textarea id={approvalsId} rows={5} value={approvals} onChange={(e) => setApprovals(e.target.value)} spellCheck={false} />
          </div>
          <div className="actions" style={{ marginTop: 10 }}>
            <button className="btn ghost" onClick={() => setApprovals(APPROVAL_HEADER)}>Delete approval</button>
            <button className="btn ghost" onClick={() => setApprovals(APPROVAL_HEADER + 'APPROVED Jesun Ahmad Ushno 19-10-2026\n')}>Malform the date</button>
          </div>
          <div className="actions" style={{ marginTop: 10 }}>
            <button className="btn primary" onClick={() => preset(true)}>Shipped records</button>
            <button className="btn ghost" onClick={() => preset(false)}>Run 1 state</button>
          </div>
        </div>
        <div>
          <pre className="term" aria-live="polite">{result ? `$ python src/gate.py\n${result.out}$ echo $?\n${result.code}` : lab.status === 'error' ? `could not load data: ${lab.error}` : 'hashing data/test_frozen.csv …'}</pre>
          <p className="hint">
            Hash in use: <span className="mono">{actualHash ? `${actualHash.slice(0, 16)}…` : '…'}</span> {actualHash && (actualHash === FROZEN_HASH ? <span className="pill good">matches test_frozen.sha256</span> : <span className="pill bad">differs from test_frozen.sha256</span>)}
          </p>
          <p className="hint">The floor is read from <code>gate_config.json</code>. An earlier version read it from the eval record itself, so editing that record could lower the bar; this branch fixed that, with a test.</p>
        </div>
      </div>
    </section>
  );
}

function Replay({ title, text, tone }) {
  const lines = text.trimEnd().split('\n');
  const [n, setN] = useState(lines.length);
  const timer = useRef(null);
  const play = () => {
    clearInterval(timer.current);
    setN(0);
    timer.current = setInterval(() => setN((k) => { if (k + 1 >= lines.length) clearInterval(timer.current); return k + 1; }), 260);
  };
  useEffect(() => () => clearInterval(timer.current), []);
  return (
    <div>
      <div className="head" style={{ marginBottom: 8 }}>
        <h3>{title}</h3>
        <button className="btn ghost" onClick={play} aria-label={`Replay ${title}`}>▶ Replay</button>
      </div>
      <pre className="term">{lines.slice(0, n).join('\n')}{n < lines.length ? '\n▌' : ''}</pre>
      <p className="hint"><span className={`pill ${tone}`}>recorded output of a real run</span></p>
    </div>
  );
}

function RecordedRuns() {
  return (
    <section id="runs" className="card window reveal" data-title="run_demo.sh — recorded">
      <p className="eyebrow">Recorded replay</p>
      <h2>The done test, as it actually printed.</h2>
      <p className="sub">
        <a href={blob('run_demo.sh')}>run_demo.sh</a> trains, sets the evidence records aside, runs the gate (blocked), writes the eval record and data card, adds one approval line, and runs the gate again (pass).
        These are the saved transcripts <code>demo/run1_blocked.txt</code> and <code>demo/run2_pass.txt</code>, imported raw. Replay only re-types them; it does not rerun Python.
      </p>
      <div className="grid2" style={{ marginTop: 16 }}>
        <Replay title="Run 1: records incomplete" text={run1} tone="bad" />
        <Replay title="Run 2: records complete" text={run2} tone="good" />
      </div>
    </section>
  );
}

const FILES = [
  ['eval_record.json', 'records/eval_record.json', evalText],
  ['data_card.md', 'records/data_card.md', dataCardText],
  ['approvals.log', 'records/approvals.log', approvalsText],
  ['model_card.md', 'models/model_v1/model_card.md', modelCardText],
  ['gate_config.json', 'gate_config.json', configText],
];

function Evidence({ lab }) {
  const [tab, setTab] = useState(0);
  const [name, path, text] = FILES[tab];
  return (
    <section id="records" className="card window reveal" data-title={`${name} — evidence records`}>
      <div className="head">
        <div>
          <p className="eyebrow">Evidence records</p>
          <h2>The three records the gate reads, shown raw.</h2>
        </div>
        <div className="seg scroll-x" role="group" aria-label="Choose a record file">
          {FILES.map(([n], i) => <button key={n} aria-pressed={tab === i} onClick={() => setTab(i)}>{n}</button>)}
        </div>
      </div>
      <div className="grid2">
        <div>
          <pre className="term light">{text}</pre>
          <p className="hint"><a href={blob(path)}>{path} on GitHub ↗</a></p>
        </div>
        <div>
          <h3>Live hash check</h3>
          <p className="hint" style={{ marginTop: 4 }}>SHA-256 of the served <code>data/test_frozen.csv</code>, computed in your browser with SubtleCrypto, against <code>records/test_frozen.sha256</code>.</p>
          <div className="tile" style={{ marginTop: 10 }}>
            <div className="hint">Recorded</div><div className="mono hash">{FROZEN_HASH}</div>
            <div className="hint" style={{ marginTop: 8 }}>Computed here</div><div className="mono hash">{lab.status === 'ready' ? lab.hash ?? 'SubtleCrypto unavailable in this browser' : lab.status === 'error' ? 'could not load' : 'hashing…'}</div>
            <div style={{ marginTop: 10 }}>{lab.status === 'ready' && lab.hash && (lab.hash === FROZEN_HASH ? <span className="pill good">● MATCH: the scored set is the frozen set</span> : <span className="pill bad">● MISMATCH</span>)}</div>
          </div>
          <p className="hint">The eval record also carries this hash (<code>test_sha256</code>), so the gate can prove the AUC was earned on the frozen rows. It carries no hash of the model file; see the findings.</p>
        </div>
      </div>
    </section>
  );
}

function Scorecard({ lab, threshold, setThreshold }) {
  const ready = lab.status === 'ready';
  const curve = useMemo(() => (ready ? roc(lab.p, lab.y) : []), [lab, ready]);
  const cm = ready ? confusion(lab.p, lab.y, threshold) : null;
  const posRate = ready ? lab.y.reduce((s, v) => s + v, 0) / lab.y.length : null;
  const W = 420, H = 380, pad = { l: 40, r: 10, t: 10, b: 34 };
  const sx = (v) => pad.l + v * (W - pad.l - pad.r), sy = (v) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const pt = cm && [cm.fp / (cm.fp + cm.tn), cm.tp / (cm.tp + cm.fn)];
  // Thin the curve for drawing; the AUC uses every point.
  const step = Math.max(1, Math.floor(curve.length / 600));
  const drawn = curve.filter((_, i) => i % step === 0 || i === curve.length - 1);
  return (
    <section id="scorecard" className="card window reveal" data-title="roc.svg — browser refit">
      <div className="head">
        <div>
          <p className="eyebrow">Model scorecard · computed in your browser</p>
          <h2>Refit the same model on the same {fmt(TRAIN_ROWS)} rows, then score the frozen {fmt(rec.test_rows)}.</h2>
          <p className="sub">
            Train rows are <code>credit.csv</code> minus the IDs in the frozen file, which reproduces the split in <a href={blob('src/train.py')}>train.py</a>. Features are standardized on train and fit with the scikit-learn objective (L2, C = 1) by Newton’s method.
            The refit is labelled <b>browser refit</b>; the gate uses the <b>recorded</b> sklearn number. The bootstrap interval and the baseline are computed here, not claimed by the repo.
          </p>
        </div>
        {ready && <span className="pill info">refit in {lab.ms.toFixed(0)} ms</span>}
      </div>
      {!ready ? <p className="hint">{lab.status === 'error' ? `Could not load data: ${lab.error}` : 'Loading data/credit.csv and refitting…'}</p> : (
        <>
          <div className="tiles" style={{ marginTop: 0 }}>
            <div className="tile"><div className="num">{lab.auc.toFixed(4)}</div><div className="cap">AUC, browser refit</div></div>
            <div className="tile"><div className="num">{rec.auc}</div><div className="cap">AUC, recorded (sklearn, eval_record.json)</div></div>
            <div className="tile"><div className="num">{lab.boot.lo.toFixed(3)}–{lab.boot.hi.toFixed(3)}</div><div className="cap">95% bootstrap interval, {lab.boot.runs} resamples, seed {lab.boot.seed} (computed here)</div></div>
            <div className="tile"><div className="num">{pct(1 - posRate)}</div><div className="cap">Accuracy of always predicting “no default” (test positive rate {pct(posRate, 2)})</div></div>
          </div>
          <div className="grid2" style={{ marginTop: 16 }}>
            <div>
              <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`ROC curve of the browser refit, AUC ${lab.auc.toFixed(3)}, with the operating point at threshold ${threshold.toFixed(2)}`}>
                {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                  <g key={t}>
                    <line className="axis" x1={sx(0)} x2={sx(1)} y1={sy(t)} y2={sy(t)} /><text x={pad.l - 6} y={sy(t) + 4} textAnchor="end">{t}</text>
                    <line className="axis" x1={sx(t)} x2={sx(t)} y1={sy(0)} y2={sy(1)} /><text x={sx(t)} y={H - 18} textAnchor="middle">{t}</text>
                  </g>
                ))}
                <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--muted)" strokeDasharray="4 4" />
                <polygon points={`${sx(0)},${sy(0)} ${drawn.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} ${sx(1)},${sy(0)}`} fill="var(--primary)" opacity="0.08" />
                <polyline fill="none" stroke="var(--primary)" strokeWidth="2.5" points={drawn.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} />
                <circle cx={sx(pt[0])} cy={sy(pt[1])} r="6" fill="var(--warn)" stroke="var(--surface-strong)" strokeWidth="2" />
                <text x={sx(0.5)} y={H - 2} textAnchor="middle">false-positive rate</text>
                <text x={sx(0.98)} y={sy(0.06)} textAnchor="end">dashed: random scorer</text>
              </svg>
            </div>
            <div>
              <Range label="Decision threshold" value={threshold} min={0.05} max={0.95} step={0.01} onChange={setThreshold} show={threshold.toFixed(2)} />
              <table style={{ marginTop: 8 }}>
                <thead><tr><th></th><th>Predicted default</th><th>Predicted no default</th></tr></thead>
                <tbody>
                  <tr><th>Defaulted</th><td className="ok">{fmt(cm.tp)} TP</td><td className="no">{fmt(cm.fn)} FN</td></tr>
                  <tr><th>Did not default</th><td className="no">{fmt(cm.fp)} FP</td><td className="ok">{fmt(cm.tn)} TN</td></tr>
                </tbody>
              </table>
              <div className="bars" style={{ marginTop: 16 }}>
                {[['Accuracy', cm.accuracy], ['Majority-class baseline', 1 - posRate], ['Precision', cm.precision], ['Recall (defaults caught)', cm.recall]].map(([k, v]) => (
                  <div className="bar-row" key={k}>
                    <div className="lab"><span>{k}</span><b>{pct(v)}</b></div>
                    <div className="track"><div className="fill" style={{ width: `${v * 100}%`, background: k.startsWith('Majority') ? 'var(--muted)' : 'var(--primary)' }} /></div>
                  </div>
                ))}
              </div>
              <p className="hint" style={{ marginTop: 12 }}>
                At 0.5 the refit is {((cm.accuracy - (1 - posRate)) * 100).toFixed(1)} points above the majority baseline{threshold === 0.5 ? '' : ' (move the slider back to 0.5 to compare with the recorded accuracy)'}. The repo fixes 0.5 without a stated rationale.
              </p>
            </div>
          </div>
        </>
      )}
    </section>
  );
}

function Audit({ lab, threshold }) {
  const [drop, setDrop] = useState([]);
  const [res, setRes] = useState(null);
  const ready = lab.status === 'ready';
  useEffect(() => {
    if (!ready) return;
    if (!drop.length) { setRes({ p: lab.p, auc: lab.auc, drop }); return; }
    const t = setTimeout(() => {
      const features = lab.features.filter((f) => !drop.includes(f));
      const p = predict(fit(lab.header, lab.train, features), lab.test);
      setRes({ p, auc: auc(p, lab.y), drop });
    }, 30);
    return () => clearTimeout(t);
  }, [drop, lab, ready]);
  const sexCol = ready ? lab.header.indexOf('SEX') : -1;
  const groups = res && [1, 2].map((g) => {
    const idx = lab.test.map((r, i) => (r[sexCol] === g ? i : -1)).filter((i) => i >= 0);
    const p = idx.map((i) => res.p[i]), y = idx.map((i) => lab.y[i]);
    const c = confusion(p, y, threshold);
    return { g, n: idx.length, pos: y.reduce((s, v) => s + v, 0) / y.length, auc: auc(p, y), flag: c.flagRate, recall: c.recall };
  });
  const toggle = (f, on) => setDrop((d) => (on ? [...d, f] : d.filter((x) => x !== f)));
  const pending = res && res.drop !== drop;
  return (
    <section id="audit" className="card window reveal" data-title="audit.ipynb — ablation · subgroups">
      <p className="eyebrow">Ablation and subgroup audit · computed in your browser</p>
      <h2>The model uses the row ID and protected attributes. What do they buy?</h2>
      <p className="sub">
        <a href={blob('src/train.py', lineOf(trainSrc, 'features = [c for c'))}>train.py</a> uses every column except the target, so ID, SEX, MARRIAGE and AGE are all inputs. Untick a feature to refit without it.
        The subgroup table splits the frozen rows by SEX (UCI coding: 1 = male, 2 = female) at the scorecard’s threshold. None of this is in the repo’s records; it is an audit computed here.
      </p>
      <div className="checks" style={{ marginTop: 16 }} role="group" aria-label="Features used by the refit">
        {PROTECTED.map((f) => <Check key={f} checked={!drop.includes(f)} onChange={(on) => toggle(f, !on)}>use <code>{f}</code></Check>)}
      </div>
      {!ready || !res ? <p className="hint">Waiting for the refit…</p> : (
        <div className="grid2" style={{ marginTop: 8 }}>
          <div>
            <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 0 }}>
              <div className="tile"><div className="num">{res.auc.toFixed(4)}</div><div className="cap">AUC with {lab.features.length - res.drop.length} features{pending ? ' (refitting…)' : ''}</div></div>
              <div className="tile"><div className="num">{(res.auc - lab.auc >= 0 ? '+' : '') + (res.auc - lab.auc).toFixed(4)}</div><div className="cap">Change vs the full {lab.features.length}-feature refit ({lab.auc.toFixed(4)})</div></div>
            </div>
            <p className="hint" style={{ marginTop: 12 }}>A change smaller than the bootstrap interval’s half-width ({((lab.boot.hi - lab.boot.lo) / 2).toFixed(3)}) is not distinguishable from noise on this split. Dropping features from v1 would be a new model, v2, and would have to pass the gate on its own records.</p>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>SEX</th><th>Rows</th><th>Default rate</th><th>AUC</th><th>Flagged at {threshold.toFixed(2)}</th><th>Recall</th></tr></thead>
              <tbody>
                {groups.map((r) => <tr key={r.g}><td>{r.g} ({r.g === 1 ? 'male' : 'female'})</td><td>{fmt(r.n)}</td><td>{pct(r.pos)}</td><td>{r.auc.toFixed(3)}</td><td>{pct(r.flag)}</td><td>{pct(r.recall)}</td></tr>)}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

function Review() {
  const items = [
    { ok: true, t: 'Floor now read from gate_config.json', f: 'src/gate.py', l: lineOf(gateSrc, 'check_eval(cfg.get("floor_auc"))'), d: 'The gate used the floor stored in the eval record, so editing the record could lower the bar. Fixed on this branch; tests/test_gate.py proves a record carrying floor 0.5 no longer passes at AUC 0.6.' },
    { ok: true, t: 'A malformed eval record blocks instead of crashing', f: 'src/gate.py', l: lineOf(gateSrc, 'if not is_number(rec.get("auc"))'), d: 'A missing auc or floor raised a TypeError traceback. It now prints BLOCKED with a reason, covered by tests.' },
    { ok: false, t: 'Eval record is not bound to the model file', f: 'src/evaluate.py', l: lineOf(evaluateSrc, 'record = {'), d: 'No hash of models/model_v1/model.pkl is recorded or checked, so the model could be swapped after evaluation and still pass.' },
    { ok: false, t: 'Approval is not bound to a model or date window', f: 'src/gate.py', l: lineOf(gateSrc, 'APPROVED\\s+'), d: 'Any APPROVED line with any well-formed date passes, for any model version.' },
    { ok: false, t: 'Row ID and protected attributes are model inputs', f: 'src/train.py', l: lineOf(trainSrc, 'features = [c for c'), d: 'ID, SEX, MARRIAGE and AGE are features. The cards now say so; the audit window above measures their effect. Changing it means a v2 model through the gate.' },
    { ok: false, t: 'One split, one seed, no interval in the records', f: 'src/evaluate.py', l: lineOf(evaluateSrc, 'auc = roc_auc_score'), d: 'The recorded AUC has no uncertainty and no baseline; this page adds a bootstrap interval and the majority baseline, labelled as computed here.' },
    { ok: false, t: 'Records are stamped with the run date', f: 'src/evaluate.py', l: lineOf(evaluateSrc, '"evaluated": date.today()'), d: `Rerunning rewrites shipped files with today’s date, so a byte-identical rerun is impossible. requirements.txt is now pinned to the versions that reproduced AUC ${rec.auc}.` },
  ];
  return (
    <section id="review" className="card window reveal" data-title="REVIEW.md — audit findings">
      <p className="eyebrow">Limitations and threats</p>
      <h2>What the gate checks, and what it does not yet.</h2>
      <p className="sub">Line numbers are computed from the source at build time. The full grading against a research rubric is in <a href={blob('docs/research-review.md')}>docs/research-review.md</a>. This is a teaching model on one public 2005 dataset, never a credit decision tool.</p>
      <div className="grid2" style={{ marginTop: 16 }}>
        {items.map((c) => (
          <div className="tile" key={c.t}>
            <div className="head" style={{ marginBottom: 6 }}><h3>{c.t}</h3><span className={`pill ${c.ok ? 'good' : 'warn'}`}>{c.ok ? 'Fixed' : 'Open'}</span></div>
            <p className="hint" style={{ margin: 0, fontSize: 13 }}>{c.d}</p>
            <p className="hint" style={{ margin: '6px 0 0' }}><a href={blob(c.f, c.l)}>{c.f}:{c.l}</a></p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Reproduce() {
  const docs = [
    ['README.md', 'README', 'What it does, the done test'],
    ['src/gate.py', 'gate.py', 'The promotion gate'],
    ['tests/test_gate.py', 'Gate tests', `${N_TESTS} stdlib behaviour tests`],
    ['records/eval_record.json', 'Eval record', 'AUC, accuracy, test hash'],
    ['run_demo.sh', 'run_demo.sh', 'Trains, blocks, then passes'],
    ['docs/research-review.md', 'Research review', 'Rubric grades and gaps'],
  ];
  return (
    <section id="reproduce" className="card window reveal" data-title="Terminal — reproduce">
      <p className="eyebrow">Reproduce it</p>
      <h2>Three commands rerun the pipeline and the gate.</h2>
      <ol className="steps" style={{ marginTop: 16 }}>
        <li><span className="n">1</span><div><b>Set up</b><div className="hint"><code>python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt</code></div></div></li>
        <li><span className="n">2</span><div><b>Run the done test</b><div className="hint"><code>./run_demo.sh</code> trains, prints BLOCKED, writes the records, prints PASS.</div></div></li>
        <li><span className="n">3</span><div><b>Test the gate</b><div className="hint"><code>python3 -m unittest discover -s tests -v</code> builds throwaway record sets and checks every way the gate must say no.</div></div></li>
      </ol>
      <div className="docs" style={{ marginTop: 20 }}>
        {docs.map(([p, t, d]) => <a key={p} href={blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}
