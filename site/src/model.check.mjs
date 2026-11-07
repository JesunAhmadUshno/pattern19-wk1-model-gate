// Known-answer self-check, run by `npm run build` before vite.
// Proves the browser code reproduces the repo: the JS gate prints exactly the two recorded
// transcripts in demo/, the frozen test file hashes to records/test_frozen.sha256, and the
// in-browser refit matches the recorded sklearn AUC and accuracy (records/eval_record.json).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import * as M from './model.js';

const repo = new URL('../../', import.meta.url);
const read = (p) => fs.readFileSync(new URL(p, repo), 'utf8');
const bytes = (p) => fs.readFileSync(new URL(p, repo));

const configText = read('gate_config.json');
const frozenHashText = read('records/test_frozen.sha256');
const actualHash = createHash('sha256').update(bytes('data/test_frozen.csv')).digest('hex');
assert.equal(actualHash, frozenHashText.trim(), 'data/test_frozen.csv hash');

const shipped = { configText, frozenHashText, actualHash, evalText: read('records/eval_record.json'), dataCard: read('records/data_card.md'), approvals: read('records/approvals.log') };
const pass = M.runGate(shipped);
assert.equal(pass.out, read('demo/run2_pass.txt'), 'JS gate vs demo/run2_pass.txt');
assert.equal(pass.code, 0);
const header = read('records/approvals.log').split('\n').filter((l) => l.startsWith('#')).join('\n') + '\n';
const blocked = M.runGate({ ...shipped, evalText: null, dataCard: null, approvals: header });
assert.equal(blocked.out, read('demo/run1_blocked.txt'), 'JS gate vs demo/run1_blocked.txt');
assert.equal(blocked.code, 1);
// Edge cases mirrored from tests/test_gate.py
assert.match(M.runGate({ ...shipped, actualHash: '0'.repeat(64) }).out, /eval earned on a swapped test set/);
assert.match(M.runGate({ ...shipped, configText: JSON.stringify({ model_version: 'model v1', floor_auc: 0.75 }) }).out, /FAIL \(AUC 0\.724 below floor 0\.75\)/);
assert.match(M.runGate({ ...shipped, approvals: header + 'APPROVED Jesun 19-10-2026\n' }).out, /no approval line/);

const rec = JSON.parse(shipped.evalText);
const credit = M.parseCsv(read('data/credit.csv'));
const frozen = M.parseCsv(read('data/test_frozen.csv'));
const { train, test } = M.split(credit, frozen);
assert.equal(train.length, 24000, 'train rows (model card)');
assert.equal(test.length, rec.test_rows, 'frozen test rows');
const features = credit.header.filter((h) => h !== M.TARGET);
assert.equal(features.length, 24, 'model card: 24 feature columns');
const yi = frozen.header.indexOf(M.TARGET);
const y = test.map((r) => r[yi]);
const p = M.predict(M.fit(credit.header, train, features), test);
const a = M.auc(p, y), acc = M.confusion(p, y, 0.5).accuracy;
// lbfgs (sklearn) stops at tol 1e-4 while Newton converges fully, so allow 0.001 of drift.
assert.ok(Math.abs(a - rec.auc) < 0.001, `refit AUC ${a} vs recorded ${rec.auc}`);
assert.ok(Math.abs(acc - rec.accuracy) < 0.001, `refit accuracy ${acc} vs recorded ${rec.accuracy}`);
// AUC by rank sum equals the area under the ROC curve built from the same scores.
const pts = M.roc(p, y);
const trap = pts.slice(1).reduce((s, [x, t], i) => s + (x - pts[i][0]) * (t + pts[i][1]) / 2, 0);
assert.ok(Math.abs(trap - a) < 1e-9, 'ROC area equals rank AUC');
const ci = M.bootstrapAuc(p, y, 200, 19);
assert.ok(ci.lo < a && a < ci.hi, 'bootstrap interval brackets the point AUC');
console.log('model ok', { hash: actualHash.slice(0, 16), trainRows: train.length, refitAuc: +a.toFixed(4), recordedAuc: rec.auc, accuracy: +acc.toFixed(4) });

// Every source line the findings window cites must still exist.
for (const [f, needle] of [['src/gate.py', 'check_eval(cfg.get("floor_auc"))'], ['src/gate.py', 'if not is_number(rec.get("auc"))'], ['src/gate.py', 'APPROVED\\s+'],
  ['src/evaluate.py', 'record = {'], ['src/evaluate.py', 'auc = roc_auc_score'], ['src/evaluate.py', '"evaluated": date.today()'], ['src/train.py', 'features = [c for c']])
  assert.ok(M.lineOf(read(f), needle) > 0, `${f} no longer contains ${needle}`);
assert.match(read('models/model_v1/model_card.md'), /Train rows: 24000/);
console.log('cited lines ok');
