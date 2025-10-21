// data-loader.js
// Minimal, dependency-free CSV loader + preprocessing for credit_risk_dataset.csv (or similar tabular credit data).
// - Detects numeric vs categorical columns automatically (target must be named 'loan_status' with 0/1 values).
// - Fits preprocessing on train (median/mode imputation, standardize numeric, one-hot categorical with top-K vocab).
// - Transforms any subsequent data using the same fitted state.
// - Produces tensors suitable for an MLP or GRU-by-time=1 model (we provide reshape helper).
// All functions are attached to window.DL for use in app.js.

(() => {
  const TOP_K = 12; // cap vocab size per categorical (rest → "Other")
  const TARGET_COL = 'loan_status';

  function parseCSV(text) {
    // Basic RFC4180-ish CSV parser for comma-separated values (no quotes handling for simplicity).
    // If your CSV has quoted commas, adapt the parser accordingly.
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length < 2) throw new Error('CSV must include header and at least 1 data row.');
    const header = lines[0].split(',').map(h => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(','); // simple split (assumes no quoted commas)
      if (parts.length !== header.length) continue; // skip malformed
      const obj = {};
      for (let j = 0; j < header.length; j++) {
        obj[header[j]] = parts[j].trim();
      }
      rows.push(obj);
    }
    return { header, rows };
  }

  function inferColumnTypes(header, rows) {
    // Any column with >90% numeric parse success is numeric (excluding target).
    // The TARGET_COL must exist and will be treated as binary label.
    if (!header.includes(TARGET_COL)) {
      throw new Error(`Target column '${TARGET_COL}' not found.`);
    }
    const numeric = [];
    const categorical = [];
    for (const col of header) {
      if (col === TARGET_COL) continue;
      let numOk = 0, nonEmpty = 0;
      for (let i = 0; i < rows.length && i < 1000; i++) {
        const v = rows[i][col];
        if (v !== '' && v != null) {
          nonEmpty++;
          if (Number.isFinite(parseFloat(v))) numOk++;
        }
      }
      if (nonEmpty > 0 && numOk / nonEmpty >= 0.9) numeric.push(col);
      else categorical.push(col);
    }
    return { numeric, categorical, target: TARGET_COL };
  }

  function computeMedian(arr) {
    const a = arr.filter(x => Number.isFinite(x)).sort((x,y)=>x-y);
    if (a.length === 0) return 0;
    const m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : (a[m-1] + a[m]) / 2;
    }

  function computeMode(arr) {
    const counts = new Map();
    for (const v of arr) {
      const vv = (v == null || v === '') ? '__MISSING__' : String(v);
      counts.set(vv, (counts.get(vv) || 0) + 1);
    }
    let best = '__MISSING__', bestC = -1;
    for (const [k,c] of counts) { if (c > bestC) { best = k; bestC = c; } }
    return best;
  }

  function topKVocab(values, K = TOP_K) {
    const counts = new Map();
    for (const v of values) {
      const key = (v == null || v === '') ? '__MISSING__' : String(v);
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    const sorted = [...counts.entries()].sort((a,b)=>b[1]-a[1]).map(([k])=>k);
    const vocab = sorted.slice(0, K - 1); // reserve 1 slot for "Other"
    if (!vocab.includes('__MISSING__')) vocab.push('__MISSING__');
    return vocab;
  }

  function fitPreprocess(rows, schema) {
    // schema includes numeric[], categorical[], target
    const prep = {
      numeric: schema.numeric.slice(),
      categorical: schema.categorical.slice(),
      target: schema.target,
      median: {},
      mean: {},
      std: {},
      mode: {},
      vocab: {}, // { col: [cats...] + 'Other' }
      featureOrder: [], // full expanded columns
    };
    // Fit numeric stats
    for (const col of prep.numeric) {
      const vals = rows.map(r => {
        const x = parseFloat(r[col]); return Number.isFinite(x) ? x : NaN;
      });
      const med = computeMedian(vals);
      // Mean/std on imputed values (use med for NaN)
      let sum = 0, sum2 = 0, n = 0;
      for (let i = 0; i < vals.length; i++) {
        const v = Number.isFinite(vals[i]) ? vals[i] : med;
        sum += v; sum2 += v*v; n++;
      }
      const mean = n ? sum / n : 0;
      const variance = n ? Math.max(1e-8, sum2 / n - mean * mean) : 1e-8;
      prep.median[col] = med;
      prep.mean[col] = mean;
      prep.std[col] = Math.sqrt(variance);
    }
    // Fit categorical vocab/mode
    for (const col of prep.categorical) {
      const vals = rows.map(r => r[col]);
      prep.mode[col] = computeMode(vals);
      const vocab = topKVocab(vals, TOP_K);
      // Ensure "Other" token exists
      if (!vocab.includes('__OTHER__')) vocab.push('__OTHER__');
      prep.vocab[col] = vocab;
    }
    // Build expanded feature order: numeric + one-hot columns
    const order = [];
    for (const col of prep.numeric) order.push(col);
    for (const col of prep.categorical) {
      const vocab = prep.vocab[col];
      for (const cat of vocab) {
        // Skip creating explicit column for '__OTHER__', it is implicit (all zeros means Other)
        if (cat === '__OTHER__') continue;
        order.push(`${col}__${cat}`);
      }
      // Track implicit OTHER column name for clarity (not included in features)
    }
    prep.featureOrder = order;
    return prep;
  }

  function transform(rows, prep) {
    const N = rows.length;
    const D = prep.featureOrder.length;
    const X = new Float32Array(N * D);
    const y = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const r = rows[i];
      // Fill numeric
      let o = 0;
      for (const col of prep.numeric) {
        const raw = parseFloat(r[col]);
        const v = Number.isFinite(raw) ? raw : prep.median[col];
        const z = (v - prep.mean[col]) / prep.std[col];
        X[i*D + o] = z; o++;
      }
      // Fill categorical one-hots (with implicit OTHER)
      for (const col of prep.categorical) {
        const vocab = prep.vocab[col];
        const val = (r[col] == null || r[col] === '') ? '__MISSING__' : String(r[col]);
        let matched = false;
        for (const cat of vocab) {
          if (cat === '__OTHER__') continue; // implicit
          const hit = (val === cat);
          if (hit) matched = true;
          X[i*D + o] = hit ? 1 : 0;
          o++;
        }
        // If not matched any, that's OTHER → keep all zeros (already 0)
      }
      // Target
      const labRaw = r[prep.target];
      const lab = (labRaw === '1' || labRaw === 1 || labRaw === true || labRaw === 'true') ? 1 : 0;
      y[i] = lab;
    }
    const xs = tf.tensor2d(X, [N, D], 'float32');
    const ys = tf.tensor2d(y, [N, 1], 'float32');
    return { xs, ys, D, N };
  }

  function stratifiedSplit(rows, targetCol = TARGET_COL, valRatio = 0.2) {
    const pos = [], neg = [];
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const y = (r[targetCol] === '1' || r[targetCol] === 1 || r[targetCol] === true || r[targetCol] === 'true') ? 1 : 0;
      (y ? pos : neg).push(r);
    }
    tf.util.shuffle(pos); tf.util.shuffle(neg);
    const posVal = Math.floor(pos.length * valRatio);
    const negVal = Math.floor(neg.length * valRatio);
    const val = pos.slice(0, posVal).concat(neg.slice(0, negVal));
    const train = pos.slice(posVal).concat(neg.slice(negVal));
    tf.util.shuffle(train); tf.util.shuffle(val);
    return { train, val };
  }

  function reshapeToSeq1(xs2d) {
    // For GRU: reshape [N, D] -> [N, 1, D]
    const [N, D] = xs2d.shape;
    return xs2d.reshape([N, 1, D]);
  }

  // Convenience loader for a single CSV file (train/val split inside).
  async function loadAndPrepareFromFile(file) {
    const text = await file.text();
    const { header, rows } = parseCSV(text);
    const schema = inferColumnTypes(header, rows);
    const { train, val } = stratifiedSplit(rows, schema.target, 0.2);
    const prep = fitPreprocess(train, schema);
    const trainT = transform(train, prep);
    const valT = transform(val, prep);
    return { header, schema, prep, train: trainT, val: valT };
  }

  // Export helpers
  window.DL = {
    parseCSV,
    inferColumnTypes,
    fitPreprocess,
    transform,
    stratifiedSplit,
    reshapeToSeq1,
    loadAndPrepareFromFile,
  };
})();
