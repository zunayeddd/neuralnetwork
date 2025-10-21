// app.js
// UI wiring for credit risk GRU classifier in the browser (TensorFlow.js + tfjs-vis).
// - Load CSV, EDA preview, preprocessing fit/transform.
// - Train GRU (sequence length = 1 over tabular features), live charts.
// - Evaluate: ROC curve, AUC, threshold slider → confusion/precision/recall/F1.
// - Predict/export: probabilities.csv and submission.csv (thresholded).
// - Save/Load model via files (no IndexedDB). All local, no network.

(() => {
  // State
  let prep = null;
  let schema = null;
  let train = null, val = null;
  let model = null;
  let threshold = 0.35;
  let featureDim = 0;

    // --- Add at very top of app.js ---
(function waitForDeps() {
  if (window.DL && window.ModelFactory && window.tf) {
    // Deps ready — now run your app bootstrap safely
    if (typeof window.__APP_WIRED__ === 'function') {
      window.__APP_WIRED__();
    }
  } else {
    setTimeout(waitForDeps, 50);
  }
})();


  // DOM helpers
  const $ = (id) => document.getElementById(id);
  const setStatus = (msg) => { const el = $('status'); if (el) el.textContent = msg; };
  const log = (msg) => { const el = $('logs'); if (!el) return; el.textContent = (el.textContent ? el.textContent + '\n' : '') + msg; };

  function clearAll() {
    if (model) { model.dispose(); model = null; }
    if (train) { train.xs?.dispose?.(); train.ys?.dispose?.(); train = null; }
    if (val) { val.xs?.dispose?.(); val.ys?.dispose?.(); val = null; }
    prep = null; schema = null; featureDim = 0;
    $('preview').innerHTML = '';
    $('metrics').innerHTML = '';
    $('rocAuc').textContent = 'AUC: –';
    $('thrVal').textContent = threshold.toFixed(2);
    $('logs').textContent = '';
    setStatus('Reset.');
  }

  // CSV preview table
  function renderPreviewTable(rows, containerId, maxRows=12) {
    const el = $(containerId);
    if (!el) return;
    el.innerHTML = '';
    if (!rows || rows.length === 0) { el.textContent = 'No rows.'; return; }
    const header = Object.keys(rows[0]);
    const table = document.createElement('table'); table.className = 'tbl';
    const thead = document.createElement('thead'); const trh = document.createElement('tr');
    for (const h of header) { const th = document.createElement('th'); th.textContent = h; trh.appendChild(th); }
    thead.appendChild(trh); table.appendChild(thead);
    const tbody = document.createElement('tbody');
    for (let i = 0; i < Math.min(maxRows, rows.length); i++) {
      const tr = document.createElement('tr');
      for (const h of header) { const td = document.createElement('td'); td.textContent = rows[i][h]; tr.appendChild(td); }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    el.appendChild(table);
  }

  function computeMissingPerc(rows) {
    const totals = {};
    const miss = {};
    if (!rows || rows.length === 0) return { missing: {}, ratios: {} };
    const header = Object.keys(rows[0]);
    for (const h of header) { totals[h] = 0; miss[h] = 0; }
    for (const r of rows) {
      for (const h of header) {
        totals[h] += 1;
        const v = r[h];
        if (v == null || v === '') miss[h] += 1;
      }
    }
    const ratios = {};
    for (const h of header) ratios[h] = totals[h] ? (miss[h] / totals[h]) : 0;
    return { missing: miss, ratios };
  }

  function makeConfusion(yTrue, yScore, thr) {
    let TP=0, FP=0, TN=0, FN=0;
    for (let i = 0; i < yTrue.length; i++) {
      const p = yScore[i] >= thr ? 1 : 0;
      const t = yTrue[i] >= 0.5 ? 1 : 0;
      if (p===1 && t===1) TP++; else if (p===1 && t===0) FP++; else if (p===0 && t===1) FN++; else TN++;
    }
    const prec = (TP+FP) ? TP/(TP+FP) : 0;
    const rec  = (TP+FN) ? TP/(TP+FN) : 0;
    const f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : 0;
    return { TP, FP, TN, FN, prec, rec, f1 };
  }

  function computeROC(yTrue, yScore) {
    // Basic ROC by sorting scores and sweeping thresholds
    const pairs = yScore.map((s,i)=>[s, yTrue[i]]);
    pairs.sort((a,b)=>b[0]-a[0]); // desc
    let P = 0, N = 0; for (const [,y] of pairs) (y>=0.5?P++:N++);
    let TP=0, FP=0;
    const roc = [[0,0]];
    let auc = 0, prevFPR=0, prevTPR=0;
    for (let i = 0; i < pairs.length; i++) {
      const y = pairs[i][1] >= 0.5 ? 1:0;
      if (y===1) TP++; else FP++;
      const TPR = P ? TP/P : 0;
      const FPR = N ? FP/N : 0;
      roc.push([FPR, TPR]);
      // Trapezoid area
      auc += (FPR - prevFPR) * (TPR + prevTPR) / 2;
      prevFPR = FPR; prevTPR = TPR;
    }
    return { roc, auc: Math.max(0, Math.min(1, auc)) };
  }

  function renderMetrics(conf) {
    const el = $('metrics'); if (!el) return;
    el.innerHTML = `
      <div class="grid3">
        <div><b>TP</b>: ${conf.TP}</div>
        <div><b>FP</b>: ${conf.FP}</div>
        <div><b>TN</b>: ${conf.TN}</div>
        <div><b>FN</b>: ${conf.FN}</div>
      </div>
      <div class="grid3">
        <div><b>Precision</b>: ${(conf.prec*100).toFixed(2)}%</div>
        <div><b>Recall</b>: ${(conf.rec*100).toFixed(2)}%</div>
        <div><b>F1</b>: ${conf.f1.toFixed(3)}</div>
      </div>`;
  }

  async function onLoadCSV() {
    try {
      setStatus('Reading CSV…');
      const file = $('csvFile').files?.[0];
      if (!file) { alert('Choose credit_risk_dataset.csv first.'); return; }
      const text = await file.text();
      const { header, rows } = DL.parseCSV(text);
      schema = DL.inferColumnTypes(header, rows);
      renderPreviewTable(rows, 'preview', 12);
      const miss = computeMissingPerc(rows);
      const missList = Object.entries(miss.ratios).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join('  ·  ');
      $('missing').textContent = missList || 'No missing values detected.';
      const split = DL.stratifiedSplit(rows, schema.target, 0.2);
      prep = DL.fitPreprocess(split.train, schema);
      train = DL.transform(split.train, prep);
      val = DL.transform(split.val, prep);
      featureDim = train.D;
      $('shape').textContent = `Train: ${train.N} × ${train.D} • Val: ${val.N} × ${val.D}`;
      setStatus('Data ready. You can Train now.');
      $('btnTrain').disabled = false;
      $('btnEval').disabled = true;
      $('btnPredict').disabled = true;
      $('btnSave').disabled = true;
    } catch (err) {
      console.error(err);
      alert('Load error: ' + (err.message || err));
      setStatus('Load failed.');
    }
  }

  async function onTrain() {
    try {
      if (!train || !prep) { alert('Load data first.'); return; }
      if (model) { model.dispose(); model = null; }
      model = ModelFactory.buildGRUModel(featureDim);
      const xsTr = DL.reshapeToSeq1(train.xs);
      const xsVa = DL.reshapeToSeq1(val.xs);
      const surface = { name: 'Training', tab: 'Charts' };
      const callbacks = tfvis.show.fitCallbacks(surface, ['loss','val_loss','acc','val_acc'], { callbacks:['onEpochEnd'] });

      setStatus('Training…');
      const t0 = performance.now();
      const history = await model.fit(xsTr, train.ys, {
        epochs: 50,
        batchSize: Math.min(32, train.N),
        validationData: [xsVa, val.ys],
        shuffle: true,
        callbacks: {
          ...callbacks,
          onEpochEnd: async (epoch, logs) => {
            $('logs').textContent += `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} acc=${(logs.acc*100).toFixed(1)}% val_acc=${(logs.val_acc*100).toFixed(1)}%\n`;
            await tf.nextFrame();
          }
        }
      });
      xsTr.dispose(); xsVa.dispose();
      const t1 = performance.now();
      $('logs').textContent += `Finished in ${(t1-t0).toFixed(0)} ms\n`;
      setStatus('Training complete. Evaluate to see ROC/AUC.');
      $('btnEval').disabled = false;
      $('btnPredict').disabled = false;
      $('btnSave').disabled = false;
    } catch (err) {
      console.error(err);
      alert('Training error: ' + (err.message || err));
      setStatus('Training failed.');
    }
  }

  async function onEvaluate() {
    try {
      if (!model || !val) { alert('Train first.'); return; }
      setStatus('Evaluating…');
      const xsVa = DL.reshapeToSeq1(val.xs);
      const probsT = model.predict(xsVa);
      const probs = Array.from(await probsT.data());
      const labels = Array.from((await val.ys.data()));
      xsVa.dispose(); probsT.dispose();
      const { roc, auc } = computeROC(labels, probs);
      $('rocAuc').textContent = `AUC: ${auc.toFixed(4)} (Val N=${labels.length})`;
      // Draw ROC into simple canvas
      drawROC(roc, 'rocCanvas');
      // Confusion at current threshold
      const conf = makeConfusion(labels, probs, threshold);
      $('thrVal').textContent = threshold.toFixed(2);
      renderMetrics(conf);
      setStatus('Evaluation done.');
      // Store for slider use
      window.__VAL_LABELS__ = labels;
      window.__VAL_PROBS__ = probs;
    } catch (err) {
      console.error(err);
      alert('Evaluation error: ' + (err.message || err));
      setStatus('Evaluation failed.');
    }
  }

  function onThrChange() {
    const inp = $('thr'); threshold = Math.max(0, Math.min(1, Number(inp.value) || 0.5));
    $('thrVal').textContent = threshold.toFixed(2);
    const labels = window.__VAL_LABELS__, probs = window.__VAL_PROBS__;
    if (labels && probs) {
      const conf = makeConfusion(labels, probs, threshold);
      renderMetrics(conf);
    }
  }

  function drawROC(roc, canvasId) {
    const c = $(canvasId); const W = c.width = 320; const H = c.height = 240; const ctx = c.getContext('2d');
    ctx.fillStyle = '#0b1020'; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle = '#243b86'; ctx.lineWidth = 1; // axes
    ctx.beginPath(); ctx.moveTo(32, 8); ctx.lineTo(32, H-24); ctx.lineTo(W-8, H-24); ctx.stroke();
    // diagonal
    ctx.strokeStyle = '#555'; ctx.beginPath(); ctx.moveTo(32, H-24); ctx.lineTo(W-8, 8); ctx.stroke();
    // ROC
    ctx.strokeStyle = '#63b3ff'; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < roc.length; i++) {
      const [fpr,tpr] = roc[i];
      const x = 32 + (W-40) * fpr;
      const y = (H-24) - (H-32) * tpr;
      if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
    ctx.fillStyle = '#cfe3ff'; ctx.font = '12px system-ui'; ctx.fillText('FPR', W-40, H-8); ctx.fillText('TPR', 4, 16);
  }

  async function onPredictExport() {
    try {
      if (!model || !prep) { alert('Train first.'); return; }
      // Use validation set as a stand-in for "test" export here; if a separate test file is provided, you could add another loader.
      const xsVa = DL.reshapeToSeq1(val.xs);
      const probsT = model.predict(xsVa);
      const probs = Array.from(await probsT.data());
      xsVa.dispose(); probsT.dispose();
      // Build CSVs (no ID col guaranteed; we output row index)
      const linesProb = ['row,probability'];
      const linesSub  = ['row,loan_status'];
      for (let i = 0; i < probs.length; i++) {
        linesProb.push(`${i},${probs[i].toFixed(6)}`);
        const pred = probs[i] >= threshold ? 1 : 0;
        linesSub.push(`${i},${pred}`);
      }
      downloadText('probabilities.csv', linesProb.join('\n'));
      downloadText('submission.csv', linesSub.join('\n'));
      setStatus('Exported probabilities.csv and submission.csv');
    } catch (err) {
      console.error(err);
      alert('Predict/export error: ' + (err.message || err));
      setStatus('Predict/export failed.');
    }
  }

  function downloadText(filename, content) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = filename; a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href), 5000);
  }

  async function onSaveModel() {
    try {
      if (!model) { alert('No model trained.'); return; }
      await model.save('downloads://credit-risk-gru');
      setStatus('Model saved to downloads.');
    } catch (err) { alert('Save error: ' + (err.message||err)); }
  }

  async function onLoadModel() {
    try {
      const jf = $('mdlJson').files?.[0];
      const bf = $('mdlBin').files?.[0];
      if (!jf || !bf) { alert('Select model.json and weights.bin'); return; }
      const m = await tf.loadLayersModel(tf.io.browserFiles([jf, bf]));
      if (model) model.dispose();
      model = m;
      setStatus('Model loaded from files.');
      $('btnEval').disabled = false;
      $('btnPredict').disabled = false;
      $('btnSave').disabled = false;
    } catch (err) { alert('Load model error: ' + (err.message||err)); }
  }

  function wire() {
    $('btnLoad').addEventListener('click', onLoadCSV);
    $('btnTrain').addEventListener('click', onTrain);
    $('btnEval').addEventListener('click', onEvaluate);
    $('btnPredict').addEventListener('click', onPredictExport);
    $('btnSave').addEventListener('click', onSaveModel);
    $('btnLoadModel').addEventListener('click', onLoadModel);
    $('btnReset').addEventListener('click', clearAll);
    $('thr').addEventListener('input', onThrChange);
    $('thrVal').textContent = threshold.toFixed(2);
    $('btnTrain').disabled = true; $('btnEval').disabled = true; $('btnPredict').disabled = true; $('btnSave').disabled = true;
  }

 // Replace your previous DOMContentLoaded listener with this:
window.__APP_WIRED__ = function () {
  // put your previous wire() body here or call wire()
  // e.g.:
  if (typeof wire === 'function') wire();
};



})();
