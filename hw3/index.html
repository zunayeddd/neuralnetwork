<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MNIST tfjs — CSV Upload Demo (Classifier & Denoiser)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- CDNs (latest as requested) -->
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest"></script>
  <style>
    :root {
      --bg:#0b1020; --panel:#121933; --ink:#eaf0ff; --muted:#a7b0d7;
      --border:#223162; --btn:#1b2750; --btn2:#2a47b8;
      --good:#9cfcc3; --bad:#ff9ba0;
    }
    html, body { margin:0; height:100%; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Arial }
    .wrap { max-width:1200px; margin:24px auto; padding:0 16px }
    h1 { font-size:22px; margin:0 0 8px }
    .muted { color:var(--muted) }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:18px; align-items:start }
    .card { background:linear-gradient(180deg,#111833,#0d142b); border:1px solid var(--border); border-radius:14px; box-shadow:0 8px 30px rgba(0,0,0,.25) inset, 0 8px 24px rgba(0,0,0,.2) }
    .card .hd { padding:12px 14px; border-bottom:1px solid var(--border); font-weight:700 }
    .card .bd { padding:14px }
    .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center }
    .stack { display:grid; gap:10px }
    label { display:flex; align-items:center; gap:8px; background:#0f1735; border:1px solid var(--border); padding:8px 10px; border-radius:12px; font-size:13px }
    input[type="file"] { inline-size:230px; color:var(--ink) }
    input[type="number"] { width:90px; background:#0f1735; border:1px solid var(--border); color:var(--ink); padding:6px 8px; border-radius:10px }
    input[type="checkbox"] { transform: scale(1.1); }
    button { appearance:none; background:var(--btn); color:var(--ink); border:1px solid var(--border); padding:10px 14px; border-radius:12px; font-weight:700; cursor:pointer }
    button.primary { background:linear-gradient(180deg,var(--btn2),#1b2e87); border-color:#3553a1 }
    button:disabled { opacity:.5; cursor:not-allowed }
    .two-col { display:grid; grid-template-columns: 1fr 1fr; gap:12px }
    .status { font-variant-numeric: tabular-nums }
    #previewRow { display:grid; grid-template-columns: repeat(5, auto); gap:12px; align-items:start; justify-content:flex-start }
    .thumb { display:grid; justify-items:center; gap:6px }
    .pair { display:grid; gap:6px } /* stack noisy + denoised */
    canvas.preview { image-rendering: pixelated; border-radius:10px; border:1px solid var(--border); background:#0a0f25; width:112px; height:112px } /* 28*4 */
    .pred { font-weight:700; padding:4px 8px; border-radius:999px; border:1px solid var(--border) }
    .pred.good { color:#064; border-color:#194 }
    .pred.bad { color:#600; border-color:#944 }
    pre { white-space:pre-wrap; word-break:break-word; margin:0; font-size:13px; color:#d8defa }
    .small { font-size:12px }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>MNIST (TensorFlow.js) — CSV Upload • Classifier & Denoiser (Autoencoder)</h1>
    <div class="muted small">Upload <b>mnist_train.csv</b> and <b>mnist_test.csv</b> (label, 784 pixels, no header). No network fetch. Trains and runs fully client-side.</div>

    <div class="grid" style="margin-top:14px">
      <!-- LEFT: Controls -->
      <div class="card">
        <div class="hd">Controls</div>
        <div class="bd stack">
          <div class="stack">
            <div class="row">
              <label>Upload Train CSV <input id="trainCsv" type="file" accept=".csv,text/csv" /></label>
              <label>Upload Test CSV <input id="testCsv" type="file" accept=".csv,text/csv" /></label>
            </div>
            <div class="row">
              <button id="btnLoad" class="primary" type="button">Load Data</button>
              <button id="btnTrain" type="button" disabled>Train</button>
              <button id="btnEval" type="button" disabled>Evaluate</button>
              <button id="btnTest5" type="button" disabled>Test 5 Random</button>
            </div>
          </div>

          <div class="stack">
            <div class="two-col">
              <label>Epochs <input id="epochs" type="number" min="1" max="30" value="8"></label>
              <label>Batch Size <input id="batch" type="number" min="16" max="512" value="128"></label>
            </div>
            <div class="two-col">
              <label><input id="denoiseMode" type="checkbox"> Denoising Mode (Autoencoder)</label>
              <label>Noise Std (denoise) <input id="noiseStd" type="number" step="0.01" min="0" max="1" value="0.30"></label>
            </div>
          </div>

          <div class="stack">
            <div class="row">
              <button id="btnSave" type="button" disabled>Save Model (Download)</button>
              <button id="btnLoadModel" type="button">Load Model (From Files)</button>
              <button id="btnVerify" type="button" disabled>Verify Loaded (Compare)</button>
              <button id="btnReset" type="button">Reset</button>
              <button id="btnVisor" type="button">Toggle Visor</button>
            </div>
            <div class="row">
              <label>model.json <input id="mdlJson" type="file" accept=".json,application/json" /></label>
              <label>weights.bin <input id="mdlBin" type="file" accept=".bin,application/octet-stream" /></label>
              <label>verify.json <input id="verifyJson" type="file" accept=".json,application/json" /></label>
            </div>
          </div>
        </div>
      </div>

      <!-- RIGHT: Status / Logs -->
      <div class="card">
        <div class="hd">Data Status</div>
        <div class="bd">
          <div id="dataStatus" class="status">Waiting for CSV uploads…</div>
        </div>
      </div>
    </div>

    <div class="grid" style="margin-top:14px">
      <div class="card">
        <div class="hd">Training Logs</div>
        <div class="bd">
          <div id="trainLogs" class="status small">–</div>
        </div>
      </div>
      <div class="card">
        <div class="hd">Metrics</div>
        <div class="bd stack">
          <div id="metricOverall" class="status">Overall metric: –</div>
          <div class="row small muted">Live charts appear in the tfjs-vis Visor (click “Toggle Visor”).</div>
        </div>
      </div>
    </div>

    <div class="card" style="margin-top:14px">
      <div class="hd">Random 5 Preview</div>
      <div class="bd">
        <div id="previewRow"></div>
      </div>
    </div>

    <div class="card" style="margin-top:14px; margin-bottom:30px">
      <div class="hd">Model Info</div>
      <div class="bd">
        <pre id="modelInfo">–</pre>
      </div>
    </div>
  </div>

  <!-- IMPORTANT: Keep this order. Use 'defer' so both scripts run after parsing, preserving order. -->
  <script defer src="data-loader.js"></script>
  <script defer src="app.js"></script>
</body>
</html>
