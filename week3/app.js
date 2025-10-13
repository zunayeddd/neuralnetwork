/* app.js
   MNIST CSV upload demo with:
   - Classifier (CNN softmax)
   - Denoiser (CNN Autoencoder)
   - tfjs-vis charts, file-based Save/Load
   - "Test 5 Random" preview (classifier or denoiser mode)
   FIX: Avoid rank-mismatch in slicing by always slicing 4D tensors; no premature squeeze.
*/
(() => {
  // ---------- STATE & DOM ----------
  let model = null;
  let fullTrainXs = null, fullTrainYs = null;
  let valXs = null, valYs = null;
  let testXs = null, testYs = null;

  const $ = (id) => document.getElementById(id);
  const setStatus = (msg) => { const el=$('dataStatus'); if (el) el.textContent = msg; };
  const logTrain = (msg) => {
    const el = $('trainLogs'); if (!el) return;
    el.textContent = (el.textContent && el.textContent !== '–') ? (el.textContent + '\n' + msg) : msg;
  };
  const clearLogs = () => { const el=$('trainLogs'); if (el) el.textContent = '–'; };
  const setOverallMetric = (msg) => { const el=$('metricOverall'); if (el) el.textContent = `Overall metric: ${msg}`; };
  const setModelInfo = (text) => { const el=$('modelInfo'); if (el) el.textContent = text; };

  function enableTrainingUI(enabled) {
    if ($('btnTrain')) $('btnTrain').disabled = !enabled;
    if ($('btnEval')) $('btnEval').disabled = !enabled || !testXs;
    if ($('btnTest5')) $('btnTest5').disabled = !enabled || !testXs;
    if ($('btnSave')) $('btnSave').disabled = !enabled || !model;
    if ($('btnVerify')) $('btnVerify').disabled = !enabled || !model || !testXs;
  }

  function disposeAllTensors() {
    try {
      fullTrainXs?.dispose(); fullTrainYs?.dispose();
      valXs?.dispose(); valYs?.dispose();
      testXs?.dispose(); testYs?.dispose();
    } catch {}
    fullTrainXs = fullTrainYs = valXs = valYs = testXs = testYs = null;
  }

  async function onReset() {
    enableTrainingUI(false);
    setStatus('Resetting…');
    if (model) { model.dispose(); model = null; }
    disposeAllTensors();
    clearLogs();
    setOverallMetric('–');
    if ($('previewRow')) $('previewRow').innerHTML = '';
    setModelInfo('–');
    setStatus('Waiting for CSV uploads…');
  }

  // ---------- MODELS ----------
  function buildClassifierModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', padding:'same', inputShape:[28,28,1]}));
    m.add(tf.layers.conv2d({filters:64, kernelSize:3, activation:'relu', padding:'same'}));
    m.add(tf.layers.maxPooling2d({poolSize:2}));
    m.add(tf.layers.dropout({rate:0.25}));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({units:128, activation:'relu'}));
    m.add(tf.layers.dropout({rate:0.5}));
    m.add(tf.layers.dense({units:10, activation:'softmax'}));
    m.compile({optimizer:'adam', loss:'categoricalCrossentropy', metrics:['accuracy']});
    return m;
  }

  function buildAutoencoder() {
    const input = tf.input({ shape:[28,28,1] });
    let x = tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', padding:'same'}).apply(input);
    x = tf.layers.maxPooling2d({poolSize:2, padding:'same'}).apply(x);
    x = tf.layers.conv2d({filters:16, kernelSize:3, activation:'relu', padding:'same'}).apply(x);
    x = tf.layers.maxPooling2d({poolSize:2, padding:'same'}).apply(x);
    x = tf.layers.conv2d({filters:8, kernelSize:3, activation:'relu', padding:'same'}).apply(x); // 7x7x8 latent

    x = tf.layers.upSampling2d({size:[2,2]}).apply(x);
    x = tf.layers.conv2d({filters:16, kernelSize:3, activation:'relu', padding:'same'}).apply(x);
    x = tf.layers.upSampling2d({size:[2,2]}).apply(x);
    const decoded = tf.layers.conv2d({filters:1, kernelSize:3, activation:'sigmoid', padding:'same'}).apply(x);

    const m = tf.model({ inputs: input, outputs: decoded });
    m.compile({ optimizer:'adam', loss:'meanSquaredError' });
    return m;
  }

  function getModelSummaryText(m) {
    const lines = [];
    lines.push('Model Summary');
    lines.push('=====================================================');
    let totalParams = 0, trainableParams = 0, nonTrainableParams = 0;
    for (const layer of m.layers) {
      const p = layer.countParams();
      totalParams += p;
      trainableParams += (layer.trainable ? p : 0);
      nonTrainableParams += (layer.trainable ? 0 : p);
      lines.push(`${layer.name.padEnd(24)} outputShape=${JSON.stringify(layer.outputShape)}  params=${p}`);
    }
    lines.push('-----------------------------------------------------');
    lines.push(`Total params: ${totalParams.toLocaleString()}`);
    lines.push(`Trainable params: ${trainableParams.toLocaleString()}`);
    lines.push(`Non-trainable params: ${nonTrainableParams.toLocaleString()}`);
    return lines.join('\n');
  }

  // ---------- UI HANDLERS ----------
  async function onLoadData() {
    try {
      setStatus('Reading CSV files…');
      clearLogs();
      setOverallMetric('–');
      if ($('previewRow')) $('previewRow').innerHTML = '';

      const trainFile = $('trainCsv').files?.[0];
      const testFile  = $('testCsv').files?.[0];
      if (!trainFile || !testFile) {
        setStatus('Please select both mnist_train.csv and mnist_test.csv.');
        return;
      }

      const t0 = performance.now();
      const train = await DL.loadTrainFromFiles(trainFile);
      const test  = await DL.loadTestFromFiles(testFile);

      const { trainXs, trainYs, valXs: vx, valYs: vy } = DL.splitTrainVal(train.xs, train.ys, 0.1);
      train.xs.dispose(); train.ys.dispose();

      disposeAllTensors();
      fullTrainXs = trainXs; fullTrainYs = trainYs;
      valXs = vx; valYs = vy;
      testXs = test.xs; testYs = test.ys;

      const t1 = performance.now();
      setStatus(`Loaded: train ${fullTrainXs.shape[0]} / val ${valXs.shape[0]} / test ${testXs.shape[0]}  • ${(t1 - t0).toFixed(0)} ms`);
      enableTrainingUI(true);
    } catch (err) {
      console.error(err);
      setStatus(`Error loading data: ${err.message || err}`);
    }
  }

  async function onTrain() {
    if (!fullTrainXs || !fullTrainYs) return;
    try {
      enableTrainingUI(false);
      if ($('btnSave')) $('btnSave').disabled = true;
      clearLogs();

      if (model) { model.dispose(); model = null; }

      const denoise = $('denoiseMode').checked;
      const epochs = Math.max(1, Math.min(30, parseInt($('epochs').value, 10) || 8));
      const batchSize = Math.max(16, Math.min(512, parseInt($('batch').value, 10) || 128));
      const noiseStd = Math.max(0, Math.min(1, Number($('noiseStd').value) || 0.3));

      const surface = { name: denoise ? 'Denoiser Training' : 'Classifier Training', tab: 'Training' };
      const metrics = denoise
        ? ['loss', 'val_loss']
        : ['loss', 'val_loss', 'acc', 'val_acc', 'accuracy', 'val_accuracy'];
      const fitCallbacks = tfvis.show.fitCallbacks(surface, metrics, { callbacks: ['onEpochEnd'] });

      if (denoise) {
        model = buildAutoencoder();
        setModelInfo(getModelSummaryText(model));
        logTrain(`Training Denoiser • epochs=${epochs}, batchSize=${batchSize}, noiseStd=${noiseStd}`);

        const t0 = performance.now();
        const noisyTrain = DL.makeNoisyCopy(fullTrainXs, noiseStd);
        const history = await model.fit(noisyTrain, fullTrainXs, {
          epochs, batchSize, shuffle: true,
          validationData: valXs ? [DL.makeNoisyCopy(valXs, noiseStd), valXs] : null,
          callbacks: {
            ...fitCallbacks,
            onEpochEnd: async (epoch, logs) => {
              logTrain(`Epoch ${epoch + 1}/${epochs} — denoise loss=${logs.loss.toFixed(5)}  ` +
                       (logs.val_loss != null ? `val_loss=${logs.val_loss.toFixed(5)}` : ''));
              await tf.nextFrame();
            }
          }
        });
        noisyTrain.dispose();
        const t1 = performance.now();
        const bestVal = Math.min(...(history.history.val_loss || [Number.POSITIVE_INFINITY]));
        logTrain(`Denoiser training complete • ${(t1 - t0).toFixed(0)} ms • best val mse=${(bestVal).toFixed(5)}`);
        setStatus('Denoiser ready. Use Evaluate or Test 5 Random to see reconstructions.');
        setOverallMetric('– (run Evaluate for MSE)');
      } else {
        model = buildClassifierModel();
        setModelInfo(getModelSummaryText(model));
        logTrain(`Training Classifier • epochs=${epochs}, batchSize=${batchSize}`);

        const t0 = performance.now();
        const history = await model.fit(fullTrainXs, fullTrainYs, {
          epochs, batchSize, shuffle: true,
          validationData: valXs && valYs ? [valXs, valYs] : null,
          callbacks: {
            ...fitCallbacks,
            onEpochEnd: async (epoch, logs) => {
              const trAcc = logs.acc ?? logs.accuracy;
              const vaAcc = logs.val_acc ?? logs.val_accuracy;
              logTrain(`Epoch ${epoch + 1}/${epochs} — loss=${logs.loss.toFixed(4)}  acc=${(trAcc*100).toFixed(2)}%  ` +
                       (vaAcc != null ? `val_acc=${(vaAcc*100).toFixed(2)}%` : ''));
              await tf.nextFrame();
            }
          }
        });
        const t1 = performance.now();
        const bestValAcc = Math.max(...(history.history.val_acc || history.history.val_accuracy || [0]));
        logTrain(`Classifier training complete • ${(t1 - t0).toFixed(0)} ms • best val acc=${(bestValAcc*100 || 0).toFixed(2)}%`);
        setStatus('Classifier ready. Use Evaluate or Test 5 Random.');
        setOverallMetric('– (run Evaluate for accuracy)');
      }

      if ($('btnSave')) $('btnSave').disabled = false;
      enableTrainingUI(true);
    } catch (err) {
      console.error(err);
      setStatus(`Training error: ${err.message || err}`);
      enableTrainingUI(true);
    }
  }

  async function onEvaluate() {
    if (!model || !testXs || !testYs) return;
    try {
      enableTrainingUI(false);
      const denoise = $('denoiseMode').checked;
      const noiseStd = Math.max(0, Math.min(1, Number($('noiseStd').value) || 0.3));

      if (denoise) {
        setStatus('Evaluating denoiser (MSE on noisy→clean)…');
        const N = testXs.shape[0];
        const B = 256;
        let totalMSE = 0, count = 0;
        for (let i = 0; i < N; i += B) {
          const n = Math.min(B, N - i);
          const clean = tf.tidy(() => testXs.slice([i,0,0,0], [n,28,28,1]));
          const noisy = DL.addGaussianNoise(clean, noiseStd);
          const recon = tf.tidy(() => model.predict(noisy));
          const mse = tf.tidy(() => recon.sub(clean).square().mean());
          const val = (await mse.data())[0];
          totalMSE += val * n; count += n;
          tf.dispose([clean, noisy, recon, mse]);
          if (i % (B*2) === 0) await tf.nextFrame();
        }
        const avgMSE = totalMSE / count;
        setOverallMetric(`MSE ${(avgMSE).toFixed(5)} on ${count} samples (noiseStd=${noiseStd})`);
        setStatus('Denoiser evaluation complete.');
      } else {
        setStatus('Evaluating classifier (accuracy & confusion)…');
        const N = testXs.shape[0];
        const B = 256;
        const numClasses = 10;
        const conf = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
        let correct = 0;

        for (let i = 0; i < N; i += B) {
          const count = Math.min(B, N - i);
          const xs = tf.tidy(() => testXs.slice([i,0,0,0],[count,28,28,1]));
          const ys = tf.tidy(() => testYs.slice([i,0],[count,10]));
          const preds = tf.tidy(() => model.predict(xs).argMax(-1));
          const labels = tf.tidy(() => ys.argMax(-1));
          const p = await preds.data();
          const l = await labels.data();
          for (let k = 0; k < count; k++) {
            conf[l[k]][p[k]] += 1;
            if (p[k] === l[k]) correct++;
          }
          tf.dispose([xs, ys, preds, labels]);
          if (i % (B*4) === 0) await tf.nextFrame();
        }
        const acc = correct / N;
        setOverallMetric(`${(acc*100).toFixed(2)}% accuracy on ${N} samples`);
        setStatus('Classifier evaluation complete. See Visor for confusion matrix & per-class accuracy.');
        const labelsStr = [...Array(numClasses)].map((_, i) => String(i));
        await tfvis.render.confusionMatrix(
          { name: 'Confusion Matrix', tab: 'Evaluation' },
          { values: conf, tickLabels: labelsStr },
          { width: 480, height: 400 }
        );
      }
    } catch (err) {
      console.error(err);
      setStatus(`Evaluation error: ${err.message || err}`);
    } finally {
      enableTrainingUI(true);
    }
  }

  /** FIXED: Always slice as 4D [1,28,28,1] when drawing; do not squeeze before slicing.
      This avoids rank mismatches when batch size is 1. */
  async function onTestFive() {
    if (!testXs || !testYs || !model) return;
    const denoise = $('denoiseMode').checked;
    const row = $('previewRow');
    if (row) row.innerHTML = '';
    try {
      if (denoise) {
        const noiseStd = Math.max(0, Math.min(1, Number($('noiseStd').value) || 0.3));
        const { noisyXs, cleanXs, ys } = DL.getRandomNoisyCleanPairBatch(testXs, testYs, 5, noiseStd);
        const recon = tf.tidy(() => model.predict(noisyXs)); // [k,28,28,1]

        const gt = tf.tidy(() => ys.argMax(-1));
        const gtArr = Array.from(await gt.data());
        const k = noisyXs.shape[0];

        for (let i = 0; i < k; i++) {
          const tile = document.createElement('div');
          tile.className = 'thumb';

          const pair = document.createElement('div');
          pair.className = 'pair';

          // Slice 4D -> [1,28,28,1] for each sample
          const noisyOne = tf.tidy(() => noisyXs.slice([i,0,0,0],[1,28,28,1]));
          const reconOne = tf.tidy(() => recon.slice([i,0,0,0],[1,28,28,1]));
          const cleanOne = tf.tidy(() => cleanXs.slice([i,0,0,0],[1,28,28,1]));

          const cNoisy = document.createElement('canvas');
          cNoisy.className = 'preview';
          await DL.draw28x28ToCanvas(noisyOne, cNoisy, 4);

          const cDeno = document.createElement('canvas');
          cDeno.className = 'preview';
          await DL.draw28x28ToCanvas(reconOne, cDeno, 4);

          pair.appendChild(cNoisy);
          pair.appendChild(cDeno);
          tile.appendChild(pair);

          const mseVal = await tf.tidy(() => reconOne.sub(cleanOne).square().mean()).data();
          const cap = document.createElement('div');
          cap.className = 'pred';
          cap.textContent = `gt ${gtArr[i]} • MSE ${mseVal[0].toFixed(4)}`;
          tile.appendChild(cap);

          row.appendChild(tile);

          tf.dispose([noisyOne, reconOne, cleanOne]);
        }

        tf.dispose([noisyXs, cleanXs, ys, recon, gt]);
      } else {
        const { xs, ys } = DL.getRandomTestBatch(testXs, testYs, 5, null); // xs: [k,28,28,1]
        const preds = tf.tidy(() => model.predict(xs).argMax(-1));
        const labs = tf.tidy(() => ys.argMax(-1));
        const p = await preds.data();
        const l = await labs.data();
        const k = xs.shape[0];

        for (let i = 0; i < k; i++) {
          const c = document.createElement('canvas');
          c.className = 'preview';
          const one = tf.tidy(() => xs.slice([i,0,0,0],[1,28,28,1]));
          await DL.draw28x28ToCanvas(one, c, 4);

          const tile = document.createElement('div');
          tile.className = 'thumb';

          const ok = p[i] === l[i];
          const tag = document.createElement('div');
          tag.className = 'pred ' + (ok ? 'good' : 'bad');
          tag.textContent = `pred ${p[i]} • gt ${l[i]}${ok ? '' : ' ✗'}`;

          tile.appendChild(c);
          tile.appendChild(tag);
          row.appendChild(tile);

          one.dispose();
        }
        tf.dispose([xs, ys, preds, labs]);
      }
    } catch (err) {
      console.error(err);
      setStatus(`Preview error: ${err.message || err}`);
    }
  }

  async function onSaveDownload() {
    if (!model) return;
    try {
      setStatus('Saving model to downloads…');
      await model.save('downloads://mnist-cnn'); // downloads model.json + weights.bin
      setStatus('Model saved (check your downloads).');
    } catch (err) {
      console.error(err);
      setStatus(`Save error: ${err.message || err}`);
    }
  }

  async function onLoadFromFiles() {
    try {
      const jf = $('mdlJson').files?.[0];
      const bf = $('mdlBin').files?.[0];
      if (!jf || !bf) {
        setStatus('Select both model.json and weights.bin to load a model.');
        return;
      }
      setStatus('Loading model from files…');
      const m = await tf.loadLayersModel(tf.io.browserFiles([jf, bf]));
      if (model) model.dispose();
      model = m;
      setModelInfo(getModelSummaryText(model));
      if ($('btnSave')) $('btnSave').disabled = false;
      enableTrainingUI(true);
      setStatus('Model loaded from files.');
    } catch (err) {
      console.error(err);
      setStatus(`Model load error: ${err.message || err}`);
    }
  }

  function onToggleVisor() { tfvis.visor().toggle(); }

  // ---------- BOOTSTRAP ----------
  window.addEventListener('DOMContentLoaded', () => {
    // Wire events (DL is global from data-loader.js)
    $('btnLoad').addEventListener('click', onLoadData);
    $('btnTrain').addEventListener('click', onTrain);
    $('btnEval').addEventListener('click', onEvaluate);
    $('btnTest5').addEventListener('click', onTestFive);
    $('btnSave').addEventListener('click', onSaveDownload);
    $('btnLoadModel').addEventListener('click', onLoadFromFiles);
    $('btnReset').addEventListener('click', onReset);
    $('btnVisor').addEventListener('click', onToggleVisor);
  });
})();
