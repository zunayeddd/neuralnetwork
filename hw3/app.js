/* app.js
   UI wiring for MNIST CSV upload demo with two modes:
   1) Classifier (CNN softmax): Train/Evaluate/Test5 as usual with accuracy & confusion matrix.
   2) Denoiser (CNN Autoencoder): Train to reconstruct clean images from noisy inputs; Evaluate with MSE; Test5 shows noisy vs denoised outputs.

   NEW (Reproducibility & Verification):
   - Save Model (Download) now also downloads a verify JSON signature computed on a fixed test slice.
   - Load Model (From Files) + Verify Loaded (Compare): load model.json/weights.bin and an optional verify.json, then recompute signature and compare with tolerances.

   Notes:
   - All browser-only; file-based Save/Load (downloads & browserFiles).
   - Uses tfjs-vis for live fit charts and evaluation visualizations.
*/

(() => {
  // ---- State
  let model = null; // Either classifier or autoencoder depending on mode
  let fullTrainXs = null, fullTrainYs = null;
  let valXs = null, valYs = null;
  let testXs = null, testYs = null;

  const $ = (id) => document.getElementById(id);

  // ---- Logging helpers
  function setStatus(msg) { $('dataStatus').textContent = msg; }
  function logTrain(msg) {
    const el = $('trainLogs');
    el.textContent = (el.textContent && el.textContent !== '–') ? (el.textContent + '\n' + msg) : msg;
  }
  function clearLogs() { $('trainLogs').textContent = '–'; }
  function setOverallMetric(msg) { $('metricOverall').textContent = `Overall metric: ${msg}`; }
  function setModelInfo(text) { $('modelInfo').textContent = text; }

  function enableTrainingUI(enabled) {
    $('btnTrain').disabled = !enabled;
    $('btnEval').disabled = !enabled || !testXs;
    $('btnTest5').disabled = !enabled || !testXs;
    $('btnSave').disabled = !enabled || !model;
    $('btnVerify').disabled = !enabled || !model || !testXs;
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
    $('previewRow').innerHTML = '';
    setModelInfo('–');
    setStatus('Waiting for CSV uploads…');
  }

  // ---- Models
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

  /** Convolutional autoencoder for denoising */
  function buildAutoencoder() {
    const input = tf.input({ shape:[28,28,1] });
    let x = tf.layers.conv2d({filters:32, kernelSize:3, activation:'relu', padding:'same'}).apply(input);
    x = tf.layers.maxPooling2d({poolSize:2, padding:'same'}).apply(x);
    x = tf.layers.conv2d({filters:16, kernelSize:3, activation:'relu', padding:'same'}).apply(x);
    x = tf.layers.maxPooling2d({poolSize:2, padding:'same'}).apply(x);
    x = tf.layers.conv2d({filters:8, kernelSize:3, activation:'relu', padding:'same'}).apply(x); // latent

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

  // ---- Deterministic verification signatures (fixed first K samples)
  function makeClassifierSignature(m, xs, ys, K = 200) {
    const count = Math.min(K, xs.shape[0]);
    const sliceXs = tf.tidy(() => xs.slice([0,0,0,0], [count,28,28,1]));
    const sliceYs = tf.tidy(() => ys.slice([0,0], [count,10]));
    const preds = tf.tidy(() => m.predict(sliceXs).argMax(-1));
    const labels = tf.tidy(() => sliceYs.argMax(-1));
    const sig = { mode:'classifier', count, acc: 0, preds: [] };
    return (async () => {
      const p = await preds.data();
      const l = await labels.data();
      let correct = 0;
      for (let i=0;i<count;i++){ if (p[i]===l[i]) correct++; sig.preds.push(p[i]); }
      sig.acc = correct / count;
      tf.dispose([sliceXs, sliceYs, preds, labels]);
      return sig;
    })();
  }

  function makeDenoiseSignature(m, cleanXs, noiseStd, K = 200) {
    const count = Math.min(K, cleanXs.shape[0]);
    const sliceClean = tf.tidy(() => cleanXs.slice([0,0,0,0], [count,28,28,1]));
    const noisy = DL.addGaussianNoise(sliceClean, noiseStd);
    const recon = tf.tidy(() => m.predict(noisy));
    const mse = tf.tidy(() => recon.sub(sliceClean).square().mean());
    // simple checksum: sum of first 10,000 elements (or less) to reduce sensitivity but catch mismatches
    const checksum = tf.tidy(() => recon.flatten().slice([0],[Math.min(count*28*28, 10000)]).sum());
    return (async () => {
      const avgMSE = (await mse.data())[0];
      const sum = (await checksum.data())[0];
      tf.dispose([sliceClean, noisy, recon, mse, checksum]);
      return { mode:'denoiser', count, noiseStd, avgMSE, sum };
    })();
  }

  function downloadJson(obj, filename) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  // ---- UI handlers
  async function onLoadData() {
    try {
      setStatus('Reading CSV files…');
      clearLogs();
      setOverallMetric('–');
      $('previewRow').innerHTML = '';

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
      $('btnSave').disabled = true;
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
        // ---- Train Autoencoder: inputs=noisy, targets=clean
        model = buildAutoencoder();
        setModelInfo(getModelSummaryText(model));
        logTrain(`Training Denoiser • epochs=${epochs}, batchSize=${batchSize}, noiseStd=${noiseStd}`);

        const t0 = performance.now();
        const noisyTrain = DL.makeNoisyCopy(fullTrainXs, noiseStd); // new tensor
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
        // ---- Train Classifier
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

      $('btnSave').disabled = false;
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

        await tfvis.render.valuesDistribution(
          { name: 'Reconstruction Error (MSE) — Summary', tab: 'Evaluation' },
          { values: [avgMSE], series: ['avg MSE'] },
          { width: 360 }
        );
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
        const perClass = conf.map((row, i) => {
          const total = row.reduce((a,b)=>a+b,0) || 1;
          return { index:i, acc: row[i]/total };
        });
        await tfvis.render.barchart(
          { name: 'Per-class Accuracy', tab: 'Evaluation' },
          perClass.map(d => ({ name: String(d.index), value: d.acc })),
          { xLabel:'Class', yLabel:'Accuracy', yAxisDomain:[0,1] }
        );
      }
    } catch (err) {
      console.error(err);
      setStatus(`Evaluation error: ${err.message || err}`);
    } finally {
      enableTrainingUI(true);
    }
  }

  /** Test 5 Random: denoise mode stacks (noisy / denoised); classifier shows prediction vs gt */
  async function onTestFive() {
    if (!testXs || !testYs || !model) return;
    const denoise = $('denoiseMode').checked;
    const row = $('previewRow');
    row.innerHTML = '';
    try {
      if (denoise) {
        const noiseStd = Math.max(0, Math.min(1, Number($('noiseStd').value) || 0.3));
        const { noisyXs, cleanXs, ys } = DL.getRandomNoisyCleanPairBatch(testXs, testYs, 5, noiseStd);
        const recon = tf.tidy(() => model.predict(noisyXs));

        const noisy3d = noisyXs.squeeze();
        const recon3d = recon.squeeze();
        const gt = tf.tidy(() => ys.argMax(-1));
        const gtArr = Array.from(await gt.data());

        for (let i = 0; i < noisy3d.shape[0]; i++) {
          const tile = document.createElement('div');
          tile.className = 'thumb';

          const pair = document.createElement('div');
          pair.className = 'pair';

          const cNoisy = document.createElement('canvas');
          cNoisy.className = 'preview';
          await DL.draw28x28ToCanvas(noisy3d.slice([i,0,0],[1,28,28]), cNoisy, 4);

          const cDeno = document.createElement('canvas');
          cDeno.className = 'preview';
          await DL.draw28x28ToCanvas(recon3d.slice([i,0,0],[1,28,28]), cDeno, 4);

          pair.appendChild(cNoisy);
          pair.appendChild(cDeno);
          tile.appendChild(pair);

          const mseT = tf.tidy(() =>
            recon3d.slice([i,0,0],[1,28,28]).sub(cleanXs.squeeze().slice([i,0,0],[1,28,28])).square().mean()
          );
          const mseVal = (await mseT.data())[0];
          mseT.dispose();

          const cap = document.createElement('div');
          cap.className = 'pred';
          cap.textContent = `gt ${gtArr[i]} • MSE ${mseVal.toFixed(4)}`;
          tile.appendChild(cap);

          row.appendChild(tile);
        }

        tf.dispose([noisyXs, cleanXs, ys, recon, noisy3d, recon3d, gt]);
      } else {
        const { xs, ys } = DL.getRandomTestBatch(testXs, testYs, 5, null);
        const preds = tf.tidy(() => model.predict(xs).argMax(-1));
        const labs = tf.tidy(() => ys.argMax(-1));
        const p = await preds.data();
        const l = await labs.data();

        const xs3d = xs.squeeze();
        for (let i = 0; i < xs3d.shape[0]; i++) {
          const tile = document.createElement('div');
          tile.className = 'thumb';

          const c = document.createElement('canvas');
          c.className = 'preview';
          await DL.draw28x28ToCanvas(xs3d.slice([i,0,0],[1,28,28]), c, 4);

          const ok = p[i] === l[i];
          const tag = document.createElement('div');
          tag.className = 'pred ' + (ok ? 'good' : 'bad');
          tag.textContent = `pred ${p[i]} • gt ${l[i]}${ok ? '' : ' ✗'}`;

          tile.appendChild(c);
          tile.appendChild(tag);
          row.appendChild(tile);
        }
        tf.dispose([xs, ys, preds, labs, xs3d]);
      }
    } catch (err) {
      console.error(err);
      setStatus(`Preview error: ${err.message || err}`);
    }
  }

  // ---- Save, Load, and Verification
  async function onSaveDownload() {
    if (!model) return;
    try {
      setStatus('Saving model & verification signature to downloads…');
      await model.save('downloads://mnist-cnn'); // downloads model.json + weights.bin

      // Build a deterministic signature on a fixed slice (first 200 test samples)
      const denoise = $('denoiseMode').checked;
      const noiseStd = Math.max(0, Math.min(1, Number($('noiseStd').value) || 0.3));
      let signature;
      if (denoise) {
        signature = await makeDenoiseSignature(model, testXs, noiseStd, 200);
      } else {
        signature = await makeClassifierSignature(model, testXs, testYs, 200);
      }
      signature.timestamp = new Date().toISOString();
      signature.meta = {
        mode: denoise ? 'denoiser' : 'classifier',
        noiseStd: denoise ? noiseStd : null,
        note: 'Deterministic verification computed on the first 200 test samples'
      };
      downloadJson(signature, 'mnist-cnn-verify.json');

      setStatus('Model and verify JSON saved (check your downloads).');
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
      $('btnSave').disabled = false;
      enableTrainingUI(true);
      setStatus('Model loaded from files. Optionally pick verify.json and click "Verify Loaded (Compare)".');
    } catch (err) {
      console.error(err);
      setStatus(`Model load error: ${err.message || err}`);
    }
  }

  async function onVerify() {
    try {
      if (!model || !testXs) {
        setStatus('Load model and data first, then choose verify.json.');
        return;
      }
      const vf = $('verifyJson').files?.[0];
      if (!vf) {
        setStatus('Please choose a verify.json file to compare.');
        return;
      }
      const verifyObj = await (async () => {
        const txt = await vf.text();
        return JSON.parse(txt);
      })();

      const tolAcc = 0.005; // 0.5% tolerance for accuracy
      const tolMSE = 1e-3;  // MSE tolerance
      const tolSum = 1e-2;  // checksum tolerance

      setStatus('Computing verification signature on current model…');
      let current;
      if (verifyObj.mode === 'classifier' || verifyObj?.meta?.mode === 'classifier') {
        current = await makeClassifierSignature(model, testXs, testYs, verifyObj.count || 200);
        const accDiff = Math.abs(current.acc - verifyObj.acc);
        const predsMatch = Array.isArray(verifyObj.preds)
          ? (verifyObj.preds.length === current.preds.length &&
             verifyObj.preds.every((v, i) => v === current.preds[i]))
          : true; // if preds not present, skip strict check
        const ok = accDiff <= tolAcc && predsMatch;
        setOverallMetric(`Verify (classifier): ${ok ? 'OK ✅' : 'MISMATCH ❌'} • acc Δ=${accDiff.toFixed(4)}`);
        setStatus(ok ? 'Verification passed.' : 'Verification mismatch — model predictions differ.');
      } else {
        // denoiser
        const noiseStd = verifyObj.noiseStd ?? verifyObj?.meta?.noiseStd ?? 0.3;
        current = await makeDenoiseSignature(model, testXs, noiseStd, verifyObj.count || 200);
        const mseDiff = Math.abs(current.avgMSE - verifyObj.avgMSE);
        const sumDiff = Math.abs(current.sum - verifyObj.sum);
        const ok = mseDiff <= tolMSE && sumDiff <= tolSum;
        setOverallMetric(`Verify (denoiser): ${ok ? 'OK ✅' : 'MISMATCH ❌'} • mse Δ=${mseDiff.toFixed(6)} • sum Δ=${sumDiff.toFixed(4)}`);
        setStatus(ok ? 'Verification passed.' : 'Verification mismatch — reconstructions differ.');
      }
    } catch (err) {
      console.error(err);
      setStatus(`Verify error: ${err.message || err}`);
    }
  }

  function onToggleVisor() {
    tfvis.visor().toggle();
  }

  // ---- Wire events
  window.addEventListener('DOMContentLoaded', () => {
    $('btnLoad').addEventListener('click', onLoadData);
    $('btnTrain').addEventListener('click', onTrain);
    $('btnEval').addEventListener('click', onEvaluate);
    $('btnTest5').addEventListener('click', onTestFive);
    $('btnSave').addEventListener('click', onSaveDownload);
    $('btnLoadModel').addEventListener('click', onLoadFromFiles);
    $('btnVerify').addEventListener('click', onVerify);
    $('btnReset').addEventListener('click', onReset);
    $('btnVisor').addEventListener('click', onToggleVisor);
  });
})();
