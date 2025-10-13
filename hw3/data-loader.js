/* data-loader.js
   File-based CSV parsing utilities for MNIST in the browser.
   - No network fetch. Users provide `mnist_train.csv` and `mnist_test.csv`.
   - CSV format: each row = label (0–9), then 784 pixel values (0–255), no header.
   - We normalize pixels to [0,1], reshape to [N,28,28,1], and one-hot labels (depth 10).
   - Exposes a global namespace: window.DL
*/
(function () {
  const IMAGE_SIZE = 28 * 28;
  const NUM_CLASSES = 10;

  function parseCsvTextToTensors(text) {
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    const N = lines.length;
    if (N === 0) throw new Error('CSV appears empty after filtering blank lines.');
    const px = new Float32Array(N * IMAGE_SIZE);
    const labels = new Int32Array(N);
    let p = 0;
    for (let i = 0; i < N; i++) {
      const parts = lines[i].split(/[,;\s]+/).filter(x => x !== '');
      if (parts.length !== 1 + IMAGE_SIZE) {
        throw new Error(`Row ${i} has ${parts.length} fields (expected ${1 + IMAGE_SIZE}).`);
      }
      const lbl = parseInt(parts[0], 10);
      if (!Number.isFinite(lbl) || lbl < 0 || lbl >= NUM_CLASSES) {
        throw new Error(`Row ${i} has invalid label: ${parts[0]}`);
      }
      labels[i] = lbl;
      for (let j = 0; j < IMAGE_SIZE; j++, p++) {
        const v = Number(parts[1 + j]);
        px[p] = Number.isFinite(v) ? (v / 255) : 0;
      }
    }
    const xs = tf.tensor4d(px, [N, 28, 28, 1], 'float32');
    const ys = tf.tidy(() => tf.oneHot(tf.tensor1d(labels, 'int32'), NUM_CLASSES).toFloat());
    return { xs, ys };
  }

  function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onerror = () => reject(fr.error || new Error('File read failed.'));
      fr.onload = () => resolve(fr.result);
      fr.readAsText(file);
    });
  }

  async function loadTrainFromFiles(file) {
    if (!file) throw new Error('No train CSV file provided.');
    const text = await readFileAsText(file);
    return parseCsvTextToTensors(text);
  }

  async function loadTestFromFiles(file) {
    if (!file) throw new Error('No test CSV file provided.');
    const text = await readFileAsText(file);
    return parseCsvTextToTensors(text);
  }

  function splitTrainVal(xs, ys, valRatio = 0.1) {
    const N = xs.shape[0];
    const idx = Array.from({ length: N }, (_, i) => i);
    tf.util.shuffle(idx);
    const valCount = Math.max(1, Math.floor(N * valRatio));
    const trainCount = N - valCount;

    const gather = (sourceXs, sourceYs, indices) => tf.tidy(() => {
      const k = indices.length;
      const xsBuf = new Float32Array(k * IMAGE_SIZE);
      const ysBuf = new Float32Array(k * 10);
      const xsData = sourceXs.dataSync();
      const ysData = sourceYs.dataSync();
      let px = 0, py = 0;
      for (const id of indices) {
        xsBuf.set(xsData.subarray(id * IMAGE_SIZE, (id + 1) * IMAGE_SIZE), px); px += IMAGE_SIZE;
        ysBuf.set(ysData.subarray(id * 10, (id + 1) * 10), py); py += 10;
      }
      const gx = tf.tensor4d(xsBuf, [k, 28, 28, 1], 'float32');
      const gy = tf.tensor2d(ysBuf, [k, 10], 'float32');
      return { gx, gy };
    });

    const trainIdx = idx.slice(0, trainCount);
    const valIdx = idx.slice(trainCount);

    const { gx: trainXs, gy: trainYs } = gather(xs, ys, trainIdx);
    const { gx: valXs, gy: valYs } = gather(xs, ys, valIdx);

    return { trainXs, trainYs, valXs, valYs };
  }

  function addGaussianNoise(xs, std = 0.1) {
    return tf.tidy(() => xs.add(tf.randomNormal(xs.shape, 0, std, 'float32')).clipByValue(0, 1));
  }

  function makeNoisyCopy(xs, std = 0.1) {
    return addGaussianNoise(xs, std);
  }

  function getRandomTestBatch(xs, ys, k = 5, noiseStd = null) {
    const IMAGE_SIZE_FLAT = 28 * 28;
    const N = xs.shape[0];
    const sel = new Set();
    while (sel.size < Math.min(k, N)) sel.add(Math.floor(Math.random() * N));
    const indices = Array.from(sel);
    return tf.tidy(() => {
      const xsData = xs.dataSync();
      const ysData = ys.dataSync();
      const xsBuf = new Float32Array(indices.length * IMAGE_SIZE_FLAT);
      const ysBuf = new Float32Array(indices.length * 10);
      let px = 0, py = 0;
      for (const id of indices) {
        xsBuf.set(xsData.subarray(id * IMAGE_SIZE_FLAT, (id + 1) * IMAGE_SIZE_FLAT), px); px += IMAGE_SIZE_FLAT;
        ysBuf.set(ysData.subarray(id * 10, (id + 1) * 10), py); py += 10;
      }
      let batchXs = tf.tensor4d(xsBuf, [indices.length, 28, 28, 1], 'float32');
      if (typeof noiseStd === 'number' && Number.isFinite(noiseStd) && noiseStd > 0) {
        const noisy = addGaussianNoise(batchXs, noiseStd);
        batchXs.dispose();
        batchXs = noisy;
      }
      const batchYs = tf.tensor2d(ysBuf, [indices.length, 10], 'float32');
      return { xs: batchXs, ys: batchYs };
    });
  }

  function getRandomNoisyCleanPairBatch(xs, ys, k = 5, noiseStd = 0.3) {
    const IMAGE_SIZE_FLAT = 28 * 28;
    const N = xs.shape[0];
    const sel = new Set();
    while (sel.size < Math.min(k, N)) sel.add(Math.floor(Math.random() * N));
    const indices = Array.from(sel);
    return tf.tidy(() => {
      const xsData = xs.dataSync();
      const ysData = ys.dataSync();
      const cleanBuf = new Float32Array(indices.length * IMAGE_SIZE_FLAT);
      const ysBuf = new Float32Array(indices.length * 10);
      let px = 0, py = 0;
      for (const id of indices) {
        cleanBuf.set(xsData.subarray(id * IMAGE_SIZE_FLAT, (id + 1) * IMAGE_SIZE_FLAT), px); px += IMAGE_SIZE_FLAT;
        ysBuf.set(ysData.subarray(id * 10, (id + 1) * 10), py); py += 10;
      }
      const cleanXs = tf.tensor4d(cleanBuf, [indices.length, 28, 28, 1], 'float32');
      const noisyXs = addGaussianNoise(cleanXs, noiseStd);
      const batchYs = tf.tensor2d(ysBuf, [indices.length, 10], 'float32');
      return { noisyXs, cleanXs, ys: batchYs };
    });
  }

  async function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    canvas.width = 28;
    canvas.height = 28;
    const img2d = tf.tidy(() => tensor.squeeze());
    await tf.browser.toPixels(img2d, canvas);
    img2d.dispose();
    canvas.style.width = `${28 * scale}px`;
    canvas.style.height = `${28 * scale}px`;
  }

  // Explicitly attach to window to avoid "DL is not defined"
  window.DL = {
    loadTrainFromFiles,
    loadTestFromFiles,
    splitTrainVal,
    getRandomTestBatch,
    getRandomNoisyCleanPairBatch,
    draw28x28ToCanvas,
    addGaussianNoise,
    makeNoisyCopy,
  };
})();
