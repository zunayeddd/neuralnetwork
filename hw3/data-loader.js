
*/

const DL = (() => {
  const IMAGE_SIZE = 28 * 28;
  const NUM_CLASSES = 10;

  /** Parse a CSV text buffer into {xs, ys} tensors.
   *  xs: [N, 28, 28, 1] float32 in [0,1]
   *  ys: [N, 10] one-hot float32
   */
  function parseCsvTextToTensors(text) {
    // Split lines; filter empty.
    const lines = text.split(/\r?\n/).filter(line => line.trim().length > 0);
    const N = lines.length;
    if (N === 0) {
      throw new Error('CSV appears empty after filtering blank lines.');
    }

    // Preallocate numeric buffers: pixels normalized [0,1], labels int (0..9).
    const px = new Float32Array(N * IMAGE_SIZE);
    const labels = new Int32Array(N);

    // Parse rows (robustly ignore extra commas/spaces).
    let p = 0;
    for (let i = 0; i < N; i++) {
      const parts = lines[i].split(/[,;\s]+/).filter(x => x !== '');
      if (parts.length !== 1 + IMAGE_SIZE) {
        throw new Error(`Row ${i} has ${parts.length} fields (expected ${1 + IMAGE_SIZE}).`);
      }
      const lbl = parseInt(parts[0], 10);
      if (Number.isNaN(lbl) || lbl < 0 || lbl >= NUM_CLASSES) {
        throw new Error(`Row ${i} has invalid label: ${parts[0]}`);
      }
      labels[i] = lbl;
      for (let j = 0; j < IMAGE_SIZE; j++, p++) {
        const v = Number(parts[1 + j]);
        // Normalize 0..255 to 0..1; guard NaN.
        px[p] = Number.isFinite(v) ? (v / 255) : 0;
      }
    }

    // Create tensors: xs [N,28,28,1]; ys one-hot [N,10].
    const xs = tf.tensor4d(px, [N, 28, 28, 1], 'float32');
    const ys = tf.tidy(() => tf.oneHot(tf.tensor1d(labels, 'int32'), NUM_CLASSES).toFloat());

    return { xs, ys };
  }

  /** Read a File object as text. */
  function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onerror = () => reject(fr.error || new Error('File read failed.'));
      fr.onload = () => resolve(fr.result);
      fr.readAsText(file);
    });
  }

  /** Public: Load TRAIN CSV -> tensors. */
  async function loadTrainFromFiles(file) {
    if (!file) throw new Error('No train CSV file provided.');
    const text = await readFileAsText(file);
    return parseCsvTextToTensors(text);
  }

  /** Public: Load TEST CSV -> tensors. */
  async function loadTestFromFiles(file) {
    if (!file) throw new Error('No test CSV file provided.');
    const text = await readFileAsText(file);
    return parseCsvTextToTensors(text);
  }

  /** Split train tensors into train/val sets by ratio (default 90/10).
   *  Returns tensors: {trainXs, trainYs, valXs, valYs}
   *  – Performs a shuffle beforehand to randomize split.
   */
  function splitTrainVal(xs, ys, valRatio = 0.1) {
    const N = xs.shape[0];
    const idx = Array.from({ length: N }, (_, i) => i);
    tf.util.shuffle(idx);
    const valCount = Math.max(1, Math.floor(N * valRatio));
    const trainCount = N - valCount;

    // Helper: gather rows from xs and ys based on index list.
    const gather = (sourceXs, sourceYs, indices) => tf.tidy(() => {
      const k = indices.length;
      const xsBuf = new Float32Array(k * IMAGE_SIZE);
      const ysBuf = new Float32Array(k * 10);

      // Pull to CPU once, then copy.
      const xsData = sourceXs.dataSync(); // [N,28,28,1]
      const ysData = sourceYs.dataSync(); // [N,10]
      let px = 0, py = 0;
      for (const id of indices) {
        xsBuf.set(xsData.subarray(id * IMAGE_SIZE, (id + 1) * IMAGE_SIZE), px);
        px += IMAGE_SIZE;
        ysBuf.set(ysData.subarray(id * 10, (id + 1) * 10), py);
        py += 10;
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

  /** Gaussian noise utility:
   *  Returns a NEW tensor = clip(xs + N(0,std), 0, 1).
   *  – Keeps original xs unchanged.
   *  – Uses tf.tidy to avoid leaks.
   */
  function addGaussianNoise(xs, std = 0.1) {
    return tf.tidy(() => {
      const noise = tf.randomNormal(xs.shape, 0, std, 'float32');
      // Clip to valid image range [0,1]
      return xs.add(noise).clipByValue(0, 1);
    });
  }

  /** Make a full noisy copy of a dataset (e.g., to evaluate noise robustness). */
  function makeNoisyCopy(xs, std = 0.1) {
    // Simply return addGaussianNoise(xs); caller should dispose the returned tensor when done.
    return addGaussianNoise(xs, std);
  }

  /** Get a random test batch (for preview).
   *  Returns {xs, ys} tensors of shapes [k,28,28,1] and [k,10].
   *  New optional param noiseStd: if provided (e.g., 0.15), additive Gaussian noise is applied.
   */
  function getRandomTestBatch(xs, ys, k = 5, noiseStd = null) {
    const N = xs.shape[0];
    const sel = new Set();
    while (sel.size < Math.min(k, N)) {
      sel.add(Math.floor(Math.random() * N));
    }
    const indices = Array.from(sel);
    return tf.tidy(() => {
      const xsData = xs.dataSync();
      const ysData = ys.dataSync();
      const xsBuf = new Float32Array(indices.length * IMAGE_SIZE);
      const ysBuf = new Float32Array(indices.length * 10);
      let px = 0, py = 0;
      for (const id of indices) {
        xsBuf.set(xsData.subarray(id * IMAGE_SIZE, (id + 1) * IMAGE_SIZE), px); px += IMAGE_SIZE;
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

  /** Draw a single [28,28,1] grayscale image tensor to a canvas (scaled by `scale`). */
  async function draw28x28ToCanvas(tensor, canvas, scale = 4) {
    canvas.width = 28;
    canvas.height = 28;
    const img2d = tf.tidy(() => tensor.squeeze()); // [28,28]
    await tf.browser.toPixels(img2d, canvas);
    img2d.dispose();
    canvas.style.width = `${28 * scale}px`;
    canvas.style.height = `${28 * scale}px`;
  }

  return {
    loadTrainFromFiles,
    loadTestFromFiles,
    splitTrainVal,
    getRandomTestBatch,  // now supports optional noiseStd
    draw28x28ToCanvas,
    addGaussianNoise,    // exported for app-level evaluation with noise
    makeNoisyCopy,       // exported convenience helper
  };
})();
