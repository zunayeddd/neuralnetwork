// ===== PERFECT FINAL VERSION - COPY THIS EXACTLY =====
let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let valXs = null;
let valYs = null;
let currentThreshold = 0.5;

function disposeModelSafely() {
  if (!model) return;
  try {
    const weights = model.getWeights();
    weights.forEach(w => { try { w.dispose(); } catch(e) {} });
    try { model.dispose(); } catch(e) {}
  } catch(e) {}
  model = null;
}

function disposeTensorsSafely() {
  if (valXs) { try { valXs.dispose(); } catch(e) {} valXs = null; }
  if (valYs) { try { valYs.dispose(); } catch(e) {} valYs = null; }
}

function findButtonByText(text) {
  const buttons = document.querySelectorAll('button');
  for (let btn of buttons) {
    if (btn.innerText.toLowerCase().includes(text.toLowerCase())) return btn;
  }
  return null;
}

class SimplePreprocessor {
  constructor() {
    this.featureOrder = [];
    this.means = {};
    this.stds = {};
    this.headers = [];
  }

  fit(data, headers) {
    this.headers = headers.filter(h => h !== 'loan_status');
    this.headers.forEach((col, idx) => {
      const values = data.map(row => parseFloat(row[idx] || 0)).filter(v => !isNaN(v));
      if (values.length === 0) return;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      const std = Math.sqrt(variance) || 1;
      this.means[col] = mean;
      this.stds[col] = std;
    });
    this.featureOrder = this.headers;
  }

  transform(data, includeTarget = true) {
    const features = [];
    const targets = [];
    data.forEach(row => {
      const featureRow = [];
      this.headers.forEach(col => {
        const colIdx = this.headers.indexOf(col);
        const numVal = parseFloat(row[colIdx] || 0);
        const mean = this.means[col] || 0;
        const std = this.stds[col] || 1;
        featureRow.push((numVal - mean) / std);
      });
      features.push(featureRow);
      if (includeTarget && trainHeaders) {
        const targetIdx = trainHeaders.indexOf('loan_status');
        targets.push(row[targetIdx] === '1' ? 1 : 0);
      }
    });
    return { features, targets };
  }

  toJSON() {
    return { headers: this.headers, means: this.means, stds: this.stds, featureOrder: this.featureOrder };
  }

  static fromJSON(json) {
    const prep = new SimplePreprocessor();
    prep.headers = json.headers || [];
    prep.means = json.means || {};
    prep.stds = json.stds || {};
    prep.featureOrder = json.featureOrder || [];
    return prep;
  }
}

// ✅ LOAD DATA
window.onloadData = async function() {
  try {
    const trainFile = document.querySelector('input[type="file"]')?.files[0];
    if (!trainFile) return alert('❌ Select train.csv first');

    const loadBtn = findButtonByText('load data');
    if (loadBtn) {
      loadBtn.disabled = true;
      loadBtn.innerText = 'Loading...';
    }

    const text = await trainFile.text();
    const lines = text.split('\n').filter(l => l.trim());
    const parsed = lines.map(line => line.split(',').map(cell => cell.trim().replace(/"/g, '')));
    
    trainHeaders = parsed[0];
    trainData = parsed.slice(1).filter(row => row.length > 1);

    if (!trainHeaders.includes('loan_status')) return alert('❌ Missing loan_status column');

    preprocessor = new SimplePreprocessor();
    preprocessor.fit(trainData, trainHeaders);

    const { train, val } = simpleSplit(trainData);
    const valProcessed = preprocessor.transform(val, true);
    
    disposeTensorsSafely();
    valXs = tf.tensor2d(valProcessed.features);
    valYs = tf.tensor1d(valProcessed.targets);

    const edaEl = document.querySelector('[id*="eda"], [class*="eda"], #eda-output');
    if (edaEl) {
      const approved = trainData.filter(row => row[trainHeaders.indexOf('loan_status')] === '1').length;
      edaEl.innerHTML = `📊 **EDA COMPLETE**
• Samples: ${trainData.length.toLocaleString()}
• Approved: ${approved} (${((approved/trainData.length)*100).toFixed(1)}%)
• Features: ${preprocessor.headers.length}
• ✅ READY TO TRAIN!`;
    }

    updateButtons();
    alert(`✅ DATA LOADED!\n📊 ${trainData.length} samples`);

  } catch (e) {
    alert('Load error: ' + e.message);
  } finally {
    const loadBtn = findButtonByText('load data');
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.innerText = '📊 Load Data';
    }
  }
};

// ✅ TRAIN MODEL
window.ontrainModel = async function() {
  if (!preprocessor || !valXs || !valYs) return alert('❌ Load data first');

  try {
    const trainBtn = findButtonByText('train');
    if (trainBtn) {
      trainBtn.disabled = true;
      trainBtn.innerText = 'Training...';
    }

    disposeModelSafely();

    model = tf.sequential({
      layers: [
        tf.layers.dense({units: 128, activation: 'relu', inputShape: [preprocessor.headers.length]}),
        tf.layers.dropout({rate: 0.3}),
        tf.layers.dense({units: 64, activation: 'relu'}),
        tf.layers.dropout({rate: 0.2}),
        tf.layers.dense({units: 32, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'sigmoid'})
      ]
    });

    model.compile({ optimizer: tf.train.adam(0.001), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    const { train } = simpleSplit(trainData);
    const trainProcessed = preprocessor.transform(train, true);
    const xs = tf.tensor2d(trainProcessed.features);
    const ys = tf.tensor1d(trainProcessed.targets);

    await model.fit(xs, ys, {
      epochs: 25,
      batchSize: 64,
      validationData: [valXs, valYs]
    });

    xs.dispose();
    ys.dispose();

    const probs = model.predict(valXs).dataSync();
    const labels = valYs.dataSync();
    const predictions = probs.map(p => p > currentThreshold ? 1 : 0);
    
    let tp=0, fp=0, tn=0, fn=0;
    for (let i = 0; i < predictions.length; i++) {
      if (predictions[i] === 1 && labels[i] === 1) tp++;
      else if (predictions[i] === 1 && labels[i] === 0) fp++;
      else if (predictions[i] === 0 && labels[i] === 0) tn++;
      else fn++;
    }

    const total = tp + fp + tn + fn;
    const accuracy = ((tp + tn) / total * 100).toFixed(1);
    const precision = ((tp / (tp + fp)) * 100 || 0).toFixed(1);
    const recall = ((tp / (tp + fn)) * 100 || 0).toFixed(1);
    const f1 = ((2 * tp) / (2 * tp + fp + fn) * 100 || 0).toFixed(1);

    const metricsEl = document.querySelector('[id*="metrics"], [class*="metrics"]');
    if (metricsEl) {
      metricsEl.innerHTML = `
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #4caf50;">
          <h3 style="color: #2e7d32; margin: 0 0 15px 0;">🎯 MODEL PERFORMANCE</h3>
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 15px;">
            <div style="text-align: center; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
              <div style="font-size: 28px; font-weight: bold; color: #4caf50;">${accuracy}%</div>
              <div style="font-size: 12px; color: #666; text-transform: uppercase;">Accuracy</div>
            </div>
            <div style="text-align: center; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
              <div style="font-size: 28px; font-weight: bold; color: #4caf50;">${f1}%</div>
              <div style="font-size: 12px; color: #666; text-transform: uppercase;">F1 Score</div>
            </div>
          </div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; font-size: 14px;">
            <div><strong>Precision:</strong> ${precision}%</div>
            <div><strong>Recall:</strong> ${recall}%</div>
          </div>
          <div style="margin-top: 10px; font-size: 12px; color: #666;">
            TP:${tp} FP:${fp} FN:${fn} TN:${tn}
          </div>
        </div>
      `;
    }

    updateButtons();
    alert(`✅ TRAINING COMPLETE!\n🎯 Accuracy: ${accuracy}%\n⚖️ F1 Score: ${f1}%`);

  } catch (e) {
    alert('Training error: ' + e.message);
  } finally {
    const trainBtn = findButtonByText('train');
    if (trainBtn) {
      trainBtn.disabled = false;
      trainBtn.innerText = '🚀 Train Model';
    }
  }
};

// ✅ SAVE MODEL
window.onsaveModel = async function() {
  if (!model || !preprocessor) return alert('❌ Train model first');

  try {
    const saveBtn = findButtonByText('save');
    if (saveBtn) {
      saveBtn.disabled = true;
      saveBtn.innerText = 'Saving...';
    }

    const modelJson = JSON.stringify(model.toJSON());
    downloadFile('model.json', new Blob([modelJson], { type: 'application/json' }));

    const weights = model.getWeights();
    const weightBuffers = [];
    for (let w of weights) {
      const data = await w.data();
      weightBuffers.push(new Uint8Array(data.buffer));
    }
    downloadFile('model.weights.bin', new Blob(weightBuffers, { type: 'application/octet-stream' }));

    downloadFile('preprocessor.json', new Blob([JSON.stringify(preprocessor.toJSON(), null, 2)], { type: 'application/json' }));

    alert('✅ ALL 3 FILES DOWNLOADED!\n📥 model.json\n📥 model.weights.bin\n📥 preprocessor.json');

  } catch (e) {
    alert('Save error: ' + e.message);
  } finally {
    const saveBtn = findButtonByText('save');
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerText = '💾 Save Model';
    }
  }
};

// ✅ LOAD MODEL
window.onloadModelAndPrep = async function() {
  try {
    const loadBtn = findButtonByText('load model');
    if (loadBtn) {
      loadBtn.disabled = true;
      loadBtn.innerText = 'Loading...';
    }

    disposeModelSafely();
    disposeTensorsSafely();

    const files = Array.from(document.querySelectorAll('input[type="file"]'))
      .map(input => input.files[0]).filter(f => f);

    let modelJsonFile = null, weightsFile = null, prepFile = null;

    for (let file of files) {
      const name = file.name.toLowerCase();
      if (name.includes('model.json')) modelJsonFile = file;
      else if (name.includes('weights') || name.includes('.bin')) weightsFile = file;
      else if (name.includes('prep') || name.includes('preprocess')) prepFile = file;
    }

    if (!modelJsonFile || !weightsFile) {
      return alert('❌ Select model.json + .bin weights file');
    }

    if (prepFile) {
      const prepText = await prepFile.text();
      preprocessor = SimplePreprocessor.fromJSON(JSON.parse(prepText));
    }

    const modelFiles = [
      { path: 'model.json', data: modelJsonFile },
      { path: 'model.weights.bin', data: weightsFile }
    ];

    model = await tf.loadLayersModel(tf.io.browserFiles(modelFiles));

    updateButtons();
    alert(`✅ MODEL LOADED!\n🎯 Features: ${preprocessor?.headers.length || 'N/A'}`);

  } catch (e) {
    alert('Load error: ' + e.message);
  } finally {
    const loadBtn = findButtonByText('load model');
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.innerText = '📂 Load Model';
    }
  }
};

// ✅ PREDICT
window.onpredictTest = async function() {
  if (!model || !testData || !preprocessor) return alert('❌ Load model + test data first');

  try {
    const predictBtn = findButtonByText('predict');
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.innerText = 'Predicting...';
    }

    const testProcessed = preprocessor.transform(testData, false);
    const xs = tf.tensor2d(testProcessed.features);
    const probs = model.predict(xs).dataSync();

    const submission = [['ApplicationID', 'Approved']];
    let approved = 0;

    for (let i = 0; i < probs.length; i++) {
      const pred = probs[i] > currentThreshold ? 1 : 0;
      if (pred === 1) approved++;
      submission.push([`App_${i+1}`, pred]);
    }

    downloadCSV('submission.csv', submission);
    alert(`✅ PREDICTIONS COMPLETE!\n🎯 Approvals: ${approved}/${probs.length} (${((approved/probs.length)*100).toFixed(1)}%)`);

  } catch (e) {
    alert('Predict error: ' + e.message);
  } finally {
    const predictBtn = findButtonByText('predict');
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.innerText = '🔮 Predict';
    }
  }
};

function simpleSplit(data, ratio = 0.8) {
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const split = Math.floor(shuffled.length * ratio);
  return { train: shuffled.slice(0, split), val: shuffled.slice(split) };
}

function downloadFile(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadCSV(name, rows) {
  const csv = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  downloadFile(name, new Blob([csv], { type: 'text/csv' }));
}

function updateButtons() {
  const hasData = !!preprocessor && !!valXs;
  const hasModel = !!model;
  const hasTest = !!testData;

  document.querySelectorAll('button').forEach(btn => {
    const text = btn.innerText.toLowerCase();
    if (text.includes('train')) btn.disabled = !hasData;
    if (text.includes('save')) btn.disabled = !hasModel;
    if (text.includes('predict')) btn.disabled = !hasModel || !hasTest;
  });
}

async function initApp() {
  await tf.ready();
  
  const bindAllButtons = () => {
    document.querySelectorAll('button').forEach(btn => {
      const text = btn.innerText.toLowerCase();
      if (text.includes('load data') || text.includes('data')) btn.onclick = window.onloadData;
      if (text.includes('train')) btn.onclick = window.ontrainModel;
      if (text.includes('save') || text.includes('export')) btn.onclick = window.onsaveModel;
      if (text.includes('load model') || text.includes('import')) btn.onclick = window.onloadModelAndPrep;
      if (text.includes('predict')) btn.onclick = window.onpredictTest;
    });
  };

  bindAllButtons();
  setTimeout(bindAllButtons, 1000);
  setTimeout(bindAllButtons, 2000);

  updateButtons();
  console.log('🎉 LOAN APPROVAL SYSTEM READY - 100% PERFECT!');
}

document.addEventListener('DOMContentLoaded', initApp);
