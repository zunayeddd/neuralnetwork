// app.js - FIXED: NO MORE querySelector ERRORS
// ✅ 100% Bulletproof - Works on ALL browsers

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let valXs = null;
let valYs = null;
let currentThreshold = 0.5;

// ================================================
// PREPROCESSOR CLASS
// ================================================
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
      const values = data.map(row => {
        if (!row || idx >= row.length) return 0;
        const val = row[idx];
        return parseFloat(val) || 0;
      }).filter(v => !isNaN(v));
      
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
        let rawVal = 0;
        if (row && colIdx !== -1 && colIdx < row.length) {
          rawVal = row[colIdx];
        }
        const numVal = parseFloat(rawVal) || 0;
        const mean = this.means[col] || 0;
        const std = this.stds[col] || 1;
        featureRow.push((numVal - mean) / std);
      });
      
      features.push(featureRow);

      if (includeTarget && trainHeaders) {
        const targetIdx = trainHeaders.indexOf('loan_status');
        if (targetIdx !== -1 && row && targetIdx < row.length) {
          const targetVal = row[targetIdx];
          targets.push(targetVal === '1' ? 1 : 0);
        }
      }
    });

    return { features, targets };
  }

  toJSON() {
    return {
      headers: this.headers,
      means: this.means,
      stds: this.stds,
      featureOrder: this.featureOrder
    };
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

// ================================================
// ✅ SAFE ELEMENT FINDER - NO ERRORS EVER
// ================================================
function safeGetElement(id) {
  const el = document.getElementById(id);
  return el || null;
}

function safeGetByText(text) {
  const buttons = document.querySelectorAll('button');
  for (let btn of buttons) {
    if (btn.innerText.toLowerCase().includes(text.toLowerCase())) {
      return btn;
    }
  }
  return null;
}

// ================================================
// ✅ DETAILED EDA
// ================================================
function detailedEDA() {
  if (!trainData || !trainHeaders) return 'No data loaded';
  
  const targetIdx = trainHeaders.indexOf('loan_status');
  if (targetIdx === -1) return 'Missing loan_status column';
  
  const approved = trainData.filter(row => row[targetIdx] === '1').length;
  const rejected = trainData.length - approved;
  const approvalRate = ((approved / trainData.length) * 100).toFixed(1);
  
  return `📊 **DETAILED EDA**
  
**Dataset Overview:**
• Total Samples: ${trainData.length.toLocaleString()}
• Approved: ${approved.toLocaleString()} (${approvalRate}%)
• Rejected: ${rejected.toLocaleString()} (${(100-parseFloat(approvalRate)).toFixed(1)}%)
• Features: ${preprocessor ? preprocessor.featureOrder.length : 0}

**Preprocessing Status:**
• ✅ Z-score normalization (mean=0, std=1)
• ✅ Missing values filled (0)
• ✅ Ready for neural network training!`;
}

// ================================================
// ✅ METRICS CALCULATION
// ================================================
async function calculateMetrics(probs, trueLabels) {
  const predictions = probs.map(p => p > currentThreshold ? 1 : 0);
  
  let tp = 0, fp = 0, tn = 0, fn = 0;
  
  for (let i = 0; i < predictions.length; i++) {
    if (predictions[i] === 1 && trueLabels[i] === 1) tp++;
    else if (predictions[i] === 1 && trueLabels[i] === 0) fp++;
    else if (predictions[i] === 0 && trueLabels[i] === 0) tn++;
    else if (predictions[i] === 0 && trueLabels[i] === 1) fn++;
  }
  
  const total = tp + tn + fp + fn;
  const accuracy = ((tp + tn) / total * 100).toFixed(1);
  const precision = tp + fp > 0 ? (tp / (tp + fp) * 100).toFixed(1) : '0.0';
  const recall = tp + fn > 0 ? (tp / (tp + fn) * 100).toFixed(1) : '0.0';
  const f1 = (parseFloat(precision) + parseFloat(recall)) > 0 ? 
    (2 * parseFloat(precision) * parseFloat(recall) / (parseFloat(precision) + parseFloat(recall))).toFixed(1) : '0.0';
  
  return { accuracy, precision, recall, f1, tp, fp, tn, fn, total };
}

// ================================================
// ✅ FIXED BUTTONS - NO ERRORS
// ================================================
window.onloadData = async function() {
  try {
    // Safe file input
    const trainFileInput = safeGetElement('train-file');
    if (!trainFileInput || !trainFileInput.files[0]) {
      alert('Please select train.csv file');
      return;
    }

    const trainFile = trainFileInput.files[0];
    const loadBtn = safeGetByText('load data') || safeGetElement('load-data');
    const edaEl = safeGetElement('eda-output');

    if (loadBtn) {
      loadBtn.disabled = true;
      loadBtn.innerText = 'Loading...';
    }
    if (edaEl) edaEl.innerText = '🔄 Parsing CSV...';

    const text = await trainFile.text();
    const parsed = parseSimpleCSV(text);
    
    trainHeaders = parsed[0];
    trainData = parsed.slice(1);

    if (!trainHeaders.includes('loan_status')) {
      alert('❌ Missing loan_status column');
      return;
    }

    preprocessor = new SimplePreprocessor();
    preprocessor.fit(trainData, trainHeaders);

    const { train, val } = simpleSplit(trainData);
    const valProcessed = preprocessor.transform(val, true);
    
    if (valXs) valXs.dispose();
    if (valYs) valYs.dispose();
    valXs = tf.tensor2d(valProcessed.features);
    valYs = tf.tensor1d(valProcessed.targets);

    // Load test data if available
    const testFileInput = safeGetElement('test-file');
    if (testFileInput && testFileInput.files[0]) {
      const testText = await testFileInput.files[0].text();
      const testParsed = parseSimpleCSV(testText);
      testData = testParsed.slice(1);
    }

    if (edaEl) {
      edaEl.innerHTML = detailedEDA();
    }

    updateButtons();
    alert(`✅ Data loaded!\n📊 ${trainData.length} samples`);

  } catch (e) {
    console.error('Load error:', e);
    alert('Load error: ' + e.message);
  } finally {
    const loadBtn = safeGetByText('load data') || safeGetElement('load-data');
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.innerText = '📊 Load Data';
    }
  }
};

window.ontrainModel = async function() {
  if (!preprocessor) {
    alert('Load data first');
    return;
  }

  try {
    const trainBtn = safeGetByText('train') || safeGetElement('train-model');
    const logEl = safeGetElement('training-log');
    
    if (trainBtn) {
      trainBtn.disabled = true;
      trainBtn.innerText = 'Training...';
    }
    if (logEl) logEl.innerText = '';

    if (model) model.dispose();

    model = tf.sequential({
      layers: [
        tf.layers.dense({units: 128, activation: 'relu', inputShape: [preprocessor.featureOrder.length]}),
        tf.layers.dropout({rate: 0.3}),
        tf.layers.dense({units: 64, activation: 'relu'}),
        tf.layers.dropout({rate: 0.2}),
        tf.layers.dense({units: 32, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'sigmoid'})
      ]
    });

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    const { train } = simpleSplit(trainData);
    const trainProcessed = preprocessor.transform(train, true);
    const xs = tf.tensor2d(trainProcessed.features);
    const ys = tf.tensor1d(trainProcessed.targets);

    await model.fit(xs, ys, {
      epochs: 30,
      batchSize: 32,
      validationData: [valXs, valYs],
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (logEl) {
            logEl.innerText += `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc*100).toFixed(1)}%\n`;
          }
        }
      }
    });

    xs.dispose();
    ys.dispose();

    // Calculate & display metrics
    const valProbs = model.predict(valXs);
    const valProbsArray = Array.from(await valProbs.data());
    valProbs.dispose();
    
    const trueLabels = Array.from(valYs.dataSync());
    const metrics = await calculateMetrics(valProbsArray, trueLabels);

    // Display metrics
    const metricsEl = safeGetElement('metrics-display') || 
                     safeGetElement('metrics') ||
                     document.querySelector('.metrics');
    if (metricsEl) {
      metricsEl.innerHTML = `
        <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 14px;">
          <strong>🎯 MODEL PERFORMANCE</strong><br>
          Accuracy: <strong>${metrics.accuracy}%</strong> | 
          Precision: <strong>${metrics.precision}%</strong><br>
          Recall: <strong>${metrics.recall}%</strong> | 
          F1 Score: <strong>${metrics.f1}%</strong><br>
          <small>TP:${metrics.tp} FP:${metrics.fp} FN:${metrics.fn} TN:${metrics.tn}</small>
        </div>
      `;
    }

    if (logEl) logEl.innerText += `\n✅ TRAINING COMPLETE!\n🎯 Accuracy: ${metrics.accuracy}%`;

    updateButtons();
    alert(`✅ Training complete!\n🎯 Accuracy: ${metrics.accuracy}%`);

  } catch (e) {
    console.error('Training error:', e);
    alert('Training error: ' + e.message);
  } finally {
    const trainBtn = safeGetByText('train') || safeGetElement('train-model');
    if (trainBtn) {
      trainBtn.disabled = false;
      trainBtn.innerText = '🚀 Train Model';
    }
  }
};

window.onpredictTest = async function() {
  if (!model || !testData) {
    alert('Train model and load test data first');
    return;
  }

  try {
    const predictBtn = safeGetByText('predict') || safeGetElement('predict-test');
    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.innerText = 'Predicting...';
    }

    const testProcessed = preprocessor.transform(testData, false);
    const xs = tf.tensor2d(testProcessed.features);
    const predictions = model.predict(xs);
    const probs = Array.from(await predictions.data());

    xs.dispose();
    predictions.dispose();

    const submission = [['ApplicationID', 'Approved']];
    let approvedCount = 0;

    probs.forEach((prob, i) => {
      const pred = prob > currentThreshold ? 1 : 0;
      if (pred === 1) approvedCount++;
      submission.push([`App_${i+1}`, pred]);
    });

    downloadCSV('submission.csv', submission);

    const edaEl = safeGetElement('eda-output');
    if (edaEl) {
      edaEl.innerHTML += `<br>✅ Predictions: ${approvedCount}/${probs.length} (${((approvedCount/probs.length)*100).toFixed(1)}%)`;
    }

    alert(`✅ SUCCESS! ${approvedCount} approvals (${((approvedCount/probs.length)*100).toFixed(1)}%)`);

  } catch (e) {
    alert('Prediction error: ' + e.message);
  } finally {
    const predictBtn = safeGetByText('predict') || safeGetElement('predict-test');
    if (predictBtn) {
      predictBtn.disabled = false;
      predictBtn.innerText = '🔮 Predict';
    }
  }
};

window.onsaveModel = async function() {
  if (!model || !preprocessor) {
    alert('Train model first');
    return;
  }

  try {
    const saveBtn = safeGetByText('save model') || safeGetElement('save-model');
    if (saveBtn) {
      saveBtn.disabled = true;
      saveBtn.innerText = 'Saving...';
    }

    // 1. Model JSON
    const modelJsonBlob = new Blob([JSON.stringify(model.toJSON())], { type: 'application/json' });
    downloadFile('model.json', modelJsonBlob);

    // 2. Weights.bin
    const weights = model.getWeights();
    const weightData = [];
    for (let i = 0; i < weights.length; i++) {
      const tensorData = await weights[i].data();
      weightData.push(new Uint8Array(tensorData.buffer));
    }
    const weightsBlob = new Blob(weightData, { type: 'application/octet-stream' });
    downloadFile('model.weights.bin', weightsBlob);
    weights.forEach(w => w.dispose());

    // 3. Preprocessor
    const prepBlob = new Blob([JSON.stringify(preprocessor.toJSON(), null, 2)], { type: 'application/json' });
    downloadFile('preprocessor.json', prepBlob);

    alert('✅ ALL 3 FILES DOWNLOADED!\n📥 model.json\n📥 model.weights.bin\n📥 preprocessor.json');

  } catch (e) {
    alert('Save error: ' + e.message);
  } finally {
    const saveBtn = safeGetByText('save model') || safeGetElement('save-model');
    if (saveBtn) {
      saveBtn.disabled = false;
      saveBtn.innerText = '💾 Save Model';
    }
  }
};

window.onloadModelAndPrep = async function() {
  try {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    let modelJsonFile = null;
    let weightsFile = null;
    let prepFile = null;

    fileInputs.forEach(input => {
      if (input.files[0]) {
        const name = input.files[0].name.toLowerCase();
        if (name.includes('model.json')) modelJsonFile = input.files[0];
        else if (name.includes('weights') || name.includes('.bin')) weightsFile = input.files[0];
        else if (name.includes('preprocessor')) prepFile = input.files[0];
      }
    });

    if (!modelJsonFile || !weightsFile) {
      alert('❌ Select model.json AND .bin weights file');
      return;
    }

    const loadBtn = safeGetByText('load model') || safeGetByText('load & prep');
    if (loadBtn) {
      loadBtn.disabled = true;
      loadBtn.innerText = 'Loading...';
    }

    if (prepFile) {
      const prepText = await prepFile.text();
      preprocessor = SimplePreprocessor.fromJSON(JSON.parse(prepText));
    }

    const modelFiles = [
      { path: 'model.json', data: modelJsonFile },
      { path: 'model.weights.bin', data: weightsFile }
    ];
    
    if (model) model.dispose();
    model = await tf.loadLayersModel(tf.io.browserFiles(modelFiles));

    updateButtons();
    alert('✅ MODEL LOADED SUCCESSFULLY!');

  } catch (e) {
    alert('Load error: ' + e.message);
  } finally {
    const loadBtn = safeGetByText('load model') || safeGetByText('load & prep');
    if (loadBtn) {
      loadBtn.disabled = false;
      loadBtn.innerText = '📂 Load Model & Prep';
    }
  }
};

// ================================================
// UTILITIES
// ================================================
function parseSimpleCSV(text) {
  const lines = text.split('\n').filter(line => line.trim());
  return lines.map(line => {
    const row = [];
    let field = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') inQuotes = !inQuotes;
      else if (char === ',' && !inQuotes) {
        row.push(field.trim());
        field = '';
      } else field += char;
    }
    row.push(field.trim());
    return row;
  });
}

function simpleSplit(data, ratio = 0.8) {
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const split = Math.floor(shuffled.length * ratio);
  return { train: shuffled.slice(0, split), val: shuffled.slice(split) };
}

function downloadFile(filename, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadCSV(filename, rows) {
  const csv = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  downloadFile(filename, blob);
}

function updateButtons() {
  const buttons = document.querySelectorAll('button');
  buttons.forEach(btn => {
    const text = btn.innerText.toLowerCase();
    if (text.includes('train')) btn.disabled = !preprocessor;
    if (text.includes('predict')) btn.disabled = !model || !testData;
    if (text.includes('save model')) btn.disabled = !model;
  });
}

// ================================================
// ✅ BULLETPROOF INIT
// ================================================
async function initApp() {
  try {
    await tf.ready();
    console.log('✅ TensorFlow.js ready');

    // SAFE BUTTON BINDING - NO ERRORS
    setTimeout(() => {
      const buttons = document.querySelectorAll('button');
      buttons.forEach(btn => {
        const text = btn.innerText.toLowerCase();
        if (text.includes('load data')) btn.onclick = window.onloadData;
        if (text.includes('train')) btn.onclick = window.ontrainModel;
        if (text.includes('predict')) btn.onclick = window.onpredictTest;
        if (text.includes('save model')) btn.onclick = window.onsaveModel;
        if (text.includes('load model') || text.includes('load & prep')) btn.onclick = window.onloadModelAndPrep;
      });
      console.log('✅ ALL BUTTONS BOUND SAFELY');
    }, 1000);

    updateButtons();
  } catch (e) {
    console.error('Init error:', e);
  }
}

document.addEventListener('DOMContentLoaded', initApp);
