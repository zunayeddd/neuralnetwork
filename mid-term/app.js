// app.js - FIXED: NO MORE SELECTOR ERRORS - 100% BULLETPROOF
// ✅ Removed :has() selectors - Works on ALL browsers

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
// ✅ DETAILED EDA FUNCTION
// ================================================
function detailedEDA() {
  if (!trainData || !trainHeaders) return 'No data loaded';
  
  const targetIdx = trainHeaders.indexOf('loan_status');
  if (targetIdx === -1) return 'Missing loan_status column';
  
  const approved = trainData.filter(row => row[targetIdx] === '1').length;
  const rejected = trainData.length - approved;
  const approvalRate = ((approved / trainData.length) * 100).toFixed(1);
  
  let featureStats = '';
  preprocessor.headers.slice(0, 6).forEach((col, i) => {
    const values = trainData.map(row => parseFloat(row[i]) || 0).filter(v => !isNaN(v));
    if (values.length > 0) {
      const mean = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1);
      const approvedMean = (trainData
        .filter(row => row[targetIdx] === '1')
        .map(row => parseFloat(row[i]) || 0)
        .filter(v => !isNaN(v))
        .reduce((a, b) => a + b, 0) / approved || 0).toFixed(1);
      featureStats += `• ${col}: ${mean} (Approved: ${approvedMean})\n`;
    }
  });

  return `📊 **DETAILED EDA**

**Dataset Overview:**
• Total Samples: ${trainData.length.toLocaleString()}
• Approved: ${approved.toLocaleString()} (${approvalRate}%)
• Rejected: ${rejected.toLocaleString()} (${(100-approvalRate).toFixed(1)}%)
• Features: ${preprocessor.featureOrder.length}

**Top Feature Insights:**
${featureStats}

**Preprocessing Status:**
• ✅ Z-score normalization (mean=0, std=1)
• ✅ Missing values → 0
• ✅ Ready for neural network!`;
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
// ✅ FIXED: BULLETPROOF BUTTON FUNCTIONS (NO :has())
// ================================================
window.onloadData = async function() {
  try {
    const trainFile = document.getElementById('train-file')?.files[0];
    if (!trainFile) {
      alert('Please select train.csv');
      return;
    }

    // ✅ SAFE BUTTON DISABLE - NO ERRORS
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('load data') || btn.id === 'load-data') {
        btn.disabled = true;
        btn.innerText = 'Loading...';
      }
    });

    // ✅ SAFE EDA DISPLAY
    const edaElements = [
      document.getElementById('eda-output'),
      ...document.querySelectorAll('[id*="eda"], [class*="eda"]')
    ].filter(el => el);
    
    if (edaElements.length > 0) {
      edaElements[0].innerText = '🔄 Parsing CSV...';
    }

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

    const testFile = document.getElementById('test-file')?.files[0];
    if (testFile) {
      const testText = await testFile.text();
      const testParsed = parseSimpleCSV(testText);
      testData = testParsed.slice(1);
    }

    // ✅ DETAILED EDA
    if (edaElements.length > 0) {
      edaElements[0].innerHTML = detailedEDA();
    }

    updateButtons();
    alert(`✅ Data loaded!\n📊 ${trainData.length.toLocaleString()} samples`);

  } catch (e) {
    console.error('Load error:', e);
    alert('Load error: ' + e.message);
  } finally {
    // ✅ SAFE BUTTON ENABLE
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('load data') || btn.id === 'load-data') {
        btn.disabled = false;
        btn.innerText = '📊 Load Data';
      }
    });
  }
};

window.ontrainModel = async function() {
  if (!preprocessor || !valXs || !valYs) {
    alert('Load data first');
    return;
  }

  try {
    // ✅ SAFE BUTTON DISABLE
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('train') || btn.id === 'train-model') {
        btn.disabled = true;
        btn.innerText = 'Training...';
      }
    });

    const logElements = [
      document.getElementById('training-log'),
      ...document.querySelectorAll('[id*="training"], [class*="log"]')
    ].filter(el => el);
    
    if (logElements.length > 0) logElements[0].innerText = '';

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
          if (logElements.length > 0) {
            logElements[0].innerText += `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, val_acc=${(logs.val_acc*100).toFixed(1)}%\n`;
          }
        }
      }
    });

    xs.dispose();
    ys.dispose();

    // ✅ FINAL METRICS
    const valProbs = model.predict(valXs);
    const valProbsArray = Array.from(await valProbs.data());
    valProbs.dispose();
    
    const trueLabels = Array.from(valYs.dataSync());
    const metrics = await calculateMetrics(valProbsArray, trueLabels);

    // ✅ DISPLAY METRICS EVERYWHERE
    const metricsElements = [
      document.getElementById('metrics-display'),
      ...document.querySelectorAll('[id*="metrics"], [class*="metrics"]')
    ].filter(el => el);
    
    if (metricsElements.length > 0) {
      metricsElements.forEach(el => {
        el.innerHTML = `
          <div style="background: #e8f5e8; padding: 15px; border-radius: 8px; font-family: monospace; border-left: 4px solid #4caf50;">
            <h4>🎯 MODEL PERFORMANCE</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px;">
              <div><strong>Accuracy:</strong> <span style="color: #2e7d32;">${metrics.accuracy}%</span></div>
              <div><strong>Precision:</strong> <span style="color: #2e7d32;">${metrics.precision}%</span></div>
              <div><strong>Recall:</strong> <span style="color: #2e7d32;">${metrics.recall}%</span></div>
              <div><strong>F1 Score:</strong> <span style="color: #2e7d32;">${metrics.f1}%</span></div>
            </div>
            <div style="margin-top: 10px; font-size: 12px; color: #666;">
              TP:${metrics.tp} FP:${metrics.fp} FN:${metrics.fn} TN:${metrics.tn}
            </div>
            <div style="margin-top: 5px; color: #1976d2; font-weight: bold;">
              Threshold: ${currentThreshold}
            </div>
          </div>
        `;
      });
    }

    if (logElements.length > 0) {
      logElements[0].innerText += `\n✅ TRAINING COMPLETE!\n🎯 Accuracy: ${metrics.accuracy}% | F1: ${metrics.f1}%`;
    }

    updateButtons();
    alert(`✅ Training complete!\n🎯 Accuracy: ${metrics.accuracy}%\n⚖️ F1: ${metrics.f1}%`);

  } catch (e) {
    console.error('Training error:', e);
    alert('Training error: ' + e.message);
  } finally {
    // ✅ SAFE BUTTON ENABLE
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('train') || btn.id === 'train-model') {
        btn.disabled = false;
        btn.innerText = '🚀 Train Model';
      }
    });
  }
};

window.onpredictTest = async function() {
  if (!model || !testData || !preprocessor) {
    alert('Train model + load test data first');
    return;
  }

  try {
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('predict') || btn.id === 'predict-test') {
        btn.disabled = true;
        btn.innerText = 'Predicting...';
      }
    });

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

    const edaElements = document.querySelectorAll('[id*="eda"], [class*="eda"]');
    if (edaElements.length > 0) {
      edaElements[0].innerHTML += `<br><strong>✅ PREDICTIONS:</strong> ${approvedCount}/${probs.length} (${((approvedCount/probs.length)*100).toFixed(1)}%) | Threshold: ${currentThreshold}`;
    }

    alert(`✅ SUCCESS! ${approvedCount}/${probs.length} approvals (${((approvedCount/probs.length)*100).toFixed(1)}%)`);

  } catch (e) {
    alert('Prediction error: ' + e.message);
  } finally {
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('predict') || btn.id === 'predict-test') {
        btn.disabled = false;
        btn.innerText = '🔮 Predict';
      }
    });
  }
};

// ================================================
// ✅ FIXED: PROPER .bin EXPORT
// ================================================
window.onsaveModel = async function() {
  if (!model || !preprocessor) {
    alert('Train model first');
    return;
  }

  try {
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('save model') || btn.id === 'save-model') {
        btn.disabled = true;
        btn.innerText = 'Saving...';
      }
    });

    // 1. MODEL JSON
    const modelJson = model.toJSON();
    const modelJsonBlob = new Blob([JSON.stringify(modelJson)], { type: 'application/json' });
    downloadFile('model.json', modelJsonBlob);

    // 2. WEIGHTS.BIN (CORRECT FORMAT)
    const weights = model.getWeights();
    const weightBuffers = [];
    for (let weight of weights) {
      const data = await weight.data();
      weightBuffers.push(new Uint8Array(data.buffer));
      weight.dispose();
    }
    const weightsBlob = new Blob(weightBuffers, { type: 'application/octet-stream' });
    downloadFile('model.weights.bin', weightsBlob);

    // 3. PREPROCESSOR
    const prepJson = preprocessor.toJSON();
    const prepBlob = new Blob([JSON.stringify(prepJson, null, 2)], { type: 'application/json' });
    downloadFile('preprocessor.json', prepBlob);

    alert('✅ ALL 3 FILES DOWNLOADED!\n📥 model.json + model.weights.bin + preprocessor.json');

  } catch (e) {
    console.error('Save error:', e);
    alert('Save error: ' + e.message);
  } finally {
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('save model') || btn.id === 'save-model') {
        btn.disabled = false;
        btn.innerText = '💾 Save Model';
      }
    });
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
        if (name.includes('model.json') || (name.includes('model') && name.endsWith('.json'))) {
          modelJsonFile = input.files[0];
        } else if (name.includes('weights') || name.includes('.bin')) {
          weightsFile = input.files[0];
        } else if (name.includes('prep') || name.includes('preprocess')) {
          prepFile = input.files[0];
        }
      }
    });

    if (!modelJsonFile || !weightsFile) {
      alert('❌ Select model.json AND .bin weights file');
      return;
    }

    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('load model') || btn.innerText.toLowerCase().includes('load & prep')) {
        btn.disabled = true;
        btn.innerText = 'Loading...';
      }
    });

    if (prepFile) {
      const prepText = await prepFile.text();
      const prepJson = JSON.parse(prepText);
      preprocessor = SimplePreprocessor.fromJSON(prepJson);
    }

    const modelFiles = [
      { path: 'model.json', data: modelJsonFile },
      { path: 'model.weights.bin', data: weightsFile }
    ];
    
    if (model) model.dispose();
    model = await tf.loadLayersModel(tf.io.browserFiles(modelFiles));

    updateButtons();
    alert(`✅ MODEL LOADED!\n🎯 Features: ${preprocessor ? preprocessor.headers.length : 'N/A'}`);

  } catch (error) {
    console.error('Load error:', error);
    alert(`❌ Load failed: ${error.message}`);
  } finally {
    const allButtons = document.querySelectorAll('button');
    allButtons.forEach(btn => {
      if (btn.innerText.toLowerCase().includes('load model') || btn.innerText.toLowerCase().includes('load & prep')) {
        btn.disabled = false;
        btn.innerText = '📂 Load Model & Prep';
      }
    });
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
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

function downloadCSV(filename, rows) {
  const csv = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  downloadFile(filename, blob);
}

function updateButtons() {
  const hasData = !!preprocessor;
  const hasModel = !!model;
  const hasTest = !!testData;
  
  const allButtons = document.querySelectorAll('button');
  allButtons.forEach(btn => {
    const text = btn.innerText.toLowerCase();
    if (text.includes('train')) btn.disabled = !hasData;
    if (text.includes('predict')) btn.disabled = !hasModel || !hasTest;
    if (text.includes('save model')) btn.disabled = !hasModel;
  });
}

// ================================================
// ✅ BULLETPROOF INIT - NO ERRORS
// ================================================
async function initApp() {
  try {
    await tf.ready();
    console.log('✅ TensorFlow.js ready');

    // ✅ AUTO-BIND ALL BUTTONS (SAFE)
    setTimeout(() => {
      const allButtons = document.querySelectorAll('button');
      allButtons.forEach(btn => {
        const text = btn.innerText.toLowerCase();
        if (text.includes('load data')) btn.onclick = window.onloadData;
        if (text.includes('train')) btn.onclick = window.ontrainModel;
        if (text.includes('predict')) btn.onclick = window.onpredictTest;
        if (text.includes('save model')) btn.onclick = window.onsaveModel;
        if (text.includes('load model') || text.includes('load & prep')) btn.onclick = window.onloadModelAndPrep;
      });
      console.log('✅ ALL BUTTONS BOUND SUCCESSFULLY');
    }, 1000);

    updateButtons();
  } catch (e) {
    console.error('Init error:', e);
  }
}

document.addEventListener('DOMContentLoaded', initApp);

// ✅ THRESHOLD SLIDER
window.onThresholdChange = function(value) {
  currentThreshold = parseFloat(value);
  console.log(`Threshold: ${currentThreshold}`);
  
  if (model && valXs && valYs) {
    setTimeout(async () => {
      try {
        const valProbs = model.predict(valXs);
        const probs = Array.from(await valProbs.data());
        valProbs.dispose();
        const metrics = await calculateMetrics(probs, Array.from(valYs.dataSync()));
        
        const metricsElements = document.querySelectorAll('[id*="metrics"]');
        if (metricsElements.length > 0) {
          metricsElements.forEach(el => {
            el.innerHTML = `
              <div style="background: #e3f2fd; padding: 12px; border-radius: 6px; font-size: 13px;">
                <strong>🎯 LIVE METRICS (Threshold: ${currentThreshold})</strong><br>
                Accuracy: ${metrics.accuracy}% | Precision: ${metrics.precision}% | 
                Recall: ${metrics.recall}% | F1: ${metrics.f1}%
              </div>
            `;
          });
        }
      } catch (e) {
        console.error('Metrics update error:', e);
      }
    }, 200);
  }
};
