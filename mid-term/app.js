// app.js - FIXED: METRICS DISPLAY + THRESHOLD SLIDER
// ✅ Shows Accuracy, Precision, Recall, F1 + Real-time threshold adjustment

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let valXs = null;
let valYs = null;
let currentThreshold = 0.5;

// ================================================
// ✅ NEW: Metrics Calculation
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
  
  const accuracy = ((tp + tn) / (tp + tn + fp + fn) * 100).toFixed(1);
  const precision = tp > 0 ? (tp / (tp + fp) * 100).toFixed(1) : 0;
  const recall = tp > 0 ? (tp / (tp + fn) * 100).toFixed(1) : 0;
  const f1 = (tp > 0 && (tp + fp + fn) > 0) ? (2 * tp / (2 * tp + fp + fn) * 100).toFixed(1) : 0;
  
  return { accuracy, precision, recall, f1, tp, fp, tn, fn };
}

// ================================================
// ✅ FIXED: Training with Metrics Display
// ================================================
window.ontrainModel = async function() {
  if (!preprocessor) {
    alert('Load data first');
    return;
  }

  try {
    const btn = document.getElementById('train-model');
    btn.disabled = true;
    btn.innerText = 'Training...';
    const logEl = document.getElementById('training-log');
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
        onEpochEnd: async (epoch, logs) => {
          const logEl = document.getElementById('training-log');
          if (logEl) {
            logEl.innerText += `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}, acc=${(logs.acc*100).toFixed(1)}%\n`;
          }
        }
      }
    });

    xs.dispose();
    ys.dispose();

    // ✅ CALCULATE FINAL METRICS
    const valProbs = model.predict(valXs);
    const valProbsArray = Array.from(await valProbs.data());
    valProbs.dispose();

    const metrics = await calculateMetrics(valProbsArray, valYs.dataSync());
    
    // ✅ DISPLAY METRICS
    const metricsEl = document.getElementById('metrics-display') || 
                     document.querySelector('[id*="metrics"], [class*="metrics"]') ||
                     document.getElementById('metrics');
    
    if (metricsEl) {
      metricsEl.innerHTML = `
        <div style="font-size: 14px; line-height: 1.6;">
          <strong>✅ FINAL METRICS (Validation Set)</strong><br>
          🎯 Accuracy: <strong>${metrics.accuracy}%</strong><br>
          ⚡ Precision: <strong>${metrics.precision}%</strong><br>
          🔍 Recall: <strong>${metrics.recall}%</strong><br>
          ⚖️ F1 Score: <strong>${metrics.f1}%</strong><br>
          <small>TP:${metrics.tp} FP:${metrics.fp} FN:${metrics.fn} TN:${metrics.tn}</small>
        </div>
      `;
    }

    if (logEl) logEl.innerText += '✅ TRAINING COMPLETE + METRICS SHOWN!';
    updateButtons();
    alert(`✅ Model trained!\n🎯 Accuracy: ${metrics.accuracy}%\n⚖️ F1: ${metrics.f1}%`);

  } catch (e) {
    alert('Training error: ' + e.message);
  } finally {
    const btn = document.getElementById('train-model');
    if (btn) {
      btn.disabled = false;
      btn.innerText = '🚀 Train Model';
    }
  }
};

// ================================================
// ✅ NEW: Threshold Slider Handler
// ================================================
window.onThresholdChange = function(value) {
  currentThreshold = parseFloat(value);
  console.log(`Threshold changed to: ${currentThreshold}`);
  
  // Update slider display
  const slider = document.querySelector('input[type="range"]');
  const thresholdLabel = document.querySelector('[for="threshold"], .threshold-label');
  if (thresholdLabel) {
    thresholdLabel.innerText = `Threshold: ${currentThreshold}`;
  }
  
  // Recalculate metrics if model loaded
  if (model && valXs && valYs) {
    updateMetricsDisplay();
  }
};

// ================================================
// ✅ NEW: Update Metrics Display
// ================================================
async function updateMetricsDisplay() {
  if (!model || !valXs || !valYs) return;
  
  try {
    const valProbs = model.predict(valXs);
    const valProbsArray = Array.from(await valProbs.data());
    valProbs.dispose();
    
    const metrics = await calculateMetrics(valProbsArray, Array.from(valYs.dataSync()));
    
    const metricsEl = document.getElementById('metrics-display') || 
                     document.querySelector('[id*="metrics"]');
    
    if (metricsEl) {
      metricsEl.innerHTML = `
        <div style="font-size: 14px; line-height: 1.6;">
          <strong>🎯 METRICS (Threshold: ${currentThreshold})</strong><br>
          🎯 Accuracy: <strong>${metrics.accuracy}%</strong><br>
          ⚡ Precision: <strong>${metrics.precision}%</strong><br>
          🔍 Recall: <strong>${metrics.recall}%</strong><br>
          ⚖️ F1 Score: <strong>${metrics.f1}%</strong>
        </div>
      `;
    }
  } catch (e) {
    console.error('Metrics update error:', e);
  }
}

// ================================================
// FIXED: Predict with Threshold
// ================================================
window.onpredictTest = async function() {
  if (!model || !testData) {
    alert('Train model + load test data first');
    return;
  }

  try {
    const btn = document.getElementById('predict-test');
    btn.disabled = true;
    btn.innerText = 'Predicting...';

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

    const edaEl = document.getElementById('eda-output');
    if (edaEl) {
      edaEl.innerText += `\n✅ Predictions: ${approvedCount}/${probs.length} (${((approvedCount/probs.length)*100).toFixed(1)}%) | Threshold: ${currentThreshold}`;
    }

    alert(`✅ SUCCESS! ${approvedCount} approvals (${((approvedCount/probs.length)*100).toFixed(1)}%) | Threshold: ${currentThreshold}`);

  } catch (e) {
    alert('Prediction error: ' + e.message);
  } finally {
    const btn = document.getElementById('predict-test');
    if (btn) {
      btn.disabled = false;
      btn.innerText = '🔮 Predict';
    }
  }
};

// ================================================
// ALL OTHER FUNCTIONS (unchanged)
// ================================================
window.onloadData = async function() {
  try {
    const trainFile = document.getElementById('train-file').files[0];
    if (!trainFile) {
      alert('Please select train.csv');
      return;
    }

    document.getElementById('load-data').disabled = true;
    document.getElementById('load-data').innerText = 'Loading...';
    document.getElementById('eda-output').innerText = 'Parsing CSV...';

    const text = await trainFile.text();
    const parsed = parseSimpleCSV(text);
    
    trainHeaders = parsed[0];
    trainData = parsed.slice(1);

    if (!trainHeaders.includes('loan_status')) {
      alert('Missing loan_status column');
      return;
    }

    preprocessor = new SimplePreprocessor();
    preprocessor.fit(trainData, trainHeaders);

    const targetIdx = trainHeaders.indexOf('loan_status');
    const approved = trainData.filter(row => row[targetIdx] === '1').length;
    const total = trainData.length;

    document.getElementById('eda-output').innerText = 
      `✅ SUCCESS!\nRows: ${total}\nApproved: ${((approved/total)*100).toFixed(1)}%\nFeatures: ${preprocessor.featureOrder.length}`;

    const { train, val } = simpleSplit(trainData);
    const valProcessed = preprocessor.transform(val, true);
    
    if (valXs) valXs.dispose();
    if (valYs) valYs.dispose();
    valXs = tf.tensor2d(valProcessed.features);
    valYs = tf.tensor1d(valProcessed.targets);

    const testFile = document.getElementById('test-file').files[0];
    if (testFile) {
      const testText = await testFile.text();
      const testParsed = parseSimpleCSV(testText);
      testData = testParsed.slice(1);
      document.getElementById('eda-output').innerText += `\nTest: ${testData.length} rows`;
    }

    updateButtons();
    alert(`✅ Loaded ${total} samples!`);

  } catch (e) {
    alert('Load error: ' + e.message);
  } finally {
    document.getElementById('load-data').disabled = false;
    document.getElementById('load-data').innerText = '📊 Load Data';
  }
};

// Save/Load functions remain the same...
window.onsaveModel = async function() {
  if (!model || !preprocessor) {
    alert('Train model first');
    return;
  }

  try {
    const btn = document.getElementById('save-model');
    if (btn) {
      btn.disabled = true;
      btn.innerText = 'Saving...';
    }

    // Model JSON
    const modelJsonBlob = new Blob([JSON.stringify(model.toJSON())], {type: 'application/json'});
    downloadFile('model.json', modelJsonBlob);

    // Weights.bin
    const weightTensors = model.getWeights();
    const weightBlobs = await Promise.all(weightTensors.map(async (tensor) => {
      const data = await tensor.data();
      return new Blob([data.buffer]);
    }));
    const weightsBlob = new Blob(weightBlobs);
    downloadFile('weights.bin', weightsBlob);
    weightTensors.forEach(tensor => tensor.dispose());

    // Preprocessor
    const prepJSON = preprocessor.toJSON();
    const prepBlob = new Blob([JSON.stringify(prepJSON, null, 2)], {type: 'application/json'});
    downloadFile('preprocessor.json', prepBlob);

    alert('✅ ALL FILES DOWNLOADED!\n📥 model.json + weights.bin + preprocessor.json');

  } catch (e) {
    alert('Save error: ' + e.message);
  } finally {
    const btn = document.getElementById('save-model');
    if (btn) {
      btn.disabled = false;
      btn.innerText = '💾 Save Model';
    }
  }
};

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

// ... (other utility functions unchanged)

async function initApp() {
  await tf.ready();
  console.log('✅ TensorFlow.js ready');

  // Bind threshold slider
  const slider = document.querySelector('input[type="range"]');
  if (slider) {
    slider.oninput = () => window.onThresholdChange(slider.value);
    currentThreshold = parseFloat(slider.value);
  }

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
    console.log('✅ ALL BUTTONS + SLIDER BOUND');
  }, 1000);

  updateButtons();
}

document.addEventListener('DOMContentLoaded', initApp);
