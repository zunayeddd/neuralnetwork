// app.js - FIXED: Load Model & Prep - SIMPLE & BULLETPROOF
// NO MORE "Please select preprocessor.json" ERROR

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let valXs = null;
let valYs = null;

// ================================================
// PREPROCESSOR CLASS (unchanged)
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
      });
      
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

    data.forEach((row, rowIndex) => {
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
// ✅ FIXED: Load Model & Prep - SIMPLE VERSION
// NO FILE DETECTION PROBLEMS
// ================================================
window.onloadModelAndPrep = async function() {
  try {
    // SIMPLIFIED: Just find ANY selected JSON files
    const allFileInputs = document.querySelectorAll('input[type="file"]');
    let modelJsonFile = null;
    let modelWeightsFile = null;
    let prepJsonFile = null;

    allFileInputs.forEach(input => {
      if (input.files[0]) {
        const filename = input.files[0].name.toLowerCase();
        if (filename.includes('model.json') || filename.includes('model') && filename.endsWith('.json')) {
          modelJsonFile = input.files[0];
        } else if (filename.includes('weights') || filename.endsWith('.bin')) {
          modelWeightsFile = input.files[0];
        } else if (filename.includes('prep') || filename.includes('preprocess') || filename.endsWith('.json')) {
          prepJsonFile = input.files[0];
        }
      }
    });

    console.log('Detected files:', {
      modelJson: modelJsonFile?.name,
      modelWeights: modelWeightsFile?.name,
      prepJson: prepJsonFile?.name
    });

    // ✅ FLEXIBLE CHECKING - WORKS EVEN IF FILES MISSING
    if (!prepJsonFile) {
      alert('⚠️ No preprocessor file found. Loading model only...');
      // Continue without preprocessor - use default
    } else {
      // Load preprocessor FIRST
      const prepText = await prepJsonFile.text();
      const prepJSON = JSON.parse(prepText);
      preprocessor = SimplePreprocessor.fromJSON(prepJSON);
      console.log('✅ Preprocessor loaded');
    }

    if (!modelJsonFile || !modelWeightsFile) {
      alert('❌ Need model.json AND weights.bin files');
      return;
    }

    // Load model
    const modelFiles = [
      {path: 'model.json', data: modelJsonFile},
      {path: 'weights.bin', data: modelWeightsFile}
    ];
    
    model = await tf.loadLayersModel(tf.io.browserFiles(modelFiles));
    console.log('✅ Model loaded');

    // Cleanup
    if (valXs) valXs.dispose();
    if (valYs) valYs.dispose();

    updateButtons();
    
    if (prepJsonFile) {
      alert(`✅ SUCCESS!\n📊 Features: ${preprocessor.headers.length}\n🎯 Model ready!`);
    } else {
      alert(`✅ Model loaded!\n⚠️ No preprocessor - predictions may not work perfectly`);
    }

  } catch (error) {
    console.error('Load error:', error);
    alert(`❌ Load failed: ${error.message}`);
  }
};

// ================================================
// ALL OTHER BUTTONS (unchanged - WORKING)
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

window.ontrainModel = async function() {
  if (!preprocessor) {
    alert('Load data first');
    return;
  }

  try {
    const btn = document.getElementById('train-model');
    btn.disabled = true;
    btn.innerText = 'Training...';
    document.getElementById('training-log').innerText = '';

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
          const logEl = document.getElementById('training-log');
          if (logEl) logEl.innerText += `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}\n`;
        }
      }
    });

    xs.dispose();
    ys.dispose();

    document.getElementById('training-log').innerText += '✅ TRAINING COMPLETE!';
    updateButtons();
    alert('✅ Model trained!');

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
      const pred = prob > 0.5 ? 1 : 0;
      if (pred === 1) approvedCount++;
      submission.push([`App_${i+1}`, pred]);
    });

    downloadCSV('submission.csv', submission);

    const edaEl = document.getElementById('eda-output');
    if (edaEl) {
      edaEl.innerText += `\n✅ Predictions: ${approvedCount}/${probs.length} (${((approvedCount/probs.length)*100).toFixed(1)}%)`;
    }

    alert(`✅ SUCCESS! ${approvedCount} approvals`);

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

    await model.save('downloads://loan-approval-model');
    
    const prepJSON = preprocessor.toJSON();
    const prepBlob = new Blob([JSON.stringify(prepJSON, null, 2)], {type: 'application/json'});
    const prepUrl = URL.createObjectURL(prepBlob);
    const prepLink = document.createElement('a');
    prepLink.href = prepUrl;
    prepLink.download = 'preprocessor.json';
    prepLink.click();
    URL.revokeObjectURL(prepUrl);

    alert('✅ SAVED! Check Downloads folder');

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

window.onsavePreprocessor = function() {
  if (!preprocessor) {
    alert('Load data first');
    return;
  }

  try {
    const prepJSON = preprocessor.toJSON();
    const prepBlob = new Blob([JSON.stringify(prepJSON, null, 2)], {type: 'application/json'});
    const prepUrl = URL.createObjectURL(prepBlob);
    const prepLink = document.createElement('a');
    prepLink.href = prepUrl;
    prepLink.download = 'preprocessor.json';
    prepLink.click();
    URL.revokeObjectURL(prepUrl);

    alert('✅ Preprocessor saved!');
  } catch (e) {
    alert('Save error: ' + e.message);
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
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        row.push(field.trim());
        field = '';
      } else {
        field += char;
      }
    }
    row.push(field.trim());
    return row;
  });
}

function simpleSplit(data, ratio = 0.8) {
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const split = Math.floor(shuffled.length * ratio);
  return {
    train: shuffled.slice(0, split),
    val: shuffled.slice(split)
  };
}

function downloadCSV(filename, rows) {
  const csv = rows.map(row => 
    row.map(cell => `"${cell}"`).join(',')
  ).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function updateButtons() {
  const hasData = !!preprocessor;
  const hasModel = !!model;
  const hasTest = !!testData;
  
  ['train-model', 'save-prep'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.disabled = !hasData;
  });
  
  const predictBtn = document.getElementById('predict-test');
  if (predictBtn) predictBtn.disabled = !hasModel || !hasTest;
  
  const saveModelBtn = document.getElementById('save-model');
  if (saveModelBtn) saveModelBtn.disabled = !hasModel;
}

// ================================================
// BULLETPROOF INIT
// ================================================
async function initApp() {
  try {
    await tf.ready();
    console.log('✅ TensorFlow.js ready');

    // AUTO-BIND BUTTONS BY TEXT
    setTimeout(() => {
      const buttons = document.querySelectorAll('button');
      buttons.forEach(btn => {
        const text = btn.innerText.toLowerCase();
        if (text.includes('load data')) btn.onclick = window.onloadData;
        if (text.includes('train')) btn.onclick = window.ontrainModel;
        if (text.includes('predict')) btn.onclick = window.onpredictTest;
        if (text.includes('save model')) btn.onclick = window.onsaveModel;
        if (text.includes('save prep')) btn.onclick = window.onsavePreprocessor;
        if (text.includes('load model') || text.includes('load & prep')) btn.onclick = window.onloadModelAndPrep;
      });
      console.log('✅ ALL BUTTONS BOUND');
    }, 1000);

    updateButtons();
  } catch (e) {
    console.error('Init error:', e);
  }
}

document.addEventListener('DOMContentLoaded', initApp);
