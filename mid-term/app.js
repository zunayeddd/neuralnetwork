// app.js - SIMPLIFIED: NO RECOVERY MODE + BULLETPROOF + REAL PREDICTIONS
// Removed ALL complex error handling - Pure, simple, working code

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let valXs = null;
let valYs = null;

// ================================================
// SIMPLE WORKING PREPROCESSOR
// ================================================
class SimplePreprocessor {
  constructor() {
    this.featureOrder = [];
    this.means = {};
    this.stds = {};
  }

  fit(data, headers) {
    this.headers = headers.filter(h => h !== 'loan_status');
    
    // Simple: Use ALL columns as numeric (most reliable)
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
    console.log(`✅ Preprocessor ready: ${this.featureOrder.length} features`);
  }

  transform(data, includeTarget = true) {
    const features = [];
    const targets = [];

    data.forEach(row => {
      const featureRow = [];
      
      this.headers.forEach(col => {
        const colIdx = this.headers.indexOf(col);
        if (colIdx === -1 || colIdx >= row.length) {
          featureRow.push(0);
          return;
        }
        
        const rawVal = row[colIdx];
        const numVal = parseFloat(rawVal) || 0;
        const mean = this.means[col] || 0;
        const std = this.stds[col] || 1;
        
        featureRow.push((numVal - mean) / std);
      });
      
      features.push(featureRow);

      if (includeTarget) {
        const targetIdx = this.headers.length; // loan_status is after features
        const targetVal = row[targetIdx];
        targets.push(targetVal === '1' ? 1 : 0);
      }
    });

    return { features, targets };
  }
}

// ================================================
// SIMPLE LOAD DATA
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
    
    if (parsed.length < 2) {
      alert('Invalid CSV file');
      return;
    }

    trainHeaders = parsed[0];
    trainData = parsed.slice(1);

    if (!trainHeaders.includes('loan_status')) {
      alert('Missing loan_status column');
      return;
    }

    // Create preprocessor
    preprocessor = new SimplePreprocessor();
    preprocessor.fit(trainData, trainHeaders);

    // Show EDA
    const targetIdx = trainHeaders.indexOf('loan_status');
    const approved = trainData.filter(row => row[targetIdx] === '1').length;
    const total = trainData.length;

    document.getElementById('eda-output').innerText = 
      `✅ SUCCESS!\n` +
      `Rows: ${total}\n` +
      `Approved: ${((approved/total)*100).toFixed(1)}%\n` +
      `Features: ${preprocessor.featureOrder.length}`;

    document.getElementById('feature-dim').innerText = 
      `Features: ${preprocessor.featureOrder.length}`;

    // Validation split
    const { train, val } = simpleSplit(trainData);
    const valProcessed = preprocessor.transform(val, true);
    
    if (valXs) valXs.dispose();
    if (valYs) valYs.dispose();
    valXs = tf.tensor2d(valProcessed.features);
    valYs = tf.tensor1d(valProcessed.targets);

    // Load test data
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

// ================================================
// SIMPLE TRAINING
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
    document.getElementById('training-log').innerText = '';

    // Simple reliable model
    model = tf.sequential({
      layers: [
        tf.layers.dense({units: 64, activation: 'relu', inputShape: [preprocessor.featureOrder.length]}),
        tf.layers.dropout({rate: 0.3}),
        tf.layers.dense({units: 32, activation: 'relu'}),
        tf.layers.dropout({rate: 0.2}),
        tf.layers.dense({units: 1, activation: 'sigmoid'})
      ]
    });

    model.compile({
      optimizer: 'adam',
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // Train data
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
          document.getElementById('training-log').innerText += 
            `Epoch ${epoch+1}: loss=${logs.loss.toFixed(4)}\n`;
        }
      }
    });

    xs.dispose();
    ys.dispose();

    document.getElementById('training-log').innerText += '✅ DONE!';
    updateButtons();
    alert('✅ Training complete!');

  } catch (e) {
    alert('Training error: ' + e.message);
  } finally {
    document.getElementById('train-model').disabled = false;
    document.getElementById('train-model').innerText = '🚀 Train Model';
  }
};

// ================================================
// SIMPLE PREDICTION
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

    // Create submission
    const submission = [['ApplicationID', 'Approved']];
    let approvedCount = 0;

    probs.forEach((prob, i) => {
      const pred = prob > 0.5 ? 1 : 0;
      if (pred === 1) approvedCount++;
      submission.push([`App_${i+1}`, pred]);
    });

    // Download
    downloadCSV('submission.csv', submission);

    document.getElementById('eda-output').innerText += 
      `\n✅ Predictions: ${approvedCount}/${probs.length} approved (${((approvedCount/probs.length)*100).toFixed(1)}%)`;

    alert(`✅ Done! ${approvedCount} approvals`);

  } catch (e) {
    alert('Prediction error: ' + e.message);
  } finally {
    document.getElementById('predict-test').disabled = false;
    document.getElementById('predict-test').innerText = '🔮 Predict';
  }
};

// ================================================
// SIMPLE UTILITIES
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
  
  document.getElementById('train-model').disabled = !hasData;
  document.getElementById('predict-test').disabled = !hasModel || !hasTest;
}

// ================================================
// SIMPLE INIT
// ================================================
async function init() {
  try {
    await tf.ready();
    console.log('✅ Ready');
    
    document.getElementById('load-data').onclick = window.onloadData;
    document.getElementById('train-model').onclick = window.ontrainModel;
    document.getElementById('predict-test').onclick = window.onpredictTest;
    
  } catch (e) {
    alert('Init error');
  }
}

init();
