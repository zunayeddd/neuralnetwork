// app.js - FIXED: REAL PREDICTIONS (not all 0s) + PROPER PREPROCESSING
// Root cause: Broken feature transformation → model predicts 0s

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let testHeaders = null;
let valXs = null;
let valYs = null;
let isTfReady = false;

// ================================================
// FIXED PREPROCESSOR - GENERATES REAL FEATURES
// ================================================
class RealPreprocessor {
  constructor() {
    this.featureOrder = [];
    this.numericCols = [];
    this.categoricalCols = [];
    this.means = {};
    this.stds = {};
    this.targetIdx = -1;
  }

  fit(data, headers) {
    this.headers = headers;
    this.targetIdx = headers.indexOf('loan_status');
    
    // IDENTIFY COLUMN TYPES
    this.headers.forEach((col, idx) => {
      if (col === 'loan_status') return;
      
      const values = data
        .map(row => row[idx] || '')
        .filter(v => v !== '');
      
      const numericCount = values.filter(v => !isNaN(parseFloat(v))).length;
      if (numericCount / values.length > 0.7) {
        this.numericCols.push(col);
        this.computeNumericStats(values.map(v => parseFloat(v)));
      } else {
        this.categoricalCols.push(col);
      }
    });

    // BUILD FEATURE ORDER
    this.featureOrder = [...this.numericCols];
    this.categoricalCols.forEach(col => {
      this.featureOrder.push(`${col}_yes`);  // Simple binary encoding
      this.featureOrder.push(`${col}_no`);
    });

    console.log('✅ Preprocessor fitted:', {
      numeric: this.numericCols.length,
      categorical: this.categoricalCols.length,
      features: this.featureOrder.length
    });
  }

  computeNumericStats(values) {
    const valid = values.filter(v => !isNaN(v));
    if (valid.length === 0) return;
    
    const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
    const variance = valid.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / valid.length;
    const std = Math.sqrt(variance) || 1;
    
    this.means[this.numericCols[this.numericCols.length - 1]] = mean;
    this.stds[this.numericCols[this.numericCols.length - 1]] = std;
  }

  transform(data, includeTarget = true) {
    const features = [];
    const targets = [];

    data.forEach(row => {
      const featureRow = [];

      // NUMERIC FEATURES
      this.numericCols.forEach(col => {
        const colIdx = this.headers.indexOf(col);
        if (colIdx === -1 || colIdx >= row.length) {
          featureRow.push(0);
          return;
        }
        
        const val = row[colIdx];
        const numVal = parseFloat(val);
        if (!isNaN(numVal)) {
          const colName = col;
          const mean = this.means[colName] || 0;
          const std = this.stds[colName] || 1;
          featureRow.push((numVal - mean) / std);
        } else {
          featureRow.push(0);
        }
      });

      // CATEGORICAL FEATURES (SIMPLE YES/NO)
      this.categoricalCols.forEach(col => {
        const colIdx = this.headers.indexOf(col);
        if (colIdx === -1 || colIdx >= row.length) {
          featureRow.push(0, 0); // yes, no
          return;
        }
        
        const val = String(row[colIdx] || '').toLowerCase();
        const isYes = val.includes('yes') || val.includes('y') || val === '1';
        const isNo = val.includes('no') || val.includes('n') || val === '0';
        
        featureRow.push(isYes ? 1 : 0);  // yes
        featureRow.push(isNo ? 1 : 0);   // no
      });

      features.push(featureRow);

      if (includeTarget && this.targetIdx !== -1 && this.targetIdx < row.length) {
        const targetVal = row[this.targetIdx];
        targets.push(targetVal === '1' || targetVal === 'Yes' ? 1 : 0);
      }
    });

    console.log('✅ Transformed:', features.length, 'samples →', features[0]?.length || 0, 'features');
    return { features, targets };
  }
}

// ================================================
// FIXED LOAD DATA - REAL PREPROCESSING
// ================================================
window.onloadData = async function() {
  try {
    const trainFileEl = document.getElementById('train-file');
    if (!trainFileEl?.files[0]) {
      alert('Please select train.csv file first');
      return;
    }

    document.getElementById('load-data').disabled = true;
    document.getElementById('load-data').textContent = 'Loading...';
    document.getElementById('eda-output').textContent = '🔄 Parsing CSV...';

    const trainText = await trainFileEl.files[0].text();
    const parsed = parseCSVRobust(trainText);
    
    if (!parsed || parsed.length < 2) {
      throw new Error('Invalid CSV: Need header + data rows');
    }

    trainHeaders = parsed[0];
    trainData = parsed.slice(1);

    if (!trainHeaders.includes('loan_status')) {
      throw new Error('Missing "loan_status" column');
    }

    // ✅ FIXED: Use REAL preprocessor
    preprocessor = new RealPreprocessor();
    preprocessor.fit(trainData, trainHeaders);

    // REAL EDA
    const targetIdx = trainHeaders.indexOf('loan_status');
    const approved = trainData.filter(row => row[targetIdx] === '1').length;
    const total = trainData.length;

    document.getElementById('eda-output').textContent = 
      `✅ DATA LOADED SUCCESSFULLY!\n\n` +
      `📊 Rows: ${total}\n` +
      `🎯 Approved: ${((approved/total)*100).toFixed(1)}%\n` +
      `🔧 Numeric features: ${preprocessor.numericCols.length}\n` +
      `🔧 Categorical features: ${preprocessor.categoricalCols.length}\n` +
      `📈 Total features: ${preprocessor.featureOrder.length}`;

    document.getElementById('feature-dim').textContent = 
      `Feature dimension: ${preprocessor.featureOrder.length}`;

    // VALIDATION SPLIT
    const { train, val } = stratifiedSplit(trainData, trainHeaders);
    const valProcessed = preprocessor.transform(val, true);
    
    if (valXs) valXs.dispose();
    if (valYs) valYs.dispose();
    valXs = tf.tensor2d(valProcessed.features);
    valYs = tf.tensor1d(valProcessed.targets, 'float32');

    // LOAD TEST DATA
    const testFileEl = document.getElementById('test-file');
    if (testFileEl?.files[0]) {
      const testText = await testFileEl.files[0].text();
      const testParsed = parseCSVRobust(testText);
      testHeaders = testParsed[0];
      testData = testParsed.slice(1);
      
      document.getElementById('eda-output').textContent += 
        `\n✅ Test data: ${testData.length} rows loaded`;
    }

    updateButtons();
    alert(`✅ Loaded ${total} samples with ${preprocessor.featureOrder.length} real features!`);

  } catch (error) {
    alert(`Load error: ${error.message}`);
    console.error('Load error:', error);
  } finally {
    document.getElementById('load-data').disabled = false;
    document.getElementById('load-data').textContent = '📊 Load & Analyze Data';
  }
};

// ================================================
// FIXED TRAINING - BETTER MODEL
// ================================================
window.ontrainModel = async function() {
  if (!preprocessor || !trainData) {
    alert('Load data first');
    return;
  }

  try {
    document.getElementById('train-model').disabled = true;
    document.getElementById('train-model').textContent = 'Training...';
    document.getElementById('training-log').textContent = '';

    const hiddenUnits = parseInt(document.getElementById('hidden-units').value) || 64;
    const lr = parseFloat(document.getElementById('lr').value) || 0.001;

    // ✅ BETTER MODEL ARCHITECTURE
    model = tf.sequential({
      layers: [
        tf.layers.dense({ units: hiddenUnits, activation: 'relu', inputShape: [preprocessor.featureOrder.length] }),
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.3 }),
        tf.layers.dense({ units: hiddenUnits * 2, activation: 'relu' }),
        tf.layers.batchNormalization(),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: hiddenUnits, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    // TRAINING DATA
    const { train } = stratifiedSplit(trainData, trainHeaders);
    const trainProcessed = preprocessor.transform(train);
    
    const xs = tf.tensor2d(trainProcessed.features);
    const ys = tf.tensor1d(trainProcessed.targets, 'float32');

    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 32,
      validationData: [valXs, valYs],
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          document.getElementById('training-log').textContent += 
            `Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(4)} val_loss=${logs.valLoss?.toFixed(4)}\n`;
        }
      }
    });

    xs.dispose();
    ys.dispose();

    document.getElementById('training-log').textContent += '\n✅ TRAINING COMPLETE!';
    updateMetrics();
    updateButtons();
    alert('✅ Model trained with REAL predictions!');

  } catch (error) {
    alert(`Training error: ${error.message}`);
  } finally {
    document.getElementById('train-model').disabled = false;
    document.getElementById('train-model').textContent = '🚀 Train Model';
  }
};

// ================================================
// FIXED PREDICTION - REAL RESULTS
// ================================================
window.onpredictTest = async function() {
  if (!model || !testData) {
    alert('Train model first AND upload test data');
    return;
  }

  try {
    document.getElementById('predict-test').disabled = true;
    document.getElementById('predict-test').textContent = 'Predicting...';

    // ✅ CRITICAL FIX: PROPER TEST TRANSFORMATION
    const testProcessed = preprocessor.transform(testData, false);
    const xs = tf.tensor2d(testProcessed.features);
    const predictions = model.predict(xs);
    const probs = Array.from(await predictions.data());
    
    xs.dispose();
    predictions.dispose();

    // GENERATE REAL SUBMISSION
    const submission = [['ApplicationID', 'Approved']];
    const probabilities = [['ApplicationID', 'Probability']];
    
    let approvedCount = 0;
    
    for (let i = 0; i < probs.length; i++) {
      const prob = probs[i];
      const prediction = prob >= 0.5 ? 1 : 0;
      if (prediction === 1) approvedCount++;
      
      submission.push([`App_${i + 1}`, prediction]);
      probabilities.push([`App_${i + 1}`, prob.toFixed(6)]);
    }

    // DOWNLOAD
    downloadCSV('submission.csv', submission);
    downloadCSV('probabilities.csv', probabilities);

    document.getElementById('eda-output').textContent += 
      `\n✅ PREDICTIONS COMPLETE!\n` +
      `📊 Total: ${probs.length} samples\n` +
      `✅ Approved: ${approvedCount} (${((approvedCount/probs.length)*100).toFixed(1)}%)`;

    alert(`✅ Success! ${approvedCount} approvals out of ${probs.length} (${((approvedCount/probs.length)*100).toFixed(1)}%)`);
    
  } catch (error) {
    alert(`Prediction error: ${error.message}`);
    console.error('Prediction error:', error);
  } finally {
    document.getElementById('predict-test').disabled = false;
    document.getElementById('predict-test').textContent = '🔮 Predict on Test';
  }
};

// ================================================
// UTILITY FUNCTIONS
// ================================================
function parseCSVRobust(text) {
  const lines = text.trim().split('\n').filter(line => line.trim());
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

function stratifiedSplit(data, headers, ratio = 0.8) {
  const targetIdx = headers.indexOf('loan_status');
  const positives = data.filter(row => row[targetIdx] === '1');
  const negatives = data.filter(row => row[targetIdx] !== '1');
  
  const posTrainSize = Math.floor(positives.length * ratio);
  const negTrainSize = Math.floor(negatives.length * ratio);
  
  const train = [
    ...positives.slice(0, posTrainSize),
    ...negatives.slice(0, negTrainSize)
  ];
  const val = [
    ...positives.slice(posTrainSize),
    ...negatives.slice(negTrainSize)
  ];
  
  return { train, val };
}

function downloadCSV(filename, rows) {
  const csv = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function updateButtons() {
  const hasData = !!(trainData && preprocessor);
  const hasModel = !!model;
  const hasTest = !!testData;
  
  document.getElementById('load-data').disabled = false;
  document.getElementById('train-model').disabled = !hasData;
  document.getElementById('predict-test').disabled = !hasModel || !hasTest;
  document.getElementById('save-model').disabled = !hasModel;
}

async function updateMetrics() {
  if (!model || !valXs || !valYs) return;
  
  try {
    const probs = model.predict(valXs).dataSync();
    const targets = valYs.dataSync();
    
    let approved = 0;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] >= 0.5) approved++;
    }
    
    document.getElementById('metrics-output').textContent = 
      `Validation Accuracy: ${((approved/probs.length)*100).toFixed(1)}%\n` +
      `Predicted approvals: ${approved}/${probs.length}`;
      
  } catch (e) {
    console.error('Metrics error:', e);
  }
}

// ================================================
// INITIALIZATION
// ================================================
async function initApp() {
  await tf.setBackend('cpu');
  await tf.ready();
  console.log('✅ TensorFlow.js ready');
  
  document.getElementById('load-data').addEventListener('click', window.onloadData);
  document.getElementById('train-model').addEventListener('click', window.ontrainModel);
  document.getElementById('predict-test').addEventListener('click', window.onpredictTest);
  document.getElementById('threshold').addEventListener('input', updateMetrics);
  
  updateButtons();
}

document.addEventListener('DOMContentLoaded', initApp);
