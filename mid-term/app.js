// app.js - FIXED: row.join is not a function error resolved
// UI, model training, evaluation, and prediction for Loan Approval Predictor

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let testHeaders = null;
let valXs = null;
let valYs = null;

// Initialize TensorFlow.js and UI event listeners
async function initApp() {
  try {
    await tf.setBackend('cpu');
    await tf.ready();
    console.log('TensorFlow.js ready with backend:', tf.getBackend());
    
    document.getElementById('load-data').addEventListener('click', loadData);
    document.getElementById('train-model').addEventListener('click', trainModel);
    document.getElementById('predict-test').addEventListener('click', predictTest);
    document.getElementById('save-model').addEventListener('click', saveModel);
    document.getElementById('save-prep').addEventListener('click', savePreprocessor);
    document.getElementById('load-model').addEventListener('click', loadModel);
    document.getElementById('reset').addEventListener('click', reset);
    document.getElementById('threshold').addEventListener('input', updateMetrics);
    document.getElementById('toggle-visor').addEventListener('click', () => {
      if (tfvis && tfvis.visor) {
        tfvis.visor().toggle();
      }
    });
    
    enableButtons();
  } catch (error) {
    console.error('Failed to initialize TensorFlow.js:', error);
    alert('Failed to initialize TensorFlow.js. Please refresh the page.');
  }
}

document.addEventListener('DOMContentLoaded', initApp);

// ✅ FIXED: Robust data loading with proper array validation
async function loadData() {
  try {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) throw new Error('Please upload train.csv');

    document.getElementById('load-data').disabled = true;
    document.getElementById('eda-output').textContent = 'Loading train data...';

    // Load TRAIN data
    const trainReader = new FileReader();
    trainReader.onload = async (e) => {
      try {
        const rawData = e.target.result;
        trainData = parseCSV(rawData);
        
        if (!Array.isArray(trainData) || trainData.length < 2) {
          throw new Error('Invalid train CSV format - no data rows found');
        }
        
        trainHeaders = Array.isArray(trainData[0]) ? trainData[0] : trainData[0].split(',');
        trainData = trainData.slice(1).filter(row => 
          Array.isArray(row) && row.length > 0
        );

        console.log('Train headers:', trainHeaders);
        console.log('Train rows:', trainData.length);

        if (!trainHeaders.includes('loan_status')) {
          throw new Error('train.csv missing loan_status column');
        }

        // Create preprocessor
        preprocessor = new Preprocessor();

        // Engineer features for train data if enabled
        if (document.getElementById('engineer-features').checked) {
          const result = preprocessor.engineerFeatures(trainData, trainHeaders);
          trainData = result.data;
          trainHeaders = result.headers;
        }

        showEDA();
        fitPreprocessor();
        createValidationSplit();

        // Load TEST data if provided
        if (testFile) {
          loadTestData(testFile);
        } else {
          document.getElementById('eda-output').textContent += '\n⚠️ No test file provided';
          enableButtons();
        }

      } catch (err) {
        alert(`Error loading train data: ${err.message}`);
        console.error('Train data error:', err);
      } finally {
        document.getElementById('load-data').disabled = false;
      }
    };
    trainReader.readAsText(trainFile);

  } catch (err) {
    alert(`Error: ${err.message}`);
    document.getElementById('load-data').disabled = false;
  }
}

// ✅ FIXED: Robust test data loading
function loadTestData(testFile) {
  const testReader = new FileReader();
  testReader.onload = (e) => {
    try {
      const rawData = e.target.result;
      testData = parseCSV(rawData);
      
      if (!Array.isArray(testData) || testData.length < 2) {
        throw new Error('Invalid test CSV format');
      }
      
      testHeaders = Array.isArray(testData[0]) ? testData[0] : testData[0].split(',');
      testData = testData.slice(1).filter(row => 
        Array.isArray(row) && row.length > 0
      );

      // Engineer features for test data if enabled
      if (document.getElementById('engineer-features').checked && preprocessor) {
        const result = preprocessor.engineerFeatures(testData, testHeaders);
        testData = result.data;
        testHeaders = result.headers;
      }
      
      console.log('✅ Test data loaded:', testData.length, 'rows');
      document.getElementById('eda-output').textContent += `\n✅ Test data loaded: ${testData.length} rows`;
      enableButtons();
      
    } catch (err) {
      console.error('Test data error:', err);
      document.getElementById('eda-output').textContent += `\n❌ Test data error: ${err.message}`;
      enableButtons();
    }
  };
  testReader.readAsText(testFile);
}

function showEDA() {
  const edaOutput = document.getElementById('eda-output');
  const targetIdx = trainHeaders.indexOf('loan_status');
  const targetCount = trainData.reduce((sum, row) => {
    const targetVal = row[targetIdx];
    return sum + (targetVal === '1' || targetVal === 'Y' || targetVal === 'yes' ? 1 : 0);
  }, 0);
  
  const targetRate = targetCount / trainData.length;
  const shape = `${trainData.length} rows, ${trainHeaders.length} columns`;
  
  edaOutput.textContent = `✅ Train data loaded!\nShape: ${shape}\nTarget Rate: ${(targetRate * 100).toFixed(2)}%\n\nPreprocessing...`;
}

function fitPreprocessor() {
  preprocessor.fit(trainData, trainHeaders);
  document.getElementById('feature-dim').textContent = `Feature Dimension: ${preprocessor.featureOrder.length}`;
  
  // Data preview
  const previewTable = document.getElementById('preview-table');
  previewTable.innerHTML = '';
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  Math.min(8, trainHeaders.length).forEach((h, i) => {
    const th = document.createElement('th');
    th.textContent = trainHeaders[i];
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  previewTable.appendChild(thead);
  
  const tbody = document.createElement('tbody');
  Math.min(10, trainData.length).forEach((row, rowIdx) => {
    const tr = document.createElement('tr');
    Math.min(8, row.length).forEach((cell, cellIdx) => {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  previewTable.appendChild(tbody);
}

function createValidationSplit() {
  const { train, val } = stratifiedSplit(trainData, trainHeaders);
  const valProcessed = preprocessor.transform(val);
  
  if (valXs) valXs.dispose();
  if (valYs) valYs.dispose();
  
  valXs = tf.tensor2d(valProcessed.features);
  valYs = tf.tensor1d(valProcessed.targets, 'float32');
}

// ✅ FIXED: Prediction with robust array handling
async function predictTest() {
  if (!model) {
    alert('Please train a model first');
    return;
  }
  
  if (!testData || !Array.isArray(testData) || testData.length === 0) {
    alert('No valid test data loaded. Please upload test.csv and click "Load Data"');
    return;
  }

  try {
    document.getElementById('predict-test').disabled = true;
    document.getElementById('eda-output').textContent = '🎯 Generating predictions...';

    // ✅ FIXED: Ensure features is always array of arrays
    const processed = preprocessor.transform(testData, false, false);
    if (!Array.isArray(processed.features) || processed.features.length === 0) {
      throw new Error('No valid features extracted from test data');
    }

    const xs = tf.tensor2d(processed.features);
    const probs = model.predict(xs).dataSync();
    const threshold = parseFloat(document.getElementById('threshold').value);
    const preds = probs.map(p => p >= threshold ? 1 : 0);

    // ✅ FIXED: Robust CSV generation
    const submissionRows = [
      ['ApplicationID', 'Approved']
    ];
    
    for (let i = 0; i < testData.length; i++) {
      submissionRows.push([`App${i}`, preds[i]]);
    }
    
    const submissionCSV = submissionRows.map(row => 
      row.map(cell => String(cell)).join(',')
    ).join('\n');
    
    const submissionBlob = new Blob([submissionCSV], { type: 'text/csv' });
    const submissionUrl = URL.createObjectURL(submissionBlob);
    const submissionLink = document.getElementById('download-submission');
    submissionLink.href = submissionUrl;
    submissionLink.download = 'submission.csv';
    submissionLink.style.display = 'inline-block';
    submissionLink.textContent = '✅ Download submission.csv';

    // Probabilities CSV
    const probRows = [
      ['ApplicationID', 'Probability']
    ];
    
    for (let i = 0; i < probs.length; i++) {
      probRows.push([`App${i}`, probs[i].toFixed(6)]);
    }
    
    const probabilitiesCSV = probRows.map(row => 
      row.map(cell => String(cell)).join(',')
    ).join('\n');
    
    const probabilitiesBlob = new Blob([probabilitiesCSV], { type: 'text/csv' });
    const probabilitiesUrl = URL.createObjectURL(probabilitiesBlob);
    const probabilitiesLink = document.getElementById('download-probabilities');
    probabilitiesLink.href = probabilitiesUrl;
    probabilitiesLink.download = 'probabilities.csv';
    probabilitiesLink.style.display = 'inline-block';
    probabilitiesLink.textContent = '✅ Download probabilities.csv';

    xs.dispose();
    
    document.getElementById('eda-output').textContent += `\n🎉 Predictions complete! ${testData.length} samples processed`;
    console.log('✅ Prediction complete:', probs.length, 'samples');

  } catch (err) {
    console.error('Prediction error:', err);
    alert(`Prediction error: ${err.message}`);
  } finally {
    document.getElementById('predict-test').disabled = false;
  }
}

// Training function (simplified and robust)
async function trainModel() {
  if (!preprocessor || !trainData) {
    alert('Please load data first');
    return;
  }

  try {
    document.getElementById('train-model').disabled = true;
    const trainLog = document.getElementById('training-log');
    trainLog.textContent = '🚀 Starting training...\n';

    const hiddenUnits = parseInt(document.getElementById('hidden-units').value);
    const lr = parseFloat(document.getElementById('lr').value);

    model = tf.sequential({
      layers: [
        tf.layers.dense({ 
          units: hiddenUnits, 
          activation: 'relu', 
          inputShape: [preprocessor.featureOrder.length] 
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: Math.floor(hiddenUnits / 2), activation: 'relu' }),
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(lr),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    const summary = [];
    model.summary({ printFn: (line) => summary.push(line) });
    document.getElementById('model-summary').textContent = summary.join('\n');

    const { train } = stratifiedSplit(trainData, trainHeaders);
    const processed = preprocessor.transform(train);
    const xs = tf.tensor2d(processed.features);
    const ys = tf.tensor1d(processed.targets, 'float32');

    await model.fit(xs, ys, {
      epochs: 30,
      batchSize: 32,
      shuffle: true,
      validationData: [valXs, valYs],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          const acc = logs.acc || logs.accuracy || 0;
          trainLog.textContent += `Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(4)}, acc=${acc.toFixed(4)}\n`;
          await tf.nextFrame();
        }
      }
    });

    xs.dispose();
    ys.dispose();
    trainLog.textContent += '✅ Training complete!';
    if (valXs && valYs) updateMetrics();
    enableButtons();

  } catch (err) {
    console.error('Training error:', err);
    alert(`Training error: ${err.message}`);
  } finally {
    document.getElementById('train-model').disabled = false;
  }
}

async function updateMetrics() {
  if (!model || !valXs || !valYs) return;

  try {
    const probs = model.predict(valXs).dataSync();
    const targets = valYs.dataSync();
    const { auc } = computeROC(probs, targets);
    const threshold = parseFloat(document.getElementById('threshold').value);
    const metrics = computeMetrics(probs, targets, threshold);

    document.getElementById('metrics-output').textContent = 
      `AUC: ${auc.toFixed(4)}\nPrecision: ${metrics.precision.toFixed(4)}\nRecall: ${metrics.recall.toFixed(4)}\nF1: ${metrics.f1.toFixed(4)}`;

  } catch (err) {
    console.error('Metrics error:', err);
  }
}

function reset() {
  if (model) { model.dispose(); model = null; }
  if (valXs) { valXs.dispose(); valXs = null; }
  if (valYs) { valYs.dispose(); valYs = null; }
  
  preprocessor = null;
  trainData = null;
  testData = null;
  trainHeaders = null;
  testHeaders = null;

  // Clear UI
  ['eda-output', 'preview-table', 'feature-dim', 'model-summary', 
   'training-log', 'metrics-output', 'confusion-output'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      if (id === 'preview-table') el.innerHTML = '';
      else el.textContent = '';
    }
  });
  
  document.getElementById('download-submission').style.display = 'none';
  document.getElementById('download-probabilities').style.display = 'none';
  
  ['train-file', 'test-file', 'model-json', 'model-bin', 'prep-json'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = '';
  });
  
  enableButtons();
}

function enableButtons() {
  document.getElementById('load-data').disabled = false;
  document.getElementById('train-model').disabled = !trainData || !preprocessor;
  document.getElementById('predict-test').disabled = !model || !testData || !Array.isArray(testData) || testData.length === 0;
  document.getElementById('save-model').disabled = !model;
  document.getElementById('save-prep').disabled = !preprocessor;
  document.getElementById('load-model').disabled = false;
}

async function saveModel() {
  if (!model) return alert('No model to save');
  try {
    await model.save('downloads://loan-approval-model');
    alert('✅ Model saved!');
  } catch (err) {
    alert('Save failed: ' + err.message);
  }
}

function savePreprocessor() {
  if (!preprocessor) return alert('No preprocessor to save');
  try {
    const json = JSON.stringify(preprocessor.toJSON(), null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'preprocessor.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    alert('✅ Preprocessor saved!');
  } catch (err) {
    alert('Save failed: ' + err.message);
  }
}

async function loadModel() {
  alert('Model loading feature coming soon!');
}
