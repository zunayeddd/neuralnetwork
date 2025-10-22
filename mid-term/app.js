// app.js - FIXED: Predict on Test button now works properly
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

// FIXED: Load and preprocess data with proper test data handling
async function loadData() {
  try {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) throw new Error('Please upload train.csv');

    document.getElementById('load-data').disabled = true;
    document.getElementById('eda-output').textContent = 'Loading train data...';

    // Load TRAIN data first
    const trainReader = new FileReader();
    trainReader.onload = async (e) => {
      try {
        trainData = parseCSV(e.target.result);
        if (!trainData || trainData.length === 0) {
          throw new Error('Invalid train CSV format');
        }
        
        trainHeaders = trainData[0];
        trainData = trainData.slice(1);

        if (!trainHeaders.includes('loan_status')) {
          throw new Error('train.csv missing loan_status column');
        }

        // Create preprocessor
        preprocessor = new Preprocessor();

        // Engineer features for train data
        if (document.getElementById('engineer-features').checked) {
          const { data, headers } = preprocessor.engineerFeatures(trainData, trainHeaders);
          trainData = data;
          trainHeaders = headers;
        }

        // Show EDA
        showEDA();
        fitPreprocessor();
        createValidationSplit();

        // Load TEST data if provided
        if (testFile) {
          document.getElementById('eda-output').textContent += '\nLoading test data...';
          loadTestData(testFile);
        } else {
          document.getElementById('eda-output').textContent += '\nNo test file provided';
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

// NEW: Separate test data loading function
function loadTestData(testFile) {
  const testReader = new FileReader();
  testReader.onload = (e) => {
    try {
      testData = parseCSV(e.target.result);
      if (!testData || testData.length === 0) {
        throw new Error('Invalid test CSV format');
      }
      
      testHeaders = testData[0];
      testData = testData.slice(1);
      
      // Engineer features for test data if enabled
      if (document.getElementById('engineer-features').checked && preprocessor) {
        const { data, headers } = preprocessor.engineerFeatures(testData, testHeaders);
        testData = data;
        testHeaders = headers;
      }
      
      console.log('✅ Test data loaded successfully:', testData.length, 'rows');
      document.getElementById('eda-output').textContent += `\n✅ Test data loaded: ${testData.length} rows`;
      enableButtons();
      
    } catch (err) {
      console.error('Test data error:', err);
      document.getElementById('eda-output').textContent += `\n❌ Test data error: ${err.message}`;
      enableButtons(); // Still enable other buttons
    }
  };
  testReader.readAsText(testFile);
}

// Show EDA statistics
function showEDA() {
  const edaOutput = document.getElementById('eda-output');
  const shape = `${trainData.length} rows, ${trainHeaders.length} columns`;
  const targetIdx = trainHeaders.indexOf('loan_status');
  const targetRate = trainData.reduce((sum, row) => sum + (row[targetIdx] === '1' ? 1 : 0), 0) / trainData.length;
  
  edaOutput.textContent = `✅ Train data loaded!\nShape: ${shape}\nTarget Rate: ${(targetRate * 100).toFixed(2)}%\n\nPreprocessing...`;
}

// Fit preprocessor
function fitPreprocessor() {
  preprocessor.fit(trainData, trainHeaders);
  document.getElementById('feature-dim').textContent = `Feature Dimension: ${preprocessor.featureOrder.length}`;
  
  // Show data preview
  const previewTable = document.getElementById('preview-table');
  previewTable.innerHTML = '';
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  trainHeaders.slice(0, 8).forEach(h => { // Show first 8 columns
    const th = document.createElement('th');
    th.textContent = h;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  previewTable.appendChild(thead);
  
  const tbody = document.createElement('tbody');
  trainData.slice(0, 10).forEach(row => {
    const tr = document.createElement('tr');
    row.slice(0, 8).forEach(cell => {
      const td = document.createElement('td');
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  previewTable.appendChild(tbody);
}

// Create validation split
function createValidationSplit() {
  const { train, val } = stratifiedSplit(trainData, trainHeaders);
  const valProcessed = preprocessor.transform(val);
  
  if (valXs) valXs.dispose();
  if (valYs) valYs.dispose();
  
  valXs = tf.tensor2d(valProcessed.features);
  valYs = tf.tensor1d(valProcessed.targets, 'float32');
}

// FIXED: Predict on test data
async function predictTest() {
  if (!model) {
    alert('Please train a model first');
    return;
  }
  
  if (!testData || testData.length === 0) {
    alert('No test data loaded. Please upload test.csv and click "Load Data"');
    return;
  }

  try {
    document.getElementById('predict-test').disabled = true;
    document.getElementById('eda-output').textContent = 'Generating predictions...';

    const { features } = preprocessor.transform(testData, false, false);
    const xs = tf.tensor2d(features);
    const probs = model.predict(xs).dataSync();
    const threshold = parseFloat(document.getElementById('threshold').value);
    const preds = probs.map(p => p >= threshold ? 1 : 0);

    // Create submission.csv
    const submission = [['ApplicationID', 'Approved'], 
      ...probs.map((_, i) => [`App${i}`, preds[i]])];
    const submissionCSV = submission.map(row => row.join(',')).join('\n');
    const submissionBlob = new Blob([submissionCSV], { type: 'text/csv' });
    const submissionUrl = URL.createObjectURL(submissionBlob);
    const submissionLink = document.getElementById('download-submission');
    submissionLink.href = submissionUrl;
    submissionLink.download = 'submission.csv';
    submissionLink.style.display = 'inline-block';
    submissionLink.textContent = '✅ Download submission.csv';

    // Create probabilities.csv
    const probabilities = [['ApplicationID', 'Probability'], 
      ...probs.map((p, i) => [`App${i}`, p.toFixed(6)])];
    const probabilitiesCSV = probabilities.map(row => row.join(',')).join('\n');
    const probabilitiesBlob = new Blob([probabilitiesCSV], { type: 'text/csv' });
    const probabilitiesUrl = URL.createObjectURL(probabilitiesBlob);
    const probabilitiesLink = document.getElementById('download-probabilities');
    probabilitiesLink.href = probabilitiesUrl;
    probabilitiesLink.download = 'probabilities.csv';
    probabilitiesLink.style.display = 'inline-block';
    probabilitiesLink.textContent = '✅ Download probabilities.csv';

    xs.dispose();
    
    document.getElementById('eda-output').textContent += `\n✅ Predictions complete! ${testData.length} samples processed`;
    console.log('✅ Prediction complete:', probs.length, 'samples');

  } catch (err) {
    alert(`Prediction error: ${err.message}`);
    console.error('Prediction error:', err);
  } finally {
    document.getElementById('predict-test').disabled = false;
  }
}

// Rest of the functions remain the same (trainModel, updateMetrics, etc.)
async function trainModel() {
  if (!preprocessor || !trainData) {
    alert('Please load data first');
    return;
  }

  try {
    document.getElementById('train-model').disabled = true;
    const trainLog = document.getElementById('training-log');
    trainLog.textContent = 'Starting training...\n';

    const hiddenUnits = parseInt(document.getElementById('hidden-units').value);
    const lr = parseFloat(document.getElementById('lr').value);

    model = tf.sequential({
      layers: [
        tf.layers.dense({ units: hiddenUnits, activation: 'relu', inputShape: [preprocessor.featureOrder.length] }),
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
    const { features, targets } = preprocessor.transform(train);
    const xs = tf.tensor2d(features);
    const ys = tf.tensor1d(targets, 'float32');

    await model.fit(xs, ys, {
      epochs: 30,
      batchSize: 32,
      shuffle: true,
      validationData: [valXs, valYs],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          trainLog.textContent += `Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(4)}, acc=${logs.acc?.toFixed(4)}\n`;
          await tf.nextFrame();
        }
      }
    });

    xs.dispose();
    ys.dispose();
    trainLog.textContent += '✅ Training complete!';
    updateMetrics();
    enableButtons();

  } catch (err) {
    alert(`Training error: ${err.message}`);
    console.error('Training error:', err);
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
    const { precision, recall, f1 } = computeMetrics(probs, targets, threshold);

    document.getElementById('metrics-output').textContent = 
      `AUC: ${auc.toFixed(4)}\nPrecision: ${precision.toFixed(4)}\nRecall: ${recall.toFixed(4)}\nF1: ${f1.toFixed(4)}`;

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
  testData = null;  // ✅ This was the issue!
  trainHeaders = null;
  testHeaders = null;

  // Clear UI
  document.getElementById('eda-output').textContent = '';
  document.getElementById('preview-table').innerHTML = '';
  document.getElementById('feature-dim').textContent = '';
  document.getElementById('model-summary').textContent = '';
  document.getElementById('training-log').textContent = '';
  document.getElementById('metrics-output').textContent = '';
  document.getElementById('confusion-output').textContent = '';
  document.getElementById('download-submission').style.display = 'none';
  document.getElementById('download-probabilities').style.display = 'none';
  
  // Clear files
  ['train-file', 'test-file', 'model-json', 'model-bin', 'prep-json'].forEach(id => {
    document.getElementById(id).value = '';
  });
  
  enableButtons();
}

function enableButtons() {
  document.getElementById('load-data').disabled = false;
  document.getElementById('train-model').disabled = !trainData || !preprocessor;
  document.getElementById('predict-test').disabled = !model || !testData || testData.length === 0;
  document.getElementById('save-model').disabled = !model;
  document.getElementById('save-prep').disabled = !preprocessor;
  document.getElementById('load-model').disabled = false;
}

// Placeholder functions (keep existing ones from data-loader.js)
async function saveModel() {
  if (!model) return alert('No model to save');
  try {
    await model.save('downloads://loan-approval-model');
    alert('Model saved!');
  } catch (err) {
    alert('Save failed: ' + err.message);
  }
}

function savePreprocessor() {
  if (!preprocessor) return alert('No preprocessor to save');
  const json = JSON.stringify(preprocessor.toJSON(), null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'preprocessor.json';
  a.click();
  URL.revokeObjectURL(url);
  alert('Preprocessor saved!');
}

async function loadModel() {
  alert('Model loading feature coming soon!');
}
