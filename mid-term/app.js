// app.js - FIXED: Added type checks to prevent "join is not a function" and "forEach is not a function" errors
// Ensured all rows are arrays and added debug logs for loading issues

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

// Load and preprocess data with additional logging
async function loadData() {
  try {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile) throw new Error('Please upload train.csv');

    document.getElementById('load-data').disabled = true;
    document.getElementById('eda-output').textContent = 'Loading train data...';

    const trainReader = new FileReader();
    trainReader.onload = async (e) => {
      try {
        console.log('Parsing train CSV...');
        trainData = parseCSV(e.target.result);
        if (!Array.isArray(trainData) || trainData.length === 0) {
          throw new Error('Invalid train CSV format - not an array');
        }
        
        trainHeaders = trainData[0];
        if (!Array.isArray(trainHeaders)) {
          throw new Error('Headers not an array');
        }
        
        trainData = trainData.slice(1);
        if (!trainHeaders.includes('loan_status')) {
          throw new Error('train.csv missing loan_status column');
        }

        // Create preprocessor
        preprocessor = new Preprocessor();

        // Engineer features for train data
        if (document.getElementById('engineer-features').checked) {
          console.log('Engineering features for train...');
          const { data, headers } = preprocessor.engineerFeatures(trainData, trainHeaders);
          trainData = data;
          trainHeaders = headers;
        }

        // Show EDA
        console.log('Generating EDA...');
        showEDA();
        console.log('Fitting preprocessor...');
        fitPreprocessor();
        console.log('Creating validation split...');
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

// Load test data with logging
function loadTestData(testFile) {
  const testReader = new FileReader();
  testReader.onload = (e) => {
    try {
      console.log('Parsing test CSV...');
      testData = parseCSV(e.target.result);
      if (!Array.isArray(testData) || testData.length === 0) {
        throw new Error('Invalid test CSV format - not an array');
      }
      
      testHeaders = testData[0];
      if (!Array.isArray(testHeaders)) {
        throw new Error('Test headers not an array');
      }
      
      testData = testData.slice(1);
      
      if (document.getElementById('engineer-features').checked && preprocessor) {
        console.log('Engineering features for test...');
        const { data, headers } = preprocessor.engineerFeatures(testData, testHeaders);
        data.forEach(d => {
          if (!Array.isArray(d)) {
            console.error('Non-array row in engineered data:', d);
            throw new Error('Engineered row is not array');
          }
        });
        testData = data;
        testHeaders = headers;
      }
      
      console.log('✅ Test data loaded successfully:', testData.length, 'rows');
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

// Show EDA with checks
function showEDA() {
  const edaOutput = document.getElementById('eda-output');
  const shape = `${trainData.length} rows, ${trainHeaders.length} columns`;
  const targetIdx = trainHeaders.indexOf('loan_status');
  const targetRate = trainData.reduce((sum, row) => {
    if (!Array.isArray(row)) throw new Error('Train row is not array');
    return sum + (row[targetIdx] === '1' ? 1 : 0);
  }, 0) / trainData.length;
  
  edaOutput.textContent = `✅ Train data loaded!\nShape: ${shape}\nTarget Rate: ${(targetRate * 100).toFixed(2)}%\n\nPreprocessing...`;
}

// Fit preprocessor with additional checks
function fitPreprocessor() {
  preprocessor.fit(trainData, trainHeaders);
  document.getElementById('feature-dim').textContent = `Feature Dimension: ${preprocessor.featureOrder.length}`;
  
  // Show data preview
  const previewTable = document.getElementById('preview-table');
  previewTable.innerHTML = '';
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  trainHeaders.slice(0, 8).forEach(h => {
    const th = document.createElement('th');
    th.textContent = h;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  previewTable.appendChild(thead);
  
  const tbody = document.createElement('tbody');
  trainData.slice(0, 10).forEach(row => {
    if (!Array.isArray(row)) {
      console.error('Non-array preview row:', row);
      throw new Error('Preview row is not array');
    }
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

// FIXED: Predict on test data with type checks
async function predictTest() {
  if (!model) {
    alert('Please train a model first');
    return;
  }
  
  if (!testData || !Array.isArray(testData) || testData.length === 0) {
    alert('No test data loaded. Please upload test.csv and click "Load Data"');
    return;
  }

  try {
    document.getElementById('predict-test').disabled = true;
    document.getElementById('eda-output').textContent = 'Generating predictions...';

    const { features } = preprocessor.transform(testData, false, false);
    if (!Array.isArray(features)) {
      throw new Error('Features is not an array');
    }
    features.forEach(f => {
      if (!Array.isArray(f)) {
        console.error('Non-array feature row:', f);
        throw new Error('Feature row is not array');
      }
    });
    
    const xs = tf.tensor2d(features);
    const probsTensor = model.predict(xs);
    const probs = probsTensor.dataSync();
    probsTensor.dispose();
    const threshold = parseFloat(document.getElementById('threshold').value);
    const preds = Array.from(probs).map(p => p >= threshold ? 1 : 0);

    // Create mapped rows with type check
    const mappedRows = Array.from(probs).map((p, i) => [`App${i}`, preds[i]]);
    mappedRows.forEach((row, idx) => {
      if (!Array.isArray(row)) {
        console.error('Non-array mapped row at index', idx, ':', row);
        throw new Error('Mapped row is not array');
      }
    });

    const submission = [['ApplicationID', 'Approved'], ...mappedRows];
    const submissionCSV = submission.map((row, idx) => {
      if (!Array.isArray(row)) {
        console.error('Non-array submission row at index', idx, ':', row);
        throw new Error('Submission row is not array');
      }
      return row.join(',');
    }).join('\n');
    
    const submissionBlob = new Blob([submissionCSV], { type: 'text/csv' });
    const submissionUrl = URL.createObjectURL(submissionBlob);
    const submissionLink = document.getElementById('download-submission');
    submissionLink.href = submissionUrl;
    submissionLink.download = 'submission.csv';
    submissionLink.style.display = 'inline-block';
    submissionLink.textContent = '✅ Download submission.csv';

    const probabilities = [['ApplicationID', 'Probability'], ...Array.from(probs).map((p, i) => [`App${i}`, p.toFixed(6)])];
    const probabilitiesCSV = probabilities.map((row, idx) => {
      if (!Array.isArray(row)) {
        console.error('Non-array probabilities row at index', idx, ':', row);
        throw new Error('Probabilities row is not array');
      }
      return row.join(',');
    }).join('\n');
    
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

// The rest of the code remains the same as before

// Train model (unchanged)
async function trainModel() {
  // ... (keep the previous version)
}

// Update metrics (unchanged)
async function updateMetrics() {
  // ... (keep the previous version)
}

// Reset (unchanged)
function reset() {
  // ... (keep the previous version)
}

// Enable buttons (unchanged)
function enableButtons() {
  // ... (keep the previous version)
}

// Save/load functions (unchanged)
async function saveModel() {
  // ... (keep the previous version)
}

function savePreprocessor() {
  // ... (keep the previous version)
}

async function loadModel() {
  // ... (keep the previous version)
}
