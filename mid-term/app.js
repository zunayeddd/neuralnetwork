// app.js with fix for preprocessor undefined
// UI, model training, evaluation, and prediction for Loan Approval Predictor

let model = null;
let preprocessor = null;
let trainData = null;
let testData = null;
let trainHeaders = null;
let testHeaders = null;
let valXs = null, valYs = null;

// Initialize UI event listeners
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('load-data').addEventListener('click', loadData);
  document.getElementById('train-model').addEventListener('click', trainModel);
  document.getElementById('predict-test').addEventListener('click', predictTest);
  document.getElementById('save-model').addEventListener('click', saveModel);
  document.getElementById('save-prep').addEventListener('click', savePreprocessor);
  document.getElementById('load-model').addEventListener('click', loadModel);
  document.getElementById('reset').addEventListener('click', reset);
  document.getElementById('threshold').addEventListener('input', updateMetrics);
  document.getElementById('toggle-visor').addEventListener('click', () => tfvis.visor().toggle());
});

// Load and preprocess data
async function loadData() {
  tf.tidy(() => {
    try {
      const trainFile = document.getElementById('train-file').files[0];
      const testFile = document.getElementById('test-file').files[0];
      if (!trainFile) throw new Error('Please upload train.csv');

      const reader = new FileReader();
      reader.onload = async e => {
        try {
          trainData = parseCSV(e.target.result);
          trainHeaders = trainData[0];
          trainData = trainData.slice(1);

          if (!trainHeaders.includes('loan_status')) {
            throw new Error('train.csv missing loan_status column');
          }

          // Create preprocessor early
          preprocessor = new Preprocessor();

          // Engineer features if enabled
          if (document.getElementById('engineer-features').checked) {
            const { data, headers } = preprocessor.engineerFeatures(trainData, trainHeaders);
            trainData = data;
            trainHeaders = headers;
          }

          // EDA
          const edaOutput = document.getElementById('eda-output');
          const shape = `${trainData.length} rows, ${trainHeaders.length} columns`;
          const targetRate = trainData.reduce((sum, row) => sum + (row[trainHeaders.indexOf('loan_status')] === '1' ? 1 : 0), 0) / trainData.length;
          const missing = trainHeaders.map((col, idx) => {
            const missingCount = trainData.filter(row => row[idx] === '' || row[idx] === null).length;
            return `${col}: ${(missingCount / trainData.length * 100).toFixed(2)}%`;
          }).join('\n');

          edaOutput.textContent = `Shape: ${shape}\nTarget Rate: ${(targetRate * 100).toFixed(2)}%\nMissing Values:\n${missing}`;

          // Preview table
          const previewTable = document.getElementById('preview-table');
          previewTable.innerHTML = '';
          const thead = document.createElement('thead');
          const headerRow = document.createElement('tr');
          trainHeaders.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            headerRow.appendChild(th);
          });
          thead.appendChild(headerRow);
          previewTable.appendChild(thead);
          const tbody = document.createElement('tbody');
          trainData.slice(0, 20).forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
              const td = document.createElement('td');
              td.textContent = cell;
              tr.appendChild(td);
            });
            tbody.appendChild(tr);
          });
          previewTable.appendChild(tbody);

          // Fit preprocessor
          preprocessor.fit(trainData, trainHeaders);
          document.getElementById('feature-dim').textContent = `Feature Dimension: ${preprocessor.featureOrder.length}`;

          // Split data
          const { train, val } = stratifiedSplit(trainData, trainHeaders);
          const trainProcessed = preprocessor.transform(train);
          const valProcessed = preprocessor.transform(val);
          valXs = tf.tensor2d(valProcessed.features);
          valYs = tf.tensor1d(valProcessed.targets, 'float32');

          // Load test data if provided
          if (testFile) {
            const testReader = new FileReader();
            testReader.onload = e => {
              testData = parseCSV(e.target.result);
              testHeaders = testData[0];
              testData = testData.slice(1);
              if (document.getElementById('engineer-features').checked) {
                const { data, headers } = preprocessor.engineerFeatures(testData, testHeaders);
                testData = data;
                testHeaders = headers;
              }
            };
            testReader.readAsText(testFile);
          }

          enableButtons();
        } catch (err) {
          alert(`Error loading data: ${err.message}`);
        }
      };
      reader.readAsText(trainFile);
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  });
}

// Create and train the model
async function trainModel() {
  tf.tidy(async () => {
    try {
      document.getElementById('train-model').disabled = true;
      const hiddenUnits = parseInt(document.getElementById('hidden-units').value);
      const lr = parseFloat(document.getElementById('lr').value);

      // Create model
      model = tf.sequential();
      model.add(tf.layers.dense({
        units: hiddenUnits,
        activation: 'relu',
        inputShape: [preprocessor.featureOrder.length]
      }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      model.add(tf.layers.dense({
        units: Math.floor(hiddenUnits / 2),
        activation: 'relu'
      }));
      model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
      }));

      model.compile({
        optimizer: tf.train.adam(lr),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });

      // Show model summary
      const summary = [];
      model.summary({
        printFn: line => summary.push(line)
      });
      document.getElementById('model-summary').textContent = summary.join('\n');

      // Prepare training data
      const { train } = stratifiedSplit(trainData, trainHeaders);
      const { features, targets } = preprocessor.transform(train);
      const xs = tf.tensor2d(features);
      const ys = tf.tensor1d(targets, 'float32');

      // Train with tfjs-vis
      const surface = { name: 'Training Metrics', tab: 'Training' };
      const history = [];
      await model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        shuffle: true,
        validationData: [valXs, valYs],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            history.push(logs);
            await tfvis.show.history(surface, history, ['loss', 'val_loss', 'acc', 'val_acc']);
            document.getElementById('training-log').textContent += `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc.toFixed(4)}\n`;
            await tf.nextFrame();
          },
          onTrainEnd: () => {
            document.getElementById('training-log').textContent += 'Training complete\n';
            updateMetrics();
          }
        }
      });

      // Early stopping simulation (manual check)
      const patience = 5;
      let bestValLoss = Infinity;
      let patienceCount = 0;
      for (let log of history) {
        if (log.val_loss < bestValLoss) {
          bestValLoss = log.val_loss;
          patienceCount = 0;
        } else {
          patienceCount++;
          if (patienceCount >= patience) {
            document.getElementById('training-log').textContent += 'Early stopping triggered\n';
            break;
          }
        }
      }

      xs.dispose();
      ys.dispose();
      document.getElementById('train-model').disabled = false;
    } catch (err) {
      alert(`Training error: ${err.message}`);
      document.getElementById('train-model').disabled = false;
    }
  });
}

// Update metrics and ROC curve
async function updateMetrics() {
  tf.tidy(() => {
    try {
      if (!model || !valXs || !valYs) return;
      const probs = model.predict(valXs).dataSync();
      const targets = valYs.dataSync();
      const { tpr, fpr, auc } = computeROC(probs, targets);
      const threshold = parseFloat(document.getElementById('threshold').value);
      document.getElementById('threshold-value').textContent = threshold.toFixed(2);
      const { tp, fp, tn, fn, precision, recall, f1 } = computeMetrics(probs, targets, threshold);

      // Update metrics
      document.getElementById('metrics-output').textContent = `AUC: ${auc.toFixed(4)}\nValidation Accuracy: ${(tp + tn) / (tp + tn + fp + fn).toFixed(4)}`;

      // Update confusion matrix
      document.getElementById('confusion-output').textContent = `Confusion Matrix:\nTP: ${tp}, FP: ${fp}\nFN: ${fn}, TN: ${tn}\nPrecision: ${precision.toFixed(4)}\nRecall: ${recall.toFixed(4)}\nF1: ${f1.toFixed(4)}`;

      // Draw ROC curve
      const canvas = document.getElementById('roc-canvas');
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      ctx.moveTo(0, canvas.height);
      for (let i = 0; i < fpr.length; i++) {
        ctx.lineTo(fpr[i] * canvas.width, (1 - tpr[i]) * canvas.height);
      }
      ctx.strokeStyle = 'blue';
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, canvas.height);
      ctx.lineTo(canvas.width, 0);
      ctx.strokeStyle = 'gray';
      ctx.setLineDash([5, 5]);
      ctx.stroke();
      ctx.setLineDash([]);
    } catch (err) {
      alert(`Metrics error: ${err.message}`);
    }
  });
}

// Predict on test data and export
async function predictTest() {
  tf.tidy(() => {
    try {
      if (!model || !testData) {
        throw new Error('Model or test data not loaded');
      }
      document.getElementById('predict-test').disabled = true;

      const { features, ids } = preprocessor.transform(testData, false, true);
      const xs = tf.tensor2d(features);
      const probs = model.predict(xs).dataSync();
      const threshold = parseFloat(document.getElementById('threshold').value);
      const preds = probs.map(p => p >= threshold ? 1 : 0);

      // Create submission.csv
      const submission = [['ApplicationID', 'Approved'], ...ids.map((id, i) => [id || `App${i}`, preds[i]])];
      const submissionCSV = submission.map(row => row.join(',')).join('\n');
      const submissionBlob = new Blob([submissionCSV], { type: 'text/csv' });
      const submissionLink = document.getElementById('download-submission');
      submissionLink.href = URL.createObjectURL(submissionBlob);
      submissionLink.download = 'submission.csv';
      submissionLink.style.display = 'block';

      // Create probabilities.csv
      const probabilities = [['ApplicationID', 'Probability'], ...ids.map((id, i) => [id || `App${i}`, probs[i].toFixed(6)])];
      const probabilitiesCSV = probabilities.map(row => row.join(',')).join('\n');
      const probabilitiesBlob = new Blob([probabilitiesCSV], { type: 'text/csv' });
      const probabilitiesLink = document.getElementById('download-probabilities');
      probabilitiesLink.href = URL.createObjectURL(probabilitiesBlob);
      probabilitiesLink.download = 'probabilities.csv';
      probabilitiesLink.style.display = 'block';

      xs.dispose();
      document.getElementById('predict-test').disabled = false;
    } catch (err) {
      alert(`Prediction error: ${err.message}`);
      document.getElementById('predict-test').disabled = false;
    }
  });
}

// Save model
async function saveModel() {
  try {
    if (!model) throw new Error('No model to save');
    await model.save('downloads://loan-approval-mlp');
  } catch (err) {
    alert(`Save model error: ${err.message}`);
  }
}

// Save preprocessor
async function savePreprocessor() {
  try {
    if (!preprocessor) throw new Error('No preprocessor to save');
    const json = JSON.stringify(preprocessor.toJSON());
    const blob = new Blob([json], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'prep.json';
    link.click();
  } catch (err) {
    alert(`Save preprocessor error: ${err.message}`);
  }
}

// Load model and preprocessor
async function loadModel() {
  try {
    const jsonFile = document.getElementById('model-json').files[0];
    const binFile = document.getElementById('model-bin').files[0];
    const prepFile = document.getElementById('prep-json').files[0];
    if (!jsonFile || !binFile || !prepFile) {
      throw new Error('Please upload model.json, weights.bin, and prep.json');
    }

    const prepReader = new FileReader();
    prepReader.onload = async e => {
      preprocessor = Preprocessor.fromJSON(JSON.parse(e.target.result));
      document.getElementById('feature-dim').textContent = `Feature Dimension: ${preprocessor.featureOrder.length}`;

      if (model) model.dispose();
      model = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));
      const summary = [];
      model.summary({ printFn: line => summary.push(line) });
      document.getElementById('model-summary').textContent = summary.join('\n');
      updateMetrics();
    };
    prepReader.readAsText(prepFile);
  } catch (err) {
    alert(`Load model error: ${err.message}`);
  }
}

// Reset app state
function reset() {
  tf.tidy(() => {
    if (model) model.dispose();
    model = null;
    preprocessor = null;
    trainData = null;
    testData = null;
    trainHeaders = null;
    testHeaders = null;
    if (valXs) valXs.dispose();
    if (valYs) valYs.dispose();
    valXs = null;
    valYs = null;

    document.getElementById('eda-output').textContent = '';
    document.getElementById('preview-table').innerHTML = '';
    document.getElementById('feature-dim').textContent = '';
    document.getElementById('model-summary').textContent = '';
    document.getElementById('training-log').textContent = '';
    document.getElementById('metrics-output').textContent = '';
    document.getElementById('confusion-output').textContent = '';
    document.getElementById('download-submission').style.display = 'none';
    document.getElementById('download-probabilities').style.display = 'none';
    document.getElementById('train-file').value = '';
    document.getElementById('test-file').value = '';
    document.getElementById('model-json').value = '';
    document.getElementById('model-bin').value = '';
    document.getElementById('prep-json').value = '';
    enableButtons();
  });
}

// Enable/disable buttons
function enableButtons() {
  document.getElementById('load-data').disabled = false;
  document.getElementById('train-model').disabled = !trainData;
  document.getElementById('predict-test').disabled = !model || !testData;
  document.getElementById('save-model').disabled = !model;
  document.getElementById('save-prep').disabled = !preprocessor;
  document.getElementById('load-model').disabled = false;
}
