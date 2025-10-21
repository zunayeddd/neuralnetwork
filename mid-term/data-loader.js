// data-loader.js
// CSV parsing, preprocessing, and data utilities for Loan Approval Predictor

// Parse CSV with robust handling for quotes, commas, and malformed rows
function parseCSV(text) {
  const rows = [];
  let currentRow = [];
  let currentField = '';
  let inQuotes = false;
  let i = 0;

  while (i < text.length) {
    const char = text[i];
    if (char === '"') {
      inQuotes = !inQuotes;
      i++;
    } else if (char === ',' && !inQuotes) {
      currentRow.push(currentField.trim());
      currentField = '';
      i++;
    } else if (char === '\n' && !inQuotes) {
      currentRow.push(currentField.trim());
      if (currentRow.length > 1 || currentRow[0] !== '') {
        rows.push(currentRow);
      }
      currentRow = [];
      currentField = '';
      i++;
    } else {
      currentField += char;
      i++;
    }
  }
  if (currentField || currentRow.length) {
    currentRow.push(currentField.trim());
    if (currentRow.length > 1 || currentRow[0] !== '') {
      rows.push(currentRow);
    }
  }
  return rows;
}

// Preprocessing class to handle imputation, scaling, encoding
class Preprocessor {
  constructor() {
    this.means = {};
    this.stds = {};
    this.medians = {};
    this.modes = {};
    this.vocabs = {};
    this.featureOrder = [];
    this.topK = 12;
    this.epsilon = 1e-6;
  }

  // Detect column types (numeric vs categorical)
  detectType(values) {
    const nonEmpty = values.filter(v => v !== '' && v !== null && v !== undefined);
    if (nonEmpty.length === 0) return 'categorical';
    const numericCount = nonEmpty.filter(v => !isNaN(parseFloat(v)) && isFinite(v)).length;
    return numericCount / nonEmpty.length > 0.9 ? 'numeric' : 'categorical';
  }

  // Fit preprocessor on training data
  fit(data, headers, targetCol = 'loan_status', idCol = 'ApplicationID') {
    this.targetCol = targetCol;
    this.idCol = idCol;
    this.headers = headers.filter(h => h !== targetCol && h !== idCol);

    // Detect types
    const types = {};
    this.headers.forEach((col, idx) => {
      const values = data.map(row => row[idx]);
      types[col] = this.detectType(values);
    });

    // Compute means, stds, medians, modes, and vocabularies
    this.headers.forEach((col, idx) => {
      const values = data.map(row => row[idx]);
      if (types[col] === 'numeric') {
        const nums = values.map(v => parseFloat(v)).filter(v => !isNaN(v));
        this.medians[col] = nums.sort((a, b) => a - b)[Math.floor(nums.length / 2)] || 0;
        this.means[col] = nums.reduce((sum, v) => sum + v, 0) / nums.length || 0;
        const variance = nums.reduce((sum, v) => sum + Math.pow(v - this.means[col], 2), 0) / nums.length;
        this.stds[col] = Math.sqrt(variance) || 1;
      } else {
        const counts = {};
        values.forEach(v => counts[v] = (counts[v] || 0) + 1);
        this.modes[col] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Unknown';
        this.vocabs[col] = Object.keys(counts)
          .sort((a, b) => counts[b] - counts[a])
          .slice(0, this.topK);
      }
    });

    // Set feature order: numeric first, then one-hot expansions
    this.featureOrder = [];
    this.headers.forEach(col => {
      if (types[col] === 'numeric') {
        this.featureOrder.push(col);
      }
    });
    this.headers.forEach(col => {
      if (types[col] === 'categorical') {
        this.vocabs[col].forEach(val => this.featureOrder.push(`${col}_${val}`));
        this.featureOrder.push(`${col}_Other`);
      }
    });
  }

  // Transform data to feature matrix
  transform(data, includeTarget = true, includeId = true) {
    const features = [];
    const targets = includeTarget ? [] : null;
    const ids = includeId ? [] : null;

    data.forEach(row => {
      const featureRow = [];
      this.headers.forEach(col => {
        const val = row[this.headers.indexOf(col)];
        if (this.means[col] !== undefined) {
          // Numeric: impute with median, standardize
          const num = val !== '' && !isNaN(parseFloat(val)) ? parseFloat(val) : this.medians[col];
          featureRow.push((num - this.means[col]) / this.stds[col]);
        } else {
          // Categorical: one-hot encode
          const oneHot = Array(this.vocabs[col].length + 1).fill(0);
          const idx = this.vocabs[col].indexOf(val);
          oneHot[idx !== -1 ? idx : this.vocabs[col].length] = 1;
          featureRow.push(...oneHot);
        }
      });
      features.push(featureRow);
      if (includeTarget) {
        const targetVal = row[this.headers.indexOf(this.targetCol)];
        targets.push(targetVal === '1' || targetVal === 'Yes' || targetVal === 'True' ? 1 : 0);
      }
      if (includeId && this.idCol) {
        ids.push(row[this.headers.indexOf(this.idCol)]);
      }
    });

    return { features, targets, ids };
  }

  // Engineer features: IsHighDTI, LoanToIncome
  engineerFeatures(data, headers) {
    const dtiCol = headers.indexOf('loan_percent_income');
    const loanCol = headers.indexOf('loan_amnt');
    const incomeCol = headers.indexOf('person_income');

    const newData = data.map(row => {
      const newRow = [...row];
      newRow.push(parseFloat(row[dtiCol]) > 0.4 ? '1' : '0'); // IsHighDTI
      newRow.push((parseFloat(row[loanCol]) / Math.max(parseFloat(row[incomeCol]), this.epsilon)).toString()); // LoanToIncome
      return newRow;
    });

    const newHeaders = [...headers, 'IsHighDTI', 'LoanToIncome'];
    return { data: newData, headers: newHeaders };
  }

  // Save preprocessing state
  toJSON() {
    return {
      means: this.means,
      stds: this.stds,
      medians: this.medians,
      modes: this.modes,
      vocabs: this.vocabs,
      featureOrder: this.featureOrder,
      headers: this.headers,
      targetCol: this.targetCol,
      idCol: this.idCol
    };
  }

  // Load preprocessing state
  static fromJSON(json) {
    const prep = new Preprocessor();
    Object.assign(prep, json);
    return prep;
  }
}

// Stratified train/validation split
function stratifiedSplit(data, headers, targetCol = 'loan_status', splitRatio = 0.8) {
  const positive = data.filter(row => row[headers.indexOf(targetCol)] === '1');
  const negative = data.filter(row => row[headers.indexOf(targetCol)] === '0');

  const shuffle = arr => arr.sort(() => Math.random() - 0.5);
  const posShuffled = shuffle([...positive]);
  const negShuffled = shuffle([...negative]);

  const posSplit = Math.floor(posShuffled.length * splitRatio);
  const negSplit = Math.floor(negShuffled.length * splitRatio);

  const train = [
    ...posShuffled.slice(0, posSplit),
    ...negShuffled.slice(0, negSplit)
  ];
  const val = [
    ...posShuffled.slice(posSplit),
    ...negShuffled.slice(negSplit)
  ];

  return { train, val };
}

// Compute ROC curve and AUC
function computeROC(probs, targets) {
  const sorted = probs.map((p, i) => [p, targets[i]]).sort((a, b) => b[0] - a[0]);
  let tpr = [0], fpr = [0];
  let tp = 0, fp = 0;
  const P = targets.reduce((sum, t) => sum + t, 0);
  const N = targets.length - P;

  let auc = 0;
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i][1] === 1) {
      tp++;
    } else {
      fp++;
      if (i > 0) {
        auc += (tpr[tpr.length - 1] * (fpr[fpr.length - 1] - fp / N));
      }
    }
    tpr.push(tp / P);
    fpr.push(fp / N);
  }
  auc += (tpr[tpr.length - 1] * (1 - fpr[fpr.length - 1]));

  return { tpr, fpr, auc };
}

// Compute confusion matrix and metrics
function computeMetrics(probs, targets, threshold) {
  const preds = probs.map(p => p >= threshold ? 1 : 0);
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === 1 && targets[i] === 1) tp++;
    else if (preds[i] === 1 && targets[i] === 0) fp++;
    else if (preds[i] === 0 && targets[i] === 0) tn++;
    else fn++;
  }
  const precision = tp / (tp + fp + 1e-6);
  const recall = tp / (tp + fn + 1e-6);
  const f1 = 2 * precision * recall / (precision + recall + 1e-6);
  return { tp, fp, tn, fn, precision, recall, f1 };
}
