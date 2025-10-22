// data-loader.js - FIXED: Preprocessor.fit() method now handles empty arrays properly

// Parse CSV with robust handling
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
      if (currentRow.length > 0 && !(currentRow.length === 1 && currentRow[0] === '')) {
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
    if (currentRow.length > 0 && !(currentRow.length === 1 && currentRow[0] === '')) {
      rows.push(currentRow);
    }
  }
  return rows;
}

// FIXED Preprocessor class
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

  detectType(values) {
    const nonEmpty = values.filter(v => v !== '' && v !== null && v !== undefined);
    if (nonEmpty.length === 0) return 'categorical';
    const numericCount = nonEmpty.filter(v => !isNaN(parseFloat(v)) && isFinite(parseFloat(v))).length;
    return numericCount / nonEmpty.length > 0.9 ? 'numeric' : 'categorical';
  }

  // FIXED: fit() method with proper empty array handling
  fit(data, headers, targetCol = 'loan_status', idCol = 'ApplicationID') {
    this.targetCol = targetCol;
    this.idCol = idCol;
    this.headers = headers.filter(h => h !== targetCol && h !== idCol);

    const types = {};
    this.headers.forEach((col, idx) => {
      const values = data.map(row => row[idx] || '');
      types[col] = this.detectType(values);
    });

    // FIXED: Proper numeric statistics calculation
    this.headers.forEach((col, idx) => {
      const values = data.map(row => {
        const val = row[idx];
        return val !== '' && !isNaN(parseFloat(val)) ? parseFloat(val) : null;
      }).filter(v => v !== null); // Only valid numbers

      if (types[col] === 'numeric' && values.length > 0) {
        values.sort((a, b) => a - b);
        const mid = Math.floor(values.length / 2);
        this.medians[col] = values.length % 2 === 0 
          ? (values[mid - 1] + values[mid]) / 2 
          : values[mid];
        
        this.means[col] = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - this.means[col], 2), 0) / values.length;
        this.stds[col] = Math.sqrt(variance) || 1;
      } else {
        // Default values for numeric columns with no valid data
        this.medians[col] = 0;
        this.means[col] = 0;
        this.stds[col] = 1;
      }

      // Categorical statistics
      if (types[col] === 'categorical') {
        const counts = {};
        data.forEach(row => {
          const val = row[idx] || 'Unknown';
          counts[val] = (counts[val] || 0) + 1;
        });
        const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        this.modes[col] = sorted[0]?.[0] || 'Unknown';
        this.vocabs[col] = sorted.slice(0, this.topK).map(entry => entry[0]);
      }
    });

    // Build feature order
    this.featureOrder = [];
    this.headers.forEach(col => {
      if (types[col] === 'numeric') {
        this.featureOrder.push(col);
      } else if (this.vocabs[col]) {
        this.vocabs[col].forEach(val => this.featureOrder.push(`${col}_${val}`));
        this.featureOrder.push(`${col}_Other`);
      }
    });
  }

  // FIXED: transform() method with proper array handling
  transform(data, includeTarget = true, includeId = true) {
    const features = [];
    const targets = includeTarget ? [] : null;
    const ids = includeId ? [] : null;

    data.forEach(row => {
      if (!Array.isArray(row)) return; // Skip invalid rows
      
      const featureRow = [];
      
      this.headers.forEach(col => {
        const colIdx = this.headers.indexOf(col);
        if (colIdx === -1 || colIdx >= row.length) return;
        
        const val = row[colIdx];
        
        if (this.means[col] !== undefined) {
          // Numeric: impute with median, standardize
          const num = val !== '' && !isNaN(parseFloat(val)) 
            ? parseFloat(val) 
            : (this.medians[col] || 0);
          const standardized = (num - (this.means[col] || 0)) / (this.stds[col] || 1);
          featureRow.push(standardized);
        } else {
          // Categorical: one-hot encode
          const vocab = this.vocabs[col] || [];
          const oneHot = new Array(vocab.length + 1).fill(0);
          const idx = vocab.indexOf(val);
          if (idx !== -1) {
            oneHot[idx] = 1;
          } else {
            oneHot[vocab.length] = 1; // Other category
          }
          featureRow.push(...oneHot);
        }
      });
      
      features.push(featureRow);
      
      if (includeTarget) {
        const targetIdx = this.headers.indexOf(this.targetCol);
        if (targetIdx !== -1 && targetIdx < row.length) {
          const targetVal = row[targetIdx];
          targets.push(targetVal === '1' || targetVal === 'Yes' || targetVal === 'True' ? 1 : 0);
        }
      }
    });

    return { features, targets, ids };
  }

  // Simplified feature engineering (optional)
  engineerFeatures(data, headers) {
    try {
      const newData = data.map(row => [...row]); // Copy rows
      const newHeaders = [...headers];
      newData.forEach(row => {
        row.push('0'); // IsHighDTI
        row.push('0'); // LoanToIncome
      });
      newHeaders.push('IsHighDTI', 'LoanToIncome');
      return { data: newData, headers: newHeaders };
    } catch (e) {
      console.warn('Feature engineering failed, using original data');
      return { data, headers };
    }
  }

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

  static fromJSON(json) {
    const prep = new Preprocessor();
    Object.assign(prep, json);
    return prep;
  }
}

// FIXED: Stratified split with proper array handling
function stratifiedSplit(data, headers, targetCol = 'loan_status', splitRatio = 0.8) {
  const targetIdx = headers.indexOf(targetCol);
  if (targetIdx === -1) {
    console.warn('Target column not found, using random split');
    return randomSplit(data, splitRatio);
  }

  const positive = data.filter(row => {
    if (!Array.isArray(row) || row.length <= targetIdx) return false;
    return row[targetIdx] === '1';
  });
  
  const negative = data.filter(row => {
    if (!Array.isArray(row) || row.length <= targetIdx) return false;
    return row[targetIdx] !== '1';
  });

  const shuffle = arr => {
    const shuffled = [...arr];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  };

  const posShuffled = shuffle(positive);
  const negShuffled = shuffle(negative);

  const posSplit = Math.floor(posShuffled.length * splitRatio);
  const negSplit = Math.floor(negShuffled.length * splitRatio);

  const train = [...posShuffled.slice(0, posSplit), ...negShuffled.slice(0, negSplit)];
  const val = [...posShuffled.slice(posSplit), ...negShuffled.slice(negSplit)];

  return { train, val };
}

function randomSplit(data, splitRatio = 0.8) {
  const shuffled = [...data];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  
  const split = Math.floor(shuffled.length * splitRatio);
  return {
    train: shuffled.slice(0, split),
    val: shuffled.slice(split)
  };
}

// Metrics functions
function computeROC(probs, targets) {
  const sorted = Array.from(probs).map((p, i) => [p, targets[i]]).sort((a, b) => b[0] - a[0]);
  let tpr = [0], fpr = [0];
  let tp = 0, fp = 0;
  const P = targets.filter(t => t === 1).length;
  const N = targets.length - P;
  let auc = 0;

  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i][1] === 1) tp++;
    else fp++;
    tpr.push(P > 0 ? tp / P : 1);
    fpr.push(N > 0 ? fp / N : 1);
    if (i > 0 && sorted[i][1] === 0) {
      auc += (tpr[i] * (fpr[i] - fpr[i-1]));
    }
  }
  return { tpr, fpr, auc };
}

function computeMetrics(probs, targets, threshold = 0.5) {
  const preds = Array.from(probs).map(p => p >= threshold ? 1 : 0);
  let tp = 0, fp = 0, tn = 0, fn = 0;
  
  for (let i = 0; i < preds.length; i++) {
    if (preds[i] === 1 && targets[i] === 1) tp++;
    else if (preds[i] === 1 && targets[i] === 0) fp++;
    else if (preds[i] === 0 && targets[i] === 0) tn++;
    else fn++;
  }
  
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
  
  return { tp, fp, tn, fn, precision, recall, f1 };
}
