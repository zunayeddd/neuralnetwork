/**
 * Diabetes Dashboard - Complete JavaScript Script (FIXED VERSION)
 * This file contains all the functionality for the diabetes prediction dashboard
 * 
 * KEY FIXES:
 * 1. Reduced data sampling to 5000 rows (instead of 25000)
 * 2. Reduced epochs from 50 to 20 for faster training
 * 3. Added batch size for efficient processing
 * 4. Added progress indicators during training
 */

// ==================== GLOBAL VARIABLES ====================
let csvData = [];
let models = { logistic: null, neuralNet: null };
let classStats = null;
let logisticWeights = null;
let charts = { class: null, feature: null };

const featureNames = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'];
const smokingMap = { 
    'Never': 0, 
    'No Info': 1, 
    'Current': 2, 
    'Former': 3, 
    'Ever': 4, 
    'Not Current': 5 
};

// ==================== FILE UPLOAD HANDLER ====================
/**
 * Handles CSV file upload and parsing
 * Displays dataset statistics and enables training button
 */
function handleFileUpload() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file');
        return;
    }

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
            // Filter out empty rows
            csvData = results.data.filter(row => Object.values(row).some(val => val));
            
            if (csvData.length === 0) {
                alert('CSV file is empty');
                return;
            }

            // Calculate class statistics
            const positives = csvData.filter(row => parseInt(row.diabetes) === 1).length;
            const negatives = csvData.length - positives;
            classStats = { positives, negatives, total: csvData.length };

            // Update UI with dataset information
            document.getElementById('uploadStatus').innerHTML = `
                <strong>✓ Dataset loaded successfully!</strong><br>
                Total Rows: <strong style="color: var(--primary);">${classStats.total}</strong><br>
                Diabetes Cases: <strong style="color: var(--accent);">${classStats.positives}</strong><br>
                Distribution: ${((negatives/classStats.total)*100).toFixed(1)}% negative, ${((positives/classStats.total)*100).toFixed(1)}% positive
            `;
            
            // Enable training button
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('chartsContainer').classList.remove('hidden');
            
            // Draw initial class distribution chart
            drawClassChart();
        },
        error: (error) => {
            alert('Error parsing CSV: ' + error.message);
        }
    });
}

// ==================== CHART FUNCTIONS ====================
/**
 * Draws the class distribution bar chart
 * Shows count of diabetes vs non-diabetes cases
 */
function drawClassChart() {
    if (!classStats) return;

    const ctx = document.getElementById('classChart').getContext('2d');
    
    // Destroy existing chart if present
    if (charts.class) {
        charts.class.destroy();
    }

    charts.class = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['No Diabetes (0)', 'Diabetes (1)'],
            datasets: [{
                label: 'Count',
                data: [classStats.negatives, classStats.positives],
                backgroundColor: ['rgba(102, 187, 106, 0.6)', 'rgba(66, 165, 245, 0.6)'],
                borderColor: ['rgba(102, 187, 106, 1)', 'rgba(66, 165, 245, 1)'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(31, 31, 31, 0.95)',
                    titleColor: '#e0e0e0',
                    bodyColor: '#e0e0e0',
                    borderColor: 'rgba(66, 165, 245, 0.5)',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#e0e0e0' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#e0e0e0' }
                }
            }
        }
    });
}

/**
 * Draws the feature importance bar chart
 * Shows model weights for each feature
 */
function drawFeatureChart() {
    if (!logisticWeights) return;

    const ctx = document.getElementById('featureChart').getContext('2d');
    
    // Destroy existing chart if present
    if (charts.feature) {
        charts.feature.destroy();
    }

    // Color bars based on positive/negative weights
    const backgroundColors = logisticWeights.map(w => 
        w >= 0 ? 'rgba(66, 165, 245, 0.6)' : 'rgba(102, 187, 106, 0.6)'
    );
    const borderColors = logisticWeights.map(w => 
        w >= 0 ? 'rgba(66, 165, 245, 1)' : 'rgba(102, 187, 106, 1)'
    );

    charts.feature = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'Weight',
                data: logisticWeights,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(31, 31, 31, 0.95)',
                    titleColor: '#e0e0e0',
                    bodyColor: '#e0e0e0',
                    borderColor: 'rgba(66, 165, 245, 0.5)',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#e0e0e0' }
                },
                x: {
                    grid: { display: false },
                    ticks: { 
                        color: '#e0e0e0', 
                        maxRotation: 45, 
                        minRotation: 45, 
                        autoSkip: false 
                    }
                }
            }
        }
    });
}

// ==================== TRAINING FUNCTION (FIXED) ====================
/**
 * FIXED VERSION - Trains both logistic regression and neural network models
 * 
 * KEY FIXES:
 * 1. Samples data to 5000 rows (Line ~90)
 * 2. Reduces epochs to 20 (Lines ~115, ~135)
 * 3. Adds batch size for efficiency (Lines ~115, ~135)
 * 4. Shows progress updates (Lines ~110, ~130, ~145)
 */
async function trainModels() {
    if (csvData.length === 0) {
        alert('Please upload a CSV file first');
        return;
    }

    document.getElementById('trainBtn').disabled = true;
    document.getElementById('trainingStatus').innerHTML = '<div class="loading" style="display: inline-block;"></div> Training models...';
    document.getElementById('trainingProgress').classList.remove('hidden');

    try {
        // ========== FIX #1: SAMPLE DATA TO 5000 ROWS ==========
        // This prevents browser from hanging on large datasets
        const sampleSize = Math.min(5000, csvData.length);
        const sampledData = csvData.slice(0, sampleSize);
        
        console.log(`Training on ${sampleSize} samples out of ${csvData.length} total rows`);
        
        // Prepare features and labels
        const features = sampledData.map(row => 
            featureNames.map(f => {
                const val = parseFloat(row[f]);
                return isNaN(val) ? 0 : val;
            })
        );
        const labels = sampledData.map(row => parseInt(row.diabetes) || 0);

        // Split data: 80% training, 20% testing
        const splitIdx = Math.floor(features.length * 0.8);
        const xTrain = tf.tensor2d(features.slice(0, splitIdx));
        const yTrain = tf.tensor2d(labels.slice(0, splitIdx), [splitIdx, 1]);
        const xTest = tf.tensor2d(features.slice(splitIdx));
        const yTest = tf.tensor2d(labels.slice(splitIdx), [labels.length - splitIdx, 1]);

        // ========== TRAIN LOGISTIC REGRESSION ==========
        document.getElementById('trainingProgress').innerHTML = '📊 Training Logistic Regression...';

        const logModel = tf.sequential({
            layers: [
                tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [8] })
            ]
        });
        logModel.compile({ 
            optimizer: 'adam', 
            loss: 'binaryCrossentropy', 
            metrics: ['accuracy'] 
        });
        
        // ========== FIX #2: REDUCED EPOCHS FROM 50 TO 20 ==========
        // ========== FIX #3: ADDED BATCH SIZE ==========
        await logModel.fit(xTrain, yTrain, { 
            epochs: 20,           // REDUCED from 50
            verbose: 0,
            batchSize: 32         // ADDED for efficiency
        });

        // ========== TRAIN NEURAL NETWORK ==========
        document.getElementById('trainingProgress').innerHTML = '🧠 Training Neural Network...';

        const nnModel = tf.sequential({
            layers: [
                tf.layers.dense({ units: 16, activation: 'relu', inputShape: [8] }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 8, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 1, activation: 'sigmoid' })
            ]
        });
        nnModel.compile({ 
            optimizer: 'adam', 
            loss: 'binaryCrossentropy', 
            metrics: ['accuracy'] 
        });
        
        // ========== FIX #2: REDUCED EPOCHS FROM 50 TO 20 ==========
        // ========== FIX #3: ADDED BATCH SIZE ==========
        await nnModel.fit(xTrain, yTrain, { 
            epochs: 20,           // REDUCED from 50
            verbose: 0,
            batchSize: 32         // ADDED for efficiency
        });

        // ========== EVALUATE MODELS ==========
        document.getElementById('trainingProgress').innerHTML = '📈 Evaluating models...';

        const logEval = logModel.evaluate(xTest, yTest);
        const nnEval = nnModel.evaluate(xTest, yTest);

        const logLoss = (await logEval[0].data())[0];
        const logAcc = (await logEval[1].data())[0];
        const nnLoss = (await nnEval[0].data())[0];
        const nnAcc = (await nnEval[1].data())[0];

        // Get weights for feature importance
        const weights = logModel.getWeights()[0];
        logisticWeights = Array.from(await weights.data());

        // Store models for later use
        models.logistic = logModel;
        models.neuralNet = nnModel;

        // Update UI with results
        document.getElementById('logAccuracy').textContent = (logAcc * 100).toFixed(2) + '%';
        document.getElementById('logLoss').textContent = logLoss.toFixed(4);
        document.getElementById('nnAccuracy').textContent = (nnAcc * 100).toFixed(2) + '%';
        document.getElementById('metricsContainer').classList.remove('hidden');
        document.getElementById('featureChartContainer').classList.remove('hidden');
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('trainingStatus').innerHTML = '<strong style="color: var(--accent);">✓ Models trained successfully!</strong>';
        document.getElementById('trainingProgress').classList.add('hidden');

        // Draw feature importance chart
        drawFeatureChart();

        // Cleanup tensors to free memory
        xTrain.dispose();
        yTrain.dispose();
        xTest.dispose();
        yTest.dispose();
        logEval[0].dispose();
        logEval[1].dispose();
        nnEval[0].dispose();
        nnEval[1].dispose();
        weights.dispose();

    } catch (error) {
        console.error('Training error:', error);
        alert('Error training models: ' + error.message);
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('trainingStatus').innerHTML = '<strong style="color: var(--danger);">✗ Training failed</strong>';
        document.getElementById('trainingProgress').classList.add('hidden');
    }
}

// ==================== PREDICTION FUNCTION ====================
/**
 * Handles form submission and makes diabetes risk prediction
 */
function handlePredict(event) {
    event.preventDefault();

    if (!models.neuralNet) {
        alert('Please train models first');
        return;
    }

    try {
        // Get form data
        const input = {
            gender: document.getElementById('gender').value === 'Female' ? 0 : 1,
            age: parseFloat(document.getElementById('age').value),
            hypertension: document.getElementById('hypertension').value === 'Yes' ? 1 : 0,
            heart_disease: document.getElementById('heartDisease').value === 'Yes' ? 1 : 0,
            smoking_history: smokingMap[document.getElementById('smokingHistory').value],
            bmi: parseFloat(document.getElementById('bmi').value),
            HbA1c_level: parseFloat(document.getElementById('hba1c').value),
            blood_glucose_level: parseFloat(document.getElementById('bloodGlucose').value)
        };

        // Prepare input tensor in correct order
        const inputArray = featureNames.map(f => input[f]);
        const inputTensor = tf.tensor2d([inputArray]);

        // Make prediction using neural network
        const prediction = models.neuralNet.predict(inputTensor);
        const riskProbability = Array.from(prediction.dataSync())[0];

        // Display results
        displayResult(riskProbability, input);

        // Cleanup
        inputTensor.dispose();
        prediction.dispose();

    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
}

// ==================== RESULT DISPLAY FUNCTION ====================
/**
 * Displays prediction results with risk factors analysis
 */
function displayResult(probability, input) {
    // Determine risk level
    const riskLevel = probability > 0.5 ? 'high' : probability > 0.3 ? 'moderate' : 'low';
    const riskPercentage = (probability * 100).toFixed(2);

    // Update probability display
    document.getElementById('riskProbability').textContent = riskPercentage + '%';
    document.getElementById('progressFill').style.setProperty('--progress-width', riskPercentage + '%');
    document.getElementById('riskLevel').textContent = riskLevel;

    // Update badge color based on risk level
    const badge = document.getElementById('resultBadge');
    badge.textContent = riskLevel.toUpperCase();
    badge.className = 'result-badge ' + (riskLevel === 'high' ? 'high' : 'negative');

    // Analyze risk factors
    const riskFactors = [];
    
    // Risk factors (increase diabetes risk)
    if (input.HbA1c_level > 6.5) 
        riskFactors.push({ name: 'High HbA1c Level', value: input.HbA1c_level, type: 'risk' });
    if (input.blood_glucose_level > 125) 
        riskFactors.push({ name: 'High Blood Glucose', value: input.blood_glucose_level, type: 'risk' });
    if (input.bmi > 30) 
        riskFactors.push({ name: 'Overweight (BMI > 30)', value: input.bmi, type: 'risk' });
    if (input.hypertension === 1) 
        riskFactors.push({ name: 'Hypertension Present', value: 'Yes', type: 'risk' });
    if (input.heart_disease === 1) 
        riskFactors.push({ name: 'Heart Disease Present', value: 'Yes', type: 'risk' });
    if (input.age > 45) 
        riskFactors.push({ name: 'Age > 45 years', value: input.age, type: 'risk' });
    
    // Protective factors (decrease diabetes risk)
    if (input.bmi < 25) 
        riskFactors.push({ name: 'Healthy BMI', value: input.bmi, type: 'protective' });
    if (input.HbA1c_level < 5.7) 
        riskFactors.push({ name: 'Normal HbA1c', value: input.HbA1c_level, type: 'protective' });
    if (input.blood_glucose_level < 100) 
        riskFactors.push({ name: 'Normal Blood Glucose', value: input.blood_glucose_level, type: 'protective' });

    // Display factors in UI
    const factorsList = document.getElementById('riskFactorsList');
    factorsList.innerHTML = riskFactors.map(factor => `
        <li class="feature-item">
            <span class="feature-name">${factor.name}</span>
            <span class="feature-value ${factor.type === 'risk' ? 'risk-factor' : 'protective-factor'}">
                ${typeof factor.value === 'number' ? factor.value.toFixed(2) : factor.value}
            </span>
        </li>
    `).join('');

    // Show result card
    document.getElementById('resultContainer').classList.remove('hidden');
}

// ==================== INITIALIZATION ====================
/**
 * Initialize dashboard when DOM is ready
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('✓ Diabetes Dashboard initialized');
    console.log('✓ Ready for CSV upload and model training');
});

// ==================== EXPORT FOR MODULE USE ====================
// Uncomment if using as a module
// export { handleFileUpload, trainModels, handlePredict, drawClassChart, drawFeatureChart };
