/**
 * Diabetes Dashboard - Complete JavaScript Script
 */

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
            csvData = results.data.filter(row => Object.values(row).some(val => val));
            
            if (csvData.length === 0) {
                alert('CSV file is empty');
                return;
            }

            const positives = csvData.filter(row => parseInt(row.diabetes) === 1).length;
            const negatives = csvData.length - positives;
            classStats = { positives, negatives, total: csvData.length };

            document.getElementById('uploadStatus').innerHTML = `
                <strong>✓ Dataset loaded successfully!</strong><br>
                Total Rows: <strong style="color: var(--primary);">${classStats.total}</strong><br>
                Diabetes Cases: <strong style="color: var(--accent);">${classStats.positives}</strong><br>
                Distribution: ${((negatives/classStats.total)*100).toFixed(1)}% negative, ${((positives/classStats.total)*100).toFixed(1)}% positive
            `;
            
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('chartsContainer').classList.remove('hidden');
            drawClassChart();
        },
        error: (error) => {
            alert('Error parsing CSV: ' + error.message);
        }
    });
}

function drawClassChart() {
    if (!classStats) return;

    const ctx = document.getElementById('classChart').getContext('2d');
    
    if (charts.class) charts.class.destroy();

    charts.class = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['No Diabetes', 'Diabetes'],
            datasets: [{
                label: 'Count',
                data: [classStats.negatives, classStats.positives],
                backgroundColor: ['rgba(39, 174, 96, 0.6)', 'rgba(52, 152, 219, 0.6)'],
                borderColor: ['rgba(39, 174, 96, 1)', 'rgba(52, 152, 219, 1)'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#2c3e50' } },
                x: { grid: { display: false }, ticks: { color: '#2c3e50' } }
            }
        }
    });
}

function drawFeatureChart() {
    if (!logisticWeights) return;

    const ctx = document.getElementById('featureChart').getContext('2d');
    if (charts.feature) charts.feature.destroy();

    const bgColors = logisticWeights.map(w => w >= 0 ? 'rgba(52, 152, 219, 0.6)' : 'rgba(39, 174, 96, 0.6)');
    const borderColors = logisticWeights.map(w => w >= 0 ? 'rgba(52, 152, 219, 1)' : 'rgba(39, 174, 96, 1)');

    charts.feature = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'Weight',
                data: logisticWeights,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: false, grid: { color: 'rgba(0,0,0,0.05)' }, ticks: { color: '#2c3e50' } },
                x: { grid: { display: false }, ticks: { color: '#2c3e50', maxRotation: 45, minRotation: 45 } }
            }
        }
    });
}

async function trainModels() {
    if (csvData.length === 0) {
        alert('Please upload a CSV file first');
        return;
    }

    document.getElementById('trainBtn').disabled = true;
    document.getElementById('trainingStatus').innerHTML = '<div class="loading"></div> Training models...';
    document.getElementById('trainingProgress').classList.remove('hidden');

    try {
        const sampleSize = Math.min(5000, csvData.length);
        const sampledData = csvData.slice(0, sampleSize);
        
        const features = sampledData.map(row => 
            featureNames.map(f => {
                const val = parseFloat(row[f]);
                return isNaN(val) ? 0 : val;
            })
        );
        const labels = sampledData.map(row => parseInt(row.diabetes) || 0);

        const splitIdx = Math.floor(features.length * 0.8);
        const xTrain = tf.tensor2d(features.slice(0, splitIdx));
        const yTrain = tf.tensor2d(labels.slice(0, splitIdx), [splitIdx, 1]);
        const xTest = tf.tensor2d(features.slice(splitIdx));
        const yTest = tf.tensor2d(labels.slice(splitIdx), [labels.length - splitIdx, 1]);

        document.getElementById('trainingProgress').innerHTML = '📊 Training Logistic Regression...';

        const logModel = tf.sequential({
            layers: [tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [8] })]
        });
        logModel.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        await logModel.fit(xTrain, yTrain, { epochs: 20, verbose: 0, batchSize: 32 });

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
        nnModel.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        await nnModel.fit(xTrain, yTrain, { epochs: 20, verbose: 0, batchSize: 32 });

        document.getElementById('trainingProgress').innerHTML = '📈 Evaluating models...';

        const logEval = logModel.evaluate(xTest, yTest);
        const nnEval = nnModel.evaluate(xTest, yTest);

        const logLoss = (await logEval[0].data())[0];
        const logAcc = (await logEval[1].data())[0];
        const nnAcc = (await nnEval[1].data())[0];

        const weights = logModel.getWeights()[0];
        logisticWeights = Array.from(await weights.data());

        models.logistic = logModel;
        models.neuralNet = nnModel;

        document.getElementById('logAccuracy').textContent = (logAcc * 100).toFixed(2) + '%';
        document.getElementById('logLoss').textContent = logLoss.toFixed(4);
        document.getElementById('nnAccuracy').textContent = (nnAcc * 100).toFixed(2) + '%';
        document.getElementById('metricsContainer').classList.remove('hidden');
        document.getElementById('featureChartContainer').classList.remove('hidden');
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('trainingStatus').innerHTML = '<strong style="color: var(--accent);">✓ Models trained successfully!</strong>';
        document.getElementById('trainingProgress').classList.add('hidden');

        drawFeatureChart();

        // Cleanup
        xTrain.dispose(); yTrain.dispose(); xTest.dispose(); yTest.dispose();
        logEval[0].dispose(); logEval[1].dispose(); nnEval[0].dispose(); nnEval[1].dispose();
        weights.dispose();

    } catch (error) {
        console.error('Training error:', error);
        alert('Error training models: ' + error.message);
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('trainingStatus').innerHTML = '<strong style="color: var(--danger);">✗ Training failed</strong>';
        document.getElementById('trainingProgress').classList.add('hidden');
    }
}

function handlePredict(event) {
    event.preventDefault();
    if (!models.neuralNet) {
        alert('Please train models first');
        return;
    }

    try {
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

        const inputArray = featureNames.map(f => input[f]);
        const inputTensor = tf.tensor2d([inputArray]);

        const prediction = models.neuralNet.predict(inputTensor);
        const riskProbability = Array.from(prediction.dataSync())[0];

        displayResult(riskProbability, input);

        inputTensor.dispose();
        prediction.dispose();

    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
}

function displayResult(probability, input) {
    const riskLevel = probability > 0.5 ? 'high' : probability > 0.3 ? 'moderate' : 'low';
    const riskPercentage = (probability * 100).toFixed(2);

    document.getElementById('riskProbability').textContent = riskPercentage + '%';
    document.getElementById('progressFill').style.setProperty('--progress-width', riskPercentage + '%');
    document.getElementById('riskLevel').textContent = riskLevel;

    const badge = document.getElementById('resultBadge');
    badge.textContent = riskLevel.toUpperCase();
    badge.className = 'result-badge ' + (riskLevel === 'high' ? 'high' : 'negative');

    const riskFactors = [];
    if (input.HbA1c_level > 6.5) riskFactors.push({ name: 'High HbA1c Level', value: input.HbA1c_level, type: 'risk' });
    if (input.blood_glucose_level > 125) riskFactors.push({ name: 'High Blood Glucose', value: input.blood_glucose_level, type: 'risk' });
    if (input.bmi > 30) riskFactors.push({ name: 'Overweight (BMI > 30)', value: input.bmi, type: 'risk' });
    if (input.hypertension === 1) riskFactors.push({ name: 'Hypertension Present', value: 'Yes', type: 'risk' });
    if (input.heart_disease === 1) riskFactors.push({ name: 'Heart Disease Present', value: 'Yes', type: 'risk' });
    if (input.age > 45) riskFactors.push({ name: 'Age > 45 years', value: input.age, type: 'risk' });
    if (input.bmi < 25) riskFactors.push({ name: 'Healthy BMI', value: input.bmi, type: 'protective' });
    if (input.HbA1c_level < 5.7) riskFactors.push({ name: 'Normal HbA1c', value: input.HbA1c_level, type: 'protective' });
    if (input.blood_glucose_level < 100) riskFactors.push({ name: 'Normal Blood Glucose', value: input.blood_glucose_level, type: 'protective' });

    document.getElementById('riskFactorsList').innerHTML = riskFactors.map(f => `
        <li class="feature-item">
            <span class="feature-name">${f.name}</span>
            <span class="feature-value ${f.type === 'risk' ? 'risk-factor' : 'protective-factor'}">
                ${typeof f.value === 'number' ? f.value.toFixed(2) : f.value}
            </span>
        </li>
    `).join('');

    document.getElementById('resultContainer').classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('Diabetes Dashboard ready!');
});
