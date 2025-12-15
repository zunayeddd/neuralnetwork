let csvData = [];
let models = { logistic: null, neuralNet: null };
let classStats = null;
let charts = { log: null, nn: null };

const featureNames = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level'];
const smokingMap = { 'Never': 0, 'No Info': 1, 'Current': 2, 'Former': 3, 'Ever': 4, 'Not Current': 5 };

// Health Recommendations
const recommendations = {
    low: {
        title: "✅ Low Risk - Keep It Up!",
        tips: [
            "🥗 <strong>Maintain Healthy Diet:</strong> Continue eating balanced meals with plenty of vegetables, whole grains, and lean proteins.",
            "🏃 <strong>Regular Exercise:</strong> Aim for 150 minutes of moderate exercise per week to maintain good health.",
            "⚖️ <strong>Maintain Healthy Weight:</strong> Keep your BMI in the normal range (18.5-24.9).",
            "🩺 <strong>Regular Check-ups:</strong> Visit your doctor annually for preventive health screenings.",
            "💤 <strong>Quality Sleep:</strong> Get 7-9 hours of sleep per night for optimal health.",
            "🚫 <strong>Avoid Smoking:</strong> Continue to stay away from tobacco and smoking."
        ]
    },
    medium: {
        title: "⚠️ Medium Risk - Take Action!",
        tips: [
            "🥗 <strong>Improve Diet:</strong> Reduce sugar and refined carbohydrates. Focus on high-fiber foods and lean proteins.",
            "🏃 <strong>Increase Physical Activity:</strong> Aim for at least 150 minutes of moderate exercise weekly.",
            "⚖️ <strong>Weight Management:</strong> If overweight, aim to lose 5-10% of your body weight gradually.",
            "🩺 <strong>Monitor Blood Sugar:</strong> Check your blood glucose levels regularly and keep records.",
            "💊 <strong>Consult Healthcare Provider:</strong> Discuss your risk factors with your doctor for personalized advice.",
            "🧂 <strong>Reduce Salt & Stress:</strong> Lower sodium intake and practice stress management techniques like meditation."
        ]
    },
    high: {
        title: "🚨 High Risk - Seek Medical Attention!",
        tips: [
            "🏥 <strong>Consult Endocrinologist:</strong> Schedule an appointment with a diabetes specialist immediately.",
            "🩺 <strong>Get Tested:</strong> Get a formal diabetes screening (Fasting Glucose, Oral Glucose Tolerance Test).",
            "💊 <strong>Consider Medication:</strong> Your doctor may recommend medication to prevent or delay diabetes onset.",
            "🥗 <strong>Strict Diet Changes:</strong> Follow a diabetes-prevention diet with minimal sugar and processed foods.",
            "🏃 <strong>Intensive Exercise:</strong> Aim for 300 minutes of moderate exercise per week if possible.",
            "⚖️ <strong>Aggressive Weight Loss:</strong> Losing 7% of body weight can significantly reduce diabetes risk.",
            "🚫 <strong>Quit Smoking:</strong> If you smoke, seek help to quit immediately.",
            "📊 <strong>Regular Monitoring:</strong> Monitor blood glucose, blood pressure, and cholesterol regularly."
        ]
    }
};

// Handle File Upload
function handleFileUpload() {
    const file = document.getElementById('csvFile').files[0];
    if (!file) return;

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

            const statusEl = document.getElementById('uploadStatus');
            statusEl.innerHTML = `✓ Dataset loaded: ${classStats.total} samples (${classStats.positives} with diabetes)`;
            statusEl.classList.remove('hidden');
            
            document.getElementById('trainBtn').disabled = false;
        }
    });
}

// Initialize Charts
function initializeCharts() {
    const logCtx = document.getElementById('logChart').getContext('2d');
    if (charts.log) charts.log.destroy();
    charts.log = new Chart(logCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointBackgroundColor: '#6366f1'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(0, 0, 0, 0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });

    const nnCtx = document.getElementById('nnChart').getContext('2d');
    if (charts.nn) charts.nn.destroy();
    charts.nn = new Chart(nnCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Loss',
                data: [],
                borderColor: '#ec4899',
                backgroundColor: 'rgba(236, 72, 153, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 3,
                pointBackgroundColor: '#ec4899'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(0, 0, 0, 0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

// Update Metrics
function updateMetrics(modelType, epoch, loss, accuracy) {
    if (modelType === 'logistic') {
        document.getElementById('logLoss').textContent = loss.toFixed(4);
        document.getElementById('logAccuracy').textContent = (accuracy * 100).toFixed(1) + '%';
        document.getElementById('logEpoch').textContent = `${epoch}/20`;
        document.getElementById('logProgressBar').style.width = `${(epoch / 20) * 100}%`;
        document.getElementById('logProgress').textContent = `${Math.round((epoch / 20) * 100)}%`;
        document.getElementById('logEpochStatus').textContent = `Epoch ${epoch} complete`;
        
        charts.log.data.labels.push(epoch);
        charts.log.data.datasets[0].data.push(loss);
        charts.log.update();
    } else {
        document.getElementById('nnLoss').textContent = loss.toFixed(4);
        document.getElementById('nnAccuracy').textContent = (accuracy * 100).toFixed(1) + '%';
        document.getElementById('nnEpoch').textContent = `${epoch}/20`;
        document.getElementById('nnProgressBar').style.width = `${(epoch / 20) * 100}%`;
        document.getElementById('nnProgress').textContent = `${Math.round((epoch / 20) * 100)}%`;
        document.getElementById('nnEpochStatus').textContent = `Epoch ${epoch} complete`;
        
        charts.nn.data.labels.push(epoch);
        charts.nn.data.datasets[0].data.push(loss);
        charts.nn.update();
    }
}

// Train Models
async function trainModels() {
    if (csvData.length === 0) {
        alert('Please upload a CSV file first');
        return;
    }

    document.getElementById('trainBtn').disabled = true;
    document.getElementById('liveTrainingContainer').style.display = 'grid';
    
    initializeCharts();

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

        // Train Logistic Regression
        const logModel = tf.sequential({
            layers: [tf.layers.dense({ units: 1, activation: 'sigmoid', inputShape: [8] })]
        });
        logModel.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
        
        for (let epoch = 1; epoch <= 20; epoch++) {
            const history = await logModel.fit(xTrain, yTrain, { epochs: 1, verbose: 0, batchSize: 32 });
            updateMetrics('logistic', epoch, history.history.loss[0], history.history.accuracy[0]);
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        document.getElementById('logStatus').classList.remove('training');
        document.getElementById('logStatus').classList.add('complete');

        // Train Neural Network
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
        
        for (let epoch = 1; epoch <= 20; epoch++) {
            const history = await nnModel.fit(xTrain, yTrain, { epochs: 1, verbose: 0, batchSize: 32 });
            updateMetrics('neural', epoch, history.history.loss[0], history.history.accuracy[0]);
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        document.getElementById('nnStatus').classList.remove('training');
        document.getElementById('nnStatus').classList.add('complete');

        // Evaluate Models
        const logEval = logModel.evaluate(xTest, yTest);
        const nnEval = nnModel.evaluate(xTest, yTest);

        const logLoss = (await logEval[0].data())[0];
        const logAcc = (await logEval[1].data())[0];
        const nnLoss = (await nnEval[0].data())[0];
        const nnAcc = (await nnEval[1].data())[0];

        models.logistic = logModel;
        models.neuralNet = nnModel;

        document.getElementById('logFinalAccuracy').textContent = (logAcc * 100).toFixed(2) + '%';
        document.getElementById('logFinalLoss').textContent = logLoss.toFixed(4);
        document.getElementById('nnFinalAccuracy').textContent = (nnAcc * 100).toFixed(2) + '%';
        document.getElementById('nnFinalLoss').textContent = nnLoss.toFixed(4);
        document.getElementById('resultsContainer').classList.add('active');

        document.getElementById('predictBtn').disabled = false;

        xTrain.dispose();
        yTrain.dispose();
        xTest.dispose();
        yTest.dispose();
        logEval[0].dispose();
        logEval[1].dispose();
        nnEval[0].dispose();
        nnEval[1].dispose();

    } catch (error) {
        console.error('Training error:', error);
        alert('Error: ' + error.message);
        document.getElementById('trainBtn').disabled = false;
    }
}

// Handle Prediction
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

        let riskLevel = 'low';
        if (riskProbability > 0.5) riskLevel = 'high';
        else if (riskProbability > 0.3) riskLevel = 'medium';

        const riskPercentage = (riskProbability * 100).toFixed(1);

        document.getElementById('riskProbability').textContent = riskPercentage + '%';
        document.getElementById('progressFill').style.width = riskPercentage + '%';

        const badge = document.getElementById('resultBadge');
        badge.textContent = riskLevel.toUpperCase();
        badge.className = `risk-badge ${riskLevel}`;

        const recsContainer = document.getElementById('recommendationsContainer');
        const recsData = recommendations[riskLevel];
        recsContainer.innerHTML = `
            <div class="recommendations-title">${recsData.title}</div>
            ${recsData.tips.map(tip => `
                <div class="recommendation-item ${riskLevel}-risk">
                    <div class="recommendation-text">${tip}</div>
                </div>
            `).join('')}
        `;

        document.getElementById('resultContainer').classList.remove('hidden');

        inputTensor.dispose();
        prediction.dispose();

    } catch (error) {
        alert('Error: ' + error.message);
    }
}
