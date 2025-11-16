// ===== Global state =====
let rawRows = [];
let scaler = { mean: [], std: [] };
let logModel = null;
let nnModel = null;
let modelsReady = false;

let classStats = null;
let classChart = null;
let importanceChart = null;

const featureOrder = [
    "genderNum",
    "age",
    "hypertension",
    "heart_disease",
    "smokingCode",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
];

const featureNames = [
    "Gender (Male=1)",
    "Age",
    "Hypertension",
    "Heart disease",
    "Smoking history code",
    "BMI",
    "HbA1c level",
    "Blood glucose level"
];

const smokingMap = {
    "never": 0,
    "No Info": 1,
    "former": 2,
    "current": 3,
    "not current": 4,
    "ever": 5
};

const datasetInput = document.getElementById("datasetInput");
const trainButton = document.getElementById("trainButton");
const edaDiv = document.getElementById("eda");
const metricsDiv = document.getElementById("metrics");
const predictForm = document.getElementById("predictForm");
const predictButton = document.getElementById("predictButton");
const predictionOutput = document.getElementById("predictionOutput");

// ===== 1. Load dataset =====
datasetInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: function (results) {
            rawRows = cleanRows(results.data);
            showEDA(rawRows);
            trainButton.disabled = rawRows.length === 0;
            metricsDiv.innerHTML = "";
            destroyCharts();
        }
    });
});

function cleanRows(rows) {
    const cleaned = [];
    for (const r of rows) {
        if (r.diabetes !== 0 && r.diabetes !== 1) continue;
        if (r.gender !== "Male" && r.gender !== "Female") continue;
        const smoking = r.smoking_history;
        if (!(smoking in smokingMap)) continue;

        const vals = [
            r.age,
            r.hypertension,
            r.heart_disease,
            r.bmi,
            r.HbA1c_level,
            r.blood_glucose_level
        ];
        if (vals.some(v => v === null || v === undefined || Number.isNaN(v))) continue;

        const genderNum = r.gender === "Male" ? 1 : 0;
        const smokingCode = smokingMap[smoking];

        cleaned.push({
            genderNum,
            age: r.age,
            hypertension: r.hypertension,
            heart_disease: r.heart_disease,
            smokingCode,
            bmi: r.bmi,
            HbA1c_level: r.HbA1c_level,
            blood_glucose_level: r.blood_glucose_level,
            diabetes: r.diabetes
        });
    }
    return cleaned;
}

function showEDA(rows) {
    if (rows.length === 0) {
        edaDiv.innerHTML = "<p>No valid rows found in dataset.</p>";
        classStats = null;
        return;
    }
    const n = rows.length;
    let positives = 0;
    let ageSum = 0;
    let bmiSum = 0;
    for (const r of rows) {
        if (r.diabetes === 1) positives++;
        ageSum += r.age;
        bmiSum += r.bmi;
    }
    const neg = n - positives;
    const posPct = (positives / n * 100).toFixed(2);
    const negPct = (neg / n * 100).toFixed(2);
    const meanAge = (ageSum / n).toFixed(1);
    const meanBmi = (bmiSum / n).toFixed(1);

    classStats = {
        total: n,
        positives,
        negatives: neg,
        posPct,
        negPct
    };

    edaDiv.innerHTML = `
        <p><strong>Rows:</strong> ${n}</p>
        <p><strong>Class distribution:</strong> 0 → ${neg} (${negPct}%), 1 → ${positives} (${posPct}%)</p>
        <p><strong>Mean age:</strong> ${meanAge}</p>
        <p><strong>Mean BMI:</strong> ${meanBmi}</p>
    `;
}

function destroyCharts() {
    if (classChart) {
        classChart.destroy();
        classChart = null;
    }
    if (importanceChart) {
        importanceChart.destroy();
        importanceChart = null;
    }
}

// ===== 2. Train models =====
trainButton.addEventListener("click", async function () {
    if (rawRows.length === 0) return;
    trainButton.disabled = true;
    trainButton.textContent = "Training...";

    try {
        const { XTrain, yTrain, XTest, yTest } = prepareTensors(rawRows);
        const inputDim = featureOrder.length;

        // Logistic regression (1-layer NN)
        logModel = tf.sequential();
        logModel.add(tf.layers.dense({
            units: 1,
            activation: "sigmoid",
            inputShape: [inputDim]
        }));
        logModel.compile({
            optimizer: tf.train.adam(0.01),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });

        await logModel.fit(XTrain, yTrain, {
            epochs: 20,
            batchSize: 64,
            shuffle: true,
            verbose: 0
        });

        const logEval = logModel.evaluate(XTest, yTest);
        const logLoss = (await logEval[0].data())[0];
        const logAcc = (await logEval[1].data())[0];

        // Neural network with hidden layers
        nnModel = tf.sequential();
        nnModel.add(tf.layers.dense({
            units: 32,
            activation: "relu",
            inputShape: [inputDim]
        }));
        nnModel.add(tf.layers.dense({
            units: 16,
            activation: "relu"
        }));
        nnModel.add(tf.layers.dense({
            units: 1,
            activation: "sigmoid"
        }));
        nnModel.compile({
            optimizer: tf.train.adam(0.005),
            loss: "binaryCrossentropy",
            metrics: ["accuracy"]
        });

        await nnModel.fit(XTrain, yTrain, {
            epochs: 25,
            batchSize: 64,
            shuffle: true,
            verbose: 0
        });

        const nnEval = nnModel.evaluate(XTest, yTest);
        const nnLoss = (await nnEval[0].data())[0];
        const nnAcc = (await nnEval[1].data())[0];

        // Get logistic regression weights for feature importance
        const kernelTensor = logModel.getWeights()[0];
        const kernel = await kernelTensor.array(); // shape [inputDim, 1]
        const weights = kernel.map(row => row[0]); // flatten

        // Feature importance for "no diabetes" = negative weights (protective)
        const pairs = featureNames.map((name, i) => ({ name, weight: weights[i] }));
        const protective = pairs
            .filter(p => p.weight < 0)
            .sort((a, b) => a.weight - b.weight) // most negative first
            .slice(0, 3);

        const protectiveText = protective.length
            ? protective.map(p => `${p.name} (weight ${p.weight.toFixed(3)})`).join(", ")
            : "No clear protective features (all weights ≥ 0).";

        metricsDiv.innerHTML = `
            <p><strong>Logistic regression:</strong> accuracy ${(logAcc * 100).toFixed(2)}%, loss ${logLoss.toFixed(4)}</p>
            <p><strong>Neural network:</strong> accuracy ${(nnAcc * 100).toFixed(2)}%, loss ${nnLoss.toFixed(4)}</p>
            <p><strong>Most important features associated with <u>no diabetes</u> (protective):</strong> ${protectiveText}</p>
            <p class="hint">Positive weight → increases diabetes risk; negative weight → associated with lower risk.</p>
        `;

        renderCharts(weights);
        modelsReady = true;
        predictButton.disabled = false;
    } catch (err) {
        console.error(err);
        metricsDiv.innerHTML = "<p>Error while training models. Open console for details.</p>";
    } finally {
        trainButton.textContent = "Train models";
        trainButton.disabled = false;
    }
});

function prepareTensors(rows) {
    const maxRows = 5000;
    let data = rows;
    if (rows.length > maxRows) {
        data = shuffle(rows).slice(0, maxRows);
    }

    const X = [];
    const y = [];
    for (const r of data) {
        X.push([
            r.genderNum,
            r.age,
            r.hypertension,
            r.heart_disease,
            r.smokingCode,
            r.bmi,
            r.HbA1c_level,
            r.blood_glucose_level
        ]);
        y.push(r.diabetes);
    }

    const n = X.length;
    const indices = [...Array(n).keys()];
    shuffle(indices);

    const testRatio = 0.2;
    const testSize = Math.floor(n * testRatio);
    const testIdx = new Set(indices.slice(0, testSize));

    const XTrain = [];
    const yTrain = [];
    const XTest = [];
    const yTest = [];

    for (let i = 0; i < n; i++) {
        if (testIdx.has(i)) {
            XTest.push(X[i]);
            yTest.push(y[i]);
        } else {
            XTrain.push(X[i]);
            yTrain.push(y[i]);
        }
    }

    scaler = computeScaler(XTrain);

    const XTrainScaled = applyScaler(XTrain, scaler);
    const XTestScaled = applyScaler(XTest, scaler);

    const XTrainTensor = tf.tensor2d(XTrainScaled);
    const yTrainTensor = tf.tensor2d(yTrain, [yTrain.length, 1]);
    const XTestTensor = tf.tensor2d(XTestScaled);
    const yTestTensor = tf.tensor2d(yTest, [yTest.length, 1]);

    return { XTrain: XTrainTensor, yTrain: yTrainTensor, XTest: XTestTensor, yTest: yTestTensor };
}

function computeScaler(X) {
    const n = X.length;
    const d = X[0].length;
    const mean = new Array(d).fill(0);
    const std = new Array(d).fill(0);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            mean[j] += X[i][j];
        }
    }
    for (let j = 0; j < d; j++) {
        mean[j] /= n;
    }
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            const diff = X[i][j] - mean[j];
            std[j] += diff * diff;
        }
    }
    for (let j = 0; j < d; j++) {
        std[j] = Math.sqrt(std[j] / n) || 1;
    }
    return { mean, std };
}

function applyScaler(X, scaler) {
    const n = X.length;
    const d = X[0].length;
    const out = new Array(n);

    for (let i = 0; i < n; i++) {
        const row = new Array(d);
        for (let j = 0; j < d; j++) {
            row[j] = (X[i][j] - scaler.mean[j]) / scaler.std[j];
        }
        out[i] = row;
    }
    return out;
}

function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// ===== 3. Charts =====
function renderCharts(weights) {
    if (classStats) {
        const ctx1 = document.getElementById("classChart").getContext("2d");
        if (classChart) classChart.destroy();
        classChart = new Chart(ctx1, {
            type: "bar",
            data: {
                labels: ["No diabetes (0)", "Diabetes (1)"],
                datasets: [{
                    data: [classStats.negatives, classStats.positives]
                }]
            },
            options: {
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    const ctx2 = document.getElementById("importanceChart").getContext("2d");
    if (importanceChart) importanceChart.destroy();
    importanceChart = new Chart(ctx2, {
        type: "bar",
        data: {
            labels: featureNames,
            datasets: [{
                data: weights
            }]
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    ticks: {
                        autoSkip: false,
                        maxRotation: 60,
                        minRotation: 30
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// ===== 4. User prediction (fixed for multiple runs) =====
predictForm.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (!modelsReady || !nnModel) {
        predictionOutput.innerHTML = "<p>Train the models first.</p>";
        return;
    }

    try {
        const genderVal = document.getElementById("gender").value;
        const age = parseFloat(document.getElementById("age").value);
        const hypertension = parseInt(document.getElementById("hypertension").value);
        const heart = parseInt(document.getElementById("heart_disease").value);
        const smokingVal = document.getElementById("smoking_history").value;
        const bmi = parseFloat(document.getElementById("bmi").value);
        const hba1c = parseFloat(document.getElementById("hba1c").value);
        const glucose = parseFloat(document.getElementById("glucose").value);

        const genderNum = genderVal === "Male" ? 1 : 0;
        const smokingCode = smokingMap[smokingVal];

        const feats = [
            genderNum,
            age,
            hypertension,
            heart,
            smokingCode,
            bmi,
            hba1c,
            glucose
        ];

        const scaledRow = applyScaler([feats], scaler)[0];

        const xTensor = tf.tensor2d([scaledRow]);
        const probTensor = nnModel.predict(xTensor);
        const probArr = await probTensor.data();
        const prob = probArr[0];

        xTensor.dispose();
        probTensor.dispose();

        const label = prob >= 0.5 ? "Positive" : "Negative";
        const msg = label === "Positive"
            ? "You may be at higher risk of diabetes. Please consult a doctor for medical advice."
            : "Your predicted diabetes risk is low based on this model. This does not replace medical tests.";

        predictionOutput.innerHTML = `
            <p><strong>Prediction:</strong> Diabetes risk ${label}</p>
            <p><strong>Probability:</strong> ${(prob * 100).toFixed(2)}%</p>
            <p>${msg}</p>
        `;
    } catch (err) {
        console.error(err);
        predictionOutput.innerHTML = "<p>Error while predicting. Check console.</p>";
    }
});
