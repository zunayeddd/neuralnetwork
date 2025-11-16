<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI-Based Diabetes Risk Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <!-- Google Fonts for modern typography -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <!-- TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.21.0/dist/tf.min.js"></script>
    <!-- PapaParse for CSV -->
    <script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>
    <!-- Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar">
        <div class="navbar-container">
            <a href="#" class="logo">Diabetes AI Predictor</a>
            <ul class="nav-links">
                <li><a href="#">Home</a></li>
                <li><a href="#features">Features</a></li>
                <li><a href="#prediction">Prediction</a></li>
                <li><a href="#faqs">FAQs</a></li>
                <li><a href="#about">About Us</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
        </div>
    </nav>

    <div class="container">
        <h1>AI-Based Diabetes Risk Prediction</h1>

        <!-- 1. Dataset upload and training -->
        <section id="features" class="card">
            <h2>Step 1: Upload Dataset and Train Models</h2>
            <p class="hint">
                Upload the diabetes CSV dataset with columns: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes.
            </p>
            <input type="file" id="datasetInput" accept=".csv" aria-label="Upload CSV dataset">
            <button id="trainButton" disabled>Train Models</button>

            <div id="eda"></div>
            <div id="metrics"></div>

            <div class="charts">
                <div class="chart-container">
                    <h3>Class Distribution</h3>
                    <canvas id="classChart" height="180"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Feature Importance (Logistic Regression)</h3>
                    <canvas id="importanceChart" height="180"></canvas>
                </div>
            </div>
        </section>

        <!-- 2. User prediction -->
        <section id="prediction" class="card">
            <h2>Step 2: Check Your Diabetes Risk</h2>
            <p class="hint">Enter your details below for an AI-powered prediction. Train models first for accurate results.</p>

            <form id="predictForm">
                <div class="grid">
                    <div class="field">
                        <label for="gender">Gender</label>
                        <select id="gender" required aria-required="true">
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="age">Age (years)</label>
                        <input type="number" id="age" step="0.1" required aria-required="true">
                    </div>

                    <div class="field">
                        <label for="hypertension">Hypertension</label>
                        <select id="hypertension">
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="heart_disease">Heart Disease</label>
                        <select id="heart_disease">
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="smoking_history">Smoking History</label>
                        <select id="smoking_history">
                            <option value="never">Never</option>
                            <option value="former">Former</option>
                            <option value="current">Current</option>
                            <option value="not current">Not Current</option>
                            <option value="ever">Ever</option>
                            <option value="No Info">No Info</option>
                        </select>
                    </div>

                    <div class="field">
                        <label for="bmi">BMI</label>
                        <input type="number" id="bmi" step="0.1" required aria-required="true">
                    </div>

                    <div class="field">
                        <label for="hba1c">HbA1c Level</label>
                        <input type="number" id="hba1c" step="0.1" required aria-required="true">
                    </div>

                    <div class="field">
                        <label for="glucose">Blood Glucose Level</label>
                        <input type="number" id="glucose" required aria-required="true">
                    </div>
                </div>

                <div class="form-actions">
                    <button type="submit" id="predictButton" disabled>Predict Risk</button>
                    <button type="reset" class="reset-button">Reset</button>
                </div>
            </form>

            <div id="predictionOutput" class="prediction-output">
                <!-- JS will populate this with results, including potential progress circle -->
            </div>
        </section>

        <!-- Additional Sections (Placeholders for FAQs, About, Contact) -->
        <section id="faqs" class="card">
            <h2>FAQs</h2>
            <p>Common questions about diabetes prediction and how our AI works.</p>
            <!-- Add accordion or list here if expanding -->
        </section>

        <section id="about" class="card">
            <h2>About Us</h2>
            <p>We are dedicated to using AI for health insights.</p>
        </section>

        <section id="contact" class="card">
            <h2>Contact</h2>
            <p>Get in touch for more information.</p>
        </section>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="footer-container">
            <p>&copy; 2025 Diabetes AI Predictor. All rights reserved.</p>
            <ul class="footer-links">
                <li><a href="#">Privacy Policy</a></li>
                <li><a href="#">Terms of Service</a></li>
                <li><a href="#contact">Contact Us</a></li>
            </ul>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>
