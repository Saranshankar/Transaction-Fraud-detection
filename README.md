# Transaction-Fraud-detection
Smart fraud detection web app that analyzes transaction details and predicts fraud risk using Machine Learning. Includes real-time predictions, probability scoring, and interactive UI.

Online Transaction Fraud Detection System

 Overview:

This project is a Machine Learning-based Web Application that detects fraudulent online transactions. It uses a trained Random Forest Classifier to analyze transaction details and predict whether a transaction is fraudulent or legitimate.

The system is built using Flask (backend) and a simple HTML interface (frontend), making it easy to test predictions in real-time.

🚀 Features

🔍 Detects fraud transactions using ML model
📊 Displays fraud probability percentage
🌐 User-friendly web interface
⚡ Real-time prediction with Flask
📈 Data visualization for fraud vs normal transactions
🧠 Feature importance analysis

🛠️ Technologies Used

Python
Flask
NumPy & Pandas
Scikit-learn (Random Forest)
Matplotlib
HTML & CSS
Pickle (Model saving/loading)

📂 Project Structure
├── dataset/
│   └── creditcard_reduced.csv
├── model/
│   └── fraud_model.pkl
├── templates/
│   └── index.html
├── app.py
├── train_model.py
└── README.md

⚙️ How It Works

1️⃣ Data Processing
Load transaction dataset
Separate features (X) and target (y)
Perform train-test split

2️⃣ Model Training
Train using RandomForestClassifier
Evaluate using accuracy score
Save model using pickle

3️⃣ Prediction
User inputs transaction details
Data is converted into feature vector (30 features)
Model predicts:
Fraud (1)
Normal (0)
Displays probability of fraud

📊 Input Features
Transaction Amount
Transaction Time
Location Risk Score
Merchant Category Score
Transaction Risk Score


📈 Output
✅ Normal Transaction
⚠ Fraud Transaction Detected
📊 Fraud Probability (%)

