# Real-time Credit Card Fraud Detection System
**SMOTE + Ensemble ML (SVM · Random Forest · XGBoost) + Flask API**

**Student:** KOGANTI VENKATA ANAND | 24FE1F0071  
**Programme:** Master of Computer Applications  
**Institute:** VLITS | Guide: P. PADMINI RANI

---

## Project Structure

```
fraud_detection/
├── train_model.py       ← Step 1: Train the model
├── app.py               ← Step 2: Run the Flask API
├── templates/
│   └── index.html       ← Web UI for demo
├── requirements.txt     ← All dependencies
├── creditcard.csv       ← Place Kaggle dataset here (you download this)
├── fraud_model.pkl      ← Generated after training
└── scaler.pkl           ← Generated after training
```

---

## Setup (One-time)

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
- Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Download `creditcard.csv`
- Place it in this folder (same folder as `train_model.py`)

---

## How to Run

### Step 1: Train the Model
```bash
python train_model.py
```
This will:
- Load the 284,807-transaction dataset
- Apply StandardScaler to Amount and Time
- Apply SMOTE to balance fraud samples
- Train SVM + Random Forest + XGBoost ensemble
- Print full evaluation (Accuracy, F1, AUC-ROC)
- Save `fraud_model.pkl` and `scaler.pkl`

**Expected output:**
```
Accuracy          : 99.9x%
Precision (Fraud) : ~95%
Recall (Fraud)    : ~85-90%
F1-Score          : ~0.90+
AUC-ROC           : ~0.97+
```

### Step 2: Start the Flask API
```bash
python app.py
```

### Step 3: Open the Web UI
Visit: **http://127.0.0.1:5000/**

Click "Sample Legit" or "Sample Fraud" to test instantly.

---

## API Reference

### POST /predict
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [v1,v2,...,v28,amount,time]}'
```

**Response:**
```json
{
  "prediction": 1,
  "result": "FRAUD",
  "fraud_probability": 0.9821,
  "confidence_score": 0.9821,
  "risk_level": "CRITICAL",
  "recommendation": "BLOCK transaction immediately",
  "top_features": [
    {"feature": "V14", "value": -9.23, "impact": 9.23}
  ]
}
```

### POST /batch_predict
Send multiple transactions at once:
```json
{"transactions": [[30 values], [30 values], ...]}
```

### GET /health
```json
{"status": "ok", "model": "loaded"}
```

---

## Why This Approach Works

| Problem | Our Solution |
|---------|-------------|
| 0.17% fraud (imbalanced) | SMOTE generates synthetic fraud samples |
| Single model bias | Ensemble of SVM + RF + XGBoost |
| Static research code | Flask API for real-time deployment |
| Binary output only | Probability + risk level + feature insights |

---

## Metrics Explanation (for your report)

- **Accuracy** – Overall correct predictions (can be misleading on imbalanced data)
- **Precision** – Of flagged frauds, how many were actually fraud
- **Recall** – Of actual frauds, how many did we catch (most important!)
- **F1-Score** – Harmonic mean of precision and recall
- **AUC-ROC** – Model's ability to distinguish fraud from legit across all thresholds
