# 💳 Real-Time Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-API-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20SVM%20%7C%20RandomForest-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-99.9%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/github/license/VenkatAnand07/credit_card_fraud_detection?style=for-the-badge"/>
</p>

<p align="center">
  A Machine Learning-based system to detect fraudulent credit card transactions in real-time using <strong>SMOTE + Ensemble Models</strong>, deployed with a <strong>Flask Web Application</strong>.
</p>

<p align="center">
  <a href="https://github.com/VenkatAnand07/credit_card_fraud_detection">
    <img src="https://img.shields.io/badge/🔗 View%20on%20GitHub-181717?style=for-the-badge&logo=github"/>
  </a>
  <!-- Replace the URL below with your deployed app link when available -->
  <!-- <a href="https://your-deployed-app-url.com">
    <img src="https://img.shields.io/badge/🌐 Live%20Demo-00C7B7?style=for-the-badge&logo=netlify&logoColor=white"/>
  </a> -->
</p>

---

## 🚀 Key Highlights

| Feature | Description |
|--------|-------------|
| ⚡ Real-time Detection | Predicts fraud instantly via Flask API |
| 📊 Imbalanced Data Handling | Uses **SMOTE** to oversample minority (fraud) class |
| 🤖 Ensemble Learning | Combines **SVM, Random Forest & XGBoost** |
| 🎯 High Accuracy | Achieves ~99.9% accuracy with ~0.97+ AUC-ROC |
| 📈 Explainable Output | Returns fraud probability, risk level & feature impact |

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| ✅ Accuracy | ~99.9% |
| 🎯 Precision (Fraud) | ~95% |
| 🔍 Recall (Fraud) | ~85–90% |
| 📐 F1 Score | ~0.90+ |
| 📉 AUC-ROC | ~0.97+ |

> ⚠️ **Note:** Accuracy alone can be misleading for imbalanced datasets. **Recall** and **F1-score** are more critical metrics for fraud detection.

---

## 🧠 Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Serialization | Pickle |

---

## 📁 Project Structure

```
credit_card_fraud_detection/
│
├── app.py                  # Flask API for real-time predictions
├── train_model.py          # Model training script
├── fraud_model.pkl         # Trained ensemble model
├── scaler.pkl              # Feature scaler
├── feature_cols.pkl        # Feature column names
├── demo_samples.pkl        # Sample data for demo
├── confusion_matrix.png    # Model evaluation visualization
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI for prediction
└── creditcard.csv          # Dataset (not pushed to GitHub)
```

---

## ⚡ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/VenkatAnand07/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Optional — pre-trained model included)
```bash
python train_model.py
```

### 4️⃣ Run the Flask App
```bash
python app.py
```

### 5️⃣ Open in Browser
```
http://127.0.0.1:5000
```

---

## 🌐 Live Demo

> 🚧 Deployment coming soon! The app is currently running locally.  
> Once deployed, the live link will be updated here.

---

## 📌 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Fraud Rate:** ~0.17% (highly imbalanced)
- **Features:** 30 anonymized PCA features (V1–V28, Amount, Time)

---

## 👤 Author

**VenkatAnand07**  
🔗 GitHub: [@VenkatAnand07](https://github.com/VenkatAnand07)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).