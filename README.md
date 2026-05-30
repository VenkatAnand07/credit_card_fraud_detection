# 💳 FraudShield: Real-Time Credit Card Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-API-black?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-ESMOTE--GAN%20%7C%20Random%20Forest%20%7C%20XGBoost-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.9771-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/github/license/VenkatAnand07/credit_card_fraud_detection?style=for-the-badge"/>
</p>

An enhanced Machine Learning-based system to detect fraudulent credit card transactions in real-time. It is based on **ESMOTE-GAN** (Ensemble Synthesized Minority Oversampling-Based Generative Adversarial Networks) and a **Weighted Random Forest Ensemble**, boosted with a **global XGBoost classifier**, and served via an interactive **Flask Web Dashboard**.

---

## 📖 Research Foundation & Paper Reference

This project is an extended and enhanced implementation of the base research paper:
> **Paper Title:** *"Ensemble Synthesized Minority Oversampling-Based Generative Adversarial Networks and Random Forest Algorithm for Credit Card Fraud Detection"*  
> **Journal:** IEEE Access (2023)  
> **DOI:** [10.1109/ACCESS.2023.3306621](https://doi.org/10.1109/ACCESS.2023.3306621)

---

## 🚀 Key Enhancements over the Base Paper

The base paper focuses primarily on offline SMOTE-GAN generation and Random Forest classification. This project introduces four major production-grade enhancements:

1. **Hybrid XGBoost Ensemble Boosting:** A global XGBoost classifier trained on all augmented data is integrated alongside the five RF sub-classifiers via a weighted probabilistic voting scheme.
2. **Adaptive PR-Optimal Thresholding:** Instead of using a standard fixed `0.5` decision boundary (which leads to sub-optimal F1 scores on highly imbalanced fraud datasets), the model automatically searches the Precision-Recall curve to find the threshold that maximizes the F1-Score (`0.9765`).
3. **Interactive Dark Cybernetic Dashboard:** A real-time web portal ("*Be Aware Of Frauds*") built with Flask, styled using a premium, dark-mode visual interface with a dynamic cyber grid layout.
4. **Explainable AI (XAI) Reason Logs:** Every prediction returns natural-language explanations explaining *why* a transaction was flagged, mapping anonymized PCA variables to human-readable spending, velocity, and authentication signals.

---

## 📈 Model Performance & Metrics

The ensemble was evaluated on the Kaggle Credit Card Fraud Detection dataset (Stratified 80/20 train/test split). 

Below are the actual metrics stored in the trained model (`fraud_model.pkl`):

| Metric | Score | Key Takeaway |
|:---|:---|:---|
| **AUC-ROC** | **0.9771** | Exceptional class separation ability |
| **Precision** | **86.21%** | Highly reliable fraud alerts with low false alarms |
| **Recall** | **76.53%** | Captures over 3/4 of all true fraud occurrences |
| **F1-Score** | **0.8108** | Strong balanced performance on the minority class |
| **False Alarm Rate (FPR)** | **0.02%** | Legitimate transactions are rarely misclassified |
| **Decision Threshold** | **0.9765** | Dynamically tuned to maximize precision & recall trade-off |

---

## 🧠 System Architecture

The model uses a multi-stage training pipeline:
1. **ESMOTE Partitioning:** Splits the imbalanced training data into 5 diverse subsets using combination undersampling (to ~10:1) and SMOTE (to ~2:1).
2. **Lightweight GAN Generation:** Fits a custom MLP Generative Adversarial Network (Generator + Discriminator) on the fraud samples of each subset to synthesize high-fidelity fake fraud samples.
3. **Sub-Ensemble Training:** Trains 5 independent Random Forest models on each augmented subset, weighting their votes using their validation AUC.
4. **Global XGBoost Integration:** Fits a global gradient-boosted tree model over all subsets.
5. **Weighted Probability Fusion:** Melds the outputs of the RF Sub-Ensembles and the XGBoost model to produce the final fraud probability.

---

## 📁 Project Structure

```
credit_card_fraud_detection/
│
├── app.py                  # Flask Web App & Real-Time Prediction API
├── train_model.py          # Model training pipeline (ESMOTE-GAN + Weighted RF + XGBoost)
├── fraud_model.pkl         # Serialized final ensemble model, weights, & metrics
├── scaler.pkl              # Fitted StandardScaler for 'Time' & 'Amount'
├── feature_cols.pkl        # Serialized feature column order
├── demo_samples.pkl        # Pre-extracted sample data for UI demo buttons
├── confusion_matrix.png    # Heatmap visualization of final test set results
├── requirements.txt        # Python dependency manifest
├── templates/
│   └── index.html          # Cyberpunk-style interactive Web UI
└── creditcard.csv          # Dataset file (not committed to Git)
```

---

## ⚡ Setup & Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/VenkatAnand07/credit_card_fraud_detection.git
cd credit_card_fraud_detection
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.10+ installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model (Optional)
The pre-trained model is already included. If you want to retrain the pipeline using the dataset:
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Save it in the project root directory.
3. Run:
```bash
python train_model.py
```

### 4️⃣ Launch the Flask Dashboard
Start the application:
```bash
python app.py
```

### 5️⃣ Access the App
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🖥️ Web Interface Features

- **Live System Health Check:** Queries `/model_info` and `/health` to show online status.
- **One-Click Legit/Fraud Demos:** Automatically pre-fills inputs using real test-set samples to demonstrate model behavior instantly.
- **Interpolative Feature Construction:** Dynamic HTML dropdowns map to semantic categories (Time of Day, Location, Velocity, Card Age, and Transaction Type) which automatically adjust PCA features ($V_1$ to $V_{28}$) behind the scenes to test custom manual scenarios.
- **Scan Visualization:** Runs a live scan line micro-animation while processing predictions.
- **Risk Level Alerts:** Displays risk levels dynamically: `LOW` (green), `MEDIUM` (amber), `HIGH` (red), and `CRITICAL` (deep red).
- **Explainable Insights:** Outlines the top-5 feature influences and provides natural language actionable advice (e.g., *Allow*, *Monitor*, *Flag for review*, *Block immediately*).

---

## 👤 Author & Academic Details

* **Author:** Koganti Venkata Anand  
* **Roll Number:** 24FE1F0071  
* **Degree:** Master of Computer Applications (MCA)  
* **Institution:** Vignan's Lara Institute of Technology & Science (VLITS)  
* **GitHub:** [@VenkatAnand07](https://github.com/VenkatAnand07)

---

## 📄 License

This project is open-source and released under the [MIT License](LICENSE).