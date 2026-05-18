"""
Credit Card Fraud Detection - train_model.py
Author: KOGANTI VENKATA ANAND | 24FE1F0071 | MCA | VLITS
Ensemble: Random Forest + XGBoost (SVM removed - causes bias on sample training)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)
df = pd.read_csv('creditcard.csv')
print(f"  Total rows     : {len(df):,}")
print(f"  Legitimate (0) : {(df['Class']==0).sum():,}")
print(f"  Fraud (1)      : {(df['Class']==1).sum():,}")
print(f"  Fraud %        : {(df['Class']==1).sum()/len(df)*100:.4f}%")
print(f"  Column order   : {list(df.columns)}")


print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)
X = df.drop('Class', axis=1)
y = df['Class']

COLS = list(X.columns)
joblib.dump(COLS, 'feature_cols.pkl')
print(f"  Column order: {COLS[:3]} ... {COLS[-3:]}")

scaler = StandardScaler()
X = X.copy()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
joblib.dump(scaler, 'scaler.pkl')
print("  scaler.pkl saved")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")


print("\n" + "=" * 60)
print("STEP 3: Applying SMOTE")
print("=" * 60)
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"  Before -> Legit:{(y_train==0).sum():,} | Fraud:{(y_train==1).sum():,}")
print(f"  After  -> Legit:{(y_res==0).sum():,} | Fraud:{(y_res==1).sum():,}")


print("\n" + "=" * 60)
print("STEP 4: Training Models (RF + XGBoost)")
print("=" * 60)

print("  [1/2] Training Random Forest (full data)...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    random_state=42, n_jobs=-1
)
rf.fit(X_res, y_res)
print("  Done!")

print("  [2/2] Training XGBoost (full data)...")
xgb = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, n_jobs=-1
)
xgb.fit(X_res, y_res)
print("  Done!")


print("\n" + "=" * 60)
print("STEP 5: Finding demo samples")
print("=" * 60)

def get_proba(row_df):
    rf_p  = float(rf.predict_proba(row_df)[0][1])
    xgb_p = float(xgb.predict_proba(row_df)[0][1])
    return rf_p * 0.4 + xgb_p * 0.6

# Find best fraud sample with non-zero amount
best_p, best_idx = 0, None
for idx_row, row in X_test[y_test==1].iterrows():
    orig = df.loc[idx_row, COLS]
    row_df = pd.DataFrame([row], columns=COLS)
    p = get_proba(row_df)
    if orig['Amount'] > 0 and p > best_p:
        best_p = p
        best_idx = idx_row
        if p >= 0.95:
            break

fraud_sample = df.loc[best_idx, COLS].tolist()
print(f"  Fraud sample: index={best_idx} Amount={df.loc[best_idx,'Amount']} prob={best_p:.4f}")

# Find legit sample
legit_idx = None
for idx_row, row in X_test[y_test==0].iterrows():
    row_df = pd.DataFrame([row], columns=COLS)
    p = get_proba(row_df)
    if p <= 0.02:
        legit_idx = idx_row
        break

legit_sample = df.loc[legit_idx, COLS].tolist()
print(f"  Legit sample: index={legit_idx} Amount={df.loc[legit_idx,'Amount']} prob={get_proba(pd.DataFrame([X_test.loc[legit_idx]], columns=COLS)):.4f}")

joblib.dump({'fraud': fraud_sample, 'legit': legit_sample, 'cols': COLS}, 'demo_samples.pkl')
print("  demo_samples.pkl saved")


print("\n" + "=" * 60)
print("STEP 6: Evaluation")
print("=" * 60)
rf_prob  = rf.predict_proba(X_test)[:,1]
xgb_prob = xgb.predict_proba(X_test)[:,1]
y_prob   = xgb_prob*0.6 + rf_prob*0.4
y_pred   = (y_prob >= 0.5).astype(int)

print(f"  Accuracy  : {accuracy_score(y_test,y_pred)*100:.2f}%")
print(f"  Precision : {precision_score(y_test,y_pred)*100:.2f}%")
print(f"  Recall    : {recall_score(y_test,y_pred)*100:.2f}%")
print(f"  F1-Score  : {f1_score(y_test,y_pred):.4f}")
print(f"  AUC-ROC   : {roc_auc_score(y_test,y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Legitimate','Fraud']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate','Fraud'],
            yticklabels=['Legitimate','Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("  confusion_matrix.png saved")


print("\n" + "=" * 60)
print("STEP 7: Saving Models")
print("=" * 60)
joblib.dump({'rf': rf, 'xgb': xgb}, 'fraud_model.pkl')
print("  fraud_model.pkl saved")

print("\n" + "=" * 60)
print("ALL DONE! Now run: python app.py")
print("=" * 60)
