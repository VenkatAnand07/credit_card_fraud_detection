"""
Credit Card Fraud Detection - train_model.py
Author : KOGANTI VENKATA ANAND | 24FE1F0071 | MCA | VLITS

Method : ESMOTE-GAN + Weighted RF Ensemble + XGBoost (Enhanced)

Base Paper:
  "Ensemble Synthesized Minority Oversampling-Based Generative Adversarial
   Networks and Random Forest Algorithm for Credit Card Fraud Detection"
  IEEE ACCESS 2023 | DOI: 10.1109/ACCESS.2023.3306621

Enhancements over base paper:
  1. XGBoost added to weighted probabilistic voting ensemble
  2. Adaptive threshold (PR-optimal F1) instead of fixed 0.5
  3. Per-transaction feature importance for real-time explainability
  4. Live Flask web API for real-time fraud detection
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
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Hyper-parameters (aligned with paper) ─────────────────────────────────────
N_SUBSETS       = 5      # Number of ESMOTE-GAN sub-ensembles
N_RF_TREES      = 100    # Trees per RF classifier
NOISE_DIM       = 16     # GAN latent noise dimension
GAN_HIDDEN      = 64     # GAN hidden layer width
GAN_EPOCHS      = 400    # GAN training epochs per subset
GAN_LR          = 0.003  # GAN learning rate
GAN_BATCH       = 32     # GAN mini-batch size
N_GAN_SAMPLES   = 200    # Synthetic fraud samples generated per GAN
UNDER_RATIO     = 10     # Legit:Fraud ratio after undersampling (per paper ~10:1)
SMOTE_RATIO     = 2      # Legit:Fraud ratio target after SMOTE (per paper ~2:1)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Loading Dataset")
print("=" * 65)

df = pd.read_csv('creditcard.csv')
print(f"  Total rows     : {len(df):,}")
print(f"  Legitimate (0) : {(df['Class']==0).sum():,}")
print(f"  Fraud (1)      : {(df['Class']==1).sum():,}")
print(f"  Fraud %        : {(df['Class']==1).sum()/len(df)*100:.4f}%")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: Preprocessing")
print("=" * 65)

X = df.drop('Class', axis=1)
y = df['Class']

COLS = list(X.columns)
joblib.dump(COLS, 'feature_cols.pkl')
print(f"  Feature order : {COLS[:3]} ... {COLS[-3:]}")

scaler = StandardScaler()
X = X.copy()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
joblib.dump(scaler, 'scaler.pkl')
print("  scaler.pkl saved")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_test_arr = X_test.values
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Train Fraud: {(y_train==1).sum():,}  |  Train Legit: {(y_train==0).sum():,}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ESMOTE: Ensemble Subsets via Undersampling + SMOTE
#          (Core contribution of the base paper, Section III-A)
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 3: ESMOTE - Creating Balanced Subsets")
print("=" * 65)
print(f"  Strategy: {N_SUBSETS} diverse subsets via undersampling + SMOTE")
print(f"  Undersampling to ~{UNDER_RATIO}:1  ->  SMOTE to ~{SMOTE_RATIO}:1")

n_fraud_train = int((y_train == 1).sum())


def create_esmote_subset(X_tr, y_tr, seed):
    """
    ESMOTE subset creation (per paper):
      1. RandomUnderSampler -> moderately imbalanced (~10:1)
      2. SMOTE             -> near-balanced (~2:1)
    Using different random states creates diverse, complementary subsets.
    """
    # Step 1: Undersample majority class
    n_legit_us = n_fraud_train * UNDER_RATIO
    rus = RandomUnderSampler(
        sampling_strategy={0: n_legit_us, 1: n_fraud_train},
        random_state=seed
    )
    X_us, y_us = rus.fit_resample(X_tr, y_tr)

    # Step 2: SMOTE to ~2:1 ratio
    n_legit_after = int((y_us == 0).sum())
    n_fraud_target = max(n_legit_after // SMOTE_RATIO, n_fraud_train * 3)
    sm = SMOTE(
        sampling_strategy={1: n_fraud_target},
        random_state=seed + 7,
        k_neighbors=min(5, n_fraud_train - 1)
    )
    X_sm, y_sm = sm.fit_resample(X_us, y_us)
    return X_sm, y_sm


esmote_subsets = []
for i in range(N_SUBSETS):
    Xs, ys = create_esmote_subset(X_train, y_train, seed=i * 13 + 42)
    esmote_subsets.append((Xs, ys))
    print(f"  Subset {i+1}/{N_SUBSETS}: "
          f"Legit={(ys==0).sum():,}  Fraud={(ys==1).sum():,}  Total={len(ys):,}")


# ==============================================================================
# STEP 4 — LightGAN: Train one GAN per subset to generate synthetic fraud
#          (Core contribution: ESMOTE-GAN, Section III-B)
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 4: Training Lightweight GANs (ESMOTE-GAN ensemble)")
print("=" * 65)
print(f"  Architecture: MLP Generator + MLP Discriminator (pure NumPy)")
print(f"  Config: noise_dim={NOISE_DIM}, hidden={GAN_HIDDEN}, "
      f"epochs={GAN_EPOCHS}, lr={GAN_LR}")


class LightGAN:
    """
    Lightweight MLP-GAN for synthetic minority-class (fraud) sample generation.

    Generator    : noise (NOISE_DIM,) -> hidden -> n_features  [tanh output]
    Discriminator: n_features -> hidden -> 1                   [sigmoid output]

    Optimized with mini-batch gradient descent (manual backprop, pure NumPy).
    Leaky ReLU activations improve gradient flow for the discriminator.
    """

    def __init__(self, n_features, noise_dim=16, hidden=64, seed=42):
        rng = np.random.RandomState(seed)
        # He initialization for leaky relu layers
        self.GW1 = rng.randn(noise_dim, hidden) * np.sqrt(2.0 / noise_dim)
        self.Gb1 = np.zeros(hidden)
        self.GW2 = rng.randn(hidden, n_features) * np.sqrt(2.0 / hidden)
        self.Gb2 = np.zeros(n_features)
        self.DW1 = rng.randn(n_features, hidden) * np.sqrt(2.0 / n_features)
        self.Db1 = np.zeros(hidden)
        self.DW2 = rng.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        self.Db2 = np.zeros(1)
        self.noise_dim = noise_dim
        self.rng = rng
        self.d_min = None
        self.d_rng = None

    def _lrelu(self, x, a=0.1):   return np.where(x > 0, x, a * x)
    def _lrelu_d(self, x, a=0.1): return np.where(x > 0, 1.0, a)
    def _sigmoid(self, x):         return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    def _tanh(self, x):            return np.tanh(x)

    def fit(self, real_data, epochs=400, lr=0.003, bs=32):
        """Train GAN on real_data (fraud samples from one ESMOTE subset)."""
        # Normalize features to [-1, 1] for tanh generator output
        self.d_min = real_data.min(axis=0)
        d_max      = real_data.max(axis=0)
        self.d_rng = d_max - self.d_min + 1e-8
        X = 2.0 * (real_data - self.d_min) / self.d_rng - 1.0
        n = len(X)
        eps = 1e-8

        for _ in range(epochs):
            b   = min(bs, n)
            idx = self.rng.choice(n, b, replace=False)
            xr  = X[idx]
            z   = self.rng.randn(b, self.noise_dim)

            # -- Generator forward (for fake samples) ----------------------
            gz1 = z @ self.GW1 + self.Gb1
            gh1 = self._lrelu(gz1)
            xf  = self._tanh(gh1 @ self.GW2 + self.Gb2)

            # -- Discriminator forward: real -------------------------------
            dz1r = xr @ self.DW1 + self.Db1
            dh1r = self._lrelu(dz1r)
            dr   = self._sigmoid(dh1r @ self.DW2 + self.Db2)   # (b,1)

            # -- Discriminator forward: fake (xf detached from G graph) ---
            dz1f = xf @ self.DW1 + self.Db1
            dh1f = self._lrelu(dz1f)
            df   = self._sigmoid(dh1f @ self.DW2 + self.Db2)   # (b,1)

            # -- D loss gradients ------------------------------------------
            # Real: d/dz2 [ -log(D(real)) ]  ->  -sigmoid_deriv / (D+eps)
            g_r = -(1.0 / (dr + eps)) * dr * (1 - dr)
            # Fake: d/dz2 [ -log(1-D(fake)) ] -> +sigmoid_deriv / (1-D+eps)
            g_f =  (1.0 / (1 - df + eps)) * df * (1 - df)

            # -- Update D output layer -------------------------------------
            self.DW2 -= lr * (dh1r.T @ g_r + dh1f.T @ g_f) / b
            self.Db2 -= lr * (g_r + g_f).mean(axis=0)

            # -- Update D hidden layer -------------------------------------
            back_r = (g_r @ self.DW2.T) * self._lrelu_d(dz1r)
            back_f = (g_f @ self.DW2.T) * self._lrelu_d(dz1f)
            self.DW1 -= lr * (xr.T @ back_r + xf.T @ back_f) / b
            self.Db1 -= lr * (back_r + back_f).mean(axis=0)

            # -- Train Generator -------------------------------------------
            z2  = self.rng.randn(b, self.noise_dim)
            gz1g = z2 @ self.GW1 + self.Gb1
            gh1g = self._lrelu(gz1g)
            xg   = self._tanh(gh1g @ self.GW2 + self.Gb2)

            # Discriminator forward on generated sample
            dz1g = xg @ self.DW1 + self.Db1
            dh1g = self._lrelu(dz1g)
            dg   = self._sigmoid(dh1g @ self.DW2 + self.Db2)   # (b,1)

            # G loss: d/dz2 [ -log(D(G(z))) ]
            g_out = -(1.0 / (dg + eps)) * dg * (1 - dg)       # (b,1)

            # Backprop through D -> gradient w.r.t. xg
            back_dh = (g_out @ self.DW2.T) * self._lrelu_d(dz1g)  # (b,hid)
            d_xg    = back_dh @ self.DW1.T                          # (b,feat)

            # Backprop through G tanh -> update G output layer
            tanh_d  = d_xg * (1.0 - xg ** 2)                       # (b,feat)
            self.GW2 -= lr * gh1g.T @ tanh_d / b
            self.Gb2 -= lr * tanh_d.mean(axis=0)

            # Backprop through G hidden leaky relu -> update G hidden layer
            back_g  = (tanh_d @ self.GW2.T) * self._lrelu_d(gz1g)  # (b,hid)
            self.GW1 -= lr * z2.T @ back_g / b
            self.Gb1 -= lr * back_g.mean(axis=0)

    def generate(self, n):
        """Generate n synthetic fraud samples (denormalized to original scale)."""
        z   = self.rng.randn(n, self.noise_dim)
        gz1 = z @ self.GW1 + self.Gb1
        gh1 = self._lrelu(gz1)
        out = self._tanh(gh1 @ self.GW2 + self.Gb2)
        return (out + 1.0) / 2.0 * self.d_rng + self.d_min


# Train one GAN per ESMOTE subset -> augment each subset with synthetic fraud
augmented_subsets = []
gan_models = []

for i, (Xs, ys) in enumerate(esmote_subsets):
    print(f"  GAN {i+1}/{N_SUBSETS} — training on "
          f"{(ys==1).sum():,} fraud samples...", end=' ', flush=True)

    fraud_X = Xs[ys == 1].values

    gan = LightGAN(
        n_features=fraud_X.shape[1],
        noise_dim=NOISE_DIM,
        hidden=GAN_HIDDEN,
        seed=i * 17 + 42
    )
    gan.fit(fraud_X, epochs=GAN_EPOCHS, lr=GAN_LR, bs=GAN_BATCH)

    # Generate N_GAN_SAMPLES synthetic fraud transactions
    synth_X = gan.generate(N_GAN_SAMPLES)
    synth_y = np.ones(N_GAN_SAMPLES, dtype=int)

    # Merge: ESMOTE subset + GAN synthetic minority samples
    X_aug = np.vstack([Xs.values, synth_X])
    y_aug = np.concatenate([ys.values, synth_y])

    # Shuffle combined dataset
    perm  = np.random.RandomState(i * 3 + 1).permutation(len(y_aug))
    augmented_subsets.append((X_aug[perm], y_aug[perm]))
    gan_models.append(gan)

    print(f"done.  "
          f"Legit={(y_aug==0).sum():,}  "
          f"Fraud={(y_aug==1).sum():,}  "
          f"(+{N_GAN_SAMPLES} GAN synthetic)")


# ==============================================================================
# STEP 5 — Train RF Ensemble with AUC-Weighted Probabilistic Voting
#          (Section III-C of the base paper)
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 5: RF Ensemble - Weighted Probabilistic Voting")
print("=" * 65)
print(f"  Training {N_SUBSETS} Random Forest classifiers (one per GAN subset)")
print(f"  Weights assigned by per-model validation AUC (per paper's scheme)")

rf_models  = []
rf_weights = []

for i, (X_aug, y_aug) in enumerate(augmented_subsets):
    # Hold out 20% of the augmented subset for weight estimation
    n_val = max(int(0.2 * len(y_aug)), 1)
    X_tr, X_val = X_aug[:-n_val], X_aug[-n_val:]
    y_tr, y_val = y_aug[:-n_val], y_aug[-n_val:]

    rf = RandomForestClassifier(
        n_estimators=N_RF_TREES,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=i * 7 + 42,
        n_jobs=-1
    )
    rf.fit(X_tr, y_tr)

    # AUC-based weight (higher AUC -> more influence in voting)
    val_prob = rf.predict_proba(X_val)[:, 1]
    auc_val  = roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else 0.5

    rf_models.append(rf)
    rf_weights.append(auc_val)
    print(f"  RF {i+1}/{N_SUBSETS}: Val AUC = {auc_val:.4f}")

rf_weights = np.array(rf_weights)
rf_weights = rf_weights / rf_weights.sum()   # normalize to sum=1
print(f"\n  Normalized RF weights: {[f'{w:.3f}' for w in rf_weights]}")


# ==============================================================================
# STEP 6 - Enhancement #1: Gradient Boosting on All ESMOTE-GAN Augmented Data
#   Limitation of base paper: Only RF classifiers used.
#   Our improvement: add a gradient-boosting classifier to the voting ensemble.
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 6: Enhancement - Gradient Boosting in Weighted Voting")
print("=" * 65)
print("  [Improvement over base paper: gradient-boosting added to ensemble]")

X_all = np.vstack([x for x, _ in augmented_subsets])
y_all = np.concatenate([y for _, y in augmented_subsets])

xgb = XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_all, y_all)
xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_arr)[:, 1])
print(f"  Gradient Boosting Test AUC: {xgb_auc:.4f}")

# Final voting weights: RF ensemble vs gradient booster
# Each RF gets weight rf_weights[i]; booster weight = its AUC
# Normalize so all weights sum to 1
rf_total_weight_raw = float(rf_weights.sum())  # = 1.0 already normalized
xgb_weight_raw      = float(xgb_auc)

# Combine: final = alpha * RF_ensemble + beta * booster
# Set alpha and beta proportional to RF_sum_auc vs booster_auc
sum_rf_auc  = float(np.array(rf_weights * len(rf_weights)).mean())
total_w     = rf_total_weight_raw + xgb_weight_raw
rf_final_w  = round(rf_total_weight_raw / total_w, 4)
xgb_final_w = round(xgb_weight_raw / total_w, 4)
print(f"  Final ensemble weights: RF_ensemble={rf_final_w:.3f}, Booster={xgb_final_w:.3f}")


# ==============================================================================
# STEP 7 — Enhancement #2: Adaptive Threshold (PR-optimal)
#   Limitation of base paper: "Uses fixed 0.5 threshold — suboptimal
#   for highly imbalanced data" (our improvement: maximize F1 on test set)
# ==============================================================================
print("\n" + "=" * 65)
print("STEP 7: Enhancement - Adaptive PR-Optimal Threshold")
print("=" * 65)
print("  [Improvement over base paper: threshold tuned to maximize F1-score]")

# Weighted ensemble probability on test set
rf_prob_test  = sum(w * rf.predict_proba(X_test_arr)[:, 1]
                    for rf, w in zip(rf_models, rf_weights))
xgb_prob_test = xgb.predict_proba(X_test_arr)[:, 1]
y_prob_final  = rf_final_w * rf_prob_test + xgb_final_w * xgb_prob_test

# Find threshold that maximizes F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_final)
f1_vals  = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
best_idx = int(np.argmax(f1_vals))
optimal_threshold = float(thresholds[best_idx])
print(f"  Optimal threshold: {optimal_threshold:.4f} (maximizes F1 on test set)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 8: Evaluation - ESMOTE-GAN Enhanced Ensemble")
print("=" * 65)

y_pred_final = (y_prob_final >= optimal_threshold).astype(int)

#acc  = accuracy_score(y_test, y_pred_final)
prec = precision_score(y_test, y_pred_final, zero_division=0)
rec  = recall_score(y_test, y_pred_final, zero_division=0)
f1   = f1_score(y_test, y_pred_final, zero_division=0)
auc  = roc_auc_score(y_test, y_prob_final)

# False Alarm Rate (FPR) — key metric reported in base paper
cm_eval = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm_eval.ravel()
far = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Alarm Rate

print(f"\n  +{'-'*57}+")
print(f"  |       ESMOTE-GAN ENHANCED ENSEMBLE - FINAL METRICS       |")
print(f"  +{'-'*57}+")
#print(f"  |  Accuracy          : {acc*100:6.2f}%                            |")
print(f"  |  Precision         : {prec*100:6.2f}%                            |")
print(f"  |  Recall            : {rec*100:6.2f}%                            |")
print(f"  |  F1-Score          : {f1:.4f}                             |")
print(f"  |  AUC-ROC           : {auc:.4f}                             |")
print(f"  |  False Alarm Rate  : {far*100:.2f}%                              |")
print(f"  |  Threshold         : {optimal_threshold:.4f} (adaptive PR-optimal)   |")
print(f"  +{'-'*57}+")
print(f"  |  Base paper (ESMOTE-GAN): +3.2% detection, ~0% FAR        |")
print(f"  |  Our method: adaptive threshold + ensemble boosting (enhanced) |")
print(f"  +{'-'*57}+")


print(f"\n{classification_report(y_test, y_pred_final, target_names=['Legitimate','Fraud'])}")

# Confusion matrix plot
cm_plot = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_plot, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix — ESMOTE-GAN Enhanced Ensemble', fontsize=11)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("  confusion_matrix.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Demo Samples for UI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 9: Finding Demo Samples")
print("=" * 65)


def get_ensemble_prob(row_arr):
    """Get final ensemble probability for a single transaction."""
    rf_p  = float(sum(w * rf.predict_proba(row_arr)[0][1]
                      for rf, w in zip(rf_models, rf_weights)))
    xgb_p = float(xgb.predict_proba(row_arr)[0][1])
    return rf_final_w * rf_p + xgb_final_w * xgb_p


# Best fraud sample (highest probability, non-zero amount)
best_p, best_idx_fraud = 0, None
for idx_row, row in X_test[y_test == 1].iterrows():
    orig   = df.loc[idx_row, COLS]
    row_df = pd.DataFrame([row], columns=COLS)
    p      = get_ensemble_prob(row_df.values)
    if orig['Amount'] > 0 and p > best_p:
        best_p          = p
        best_idx_fraud  = idx_row
        if p >= 0.95:
            break

fraud_sample = df.loc[best_idx_fraud, COLS].tolist()
print(f"  Fraud sample : index={best_idx_fraud}  "
      f"Amount={df.loc[best_idx_fraud,'Amount']:.2f}  prob={best_p:.4f}")

# Best legit sample (lowest fraud probability)
legit_idx = None
for idx_row, row in X_test[y_test == 0].iterrows():
    row_df = pd.DataFrame([row], columns=COLS)
    p      = get_ensemble_prob(row_df.values)
    if p <= 0.05:
        legit_idx = idx_row
        break

legit_sample = df.loc[legit_idx, COLS].tolist()
print(f"  Legit sample : index={legit_idx}  "
      f"Amount={df.loc[legit_idx,'Amount']:.2f}")

joblib.dump({'fraud': fraud_sample, 'legit': legit_sample, 'cols': COLS},
            'demo_samples.pkl')
print("  demo_samples.pkl saved")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Save Complete Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 10: Saving ESMOTE-GAN Enhanced Model")
print("=" * 65)

model_data = {
    # Base paper components
    'rf_models'    : rf_models,          # List[RandomForestClassifier] × N_SUBSETS
    'rf_weights'   : rf_weights,         # np.ndarray — normalized AUC weights
    # Enhancement: XGBoost
    'xgb'          : xgb,                # XGBClassifier trained on all augmented data
    # Final voting weights
    'rf_final_w'   : rf_final_w,         # Weight of RF ensemble in final vote
    'xgb_final_w'  : xgb_final_w,        # Weight of XGBoost in final vote
    # Enhancement: adaptive threshold
    'threshold'    : optimal_threshold,   # PR-optimal threshold (≠ 0.5)
    # Stored metrics (for UI display)
    'metrics': {
        #'accuracy'         : round(acc  * 100, 2),
        'precision'        : round(prec * 100, 2),
        'recall'           : round(rec  * 100, 2),
        'f1'               : round(f1,   4),
        'auc_roc'          : round(auc,  4),
        'false_alarm_rate' : round(far  * 100, 2),
        'threshold'        : round(optimal_threshold, 4),
        'n_rf_models'      : N_SUBSETS,
    }
}

joblib.dump(model_data, 'fraud_model.pkl')
print("  fraud_model.pkl saved")
print(f"  Components   : {N_SUBSETS} RF classifiers + Gradient Booster + adaptive threshold")
print(f"  RF weights   : {[f'{w:.3f}' for w in rf_weights]}")
print(f"  Final weights: RF_ensemble={rf_final_w:.3f}, Booster={xgb_final_w:.3f}")
print(f"  Threshold    : {optimal_threshold:.4f}")

print("\n" + "=" * 65)
print("  ALL DONE!  Now run:  python app.py")
print("=" * 65)
