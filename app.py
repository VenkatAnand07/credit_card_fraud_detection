from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load models ────────────────────────────────────────────────
MODEL_PATH  = 'fraud_model.pkl'
SCALER_PATH = 'scaler.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("fraud_model.pkl not found. Run train_model.py first!")

saved   = joblib.load(MODEL_PATH)   # dict: {'rf': ..., 'xgb': ...}
rf_model  = saved['rf']
xgb_model = saved['xgb']
scaler    = joblib.load(SCALER_PATH)  # fitted on ['Time', 'Amount']

FEATURE_COLS = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
print(f"Models loaded. RF: {type(rf_model).__name__}, XGB: {type(xgb_model).__name__}")
print(f"Scaler features: {list(scaler.feature_names_in_)}")


# ── Helpers ────────────────────────────────────────────────────
def risk_level(prob):
    if prob >= 0.80: return 'CRITICAL'
    if prob >= 0.60: return 'HIGH'
    if prob >= 0.35: return 'MEDIUM'
    return 'LOW'

def top_features(arr, n=5):
    arr = np.array(arr).flatten()
    idx = np.abs(arr).argsort()[::-1][:n]
    return [{'feature': FEATURE_COLS[i],
             'value':  round(float(arr[i]), 4),
             'impact': round(float(abs(arr[i])), 4)} for i in idx]

def scale_and_predict(features_30):
    """
    features_30: list/array of 30 values [Time, V1..V28, Amount]
    Returns (prediction, fraud_probability)
    """
    arr = np.array(features_30, dtype=float).reshape(1, -1)

    # Scale Time (index 0) and Amount (index 28) using a DataFrame
    # so the scaler gets the feature names it was fitted with
    scale_df = pd.DataFrame([[arr[0, 0], arr[0, 28]]],
                             columns=['Time', 'Amount'])
    scaled = scaler.transform(scale_df)

    arr[0, 0]  = scaled[0, 0]   # Time scaled
    arr[0, 28] = scaled[0, 1]   # Amount scaled

    # Wrap in DataFrame with correct column names for the models
    df = pd.DataFrame(arr, columns=FEATURE_COLS)

    # Get probabilities from each model
    rf_prob  = float(rf_model.predict_proba(df)[0][1])
    xgb_prob = float(xgb_model.predict_proba(df)[0][1])

    # Weighted average: RF weight=2, XGB weight=3
    final_prob = (rf_prob * 2 + xgb_prob * 3) / 5

    print(f"RF:{rf_prob:.4f}  XGB:{xgb_prob:.4f}")
    print(f"Final: {final_prob:.4f} -> {'FRAUD' if final_prob >= 0.5 else 'LEGIT'}")

    prediction = 1 if final_prob >= 0.5 else 0
    return prediction, final_prob


# ── Routes ─────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/demo_samples', methods=['GET'])
def demo_samples():
    CSV = 'creditcard.csv'
    if not os.path.exists(CSV):
        legit = [-1.3598071, -0.0727811,  2.5363467,  1.3781552, -0.3383208,
                  0.4623878,  0.2395986,  0.0986979,  0.3637870,  0.0907941,
                 -0.5515995, -0.6178009, -0.9913898, -0.3111695,  1.4681770,
                 -0.4704005,  0.2079708,  0.0257905,  0.4039936,  0.2514121,
                 -0.0183067,  0.2778375, -0.1104749,  0.0669281,  0.1285394,
                 -0.1891148,  0.1335584, -0.0210530,  149.62,     0.0]
        fraud  = [-1.3598071, -1.3404920,  1.7732093,  0.3797796, -0.5031970,
                   1.8004940,  0.7914580,  0.2476986, -1.5146543,  0.2076429,
                   0.6245015,  0.0660703,  0.7172927, -0.1656718,  2.3459989,
                  -2.8900832,  1.1099773, -0.1213592, -2.2618575,  0.5249797,
                   0.2479819,  0.7716376,  0.9094082, -0.6892811, -0.3276418,
                  -0.1390265, -0.0553528, -0.0597519,  529.00,    406.0]
        return jsonify({'legit': legit, 'fraud': fraud})

    df        = pd.read_csv(CSV)
    legit_row = df[df['Class'] == 0].iloc[0]
    fraud_row = df[df['Class'] == 1].iloc[0]
    return jsonify({
        'legit': legit_row[FEATURE_COLS].tolist(),
        'fraud': fraud_row[FEATURE_COLS].tolist()
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json(force=True)
        features = data['features']

        if len(features) != 30:
            return jsonify({'error': f'Expected 30 features, got {len(features)}'}), 400

        prediction, proba = scale_and_predict(features)

        confidence = round(proba if prediction == 1 else 1 - proba, 4)
        risk       = risk_level(proba)

        return jsonify({
            'prediction':        prediction,
            'result':            'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': round(proba, 4),
            'confidence_score':  confidence,
            'risk_level':        risk,
            'top_features':      top_features(features),
            'recommendation': (
                'BLOCK transaction immediately' if risk == 'CRITICAL' else
                'Flag for manual review'         if risk == 'HIGH'     else
                'Monitor closely'                if risk == 'MEDIUM'   else
                'Allow transaction'
            )
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'models': ['rf', 'xgb']}), 200


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data         = request.get_json(force=True)
        transactions = data['transactions']

        results = []
        for idx, t in enumerate(transactions):
            pred, prob = scale_and_predict(t)
            results.append({
                'transaction_id':    idx,
                'prediction':        pred,
                'result':            'FRAUD' if pred == 1 else 'LEGITIMATE',
                'fraud_probability': round(prob, 4),
                'risk_level':        risk_level(prob)
            })

        return jsonify({
            'total_transactions': len(results),
            'fraud_detected':     sum(1 for r in results if r['prediction'] == 1),
            'results':            results
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  FraudShield API running!")
    print("  Open -> http://127.0.0.1:5000/")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
