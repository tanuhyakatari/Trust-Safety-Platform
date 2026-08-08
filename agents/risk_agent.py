import joblib
import time
import pandas as pd

model = joblib.load("models/risk_model.pkl")
columns = joblib.load("models/risk_model_columns.pkl")

def score_transaction(transaction_dict: dict):
    """
    Takes a dict of transaction features, returns fraud probability + latency.
    """
    start = time.time()

    row = pd.DataFrame([transaction_dict])
    for col in columns:
        if col not in row.columns:
            row[col] = -999
    row = row[columns].fillna(-999)

    prob = model.predict_proba(row)[0][1]
    latency_ms = (time.time() - start) * 1000

    if prob >= 0.7:
        decision = "block"
        reason = f"High fraud probability ({prob:.2f}) — auto-blocked"
    elif prob >= 0.3:
        decision = "human_review"
        reason = f"Moderate fraud probability ({prob:.2f}) — sent to review queue"
    else:
        decision = "approve"
        reason = f"Low fraud probability ({prob:.2f}) — auto-approved"

    return {
        "fraud_probability": round(float(prob), 4),
        "decision": decision,
        "reason": reason,
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    sample_row = pd.read_csv("data/ieee_fraud_sample.csv").iloc[0].to_dict()
    result = score_transaction(sample_row)
    print(result)
