"""
=========================================================
Risk Scoring Agent
Project : Trust Safety Platform
=========================================================

This agent:
1. Loads the trained fraud detection model
2. Accepts transaction features
3. Predicts fraud probability
4. Returns a structured response
"""

from pathlib import Path
import joblib
import pandas as pd


class RiskAgent:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        self.model_path = BASE_DIR / "saved_models" / "risk_model.pkl"

        print("=" * 60)
        print("Loading Risk Model...")
        print("=" * 60)

        self.model = joblib.load(self.model_path)

        print("Risk Model Loaded Successfully!")

    ####################################################
    # Predict Risk
    ####################################################

    def predict(self, transaction):

        """
        transaction should be:
        - dictionary
        - pandas DataFrame (1 row)
        """

        # Convert dictionary to DataFrame
        if isinstance(transaction, dict):
            transaction = pd.DataFrame([transaction])

        probability = self.model.predict_proba(transaction)[0][1]

        prediction = self.model.predict(transaction)[0]

        if prediction == 1:
            decision = "Fraud"
        else:
            decision = "Legitimate"

        if probability >= 0.80:
            confidence = "High"

        elif probability >= 0.50:
            confidence = "Medium"

        else:
            confidence = "Low"

        result = {

            "prediction": decision,

            "risk_score": round(float(probability), 4),

            "confidence": confidence

        }

        return result


#########################################################
# MAIN (Testing)
#########################################################

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    dataset = BASE_DIR / "datasets" / "fraud" / "cleaned_fraud_dataset.csv"

    print("\nLoading sample transaction...\n")

    df = pd.read_csv(dataset)

    # Remove target column
    sample_transaction = df.drop("isFraud", axis=1).iloc[[0]]

    agent = RiskAgent()

    result = agent.predict(sample_transaction)

    print("\nPrediction Result")
    print("=" * 40)

    print(result)