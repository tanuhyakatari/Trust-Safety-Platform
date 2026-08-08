import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/ieee_fraud_sample.csv")

# Keep only numeric columns for speed (skip heavy categorical encoding for MVP)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols.remove('isFraud')
numeric_cols.remove('TransactionID')

X = df[numeric_cols].fillna(-999)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

joblib.dump(model, "models/risk_model.pkl")
joblib.dump(numeric_cols, "models/risk_model_columns.pkl")
print("Model saved.")
