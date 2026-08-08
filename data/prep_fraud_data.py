import pandas as pd

df = pd.read_csv("data/train_transaction.csv")
df_sample = df.head(10000)
df_sample.to_csv("data/ieee_fraud_sample.csv", index=False)
print(df_sample.shape)
print(df_sample['isFraud'].value_counts())
