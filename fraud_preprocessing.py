"""
===========================================================
Fraud Dataset Preprocessing Module
Project : Trust Safety Platform
Agent   : Risk Scoring Agent
===========================================================

This module performs:

1. Load IEEE-CIS datasets
2. Merge transaction and identity datasets
3. Explore the dataset
4. Analyze missing values
5. Analyze fraud distribution
6. Identify numerical & categorical columns
7. Save merged dataset

Author : Harsh
"""

from pathlib import Path
import pandas as pd


class FraudPreprocessor:

    def __init__(self, transaction_path, identity_path):

        self.transaction_path = Path(transaction_path)
        self.identity_path = Path(identity_path)
        self.data = None

    #########################################################
    # Load Dataset
    #########################################################

    def load_data(self):

        print("=" * 60)
        print("Loading IEEE-CIS Fraud Dataset")
        print("=" * 60)

        print("\nLoading Transaction Dataset...")

        transaction = pd.read_csv(self.transaction_path)

        print("Transaction Shape :", transaction.shape)

        print("\nLoading Identity Dataset...")

        identity = pd.read_csv(self.identity_path)

        print("Identity Shape :", identity.shape)

        print("\nMerging datasets using TransactionID...")

        self.data = transaction.merge(
            identity,
            on="TransactionID",
            how="left"
        )

        print("\nDataset Loaded Successfully!")

        print("Merged Shape :", self.data.shape)

        return self.data

    #########################################################
    # Dataset Exploration
    #########################################################

    def explore_data(self):

        print("\n")
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)

        print("\nDataset Shape")
        print(self.data.shape)

        print("\n")

        print("=" * 60)
        print("COLUMN NAMES")
        print("=" * 60)

        print(self.data.columns.tolist())

        print("\n")

        print("=" * 60)
        print("DATA TYPES")
        print("=" * 60)

        print(self.data.dtypes)

        print("\n")

        print("=" * 60)
        print("MISSING VALUES (TOP 20)")
        print("=" * 60)

        missing = self.data.isnull().sum()
        missing = missing.sort_values(ascending=False)

        print(missing.head(20))

        print("\n")

        print("=" * 60)
        print("FRAUD DISTRIBUTION")
        print("=" * 60)

        print(self.data["isFraud"].value_counts())

        print("\n")

        print("=" * 60)
        print("FRAUD PERCENTAGE")
        print("=" * 60)

        print(self.data["isFraud"].value_counts(normalize=True) * 100)

        print("\n")

        print("=" * 60)
        print("NUMERICAL FEATURES")
        print("=" * 60)

        numerical = self.data.select_dtypes(
            include=["int64", "float64"]
        ).columns

        print("Total Numerical Features :", len(numerical))
        print(numerical.tolist())

        print("\n")

        print("=" * 60)
        print("CATEGORICAL FEATURES")
        print("=" * 60)

        categorical = self.data.select_dtypes(
            include=["object"]
        ).columns

        print("Total Categorical Features :", len(categorical))
        print(categorical.tolist())

    #########################################################
    # Dataset Summary
    #########################################################

    def summary(self):

        print("\n")

        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print("Rows :", self.data.shape[0])
        print("Columns :", self.data.shape[1])

        print("Missing Values :",
              self.data.isnull().sum().sum())

        fraud = self.data["isFraud"].sum()

        print("Fraud Transactions :", fraud)

        print("Normal Transactions :",
              len(self.data) - fraud)

    #########################################################
    # Save Dataset
    #########################################################

    def save_dataset(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        output_path = BASE_DIR / "datasets" / "fraud" / "merged_fraud_dataset.csv"

        self.data.to_csv(output_path, index=False)

        print("\nMerged dataset saved successfully!")

        print(output_path)


#########################################################
# MAIN
#########################################################

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    transaction_path = BASE_DIR / "datasets" / "fraud" / "train_transaction.csv"

    identity_path = BASE_DIR / "datasets" / "fraud" / "train_identity.csv"

    processor = FraudPreprocessor(
        transaction_path,
        identity_path
    )

    # Step 1 - Load and Merge Dataset
    processor.load_data()

    # Step 2 - Explore Dataset
    processor.explore_data()

    # Step 3 - Print Summary
    processor.summary()

    # Step 4 - Save Merged Dataset
    processor.save_dataset()