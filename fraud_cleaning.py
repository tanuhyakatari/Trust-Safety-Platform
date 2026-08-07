"""
==========================================================
Fraud Data Cleaning Module
Project : Trust Safety Platform
==========================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FraudCleaner:

    def __init__(self, input_file):

        self.input_file = Path(input_file)

        self.data = None

    ####################################################
    # Load merged dataset
    ####################################################

    def load_dataset(self):

        print("=" * 60)
        print("Loading merged dataset...")
        print("=" * 60)

        self.data = pd.read_csv(self.input_file)

        print("Dataset Shape :", self.data.shape)

    ####################################################
    # Remove columns with too many missing values
    ####################################################

    def remove_missing_columns(self):

        print("\nRemoving columns with >80% missing values...")

        threshold = len(self.data) * 0.80

        self.data = self.data.dropna(
            axis=1,
            thresh=threshold
        )

        print("Remaining Columns :", self.data.shape[1])

    ####################################################
    # Fill missing values
    ####################################################

    def fill_missing_values(self):

        print("\nFilling Missing Values...")

        numeric_columns = self.data.select_dtypes(
            include=["int64", "float64"]
        ).columns

        categorical_columns = self.data.select_dtypes(
            include=["object"]
        ).columns

        for col in numeric_columns:

            self.data[col] = self.data[col].fillna(
                self.data[col].median()
            )

        for col in categorical_columns:

            self.data[col] = self.data[col].fillna(
                "Unknown"
            )

        print("Missing values handled.")

    ####################################################
    # Encode categorical columns
    ####################################################

    def encode_categories(self):

        print("\nEncoding categorical columns...")

        encoder = LabelEncoder()

        categorical_columns = self.data.select_dtypes(
            include=["object"]
        ).columns

        for col in categorical_columns:

            self.data[col] = encoder.fit_transform(
                self.data[col].astype(str)
            )

        print("Encoding completed.")

    ####################################################
    # Split features and target
    ####################################################

    def split_dataset(self):

        print("\nSplitting Features and Target...")

        X = self.data.drop("isFraud", axis=1)

        y = self.data["isFraud"]

        print("Feature Shape :", X.shape)

        print("Target Shape :", y.shape)

        return X, y

    ####################################################
    # Save cleaned dataset
    ####################################################

    def save_dataset(self, output_file):

        self.data.to_csv(output_file, index=False)

        print("\nCleaned dataset saved to")

        print(output_file)


#########################################################
# MAIN
#########################################################

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    input_file = BASE_DIR / "datasets" / "fraud" / "merged_fraud_dataset.csv"

    output_file = BASE_DIR / "datasets" / "fraud" / "cleaned_fraud_dataset.csv"

    cleaner = FraudCleaner(input_file)

    cleaner.load_dataset()

    cleaner.remove_missing_columns()

    cleaner.fill_missing_values()

    cleaner.encode_categories()

    cleaner.split_dataset()

    cleaner.save_dataset(output_file)