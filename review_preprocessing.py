"""
==========================================================
Review Dataset Preprocessing Module
Project : Trust Safety Platform
Agent   : Review Analysis Agent
==========================================================

This module performs:

1. Load Fake Reviews Dataset
2. Display dataset information
3. Handle missing values
4. Remove duplicate reviews
5. Identify review text and label columns
6. Save preprocessed dataset

Author : Harsh
"""

from pathlib import Path
import pandas as pd


class ReviewPreprocessor:

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)
        self.data = None

    #########################################################
    # Load Dataset
    #########################################################

    def load_data(self):

        print("=" * 60)
        print("Loading Fake Reviews Dataset")
        print("=" * 60)

        self.data = pd.read_csv(self.dataset_path)

        print("\nDataset Loaded Successfully!")
        print("Dataset Shape :", self.data.shape)

        return self.data

    #########################################################
    # Explore Dataset
    #########################################################

    def explore_data(self):

        print("\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)

        print("\nShape:")
        print(self.data.shape)

        print("\nColumns:")
        print(self.data.columns.tolist())

        print("\nData Types:")
        print(self.data.dtypes)

        print("\nFirst 5 Rows:")
        print(self.data.head())

        print("\nMissing Values:")
        print(self.data.isnull().sum())

    #########################################################
    # Remove Missing Values
    #########################################################

    def clean_data(self):

        print("\nRemoving Missing Values...")

        self.data.dropna(inplace=True)

        print("Shape after removing missing values :",
              self.data.shape)

        print("\nRemoving Duplicate Reviews...")

        self.data.drop_duplicates(inplace=True)

        print("Shape after removing duplicates :",
              self.data.shape)

    #########################################################
    # Detect Review and Label Columns
    #########################################################

    def detect_columns(self):

        print("\nDetecting Review and Label Columns...")

        review_candidates = [
            "review",
            "reviewText",
            "text",
            "content",
            "review_text"
        ]

        label_candidates = [
            "label",
            "class",
            "fake",
            "sentiment"
        ]

        review_column = None
        label_column = None

        for col in self.data.columns:

            if col in review_candidates:
                review_column = col

            if col in label_candidates:
                label_column = col

        print("Review Column :", review_column)
        print("Label Column :", label_column)

        return review_column, label_column

    #########################################################
    # Save Dataset
    #########################################################

    def save_dataset(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        output_path = (
            BASE_DIR /
            "datasets" /
            "reviews" /
            "cleaned_reviews_dataset.csv"
        )

        self.data.to_csv(output_path, index=False)

        print("\nCleaned Review Dataset Saved Successfully!")

        print(output_path)


#########################################################
# MAIN
#########################################################

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    dataset_path = (
        BASE_DIR /
        "datasets" /
        "reviews" /
        "fake_reviews_dataset.csv"
    )

    processor = ReviewPreprocessor(dataset_path)

    processor.load_data()

    processor.explore_data()

    processor.clean_data()

    processor.detect_columns()

    processor.save_dataset()