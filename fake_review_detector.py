from pathlib import Path
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class FakeReviewDetector:

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000
        )

    #########################################################
    # Load Features and Labels
    #########################################################

    def load_data(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        feature_path = (
            BASE_DIR /
            "models" /
            "tfidf_features.pkl"
        )

        dataset_path = (
            BASE_DIR /
            "datasets" /
            "reviews" /
            "cleaned_reviews_dataset.csv"
        )

        with open(feature_path, "rb") as f:
            X = pickle.load(f)

        data = pd.read_csv(dataset_path)

        y = data["label"]

        print("Features Loaded:", X.shape)
        print("Labels Loaded:", y.shape)

        return X, y


    #########################################################
    # Train Model
    #########################################################

    def train(self, X, y):

        print("\nSplitting Dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("Training Model...")

        self.model.fit(
            X_train,
            y_train
        )

        print("Training Completed!")

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("\nModel Accuracy:", accuracy)

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions
            )
        )


    #########################################################
    # Save Model
    #########################################################

    def save_model(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        model_path = (
            BASE_DIR /
            "models" /
            "fake_review_model.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(
                self.model,
                f
            )

        print("\nFake Review Model Saved Successfully!")
        print(model_path)



if __name__ == "__main__":

    detector = FakeReviewDetector()

    X, y = detector.load_data()

    detector.train(
        X,
        y
    )

    detector.save_model()