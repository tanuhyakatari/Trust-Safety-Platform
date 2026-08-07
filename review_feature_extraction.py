from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class ReviewFeatureExtractor:

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.data = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )

    def load_data(self):
        print("Loading cleaned review dataset...")

        self.data = pd.read_csv(self.dataset_path)

        print("Dataset Loaded")
        print("Shape:", self.data.shape)


    def extract_features(self):

        print("\nExtracting TF-IDF features...")

        X = self.vectorizer.fit_transform(
            self.data["text"]
        )

        y = self.data["label"]

        print("Feature Shape:", X.shape)

        return X, y


    def save_features(self, X):

        BASE_DIR = Path(__file__).resolve().parent.parent

        model_dir = BASE_DIR / "models"

        model_dir.mkdir(exist_ok=True)

        with open(model_dir / "tfidf_features.pkl", "wb") as f:
            pickle.dump(X, f)

        with open(model_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        print("\nFeatures Saved Successfully!")


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    dataset_path = (
        BASE_DIR /
        "datasets" /
        "reviews" /
        "cleaned_reviews_dataset.csv"
    )

    extractor = ReviewFeatureExtractor(dataset_path)

    extractor.load_data()

    X, y = extractor.extract_features()

    extractor.save_features(X)